"""Readable local inference for five fault families and three input modalities."""
# 交付程序的三个入口按传感器布局划分，而不是按真实故障标签划分。
# bearing 接收 CWRU 驱动端振动；current 接收三相电流；gear 接收齿轮箱八通道记录。
# 五种故障大类为轴承、齿轮、电气、转子和复合，健康状态另外保留。
# 所有任务共享同一组基础语言模型权重和最终 LoRA 权重。
# 各任务的线性适配器负责将不同维度的频谱转换成八个信号 token。
# 图像分支读取真实像素，输出与数值分支相同维度的连续表示。
# 默认图像由输入信号生成；用户也可以显式提供配对 PNG。
# 自动图像与原信号是同源表示，不能当成独立传感器重复计数证据。
# 文本字段保留用户提供的工况或观测说明；空文本不会自动生成维修记录。
# 输出中的 engine、modalities 和 image_provenance 标明本次实际调用方式。
# 模型配置、归一化参数和权重均在交付目录中显式列出，可以独立检查。
# 基础模型路径可在 model_config.json 修改；本程序不依赖实验脚本。
# 评价结果及逐条生成文本位于 reports，示例输入位于 examples。
# 本地示例是演示材料，完整性能以预先划分的测试记录为准。
# 无效 JSON 在评价中计为错误，本程序也保留同一处理方式。
# 频谱计算公式、通道顺序和缺失修复方式必须与训练时保持一致。
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import signal,stats

# Paths are relative to this delivery folder, not the shell working directory.
ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'model_config.json').read_text(encoding='utf-8'))
# The existing local foundation dependencies are reused without downloading packages.
sys.path.insert(0,str((ROOT/CONFIG['dependency_path']).resolve()))
# Offline mode also prevents a misspelled model path from becoming a network request.
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
from torch import nn
from transformers import AutoTokenizer,AutoModelForCausalLM


def raw_features(task,path,fs):
    # NPY inputs contain actual sensor values, never precomputed fault labels.
    # Disallow pickle because the input contract is a numerical array.
    raw=np.load(path,allow_pickle=False)
    if raw.dtype.kind not in 'fiu':
        raise ValueError('Input must contain real numerical samples')
    if task=='bearing':
        # CWRU uses one 2048-point drive-end window at 12 kHz.
        # Anti-alias resampling, when needed, happens before this interface.
        if raw.shape!=(2048,) or fs!=12000 or not np.isfinite(raw).all():
            raise ValueError('Bearing input requires finite (2048,) samples at 12000 Hz')
        x=raw.astype(np.float64)
        # Remove DC without removing fault-related vibration amplitude.
        x=x-x.mean()
        # The Hann taper and four-bin averaging match the trained CWRU adapter.
        power=abs(np.fft.rfft(x*np.hanning(2048)))**2
        return np.log10(power[1:1025].reshape(256,4).mean(1)+1e-12)
    if task=='current':
        # 允许的采样率浮动只覆盖源时间坐标造成的微小差异。
        # 这个范围不构成对其他采样率或任意电机的适配支持。
        # LIMAN input order is the original source channels 1,2,3.
        # Do not supply already reordered ABC signals to this raw interface.
        if raw.shape!=(16384,3) or not 4095<=fs<=4097:
            raise ValueError('Current input requires (16384,3) samples near 4096 Hz')
        # Publisher mapping: A=1, B=channel3 shifted forward one sample, C=2.
        x=np.stack([raw[:-1,0],raw[1:,2],raw[:-1,1]],axis=1).astype(float)
        for c in range(3):
            missing=np.flatnonzero(~np.isfinite(x[:,c]))
            # The maximum allowed gap is three samples, per the source policy.
            runs=np.split(missing,np.flatnonzero(np.diff(missing)>1)+1)
            if max(map(len,runs))>3:
                raise ValueError('Current gap exceeds three samples')
            good=np.flatnonzero(np.isfinite(x[:,c]))
            # Uniform-time interpolation also supplies the nearest finite edge value.
            x[missing,c]=np.interp(missing,good,x[good,c])
        x=x-x.mean(axis=0)
        # 三个相位分别去均值，避免直流偏置影响交流频谱。
        # RMS 特征保留原始幅值，不做逐记录单位方差缩放。
        rms=np.sqrt(np.mean(x*x,axis=0))
        # Long Welch segments retain narrow current sidebands.
        f,psd=signal.welch(x,fs=fs,nperseg=8192,axis=0)
        edges=np.linspace(0,1024,513)
        # Frequency-major flattening is intentional; it matches the checkpoint.
        bins=np.stack([psd[(f>=a)&(f<b)].mean(axis=0) for a,b in zip(edges[:-1],edges[1:])])
        band=(f>=40)&(f<=70)
        # 供电基频通过限定频带内的总能量估计，避免高次谐波占优时误选。
        fundamental=f[band][np.argmax(psd[band].sum(axis=1))]
        harmonics=[]
        # Harmonics are measured from the current, not supplied from a class label.
        for k in range(1,16):
            mask=abs(f-k*fundamental)<=2
            harmonics.extend(np.log10(psd[mask].sum(axis=0)/(rms*rms+1e-15)+1e-15))
        physical=np.r_[np.log(rms+1e-12),rms/(rms.mean()+1e-12),
            stats.kurtosis(x,axis=0),np.max(abs(x),axis=0)/(rms+1e-12),
            np.corrcoef(x.T)[np.triu_indices(3,1)],harmonics]
        # 对数中的固定小量只用于数值稳定，不使用验证或测试集统计量。
        return np.r_[np.log10(bins+1e-15).ravel(),physical]
    # THU columns: tachometer, torque, motor xyz, gearbox xyz.
    # Vibration amplitudes use the source's g convention.
    if raw.shape!=(25600,8) or fs!=12800 or not np.isfinite(raw).all():
        raise ValueError('Gear input requires finite (25600,8) samples at 12800 Hz')
    edges=np.flatnonzero(np.diff((raw[:,0]>2).astype(int))==1)+1
    # Retain the same running-window eligibility rule used in evaluation.
    if len(edges)<18:
        raise ValueError('Too few tachometer pulses in this window')
    v=raw[:,2:8].astype(float)
    # 转速脉冲和扭矩不是加速度通道，不能混入六通道振动频谱。
    v-=v.mean(axis=0)
    rms=np.sqrt(np.mean(v*v,axis=0))
    _,power=signal.welch(v,fs=12800,nperseg=4096,axis=0)
    # Six channels remain distinct through the 256-band representation.
    bands=power[:2048].reshape(256,8,6).mean(axis=1)
    physical=np.r_[np.log(rms+1e-12),stats.kurtosis(v,axis=0),
        np.max(abs(v),axis=0)/(rms+1e-12),np.log(np.mean(raw[:,1]**2)+1e-12)]
    # 扭矩能量作为运行状态特征保留；故障名称和文件路径不进入特征。
    return np.r_[np.log10(bands+1e-16).ravel(),physical]


def render_spectrum(x,task):
    # x 已使用训练集均值和标准差归一化，所以显示上下限在所有数据中固定。
    # 灰度值越亮，代表相对于训练参考的对数功率越大。
    # current 原频谱包含512个频率区间，显示时插值到256列。
    # 物理统计量不绘入图像，图像分支仅观察频谱部分。
    # Render the trained frequency/channel layout without a class name or caption.
    # Images are derived views of numerical measurements, not independent sensors.
    if task=='bearing':
        grid=x[:256].reshape(1,256)
    elif task=='current':
        grid=x[:1536].reshape(512,3).T
    else:
        grid=x[:1536].reshape(256,6).T
    # Fixed training-standardized limits make brightness comparable between samples.
    pixels=np.round((np.clip(grid,-4,4)+4)*255/8).astype('uint8')
    return np.array(Image.fromarray(pixels).resize((256,64),Image.Resampling.BILINEAR))


class LoRA(nn.Module):
    # a 的形状为 rank×输入维度，b 的形状为输出维度×rank。
    # 基础投影使用 bfloat16；小矩阵计算使用 float32 以保持与训练一致。
    # forward 中的系数2对应 alpha/rank=16/8。
    # Rank and scale must match the shared language-model checkpoints.
    def __init__(self,base):
        super().__init__()
        self.base=base
        self.a=nn.Parameter(torch.empty(8,base.in_features,device='cuda'))
        self.b=nn.Parameter(torch.empty(base.out_features,8,device='cuda'))

    def forward(self,x):
        # All low-rank weights are restored before inference begins.
        return self.base(x)+(2*((x.float()@self.a.T)@self.b.T)).to(x.dtype)


class Vision(nn.Module):
    # 输入张量为 batch×1×64×256，灰度已经缩放到[0,1]。
    # 三次步长2卷积将空间尺寸变为8×32。
    # 固定2×1平均池化得到4×32，64通道展平后共有8192个值。
    # 共享卷积骨干学习频谱结构，各任务最后一层解释自身传感器排列。
    # CNN parameters interpret the actual PNG pixels supplied to the interface.
    def __init__(self,width):
        super().__init__()
        self.backbone=nn.Sequential(nn.Conv2d(1,16,3,padding=1,stride=2),nn.GELU(),
            nn.Conv2d(16,32,3,padding=1,stride=2),nn.GELU(),
            nn.Conv2d(32,64,3,padding=1,stride=2),nn.GELU(),
            nn.AvgPool2d((2,1)),nn.Flatten(),nn.Linear(8192,256),nn.GELU())
        # Sensor-specific heads preserve the three distinct acquisition layouts.
        self.heads=nn.ModuleDict({t:nn.Linear(256,8*width) for t in CONFIG['dimensions']})

    def forward(self,image,task):
        return self.heads[task](self.backbone(image))


class DiagnosisModel(nn.Module):
    # 推理对象不包含训练优化器、数据集加载器或基线分类器。
    # 两个小权重文件按顺序组合：先数值训练结果，再多模态增量。
    # 基础模型的大文件只读取一次，避免为不同任务复制三套大模型。
    # One foundation model and one shared LoRA serve every fault family.
    def __init__(self):
        super().__init__()
        base=(ROOT/CONFIG['foundation_path']).resolve()
        self.tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
        self.lm=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
        width=self.lm.config.hidden_size
        for layer in self.lm.model.layers:
            for name in ['q_proj','v_proj']:
                setattr(layer.self_attn,name,LoRA(getattr(layer.self_attn,name)))
        self.adapter=nn.ModuleDict({t:nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Linear(256,8*width)).cuda()
                                   for t,d in CONFIG['dimensions'].items()})
        self.vision=Vision(width).cuda()
        params=dict(self.named_parameters())
        # Numerical checkpoint initializes adapters and the shared language update.
        # Multimodal checkpoint replaces the shared update and supplies the vision branch.
        sets=[('numerical.pt',{n for n in params if n.startswith('adapter.') or n.endswith(('.a','.b'))}),
              ('multimodal.pt',{n for n in params if n.startswith('vision.') or n.endswith(('.a','.b'))})]
        with torch.no_grad():
            for name,expected in sets:
                state=torch.load(ROOT/'weights'/name,weights_only=True,map_location='cpu')
                # Exact key matching catches incomplete or incompatible model files.
                if set(state)!=expected:
                    raise ValueError('Checkpoint parameter mismatch: '+name)
                for key,value in state.items():
                    params[key].copy_(value)
        # Inference cannot silently update normalization or model parameters.
        self.requires_grad_(False)
        # eval 同时固定基础模型和图像分支的推理行为。
        self.eval()

    @torch.no_grad()
    def generate(self,task,x,image,text):
        # 一批样本必须属于同一设备任务并共享同一段背景文本。
        # 本命令行每次输入一个样本，内部仍保留批量维度以便应用复用。
        numeric=self.adapter[task](x)
        visual=self.vision(image,task)
        # The visual contribution is fixed at the trained coefficient of 0.1.
        weight=CONFIG['visual_weight']
        signal=((1-weight)*numeric+weight*visual).reshape(len(x),8,-1).to(torch.bfloat16)
        prompt=CONFIG['prompts'][task]
        if text:
            # User text is supplied as observation context, not rewritten into a label.
            prompt+='\n补充文本仅作为观测背景，不改变输出格式。\n'+text
        ids=self.tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True)
        ids=torch.tensor(ids,device='cuda')
        embeddings=self.lm.get_input_embeddings()(ids)[None].expand(len(x),-1,-1)
        # 信号 token 位于文本之前，让解码器能够同时关注两者。
        prefix=torch.cat([signal,embeddings],1)
        out=self.lm.model(inputs_embeds=prefix,use_cache=True,return_dict=True)
        cache,hidden=out.past_key_values,out.last_hidden_state[:,-1]
        eos=self.tokenizer.eos_token_id
        done=torch.zeros(len(x),device='cuda',dtype=torch.bool)
        tokens=[]
        for _ in range(64):
            # 最大生成长度固定，与真实故障类别无关。
            # Greedy native decoding: no candidate ranking or SVM output substitution.
            nxt=self.lm.lm_head(hidden).argmax(-1)
            nxt=torch.where(done,torch.full_like(nxt,eos),nxt)
            tokens.append(nxt.cpu());done=done|(nxt==eos)
            if bool(done.all()):
                # 所有样本产生结束标记后立即停止，避免无意义的额外生成。
                break
            out=self.lm.model(input_ids=nxt[:,None],past_key_values=cache,use_cache=True,return_dict=True)
            cache,hidden=out.past_key_values,out.last_hidden_state[:,-1]
        results=[]
        for row in torch.stack(tokens,1).tolist():
            ended=eos in row
            row=row[:row.index(eos)] if ended else row
            results.append((self.tokenizer.decode(row,skip_special_tokens=False),ended))
        return results


def decode(text,task):
    # 字典键值必须完整匹配该任务的诊断输出定义。
    # 轴承尺寸要求整数，避免把 JSON 布尔值误当成数值标签。
    # Preserve invalid JSON as an invalid prediction, rather than guessing a class.
    def pairs(items):
        value={}
        for key,item in items:
            if key in value:
                raise ValueError('duplicate key')
            value[key]=item
        return value
    try:
        value=json.loads(text,object_pairs_hook=pairs)
        if task=='bearing' and type(value.get('diameter_mil')) is not int:
            return None
        return next((target for target in CONFIG['targets'][task] if target==value),None)
    except (ValueError,AttributeError):
        return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',choices=list(CONFIG['dimensions']),required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--sample-rate',type=float,required=True)
    parser.add_argument('--image',type=Path)
    parser.add_argument('--text',default='')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    # Reports may not overwrite the numerical or image inputs.
    if args.output.resolve() in [args.input.resolve()]+([args.image.resolve()] if args.image else []):
        raise ValueError('Output must be a separate file')
    if len(args.text)>1000:
        raise ValueError('Text context must be at most 1000 characters')
    feature=raw_features(args.task,args.input,args.sample_rate)
    normal=np.load(ROOT/'weights'/f'{args.task}_normalization.npz')
    # These means and scales were fitted only on training records.
    x=(feature-normal['mu'])/normal['sd']
    if not np.isfinite(x).all():
        raise ValueError('Input does not produce finite spectral features')
    if args.image:
        # 输入图像必须与当前数值窗口对应；程序无法替用户证明配对关系。
        with Image.open(args.image) as im:
            # External PNGs must follow the documented trained heatmap layout.
            if im.size!=(256,64):
                raise ValueError('Expected a 256x64 spectrum heatmap')
            pixels=np.array(im.convert('L'))
        provenance='user_supplied_spectrum_image'
    else:
        pixels=render_spectrum(x,args.task)
        provenance='spectrum_image_derived_from_numerical_input'
    torch.set_num_threads(4)
    model=DiagnosisModel()
    inputs=torch.tensor(x[None],dtype=torch.float32,device='cuda')
    image=torch.tensor(pixels[None,None]/255,dtype=torch.float32,device='cuda')
    generated,eos=model.generate(args.task,inputs,image,args.text)[0]
    diagnosis=decode(generated,args.task)
    family=None
    if diagnosis:
        fault=diagnosis['fault']
        # The public family field unifies task-specific diagnostic labels.
        family='healthy' if fault=='normal' else 'bearing' if args.task=='bearing' else {
            'bearing':'bearing','rotor_bar':'rotor','interturn_short':'electrical','gear':'gear','compound':'compound'}[fault]
    result=dict(engine='multimodal shared Qwen2.5-0.5B v0.4',task=args.task,
        modalities=['numerical','image']+(['text'] if args.text else []),
        image_provenance=provenance,text_context=args.text,raw_generation=generated,
        eos_terminated=eos,valid=diagnosis is not None,diagnosis=diagnosis,fault_family=family)
    # fault_family 用于统一展示，diagnosis 保留任务原生的具体输出。
    # 输出文件使用 UTF-8，方便在 Python、浏览器或其他可编辑工具中读取。
    # Raw generated text remains available for independent format checks.
    # No probability or repair recommendation is fabricated from the class label.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
