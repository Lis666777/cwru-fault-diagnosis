"""CWRU spectrum-conditioned Qwen: independent native JSON experiment."""
import os
import sys
import json
import random
import hashlib
from pathlib import Path
import numpy as np
from baseline import features, metrics

# Reuse only the locally downloaded foundation libraries and foundation weights.
# 运行时所有模型资产均从交付目录定位。
# 移动整个文件夹无需改写基础权重路径。
# 本地基础模型和依赖随包提供，Python及CUDA运行时另行配置。
# 关闭自动联网下载，避免版本漂移影响复现。
# 包内基础权重保留其原始许可证和模型卡。
ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
sys.path.insert(0, str(ASSETS / 'deps'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# This pilot deliberately accepts one numerical modality and one fixed prompt.
# 这是固定任务提示，不是从样本标签拼接的问题。
# fault为故障位置，diameter_mil为缺陷直径的千分之一英寸数。
# 健康状态直径为0；三种缺陷尺寸为7、14、21。
# 生成内容不含轴承寿命、故障概率或维修建议。
# 模板中的候选描述对所有样本相同，不提供样本答案。
PROMPT = '根据输入的轴承振动频谱诊断状态。仅输出JSON，字段为fault、diameter_mil。fault取normal、inner、ball、outer；健康时diameter_mil为0，故障时为7、14或21。'
TARGETS = [dict(fault='normal', diameter_mil=0)] + [dict(fault=f, diameter_mil=d) for d in [7,14,21] for f in ['inner','ball','outer']]

# 训练日志、协议和选择结果统一写成可读JSON。
# 保存真实结果，不将小数截断后再判断是否达标。
def save(path, value):
    # Human-readable reports preserve the exact observed output and selection rule.
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

# 训练协议记录基础权重、代码和数据协议的摘要。
# 基础模型较大，使用分块读取控制内存。
def sha(path):
    # Hash artifacts in blocks so the foundation file need not be loaded twice.
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

# 低秩增量由A和B两个矩阵组成。
# 秩为8，alpha为16，因此增量乘2。
# B初始为零，初始投影等于冻结的基础模型。
# A随机初始化，随后与B共同接受答案损失梯度。
# 只改动注意力中的q和v投影，不训练全量基础模型。
class LoRA(nn.Module):
    # The foundation projection stays frozen; rank-eight increments start at zero.
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(8, base.in_features, device='cuda'))
        self.b = nn.Parameter(torch.zeros(base.out_features, 8, device='cuda'))
        nn.init.kaiming_uniform_(self.a, a=5**.5)

    def forward(self, x):
        # FP32 low-rank parameters are converted back to the base activation type.
        return self.base(x) + (2 * ((x.float() @ self.a.T) @ self.b.T)).to(x.dtype)

# 接口接收标准化后的256维频谱，不接收类别预测。
# 模型包含适配器、冻结基础模型和可训练LoRA。
# 权重文件仅保存可训练参数，基础模型在assets内单独保存。
# 模型结构尺寸取自基础配置，必须与所选权重一致。
# 当前实现经本机NVIDIA CUDA环境验证，不承诺CPU推理。
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(ASSETS/'base_model', local_files_only=True)
        self.lm = AutoModelForCausalLM.from_pretrained(ASSETS/'base_model', local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
        self.lm.requires_grad_(False)
        # 适配器先映射至256维隐藏层，再输出8个语言模型宽度的token。
        # 这8个token是数值特征嵌入，不是8个故障标签。
        # 所有频谱频段经同一个适配器联合处理。
        width = self.lm.config.hidden_size
        # The 256 spectral bins produce eight learned signal tokens, without labels.
        self.adapter = nn.Sequential(nn.Linear(256,256), nn.GELU(), nn.Linear(256,8*width)).cuda()
        for layer in self.lm.model.layers:
            for name in ['q_proj','v_proj']:
                setattr(layer.self_attn, name, LoRA(getattr(layer.self_attn, name)))
        # Every sample receives the same instruction; no filename or load enters it.
        ids = self.tokenizer.apply_chat_template([dict(role='user',content=PROMPT)], tokenize=True, add_generation_prompt=True)
        self.register_buffer('prompt', torch.tensor(ids, device='cuda'))
        answers = [self.tokenizer.encode(json.dumps(t,separators=(',',':')), add_special_tokens=False)+[self.tokenizer.eos_token_id] for t in TARGETS]
        # 训练答案长度不同，统一填充后使用布尔掩码。
        # EOS参与监督，使模型学习在JSON后结束。
        # 填充位置不参与损失，也不影响有效答案的注意力。
        # 标签只能在训练监督路径中取目标答案。
        size = max(map(len,answers))
        ids = torch.full((10,size), self.tokenizer.eos_token_id, device='cuda', dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, answer in enumerate(answers):
            ids[i,:len(answer)] = torch.tensor(answer,device='cuda')
            mask[i,:len(answer)] = True
        # Answers are used only in supervised loss, never as generation candidates.
        self.register_buffer('answers',ids)
        self.register_buffer('mask',mask)

    # 前缀先放信号token，再放固定文本提示。
    # 信号嵌入转换为BF16以匹配基础模型计算类型。
    # 输入张量应为批大小乘256，批次之间不共享标签信息。
    def prefix(self, x):
        # Numerical features are continuous embeddings, not SVM predictions.
        signal = self.adapter(x).reshape(len(x),8,-1).to(torch.bfloat16)
        text = self.lm.get_input_embeddings()(self.prompt).unsqueeze(0).expand(len(x),-1,-1)
        return torch.cat([signal,text],1)

    # 训练使用教师强制，目标答案嵌入仅用于因果预测。
    # 因果掩码防止当前位置看到未来答案token。
    # 答案第一个token对应前缀最后一个隐藏状态。
    # 返回总损失而非均值，外层按有效token总数归一化。
    # 分块输出头仍使用完整词表，不缩小候选答案集合。
    def loss(self, x, y):
        prefix = self.prefix(x)
        ids, mask = self.answers[y], self.mask[y]
        attention = torch.cat([torch.ones(prefix.shape[:2],device='cuda',dtype=torch.bool),mask],1)
        hidden = self.lm.model(inputs_embeds=torch.cat([prefix,self.lm.get_input_embeddings()(ids)],1),
            attention_mask=attention,use_cache=False,return_dict=True).last_hidden_state
        # The last prefix position predicts the first answer token; pad loss is zero.
        selected = hidden[:,prefix.shape[1]-1:prefix.shape[1]+ids.shape[1]-1][mask]
        targets = ids[mask]
        loss = selected.sum()*0
        for start in range(0,len(targets),128):
            # Full vocabulary loss is chunked only to reduce peak GPU memory.
            logits = self.lm.lm_head(selected[start:start+128]).float()
            loss = loss + nn.functional.cross_entropy(logits,targets[start:start+128],reduction='sum')
        return loss

    # 生成关闭梯度，避免不必要的显存占用。
    # 生成接口不接受真实标签，也不读取训练答案长度。
    # 最大64个新token是固定上限，并非依照样本标签设置。
    # 已遇EOS的样本仅填充EOS，不继续改变其最终输出。
    @torch.no_grad()
    def generate(self, x):
        # Fixed maximum generation length is independent of the true class.
        self.eval()
        out = self.lm.model(inputs_embeds=self.prefix(x),use_cache=True,return_dict=True)
        cache, hidden = out.past_key_values, out.last_hidden_state[:,-1]
        eos = self.tokenizer.eos_token_id
        done = torch.zeros(len(x),device='cuda',dtype=torch.bool)
        tokens = []
        for _ in range(64):
            # No constrained decoding or nearest-valid-answer correction is used.
            nxt = self.lm.lm_head(hidden).argmax(-1)
            nxt = torch.where(done,torch.full_like(nxt,eos),nxt)
            tokens.append(nxt.cpu())
            done = done | (nxt==eos)
            if bool(done.all()):
                break
            out = self.lm.model(input_ids=nxt[:,None],past_key_values=cache,use_cache=True,return_dict=True)
            cache, hidden = out.past_key_values, out.last_hidden_state[:,-1]
        # 返回原始字符串和是否正常结束两项证据。
        # 解码不隐藏特殊token，异常生成会保留给格式检查。
        # 不得把无效文本自动改写为合法的类别JSON。
        texts, ended = [], []
        for row in torch.stack(tokens,1).tolist():
            ended.append(eos in row)
            row = row[:row.index(eos)] if eos in row else row
            texts.append(self.tokenizer.decode(row,skip_special_tokens=False))
        return texts, ended

    # 保存CPU副本，使存盘不依赖当前CUDA张量生命周期。
    # 只保存需要梯度的参数，避免重复封装基础模型。
    def checkpoint(self):
        # Base weights stay external; the saved state contains all trained parameters.
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}

    # 加载时仅允许安全权重格式，不执行任意对象反序列化。
    # 参数名称集合必须完全一致，缺失参数不能静默忽略。
    # 参数复制在无梯度上下文完成，不改变训练参数开关。
    def restore(self,path):
        state = torch.load(path,weights_only=True,map_location='cpu')
        params = {n:p for n,p in self.named_parameters() if p.requires_grad}
        assert set(state)==set(params)
        with torch.no_grad():
            for n,p in params.items():
                p.copy_(state[n])

# Python默认接受重复JSON键，因此这里使用专用键检查。
# 拒绝NaN和Infinity等非标准JSON数值。
# diameter_mil必须是整数，字符串或布尔值不等价。
# 返回-1表示格式或目标语义非法，评价时计作错误。
def parse(text):
    # Duplicate keys and non-standard numeric constants are invalid, not repaired.
    def pairs(items):
        obj = {}
        for k,v in items:
            if k in obj:
                raise ValueError('duplicate key')
            obj[k] = v
        return obj
    def reject(value):
        raise ValueError(value)
    try:
        obj = json.loads(text,object_pairs_hook=pairs,parse_constant=reject)
    except ValueError:
        return -1
    if not isinstance(obj,dict) or set(obj)!={'fault','diameter_mil'} or type(obj['diameter_mil']) is not int:
        return -1
    return next((i for i,t in enumerate(TARGETS) if obj==t),-1)

# 分批推理以控制显存，批大小不改变样本顺序。
# 真实标签只用于生成完成后的统计，不传入generate。
# 保留每一个样本的原始文本，不只导出正确样本。
# 混淆矩阵最后一列为非法输出，不能从分母删去。
# 宏平均指标与逐类指标一起报告，便于识别弱类。
def evaluate(model,x,y,out,name):
    texts, ended = [], []
    for start in range(0,len(x),32):
        t,e = model.generate(x[start:start+32].cuda())
        texts.extend(t)
        ended.extend(e)
    pred = np.array([parse(t) for t in texts])
    result = metrics(y,pred)
    # Invalid generated strings count as errors and have a separate confusion column.
    cm = np.zeros((10,11),dtype=int)
    np.add.at(cm,(y,np.where(pred<0,10,pred)),1)
    result['confusion_matrix_with_invalid_column'] = cm.tolist()
    result['invalid_rate'] = float(np.mean(pred<0))
    result['eos_rate'] = float(np.mean(ended))
    save(out/(name+'_metrics.json'),result)
    with (out/(name+'_generations.jsonl')).open('w',encoding='utf-8') as stream:
        for i,(t,p,e) in enumerate(zip(texts,pred,ended)):
            stream.write(json.dumps(dict(index=i,truth=int(y[i]),prediction=int(p),generated=t,eos=e),ensure_ascii=False)+'\n')
    return result

# 重新训练默认写入runs下的新目录，不覆盖已发布权重。
# 重复目录会报错，防止实验结果混合。
# 种子、训练轮数与验证选择点沿用原实验方案。
# 基础权重始终冻结，新适配器和LoRA从初始化开始训练。
# 训练仅访问train和val，选择权重前不打开test。
# 重训结果是新的实验记录，不自动替换当前交付模型。
def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=ROOT/'runs/retrain0')
    args = parser.parse_args()
    out = args.run_dir
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    # 保存实际执行源码，供独立核对训练协议中的代码摘要。
    (out/'executed_llm.py').write_bytes(Path(__file__).read_bytes())
    # Pilot budget and selection rule are fixed before fitting; test stays unopened.
    save(out/'protocol.json',dict(epochs=6,seed=0,selection_epochs=[2,4,6],
        selection='maximum validation macro F1, earliest tie',input='256 log-power bins, eight learned tokens',
        adapter_lr=.001,lora_lr=.0001,lora_rank=8,lora_alpha=16,batch=32,microbatch=8,
        prompt=PROMPT,targets=TARGETS,test_policy='evaluate once only if validation passes all four thresholds',
        dataset_protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(Path(__file__)),
        foundation_sha256=sha(ASSETS/'base_model/model.safetensors')))
    train, val = np.load(ROOT/'data/train.npz'),np.load(ROOT/'data/val.npz')
    # 归一化均值和标准差仅由训练频谱计算。
    # 标准差下限防止常数特征发生除零。
    # 验证与测试应用同一组参数，不重新拟合。
    f = features(train['x'])
    mu, sd = f.mean(0),np.maximum(f.std(0),1e-8)
    np.savez(out/'normalization.npz',mu=mu,sd=sd)
    # Normalization uses training only, independently from the SVM fitted pipeline.
    x = torch.tensor((f-mu)/sd,dtype=torch.float32)
    vx = torch.tensor((features(val['x'])-mu)/sd,dtype=torch.float32)
    y = torch.tensor(train['y'],device='cuda')
    model = Model()
    opt = torch.optim.AdamW([dict(params=model.adapter.parameters(),lr=.001),
        dict(params=[p for p in model.lm.parameters() if p.requires_grad],lr=.0001)],weight_decay=.01)
    params = [p for p in model.parameters() if p.requires_grad]
    history, best = [], -1
    # 每轮仅打乱训练样本顺序，不重分数据集合。
    # 有效批大小为32，通过8样本微批累积梯度。
    # 损失按整批有效token数归一化，避免短答案权重失衡。
    # 冻结模型保持eval模式，但可训练参数仍可反向传播。
    for epoch in range(1,7):
        model.eval()
        order = torch.randperm(len(x))
        total, count = 0.,0
        for start in range(0,len(order),32):
            ix = order[start:start+32]
            denom = model.mask[y[ix.cuda()]].sum()
            opt.zero_grad(set_to_none=True)
            for offset in range(0,len(ix),8):
                batch = ix[offset:offset+8]
                loss = model.loss(x[batch].cuda(),y[batch.cuda()])
                assert torch.isfinite(loss)
                (loss/denom).backward()
                total += float(loss.detach())
            # Clip the accumulated gradient once per effective batch.
            norm = nn.utils.clip_grad_norm_(params,1.)
            assert torch.isfinite(norm)
            opt.step()
            count += int(denom)
        row = dict(epoch=epoch,token_loss=total/count)
        if epoch%2==0:
            result = evaluate(model,vx,val['y'],out,'val_epoch'+str(epoch))
            row['validation'] = result
            if result['f1_macro']>best:
                best = result['f1_macro']
                torch.save(model.checkpoint(),out/'learned.pt')
                save(out/'selection.json',dict(epoch=epoch,validation=result,checkpoint_sha256=sha(out/'learned.pt')))
        history.append(row)
        save(out/'history.json',history)
        print('QWEN',epoch,round(total/count,5),row.get('validation',{}).get('accuracy'),flush=True)
    # 最佳权重仅由验证宏F1确定，同分保留较早轮次。
    # 重新加载后复现验证结果，检查保存加载没有丢参数。
    # 只有验证四项门槛同时通过，才运行预定测试评价。
    # 测试之后不能以该成绩继续选择参数并仍称独立测试。
    model.restore(out/'learned.pt')
    # Recheck the saved checkpoint before exposing the held-out test partition.
    selected = json.loads((out/'selection.json').read_text(encoding='utf-8'))
    checked = evaluate(model,vx,val['y'],out,'val_selected')
    assert checked==selected['validation']
    if checked['thresholds_pass']:
        test = np.load(ROOT/'data/test.npz')
        tx = torch.tensor((features(test['x'])-mu)/sd,dtype=torch.float32)
        result = evaluate(model,tx,test['y'],out,'test')
        print('QWEN_TEST',json.dumps({k:v for k,v in result.items() if isinstance(v,(float,bool))}),flush=True)
    save(out/'status.json',dict(stage='complete',test_evaluated=checked['thresholds_pass']))

if __name__=='__main__':
    main()
