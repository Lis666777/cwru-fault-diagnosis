"""Run one 12 kHz, 2048-point DE vibration segment through the selected model."""
import argparse
import json
from pathlib import Path
import numpy as np
from baseline import features

# The published interface expects already resampled waveform data, not features.
ROOT = Path(__file__).resolve().parent

# 默认运行已冻结的Qwen模型，不会自动训练。
# 输入样本必须来自驱动端振动通道，幅值口径与CWRU一致。
# 不允许通过文件名推断标签，也不把文件名送入模型。
# 采样率检查属于接口约束，程序无法从波形自动证明采样率。
# 使用者应先按说明完成48kHz到12kHz的抗混叠重采样。
# JSON同时保留原始文本和解析结果，便于发现格式错误。
# 无效输出返回空诊断，不猜测最相近的故障类型。
# 本工具尚未验证其他机械设备或未知故障的输入。
# 输出为诊断实验结果，不包含剩余寿命或维修决策。
# 包内权重由复核清单标识，更换权重后应重新评价。
def main():
    # Requiring the sampling rate prevents silently interpreting 48 kHz as 12 kHz.
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--sample-rate',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',choices=['qwen','baseline'],default='qwen')
    args = parser.parse_args()
    # Disable object deserialization: input is a numeric NPY array only.
    x = np.load(args.input,allow_pickle=False)
    if args.sample_rate!=12000 or x.shape!=(2048,):
        raise ValueError('Require one DE waveform: shape (2048,), sampling rate 12000 Hz.')
    # Reject incomplete, complex or non-finite input rather than filling it silently.
    if x.dtype.kind not in 'fiu' or not np.isfinite(x).all():
        raise ValueError('Require finite real numerical values.')
    # Keep output identity explicit so the SVM cannot be mistaken for the large model.
    result = dict(model=args.model,input_shape=list(x.shape),sample_rate_hz=12000)
    if args.model=='qwen':
        import llm
        # This local experiment is verified on the existing NVIDIA CUDA environment.
        llm.torch.set_num_threads(4)
        model = llm.Model()
        model.restore(ROOT/'qwen_v1/learned.pt')
        normal = np.load(ROOT/'qwen_v1/normalization.npz')
        # Apply precisely the training-fitted spectral normalization.
        f = (features(x[None])-normal['mu'])/normal['sd']
        texts,ended = model.generate(llm.torch.tensor(f,dtype=llm.torch.float32,device='cuda'))
        label = llm.parse(texts[0])
        # Raw generation is preserved even when it is invalid or out of taxonomy.
        result.update(raw_generation=texts[0],eos_terminated=ended[0],valid=label>=0,
                      diagnosis=llm.TARGETS[label] if label>=0 else None)
    else:
        import joblib
        # Only load the model artifact produced by this trusted local experiment.
        model = joblib.load(ROOT/'baseline_v1/model.joblib')
        label = int(model.predict(features(x[None]))[0])
        classes = json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))['classes']
        result.update(class_id=label,diagnosis=classes[label])
    # No calibrated confidence or maintenance recommendation is invented.
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))

if __name__=='__main__':
    main()
