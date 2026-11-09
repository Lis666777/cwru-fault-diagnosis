"""Audit package evidence and optionally reproduce frozen model predictions."""
import argparse
import ast
import hashlib
import json
import platform
from pathlib import Path
import numpy as np

# 所有路径由脚本位置决定，与当前终端工作目录无关。
# 审计不修改原始数据、权重或归档的生成文本。
# 实际推理需要CUDA；纯数据和源码审计不加载基础模型。
ROOT = Path(__file__).resolve().parent

def read(path):
    # 统一UTF-8，避免中文目录下的系统编码差异。
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    # 大权重使用分块哈希，避免额外占用接近1GB内存。
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def audit():
    # 源码审计检查发布执行文件，不重复计算同一逻辑的历史快照。
    # 发布清单同时覆盖快照，排除注释统计不表示排除完整性检查。
    # 注释比例只是可读性的一项量化条件，仍需人工理解算法。
    # 第三方源码虽然随包附带，其数量不能提高自有源码的注释比例。
    # 验证失败时由异常给出具体文件名，便于定位交付缺失项。
    # 统计口径：整行#注释数除以非空物理行数。
    # 不把说明文档、空行、docstring或第三方代码当成注释。
    # 根目录是本版自有可执行源码；qwen_v1内执行快照仅作历史证据。
    # Python和PowerShell逐文件检查，避免以一个长注释文件拉高总比例。
    sources = {}
    for file in sorted(list(ROOT.glob('*.py'))+list(ROOT.glob('*.ps1'))):
        text = file.read_text(encoding='utf-8-sig')
        lines = [s for s in text.splitlines() if s.strip()]
        comments = sum(s.lstrip().startswith('#') for s in lines)
        sources[file.name] = dict(comments=comments,nonblank=len(lines),ratio=comments/len(lines))
        if file.suffix=='.py':
            # AST和compile仅检查语法，不执行训练或联网下载。
            ast.parse(text)
            compile(text,str(file),'exec')
        if comments/len(lines)<.30:
            raise ValueError('Comment ratio below 30%: '+file.name)
    registry = read(ROOT/'registry.json')
    # 清单中的分组必须与原协议的负载映射一致。
    # 同一文件出现在两组中时，不能以波形切片不同为理由通过。
    # 还要核对每个窗口引用的原文件确实属于该组。
    by_file = {r['file']:r for r in registry}
    windows = read(ROOT/'windows.json')
    seen = set()
    # 官方原文件与记录中的SHA256逐一核对。
    # 每个原文件只能分配到一个集合，不允许切片后再随机分组。
    if len(registry)!=40 or len({r['file'] for r in registry})!=40:
        raise ValueError('Unexpected recording registry')
    for record in registry:
        if record['split']!=['train','train','val','test'][record['load_hp']]:
            raise ValueError('Recording split violates the frozen protocol')
        if digest(ROOT/'raw'/record['file'])!=record['sha256']:
            raise ValueError('Raw file mismatch: '+record['file'])
    # 窗口摘要核对实际NPZ数组，而非仅相信清单中的重复计数。
    # 标签、顺序和数据长度也必须与清单一致。
    for split in ['train','val','test']:
        # 逐窗口检查偏移，确保重建时没有引入重叠采样。
        # 正常信号的偏移指重采样后序列，不是48kHz原始偏移。
        # 类别由原记录决定，窗口清单不能悄悄修改某个样本标签。
        offsets = {}
        data = np.load(ROOT/'data'/f'{split}.npz',allow_pickle=False)
        manifest = [r for r in windows if r['split']==split]
        if len(data['x'])!=len(manifest):
            raise ValueError('Window count mismatch')
        for x,y,row in zip(data['x'],data['y'],manifest):
            source = by_file[row['file']]
            if source['split']!=split or source['class_id']!=int(y):
                raise ValueError('Window source assignment mismatch')
            if row['start_at_12khz']!=offsets.get(row['file'],0) or row['length']!=2048:
                raise ValueError('Window overlap or offset mismatch')
            offsets[row['file']] = row['start_at_12khz']+2048
            h = hashlib.sha256(x.tobytes()).hexdigest()
            if h!=row['sha256'] or h in seen or int(y)!=row['class_id']:
                raise ValueError('Window hash, label or duplication mismatch')
            seen.add(h)
    # 选择记录锁定正式权重，不能静默换成另一个训练轮次。
    selection = read(ROOT/'qwen_v1/selection.json')
    if digest(ROOT/'qwen_v1/learned.pt')!=selection['checkpoint_sha256']:
        raise ValueError('Selected checkpoint mismatch')
    # 基础权重仍然必须匹配训练时的记录。
    protocol = read(ROOT/'qwen_v1/protocol.json')
    if digest(ROOT/'assets/base_model/model.safetensors')!=protocol['foundation_sha256']:
        raise ValueError('Foundation checkpoint mismatch')
    if digest(ROOT/'qwen_v1/executed_llm.py')!=protocol['code_sha256']:
        raise ValueError('Historical training source mismatch')
    # 发布清单存在时核对全部列出的文件，生成报告不计入静态清单。
    # 清单自校验不替代数字签名，作用是发现缺文件或传输损坏。
    manifest = ROOT/'release_manifest.json'
    if manifest.exists():
        for name,expected in read(manifest)['files'].items():
            if digest(ROOT/name)!=expected:
                raise ValueError('Release manifest mismatch: '+name)
    return dict(source_comments=sources,raw_files_verified=40,unique_windows_verified=len(seen),
                checkpoint_verified=True,foundation_verified=True,python=platform.python_version())

def reproduce():
    # 必须先完成审计，再加载模型，以免评价被改动的数据或权重。
    # 特征变换与正式接口共享baseline.features函数。
    # 特征标准化使用训练时保存的均值和标准差。
    # 这里不重新拟合标准化，也不观察测试标签构造输入。
    # 报告保留无效输出比例，格式错误不能被忽略。
    # 四项平均值相同不代表每一条预测相同，因此必须逐条比对。
    # 任何生成文本差异都导致重放失败，即使分类恰好相同。
    # 跨硬件浮点差异可能影响逐字重放，需要记录新环境而非掩盖差异。
    # 固定模型重放不会选择权重、优化参数或重新训练。
    # 输出到reports，不覆盖原test_generations.jsonl。
    import llm
    llm.torch.set_num_threads(4)
    llm.torch.backends.cuda.matmul.allow_tf32 = False
    llm.torch.use_deterministic_algorithms(True)
    model = llm.Model()
    model.restore(ROOT/'qwen_v1/learned.pt')
    norm = np.load(ROOT/'qwen_v1/normalization.npz')
    data = np.load(ROOT/'data/test.npz')
    x = llm.torch.tensor((llm.features(data['x'])-norm['mu'])/norm['sd'],dtype=llm.torch.float32)
    # 重放仍使用全词表贪心生成，不读取参考答案来约束解码。
    report = llm.evaluate(model,x,data['y'],ROOT/'reports','replayed_test')
    expected = read(ROOT/'qwen_v1/test_metrics.json')
    if report!=expected:
        raise ValueError('Replayed metrics differ from the original evaluation')
    # 逐个文本和结束标记一致，比只核对四个汇总数更严格。
    original = [json.loads(s) for s in (ROOT/'qwen_v1/test_generations.jsonl').read_text(encoding='utf-8').splitlines()]
    replay = [json.loads(s) for s in (ROOT/'reports/replayed_test_generations.jsonl').read_text(encoding='utf-8').splitlines()]
    if original!=replay:
        raise ValueError('Generated text or ordering differs from archive')
    return dict(exact_generation_matches=len(replay),metrics=report)

def main():
    # 复核命令不改变模型的正式输出目录或训练选择记录。
    # 每次运行生成可读报告，失败时终端保留异常详情。
    # reports目录用于运行产物，原始证据目录是只读使用的。
    # rebuild核对数组内容，不要求压缩ZIP容器字节完全一致。
    # 这种比较覆盖波形、标签及顺序三个方面。
    # 重建核对通过只说明预处理复现，不等于重新训练成绩复现。
    # 原训练过程保留完整代码快照，供需要时独立重训。
    # 本版不通过修改训练次数或更换划分去改善已有测试分数。
    # 默认只审计；--evaluate显式执行全部591条GPU生成。
    # --rebuilt用于核对prepare输出，读取但不修改重建文件。
    # 本机实测环境与复核结果一起保存，便于定位跨机器差异。
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--rebuilt',type=Path)
    args = parser.parse_args()
    (ROOT/'reports').mkdir(exist_ok=True)
    result = audit()
    if args.rebuilt:
        for split in ['train','val','test']:
            original = np.load(ROOT/'data'/f'{split}.npz')
            rebuilt = np.load(args.rebuilt/'data'/f'{split}.npz')
            if any(not np.array_equal(original[k],rebuilt[k]) for k in ['x','y']):
                raise ValueError('Rebuilt arrays mismatch: '+split)
        result['rebuilt_arrays_exact'] = True
    if args.evaluate:
        result['reproduction'] = reproduce()
    # 审计异常直接导致非零退出码，不输出虚假的通过状态。
    (ROOT/'reports/verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True),flush=True)

if __name__=='__main__':
    main()
