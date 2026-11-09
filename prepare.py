"""Download official CWRU records and audit a recording-disjoint protocol."""
import hashlib
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

# Everything stays in this new experiment directory; old results are untouched.
ROOT = Path(__file__).resolve().parent
BASE = 'https://engineering.case.edu'
CLASSES = ['Normal', 'IR007', 'B007', 'OR007@6', 'IR014', 'B014', 'OR014@6', 'IR021', 'B021', 'OR021@6']

# UTF-8记录可直接人工检查，不依赖专用数据库。
# JSON中的类别顺序必须与模型目标表一致。
def write(path, value):
    # Save readable metadata independently of binary sample arrays.
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

# 返回内容摘要而不是文件名摘要。
# 摘要用于完整性检查，不能证明不同记录来自独立设备。
def sha(data):
    # Content fingerprints detect accidental duplicate records and windows.
    return hashlib.sha256(data).hexdigest()

# 官方链接由网页解析获得，不根据猜测拼接文件编号。
# 已存在的缓存不联网覆盖，复核时另对照发布清单。
# 网络失败抛出异常，禁止缺文件时悄悄缩减类别。
# 先写临时文件再替换，避免中断后误读半个MAT文件。
def download(url, path):
    # Reuse completed downloads; interrupted writes use a separate suffix.
    if not path.exists():
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        temporary = path.with_suffix(path.suffix + '.part')
        temporary.write_bytes(response.content)
        temporary.replace(path)
    return path.read_bytes()

# 数据重建必须写入新目录，不能改动包内参考数据。
# 从随包原始记录重建时不需重新联网。
# 若需重新获取源数据，应在独立实验中明确记录来源变化。
# 窗口按记录顺序排列，便于预测结果回溯到原始采样点。
# 分组以电机负载为依据，不以窗口随机划分。
# 同一轴承可能跨负载重复使用，报告不声称个体独立。
def main():
    # 新目录必须不存在，防止覆盖已有重建结果。
    # 原始文件先复制再读取，包内原件不做修改。
    # destination支持带中文或空格的路径。
    # ROOT在参数解析后切换，只影响本次进程。
    # 重建完成后用verify检查三个划分的逐数组一致性。
    # 自带缓存完整时download不会发起网络请求。
    # 不允许以缺少某个类别为理由减少40个源文件要求。
    global ROOT
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    source = ROOT
    ROOT = args.destination.resolve()
    ROOT.mkdir(parents=True, exist_ok=False)
    import shutil
    for folder in ['raw','sources']:
        shutil.copytree(source/folder, ROOT/folder)
    # File-to-class mapping comes from the official link labels, not guessed IDs.
    for folder in ['sources', 'raw', 'data']:
        (ROOT / folder).mkdir(exist_ok=True)
    # 注册表同时记录源页面、二进制链接、类别与负载。
    # 页面链接名与原始文件内部变量名分别核对。
    # 仅选三种缺陷尺寸及外圈@6方向，不混入其他方向。
    registry = []
    for page in ['normal-baseline-data', '12k-drive-end-bearing-fault-data']:
        url = BASE + '/bearingdatacenter/' + page
        html = download(url, ROOT / 'sources' / (page + '.html')).decode('utf-8')
        for link, label in re.findall(r'<a\s+[^>]*href="([^"]+\.mat)"[^>]*>([^<]+)</a>', html):
            label = label.strip()
            match = re.fullmatch(r'(.+)_([0-3])', label)
            if not match or match[1] not in CLASSES:
                continue
            name, load = match[1], int(match[2])
            registry.append(dict(label=name, class_id=CLASSES.index(name), load_hp=load,
                                 split=['train', 'train', 'val', 'test'][load], url=link,
                                 file=link.rsplit('/', 1)[1], source_page=url,
                                 source_fs=48000 if name == 'Normal' else 12000))
    # Exactly four load conditions must exist for every selected class.
    assert len(registry) == 40, len(registry)
    assert len({(r['label'], r['load_hp']) for r in registry}) == 40
    # 排序保证不同机器重建时样本顺序一致。
    # 排序不改变负载分组，不进行按成绩选择样本。
    registry.sort(key=lambda r: (r['class_id'], r['load_hp']))
    # Pin the protocol before feature extraction or any model fitting.
    protocol = dict(version=1, classes=CLASSES, target_fs=12000, window=2048, stride=2048,
                    split={'train':[0,1], 'val':[2], 'test':[3]},
                    normal_fs_evidence='Smith & Randall (2015), DOI 10.1016/j.ymssp.2015.04.021, Table A1: files 97-100, 48 kHz',
                    normal_resampling='scipy.signal.resample_poly(up=1, down=4); anti-alias FIR',
                    metric_average='macro; zero_division=0; all 10 labels',
                    baseline='fixed log-power spectrum + StandardScaler + SVC(C=10, gamma=scale); seed=0',
                    test_policy='No test-driven hyperparameter selection',
                    limitation='Recording/load disjoint, not independent bearing specimens; one numerical modality; three fault locations')
    protocol_path = ROOT / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding='utf-8')) == protocol
    else:
        write(protocol_path, protocol)
    # Downloads are independent; failures propagate rather than silently skipping a class.
    def fetch(r):
        raw = download(r['url'], ROOT / 'raw' / r['file'])
        return sha(raw)
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = list(pool.map(fetch, registry))
    assert len(set(hashes)) == 40, 'Duplicate raw recordings'
    # 训练、验证、测试分别累积，最终保存三个数组文件。
    # 原始记录哈希与选中通道哈希分别保留。
    # 99.mat包含额外X098变量，不能用第一个DE变量代替匹配。
    arrays = {s: [] for s in ['train', 'val', 'test']}
    labels = {s: [] for s in arrays}
    windows, seen = [], {}
    for r, digest in zip(registry, hashes):
        # Only DE waveform enters the model: filenames, RPM and labels are metadata.
        mat = loadmat(ROOT / 'raw' / r['file'])
        keys = [k for k in mat if k.endswith('_DE_time')]
        # Official 99.mat also embeds X098; never select the first available key.
        expected_key = 'X' + Path(r['file']).stem.zfill(3) + '_DE_time'
        assert expected_key in keys, (r['file'], keys)
        signal = mat[expected_key].reshape(-1).astype(np.float64)
        assert np.isfinite(signal).all()
        r.update(sha256=digest, de_key=expected_key, raw_samples=len(signal),
                 ignored_de_keys=[k for k in keys if k != expected_key], signal_sha256=sha(signal.tobytes()))
        # Healthy records are 48 kHz; faults selected from the official 12 kHz table.
        if r['source_fs'] == 48000:
            signal = resample_poly(signal, 1, 4)
        # 正常信号经抗混叠重采样后再切片。
        # 不足一个完整窗口的尾部丢弃，数量保存在注册表中。
        # 不填充尾部，避免把人工填充值引入模型输入。
        # 2048点在12kHz下约为0.1707秒。
        n = len(signal) // 2048
        r.update(windows=n, discarded_tail_samples=len(signal) % 2048)
        for i in range(n):
            # Non-overlap is enforced within each recording; split was assigned first.
            x = signal[i*2048:(i+1)*2048].astype(np.float32)
            h = sha(x.tobytes())
            assert h not in seen, ('Duplicate window', r['file'], seen.get(h))
            seen[h] = r['file']
            arrays[r['split']].append(x)
            labels[r['split']].append(r['class_id'])
            windows.append(dict(file=r['file'], split=r['split'], class_id=r['class_id'],
                                start_at_12khz=i*2048, length=2048, sha256=h))
    # MAT container hashes alone cannot detect duplicated embedded waveforms.
    assert len({r['signal_sha256'] for r in registry}) == 40
    for split in arrays:
        # Separate files let training avoid opening the held-out test samples.
        np.savez_compressed(ROOT / 'data' / (split + '.npz'),
                            x=np.stack(arrays[split]), y=np.array(labels[split]))
    # 窗口清单中的起点均指重采样后的12kHz序列。
    # 重建前后应逐数组比较，而不是仅比较压缩文件大小。
    # 记录数、重复检查和各类样本量一起构成数据审计证据。
    # 十个标签仍只覆盖健康及三种轴承缺陷位置。
    # 频谱由同一数值信号计算，不能据此计算成三种模态。
    write(ROOT / 'registry.json', registry)
    write(ROOT / 'windows.json', windows)
    audit = dict(raw_files=40, duplicate_raw_files=0, duplicate_windows=0,
                 protocol_sha256=sha(protocol_path.read_bytes()),
                 counts={s: np.bincount(labels[s], minlength=10).tolist() for s in arrays},
                 known_quality_caveat='Files 236/237 have reported clipping (Smith & Randall); retained without score-driven exclusion.',
                 split_unit='original recording, grouped by load', independent_specimens=False)
    write(ROOT / 'data_audit.json', audit)
    print(json.dumps(audit, ensure_ascii=True), flush=True)

if __name__ == '__main__':
    main()
