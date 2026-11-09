"""One fixed signal baseline; no hyperparameter search on the test set."""
import json
import hashlib
import platform
from pathlib import Path
import joblib
import numpy as np
import scipy
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parent

# 输入为二维数组，每行一个2048点窗口。
# 特征函数没有拟合状态，不接收标签或负载编号。
# 输出256个按频率升序排列的对数功率值。
# 同一个函数供基线与语言模型使用，避免前处理不一致。
def features(x):
    # Remove window DC, but retain vibration amplitude information.
    x = x.astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    # A Hann taper reduces spectral leakage at the boundaries of each segment.
    spectrum = np.abs(np.fft.rfft(x * np.hanning(2048), axis=1)) ** 2
    # Average adjacent bins for a compact 256-bin representation; discard DC.
    power = spectrum[:, 1:1025].reshape(-1, 256, 4).mean(axis=2)
    # Fixed numerical floor does not depend on validation or test statistics.
    return np.log10(power + 1e-12)

# 标签固定为0到9，未预测到的类别也纳入宏平均。
# 宏平均不按样本数量加权，因此须同时报告逐类指标。
# 无效预测使用负数，仍进入召回率和准确率的分母。
# 结果包含严格大于门槛的判断，等于门槛不能算通过。
# 混淆矩阵此处仅含合法类别，语言模型评价另加无效列。
def metrics(y, pred):
    # Macro averages keep each of the ten labels equally important.
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=list(range(10)), average='macro', zero_division=0)
    # The four thresholds are strict inequalities, as written in the project.
    values = dict(accuracy=float(accuracy_score(y, pred)), precision_macro=float(p), recall_macro=float(r), f1_macro=float(f))
    values['thresholds_pass'] = all(values[k] > v for k, v in zip(values, [.95, .93, .90, .92]))
    # Preserve class-level results so aggregate numbers cannot hide a failed class.
    pc, rc, fc, support = precision_recall_fscore_support(y, pred, labels=list(range(10)), zero_division=0)
    values['per_class'] = [dict(class_id=i, precision=float(pc[i]), recall=float(rc[i]), f1=float(fc[i]), support=int(support[i])) for i in range(10)]
    values['confusion_matrix'] = confusion_matrix(y, pred, labels=list(range(10))).tolist()
    return values

# 此程序是传统对照，不参与语言模型生成答案。
# 参数C和gamma在首次测试前固定，未作测试集调参。
# 目录已存在时拒绝覆盖，避免丢失原始实验。
# 训练数据标准化由流水线拟合并与分类器一起保存。
# 验证和测试只调用predict，不更新流水线参数。
def main():
    out = ROOT / 'baseline_v1'
    # A completed result cannot be silently replaced by a new experiment.
    out.mkdir(exist_ok=False)
    train = np.load(ROOT / 'data/train.npz')
    # Scaling is fitted only on training features, inside the saved pipeline.
    model = make_pipeline(StandardScaler(), SVC(C=10, gamma='scale', random_state=0))
    model.fit(features(train['x']), train['y'])
    joblib.dump(model, out / 'model.joblib')
    # Freeze the fitted model before reading either evaluation partition.
    report = dict(model='Fixed spectral SVM baseline; NOT a vertical large model',
                  protocol_sha256=hashlib.sha256((ROOT / 'protocol.json').read_bytes()).hexdigest(),
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__))
    for split in ['val', 'test']:
        data = np.load(ROOT / 'data' / (split + '.npz'))
        pred = model.predict(features(data['x']))
        report[split] = metrics(data['y'], pred)
        # Export every prediction for manual inspection and later model comparisons.
        np.savez(out / (split + '_predictions.npz'), truth=data['y'], prediction=pred)
    # Check that the actual saved artifact reproduces the in-memory predictions.
    restored = joblib.load(out / 'model.joblib')
    assert np.array_equal(restored.predict(features(data['x'])), pred)
    report['saved_model_reproduction'] = True
    (out / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({s: {k:v for k,v in report[s].items() if k not in ['per_class','confusion_matrix']} for s in ['val','test']}), flush=True)

if __name__ == '__main__':
    main()
