# CWRU fault diagnosis with Qwen

Editable Python and PowerShell source for a vibration-conditioned Qwen2.5-0.5B-Instruct model. The model generates diagnosis JSON directly; SVM predictions are not supplied to the language model.

## Validated experiment

The CWRU experiment uses drive-end vibration, ten labels (healthy and three fault locations at three defect diameters), and non-overlapping 2048-point windows at 12 kHz. Complete recordings at 0/1 HP form training, 2 HP validation, and 3 HP test. Healthy 48 kHz recordings are anti-alias resampled. The loader selects `X099_DE_time` explicitly because official `99.mat` also contains `X098` signals.

| Test metric | Result |
|---|---:|
| Accuracy | 97.4619% |
| Macro precision | 97.7709% |
| Macro recall | 97.4576% |
| Macro F1 | 97.4786% |

These are one-seed, one-protocol results on 591 test windows, with 15 errors. Recording/load separation does not establish independent bearing specimens. These are not vehicle-test results or evidence of five component-level fault families. Per-class performance is weaker than the aggregate for some labels.

## Run from the validated local delivery bundle

Prepare a compatible Windows/Python 3.12/CUDA environment, then run:

```powershell
python bootstrap.py --from-delivery "D:\path\to\CWRU_故障诊断交付候选_v0.2"
.\运行.ps1 -Python "C:\path\to\python.exe"
.\运行.ps1 -Action audit -Python "C:\path\to\python.exe"
.\运行.ps1 -Action evaluate -Python "C:\path\to\python.exe"
```

The bundle is a separate local artifact; this source repository does not claim to host trained weights. The offline bootstrap copies only explicit public-experiment assets. It does not collect private vehicle records or project documents. Existing asset directories are not overwritten.

## Rebuild from public sources

Install the matching PyTorch CUDA 12.8 wheel and the versions in `requirements-runtime.txt`. Then:

```powershell
python bootstrap.py
python llm.py --run-dir qwen_v1
python predict.py --input your_2048_point_window.npy --sample-rate 12000 --output result.json
```

This path downloads the public foundation model at commit `7ae557604adf67be50417f59c2c2f167def9a775`, verifies its SHA256, downloads and verifies 40 official CWRU files, and prepares the fixed data split. It trains a new adapter/LoRA rather than downloading project-trained weights. It requires network access for preparation and a compatible NVIDIA GPU for training/inference. New training results must be measured; identical results on arbitrary hardware are not guaranteed.

## Interface and checks

Input is a real-valued NPY array of shape `(2048,)`, representing one 12 kHz drive-end vibration window in the dataset's amplitude convention. Wrong shape, non-finite values and other sampling rates are rejected. Input filenames and labels do not enter model inference.

Output preserves `raw_generation`, `eos_terminated`, `valid`, and `diagnosis`. Diagnosis contains `fault` (`normal`, `inner`, `ball`, `outer`) and `diameter_mil` (`0`, `7`, `14`, `21`). Invalid generated JSON is counted as an error and is not repaired into a valid answer. There is no calibrated confidence or remaining-life estimate.

`verify.py` checks source comments, data/window hashes, selected weights, and optionally exact generation replay when the archived delivery evidence is present. Source comment accounting uses whole-line comments divided by nonblank physical lines and excludes third-party sources and historical duplicate snapshots. Raw data and environment assets are deliberately excluded from Git.

## License and data attribution

Project-owned source is distributed under Apache License 2.0; see `LICENSE` and `NOTICE.txt`. This does not relicense Qwen, third-party dependencies, or CWRU data.

- [CWRU normal baseline data](https://engineering.case.edu/bearingdatacenter/normal-baseline-data)
- [CWRU 12 kHz drive-end faults](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- Smith & Randall (2015), DOI: [10.1016/j.ymssp.2015.04.021](https://doi.org/10.1016/j.ymssp.2015.04.021).
