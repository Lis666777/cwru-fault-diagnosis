# Five-family, three-modality diagnosis — v0.4 source

Editable Apache-2.0 inference source for a shared Qwen2.5-0.5B model with numerical adapters, a spectrum-image encoder and observation text. Fault families are bearing, gear, electrical, rotor and compound. Healthy is a separate state. The language model generates JSON directly; inference does not use SVM predictions.

## Measured test results

| Task | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| CWRU ten bearing states | 99.15% | 99.35% | 99.15% | 99.24% |
| LIMAN-C four motor states | 100% | 100% | 100% | 100% |
| MCC5-THU healthy / gear / compound | 99.57% | 97.44% | 99.78% | 98.56% |

Splits use complete source recordings. The fixed test sets had previously been evaluated by numerical baselines. CWRU includes one invalid generated JSON counted as an error. LIMAN state and source motor group are confounded; its result describes the supplied motor groups across load settings, not unseen-motor generalization.

Spectrum images are derived from the same numerical signal. Removing images did not change validation class predictions, so no independent accuracy gain from the image branch is claimed. Text used for evaluation describes recorded operating conditions, not fabricated maintenance events.

## Files and setup

- `diagnose.py`: fully readable inference implementation; 92 comment lines / 306 nonblank lines (30.07%).
- `model_config.json`: task dimensions, prompts, targets, foundation revision and local paths.
- `weights_manifest.json`: checksums of the separately supplied trained increments and normalization files.
- `requirements-runtime.txt`: versions used in the validated Windows / Python 3.12 / RTX 4070 environment.

Place the supplied checkpoint and normalization files under `weights/`. The local v0.4 delivery includes them; this source directory does not embed trained weights or datasets. Set `foundation_path` in `model_config.json` to a local `Qwen/Qwen2.5-0.5B-Instruct` checkout at revision `7ae557604adf67be50417f59c2c2f167def9a775`. `dependency_path` may point to the supplied dependency folder; otherwise install the listed packages in your environment. Install the CUDA 12.8 PyTorch wheel from the official PyTorch index.

```sh
python diagnose.py --task current --input current.npy --sample-rate 4096 --image spectrum.png --text "Three-phase current; source load setting 100." --output result.json
```

Supported arrays:

| task | Shape | Sample rate | Channel layout |
|---|---|---|---|
| bearing | `(2048,)` | 12000 Hz | CWRU drive-end vibration |
| current | `(16384,3)` | approximately 4096 Hz | LIMAN original source channels 1,2,3 |
| gear | `(25600,8)` | 12800 Hz | tachometer, torque, motor xyz, gearbox xyz |

Optional images must use the trained 256×64 grayscale frequency/channel heatmap layout. Omitting `--image` renders the paired spectrum automatically. `--text` accepts observation context; omitted text is reported as absent. JSON output retains the native generation, parsed diagnosis, fault family and actual modalities.

## Attribution

- Qwen Team: Qwen2.5-0.5B-Instruct foundation model, separately licensed.
- Case Western Reserve University Bearing Data Center: CWRU bearing measurements.
- Khizhik, Ali, Ryzhikov and Derkach: [LIMAN-C v1](https://data.mendeley.com/datasets/kccmrf3864/1), CC BY 4.0.
- Chen, Liu, He, Zou and Zhou: [MCC5-THU v2](https://data.mendeley.com/datasets/p92gj2732w/2), CC BY 4.0, DOI 10.1016/j.dib.2024.110453.

The repository's Apache-2.0 license applies to project-owned source, not to the separately licensed datasets or foundation model. No vehicle logs or project documents are included.
