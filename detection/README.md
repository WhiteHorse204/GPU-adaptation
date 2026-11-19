# Detection project requirements

```text
model inference for GPU adaptation/
├── dataset/                        # Existing shared datasets
└── detection/
    ├── evaluate.py                 # Shared COCO bbox AP evaluator
    └── gaoyonghan/
        ├── environment/            # Dependency manifests and model mapping
        └── models/
            └── <model_name>/
                ├── inference.py   # Required public inference entry
                ├── weights/       # Required model weights
                ├── result/        # Required inference and evaluation outputs
                ├── inference/     # Existing model-specific internals
                └── README.md
```

Other contributors use `detection/<owner>/models/<model_name>/`.

## Required contract

1. Each model has `inference.py`, `weights/` and `result/` in its root directory.
2. `python inference.py` runs the complete model-specific workflow including
   output conversion. Internal scripts and frameworks remain model-specific.
3. Default outputs go to `result/run_001/`. `inference_output.json` records
   completion and relative paths to standardized outputs. Scoring-ready bundles
   contain `annotations.json`, `predictions.json` and `protocol.json`.
4. `detection/evaluate.py` contains the common input validation and COCO bbox AP
   scoring logic. Every AP model must emit the project-standard COCO bbox output
   from its own `inference.py`. Models do not have an `evaluation/` or `vendor/`
   directory.
5. Default paths resolve against script locations and work from other working
   directories. Use the original project-level `dataset/` with the model-relative
   path `../../../../dataset/`. No symbolic links or dataset copies are required.
6. Preserve native preprocessing, predictions, category mapping, selected samples,
   score thresholds and evaluation rules. Environment consolidation is separate.
7. Historical skipped models retain their status. Prediction-only workflows
   without a validated scoring protocol remain explicitly marked as such.

## Example

From this detection directory:

```bash
python gaoyonghan/models/facebook--detr-resnet-50/inference.py
python evaluate.py --run-dir gaoyonghan/models/facebook--detr-resnet-50/result/run_001
```

Relative evaluator arguments resolve against this detection directory. Absolute
paths are accepted. `--bundle` evaluates one prepared scoring bundle. The model
entry supports `--show-command` and `--prepare-only`; other native options remain
model-specific. There is no central model inference dispatcher.

## Evaluation entry point

All detection models that participate in the project evaluation use:

    python detection/evaluate.py --run-dir "/mnt/aoss/model inference for GPU adaptation/detection/<contributor>/models/<model-name>/result/run_001"

Each model's inference.py must convert its native output to the project
standard COCO bbox format before evaluation. detection/evaluate.py computes
the common AP/AR metrics. LVIS-specific and native evaluator branches are not
part of the project evaluation path.

Models 6 and 15 are table-structure models. Their primary metric is GriTS, so
they are recorded for reference and excluded from the AP run.


## Environment dependency requirement

`detection/gaoyonghan/environment/` must contain a README, at least one Conda or
pip dependency manifest, and `model_environment.csv`. The CSV must map every
detection model to the exact manifest it uses. If the models require two
environments, provide two manifests and record the model assignment for both; do
not rely on an unrecorded personal environment.
