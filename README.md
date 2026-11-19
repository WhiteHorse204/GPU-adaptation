# model inference for GPU adaptation

Gaoyonghan's existing model folders use the same public structure for classification
(59 folders) and detection (70 folders):

```text
model inference for GPU adaptation/
├── dataset/                        # Original shared datasets
├── classification/
│   ├── evaluate.py
│   └── gaoyonghan/
│       ├── environment/              # Dependency manifests and model mapping
│       └── models/<model_name>/
│       ├── inference.py
│       ├── weights/
│       ├── result/
│       ├── inference/
│       └── README.md
├── detection/
│   ├── evaluate.py
│   └── gaoyonghan/
│       ├── environment/              # Dependency manifests and model mapping
│       └── models/<model_name>/
│       ├── inference.py
│       ├── weights/
│       ├── result/
│       ├── inference/
│       └── README.md
└── model/                          # Other existing contributors and tasks
```

Each model's `inference.py` calls its preserved model-specific workflow and converts
native output into the existing standardized format. Classification uses `classification/evaluate.py` and detection uses
`detection/evaluate.py`. Detection evaluation uses the project-standard COCO
bbox AP protocol; LVIS-specific evaluation is not a project entry point.
See each task README for required entry points and output contracts.

The original `dataset/` remains shared. Each model references it using the relative
path `../../../../dataset/`; there are no duplicated task datasets or symbolic
links. Existing weights, results, qualification records and other contributors'
files are preserved. `模型清单.json` and `模型清单.csv` record the updated locations.
Skipped and prediction-only workflows retain their historical status.

This change organizes files, entry points and paths. It does not claim a new full
inference pass. Environment consolidation and installation are separate work;
no environment directory is introduced by this layout change.

## Task entry points

Each task has one shared evaluation entry point with the same filename:

classification/evaluate.py --run-dir "/mnt/aoss/model inference for GPU adaptation/classification/<contributor>/models/<model-name>/result/run_001"
detection/evaluate.py --run-dir "/mnt/aoss/model inference for GPU adaptation/detection/<contributor>/models/<model-name>/result/run_001"

Models must produce their task-standard output from their own inference.py.
Classification and detection use separate evaluators because their prediction
schemas and metrics differ. Detection models 6 and 15 are recorded as GriTS-only
models and are excluded from the AP batch. No model-level `evaluation/` or
`vendor/` directory is required.


## Environment dependency requirement

Each contributor directory must contain an `environment/` directory with a README,
at least one dependency manifest (`.yml` or `.txt`), and `model_environment.csv`.
The mapping records the exact environment used by every model. If two or more
environments are needed, each manifest and model-to-environment assignment must
be listed explicitly.
