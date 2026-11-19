# Classification project requirements

```text
model inference for GPU adaptation/
├── dataset/                        # Existing shared datasets
└── classification/
    ├── README.md
    ├── evaluate.py                 # Shared classification evaluator
    └── gaoyonghan/
        ├── environment/            # Dependency manifests and model mapping
        └── models/
            └── <model_name>/
                ├── inference.py   # Required public inference entry point
                ├── weights/       # Required model weights
                ├── result/        # Required inference and evaluation outputs
                ├── README.md
                ├── inference/     # Model-specific implementation, if needed
                └── ...            # Other model-specific files are unrestricted
```

Other contributors use `classification/<owner>/models/<model_name>/`.
The classification evaluator is shared at the task level. Existing datasets
remain in the original project-level `dataset/` directory and are shared with
other model workflows.

## Required contract

1. Every model directory contains `inference.py`, `weights/`, and `result/`.
2. Running `python inference.py` performs the
   complete model inference workflow, including output conversion. Internal
   scripts, frameworks, and implementation structure remain model-specific.
3. The default run directory is `result/run_001/`. Its `inference_output.json`
   records `state: completed`, `task: classification`, and relative bundle paths.
   Each bundle contains `predictions.json`, `annotations.json`, and `protocol.json`
   in the existing `model-eval/score-classification/v1` format.
4. `classification/evaluate.py` contains the complete common evaluation code
   and evaluates those bundles. Model directories do not contain their own
   evaluation entry points.
   Metrics are saved with the run.
5. Resolve default paths against the script location. A model must work when
   invoked from another working directory. Keep weights inside the model and
   datasets in the original project-level `dataset/` directory. Do not require symbolic links or
   machine-specific absolute paths.
6. Document any mandatory model-specific options in the model README.
   Environment consolidation and deployment are a separate follow-up.
7. Preserve verified preprocessing, model execution, labels, splits, and scoring
   rules. Directory reorganization is not evidence of a new inference pass.
8. Previously skipped models remain explicitly marked as skipped. A structural
   placeholder is not a verified runnable model.

## Run an existing model

From this classification directory:

```bash
python gaoyonghan/models/Ahmed9275--Vit-Cifar100/inference.py
python evaluate.py --run-dir gaoyonghan/models/Ahmed9275--Vit-Cifar100/result/run_001
```

Inference options and native arguments are documented in each model. For the
existing models, `--show-command` prints the preserved command without running
inference. `--output-root result/run_002` selects another model-relative run
directory. `--prepare-only` converts a completed native run without inference.
The evaluator resolves relative `--run-dir` and `--bundle` arguments against
this classification directory. Absolute paths are also accepted.

The public entry name enables later batching; no central model dispatcher is
required. Historical results are retained under each model's `result/` folder.


## Environment dependency requirement

`classification/gaoyonghan/environment/` must contain a README, at least one
Conda or pip dependency manifest, and `model_environment.csv`. The CSV must map
every classification model to the exact manifest it uses. If the models require
two environments, provide two manifests and record the model assignment for both;
do not rely on an unrecorded personal environment.
