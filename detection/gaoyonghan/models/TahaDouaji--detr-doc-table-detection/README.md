# TahaDouaji/detr-doc-table-detection

From this model directory:

```bash
python inference.py
```

The public entry calls the existing model-specific inference workflow and converts
its outputs automatically. Default outputs are saved in `result/run_001/`, with
`inference_output.json` recording the completed outputs and scoring bundles.
`--output-root result/run_002` chooses another model-relative run directory.
`--show-command` prints the preserved native command without executing it.
`--prepare-only` converts an already completed native run without model inference.
Native options and required weight variants remain those recorded in
`inference/entrypoint.json` and the historical `provenance/` files.

Run the common detector evaluator from the detection directory:

```bash
python evaluate.py --run-dir gaoyonghan/models/TahaDouaji--detr-doc-table-detection/result/run_001
```

Weights remain under `weights/`; shared datasets remain at `../../../../dataset/`
relative to this model directory. Existing results are retained under `result/`.
This reorganization preserves model computation, sample selection and scoring
rules. Environment consolidation is handled separately; no new full inference
pass is implied by this layout.
