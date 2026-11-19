# viddexa/nsfw-detection-nano

This model was historically skipped and has no verified runnable workflow.
Reason: Historical record: skipped. No model weights or verified inference entry were available in the source project. This folder preserves the provenance and skipped status; it is not presented as runnable or scored.

From this model directory:

```bash
python inference.py
```

The public entry invokes this model's preserved native workflow and converts its
outputs automatically. The default output is `result/run_001/`, including
`inference_output.json` and the standardized scoring bundles. Each bundle
contains `predictions.json`, `annotations.json`, and `protocol.json`.
`--output-root result/run_002` selects another model-relative run directory.
`--show-command` prints the command without inference. `--prepare-only` converts
an existing completed native run. Native arguments remain model-specific;
the recorded commands are in `inference/entrypoint.json` and `provenance/`.

From the classification directory, evaluate this output:

```bash
python evaluate.py --run-dir gaoyonghan/models/viddexa--nsfw-detection-nano/result/run_001
```

Weights belong in `weights/`; the dataset location is
`../../../../dataset/`, relative to this model. Historical outputs are preserved
under `result/`. Dependencies, provenance, and historical qualification records
remain available. This organization change does not alter model computation or
claim a new full inference pass.
