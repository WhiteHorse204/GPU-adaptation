# dima806/closed_eyes_image_detection

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
python evaluate.py --run-dir gaoyonghan/models/dima806--closed_eyes_image_detection/result/run_001
```

Weights belong in `weights/`; the dataset location is
`../../../../dataset/`, relative to this model. Historical outputs are preserved
under `result/`. Dependencies, provenance, and historical qualification records
remain available. This organization change does not alter model computation or
claim a new full inference pass.
