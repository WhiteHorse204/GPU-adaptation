# microsoft/table-transformer-detection

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
python evaluate.py --run-dir gaoyonghan/models/microsoft--table-transformer-detection/result/run_001
```

Weights remain under `weights/`; shared datasets remain at `../../../../dataset/`
relative to this model directory. Existing results are retained under `result/`.
This reorganization preserves model computation, sample selection and scoring
rules. Environment consolidation is handled separately; no new full inference
pass is implied by this layout.

## A100 两步小样本测试

环境：e26（/opt/model-envs/e26）。已验证原固定前 10 张。

```bash
source /opt/model-envs/e26/bin/activate
python inference.py --phase smoke --output-root result/smoke_001
python ../../../evaluate.py --run-dir "/mnt/aoss/model inference for GPU adaptation/detection/gaoyonghan/models/microsoft--table-transformer-detection/result/smoke_001"
```

全量使用 --phase full；这次通过证据是小样本两步测试，不代表完成全量精度验收。
