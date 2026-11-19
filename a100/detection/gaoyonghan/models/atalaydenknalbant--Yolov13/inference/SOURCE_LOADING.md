# Model-local framework source

The native inference script loads `resources/runtime/assets/workspace/official_sources/object_detection/yolov13` through `native/model_source.py` before importing `ultralytics`. This location is resolved relative to the model directory, so no A100 or MX project path is hard-coded. Each model still runs through its own `inference.py`.

The loader changes only the current process and its child-process search paths. It does not install or replace a shared environment package, and it does not change inference computations. NVIDIA and MX still require their respective PyTorch/device runtime; copying this loader does not certify a full model run on either platform.
