# racineai/UI-DETR-1

Run `python inference/run.py` from this model folder. The preserved native inference script and its support files are in `inference/native/`, with real weights in `weights/`. The project-level `dataset/` is referenced through this model's relative paths. Dependencies and historical commands are recorded in `requirements.txt`, `inference/environment.json`, and `provenance/`.

`evaluation/prepare.py` preserves the available format-conversion step. The historical record has no validated scoring protocol for this model, so this folder does not claim a scoring result or provide a fabricated `evaluation/score.py`. The reorganized layout has not undergone a full inference or evaluation run.
