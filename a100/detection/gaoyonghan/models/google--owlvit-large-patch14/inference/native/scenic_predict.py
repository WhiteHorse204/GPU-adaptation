"""Use installed pinned Scenic prediction functions; publish completed image coverage."""
import argparse,hashlib,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);p.add_argument('--native-data-root',required=True);p.add_argument('--model-root',required=True);p.add_argument('--phase',choices=['smoke','full','preflight'],required=True);a=p.parse_args()
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false';os.environ['FLAX_USE_ORBAX_CHECKPOINTING']='0'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU');tf.config.threading.set_inter_op_parallelism_threads(2);tf.config.threading.set_intra_op_parallelism_threads(2)
import jax
import scenic
from scenic.projects.owl_vit import evaluator
from scenic.projects.owl_vit.clip import tokenizer
from scenic.projects.owl_vit.configs import clip_l14
from scenic.projects.owl_vit.preprocessing import modalities
import clip
import numpy as np
D=Path(a.native_data_root);O=Path(a.output_root);O.mkdir(parents=True,exist_ok=True)
report=json.loads((D/'checkpoint_conversion_report.json').read_text())
assert report['status']=='verified' and report['roundtrip_exact'] is True and report['roundtrip_max_abs_error']==0
checkpoint=Path(a.model_root)/'scenic/checkpoint_0';assert checkpoint.is_file()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(4194304):h.update(b)
 return h.hexdigest()
weight=Path(a.model_root)/'google/owlvit-large-patch14/pytorch_model.bin'
assert sha(weight)==report['source_weight_sha256']
bpe=Path(clip.__file__).parent/'bpe_simple_vocab_16e6.txt.gz';assert sha(bpe)=='924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a'
assert any(x.platform=='gpu' for x in jax.devices()),str(jax.devices())
fn=tokenizer.build_tokenizer.__wrapped__;fn.__defaults__=(str(bpe),*fn.__defaults__[1:])
split='validation[:10]' if a.phase=='smoke' else 'validation'
evaluator.FLAGS(['scenic-project-inference','--config=clip_l14','--checkpoint_path='+str(checkpoint.parent),'--output_dir='+str(O),'--tfds_data_dir='+str(D/'tfds'),'--tfds_download_dir='+str(D/'downloads'),'--data_format=lvis','--platform=gpu'])
if a.phase=='preflight':print('Scenic environment, local weights and tokenizer verified');raise SystemExit(0)
completed=[];original_dataset=evaluator.get_dataset
class RecordedDataset:
 def __init__(self,dataset):self.dataset=dataset
 def cardinality(self):return self.dataset.cardinality()
 def as_numpy_iterator(self):
  for batch in self.dataset.as_numpy_iterator():
   ids=[int(v) for v in np.asarray(batch[modalities.IMAGE_ID]).reshape(-1)]
   yield batch
   completed.extend(ids)
   (O/'coverage_progress.json').write_text(json.dumps({'completed_image_ids':completed}))
def get_dataset(*args,**kwargs):
 ds,names=original_dataset(*args,**kwargs);return RecordedDataset(ds),names
evaluator.get_dataset=get_dataset
(O/'status.json').write_text(json.dumps({'state':'running','phase':a.phase,'split':split,'scenic_module':scenic.__file__}))
predictions=evaluator.get_predictions(config=clip_l14.get_config(),checkpoint_path=str(checkpoint.parent),tfds_name='lvis',split=split,label_shift=1)
assert completed and len(completed)==len(set(completed))
assert len(completed)==(10 if a.phase=='smoke' else 19809)
assert {x['image_id'] for x in predictions}<=set(completed)
(O/'predictions_validation.json').write_text(json.dumps(predictions))
(O/'expected_ids.json').write_text(json.dumps(completed))
(O/'status.json').write_text(json.dumps({'state':'partial_complete' if a.phase=='smoke' else 'complete','phase':a.phase,'completed_images':len(completed),'total_images':19809,'image_ids':completed,'source':'pinned Scenic 08ee9dc4 original get_predictions','checkpoint_sha256':sha(checkpoint),'scenic_module':scenic.__file__,'scoring':'Project COCO bbox AP in task evaluate.py'}))
print('Scenic predictions completed:',len(completed),len(predictions))
