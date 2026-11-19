"""Public native route: preserve pinned Scenic inference and expose raw predictions."""
import argparse,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);p.add_argument('--scenic-root',required=True);p.add_argument('--native-data-root',required=True);p.add_argument('--dataset-root',required=True);p.add_argument('--model-root',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--phase',choices=['smoke','full','preflight'],default='full');a=p.parse_args()
assert a.device.startswith('cuda:') and a.device[5:].isdigit()
env={k:v for k,v in os.environ.items() if not k.startswith('CODEX_TIMING_') and k not in ['PYTHONPATH','PYTHONHOME','JAX_PLATFORMS']}
env.update(CUDA_VISIBLE_DEVICES=a.device[5:],XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
env['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
env['LD_LIBRARY_PATH'] = str(Path(sys.prefix) / 'lib/system-runtime') + ':' + env.get('LD_LIBRARY_PATH', '')
command=[sys.executable,str(Path(__file__).with_name('scenic_predict.py')),'--output-root',a.output_root,'--native-data-root',a.native_data_root,'--model-root',a.model_root,'--phase',a.phase]
subprocess.run(command,env=env,check=True)
