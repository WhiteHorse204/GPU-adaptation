"""Permit the recorded existing A100 environment only for bounded smoke runs."""
import importlib.metadata,json,sys
from pathlib import Path

def allow(actual=None):
    args=sys.argv[1:]
    bounded=False
    for flag in ['--max-images','--phase']:
        if flag in args and args.index(flag)+1<len(args):
            value=args[args.index(flag)+1]
            bounded |= (value=='smoke') if flag=='--phase' else (value.isdigit() and 0<int(value)<=10)
    if not bounded:return False
    root=Path(__file__).resolve().parents[4]
    path=root/'environment/dual_smoke_20261116/existing_environment.json'
    if not path.exists():raise RuntimeError('Missing captured A100 environment: '+str(path))
    recorded=json.loads(path.read_text())
    if sys.executable!=recorded['executable'] or sys.version!=recorded['python']:return False
    for name,version in (actual or {}).items():
        if recorded['packages'].get(name.lower().replace('_','-'))!=version:return False
    print('WARNING: bounded smoke using recorded existing A100 environment; original reference versions differ: '+json.dumps(actual,ensure_ascii=False),flush=True)
    return True
