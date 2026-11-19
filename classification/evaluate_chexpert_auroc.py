"""Offline CheXpert 14-label AUROC. No model, image loading, or forward timing."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
LABELS=['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def score(bundle):
 protocol=read(bundle/'protocol.json');ann=read(bundle/'annotations.json');pred=read(bundle/'predictions.json')
 if protocol.get('schema')!='chexpert-auroc-v1' or protocol.get('task')!='multi_label_classification' or protocol.get('scoring_ready') is not True:raise ValueError('Unsupported or incomplete CheXpert protocol')
 for name in ['annotations.json','predictions.json']:
  if protocol.get('artifact_sha256',{}).get(name)!=hashlib.sha256((bundle/name).read_bytes()).hexdigest():raise ValueError('Artifact checksum mismatch')
 if ann['labels']!=[dict(id=i,name=n) for i,n in enumerate(LABELS)]:raise ValueError('Expected exact CheXpert 14-label order')
 ids=[r['sample_id'] for r in ann['samples']];lookup={r['sample_id']:r for r in pred}
 if not ids or len(set(ids))!=len(ids) or len(lookup)!=len(pred) or set(ids)!=set(lookup):raise ValueError('Missing/duplicate sample IDs')
 if protocol.get('prediction_count')!=len(ids) or sorted(protocol.get('completed_sample_ids',[]))!=sorted(ids):raise ValueError('Completion counts disagree')
 truth=np.asarray([r['labels'] for r in ann['samples']]);prob=np.asarray([lookup[i]['probabilities'] for i in ids],dtype=np.float64)
 if truth.shape!=(len(ids),14) or prob.shape!=truth.shape:raise ValueError('Expected [N,14] GT and probabilities')
 if not np.isin(truth,[0,1]).all() or not np.isfinite(prob).all() or not ((prob>=0)&(prob<=1)).all():raise ValueError('Invalid binary GT or probabilities; no uncertain-label conversion')
 per=[]
 for i,n in enumerate(LABELS):
  both=len(np.unique(truth[:,i]))==2
  per.append({'id':i,'name':n,'auroc':float(roc_auc_score(truth[:,i],prob[:,i])) if both else None,'status':'measured' if both else 'unavailable_single_gt_class','positive_count':int(truth[:,i].sum()),'negative_count':int(len(ids)-truth[:,i].sum())})
 valid=[x['auroc'] for x in per if x['auroc'] is not None]
 result={'task':'multi_label_classification','protocol':'chexpert-auroc-v1','sample_count':len(ids),'num_labels':14,'metric_unit':'fraction','metrics':{'mean_auroc_over_available_labels':float(np.mean(valid)) if valid else None},'available_label_count':len(valid),'unavailable_label_count':14-len(valid),'status':'measured' if valid else 'auc_unavailable','per_label':per,'scope':'14-label AUROC, not official five-pathology mean; debug subset does not establish full-test accuracy','inference_executed':False}
 write(bundle/'metrics.json',result);return result
def main():
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--run-dir',type=Path);g.add_argument('--bundle',type=Path);a=p.parse_args()
 if a.run_dir:
  root=a.run_dir.resolve();m=read(root/'inference_output.json')
  if m.get('state')!='completed' or m.get('task')!='classification':raise ValueError('Inference incomplete')
  bundles=[(root/x).resolve() for x in m['bundles']]
  if any(not b.is_relative_to(root) for b in bundles):raise ValueError('Bundle escapes run')
 else:bundles=[a.bundle.resolve()]
 if not bundles or len(set(bundles))!=len(bundles):raise ValueError('No unique scoring bundles')
 results=[{'bundle':str(b),'metrics':score(b)} for b in bundles]
 if a.run_dir:write(root/'evaluation_summary.json',{'state':'completed','task':'multi_label_classification','results':results})
 print(json.dumps({'state':'completed','scored_bundles':len(results),'available_label_counts':[r['metrics']['available_label_count'] for r in results]}))
if __name__=='__main__':main()
