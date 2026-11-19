"""独立CPU多标签评分器，只依赖NumPy；不加载模型、不做前向、不读取原图。"""
import argparse
import json
from pathlib import Path
import numpy as np


def score(bundle):
    bundle = Path(bundle)
    annotation = json.loads((bundle / 'annotations.json').read_text())
    predictions = json.loads((bundle / 'predictions.json').read_text())
    protocol = json.loads((bundle / 'protocol.json').read_text())
    if protocol.get('schema') != 'multilabel-v1' or protocol.get('task') != 'multi_label_classification':
        raise ValueError('不是受支持的多标签评分包')
    samples, labels = annotation['samples'], annotation['labels']
    ids = [r['sample_id'] for r in samples]
    lookup = {r['sample_id']: r for r in predictions}
    if not ids or len(set(ids)) != len(ids) or len(lookup) != len(predictions) or set(ids) != set(lookup):
        raise ValueError('GT/预测ID必须唯一且完整对齐')
    if [r['id'] for r in labels] != list(range(len(labels))):
        raise ValueError('标签表必须按输出列编号排列')
    truth = np.asarray([r['labels'] for r in samples])
    pred = np.asarray([lookup[sid]['prediction'] for sid in ids])
    if truth.shape != (len(ids), len(labels)) or pred.shape != truth.shape or not len(labels):
        raise ValueError('GT/预测应为同样的[N,L]矩阵')
    if not np.isin(truth, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError('GT/预测必须为0或1，不接受未处理的不确定标签')
    truth, pred = truth.astype(bool), pred.astype(bool)
    equal = truth == pred
    tp = (truth & pred).sum(0); fp = (~truth & pred).sum(0); fn = (truth & ~pred).sum(0)
    def ratio(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    precision = ratio(tp, tp + fp); recall = ratio(tp, tp + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    result = {'task': 'multi_label_classification', 'sample_count': len(ids), 'num_labels': len(labels),
              'correct_label_count': int(equal.sum()), 'total_label_count': int(equal.size),
              'metrics': {'label_accuracy': float(equal.mean()),
                          'subset_accuracy': float(equal.all(1).mean()),
                          'micro_f1': float(ratio(2 * tp.sum(), 2 * tp.sum() + fp.sum() + fn.sum())),
                          'macro_f1': float(f1.mean())},
              'per_label': [{'id': x['id'], 'name': x['name'], 'positive_support': int(truth[:, i].sum()),
                             'accuracy': float(equal[:, i].mean()), 'precision': float(precision[i]),
                             'recall': float(recall[i]), 'f1': float(f1[i])} for i, x in enumerate(labels)]}
    (bundle / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    choice=p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--bundle',type=Path)
    choice.add_argument('--run-dir',type=Path)
    a=p.parse_args()
    if a.bundle:
        bundles=[a.bundle.resolve()]
    else:
        root=a.run_dir.resolve(); manifest=json.loads((root/'inference_output.json').read_text())
        if manifest.get('state')!='completed' or manifest.get('task')!='classification':raise ValueError('Inference not completed')
        bundles=[(root/x).resolve() for x in manifest['bundles']]
        if any(not b.is_relative_to(root) for b in bundles):raise ValueError('Invalid bundle path')
    if not bundles:raise ValueError('No scoring bundles')
    results=[{'bundle':str(b),'metrics':score(b)} for b in bundles]
    if a.run_dir:(a.run_dir/'evaluation_summary.json').write_text(json.dumps({'state':'completed','task':'multi_label_classification','results':results},ensure_ascii=False,indent=2))
    print(json.dumps({'state':'completed','scored_bundles':len(results),'task':'multi_label_classification'}))
