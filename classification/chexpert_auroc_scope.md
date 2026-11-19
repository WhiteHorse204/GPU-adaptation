# CheXpert 14标签 AUROC 范围与方法

适用JinPeng/chexpert-densenet121，保留原14标签顺序和validation数据。不是现有vit-chest-xray的5标签阈值指标，也不是官方5病种AUROC。包schema=chexpert-auroc-v1；annotations.json含标签表及每图[N,14]二值GT，predictions.json含逐图sigmoid概率，protocol.json记录范围及校验值。缺失/不确定GT直接拒绝，不隐式补0或换标签顺序。

```bash
python evaluate_chexpert_auroc.py --run-dir /完整模型/result/run
# 或 --bundle /完整/scoring_bundle
```

只依赖NumPy/scikit-learn，不加载模型或原图。每标签同时有正负GT才计算AUROC；单一GT类标记 unavailable_single_gt_class。报告可用标签的AUROC均值及有效/无效标签数，无法计算的值为null；两图成功不代表14标签均有有效AUC或官方测试精度。输出scoring_bundle/metrics.json及run/evaluation_summary.json，0..1为比例。不重测forward，不将评分时间计入推理。现有evaluate.py和evaluate_multilabel.py保持不变。
