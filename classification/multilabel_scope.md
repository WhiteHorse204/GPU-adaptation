# 多标签评分使用情况说明——范围界定
当前aoss/model inference for GPU adaptation/classification/evaluate_multilabel.py，仅仅使用在vit-chest-xray模型（从属zehong），由于该任务固定5个输出标签，sigmoid(logits)>0.5逐标签判定；不能用softmax/argmax。标签顺序、GT空值补0及遇-1拒绝沿用原项目。新增任务根evaluate_multilabel.py为独立CPU脚本，仅依赖NumPy；


# 使用方法
python /任务根/evaluate_multilabel.py --run-dir /完整模型/result/run
或 --bundle /完整/scoring_bundle。包包含annotations.json（标签表与逐图0/1向量GT）、predictions.json（逐图0/1预测及概率）、protocol.json（模型、数据、阈值）。结果为包metrics.json，使用run-dir还会生成evaluation_summary.json。不需要模型、原图或显卡。

# 指标定义
label_accuracy=正确标签判定数/(图数×标签数)；subset_accuracy=整组标签完全正确图数/图数；micro_F1=2TP/(2TP+FP+FN)，先跨标签汇总计数；macro_F1为5个标签各自F1的平均。分母0的F1按0处理，所以仅有某一个正标签的两图样例即使全部预测正确，macro_F1也可能是0.2，不能误解为整体准确率20%。这些值为0到1的比例，不与单标签Top1直接混用。
