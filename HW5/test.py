import pandas as pd
from sklearn.metrics import roc_auc_score

# 读取CSV文件
data = pd.read_csv('test_label-me.csv')

# 假设'Class'列是预测结果和真实标签
true_labels = data['Class']
predicted_probs = data['Class']  # 使用相同的列作为预测值

# 计算AUC得分
auc_score = roc_auc_score(true_labels, predicted_probs)

print("AUC Score:", auc_score)
