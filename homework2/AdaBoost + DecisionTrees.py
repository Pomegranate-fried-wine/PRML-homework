import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 加载数据
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

X_train, y_train = train_df[['x', 'y', 'z']], train_df['label']
X_test, y_test = test_df[['x', 'y', 'z']], test_df['label']

# 2. 设置基础分类器
base_estimator = DecisionTreeClassifier(max_depth=5)

# 3. 初始化 AdaBoost
# n_estimators: 迭代次数
# learning_rate: 学习率
ada_model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# 4. 训练模型
ada_model.fit(X_train, y_train)

# 5. 预测
y_pred_ada = ada_model.predict(X_test)

# 6. 评估结果
print(f"AdaBoost + Decision Tree 测试集准确率: {accuracy_score(y_test, y_pred_ada):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred_ada))
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred_ada))

