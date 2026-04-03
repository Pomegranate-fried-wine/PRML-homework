import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. 加载数据
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

X_train, y_train = train_df[['x', 'y', 'z']], train_df['label']
X_test, y_test = test_df[['x', 'y', 'z']], test_df['label']

# 2. 定义三种不同的核函数模型
kernels = {
    "SVM (Linear)": SVC(kernel='linear'),
    "SVM (Polynomial)": SVC(kernel='poly', degree=3), # 3次多项式
    "SVM (RBF)": SVC(kernel='rbf', gamma='scale')    # 默认 RBF
}

print(f"{'Kernel Type':<20} | {'Accuracy':<10}")
print("-" * 35)

# 3. 循环训练并评估
for name, model in kernels.items():
    # 训练
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)
    # 评估
    acc = accuracy_score(y_test, y_pred)
    print(f"{name:<20} | {acc:.4f}")

    # 具体的分类报告：
    print(f"\n--- {name} Classification Report ---")
    print(classification_report(y_test, y_pred))

