import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  1.读取xlsx文件
def load_and_prepare_data(filepath):
    try:
        df = pd.read_excel(filepath)
        X_raw = df.iloc[:, 0].values.reshape(-1, 1)
        Y = df.iloc[:, 1].values.reshape(-1, 1)
        
        # 构造增广矩阵 X_b = [x, 1] y=[x,1]*[w  
        #                                   b]
        ones = np.ones((X_raw.shape[0], 1))
        X_b = np.hstack((X_raw, ones))
        return X_raw, X_b, Y
    except Exception as e:
        print(f"读取文件 {filepath} 失败: {e}")
        return None, None, None
#mse误差作为拟合衡量结果与最小二乘法计算依据
def compute_mse(X_b, Y, theta):
    m = len(Y)
    predictions = X_b.dot(theta)
    return np.sum((predictions - Y) ** 2) / m

# 2.三种算法
#最小二乘法精确解析解
def least_squares(X_b, Y):
    # theta = (X^T * X)^-1 * X^T * Y
    return np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)

#梯度下降
def gradient_descent(X_b, Y, learning_rate=0.01, iterations=300):
    m = len(Y)
    theta = np.zeros((X_b.shape[1], 1))
    for _ in range(iterations):
        gradients = (2/m) * X_b.T.dot(X_b.dot(theta) - Y)
        theta = theta - learning_rate * gradients
    return theta

# 牛顿法找导数的零点
"""def newton_method(X_b, Y):
    m = len(Y)
    theta = np.zeros((X_b.shape[1], 1))
    gradients = (2/m) * X_b.T.dot(X_b.dot(theta) - Y)
    H = (2/m) * X_b.T.dot(X_b)
    theta = theta - np.linalg.inv(H).dot(gradients)
    return theta"""
def newton_method(X_b, Y, iterations=3):
    m = len(Y)
    # 初始点任取，比如全0
    theta = np.zeros((X_b.shape[1], 1))
    
    for i in range(iterations):
        # 1. 计算当前的 "y值" (这里的 y 是导数值，即梯度)
        g = (2/m) * X_b.T.dot(X_b.dot(theta) - Y)
        # 2. 计算当前的 "切线斜率" (这里的斜率是二阶导，即海森矩阵)
        H = (2/m) * X_b.T.dot(X_b)
        # 3. 迭代：下一点 = 当前点 - 导数值 / 二阶导数值
        # 矩阵形式下，除法变成乘以逆矩阵
        new_theta = theta - np.linalg.inv(H).dot(g)
        theta = new_theta
    return theta

# 3.输出
def print_formula(name, theta):
    w = theta[0][0]
    b = theta[1][0]
    sign = "+" if b >= 0 else "-"
    print(f"[{name}] 拟合函数: y = {w:.4f}x {sign} {abs(b):.4f}")

def main():
    X_train_raw, X_train_b, Y_train = load_and_prepare_data('training data.xlsx')
    X_test_raw, X_test_b, Y_test = load_and_prepare_data('test data.xlsx')
    
    if X_train_raw is None or X_test_raw is None: return

    # 训练
    t_ls = least_squares(X_train_b, Y_train)
    t_gd = gradient_descent(X_train_b, Y_train)
    t_nt = newton_method(X_train_b, Y_train)

    # 打印函数表达式
    print("--- 拟合结果 ---")
    print_formula("最小二乘法", t_ls)
    print_formula("梯度下降法", t_gd)
    print_formula("牛顿法    ", t_nt)
    print("-" * 30)

    # 打印误差
    methods = [("最小二乘法", t_ls), ("梯度下降法", t_gd), ("牛顿法", t_nt)]
    for name, t in methods:
        train_err = compute_mse(X_train_b, Y_train, t)
        test_err = compute_mse(X_test_b, Y_test, t)
        print(f"{name}: 训练误差={train_err:.6f}, 测试误差={test_err:.6f}")

    # 绘图
    plt.figure(figsize=(12, 6))
    x_range = np.linspace(0, 10, 100).reshape(-1, 1)
    x_range_b = np.hstack((x_range, np.ones((100, 1))))

    # 训练集图
    plt.subplot(1, 2, 1)
    plt.scatter(X_train_raw, Y_train, c='blue', alpha=0.6, label='Train Data')
    plt.plot(x_range, x_range_b.dot(t_ls), 'r-', label='LS Fit')
    plt.plot(x_range, x_range_b.dot(t_gd), 'g--', label='GD Fit')
    plt.title('Training Set Fitting')
    plt.ylim(-5, 5) # 设置 Y 轴显示范围
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    # 测试集图
    plt.subplot(1, 2, 2)
    plt.scatter(X_test_raw, Y_test, c='orange', alpha=0.6, label='Test Data')
    plt.plot(x_range, x_range_b.dot(t_ls), 'r-', label='LS Fit')
    plt.plot(x_range, x_range_b.dot(t_nt), 'y:', linewidth=3, label='Newton Fit')
    plt.title('Test Set Fitting')
    plt.ylim(-5, 5) # 设置 Y 轴显示范围
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
