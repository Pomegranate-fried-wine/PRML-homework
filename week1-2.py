import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 复合拟合函数: y = k*x + a*sin(b*x + c) + d
class SineFitter:
    def __init__(self):
        # 线性斜率k, 振幅a, 频率b, 相位c, 偏置d
        self.k, self.a, self.b, self.c, self.d = [None] * 5

    def compute_mse(self, X, Y):
        """计算均方误差 MSE"""
        y_pred = self.predict(X)
        return np.mean((y_pred - Y)**2)

    def smart_init(self, X, Y):
        
        # 1. 估算全局斜率 k (取末尾均值减去开头均值)
        self.k = (np.mean(Y[-10:]) - np.mean(Y[:10])) / (np.max(X) - np.min(X))
        
        # 2. 去除线性趋势后估算 d 和 a
        Y_detrended = Y - self.k * X
        self.d = np.mean(Y_detrended)
        self.a = (np.max(Y_detrended) - np.min(Y_detrended)) / 2
        
        # 3. FFT 识别主频率 b
        n = len(X)
        y_fft = np.fft.fft(Y_detrended.flatten() - self.d)
        sample_dist = (X[1]-X[0])[0] if len(X)>1 else 1
        freqs = np.fft.fftfreq(n, d=sample_dist)
        idx = np.argmax(np.abs(y_fft[1:n//2])) + 1
        self.b = 2 * np.pi * abs(freqs[idx])
        
        # 4. 初始相位 c
        self.c = 0.0
        
        print("--- 初始参数估算 ---")
        print(f"k={self.k:.4f}, a={self.a:.4f}, b={self.b:.4f}, c={self.c:.4f}, d={self.d:.4f}")

    def fit(self, X, Y, lr=0.001, epochs=100000):
        self.smart_init(X, Y)
        m = len(X)
        
        print("\n--- 开始梯度下降优化 ---")
        for i in range(epochs):
            # 1. 前向传播
            arg = self.b * X + self.c
            sin_val = np.sin(arg)
            cos_val = np.cos(arg) # 这就是链式法则需要的导数部分
            
            y_pred = self.k * X + self.a * sin_val + self.d
            error = y_pred - Y
            
            # 2. 计算梯度 (MSE对各参数的偏导数)
            dk = (2/m) * np.sum(error * X)
            da = (2/m) * np.sum(error * sin_val)
            db = (2/m) * np.sum(error * self.a * cos_val * X) # b在sin内部，且乘了x
            dc = (2/m) * np.sum(error * self.a * cos_val)     # c在sin内部
            dd = (2/m) * np.sum(error)
            
            # 3. 更新参数
            self.k -= lr * dk
            self.a -= lr * da
            self.b -= lr * db
            self.c -= lr * dc
            self.d -= lr * dd
            
            # 每 20000 次打印一次 MSE 和学习率衰减
            if i % 20000 == 0:
                current_mse = np.mean(error**2)
                print(f"迭代 {i:6d} | MSE: {current_mse:.6f}")
                lr *= 0.7 

    def predict(self, X):
        return self.k * X + self.a * np.sin(self.b * X + self.c) + self.d

# ==================== 主程序逻辑 ====================
def main():
    # 1. 加载数据
    try:
        train_df = pd.read_excel('training data.xlsx')
        test_df = pd.read_excel('test data.xlsx')
        X_train = train_df.iloc[:, 0].values.reshape(-1, 1)
        Y_train = train_df.iloc[:, 1].values.reshape(-1, 1)
        X_test = test_df.iloc[:, 0].values.reshape(-1, 1)
        Y_test = test_df.iloc[:, 1].values.reshape(-1, 1)
    except Exception as e:
        print(f"文件加载失败: {e}"); return

    # 2. 实例化并训练
    model = SineFitter()
    model.fit(X_train, Y_train, lr=0.001, epochs=50000)

    # 3. 计算最终 MSE
    train_mse = model.compute_mse(X_train, Y_train)
    test_mse = model.compute_mse(X_test, Y_test)

    # 4. 打印结果报告
    print("\n" + "="*50)
    print(f"最终方程: y = {model.k:.4f}x + {model.a:.4f}*sin({model.b:.4f}x + {model.c:.4f}) + {model.d:.4f}")
    print(f"训练集 MSE: {train_mse:.6f}")
    print(f"测试集 MSE: {test_mse:.6f}")
    print("="*50)

    # 5. 可视化
    plt.figure(figsize=(15, 6))
    x_smooth = np.linspace(X_train.min(), X_train.max(), 1000).reshape(-1, 1)
    y_smooth = model.predict(x_smooth)

    # 子图1: 训练集
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, Y_train, color='blue', alpha=0.4, label='Training Data')
    plt.plot(x_smooth, y_smooth, 'r-', lw=2.5, label=f'Trend+Sine Fit\nMSE={train_mse:.4f}')
    plt.title('Training Data Fitting')
    plt.ylim(-5, 5) # 设置 Y 轴显示范围
    plt.grid(True, ls=':', alpha=0.6); plt.legend()

    # 子图2: 测试集
    plt.subplot(1, 2, 2)
    plt.scatter(X_test, Y_test, color='orange', alpha=0.4, label='Test Data')
    plt.plot(x_smooth, y_smooth, 'r-', lw=2.5, label=f'Trend+Sine Fit\nMSE={test_mse:.4f}')
    plt.title('Test Data Generalization')
    plt.ylim(-5, 5) # 设置 Y 轴显示范围
    plt.grid(True, ls=':', alpha=0.6); plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()