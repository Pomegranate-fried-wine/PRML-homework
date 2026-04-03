# Generating 3D make-moons data

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def make_moons_3d(n_samples=500, noise=0.1):
    # Generate the original 2D make_moons data
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # Adding a sinusoidal variation in the third dimension

    # Concatenating the positive and negative moons with an offset and noise
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # Adding Gaussian noise
    X += np.random.normal(scale=noise, size=X.shape)

    return X, y

# Generate the data (1000 datapoints)
X, labels = make_moons_3d(n_samples=500, noise=0.2)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', marker='o')
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Make Moons')
plt.show()

import pandas as pd

def save_to_csv(X, y, filename="test_data.csv"):
    """
    将生成的 3D 数据和标签保存为 CSV 文件
    """
    # 创建 DataFrame，包含坐标 (x, y, z) 和分类标签 (label)
    df = pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1],
        'z': X[:, 2],
        'label': y.astype(int)
    })
    
    # 输出为 CSV 文件，不保留索引
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"成功！已生成包含 {len(df)} 条数据的文件: {filename}")

# 执行保存
save_to_csv(X, labels)