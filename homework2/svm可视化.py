import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from skimage import measure  # 用于计算 3D 等值面
import warnings

# 忽略 marching_cubes 的警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 1. 加载数据
try:
    train_df = pd.read_csv("train_data.csv")
    test_df = pd.read_csv("test_data.csv")
    X_train, y_train = train_df[['x', 'y', 'z']], train_df['label']
    X_test, y_test = test_df[['x', 'y', 'z']], test_df['label']
except FileNotFoundError:
    print("错误：请确保当前目录下有 train_data.csv 和 test_data.csv")
    exit()

# 2. 定义并训练模型
# 重要：必须设置 probability=True 才能绘制平滑的决策曲面
kernels = {
    "SVM (Linear)": SVC(kernel='linear', probability=True),
    "SVM (Polynomial)": SVC(kernel='poly', degree=3, probability=True),
    "SVM (RBF)": SVC(kernel='rbf', gamma='scale', probability=True)
}

print(f"{'Kernel Type':<20} | {'Accuracy':<10}")
print("-" * 35)

for name, model in kernels.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name:<20} | {acc:.4f}")

# 3. 定义 3D 决策曲面绘制函数
def plot_svm_decision_surfaces_3d(models_dict, X, y, resolution=40):
    fig = plt.figure(figsize=(20, 7))
    
    x_min, x_max = X['x'].min() - 0.5, X['x'].max() + 0.5
    y_min, y_max = X['y'].min() - 0.5, X['y'].max() + 0.5
    z_min, z_max = X['z'].min() - 0.5, X['z'].max() + 0.5
    
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    z_grid = np.linspace(z_min, z_max, resolution)
    xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    for i, (name, model) in enumerate(models_dict.items()):
        ax = fig.add_subplot(131 + i, projection='3d')
        
        # 计算概率场
        probs = model.predict_proba(grid_points)[:, 1]
        vol = probs.reshape(xx.shape)
        
        try:
            # 提取 P=0.5 的等值面
            verts, faces, normals, values = measure.marching_cubes(vol, level=0.5)
            
            # 映射回原始坐标系
            verts[:, 0] = verts[:, 0] * (x_max - x_min) / (resolution - 1) + x_min
            verts[:, 1] = verts[:, 1] * (y_max - y_min) / (resolution - 1) + y_min
            verts[:, 2] = verts[:, 2] * (z_max - z_min) / (resolution - 1) + z_min
            
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                            color='gray', alpha=0.3, lw=0)
            ax.set_title(name)
        except Exception as e:
            ax.set_title(f"{name} (No Surface)")

        ax.scatter(X['x'], X['y'], X['z'], c=y, cmap='coolwarm', s=10, alpha=0.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=45)

    plt.suptitle("SVM Decision Hyperplanes Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()

# 4. 执行可视化 (直接传入训练好的 kernels 字典)
plot_svm_decision_surfaces_3d(kernels, X_test, y_test, resolution=45)