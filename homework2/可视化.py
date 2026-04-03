import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from skimage import measure  # 用于计算 3D 等值面

# 1. 加载数据 (确保工作目录下有这两个文件)
try:
    train_df = pd.read_csv("train_data.csv")
    test_df = pd.read_csv("test_data.csv")
    X_train = train_df[['x', 'y', 'z']]
    y_train = train_df['label']
    X_test = test_df[['x', 'y', 'z']]
    y_test = test_df['label']
except FileNotFoundError:
    print("错误：未找到 CSV 文件，请先生成 train_data.csv 和 test_data.csv")
    exit()

# 2. 训练三个具有代表性的模型
print("正在训练模型中...")
models = {
    "Decision Tree (Depth 5)": DecisionTreeClassifier(max_depth=5, random_state=42),
    "AdaBoost (100 Trees)": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=5), 
        n_estimators=100, 
        learning_rate=0.1, 
        random_state=42
    ),
    "SVM (RBF Kernel)": SVC(kernel='rbf', gamma='auto', probability=True)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} 训练完成。")

# 3. 定义 3D 决策曲面绘制函数
def plot_3d_decision_surfaces(models, X, y, resolution=40):
    fig = plt.figure(figsize=(20, 7))
    
    # 确定空间范围
    x_min, x_max = X['x'].min() - 0.5, X['x'].max() + 0.5
    y_min, y_max = X['y'].min() - 0.5, X['y'].max() + 0.5
    z_min, z_max = X['z'].min() - 0.5, X['z'].max() + 0.5
    
    # 生成 3D 网格
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    z_grid = np.linspace(z_min, z_max, resolution)
    xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    for i, (name, model) in enumerate(models.items()):
        ax = fig.add_subplot(131 + i, projection='3d')
        
        # 预测整个网格的分类结果 (0 或 1)
        # 对于决策边界，我们寻找预测概率为 0.5 的地方
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(grid_points)[:, 1]
        else:
            probs = model.predict(grid_points)
        
        vol = probs.reshape(xx.shape)
        
        # 使用 Marching Cubes 提取 0.5 分类边界 (等值面)
        try:
            # 这里的 level=0.5 即决策边界
            verts, faces, normals, values = measure.marching_cubes(vol, level=0.5)
            
            # 缩放顶点到原始坐标空间
            verts[:, 0] = verts[:, 0] * (x_max - x_min) / (resolution - 1) + x_min
            verts[:, 1] = verts[:, 1] * (y_max - y_min) / (resolution - 1) + y_min
            verts[:, 2] = verts[:, 2] * (z_max - z_min) / (resolution - 1) + z_min
            
            # 绘制提取出的曲面
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                            color='gray', alpha=0.3, lw=1)
        except Exception as e:
            print(f"无法为 {name} 生成曲面: {e}")

        # 叠加原始散点（降低透明度以便观察内部曲面）
        ax.scatter(X['x'], X['y'], X['z'], c=y, cmap='coolwarm', s=10, alpha=0.2)
        
        ax.set_title(name)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=45)

    plt.suptitle("Comparison of 3D Decision Surfaces (Boundary at P=0.5)", fontsize=16)
    plt.tight_layout()
    plt.show()

# 执行可视化
plot_3d_decision_surfaces(models, X_test, y_test)