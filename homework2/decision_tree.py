import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 加载 CSV 数据
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

# 2. 提取特征 (X, Y, Z) 和 标签 (label)
X_train = train_df[['x', 'y', 'z']]
y_train = train_df['label']

X_test = test_df[['x', 'y', 'z']]
y_test = test_df['label']

# 3. 初始化决策树模型
# max_depth 是一个重要参数
dt_model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)

# 4. 训练模型
dt_model.fit(X_train, y_train)

# 5. 在测试集上进行预测
y_pred = dt_model.predict(X_test)

# 6. 输出评估
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree 测试集准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 7. 查看混淆矩阵
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# --- 1. 快速准备数据和模型 (确保代码可独立运行) ---
try:
    train_df = pd.read_csv("train_data.csv")
    X_train = train_df[['x', 'y', 'z']]
    y_train = train_df['label']
except FileNotFoundError:
    print("请确保 train_data.csv 文件存在。")
    exit()

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42) # 深度设为4，图面更清晰
dt_model.fit(X_train, y_train)

# --- 2. 核心可视化函数：绘制 3D 盒子 ---
def plot_3d_decision_boxes(model, X_data, y_data):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 获取数据边界
    x_min, x_max = X_data['x'].min(), X_data['x'].max()
    y_min, y_max = X_data['y'].min(), X_data['y'].max()
    z_min, z_max = X_data['z'].min(), X_data['z'].max()

    # 2. 提取决策树的叶子节点区域
    # 这需要一些技巧来解析 tree_ 对象
    tree = model.tree_
    node_count = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value

    # 用于存储每个节点的边界 [xmin, xmax, ymin, ymax, zmin, zmax]
    node_bounds = np.zeros((node_count, 6))
    node_bounds[0] = [x_min, x_max, y_min, y_max, z_min, z_max] # 根节点覆盖全空间

    # 递归计算所有节点的边界
    def get_bounds(node_id):
        if children_left[node_id] != -1: # 不是叶子节点
            feat = feature[node_id]
            thresh = thresholds = threshold[node_id]
            
            # 继承父节点边界
            node_bounds[children_left[node_id]] = node_bounds[node_id].copy()
            node_bounds[children_right[node_id]] = node_bounds[node_id].copy()
            
            # 根据切分修正边界
            node_bounds[children_left[node_id], feat * 2 + 1] = thresh # 修正上限
            node_bounds[children_right[node_id], feat * 2] = thresh    # 修正下限
            
            get_bounds(children_left[node_id])
            get_bounds(children_right[node_id])

    get_bounds(0)

    # 3. 绘制叶子节点的盒子
    leaf_nodes = np.where(children_left == -1)[0]
    
    for node_id in leaf_nodes:
        # 获取该盒子的预测类别
        box_class = np.argmax(value[node_id])
        color = 'red' if box_class == 0 else 'blue'
        
        # 获取盒子边界
        b = node_bounds[node_id]
        
        # 定义 8 个顶点
        vertices = np.array([
            [b[0], b[2], b[4]], [b[1], b[2], b[4]], [b[1], b[3], b[4]], [b[0], b[3], b[4]], # 底面
            [b[0], b[2], b[5]], [b[1], b[2], b[5]], [b[1], b[3], b[5]], [b[0], b[3], b[5]]  # 顶面
        ])
        
        # 定义 6 个面
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]], # 底
            [vertices[4], vertices[5], vertices[6], vertices[7]], # 顶
            [vertices[0], vertices[1], vertices[5], vertices[4]], # 前
            [vertices[2], vertices[3], vertices[7], vertices[6]], # 后
            [vertices[0], vertices[3], vertices[7], vertices[4]], # 左
            [vertices[1], vertices[2], vertices[6], vertices[5]]  # 右
        ]
        
        # 绘制半透明盒子
        poly3d = Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='k', alpha=0.15)
        ax.add_collection3d(poly3d)

    # 4. 叠加少量原始散点以做对比 (可选，设得很透)
    ax.scatter(X_data['x'], X_data['y'], X_data['z'], c=y_data, cmap='coolwarm', s=10, alpha=0.3, edgecolors='none')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    plt.title(f"3D Decision Boundary: Tree Leaves as Boxes (Depth={model.max_depth})")
    
    # 视角调整
    ax.view_init(elev=20, azim=45)
    plt.show()

# --- 3. 执行可视化 ---
# 为了清晰，我把 max_depth 设为了 4，你可以尝试设回 5
plot_3d_decision_boxes(dt_model, X_train, y_train)