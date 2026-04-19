import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 1. 读取数据
# =========================
data_path = 'archive/LSTM-Multivariate_pollution.csv'
dataset = pd.read_csv(data_path)

dataset['date'] = pd.to_datetime(dataset['date'])
dataset.set_index('date', inplace=True)

# =========================
# 2. 类别变量处理
# =========================
encoder = LabelEncoder()
dataset['wnd_dir'] = encoder.fit_transform(dataset['wnd_dir'])

dataset.fillna(0, inplace=True)

# =========================
# 3. 归一化
# =========================
values = dataset.values
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

# =========================
# 4. 构造时间序列
# =========================
def create_dataset(data, look_back=24):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, :])
        y.append(data[i, 0])  # pollution
    return np.array(X), np.array(y)

look_back = 12
X, y = create_dataset(scaled, look_back)

# =========================
# 5. 时间划分
# =========================
train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# =========================
# 6. 升级模型（堆叠LSTM + Dropout）
# =========================
model = Sequential()

# 第一层LSTM（返回序列）
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))

# 第二层LSTM（压缩时序信息）
model.add(LSTM(32))
model.add(Dropout(0.2))

# 输出层
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')

# =========================
# 7. EarlyStopping（防止过拟合）
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# =========================
# 8. 训练
# =========================
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test, y_test),
    shuffle=False,
    verbose=2,
    callbacks=[early_stop]
)

# =========================
# 9. Loss曲线
# =========================
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

# =========================
# 10. 预测
# =========================
y_pred = model.predict(X_test)

# =========================
# 11. 反归一化
# =========================
X_test_last = X_test[:, -1, :]

pred_concat = np.concatenate((y_pred, X_test_last[:, 1:]), axis=1)
true_concat = np.concatenate((y_test.reshape(-1, 1), X_test_last[:, 1:]), axis=1)

inv_y_pred = scaler.inverse_transform(pred_concat)[:, 0]
inv_y_test = scaler.inverse_transform(true_concat)[:, 0]

# =========================
# 12. RMSE
# =========================
rmse = np.sqrt(mean_squared_error(inv_y_test, inv_y_pred))
print("Test RMSE:", rmse)

# =========================
# 13. 全局预测对比
# =========================
plt.figure(figsize=(12,5))
plt.plot(inv_y_test, label='True')
plt.plot(inv_y_pred, label='Predicted')
plt.legend()
plt.title('PM2.5 Prediction (Full Test Set)')
plt.show()

# =========================
# 14. 局部放大
# =========================
plt.figure(figsize=(12,5))
plt.plot(inv_y_test[:200], label='True')
plt.plot(inv_y_pred[:200], label='Predicted')
plt.legend()
plt.title('Result of First 200 Samples')
plt.show()

# =========================
# 15. 误差分布
# =========================
error = inv_y_test - inv_y_pred

plt.figure()
plt.hist(error, bins=50)
plt.title('Prediction Error Distribution')
plt.show()