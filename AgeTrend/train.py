import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 定义改进版的LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        # 使用Dropout来防止过拟合
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 1)  # 输出层，预测一个值（人口增长率）

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 增加时间维度，变成 (batch_size, 1, input_size)
        out, _ = self.lstm(x)  # 获取LSTM的输出
        out = out[:, -1, :]  # 只取最后一个时间步的输出
        out = self.fc(out)  # 通过全连接层得到预测值
        return out

def train_model(X_train, y_train, X_test, y_test, df_test):
    # 转换为Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 初始化模型
    input_size = X_train.shape[1]  # 输入特征的数量
    hidden_size = 128  # 增加隐藏层的大小
    num_layers = 3  # 增加LSTM的层数
    model = LSTMModel(input_size, hidden_size, num_layers)

    # 使用Adam优化器，调整学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    criterion = nn.MSELoss()  # 使用均方误差损失函数

    # 训练循环
    epochs = 200  # 增加训练的轮数
    patience = 10  # 提前停止的耐心轮数（early stopping）
    best_loss = float('inf')
    no_improvement_epochs = 0  # 跟踪没有改进的轮数

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()  # 清零梯度
            output = model(data)  # 模型预测
            loss = criterion(output.squeeze(), target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        # 每10个epoch打印一次损失
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        # 进行早停判断
        if loss.item() < best_loss:
            best_loss = loss.item()
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print(f'Early stopping at epoch {epoch + 1} due to no improvement')
            break

    # 调用评估函数
    evaluate_model(model, X_test, y_test, df_test)

    return model

def evaluate_model(model, X_test, y_test, df_test):
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 评估时不计算梯度
        predictions = model(X_test_tensor)  # 模型预测
        predictions = predictions.squeeze().numpy()  # 转换为numpy数组

    # 计算均方误差
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    # 获取测试集中的年份信息
    test_years = df_test['年份'].values

    # 可视化实际值和预测值的对比
    plt.figure(figsize=(10, 6))
    plt.plot(test_years, y_test, label="Actual Population Growth Rate", color='b')
    plt.plot(test_years, predictions, label="Predicted Population Growth Rate", color='r', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel('Population Growth Rate (%)')
    plt.title('Actual vs Predicted Population Growth Rate')
    plt.legend()
    plt.show()

def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)
    print(f'Model saved as {file_name}')
