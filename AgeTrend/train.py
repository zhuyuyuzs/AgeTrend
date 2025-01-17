import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 定义DNN模型
class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # 输出层，预测人口增长率

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 训练模型
def train_model(X_train, y_train, X_test, y_test, df_test):
    # 转换为Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # 初始化模型
    input_size = X_train.shape[1]  # 特征数量
    hidden_size = 128  # 隐藏层大小
    model = DNNModel(input_size, hidden_size)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    epochs = 200  # 训练轮数
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()  # 清零梯度
        output = model(X_train_tensor)  # 模型预测
        loss = criterion(output.squeeze(), y_train_tensor)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 每10个epoch打印一次损失
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # 评估模型
    evaluate_model(model, X_test_tensor, y_test_tensor, df_test)

    return model

# 评估模型，计算误差并可视化
def evaluate_model(model, X_test, y_test, df_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predictions = predictions.squeeze().numpy()

    # 计算均方误差
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    # 获取测试集中的年份信息，仅提取与测试集对应的年份
    test_years = df_test['年份'].values[:len(y_test)]  # 确保年份的长度和测试集长度一致

    # 可视化实际值和预测值的对比
    plt.figure(figsize=(10, 6))
    plt.plot(test_years, y_test, label="Actual Growth Rate", color='b')
    plt.plot(test_years, predictions, label="Predicted Growth Rate", color='r', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel('Population Growth Rate (%)')
    plt.title('Actual vs Predicted Population Growth Rate')
    plt.legend()
    plt.show()

# 保存模型
def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)
    print(f'Model saved as {file_name}')
