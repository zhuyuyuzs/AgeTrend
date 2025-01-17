import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from train import train_model, save_model, evaluate_model

# 数据加载
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 数据预处理
def preprocess_data(df):
    # 提取特征和目标
    features = df[['出生人口(万)', '总人口(万人)', '中国人均GPA(美元计)', '中国性别比例(按照女生=100)',
                   '城镇人口(城镇+乡村=100)', '乡村人口', '美元兑换人民币汇率', '中国就业人口(万人)']].values
    target = df['自然增长率(%)'].values

    # 特征标准化
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # 将年份列加入特征
    scaled_features = np.hstack((scaled_features, df[['年份']].values))

    return scaled_features, target, df

def main():
    # 文件路径
    file_path = 'population_data.csv'

    # 加载和预处理数据
    df = load_data(file_path)
    X, y, df_scaled = preprocess_data(df)

    # 将数据分为训练集和测试集（1970-2000年训练集，2000-2020年测试集）
    train_df = df[df['年份'] <= 2000]
    test_df = df[df['年份'] > 2000]

    # 训练集数据
    X_train = train_df[['出生人口(万)', '总人口(万人)', '中国人均GPA(美元计)', '中国性别比例(按照女生=100)',
                        '城镇人口(城镇+乡村=100)', '乡村人口', '美元兑换人民币汇率', '中国就业人口(万人)', '年份']].values
    y_train = train_df['自然增长率(%)'].values

    # 测试集数据
    X_test = test_df[['出生人口(万)', '总人口(万人)', '中国人均GPA(美元计)', '中国性别比例(按照女生=100)',
                      '城镇人口(城镇+乡村=100)', '乡村人口', '美元兑换人民币汇率', '中国就业人口(万人)', '年份']].values
    y_test = test_df['自然增长率(%)'].values

    # 数据标准化
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练模型
    model = train_model(X_train_scaled, y_train, X_test_scaled, y_test, test_df)

    # 保存模型
    save_model(model, 'population_growth_rate_model.pth')

    # 评估并计算误差
    evaluate_model(model, X_test_scaled, y_test, test_df)

    # 预测未来数据（2020年后的人口增长率）
    future_years = np.array(range(2021, 2031)).reshape(-1, 1)
    future_data = np.hstack([np.zeros((future_years.shape[0], X_train.shape[1]-1)), future_years])  # 使用年份数据进行预测
    future_data_scaled = scaler.transform(future_data)  # 标准化
    future_predictions = model(torch.tensor(future_data_scaled, dtype=torch.float32)).detach().numpy()  # 模型预测

    # 可视化2020年后的预测
    plt.plot(future_years, future_predictions, label="Predicted Growth Rate", color='g', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel('Population Growth Rate (%)')
    plt.title('Predicted Population Growth Rate (2021-2030)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
