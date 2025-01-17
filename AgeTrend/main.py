import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from train import train_model, save_model, evaluate_model  # 导入train.py中的函数

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
    file_path = 'population_data.csv'  # 请修改为你的文件路径

    # 加载和预处理数据
    df = load_data(file_path)
    X, y, df_scaled = preprocess_data(df)

    # 数据分割
    train_data = df[df['年份'] < 2000]  # 训练集：2000年之前的数据
    test_data = df[df['年份'] >= 2000]  # 测试集：2000年之后的数据

    # 特征和目标划分
    X_train = train_data[['出生人口(万)', '总人口(万人)', '中国人均GPA(美元计)', '中国性别比例(按照女生=100)',
                          '城镇人口(城镇+乡村=100)', '乡村人口', '美元兑换人民币汇率', '中国就业人口(万人)']].values
    y_train = train_data['自然增长率(%)'].values
    X_test = test_data[['出生人口(万)', '总人口(万人)', '中国人均GPA(美元计)', '中国性别比例(按照女生=100)',
                        '城镇人口(城镇+乡村=100)', '乡村人口', '美元兑换人民币汇率', '中国就业人口(万人)']].values
    y_test = test_data['自然增长率(%)'].values

    # 特征标准化
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 将年份列加入特征
    X_train_scaled = np.hstack((X_train_scaled, train_data[['年份']].values))
    X_test_scaled = np.hstack((X_test_scaled, test_data[['年份']].values))

    # 训练模型
    model = train_model(X_train_scaled, y_train, X_test_scaled, y_test, test_data)

    # 评估模型
    evaluate_model(model, torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32), test_data)

    # 保存模型
    save_model(model, 'population_growth_rate_model.pth')

if __name__ == '__main__':
    main()
