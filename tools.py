import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sympy.physics.quantum.density import entropy

data_path = './data/clean_data.xlsx'
normalized_data_path = './data/normalized_data.xlsx'
# 读取数据进行归一化
def normalize_data(path):
    df = pd.read_excel(path)
    # 资产负债率 和 营业成本率 为负向指标

    # 先将正指标标准化 减去最小值 除以值域
    # 负向指标 先减去最大值 除以值域
    for col in df.columns:
        if col not in ['资产负债率', '营业成本率', '股票代码', '年份', '股票简称']:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-6)
        elif col in ['资产负债率', '营业成本率']:
            df[col] = (df[col].max() - df[col]) / (df[col].max() - df[col].min() + 1e-6)

    # save
    df.to_excel('./data/normalized_data.xlsx', index=False)

# 计算熵权
def entropy_weight(path):
    df = pd.read_excel(path)
    df = df.drop(columns=['股票代码', '年份', '股票简称'])
    columns  = df.columns
    data = df[columns ].values

    # 计算比重
    m, n = data.shape
    p = data / np.sum(data, axis=0)

    p = np.where(p == 0, 1e-6, p)  # 避免对数计算中的零值

    # 计算熵值
    entropy = -np.sum(p * np.log(p), axis=0) / np.log(m)

    # 计算差异系数
    diff = 1 - entropy

    # 计算权重
    weights = diff / np.sum(diff)

    weights_dict = dict(zip(columns, weights))

    df['综合得分'] = df[columns].dot(weights)

    df.to_excel('./data/entropy_weights.xlsx', index=False)

    return weights_dict, df

# 根据年份将数据划分

def load_data():
    data = pd.read_excel('./data/entropy_weights.xlsx')
    x = data.iloc[:, :-2].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

    # 插值生成更多的样本
    def interpolate_data(data, num_interpolations = 49):
        interpolated_data = []
        for i in range(len(data) - 1):
            for t in range(num_interpolations + 1):
                interpolated_point = data[i] + (data[i + 1] - data[i]) * (t / (num_interpolations + 1))
                interpolated_data.append(interpolated_point)
        return np.array(interpolated_data)
    x = interpolate_data(x)
    y = interpolate_data(y)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    return X_train, X_test, Y_train, Y_test

if __name__ == "__main__":
    # normalize_data(data_path)
    print("==================================================")
    # entropy_weight(normalized_data_path)
    #
    # weigthts, df = entropy_weight(normalized_data_path)
    #
    # print("权重：")
    # for col, weight in weigthts.items():
    #     print(f"{col}: {weight:.4f}")

    X_train, X_test, Y_train, Y_test = load_data()

    print("训练集：", X_train.shape, Y_train.shape)
    print("测试集：", X_test.shape, Y_test.shape)

