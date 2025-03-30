import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def entropy_weight(data):
    """熵权法计算权重（带数据标准化）"""
    # 标准化处理
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-6)

    # 计算比重
    p = data_norm / data_norm.sum()

    # 计算熵值
    entropy = (-p * np.log(p + 1e-6)).sum(axis=0) / np.log(len(data))

    # 计算权重
    weights = (1 - entropy) / (1 - entropy).sum()
    return weights.values


def normalize_series(s):
    """将单个序列归一化到0-1"""
    return (s - s.min()) / (s.max() - s.min() + 1e-6)


def calculate_resilience(df):
    # 创建副本避免修改原始数据
    df = df.copy()

    # 负向指标正向化处理
    df['资产负债率'] = 1 - df['资产负债率']
    df['营业成本率'] = 1 - df['营业成本率']

    # 抵抗能力指标集（已正向化）
    resistance_cols = ['员工人数', '总资产', '流动比率', '速冻比率', '资产负债率']
    resistance_data = df[resistance_cols]

    # 恢复能力指标集
    recovery_cols = ['营业收入', '员工人均创收', '营业毛利率', '营业净利率', '营业成本率']
    recovery_data = df[recovery_cols]

    # 计算熵权
    resistance_weights = entropy_weight(resistance_data)
    recovery_weights = entropy_weight(recovery_data)

    # 计算能力得分（保持原始量纲）
    resistance_scores = (resistance_data * resistance_weights).sum(axis=1)
    recovery_scores = (recovery_data * recovery_weights).sum(axis=1)

    # 双维度归一化
    df['抵抗能力'] = MinMaxScaler().fit_transform(resistance_scores.values.reshape(-1, 1))
    df['恢复能力'] = MinMaxScaler().fit_transform(recovery_scores.values.reshape(-1, 1))

    # 计算韧性指数（加权平均）
    df['韧性指数'] = (df['抵抗能力'] + df['恢复能力']) / 2

    return df


if __name__ == "__main__":
    # 读取数据
    df = pd.read_excel('./data/clean_data.xlsx')

    # 确保数值类型
    numeric_cols = df.columns.drop(['股票代码', '年份', '股票简称'])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # 执行计算
    result_df = calculate_resilience(df)

    # 保存结果
    output_path = './data/clean_data_with_resilience.xlsx'
    result_df.to_excel(output_path, index=False)

    # 验证输出
    print(f"数据已保存至 {output_path}")
    print("\n韧性指数范围验证：")
    print(f"最小值：{result_df['韧性指数'].min():.4f}")
    print(f"最大值：{result_df['韧性指数'].max():.4f}")
    print("\n结果样例：")
    print(result_df[['股票代码', '年份', '抵抗能力', '恢复能力', '韧性指数']].head())