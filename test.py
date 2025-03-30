import pandas as pd

df = pd.read_excel('./data/clean_data_with_resilience.xlsx')

# 拿到最后一列数据
y = df.iloc[:, -1].values.reshape(-1, 1)

# 从小到大排序输出
print("排序前：", y)
y.sort()
print("排序后：", y)