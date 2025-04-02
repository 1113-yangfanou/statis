import pandas as pd
import torch

from tools import load_data

X_train, X_test, Y_train, Y_test = load_data()

print(X_train.shape, Y_train.shape)

weights = torch.load('ga_optimized_weights.pth')

for name, param in weights.items():
    print(f"参数名: {name}, 数据类型: {param.dtype}, 形状: {param.shape}")