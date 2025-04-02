# 创建一个三层BP神经网络用pytorch
import torch
import torch.nn as nn

class BpNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BpNetwork, self).__init__()
        # 定义输入层到隐藏层的线性变换
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 定义隐藏层到输出层的线性变换
        self.fc2 = nn.Linear(hidden_size, output_size)
        # 定义激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播
        x = self.fc1(x)  # 输入层到隐藏层
        x = self.relu(x)  # 激活函数
        x = self.fc2(x)  # 隐藏层到输出层
        return x
# 获取模型
def get_model(input_size=13, hidden_size=25, output_size=1):
    model = BpNetwork(input_size, hidden_size, output_size)
    return model