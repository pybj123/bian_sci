import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# 定义神经网络模型类
class MyModule(nn.Module):
    # 初始化层数
    def __init__(self):
        super(MyModule, self).__init__()
        self.fc1 = nn.Linear(6, 64)  # 6个值的输入
        self.fc2 = nn.Linear(64, 32)  # 第1个隐藏层
        self.fc3 = nn.Linear(32, 16)   # 第2个隐藏层
        self.fc4 = nn.Linear(16, 16)  # 第1个隐藏层
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, 1)    # 输出层

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = self.fc6(x)
        return x

# 创建神经网络实例
myModule = MyModule()

# 加载保存的模型参数
myModule.load_state_dict(torch.load("tanh+mae+0.001+2000.pth"))

# 设置模型为评估模式
myModule.eval()


input_data = [0.097493036	,0.151515152	,0.747474747

,
0.125348189	,0.03030303	,0.505050505
]




input_tensor = torch.tensor(input_data, dtype=torch.float32)
predicted_scores = myModule(input_tensor)
print(f"预测的两个分数为: {predicted_scores}")

