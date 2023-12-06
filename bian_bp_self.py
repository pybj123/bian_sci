import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch.utils.data as Data


# 读取数据
data = pd.read_excel("bian_1.xlsx")

colors = data.iloc[:, 0:6]

scores = data.iloc[:, -1]
scores_np = np.array(scores)



# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_color = scaler.fit_transform(colors)
scaler_score = scaler.fit_transform(scores_np.reshape(-1, 1))

# 数据集划分
colors_train, colors_test, scores_train, scores_test = train_test_split(scaled_color,
                                                                        scaler_score,
                                                                        test_size=0.2,
                                                                        random_state=250)

# 构建张量数据集
train_dataset = Data.TensorDataset(torch.from_numpy(colors_train),
                                   torch.from_numpy(scores_train))

test_dataset = Data.TensorDataset(torch.from_numpy(colors_test),
                                  torch.from_numpy(scores_test))

# 构建数据加载器
size = 64
train_loader = Data.DataLoader(train_dataset, batch_size=size)
test_loader = Data.DataLoader(test_dataset, batch_size=size)

# 搭建神经网络
class MyModule(nn.Module):
    # 初始化层数
    def __init__(self):
        super(MyModule, self).__init__()
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, 1)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = self.fc6(x)
        return x

myModule = MyModule()
print(myModule)

# 放到GPU上
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModule.to(device)

# 损失函数
myLoss = nn.L1Loss()
# 优化器
optimizer = optim.Adam(myModule.parameters(), lr=0.001)

# 训练次数
num_epoch = 2000

#损失数组
my_loss_arr = []

for epoch in range(num_epoch):
    myModule.train()
    total_loss = 0.0
    for data, label in train_loader:
        # 将数据放入显存
        data = data.to(device, dtype=torch.float32)  # 将数据转换为 Float 类型
        label = label.to(device, dtype=torch.float32)  # 将标签也转换为 Float 类型

        # 计算出结果
        output = myModule(data)
        # 计算损失
        loss = myLoss(output, label)

        # 清除梯度
        optimizer.zero_grad()
        # 反向传播loss
        loss.backward()

        # 更新
        optimizer.step()
        # 加和loss
        total_loss += loss.item()
    #每10次打印平均方差
    if (epoch+1) % 10 == 0:
        av_loss = total_loss / len(train_loader)
        # 记录损失
        my_loss_arr.append(av_loss)
        print(f'Epoch [{epoch + 1}/{num_epoch}], Average Loss (MSE): {av_loss:.10f}')
#保存模型到文件
torch.save(myModule.state_dict(),"color_score.pth")
#调整到评估模式
myModule.eval()
#预测值数组
p_values = []
#实际值
a_values = []

with torch.no_grad():
    for data,label in test_loader:
        data = data.to(device, dtype=torch.float32)  # 将数据转换为 Float 类型
        label = label.to(device, dtype=torch.float32)  # 将标签也转换为 Float 类型
        output = myModule(data)
        p_values.extend(output.cpu().numpy())
        a_values.extend(label.cpu().numpy())


#处理预测值，反归一化预测值和真实值
p_values_1 = []


p_values = scaler.inverse_transform(p_values)

for s in p_values:
    if s < 1.5:
        p_values_1.append(1)
    elif s < 2.5:
        p_values_1.append(2)
    elif s < 3.5:
        p_values_1.append(3)
    elif s < 4.5:
        p_values_1.append(4)
    else:
        p_values_1.append(5)

a_values = scaler.inverse_transform(a_values)
#计算正确率
right = 0
for (p,a) in zip(p_values_1,a_values):
    if p == a :
        right+=1


print("正确率："+str(right/len(a_values)))
sample_interval =50  # 每隔10个数据点绘制一个点
plt.figure(figsize=(20, 10))
plt.plot(range(0, len(p_values_1), sample_interval), p_values_1[::sample_interval], label="Predicted Values", marker='o',linewidth = 5 , markersize = 10)
plt.plot(range(0, len(a_values), sample_interval), a_values[::sample_interval], label="Actual Values", marker='x' , linewidth = 5 ,markersize = 10 , markeredgewidth=5)
plt.xlabel("Data Point" ,  fontsize = 20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Score (0-6)" , fontsize = 20)
plt.legend()
plt.show()

#画训练时的损失图
sample_interval_1 = 1
plt.figure(figsize=(20,10))
plt.plot(my_loss_arr,label = "Losses while training" , marker = 'o', linewidth = 5)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Data Piont" ,  fontsize = 20)
plt.ylabel("loss" ,  fontsize = 20)
plt.legend()
plt.show()
