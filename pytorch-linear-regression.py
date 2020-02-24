# 导入 numpy 
import numpy as np
# 导入 pytorch 库
import torch


### 1. create training inputs.

# 作为一家汽车公司，我们从以前的销售中收集这些数据
# 定义汽车价格
car_prices_array = [3,4,5,6,7,8,9]
car_prices_array

car_price_np = np.array(car_prices_array,dtype=np.float32)
car_price_np

# transform the array into a matrix.
# each line is a label.
car_price_np = car_price_np.reshape(-1,1)
car_price_np

car_price_tensor = Variable(torch.from_numpy(car_price_np))
car_price_tensor


# 定义汽车销售量
number_of_car_sell_array = [ 7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5]
number_of_car_sell_np = np.array(number_of_car_sell_array,dtype=np.float32)
number_of_car_sell_np = number_of_car_sell_np.reshape(-1,1)
number_of_car_sell_tensor = Variable(torch.from_numpy(number_of_car_sell_np))
number_of_car_sell_tensor

# 将数据可视化
import matplotlib.pyplot as plt
plt.scatter(car_prices_array, number_of_car_sell_array)
plt.xlabel("Car Price $")
plt.ylabel("Number of Car Sell")
plt.title("Car Price$ VS Number of Car Sell")
plt.show()


### 2. build the model
# Pytorch线性回归

# 导入库
import torch      
from torch.autograd import Variable     
import torch.nn as nn 
import warnings
warnings.filterwarnings("ignore")

# 创建LinearRegression类 网络结构
class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size):
        
        # 超级函数，继承自nn.Module
        super(LinearRegression,self).__init__()
        
        # 线性函数 层
        self.linear = nn.Linear(input_dim,output_dim)

        # 层之间的顺序
    def forward(self,x):
        return self.linear(x)

# 定义模型
input_dim = 1
output_dim = 1
# instance the models
model = LinearRegression(input_dim,output_dim) # 输入和输出大小为1


# loss function MSE/均方差
mse = nn.MSELoss()

# optimizer 优化(找到最小化差值的参数)
learning_rate = 0.02   # 学习率, 达到最佳参数的速度有多快
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)


# 训练模型
loss_list = []
iteration_number = 200
for iteration in range(iteration_number):

    # 优化
    optimizer.zero_grad() 

    # 计算模型输出
    results = model(car_price_tensor)

    # 计算损失/差值
    loss = mse(results, number_of_car_sell_tensor)

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 保存损失/差值
    loss_list.append(loss.data)

    # 打印损失
    if(iteration % 50 == 0):
        print('epoch {}, loss {}'.format(iteration, loss.data))

plt.plot(range(iteration_number),loss_list)
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.show()





# 预测汽车价格
predicted = model(car_price_tensor).data.numpy()
plt.scatter(car_prices_array,number_of_car_sell_array,label = "original data",color ="red")
plt.scatter(car_prices_array,predicted,label = "predicted data",color ="blue")

# 预测一下，如果汽车的价格是10美元，那么汽车的销量会是多少
# predicted_10 = model(torch.from_numpy(np.array([10]))).data.numpy()
#plt.scatter(10,predicted_10.data,label = "car price 10$",color ="green")
plt.legend()
plt.xlabel("Car Price $")
plt.ylabel("Number of Car Sell")
plt.title("Original vs Predicted values")
plt.show()
