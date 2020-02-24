from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)



# 导入 numpy 
import numpy as np

# numpy 数组
array = [[1,2,3],[4,5,6]]

# pytorch 数组
tensor = torch.Tensor(array)

print("Array Type: {}".format(tensor.type)) # type
print("Array Shape: {}".format(tensor.shape)) # shape
print(tensor)


# pytorch random
print(torch.rand(2,3)) # 从区间[0, 1)的均匀分布中抽取的一组随机数。


# 让我们从一个简单的张量中创建一个变量:
# 从pytorch autograd库中导入变量
from torch.autograd import Variable

# 定义变量
x = Variable(torch.ones(2, 2) * 2, requires_grad=True)
x
# 在上面的变量声明中，传入了一个张量，同时指明这个变量需要一个梯度，这意味着这个变量是可以训练的。如果requires_grad标志设置为False，则不会对变量进行训练。

#变量和张量之间的区别是变量中包含梯度，变量也可以进行数学运算。