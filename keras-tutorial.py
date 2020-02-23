
# ref ： Keras 导入库与模块 | 奇客谷教程 💯
# https://www.qikegu.com/docs/4158


# 从Keras导入Sequential模型类型。这是一个简单的线性神经网络层的栈
from keras.models import Sequential
# 从Keras导入核心层
from keras.layers import Dense, Dropout, Activation, Flatten
# 将从Keras导入CNN层
from keras.layers import Convolution2D, MaxPooling2D
# 导入一些实用程序，用于转换数据
from keras.utils import np_utils
# 导入backend，获取底层实现库的信息，例如可以获取支持的图像格式：
from keras import backend as K
# 导入numpy开始，并为计算机的伪随机数生成器设置一个种子，相同种子可以产生同系列的随机数
import numpy as np
np.random.seed(123)  # 种子相同，随机数产生可以重现





###1. 数据预处理###

#Keras库已经包含了这个数据集，可以从Keras库中加载:
from keras.datasets import mnist

# 将预打乱的MNIST数据加载到训练和测试集中
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 查看数据集的形状
print (X_train.shape)
# 可以看到，训练集中有60000个样本，每个图像都是28像素x28像素。


# 要查看手写数字图像，可以使用matplotlib绘制，下面绘制MNIST数据集中的第一个图像：
from matplotlib import pyplot as plt
plt.imshow(X_train[0])
plt.show()

# MNIST是灰度图像，位深为1，我们将数据集从形状(n，宽度，高度)转换为(n，位深，宽度，高度)。
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

# 重新打印X_train的形状:
print (X_train.shape)

# 将数据类型转换为float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 将数据值归一化到[0,1]范围内。
X_train /= 255
X_test /= 255

# 让我们看看分类标签数据:
print (y_train.shape)
# (60000,) 等于（60000，1）
print (y_train[:10])
# [5 0 4 1 9 2 1 3 1 4]

# 将一维类数组转换为10维分类矩阵
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

print (Y_train.shape)
# (60000, 10)

print (Y_train[:10])



###2. 定义神经网络模型架构###

# 首先声明一个Sequential模型格式:
model = Sequential()
# 接下来，声明输入层:
model.add(Convolution2D(32, 3, 3, activation='relu',  input_shape=input_shape))
# 32, 3, 3 这3个参数分别对应于要使用的卷积滤波器的数量、每个卷积核中的行数和列数。

# 打印当前模型输出的形状
print (model.output_shape)
# (None, 32, 26, 26)

# 接下来，可以简单地在模型中添加更多层，就像搭积木一样:
model.add(Convolution2D(32, 3, 3, activation='relu'))
# MaxPooling2D层是池化层，进一步降低数据量提取特征。
model.add(MaxPooling2D(pool_size=(2,2)))
# Dropout层的作用是防止过拟合
model.add(Dropout(0.25))

# 到目前为止，对于模型参数，已经添加了2个卷积层。为完成CNN模型架构，还需添加一个全连接层和输出层:
model.add(Flatten()) # 将卷积层的权值传递到全连接层之前，必须将卷积层的权值压平(使其为一维)。
model.add(Dense(128, activation='relu')) # 对于全连接层/稠密层，第一个参数是该层的输出大小。Keras自动处理层之间的连接。
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) # 最后一层的输出大小为10，对应于0~9的10个数字。



###3. 定义损失函数和优化器###
# 编译模型。在编译模型时，设置损失函数与优化器。
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



###4. 开始训练模型###
# 训练模型，即根据训练数据拟合模型的过程。
# 为了拟合这个模型，需要设置训练的批次大小和训练周期(epoch)数，
# 另外，当然需要传递训练数据。
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=10, verbose=1)
# Epoch 1/10
# 7744/60000 [==>...........................] - ETA: 96s - loss: 0.5806 - acc: 0.8164

###5. 使用测试数据评估模型的性能。###

score = model.evaluate(X_test, Y_test, verbose=0)