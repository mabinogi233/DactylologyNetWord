from PIL import Image
import numpy as np
from keras.layers import Dense,LSTM,Conv2D,Reshape,Flatten,MaxPooling2D
from keras.models import Sequential
from keras.optimizers import RMSprop


def readImage(filePath):
    image = Image.open(filePath)
    return image


def image2narray(image):
    return np.asarray(image)


def build():
    # 图片序列中图片的最大张数
    time_steps = 1
    # 一维化图片的像素点数（30*30*4）
    input_vector = 3600
    # LSTM输出的维数（20*20）
    output_dim = 400
    # 卷积核大小(3*3)
    conv_div = (3, 3)
    # 卷积核个数
    conv_core = 4
    # 池化层大小
    pool_size = (2, 2)
    # 全链接层维度
    dense_size = 64

    NetWordModel = Sequential([
        # input_shape=(n_features, )
        LSTM(output_dim=output_dim, input_shape=(time_steps, input_vector)),
        # 将一维图片变为二维图片（reshape）
        Reshape((int(output_dim ** (1 / 2)), int(output_dim ** (1 / 2)), 1), input_shape=(output_dim,)),
        # 卷积层
        Conv2D(conv_core, conv_div, activation='relu',
               input_shape=(int(output_dim ** (1 / 2)), int(output_dim ** (1 / 2)), 1)),
        MaxPooling2D(pool_size=pool_size),
        # 第二卷积层
        Conv2D(conv_core * 2, conv_div, activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        # 图像一维化
        Flatten(),
        # 全连接层
        Dense(dense_size, activation='relu'),
        # 输出层
        Dense(100, activation='softmax'),
    ])
    return NetWordModel


def inputOperator(input):
    print("输入预处理")
    input = input.reshape(1,1,3600)
    return input



if __name__ == "__main__":

    #image = readImage(r"C:\Users\LiuWenze\Pictures\Camera Roll\QQ截图20210310212839.png")
    '''
    image = image.resize((40,40),Image.ANTIALIAS)
    image_array = image2narray(image)
    image_array = image_array.reshape((6400,))
    image_array = image_array[np.newaxis, :]
    print(image_array.shape)
    NetWordModel = build()
    print(NetWordModel.summary())
    print(NetWordModel.predict(inputOperator(image_array)).shape)
    '''
    print(np.array([]).shape)


