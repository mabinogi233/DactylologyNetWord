from Environment import Environment
from PIL import Image
import numpy as np
import os
import random



class dactyloloEnvironment(Environment):

    # 图片一维拉伸后的大小 40 * 40 * 10（张数） * 通道数
    n_features = 48000

    time_step = 10

    # 标签的个数
    n_actions = 29


    def __init__(self,isRun = False):
        super(dactyloloEnvironment,self).__init__()
        if(not isRun):
            self.loadPackageImage()
        else:
            self.loadPackageImageOnlyY()
        #保存上一个样本的索引位置
        self.old_index = 0
        #保存步数
        self.temp = 0
        #多少步后终止
        self.maxTemp = 10000


    '''
    初始化环境，返回初始观测值
    :return 
    '''
    def reset(self):
        self.temp = 0
        self.old_index = random.randint(0,len(self.trainX)-1)
        return self.trainX[self.old_index]




    '''
    根据当前状态和执行的action更新环境状态，
    返回观测值（np.array，shape=（n_features，）、奖励(double)、是否结束标志(boolean)
    :param action 行为的索引，表示执行第action个行为
    :return s_ 下一个状态的观测值
    :return reward 奖励
    :return done 是否终止
    '''
    def step(self, action):
        self.temp+=1
        reward = 0
        if(action==self.trainY[self.old_index]):
            #判断成功，获得奖励
            reward = 10
        #选择下一个样本
        self.old_index = random.randint(0, len(self.trainX) - 1)
        s_ = self.trainX[self.old_index]
        #判断是否结束
        done = False
        if(self.maxTemp<self.temp):
            done = True

        return s_, reward, done

    '''
    销毁此环境
    '''
    def destroy(self):
        pass




    '''
    加载路径指定的图片
    '''
    def loadImage(self,filePath):
        image = Image.open(filePath)
        image = image.resize((40, 40), Image.ANTIALIAS)

        #返回（高度*宽度*通道数）= n_features的图
        return image


    '''
    将图片转化为一维矩阵
    '''
    def image2narray(self,image):

        return np.asarray(image).reshape((self.n_features//self.time_step,))


    '''
    读取目录下的全部视频（视频为一系列图片，存于一个文件夹内），并转化为一维矩阵,并将文件名作为标签
    '''
    def loadPackageImage(self,rootdir=r"C:\Users\LiuWenze\Desktop\ResourceSpace"):
        listfiles1 = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        trainX = []
        trainY = []
        #id和name一一对应
        #存放 key=id value=名称的键值对
        self.action_id_ditc = {}
        # 存放 key=名称 value=id的键值对
        self.action_name_ditc = {}

        for i in range(0,len(listfiles1)):
            filePath1 = os.path.join(rootdir,listfiles1[i])
            if (os.path.isdir(filePath1)):
                #此文件夹(filePath1)对应一个视频
                print("正在收集：",filePath1)
                fileName = listfiles1[i].split(".")[0]
                if (fileName in self.action_name_ditc.keys()):
                    trainY.append(self.action_name_ditc[fileName])
                else:
                    image_id = self.n_actions
                    self.n_actions += 1
                    self.action_name_ditc[fileName] = image_id
                    self.action_id_ditc[image_id] = fileName
                    trainY.append(image_id)
                    # 加载视频
                    one_data_x = self.readVedio(filePath1)
                    trainX.append(one_data_x)

        self.trainX = trainX
        self.trainY = trainY

    def loadPackageImageOnlyY(self, rootdir=r"C:\Users\LiuWenze\Desktop\ResourceSpace"):
        listfiles1 = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        trainY = []
        #id和name一一对应
        #存放 key=id value=名称的键值对
        self.action_id_ditc = {}
        # 存放 key=名称 value=id的键值对
        self.action_name_ditc = {}
        for i in range(0,len(listfiles1)):
            filePath1 = os.path.join(rootdir,listfiles1[i])
            if (os.path.isdir(filePath1)):
                #此文件夹(filePath1)对应一个视频
                print("正在收集：",filePath1)
                fileName = listfiles1[i].split(".")[0]
                if (fileName in self.action_name_ditc.keys()):
                    trainY.append(self.action_name_ditc[fileName])
                else:
                    image_id = self.n_actions
                    self.n_actions += 1
                    self.action_name_ditc[fileName] = image_id
                    self.action_id_ditc[image_id] = fileName
                    trainY.append(image_id)
        self.trainX = []
        self.trainY = trainY

    '''
    加载预测数据
    '''
    def getPrecObservation(self,filePath):
        return self.readVedio(filePath)

    def readVedio(self,filePath):
        #filePath为文件夹
        rList = np.array([])
        #取出文件夹内部的图片
        listfiles = os.listdir(filePath)
        for i in range(0, len(listfiles)):
            fname = listfiles[i].split(".")[1]
            if (fname == "png"):
                path = os.path.join(filePath, listfiles[i])
                rList = np.hstack((rList,self.image2narray(self.loadImage(path))))
        return rList
    '''
    根据行为的索引获取行为的值
    '''
    def getNameByActionIndex(self,index):
        return self.action_id_ditc[index]





