import numpy as np

import random


class Environment():

    #状态观测值的维度
    n_features = 24

    #行为的个数
    n_actions = 2

    def __init__(self):
        #默认参数，无需改动
        self.old_index = 0
        self.temp = 0





    '''
    初始化环境，返回初始观测值
    :return 
    '''
    def reset(self):

        pass
        #return s

    '''
    根据当前状态和执行的action更新环境状态，
    返回观测值（np.array，shape=（n_features，）、奖励(double)、是否结束标志(boolean)
    :param action 行为的索引，表示执行第action个行为
    :return s_ 下一个状态的观测值
    :return reward 奖励
    :return done 是否终止
    '''
    def step(self, action):
        pass

        #return s_, reward, done


    '''
    销毁此环境
    '''
    def destroy(self):
        pass







