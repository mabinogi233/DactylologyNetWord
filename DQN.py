
from keras.layers import Dense,LSTM,Conv2D,Reshape,Flatten,MaxPooling2D
from keras.models import Sequential
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class DQN:
    '''
    初始化DQN，初始化参数
    :param  n_actions  需要输出多少个action的值，就是控制的动作 如左右
    :param  n_features  观测状态的维数，需要将观测状态reshape为(n_features,)
    :param  learning_rate  学习率
    :param  reward_decay  奖励的衰减率gamma
    :param  e_greedy  e_greedy算法参数，代表有e_greedy概率选择最优action  1-e_greedy的概率进行探索，即随机选择其他的action
    :param  replace_target_iter  隔多少步后，把MainNet神经网络的值全部复制给TargetNet神经网络
    :param  memory_size  记忆存储数据的最大容量
    :param  batch_size  一次随机梯度下降的样本数
    :param  e_greedy_increment  e_greedy的增长率，当此参数不为0时 e_greedy从0开始按此增长率线性增长
    '''
    def __init__(self,n_actions,n_features,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9,
                 replace_target_iter=300,memory_size=500,batch_size=32,e_greedy_increment=0):
        self.model_eval_file_path = r"D:\试验田\pyModel\eval_checkpoint"
        self.model_target_file_path = r"D:\试验田\pyModel\target_checkpoint"
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        if(e_greedy_increment==0):
            self.epsilon = 0
        else:
            self.epsilon = self.epsilon_max
        # 一共学习了多少步
        self.learn_step_counter = 0
        # 初始化记忆库
        # 每一行为四元组(st,at,rt,st+1),对应(当前观测状态，当前执行的行为，执行当前行为得到的奖励，执行行为后的状态)
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        # 建立神经网络
        self.build_net()
        #记忆过的样本数
        self.memory_counter = 0
        #绘图使用
        self.cost_draw = []
        self.acc_draw = []

    '''
    创建MainNet和TargetNet网络
    '''
    def build_net(self):

        #输出Q(s,a;θi)
        # 学习Q矩阵，输入状态，返回该状态下执行每个行为对应的Q值
        self.model_eval = self.build_model()

        # 选择rms优化器，输入学习率参数
        rmsprop = RMSprop(lr=self.lr, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model_eval.compile(loss='mse',optimizer=rmsprop,metrics=['accuracy'])

        #Target网络，输出Q(s',a';θ),与MainNet结构相同
        self.model_target = self.build_model()

    '''
    :param  s  当前状态
    :param  a  当前执行的动作
    :param  r  在s,a情况下得到的奖励
    :param  s_ 在s,a情况下下一步的状态s_ 
    '''
    def store_transition(self,s,a,r,s_):
        transition = np.hstack((s, [a, r], s_))
        #覆盖最早的历史
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1


    '''
    学习函数，执行一次学习过程
    '''
    def learn(self):
        # 经过一定的步数来做参数替换
        if (self.learn_step_counter % self.replace_target_iter == 0):
            self.model_target.set_weights(self.model_eval.get_weights())

        # 随机取出记忆，随机抽取memory_counter条记录训练
        # 随机生成索引
        if (self.memory_counter > self.memory_size):
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        #按随机生成的索引抽取记忆作为训练数据
        batch_memory = self.memory[sample_index, :]

        # TargetQ = r + γmaxQ(s',a';θ)
        # L(θ)=E[ (TargetQ − Q(s,a;θ))2]
        # 其中，Q(s',a';θ)为TargetNet的输出，Q(s,a;θ)为MainNet的输出

        #Q为二维矩阵，行表示每一条数据，列表示这条数据s(t)对应的每个action的Q值
        #取前n_features列为s(t)
        q_eval = self.model_eval.predict(self.inputOperator(batch_memory[:, :self.n_features]), batch_size=self.batch_size)

        # 取后n_features列为s(t+1)
        q_next = self.model_target.predict(self.inputOperator(batch_memory[:, -self.n_features:]), batch_size=self.batch_size)

        #初始化与q_eval一样大小的矩阵
        q_target = q_eval.copy()

        # 取出对应的奖赏r
        reward = batch_memory[:, self.n_features + 1]

        #序号,定位哪一条训练记录
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        #定位action的索引，表示是第几个策略对应的Q
        eval_act_index = batch_memory[:, self.n_features].astype(int)

        #计算目标Q，即作为神经网络的标签
        #axis=1按列进行max，选择出每一行的最大值
        #每次仅修改一列对应的Q
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # 训练估计网络，用的是当前观察值训练，并且训练选择到的数据加奖励训练而不是没选择的
        # 训练集输入x=s(t)    y = TargetQ
        self.cost = self.model_eval.train_on_batch(self.inputOperator(batch_memory[:, :self.n_features]), q_target)

        #绘图
        self.cost_draw.append(self.cost[0])
        self.acc_draw.append(self.cost[1])

        # 增大epsilon贪心选择最优action的概率
        if(self.epsilon <= self.epsilon_max and self.epsilon + self.epsilon_increment <= self.epsilon_max):
            self.epsilon = self.epsilon + self.epsilon_increment
        else:
            self.epsilon = self.epsilon_max

        #学习次数加1
        self.learn_step_counter += 1

    '''
    根据观测的状态值选择下一步的行为
    :param  observation  观测状态值
    '''
    def choose_action(self,observation):
        observation = observation[np.newaxis, :]
        #概率选择经验或者随机探索
        if np.random.uniform() < self.epsilon:
            actions_value = self.model_eval.predict(self.inputOperator(observation))
            action_index = np.argmax(actions_value)
        else:
            action_index = np.random.randint(0, self.n_actions)
        return action_index

    '''
    根据观测的状态值选择下一步的行为，实际使用时调用此方法
    '''
    def choose_action_only_experience(self,observation):
        observation = observation[np.newaxis, :]
        #选择经验
        actions_value = self.model_eval.predict(self.inputOperator(observation))
        action_index = np.argmax(actions_value)

        return action_index

    '''
    绘图
    '''
    def plot_cost(self,param='cost'):
        if(param=='cost'):
            plt.plot(np.arange(len(self.cost_draw)), self.cost_draw)
            plt.ylabel('Cost')
            plt.xlabel('training steps')
            plt.show()
        elif(param=='acc'):
            plt.plot(np.arange(len(self.acc_draw)), self.acc_draw)
            plt.ylabel('acc')
            plt.xlabel('training steps')
            plt.show()


    def plot_hist(self):
        list_x = ['0~0.1','0.1~0.2','0.2~0.3','0.3~0.4','0.4~0.5','0.5~0.6','0.6~0.7','0.7~0.8','0.8~0.9','0.9~1.0']
        list_y = [0 for i in range(10)]
        for acc in self.acc_draw:
            if(int(acc/0.1)>=0 and int(acc/0.1)<10):
                list_y[int(acc/0.1)] +=1
            elif(int(acc/0.1)==10):
                list_y[9] +=1

        plt.bar(list_x, list_y, label="acc", color='red')
        plt.xticks()
        plt.legend(loc="upper left")  # 防止label和图像重合显示不出来
        plt.ylabel('conut')
        plt.xlabel('acc')
        plt.show()



    def save_model(self):
        self.model_eval.save_weights(self.model_eval_file_path)
        self.model_target.save_weights(self.model_target_file_path)


    def load_model(self):
        self.model_eval.load_weights(self.model_eval_file_path)
        self.model_target.load_weights(self.model_target_file_path)


    '''
    钩子方法，子类可重写
    '''
    def build_model(self):
        #图片序列中图片的最大张数
        self.time_steps = 10
        #一维化图片的像素点数（40*40*3)
        self.input_vector = 4800
        #LSTM输出的维数（10*10）
        output_dim = 100
        #卷积核大小(3*3)
        conv_div = (3,3)
        #卷积核个数
        conv_core = 4
        #池化层大小
        pool_size = (2, 2)
        #全链接层维度
        dense_size = 64

        NetWordModel =  Sequential([
            #input_shape=(n_features, )
            LSTM(output_dim=output_dim, input_shape=(self.time_steps, self.input_vector)),
            #将一维图片变为二维图片（reshape）
            Reshape((int(output_dim**(1/2)),int(output_dim**(1/2)),1),input_shape=(output_dim, )),
            #卷积层
            Conv2D(conv_core, conv_div, activation='relu', input_shape=(int(output_dim**(1/2)), int(output_dim**(1/2)), 1)),
            MaxPooling2D(pool_size=pool_size),
            #第二卷积层
            Conv2D(conv_core*2, conv_div,activation='relu'),
            MaxPooling2D(pool_size=pool_size),
            #图像一维化
            Flatten(),
            #全连接层
            Dense(dense_size, activation='relu'),
            #输出层
            Dense(self.n_actions, activation='softmax'),
        ])
        return NetWordModel

    '''
    钩子方法，子类可重写，input为二维输入，第一个维度表示样本个数，第二个维度为faltten后的数据的特征值数目
    需要将其reshape为build_model中model的输入形式
    '''
    def inputOperator(self,input):
        # print("输入预处理")
        return input.reshape((input.shape[0],self.time_steps,self.input_vector))



