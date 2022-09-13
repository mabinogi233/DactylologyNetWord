from DQN import DQN
from Environment import Environment
from dactylologyEnvironment import dactyloloEnvironment
import time
from PIL import Image
import numpy as np
import sys
import os

printflag = True



class RunDQN:
    def __init__(self):
        pass

    @classmethod
    def run_learn(cls,env = Environment(),zl = False):
        if(printflag):
            print("开始训练")
        start = time.process_time()
        RL = DQN(env.n_actions, env.n_features,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9,
                 replace_target_iter=10,memory_size=4000)
        if(zl):
            RL.load_model()
        RunDQN._run(env, RL)
        RL.save_model()
        end = time.process_time()
        if(printflag):
            print("程序运行时间为：",(end-start),"秒")
            print("程序运行时间为：", (end-start) / 60, "分钟")
            print("程序运行时间为：", (end-start) / 3600, "小时")
            RL.plot_cost()
            RL.plot_cost('acc')
            RL.plot_hist()




    @classmethod
    def run_prec_dactylology(cls,filePath):
        env = dactyloloEnvironment(True)
        observation = env.getPrecObservation(filePath)
        RL = DQN(env.n_actions, env.n_features, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=10, memory_size=4000)
        RL.load_model()
        action_index = RL.choose_action_only_experience(observation)
        return env.getNameByActionIndex(action_index)


    @classmethod
    def _run(cls,env, RL, verbose=20, start_learn_step=100, step_interval=5):
        step = 0  # 用来控制什么时候学习


        for episode in range(verbose):
            if(printflag):
                print("第",episode+1,"次模拟")
            # 初始化环境
            observation = env.reset()

            while (True):
                # DQN 根据观测值选择行为
                action = RL.choose_action(observation)

                # 在当前环境下根据行为给出下一个 state, reward, 是否终止标志done
                observation_, reward, done = env.step(action)

                # DQN 存储记忆
                RL.store_transition(observation, action, reward, observation_)

                # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
                if (step > start_learn_step) and (step % step_interval == 0):
                    if (printflag):
                        print("学习中")
                    RL.learn()
                    if (printflag):
                        print("学习结束")

                # 将下一个 state_ 变为 下次循环的 state
                observation = observation_

                # 如果终止, 就跳出循环
                if done:
                    break
                step += 1  # 总步数

                print(step)
            RL.save_model()
            if (printflag):
                print("保存模型")
        if (printflag):
            print('训练结束')

        env.destroy()



if __name__ == "__main__":
    '''
    与后端连接时启动
    filePath = str(sys.argv[1])
    print(RunDQN.run_prec_dactylology(filePath))
    '''
    #训练用
    #RunDQN.run_learn(env=dactyloloEnvironment(),zl=True)

    #测试用
    filePath = r"C:\Users\LiuWenze\Desktop\ResourceSpace\盎"
    print(RunDQN.run_prec_dactylology(filePath))
