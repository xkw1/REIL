import gym
from GAIL_adv1_1 import GAIL
from PIL import Image
import os
import json
import shutil
from utils import ExpertTraj
from copy import deepcopy
import numpy as np
import torch

list = ['FGSM','PGD', 'MiFGSM', 'random_noise']
x = [0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.51, 0.61, 0.91, 0.99, 1.01, 2.01, 3.01, 4.01, 5.01]
y = [0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.53, 0.63, 0.93, 1.01, 1.03, 2.03, 3.03, 4.03, 5.03]
list2 = [3]
for j in range(4):
    if j == 3:
        c = 0  # if c=0 执行random_noise攻击方法    c=1 执行其他攻击方法
    else:
        c = 1
    method = list[j]
    for i in range(len(x)):
        a = x[i]
        b = y[i]
        if a == 0.13 and b == 0.15:
            L = str(0.14)
            L0 = '0_14'
        elif a == 0.15 and b == 0.17:
            L = str(0.16)
            L0 = '0_16'
        elif a == 0.17 and b == 0.19:
            L = str(0.18)
            L0 = '0_18'
        elif a == 0.19 and b == 0.21:
            L = str(0.20)
            L0 = '0_20'
        elif a == 0.21 and b == 0.23:
            L = str(0.22)
            L0 = '0_22'
        elif a == 0.23 and b == 0.25:
            L = str(0.24)
            L0 = '0_24'
        elif a == 0.25 and b == 0.27:
            L = str(0.26)
            L0 = '0_26'
        elif a == 0.27 and b == 0.29:
            L = str(0.28)
            L0 = '0_28'
        elif a == 0.29 and b == 0.31:
            L = str(0.30)
            L0 = '0_30'
        elif a == 0.51 and b == 0.53:
            L = str(0.52)
            L0 = '0_52'
        elif a == 0.61 and b == 0.63:
            L = str(0.62)
            L0 = '0_62'
        elif a == 0.91 and b == 0.93:
            L = str(0.92)
            L0 = '0_92'
        elif a == 0.99 and b == 1.01:
            L = str(1.00)
            L0 = '1_00'
        elif a == 1.01 and b == 1.03:
            L = str(1.02)
            L0 = '1_02'
        elif a == 2.01 and b == 2.03:
            L = str(2.02)
            L0 = '2_02'
        elif a == 3.01 and b == 3.03:
            L = str(3.02)
            L0 = '3_02'
        elif a == 4.01 and b == 4.03:
            L = str(4.02)
            L0 = '5_02'
        elif a == 5.01 and b == 5.03:
            L = str(5.02)
            L0 = '5_02'
        type1 = 'adv_vari'
        output = 'file'
        output_dir = os.path.join(output, str(type1))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        def list2Str(data,split):
            result = ""
            for item in data:
                result = result + str(item)
                result = result + split
            return result[0:len(result)-1]

        def test():
            env_name = "BipedalWalker-v2"
            #env_name = "LunarLanderContinuous-v2"
            random_seed = 0
            lr = 0.0002
            betas = (0.5, 0.999)

            directory = "/home/wangxueke/wxk/code/Deterministic-GAIL-PyTorch-CE2/preTrained/{}".format(env_name)
            filename = "GAIL_{}_{}".format(env_name, random_seed)

            env = gym.make(env_name)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            max_action = float(env.action_space.high[0])
            # 加载模型
            policy = GAIL(env_name, state_dim, action_dim, max_action, lr, betas)
            policy.load(directory, filename)
            env_name = "BipedalWalker-v2"
            exp_states = np.loadtxt(
                "/home/wangxueke/wxk/code/Deterministic-GAIL-PyTorch-CE2/expert_traj/{}/{}_expert_states.dat".format(env_name,
                                                                                                                     env_name))
            for i in range(500):
                state = deepcopy(exp_states[i])
                action, state = policy.select_action(state, c, j, a, b ,output_dir)
                action = np.array(action, copy=False)
                state = state.cpu().detach().numpy()

                with open(os.path.join(output_dir, r'BipedalWalker-v2_adv_expert_actions.dat'), 'a', encoding='utf-8') as outfile:
                    outfile.write(list2Str(action, '\t') + '\n')
                with open(os.path.join(output_dir, r'BipedalWalker-v2_adv_expert_states.dat'), 'a', encoding='utf-8') as outfile:
                    outfile.write(list2Str(state[0], '\t') + '\n')

        if __name__ == '__main__':
            test()


