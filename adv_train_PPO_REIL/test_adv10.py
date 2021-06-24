import gym
from ppo_adv import PPO, Memory
from PIL import Image
import torch
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cpu")
torch.set_default_tensor_type(torch.DoubleTensor)

from PIL import Image
import json
import shutil
import gym
import numpy as np
import random
import os
import warnings
from copy import deepcopy
warnings.filterwarnings('ignore')
K = '_adppo10'
N = '_7000'
directory = "./preTrained_advppo10/"

list = ['FGSM','PGD', 'MiFGSM', 'random_noise']
list2 = [0, 1, 2]
x = [0.01, 0.03, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.51, 0.61, 0.91]#, 0.99, 1.01, 2.01, 3.01, 4.01, 5.01]
y = [0.03, 0.05, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.53, 0.63, 0.93]#, 1.01, 1.03, 2.03, 3.03, 4.03, 5.03]
def noiseExampleSelect(i,j):

    if j == 3:
        c = 0   # if c=0 执行random_noise攻击方法    c=1 执行其他攻击方法
    else:
        c = 1
    method = list[j]
    a = x[i]
    b = y[i]
    if a == 0.01 and b == 0.03:
        L = str(0.02)
        L0 = '0_02'
    elif a == 0.03 and b == 0.05:
        L = str(0.04)
        L0 = '0_04'
    elif a == 0.05 and b == 0.07:
        L = str(0.06)
        L0 = '0_06'
    elif a == 0.07 and b == 0.09:
        L = str(0.08)
        L0 = '0_08'
    elif a == 0.09 and b == 0.11:
        L = str(0.10)
        L0 = '0_1'
    elif a == 0.11 and b == 0.13:
        L = str(0.12)
        L0 = '0_12'
    elif a == 0.13 and b == 0.15:
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
    print (a, b, L, method)
    type1 = 'adv'
    output = 'file_ppo10'
    output_dir1 = os.path.join(output, str(type1 + L0))
    output_dir = os.path.join(output_dir1, str(method))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return c, j, a, b, output_dir

for j in list2:
    if j == 3:
        c = 0
    else:
        c = 1
    method = list[j]
    for i in range(len(x)):
        c, j, ax, by, output_dir = noiseExampleSelect(i, j)
        def test():
            ############## Hyperparameters ##############
            env_name = "BipedalWalker-v2"
            env = gym.make(env_name)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            # filename and directory to load model from
            filename = "PPO_continuous_" + env_name + N + ".pth"

            n_episodes = 100  # num of episodes to run
            max_timesteps = 1500  # max timesteps in one episode
            render = False  # render the environment
            save_gif = False  # png images are saved in gif folder

            action_std = 0.5  # constant std for action distribution (Multivariate Normal)
            # K_epochs = 80  # update policy for K epochs
            K_epochs = 1  # update policy for K epochs
            eps_clip = 0.2  # clip parameter for PPO
            gamma = 0.99  # discount factor

            lr = 0.0003  # parameters for Adam optimizer
            betas = (0.9, 0.999)
            #############################################

            memory = Memory()
            memory0 = Memory()
            ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
            ppo.policy_old.load_state_dict(torch.load(directory + filename))

            for ep in range(1, n_episodes + 1):
                ep_reward = 0
                time_step = 0
                state = env.reset()
                for t in range(max_timesteps):
                    time_step += 1
                    a = ppo.select_action0(state, memory)
                    state0 = state
                    state1, reward0, done0, _ = env.step(a)
                    state = ppo.select_action1(state0, a, reward0, memory, done0, c, j, ax, by)
                    memory.clear_memory()
                    action = ppo.select_action(state, memory0)
                    state, reward, done, _ = env.step(action)
                    ep_reward += reward
                    with open(os.path.join(output_dir, r"avg_action{}.txt").format(K), "a", encoding='utf-8') as outfile:
                        outfile.write(str(action) + '\n')
                    if render:
                        env.render()
                    if save_gif:
                        img = env.render(mode='rgb_array')
                        img = Image.fromarray(img)
                        img.save('./gif/{}.jpg'.format(t))
                    if done:
                        break
                with open(os.path.join(output_dir, r"avg_reward{}.txt").format(K), "a", encoding='utf-8') as outfile:
                    outfile.write(str(ep_reward) + '\n')
                with open(os.path.join(output_dir, r"avg_action{}.txt").format(K), "a", encoding='utf-8') as outfile:
                    outfile.write(str(action) + '\n')
                print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
                ep_reward = 0
                time_step = 0
                env.close()

        if __name__ == '__main__':
            test()
