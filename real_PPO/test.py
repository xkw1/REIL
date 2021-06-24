import gym
from ppo import PPO, Memory
from PIL import Image
import torch
# from AdvStateGenerate import generate
import warnings
warnings.filterwarnings('ignore')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.set_default_tensor_type(torch.DoubleTensor)
# print(1111111111111111111)

# from GAIL_adv1_1 import GAIL
# from PIL import Image
# import json
# import shutil
# import gym
# import numpy as np
# import random
# import os
# import warnings
# from copy import deepcopy
# warnings.filterwarnings('ignore')
# from utils import ExpertTraj
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# def noiseExampleSelect(i,j):
#     # c = 1   # if c=0 执行random_noise攻击方法    c=1 执行其他攻击方法
#     list = ['FGSM','PGD', 'MiFGSM', 'random_noise']
#     x = [0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29]
#     y = [0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31]
#     # list2 = [3,0,1,2]
#     # # for j in range(4):
#     # for j in list2:
#     # # list22 = [2]
#     # # for j in list22:
#     if j == 3:
#         c = 0
#     else:
#         c = 1
#     method = list[j]
#     # for i in range(1):
#     a = x[i]
#     b = y[i]
#     if a == 0.01 and b == 0.03:
#         L = str(0.02)
#         L0 = '0_02'
#     elif a == 0.03 and b == 0.05:
#         L = str(0.04)
#         L0 = '0_04'
#     elif a == 0.05 and b == 0.07:
#         L = str(0.06)
#         L0 = '0_06'
#     elif a == 0.07 and b == 0.09:
#         L = str(0.08)
#         L0 = '0_08'
#     elif a == 0.09 and b == 0.11:
#         L = str(0.10)
#         L0 = '0_1'
#     elif a == 0.11 and b == 0.13:
#         L = str(0.12)
#         L0 = '0_12'
#     elif a == 0.13 and b == 0.15:
#         L = str(0.14)
#         L0 = '0_14'
#     elif a == 0.15 and b == 0.17:
#         L = str(0.16)
#         L0 = '0_16'
#     elif a == 0.17 and b == 0.19:
#         L = str(0.18)
#         L0 = '0_18'
#     elif a == 0.19 and b == 0.21:
#         L = str(0.20)
#         L0 = '0_20'
#     elif a == 0.21 and b == 0.23:
#         L = str(0.22)
#         L0 = '0_22'
#     elif a == 0.23 and b == 0.25:
#         L = str(0.24)
#         L0 = '0_24'
#     elif a == 0.25 and b == 0.27:
#         L = str(0.26)
#         L0 = '0_26'
#     elif a == 0.27 and b == 0.29:
#         L = str(0.28)
#         L0 = '0_28'
#     elif a == 0.29 and b == 0.31:
#         L = str(0.30)
#         L0 = '0_30'
#     # print (a, b, L, method)
#     type1 = 'adv'
#     output = 'file'
#     output_dir = os.path.join(output, str(type1))
#     # output_dir = os.path.join(output_dir1, str(method))
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
# #     return c, j, a, b, output_dir
#
# def list2Str(data,split):
#     result = ""
#     for item in data:
#         result = result + str(item)
#         result = result + split
#     #return result.strip()
#     return result[0:len(result)-1]

# def generate(state, policy):
#     # i = random.randint(0,8)
#     # j = random.randint(0,3)
#     i = 8
#     j = 3
#     c, j, a, b, output_dir = noiseExampleSelect(i, j)
#     action, state = policy.select_action(state, c, j, a, b ,output_dir)
#     # print(type(state))
#     # state = np.array(state, copy=False)
#     action = np.array(action, copy=False)
#     state = state.cpu().detach().numpy()
#     # print(type(state))
#     # action = list2Str(action, '\t')
#     # state = list2Str(state[0], '\t')
#     return state

def test():
    ############## Hyperparameters ##############
    env_name = "BipedalWalker-v2"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    n_episodes = 10  # num of episodes to run
    max_timesteps = 1500  # max timesteps in one episode
    render = False  # render the environment
    save_gif = False  # png images are saved in gif folder

    # filename and directory to load model from
    # filename = "PPO_continuous_" + env_name+ "_5000" + ".pth"
    # filename = "PPO_continuous_" + env_name+ "_4000" + ".pth"
    # filename = "PPO_continuous_" + env_name+ "_6000" + ".pth"
    filename = "PPO_continuous_" + env_name+ "_7000" + ".pth"
    directory = "./"

    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.0003  # parameters for Adam optimizer
    betas = (0.9, 0.999)
    #############################################

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy_old.load_state_dict(torch.load(directory + filename))

    #############################################
    # imitation learning model load
    # env_name = "BipedalWalker-v2"
    # random_seed1 = 0
    # lr1 = 0.0002
    # betas1 = (0.5, 0.999)
    # directory1 = "/home/wangxueke/wxk/code/DRL_IL_advtrain/DynamicAdv_train_DRLPPO_FGSM00/preTrained/{}".format(
    #     env_name)
    # filename1 = "GAIL_{}_{}".format(env_name, random_seed1)
    # env = gym.make(env_name)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # max_action = float(env.action_space.high[0])
    # # 加载模型
    # policy = GAIL(env_name, state_dim, action_dim, max_action, lr1, betas1)
    # policy.load(directory1, filename1)
    ##########################
    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        time_step = 0
        state = env.reset()
        for t in range(max_timesteps):
            time_step += 1
            # if time_step % 10 == 0:
            #     state = generate(state, policy)
            #     action = ppo.select_action(state, memory)
            #     state, reward, done, _ = env.step(action)
            # else:
            #     action = ppo.select_action(state, memory)
            #     state, reward, done, _ = env.step(action)
            # state = generate(state, policy)
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            if save_gif:
                img = env.render(mode='rgb_array')
                img = Image.fromarray(img)
                img.save('./gif/{}.jpg'.format(t))
            if done:
                break
        # with open('file/avg_reward.txt', 'a', encoding='utf-8') as outfile:
        #     outfile.write(str(ep_reward) + '\n')
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        time_step = 0
        env.close()


if __name__ == '__main__':
    test()
