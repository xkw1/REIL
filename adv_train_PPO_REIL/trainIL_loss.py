import torch
import gym
import numpy as np
from GAIL import GAIL
import matplotlib.pyplot as plt
from ppo import PPO, Memory
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import datetime
start = datetime.datetime.now()

env_name = "BipedalWalker-v2"
env0 = gym.make(env_name)
state_dim = env0.observation_space.shape[0]
action_dim = env0.action_space.shape[0]
n_episodes = 10  # num of episodes to run
max_timesteps = 1500  # max timesteps in one episode
render = False  # render the environment
save_gif = False  # png images are saved in gif folder

filename = "PPO_continuous_" + env_name+ "_4000" + ".pth"
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
env0.reset()

def Generate_pertur():
    # from GAIL_test import GAIL
    from GAIL import GAIL
    epochi = 1800
    random_seed = 0
    batch_size = 1

    directory = "./preTrained/{}".format(env_name)
    filename = "GAIL_{}_{}".format(env_name, random_seed)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = GAIL(env_name, state_dim, action_dim, max_action, lr, betas)

    policy.load(directory, filename, epochi)
    ad_state, ad_action = policy.perturbation_select(batch_size, state_dim)
    print ('ad_state', ad_state)
    print ('ad_action', ad_action)
    return ad_state, ad_action

def train():
    ######### Hyperparameters #########
    env_name = "BipedalWalker-v2"
    # env_name = "LunarLanderContinuous-v2"
    solved_reward = 300  # stop training if solved_reward > avg_reward
    random_seed = 0
    # max_timesteps = 1400        # max time steps in one episode
    max_timesteps = 3000  # max time steps in one episode
    n_eval_episodes = 20  # evaluate average reward over n episodes
    lr = 0.0002  # learing rate
    betas = (0.5, 0.999)  # betas for adam optimizer
    n_epochs = 5000  # number of epochs
    n_iter = 100  # updates per epoch
    batch_size = 100  # num of transitions sampled from expert
    directory = "./preTrained/{}".format(env_name)  # save trained models
    filename = "GAIL_{}_{}".format(env_name, random_seed)
    ###################################

    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = GAIL(env_name, state_dim, action_dim, max_action, lr, betas)
    # graph logging variables:
    epochs = []
    rewards = []

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    # training procedure
    for epoch in range(1, n_epochs + 1):
        # update policy n_iter times
        policy.update(n_iter, batch_size, ppo, env0, memory)
        # evaluate in environment
        total_reward = 0
        for episode in range(n_eval_episodes):
            state = env.reset()
            for t in range(max_timesteps):
                action = policy.select_action(state)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                # env.render()
                if done:
                    break
            with open('file/total_reward.txt', 'a', encoding='utf-8') as outfile:
                outfile.write(str(t) + " " + str(total_reward) + '\n')

        avg_reward = int(total_reward / n_eval_episodes)
        print("Epoch: {}\tAvg Reward: {}".format(epoch, avg_reward))

        with open('file/avg_reward.txt', 'a', encoding='utf-8') as outfile:
            outfile.write(str(avg_reward) + '\n')

        # add data for graph
        epochs.append(epoch)
        rewards.append(avg_reward)

        # if avg_reward > solved_reward:
        if epoch % 200 == 0:
            policy.save(directory, filename, epoch)

    # 取每50回合的平均值#########################################
    len_rewards = int(len(rewards) / 50)
    sum = 0
    aves_reward = []
    for i in range(len_rewards):
        j = rewards[50 * i:50 * (i + 1)]
        for k in j:
            sum = sum + k
        ave = int(sum / 50)
        sum = 0
        aves_reward.append(ave)
        with open('aves_reward.txt', 'a', encoding='utf-8') as outfile:
            outfile.write(str(ave) + '\n')

    len_aves_reward = len(aves_reward)
    aves_epochs = epochs[:len_aves_reward]

    plt.plot(aves_epochs, aves_reward)
    plt.xlabel('Epochs')
    plt.ylabel('Average Reward')
    plt.title('{}'.format(env_name))
    plt.savefig('./gif/graph_ave50_{}.png'.format(env_name))
    ####################################################################
    # plot and save graph
    plt.plot(epochs, rewards)
    plt.xlabel('Epochs')
    plt.ylabel('Average Reward')
    plt.title('{}  {}  {} '.format(env_name, lr, betas))
    plt.savefig('./gif/graph_{}.png'.format(env_name))
    end = time.time()
    time2 = []
    time1 = end - start
    time2.append(start)
    time2.append(end)
    time2.append(time1)
    with open('file/time_end.txt', 'a', encoding='utf-8') as outfile:
        outfile.write(str(time2) + '\n')


if __name__ == '__main__':
    train()
