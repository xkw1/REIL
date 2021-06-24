import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
import warnings
from copy import deepcopy
warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cpu")
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()
    def act0(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = torch.squeeze(self.actor(state))

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    torch.set_default_tensor_type(torch.DoubleTensor)

    def FGSM(self, state, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)   #计算折扣奖励
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards).to(device)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()
        old_states.requires_grad_(True)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            data_grad = old_states.grad.data
            sign_data_grad = data_grad.sign()   #参考pytorch的FGSM ： https://blog.csdn.net/zhjm07054115/article/details/104831924
        return sign_data_grad

    def FGM(self, state, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards).to(device)
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()
        old_states.requires_grad_(True)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            data_grad = old_states.grad.data
        return data_grad

    def PGD(self, state, pertubation, memory):
        clip_min = 0
        clip_max = 1.0
        ord = np.inf
        eps = 0.2
        adv_state = state + pertubation
        pertubation = self.FGSM(state, memory)
        adv_state = adv_state + pertubation
        state = state.cpu().detach().numpy()
        adv_state = adv_state.cpu().detach().numpy()

        pertubation = np.clip(adv_state, clip_min, clip_max) - state
        pertubation = np.clip(pertubation, eps, ord)
        return pertubation

    def mifgsm(self, state, memory):
        nb_iter = 5
        g = 0
        delta = 0
        s_t_temp = deepcopy(state)
        for i in range(nb_iter):
            # print(delta)
            decay_factor = 1
            iter_eps = 0.0005 / 5
            disrupt_magnitude = 0.0005
            clip_min = 0.0
            clip_max = 1.0
            img_adv = s_t_temp + delta
            g_plus = self.FGM(img_adv, memory)
            g_plus = g_plus / torch.norm(g_plus, p=1)
            g = decay_factor * g + g_plus
            delta += iter_eps * g.sign()
            delta = torch.clamp(delta, -disrupt_magnitude, disrupt_magnitude)
            delta = torch.clamp(state + delta, clip_min, clip_max) - state
        sign_g_grad = g.sign()
        return sign_g_grad
    def ran_noise(self, a, b):
        if a == 0.01 and b == 0.03:
            s = 0.0014
        elif a == 0.03 and b == 0.05:
            s = 0.0025
        elif a == 0.05 and b == 0.07:
            s = 0.0038
        elif a == 0.07 and b == 0.09:
            s = 0.0052
        elif a == 0.09 and b == 0.11:
            s = 0.0072
        elif a == 0.11 and b == 0.13:
            s = 0.008
        elif a == 0.13 and b == 0.15:
            s = 0.01
        elif a == 0.15 and b == 0.17:
            s = 0.012
        elif a == 0.17 and b == 0.19:
            s = 0.014
        elif a == 0.19 and b == 0.21:
            s = 0.016
        elif a == 0.21 and b == 0.23:
            s = 0.018
        elif a == 0.23 and b == 0.25:
            s = 0.021
        elif a == 0.25 and b == 0.27:
            s = 0.023
        elif a == 0.27 and b == 0.29:
            s = 0.025
        elif a == 0.29 and b == 0.31:
            s = 0.027
        elif a == 0.51 and b == 0.53:
            s = 0.038
        elif a == 0.61 and b == 0.63:
            s = 0.048
        elif a == 0.91 and b == 0.93:
            s = 0.056
        elif a == 0.99 and b == 1.01:
            s = 0.06
        elif a == 1.01 and b == 1.03:
            s = 0.062
        elif a == 2.01 and b == 2.03:
            s = 0.11
        elif a == 3.01 and b == 3.03:
            s = 0.2
        elif a == 4.01 and b == 4.03:
            s = 0.29
        elif a == 5.01 and b == 5.03:
            s = 0.34
        e = 0
        while e <= a or e >= b:
            l = s * np.random.uniform(1, 5, (24,))
            ll = torch.from_numpy(l)
            e = torch.norm(ll, p=2)
            if e <= a or e >= b:
                continue
        return l, e

    def select_action(self, state, memory):
        state = torch.DoubleTensor(state.reshape(1, -1)).to(device)
        a = self.policy_old.act(state, memory).cpu().data.numpy().flatten()
        return a

    def select_action1(self, state, action, reward, memory, done, c, j, a, b):
        state = torch.DoubleTensor(state.reshape(1, -1)).to(device)
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        if c == 1:
            if j == 0:
                att = self.FGSM(state, memory)
            elif j == 1:
                pertubation = torch.zeros(state.shape).type_as(state).to(device)
                for i in range(5):
                    pertubation = self.PGD(state, pertubation, memory)
                    pertubation = torch.Tensor(pertubation).type_as(state).to(device)

                att = pertubation
            elif j == 2:
                att = self.mifgsm(state, memory)
            e = torch.norm(att, p=2)
            while e < a or e > b:
                if e < a:
                    att = 1.1 * att
                    e = torch.norm(att, p=2)
                    continue
                if e > b:
                    att = 0.9 * att
                    e = torch.norm(att, p=2)
                    continue
                e = torch.norm(att, p=2)
        else:
            if j == 3:
                att, e = self.ran_noise(a, b)
                att = torch.FloatTensor(att).to(device)
        state1 = state + att
        return state1
    def select_action0(self, state, memory):
        state = torch.DoubleTensor(state.reshape(1, -1)).to(device)
        a = self.policy_old.act0(state, memory).cpu().data.numpy().flatten()
        return a

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


def main():
    ############## Hyperparameters ##############
    env_name = "BipedalWalker-v2"
    render = False
    solved_reward = 300  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 10000  # max training episodes
    max_timesteps = 1000  # max timesteps in one episode

    update_timestep = 3000  # update policy every n timesteps
    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.0003  # parameters for Adam optimizer
    betas = (0.9, 0.999)
    retain = True

    random_seed = None
    #############################################
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    if retain:
        start = max_episodes + 1
        end = start + max_episodes+1
        filename = "PPO_continuous_" + env_name + "_9500" + ".pth"
        directory = "./"
        try:
            ppo.policy_old.load_state_dict(torch.load(directory + filename), strict=True)
            print("加载模型成功!继续训练模型")
        except:
            print("加载模型失败!开始训练一个新模型")
            
    else:
        start = 1
        end = max_timesteps + 1

    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(start, end):
        state = env.reset()
        for t in range(max_timesteps):
            time_step += 1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
            break

        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}_{}.pth'.format(env_name, i_episode))


        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            with open('file/Episode.txt', 'a', encoding='utf-8') as outfile:
                outfile.write(str(i_episode) + '\n')
            with open('file/avg_reward.txt', 'a', encoding='utf-8') as outfile:
                outfile.write(str(running_reward) + '\n')
            with open('file/avg_length.txt', 'a', encoding='utf-8') as outfile:
                outfile.write(str(avg_length) + '\n')
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()
