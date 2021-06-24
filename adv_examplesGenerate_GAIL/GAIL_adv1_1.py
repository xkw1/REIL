import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ExpertTraj
from torch.nn import init, Parameter   ##############
from torch.autograd import Variable     ##############
import math                              ##############
import numpy as np
from copy import deepcopy
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x = 'FGSM_01_012'

class NoisyLinear(nn.Linear):
  def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
    super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
    # µ^w and µ^b reuse self.weight and self.bias
    self.sigma_init = sigma_init
    self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
    self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
    self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
    self.register_buffer('epsilon_bias', torch.zeros(out_features))
    self.reset_parameters()

  def reset_parameters(self):
    if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
      init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      init.constant(self.sigma_weight, self.sigma_init)
      init.constant(self.sigma_bias, self.sigma_init)

  def forward(self, input):
    return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), self.bias + self.sigma_bias * Variable(self.epsilon_bias))

  def sample_noise(self):
    self.epsilon_weight = torch.randn(self.out_features, self.in_features)
    self.epsilon_bias = torch.randn(self.out_features)

  def remove_noise(self):
    self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
    self.epsilon_bias = torch.zeros(self.out_features)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        # self.l3 = NoisyLinear(300, action_dim, sigma_init=0.017)
        self.max_action = max_action
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.max_action
        return x


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        
        self.l1 = nn.Linear(state_dim+action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
    
    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        x = torch.tanh(self.l1(state_action))
        x = torch.tanh(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x
    
    
class GAIL:
    def __init__(self, env_name, state_dim, action_dim, max_action, lr, betas):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=betas)
        
        self.discriminator = Discriminator(state_dim, action_dim).to(device)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        
        self.max_action = max_action
        self.expert = ExpertTraj(env_name)
        
        self.loss_fn = nn.BCELoss()

    def FGSM(self, state):
        batch_size = 1
        # state = torch.FloatTensor(state).to(device)
        state.requires_grad_(True)
        action = self.actor(state)
        #######################
        # update discriminator
        #######################
        self.optim_discriminator.zero_grad()
        # label tensors
        policy_label = torch.full((batch_size,1), 0, device=device)
        prob_policy = self.discriminator(state, action.detach())
        # for param in self.discriminator.parameters():
        #     param.requires_grad_(False)
        # for param in self.actor.parameters():
        #     param.requires_grad_(False)
        loss = self.loss_fn(prob_policy, policy_label)
        # take gradient step
        loss.backward()
        # loss_actor = -self.discriminator(state, action)
        # loss_actor.mean().backward()
        # self.optim_actor.step()
        data_grad = state.grad.data
        sign_data_grad = data_grad.sign()   #参考pytorch的FGSM ： https://blog.csdn.net/zhjm07054115/article/details/104831924
        return sign_data_grad

    def FGM(self, state):  # FGM VS FGSM  https://blog.csdn.net/tangyaruo/article/details/103423542
        batch_size = 1
        # state = torch.FloatTensor(state).to(device)
        state.requires_grad_(True)
        action = self.actor(state)
        #######################
        # update discriminator
        #######################
        self.optim_discriminator.zero_grad()
        # label tensors
        policy_label = torch.full((batch_size,1), 0, device=device)
        prob_policy = self.discriminator(state, action.detach())
        # for param in self.discriminator.parameters():
        #     param.requires_grad_(False)
        # for param in self.actor.parameters():
        #     param.requires_grad_(False)
        loss = self.loss_fn(prob_policy, policy_label)
        # take gradient step
        loss.backward()
        # loss_actor = -self.discriminator(state, action)
        # loss_actor.mean().backward()
        # self.optim_actor.step()
        data_grad = state.grad.data
        # 添加剪切以维持【0, 1]范围   https://blog.csdn.net/weixin_42182906/article/details/107986371
        # data_grad = torch.clamp(data_grad, 0, 1)   # 0和1表示限制范围的下限和上限   https://blog.csdn.net/xiaoqiaoliushuiCC/article/details/105471056?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~first_rank_v2~rank_v25-18-105471056.nonecase&utm_term=pytorch样本攻击
        return data_grad

    def PGD(self, state, pertubation):
        clip_min = 0
        clip_max = 1.0
        ord = np.inf
        eps = 0.3
        adv_state = state + pertubation
        pertubation = self.FGSM(state)
        adv_state = adv_state + pertubation
        state = state.cpu().detach().numpy()
        adv_state = adv_state.cpu().detach().numpy()

        pertubation = np.clip(adv_state, clip_min, clip_max) - state
        # pertubation = pertubation.cpu().numpy()
        pertubation = np.clip(pertubation, ord, eps)
        return pertubation

    def mifgsm(self, state):
        nb_iter = 10
        g = 0
        delta = 0
        s_t_temp = deepcopy(state)
        for i in range(nb_iter):
            # print(delta)
            decay_factor = 1
            iter_eps = 0.0005 / 20
            disrupt_magnitude = 0.0005
            clip_min = 0.0
            clip_max = 1.0
            img_adv = s_t_temp + delta
            g_plus = self.FGM(img_adv)
            g_plus = g_plus / torch.norm(g_plus, p=1)
            g = decay_factor * g + g_plus
            # print(g_plus)
            # g[np.isnan(g)] = 0
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

    def select_action(self, state, c, j, a, b, output_dir):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if c == 1:
            if j == 0:
                att = self.FGSM(state)
            # elif j == 1:
            #     att = self.FGM(state)
            elif j == 1:
                # att = self.PGD(state)
                pertubation = torch.zeros(state.shape).type_as(state).to(device)
                for i in range(10):
                    pertubation = self.PGD(state, pertubation=pertubation)
                    pertubation = torch.Tensor(pertubation).type_as(state).to(device)
                att = pertubation
            elif j == 2:
                att = self.mifgsm(state)
            # ll = torch.from_numpy(att)
            att = 0.1 * att
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
            with open(os.path.join(output_dir, r"e.txt"), "a", encoding='utf-8') as outfile:
                outfile.write(str(e) + '\n')
        else:
            if j == 3:
                att, e = self.ran_noise(a, b)
                # att = l
                att = 0.1 * att
                with open(os.path.join(output_dir, r"e.txt"), "a", encoding='utf-8') as outfile:
                    outfile.write(str(e) + '\n')
                att = torch.FloatTensor(att).to(device)
        state1 = state + att

        return self.actor(state).cpu().data.numpy().flatten(), state1

    # def select_action(self, state):
    #     state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    #     # att = self.FGSM(state)
    #     # att = self.FGM(state)
    #     # att = self.PGD(state)
    #     att = self.mifgsm(state)
    #     # ll = torch.from_numpy(att)
    #     att = 0.1 * att
    #     e = torch.norm(att, p=2)
    #     # a = 0.2
    #     # b = 0.3
    #     a = 0.09  # FGSM
    #     b = 0.11  #
    #     while e < a or e > b:
    #         if e < a:
    #             att = 1.1 * att
    #             e = torch.norm(att, p=2)
    #             continue
    #         if e > b:
    #             att = 0.9 * att
    #             e = torch.norm(att, p=2)
    #             continue
    #         e = torch.norm(att, p=2)
    #     # print(e)
    #     with open('file/adv/e.txt', 'a', encoding='utf-8') as outfile:
    #         outfile.write(str(e) + '\n')
    #     state = state + att
    #     return self.actor(state).cpu().data.numpy().flatten()
        
    def update(self, n_iter, batch_size=100):
        for i in range(n_iter):
            # sample expert transitions
            exp_state, exp_action = self.expert.sample(batch_size)
            exp_state = torch.FloatTensor(exp_state).to(device)
            exp_action = torch.FloatTensor(exp_action).to(device)
            
            # sample expert states for actor
            state, _ = self.expert.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = self.actor(state)
            
            #######################
            # update discriminator
            #######################
            self.optim_discriminator.zero_grad()
            
            # label tensors
            exp_label= torch.full((batch_size,1), 1, device=device)
            policy_label = torch.full((batch_size,1), 0, device=device)
            
            # with expert transitions
            prob_exp = self.discriminator(exp_state, exp_action)
            loss = self.loss_fn(prob_exp, exp_label)

            # with policy transitions
            prob_policy = self.discriminator(state, action.detach())
            loss += self.loss_fn(prob_policy, policy_label)

            # take gradient step
            loss.backward()
            self.optim_discriminator.step()
            
            ################
            # update policy
            ################
            self.optim_actor.zero_grad()
            
            loss_actor = -self.discriminator(state, action)
            loss_actor.mean().backward()
            self.optim_actor.step()
            
            
    # def save(self, directory='./preTrained', name='GAIL'):
    #     torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(directory,name))
    #     torch.save(self.discriminator.state_dict(), '{}/{}_discriminator.pth'.format(directory,name))
        
    def load(self, directory='/home/wangxueke/wxk/code/Deterministic-GAIL-PyTorch-CE2/10000-20-3/preTrained', name='GAIL'):
        self.actor.load_state_dict(torch.load('{}/{}_actor.pth'.format(directory,name)))
        self.discriminator.load_state_dict(torch.load('{}/{}_discriminator.pth'.format(directory,name)))
