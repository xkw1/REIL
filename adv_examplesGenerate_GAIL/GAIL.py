import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ExpertTraj
import numpy as np
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
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

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
        
    def update(self, n_iter, batch_size=100):
        for i in range(n_iter):
            # sample expert transitions
            exp_state, exp_action = self.expert.sample(batch_size)
            exp_state = torch.FloatTensor(exp_state).to(device)
            exp_action = torch.FloatTensor(exp_action).to(device)
            
            # sample expert states for actor
            state, _ = self.expert.sample(batch_size)
            def sample1(num):
                env_name = "BipedalWalker-v2"
                exp_states = np.loadtxt(
                    "./file/adv_vari/{}_adv_expert_states.dat".format(env_name))
                exp_actions = np.loadtxt(
                    "./file/adv_vari/{}_adv_expert_actions.dat".format(env_name))
                state = np.array(exp_states[num])
                action = np.array(exp_actions[num])

                return np.array(state), np.array(action)
            if i % 1 == 0:
                num = random.randint(0, 999)
                mid_num = random.randint(0,98)
                state_num, action_num = sample1(num)
                action_number = torch.FloatTensor(action_num).to(device)
                state_num = torch.FloatTensor(state_num).to(device)
                state_num = torch.reshape(state_num, (1, 24))
                action_number = torch.reshape(action_number, (1,4))
                state = torch.FloatTensor(state).to(device)
                action = self.actor(state)
                state = torch.cat((state[:mid_num], state_num, state[mid_num+1:]), 0)
                action = torch.cat((action[:mid_num], action_number, action[mid_num+1:]), 0)

            else:
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
            
            
    def save(self, directory='./preTrained', name='GAIL'):
        torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(directory,name))
        torch.save(self.discriminator.state_dict(), '{}/{}_discriminator.pth'.format(directory,name))
        
    def load(self, directory='./preTrained', name='GAIL'):
        self.actor.load_state_dict(torch.load('{}/{}_actor.pth'.format(directory,name)))
        self.discriminator.load_state_dict(torch.load('{}/{}_discriminator.pth'.format(directory,name)))
