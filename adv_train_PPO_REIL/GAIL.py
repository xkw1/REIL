import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ExpertTraj
from torch.autograd import Variable
import numpy as np
import gym
from ppo import PPO, Memory
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
# generate
class generator(nn.Module):
    def __init__(self, state_dim):
        super(generator, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, state_dim)

    def forward(self, state):
        ad_state = torch.cat([state], 1)
        x = F.relu(self.l1(ad_state))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x

class GAIL:
    def __init__(self, env_name, state_dim, action_dim, max_action, lr, betas):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=betas)
        
        self.discriminator = Discriminator(state_dim, action_dim).to(device)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        self.G = generator(state_dim).to(device)
        self.optim_generator = torch.optim.Adam(self.G.parameters(), lr=lr, betas=betas)

        self.max_action = max_action
        self.expert = ExpertTraj(env_name)
        
        self.loss_fn = nn.BCELoss()

    def perturbation_select(self, batch_size, state_dim):
        z = Variable(torch.randn(batch_size, state_dim)).to(device)
        ad_action = self.actor(z).cpu().data.numpy().flatten()
        return self.G(z), ad_action

    def select_action(self, state):
        state = torch.DoubleTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, n_iter, batch_size,ppo,env0, memory):
        env_name = "BipedalWalker-v2"
        env = gym.make(env_name)
        discriminator_loss0 = 0
        loss_generator0 = 0
        loss_genrrator_total0 = 0
        testLoss0 = 0
        loss_actor0 = 0
        for i in range(n_iter):
            # sample expert transitions
            exp_state, exp_action = self.expert.sample(batch_size)
            exp_state = torch.DoubleTensor(exp_state).to(device)
            exp_action = torch.DoubleTensor(exp_action).to(device)
            
            # sample expert states for actor
            state, _ = self.expert.sample(batch_size)
            state = torch.DoubleTensor(state).to(device)
            action = self.actor(state)
            #### adversarial examples
            state_dim = state.size(1)
            z = Variable(torch.randn(batch_size, state_dim)).to(device)
            ad_state = self.G(z)

            ad_action = self.actor(ad_state)
            
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

            # with ad_state transitions
            prob_ad_policy = self.discriminator(ad_state, ad_action.detach())
            loss += self.loss_fn(prob_ad_policy, policy_label)
            discriminator_loss0 += loss
            # take gradient step
            loss.backward(retain_graph=True)

            self.optim_discriminator.step()
            ################
            # update G
            ################
            self.optim_generator.zero_grad()
            # calculate DRLtest_loss
            state0 = torch.randn(100, 24, requires_grad=False)
            state0.copy_(state)
            testLoss = ppo.return_testLoss(state0,memory,env0)
            env.close()

            prob_generator = self.discriminator(ad_state, ad_action)
            loss_generator = self.loss_fn(prob_generator, policy_label)
            # loss_generator.mean().backward()
            loss_generator_total = loss_generator.mean() - testLoss

            loss_generator0 += loss_generator.mean()
            loss_genrrator_total0 += loss_generator_total
            testLoss0 += testLoss

            loss_generator_total.backward(retain_graph=True)
            self.optim_generator.step()

            ################
            # update policy
            ################
            self.optim_actor.zero_grad()
            
            loss_actor = -self.discriminator(state, action)
            loss_actor.mean().backward()
            loss_actor0 += loss_actor.mean()
            self.optim_actor.step()

            with open('file/discriminator_loss.txt', 'a', encoding='utf-8') as outfile:
                outfile.write(str(loss) + '\n')
            with open('file/loss_generator.txt', 'a', encoding='utf-8') as outfile:
                outfile.write(str(loss_generator.mean() ) + '\n')
            with open('file/testLoss.txt', 'a', encoding='utf-8') as outfile:
                outfile.write(str(testLoss) + '\n')
            with open('file/loss_generator_total.txt', 'a', encoding='utf-8') as outfile:
                outfile.write(str(loss_generator_total) + '\n')
            with open('file/loss_actor.txt', 'a', encoding='utf-8') as outfile:
                outfile.write(str(loss_actor.mean()) + '\n')

        def re_average(x):
            ave_x = x / n_iter
            return ave_x

        discriminator_loss0 = re_average(discriminator_loss0)
        loss_generator0 = re_average(loss_generator0 / n_iter)
        testLoss0 = re_average(testLoss0 / n_iter)
        loss_genrrator_total0 = re_average(loss_genrrator_total0 / n_iter)
        loss_actor0 = re_average(loss_actor0 / n_iter)
        with open('file/discriminator_loss_inter.txt', 'a', encoding='utf-8') as outfile:
            outfile.write(str(discriminator_loss0) + '\n')
        with open('file/loss_generator_inter.txt', 'a', encoding='utf-8') as outfile:
            outfile.write(str(loss_generator0) + '\n')
        with open('file/testLoss_inter.txt', 'a', encoding='utf-8') as outfile:
            outfile.write(str(testLoss0) + '\n')
        with open('file/loss_generator_total_inter.txt', 'a', encoding='utf-8') as outfile:
            outfile.write(str(loss_genrrator_total0) + '\n')
        with open('file/loss_actor_inter.txt', 'a', encoding='utf-8') as outfile:
            outfile.write(str(loss_actor0) + '\n')

    def save(self, directory='./preTrained', name='GAIL', epoch=200):
        torch.save(self.actor.state_dict(), '{}/{}_actor{}.pth'.format(directory,name,epoch))
        torch.save(self.discriminator.state_dict(), '{}/{}_discriminator{}.pth'.format(directory,name,epoch))
        torch.save(self.G.state_dict(), '{}/{}_generator{}.pth'.format(directory,name,epoch))

    def load(self, directory='./preTrained', name='GAIL', epoch=200):
        self.actor.load_state_dict(torch.load('{}/{}_actor{}.pth'.format(directory,name,epoch)))
        self.discriminator.load_state_dict(torch.load('{}/{}_discriminator{}.pth'.format(directory,name,epoch)))
        self.G.load_state_dict(torch.load('{}/{}_generator{}.pth'.format(directory,name,epoch)))