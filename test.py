
import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from common.layers import NoisyLinear
from common.replay_buffer import ReplayBuffer
from common.replay_buffer import PrioritizedReplayBuffer

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,**kwargs)


from utils.hyperparameters import Config

config = Config()

config.num_atoms = 51
config.Vmin = -200
config.Vmax = 200

config.num_frames = 400000
config.batch_size = 32
config.gamma = 0.99

config.epsilon_start = 1.0
config.epsilon_final = 0.01
config.epsilon_decay = 500

config.learning_rate=1e-3
config.gamma=0.99
config.buffer_size=10000
config.replace_iter_num=1000
config.use_noisy=False
config.n_step=3

config.prioritized_replay=True
config.prioritized_replay_alpha =0.6
config.prioritized_replay_beta= 0.4

config.hidden_before=[32,64]
config.hidden_after=[64]


class Agent:
    def __init__(self,oEnv):
        self.num_atoms = config.num_atoms
        self.Vmin = config.Vmin
        self.Vmax = config.Vmax

        self.num_frames = config.num_frames
        self.batch_size = config.batch_size
        self.gamma = config.gamma

        self.epsilon_start = config.epsilon_start
        self.epsilon_final =config.epsilon_final
        self.epsilon_decay = config.epsilon_decay

        self.learning_rate = config.learning_rate
        self.gamma=config.gamma
        self.buffer_size = config.buffer_size
        self.replace_iter_num =config.replace_iter_num
        self.use_noisy =config.use_noisy
        self.n_step = config.n_step

        self.prioritized_replay=config.prioritized_replay
        self.prioritized_replay_alpha=config.prioritized_replay_alpha
        self.prioritized_replay_beta=config.prioritized_replay_beta

        self.hidden_before=config.hidden_before
        self.hidden_after=config.hidden_after


        self.epsilon_by_frame = lambda frame_idx: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay)

        self.env=oEnv
        self.nstep_buffer=[]

        if self.prioritized_replay:
            self.replay_buffer=PrioritizedReplayBuffer(self.buffer_size,alpha=self.prioritized_replay_alpha)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.current_model = RainbowDQN(oEnv,oEnv.observation_space.shape[0], oEnv.action_space.n, self.num_atoms,
                                        self.Vmin, self.Vmax,self.use_noisy,hidden_before=self.hidden_before,hidden_after=self.hidden_after)
        self.target_model = RainbowDQN(oEnv,oEnv.observation_space.shape[0], oEnv.action_space.n, self.num_atoms,
                                       self.Vmin, self.Vmax,self.use_noisy,hidden_before=self.hidden_before,hidden_after=self.hidden_after)
        if USE_CUDA:
            self.current_model = self.current_model.cuda()
            self.target_model = self.target_model.cuda()

        self.optimizer = optim.Adam(self.current_model.parameters(), self.learning_rate)

        self.update_target(self.current_model, self.target_model)


    def update_target(self,current_model, target_model):
            target_model.load_state_dict(current_model.state_dict())

    def projection_distribution(self,next_state, rewards, dones):
        batch_size = next_state.size(0)

        delta_z = float(self.Vmax - self.Vmin) / (self.num_atoms - 1)
        support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms)

        next_dist = self.target_model(next_state).data.cpu()

        next_dist_action = next_dist * support
        # print("next",next_dist.shape,next_dist_action.shape)
        next_action = next_dist_action.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
        next_dist = next_dist.gather(1, next_action).squeeze(1)

        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones = dones.unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)

        Tz = rewards + (1 - dones) * self.gamma * support
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
        b = (Tz - self.Vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()
        equal = l == u
        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).long().unsqueeze(1).expand(batch_size,self.num_atoms)

        proj_dist = torch.zeros(next_dist.size())
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1),
                                      (next_dist * (u.float() - b + equal.float())).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        return proj_dist

    def compute_td_loss(self,batch_size):
        if self.prioritized_replay:
            state, action, reward, next_state, done, weights, indices = self.replay_buffer.sample(batch_size, self.prioritized_replay_beta)
            weights=Variable(torch.FloatTensor(weights))
        else:
            state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
            weights=164

        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        action = Variable(torch.LongTensor(action))

        if self.num_atoms > 1:
            reward = torch.FloatTensor(reward)
            done = torch.FloatTensor(np.float32(done))
            proj_dist = self.projection_distribution(next_state, reward, done)
            dist = self.current_model(state)
            action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, self.num_atoms)
            dist = dist.gather(1, action).squeeze(1)
            dist.data.clamp_(0.01, 0.99)
            loss = -(Variable(proj_dist) * dist.log()).sum(1)*weights
            prios = loss + 1e-5
        else:
            reward = Variable(torch.FloatTensor(reward))
            done = Variable(torch.FloatTensor(np.float32(done)))
            q_values = self.current_model(state)
            next_q_values = self.target_model(next_state)

            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_values.max(1)[0]

            expected_q_value = reward + self.gamma * next_q_value * (1 - done)
            loss = (q_value - Variable(expected_q_value.detach())).pow(2)*weights
            prios = loss + 1e-5

        loss = loss.mean()

        self.optimizer.zero_grad()

        loss.backward()

        # for name,para in self.current_model.named_parameters():
        #     print("i",name,para.grad)
        self.optimizer.step()




        if self.prioritized_replay:
            self.replay_buffer.update_priorities(indices,prios.data.cpu().numpy())

        self.current_model.reset_noise()
        self.target_model.reset_noise()

        return loss

    def append_to_replay(self,s, a, r, s_, done):
        self.nstep_buffer.append((s, a, r, s_, done))
        if len(self.nstep_buffer) < self.n_step:
            return
        R = sum(self.nstep_buffer[i][2] * self.gamma ** i for i in range(self.n_step))
        state, action, _, _, _ = self.nstep_buffer.pop(0)
        self.replay_buffer.push(state, action, R, s_, done)

    def finish_nstep(self,):
        while (len(self.nstep_buffer) > 0):
            R = sum(self.nstep_buffer[i][2] * self.gamma * i for i in range(len(self.nstep_buffer)))
            state, action, reward, state_, _ = self.nstep_buffer.pop(0)
            self.replay_buffer.push(state, action, R, state_, True)

    def Train(self,):
        losses = []
        all_rewards = []
        episode_reward = 0
        oEnv=self.env

        state = oEnv.reset()
        for frame_idx in range(1, self.num_frames + 1):
            epsilon = self.epsilon_by_frame(frame_idx)
            action = self.current_model.act(state, epsilon)

            next_state, reward, done, _ = oEnv.step(action)
            self.append_to_replay(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                self.finish_nstep()
                state = oEnv.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0

            if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_td_loss(self.batch_size)
                losses.append(loss.data[0])

            if frame_idx % self.replace_iter_num == 0:
                self.update_target(self.current_model, self.target_model)

            if frame_idx % 3000 == 0:
                print("frame", frame_idx)

class RainbowDQN(nn.Module):
    def __init__(self, oEnv,num_inputs, num_actions, num_atoms, Vmin, Vmax,use_noisy,hidden_before,hidden_after):
        super(RainbowDQN, self).__init__()
        self.env=oEnv
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.use_noisy=use_noisy
        self.hidden_before=hidden_before
        self.hidden_after=hidden_after


        #pre_hidden
        # iPre = num_inputs
        # for iIndex, iHidden in enumerate(hidden_before, 1):
        #     sAttr = "linear%s" % iIndex
        #     setattr(self, sAttr, nn.Linear(iPre, iHidden))
        #     iPre = iHidden



        # self.linear1 = nn.Linear(num_inputs, 32)
        # self.linear2 = nn.Linear(32, 64)

        setattr(self, "linear1", nn.Linear(num_inputs, 32))
        setattr(self, "linear2", nn.Linear(32, 64))



        if self.use_noisy:
            #value
            # iPre = hidden_before[-1]
            # for iIndex, iHidden in enumerate(hidden_after, 1):
            #     sAttr = "value%s" % iIndex
            #     setattr(self, sAttr,  NoisyLinear(iPre, iHidden, use_cuda=USE_CUDA))
            #     iPre = iHidden
            # sAttr = "value%s" % (iIndex+1)
            # setattr(self, sAttr, NoisyLinear(iPre, num_atoms, use_cuda=USE_CUDA))
            #
            # #advantage
            # iPre = hidden_before[-1]
            # for iIndex, iHidden in enumerate(hidden_after, 1):
            #     sAttr = "advantage%s" % iIndex
            #     setattr(self, sAttr,  NoisyLinear(iPre, iHidden, use_cuda=USE_CUDA))
            #     iPre = iHidden
            # sAttr = "advantage%s" % (iIndex+1)
            # setattr(self, sAttr, NoisyLinear(iPre, self.num_atoms * self.num_actions, use_cuda=USE_CUDA))

            # self.value1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
            # self.value2 = NoisyLinear(64, self.num_atoms, use_cuda=USE_CUDA)
            #
            # self.advantage1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
            # self.advantage2 = NoisyLinear(64, self.num_atoms * self.num_actions, use_cuda=USE_CUDA)

            setattr(self, "value1", NoisyLinear(64, 64, use_cuda=USE_CUDA))
            setattr(self, "value2", NoisyLinear(64, self.num_atoms, use_cuda=USE_CUDA))
            setattr(self, "advantage1",  NoisyLinear(64, 64, use_cuda=USE_CUDA))
            setattr(self, "advantage2", NoisyLinear(64, self.num_atoms * self.num_actions, use_cuda=USE_CUDA))

        else:
            # value
            # iPre = hidden_before[-1]
            # for iIndex, iHidden in enumerate(hidden_after, 1):
            #     sAttr = "value%s" % iIndex
            #     setattr(self, sAttr, nn.Linear(iPre, iHidden))
            #     iPre = iHidden
            # sAttr = "value%s" % (iIndex + 1)
            # setattr(self, sAttr, nn.Linear(iPre, num_atoms))
            #
            # # advantage
            # iPre = hidden_before[-1]
            # for iIndex, iHidden in enumerate(hidden_after, 1):
            #
            #     sAttr = "advantage%s" % iIndex
            #     setattr(self, sAttr, nn.Linear(iPre, iHidden))
            #     iPre = iHidden
            # sAttr = "advantage%s" % (iIndex + 1)
            # setattr(self, sAttr, nn.Linear(iPre, self.num_atoms * self.num_actions))



            # self.value1= nn.Linear(64, 64)
            # self.value2= nn.Linear(64, self.num_atoms)
            # self.advantage1=nn.Linear(64, 64)
            # self.advantage2=nn.Linear(64,  self.num_atoms * self.num_actions)

            setattr(self, "value1", nn.Linear(64, 64))
            setattr(self, "value2", nn.Linear(64, self.num_atoms))
            setattr(self, "advantage1", nn.Linear(64, 64))
            setattr(self, "advantage2", nn.Linear(64,  self.num_atoms * self.num_actions))


    def forward(self, x):
        batch_size = x.size(0)

        # for iIndex,iHidden in enumerate(self.hidden_before,1):
        #     sAttr="linear%s"%iIndex
        #     layer=getattr(self,sAttr)
        #     x = F.relu(layer(x))
        #
        # for iIndex,_ in enumerate(range(len(self.hidden_after)+1),1):
        #     sAttr="value%s"%iIndex
        #     layer=getattr(self,sAttr)
        #     if iIndex==1:
        #         value = F.relu(layer(x))
        #     else:
        #         value=F.relu(layer(value))
        #
        # for iIndex, _ in enumerate(range(len(self.hidden_after) + 1), 1):
        #     sAttr="advantage%s"%iIndex
        #     layer=getattr(self,sAttr)
        #     if iIndex == 1:
        #         advantage = F.relu(layer(x))
        #     else:
        #         advantage = F.relu(layer(advantage))




        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        #
        # value = F.relu(self.value1(x))
        # value = self.value2(value)
        #
        # advantage = F.relu(self.advantage1(x))
        # advantage = self.advantage2(advantage)

        layer = getattr(self, "linear1")
        x = F.relu(layer(x))
        layer = getattr(self, "linear2")
        x = F.relu(layer(x))

        layer = getattr(self, "value1")

        value = F.relu(layer(x))

        layer = getattr(self, "value2")
        value = layer(value)

        layer = getattr(self, "advantage1")

        advantage = F.relu(layer(x))
        layer = getattr(self, "advantage2")
        advantage = layer(advantage)



        if self.num_atoms>1:
            value = value.view(batch_size, 1, self.num_atoms)
            advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

            x = value + advantage - advantage.mean(1, keepdim=True)
            x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)
        else:
            x=value+advantage-advantage.mean()

        return x

    def reset_noise(self):
        if self.use_noisy:
            for iIndex, iHidden in enumerate(self.hidden_after, 1):
                sAttr = "value%s" % iIndex
                layer = getattr(self, sAttr)
                layer.reset_noise()

            for iIndex, iHidden in enumerate(self.hidden_after, 1):
                sAttr = "advantage%s" % iIndex
                layer = getattr(self, sAttr)
                layer.reset_noise()

            # self.value1.reset_noise()
            # self.value2.reset_noise()
            # self.advantage1.reset_noise()
            # self.advantage2.reset_noise()

    def act(self, state,epsilon=0.02):
        if random.random() < epsilon:
            action = random.randrange(self.env.action_space.n)
        else:
            if self.num_atoms>1:
                state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
                dist = self.forward(state).data.cpu()
                dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
                action = dist.sum(2).max(1)[1].numpy()[0]
            else:
                state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
                q_value = self.forward(state)
                action = q_value.max(1)[1].cpu().numpy()[0]

        return action


class MyReward(gym.Wrapper):
    def __init__(self, env):
        super(MyReward, self).__init__(env)
        self.m_RwardList = []
        self.m_count = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # reward=0
        self.m_count += 1
        # print("reware",self.m_count,action,reward,done,info)
        if not done:
            self.m_RwardList.append(reward)
        else:
            # if info["done"]:
            #     reward=100
            #     self.m_RwardList.append(reward)
            iMeanReward = np.sum(self.m_RwardList)
            print("mean_reward", iMeanReward)
            self.m_RwardList = []
        return obs, reward, done, info


def Train():
    oEnv = gym.make("MountainCar-v0")
    oEnv = MyReward(oEnv)
    agent = Agent(oEnv)
    agent.Train()



if __name__=="__main__":
    Train()