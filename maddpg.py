from model import Critic, Actor
import torch as th
from copy import deepcopy
from torch.optim import Adam
from memory import ReplayMemory, Experience
from randomProcess import OrnsteinUhlenbeckProcess
from torch.autograd import Variable

import torch.nn as nn
import numpy as np
from params import scale_reward


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train):
        sum_obs_dim = 0
        self.actors = []
        self.critics = []
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs,
                               dim_act) for i in range(n_agents)]
        #for i in range(n_agents):
            # print('dim_act',dim_act)
            # print('len(dim_obs[i])',len(dim_obs[i]))
            # print('dim_act[i].shape',dim_act[i].shape)
            # print('dim_act[i]',dim_act[i])
            #self.actors.append(Actor(len(dim_obs[i]), dim_act))
            #sum_obs_dim += len(dim_obs[i])
            # self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
            # sum_obs_dim += len(dim_obs[i])
        #for i in range(n_agents):
            #self.critics.append(Critic(n_agents, sum_obs_dim, dim_act))
        # self.critics = [Critic(n_agents, dim_obs,
        #                       dim_act) for i in range(n_agents)]

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act

        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]

        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self,i_episode):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []
        a_loss = []

        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            
            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = Variable(th.stack(batch.states).type(FloatTensor))
            #whole_list = []
            # next_whole_list = []
            # next_state_count = 0
            #print(len(batch.states) == len(batch.next_states))
            # for i in range(len(batch.states)):
            #     n_list = []
            #     for j in range(4):
            #         for k in range(len(batch.states[i][j])):
            #             n_list.append(batch.states[i][j][k].data.numpy())
            #             #if batch.next_states[i] != None:
            #                 #print('batch.next_states[i][j][k]',type(batch.next_states[i][j][k]),i,j,k)
            #                 # next_state_count += 1
            #     n_array = np.asarray(n_list)
            #     # print('n_array',type(n_array))
            #     n_tensor = th.from_numpy(n_array).float()
            #     n_variable = Variable(n_tensor).type(FloatTensor)
            #     whole_list.append(n_variable.data.numpy())
            # whole_array = np.asarray(whole_list)
            # whole_tensor = th.from_numpy(whole_array).float()
            
            # for i in range(len(batch.states)):
            #     next_list = []
            #     if batch.next_states[i] != None:
            #         for j in range(4):
            #             for k in range(len(batch.next_states[i][j])):
            #                 #print('batch.next_states[i][j][k]',batch.next_states[i][j][k],i,j,k)
            #                 next_list.append(batch.next_states[i][j][k].data.numpy())
            #         next_array = np.asarray(next_list)
            #         next_tensor = th.from_numpy(next_array).float()
            #         next_variable = Variable(next_tensor).type(FloatTensor)
            #         next_whole_list.append(th.t(next_variable).data.numpy())
            # next_whole_array = np.asarray(next_whole_list)
            # next_whole_tensor = th.from_numpy(next_whole_array).float()
                        
            #state_batch = Variable(th.stack(whole_tensor).type(FloatTensor))
            # print('state_batch',state_batch)  #[torch.FloatTensor of size 100x62x1]
            # state_batch = Variable(th.stack(batch.states).type(FloatTensor))
            action_batch = Variable(th.stack(batch.actions).type(FloatTensor))
            reward_batch = Variable(th.stack(batch.rewards).type(FloatTensor))
            # : (batch_size_non_final) x n_agents x dim_obs
            # print('next_whole_tensor',next_whole_tensor)
            #non_final_next_states = Variable(th.stack(next_whole_tensor).type(FloatTensor))
            non_final_next_states = Variable(th.stack(
               [s for s in batch.next_states
                if s is not None]).type(FloatTensor))
            # print('non_final_next_states',non_final_next_states)  [torch.FloatTensor of size 99x1x62]
            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            # print('whole_state',whole_state)  [torch.FloatTensor of size 100x62]
            whole_action = action_batch.view(self.batch_size, -1)
            # non_final_next_states = non_final_next_states.view(next_state_count,-1)
            # print('non_final_next_states',non_final_next_states)
            self.critic_optimizer[agent].zero_grad()

            current_Q = self.critics[agent](whole_state, whole_action)

            # non_final_next_actions = []

            # for a in range(self.n_agents):
            #     batch_obs = []
            #     # for j in range(self.n_agents):
            #     for i in range(len(batch.next_states)):
            #         if batch.next_states[i] is not None:
            #             batch_obs.append(batch.next_states[i][a].data.numpy())
            #             # print('batch_obs',type(batch.next_states[i][a]))  'torch.autograd.variable.Variable'
            #     batch_obs = np.asarray(batch_obs)
            #     batch_obs = th.from_numpy(batch_obs).float()
            #     batch_obs = Variable(batch_obs).type(FloatTensor)
            #     # print('batch_obs',batch_obs)  [torch.FloatTensor of size 99x16]
            #     non_final_next_actions.append(self.actors_target[a](batch_obs))
                # print('non_final_next_actions',non_final_next_actions)

            non_final_next_actions = [      #[torch.FloatTensor of size 989x2]
                 self.actors_target[i](non_final_next_states[:,    #[torch.FloatTensor of size 989x213]
                                      i,
                                       :]) for i in range(
                    self.n_agents)]
            non_final_next_actions = th.stack(non_final_next_actions)
            # non_final_next_actions = Variable(non_final_next_actions)
            non_final_next_actions = (
                non_final_next_actions.transpose(0,
                                                 1).contiguous())
            target_Q = Variable(th.zeros(
                self.batch_size).type(FloatTensor))
            # print('non_final_mask',non_final_mask)
            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view(-1,
                                            self.n_agents * self.n_actions))

            # scale_reward: to scale reward in Q functions
            target_Q = (target_Q * self.GAMMA) + (
                reward_batch[:, agent] * scale_reward)

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            # state_i = []
            # for i in range(len(state_batch)):
            #     state_i.append(batch.states[i][agent].data.numpy())
                # print('batch_obs',type(batch.next_states[i][a]))  'torch.autograd.variable.Variable'
            #state_i = np.asarray(state_i)
            #state_i = th.from_numpy(state_i).float()
            #state_i = Variable(state_i).type(FloatTensor)
            # print('state_i',state_i)  [torch.FloatTensor of size 100x1]
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)
        if i_episode % 100 == 0:
            for i in range(self.n_agents):
                th.save(self.critics[i], 'critic[' + str(i) + '].pkl_episode' + str(i_episode))
                th.save(self.actors[i], 'actors[' + str(i) + '].pkl_episode' + str(i_episode))
        return c_loss, a_loss

    def select_action(self, state_batch):
        # state_batch: n_agents x state_dim
        actions = Variable(th.zeros(
            self.n_agents,
            self.n_actions))
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_agents):
            sb = state_batch[i].detach()
            act = self.actors[i](sb.unsqueeze(0)).squeeze()        
            act += Variable(
                th.from_numpy(
                    np.random.randn(2) * self.var[i]).type(FloatTensor))

            if self.episode_done > self.episodes_before_train and \
                            self.var[i] > 0.05:
                self.var[i] *= 0.999998
            act = th.clamp(act, -1.0, 1.0)

            actions[i, :] = act          
        self.steps_done += 1

        return actions
