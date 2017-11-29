from torch.autograd import Variable
import numpy as np
import torch as th
from params import scale_reward
import datetime
import visdom

import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

import multiagent
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from maddpg import MADDPG

# args = parser.parse_args()
# load scenario from script
scenario = scenarios.load('/home/yexm/maddpg_mpe/multiagent-particle-envs/multiagent/scenarios/simple_tag.py').Scenario()

# create world
world = scenario.make_world()
# create multiagent environment
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                    shared_viewer=False)

n_agents = env.n
# n_states = env.observation_space
n_actions = world.dim_p
capacity = 1000000
batch_size = 1000
totalTime = 0

vis = visdom.Visdom(port=8097)
win = None
param = None

np.random.seed(1234)
th.manual_seed(1234)

n_episode = 20000
max_steps = 1200
episode_before_train = 100
obs = env.reset()
n_states = len(obs[0])

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity, episode_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

for i_episode in range(n_episode):
    startTime = datetime.datetime.now()
    obs = env.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
       obs = th.from_numpy(obs).float()
    #obs = np.asarray(obs)
    reward_record = []
    adversaries_reward_record = []
    agent_reward_record = []

    # for i in range(len(obs)):
    #     if isinstance(obs[i], np.ndarray):
    #         obs[i] = th.from_numpy(obs[i]).float()
    #         obs[i] = Variable(obs[i]).type(FloatTensor)
            

    total_reward = 0.0
    adversaries_reward = 0.0
    agent_reward = 0.0

    rr = np.zeros((n_agents,))

    for t in range(max_steps):
        #for j in range(len(obs)):
        #    obs = Variable(obs).type(FloatTensor)
            
        obs = Variable(obs).type(FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
        obs_, reward, done, _ = env.step(action.numpy())
        #obs_ = np.asarray(obs_)
        # for p in range(len(obs_)):
        #     if isinstance(obs_[p], np.ndarray):
        #        obs_[p] = th.from_numpy(obs_[p]).float()
        #        obs_[p] = Variable(obs_[p]).type(FloatTensor)
        #for q in range(len(obs_)):
        #    print('obs_[q]',type(obs_[q]))
        #    obs_[q] = Variable(obs_[q]).type(FloatTensor)
        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        #obs_ = np.asarray(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None

        total_reward += reward.sum()
        adversaries_reward += reward[0:2].sum()
        agent_reward = reward[3]
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, action, next_obs, reward)
        #for i in range(len(next_obs)):
        #        for j in range(4):
        #            for k in range(len(next_obs[i][j])):
        #                if next_obs[i] != None:
        #                    print('next_obs[i][j][k]',type(next_obs[i][j][k]),i,j,k)
        #print('next_obs',len(next_obs))  4 ndarray  next_obs[0] <class 'torch.FloatTensor'> len(next_obs[0]) 16
        obs = next_obs
        c_loss, a_loss = maddpg.update_policy(i_episode)
        #env.render()
    maddpg.episode_done += 1
    endTime = datetime.datetime.now()
    runTime = (endTime - startTime).seconds
    totalTime = totalTime+runTime
    print('Episode:%d,reward = %f' % (i_episode, total_reward))
    print('Episode:%d,adversaries_reward = %f' % (i_episode, adversaries_reward))
    print('Episode:%d,agent_reward = %f' % (i_episode, agent_reward))
    print('this episode run time:'+ str(runTime))
    print('totalTime:'+ str(totalTime))
    reward_record.append(total_reward)
    adversaries_reward_record.append(adversaries_reward)
    agent_reward_record.append(agent_reward)

    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')
        print('MADDPG on WaterWorld\n' +
              'scale_reward=%f\n' % scale_reward +
              'agent=%d' % n_agents )


    if win is None:
        win = vis.line(X=np.arange(i_episode, i_episode+1),
                       Y=np.array([
                           np.append(adversaries_reward, rr)]),
                       opts=dict(
                           ylabel='Reward',
                           xlabel='Episode',
                           title='MADDPG on MOE\n' +
                           'agent=%d' % n_agents +
                           ', sensor_range=0.2\n',
                           legend=['Total'] +
                           ['Agent-%d' % i for i in range(n_agents)]))
    else:
        vis.line(X=np.array(
            [np.array(i_episode).repeat(n_agents+1)]),
                 Y=np.array([np.append(total_reward,
                                       rr)]),
                 win=win,
                 update='append')
    if param is None:
        param = vis.line(X=np.arange(i_episode, i_episode+1),
                         Y=np.array([maddpg.var[0]]),
                         opts=dict(
                             ylabel='Var',
                             xlabel='Episode',
                             title='MADDPG on MPE: Exploration',
                             legend=['Variance']))
    else:
        vis.line(X=np.array([i_episode]),
                 Y=np.array([maddpg.var[0]]),
                 win=param,
                 update='append')
