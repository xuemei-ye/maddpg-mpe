# maddpg-mpe
Transplant a implementation of MADDPG to the environment provided by openAI (multiagent-particle-envs).

## Introduction

Transplant a pytorch implementation [pytorch-maddpg](https://github.com/xuehy/pytorch-maddpg]) of MADDPG.

paper : [multi-agent deep deterministic policy gradient algorithm](https://arxiv.org/abs/1706.02275).

environment : [multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs). 
(tested it with the simple tag environment and didn't use communication property c).


## Dependency

- [pytorch](https://github.com/pytorch/pytorch)
- [visdom](https://github.com/facebookresearch/visdom)
- python 3 (recommend using the anaconda/miniconda)

## Install

- git clone and there are a number of other requirements which can be found in multiagent-particle-envs/environment.yml file if using anaconda distribution.
- add directories to PYTHONPATH: 
      
       export PYTHONPATH=$(pwd):$(pwd)/multiagent
- python main.py

## result

![image](https://github.com/yexme/maddpg-mpe/blob/master/picture/Waterworld_beforTrain.gif)：


Trained 1000 episodes：

![image](https://github.com/yexme/maddpg-mpe/blob/master/picture/Waterworld_Trained.gif)

Two purple spots are agents, red spots are poison, and green spots are food. It can be seen that before the training, the movement of the agent is random. After 1000 iterations, the agent has the actions of chasing, avoiding and cooperating.

read more:
- [MADDPG note](https://zhuanlan.zhihu.com/p/30527842)
- [Transplant note](https://zhuanlan.zhihu.com/p/31175608)




