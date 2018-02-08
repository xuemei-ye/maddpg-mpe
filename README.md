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

![image](https://github.com/yexme/maddpg-mpe/blob/master/Waterworld_Trained.gif)
Before train：(click the picture and you will see the video.)

[![IMAGE ALT TEXT](http://oyf4unfbt.bkt.clouddn.com/runtime.png)](http://v.youku.com/v_show/id_XMzI4MjgyODU2MA==.html?spm=a2h3j.8428770.3416059.1)


Trained 1000 episodes：

[![IMAGE ALT TEXT](http://oyf4unfbt.bkt.clouddn.com/runtime.png)](http://v.youku.com/v_show/id_XMzI4MjgzMDAyNA==.html?spm=a2h3j.8428770.3416059.1)





