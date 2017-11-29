# maddpg-mpe
Transplant a implementation of MADDPG to the environment provided by openAI(multiagent-particle-envs).

# maddpg-mpe
Transplant a implementation of MADDPG to the environment provided by openAI(multiagent-particle-envs).

## 1. Introduction

Transplant a pytorch implementation [pytorch-maddpg](https://github.com/xuehy/pytorch-maddpg]) of MADDPG.

paper : [multi-agent deep deterministic policy gradient algorithm](https://arxiv.org/abs/1706.02275)

environment : [multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs). 
(tested it with the simple tag environment and didn't use communication property c.)


## 2. Dependency

- [pytorch](https://github.com/pytorch/pytorch)
- [visdom](https://github.com/facebookresearch/visdom)
- python==3.6.1 (recommend using the anaconda/miniconda)

## 3. Install

- git clone and there are a number of other requirements which can be found in multiagent-particle-envs/environment.yml file if using anaconda distribution.
- add directories to PYTHONPATH: 
      
       export PYTHONPATH=$(pwd):$(pwd)/multiagent
- python main.py

## 4. TODO

Adjust parameters to improve training results.
And for decrease training time,I change local observation to whole observation,this can be improvement.





