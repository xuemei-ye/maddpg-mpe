# maddpg-mpe
Transplant a implementation of MADDPG to the environment provided by openAI(multiagent-particle-envs).

* 1. Introduction

Transplant a pytorch implementation([[https://github.com/xuehy/pytorch-maddpg][pytorch-maddpg]]) of MADDPG.
paper:[[https://arxiv.org/abs/1706.02275][multi-agent deep deterministic policy gradient algorithm]] 
environment:[[https://github.com/openai/multiagent-particle-envs][multiagent-particle-envs]]. 
(Tested it with the simple tag environment and didn't use communication property c.)


* 2. Dependency

- [[https://github.com/pytorch/pytorch][pytorch]]
- [[https://github.com/facebookresearch/visdom][visdom]]
- =python==3.6.1= (recommend using the anaconda/miniconda)

* 3. Install

- git clone and there are a number of other requirements which can be found in multiagent-particle-envs/environment.yml file if using anaconda distribution.
- add directories to PYTHONPATH: =export PYTHONPATH=$(pwd):$(pwd)/multiagent=
- =python main.py=

* 4. TODO

Adjust parameters to improve training results.
And for decrease training time,I change local observation to whole observation,this can be improvement.

