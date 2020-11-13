# Deep-RL
In this project I trained a DQN (Deep Q-Network) agent 
to learn how to play the popular Atari game "Breakout".

the training was done on a 8cpu GCP VM, 
and it took a **whole day** to train the agent for **1M**
iteration. 

The agent reached a superhuman level
(meaning it never loose) by just trail
and error and using only the game image as an input
(Like any human player).  

The Architecture and hyperparams of the DQN network
are the same used in the original DQN paper by Deepmind
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

the trained agent in action :

<img src="breakout.gif" width="260" height="310"/>


lot of this work is inspired by Auélien Géron's Book
"Hands-On Machine learning with Scikit-Learn, Keras
 & TensorFlow" https://github.com/ageron/handson-ml2/

#### Run project locally

create a conda environment using the requirement.txt file:

conda create --name rl_env --file requirement.txt

This uses Tensorflow 2.0 and TF-agents 0.3. 
Workes and tested on Window10 and Linux as well  


#### Technologies

* Tensorflow
* TF-agents
* OpenAI's Gym[atari] environment
