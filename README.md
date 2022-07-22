[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# DQN - Banana Hunting

### Introduction

To train a reinforcement learning agent to hunt healthy bananas.

![Trained Agent][image1]

### Project Details

- This environment is simulated using Unity's Reacher Environment
- The observation space consists of 37 variables contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.
- The action space consist of 4 variables corrensponding to move forward, backward, left, and right.
- A position reward, +1 is given if the agent collected yellow healthy banana.
- A negative reward, -1 is given if the agent collected blue poisoned banana.
- In order to solve the environment, your agent must get an average score of +13 over 100 consecuitive episodes.

### Getting Started

1. Clone this repo:
```shell
$ git clone https://github.com/junisabot/dqn-banana-hunting.git
```

2. Install python dependencies through pip:
```shell
$ pip install -r requirement.txt
```

3. Download Unity environment according your OS to the root folder of this repo and unzip the file.
- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

### Project Structure

1. report.html is a report of this project
2. train.ipynb is the notebook to train DQN network with this project
3. agent.py contains the structure of DQN learning agent.
4. network/dqn.py contains deep q neural network.
5. config.py contains all the adjustable parameters of this project.
6. pretrained models are provided in directory ./pretrained_model
