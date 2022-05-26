# Bananas

Project to train an agent able to solve the Navigation project in https://github.com/udacity/deep-reinforcement-learning#dependencies

The agent aims at collecting yellow bananas (+1 reward) and avoid blue bananas (-1 reward).

The environment has an action space of 4:
- move forward,
- move backward,
- move right,
- move left.

The environment state has 37 dimensions including agent's velocity, along with ray-based perception of objects around the agent's forward direction

The environment is considered solved when achieving a positive score of +13 during 100 episodes.

## Usage

Execute train.ipynb notebook to train the agent.

Execute test.ipynb notebook to test and view the saved agent model.

Note that you can find an already trained model in checkpoints/checkpoint.pth

## Dependencies

Refer to https://github.com/udacity/deep-reinforcement-learning#dependencies to install any needed dependencies.
