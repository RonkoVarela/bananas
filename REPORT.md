# REPORT

This document describes the details the agent implementation and ideas for future work.

## The Agent model

The agent is a dueling neural network with 2 hidden layers with ReLU activation.

## Training features

The training consists of a deep q-learning with some improved features:
- experience replay in batches extracted from the previous n steps,
- fixed Q-target with doubleDQN, where the target network is updated every n steps,
- prioritized experience replay,
- gradient clipping.
