import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.optim as optim


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(object):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 buffer_size=10 ** 5,
                 batch_size=64,
                 gamma=.99,
                 learning_rate=1e-3,
                 learn_every=4,
                 update_every=1000,
                 grad_clipping_norm=1.,
                 ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            learning_rate (float)
            learn_every (int): how many steps to learn a batch
            update_every (int): how many steps to update the target network
            grad_clipping_norm (int): how many steps to learn a batch
        """

        self.state_size = state_size
        self.action_size = action_size

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size).to(DEVICE)
        self.qnetwork_target = QNetwork(state_size, action_size).to(DEVICE)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # learning parameters
        self.learn_every = learn_every
        self.update_every = update_every
        self.gamma = gamma
        self.grad_clipping_norm = grad_clipping_norm
    
    def step(self, state, action, reward, next_state, done):
        # compute delta to do replay prioritization
        delta = self.get_experience_delta(state, action, reward, next_state, done)
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, delta)
        
        # Learn every LEARN_EVERY time steps.
        self.t_step += 1
        if self.t_step % self.learn_every == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.prioritized_sample()
                self.learn(experiences)

        # update the network
        if self.t_step % self.update_every == 0:
            self.update_target_model()

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def get_experience_delta(self, state, action, reward, next_state, done):
        """ Compute the delta between the target and local q_value. """
        experiences = [
            torch.Tensor([state], dtype=torch.float).to(DEVICE),
            torch.Tensor([[action]], dtype=torch.long).to(DEVICE),
            torch.Tensor([reward], dtype=torch.float).to(DEVICE),
            torch.Tensor([next_state], dtype=torch.float).to(DEVICE),
            torch.Tensor([done], dtype=torch.float).to(DEVICE),
        ]
        with torch.no_grad():
            return self.delta(experiences)[0, 0].cpu().numpy()

    def delta(self, experiences):
        """
        Compute the difference between expected q value and predicted q value.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        # use double-Q
        # expected actions gathered from local network
        max_value_actions = self.qnetwork_local(next_states).max(1)[1].reshape(-1, 1)
        # expected value gathered from target network
        q_target_next = self.qnetwork_target(next_states).detach().gather(1, max_value_actions)
        # Compute Q targets for current states 
        q_target_expected = rewards + (self.gamma * q_target_next * (1 - dones))

        # Get expected Q values from local model
        q_predicted = self.qnetwork_local(states).gather(1, actions)
        return q_target_expected - q_predicted

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        """
        delta = self.delta(experiences)

        # Compute loss (MSE)
        loss = (delta ** 2).mean()
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.grad_clipping_norm)
        self.optimizer.step()

    def update_target_model(self):
        """ Update target model parameters. """
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(local_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, eps=1e-6, a=.5):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.deltas = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # parameters to control the prioritized sampling
        self.eps = eps
        self.a = a
    
    def add(self, state, action, reward, next_state, done, delta):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.deltas.append((abs(delta) + self.eps) ** self.a)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
  
        return self.format_experiences(experiences)
    
    def prioritized_sample(self):
        """Prioritized sample a batch of experiences from memory."""
        probs = np.array(self.deltas)
        probs /= probs.sum()
        idx = np.random.choice(len(self.memory), size=self.batch_size)
        experiences = [self.memory[i] for i in idx]
        return self.format_experiences(experiences)

    @staticmethod
    def format_experiences(experiences):
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
