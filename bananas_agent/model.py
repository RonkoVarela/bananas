import torch.nn as nn


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self,
                 state_size,
                 action_size,
                 first_layer_size=128,
                 advantage_layer_size=128,
                 value_layer_size=32,
                 ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            first_layer_size (int): Hidden neurons in first layer
            action_layer_size (int): Hidden neurons in advantage layer
            value_layer_size (int): Hidden neurons in value layer
        """
        super(QNetwork, self).__init__()

        # 1st FC layer
        fc1_in = state_size
        fc1_out = first_layer_size
        self.fc1 = nn.Linear(in_features=fc1_in, out_features=fc1_out, bias=True)

        # DUELING NETWORK: two fully connected output-layers
        # 2nd FC layers
        fc2_in = fc1_out
        fc2_out_a = advantage_layer_size
        fc2_out_v = value_layer_size
        self.fc2_a = nn.Linear(in_features=fc2_in, out_features=fc2_out_a)
        self.fc2_v = nn.Linear(in_features=fc2_in, out_features=fc2_out_v)
        
        # output FC layers
        self.fc_out_a = nn.Linear(in_features=fc2_out_a, out_features=action_size)
        self.fc_out_v = nn.Linear(in_features=fc2_out_v, out_features=1)

        # init gaussian weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2_a.weight)
        nn.init.xavier_uniform_(self.fc2_v.weight)
        nn.init.xavier_uniform_(self.fc_out_a.weight)
        nn.init.xavier_uniform_(self.fc_out_v.weight)
        
        # create two sequential
        self.sequential_a = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2_a,
            nn.ReLU(),
            self.fc_out_a,
        )
        self.sequential_v = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2_v,
            nn.ReLU(),
            self.fc_out_v,
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        advantage_value = self.sequential_a(state)
        advantage_value = advantage_value - advantage_value.mean(0)[0]
        return advantage_value + self.sequential_v(state)
