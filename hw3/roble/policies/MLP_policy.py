import abc
import copy
import itertools
import numpy as np
import torch
import hw1.roble.util.class_util as classu

from hw1.roble.infrastructure import pytorch_util as ptu
from hw1.roble.policies.base_policy import BasePolicy 
from hw1.roble.policies.MLP_policy import MLPPolicy 
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions

class ConcatMLP(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, dim=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self._dim)
        return super().forward(flat_inputs, **kwargs)

class MLPPolicyDeterministic(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, *args, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        kwargs['deterministic'] = True
        kwargs['network']['output_activation']='tanh'
        super().__init__(*args, **kwargs)
        
    def update(self, observations, q_fun):

        observations = ptu.from_numpy(observations)

        self._optimizer.zero_grad()
        # TODO: update the policy and return the loss
        ## Hint you will need to use the q_fun for the loss
        ## Hint: do not update the parameters for q_fun in the loss
        q_values = q_fun._q_net(observations, self(observations))

        loss = -q_values.mean()

        # Compute gradients
        loss.backward()

        # Update policy parameters
        self._optimizer.step()

        return {"Loss": loss.item()}
    
    def forward(self, observation: torch.FloatTensor):
        return self._max_action_value * super().forward(observation)
    
class MLPPolicyStochastic(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, entropy_coeff, *args, **kwargs):
        kwargs['deterministic'] = False
        super().__init__(*args, **kwargs)
        self.entropy_coeff = entropy_coeff

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: sample actions from the gaussian distribrution given by MLPPolicy policy when providing the observations.
        # Hint: make sure to use the reparameterization trick to sample from the distribution

        obs = ptu.from_numpy(obs).float()
        # action_distribution = self.forward(obs)
        # action = action_distribution.sample()
        action, _ = self.forward(obs)
        return ptu.to_numpy(action)
        
    def update(self, observations, q_fun):
        # TODO: update the policy and return the loss
        ## Hint you will need to use the q_fun for the loss
        ## Hint: do not update the parameters for q_fun in the loss
        ## Hint: you will have to add the entropy term to the loss using self.entropy_coeff
        observations = ptu.from_numpy(observations).float()
        # action_distribution = self.forward(observations)
        # actions = action_distribution.rsample()
        # log_probs = action_distribution.log_prob(actions)

        actions, log_probs = self.forward(observations)

        q_values = q_fun._q_net(observations, actions)
        loss = -(q_values - self.entropy_coeff * log_probs).mean()

        # Optimize the policy
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return {"Loss": loss.item()}
    
    def forward(self, observation: torch.FloatTensor):
        action_distribution = super().forward(observation)
        actions = action_distribution.rsample()
        log_probs = action_distribution.log_prob(actions)
        actions = torch.tanh(actions)
        log_probs -= torch.log(1 - actions.pow(2) + 1e-7)
        actions = actions * self._max_action_value
        return actions, log_probs.detach()
    
#####################################################