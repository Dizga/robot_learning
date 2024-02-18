import numpy as np

from hw3.roble.critics.dqn_critic import DQNCritic


class ArgMaxPolicy(object):

    def __init__(self, critic : DQNCritic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        q_values = self.critic.qa_values(observation).squeeze()    
        action = np.random.choice(np.argwhere(q_values == q_values.max()).flatten())
        return action
        # return action.squeeze()