import numpy as np
import copy

from hw3.roble.policies.MLP_policy import MLPPolicyStochastic
from hw3.roble.critics.sac_critic import SACCritic
from hw3.roble.agents.ddpg_agent import DDPGAgent

class SACAgent(DDPGAgent):
    
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, env, **kwargs):

        super().__init__(env, **kwargs)
        
        self._actor = MLPPolicyStochastic(
            **kwargs
        )

        self._q_fun = SACCritic(self._actor, 
                               **kwargs)
        
    # def step_env(self):
    #     """
    #         Step the env and store the transition
    #         At the end of this block of code, the simulator should have been
    #         advanced one step, and the replay buffer should contain one more transition.
    #         Note that self._last_obs must always point to the new latest observation.
    #     """        

    #     # TODO: Take the code from DDPG Agent and make sure the remove the exploration noise
    #     return
    
    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self._last_obs must always point to the new latest observation.
        """        

        # TODO store the latest observation ("frame") into the replay buffer
        # HINT: the replay buffer used here is `MemoryOptimizedReplayBuffer`
            # in dqn_utils.py
        # self._replay_buffer_idx = -1
        self.replay_buffer_idx = self._replay_buffer.store_frame(self._last_obs)

        # TODO add noise to the deterministic policy
        action = self._actor.get_action(self._replay_buffer.encode_recent_observation())

        # noise_scale = 0.1  # Example noise scale, adjust based on environment
        # random_action = np.random.randn(self._num_actions) * noise_scale
        # action = self._actor.get_action(self._last_obs) + random_action
        
        # TODO take a step in the environment using the action from the policy
        # HINT1: remember that self._last_obs must always point to the newest/latest observation
        # HINT2: remember the following useful function that you've seen before:
            #obs, reward, done, info = env.step(action)
        self._last_obs, reward, done, info = self._env.step(action)

        # TODO store the result of taking this action into the replay buffer
        # HINT1: see your replay buffer's `store_effect` function
        # HINT2: one of the arguments you'll need to pass in is self._replay_buffer_idx from above
        self._replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        if done:
            self._last_obs = self._env.reset()