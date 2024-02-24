import numpy as np

from hw3.roble.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from hw3.roble.infrastructure.utils import add_noise
from hw3.roble.policies.MLP_policy import MLPPolicyDeterministic
from hw3.roble.critics.ddpg_critic import DDPGCritic
import copy

class DDPGAgent(object):
    
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, env, **kwargs):

        self._last_obs = self._env.reset()
        self._cumulated_rewards = 0
        self._rewards = []
        self._num_actions = self._env.action_space.shape[0]

        self._replay_buffer_idx = None
        
        self._actor = MLPPolicyDeterministic(
            **kwargs
        )
        ## Create the Q function
        # self._agent_params['optimizer_spec'] = self._optimizer_spec
        self._q_fun = DDPGCritic(self._actor, **kwargs)

        ## Hint: We can use the Memory optimized replay buffer but now we have continuous actions
        self._replay_buffer = MemoryOptimizedReplayBuffer(
            self._replay_buffer_size, self._frame_history_len, lander=True,
            continuous_actions=True, ac_dim=self._ac_dim)
        self._t = 0
        self._num_param_updates = 0
        self._step_counter = 0
        
    def add_to_replay_buffer(self, paths):
        pass

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
        perform_random_action = self._actor.get_action(self._replay_buffer.encode_recent_observation())
        # HINT: take random action
        # if perform_random_action < -3 or perform_random_action > 3:
        #     print('test')
        action = add_noise(perform_random_action)

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

    def get_replay_buffer(self):
        return self._replay_buffer
        
    def sample(self, batch_size):
        if self._replay_buffer.can_sample(self._train_batch_size):
            return self._replay_buffer.sample(batch_size)
        else:
            # print("Need more experience in the replay buffer")
            return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self._t > self._learning_starts
                and self._t % self._learning_freq == 0
                and self._replay_buffer.can_sample(self._train_batch_size)
        ):
            # TODO fill in the call to the update function using the appropriate tensors
            update_q_fun = self._num_param_updates % self._num_critic_updates_per_agent_update == 0
            if update_q_fun:
                log = self._q_fun.update(
                    ob_no, ac_na, next_ob_no, re_n, terminal_n
                )
            
            # TODO fill in the call to the update function using the appropriate tensors
            ## Hint the actor will need a copy of the q_net to maximize the Q-function
            log = self._actor.update(
                ob_no, self._q_fun
            )

            self._q_fun.update_target_network(update_q_fun)

            # TODO update the target network periodically 
            # HINT: your critic already has this functionality implemented
            # if self._num_param_updates % self._target_update_freq == 0:
            #     self._q_fun.update_target_network()

            self._num_param_updates += 1
        self._t += 1
        return log
