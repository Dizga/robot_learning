import numpy as np

from hw3.roble.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from hw3.roble.policies.argmax_policy import ArgMaxPolicy
from hw3.roble.critics.dqn_critic import DQNCritic

class DQNAgent(object):
    
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    # def __init__(self, env, agent_params):
    def __init__(self, env, **kwargs):

        self.env = env
        # self.agent_params = agent_params
        # self.batch_size = agent_params['alg']['train_batch_size']
        # import ipdb; ipdb.set_trace()
        self._last_obs = self.env.reset()

        self._num_actions = self.env.action_space.n
        # self.learning_starts = agent_params['alg']['learning_starts']
        # self.learning_freq = agent_params['alg']['learning_freq']
        # self.target_update_freq = agent_params['alg']['target_update_freq']

        self._replay_buffer_idx = None
        # self.exploration = agent_params['exploration_schedule']
        # self.optimizer_spec = agent_params['optimizer_spec']

        self._critic = DQNCritic(**kwargs)
        self._actor = ArgMaxPolicy(self._critic)

        lander = self._env_name.startswith('LunarLander')
        self._replay_buffer = MemoryOptimizedReplayBuffer(
            self._replay_buffer_size, self._frame_history_len, lander=lander)
        self._t = 0
        self._num_param_updates = 0

        # self.env = env
        # self.agent_params = self._params
        # self.batch_size = self._params['alg']['train_batch_size']
        # # import ipdb; ipdb.set_trace()
        # self.last_obs = self.env.reset()

        # self.num_actions = self.env.action_space.n
        # self.learning_starts = agent_params['alg']['learning_starts']
        # self.learning_freq = agent_params['alg']['learning_freq']
        # self.target_update_freq = agent_params['alg']['target_update_freq']

        # self.replay_buffer_idx = None
        # self.exploration = agent_params['exploration_schedule']
        # self.optimizer_spec = agent_params['optimizer_spec']

        # self.critic = DQNCritic(agent_params, self.optimizer_spec)
        # self.actor = ArgMaxPolicy(self.critic)

        # lander = agent_params['env']['env_name'].startswith('LunarLander')
        # self.replay_buffer = MemoryOptimizedReplayBuffer(
        #     agent_params['alg']['replay_buffer_size'], agent_params['alg']['frame_history_len'], lander=lander)
        # self.t = 0
        # self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        pass

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """        

        # TODO store the latest observation ("frame") into the replay buffer
        # HINT: the replay buffer used here is `MemoryOptimizedReplayBuffer`
            # in dqn_utils.py
        # self.replay_buffer_idx = -1
        self.replay_buffer_idx = self._replay_buffer.store_frame(self._last_obs)

        eps = self._exploration_schedule.value(self._t)

        # TODO use epsilon greedy exploration when selecting action
        perform_random_action = np.random.random() < eps
        if perform_random_action:
            # HINT: take random action 
                # with probability eps (see np.random.random())
                # OR if your current step number (see self.t) is less that self.learning_starts
            action = self.env.action_space.sample()
        else:
            # HINT: Your actor will take in multiple previous observations ("frames") in order
                # to deal with the partial observability of the environment. Get the most recent 
                # `frame_history_len` observations using functionality from the replay buffer,
                # and then use those observations as input to your actor. 
            # action = self._actor.get_action(self._last_obs)
            action = self._actor.get_action(self._replay_buffer.encode_recent_observation())
        
        # TODO take a step in the environment using the action from the policy
        # HINT1: remember that self.last_obs must always point to the newest/latest observation
        # HINT2: remember the following useful function that you've seen before:
            #obs, reward, done, info = env.step(action)

        self._last_obs, reward, done, info = self.env.step(action)

        # TODO store the result of taking this action into the replay buffer
        # HINT1: see your replay buffer's `store_effect` function
        # HINT2: one of the arguments you'll need to pass in is self.replay_buffer_idx from above
        self._replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        if done:
            self._last_obs = self.env.reset()



    def sample(self, batch_size):
        if self._replay_buffer.can_sample(batch_size):
            return self._replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        # if (self.t > self._learning_starts
        #         and self._t % self._learning_freq == 0
        #         and self._replay_buffer.can_sample(self.batch_size)
        # ):
        if (self._t > self._learning_starts
                and self._t % self._learning_freq == 0
        ):
            # TODO fill in the call to the update function using the appropriate tensors
            log = self._critic.update(
                ob_no, ac_na, next_ob_no, re_n, terminal_n
            )

            # TODO update the target network periodically 
            # HINT: your critic already has this functionality implemented
            if self._num_param_updates % self._target_update_freq == 0:
                self._critic.update_target_network()

            self._num_param_updates += 1
        self._t += 1
        return log
