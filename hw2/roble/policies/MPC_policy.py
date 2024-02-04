# hw1 imports
from hw1.roble.policies.base_policy import BasePolicy

import numpy as np

class MPCPolicy(BasePolicy):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self,
                 env,
                 dyn_models,
                 mpc_horizon,
                 mpc_num_action_sequences,
                 mpc_action_sampling_strategy='random',
                 **kwargs
                 ):
        super().__init__()

        # init vars
        self._data_statistics = None  # NOTE must be updated from elsewhere


        # action space
        self._ac_space = self._env.action_space
        self._low = self._ac_space.low
        self._high = self._ac_space.high

        # self.models = dyn_models

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert self._mpc_action_sampling_strategy in allowed_sampling, f"self._mpc_action_sampling_strategy must be one of the following: {allowed_sampling}"

        print(f"Using action sampling strategy: {self._mpc_action_sampling_strategy}")
        if self._mpc_action_sampling_strategy == 'cem':
            print(f"CEM params: alpha={self._cem_alpha}, "
                + f"num_elites={self._cem_num_elites}, iterations={self._cem_iterations}")

    def get_random_actions(self, num_sequences, horizon):
       return np.random.uniform(low=self._ac_space.low, high=self._ac_space.high,
						size=(horizon,num_sequences))
						
    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self._mpc_action_sampling_strategy == 'random' \
            or (self._mpc_action_sampling_strategy == 'cem' and obs is None):
            # TODO (Q1) uniformly sample trajectories and return an array of
            # dimensions (num_sequences, horizon, self._ac_dim) in the range
            # [self._low, self._high]
            random_action_sequences = np.random.uniform(low=self._low, high=self._high,
                                                        size=(num_sequences, horizon, self._ac_space.shape[0]))
            return random_action_sequences
        elif self._mpc_action_sampling_strategy == 'cem':
            # TODO(Q5): Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf

            for i in range(self._cem_iterations):
                # - Sample candidate sequences from a Gaussian with the current
                #   elite mean and variance
                #     (Hint: remember that for the first iteration, we instead sample
                #      uniformly at random just like we do for random-shooting)
                # - Get the top `self._cem_num_elites` elites
                #     (Hint: what existing function can we use to compute rewards for
                #      our candidate sequences in order to rank them?)
                # - Update the elite mean and variance
                
                if i == 0:
                    candidates = np.random.uniform(low=self._low, high=self._high,
                                                        size=(num_sequences, horizon, self._ac_space.shape[0]))
                else:
                    candidates = np.random.normal(elites_mean, elites_std,(num_sequences, horizon, self._ac_space.shape[0]))
            
                scores = self.evaluate_candidate_sequences(candidates, obs)

                elites_ids = scores.argsort()[-self._cem_num_elites:]

                elites = candidates[elites_ids]

                elites_mean = elites.mean(axis=0)
                elites_std = elites.mean(axis=0)

                

            # TODO(Q5): Set `cem_action` to the appropriate action sequence chosen by CEM.
            # The shape should be (horizon, self._ac_dim)  
            cem_action = elites_mean
            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self._mpc_action_sampling_strategy}")
        
    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self._mpc_action_sampling_strategy == 'random' \
            or (self._mpc_action_sampling_strategy == 'cem' and obs is None):
            # TODO (Q1) uniformly sample trajectories and return an array of
            # dimensions (num_sequences, horizon, self._ac_dim) in the range
            # [self._low, self._high]
            random_action_sequences = np.random.uniform(low=self._low, high=self._high,
                                                        size=(num_sequences, horizon, self._ac_space.shape[0]))
            return random_action_sequences
        elif self._mpc_action_sampling_strategy == 'cem':
            # TODO(Q5): Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf
            for i in range(self._cem_iterations):
                # - Sample candidate sequences from a Gaussian with the current
                #   elite mean and variance
                #     (Hint: remember that for the first iteration, we instead sample
                #      uniformly at random just like we do for random-shooting)
                # - Get the top `self._cem_num_elites` elites
                #     (Hint: what existing function can we use to compute rewards for
                #      our candidate sequences in order to rank them?)
                # - Update the elite mean and variance
                pass

            # TODO(Q5): Set `cem_action` to the appropriate action sequence chosen by CEM.
            # The shape should be (horizon, self._ac_dim)  
            cem_action = None
            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self._mpc_action_sampling_strategy}")
        
    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # TODO(Q2): for each model in ensemble, compute the predicted sum of rewards
        # for each candidate action sequence.
        #
        # Then, return the mean predictions across all ensembles.
        # Hint: the return value should be an array of shape (N,)
        N, _, _ = candidate_action_sequences.shape
        predicted_rewards = []

        for model in self._dyn_models:
            predicted_rewards.append(self.calculate_sum_of_rewards(obs, candidate_action_sequences, model))

        return np.array(predicted_rewards).mean(axis=0)

    def get_action(self, obs):
        if self._data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self._mpc_num_action_sequences, horizon=self._mpc_horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)
            # pick the action sequence and return the 1st element of that sequence
            best_sequence_idx = np.argmax(predicted_rewards)
            best_action_sequence = candidate_action_sequences[best_sequence_idx]
            action_to_take = best_action_sequence[0]
            return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        sum_of_rewards = None  # TODO (Q2)
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self._env.get_reward(predicted_obs, action)` at each step.
        # You should sum across `self._mpc_horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.

        # obs_sequences = []
        # obs_sequences.append(np.repeat(obs, candidate_action_sequences.shape[0]))

        # cas = np.transpose(candidate_action_sequences, (1, 0, 2))
        # for actions in cas:
        #     predicted_obs = model(actions)
        #     self._env.get_reward(predicted_obs, action)

        # model
        
        # return sum_of_rewards



        # N, H, _ = candidate_action_sequences.shape  # N: Number of sequences, H: Horizon
        # sum_of_rewards = np.zeros(N)  # Initialize the sum of rewards for each sequence
        
        # # Iterate over each action sequence
        # for i in range(N):
        #     cumulative_reward = 0  # Initialize cumulative reward for the sequence
        #     current_obs = obs.copy()  # Start with the initial observation
            
        #     # Iterate over each time step in the horizon
        #     for h in range(H):
        #         # Extract the current action from the sequence
        #         action = candidate_action_sequences[i, h, :]
                
        #         # Predict the next state based on the current state and action
        #         next_obs_pred = model.get_prediction(current_obs[None, :], action[None, :], self._data_statistics)
                
        #         # Calculate the reward for the predicted state and current action
        #         reward, _ = self._env.get_reward(current_obs[None, :], action[None, :])
                
        #         cumulative_reward += reward  # Accumulate the reward
                
        #         # Update the current observation to the predicted next observation
        #         current_obs = next_obs_pred.squeeze()  # Assuming next_obs_pred is of shape (1, D_obs)
            
        #     # Store the sum of rewards for this action sequence
        #     sum_of_rewards[i] = cumulative_reward
        
        # return sum_of_rewards


        N, H, _ = candidate_action_sequences.shape
        candidate_action_sequences = candidate_action_sequences.transpose(1, 0, 2)
        sum_of_rewards = np.zeros(N)
        current_obs = np.tile(obs, (N, 1))

        for h in range(H):
            actions = candidate_action_sequences[h]
            next_obs_pred = model.get_prediction(current_obs, actions, self._data_statistics)

            rewards, _ = self._env.get_reward(current_obs, actions)
            sum_of_rewards += rewards.squeeze()

            current_obs = next_obs_pred

        return sum_of_rewards
