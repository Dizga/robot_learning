# Commands used to make the experiments.

## Question 1


Network of 1x32 hidden layers, with 500 training steps
```
python run_hw2_mb.py env.exp_name=q1_cheetah_n500_arch1x32 env.env_name=cheetah-roble-v0 alg.num_agent_train_steps_per_iter=500 alg.network.layer_sizes='[32]' alg.n_iter=1
```

Network of 2x256 hidden layers, with 5 training steps
```
python run_hw2_mb.py env.exp_name=q1_cheetah_n5_arch2x256 env.env_name=cheetah-roble-v0 alg.num_agent_train_steps_per_iter=5 alg.network.layer_sizes='[256,256]' alg.n_iter=1 logging.video_log_freq=1
```

Network of 2x256 hidden layers, with 500 training steps
```
python run_hw2_mb.py env.exp_name=q1_cheetah_n500_arch2x256 env.env_name=cheetah-roble-v0 alg.num_agent_train_steps_per_iter=500 alg.network.layer_sizes='[256,256]' alg.n_iter=1 logging.video_log_freq=1
```

## Question 2

```
python run_hw2_mb.py env.exp_name=q2_obstacles_singleiteration env.env_name=obstacles-roble-v0 alg.num_agent_train_steps_per_iter=20 alg.batch_size_initial=5000 alg.batch_size=1000 alg.mpc_horizon=10 alg.n_iter=1 logging.video_log_freq=1
```

## Question 3


MBRL run on obstacles env.
```
python run_hw2_mb.py env.exp_name=q3_obstacles env.env_name=obstacles-roble-v0 alg.num_agent_train_steps_per_iter=20 alg.batch_size_initial=5000 alg.batch_size=2000 alg.mpc_horizon=10 alg.n_iter=12 logging.video_log_freq=11
```

MBRL run on reacher env.
```
python run_hw2_mb.py env.exp_name=q3_reacher env.env_name=reacher-roble-v0 alg.mpc_horizon=10 alg.num_agent_train_steps_per_iter=1000 alg.batch_size_initial=5000 alg.batch_size=5000 alg.n_iter=15 logging.video_log_freq=14
```

MBRL run on cheetah env.
```
python run_hw2_mb.py env.exp_name=q3_cheetah env.env_name=cheetah-roble-v0 alg.mpc_horizon=15  alg.num_agent_train_steps_per_iter=1500 alg.batch_size_initial=5000 alg.batch_size=5000 alg.n_iter=20 logging.video_log_freq=19
```

## Question 4


Horizon 5 steps.
```
python run_hw2_mb.py env.exp_name=q4_reacher_horizon5 env.env_name=reacher-roble-v0 alg.add_sl_noise=true alg.mpc_horizon=5 alg.mpc_action_sampling_strategy='random' alg.num_agent_train_steps_per_iter=1000 alg.batch_size=800 alg.n_iter=15 logging.video_log_freq=14
```

Horizon 15 steps.
```
python run_hw2_mb.py env.exp_name=q4_reacher_horizon15 env.env_name=reacher-roble-v0 alg.add_sl_noise=true alg.mpc_horizon=15 alg.mpc_action_sampling_strategy='random' alg.num_agent_train_steps_per_iter=1000 alg.batch_size=800 alg.n_iter=15 logging.video_log_freq=14
```

Horizon 30 steps.
```
python run_hw2_mb.py env.exp_name=q4_reacher_horizon30 env.env_name=reacher-roble-v0 alg.add_sl_noise=true alg.mpc_horizon=30 alg.mpc_action_sampling_strategy='random' alg.num_agent_train_steps_per_iter=1000 alg.batch_size=800 alg.n_iter=15 logging.video_log_freq=14
```

100 action sequences.
```
python run_hw2_mb.py env.exp_name=q4_reacher_numseq100 env.env_name=reacher-roble-v0 alg.add_sl_noise=true alg.mpc_horizon=10 alg.mpc_action_sampling_strategy='random' alg.num_agent_train_steps_per_iter=1000 alg.batch_size=800 alg.n_iter=15 alg.mpc_num_action_sequences=100 logging.video_log_freq=14
```

1000 action sequences.
```
python run_hw2_mb.py env.exp_name=q4_reacher_numseq1000 env.env_name=reacher-roble-v0 alg.add_sl_noise=true alg.mpc_horizon=10 alg.mpc_action_sampling_strategy='random' alg.num_agent_train_steps_per_iter=1000 alg.batch_size=800 alg.n_iter=15 alg.mpc_num_action_sequences=1000 logging.video_log_freq=14
```

No ensemble.
```
python run_hw2_mb.py env.exp_name=q4_reacher_ensemble1 env.env_name=reacher-roble-v0 alg.ensemble_size=1 alg.add_sl_noise=true alg.mpc_horizon=10 alg.mpc_action_sampling_strategy='random' alg.num_agent_train_steps_per_iter=1000 alg.batch_size=800 alg.n_iter=15 logging.video_log_freq=14
```

Ensemble size 3.
```
python run_hw2_mb.py env.exp_name=q4_reacher_ensemble3 env.env_name=reacher-roble-v0 alg.ensemble_size=3 alg.add_sl_noise=true alg.mpc_horizon=10 alg.mpc_action_sampling_strategy='random' alg.num_agent_train_steps_per_iter=1000 alg.batch_size=800 alg.n_iter=15 logging.video_log_freq=14
```

## Question 5

MBRL with random shooting
```
python run_hw2_mb.py env.exp_name=q5_cheetah_random env.env_name='cheetah-roble-v0' alg.mpc_horizon=15  alg.num_agent_train_steps_per_iter=1500 alg.batch_size_initial=5000 alg.batch_size=5000 alg.n_iter=5 logging.video_log_freq=4 alg.mpc_action_sampling_strategy='random'
```

MBRL with CEM 2
```
python run_hw2_mb.py env.exp_name=q5_cheetah_cem_2 env.env_name='cheetah-roble-v0' alg.mpc_horizon=15 alg.add_sl_noise=true alg.num_agent_train_steps_per_iter=1500 alg.batch_size_initial=5000 alg.batch_size=5000 alg.n_iter=5 logging.video_log_freq=4 alg.mpc_action_sampling_strategy='cem' alg.cem_iterations=2
```


MBRL with CEM 4
```
python run_hw2_mb.py env.exp_name=q5_cheetah_cem_4 env.env_name='cheetah-roble-v0' alg.mpc_horizon=15 alg.add_sl_noise=true alg.num_agent_train_steps_per_iter=1500 alg.batch_size_initial=5000 alg.batch_size=5000 alg.n_iter=5 logging.video_log_freq=4 alg.mpc_action_sampling_strategy='cem' alg.cem_iterations=4
```