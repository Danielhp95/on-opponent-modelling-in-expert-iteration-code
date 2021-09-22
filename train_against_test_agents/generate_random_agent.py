import regym

from regym.environments import generate_task, EnvType
from regym.rl_algorithms import build_PPO_Agent
import gym_connect4
import torch

config = dict()
config['discount'] = 0.99
config['use_gae'] = False
config['use_cuda'] = False
config['gae_tau'] = 0.95
config['entropy_weight'] = 0.01
config['gradient_clip'] = 5
config['optimization_epochs'] = 10
config['mini_batch_size'] = 32
config['ppo_ratio_clip'] = 0.2
config['learning_rate'] = 3.0e-4
config['adam_eps'] = 1.0e-5
config['horizon'] = 128
config['phi_arch'] = 'CNN-MLP'
config['use_batch_normalization'] = False
config['preprocessed_input_dimensions'] = [7, 6]
config['channels'] = [3, 32]
config['kernel_sizes'] = [3]
config['paddings'] = [1]
config['strides'] = [1]
config['final_feature_dim'] = 512
config['body_mlp_hidden_units'] = [64]
# Important
config['state_preprocess_fn'] = 'turn_into_single_element_batch'

task = generate_task('Connect4-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION)
random_ppo_agent = build_PPO_Agent(task, config, 'RandomAgent')
# making sure that it works
random_ppo_agent.model_free_take_action(task.env.observation_space.sample()[0])
torch.save(random_ppo_agent, 'random_ppo_agent.pt')

