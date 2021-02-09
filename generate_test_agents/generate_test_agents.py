from typing import List, Dict
from functools import partial
import time
import yaml
import argparse
import logging
import os

import dill
import numpy as np
import gym_connect4
import torch

from torch.utils.tensorboard import SummaryWriter

import regym
from regym.rl_algorithms import load_population_from_path
from regym.environments.wrappers import FrameStack
from regym.training_schemes import SelfPlayTrainingScheme
from regym.environments import generate_task, EnvType
from regym.util.experiment_parsing import initialize_agents
from regym.util.experiment_parsing import initialize_training_schemes
from regym.util.experiment_parsing import filter_relevant_configurations
from regym.rl_loops.multiagent_loops import self_play_training


def generate_test_agents(task: 'Task', sp_schemes: List, agents: List,
                         initial_menagerie: List['Agent'],
                         exper_config: Dict):
    for sp_scheme in sp_schemes:
        for agent in agents:
            train(task, agent, sp_scheme, initial_menagerie,
                  training_episodes=int(exper_config['training_episodes']),
                  save_interval=int(exper_config['save_interval']),
                  base_path=f"{exper_config['experiment_id']}")
            dill.dump(sp_scheme, open(f"{exper_config['experiment_id']}/{sp_scheme.name}.dill", 'wb'))


def train(task: 'Task', training_agent: 'Agent',
          self_play_scheme: SelfPlayTrainingScheme,
          initial_menagerie: List['Agent'],
          training_episodes: int, save_interval: int,
          base_path: str):
    """
    :param task: Task on which agents will be trained
    :param training_agent: agent representation + training algorithm which will be trained in this process
    :param self_play_scheme: self play scheme used to meta train the param training_agent.
    :param training_episodes: total amount of training episodes
    :param save_interval: Number of episodes to elapse before saving :param: training_agent to disk
    :param base_path: Base directory from where subdirectories will be accessed to reach menageries, save episodic rewards and save checkpoints of agents.
    """

    logger = logging.getLogger(f'TRAINING: Task: {task.name}. SP: {self_play_scheme.name}. Agent: {training_agent.name}')
    logger.info('Started')

    menagerie, menagerie_path = initial_menagerie, f'{base_path}/menagerie'
    agents_to_benchmark = [] # Come up with better name

    if not os.path.exists(base_path):
        os.makedirs(base_path)
        os.mkdir(menagerie_path)

    completed_iterations, start_time = task.total_episodes_run, time.time()

    trained_policy_save_directory = base_path

    for target_iteration in range(completed_iterations + save_interval, training_episodes+1, save_interval):
        next_training_iterations = target_iteration - completed_iterations
        (menagerie, trained_agent,
         trajectories) = train_for_given_iterations(task, training_agent, self_play_scheme,
                                                    menagerie, menagerie_path,
                                                    next_training_iterations, completed_iterations, logger)
        logger.info('Training completion: {}%'.format(100 * target_iteration / training_episodes))
        del trajectories # we are not using them here
        completed_iterations += next_training_iterations
        save_path= f'{trained_policy_save_directory}/{target_iteration}_iterations.pt'
        torch.save(trained_agent, save_path)
        logger.info(f'Saving agent \'{trained_agent.name}\' in \'{save_path}\'')

        agents_to_benchmark += [trained_agent.clone()]
        training_agent = trained_agent # Updating

    logger.info('FINISHED training. Total duration: {} seconds'.format(time.time() - start_time))
    return agents_to_benchmark


def train_for_given_iterations(task, training_agent, self_play_scheme,
                               menagerie, menagerie_path,
                               next_training_iterations, completed_iterations,
                               logger):

    training_start = time.time()
    (menagerie, trained_agent,
     trajectories) = self_play_training(task=task, training_agent=training_agent, self_play_scheme=self_play_scheme,
                                        target_episodes=next_training_iterations, initial_episode=completed_iterations,
                                        menagerie=menagerie, menagerie_path=menagerie_path)
    training_duration = time.time() - training_start
    logger.info('Training between iterations [{}, {}]: {:.2} seconds'.format(
                completed_iterations, completed_iterations + next_training_iterations,
                training_duration))
    summary_writer.add_scalar('Timing/Training_iterations', training_duration, completed_iterations + next_training_iterations)
    return menagerie, training_agent, trajectories


def create_task_from_config(environment_config):
    wrappers = create_wrappers(environment_config)
    task = generate_task(environment_config['name'],
                         EnvType(environment_config['env_type']),
                         wrappers=wrappers)
    return task


def initialize_experiment(experiment_config, agents_config, self_play_configs):
    task = create_task_from_config(experiment_config['environment'])
    sp_schemes = initialize_training_schemes(self_play_configs, task)
    agents = initialize_agents(task, agents_config)
    initial_menagerie = []

    base_path = experiment_config['experiment_id']
    menagerie_path = f"{base_path}/menagerie/{sp_schemes[0].name}-{experiment_config['algorithms'][0]}"
    # Load pre-trained agent, if there is any (there might be a menagerie but not a trained agent)
    if os.path.exists(base_path) and (os.listdir(base_path) != ['menagerie']):
        logger = logging.getLogger('LOADING AGENT AND MENAGERIE')
        logger.info(f"Attempting to load agent from: {base_path}/")
        agent = load_existing_agent_and_update_task(base_path, task)
        assert os.path.exists(menagerie_path), f'Menagerie should be present at {menagerie_path}'
        initial_menagerie = load_population_from_path(menagerie_path, show_progress=True)
        initial_menagerie.sort(key=lambda agent: agent.finished_episodes)
        logger.info(f'Loaded agent, with {agent.finished_episodes} episodes under its belt')
        logger.info(f'Loaded menagerie containing {len(initial_menagerie)} agents')

    return task, sp_schemes, agents, initial_menagerie


def load_existing_agent_and_update_task(agent_directory_path: str, task) -> 'Agent':
    if not os.path.isdir(agent_directory_path):
        raise ValueError(f'Path {agent_directory_path} does not exist')

    all_agent_checkpoints = list(filter(lambda file: file.endswith('.pt'),
                                        os.listdir(agent_directory_path)))
    latest_agent = max(
        all_agent_checkpoints,
        # Following the convention of {iteration}_iterations.pt
        # We sort by the {iterations} number
        key=(lambda file: int(file.split('_')[0]))
    )
    agent = torch.load(f'{agent_directory_path}/{latest_agent}')
    task.total_episodes_run = agent.finished_episodes
    return agent


def create_wrappers(environment_config):
    wrappers = []
    if 'frame_stack' in environment_config:
        wrappers.append(partial(
            FrameStack,
            num_stack=int(environment_config['frame_stack'])
            )
        )
    return wrappers


def load_configs(config_file_path: str):
    all_configs = yaml.load(open(config_file_path), Loader=yaml.FullLoader)
    experiment_config = all_configs['experiment']
    self_play_configs = filter_relevant_configurations(experiment_config,
                                                       target_configs=all_configs['self_play_training_schemes'],
                                                       target_key='self_play_training_schemes')
    agents_config = filter_relevant_configurations(experiment_config,
                                                   target_configs=all_configs['agents'],
                                                   target_key='algorithms')
    return experiment_config, agents_config, self_play_configs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Generates test agents for experiments of paper "On Opponent Modelling in Expert Iteration"')
    parser.add_argument('--config', required=True, help='path to YAML config file containing info about environment, self-play algorithms and agents')
    args = parser.parse_args()

    exper_config, agents_config, self_play_config = load_configs(args.config)
    task, sp_schemes, agents, initial_menagerie = \
            initialize_experiment(exper_config, agents_config, self_play_config)

    log_path = f"{exper_config['experiment_id']}_logs"
    summary_writer = SummaryWriter(log_path)
    for agent in agents: agent.summary_writer = summary_writer

    generate_test_agents(task, sp_schemes, agents, initial_menagerie, exper_config)
