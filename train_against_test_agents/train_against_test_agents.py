from typing import List, Dict, Optional
import math
import time
import yaml
import argparse
import logging
import os
from functools import reduce, partial

import numpy as np
import gym_connect4
import multiprocessing
import torch
from torch.utils.tensorboard import SummaryWriter

import regym
from regym.environments.wrappers import FrameStack
from regym.environments import Task
from regym.rl_loops import compute_winrates
from regym.networks.preprocessing import (batch_vector_observation,
                                          keep_last_stack_and_batch_vector_observation)
from regym.rl_algorithms import AgentHook, load_population_from_path
from regym.rl_algorithms.agents import build_NeuralNet_Agent
from regym.environments import generate_task, EnvType
from regym.util.experiment_parsing import initialize_agents
from regym.util.experiment_parsing import filter_relevant_configurations


def train_against_fixed_agent(task: 'Task',
                              agent: 'Agent',
                              test_agents: List['Agent'],
                              base_path: str,
                              exper_config: Dict,
                              summary_writer: SummaryWriter):
    for test_agent in test_agents:
        logger = logging.getLogger(f'TRAINING: Task: {task.name}. Agent: {agent.name} Opponent: {test_agent.name}')
        logger.info('Started')

        os.makedirs(base_path, exist_ok=True)

        train_to_a_desired_winrate(
            task,
            training_agent=agent,
            opponent=test_agent,
            desired_winrate=exper_config['desired_winrate'],
            training_episodes=exper_config['training_episodes'],
            benchmarking_episodes=exper_config['benchmarking_episodes'],
            agent_position=0,   # HACK: THIS IS VERY IMPORTANT
            num_envs=exper_config['num_envs'],
            base_path=base_path,
            logger=logger,
            summary_writer=summary_writer)


def train_to_a_desired_winrate(task: 'Task',
                               training_agent: 'Agent',
                               opponent: 'Agent',
                               desired_winrate: float,
                               training_episodes: int,
                               benchmarking_episodes: int,
                               agent_position: int,
                               num_envs: int,
                               base_path: str,
                               logger,
                               summary_writer: SummaryWriter):
    """
    :param task: Task on which agents will be trained
    :param training_agent: agent representation + training algorithm which will be trained in this process
    :param opponent: Fixed agent against which :param: training_agent will train against
    :param desired_winrate: Winrate at which the learning agent will stop training
    :param training_episodes: Number of episodes which will be played before benchmarking
    :param benchmarking_episodes: Number of episodes used to compute winrate (termination condition)
    :param agent_position: Index on the agent vector where the agent will be placed
    :param base_path: Base directory from where subdirectories will be accessed to reach menageries, save episodic rewards and save checkpoints of agents.
    """
    completed_iterations, start_time, benchmark_winrate = 0, time.time(), -math.inf

    while benchmark_winrate < desired_winrate:
        logger.info(f'Completed_episodes: {task.total_episodes_run}. Training for extra {training_episodes}')
        trajectories = train_for_given_iterations(
            task,
            training_agent,
            opponent,
            agent_position,
            training_episodes,
            num_envs,
            logger)
        completed_iterations += len(trajectories)
        winrates_during_training = compute_winrates(trajectories[-benchmarking_episodes:])
        tolerance = 0.1
        logger.info(f'Winrate during training {winrates_during_training[agent_position]}, desired - tolerance: {desired_winrate - tolerance}')
        summary_writer.add_scalar('Training/Winrate', winrates_during_training[agent_position], completed_iterations)
        if winrates_during_training[agent_position] >= desired_winrate - tolerance:
            logger.info(f'Proceding to benchmark for {benchmarking_episodes} episodes')
            benchmark_winrate = benchmark_agent(
                task=task,
                agent=training_agent,
                opponent=opponent,
                agent_position=agent_position,
                benchmarking_episodes=benchmarking_episodes,
                starting_episode=task.total_episodes_run,
                logger=logger,
                summary_writer=summary_writer)

        del trajectories # Off you go!

        save_trained_policy(
            training_agent,
            save_path=f'{base_path}/{training_agent.name}_{completed_iterations}_iterations.pt',
            logger=logger)

    logger.info('FINISHED training to reach {}. Total duration: {} seconds'.format(desired_winrate, time.time() - start_time))


def train_for_given_iterations(task,
                               training_agent: 'Agent',
                               opponent: 'Agent',
                               agent_position: int,
                               training_episodes: int,
                               num_envs: int,
                               logger) -> List:
    agent_vector = [opponent]
    agent_vector.insert(agent_position, training_agent)
    training_start = time.time()
    trajectories = task.run_episodes(agent_vector,
                                     training=True,
                                     num_envs=num_envs,  # Max number of environments
                                     num_episodes=training_episodes,
                                     show_progress=False,
                                     summary_writer=summary_writer)
    training_duration = time.time() - training_start
    logger.info('Training for {} took {:.2} seconds'.format(
                training_episodes, training_duration))
    return trajectories


def benchmark_agent(task: Task, agent: 'Agent', opponent: 'Agent',
                    agent_position, benchmarking_episodes,
                    starting_episode: int,
                    logger,
                    summary_writer: Optional[SummaryWriter] = None):
    agent_vector = [opponent]
    agent_vector.insert(agent_position, agent)
    training_start = time.time()
    trajectories = task.run_episodes(agent_vector, training=False,
                                     num_envs=-1,  # Max number of environments
                                     num_episodes=benchmarking_episodes)
    benchmarking_time = time.time() - training_start
    logger.info('Benchmarking for {} took {:.2} seconds'.format(
                benchmarking_episodes, benchmarking_time))

    # How can we also print this info in a useful way?
    winrate = len(list(filter(lambda t: t.winner == agent_position,
                              trajectories))) / len(trajectories)
    logger.info(f'Benchmarking winrate {winrate}')
    avg_episode_length = reduce(lambda acc, t: acc + len(t), trajectories, 0) / len(trajectories)
    avg_episode_reward = reduce(lambda acc, t: acc + t.agent_specific_cumulative_reward(agent_position),
                                trajectories, 0) / len(trajectories)
    if summary_writer:
        summary_writer.add_scalar('Benchmarking/Winrate', winrate, starting_episode)
        summary_writer.add_scalar('Benchmarking/Average_episode_length', avg_episode_length, starting_episode)
        summary_writer.add_scalar('Benchmarking/Average_episode_reward', avg_episode_reward, starting_episode)
    return winrate


def save_trained_policy(trained_agent, save_path: str, logger):
    logger.info(f'Saving agent \'{trained_agent.name}\' in \'{save_path}\'')
    torch.save(trained_agent, save_path)


def initialize_experiment(experiment_config, agents_config, args):
    task = create_task_from_config(experiment_config['environment'])
    agents = initialize_agents(task, agents_config)
    test_agents = load_test_agents(task, args.opponents_path)
    return task, agents, test_agents

def create_task_from_config(environment_config):
    wrappers = create_wrappers(environment_config)
    task = generate_task(environment_config['name'],
                         EnvType(environment_config['env_type']),
                         wrappers=wrappers)
    return task

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
    agents_config = filter_relevant_configurations(experiment_config,
                                                   target_configs=all_configs['agents'],
                                                   target_key='algorithms')
    return experiment_config, agents_config


def load_agent_and_update_task(agent_directory_path: str, task) -> 'Agent':
    if not os.path.isdir(agent_directory_path):
        raise ValueError(f'Path {agent_directory_path} does not exist')

    all_agent_checkpoints = list(filter(lambda file: file.endswith('.pt'),
                                        os.listdir(agent_directory_path)))
    latest_agent = max(
        all_agent_checkpoints,
        # Following the convention of {name}_{iteration}_iterations.pt
        # We sort by the {iterations} number
        key=(lambda file: int(file.split('_')[-2]))
    )
    agent = torch.load(f'{agent_directory_path}/{latest_agent}')
    task.total_episodes_run = agent.finished_episodes
    return agent


def load_test_agents(task, opponents_path) -> List['Agent']:
    test_agents = load_population_from_path(opponents_path)
    test_agents = [build_NeuralNet_Agent(task,
                      {'neural_net': t_a.algorithm.model,
                       #'pre_processing_fn': batch_vector_observation},
                       'state_preprocess_fn': keep_last_stack_and_batch_vector_observation},
                       f'TestAgent: {t_a.handled_experiences}')
                   for t_a in test_agents]
    return test_agents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Expert Iteration / Best Response Expert Iteration agents for experiments of paper "On Opponent Modelling in Expert Iteration"')
    parser.add_argument('--config', required=True, help='path to YAML config file containing info about environment and agents')
    parser.add_argument('--opponents_path', required=True, help='path to directory containing agents to train against (opponents)')
    parser.add_argument('--agent_index', required=False, default=None, help='Optional. Index of the agent that will be kept out of all of the agents specified in config file. Useful for batch jobs in SLURM settings')
    args = parser.parse_args()

    # Spawn is required for GPU to be used inside of neural_net_server
    torch.multiprocessing.set_start_method('spawn')

    logging.basicConfig(level=logging.INFO)
    top_level_logger = logging.getLogger('BRExIt Opponent Modelling Experiment')

    exper_config, agents_config = load_configs(args.config)
    task, agents, test_agents = initialize_experiment(exper_config, agents_config, args)

    if (len(exper_config['algorithms']) > 1) and (args.agent_index is None):
        raise ValueError('More than one agent was specified. Use `agent_index` to select which one to use')

    agent_index = int(args.agent_index) if args.agent_index else 0
    agent_name = exper_config['algorithms'][agent_index]
    agent = agents[agent_index]

    base_path = f"{exper_config['experiment_id']}/{agent_name}"

    # Maybe this should go into initialize_experiment
    if os.path.exists(base_path) and (os.listdir(base_path) != []):  # Load pre-trained agent, if there is any
        top_level_logger.info(f'Attempting to load agent from: {base_path}')
        agent = load_agent_and_update_task(base_path, task)
        top_level_logger.info(f'Loaded agent, with {agent.finished_episodes} episodes under its belt')

    log_path = f"{exper_config['experiment_id']}_logs/{agent_name}"

    summary_writer = SummaryWriter(log_path)
    agent.algorithm.summary_writer = summary_writer

    train_against_fixed_agent(
        task,
        agent,
        test_agents,
        base_path,
        exper_config,
        summary_writer)
