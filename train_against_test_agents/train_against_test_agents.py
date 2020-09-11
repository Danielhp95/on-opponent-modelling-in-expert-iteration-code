from typing import List, Dict, Optional
import pickle
import math
import time
import yaml
import argparse
import logging
import os
from functools import reduce

import numpy as np
import gym_connect4
import multiprocessing
import torch
from torch.utils.tensorboard import SummaryWriter

import regym
from regym.environments import Task
from regym.util import extract_winner, trajectory_reward
from regym.networks.preprocessing import batch_vector_observation
from regym.rl_algorithms import AgentHook, load_population_from_path
from regym.rl_algorithms.agents import build_NeuralNet_Agent
from regym.environments import generate_task, EnvType
from regym.util.experiment_parsing import initialize_agents
from regym.util.experiment_parsing import filter_relevant_configurations


def train_against_fixed_agent(task: 'Task', agents: List, test_agents: List['Agent'],
                              exper_config: Dict, summary_writer: SummaryWriter):
    for test_agent in test_agents:
        for agent in agents:
            logger = logging.getLogger(f'TRAINING: Task: {task.name}. Agent: {agent.name} Opponent: {test_agent.name}')
            logger.info('Started')

            base_path = f"{exper_config['experiment_id']}"
            os.makedirs(base_path, exist_ok=True), trajectory_reward

            train_to_a_desired_winrate(task, training_agent=agent, opponent=test_agent,
                  desired_winrate=exper_config['desired_winrate'],
                  training_episodes=exper_config['training_episodes'],
                  benchmarking_episodes=exper_config['benchmarking_episodes'],
                  agent_position=0,   # HACK: THIS IS VERY IMPORTANT
                  base_path=base_path,
                  logger=logger,
                  summary_writer=summary_writer)


def train_to_a_desired_winrate(task: 'Task', training_agent: 'Agent', opponent: 'Agent',
                               desired_winrate: float,
                               training_episodes: int, benchmarking_episodes: int,
                               agent_position: int,
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
        logger.info(f'Completed_episodes: {completed_iterations} Training for extra {training_episodes}')
        trajectories = train_for_given_iterations(task, training_agent, opponent,
                                                  agent_position,
                                                  training_episodes,
                                                  completed_iterations, logger)
        completed_iterations += len(trajectories)
        winrates_during_training = compute_winrates(trajectories[-benchmarking_episodes:],
                                                    num_agents=task.num_agents)

        tolerance = 0.1
        logger.info(f'Winrate during training {winrates_during_training[agent_position]}, desired - tolerance: {desired_winrate - tolerance}')
        if winrates_during_training[agent_position] >= desired_winrate - tolerance:
            logger.info(f'Proceding to benchmark for {benchmarking_episodes} episodes')
            benchmark_winrate = benchmark_agent(task, training_agent, opponent,
                                                agent_position,
                                                benchmarking_episodes,
                                                completed_iterations, logger,
                                                summary_writer)

        del trajectories # Off you go!

        save_trained_policy(training_agent,
                            save_path=f'{base_path}/{training_agent.name}_{completed_iterations}_iterations.pt',
                            logger=logger)

    logger.info('FINISHED training to reach {}. Total duration: {} seconds'.format(desired_winrate, time.time() - start_time))


def compute_winrates(trajectories: List, num_agents: int) -> List[float]:
    winners = list(map(lambda t: extract_winner(t), trajectories))
    return [winners.count(a_i) / len(winners) for a_i in range(num_agents)]


def train_for_given_iterations(task, training_agent: 'Agent', opponent: 'Agent',
                               agent_position: int, training_episodes: int,
                               completed_iterations: int,  # TODO Do we need this param?
                               logger) -> List:
    agent_vector = [opponent]
    agent_vector.insert(agent_position, training_agent)
    training_start = time.time()
    trajectories = task.run_episodes(agent_vector, training=True,
                                     num_envs=-1,  # Max number of environments
                                     num_episodes=training_episodes)
    training_duration = time.time() - training_start
    logger.info('Training for {} took {:.2} seconds'.format(
                training_episodes, training_duration))
    return trajectories


def benchmark_agent(task: Task, agent: 'Agent', opponent: 'Agent',
                    agent_position, benchmarking_episodes,
                    starting_episode: int,
                    logger,
                    summary_writer: Optional[SummaryWriter]):
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
    winrate = len(list(filter(lambda t: extract_winner(t) == agent_position,
                              trajectories))) / len(trajectories)
    logger.info(f'Benchmarking winrate {winrate}')
    avg_episode_length = reduce(lambda acc, t: acc + len(t), trajectories, 0) / len(trajectories)
    avg_episode_reward = reduce(lambda acc, t: acc + trajectory_reward(t, agent_position),
                                trajectories, 0) / len(trajectories)
    if summary_writer:
        summary_writer.add_scalar('Benchmarking/Winrate', winrate, starting_episode)
        summary_writer.add_scalar('Benchmarking/Average_episode_length', avg_episode_length, starting_episode)
        summary_writer.add_scalar('Benchmarking/Average_episode_reward', avg_episode_reward, starting_episode)
    return winrate


def save_trained_policy(trained_agent, save_path: str, logger):
    logger.info(f'Saving agent \'{trained_agent.name}\' in \'{save_path}\'')
    torch.save(trained_agent, save_path)


def initialize_experiment(experiment_config, agents_config):
    env_name, requested_env_type = experiment_config['environment']
    task = generate_task(env_name, EnvType(requested_env_type))
    agents = initialize_agents(task, agents_config)
    return task, agents


def load_configs(config_file_path: str):
    all_configs = yaml.load(open(config_file_path), Loader=yaml.FullLoader)
    experiment_config = all_configs['experiment']
    agents_config = filter_relevant_configurations(experiment_config,
                                                   target_configs=all_configs['agents'],
                                                   target_key='algorithms')
    return experiment_config, agents_config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Trains Expert Iteration / Best Response Expert Iteration agents for experiments of paper "On Opponent Modelling in Expert Iteration"')
    parser.add_argument('--config', required=True, help='path to YAML config file containing info about environment and agents')
    parser.add_argument('--opponents_path', required=True, help='path to directory containing agents to train against (opponents)')
    args = parser.parse_args()

    multiprocessing.set_start_method('forkserver')
    exper_config, agents_config = load_configs(args.config)
    task, agents = initialize_experiment(exper_config, agents_config)

    test_agents = load_population_from_path(args.opponents_path)
    test_agents = [build_NeuralNet_Agent(task,
                      {'neural_net': t_a.algorithm.model,
                       'pre_processing_fn': batch_vector_observation},
                       f'TestAgent: {t_a.handled_experiences}')
                   for t_a in test_agents]
    summary_writer = SummaryWriter('Exit-TrainAgainstTestAgents')
    regym.rl_algorithms.expert_iteration.expert_iteration_loss.summary_writer = summary_writer

    train_against_fixed_agent(task, agents, test_agents, exper_config, summary_writer)
