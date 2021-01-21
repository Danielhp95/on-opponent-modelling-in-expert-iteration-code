from typing import List, Tuple, Dict, Optional
from copy import deepcopy
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
                                          keep_last_stack_and_batch_vector_observation,
                                          flatten_last_dim_and_batch_vector_observation)
from regym.rl_algorithms import AgentHook, load_population_from_path
from regym.rl_algorithms.agents import build_NeuralNet_Agent
from regym.environments import generate_task, EnvType
from regym.util.experiment_parsing import initialize_agents
from regym.util.experiment_parsing import filter_relevant_configurations


'''
Creates the following directory structure:

experiment_id/
    run_id/
        agent_name/
            tensorboard_logs (many files)
            winrates/
                apprentice_only.csv                 (always present)
                vanilla_exit.csv                    (always present)
                true_opponent_model.csv            (if applicable)
                learnt_opponent_model_winrate.csv  (if applicable)

This script can be called multiple times with different --run_id and agent_index
flags to populate the same `experiment_id/` directory, with different `run_id/` and
`agent_name/` subdirectories.

Each winrates/*.csv file features the following columns:
    'elapsed_episodes', 'handled_experiences', 'winrate'
'''


def train_against_fixed_agent(task: 'Task',
                              agent: 'Agent',
                              test_agent: 'Agent',
                              base_path: str,
                              run_id: int,
                              exper_config: Dict,
                              summary_writer: SummaryWriter):
    logger = logging.getLogger(f'Run: {run_id}. TRAINING: Task: {task.name}. Agent: {agent.name} Opponent: {test_agent.name}')
    logger.info('Started')

    # Create directory structure
    create_directory_structure(base_path, agent)

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
        run_id=run_id,
        logger=logger,
        summary_writer=summary_writer)


def create_directory_structure(base_path: str, agent: 'Agent'):
    '''
    Creates the directory structure defined at the top of this file
    + all csv files inside winrates/ directory, which are these four:

    apprentice_only.csv                 (always present)
    vanilla_exit.csv                    (always present)
    true_opponent_model.csv            (if applicable)
    learnt_opponent_model.csv  (if applicable)

    '''
    winrates_path = f'{base_path}/winrates/'
    os.makedirs(winrates_path, exist_ok=True)

    # Apprentice_only and vanilla_exit are necessary
    if not os.path.isfile(f'{winrates_path}/apprentice_only.csv'): open(f'{winrates_path}/apprentice_only.csv', 'w')
    if not os.path.isfile(f'{winrates_path}/vanilla_exit.csv'): open(f'{winrates_path}/vanilla_exit.csv', 'w')

    # Proper BRExIt, the agent uses true opponent models and agent modelling
    if agent.use_true_agent_models_in_mcts and agent.use_agent_modelling:
        if not os.path.isfile(f'{winrates_path}/true_opponent_model.csv'): open(f'{winrates_path}/true_opponent_model.csv', 'w')  # This creates a file
        if not os.path.isfile(f'{winrates_path}/learnt_opponent_model.csv'): open(f'{winrates_path}/learnt_opponent_model.csv', 'w')  # This creates a file

    # Soft BRExIt, only opponent models are used
    if agent.use_learnt_opponent_models_in_mcts and agent.use_agent_modelling:
        if not os.path.isfile(f'{winrates_path}/learnt_opponent_model.csv'): open(f'{winrates_path}/learnt_opponent_model.csv', 'w')  # This creates a file


def train_to_a_desired_winrate(task: 'Task',
                               training_agent: 'Agent',
                               opponent: 'Agent',
                               desired_winrate: float,
                               training_episodes: int,
                               benchmarking_episodes: int,
                               agent_position: int,
                               num_envs: int,
                               base_path: str,
                               run_id: int,
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
        logger.info(f'Run: {run_id}. Completed_episodes: {task.total_episodes_run}. Training for extra {training_episodes}')
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
        training_agent_winrate = winrates_during_training[agent_position]
        tolerance = 0.1
        logger.info(f'Winrate during training {training_agent_winrate}, desired - tolerance: {desired_winrate - tolerance}')

        summary_writer.add_scalar('Training/Winrate_episodes', training_agent_winrate, completed_iterations)
        summary_writer.add_scalar('Training/Winrate_handled_experiences', training_agent_winrate, training_agent.handled_experiences)

        logger.info(f'Proceding to benchmark for {benchmarking_episodes} episodes')
        benchmark_winrate = compute_and_log_winrates_for_all_agent_variations(
            training_agent_winrate=training_agent_winrate,
            task=task,
            agent=training_agent,
            opponent=opponent,
            agent_position=agent_position,
            benchmarking_episodes=benchmarking_episodes,
            starting_episode=task.total_episodes_run,
            save_path=f'{base_path}/winrates',
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
                                     num_episodes=training_episodes)
    training_duration = time.time() - training_start
    logger.info('Training for {} took {:.2} seconds'.format(
                training_episodes, training_duration))
    return trajectories


def compute_and_log_winrates_for_all_agent_variations(training_agent_winrate: float,
                                                      task: Task,
                                                      agent: 'Agent',
                                                      opponent: 'Agent',
                                                      agent_position, benchmarking_episodes,
                                                      starting_episode: int,
                                                      save_path: str,
                                                      logger,
                                                      summary_writer: Optional[SummaryWriter] = None) -> float:
    '''
    Benchmarks :param: agent, and its variations.
    NOTE: Not all variations will be present for all training agents:
        - Apprentice only agent: Uses apprentice (neural network) to select actions
                           instead of search.
        - Vanilla Expert Iteration: Using apprentice to compute action priors
                                    for all agent nodes.
        - BRExIt: Using true opponent models to compute priors during their
                  turns
        - Learnt opponent models agent: If :param: agent uses true opponent
                                        models during training, it benchmarks
                                        using learnt opponent models, embedded
                                        in the apprentice neural net.

    This function categorizes :param: training_agent_winrate by looking at the
    iternals of :param: agent.
    :params training_agent_winrate: Winrate obtained during training
    :returns: Winrate of apprentice only version of :param: agent against
              :param: opponent
    '''

    ## Categorize algorithm
    is_brexit = agent.use_true_agent_models_in_mcts and agent.use_agent_modelling
    is_brexit_learnt_models = agent.use_learnt_opponent_models_in_mcts and agent.use_agent_modelling
    is_exit_opponent_modelling = not(agent.use_learnt_opponent_models_in_mcts or agent.use_true_agent_models_in_mcts) and agent.use_agent_modelling
    is_vanilla_exit = not(agent.use_learnt_opponent_models_in_mcts or agent.use_true_agent_models_in_mcts) and not(agent.use_agent_modelling)

    ## Compute necessary winrates
    (true_models_winrate, learnt_opponent_model_winrate,
     vanilla_exit_winrate, apprentice_only_winrate) = None, None, None, None
    if is_brexit:
        true_models_winrate = training_agent_winrate
        learnt_opponent_model_winrate = compute_learnt_opponent_model_winrate(task, agent, opponent, agent_position, benchmarking_episodes, logger)
        vanilla_exit_winrate = compute_vanilla_exit_winrate(task, agent, opponent, agent_position, benchmarking_episodes, logger)
    elif is_brexit_learnt_models:
        learnt_opponent_model_winrate = training_agent_winrate
        vanilla_exit_winrate = compute_vanilla_exit_winrate(task, agent, opponent, agent_position, benchmarking_episodes, logger)
    elif is_exit_opponent_modelling or is_vanilla_exit:
        vanilla_exit_winrate = training_agent_winrate

    # All agent variations have to compute this
    apprentice_only_winrate = compute_apprentice_only_winrate(task, agent, opponent, agent_position, benchmarking_episodes, logger)
    log_winrates_during_benchmark(
        true_models_winrate,
        learnt_opponent_model_winrate,
        vanilla_exit_winrate,
        apprentice_only_winrate,
        starting_episode,
        agent.handled_experiences,
        summary_writer,
        logger,
        save_path
    )
    return apprentice_only_winrate


def compute_apprentice_only_winrate(task, agent, opponent, agent_position, benchmarking_episodes, logger):
    apprentice_nn_agent = build_NeuralNet_Agent(
        task,
        {'neural_net': agent.apprentice,
         'state_preprocess_fn': flatten_last_dim_and_batch_vector_observation},
         f'Apprentice_nn_agent'
    )
    apprentice_nn_winrate = benchmark_single_agent(
        agent=apprentice_nn_agent,
        task=task, opponent=opponent,
        agent_position=agent_position,
        benchmarking_episodes=benchmarking_episodes,
        logger=logger,
        caption='apprentice_only'
    )
    return apprentice_nn_winrate


def compute_learnt_opponent_model_winrate(task, agent, opponent, agent_position, benchmarking_episodes, logger):
    learnt_opponent_model_agent = deepcopy(agent)
    learnt_opponent_model_agent.use_learnt_opponent_models_in_mcts = True

    # Compute winrates
    learnt_opponent_model_winrate = benchmark_single_agent(
        agent=learnt_opponent_model_agent,
        task=task, opponent=opponent,
        agent_position=agent_position,
        benchmarking_episodes=benchmarking_episodes,
        logger=logger,
        caption='learnt_opponent_model'
    )
    return learnt_opponent_model_winrate


def compute_vanilla_exit_winrate(task, agent, opponent, agent_position, benchmarking_episodes, logger):
    vanilla_exit = deepcopy(agent)
    vanilla_exit.use_learnt_opponent_models_in_mcts = False
    vanilla_exit.use_true_agent_models_in_mcts = False

    # Compute winrates
    vanilla_exit_winrate = benchmark_single_agent(
        agent=vanilla_exit,
        task=task, opponent=opponent,
        agent_position=agent_position,
        benchmarking_episodes=benchmarking_episodes,
        logger=logger,
        caption='vanilla_exit'
    )
    return vanilla_exit_winrate


def log_winrates_during_benchmark(true_models_winrate: Optional[float],
                                  learnt_opponent_model_winrate: Optional[float],
                                  vanilla_exit_winrate: Optional[float],
                                  apprentice_only_winrate: Optional[float],
                                  elapsed_episodes: int, handled_experiences: int,
                                  summary_writer,
                                  logger,
                                  save_path):
    '''
    Logs in :param logger, :param: summary_writer and in files at
    :param: save_path the :param: winrates previously computed
    '''
    if true_models_winrate is not None:
        log_winrate_metric(true_models_winrate, elapsed_episodes, handled_experiences, 'true_opponent_model', f'{save_path}/true_opponent_model.csv', logger, summary_writer)
    if learnt_opponent_model_winrate is not None:
        log_winrate_metric(learnt_opponent_model_winrate, elapsed_episodes, handled_experiences, 'learnt_opponent_model', f'{save_path}/learnt_opponent_model.csv', logger, summary_writer)
    if vanilla_exit_winrate is not None:
        log_winrate_metric(vanilla_exit_winrate, elapsed_episodes, handled_experiences, 'vanilla_exit', f'{save_path}/vanilla_exit.csv', logger, summary_writer)
    if apprentice_only_winrate is not None:
        log_winrate_metric(apprentice_only_winrate, elapsed_episodes, handled_experiences, 'apprentice_only', f'{save_path}/apprentice_only.csv', logger, summary_writer)


def log_winrate_metric(winrate: float, elapsed_episodes: int,
                       handled_experiences: int, caption: str, file_path: str,
                       logger, summary_writer):
    logger.info('Benchmarking {}: {:.2}'.format(caption, winrate))
    if summary_writer:
        summary_writer.add_scalar(f'Benchmarking/Winrate_{caption}', winrate, elapsed_episodes)
        summary_writer.add_scalar(f'Benchmarking/Winrate_{caption}_handled_exps', winrate, handled_experiences)
    with open(file_path, 'a') as f:
        f.write(f'{elapsed_episodes},{handled_experiences},{winrate}\n')


def benchmark_single_agent(task: Task,
                           agent: 'Agent', opponent: 'Agent',
                           agent_position: int, benchmarking_episodes: int,
                           caption: str,
                           logger) -> float:
    '''
    Obtains winrate of :param: agent playing in :param: agent_position vs
    :param: opponent in :param: task.
    '''
    agent_vector = [opponent]
    agent_vector.insert(agent_position, agent)
    training_start = time.time()
    trajectories = task.run_episodes(agent_vector, training=False,
                                     num_envs=-1,  # Max number of environments
                                     num_episodes=benchmarking_episodes)
    benchmarking_time = time.time() - training_start
    logger.info('Benchmarking for {} episodes for {} took {:.2} seconds'.format(
                benchmarking_episodes, caption, benchmarking_time))

    winrate = len(list(filter(lambda t: t.winner == agent_position,
                              trajectories))) / len(trajectories)
    return winrate


def save_trained_policy(trained_agent, save_path: str, logger):
    logger.info(f'Saving agent \'{trained_agent.name}\' in \'{save_path}\'')
    torch.save(trained_agent, save_path)


def initialize_experiment(experiment_config, agents_config, args):
    task = create_task_from_config(experiment_config['environment'])
    agents = initialize_agents(task, agents_config)
    test_agent = build_NeuralNet_Agent(
        task,
        {'neural_net': torch.load(args.opponent_path).algorithm.model,
        #'pre_processing_fn': batch_vector_observation},
        'state_preprocess_fn': keep_last_stack_and_batch_vector_observation
        },
        f'TestAgent'
    )
    return task, agents, test_agent

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Expert Iteration / Best Response Expert Iteration agents for experiments of paper "On Opponent Modelling in Expert Iteration"')
    parser.add_argument('--config', required=True, help='path to YAML config file containing info about environment and agents')
    parser.add_argument('--opponent_path', required=True, help='Path to *.pt file containing an agent to train against (opponent)')
    parser.add_argument('--agent_index', required=False, default=None, help='Optional. Index of the agent that will be kept out of all of the agents specified in config file. Useful for batch jobs in SLURM settings')
    parser.add_argument('--run_id', required=True,
                        help='Identifier for the single_run that will be run. Ignoring number of runs in config file.')
    args = parser.parse_args()

    # Spawn is required for GPU to be used inside of neural_net_server
    torch.multiprocessing.set_start_method('spawn')

    logging.basicConfig(level=logging.INFO)
    top_level_logger = logging.getLogger('BRExIt Opponent Modelling Experiment')

    exper_config, agents_config = load_configs(args.config)
    task, agents, test_agent = initialize_experiment(exper_config, agents_config, args)

    if (len(exper_config['algorithms']) > 1) and (args.agent_index is None):
        raise ValueError('More than one agent was specified. Use `agent_index` to select which one to use')

    agent_index = int(args.agent_index) if args.agent_index else 0
    agent_name = exper_config['algorithms'][agent_index]
    agent = agents[agent_index]

    base_path = f"{exper_config['experiment_id']}/run-{args.run_id}/{agent_name}"

    # Maybe this should go into initialize_experiment
    if os.path.exists(base_path) and (os.listdir(base_path) != ['winrates']):  # Load pre-trained agent, if there is any
        top_level_logger.info(f'Attempting to load agent from: {base_path}')
        agent = load_agent_and_update_task(base_path, task)
        top_level_logger.info(f'Loaded agent, with {agent.finished_episodes} episodes under its belt')

    log_path = f"{exper_config['experiment_id']}_logs/run-{args.run_id}/{agent_name}"

    summary_writer = SummaryWriter(log_path)
    agent.algorithm.summary_writer = summary_writer

    train_against_fixed_agent(
        task,
        agent,
        test_agent,
        base_path,
        args.run_id,
        exper_config,
        summary_writer
    )
