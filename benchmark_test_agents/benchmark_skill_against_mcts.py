from typing import Dict, Tuple
import argparse
import logging
import gym_connect4

import pandas as pd

import regym
from regym.environments import generate_task, EnvType
from regym.util.play_matches import extract_winner
from regym.networks.preprocessing import batch_vector_observation
from regym.rl_algorithms import load_population_from_path
from regym.rl_algorithms import build_NeuralNet_Agent, build_MCTS_Agent


'''
Description:

In order to have an "objective" of agent skill, we can use MCTS agents.
By slowly increasing the computational budget of MCTS agents, we can see how much
computational power an MCTS requires to beat our agents. The higher the
computational power, the stronger our (non-MCTS) agent is.

We denote this metric as the MCTS equivalent strength of an agent
'''


def estimate_agent_strength(agent: regym.rl_algorithms.agents.Agent,
                            task: regym.environments.Task,
                            desired_winrate: float,
                            initial_mcts_config: Dict,
                            logger,
                            benchmarking_episodes: int = 200) -> Tuple[int, pd.DataFrame]:
    '''
    Computes MCTS budget for an MCTS agent (with :param: initial_mcts_config)
    required to reach a :param: desired_winrate against :param: agent in
    :param: task.

    TODO: mention that we are talking about a non-symmetrical game

    :param agent: TODO
    :param task: TODO
    :param desired_winrate: TODO
    :param initial_mcts_config: TODO
    :param logger: TODO
    :param benchmarking_episodes: TODO
    :returns pd.DataFrame containing logs about winrates observed during
             strength estimation
    '''
    df = pd.DataFrame(columns=('test_agent_id', 'mcts_budget', 'winrate_pos_0',
                               'winrate_pos_1', 'avg_winrate'))

    config = initial_mcts_config.copy()
    for budget in range(initial_mcts_config['budget'], 2000, 1):
        logger.info(f'Starting benchmarking with BUDGET: {budget}')

        config['budget'] = budget
        mcts_agent = build_MCTS_Agent(task, config, f'MCTS-{budget}')

        traj_1 = task.run_episodes([agent, mcts_agent],
                                   num_episodes=(benchmarking_episodes // 2),
                                   num_envs=-1, training=False)
        traj_2 = task.run_episodes([mcts_agent, agent],
                                   num_episodes=(benchmarking_episodes // 2),
                                   num_envs=-1, training=False)

        traj_1_winners = [extract_winner(t) for t in traj_1]  # Our agent is pos: 0
        traj_2_winners = [extract_winner(t) for t in traj_2]  # Our agent is pos: 1
        pos_0_winrate = traj_1_winners.count(0) / len(traj_1)
        pos_1_winrate = traj_2_winners.count(1) / len(traj_2)
        avg_winrate = (pos_0_winrate + pos_1_winrate) / 2

        df = df.append({'test_agent_id': agent.handled_experiences ,
                        'mcts_budget': budget,
                        'winrate_pos_0': pos_0_winrate,
                        'winrate_pos_1': pos_1_winrate,
                        'avg_winrate': avg_winrate}, ignore_index=True)

        logger.info(f'WINRATES: Total = {avg_winrate}\tPos 0 = {pos_0_winrate}\t Pos 1 = {pos_1_winrate}')

        if avg_winrate < desired_winrate:
            return budget, df


def main(path: str, logger):
    initial_mcts_config = {'budget': 10, 'rollout_budget': 100,
                           'selection_phase': 'ucb1',
                           'exploration_factor_ucb1': 1.41,
                           'use_dirichlet': False,
                           'dirichlet_alpha': None}
    task = generate_task('Connect4-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION)

    strength_estimation_df = pd.DataFrame(columns=('test_agent_id', 'mcts_budget', 'winrate_pos_0',
                               'winrate_pos_1', 'avg_winrate'))

    for agent in load_population(path):
        logger.info(f'Benchmarking agent with {agent.algorithm.num_updates} number of updates')
        nn_agent = build_NeuralNet_Agent(task,
                {'neural_net': agent.algorithm.model, 'pre_processing_fn': batch_vector_observation},
                agent_name='NeuralNet')

        agent_strength, agent_specific_strength_estimation_df = estimate_agent_strength(
                nn_agent, task, 0.5, initial_mcts_config, logger)
        strength_estimation_df = strength_estimation_df.append(
            agent_specific_strength_estimation_df, ignore_index=True)

        logger.info(f'Agent strength: {agent_strength} (MCTS budget)')
    strength_estimation_df.to_csv('mcts_equivalent_strenght_estimation_df.csv')


def load_population(path):
    sort_fn = lambda x: int(x.split('/')[-1].split('_')[0])  # PPO test training
    return load_population_from_path(path=path, sort_fn=sort_fn)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('MCTS_benchmarking')
    fh = logging.FileHandler('mcts_strength_benchmark.logs')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)


    parser = argparse.ArgumentParser(description='Estimates the skill of agents by playing against increasingly strong MCTS agents')
    parser.add_argument('--path', required=True, help='Path to directory containing trained agents to be benchmarked')
    args = parser.parse_args()

    main(args.path, logger)
