from typing import Dict, Tuple, List
from functools import partial
import argparse
import logging
import gym_connect4

import pandas as pd

import regym
from regym.environments import generate_task, EnvType
from regym.environments.wrappers import FrameStack
from regym.networks.preprocessing import batch_vector_observation
from regym.rl_algorithms import load_population_from_path
from regym.rl_algorithms import build_MCTS_Agent
from regym.networks.preprocessing import flatten_last_dim_and_batch_vector_observation


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

        winrates_1 = [len(list(filter(lambda t: t.winner == a_i, traj_1))) / len(traj_1)
            for a_i in range(2)]
        winrates_2 = [len(list(filter(lambda t: t.winner == a_i, traj_2))) / len(traj_2)
            for a_i in range(2)]

        avg_winrate = (winrates_1[0] + winrates_2[1]) / 2

        df = df.append({'test_agent_id': agent.handled_experiences ,
                        'mcts_budget': budget,
                        'winrate_pos_0': winrates_1[0],
                        'winrate_pos_1': winrates_2[1],
                        'avg_winrate': avg_winrate}, ignore_index=True)

        logger.info(f'WINRATES: Total = {avg_winrate}\tPos 0 = {winrates_1[0]}\t Pos 1 = {winrates_2[1]}')

        if avg_winrate < desired_winrate:
            return budget, df


def main(population: List['Agent'], logger, num_stack: int):
    initial_mcts_config = {'budget': 20, 'rollout_budget': 100,
                           'selection_phase': 'ucb1',
                           'exploration_factor_ucb1': 1.41,
                           'use_dirichlet': False,
                           'dirichlet_alpha': None}
    task = generate_task(
        'Connect4-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION,
        wrappers=create_wrapper(
            num_stack=num_stack
        )
    )

    strength_estimation_df = pd.DataFrame(columns=('test_agent_id', 'mcts_budget', 'winrate_pos_0',
                               'winrate_pos_1', 'avg_winrate'))

    for agent in reversed(population):
        logger.info(f'Benchmarking agent with {agent.algorithm.num_updates} number of updates and {agent.finished_episodes} finished episodes')

        agent_strength, agent_specific_strength_estimation_df = estimate_agent_strength(
                agent, task, 0.5, initial_mcts_config, logger)
        strength_estimation_df = strength_estimation_df.append(
            agent_specific_strength_estimation_df, ignore_index=True)

        logger.info(f'Agent strength: {agent_strength} (MCTS budget)')
    strength_estimation_df.to_csv('mcts_equivalent_strenght_estimation_df.csv')


def create_wrapper(num_stack: int):
    frame_stack_wrapper = partial(FrameStack, num_stack=num_stack)
    return [frame_stack_wrapper]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('MCTS_benchmarking')
    fh = logging.FileHandler('mcts_strength_benchmark.logs')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)


    parser = argparse.ArgumentParser(description='Estimates the skill of agents by playing against increasingly strong MCTS agents')
    parser.add_argument('--path', required=True, help='Path to directory containing trained agents to be benchmarked')
    parser.add_argument('--num_stack', required=True, help='Number of FrameStack(s)')
    args = parser.parse_args()

    population = load_population_from_path(path=args.path, show_progress=True)
    population.sort(key=lambda agent: agent.finished_episodes)

    for agent in population:
        agent.requires_environment_model = False
        agent.training = False
        # If not using frame stack: TODO
        # If using frame stack
        agent.state_preprocess_fn = flatten_last_dim_and_batch_vector_observation

    main(population, logger, int(args.num_stack))
