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
from regym.rl_loops import compute_winrates


'''
Description:

In order to have an "objective" of agent skill, we can use MCTS agents.
By slowly increasing the computational budget of MCTS agents, we can see how much
computational power an MCTS requires to beat our agents. The higher the
computational power, the stronger our (non-MCTS) agent is.

We denote this metric as the MCTS equivalent strength of an agent
'''


def estimate_mcts_equivalent_strength(agent: regym.rl_algorithms.agents.Agent,
                                      task: regym.environments.Task,
                                      desired_winrate: float,
                                      initial_mcts_config: Dict,
                                      max_budget: int=2000,
                                      budget_step: int=1,
                                      benchmarking_episodes: int = 1000,
                                      show_progress: bool=False) \
                                      -> Tuple[int, pd.DataFrame]:
    '''
    Computes MCTS budget for an MCTS agent (with :param: initial_mcts_config)
    required to reach a :param: desired_winrate against :param: agent in
    :param: task.

    Winrates are estimated by playing :param: benchmarking_episodes between both agents.
    If :param: desired_winrate is not met for a given buget, the budget is increased
    by :param: budget_step and the process is repeated. If MCTS agent doesn't reach
    :param: desired_winrate before :param: max_budget, this function returns.

    TODO: mention that we are talking about a non-symmetrical game

    :param agent: Agent to be benchmarked against MCTS
    :param task: Task where :param: agent play against MCTS
    :param desired_winrate: Target winrate at wich,
    :param initial_mcts_config: Initial configuration for MCTS agent.
    :param show_progress: Logger for debugging and monitoring purposes
    :param budget_step: Value to increase budget by.
    :param benchmarking_episodes: Number of episodes to be used to estimate winrate between agents
    :returns:
        - MCTS equivalent strength
        - pd.DataFrame containing logs about winrates observed during strength estimation
    '''
    config = initial_mcts_config.copy()
    dfs = []
    for budget in range(initial_mcts_config['budget'], max_budget, budget_step):
        if show_progress:
            logger.info(f'Starting benchmarking with BUDGET: {budget}')

        config['budget'] = budget
        mcts_agent = build_MCTS_Agent(task, config, f'MCTS-{budget}')

        traj_1 = task.run_episodes([agent, mcts_agent],
                                   num_episodes=(benchmarking_episodes // 2),
                                   num_envs=-1, training=False)
        traj_2 = task.run_episodes([mcts_agent, agent],
                                   num_episodes=(benchmarking_episodes // 2),
                                   num_envs=-1, training=False)

        winrates_1, winrates_2 = compute_winrates(traj_1), compute_winrates(traj_2)
        trained_agent_victories_traj_1 = [int(t.winner == 0) for t in traj_1]
        trained_agent_victories_traj_2 = [int(t.winner == 1) for t in traj_2]

        #trained_agent_avg_winrate = (winrates_1[0] + winrates_2[1]) / 2

        df_pos_1 = pd.DataFrame({
            'test_agent_id': agent.handled_experiences,
            'mcts_budget': budget,
            'position': 0,
            'victories': trained_agent_victories_traj_1
        })
        df_pos_2 = pd.DataFrame({
            'test_agent_id': agent.handled_experiences,
            'mcts_budget': budget,
            'position': 1,
            'victories': trained_agent_victories_traj_2
        })
        dfs += [pd.concat([df_pos_1, df_pos_2], ignore_index=True)]

        trained_agent_avg_winrate = (df_pos_1['victories'].mean() + df_pos_2['victories'].mean()) / 2
        if show_progress:
            logger.info(f'WINRATES: Total = {trained_agent_avg_winrate}\tPos 0 = {winrates_1[0]}\t Pos 1 = {winrates_2[1]}')

        if df_pos_1['victories'].mean() < desired_winrate:
            return budget, pd.concat(dfs)


def main(population: List['Agent'], task, winrate_threshold: float, logger) -> pd.DataFrame:
    initial_mcts_config = {'budget': 10, 'rollout_budget': 500,
                           'selection_phase': 'ucb1',
                           'exploration_factor_ucb1': 1.41,
                           'use_dirichlet': False,
                           'dirichlet_alpha': None}
    strength_estimation_dfs = []

    for agent in reversed(population):
        logger.info(f'Benchmarking agent with {agent.algorithm.num_updates} number of updates and {agent.finished_episodes} finished episodes')

        agent_strength, agent_specific_strength_estimation_df = estimate_mcts_equivalent_strength(
                agent, task, winrate_threshold, initial_mcts_config, show_progress=True)
        strength_estimation_dfs += [agent_specific_strength_estimation_df]

        logger.info(f'Agent strength: {agent_strength} (MCTS budget)')
    strength_estimation_dfs = pd.concat(strength_estimation_dfs, ignore_index=True)
    strength_estimation_dfs.to_csv('mcts_equivalent_strenght_estimation_df.csv')
    return strength_estimation_dfs


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
    parser.add_argument('--winrate_threshold', required=True, help='Winrate threshold for MCTS equivalent strength')
    args = parser.parse_args()

    population = load_population_from_path(path=args.path, show_progress=True, state_preprocess_fn=batch_vector_observation)
    population.sort(key=lambda agent: agent.finished_episodes)
    # population = [population[-1]]

    for agent in population:
        agent.requires_environment_model, agent.training = False, False

    main(population, generate_task('Connect4-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION), float(args.winrate_threshold), logger)
