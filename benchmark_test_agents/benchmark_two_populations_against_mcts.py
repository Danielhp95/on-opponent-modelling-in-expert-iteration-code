import argparse

import os
from itertools import chain
from loguru import logger

import numpy as np
import gym_connect4
import regym
from regym.environments import generate_task, EnvType
from regym.game_theory import compute_winrate_matrix_metagame, compute_nash_averaging, relative_population_performance
from regym.plotting.game_theory import plot_winrate_matrix
from regym.rl_algorithms import load_population_from_path, build_MCTS_Agent
from regym.networks.preprocessing import batch_vector_observation

from benchmark_skill_against_mcts import estimate_mcts_equivalent_strength


DESCRIPTION = \
    '''
Maybe PSRO trains better against than FullHistory, as measured by
relative_population_performance. however, does this superiority remains when
benchmarked against planning methods such as MCTS?

GOAL: Inspect if PSRO, with initial superiority, still maintains it against MCTS.

    - Generating agents
        - First we train a population of PPO agents with PSRO
        - Repeat with FullHistory
    - Bennchmarking agents
        - Inspect inter-population empirical winrate matrices / Nash Averaging as a sanity check that they are learning
        - Find MCTS equivalent strength for the agents with the largest Nash (for each population)! (yeeeeah, that's neat)
        - Compare MCTS strength for PSRO and FullHistory. Do we see superior MCTS strength on PSRO-trained agents?
    '''


def main(population_1, population_2, task):
    episodes_per_matchup = 500

    pop_1_winrate_matrix = compute_winrate_matrix_metagame(
        population_1,
        episodes_per_matchup,
        task,
        is_game_symmetrical=False,
        num_envs=-1,
        show_progress=True
    )
    pop_2_winrate_matrix = compute_winrate_matrix_metagame(
        population_2,
        episodes_per_matchup,
        task,
        is_game_symmetrical=False,
        num_envs=-1,
        show_progress=True
    )

    pop_1_maxent_nash, pop_1_nash_averaging = compute_nash_averaging(
            pop_1_winrate_matrix, perform_logodds_transformation=True)
    pop_2_maxent_nash, pop_2_nash_averaging = compute_nash_averaging(
            pop_2_winrate_matrix, perform_logodds_transformation=True)

    rel_pop_performance, cross_pop_winrate_matrix = relative_population_performance(
        population_1, population_2,
        task,
        episodes_per_matchup=episodes_per_matchup)

    top_3_pop_1 = [
        population_1[agent_id]
        for agent_id, nash in sorted(enumerate(pop_1_nash_averaging), key=lambda x: x[1], reverse=True)[:3]]
    top_3_pop_2 = [
        population_2[agent_id]
        for agent_id, nash in sorted(enumerate(pop_2_nash_averaging), key=lambda x: x[1], reverse=True)[:3]]

    initial_mcts_config = {'budget': 30, 'rollout_budget': 100,
                           'selection_phase': 'ucb1',
                           'exploration_factor_ucb1': 1.41,
                           'use_dirichlet': False,
                           'dirichlet_alpha': None}
    mcts_1, strength_estimation_df_1 = estimate_mcts_equivalent_strength(
        agent=top_3_pop_1[0], task=task,
        desired_winrate=0.8,
        initial_mcts_config=initial_mcts_config,
        benchmarking_episodes=300,
        show_progress=False
    )
    logger.info(f'MCTS strength for population 1: {mcts_1}')
    mcts_2, strength_estimation_df_2 = estimate_mcts_equivalent_strength(
        agent=top_3_pop_2[0], task=task,
        desired_winrate=0.8,
        initial_mcts_config=initial_mcts_config,
        benchmarking_episodes=300,
        show_progress=False,
    )
    logger.info(f'MCTS strength for population 2: {mcts_2}')
    return pop_1_winrate_matrix, pop_2_winrate_matrix, pop_1_nash_averaging, pop_2_nash_averaging, rel_pop_performance, cross_pop_winrate_matrix, strength_estimation_df_1, strength_estimation_df_2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--population1', required=True, help='Path to directory with agents for population 1')
    parser.add_argument('--population2', required=True, help='Path to directory with agents for population 2')
    parser.add_argument('--save_path', required=True, help='Path where directory will be created hosting all resulting info')
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=False)

    population_1 = load_population_from_path(path=args.population1, state_preprocess_fn=batch_vector_observation, show_progress=True)
    population_2 = load_population_from_path(path=args.population2, state_preprocess_fn=batch_vector_observation, show_progress=True)
    for agent in chain(population_1, population_2):
        agent.requires_environment_model, agent.training = False, False

    #initial_mcts_config = {'budget': 45, 'rollout_budget': 100,
    #                       'selection_phase': 'ucb1',
    #                       'exploration_factor_ucb1': 1.41,
    #                       'use_dirichlet': False,
    #                       'dirichlet_alpha': None}
    #mcts_2, strength_estimation_df_2 = estimate_mcts_equivalent_strength(
    #    agent=population_1[-1],
    #    desired_winrate=0.8,
    #    task=generate_task('Connect4-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION),
    #    initial_mcts_config=initial_mcts_config,
    #    benchmarking_episodes=200,
    #    budget_step=2,
    #    show_progress=False,
    #)
    #logger.info(mcts_2)
    #logger.info(strength_estimation_df_2)

    (pop_1_winrate_matrix, pop_2_winrate_matrix,
    pop_1_nash_averaging, pop_2_nash_averaging,
    rel_pop_performance, cross_pop_winrate_matrix,
    strength_estimation_df_1, strength_estimation_df_2) = main(population_1[25:43], population_2[25:43], task=generate_task('Connect4-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION))

    # Save all stuff
    logger.info(f'Saving everything in {args.save_path}')
    np.savetxt(os.path.join(args.save_path, 'pop_1_winrate_matrix.csv'), pop_1_winrate_matrix, delimiter=',')
    np.savetxt(os.path.join(args.save_path, 'pop_2_winrate_matrix.csv'), pop_2_winrate_matrix, delimiter=',')
    np.savetxt(os.path.join(args.save_path, 'pop_1_nash_averaging.csv'), pop_1_nash_averaging, delimiter=',')
    np.savetxt(os.path.join(args.save_path, 'pop_2_nash_averaging.csv'), pop_2_nash_averaging, delimiter=',')
    np.savetxt(os.path.join(args.save_path, 'cross_pop_winrate_matrix.csv'), cross_pop_winrate_matrix, delimiter=',')
    strength_estimation_df_1.to_csv(os.path.join(args.save_path, 'mcts_equivalent_strenght_estimation_df_pop_1.csv'))
    strength_estimation_df_2.to_csv(os.path.join(args.save_path, 'mcts_equivalent_strenght_estimation_df_pop_2.csv'))
