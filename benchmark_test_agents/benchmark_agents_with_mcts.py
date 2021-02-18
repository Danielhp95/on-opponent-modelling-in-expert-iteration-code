from typing import List
import os
import argparse

import matplotlib.pyplot as plt
import gym_connect4

import regym
from regym.environments import generate_task, EnvType
from regym.game_theory import compute_winrate_matrix_metagame, compute_nash_averaging
from regym.plotting.game_theory import plot_winrate_matrix
from regym.rl_algorithms import load_population_from_path, build_MCTS_Agent

import numpy as np

'''

TODO: merge this with benchmark_agents.py
This script is a copy of benchmark_agents.py but adds a list
of MCTS agents to the benchmark.

DESCRIPTION:

Once we have generated a population of agents via the
generate_test_agents.py script. We can benchmark them using this script.
By benchmarking, we mean making each combination of agent face one another
for a large number of matches, to obtain an empirical winrate for each agent
combination.

This script generates 3 .csv files:
    - Winrate matrix for all agents found in :param path:
    - Support for the maximum entropy Nash equilibrium for the strategic form
      game expressed by the aforementioned winrate matrix.
    - The associated Nash averaging
      (look up paper by David Balduzzi: Re-evaluating evaluation)
'''


def main(population: List, name: str):
    task = generate_task(
        'Connect4-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION,
        wrappers=create_wrapper(
            num_stack=4
        )
    )

    winrate_matrix = compute_winrate_matrix_metagame(
            population=population, episodes_per_matchup=200, task=task)
    maxent_nash, nash_averaging = compute_nash_averaging(
            winrate_matrix, perform_logodds_transformation=True)

    winrate_matrix = np.array(winrate_matrix)
    print('Saving winrate_matrix, max-entropy Nash equilibrium for game defined by winrate matrix and Nash averaging')
    np.savetxt(f'{name}/winrate_matrix.csv', winrate_matrix, delimiter=', ')
    np.savetxt(f'{name}/maxent_nash.csv', maxent_nash, delimiter=', ')
    np.savetxt(f'{name}/nash_averaging.csv', maxent_nash, delimiter=', ')

    ax = plot_winrate_matrix(winrate_matrix)

    plt.show()

def create_wrapper(num_stack: int):
    frame_stack_wrapper = partial(FrameStack, num_stack=num_stack)
    return [frame_stack_wrapper]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computes winrate matrices and Nash averagings for test agents of paper "On Opponent Modelling in Expert Iteration"')
    parser.add_argument('--path', required=True, help='Path to directory containing trained agents to be benchmarked')
    parser.add_argument('--name', required=True, help='Identifier, used in file creation')
    args = parser.parse_args()
    os.mkdir(args.name)

    ### To refactor at some point
    #sort_fn = lambda x: int(x.split('_')[-1][:-3])  # ExIt
    sort_fn = lambda x: int(x.split('/')[-1].split('_')[0])  # PPO test training
    sorted_population = load_population_from_path(path=args.path, sort_fn=sort_fn)
    sorted_population.sort(key=lambda agent: agent.finished_episodes)

    for agent in sorted_population:
        agent.requires_environment_model = False
        agent.training = False
    ###

    # Taken from MCTS equivalent strength benchmarking
    mcts_budgets = [29, 42, 42, 38, 45, 56, 48, 49, 51, 42, 53, 46, 35, 49, 49,
                    42, 45, 40, 45, 42, 47, 38, 42, 47, 45, 37, 42, 35, 39, 25,
                    38, 34, 33, 38, 40]
    mcts_population = []
    for budget in mcts_budgets:
        initial_mcts_config = {'budget': budget, 'rollout_budget': 100,
                               'selection_phase': 'ucb1',
                               'exploration_factor_ucb1': 1.41,
                               'use_dirichlet': False,
                               'dirichlet_alpha': None}
        mcts_population.append(
            build_MCTS_Agent(generate_task('Connect4-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION),
                             initial_mcts_config, agent_name=f'MCTS:{budget}')
        )

    main(population=sorted_population+mcts_population, name=args.name)
