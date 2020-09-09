import os
import gym_connect4
import argparse

import regym
from regym.environments import generate_task, EnvType
from regym.game_theory import compute_winrate_matrix_metagame, compute_nash_averaging
from regym.rl_algorithms import load_population_from_path

import numpy as np

'''
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


def main(path: str, name: str):
    task = generate_task('Connect4-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION)

    #sort_fn = lambda x: int(x.split('_')[-1][:-3])  # ExIt
    sort_fn = lambda x: int(x.split('/')[-1].split('_')[0])  # PPO test training
    sorted_population = load_population_from_path(path=path, sort_fn=sort_fn)

    for agent in sorted_population:
        print(agent.algorithm.num_updates)
        agent.requires_environment_model = False
        agent.training = False

    winrate_matrix = compute_winrate_matrix_metagame(
            population=sorted_population, episodes_per_matchup=1000, task=task)
    maxent_nash, nash_averaging = compute_nash_averaging(
            winrate_matrix, perform_logodds_transformation=True)

    winrate_matrix = np.array(winrate_matrix)
    print('Saving winrate_matrix, max-entropy Nash equilibrium for game defined by winrate matrix and Nash averaging')
    np.savetxt(f'{name}_winrate_matrix.csv', winrate_matrix, delimiter=', ')
    np.savetxt(f'{name}_maxent_nash.csv', maxent_nash, delimiter=', ')
    np.savetxt(f'{name}_nash_averaging.csv', maxent_nash, delimiter=', ')

    ax = plot_winrate_matrix(winrate_matrix)
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computes winrate matrices and Nash averagings for test agents of paper "On Opponent Modelling in Expert Iteration"')
    parser.add_argument('--path', required=True, help='Path to directory containing trained agents to be benchmarked')
    parser.add_argument('--name', required=True, help='Identifier, used in file creation')
    args = parser.parse_args()
    main(path=args.path, name=args.name)
