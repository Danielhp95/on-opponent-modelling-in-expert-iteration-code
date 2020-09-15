from typing import Dict
import argparse
import logging
import gym_connect4

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
'''


def estimate_agent_strength(agent: regym.rl_algorithms.agents.Agent,
                            task: regym.environments.Task,
                            desired_winrate: float,
                            initial_mcts_config: Dict,
                            logger,
                            benchmarking_episodes: int = 200) -> int:
    '''
    Computes MCTS budget required to reach a :param: desired winrate against :param: agent
    '''
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
        logger.info(f'WINRATES: Total = {avg_winrate}\tPos 0 = {pos_0_winrate}\t Pos 1 = {pos_1_winrate}')
        logger.info('')

        if avg_winrate < desired_winrate:
            return avg_winrate


def load_population(path):
    sort_fn = lambda x: int(x.split('/')[-1].split('_')[0])  # PPO test training
    return load_population_from_path(path=path, sort_fn=sort_fn)


def main(path: str, logger):
    initial_mcts_config = {'budget': 10, 'rollout_budget': 100,
                           'selection_phase': 'ucb1',
                           'exploration_factor_ucb1': 1.41,
                           'use_dirichlet': False,
                           'dirichlet_alpha': None}
    task = generate_task('Connect4-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION)
    for agent in load_population(path):
        logger.info(f'Benchmarking agent with {agent.algorithm.num_updates} number of updates')
        nn_agent = build_NeuralNet_Agent(task,
                {'neural_net': agent.algorithm.model, 'pre_processing_fn': batch_vector_observation},
                agent_name='NeuralNet')
        agent_strength = estimate_agent_strength(
                nn_agent, task, 0.5, initial_mcts_config, logger)
        logger.info(f'Agent strength: {agent_strength} (MCTS budget)')



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
