from typing import Callable, Tuple
import pickle
from copy import deepcopy
import os
import sys
import argparse
from functools import partial
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
from loguru import logger

import descriptions


def data_directory_selection_sidebar_widget(path: str):
    selected_results_directory = st.sidebar.text_input('Select results directory',
                                                       path)
    display_selected_directory(selected_results_directory)
    return selected_results_directory


def display_selected_directory(selected_dir: str):
    if st.sidebar.checkbox(f'List files in directory {selected_dir} '):
        for f in os.listdir(selected_dir): st.sidebar.markdown(f)


@st.cache
def load_results(results_dir: str):
    def load_file(base_path: str, path: str):
        file_path = f'{base_path}/{path}'
        if os.path.exists(file_path): return np.loadtxt(file_path, delimiter=', ')
        else: raise FileNotFoundError(f'File {file_path} should exist')

    results = dict()
    results['winrate_matrix_test_agents'] = load_file(results_dir, 'winrate_matrix_test_agents.csv')
    results['maxent_nash_test_agents'] = load_file(results_dir, 'maxent_nash_test_agents.csv')

    results['evolution_nash_averaging_test_agents'] = load_file(results_dir, 'evolution_nash_averaging_test_agents.csv')

    results['winrate_matrix_mcts'] = load_file(results_dir, 'winrate_matrix_mcts.csv')
    results['maxent_nash_mcts'] = load_file(results_dir, 'maxent_nash_test_agents.csv')

    results['mcts_equivalent_strength_test_agents'] = load_file(results_dir, 'mcts_equivalent_strength_test_agents.csv')

    results['winrate_matrix_test_agents_and_mcts'] = load_file(results_dir, 'winrate_matrix_test_agents_and_mcts.csv')
    results['maxent_nash_test_agents_and_mcts'] = load_file(results_dir, 'maxent_nash_test_agents.csv')
    return results


def run(path: str,
        path_1: str, path_2: str, path_3: str,
        path_1_one_hot: str, path_2_one_hot: str, path_3_one_hot: str):
    def plot_component(checkbox_ticked: bool, title: str, description: str,
                       description_key: str, plot_func: Callable):
        if checkbox_ticked:
            st.markdown(title)
            show_description = st.checkbox('Show description', key=description_key)
            if show_description:
                st.markdown(description)
            fig = plot_func()
            plt.tight_layout()
            st.pyplot(fig)

    results_dir = data_directory_selection_sidebar_widget(path)

    results = load_results(results_dir)

    # SIDEBAR
    st.sidebar.markdown('## Plots to show:')
    st.sidebar.markdown('Test agents')
    show_winrate_matrix_test_agents = st.sidebar.checkbox('Winrate matrix for test agents', True)
    show_nash_averaging_evolution_test_agents = st.sidebar.checkbox('Evolution Nash avg test agents', True)
    st.sidebar.markdown('MCTS evaluation for Test agents')
    show_plot_mcts_equivalent_test_agents = st.sidebar.checkbox('MCTS equivalent of test agents', True)
    show_winrate_matrix_mcts = st.sidebar.checkbox('Winrate matrix for MCTS agents', True)
    st.sidebar.markdown('Combined MCTS & Test agents')
    show_winrate_matrix_mcts_and_test_agents = st.sidebar.checkbox('Winrate matrix MCTS & Test agents', True)
    st.sidebar.markdown('Training time till apprentice convergence.')

    path_1 = st.text_input('Select results directory 1', path_1)
    path_2 = st.text_input('Select results directory 2', path_2)
    path_3 = st.text_input('Select results directory 3', path_3)
    path_1_one_hot = st.text_input('Select results directory 1 one hot', path_1_one_hot)
    path_2_one_hot = st.text_input('Select results directory 2 one hot', path_2_one_hot)
    path_3_one_hot = st.text_input('Select results directory 3 one hot', path_3_one_hot)

    show_convergence_times = st.sidebar.checkbox('Convergence times', True)
    winrate_threshold = st.sidebar.slider(label='Winrate threshold for convergence',
                                          min_value=0., max_value=1., step=0.01, value=0.5)

    #plot_component(show_winrate_matrix_test_agents,
    #               title='# Winrate matrix: Test agents',
    #               description=descriptions.winrate_matrix_test_agents,
    #               description_key='description 1',
    #               plot_func=partial(plot_winrate_matrix_and_support,
    #                                 results['winrate_matrix_test_agents'],
    #                                 results['maxent_nash_test_agents']))

    #plot_component(show_nash_averaging_evolution_test_agents,
    #               title='# Evolution of Nash averaging: Test agents',
    #               description=descriptions.nash_averaging_evolutions_test_agents,
    #               description_key='description 2',
    #               plot_func=partial(plot_nash_averaging_evolution_test_agents,
    #                                 results['evolution_nash_averaging_test_agents']))

    #plot_component(show_plot_mcts_equivalent_test_agents,
    #               title='# Plot: MCTS equivalent strength for Test agents',
    #               description=descriptions.mcts_equivalent_strength_test_agents,
    #               description_key='description 3',
    #               plot_func=partial(plot_mcts_equivalent_strength,
    #                                 results['mcts_equivalent_strength_test_agents']))

    #plot_component(show_winrate_matrix_mcts,
    #               title='# Winrate matrix: MCTS agents',
    #               description=descriptions.winrate_matrix_mcts_agents,
    #               description_key='description 4',
    #               plot_func=partial(plot_winrate_matrix_and_support,
    #                                 results['winrate_matrix_test_agents'],
    #                                 results['maxent_nash_mcts']))

    #plot_component(show_winrate_matrix_mcts,
    #               title='# Winrate matrix: MCTS & Test agents',
    #               description=descriptions.winrate_matrix_mcts_and_test_agents,
    #               description_key='description 5',
    #               plot_func=partial(plot_winrate_matrix_mcts_and_test_agents,
    #                                 results['winrate_matrix_test_agents_and_mcts'],
    #                                 results['maxent_nash_test_agents_and_mcts']))

    (parsed_winrates_df_test1, filtered_winrates_df_test1,
    parsed_winrates_df_test2, filtered_winrates_df_test2,
    parsed_winrates_df_test3, filtered_winrates_df_test3,
    parsed_winrates_df_test1_one_hot, filtered_winrates_df_test1_one_hot,
    parsed_winrates_df_test2_one_hot, filtered_winrates_df_test2_one_hot,
    parsed_winrates_df_test3_one_hot, filtered_winrates_df_test3_one_hot,
    all_filtered_winrates, all_parsed_winrates) = \
        load_and_process_df_against_fixed_agent(
            path_1, path_2, path_3, path_1_one_hot, path_2_one_hot, path_3_one_hot,
            winrate_threshold
        )

    plot_component(True,  # whether to show or not
                   title='# Full distribution VS Weak test agent',
                   description=descriptions.apprentice_convergence,
                   description_key='description 7',
                   plot_func=partial(plot_winrate_progression, parsed_winrates_df_test1, filtered_winrates_df_test1, winrate_threshold))

    plot_component(checkbox_ticked=True, title='# Full distribution VS Medium test agent',
                   description=descriptions.apprentice_convergence, description_key='description 8',
                   plot_func=partial(plot_winrate_progression, parsed_winrates_df_test2, filtered_winrates_df_test2, winrate_threshold))

    plot_component(checkbox_ticked=True, title='# Full distribution VS Strong test agent',
                   description=descriptions.apprentice_convergence, description_key='description 9',
                   plot_func=partial(plot_winrate_progression, parsed_winrates_df_test3, filtered_winrates_df_test3, winrate_threshold))

    plot_component(checkbox_ticked=True, title='# One Hot VS Weak test agent',
                   description=descriptions.apprentice_convergence, description_key='description 10',
                   plot_func=partial(plot_winrate_progression, parsed_winrates_df_test1_one_hot, filtered_winrates_df_test1_one_hot, winrate_threshold, True))

    plot_component(checkbox_ticked=True, title='# One Hot VS Medium test agent',
                   description=descriptions.apprentice_convergence, description_key='description 11',
                   plot_func=partial(plot_winrate_progression, parsed_winrates_df_test2_one_hot, filtered_winrates_df_test2_one_hot, winrate_threshold, True))

    plot_component(checkbox_ticked=True, title='# One Hot VS Strong test agent',
                   description=descriptions.apprentice_convergence, description_key='description 12',
                   plot_func=partial(plot_winrate_progression, parsed_winrates_df_test3_one_hot, filtered_winrates_df_test3_one_hot, winrate_threshold, True))

    plot_component(True,  # whether to show or not
                   title=f'# Comparison of convergence time to {winrate_threshold}%',
                   description='TODO',
                   description_key='description 1333',
                   plot_func=partial(plot_convergence_bar_plot_comparison,
                                     all_filtered_winrates))

    #plot_component(True,  # whether to show or not
    #               title=f'# Probability of improvement',
    #               description='TODO',
    #               description_key='description 69',
    #               plot_func=partial(plot_rliable_probability_of_improvement,
    #                                 all_parsed_winrates))


def load_and_process_df_against_fixed_agent(path_1, path_2, path_3, path_1_one_hot, path_2_one_hot, path_3_one_hot, winrate_threshold):
    logger.info('Processing winrate data for Test agent 1. Full distribution')
    parsed_winrates_df_test1, filtered_winrates_df_test1 = process_winrates_from_path_and_threshold(path_1, winrate_threshold)
    logger.info('Processing winrate data for Test agent 2. Full distribution')
    parsed_winrates_df_test2, filtered_winrates_df_test2 = process_winrates_from_path_and_threshold(path_2, winrate_threshold)
    logger.info('Processing winrate data for Test agent 3. Full distribution')
    parsed_winrates_df_test3, filtered_winrates_df_test3 = process_winrates_from_path_and_threshold(path_3, winrate_threshold)
    logger.info('Processing winrate data for Test agent 1. One hot')
    parsed_winrates_df_test1_one_hot, filtered_winrates_df_test1_one_hot = process_winrates_from_path_and_threshold(path_1_one_hot, winrate_threshold, is_one_hot=True)
    logger.info('Processing winrate data for Test agent 2. One hot')
    parsed_winrates_df_test2_one_hot, filtered_winrates_df_test2_one_hot = process_winrates_from_path_and_threshold(path_2_one_hot, winrate_threshold, is_one_hot=True)
    logger.info('Processing winrate data for Test agent 3. One hot')
    parsed_winrates_df_test3_one_hot, filtered_winrates_df_test3_one_hot = process_winrates_from_path_and_threshold(path_3_one_hot, winrate_threshold, is_one_hot=True)
    ###

    # Adding info about which opponent these agents trained against
    filtered_winrates_df_test1['test_agent'] = 'weak'
    filtered_winrates_df_test2['test_agent'] = 'medium'
    filtered_winrates_df_test3['test_agent'] = 'strong'
    filtered_winrates_df_test1_one_hot['test_agent'] = 'weak'
    filtered_winrates_df_test2_one_hot['test_agent'] = 'medium'
    filtered_winrates_df_test3_one_hot['test_agent'] = 'strong'

    ###
    all_filtered_winrates = pd.concat(
    [filtered_winrates_df_test1,
     filtered_winrates_df_test2,
     filtered_winrates_df_test3,
     filtered_winrates_df_test1_one_hot,
     filtered_winrates_df_test2_one_hot,
     filtered_winrates_df_test3_one_hot
    ])
    all_parsed_winrates = pd.concat(
    [filtered_winrates_df_test1,
     filtered_winrates_df_test2,
     filtered_winrates_df_test3,
     filtered_winrates_df_test1_one_hot,
     filtered_winrates_df_test2_one_hot,
     filtered_winrates_df_test3_one_hot
    ])

    return (parsed_winrates_df_test1, filtered_winrates_df_test1,
           parsed_winrates_df_test2, filtered_winrates_df_test2,
           parsed_winrates_df_test3, filtered_winrates_df_test3,
           parsed_winrates_df_test1_one_hot, filtered_winrates_df_test1_one_hot,
           parsed_winrates_df_test2_one_hot, filtered_winrates_df_test2_one_hot,
           parsed_winrates_df_test3_one_hot, filtered_winrates_df_test3_one_hot,
           all_filtered_winrates, all_parsed_winrates)


def plot_convergence_bar_plot_comparison(all_filtered_winrates_df: pd.DataFrame):
    # TODO: figure out a way of sorting them in ascending fashion.
    # Ask in stack overflow?
    fig, ax = plt.subplots(1, 1)

    all_filtered_winrates_df = all_filtered_winrates_df.copy()
    all_filtered_winrates_df['algorithm'] = pd.Categorical(
        all_filtered_winrates_df['algorithm'],
        categories=['BRExIt-OMS-OH', 'BRExIt-OMS', 'BRExIt-OH', 'BRExIt',  'ExIt', 'ExIt-OMFS-OH', 'ExIt-OMFS'],
        ordered=True
    )

    ax = sns.barplot(y='elapsed_episodes', x='test_agent', hue='algorithm',
                     data=all_filtered_winrates_df,
                     ax=ax)
    #plt.legend(bbox_to_anchor=(0., 1.15), loc='upper left', borderaxespad=0, ncol=3)
    plt.legend(bbox_to_anchor=(1., 1.), loc='upper right', borderaxespad=0, ncol=2)
    return fig


def plot_winrate_progression(parsed_winrates_df: pd.DataFrame,
                             filtered_winrates_df: pd.DataFrame,
                             winrate_threshold: float,
                             is_one_hot: bool=False):
    fig = plt.figure()
    gs = fig.add_gridspec(2, 1, hspace=0, wspace=0, height_ratios=[1., 1.5])
    (ax1, ax2) = gs.subplots(sharex='col')

    parsed_winrates_df = parsed_winrates_df.copy()
    filtered_winrates_df = filtered_winrates_df.copy()


    if is_one_hot:
        parsed_winrates_df['algorithm'] = pd.Categorical(
            parsed_winrates_df['algorithm'],
            categories=['BRExIt-OMS-OH', 'BRExIt-OH', 'ExIt-OMFS-OH'],
            ordered=True
        )
        filtered_winrates_df['algorithm'] = pd.Categorical(
            filtered_winrates_df['algorithm'],
            categories=['BRExIt-OMS-OH', 'BRExIt-OH', 'ExIt-OMFS-OH'],
            ordered=True
        )
    else:
        parsed_winrates_df['algorithm'] = pd.Categorical(
            parsed_winrates_df['algorithm'],
            categories=['BRExIt-OMS', 'BRExIt',  'ExIt', 'ExIt-OMFS'],
            ordered=True
        )
        filtered_winrates_df['algorithm'] = pd.Categorical(
            filtered_winrates_df['algorithm'],
            categories=['BRExIt-OMS', 'BRExIt',  'ExIt', 'ExIt-OMFS'],
            ordered=True
        )


    ax2 = sns.lineplot(
        x='elapsed_episodes',
        y='winrate',
        hue='algorithm',
        style='algorithm',
        data=parsed_winrates_df.sort_values('algorithm'),
        ax=ax2
    )

    ## Draw line at target winrate
    ax2.axhline(winrate_threshold, color='black')
    #ax2.text(s='Target', x=300, y=winrate_threshold +0.02)

    #### Adding tick next to winrate threshold
    yt = ax2.get_yticks()
    yt = np.append(yt, winrate_threshold)
    yt = np.around(yt, 2)
    yt = yt[yt >= 0.] # Remove negative values

    ytl = yt.tolist()
    ytl[-1] = winrate_threshold
    ax2.set_yticks(yt)
    ax2.set_yticklabels(ytl)
    ####

    ax2.legend(bbox_to_anchor=(1.0, 1.65),borderaxespad=0)

    ax1 = sns.barplot(x='elapsed_episodes', y='algorithm',
                      data=filtered_winrates_df, ax=ax1,
                      orient='h')
    #ax1.set_yticklabels([''] * len(filtered_winrates_df['algorithm'].unique()))  # Remove long names
    return fig


def process_winrates_from_path_and_threshold(winrates_path: str, winrate_threshold: float,
                                             is_one_hot: bool=False) \
                                             -> Tuple[pd.DataFrame, pd.DataFrame]:
    target_ablation = 'apprentice_only'
    parsed_winrates_df = create_algorithm_ablation_winrate_df(winrates_path)
    # Shortens names: A hack for better naming

    parsed_winrates_df['algorithm'] = parsed_winrates_df['algorithm'].apply(rename)
    parsed_winrates_df = parsed_winrates_df[parsed_winrates_df['ablation'] == target_ablation]

    parsed_winrates_df =  parsed_winrates_df.sort_values('algorithm')
    filtered_winrates_df = parsed_winrates_df.groupby(
        ['algorithm', 'run_id', 'ablation']
    ).apply(
        partial(get_first_value_above_threshold, threshold=winrate_threshold)
    ).dropna()

    # Renaming algorithm to show that these agents' opponent models
    # were trained with one hot encodings of agent actions
    if is_one_hot:
        filtered_winrates_df['algorithm'] += '-OH'
        parsed_winrates_df['algorithm'] += '-OH'
    return parsed_winrates_df, filtered_winrates_df


def retain_only_interquartile_range(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Following the Interquartile Mean, we throw away both
    the least and most performing 25% of runs
    '''
    # ASSUMPTION: There's a single ablation, for 'apprentice_only'
    df.reset_index(inplace=True)
    runs_to_remove_on_each_end = len(df.run_id.unique()) // 4
    runs_and_last_episode = {
        run_id: df[df['run_id'] == run_id].elapsed_episodes.max()
        for run_id in df.run_id.unique()
    }
    sorted_runs_and_last_episode = {
        run_id: max_episode
        for run_id, max_episode in
        sorted(runs_and_last_episode.items(), key=lambda item: item[1])
    }
    best_runs    = list(sorted_runs_and_last_episode.keys())[:runs_to_remove_on_each_end]
    worst_runs   = list(sorted_runs_and_last_episode.keys())[-runs_to_remove_on_each_end:]
    indices_to_drop = [
        index_to_drop
        for run_to_remove in chain(best_runs, worst_runs)
        for index_to_drop in list(df[df.run_id == run_to_remove].index)
    ]
    iq_df = df.drop(indices_to_drop)
    return iq_df


def plot_rliable_probability_of_improvement(df: pd.DataFrame):
    from rliable import library as rly
    from rliable.plot_utils import plot_probability_of_improvement
    from rliable import metrics

    #def extract_numpy_score_matrix(algorithm_name):
    #    return np.array([
    #        df[(df['test_agent'] == 'weak') & (df['algorithm'] == algorithm_name)].elapsed_episodes.to_numpy(),
    #        df[(df['test_agent'] == 'medium') & (df['algorithm'] == algorithm_name)].elapsed_episodes.to_numpy(),
    #        df[(df['test_agent'] == 'strong') & (df['algorithm'] == algorithm_name)].elapsed_episodes.to_numpy(),
    #    ]) * -1  # We multiply by -1 because we care about which algorithm takes _less_ time
    #             # Whereas rliable cares about maximizing values

    #scores = {
    #    alg: extract_numpy_score_matrix(alg)
    #    for alg in df.algorithm.unique()
    #}

    #pair_scores = {}
    #for i, a1 in enumerate(df.algorithm.unique()):
    #    for a2 in df.algorithm.unique()[i:]:
    #        if a1 == a2: continue
    #        else:
    #            pair_scores[f'{a1},{a2}'] = (scores[a1], scores[a2])

    ## Remember to multiply performances by -1, as higher is worse in our original case
    #probability_estimates, probability_confidence_intervals = rly.get_interval_estimates(
    #    pair_scores, metrics.probability_of_improvement, reps=10000
    #)
    #pickle.dump(probability_estimates, open('probability_estimates.pickle', 'wb'))
    #pickle.dump(probability_confidence_intervals, open('probability_confidence_intervals.pickle', 'wb'))

    probability_estimates = pickle.load(open('probability_estimates.pickle', 'rb'))
    probability_confidence_intervals = pickle.load(open('probability_confidence_intervals.pickle', 'rb'))
    fig, ax = plt.subplots(1, 1)
    ax = plot_probability_of_improvement(probability_estimates, probability_confidence_intervals, ax=ax)
    return fig


def get_first_value_above_threshold(d: pd.DataFrame, threshold: float):
    filtered_values = d.loc[d['winrate'] >= threshold]
    if len(filtered_values) > 0: return filtered_values.iloc[0]
    else: return None


def create_algorithm_ablation_winrate_df(path: str,
                                         use_inter_quartile_values: bool=True) -> pd.DataFrame:
    ''''
    Creates a dataframe with following structure (columns):
    'algorithm', 'run_id', **ablation

    experiment_id/
        run-id/
            agent_name/
                tensorboard_logs (many files)
                winrates/
                    apprentice_only.csv        (always present)
                    vanilla_exit.csv           (always present)
                    true_opponent_model.csv    (if applicable)
                    learnt_opponent_model.csv  (if applicable)

    Assumptions:
        - Each run-id features the same 'agent_name's
          (i.e all runs have run the same algorithms)
        - All 'agent_names' have the same number of winrates
        - There is a `run-0` directory from which initial data is parsed

    '''
    run_0_path = f'{path}/run-0/'
    algorithm_names = os.listdir(run_0_path)
    return pd.concat(
        [retain_only_interquartile_range(create_single_algorithm_dataframe(path, algorithm)) \
         if use_inter_quartile_values else \
         create_single_algorithm_dataframe(path, algorithm)
         for algorithm in algorithm_names]
    )


def create_single_algorithm_dataframe(path: str, algorithm: str) -> pd.DataFrame:
    algorithm_winrate_types = [csv_file.split('.')[0]  # Remove file extension
                               for csv_file in os.listdir(f'{path}/run-0/{algorithm}/winrates/')]
    winrate_data_frames = []
    # get all the winrates for one run
    for winrate_type in algorithm_winrate_types:
        run_dfs = [
            parse_single_algorithm_and_run_df_from_path(
                path, algorithm, run, winrate_type)
            for run in os.listdir(path) if 'run-' in run
        ]
        run_df = pd.concat(run_dfs)
        winrate_data_frames.append(run_df)
    algorithm_df = pd.concat(winrate_data_frames)
    algorithm_df['algorithm'] = algorithm
    return algorithm_df


def parse_single_algorithm_and_run_df_from_path(path: str,
                                                algorithm: str,
                                                run: str,
                                                winrate_type: str) -> pd.DataFrame:
    df = pd.read_csv(f'{path}/{run}/{algorithm}/winrates/{winrate_type}.csv')
    df['ablation'] = winrate_type
    df['run_id'] = run
    return df


def plot_winrate_matrix(ax, winrate_matrix: np.ndarray):
    sns.heatmap(winrate_matrix, annot=False, ax=ax, square=True,
                cmap=sns.color_palette('coolwarm', 50)[::-1],
                vmin=0.0, vmax=1.0, cbar=False,
                cbar_kws={'label': 'Head to head winrates'})
    ax.set_xlabel('Agent ID')
    ax.set_ylabel('Agent ID')
    ax.set_ylim(len(winrate_matrix) + 0.2, -0.2)  # Required seaborn hack
    ax.set_title('Empirical winrate matrix')
    return ax


def plot_winrate_matrix_and_support(winrate_matrix: np.ndarray, nash_support: np.ndarray):
    fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [15, 1]})
    plot_winrate_matrix(ax[0], winrate_matrix)
    plot_nash_support(ax[1], nash_support, column=True)

    plt.subplots_adjust(wspace=0, hspace=0)

    return fig


def plot_nash_support(ax, nash: np.ndarray,
                      column: bool = True, show_ticks: bool = True):
    '''
    param column: True/False Column or Row
    '''
    max_support = np.max(nash)
    nash_vector = np.reshape(nash, (nash.shape[0], 1) if column else (1, nash.shape[0]))
    sns.heatmap(nash_vector, ax=ax, annot=True,
                vmin=0, vmax=max_support, cbar=False,
                cmap=sns.color_palette('coolwarm', 50)[::-1])
    if show_ticks:
        ax.set_xticks([])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.yaxis.tick_left()
    ax.set_ylim(len(nash) + 0.2, -0.2)


def plot_nash_averaging_evolution_test_agents(evolution_nash_averaging_test_agents: np.ndarray):
    fig, ax = plt.subplots(1, 1)
    # Only show lower triangular

    ### Preprocessing to print only lower matrix
    mask = np.zeros_like(evolution_nash_averaging_test_agents, dtype=np.bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    ###

    axx = sns.heatmap(evolution_nash_averaging_test_agents, vmax=1.0, vmin=0.0, ax=ax,
                mask=mask,
                cmap=sns.color_palette('coolwarm', 50)[::-1],
                cbar_kws={'label': 'Support under Nash'},
                annot=False, annot_kws={'size': 10})

    # Workaround to prevent top and bottom of heatmaps to be cutoff
    # This is a known matplotlib bug
    ax.set_ylim(len(evolution_nash_averaging_test_agents) + 0.2, -0.2)

    ax.set_xlabel(xlabel='Agent ID', fontdict={'fontsize': 20})
    ax.set_ylabel(ylabel='benchmarking round', fontdict={'fontsize': 20})
    ax.set_title(label='Progression of Nash equilibrium during training', fontdict={'fontsize': 20})

    return fig


def plot_mcts_equivalent_strength(mcts_equivalent_strength: np.ndarray):
    test_agent_processed_timesteps = mcts_equivalent_strength[:, 0]
    test_agent_processed_timesteps = [int(x) for x in test_agent_processed_timesteps]
    mcts_equivalent_strengths = mcts_equivalent_strength[:, 1]

    fig, ax = plt.subplots(1, 1)
    ax = sns.lineplot(test_agent_processed_timesteps, mcts_equivalent_strengths,
                      dashes=True, marker='o', ax=ax)
    ax.set_xticks(test_agent_processed_timesteps)
    ax.set_xticklabels(test_agent_processed_timesteps, rotation=90, ha='center')
    ax.set_title('MCTS equivalent strength for Test agents')
    ax.set_xlabel('Test agent processed timesteps')
    ax.set_ylabel('MCTS budget')
    st.pyplot(fig)


def plot_winrate_matrix_mcts_and_test_agents(winrate_matrix: np.ndarray, nash_support: np.ndarray):
    fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [15, 1]})
    plot_winrate_matrix(ax[0], winrate_matrix)
    plot_nash_support(ax[1], nash_support, column=True)
    add_population_delimiting_lines(axes=ax, length=len(winrate_matrix))

    plt.subplots_adjust(wspace=0, hspace=0)

    return fig


def add_population_delimiting_lines(axes, length, number_populations=2):
    for i_delimiter in range(0, length,
                             int(length / number_populations)):
        axes[0].vlines(x=i_delimiter, ymin=0, ymax=length,
                     color='lime', lw=1.5)
        axes[0].hlines(y=i_delimiter, xmin=0, xmax=length,
                     color='lime', lw=1.5)
        axes[1].hlines(y=i_delimiter, xmin=0, xmax=length,
                     color='lime', lw=2)


def rename(name):
    shorter_name = name.split('expert_iteration_')[1].split('_800')[0]  # 800 is the MCTS budget in the name
    rename_dict = {
        'neural_net_opponent_modelling': 'ExIt-OMFS',
        'brexit_learnt_models': 'BRExIt-OMS',
        'brexit_true_models': 'BRExIt',
        'vanilla': 'ExIt'}
    return rename_dict[shorter_name]


if __name__ == '__main__':
    '''
    REMINDER:
    A double dash (`--`) is used to separate streamlit arguments
    from app arguments.
    '''
    DESCRIPTION=\
    ''' Generates some of the plots for BRExIt paper. If only :args path_1 is
    passed. it will be used for all other paths for debuggin purposes.
    '''

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--test_agents_results_path', required=False, help='Path to performance metrics about the test agents')
    parser.add_argument('--path_1', required=True, help='path to trained agents results (winrates) for test agent 1')
    parser.add_argument('--path_2', required=False, help='path to trained agents results (winrates) for test agent 2')
    parser.add_argument('--path_3', required=False, help='path to trained agents results (winrates) for test agent 3')
    parser.add_argument('--path_1_one_hot', required=False, help='path to trained agents (trained with one hot encoding actions) results (winrates) for test agent 1')
    parser.add_argument('--path_2_one_hot', required=False, help='path to trained agents (trained with one hot encoding actions) results (winrates) for test agent 2')
    parser.add_argument('--path_3_one_hot', required=False, help='path to trained agents (trained with one hot encoding actions) results (winrates) for test agent 3')
    args = parser.parse_args()

    # If only path 1 is presentm copy its value everywhere
    if not(args.path_2 or args.path_3 or args.path_1_one_hot or args.path_2_one_hot or args.path_3_one_hot):
        args.path_2, args.path_3, args.path_1_one_hot, args.path_2_one_hot, args.path_3_one_hot = [args.path_1] * 5
    # Add arg parse here. Make everything optional except for the first one
    # pass all of these over to `run`
    run(args.test_agents_results_path,
        args.path_1, args.path_2, args.path_3,
        args.path_1_one_hot, args.path_2_one_hot, args.path_3_one_hot)
