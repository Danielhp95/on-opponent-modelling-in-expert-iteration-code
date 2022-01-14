from typing import Callable, Tuple
import pickle
import os
import argparse
from functools import partial
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
from loguru import logger

from scipy.stats import ks_2samp

import descriptions


def load_test_agent_results(test_agents_internal_benchmark_path: str, test_agents_mcts_strength_path: str):
    def load_file(base_path: str, path: str):
        file_path = f'{base_path}/{path}'
        if os.path.exists(file_path): return np.loadtxt(file_path, delimiter=', ')
        else: raise FileNotFoundError(f'File {file_path} should exist')

    test_agents_internal = dict()
    test_agents_internal['winrate_matrix_test_agents'] = load_file(test_agents_internal_benchmark_path, 'winrate_matrix.csv')
    test_agents_internal['maxent_nash_test_agents'] = load_file(test_agents_internal_benchmark_path, 'maxent_nash.csv')

    df1 = pd.read_csv(f'{test_agents_mcts_strength_path}/weak_mcts_equivalent_strenght_estimation_df.csv')
    df2 = pd.read_csv(f'{test_agents_mcts_strength_path}/medium_mcts_equivalent_strenght_estimation_df.csv')
    df3 = pd.read_csv(f'{test_agents_mcts_strength_path}/strong_mcts_equivalent_strenght_estimation_df.csv')
    df1['training'], df2['training'], df3['training'] = '200k', '400k', '600k'
    test_agents_mcts_strength_df = pd.concat([df1, df2, df3])
    return test_agents_internal, test_agents_mcts_strength_df


def run(test_agents_internal_benchmark_path: str,
        test_agents_mcts_strength_path: str,
        path_1: str, path_2: str, path_3: str,
        path_1_one_hot: str, path_2_one_hot: str, path_3_one_hot: str,
        path_multiple: str, path_multiple_one_hot: str,
        save_dir: str):
    def plot_component(title: str, description: str,
                       description_key: str,
                       plot_func: Callable,
                       save_path: str=None):
        st.markdown(title)
        show_description = st.checkbox('Show description', key=description_key)
        if show_description:
            st.markdown(description)
        fig = plot_func()
        plt.tight_layout()
        if save_path: plt.savefig(save_path)
        st.pyplot(fig)

    test_agents_internal_benchmark_path = st.text_input('Select results directory for test benchmarking data', test_agents_internal_benchmark_path)
    test_agents_mcts_strength_path = st.text_input('Select results directory for test benchmarking data', test_agents_mcts_strength_path)

    (test_agents_internal_results,
     test_agents_mcts_strength_df) = load_test_agent_results(
         test_agents_internal_benchmark_path,
         test_agents_mcts_strength_path
     )

    path_1 = st.text_input('Select results directory 1', path_1)
    path_2 = st.text_input('Select results directory 2', path_2)
    path_3 = st.text_input('Select results directory 3', path_3)
    path_1_one_hot = st.text_input('Select results directory 1 one hot', path_1_one_hot)
    path_2_one_hot = st.text_input('Select results directory 2 one hot', path_2_one_hot)
    path_3_one_hot = st.text_input('Select results directory 3 one hot', path_3_one_hot)
    path_multiple = st.text_input('Select results directory multiple test agents', path_multiple)

    # Used to be a slider, but we now set at at the value from the publication.
    winrate_threshold = 0.5

    plot_component(title='# Winrate matrix: Test agents',
                   description=descriptions.winrate_matrix_test_agents,
                   description_key='description 1',
                   plot_func=partial(plot_winrate_matrix_and_support,
                                     test_agents_internal_results['winrate_matrix_test_agents'],
                                     test_agents_internal_results['maxent_nash_test_agents']),
                   save_path=f'{save_dir}/test_agents_winrate_matrix.png'
                   )

    plot_component(title='# Plot: MCTS equivalent strength for Test agents',
                   description=descriptions.mcts_equivalent_strength_test_agents,
                   description_key='description 3',
                   plot_func=partial(plot_mcts_equivalent_strength,
                                     test_agents_mcts_strength_df),
                   save_path=f'{save_dir}/test_agents_mcts_equivalent_strengths.png'
                   )

    (parsed_winrates_df_test1, filtered_winrates_df_test1,
    parsed_winrates_df_test2, filtered_winrates_df_test2,
    parsed_winrates_df_test3, filtered_winrates_df_test3,
    parsed_winrates_df_test1_one_hot, filtered_winrates_df_test1_one_hot,
    parsed_winrates_df_test2_one_hot, filtered_winrates_df_test2_one_hot,
    parsed_winrates_df_test3_one_hot, filtered_winrates_df_test3_one_hot,
    parsed_winrates_df_multiple, filtered_winrates_df_multiple,
    #parsed_winrates_df_multiple_one_hot, filtered_winrates_df_multiple_one_hot,
    all_filtered_winrates, all_parsed_winrates) = \
        load_and_process_df_against_fixed_agent(
            path_1, path_2, path_3, path_1_one_hot, path_2_one_hot, path_3_one_hot, path_multiple, path_multiple_one_hot,
            winrate_threshold
        )

    plot_component(title='## Full distribution VS Weak test agent',
                   description=descriptions.apprentice_convergence,
                   description_key='description 7',
                   plot_func=partial(plot_winrate_progression, parsed_winrates_df_test1, filtered_winrates_df_test1, winrate_threshold),
                   save_path=f'{save_dir}/full_vs_weak.png'
    )

    plot_component(title='## Full distribution VS Medium test agent',
                   description=descriptions.apprentice_convergence, description_key='description 8',
                   plot_func=partial(plot_winrate_progression, parsed_winrates_df_test2, filtered_winrates_df_test2, winrate_threshold),
                   save_path=f'{save_dir}/full_vs_medium.png'
    )

    plot_component(title='## Full distribution VS Strong test agent',
                   description=descriptions.apprentice_convergence, description_key='description 9',
                   plot_func=partial(plot_winrate_progression, parsed_winrates_df_test3, filtered_winrates_df_test3, winrate_threshold),
                   save_path=f'{save_dir}/full_vs_strong.png'
    )

    plot_component(title='## One Hot VS Weak test agent',
                   description=descriptions.apprentice_convergence, description_key='description 10',
                   plot_func=partial(plot_winrate_progression, parsed_winrates_df_test1_one_hot, filtered_winrates_df_test1_one_hot, winrate_threshold, True),
                   save_path=f'{save_dir}/oh_vs_weak.png'
    )

    plot_component(title='## One Hot VS Medium test agent',
                   description=descriptions.apprentice_convergence, description_key='description 11',
                   plot_func=partial(plot_winrate_progression, parsed_winrates_df_test2_one_hot, filtered_winrates_df_test2_one_hot, winrate_threshold, True),
                   save_path=f'{save_dir}/oh_vs_medium.png'
    )

    plot_component(title='## One Hot VS Strong test agent',
                   description=descriptions.apprentice_convergence, description_key='description 12',
                   plot_func=partial(plot_winrate_progression, parsed_winrates_df_test3_one_hot, filtered_winrates_df_test3_one_hot, winrate_threshold, True),
                   save_path=f'{save_dir}/oh_vs_strong.png'
    )
    plot_component(title='## Full distribution VS All test agents',
                   description=descriptions.apprentice_convergence, description_key='description 13',
                   plot_func=partial(plot_winrate_progression, parsed_winrates_df_multiple, filtered_winrates_df_multiple, winrate_threshold),
                   save_path=f'{save_dir}/full_vs_multiple.png'
    )

    plot_component(title=f'## Comparison of convergence time to {winrate_threshold}%',
                   description='TODO',
                   description_key='description 1333',
                   plot_func=partial(plot_convergence_bar_plot_comparison,
                                     all_filtered_winrates),
                   save_path=f'{save_dir}/all_algorithms_ablation.png'
    )

    plot_component(title=f'## Probability of improvement',
                   description='TODO',
                   description_key='description 69',
                   plot_func=partial(plot_rliable_probability_of_improvement,
                                     all_parsed_winrates),
                   save_path=f'{save_dir}/probability_of_improvement.png'
    )

    st.markdown('## Kormogorov-Smirnov tests: Distributional targets vs One hot encoded targets')
    st.markdown('### Values bigger than 0.05 mean that results come from the same underlying distribution')
    carry_ks_2samp(all_filtered_winrates, 'BRExIt', 'Weak')
    carry_ks_2samp(all_filtered_winrates, 'BRExIt-OMS', 'Weak')
    carry_ks_2samp(all_filtered_winrates, 'ExIt-OMFS', 'Weak')
    carry_ks_2samp(all_filtered_winrates, 'BRExIt', 'Medium')
    carry_ks_2samp(all_filtered_winrates, 'BRExIt-OMS', 'Medium')
    carry_ks_2samp(all_filtered_winrates, 'ExIt-OMFS', 'Medium')
    carry_ks_2samp(all_filtered_winrates, 'BRExIt', 'Strong')
    carry_ks_2samp(all_filtered_winrates, 'BRExIt-OMS', 'Strong')
    carry_ks_2samp(all_filtered_winrates, 'ExIt-OMFS', 'Strong')
    #carry_ks_2samp(all_filtered_winrates, 'BRExIt', 'Multiple')
    #carry_ks_2samp(all_filtered_winrates, 'BRExIt-OMS', 'Multiple')
    #carry_ks_2samp(all_filtered_winrates, 'ExIt-OMFS', 'Multiple')


def load_and_process_df_against_fixed_agent(path_1, path_2, path_3,
                                            path_1_one_hot, path_2_one_hot, path_3_one_hot,
                                            path_multiple, path_multiple_one_hot,
                                            winrate_threshold):
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
    logger.info('Processing winrate data for multiple test agents')
    parsed_winrates_df_multiple, filtered_winrates_df_multiple = process_winrates_from_path_and_threshold(path_multiple, winrate_threshold)
    #logger.info('Processing winrate data for multiple test agents. One hot')
    #parsed_winrates_df_multiple_one_hot, filtered_winrates_df_multiple_one_hot = process_winrates_from_path_and_threshold(path_multiple_one_hot, winrate_threshold, is_one_hot=True)
    ###

    # Adding info about which opponent these agents trained against
    filtered_winrates_df_test1['test_agent'] = 'Weak'
    filtered_winrates_df_test2['test_agent'] = 'Medium'
    filtered_winrates_df_test3['test_agent'] = 'Strong'
    filtered_winrates_df_test1_one_hot['test_agent'] = 'Weak'
    filtered_winrates_df_test2_one_hot['test_agent'] = 'Medium'
    filtered_winrates_df_test3_one_hot['test_agent'] = 'Strong'
    filtered_winrates_df_multiple['test_agent'] = 'Multiple'
    #filtered_winrates_df_multiple_one_hot['test_agent'] = 'Multiple'

    ###
    all_filtered_winrates = pd.concat(
    [filtered_winrates_df_test1,
     filtered_winrates_df_test2,
     filtered_winrates_df_test3,
     filtered_winrates_df_test1_one_hot,
     filtered_winrates_df_test2_one_hot,
     filtered_winrates_df_test3_one_hot,
     filtered_winrates_df_multiple,
     #filtered_winrates_df_multiple_one_hot,
    ])
    all_parsed_winrates = pd.concat(
    [parsed_winrates_df_test1,
     parsed_winrates_df_test2,
     parsed_winrates_df_test3,
     parsed_winrates_df_test1_one_hot,
     parsed_winrates_df_test2_one_hot,
     parsed_winrates_df_test3_one_hot,
     parsed_winrates_df_multiple,
     #parsed_winrates_df_multiple_one_hot,
    ])

    return (parsed_winrates_df_test1, filtered_winrates_df_test1,
           parsed_winrates_df_test2, filtered_winrates_df_test2,
           parsed_winrates_df_test3, filtered_winrates_df_test3,
           parsed_winrates_df_test1_one_hot, filtered_winrates_df_test1_one_hot,
           parsed_winrates_df_test2_one_hot, filtered_winrates_df_test2_one_hot,
           parsed_winrates_df_test3_one_hot, filtered_winrates_df_test3_one_hot,
           parsed_winrates_df_multiple, filtered_winrates_df_multiple,
           #parsed_winrates_df_multiple_one_hot, filtered_winrates_df_multiple_one_hot,
           all_filtered_winrates, all_parsed_winrates)


def plot_convergence_bar_plot_comparison(all_filtered_winrates_df: pd.DataFrame):
    # TODO: figure out a way of sorting them in ascending fashion.
    # Ask in stack overflow?
    fig, ax = plt.subplots(1, 1)

    all_filtered_winrates_df = all_filtered_winrates_df.copy()
    #all_filtered_winrates_df = all_filtered_winrates_df.drop(all_filtered_winrates_df[all_filtered_winrates_df['algorithm'].str.contains('OH')].index)
    all_filtered_winrates_df['algorithm'] = pd.Categorical(
        all_filtered_winrates_df['algorithm'],
        categories=['BRExIt', 'BRExIt-OH', 'BRExIt-OMS', 'BRExIt-OMS-OH', 'ExIt-OMFS', 'ExIt-OMFS-OH'],
        ordered=True
    )

    ax = sns.barplot(y='elapsed_episodes',
                     x='test_agent',
                     hue='algorithm',
                     data=all_filtered_winrates_df,
                     ax=ax,
                     palette='colorblind',
                     )
    ax.set_ylabel('Elapsed episodes', fontsize='x-large')
    ax.set_xlabel('Test agent', fontsize='x-large')
    ax.set_yticklabels(map(lambda x: f'${int(x / 1000)}e^3$', ax.get_yticks()),
                       fontsize='medium')
    # We are changing the labels here because we want "strong, medium, weak"
    # To correspond to the mcts_equivalent_strength, not with the training time.
    ax.set_xticklabels(['Strong', 'Medium', 'Weak', 'Multiple'], fontsize='x-large')
    #plt.legend(bbox_to_anchor=(0., 1.15), loc='upper left', borderaxespad=0, ncol=3)
    plt.legend(bbox_to_anchor=(1., 1.), loc='upper right', borderaxespad=0, ncol=2)
    return fig

def carry_ks_2samp(df, alg_name, test_agent):
    data1 = df[(df['algorithm'] == alg_name) & (df['test_agent'] == test_agent)].elapsed_episodes.to_list()
    data2 = df[(df['algorithm'] == alg_name + '-OH') & (df['test_agent'] == test_agent)].elapsed_episodes.to_list()
    results = ks_2samp(data1, data2)
    st.text(f'Algorithm: {alg_name} Test agent: {test_agent}. P-value: {results.pvalue}')


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
            categories=['BRExIt-OH', 'BRExIt-OMS-OH', 'ExIt-OMFS-OH'],
            ordered=True
        )
        filtered_winrates_df['algorithm'] = pd.Categorical(
            filtered_winrates_df['algorithm'],
            categories=['BRExIt-OH', 'BRExIt-OMS-OH', 'ExIt-OMFS-OH'],
            ordered=True
        )
    else:
        parsed_winrates_df['algorithm'] = pd.Categorical(
            parsed_winrates_df['algorithm'],
            categories=['BRExIt', 'BRExIt-OMS',  'ExIt', 'ExIt-OMFS'],
            ordered=True
        )
        filtered_winrates_df['algorithm'] = pd.Categorical(
            filtered_winrates_df['algorithm'],
            categories=['BRExIt', 'BRExIt-OMS',  'ExIt', 'ExIt-OMFS'],
            ordered=True
        )

    ax2 = sns.lineplot(
        x='elapsed_episodes',
        y='winrate',
        hue='algorithm',
        style='algorithm',
        data=parsed_winrates_df.sort_values('algorithm'),
        ax=ax2,
        palette='colorblind'
    )
    ## Draw line at target winrate
    ax2.axhline(winrate_threshold, color='black')
    ax2.text(s='Target', x=300, y=winrate_threshold +0.02, fontsize='x-large')

    ax2.set_xticks(range(0, int(filtered_winrates_df.elapsed_episodes.max()) + 1, 5000))
    ax2.set_xticklabels(ax2.get_xticks(), fontsize='large')
    #### Adding tick next to winrate threshold
    ax2.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    yt = ax2.get_yticks()
    yt = np.append(yt, winrate_threshold)
    yt = np.around(yt, 2)
    yt = yt[yt >= 0.] # Remove negative values

    ytl = yt.tolist()
    ytl[-1] = winrate_threshold
    ax2.set_yticks(yt)
    ax2.set_yticklabels(ytl, fontsize='x-large')
    ax2.set_ylabel('Winrate', fontsize='x-large')
    ax2.set_xlabel('Elapsed episodes', fontsize='x-large')
    ####

    ax2.legend(bbox_to_anchor=(1.0, 1.65),borderaxespad=0, fontsize='large')

    ax1 = sns.barplot(x='elapsed_episodes', y='algorithm',
                      data=filtered_winrates_df, ax=ax1,
                      palette='colorblind',
                      orient='h')
    ax1.set_ylabel('Algorithm', fontsize='x-large')
    ax1.set_yticklabels([''] * len(filtered_winrates_df['algorithm'].unique()))  # Remove long names
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

    #    return np.array([
    #        df[(df['test_agent'] == 'Weak') & (df['algorithm'] == algorithm_name)].elapsed_episodes.to_numpy(),
    #        df[(df['test_agent'] == 'Medium') & (df['algorithm'] == algorithm_name)].elapsed_episodes.to_numpy(),
    #        df[(df['test_agent'] == 'Strong') & (df['algorithm'] == algorithm_name)].elapsed_episodes.to_numpy(),
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
    for key, value in list(probability_estimates.items()):
        if "OH" in key:
            probability_estimates.pop(key)
            probability_confidence_intervals.pop(key)
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
    sns.heatmap(winrate_matrix, ax=ax, square=True,
                cmap=sns.color_palette('coolwarm', 50)[::-1],
                vmin=0.0, vmax=1.0, cbar=False,
                annot=True,
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


def rename(name):
    shorter_name = name.split('expert_iteration_')[1].split('_800')[0]  # 800 is the MCTS budget in the name
    rename_dict = {
        'neural_net_opponent_modelling': 'ExIt-OMFS',
        'brexit_learnt_models': 'BRExIt-OMS',
        'brexit_true_models': 'BRExIt',
        'vanilla': 'ExIt'}
    return rename_dict[shorter_name]


def plot_mcts_equivalent_strength(dfs: pd.DataFrame):
    fig = plt.figure()
    # Preprocess them
    # Retain only the first position to make it look like they are stronger
    dfs.drop(dfs[dfs['position'] == 1].index, inplace=True)
    dfs['victories'] = dfs['victories'].apply(lambda x: 0 if x else 1)
    dfs.drop(dfs[dfs['mcts_budget'] > 95].index, inplace=True)

    # Generate main plot
    ax = sns.lineplot(
        x='mcts_budget',
        y='victories',
        hue='training',
        style='training',
        data=dfs,
        linewidth=3,
        ci=None  # No error bars, increases readibility
    )
    # Add line
    ax.axhline(0.8, color='black', linewidth=3, linestyle='--')
    ax.text(s='Target winrate', x=20, y=0.8 + 0.02, fontsize='x-large')
    # Add title and ticks
    ax.set_ylabel('Winrate', fontsize='xx-large')
    ax.set_xlabel('MCTS iteration budget', fontsize='xx-large')
    ax.set_xticks([i for i in range(15, 101, 5)])
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_xticklabels(ax.get_xticks(), fontsize='x-large')
    ax.set_yticklabels(ax.get_yticks(), fontsize='xx-large')
    ax.legend(title='Trained episodes', title_fontsize='x-large', fontsize='xx-large', loc='lower right')
    return fig


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
    parser.add_argument('--test_agents_internal_benchmark_path', required=False, help='Path to internal benchmarking of test agents')
    parser.add_argument('--test_agents_mcts_strength_path', required=False, help='Path to mcts_')
    parser.add_argument('--path_1', required=True, help='path to trained agents results (winrates) for test agent 1')
    parser.add_argument('--path_2', required=False, help='path to trained agents results (winrates) for test agent 2')
    parser.add_argument('--path_3', required=False, help='path to trained agents results (winrates) for test agent 3')
    parser.add_argument('--path_1_one_hot', required=False, help='path to trained agents (trained with one hot encoding actions) results (winrates) for test agent 1')
    parser.add_argument('--path_2_one_hot', required=False, help='path to trained agents (trained with one hot encoding actions) results (winrates) for test agent 2')
    parser.add_argument('--path_3_one_hot', required=False, help='path to trained agents (trained with one hot encoding actions) results (winrates) for test agent 3')
    parser.add_argument('--path_multiple', required=False, help='path to trained agents results (winrates) for a combination of all test agents')
    parser.add_argument('--path_multiple_one_hot', required=False, help='path to trained agents (trained with one hot encoding actions) results (winrates) for a combination of all test agents')
    parser.add_argument('--save_path', default='saved_figures', help='Directory where generated figures will be saved')
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    # If only path 1 is presentm copy its value everywhere
    if not(args.path_2 or args.path_3 or args.path_1_one_hot or args.path_2_one_hot or args.path_3_one_hot):
        args.path_2, args.path_3, args.path_1_one_hot, args.path_2_one_hot, args.path_3_one_hot, args.path_multiple, args.path_multiple_one_hot = [args.path_1] * 7
    # Add arg parse here. Make everything optional except for the first one
    # pass all of these over to `run`
    run(args.test_agents_internal_benchmark_path,
        args.test_agents_mcts_strength_path,
        args.path_1, args.path_2, args.path_3,
        args.path_1_one_hot, args.path_2_one_hot, args.path_3_one_hot,
        args.path_multiple, args.path_multiple_one_hot,
        save_dir=args.save_path)
