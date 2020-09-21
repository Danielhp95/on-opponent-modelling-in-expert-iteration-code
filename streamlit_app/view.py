from typing import Callable
import os
import sys
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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
    results['evolution_nash_averaging_test_agents'] = load_file(results_dir, 'evolution_nash_averaging_test_agents.csv')
    results['winrate_matrix_mcts'] = load_file(results_dir, 'winrate_matrix_mcts.csv')
    results['maxent_nash_test_agents'] = load_file(results_dir, 'maxent_nash_test_agents.csv')
    results['mcts_equivalent_strength_test_agents'] = load_file(results_dir, 'mcts_equivalent_strength_test_agents.csv')
    return results


def run(path: str):
    # TODO: if we ever wanna refactor the things above into this
    def plot_component(checkbox_ticked: bool, title: str, description: str,
                       description_key: str, plot_func: Callable):
        if checkbox_ticked:
            st.markdown(title)
            show_description = st.checkbox('Show description', key=description_key)
            if show_description:
                st.markdown(description)
            plot_func()

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

    plot_component(show_winrate_matrix_test_agents,
                   title='# Winrate matrix: Test agents',
                   description=descriptions.winrate_matrix_test_agents,
                   description_key='description 1',
                   plot_func=partial(plot_winrate_matrix_and_support,
                                     results['winrate_matrix_test_agents'],
                                     results['maxent_nash_test_agents']))

    plot_component(show_nash_averaging_evolution_test_agents,
                   title='# Evolution of Nash averaging: Test agents',
                   description=descriptions.nash_averaging_evolutions_test_agents,
                   description_key='description 2',
                   plot_func=partial(plot_nash_averaging_evolution_test_agents,
                                     results['evolution_nash_averaging_test_agents']))

    plot_component(show_plot_mcts_equivalent_test_agents,
                   title='# Plot: MCTS equivalent strength for Test agents',
                   description=descriptions.mcts_equivalent_strength_test_agents,
                   description_key='description 3',
                   plot_func=partial(plot_mcts_equivalent_strength,
                                     results['mcts_equivalent_strength_test_agents']))

    plot_component(show_winrate_matrix_mcts,
                   title='# Winrate matrix: MCTS agents',
                   description=descriptions.winrate_matrix_mcts_agents,
                   description_key='description 4',
                   plot_func=partial(plot_winrate_matrix_and_support,
                                     results['winrate_matrix_test_agents'],
                                     results['maxent_nash_test_agents']))

    plot_component(show_winrate_matrix_mcts,
                   title='# Winrate matrix: MCTS & Test agents',
                   description=descriptions.winrate_matrix_mcts_and_test_agents,
                   description_key='description 5',
                   plot_func=partial(plot_winrate_matrix_mcts_and_test_agents,
                                     results['winrate_matrix_test_agents'],
                                     results['maxent_nash_test_agents']))


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

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    st.pyplot()


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

    st.pyplot()


def plot_mcts_equivalent_strength(mcts_equivalent_strength: np.ndarray):
    test_agent_processed_timesteps = mcts_equivalent_strength[:, 0]
    test_agent_processed_timesteps = [int(x) for x in test_agent_processed_timesteps]
    mcts_equivalent_strengths = mcts_equivalent_strength[:, 1]

    ax = sns.lineplot(test_agent_processed_timesteps, mcts_equivalent_strengths,
                      dashes=True, marker='o')
    ax.set_xticks(test_agent_processed_timesteps)
    ax.set_xticklabels(test_agent_processed_timesteps, rotation=90, ha='center')
    ax.set_title('MCTS equivalent strength for Test agents')
    ax.set_xlabel('Test agent processed timesteps')
    ax.set_ylabel('MCTS budget')
    st.pyplot()


def plot_winrate_matrix_mcts_and_test_agents(winrate_matrix: np.ndarray, nash_support: np.ndarray):
    fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [15, 1]})
    plot_winrate_matrix(ax[0], winrate_matrix)
    plot_nash_support(ax[1], nash_support, column=True)
    add_population_delimiting_lines(axes=ax, length=len(winrate_matrix))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    st.pyplot()


def add_population_delimiting_lines(axes, length, number_populations=2):
    for i_delimiter in range(0, length,
                             int(length / number_populations)):
        axes[0].vlines(x=i_delimiter, ymin=0, ymax=length,
                     color='lime', lw=1.5)
        axes[0].hlines(y=i_delimiter, xmin=0, xmax=length,
                     color='lime', lw=1.5)
        axes[1].hlines(y=i_delimiter, xmin=0, xmax=length,
                     color='lime', lw=2)


if __name__ == '__main__':
    if len(sys.argv) < 2: raise ValueError('Please specify a path to results')
    results_path = sys.argv[1]
    run(results_path)
