import os
import sys

import descriptions
import streamlit as st


def data_directory_selection_sidebar_widget(path: str):
    selected_results_directory = st.sidebar.text_input('Select results directory',
                                                       path)
    display_selected_directory(selected_results_directory)
    return selected_results_directory


def display_selected_directory(selected_dir: str):
    if st.sidebar.checkbox(f'List files in directory {selected_dir} '):
        for f in os.listdir(selected_dir): st.sidebar.markdown(f)


def load_results(results_dir: str):
    return None


def run(path: str):
    results_dir = data_directory_selection_sidebar_widget(path)

    results = load_results(results_dir)

    # SIDEBAR
    st.sidebar.markdown('## Plots to show:')
    st.sidebar.markdown('Test agents')
    show_winrate_matrix_test_agents = st.sidebar.checkbox('Winrate matrix for test agents', True)
    show_nash_averaging_evolution_test_agents = st.sidebar.checkbox('Evolution Nash avg test agents', True)
    st.sidebar.markdown('MCTS evaluation test agents')
    show_winrate_matrix_mcts = st.sidebar.checkbox('Winrate matrix for MCTS agents', True)
    show_winrate_matrix_mcts_and_test_agents = st.sidebar.checkbox('Winrate matrix MCTS & Test agents', True)
    show_nash_averaging_mcts_and_test_agents = st.sidebar.checkbox('Nash averaging MCTS & Test agents', True)

    if show_winrate_matrix_test_agents:
        st.markdown('## Winrate matrix: Test agents')
        show_description = st.checkbox('Show description', key='description 1')
        if show_description:
            st.markdown(descriptions.winrate_matrix_test_agents)
        plot_winrate_matrix_test_agents()

    if show_nash_averaging_evolution_test_agents:
        st.markdown('## Evolution of Nash averaging: Test agents')
        show_description = st.checkbox('Show description', key='description 2')
        if show_description:
            st.markdown(descriptions.nash_averaging_evolutions_test_agents)
        plot_nash_averaging_evolution_test_agents()

    if show_winrate_matrix_mcts:
        st.markdown('## Winrate matrix: MCTS agents')
        show_description = st.checkbox('Show description', key='description 3')
        if show_description:
            st.markdown(descriptions.winrate_matrix_mcts_agents)
        plot_winrate_matric_mcts_agents()

    if show_winrate_matrix_mcts_and_test_agents:
        st.markdown('## Winrate matrix: MCTS and Test agents')
        show_description = st.checkbox('Show description', key='description 4')
        if show_description:
            st.markdown(descriptions.winrate_matrix_mcts_and_test_agents)
        plot_winrate_matrix_mcts_and_test_agents()

    if show_nash_averaging_mcts_and_test_agents:
        st.markdown('## Nash averaging: MCTS and Test agents')
        show_description = st.checkbox('Show description', key='description 5')
        if show_description:
            st.markdown(descriptions.nash_averaging_mcts_and_test_agents)
        plot_mcts_and_test_agents_nash_averaging()


def plot_winrate_matrix_test_agents():
    pass


def plot_nash_averaging_evolution_test_agents():
    pass


def plot_winrate_matric_mcts_agents():
    pass


def plot_winrate_matrix_mcts_and_test_agents():
    pass


def plot_mcts_and_test_agents_nash_averaging():
    pass


if __name__ == '__main__':
    if len(sys.argv) < 2: raise ValueError('Please specify a path to results')
    results_path = sys.argv[1]
    run(results_path)
