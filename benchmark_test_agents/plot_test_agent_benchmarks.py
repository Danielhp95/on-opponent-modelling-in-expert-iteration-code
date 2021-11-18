import argparse
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from regym.game_theory import compute_nash_averaging
from regym.plotting import plot_winrate_matrix

import seaborn as sns

'''
TODO: description
'''


def compute_progression_of_nash_averagings(winrate_matrix: np.ndarray):
    '''
    Creates a lower triangular matrix
    '''
    maxent_nashes = [compute_nash_averaging(
                        winrate_matrix[:i,:i],
                        perform_logodds_transformation=True)[0]
        for i in range(1, winrate_matrix.shape[0] + 1)]
    for max_ent in maxent_nashes:
        max_ent.resize(winrate_matrix.shape[0], refcheck=False)
    return np.stack(maxent_nashes)


def plot_progression_nash_equilibriums(progression_nash: np.ndarray, show_annotations=False):
    fig, ax = plt.subplots(1, 1)
    # Only show lower triangular

    ### Preprocessing to print only lower matrix
    mask = np.zeros_like(progression_nash, dtype=np.bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    ###

    axx = sns.heatmap(progression_nash, vmax=1.0, vmin=0.0, ax=ax,
                mask=mask,
                cmap=sns.color_palette('coolwarm', 50)[::-1],
                cbar_kws={'label': 'Support under Nash'},
                annot=show_annotations, annot_kws={'size': 10})

    # Workaround to prevent top and bottom of heatmaps to be cutoff
    # This is a known matplotlib bug
    ax.set_ylim(len(progression_nash) + 0.2, -0.2)

    ax.set_xlabel(xlabel='Agent ID', fontdict={'fontsize': 30})
    ax.set_ylabel(ylabel='benchmarking round', fontdict={'fontsize': 30})
    ax.set_title(label='Progression of Nash equilibrium during training', fontdict={'fontsize': 30})
    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plots winrate_matrix and Nash averaging evoluation of "On Opponent Modelling in Expert Iteration"')
    parser.add_argument('--winrate_matrix', required=True, help='Path to csv containing winrate matrix')
    args = parser.parse_args()

    winrate_matrix = np.loadtxt(args.winrate_matrix, delimiter=', ')

    nash_avgs = compute_progression_of_nash_averagings(winrate_matrix)
    #nash_avgs = pickle.load(open('nash_avgs.pickle', 'rb'))

    matplotlib.rcParams.update({'font.size': 15})

    ax = plot_progression_nash_equilibriums(nash_avgs)
    plt.show()
    ax = plot_winrate_matrix(winrate_matrix, show_annotations=False)
    plt.show()
