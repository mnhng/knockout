#!/usr/bin/env python3
import argparse
import pathlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset
LABELS = {'Knockout': 'Knock', 'Dropout': 'Drop', 'ZeroImpute': 'ZI', 'missForest': 'mFr', 'supMIWAE': 'sMIWAE'}
COLORS = {'CB': '#1f77b4', r'CB$^*$': 'navy',
          'mFr': '#ff7f0e', 'MICE': 'orange',
          'Knock': '#2ca02c',
          'MIWAE': 'cyan', 'sMIWAE': 'gray',
          'ZI': 'pink', 'Drop': 'magenta'}

# sns.set_theme(style="ticks")
sns.set(font='DejaVu Sans')
sns.set_style('whitegrid')

def draw_axe(sf, ax):
    # Plot the orbital period with horizontal boxes
    sns.lineplot(sf, x='ko', y='MSE', hue='method', ax=ax, marker='o',
                 palette=COLORS,
                 )

    # Tweak the visual presentation
    ax.set_yscale('log')
    ax.yaxis.grid(True)
    ax.set_xticks([0, 1, 2, 3])
    ax.set(xlabel='')
    ax.set(ylabel='')
    ax.get_legend().remove()
    # ax.tick_params(axis='x', labelrotation=-15)


def main(args):
    TARGET = 'Bayes Optimal' if args.target == 'BO' else 'Observations'

    fig, axes = plt.subplots(1, 3, figsize=(9, 2), dpi=300)

    frame = pd.concat([pd.read_csv(fp) for fp in pathlib.Path(f'sim/comp').glob('*/stats_*.csv')])
    frame['method'] = frame['method'].apply(lambda x: LABELS.get(x, x))
    frame['target'] = frame['type'].map({'error': 'Observations', 'bayes': 'Bayes Optimal'})

    draw_axe(frame.loc[frame['target'] == TARGET, ['ko', 'method', 'MSE']], axes[0])
    axes[0].set_title('a) Complete')


    frame = pd.concat([pd.read_csv(fp) for fp in pathlib.Path(f'sim/MCAR').glob('*/stats_*.csv')])
    frame.loc[frame['method'] == 'CB', 'method'] = r'CB$^*$'
    frame['method'] = frame['method'].apply(lambda x: LABELS.get(x, x))
    frame['target'] = frame['type'].map({'error': 'Observations', 'bayes': 'Bayes Optimal'})

    draw_axe(frame.loc[(frame['target'] == TARGET), ['ko', 'method', 'MSE']], axes[1])
    axes[1].set_title('b) MCAR')


    frame = pd.concat([pd.read_csv(fp) for fp in pathlib.Path(f'sim/MNAR').glob('*/stats_*.csv')])
    frame.loc[frame['method'] == 'CB', 'method'] = r'CB$^*$'
    frame['method'] = frame['method'].apply(lambda x: LABELS.get(x, x))
    frame['target'] = frame['type'].map({'error': 'Observations', 'bayes': 'Bayes Optimal'})

    draw_axe(frame.loc[(frame['target'] == TARGET), ['ko', 'method', 'MSE']], axes[2])
    axes[2].set_title('c) MNAR (self-censored)')

    fig.tight_layout(pad=2.)
    part1, part2 = axes[0].get_legend_handles_labels()
    for art, lbl in zip(*axes[1].get_legend_handles_labels()):
        if lbl not in part2:
            part1.append(art)
            part2.append(lbl)

    ncol = len(part1)
    fig.legend(part1, part2,
        ncol=ncol, loc='lower center',
        borderpad=.1, labelspacing=.0, columnspacing=0.4,
        handlelength=1., handletextpad=.2, frameon=False,
    )

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', '-t', choices={'BO', 'GT'}, required=True)

    main(parser.parse_args())
