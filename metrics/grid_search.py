import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def plot_scatter(dat):
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.25, bottom=.15, right=.99, top=.90)
    sns.boxplot(y=dat[0, :], x=dat[1, :], color='indianred', boxprops=dict(alpha=.7))
    sns.stripplot(y=dat[0, :], x=dat[1, :], jitter=0.08, size=2, color='grey')
    plt.scatter(x=[-0.05], y=[0.46], s=5, marker="^", color='blue')
    plt.ylim(0, 0.8)
    plt.xticks([], [])
    plt.ylabel('auPRC', fontsize=12)
    plt.xlabel('Models', fontsize=12)
    fig.set_size_inches(2, 3)
    plt.savefig('/Users/asheesh/Desktop/iTF_revision/gridsearch/auPRCs.pdf')


if __name__ == "__main__":
    dat = np.loadtxt('/Users/asheesh/Desktop/iTF_revision/gridsearch/auPRCs.txt')
    # Converting data to long-form.
    dat = np.vstack((dat, np.repeat(1, len(dat))))
    print(dat)
    plot_scatter(dat)
