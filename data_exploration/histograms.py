import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(data, title="Histogram", x_axis_label="Value", y_axis_label="Frequency"):
    n, bins, patches = plt.hist(x=data, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.text(110, 60, "This is a generic label at x=110 y=60")
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    return plt


def demo_histogram():
    d = np.random.laplace(loc=100, scale=5, size=1000)
    plot_histogram(d)


if __name__ == "__main__":
    demo_histogram()
