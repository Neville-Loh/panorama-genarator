from matplotlib import pyplot
import numpy as np
from scipy.spatial import distance
from histograms import plot_histogram

def corner_distances(points):
    distances = distance.pdist(points, 'euclidean')
    return distances

def corner_distance_demo():
    fig1, axs1 = pyplot.subplots(2, sharey=False, tight_layout=True)
    points = np.random.randint(100, size=(2, 100))
    axs1[0].scatter(*points)
    axs1[0].set_ylim(0, 100)
    y = corner_distances(np.array(points).transpose())
    axs1[1] = plot_histogram(y, title="distribution of distances between points")
    pyplot.show()

if __name__ == "__main__":
    corner_distance_demo()