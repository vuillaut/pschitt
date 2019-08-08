from pschitt.vizualisation import plot_array
from pschitt import geometry as geo
import matplotlib.pyplot as plt


def test_plot_array():
    array = [geo.Telescope([0, i, 0], [0, 0, 1]) for i in range(4)]

    fig, ax = plt.subplots()
    ax = plot_array(array, ax=ax, c=[0, 0, 1, 2], marker='+', alpha=0.5, s=20)