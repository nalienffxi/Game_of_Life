import numpy as np
import dask.array as da
from grid import Grid
from renderer import Renderer
from numba import njit

def count_alive_neighbors(grid, x, y):
    count = da.from_array(grid, chunks=100).map_overlap(
        lambda g: np.sum(g[max(0, x-1):min(x+2, g.shape[0]), max(0, y-1):min(y+2, g.shape[1])]) - g[x, y],
        depth=(1, 1),
        boundary='none',
    ).compute()[x, y]

    return count

def new_game(grid, epochs, record=False):
    renderer = Renderer(grid)
    renderer.render(epochs, record)

    if record:
        writer = animation.FFMpegWriter(fps=self.fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
        ani.save("game_of_life.mp4", writer=writer)

        plt.show()