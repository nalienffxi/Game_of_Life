import numpy as np
import dask.array as da
from numba import njit

class Grid:
    def __init__(self, size):
        self.size = size
        self.grid = da.zeros(size, dtype=bool, chunks=(size[0] // 10, size[1] // 10))

    def set_initial_configuration(self, configuration):
        for x, y in configuration:
            self.set_alive(x, y)

    def set_alive(self, x, y):
        self.grid = self.grid.map_blocks(lambda block: set_cell_alive(block, x, y))

    def set_dead(self, x, y):
        self.grid = self.grid.map_blocks(lambda block: set_cell_dead(block, x, y))

    def get_neighbors(self, x, y):
        return get_neighbors_numba(x, y, self.size)

    def count_alive_neighbors(self, x, y):
        return count_alive_neighbors_numba(x, y, self.grid, self.size)

def set_cell_alive(block, x, y):
    block[x % block.shape[0], y % block.shape[1]] = True
    return block

def set_cell_dead(block, x, y):
    block[x % block.shape[0], y % block.shape[1]] = False
    return block

@njit
def get_neighbors_numba(x, y, size):
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            new_x = x + i
            new_y = y + j
            if 0 <= new_x < size[0] and 0 <= new_y < size[1]:
                neighbors.append((new_x, new_y))
    return neighbors

@njit
def count_alive_neighbors(grid, x, y):
    count = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            nx, ny = x + i, y + j
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny]:
                count += 1
    return count
