import numpy as np
import dask.array as da
from numba import njit

class Grid:
    def __init__(self, size, grid_data=None):
        self.size = size
        if grid_data is None:
            self.grid = da.random.randint(2, size=(size, size), chunks=size)
        else:
            if not isinstance(grid_data, da.Array):
                self.grid = da.from_array(grid_data, chunks=size)
            else:
                self.grid = grid_data

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

    def evolve(self):
        new_grid = self.grid.map_blocks(lambda block: evolve_block(block, self.size))
        self.grid = new_grid

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
def count_alive_neighbors(block, x, y):
    count = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            nx, ny = x + i, y + j
            if 0 <= nx < block.shape[0] and 0 <= ny < block.shape[1] and block[nx, ny]:
                count += 1
    return count

@njit
def evolve_block(block, size):
    new_block = block.copy()
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            alive_neighbors = count_alive_neighbors(block, i, j)
            if block[i, j]:
                new_block[i, j] = alive_neighbors in (2, 3)
            else:
                new_block[i, j] = alive_neighbors == 3
    return new_block

