import numpy as np
import dask.array as da
from numba import jit
from grid import Grid

@jit(nopython=True)
def count_alive_neighbors(grid_data, x, y):
    alive_neighbors = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            alive_neighbors += grid_data[(x + i) % grid_data.shape[0], (y + j) % grid_data.shape[1]]
    return alive_neighbors

@jit(nopython=True)
def _apply_rules_numpy(grid_data):
    new_grid_data = np.copy(grid_data)
    for x in range(grid_data.shape[0]):
        for y in range(grid_data.shape[1]):
            alive_neighbors = count_alive_neighbors(grid_data, x, y)
            if grid_data[x, y] == 1 and (alive_neighbors < 2 or alive_neighbors > 3):
                new_grid_data[x, y] = 0
            elif grid_data[x, y] == 0 and alive_neighbors == 3:
                new_grid_data[x, y] = 1
    return new_grid_data

def apply_rules(grid: Grid) -> Grid:
    grid_data = grid.grid.compute()
    new_grid_data = _apply_rules_numpy(grid_data)
    dask_grid = da.from_array(new_grid_data, chunks=(100, 100))
    return Grid(grid.size, dask_grid)
