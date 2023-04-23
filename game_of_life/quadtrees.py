import numpy as np
from numba import njit, prange
import dask.array as da

class QuadTreeNode:
    def __init__(self, level, x, y, size, grid=None):
        self.level = level
        self.x = x
        self.y = y
        self.size = size
        self.grid = grid
        self.children = []

        if self.grid is not None and self.level > 0:
            self.split()

    def split(self):
        child_size = self.size // 2
        for i in range(2):
            for j in range(2):
                subgrid = self.grid[i * child_size:(i + 1) * child_size, j * child_size:(j + 1) * child_size]
                child_node = QuadTreeNode(self.level - 1, self.x + i * child_size, self.y + j * child_size,
                                          child_size, subgrid)
                self.children.append(child_node)
                self.grid = None

class QuadTree:
    def __init__(self, grid):
        self.grid = da.from_array(grid, chunks=(grid.shape[0] // 2, grid.shape[1] // 2))
        self.max_level = int(np.log2(min(self.grid.shape)))
        self.root = self.build_tree(self.max_level, 0, 0, self.grid.shape[0], self.grid)

    def build_tree(self, level, x, y, size, grid):
        return QuadTreeNode(level, x, y, size, grid)

    def apply_rules(self):
        self.root = self.apply_rules_recursive(self.root)

    def apply_rules_recursive(self, node):
        if node.level == 0:
            new_grid = apply_rules_to_grid(node.grid.compute())
            return QuadTreeNode(node.level, node.x, node.y, node.size, new_grid)
        else:
            new_children = []
            for child in node.children:
                new_child = self.apply_rules_recursive(child)
                new_children.append(new_child)
            return QuadTreeNode(node.level, node.x, node.y, node.size, children=new_children)

    def get_grid_data(self):
        return self.grid.shape, self.quadtree_to_grid(self.root).compute()

    def quadtree_to_grid(self, node):
        if node.level == 0:
            return node.grid
        else:
            child_grids = [self.quadtree_to_grid(child) for child in node.children]
            top = da.hstack([child_grids[0], child_grids[1]])
            bottom = da.hstack([child_grids[2], child_grids[3]])
            return da.vstack([top, bottom])

@njit
def apply_rules_to_grid(grid):
    new_grid = grid.copy()
    for x in prange(grid.shape[0]):
        for y in prange(grid.shape[1]):
            alive_neighbors = count_alive_neighbors(grid, x, y)
            if grid[x, y]:
                if alive_neighbors < 2 or alive_neighbors > 3:
                    new_grid[x, y] = False
            else:
                if alive_neighbors == 3:
                    new_grid[x, y] = True
    return new_grid

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
