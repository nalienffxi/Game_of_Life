import matplotlib.pyplot as plt
import matplotlib.animation as animation
from grid import Grid
from quadtrees import QuadTree
from numba import njit, prange


def render(quadtree, epochs, record=False):
    grid = quadtree_to_grid(quadtree)
    if record:
        record_game(grid, epochs)
    else:
        display_game(grid, epochs)

def quadtree_to_grid(quadtree):
    size, grid_data = quadtree.get_grid_data()
    grid = Grid(size)
    grid.grid = grid_data
    return grid

def display_game(grid, epochs):
    fig, ax = plt.subplots()
    im = ax.imshow(grid.grid, cmap='Greys', interpolation='nearest')
    ani = animation.FuncAnimation(fig, update_display, epochs, fargs=(grid, im),
                                  interval=200, blit=False)
    plt.show()

def update_display(frame, grid, im):
    grid = apply_rules(grid)  # Make sure you have a function to apply the game of life rules
    im.set_array(grid.grid)
    return im,

def record_game(grid, epochs):
    fig, ax = plt.subplots()
    im = ax.imshow(grid.grid, cmap='Greys', interpolation='nearest')
    ani = animation.FuncAnimation(fig, update_display, epochs, fargs=(grid, im),
                                  interval=200, blit=False)
    ani.save('game_of_life.mp4', writer='ffmpeg', dpi=100)
    plt.close()

@njit(parallel=True)
def apply_rules(grid):
    new_grid = grid.grid.copy()
    for x in prange(grid.size[0]):
        for y in prange(grid.size[1]):
            alive_neighbors = count_alive_neighbors(grid.grid, x, y)
            if grid.grid[x, y]:
                if alive_neighbors < 2 or alive_neighbors > 3:
                    new_grid[x, y] = False
            else:
                if alive_neighbors == 3:
                    new_grid[x, y] = True
    grid.grid = new_grid
    return grid
