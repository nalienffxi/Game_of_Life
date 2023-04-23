from grid import Grid
from quadtrees import QuadTree
from hasher import compute_hash
from renderer import render
from renderer import apply_rules

def new_game(size, epochs):
    grid = initialize_grid(size)

    quadtree = QuadTree(grid)
    quadtree.build_tree()

    for epoch in range(epochs):
        hashed_grid = compute_hash(quadtree)
        new_quadtree = apply_rules(hashed_grid)
        new_quadtree.prune()

        render(new_quadtree)

        quadtree = new_quadtree

def initialize_grid(size):
    # Implement the logic to initialize the grid with the desired resolution (size).
    # You can use the Grid class from the grid.py file.
    pass

if __name__ == '__main__':
    size = (50, 50)
    epochs = 100
    new_game(size, epochs)
