from game import new_game
from grid import Grid

def main():
    # Define the grid size and number of epochs
    grid_size = (1920, 1080)  # Adjust the size as desired
    epochs = 1000

    # Create a new Grid instance
    grid = Grid(grid_size)

    # Add initial live cells to the grid
    # Here's an example of how to create a glider:
    grid.grid[1, 0] = True
    grid.grid[2, 1] = True
    grid.grid[0, 2] = True
    grid.grid[1, 2] = True
    grid.grid[2, 2] = True

    # Run and render the game
    new_game(grid, epochs)

if __name__ == "__main__":
    main()
