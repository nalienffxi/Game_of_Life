import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Renderer:

    def __init__(self, grid, cell_size=1):
        self.grid = grid
        self.cell_size = cell_size
        width, height = self.grid.size
        self.fig, self.ax = plt.subplots(figsize=(width * self.cell_size / 80, height * self.cell_size / 80))
        self.image = self.ax.imshow(self.grid.grid.compute(), cmap='gray_r', interpolation='nearest', aspect='auto')
        plt.axis('off')

    def update(self, *args):
        self.grid.evolve()
        self.image.set_array(self.grid.grid.compute())
        return self.image,

    def render(self, epochs, record=False):
        ani = animation.FuncAnimation(self.fig, self.update, epochs, blit=True)
        if record:
            try:
                writer = animation.writers['ffmpeg'](fps=10, metadata=dict(artist='Conway\'s Game of Life'))
                ani.save("game_of_life.mp4", writer=writer)
            except RuntimeError:
                print('Error: ffmpeg is not installed. Please install it to record the video.')
        return ani
