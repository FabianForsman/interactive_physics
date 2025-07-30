import matplotlib.pyplot as plt
import numpy as np


class CartPoleRenderer:
    def __init__(self):
        # Enable interactive mode for live updating plots
        plt.ion()
        # Create a new figure and axes for rendering
        self.fig, self.ax = plt.subplots()

        # Track if window is closed
        self.window_closed = False

        # Connect close event
        self.fig.canvas.mpl_connect("close_event", self.on_close)

    def on_close(self, event):
        """Handle window close event"""
        self.window_closed = True

    def is_window_closed(self):
        """Check if the plot window has been closed"""
        return self.window_closed

    def render(self, state):
        # Check if window is still open
        if self.window_closed:
            return

        # Clear the axes for the new frame
        self.ax.clear()
        # Unpack the state: x (cart position), _, theta (pole angle), _
        x, _, theta, _ = state
        cart_w, cart_h = 0.4, 0.2  # Cart width and height
        pole_len = 1.0  # Length of the pole

        # Set the limits for the axes
        self.ax.set_xlim(-2.4, 2.4)
        self.ax.set_ylim(-1, 1.5)

        # Draw the cart as a rectangle
        cart = plt.Rectangle((x - cart_w / 2, 0), cart_w, cart_h, color="black")
        self.ax.add_patch(cart)

        # Calculate the pole's end position
        pole_x = x + pole_len * np.sin(theta)
        pole_y = cart_h + pole_len * np.cos(theta)
        # Draw the pole as a line
        self.ax.plot([x, pole_x], [cart_h, pole_y], color="blue", linewidth=3)
        # Pause to update the plot
        plt.pause(0.001)
