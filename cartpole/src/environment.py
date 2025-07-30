import numpy as np
from .physics import compute_next_state

# Angle at which to fail the episode (in radians)
THETA_THRESHOLD_RADIANS = 20 * 2 * np.pi / 360
# Position at which to fail the episode
X_THRESHOLD = 2.4

class CartPoleEnv:
    """
    Q-learning environment for CartPole.
    Maintains state and provides reset and step methods.
    """

    def __init__(self):
        # Initialize environment state
        self.state = None
        self.reset()

    def reset(self):
        """
        Reset environment to initial state.

        Returns:
            np.array: Initial state.
        """
        # Randomly initialize the cart position and pole angle
        # Cart position (x) is between -0.05 and 0.05,
        # Pole angle (theta) is between -0.05 and 0.05 radians
        # Other state variables (x_dot, theta_dot) are initialized to 0
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state

    def step(self, action):
        """
        Apply action, update state, and return result.

        Args:
            action (int): Action to take (0 or 1).

        Returns:
            tuple: (next_state, reward, done)
                next_state (np.array): State after action.
                reward (float): Reward for action.
                done (bool): Whether episode is finished.
        """
        self.state = compute_next_state(self.state, action)
        # Unpack the state for readability. x = cart position, theta = pole angle
        x, _, theta, _ = self.state

        done = (
            x < -X_THRESHOLD
            or x > X_THRESHOLD
            or theta < -THETA_THRESHOLD_RADIANS
            or theta > THETA_THRESHOLD_RADIANS
        )

        reward = 1.0 if not done else 0.0
        return self.state, reward, done
