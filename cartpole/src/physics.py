import numpy as np

# Physical constants for the CartPole system
GRAVITY = 9.8                # Acceleration due to gravity (m/s^2)
MASSCART = 1.0               # Mass of the cart (kg)
MASSPEND = 0.1               # Mass of the pole (kg)
TOTAL_MASS = MASSCART + MASSPEND  # Total mass (cart + pole)
LENGTH = 0.5                 # Half the length of the pole (m)
POLEMASS_LENGTH = MASSPEND * LENGTH  # Mass of the pole times half its length
FORCE_MAG = 10.0             # Magnitude of the force applied to the cart
TAU = 0.02                   # Time interval for each step (s)

"""
Computes the next state of the CartPole environment based on the current state and action taken.

Args:
    state (np.array): The current state of the environment [x, x_dot, theta, theta_dot].
    action (int): The action taken by the agent (0 for left, 1 for right).

Returns:
    np.array: The next state of the environment after applying the action.
"""
def compute_next_state(state, action):
    # Unpack the state vector
    x, x_dot, theta, theta_dot = state

    # Determine the direction of the force based on the action
    force = FORCE_MAG if action == 1 else -FORCE_MAG

    # Precompute trigonometric functions for efficiency
    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    # Compute a temporary variable for acceleration calculations
    temp = (force + POLEMASS_LENGTH * theta_dot ** 2 * sintheta) / TOTAL_MASS # Formula: a = (F + m * l * θ_dot^2 * sin(θ)) / M

    # Calculate angular acceleration of the pole
    theta_acc = (GRAVITY * sintheta - costheta * temp) / \
                (LENGTH * (4.0 / 3.0 - MASSPEND * costheta ** 2 / TOTAL_MASS)) # Formula: θ_acc = (g * sin(θ) - cos(θ) * a) / (l * (4/3 - m * cos^2(θ) / M))

    # Calculate linear acceleration of the cart
    x_acc = temp - POLEMASS_LENGTH * theta_acc * costheta / TOTAL_MASS

    # Update the state using Euler's method
    x += TAU * x_dot # Update cart position
    x_dot += TAU * x_acc # Update cart velocity
    theta += TAU * theta_dot # Update pole angle
    theta_dot += TAU * theta_acc # Update pole angular velocity

    # Return the next state as a NumPy array
    return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
