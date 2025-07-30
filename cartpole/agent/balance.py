import numpy as np


class SimpleBalanceController:
    """
    An improved balance controller that uses a dead zone and considers both
    pole angle and angular velocity for better balancing behavior.
    """

    def __init__(self, dead_zone=0.05, moderate_zone=0.15):
        """
        Initialize the controller with configurable zones.

        Args:
            dead_zone (float): Angle threshold below which no action is taken
            moderate_zone (float): Angle threshold for moderate response
        """
        self.dead_zone = dead_zone  # ~2.9 degrees
        self.moderate_zone = moderate_zone  # ~8.6 degrees
        self.action_history = []  # Track recent actions for smoother control
        self.max_history = 5

    def act(self, state):
        """
        Select action based on pole angle and angular velocity with zones.

        Args:
            state (np.array): Current state [x, x_dot, theta, theta_dot]
                - x: cart position
                - x_dot: cart velocity
                - theta: pole angle (positive = tilting right)
                - theta_dot: pole angular velocity

        Returns:
            int: Action (0 = move left, 1 = move right, 2 = stay still)
                 Note: Since physics only supports 0/1, we'll map 2 to alternating or no action
        """
        # Extract state variables
        x, x_dot, theta, theta_dot = state

        # Calculate absolute angle for zone detection
        abs_theta = abs(theta)

        # Dead zone: pole is nearly upright, don't move
        if abs_theta < self.dead_zone:
            action = self._get_stabilizing_action(theta, theta_dot)

        # Moderate zone: pole tilting but not critical, gentle response
        elif abs_theta < self.moderate_zone:
            # Consider both angle and angular velocity
            if theta > 0:  # Tilting right
                if theta_dot > 0:  # And falling right
                    action = 1  # Move right to catch it
                else:  # But moving back left
                    action = self._get_gentle_action(theta, theta_dot)
            else:  # Tilting left
                if theta_dot < 0:  # And falling left
                    action = 0  # Move left to catch it
                else:  # But moving back right
                    action = self._get_gentle_action(theta, theta_dot)

        # Critical zone: pole tilting significantly, strong response needed
        else:
            # Strong corrective action
            if theta > 0:
                action = 1  # Move right strongly
            else:
                action = 0  # Move left strongly

        # Track action history for smoother control
        self.action_history.append(action)
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)

        return action

    def _get_stabilizing_action(self, theta, theta_dot):
        """
        Get a stabilizing action when in the dead zone.
        Only act if there's significant angular velocity.
        """
        if abs(theta_dot) > 0.1:  # Significant angular velocity
            if theta_dot > 0:  # Moving right
                return 1  # Move right to stabilize
            else:  # Moving left
                return 0  # Move left to stabilize
        else:
            # Very stable, alternate or choose based on recent history
            return self._get_gentle_action(theta, theta_dot)

    def _get_gentle_action(self, theta, theta_dot):
        """
        Get a gentle action that considers recent action history.
        This provides "gradual" movement by sometimes not acting.
        """
        # Calculate desired action based on physics
        if theta > 0:
            desired_action = 1
        else:
            desired_action = 0

        # Sometimes don't act to create gentler response
        # This simulates "gradual" movement in a discrete action space
        if len(self.action_history) >= 2:
            recent_actions = self.action_history[-2:]
            # If we've been acting in the same direction, occasionally pause
            if all(a == desired_action for a in recent_actions):
                # 30% chance to use opposite action for gentler control
                if np.random.random() < 0.3:
                    return 1 - desired_action

        return desired_action

    def update(self, state, action, reward, next_state, done):
        """
        This controller doesn't learn, so update does nothing.
        Keeping this method for compatibility with the existing code structure.
        """
        pass
