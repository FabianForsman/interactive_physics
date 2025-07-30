from src.environment import CartPoleEnv
from src.renderer import CartPoleRenderer
from agent.balance_controller import BalanceController

# Maximum steps per episode
max_steps = 2000

# Initialize the CartPole environment
env = CartPoleEnv()
# Initialize the renderer for visualization
renderer = CartPoleRenderer()
# Initialize the balance controller agent
agent = BalanceController()

print("Interactive CartPole Simulation")
print("Close the plot window to end the simulation.")

# Reset the environment at the start of each episode
state = env.reset()
total_reward = 0

# Step through the environment
for step in range(max_steps):
    # Check if plot window was closed
    if renderer.is_window_closed():
        print("Plot window closed. Ending simulation.")
        break

    # Agent selects an action based on the current state
    action = agent.act(state)

    # Environment returns next state, reward, and done flag
    next_state, done = env.step(action)

    # Move to the next state
    state = next_state

    # Render the current state
    renderer.render(state)

    # If the episode is done (pole fell or cart went too far), exit the loop
    if done:
        print(
            "Episode ended due to physics constraints (pole fell or cart went too far)."
        )
        break

# Print the final results
print(f"Episode finished after {step + 1} steps out of max {max_steps} steps.")

print("Simulation complete!")
