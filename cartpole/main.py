from src.environment import CartPoleEnv
from src.renderer import CartPoleRenderer
from cartpole.agent.balance import SimpleBalanceController

# Number of episodes to run the simulation
n_episodes = 1
# Maximum steps per episode
max_steps = 1000

# Initialize the CartPole environment
env = CartPoleEnv()
# Initialize the renderer for visualization
renderer = CartPoleRenderer()
# Initialize the simple balance controller
agent = SimpleBalanceController()

# Simulation loop over episodes
for ep in range(n_episodes):
    # Reset the environment at the start of each episode
    state = env.reset()
    total_reward = 0
    # Step through the environment
    for step in range(max_steps):
        # Controller selects an action based on the current pole angle
        action = agent.act(state)
        # Environment returns next state, reward, and done flag
        next_state, reward, done = env.step(action)
        # Move to the next state
        state = next_state
        # Accumulate the reward
        total_reward += reward
        # Render the current state
        renderer.render(state)
        # If the episode is done, exit the loop
        if done:
            break
    # Print the total reward for the episode
    print(f"Episode {ep + 1}, Total Reward: {total_reward}, Steps: {step + 1}")
