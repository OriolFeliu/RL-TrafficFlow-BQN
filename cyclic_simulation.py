import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from config import ENV, TRAINING
from sumolib import checkBinary
from env import Environment

if __name__ == '__main__':
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)

    print('CYCLIC SIMULATION')

    # HYPERPARAMETHERS
    MAX_STEPS = TRAINING["max_steps"]
    N_CARS = TRAINING["n_cars"]
    GREEN_DURATION = ENV["green_duration"]
    YELLOW_DURATION = ENV["yellow_duration"]
    N_INTERSECTIONS = ENV['n_branches']
    MAP_NAME = ENV['map_name']

    n_simulations = 10

    sumoBinary = checkBinary('sumo')
    sumo_cmd = [
        sumoBinary,
        '-c', os.path.join('data', 'cfg', 'sumo_config.sumocfg'),
        '--no-step-log',
        '--waiting-time-memory', str(MAX_STEPS)
    ]

    env = Environment(sumo_cmd, MAX_STEPS, N_INTERSECTIONS,
                      N_CARS, GREEN_DURATION, YELLOW_DURATION, MAP_NAME)
    # env.reset()

    total_queue_lengths = []
    total_queue_times = []

    for simulation in range(n_simulations):
        env.reset()
        done = False

        while not done:
            # Get action and step environment
            next_state, _, done = env.step_cyclic_sim()

            # Update state and reward
            state = next_state

            if not done:
                total_queue_lengths.append(env.get_queue_length_reward(next_state))
                total_queue_times.append(env.get_queue_waiting_time_reward())

    # Logging
    avg_queue_length = np.mean(total_queue_lengths)
    avg_queue_time = np.mean(total_queue_times)
    print(f'Average queue length: {avg_queue_length}')
    print(f'Average queue time: {avg_queue_time}')
    print(f'Total steps: {env.current_step}')
    print(f'Total arrived vehicles: {env.total_arrived_vehicles}')

    fig, ax1 = plt.subplots()

    # Plot total_queue_lengths on the primary y-axis
    ax1.plot(total_queue_lengths, label='Queue Length', color='blue')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Queue Length', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a secondary y-axis for total_queue_times
    ax2 = ax1.twinx()
    ax2.plot(total_queue_times, label='Queue Time', color='red')
    ax2.set_ylabel('Queue Time', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Title and grid
    plt.title('Cyclic Simulation')
    fig.tight_layout()

    plt.show()
