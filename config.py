TRAINING = {
    'n_episodes': 200,
    'max_steps': 5400,
    'batch_size': 64,
    'gamma': 0.9,
    'lr': 1e-3,
    'target_update': 10,
    'buffer_size': 2000,
    'n_cars': 100,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.99
}

ENV = {
    'state_size': 4,
    'action_size': 2,
    'green_duration': 10,
    'yellow_duration': 3,
    'n_branches': 9,
    'map_name': 'data/network/grid_9inter.net.xml',
}