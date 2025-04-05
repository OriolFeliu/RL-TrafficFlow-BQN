TRAINING = {
    'n_episodes': 200,
    'max_steps': 3600,
    'n_cars': 500,
    'batch_size': 64,
    'hidden_size': 64,
    'gamma': 0.8,
    'lr': 1e-3,
    'target_update': 10,
    'buffer_size': 2000,
    'epsilon_start': 1.0,
    'epsilon_end': 0.1,
    'epsilon_decay': 0.99
}

ENV = {
    'state_size': 4,
    'action_size': 2,
    'green_duration': 10,
    'yellow_duration': 3,
    'n_branches': 1,
    'map_name': 'data/network/grid_1inter.net.xml',
}