import torch
import traci
import numpy as np
import random
import timeit
import os
import math
import subprocess

# TL phase codes
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_EW_GREEN = 2  # action 1 code 01
PHASE_EW_YELLOW = 3


class Environment:
    def __init__(self, sumo_cmd, max_steps, n_intersections, n_cars, green_duration, yellow_duration, map_name):
        self.n_cars = n_cars
        self.sumo_cmd = sumo_cmd
        self.max_steps = max_steps
        self.n_intersections = n_intersections

        self.done = False
        self.current_step = 0
        self.total_arrived_vehicles = 0

        self.steps_in_current_phase = 0
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.old_action = [0] * self.n_intersections
        self.current_phase_steps = [0] * self.n_intersections
        self.phases = ['green'] * self.n_intersections
        self.current_phase = np.full(self.n_intersections, 0)

        self.map_name = map_name
        self.first_epoch = True
        self.seed = 0

    def reset(self):
        self.done = False
        self.current_step = 0
        self.total_arrived_vehicles = 0
        self.old_action = [0] * self.n_intersections
        self.current_phase_steps = np.full(self.n_intersections, 0)

        self.generate_routefile()
        traci.start(self.sumo_cmd)

        if self.first_epoch:
            # Store incoming lane IDs the first epoch
            self.tl_ids = traci.trafficlight.getIDList()

            # Store incoming lane IDs the first epoch
            all_edges = traci.edge.getIDList()
            all_edges = [
                edge for edge in all_edges if not edge.startswith(':')]
            incoming_edges = []
            for tl_id in self.tl_ids:
                incoming_edges.append([edge for edge in all_edges
                                       if tl_id == edge[2:4]])

            self.incoming_edges = incoming_edges

            self.first_epoch = False

        state = self.get_queue_length_state()

        self.current_phase = None
        self.steps_in_current_phase = 0
        self.seed += 1

        return state

    def step(self, action):
        for branch, tl_id in enumerate(self.tl_ids):
            phase = self.phases[branch]
            steps = self.current_phase_steps[branch]
            action_branch = action[branch]
            old_action_branch = self.old_action[branch]

            if phase == 'green' and steps >= self.green_duration and self.old_action[branch] != action_branch:
                self.set_yellow_phase(old_action_branch, tl_id)
                self.phases[branch] = 'yellow'
                self.old_action[branch] = action_branch
                self.current_phase_steps[branch] = 0
            elif phase == 'yellow' and steps >= self.yellow_duration:
                self.set_green_phase(old_action_branch, tl_id)
                self.phases[branch] = 'green'
                self.current_phase_steps[branch] = 0

        traci.simulationStep()
        self.total_arrived_vehicles += traci.simulation.getArrivedNumber()
        self.current_phase_steps += 1
        self.current_step += 1

        next_state = self.get_queue_length_state()
        reward = self.get_queue_waiting_time_reward()

        self.done = (self.current_step >= self.max_steps) or (
            self.total_arrived_vehicles >= self.n_cars)

        if self.done:
            traci.close()

        return next_state, reward, self.done

    def get_queue_length_reward(self, next_state):
        queue_length = np.sum(next_state)

        return -queue_length

    def get_queue_waiting_time_reward(self):
        total_waiting_time = 0.0
        for veh_id in traci.vehicle.getIDList():
            # Check if the vehicle is halted
            if traci.vehicle.getSpeed(veh_id) < 0.01:
                total_waiting_time += traci.vehicle.getWaitingTime(veh_id)

        return -total_waiting_time

    def get_queue_length_waiting_time_reward(self, next_state):
        # can use exp to emphasize time
        reward = self.get_queue_length_reward(
            next_state) * self.get_queue_waiting_time_reward()

        return -reward

    def get_queue_length_state(self):
        n_edges = len(self.incoming_edges[0])

        halting_vehicles = np.zeros(
            (self.n_intersections * n_edges), dtype=np.int32)

        for intersection_idx in range(self.n_intersections):
            for i, edge_id in enumerate(self.incoming_edges[intersection_idx]):
                state_idx = intersection_idx * n_edges + i
                halting_vehicles[state_idx] = \
                    traci.edge.getLastStepHaltingNumber(edge_id)

        return halting_vehicles

    def set_green_phase(self, action_branch, tl_id):
        if action_branch == 0:
            traci.trafficlight.setPhase(tl_id, PHASE_NS_GREEN)
        elif action_branch == 1:
            traci.trafficlight.setPhase(tl_id, PHASE_EW_GREEN)

    def set_yellow_phase(self, old_action_branch, tl_id):
        if old_action_branch == 0:
            traci.trafficlight.setPhase(tl_id, PHASE_NS_YELLOW)
        elif old_action_branch == 1:
            traci.trafficlight.setPhase(tl_id, PHASE_EW_YELLOW)

    def run_simulation_steps(self, n_steps):
        while n_steps > 0:
            traci.simulationStep()
            self.total_arrived_vehicles += traci.simulation.getArrivedNumber()
            self.current_step += 1

            if (self.current_step >= self.max_steps) or (self.total_arrived_vehicles >= self.n_cars):
                return True

            n_steps -= 1

        return False

        # TODO calculate reward while waiting for green

    def generate_routefile(self):
        insertion_rate = self.n_cars / (self.max_steps / 3600)
        period = (self.max_steps * 0.8) / self.n_cars

        command = [
            'python', "C:/Program Files (x86)/Eclipse/Sumo/tools/randomTrips.py",
            '-n', self.map_name, '-o', 'data/route/randomTrips.rou.xml',
            '-b', '0', '-e', str(self.max_steps),
            '--seed', str(self.seed),
            '--validate',
            '--period', str(period),
            # '--insertion-rate', str(insertion_rate),  '--random-depart',
            # '--maxtries', str(self.n_cars),
            '--allow-fringe', '--fringe-factor', 'max',
            '--trip-attributes', 'departSpeed="10.0"'
        ]
        # python "C:\Program Files (x86)\Eclipse\Sumo\tools\randomTrips.py" -n "data/network/grid_4inter.net.xml"  -o data/route/randomTrips.rou.xml -e 5400 --seed 1234 --trip-attributes="departSpeed='10.0'"

        subprocess.run(command, capture_output=True, text=True)

    def step_cyclic_sim(self):
        traci.simulationStep()
        self.total_arrived_vehicles += traci.simulation.getArrivedNumber()
        self.current_step += 1

        next_state = self.get_queue_length_state()

        self.done = (self.current_step >= self.max_steps) or (
            self.total_arrived_vehicles >= self.n_cars)
        if self.done:
            traci.close()

        return next_state, None, self.done
