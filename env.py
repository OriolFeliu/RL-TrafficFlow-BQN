import torch
import traci
import numpy as np
import random
import timeit
import os
import math
import subprocess

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Environment:
    def __init__(self, sumo_cmd, max_steps, n_intersections, n_cars, green_duration, yellow_duration, map_name):
        self.n_cars = n_cars
        self.sumo_cmd = sumo_cmd
        self.max_steps = max_steps
        self.n_intersections = n_intersections

        self.done = False
        self.current_step = 0
        self.total_arrived_vehicles = 0

        self.current_phase = None
        self.steps_in_current_phase = 0
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.old_action = [-1] * self.n_intersections

        self.map_name = map_name
        self.first_epoch = True

    def reset(self):
        self.done = False
        self.current_step = 0
        self.total_arrived_vehicles = 0
        self.old_action = [-1] * self.n_intersections

        self.generate_routefile()
        traci.start(self.sumo_cmd)

        # all_lanes = traci.lane.getIDList()
        # Filter lanes starting with 'TL' (outgoing from the junction)
        # ['E2TL_0', 'E2TL_1', 'E2TL_2', 'E2TL_3', 'N2TL_0', 'N2TL_1', 'N2TL_2', 'N2TL_3', 'S2TL_0', 'S2TL_1', 'S2TL_2', 'S2TL_3', 'W2TL_0', 'W2TL_1', 'W2TL_2', 'W2TL_3']
        # self.incoming_lanes = [lane for lane in all_lanes
        #                        if not lane.startswith(':TL') and not lane.startswith('TL')]

        # Store incoming lane IDs the first epoch
        if self.first_epoch:
            all_lanes = traci.lane.getIDList()
            all_lanes = [
                lane for lane in all_lanes if not lane.startswith(':')]
            incoming_lanes = []
            for tl_id in traci.trafficlight.getIDList():
                incoming_lanes.append([lane for lane in all_lanes
                                       if lane[2:4] == tl_id and '40.00' in lane])

            self.incoming_lanes = incoming_lanes
            self.first_epoch = False

        state = self.get_queue_length_state()

        self.current_phase = None
        self.steps_in_current_phase = 0

        return state

    def step(self, action):
        for branch in range(self.n_intersections):
            action_branch = action[branch]
            if self.current_step != 0 and self.old_action[branch] != action_branch:
                self.set_yellow_phase(branch)
                self.done = self.run_simulation_steps(self.yellow_duration)

            self.old_action[branch] = action_branch

            if not self.done:
                self.set_green_phase(action_branch)
                self.done = self.run_simulation_steps(self.green_duration)

        next_state = self.get_queue_length_state()
        reward = self.get_queue_length_reward()
        # reward = self.get_queue_waiting_time_reward()

        if self.done:
            traci.close()

        return next_state, reward, self.done

    # TODO use next state to calculate this without repeating code
    def get_queue_length_reward(self):
        # halt_N = traci.edge.getLastStepHaltingNumber('N2TL')
        # halt_S = traci.edge.getLastStepHaltingNumber('S2TL')
        # halt_E = traci.edge.getLastStepHaltingNumber('E2TL')
        # halt_W = traci.edge.getLastStepHaltingNumber('W2TL')

        # queue_length = halt_N + halt_S + halt_E + halt_W

        # queue_length = 0
        # for veh_id in traci.vehicle.getIDList():
        #     # Check if the vehicle is halted
        #     if traci.vehicle.getSpeed(veh_id) < 0.1:
        #         queue_length += 1

        queue_length = sum(traci.edge.getLastStepHaltingNumber(edge)
                           for edge in traci.edge.getIDList())

        return -queue_length

    def get_queue_waiting_time_reward(self):
        total_waiting_time = 0.0
        for veh_id in traci.vehicle.getIDList():
            # Check if the vehicle is halted
            if traci.vehicle.getSpeed(veh_id) < 0.1:
                total_waiting_time += traci.vehicle.getWaitingTime(veh_id)

        return -total_waiting_time

    def get_queue_length_waiting_time_reward(self):
        # can use exp to emphasize time
        reward = self.get_queue_length_reward() * self.get_queue_waiting_time_reward()

        return -reward

    def get_queue_length_state(self):
        halting_vehicles = np.zeros(
            (self.n_intersections, len(self.incoming_lanes[0])), dtype=np.int32)

        for intersection_idx in range(self.n_intersections):
            for i, lane_id in enumerate(self.incoming_lanes[intersection_idx]):
                halting_vehicles[intersection_idx, i] = \
                    traci.lane.getLastStepHaltingNumber(lane_id)

        return halting_vehicles

    def set_green_phase(self, action):
        if action == 0:
            traci.trafficlight.setPhase("B1", PHASE_NS_GREEN)
        elif action == 1:
            traci.trafficlight.setPhase("B1", PHASE_NSL_GREEN)
        elif action == 2:
            traci.trafficlight.setPhase("B1", PHASE_EW_GREEN)
        elif action == 3:
            traci.trafficlight.setPhase("B1", PHASE_EWL_GREEN)

    def set_yellow_phase(self, branch):
        if self.old_action[branch] == 0:
            traci.trafficlight.setPhase("B1", PHASE_NS_YELLOW)
        elif self.old_action[branch] == 1:
            traci.trafficlight.setPhase("B1", PHASE_NSL_YELLOW)
        elif self.old_action[branch] == 2:
            traci.trafficlight.setPhase("B1", PHASE_EW_YELLOW)
        elif self.old_action[branch] == 3:
            traci.trafficlight.setPhase("B1", PHASE_EWL_YELLOW)

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
        period = self.max_steps / self.n_cars

        # command = [
        #     'python', 'C:/Program Files (x86)/Eclipse/Sumo/tools/randomTrips.py',
        #     '-n', self.map_name,
        #     '-o', 'data/route/randomTrips.rou.xml',
        #     '-b', '0', '-e', str(self.max_steps), '-p', '1.0',
        #     '--binomial', '4', '--min-distance', '500',
        #     '--insertion-rate', str(random_depart),
        #     '--fringe-factor', '1.5', '--speed-exponent', '1.0', '-L',
        #     '--vehicle-class', 'passenger', '--trip-attributes', 'type=\"passenger\"',
        #     '--seed', '1234', '--validate', '--maxtries',  '200'
        # ]

        command = [
            'python', "C:/Program Files (x86)/Eclipse/Sumo/tools/randomTrips.py",
            '-n', self.map_name, '-o', 'data/route/randomTrips.rou.xml',
            '-b', '0', '-e', str(self.max_steps), '--seed', '1234', '--validate',
            # '--period', str(period),
            # '--insertion-rate', str(insertion_rate),  '--random-depart',
            # '--maxtries', str(self.n_cars),
            '--allow-fringe', '--fringe-factor', 'max',
        ]

        subprocess.run(command, capture_output=True, text=True)

    def step_cyclic_sim(self, action):
        traci.simulationStep()
        self.total_arrived_vehicles += traci.simulation.getArrivedNumber()
        self.current_step += 1

        next_state = self.get_queue_length_state()

        self.done = (self.current_step >= self.max_steps) or (
            self.total_arrived_vehicles >= self.n_cars)
        if self.done:
            traci.close()

        return next_state, None, self.done
