import numpy as np
import random
import gym

from gym import spaces
from collections import namedtuple

import numba

from numba import jit

"The initial state of Migration Environment including the initial position. distances between services and positions"

Point = namedtuple('Point', ['x', 'y'])
EnvironmentParameters = namedtuple('ServersParameter', ['trace_start_index', 'num_traces','num_base_station', 'optical_fiber_trans_rate',
                                                        'migration_coefficient_low', 'migration_coefficient_high', 'backhaul_coefficient',
                                                        'server_poisson_rate', 'server_task_data_lower_bound',
                                                       'server_frequency', 'server_task_data_higher_bound', 'ratio_lower_bound','ratio_higher_bound',
                                                       'client_poisson_rate', 'client_task_data_lower_bound', 'client_task_data_higher_bound',
                                                        'migration_size_low', 'migration_size_high',
                                                       'map_width', 'map_height', 'num_horizon_servers', 'num_vertical_servers',
                                                       'transmission_rates', 'traces_file_path', 'trace_length', 'trace_interval', 'is_full_observation',
                                                        'is_full_action'])

class MECServer(object):
    def __init__(self, index, poisson_rate, task_data_lower_bound, task_data_higher_bound,
                 ratio_lower_bound, ratio_higher_bound, frequence = 32.0):
        self.poisson_rate = poisson_rate
        self.index = index

        self.task_data_lower_bound = task_data_lower_bound
        self.task_data_higher_bound = task_data_higher_bound
        self.ratio_lower_bound = ratio_lower_bound
        self.ratio_higher_bound = ratio_higher_bound

        self.procession_task_num = 0
        self.task_load_range = 0
        self.frequence = frequence # 10 GHz

    def get_current_workload(self):
        num_arriving_tasks = np.random.poisson(self.poisson_rate)

        total_required_frequency = 0.0
        for i in range(num_arriving_tasks):
            task_data = np.random.uniform(self.task_data_lower_bound,
                                          self.task_data_higher_bound)

            task_comp_to_volume_ratio = np.random.uniform(self.ratio_lower_bound,
                                                          self.ratio_higher_bound)

            total_required_frequency += task_data * task_comp_to_volume_ratio

        total_required_frequency = total_required_frequency / (1024.0 * 1024.0 * 1024.0) # Hz --> GHz
        return total_required_frequency

    def get_estimated_running_time(self, client_workload, server_workload):
        # unit GHz
        estimated_time = (client_workload + server_workload)  / self.frequence

        return estimated_time

class BatchMigrationEnv(gym.Env):
    def __init__(self, env_parameters):
        self.is_full_action = env_parameters.is_full_action
        self.is_batch = True
        if self.is_full_action:
            self._action_spec = spaces.Discrete(env_parameters.num_base_station)
            self._action_dim = env_parameters.num_base_station
        else:
            self._action_spec = spaces.Discrete(6)
            self._action_dim = 6

        self.migration_size_low = env_parameters.migration_size_low
        self.migration_size_high = env_parameters.migration_size_high
        self.migration_coefficient_low = env_parameters.migration_coefficient_low
        self.migration_coefficient_high = env_parameters.migration_coefficient_high


        self.backhaul_coefficient = env_parameters.backhaul_coefficient
        self._num_traces = env_parameters.num_traces
        self._optical_fiber_trans_rate = env_parameters.optical_fiber_trans_rate  # Mbps
        self.is_full_observation = env_parameters.is_full_observation
        #self._state_dim = 5 + 2 * env_parameters.num_base_station
        if self.is_full_observation:
            self._state_dim = 2 * env_parameters.num_base_station + 2
        else:
            self._state_dim = 4

        self.server_poisson_rate = env_parameters.server_poisson_rate
        # The state include (distance, num_of_hops, workload_distribution, service_index, data_process)
        # this is a gym environment with Box as the observation type
        low_state = np.array( [float("-inf")]*self._state_dim)
        high_state = np.array( [float("inf")] * self._state_dim)
        self._server_frequency = env_parameters.server_frequency


        self._observation_spec = spaces.Box(low=np.float32(low_state), high=np.float32(high_state),  dtype=np.float32)

        self._state = np.zeros(self._state_dim)
        self._episode_ended = False
        self._num_base_station = env_parameters.num_base_station
        # initialize the environment
        self.server_list = []
        for i in range(env_parameters.num_base_station):
            # randomly assign the server poisson rate between 1 to env_parameters.server_poisson_rate
            server = MECServer(index=i,
                               poisson_rate=self.server_poisson_rate[i],
                               task_data_lower_bound=env_parameters.server_task_data_lower_bound,
                               task_data_higher_bound=env_parameters.server_task_data_higher_bound,
                               ratio_lower_bound=env_parameters.ratio_lower_bound,
                               ratio_higher_bound=env_parameters.ratio_higher_bound,
                               frequence=self._server_frequency)
            self.server_list.append(server)

        # initialization the task
        self.client_poisson_rate = env_parameters.client_poisson_rate
        self.client_task_data_lower_bound = env_parameters.client_task_data_lower_bound
        self.client_task_data_higher_bound = env_parameters.client_task_data_higher_bound
        self.ratio_lower_bound = env_parameters.ratio_lower_bound
        self.ratio_higher_bound = env_parameters.ratio_higher_bound

        self.map_width = env_parameters.map_width
        self.map_height = env_parameters.map_height
        self.num_horizon_servers = env_parameters.num_horizon_servers
        self.num_vertical_servers = env_parameters.num_vertical_servers
        self.transmission_rates = env_parameters.transmission_rates
        self.trace_length = env_parameters.trace_length
        self.trace_interval = env_parameters.trace_interval

        self._total_time_slot_length = env_parameters.trace_length

        self.servers_position = self._initialize_servers_position()
        # read the users traces
        # record the total trace number
        self.users_traces = self._read_traces_from_the_csv(env_parameters.traces_file_path, env_parameters.trace_start_index, self._num_traces)
        self._current_time_slot = [0] * self._num_traces
        self.batch_size = self._num_traces

    def _initialize_servers_position(self):
        delta_x = self.map_width / self.num_horizon_servers
        delta_y = self.map_height / self.num_vertical_servers

        servers_poision = []

        for i in range(self.num_vertical_servers):
            for j in range(self.num_horizon_servers):
                server_x = delta_x / 2.0 + j*delta_x
                server_y = delta_y / 2.0 + i*delta_y

                servers_poision.append(Point(x=server_x, y=server_y))

        return servers_poision

    def _read_traces_from_the_csv(self, file_path, start_index, num_of_traces):
        f = open(file_path, "r")

        users_traces = {}
        lines = f.readlines()

        for line in lines:
            if line[0] == '[':
                continue
            items = line.split()
            x = float(items[1])
            y = float(items[2])

            if items[0] not in users_traces.keys():
                users_traces[items[0]] = []
                users_traces[items[0]].append(Point(x, y))
            else:
                users_traces[items[0]].append(Point(x, y))

        f.close()
        user_names = list(users_traces.keys())
        users_traces_list = []
        for i in range(start_index, (start_index+num_of_traces)):
            user_name = user_names[i]
            one_user_trace = users_traces[user_name][:: self.trace_interval]
            one_user_trace = one_user_trace[0:self.trace_length]
            users_traces_list.append(one_user_trace)

        return users_traces_list
    # This function aims at find the area that user belongs to
    def _get_user_area_by_position(self, user_position):
        delta_x = self.map_width / float(self.num_horizon_servers)
        delta_y = self.map_height / float(self.num_vertical_servers)

        x_index = int(user_position.x / delta_x)
        y_index = int(user_position.y / delta_y)

        index = y_index * self.num_horizon_servers + x_index
        return index

    def _get_wireless_transmission_rate(self, user_position):
        servers_index = self._get_user_area_by_position(user_position)
        base_state_position = self.servers_position[servers_index]

        x_distance = abs(user_position.x - base_state_position.x)
        y_distance = abs(user_position.y - base_state_position.y)

        num_areas = len(self.transmission_rates)

        delta_x = self.map_width / float(self.num_horizon_servers)
        delta_y = self.map_height / float(self.num_vertical_servers)

        area_cover_unit_x = delta_x / 2.0 / num_areas
        area_cover_unit_y = delta_y / 2.0 / num_areas

        area_number = max(int(x_distance / area_cover_unit_x), int(y_distance / area_cover_unit_y))

        return self.transmission_rates[area_number]  # bps

    def _get_number_of_hops(self, base_one_index, base_two_index):
        base_one_index_y = int(base_one_index / self.num_horizon_servers)
        base_one_index_x = int(base_one_index % self.num_horizon_servers)

        base_two_index_y = int(base_two_index / self.num_horizon_servers)
        base_two_index_x = int(base_two_index % self.num_horizon_servers)

        # applying Manhattan Distance
        num_of_hops = abs(base_two_index_x - base_one_index_x) + abs(base_one_index_y - base_two_index_y)

        return num_of_hops

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_migration_cost(self):
        image_size = np.random.uniform(self.migration_size_low,
                                 self.migration_size_high)
        migration_cost = image_size * 8.0 / self._optical_fiber_trans_rate

        return migration_cost

    def get_migration_coefficient(self):
        return np.random.uniform(self.migration_coefficient_low, self.migration_coefficient_high)

    def _generate_client_work_loads(self):
        num_arriving_tasks = max(1, np.random.poisson(self.client_poisson_rate))

        total_required_frequency = 0.0
        task_data_volume = 0.0
        for i in range(num_arriving_tasks):
            task_data = np.random.uniform(self.client_task_data_lower_bound,
                                          self.client_task_data_higher_bound)

            task_comp_to_volume_ratio = np.random.uniform(self.ratio_lower_bound,
                                                          self.ratio_higher_bound)

            total_required_frequency += task_data * task_comp_to_volume_ratio
            task_data_volume += task_data

        total_required_frequency = total_required_frequency / (1024.0 * 1024.0 * 1024.0) # Hz --> GHz
        task_data_volume = task_data_volume / (1024.0 * 1024.0 ) # bit -- > Mb

        return total_required_frequency, task_data_volume

    def _make_state_according_to_action(self, trace_id, action):
        # this action should be one action rather than a batch of actions.
        # generate_client_task
        client_required_frequency, task_data_volume = self._generate_client_work_loads()

        user_position = self.users_traces[trace_id][self._current_time_slot[trace_id]]
        user_position_index = self._get_user_area_by_position(user_position)

        if action == None:
            service_index = user_position_index
        else:
            # the service index is the second dimension of true state
            service_index = self._state[trace_id][1]

        trans_rate = self._get_wireless_transmission_rate(user_position)

        server_workloads = []
        servers_computation_latencies = []
        for server in self.server_list:
            server_workload = server.get_current_workload()
            server_workloads.append(server_workload)
            computation_latency = float(server_workload + client_required_frequency) / server.frequence
            servers_computation_latencies.append(computation_latency)
            # servers_workloads.append(server_workload)


        self._client_required_frequency[trace_id] = client_required_frequency
        self._task_data_volume[trace_id] = task_data_volume
        self._server_workloads[trace_id] = server_workloads
        self._service_index[trace_id] = service_index
        self._user_position_index[trace_id] = user_position_index
        self._trans_rate[trace_id] = trans_rate

        servers_num_of_hops = []
        servers_migration_num_of_hops = []

        communication_costs = []

        current_migration_cost = self.get_migration_cost()
        current_migration_coefficient = self.get_migration_coefficient()
        self._migration_coefficient[trace_id] = current_migration_coefficient
        self._migration_cost[trace_id] = current_migration_cost

        for server in self.server_list:
            num_of_hops = self._get_number_of_hops(user_position_index, server.index)

            # Calculate the migration number of hops
            migration_num_of_hops = self._get_number_of_hops(service_index, server.index)

            servers_migration_num_of_hops.append(migration_num_of_hops)
            servers_num_of_hops.append(num_of_hops)

            wired_communication_cost = (task_data_volume / self._optical_fiber_trans_rate) * min(num_of_hops, 1) \
                                       + self.backhaul_coefficient * num_of_hops
            #wired_communication_cost = self._optical_fiber_trans_rate * num_of_hops
            #here we change a way to calculate the migration costs
            communication_cost = float(task_data_volume) / float(trans_rate) + wired_communication_cost + \
                                 (migration_num_of_hops * current_migration_coefficient + current_migration_cost)

            communication_costs.append(communication_cost)

        # what we should return is the observation instead of the true state
        #state = [user_position_index, service_index, trans_rate, client_required_frequency,
        #         task_data_volume] + servers_computation_latencies + servers_num_of_hops

        # when action is None, we do the reset()
        if action != None:
            if self.is_full_action:
                service_index = action
            else:
                service_index = self._get_service_index_by_action(action, service_index, user_position_index)

        # state = [self._user_position_index, self._service_index ] + servers_computation_latencies + communication_costs
        # observation = [self._user_position_index] + servers_computation_latencies + communication_costs
        # there are several state for the service and users
        state = [user_position_index, service_index] + servers_computation_latencies + communication_costs + \
                [trans_rate, client_required_frequency, task_data_volume] + server_workloads
        num_of_hops = self._get_number_of_hops(user_position_index, service_index)
        observation = [user_position_index, trans_rate, task_data_volume, client_required_frequency]

        return state, observation

    def extract_system_infomation_from_state(self, states):
        positions_vector = states[:, 0:2]
        client_side_vector = states[:, 2+self._num_base_station*2: 2+self._num_base_station*2+3]
        server_workloads = states[:, 5+self._num_base_station*2:]

        system_info_vector = np.concatenate([positions_vector, client_side_vector, server_workloads], axis=-1)

        return system_info_vector

    def reset_trace(self, trace_id):
        self._current_time_slot[trace_id] = 0

        state, observation = self._make_state_according_to_action(trace_id, action=None)

        return state, observation

    def reset(self):
        # true state [user_position_index, service_index, trans_rate, client_required_frequency, task_data_volume]
        batch_state = []
        batch_observation = []
        self._client_required_frequency = np.zeros(shape=(self._num_traces,), dtype=np.float32)
        self._task_data_volume = np.zeros(shape=(self._num_traces,), dtype=np.float32)
        self._server_workloads = np.zeros(shape=(self._num_traces, self._num_base_station), dtype=np.float32)
        self._service_index = np.zeros(shape=(self._num_traces,), dtype=np.float32)
        self._user_position_index = np.zeros(shape=(self._num_traces,), dtype=np.float32)
        self._trans_rate = np.zeros(shape=(self._num_traces,), dtype=np.float32)
        self._migration_cost = np.zeros(shape=(self._num_traces,), dtype=np.float32)
        self._migration_coefficient = np.zeros(shape=(self._num_traces,), dtype=np.float32)

        for i in range(self._num_traces):
            state, observation = self.reset_trace(i)
            batch_state.append(state)
            batch_observation.append(observation)

        # the true state space dimension is
        self._state = np.array(batch_state, dtype=np.float32)
        observation = np.array(batch_observation, dtype=np.float32)

        if self.is_full_observation:
            return self._state
        else:
            return observation

    def current_system_state(self):
        system_state = np.column_stack([self._user_position_index, self._service_index, self._trans_rate, self._client_required_frequency,
                         self._task_data_volume, self._server_workloads, self._migration_cost, self._migration_coefficient])

        return system_state

    def _reward_func(self, latency):
        return -(latency)

    def step_trace(self, trace_id, action):
        # first, calculating the computatio cost of each service node
        # user_profile, service_workloads, servers_num_of_hops = self._get_info_from_current_state()

        computation_cost = self._state[trace_id][2+action]
        communication_migration_cost = self._state[trace_id][2+self._num_base_station +action]

        # print("————————step trace ------")
        # print("env action : ", action)
        # print("get current system info: ", self.current_system_state())
        # print("state computation cost: ", computation_cost)
        # print("state communication_migration_cost", communication_migration_cost)
        #
        # print("workloads: ", self._server_workloads[trace_id])
        #
        # computation_cost = self.server_list[action].get_estimated_running_time(self._client_required_frequency[trace_id], self._server_workloads[trace_id][action])
        # communication_migration_cost = self._get_number_of_hops(self._service_index[trace_id], action) * 0.3 + \
        #                                     self._task_data_volume[trace_id] / self._trans_rate[trace_id] + \
        #                                     self._get_number_of_hops(self._user_position_index[trace_id], action) * (self._task_data_volume[trace_id] / self._optical_fiber_trans_rate)
        #
        # print("computation_cost", computation_cost)
        # print("communication_migration_cost", communication_migration_cost)

        # action = int(action)
        # computation_cost = self._state[trace_id][2+action]
        # communication_migration_cost = self._state[trace_id][2+self._num_base_station +action]
        #
        # print("step_trace: "+str(trace_id) + ": ", self._state[trace_id])
        # print("step_trace: "+str(trace_id) + " action: ", action)

        reward = self._reward_func((computation_cost + communication_migration_cost))

        self._current_time_slot[trace_id] = self._current_time_slot[trace_id] + 1
        if self._current_time_slot[trace_id] == self._total_time_slot_length:
            done = True
            state, observation = self.reset_trace(trace_id)
        else:
            done = False
            state, observation = self._make_state_according_to_action(trace_id, action=action)

        return state, observation, reward, done, state

    def step(self, action):
        states = []
        observations = []
        rewards = []
        dones = []
        env_infos = []
        for i in range(self._num_traces):
            state, observation, reward, done, env_info = self.step_trace(trace_id=i, action=action[i])
            states.append(state)
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            env_infos.append(env_info)

        self._state = np.array(states, dtype=np.float32)
        observations = np.array(observations, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)

        if self.is_full_observation:
            return self._state, rewards, dones, env_infos
        else:
            return observations, rewards, dones, env_infos
