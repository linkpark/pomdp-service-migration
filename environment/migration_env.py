import numpy as np
import random
import gym

from gym import spaces
from collections import namedtuple

import utils.logger as logger

"The initial state of Migration Environment including the initial position. distances between services and positions"

# size of input task input data d_i [0.5, 3] MB
# computation-to-volume ratio [100, 3200] cycles / byte
# MEC server's CPU frequencey f = 10 GHz
# number of hops task_load / transmission latency * hops
# transmission_rate: Mbps

# return the partial infomation

Point = namedtuple('Point', ['x', 'y'])
EnvironmentParameters = namedtuple('ServersParameter', ['trace_start_index', 'num_traces','num_base_station', 'optical_fiber_trans_rate', 'server_poisson_rate', 'server_task_data_lower_bound',
                                                       'server_frequency', 'server_task_data_higher_bound', 'ratio_lower_bound','ratio_higher_bound',
                                                       'client_poisson_rate', 'client_task_data_lower_bound', 'client_task_data_higher_bound',
                                                        'migration_cost_low', 'migration_cost_high',
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
        self.frequence = frequence

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

class MigrationEnv(gym.Env):
    def __init__(self, env_parameters):
        self.is_batch = False
        self.is_full_action = env_parameters.is_full_action
        if self.is_full_action:
            self._action_spec = spaces.Discrete(env_parameters.num_base_station)
            self._action_dim = env_parameters.num_base_station
        else:
            self._action_spec = spaces.Discrete(6)
            self._action_dim = 6

        self.migration_cost_low = env_parameters.migration_cost_low
        self.migration_cost_high = env_parameters.migration_cost_high
        self.server_poisson_rate = env_parameters.server_poisson_rate
        self._optical_fiber_trans_rate = env_parameters.optical_fiber_trans_rate  # Mbps
        self.is_full_observation = env_parameters.is_full_observation
        #self._state_dim = 5 + 2 * env_parameters.num_base_station
        if self.is_full_observation:
            self._state_dim = 2 * env_parameters.num_base_station + 2
        else:
            self._state_dim = 4

        # The state include (distance, num_of_hops, workload_distribution, service_index, data_process)
        # this is a gym environment with Box as the observation type
        low_state = np.array( [float("-inf")]*self._state_dim)
        high_state = np.array( [float("inf")] * self._state_dim)

        self._observation_spec = spaces.Box(low=np.float32(low_state), high=np.float32(high_state),  dtype=np.float32)

        self._state = np.zeros(self._state_dim)
        self._episode_ended = False
        self._num_base_station = env_parameters.num_base_station
        self._server_frequency = env_parameters.server_frequency
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

        self.servers_position = self._initialize_servers_position()
        self._num_traces = env_parameters.num_traces
        self.trace_length = env_parameters.trace_length
        self.trace_interval = env_parameters.trace_interval

        # read the users traces
        self.users_traces = self._read_traces_from_the_csv(env_parameters.traces_file_path,
                                                           env_parameters.trace_start_index,
                                                           env_parameters.num_traces)
        self._total_time_slot_length = env_parameters.trace_length
        self._current_time_slot = 0

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
        for i in range(start_index, start_index+num_of_traces):
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

    def _make_state_according_to_action(self, action, done):
        # generate_client_task
        self._client_required_frequency, self._task_data_volume = self._generate_client_work_loads()

        self._user_position = self.one_user_trace[self._current_time_slot]
        self._user_position_index = self._get_user_area_by_position(self._user_position)

        if action == None:
            self._service_index = self._user_position_index

        self._trans_rate = self._get_wireless_transmission_rate(self._user_position)
        self._server_workloads = []

        # servers_workloads = []
        servers_computation_latencies = []
        for server in self.server_list:
            server_workload = server.get_current_workload()
            self._server_workloads.append(server_workload)
            computation_latency = float(server_workload + self._client_required_frequency) / server.frequence
            servers_computation_latencies.append(computation_latency)
            # servers_workloads.append(server_workload)

        self._servers_num_of_hops = []
        self._migration_num_of_hops = []

        communication_costs = []

        # get the migration cost at each time step.
        self._current_migration_cost = np.random.uniform(self.migration_cost_low,
                                                         self.migration_cost_high)

        for server in self.server_list:
            num_of_hops = self._get_number_of_hops(self._user_position_index, server.index)
            # Calculate the migration number of hops
            migration_num_of_hops = self._get_number_of_hops(self._service_index, server.index)

            self._migration_num_of_hops.append(migration_num_of_hops)
            self._servers_num_of_hops.append(num_of_hops)

            communication_cost = float(self._task_data_volume) / float(self._trans_rate) + num_of_hops * (
                        float(self._task_data_volume) / self._optical_fiber_trans_rate) + migration_num_of_hops * self._current_migration_cost

            communication_costs.append(communication_cost)

        # what we should return is the observation instead of the true state
        #state = [user_position_index, service_index, trans_rate, client_required_frequency,
        #         task_data_volume] + servers_computation_latencies + servers_num_of_hops

        # when action is None, we do the reset()
        if action != None:
            if self.is_full_action:
                self._service_index = action
            else:
                self._service_index = self._get_service_index_by_action(action)

        # state = [self._user_position_index, self._service_index ] + servers_computation_latencies + communication_costs
        # observation = [self._user_position_index] + servers_computation_latencies + communication_costs
        # there are several state for the service and users
        self._work_loads_info = [self._user_position_index, self._service_index,
                           self._trans_rate, self._task_data_volume, self._client_required_frequency] + self._server_workloads + [self._current_migration_cost]
        full_state = [self._user_position_index, self._service_index] + \
                     servers_computation_latencies + \
                     communication_costs

        num_of_hops = self._get_number_of_hops(self._user_position_index, self._service_index)
        observation = [self._user_position_index, self._service_index, self._trans_rate, self._task_data_volume, self._client_required_frequency]

        return full_state, observation

    def _get_service_index_by_action(self, action):
        # do not migrate
        if action == 0:
            service_index = self._service_index
        # migrate to the user's position
        elif action == 1:
            service_index = self._user_position_index
        # migrate to upper base station
        elif action == 2:
            if self._user_position_index - self.num_horizon_servers < 0:
                service_index= self._user_position_index
            else:
                service_index = self._user_position_index - self.num_horizon_servers
        # migrate to left base station
        elif action == 3:
            if self._user_position_index % self.num_horizon_servers == 0:
                service_index = self._user_position_index
            else:
                service_index = self._user_position_index - 1

        # migrate to right base station
        elif action == 4:
            if self._user_position_index + self.num_horizon_servers > (self._num_base_station - 1):
                service_index = self._user_position_index
            else:
                service_index = self._user_position_index + self.num_horizon_servers
        # migrate to the down base state
        elif action == 5:
            if self._user_position_index % self.num_horizon_servers == (self.num_horizon_servers-1):
                service_index = self._user_position_index
            else:
                service_index = self._user_position_index + 1

        return service_index

    def reset(self):
        self._current_time_slot = 0
        self._episode_ended = False

        # random sample one user trace from the traces:
        trace_index = random.randint(0, len(self.users_traces)-1)
        self.one_user_trace = self.users_traces[trace_index]
        self._total_time_slot_length = len(self.one_user_trace)

        # true state [user_position_index, service_index, trans_rate, client_required_frequency, task_data_volume]

        state, observations = self._make_state_according_to_action(action=None, done=False)

        # the true state space dimension is
        self._state = np.array(state, dtype=np.float32)

        if self.is_full_observation:
            return self._state
        else:
            return np.array(observations, dtype=np.float32)

    def current_system_state(self):
        return self._work_loads_info

    def _reward_func(self, latency):
        return -(latency)

    def step(self, action):
        try:
            computation_cost = self.server_list[action].get_estimated_running_time(self._client_required_frequency,
                                                                                   self._server_workloads[action])
        except:
            raise Exception('action should be less than '+str(self._num_base_station-1)+'but the action is: '+str(action))

        communication_migration_cost = self._get_number_of_hops(self._service_index, action) * 0.3 + \
                                            self._task_data_volume / self._trans_rate + \
                                            self._get_number_of_hops(self._user_position_index, action) * (self._task_data_volume / self._optical_fiber_trans_rate)

        reward = self._reward_func((computation_cost + communication_migration_cost))

        self._current_time_slot = self._current_time_slot + 1
        if self._current_time_slot == self._total_time_slot_length:
            self._episode_ended = True

        if self._episode_ended:
            done = True
            observation = self.reset()
        else:
            done = False
            state, observation = self._make_state_according_to_action(action=action, done=done)

            self._state = np.array(state, dtype=np.float32)
            if self.is_full_observation:
                observation = np.array(state, dtype=np.float32)
            else:
                observation = np.array(observation, dtype=np.float32)

        return observation, reward, done, {}


if __name__ == "__main__":
    # Test of the MEC server
    import numpy as np
    possion_rate_vector = np.random.randint(5,21,size=64)
    env_default_parameters = EnvironmentParameters(trace_start_index=201,
                                                   num_traces=20,
                                                   num_base_station=64, optical_fiber_trans_rate=60.0,
                                                   server_poisson_rate=possion_rate_vector, client_poisson_rate=4,
                                                   server_frequency=32.0, # GHz
                                                   server_task_data_lower_bound=(0.5 * 1024.0 * 1024.0),
                                                   server_task_data_higher_bound=(5 * 1024.0 * 1024.0),
                                                   ratio_lower_bound=100.0,
                                                   client_task_data_lower_bound=(0.5 * 1024.0 * 1024.0 * 8.0), # MB
                                                   client_task_data_higher_bound=(5 * 1024.0 * 1024.0 * 8.0),  # MB
                                                   ratio_higher_bound=3200.0, map_width=8000.0, map_height=8000.0,
                                                   num_horizon_servers=8, num_vertical_servers=8,
                                                   traces_file_path='./rome_traces_coordinate.txt',
                                                   transmission_rates=[20.0, 16.0, 12.0, 8.0, 4.0],
                                                   trace_length=100,
                                                   trace_interval=10,
                                                   is_full_observation=False,
                                                   is_full_action=True)

    env = MigrationEnv(env_default_parameters)
    print("number of servers: ", len(env.servers_position))

    env.reset()
    for i in range(100):
        state, reward, done, _ = env.step(1)
        print("step "+str(i)+":", state[0:2])
