import numpy as np
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path
# def optimal_soution(env, system_infos):
#     # this function is used to calculate the optimal system infomations.
#     pass


# build the graph and calculate the shortest path for the graph.

def calculate_edge_weight(env,
                          service_index, user_position_index,
                          time_step,
                          system_infos):
    trans_rate = system_infos[time_step][2]
    client_required_frequency = system_infos[time_step][3]
    task_data_volume = system_infos[time_step][4]
    server_workloads = system_infos[time_step][5:-1]
    migration_cost = system_infos[time_step][-2]
    migration_coefficeitn = system_infos[time_step][-1]

    edge_weights = []
    computation_costs = []
    communication_migration_costs = []

    #print("migration cost: ", migration_cost)

    for action in range(env._action_dim):
        computation_cost = env.server_list[action].get_estimated_running_time(client_required_frequency,
                                                                              server_workloads[action])
        computation_costs.append(computation_cost)
        num_of_hops = env._get_number_of_hops(user_position_index, action)
        wired_communication_cost = (task_data_volume / env._optical_fiber_trans_rate) * min(num_of_hops, 1) \
                                   + env.backhaul_coefficient * num_of_hops

        communication_migration_cost = env._get_number_of_hops(service_index, action) * migration_coefficeitn + migration_cost + \
                                       task_data_volume / trans_rate + \
                                       wired_communication_cost

        total_cost = computation_cost + communication_migration_cost

        edge_weights.append(total_cost)

    # print("--------------- in optimal ---------")
    # print("trans_rate", trans_rate)
    # print("task_data_volume", task_data_volume)
    # print("client_required_frequency", client_required_frequency)
    # print("server_workloads", server_workloads)
    #
    # print(computation_costs)
    # print(communication_migration_costs)

    if time_step == 0:
        edges = [(0, action+1, {"weight": weight}) for action, weight in zip(range(env._action_dim), edge_weights)]
    else:
        edges = [( (1+ (time_step-1) * env._action_dim + service_index),
                   (1+time_step*env._action_dim + action),  {"weight": weight}) for action, weight
                 in zip(range(env._action_dim), edge_weights) ]

    return edges, edge_weights

def optimal_solution(env, system_infos):
    time_step = len(system_infos)
    action_dim = env._action_dim
    num_of_nodes = 2 + action_dim * time_step
    edge_weights_info = {}
    # build the graph based on the collected infomation
    # system_infos include user_position_index, trans_rate,
    # client_required_frequency,task_data_volume, server_workloads
    G = nx.Graph()
    G.add_nodes_from(range(num_of_nodes))

    # add first node
    service_index = system_infos[0][0]
    user_position_index = system_infos[0][0]

    edges, edge_weights = calculate_edge_weight(env,
                          service_index, user_position_index,
                          time_step=0,
                          system_infos=system_infos)

    edge_weights_info[(0, user_position_index, service_index)] = edge_weights
    G.add_edges_from(edges)

    # build the graph
    for i in range(1, time_step):
        user_position_index = system_infos[i][0]
        for service_index in range(env._action_dim):
            edges, edge_weights = calculate_edge_weight(env,
                                                        service_index,
                                                        user_position_index,
                                                        time_step=i,
                                                        system_infos=system_infos)
            edge_weights_info[(i, user_position_index, service_index)] = edge_weights
            G.add_edges_from(edges)

    # add the last node
    last_layer_edges = []
    for start_node in range(num_of_nodes-action_dim-1, num_of_nodes-1):
        edge = (start_node, num_of_nodes-1, {"weight": 0.0})
        last_layer_edges.append(edge)

    G.add_edges_from(last_layer_edges)

    length = nx.shortest_path_length(G, source=0, target=num_of_nodes-1, weight='weight')


    return G, edge_weights_info, length

def optimal_solution_for_batch_system_infos(env, system_infos):
    total_solution_lengths = 0.0
    num = float(len(system_infos))
    for system_info in system_infos:
        _, edge_weights_info, length = optimal_solution(env, system_info)

        total_solution_lengths += length

    return total_solution_lengths / num

if __name__ == "__main__":
    from environment.migration_env import MigrationEnv
    from environment.batch_migration_env import BatchMigrationEnv
    from environment.batch_migration_env import EnvironmentParameters
    from sampler.migration_sampler import EvaluationSampler
    from sampler.migration_sampler import MigrationSamplerProcess
    from baselines.linear_baseline import LinearFeatureBaseline
    from policies.always_migrate_policy import AlwaysMigratePolicy
    from policies.always_migration_solution import always_migration_solution
    from policies.no_migration_solution import no_migration_solution
    import matplotlib.pyplot as plt

    number_of_base_state = 64
    x_base_state = 8
    y_base_state = 8

    possion_rate_vector = np.random.randint(5, 21, size=number_of_base_state)
    print("possion_rate_vector is: ", repr(possion_rate_vector))
    possion_rate_vector = [11,  8, 20,  9, 18, 18,  9, 17, 12, 17,  9, 17, 14, 10,  5,  7, 12,
        8, 20, 10, 14, 12, 20, 14,  8,  6, 15,  7, 18,  9,  8, 18, 17,  7,
       11, 11, 13, 14,  8, 18, 13, 17,  6, 18, 17, 18, 18,  7,  9,  6, 12,
       10,  9,  8, 20, 14, 11, 15, 14,  6,  6, 15, 16, 20] # sanfransisco trace
    # possion_rate_vector = [17, 19, 16, 15, 16, 20, 20, 24, 19, 18, 16, 30, 15, 18, 24, 19, 19,
    #                        18, 24, 17, 21, 22, 18, 24, 22, 19, 27, 16, 18, 16, 24, 25, 21, 21,
    #                        23, 29, 17, 18, 24, 23, 30, 30, 27, 23, 24, 15, 27, 22, 19, 25, 19,
    #                        22, 18, 28, 15, 22, 19, 26, 15, 20, 16, 28, 20, 26]

    # possion_rate_vector = [7, 10, 8, 14, 15, 6, 20, 18, 11, 17, 20, 9, 8, 14, 9, 15, 8, 17, 9, 9, 10, 7, 17, 10,
    #                        13, 12, 5, 8, 10, 13, 19, 15, 10, 9, 10, 18, 12, 13, 5, 11, 7, 8, 8, 19, 15, 15, 6, 10,
    #                        5, 20, 17, 5, 5, 16, 5, 19, 19, 19, 9, 20, 17, 14, 17, 17]

    # start point (41.856, 12.442), end point (41.928,12.5387), a region in Roman, Italy.
    env_eval_parameters = EnvironmentParameters(trace_start_index=0,
                                                num_traces=10,
                                                server_frequency=128.0,  # GHz
                                                num_base_station=number_of_base_state,
                                                optical_fiber_trans_rate=1000.0,
                                                switch_coefficient = 0.02,
                                                server_poisson_rate=possion_rate_vector, client_poisson_rate=2,
                                                server_task_data_lower_bound=(4 * 1024.0 * 1024.0 * 8),
                                                server_task_data_higher_bound=(8 * 1024.0 * 1024.0 * 8),
                                                client_task_data_lower_bound=(4 * 1024.0 * 1024.0*8),
                                                client_task_data_higher_bound=(8 * 1024.0 * 1024.0*8),
                                                migration_size_low=0.5,
                                                migration_size_high=2000.0,
                                                ratio_lower_bound=200.0,
                                                ratio_higher_bound=10000.0,
                                                map_width=8000.0, map_height=8000.0,
                                                num_horizon_servers=x_base_state, num_vertical_servers=y_base_state,
                                                traces_file_path='../environment/san_traces_coordinate.txt',
                                                transmission_rates=[60.0, 48.0, 36.0, 24.0, 12.0],  # Mbps
                                                trace_length=100,
                                                trace_interval=3,
                                                is_full_observation=False,
                                                is_full_action=True)
    env = BatchMigrationEnv(env_eval_parameters)

    sampler = EvaluationSampler(env,
                                policy=AlwaysMigratePolicy(env._state_dim,
                                                         action_dim=env._action_dim),
                                                         batch_size=10,
                                                         max_path_length=100)

    for i in range(3):
        reward_collects, system_info_collects = sampler.obtain_samples(is_rnn=False)
        system_info_collects = np.array(system_info_collects)
        # for i in range(100):
        #     print(system_info_collects[0][i][0])
        #print(system_info_collects[0][1:-1][0])

        #print("processing sample's system_info shape", system_info_collects.shape)

        optimal_latency = optimal_solution_for_batch_system_infos(env, system_info_collects)
        print("optimal_solution is: ", optimal_latency)

        always_migration_latency= always_migration_solution(env, system_info_collects)
        print("always_migration_solution is: ", always_migration_latency)

        nomigration_latency = no_migration_solution(env, system_info_collects)
        print("no migration solution is: ", nomigration_latency)
