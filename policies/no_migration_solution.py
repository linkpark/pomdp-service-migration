import numpy as np
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path
# def optimal_soution(env, system_infos):
#     # this function is used to calculate the optimal system infomations.
#     pass


# build the graph and calculate the shortest path for the graph.

def calculate_cost_at_each_time_step(env,
                          service_index, user_position_index,
                          time_step,
                          system_infos):
    trans_rate = system_infos[time_step][2]
    client_required_frequency = system_infos[time_step][3]
    task_data_volume = system_infos[time_step][4]
    server_workloads = system_infos[time_step][5:-1]
    migration_cost = system_infos[time_step][-2]
    migration_coefficent = system_infos[time_step][-1]

    edge_weights = []
    computation_costs = []
    communication_costs=[]

    action = int(service_index)

    computation_cost = env.server_list[action].get_estimated_running_time(client_required_frequency,
                                                                          server_workloads[action])

    num_of_hops = env._get_number_of_hops(user_position_index, action)
    #print("no migration number of hops: ", num_of_hops)
    wired_communication_cost = (task_data_volume / env._optical_fiber_trans_rate) * min(num_of_hops, 1) \
                               + env.backhaul_coefficient * num_of_hops

    communication_migration_cost = env._get_number_of_hops(service_index, action) * migration_coefficent + migration_cost + \
                                   task_data_volume / trans_rate + \
                                   wired_communication_cost

    computation_costs.append(computation_cost)
    communication_costs.append(communication_migration_cost)
    total_cost = computation_cost + communication_migration_cost
    #print("wired_communication_cost: ", wired_communication_cost, "computation cost: ", computation_cost)
    return total_cost, action

def no_migration_solution(env, system_infos):
    time_step = len(system_infos[0])
    cost_batch = []

    for system_info in system_infos:
        service_index = system_info[0][0]
        costs = []

        for i in range(time_step):
            user_position_index = system_info[i][0]
            cost, service_index = calculate_cost_at_each_time_step(env,
                          service_index, user_position_index,
                          i,
                          system_info)
            costs.append(cost)

        cost_batch.append(costs)

    cost_batch = np.array(cost_batch)
    random_soultion = np.mean(np.sum(cost_batch, axis=-1))

    return random_soultion

if __name__ == "__main__":
    from environment.migration_env import MigrationEnv
    from environment.batch_migration_env import BatchMigrationEnv
    from sampler.migration_sampler import EvaluationSampler
    from environment.migration_env import EnvironmentParameters
    from sampler.migration_sampler import MigrationSampler
    from sampler.migration_sampler import MigrationSamplerProcess
    from baselines.linear_baseline import LinearFeatureBaseline
    from policies.always_migrate_policy import AlwaysMigratePolicy
    import matplotlib.pyplot as plt

    server_poisson_rates= [18,  8, 17, 19, 10, 13, 19, 12 , 8 ,10, 14 , 7, 17,  8, 11, 10, 16, 16,  9, 19 ,20,  8, 15,  6,
  6,  6, 17,  8, 17, 16, 15, 18,  8, 17,  5, 11, 12, 17, 10, 17, 12, 12,  9, 18,  7, 17,  9, 13,
  8, 11, 12, 19, 11,  9,  5, 16,  9,  8, 10, 12, 20, 16,  8]
    env_default_parameters = EnvironmentParameters(num_traces=10,
                                                   num_base_station=63, optical_fiber_trans_rate=60.0,
                                                   server_poisson_rate=server_poisson_rates, client_poisson_rate=4,
                                                   server_task_data_lower_bound=(0.5 * 1024.0 * 1024.0),
                                                   server_task_data_higher_bound=(5 * 1024.0 * 1024.0),
                                                   ratio_lower_bound=100.0,
                                                   client_task_data_lower_bound=(0.5 * 1024.0 * 1024.0),
                                                   client_task_data_higher_bound=(5 * 1024.0 * 1024.0),
                                                   ratio_higher_bound=3200.0, map_width=4500.0, map_height=3500.0,
                                                   num_horizon_servers=9, num_vertical_servers=7,
                                                   traces_file_path='../environment/default_scenario_LocationSnapshotReport.txt',
                                                   transmission_rates=[20.0, 16.0, 12.0, 8.0, 4.0],
                                                   trace_length=100,
                                                   trace_interval=5,
                                                   is_full_observation=False,
                                                   is_full_action=True)

    env = MigrationEnv(env_default_parameters)

    eval_sampler = EvaluationSampler(env,
                                     policy=AlwaysMigratePolicy(env._state_dim,
                                                                action_dim=env._action_dim),
                                     batch_size=240,
                                     num_environment_per_core=4,
                                     max_path_length=100,
                                     parallel=True,
                                     num_process=6)

    rewards, system_infos = eval_sampler.obtain_samples()
    system_infos = np.array(system_infos)
    print("processing sample's system_info shape", system_infos.shape)

    no_migration_latency = no_migration_solution(env, system_infos)
    print("no_migration_latency is: ", no_migration_latency)
