import numpy as np
import pickle as pickle
from multiprocessing import Process, Pipe
from multiprocessing import Pool as ProcessPool
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
import copy


def step_func(env, action):
    print("environment is: ", env)
    print("action is: ", action)
    return env.step(action)


def reset_func(env):
    return env.reset()

class MetaIterativeEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Internally, the environments are executed iteratively.
    Args:
        env (meta_policy_search.envs.base.MetaEnv): meta environment object
        meta_batch_size (int): number of meta tasks
        envs_per_task (int): number of environments per meta task
        max_path_length (int): maximum length of sampled environment paths - if the max_path_length is reached,
                               the respective environment is reset
    """

    def __init__(self, env, meta_batch_size, envs_per_task, max_path_length):
        self.envs = np.asarray([copy.deepcopy(env) for _ in range(meta_batch_size * envs_per_task)])
        self.ts = np.zeros(len(self.envs), dtype='int')  # time steps
        self.max_path_length = max_path_length

    def step(self, actions):
        """
        Steps the wrapped environments with the provided actions
        Args:
            actions (list): lists of actions, of length meta_batch_size x envs_per_task
        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool),
             env_infos (dict). Each list is of length meta_batch_size x envs_per_task
             (assumes that every task has same number of envs)
        """
        assert len(actions) == self.num_envs

        all_results = [env.step(a) for (a, env) in zip(actions, self.envs)]

        # stack results split to obs, rewards, ...
        obs, rewards, dones, env_infos = list(map(list, zip(*all_results)))

        # reset env when done or max_path_length reached
        dones = np.asarray(dones)
        self.ts += 1
        dones = np.logical_or(self.ts >= self.max_path_length, dones)
        for i in np.argwhere(dones).flatten():
            obs[i] = self.envs[i].reset()
            self.ts[i] = 0

        return obs, rewards, dones, env_infos

    def set_tasks(self, tasks):
        """
        Sets a list of tasks to each environment
        Args:
            tasks (list): list of the tasks for each environment
        """
        envs_per_task = np.split(self.envs, len(tasks))
        for task, envs in zip(tasks, envs_per_task):
            for env in envs:
                env.set_task(task)

    def reset(self):
        """
        Resets the environments
        Returns:
            (list): list of (np.ndarray) with the new initial observations.
        """
        obses = [env.reset() for env in self.envs]
        self.ts[:] = 0
        return obses

    @property
    def num_envs(self):
        """
        Number of environments
        Returns:
            (int): number of environments
        """
        return len(self.envs)

class MetaParallelEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Thereby the environments are distributed among meta_batch_size processes and
    executed in parallel.
    Args:
        env (meta_policy_search.envs.base.MetaEnv): meta environment object
        meta_batch_size (int): number of meta tasks
        envs_per_task (int): number of environments per meta task
        max_path_length (int): maximum length of sampled environment paths - if the max_path_length is reached,
                             the respective environment is reset
    """

    def __init__(self, env, meta_batch_size, envs_per_task, max_path_length):
        self.n_envs = meta_batch_size * envs_per_task
        self.meta_batch_size = meta_batch_size
        self.envs_per_task = envs_per_task
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(meta_batch_size)])
        seeds = np.random.choice(range(10**6), size=meta_batch_size, replace=False)

        self.ps = [
            Process(target=worker, args=(work_remote, remote, pickle.dumps(env), envs_per_task, max_path_length, seed))
            for (work_remote, remote, seed) in zip(self.work_remotes, self.remotes, seeds)]  # Why pass work remotes?

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        """
        Executes actions on each env
        Args:
            actions (list): lists of actions, of length meta_batch_size x envs_per_task
        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool), env_infos (dict)
                      each list is of length meta_batch_size x envs_per_task (assumes that every task has same number of envs)
        """
        assert len(actions) == self.num_envs

        # split list of actions in list of list of actions per meta tasks
        chunks = lambda l, n: [l[x: x + n] for x in range(0, len(l), n)]
        actions_per_meta_task = chunks(actions, self.envs_per_task)

        # step remote environments
        for remote, action_list in zip(self.remotes, actions_per_meta_task):
            remote.send(('step', action_list))

        results = [remote.recv() for remote in self.remotes]

        obs, rewards, dones, env_infos = map(lambda x: sum(x, []), zip(*results))

        return obs, rewards, dones, env_infos

    def reset(self):
        """
        Resets the environments of each worker
        Returns:
            (list): list of (np.ndarray) with the new initial observations.
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        return sum([remote.recv() for remote in self.remotes], [])

    def set_tasks(self, tasks=None):
        """
        Sets a list of tasks to each worker
        Args:
            tasks (list): list of the tasks for each worker
        """
        for remote, task in zip(self.remotes, tasks):
            remote.send(('set_task', task))
        for remote in self.remotes:
            remote.recv()

    @property
    def num_envs(self):
        """
        Number of environments
        Returns:
            (int): number of environments
        """
        return self.n_envs

class ParallelEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Thereby the environments are distributed among meta_batch_size processes and
    executed in parallel.
    Args:
        env (meta_policy_search.envs.base.MetaEnv): meta environment object
        meta_batch_size (int): number of meta tasks
        envs_per_task (int): number of environments per meta task
        max_path_length (int): maximum length of sampled environment paths - if the max_path_length is reached,
                             the respective environment is reset
    """

    def __init__(self, env, process_core, envs_per_core, max_path_length, is_batch_env=False):
        self.n_envs = process_core * envs_per_core
        self.process_core = process_core
        self.envs_per_task = envs_per_core
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(process_core)])
        seeds = np.random.choice(range(10**6), size=process_core, replace=False)
        self.env = env

        self.ps = [
            Process(target=worker, args=(work_remote, remote, pickle.dumps(env), envs_per_core, max_path_length, seed))
            for (work_remote, remote, seed) in zip(self.work_remotes, self.remotes, seeds)]  # Why pass work remotes?

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.is_batch_env = is_batch_env

    def step(self, actions):
        """
        Executes actions on each env
        Args:
            actions (list): lists of actions, of length meta_batch_size x envs_per_task
        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool), env_infos (dict)
                      each list is of length meta_batch_size x envs_per_task (assumes that every task has same number of envs)
        """
        if self.is_batch_env == True:
            assert len(actions) == self.num_envs * self.env.batch_size
        else:
            assert len(actions) == self.num_envs


        # split list of actions in list of list of actions per meta tasks
        chunks = lambda l, n: [l[x: x + n] for x in range(0, len(l), n)]
        if self.is_batch_env == True:
            actions_per_core = chunks(actions, self.env.batch_size)
            actions_per_core = chunks(actions_per_core, self.envs_per_task)
        else:
            actions_per_core = chunks(actions, self.envs_per_task)

        # step remote environments
        for remote, action_list in zip(self.remotes, actions_per_core):
            remote.send(('step', action_list))

        # for remote in self.remotes:
        #     obs, rewards, dones, env_infos = remote.recv()
        #     print("obs shape: ", np.array(obs).shape)
        #     print("rewards shape: ", np.array(rewards).shape)
        #     print("dones ", dones)
        #     print("env_infos ", env_infos)

        results = [remote.recv() for remote in self.remotes]

        obs, rewards, dones, env_infos = map(lambda x: sum(x, []), zip(*results))

        if self.is_batch_env == True:
            obs = np.concatenate(obs)
            rewards = np.concatenate(rewards)
            dones = sum(dones, [])
            env_infos = sum(env_infos, [])


        return obs, rewards, dones, env_infos

    def current_system_state(self):
        pass

    def reset(self):
        """
        Resets the environments of each worker
        Returns:
            (list): list of (np.ndarray) with the new initial observations.
        """
        for remote in self.remotes:
            remote.send(('reset', None))


        ret = sum([remote.recv() for remote in self.remotes], [])

        if self.is_batch_env:
            ret = np.concatenate(ret, axis=0)

        return ret

    def current_system_state(self):
        """
        Return the environment state checkpoints
        """
        for remote in self.remotes:
            remote.send(('current_system_state', None))

        ret = sum([remote.recv() for remote in self.remotes], [])

        if self.is_batch_env:
            ret = np.concatenate(ret, axis=0)

        return ret

    def set_tasks(self, tasks=None):
        """
        Sets a list of tasks to each worker
        Args:
            tasks (list): list of the tasks for each worker
        """
        for remote, task in zip(self.remotes, tasks):
            remote.send(('set_task', task))
        for remote in self.remotes:
            remote.recv()

    @property
    def num_envs(self):
        """
        Number of environments
        Returns:
            (int): number of environments
        """
        return self.n_envs


def worker(remote, parent_remote, env_pickle, n_envs, max_path_length, seed):
    """
    Instantiation of a parallel worker for collecting samples. It loops continually checking the task that the remote
    sends to it.
    Args:
        remote (multiprocessing.Connection):
        parent_remote (multiprocessing.Connection):
        env_pickle (pkl): pickled environment
        n_envs (int): number of environments per worker
        max_path_length (int): maximum path length of the task
        seed (int): random seed for the worker
    """
    parent_remote.close()

    envs = [pickle.loads(env_pickle) for _ in range(n_envs)]
    np.random.seed(seed)

    ts = np.zeros(n_envs, dtype='int')

    while True:
        # receive command and data from the remote
        cmd, data = remote.recv()

        # do a step in each of the environment of the worker
        if cmd == 'step':
            all_results = [env.step(a) for (a, env) in zip(data, envs)]
            obs, rewards, dones, infos = map(list, zip(*all_results))

            remote.send((obs, rewards, dones, infos))

        elif cmd == 'current_system_state':
            all_results = [env.current_system_state() for env in envs]
            remote.send(all_results)

        # reset all the environments of the worker
        elif cmd == 'reset':
            obs = [env.reset() for env in envs]
            ts[:] = 0
            remote.send(obs)

        # set the specified task for each of the environments of the worker
        elif cmd == 'set_task':
            for env in envs:
                env.set_task(data)
            remote.send(None)

        # close the remote and stop the worker
        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError


if __name__ == "__main__":
    from environment.migration_env import EnvironmentParameters
    from environment.migration_env import MigrationEnv
    from environment.batch_migration_env import BatchMigrationEnv
    import numpy as np

    env_default_parameters = EnvironmentParameters(num_traces=10,
                                                   num_base_station=63, optical_fiber_trans_rate=60.0,
                                                   server_poisson_rate=20, client_poisson_rate=4,
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
                                                   is_full_observation=True,
                                                   is_full_action=True)

    env = BatchMigrationEnv(env_default_parameters)
    parallel_env = ParallelEnvExecutor(env, process_core=6, envs_per_core=2, max_path_length=100, is_batch_env=True)

    obs = parallel_env.reset()
    print(np.array(obs).shape)

    system_info = parallel_env.current_system_state()

    print("system infomation: ", np.array(system_info).shape)

    obs, rewards, dones, env_infos = parallel_env.step(actions = np.ones(shape=(120,), dtype=np.int32))
    #
    print("obs shape: ", np.array(obs).shape)
    print("rewards shape: ", np.array(rewards).shape)
    print("dones lenght", len(dones))
    print("env_infos ", env_infos)
