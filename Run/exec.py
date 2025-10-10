# ===================================
# Migrated from TensorFlow 1.x to 2.x
# Original file: D:\codetest\rlpp_ori\Run\exec.py
# Migration may not be complete. 
# Please review TODO comments.
# ===================================

from Run.util import Record
import numpy as np
from tqdm import tqdm
import time


class Exec:
    def __init__(self, env, agent, scenes, paths, train=True, episodes=1, max_n_step=200):
        self.env = env
        self.agent = agent
        self.scenes = scenes
        self.paths = paths
        self.train = train
        self.episodes = episodes
        self.max_n_step = max_n_step
        self.outcomes = None

    def __call__(self, *args, **kwargs):
        """执行训练或预测"""
        if self.train:
            self._run_train()
        else:
            self._run()

    def _run_train(self):
        """训练模式 - 带进度条"""
        print(f'\ntrain_ nth_rounds: {self.episodes}')

        total_tasks = self.episodes * len(self.scenes)
        outcome_names = ['timeout', 'failure', 'border', 'success']

        # 总进度条
        with tqdm(total=total_tasks, desc="Overall Training", position=0, leave=True,
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as overall_pbar:

            for episode in range(1, self.episodes + 1):
                print(f'\n{"=" * 70}')
                print(f'  当前训练轮次: {episode}/{self.episodes}')
                print(f'{"=" * 70}')

                # 初始化episode统计
                episode_outcomes = {'success': 0, 'failure': 0, 'timeout': 0, 'border': 0}
                episode_rewards = []
                episode_steps_list = []
                episode_start_time = time.time()

                # Episode进度条
                with tqdm(total=len(self.scenes), desc=f"  Episode {episode}", position=1, leave=False,
                          bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as episode_pbar:

                    for task_id, scenario in enumerate(self.scenes):
                        # 计算总体进度
                        total_progress = ((episode - 1) * len(self.scenes) + task_id) / total_tasks

                        # 训练一个场景
                        outcome, episode_reward, episode_steps = self._train(scenario, total_progress)

                        # 统计结果
                        episode_outcomes[outcome_names[outcome]] += 1
                        episode_rewards.append(episode_reward)
                        episode_steps_list.append(episode_steps)

                        # 计算当前成功率
                        total_finished = sum(episode_outcomes.values())
                        success_rate = episode_outcomes['success'] / total_finished * 100 if total_finished > 0 else 0

                        # 更新进度条 - 显示更详细信息
                        episode_pbar.set_postfix({
                            'Succ': f"{episode_outcomes['success']:>3}/{total_finished}",
                            'Rate': f"{success_rate:>5.1f}%",
                            'Fail': f"{episode_outcomes['failure']:>3}",
                            'TO': f"{episode_outcomes['timeout']:>2}",
                            'AvgR': f"{np.mean(episode_rewards):>6.2f}",
                            'Steps': f"{episode_steps:>3}"
                        }, refresh=True)

                        episode_pbar.update(1)
                        overall_pbar.update(1)

                        # 更新总体进度条
                        overall_pbar.set_postfix({
                            'Episode': f"{episode}/{self.episodes}",
                            'Success': f"{episode_outcomes['success']}/{total_finished}",
                            'Rate': f"{success_rate:.1f}%"
                        }, refresh=True)

                # Episode结束统计
                episode_time = time.time() - episode_start_time
                total_finished = sum(episode_outcomes.values())

                print(f'\n{"=" * 70}')
                print(f'  Episode {episode} 统计结果:')
                print(f'  {"─" * 68}')
                print(f'    ⏱  耗时: {episode_time:.2f}s ({episode_time / 60:.1f}min)')
                print(
                    f'    ✓  成功: {episode_outcomes["success"]:>4}/{total_finished} ({episode_outcomes["success"] / total_finished * 100:>5.2f}%)')
                print(f'    ✗  失败: {episode_outcomes["failure"]:>4}')
                print(f'    ⌛ 超时: {episode_outcomes["timeout"]:>4}')
                print(f'    ⊗  边界: {episode_outcomes["border"]:>4}')
                print(f'    📊 平均奖励: {np.mean(episode_rewards):>8.2f}')
                print(f'    🚶 平均步数: {np.mean(episode_steps_list):>8.1f}')
                print(f'{"=" * 70}\n')

        print('\n✅ 训练完成!\n')

    def _train(self, scenario, progress):
        """
        训练一个场景

        Returns:
            outcome: 结果类型
            episode_reward: 总奖励
            episode_steps: 总步数
        """
        return self.train_on_policy(scenario, progress)

    def train_on_policy(self, scenario, progress):
        """On-policy训练"""
        observation = self.env.reset(scenario)  # ← 修改这里
        self.agent.reset()

        episode_reward = 0
        episode_steps = 0

        for step in range(self.max_n_step):
            action = self.agent.sample(observation)
            observation_, reward, done = self.env.step(action)

            c_loss, a_loss = self.agent.learn(observation, action, reward, observation_, done)

            episode_reward += reward
            episode_steps += 1
            observation = observation_

            if done:
                break
        outcome = getattr(self.env, 'result', 0)
        return outcome, episode_reward, episode_steps

    def _run(self):
        """预测/测试模式"""
        print(f'\n{"=" * 70}')
        print(f'  测试预测模式 - 轮次: {self.episodes}')
        print(f'{"=" * 70}\n')

        self.outcomes = np.zeros(self.env.n_outcomes, np.int64)

        total_tasks = self.episodes * len(self.scenes)
        outcome_names = ['timeout', 'failure', 'border', 'success']

        # 测试进度条
        with tqdm(total=total_tasks, desc="Testing Progress",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:

            for episode in range(1, self.episodes + 1):
                print(f'\n  测试轮次: {episode}/{self.episodes}')

                episode_outcomes = {'success': 0, 'failure': 0, 'timeout': 0, 'border': 0}

                for task_id, scenario in enumerate(self.scenes):
                    outcome = self.predict(scenario)
                    self.outcomes[outcome] += 1
                    episode_outcomes[outcome_names[outcome]] += 1

                    # 更新进度条
                    total_finished = sum(episode_outcomes.values())
                    success_rate = episode_outcomes['success'] / total_finished * 100 if total_finished > 0 else 0

                    pbar.set_postfix({
                        'Ep': f"{episode}/{self.episodes}",
                        'Rate': f"{success_rate:.1f}%",
                        'Succ': f"{episode_outcomes['success']:>3}",
                        'Fail': f"{episode_outcomes['failure']:>3}"
                    }, refresh=True)
                    pbar.update(1)

                print(f'    Episode {episode} 统计: Success={episode_outcomes["success"]}, '
                      f'Failure={episode_outcomes["failure"]}, Timeout={episode_outcomes["timeout"]}')

        print(f'\n{"=" * 70}')
        print(f'  最终测试结果:')
        print(f'  {"─" * 68}')
        print(f'    Timeout : {self.outcomes[0]:>4}')
        print(f'    Failure : {self.outcomes[1]:>4}')
        print(f'    Border  : {self.outcomes[2]:>4}')
        print(f'    Success : {self.outcomes[3]:>4}')
        total = np.sum(self.outcomes)
        print(f'  {"─" * 68}')
        print(f'    总成功率: {self.outcomes[3] / total * 100:.2f}%')
        print(f'{"=" * 70}\n')

    def predict(self, scenario):
        """预测一个场景"""
        observation = self.env.reset(scenario)

        for step in range(self.max_n_step):
            action = self.agent.sample(observation)
            observation, reward, done = self.env.step(action)

            if done:
                break
        outcome = getattr(self.env, 'result', 0)
        return outcome


class Train:
    def __init__(self, draw=False, episodes=200, batch_size=64, max_n_step=100):
        self.max_n_step = max_n_step
        self.draw = draw
        self.episodes = episodes
        self.batch_size = batch_size

    def __call__(self, env, agent, scenes, paths):
        self.env = env
        self.agent = agent
        self.scenes = scenes
        self.paths = paths(paths.train)
        self._train = self.train_off_policy if self.agent.off_policy else self.train_on_policy

        total_task_size = self.episodes * len(self.scenes)
        self.outcomes = np.zeros(self.env.n_outcomes, np.int64)
        time_start = time.time()
        self.current_step = 0
        for episode in range(self.episodes):
            print('current episode: ', episode+1)
            for task_id, scenario in enumerate(self.scenes):
                self._train(scenario, (episode * len(self.scenes) + task_id) / total_task_size)
        run_time = time.time() - time_start
        with open(self.paths.record_file, 'a') as fp:
            s = "date: {}\n" \
                "time: {}\n" \
                "step: {}\n" \
                "n_scenes: {}\n" \
                "episode: {}\n" \
                "timeout, failure, border, success: {}"
            s = s.format(time.ctime(), run_time, self.current_step, len(self.scenes), self.episodes, self.outcomes)
            fp.write(s)
            print(s)
        return self

    def train_off_policy(self, scenario, task_percentage):
        observation = self.env.reset(scenario)
        for i in range(self.max_n_step):
            action = self.agent.sample(observation, task_percentage=task_percentage)
            # print("goal_dis: {}, goal_dir: {}".format(self.env.goal_dis, self.env.goal_dir))
            observation_, reward, done = self.env.step(action)
            # print("out: {}, reward: {}".format(out, reward))
            if self.draw:
                self.env.render(sleep_time=0.001, show_arrow=True, show_scope=False)
            self.agent.store_exp(observation, action, reward, observation_, done)
            if self.current_step > self.batch_size and self.current_step % 5 == 0:
                c_loss, a_loss = self.agent.learn(self.batch_size)
                # if self.current_step % 100 == 0:
                #     print('c_loss: {0:10f}, a_loss: {1:10f}'.format(c_loss, a_loss))
            observation = observation_
            self.current_step += 1
            if done:
                break
        self.outcomes[self.env.result] += 1

    def train_on_policy(self, scenario, task_percentage):
        observation = self.env.reset(scenario)
        self.agent.reset()
        for i in range(self.max_n_step):
            # print(observation)
            action = self.agent.sample(observation)
            observation_, reward, done = self.env.step(action)
            if self.draw:
                self.env.render(sleep_time=0.01, show_arrow=True, show_scope=True)
            c_loss, a_loss = self.agent.learn(observation, action, reward, observation_, done)
            observation = observation_
            self.current_step += 1
            if done:
                break
        self.outcomes[self.env.result] += 1


class Predict:
    def __init__(self, draw=True, draw_rate=0.2, test_predict=True, pick=False, debug=False, max_n_step=100, save=False):
        self.save = save
        self.draw = draw
        self.draw_rate = draw_rate  # 前百分之
        self.max_n_step = max_n_step
        self.pick = pick
        self.debug = debug
        self.test_predict = test_predict
        self._run = self._pick_task if pick else self.predict

    def __call__(self, env, agent, scenes, paths):
        self.env = env
        self.agent = agent
        self.scenes = scenes
        self.paths = paths(paths.test_predict if self.test_predict else paths.train_predict)
        self._run()
        return self

    def predict(self):
        time_start = time.time()
        # 记录信息：步数，路程，角速率切换，切换次数，最大切换角速率，结果
        task_r = Record(self.paths.task_record_file, len(self.scenes), {
            'id': int, 'n_step': int, 'outcome': int,
            'total_reward': float, 'average_reward': float,
            'total_distance': float, 'total_time': float, 'linear_velocity': float,
            'total_change_yaw_rate': float, 'average_change_yaw_rate': float, 'max_change_yaw_rate': float,
            'total_yaw_rate': float, 'average_yaw_rate': float, 'max_yaw_rate': float,
            'total_turning_angle': float, 'average_turning_angle': float, 'max_turning_angle': float})
        outcomes = np.zeros(self.env.n_outcomes, np.int64)
        for scenario in self.scenes:
            observation = self.env.reset(scenario)
            task_r.init()  # __enter__
            for i in range(self.max_n_step):
                # pos, direct = self.env.current_pos, self.env.current_dir
                action = self.agent.predict(observation)
                observation_, reward, done = self.env.step(action)

                task_r.total_reward += reward
                task_r.total_distance += self.env.delta_dis
                task_r.total_time += self.env.delta_time
                task_r.total_change_yaw_rate += abs(self.env.delta_yaw_rate)
                if abs(self.env.delta_yaw_rate) > abs(task_r.max_change_yaw_rate):
                    task_r.max_change_yaw_rate = self.env.delta_yaw_rate
                task_r.total_yaw_rate += abs(self.env.yaw_rate)
                if abs(self.env.yaw_rate) > abs(task_r.max_yaw_rate):
                    task_r.max_yaw_rate = self.env.yaw_rate
                task_r.total_turning_angle += abs(self.env.delta_dir)
                if abs(self.env.delta_dir) > abs(task_r.max_turning_angle):
                    task_r.max_turning_angle = self.env.delta_dir

                if self.draw and task_r.count < self.draw_rate * len(self.scenes):
                    self.env.render(sleep_time=0.01, show_arrow=False, show_scope=True, show_pos=True)
                observation = observation_
                if done:
                    task_r.n_step = i + 1
                    break
            else:
                task_r.n_step = self.max_n_step
            outcomes[self.env.result] += 1
            # task_r.total_turning_angle = math.degrees(task_r.total_turning_angle)
            # task_r.max_turning_angle = math.degrees(task_r.max_turning_angle)
            task_r.average_turning_angle = task_r.total_turning_angle / task_r.n_step  # change_times
            task_r.average_change_yaw_rate = task_r.total_change_yaw_rate / task_r.n_step  # change_times
            task_r.average_yaw_rate = task_r.total_yaw_rate / task_r.n_step  # change_times
            task_r.average_reward = task_r.total_reward / task_r.n_step
            task_r.outcome = self.env.result
            task_r.linear_velocity = self.env.action.linear_velocity
            task_r.inc()  # 必须放在最后

        run_time = time.time() - time_start
        s = "date: {}\n" \
            "time: {}\n" \
            "n_scenes: {}\n" \
            "timeout, failure, border, success: {}\n".format(time.ctime(), run_time, len(self.scenes), outcomes)
        print(s)
        if self.save:
            with open(self.paths.record_file, 'a') as fp:
                fp.write(s)
            task_r.save()
            # setattr(self, 'task_r', task_r)  # 用于数据后续处理
        # print('predict end')

    def _pick_task(self):
        for task_id in range(100):
            if task_id < 0 or task_id >= len(self.scenes):
                break
            scenario = self.scenes[task_id]
            observation = self.env.reset(scenario)
            for i in range(self.max_n_step):
                if self.debug:
                    print("step: {}".format(i))
                    print("obs_dis: {}".format(self.env.obs_distances))
                    print("goal_dis: {}, goal_dir: {}".format(self.env.goal_dis, self.env.goal_dir))
                action = self.agent.predict(observation)
                observation_, reward, done = self.env.step(action)
                if self.debug:
                    print("reward: {}, out: {}".format(reward, action))
                    input('continue')

                self.env.render(sleep_time=0.01, show_arrow=False, show_scope=True, show_pos=True, circle_scope=False)

                observation = observation_
                if done:
                    if self.env.result == self.env.success:
                        print('success')
                        self.env.render_trajectory(self.env.residual_traj, show_pos=False)
                    break
            # input('screenshot: start')
            # time.sleep(2)
            self.env.canvas.screenshot(self.paths.picked_image_dir.join(str(task_id)))
            print('save screenshot: {}'.format(task_id))
