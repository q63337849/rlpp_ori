"""
渐进式LSTM训练 - 从简单到困难
解决训练效果差的问题
独立脚本，不依赖train_lstm_rl_gpu.py
"""

import numpy as np
import tensorflow as tf
import sys
import os
import time
from collections import deque
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Env.flight_with_dynamic_obstacles import Flight, DynamicObstacle, Scenairo
from Algo.lstm_flight_integration import LSTMFlightWrapper
from Algo.lstm_model import LSTMRL, AttentionLSTMRL
import Scene
import options


class ProgressiveConfig:
    """渐进式训练配置"""
    def __init__(self):
        # 基础参数
        self.use_attention = False
        self.lstm_hidden_dim = 50
        self.history_length = 5
        self.max_steps = 300
        self.gamma = 0.99

        # 保存路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f'./models/lstm_progressive_{timestamp}'
        self.log_dir = f'./logs/lstm_progressive_{timestamp}'

        # 场景配置
        self.train_percentage = 0.3
        self.obstacle_type = options.O_circle

        # 训练阶段
        self.stages = [
            {
                'name': '阶段1：静态环境',
                'episodes': 500,
                'dynamic_type': 'none',
                'learning_rate': 0.001,
                'eval_frequency': 20,
                'min_success': 0.5,
                'extra_episodes': 120,
                'max_repeats': 3
            },
            {
                'name': '阶段2：简单动态',
                'episodes': 150,
                'dynamic_type': 'basic',
                'learning_rate': 0.0005,
                'eval_frequency': 20,
                'min_success': 0.3,
                'extra_episodes': 80,
                'max_repeats': 3
            },
            {
                'name': '阶段3：复杂动态',
                'episodes': 200,
                'dynamic_type': 'mixed',
                'learning_rate': 0.0003,
                'eval_frequency': 20,
                'min_success': 0.2,
                'extra_episodes': 80,
                'max_repeats': 4
            },
            {
                'name': '阶段4：混乱模式',
                'episodes': 200,
                'dynamic_type': 'chaotic',
                'learning_rate': 0.0001,
                'eval_frequency': 20,
                'min_success': 0.1,
                'extra_episodes': 100,
                'max_repeats': 4
            }
        ]


class SimpleAgent:
    """简化的LSTM智能体"""
    def __init__(self, config, action_dim):
        self.config = config
        self.action_dim = action_dim

        # 创建模型
        if config.use_attention:
            self.model = AttentionLSTMRL(
                state_dim=4, goal_dim=2, obstacle_state_dim=5,
                human_num=10, action_dim=action_dim,
                lstm_hidden_dim=config.lstm_hidden_dim
            )
        else:
            self.model = LSTMRL(
                state_dim=4, goal_dim=2, obstacle_state_dim=5,
                human_num=10, action_dim=action_dim,
                lstm_hidden_dim=config.lstm_hidden_dim, use_lstm=True
            )

        self.build_graph()

    def build_graph(self):
        """构建计算图"""
        tf.reset_default_graph()

        # 输入
        self.self_state_ph = tf.placeholder(tf.float32, [None, 4])
        self.goal_state_ph = tf.placeholder(tf.float32, [None, 2])
        self.obstacle_states_ph = tf.placeholder(tf.float32, [None, 10, 5])

        # 网络
        if self.config.use_attention:
            self.action_logits, _ = self.model.build_network(
                self.self_state_ph, self.goal_state_ph, self.obstacle_states_ph
            )
        else:
            self.action_logits = self.model.build_network(
                self.self_state_ph, self.goal_state_ph, self.obstacle_states_ph
            )

        self.action_probs = tf.nn.softmax(self.action_logits)
        self.value = self.model.build_value_network(
            self.self_state_ph, self.goal_state_ph, self.obstacle_states_ph
        )

        # 训练
        self.action_ph = tf.placeholder(tf.int32, [None])
        self.advantage_ph = tf.placeholder(tf.float32, [None])
        self.target_value_ph = tf.placeholder(tf.float32, [None])

        action_one_hot = tf.one_hot(self.action_ph, self.action_dim)
        action_log_prob = tf.reduce_sum(
            action_one_hot * tf.log(self.action_probs + 1e-10), axis=1
        )

        entropy = -tf.reduce_sum(self.action_probs * tf.log(self.action_probs + 1e-10), axis=1)

        self.actor_loss = -tf.reduce_mean(action_log_prob * self.advantage_ph + 0.01 * entropy)
        self.critic_loss = tf.reduce_mean(tf.square(self.value - self.target_value_ph))
        self.total_loss = self.actor_loss + 0.5 * self.critic_loss

        # 优化器（可变学习率）
        self.learning_rate_ph = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

        grads_and_vars = self.optimizer.compute_gradients(self.total_loss)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        if grads_and_vars:
            grads, vars_ = zip(*grads_and_vars)
            clipped_grads, _ = tf.clip_by_global_norm(grads, 5.0)
            self.train_op = self.optimizer.apply_gradients(zip(clipped_grads, vars_))
        else:
            self.train_op = tf.no_op()

        # Saver
        self.saver = tf.train.Saver(max_to_keep=10)

        # 会话
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def select_action(self, lstm_state, deterministic=False, epsilon=0.0):
        feed_dict = {
            self.self_state_ph: [lstm_state['self_state']],
            self.goal_state_ph: [lstm_state['goal_state']],
            self.obstacle_states_ph: [lstm_state['obstacle_states']]
        }

        action_probs = self.sess.run(self.action_probs, feed_dict=feed_dict)[0]

        if not deterministic and np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        elif deterministic:
            return np.argmax(action_probs)
        else:
            return np.random.choice(self.action_dim, p=action_probs)

    def get_value(self, lstm_state):
        feed_dict = {
            self.self_state_ph: [lstm_state['self_state']],
            self.goal_state_ph: [lstm_state['goal_state']],
            self.obstacle_states_ph: [lstm_state['obstacle_states']]
        }
        return self.sess.run(self.value, feed_dict=feed_dict)[0][0]

    def evaluate_values(self, states):
        if not states:
            return np.zeros(0, dtype=np.float32)

        self_states = np.array([s['self_state'] for s in states], dtype=np.float32)
        goal_states = np.array([s['goal_state'] for s in states], dtype=np.float32)
        obstacle_states = np.array([s['obstacle_states'] for s in states], dtype=np.float32)

        feed_dict = {
            self.self_state_ph: self_states,
            self.goal_state_ph: goal_states,
            self.obstacle_states_ph: obstacle_states
        }

        values = self.sess.run(self.value, feed_dict=feed_dict)
        return values[:, 0]

    def train_step(self, states, actions, advantages, target_values, learning_rate):
        self_states = np.array([s['self_state'] for s in states], dtype=np.float32)
        goal_states = np.array([s['goal_state'] for s in states], dtype=np.float32)
        obstacle_states = np.array([s['obstacle_states'] for s in states], dtype=np.float32)

        feed_dict = {
            self.self_state_ph: self_states,
            self.goal_state_ph: goal_states,
            self.obstacle_states_ph: obstacle_states,
            self.action_ph: np.array(actions, dtype=np.int32),
            self.advantage_ph: np.array(advantages, dtype=np.float32),
            self.target_value_ph: np.array(target_values, dtype=np.float32),
            self.learning_rate_ph: learning_rate
        }

        _, actor_loss, critic_loss = self.sess.run(
            [self.train_op, self.actor_loss, self.critic_loss],
            feed_dict=feed_dict
        )

        return actor_loss, critic_loss

    @staticmethod
    def clone_state(lstm_state):
        """复制LSTM状态，避免被环境后续修改"""
        return {
            'self_state': np.array(lstm_state['self_state'], dtype=np.float32, copy=True),
            'goal_state': np.array(lstm_state['goal_state'], dtype=np.float32, copy=True),
            'obstacle_states': np.array(lstm_state['obstacle_states'], dtype=np.float32, copy=True)
        }

    def td_update(self, state, action, reward, next_state, done, learning_rate, gamma):
        """基于单步TD误差的更新，更接近原AC训练流程"""
        cloned_state = self.clone_state(state)
        cloned_next_state = self.clone_state(next_state)

        current_value = self.get_value(cloned_state)
        next_value = 0.0 if done else self.get_value(cloned_next_state)

        target = reward + gamma * next_value
        advantage = target - current_value

        actor_loss, critic_loss = self.train_step(
            [cloned_state], [action], [advantage], [target], learning_rate
        )

        return actor_loss, critic_loss, advantage

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.saver.save(self.sess, filepath)

    def load(self, filepath):
        self.saver.restore(self.sess, filepath)


def create_dynamic_obstacles(dynamic_type='none'):
    """创建不同难度的动态障碍物"""
    if dynamic_type == 'none':
        return []
    elif dynamic_type == 'basic':
        return [
            DynamicObstacle([150, 500], 30, 1.5, None, [50, 50, 650, 650], 0.02),
            DynamicObstacle([550, 350], 25, 1.2, None, [50, 50, 650, 650], 0.02),
            DynamicObstacle([300, 100], 35, 1.0, None, [50, 50, 650, 650], 0.02)
        ]
    elif dynamic_type == 'mixed':
        return [
            DynamicObstacle([100, 100], 20, 2.0, None, [50, 50, 650, 650], 0.03),
            DynamicObstacle([300, 400], 35, 1.5, None, [50, 50, 650, 650], 0.03),
            DynamicObstacle([550, 250], 25, 2.2, None, [50, 50, 650, 650], 0.04),
            DynamicObstacle([450, 550], 30, 1.8, None, [50, 50, 650, 650], 0.03),
            DynamicObstacle([200, 500], 20, 2.5, None, [50, 50, 650, 650], 0.04)
        ]
    elif dynamic_type == 'chaotic':
        return [
            DynamicObstacle([200, 200], 20, 3.5, None, [50, 50, 650, 650], 0.08),
            DynamicObstacle([400, 400], 30, 2.8, None, [50, 50, 650, 650], 0.06),
            DynamicObstacle([500, 200], 35, 2.5, None, [50, 50, 650, 650], 0.05),
            DynamicObstacle([150, 550], 15, 4.0, None, [50, 50, 650, 650], 0.10),
            DynamicObstacle([600, 300], 30, 3.0, None, [50, 50, 650, 650], 0.07)
        ]
    return []


def load_scenes(config, dynamic_type):
    """加载场景"""
    if config.obstacle_type == options.O_circle:
        train_task = Scene.circle_train_task4x200
        test_task = Scene.circle_test_task4x100
    else:
        train_task = Scene.line_train_task4x200
        test_task = Scene.line_test_task4x100

    scene_loader = Scene.ScenarioLoader()
    original_train = scene_loader.load_scene(**train_task, percentage=config.train_percentage)
    original_test = scene_loader.load_scene(**test_task, percentage=0.15)

    train_scenes = []
    for orig in original_train:
        enhanced = Scenairo(
            init_pos=orig.init_pos, init_dir=orig.init_dir,
            goal_pos=orig.goal_pos, goal_dir=orig.goal_dir,
            circle_obstacles=orig.circle_obstacles,
            line_obstacles=getattr(orig, 'line_obstacles', None),
            dynamic_obstacles=create_dynamic_obstacles(dynamic_type)
        )
        train_scenes.append(enhanced)

    test_scenes = []
    for orig in original_test:
        enhanced = Scenairo(
            init_pos=orig.init_pos, init_dir=orig.init_dir,
            goal_pos=orig.goal_pos, goal_dir=orig.goal_dir,
            circle_obstacles=orig.circle_obstacles,
            line_obstacles=getattr(orig, 'line_obstacles', None),
            dynamic_obstacles=create_dynamic_obstacles(dynamic_type)
        )
        test_scenes.append(enhanced)

    return train_scenes, test_scenes


def run_episode(lstm_env, agent, scenario, max_steps, epsilon=0.0, deterministic=False):
    """运行一个episode并返回结果，用于评估或收集统计信息"""
    lstm_state = lstm_env.reset(scenario)

    total_reward = 0.0
    steps = 0

    for step in range(max_steps):
        action = agent.select_action(lstm_state, deterministic, epsilon)
        next_lstm_state, reward, done = lstm_env.step(action)

        total_reward += reward
        steps = step + 1

        if done:
            break
        lstm_state = next_lstm_state

    return total_reward, lstm_env.env.result, steps


def train_stage(config, stage_config, agent, lstm_env, global_episode):
    """训练一个阶段，必要时自动追加训练以达到目标成功率"""
    print(f"\n{'='*60}")
    print(f"🎯 {stage_config['name']}")
    print(f"{'='*60}")
    print(f"初始Episodes: {stage_config['episodes']}")
    print(f"动态障碍物: {stage_config['dynamic_type']}")
    print(f"学习率: {stage_config['learning_rate']}")

    # 加载场景
    train_scenes, test_scenes = load_scenes(config, stage_config['dynamic_type'])
    print(f"场景: {len(train_scenes)} 训练, {len(test_scenes)} 测试")
    print(f"动态障碍物数量: {len(train_scenes[0].dynamic_obstacles)}\n")

    min_success = stage_config.get('min_success', 0.3)
    extra_episodes = stage_config.get('extra_episodes', 100)
    max_repeats = stage_config.get('max_repeats', 2)
    eval_frequency = stage_config.get('eval_frequency', 20)
    base_episodes = stage_config['episodes']

    reward_window = deque(maxlen=10)
    result_window = deque(maxlen=10)
    actor_loss_window = deque(maxlen=50)
    critic_loss_window = deque(maxlen=50)
    td_error_window = deque(maxlen=50)

    best_success = 0.0
    episodes_completed = 0
    attempt = 0

    while True:
        if attempt == 0:
            episodes_this_round = base_episodes
        else:
            episodes_this_round = extra_episodes
            print(f"🔁 追加训练第{attempt}轮：增加 {episodes_this_round} 个episodes，目标成功率 {min_success*100:.1f}%")

        for _ in range(episodes_this_round):
            episodes_completed += 1
            global_episode += 1
            scenario = np.random.choice(train_scenes)

            planned_budget = base_episodes + max(0, attempt) * extra_episodes
            decay_steps = max(1, int(planned_budget * 0.7))
            decay_progress = min(episodes_completed / decay_steps, 1.0)
            epsilon = max(0.05, 1.0 - decay_progress)

            lstm_state = lstm_env.reset(scenario)
            episode_reward = 0.0
            actor_losses = []
            critic_losses = []
            td_errors = []

            for _ in range(config.max_steps):
                action = agent.select_action(lstm_state, deterministic=False, epsilon=epsilon)
                next_state, reward, done = lstm_env.step(action)

                actor_loss, critic_loss, td_error = agent.td_update(
                    lstm_state, action, reward, next_state, done,
                    stage_config['learning_rate'], config.gamma
                )

                episode_reward += reward
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                td_errors.append(td_error)

                if done:
                    break

                lstm_state = next_state

            result = lstm_env.env.result

            reward_window.append(episode_reward)
            result_window.append(result)

            if actor_losses:
                actor_loss_window.append(np.mean(actor_losses))
                critic_loss_window.append(np.mean(critic_losses))
                td_error_window.append(np.mean(td_errors))

            if episodes_completed % 10 == 0:
                recent_rewards = list(reward_window)
                recent_results = list(result_window)
                success_count = sum(1 for r in recent_results if r == 3)

                print(f"Episode {episodes_completed} [全局:{global_episode}] (最近10局)")
                log_line = (
                    f"  奖励:{np.mean(recent_rewards):.3f} | 成功:{success_count}/{len(recent_results)} | "
                    f"ε:{epsilon:.3f}"
                )
                if actor_loss_window and critic_loss_window:
                    log_line += (
                        f" | ActorLoss:{np.mean(actor_loss_window):.4f}"
                        f" | CriticLoss:{np.mean(critic_loss_window):.4f}"
                    )
                if td_error_window:
                    log_line += f" | TD:{np.mean(td_error_window):.4f}"
                print(log_line)

            if episodes_completed % eval_frequency == 0:
                eval_results = []
                for eval_ep in range(min(20, len(test_scenes))):
                    _, eval_result, _ = run_episode(
                        lstm_env, agent, test_scenes[eval_ep],
                        config.max_steps, deterministic=True
                    )
                    eval_results.append(eval_result)

                success_count = sum(1 for r in eval_results if r == 3)
                success_rate = success_count / len(eval_results)

                print(f"\n  📊 评估: {success_count}/{len(eval_results)} = {success_rate*100:.1f}%")

                if success_rate > best_success:
                    best_success = success_rate
                    print(f"  🏆 新最佳: {best_success*100:.1f}%\n")

        if best_success >= min_success:
            break

        attempt += 1
        if attempt > max_repeats:
            print(f"⚠️ 达到最大追加次数，但最佳成功率仍为 {best_success*100:.1f}%，未达到目标 {min_success*100:.1f}%")
            break

    print(f"\n✓ {stage_config['name']} 完成! 最佳: {best_success*100:.1f}%\n")
    return global_episode, best_success


def main():
    """主函数"""
    config = ProgressiveConfig()

    # 快速模式
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        for stage in config.stages:
            stage['episodes'] = stage['episodes'] // 2
        print("✓ 快速模式：episodes减半")

    print("\n🎓 渐进式LSTM训练")
    print("="*60)

    # 创建环境和智能体
    base_env = Flight()
    lstm_env = LSTMFlightWrapper(base_env, history_length=config.history_length)
    agent = SimpleAgent(config, base_env.action.n_actions)

    print(f"✓ 环境创建完成 (动作维度: {base_env.action.n_actions})")
    print(f"✓ 模型保存路径: {config.save_dir}")
    print("="*60)

    start_time = time.time()
    global_episode = 0
    stage_results = []

    # 逐阶段训练
    for stage_idx, stage_config in enumerate(config.stages):
        global_episode, best_success = train_stage(
            config, stage_config, agent, lstm_env, global_episode
        )
        stage_results.append({
            'stage': stage_config['name'],
            'success': best_success
        })

        agent.save(f"{config.save_dir}/stage{stage_idx+1}_model.ckpt")

        if stage_idx < len(config.stages) - 1 and best_success < 0.3:
            print(f"⚠️ 警告: 当前阶段成功率({best_success*100:.1f}%)较低，可能影响后续阶段")

    # 完成
    elapsed = time.time() - start_time
    print(f"\n{'🎉'*30}")
    print(f"训练完成! 用时: {elapsed/60:.1f} 分钟")
    print(f"\n各阶段成功率:")
    for r in stage_results:
        print(f"  {r['stage']}: {r['success']*100:.1f}%")
    print(f"{'🎉'*30}\n")

    agent.save(f"{config.save_dir}/final_model.ckpt")


if __name__ == "__main__":
    main()


