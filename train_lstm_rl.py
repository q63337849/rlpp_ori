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
                'eval_frequency': 20
            },
            {
                'name': '阶段2：简单动态',
                'episodes': 150,
                'dynamic_type': 'basic',
                'learning_rate': 0.0005,
                'eval_frequency': 20
            },
            {
                'name': '阶段3：复杂动态',
                'episodes': 200,
                'dynamic_type': 'mixed',
                'learning_rate': 0.0003,
                'eval_frequency': 20
            },
            {
                'name': '阶段4：混乱模式',
                'episodes': 200,
                'dynamic_type': 'chaotic',
                'learning_rate': 0.0001,
                'eval_frequency': 20
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
        self.train_op = self.optimizer.minimize(self.total_loss)

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

    def train_step(self, states, actions, advantages, target_values, learning_rate):
        self_states = np.array([s['self_state'] for s in states])
        goal_states = np.array([s['goal_state'] for s in states])
        obstacle_states = np.array([s['obstacle_states'] for s in states])

        feed_dict = {
            self.self_state_ph: self_states,
            self.goal_state_ph: goal_states,
            self.obstacle_states_ph: obstacle_states,
            self.action_ph: actions,
            self.advantage_ph: advantages,
            self.target_value_ph: target_values,
            self.learning_rate_ph: learning_rate
        }

        _, actor_loss, critic_loss = self.sess.run(
            [self.train_op, self.actor_loss, self.critic_loss],
            feed_dict=feed_dict
        )

        return actor_loss, critic_loss

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
    """运行一个episode"""
    lstm_state = lstm_env.reset(scenario)
    states, actions, rewards = [], [], []
    next_states, dones = [], []

    for step in range(max_steps):
        action = agent.select_action(lstm_state, deterministic, epsilon)
        next_lstm_state, reward, done = lstm_env.step(action)

        states.append(lstm_state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_lstm_state)
        dones.append(done)

        if done:
            break
        lstm_state = next_lstm_state

    return states, actions, rewards, next_states, dones, lstm_env.env.result, step + 1


def train_stage(config, stage_config, agent, lstm_env, global_episode):
    """训练一个阶段"""
    print(f"\n{'='*60}")
    print(f"🎯 {stage_config['name']}")
    print(f"{'='*60}")
    print(f"Episodes: {stage_config['episodes']}")
    print(f"动态障碍物: {stage_config['dynamic_type']}")
    print(f"学习率: {stage_config['learning_rate']}")

    # 加载场景
    train_scenes, test_scenes = load_scenes(config, stage_config['dynamic_type'])
    print(f"场景: {len(train_scenes)} 训练, {len(test_scenes)} 测试")
    print(f"动态障碍物数量: {len(train_scenes[0].dynamic_obstacles)}\n")

    episode_rewards = []
    episode_results = []
    best_success = 0.0

    for episode in range(stage_config['episodes']):
        global_episode += 1
        scenario = np.random.choice(train_scenes)
        epsilon = max(0.05, 1.0 - episode / (stage_config['episodes'] * 0.5))

        # 运行episode
        states, actions, rewards, next_states, dones, result, num_steps = run_episode(
            lstm_env, agent, scenario, config.max_steps, epsilon
        )

        # 计算回报（Generalized Advantage Estimation）
        values = np.array([agent.get_value(s) for s in states])
        next_values = np.array([agent.get_value(ns) for ns in next_states])

        # 终止状态的bootstrap应为0
        next_values = next_values * (1 - np.array(dones, dtype=np.float32))

        advantages = []
        gae = 0.0
        lambda_ = 0.95
        gamma = config.gamma

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] - values[t]
            gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages)
        returns = advantages + values

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 训练
        actor_loss, critic_loss = agent.train_step(
            states, actions, advantages, returns, stage_config['learning_rate']
        )

        episode_rewards.append(np.sum(rewards))
        episode_results.append(result)

        # 打印
        if (episode + 1) % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            recent_results = episode_results[-10:]
            success_count = sum(1 for r in recent_results if r == 3)

            print(f"Episode {episode+1}/{stage_config['episodes']} [全局:{global_episode}]")
            print(f"  奖励:{np.mean(recent_rewards):.3f} | 成功:{success_count}/10 | ε:{epsilon:.3f}")

        # 评估
        if (episode + 1) % stage_config['eval_frequency'] == 0:
            eval_results = []
            for eval_ep in range(min(20, len(test_scenes))):
                _, _, _, _, _, result, _ = run_episode(
                    lstm_env, agent, test_scenes[eval_ep],
                    config.max_steps, deterministic=True
                )
                eval_results.append(result)

            success_count = sum(1 for r in eval_results if r == 3)
            success_rate = success_count / len(eval_results)

            print(f"\n  📊 评估: {success_count}/{len(eval_results)} = {success_rate*100:.1f}%")

            if success_rate > best_success:
                best_success = success_rate
                print(f"  🏆 新最佳: {best_success*100:.1f}%\n")

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


