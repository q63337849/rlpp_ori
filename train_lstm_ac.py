# ===================================
# Migrated from TensorFlow 1.x to 2.x
# Original file: D:\codetest\rlpp_ori\train_lstm_ac.py
# Migration may not be complete. 
# Please review TODO comments.
# ===================================

"""
训练LSTM-AC算法进行动态环境路径规划
"""
import numpy as np
import matplotlib.pyplot as plt
from Algo.LSTM_AC_Policy import LSTMAC_Agent
import os


class DynamicEnvironment:
    """
    动态障碍物环境模拟器
    可以根据你的实际环境修改此类
    """

    def __init__(self, map_size=20, num_obstacles=5, max_steps=200):
        self.map_size = map_size
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        """重置环境"""
        # 机器人起始位置和目标
        self.robot_pos = np.array([1.0, 1.0])
        self.goal_pos = np.array([18.0, 18.0])
        self.robot_vel = np.array([0.0, 0.0])

        # 随机生成动态障碍物
        self.obstacles_pos = np.random.uniform(
            2, self.map_size - 2, (self.num_obstacles, 2)
        )
        self.obstacles_vel = np.random.uniform(
            -0.5, 0.5, (self.num_obstacles, 2)
        )
        self.obstacles_radius = np.random.uniform(
            0.3, 0.8, self.num_obstacles
        )

        self.current_step = 0

        return self._get_observation()

    def _get_observation(self):
        """获取观测"""
        # 机器人状态: [x, y, vx, vy, goal_x, goal_y, distance_to_goal]
        distance_to_goal = np.linalg.norm(self.goal_pos - self.robot_pos)
        robot_state = np.concatenate([
            self.robot_pos,
            self.robot_vel,
            self.goal_pos,
            [distance_to_goal]
        ])

        # 障碍物状态: [rel_x, rel_y, vx, vy, radius] 相对位置
        obstacles_state = []
        for i in range(self.num_obstacles):
            rel_pos = self.obstacles_pos[i] - self.robot_pos
            obs_state = np.concatenate([
                rel_pos,
                self.obstacles_vel[i],
                [self.obstacles_radius[i]]
            ])
            obstacles_state.append(obs_state)

        obstacles_state = np.array(obstacles_state)

        return robot_state, obstacles_state

    def step(self, action):
        """
        执行动作
        action: 0-7 表示8个方向,8表示停止
        """
        # 动作映射
        actions_map = {
            0: np.array([1, 0]),  # 右
            1: np.array([1, 1]),  # 右上
            2: np.array([0, 1]),  # 上
            3: np.array([-1, 1]),  # 左上
            4: np.array([-1, 0]),  # 左
            5: np.array([-1, -1]),  # 左下
            6: np.array([0, -1]),  # 下
            7: np.array([1, -1]),  # 右下
            8: np.array([0, 0])  # 停止
        }

        # 更新机器人速度和位置
        if action in actions_map:
            direction = actions_map[action]
            self.robot_vel = direction * 0.5  # 固定速度
            self.robot_pos += self.robot_vel

        # 边界检查
        self.robot_pos = np.clip(self.robot_pos, 0, self.map_size)

        # 更新障碍物位置
        self.obstacles_pos += self.obstacles_vel

        # 障碍物反弹
        for i in range(self.num_obstacles):
            if self.obstacles_pos[i, 0] <= 0 or self.obstacles_pos[i, 0] >= self.map_size:
                self.obstacles_vel[i, 0] *= -1
            if self.obstacles_pos[i, 1] <= 0 or self.obstacles_pos[i, 1] >= self.map_size:
                self.obstacles_vel[i, 1] *= -1

        self.obstacles_pos = np.clip(self.obstacles_pos, 0, self.map_size)

        # 计算奖励
        reward, done = self._compute_reward()

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return self._get_observation(), reward, done

    def _compute_reward(self):
        """计算奖励"""
        done = False

        # 到达目标
        distance_to_goal = np.linalg.norm(self.goal_pos - self.robot_pos)
        if distance_to_goal < 1.0:
            return 100.0, True

        # 碰撞检测
        for i in range(self.num_obstacles):
            distance = np.linalg.norm(self.obstacles_pos[i] - self.robot_pos)
            if distance < self.obstacles_radius[i] + 0.3:  # 机器人半径0.3
                return -50.0, True

        # 距离奖励(鼓励接近目标)
        reward = -distance_to_goal * 0.1

        # 碰撞风险惩罚
        min_distance = float('inf')
        for i in range(self.num_obstacles):
            distance = np.linalg.norm(self.obstacles_pos[i] - self.robot_pos)
            min_distance = min(min_distance, distance)

        if min_distance < 2.0:
            reward -= (2.0 - min_distance) * 2.0

        # 时间惩罚
        reward -= 0.1

        return reward, done


def train_lstm_ac(episodes=1000, save_interval=100, log_interval=10):
    """
    训练LSTM-AC算法
    """
    # 环境参数
    env = DynamicEnvironment(map_size=20, num_obstacles=5, max_steps=200)

    # 智能体参数
    state_dim = 7  # [x, y, vx, vy, goal_x, goal_y, distance]
    obstacle_dim = 5  # [rel_x, rel_y, vx, vy, radius]
    action_dim = 9  # 8个方向 + 停止

    agent = LSTMAC_Agent(
        state_dim=state_dim,
        obstacle_dim=obstacle_dim,
        action_dim=action_dim,
        lstm_hidden_dim=128,
        mlp_hidden_dims=[128, 64],
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device='cuda'
    )

    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # 训练记录
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    print("开始训练LSTM-AC算法...")
    print(f"状态维度: {state_dim}, 障碍物维度: {obstacle_dim}, 动作维度: {action_dim}")
    print("=" * 60)

    for episode in range(episodes):
        robot_state, obstacles_state = env.reset()
        hidden_state = None
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # 选择动作
            action, hidden_state = agent.select_action(
                robot_state, obstacles_state, hidden_state
            )

            # 执行动作
            (next_robot_state, next_obstacles_state), reward, done = env.step(action)

            # 存储转移
            agent.store_transition(reward, done)

            episode_reward += reward
            episode_length += 1

            # 更新状态
            robot_state = next_robot_state
            obstacles_state = next_obstacles_state

        # 更新策略
        loss_info = agent.update(
            next_robot_state, next_obstacles_state, hidden_state,
            epochs=10, batch_size=64
        )

        # 记录
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if episode_reward > 50:  # 成功到达目标
            success_count += 1

        # 日志
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_length = np.mean(episode_lengths[-log_interval:])
            success_rate = success_count / log_interval

            print(f"Episode {episode + 1}/{episodes}")
            print(f"  平均奖励: {avg_reward:.2f}")
            print(f"  平均步数: {avg_length:.2f}")
            print(f"  成功率: {success_rate:.2%}")
            if loss_info:
                print(f"  总损失: {loss_info['total_loss']:.4f}")
                print(f"  策略损失: {loss_info['policy_loss']:.4f}")
                print(f"  价值损失: {loss_info['value_loss']:.4f}")
                print(f"  熵: {loss_info['entropy']:.4f}")
            print("-" * 60)

            success_count = 0

        # 保存模型
        if (episode + 1) % save_interval == 0:
            model_path = f'models/lstm_ac_episode_{episode + 1}.pth'
            agent.save(model_path)
            print(f"模型已保存至: {model_path}")

    # 绘制训练曲线
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.title('Episode Lengths')
    plt.grid(True)

    # 计算滑动平均
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = [np.mean(episode_rewards[i:i + window])
                      for i in range(len(episode_rewards) - window)]
        plt.subplot(1, 3, 3)
        plt.plot(moving_avg)
        plt.xlabel('Episode')
        plt.ylabel('Moving Average Reward')
        plt.title(f'Moving Average (window={window})')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('logs/training_curves.png')
    plt.close()

    print("\n训练完成!")
    print(f"训练曲线已保存至: logs/training_curves.png")

    return agent


def test_lstm_ac(model_path, test_episodes=10, render=False):
    """
    测试训练好的LSTM-AC模型
    """
    # 环境参数
    env = DynamicEnvironment(map_size=20, num_obstacles=5, max_steps=200)

    # 智能体参数
    state_dim = 7
    obstacle_dim = 5
    action_dim = 9

    agent = LSTMAC_Agent(
        state_dim=state_dim,
        obstacle_dim=obstacle_dim,
        action_dim=action_dim,
        lstm_hidden_dim=128,
        mlp_hidden_dims=[128, 64],
        device='cuda'
    )

    # 加载模型
    agent.load(model_path)
    print(f"模型已加载: {model_path}")

    test_rewards = []
    test_lengths = []
    success_count = 0
    collision_count = 0

    print("\n开始测试...")
    print("=" * 60)

    for episode in range(test_episodes):
        robot_state, obstacles_state = env.reset()
        hidden_state = None
        episode_reward = 0
        episode_length = 0
        done = False

        trajectory = [env.robot_pos.copy()]

        while not done:
            # 选择动作(确定性)
            action, hidden_state = agent.select_action(
                robot_state, obstacles_state, hidden_state, deterministic=True
            )

            # 执行动作
            (next_robot_state, next_obstacles_state), reward, done = env.step(action)

            trajectory.append(env.robot_pos.copy())

            episode_reward += reward
            episode_length += 1

            robot_state = next_robot_state
            obstacles_state = next_obstacles_state

        test_rewards.append(episode_reward)
        test_lengths.append(episode_length)

        # 判断结果
        if episode_reward > 50:
            result = "成功"
            success_count += 1
        elif episode_reward < -40:
            result = "碰撞"
            collision_count += 1
        else:
            result = "超时"

        print(f"测试 {episode + 1}/{test_episodes} - {result}")
        print(f"  奖励: {episode_reward:.2f}, 步数: {episode_length}")

        # 可视化轨迹
        if render and episode < 3:  # 只显示前3个episode
            visualize_trajectory(env, trajectory, episode)

    print("\n" + "=" * 60)
    print("测试结果:")
    print(f"  平均奖励: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"  平均步数: {np.mean(test_lengths):.2f} ± {np.std(test_lengths):.2f}")
    print(f"  成功率: {success_count / test_episodes:.2%}")
    print(f"  碰撞率: {collision_count / test_episodes:.2%}")


def visualize_trajectory(env, trajectory, episode_id):
    """可视化机器人轨迹"""
    plt.figure(figsize=(10, 10))

    # 绘制地图边界
    plt.xlim(0, env.map_size)
    plt.ylim(0, env.map_size)

    # 绘制起点和终点
    plt.plot(trajectory[0][0], trajectory[0][1], 'go', markersize=15, label='起点')
    plt.plot(env.goal_pos[0], env.goal_pos[1], 'r*', markersize=20, label='目标')

    # 绘制障碍物
    for i in range(env.num_obstacles):
        circle = plt.Circle(
            env.obstacles_pos[i],
            env.obstacles_radius[i],
            color='gray',
            alpha=0.5
        )
        plt.gca().add_patch(circle)

    # 绘制轨迹
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='轨迹')
    plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'bo', markersize=10)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Episode {episode_id + 1} - 机器人轨迹')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.savefig(f'logs/trajectory_episode_{episode_id + 1}.png')
    plt.close()
    print(f"  轨迹图已保存至: logs/trajectory_episode_{episode_id + 1}.png")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='LSTM-AC动态环境路径规划')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='运行模式: train或test')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='训练episodes数量')
    parser.add_argument('--model_path', type=str, default='models/lstm_ac_episode_1000.pth',
                        help='模型路径(用于测试)')
    parser.add_argument('--test_episodes', type=int, default=10,
                        help='测试episodes数量')
    parser.add_argument('--render', action='store_true',
                        help='是否可视化测试结果')

    args = parser.parse_args()

    if args.mode == 'train':
        agent = train_lstm_ac(episodes=args.episodes)
    else:
        test_lstm_ac(args.model_path, args.test_episodes, args.render)