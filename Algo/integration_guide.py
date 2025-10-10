# ===================================
# Migrated from TensorFlow 1.x to 2.x
# Original file: D:\codetest\rlpp_ori\Algo\integration_guide.py
# Migration may not be complete. 
# Please review TODO comments.
# ===================================

"""
将LSTM-AC算法集成到你的rlpp_ori项目中
这个文件展示了如何修改你的run_with_dynamic_obstacles.py
"""

import numpy as np
import torch
from LSTM_AC_Policy import LSTMAC_Agent


class LSTMACPathPlanner:
    """
    LSTM-AC路径规划器包装类
    可以直接集成到你的现有代码中
    """

    def __init__(self, map_config, agent_config=None):
        """
        Args:
            map_config: 地图配置字典
            agent_config: 智能体配置字典
        """
        self.map_config = map_config

        # 默认配置
        if agent_config is None:
            agent_config = {
                'state_dim': 7,
                'obstacle_dim': 5,
                'action_dim': 9,
                'lstm_hidden_dim': 128,
                'mlp_hidden_dims': [128, 64],
                'lr': 3e-4,
                'gamma': 0.99,
                'device': 'cuda'
            }

        self.agent_config = agent_config

        # 创建智能体
        self.agent = LSTMAC_Agent(**agent_config)

        # LSTM隐藏状态
        self.hidden_state = None

        # 路径记录
        self.path = []

    def reset(self):
        """重置规划器"""
        self.hidden_state = None
        self.path = []

    def process_state(self, robot_pos, robot_vel, goal_pos, obstacles):
        """
        处理原始状态信息,转换为网络输入格式

        Args:
            robot_pos: [x, y] 机器人位置
            robot_vel: [vx, vy] 机器人速度
            goal_pos: [x, y] 目标位置
            obstacles: list of dict, 每个障碍物包含
                       {'pos': [x, y], 'vel': [vx, vy], 'radius': r}

        Returns:
            robot_state: numpy array
            obstacles_state: numpy array
        """
        # 计算到目标的距离
        distance_to_goal = np.linalg.norm(
            np.array(goal_pos) - np.array(robot_pos)
        )

        # 机器人状态
        robot_state = np.array([
            robot_pos[0], robot_pos[1],
            robot_vel[0], robot_vel[1],
            goal_pos[0], goal_pos[1],
            distance_to_goal
        ], dtype=np.float32)

        # 障碍物状态(相对位置)
        obstacles_state = []
        for obs in obstacles:
            rel_pos = np.array(obs['pos']) - np.array(robot_pos)
            obs_state = np.array([
                rel_pos[0], rel_pos[1],
                obs['vel'][0], obs['vel'][1],
                obs['radius']
            ], dtype=np.float32)
            obstacles_state.append(obs_state)

        if len(obstacles_state) == 0:
            # 如果没有障碍物,创建一个零向量
            obstacles_state = np.zeros((1, 5), dtype=np.float32)
        else:
            obstacles_state = np.array(obstacles_state, dtype=np.float32)

        return robot_state, obstacles_state

    def plan_step(self, robot_pos, robot_vel, goal_pos, obstacles,
                  deterministic=False):
        """
        规划下一步动作

        Args:
            robot_pos: [x, y]
            robot_vel: [vx, vy]
            goal_pos: [x, y]
            obstacles: list of dict
            deterministic: 是否使用确定性策略

        Returns:
            action: int, 动作索引
            action_vector: [dx, dy], 动作向量
        """
        # 处理状态
        robot_state, obstacles_state = self.process_state(
            robot_pos, robot_vel, goal_pos, obstacles
        )

        # 选择动作
        action, self.hidden_state = self.agent.select_action(
            robot_state, obstacles_state,
            self.hidden_state, deterministic
        )

        # 动作映射为向量
        actions_map = {
            0: [1, 0],  # 右
            1: [1, 1],  # 右上
            2: [0, 1],  # 上
            3: [-1, 1],  # 左上
            4: [-1, 0],  # 左
            5: [-1, -1],  # 左下
            6: [0, -1],  # 下
            7: [1, -1],  # 右下
            8: [0, 0]  # 停止
        }

        action_vector = actions_map.get(action, [0, 0])

        # 记录路径
        self.path.append(robot_pos)

        return action, action_vector

    def load_model(self, model_path):
        """加载预训练模型"""
        self.agent.load(model_path)
        print(f"已加载模型: {model_path}")

    def save_model(self, model_path):
        """保存模型"""
        self.agent.save(model_path)
        print(f"已保存模型: {model_path}")

    def get_path(self):
        """获取规划的路径"""
        return self.path


# ============================================================
# 示例：如何在你的run_with_dynamic_obstacles.py中使用
# ============================================================

def example_integration():
    """
    这是一个示例,展示如何将LSTM-AC集成到你的代码中
    """

    # 1. 创建路径规划器
    map_config = {
        'size': 20,
        'resolution': 0.1
    }

    planner = LSTMACPathPlanner(map_config)

    # 2. 如果有预训练模型,加载它
    # planner.load_model('models/lstm_ac_episode_1000.pth')

    # 3. 在你的主循环中使用
    # 假设你有以下变量:
    robot_pos = [1.0, 1.0]
    robot_vel = [0.0, 0.0]
    goal_pos = [18.0, 18.0]

    # 动态障碍物列表
    obstacles = [
        {'pos': [5.0, 5.0], 'vel': [0.2, 0.1], 'radius': 0.5},
        {'pos': [10.0, 8.0], 'vel': [-0.1, 0.2], 'radius': 0.6},
        {'pos': [15.0, 12.0], 'vel': [0.15, -0.15], 'radius': 0.4}
    ]

    # 4. 规划路径
    max_steps = 200
    for step in range(max_steps):
        # 获取动作
        action, action_vector = planner.plan_step(
            robot_pos, robot_vel, goal_pos, obstacles,
            deterministic=True  # 测试时使用确定性策略
        )

        # 更新机器人位置
        robot_vel = [action_vector[0] * 0.5, action_vector[1] * 0.5]
        robot_pos[0] += robot_vel[0]
        robot_pos[1] += robot_vel[1]

        # 更新障碍物位置
        for obs in obstacles:
            obs['pos'][0] += obs['vel'][0]
            obs['pos'][1] += obs['vel'][1]

        # 检查是否到达目标
        distance_to_goal = np.linalg.norm(
            np.array(goal_pos) - np.array(robot_pos)
        )
        if distance_to_goal < 1.0:
            print(f"成功到达目标! 步数: {step}")
            break

        # 碰撞检测
        collision = False
        for obs in obstacles:
            distance = np.linalg.norm(
                np.array(obs['pos']) - np.array(robot_pos)
            )
            if distance < obs['radius'] + 0.3:
                print(f"碰撞! 步数: {step}")
                collision = True
                break

        if collision:
            break

    # 5. 获取完整路径
    path = planner.get_path()
    print(f"路径长度: {len(path)}")

    return path


# ============================================================
# 完整的集成示例(修改你的run_with_dynamic_obstacles.py)
# ============================================================

class ModifiedDynamicObstacleRunner:
    """
    这是一个修改版本的示例,展示如何替换你原有的路径规划算法
    """

    def __init__(self, config):
        self.config = config

        # 创建LSTM-AC规划器
        agent_config = {
            'state_dim': 7,
            'obstacle_dim': 5,
            'action_dim': 9,
            'lstm_hidden_dim': 128,
            'mlp_hidden_dims': [128, 64],
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

        self.planner = LSTMACPathPlanner(
            map_config=config['map'],
            agent_config=agent_config
        )

        # 如果有预训练模型
        if 'model_path' in config and config['model_path']:
            self.planner.load_model(config['model_path'])

    def run_episode(self, start_pos, goal_pos, dynamic_obstacles,
                    max_steps=200, visualize=False):
        """
        运行一个episode

        Args:
            start_pos: [x, y] 起始位置
            goal_pos: [x, y] 目标位置
            dynamic_obstacles: list of dict, 动态障碍物信息
            max_steps: 最大步数
            visualize: 是否可视化

        Returns:
            success: bool, 是否成功
            path: list, 路径
            statistics: dict, 统计信息
        """
        # 重置规划器
        self.planner.reset()

        # 初始化
        robot_pos = list(start_pos)
        robot_vel = [0.0, 0.0]

        statistics = {
            'steps': 0,
            'success': False,
            'collision': False,
            'timeout': False,
            'final_distance': 0.0
        }

        for step in range(max_steps):
            # 规划下一步
            action, action_vector = self.planner.plan_step(
                robot_pos, robot_vel, goal_pos, dynamic_obstacles,
                deterministic=True
            )

            # 执行动作
            robot_vel = [action_vector[0] * 0.5, action_vector[1] * 0.5]
            robot_pos[0] += robot_vel[0]
            robot_pos[1] += robot_vel[1]

            # 更新障碍物
            for obs in dynamic_obstacles:
                obs['pos'][0] += obs['vel'][0]
                obs['pos'][1] += obs['vel'][1]

                # 边界反弹
                if obs['pos'][0] <= 0 or obs['pos'][0] >= self.config['map']['size']:
                    obs['vel'][0] *= -1
                if obs['pos'][1] <= 0 or obs['pos'][1] >= self.config['map']['size']:
                    obs['vel'][1] *= -1

            # 检查目标
            distance_to_goal = np.linalg.norm(
                np.array(goal_pos) - np.array(robot_pos)
            )

            if distance_to_goal < 1.0:
                statistics['success'] = True
                statistics['steps'] = step + 1
                statistics['final_distance'] = distance_to_goal
                break

            # 检查碰撞
            for obs in dynamic_obstacles:
                distance = np.linalg.norm(
                    np.array(obs['pos']) - np.array(robot_pos)
                )
                if distance < obs['radius'] + 0.3:
                    statistics['collision'] = True
                    statistics['steps'] = step + 1
                    statistics['final_distance'] = distance_to_goal
                    break

            if statistics['collision']:
                break

        # 超时
        if not statistics['success'] and not statistics['collision']:
            statistics['timeout'] = True
            statistics['steps'] = max_steps
            statistics['final_distance'] = distance_to_goal

        # 获取路径
        path = self.planner.get_path()

        return statistics['success'], path, statistics

    def run_multiple_episodes(self, num_episodes=100):
        """运行多个episodes并统计结果"""
        results = {
            'success_rate': 0,
            'collision_rate': 0,
            'timeout_rate': 0,
            'avg_steps': 0,
            'avg_final_distance': 0
        }

        success_count = 0
        collision_count = 0
        timeout_count = 0
        total_steps = 0
        total_distance = 0

        for episode in range(num_episodes):
            # 随机生成起点和终点
            start_pos = np.random.uniform(1, 5, 2)
            goal_pos = np.random.uniform(15, 19, 2)

            # 随机生成障碍物
            num_obstacles = np.random.randint(3, 8)
            dynamic_obstacles = []
            for _ in range(num_obstacles):
                obs = {
                    'pos': list(np.random.uniform(5, 15, 2)),
                    'vel': list(np.random.uniform(-0.5, 0.5, 2)),
                    'radius': np.random.uniform(0.3, 0.8)
                }
                dynamic_obstacles.append(obs)

            # 运行episode
            success, path, stats = self.run_episode(
                start_pos, goal_pos, dynamic_obstacles
            )

            if stats['success']:
                success_count += 1
            if stats['collision']:
                collision_count += 1
            if stats['timeout']:
                timeout_count += 1

            total_steps += stats['steps']
            total_distance += stats['final_distance']

            if (episode + 1) % 10 == 0:
                print(f"已完成 {episode + 1}/{num_episodes} episodes")

        results['success_rate'] = success_count / num_episodes
        results['collision_rate'] = collision_count / num_episodes
        results['timeout_rate'] = timeout_count / num_episodes
        results['avg_steps'] = total_steps / num_episodes
        results['avg_final_distance'] = total_distance / num_episodes

        return results


if __name__ == '__main__':
    # 简单示例
    print("=" * 60)
    print("LSTM-AC路径规划器集成示例")
    print("=" * 60)

    # 运行简单示例
    path = example_integration()

    print("\n" + "=" * 60)
    print("完整集成示例")
    print("=" * 60)

    # 配置
    config = {
        'map': {
            'size': 20,
            'resolution': 0.1
        },
        'model_path': None  # 如果有预训练模型,在这里指定路径
    }

    # 创建运行器
    runner = ModifiedDynamicObstacleRunner(config)

    # 运行测试
    print("\n开始测试...")
    results = runner.run_multiple_episodes(num_episodes=20)

    print("\n测试结果:")
    print(f"  成功率: {results['success_rate']:.2%}")
    print(f"  碰撞率: {results['collision_rate']:.2%}")
    print(f"  超时率: {results['timeout_rate']:.2%}")
    print(f"  平均步数: {results['avg_steps']:.2f}")
    print(f"  平均最终距离: {results['avg_final_distance']:.2f}")