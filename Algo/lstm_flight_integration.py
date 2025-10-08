"""
将LSTM模型集成到现有的动态障碍物飞行环境中
"""

import numpy as np
import math
from collections import deque


class LSTMStatePreprocessor:
    """
    状态预处理器，将Flight环境的状态转换为LSTM模型需要的格式
    """
    def __init__(self, history_length=5):
        """
        history_length: 保存的历史状态数量
        """
        self.history_length = history_length
        self.state_history = deque(maxlen=history_length)
        
    def reset(self):
        """重置历史"""
        self.state_history.clear()
    
    def extract_dynamic_obstacle_states(self, env):
        """
        从环境中提取动态障碍物的状态
        返回: [num_obstacles, obstacle_state_dim]
        obstacle_state_dim包括: [relative_x, relative_y, vx, vy, radius]
        """
        if not env.dynamic_obstacles:
            return np.zeros((0, 5))
        
        obstacle_states = []
        current_pos = env.current_pos
        
        for obs in env.dynamic_obstacles:
            # 相对位置
            relative_pos = obs.pos - current_pos
            
            # 速度分量
            vx = obs.velocity * np.cos(obs.direction)
            vy = obs.velocity * np.sin(obs.direction)
            
            # 半径
            radius = obs.radius
            
            obstacle_state = [
                relative_pos[0] / 700.0,  # 归一化
                relative_pos[1] / 700.0,
                vx / 5.0,  # 假设最大速度为5
                vy / 5.0,
                radius / 50.0  # 假设最大半径为50
            ]
            
            obstacle_states.append(obstacle_state)
        
        return np.array(obstacle_states, dtype=np.float32)
    
    def extract_self_state(self, env):
        """
        提取自身状态
        返回: [self_state_dim]
        """
        # 归一化位置
        pos_x = env.current_pos[0] / 700.0
        pos_y = env.current_pos[1] / 700.0
        
        # 归一化方向 [-pi, pi] -> [-1, 1]
        direction = env.current_dir / math.pi
        
        # 速度（可以从action中推断）
        velocity = env.action.linear_velocity / 10.0  # 归一化
        
        return np.array([pos_x, pos_y, direction, velocity], dtype=np.float32)
    
    def extract_goal_state(self, env):
        """
        提取目标状态
        返回: [goal_state_dim]
        """
        # 目标的相对位置
        goal_vector = env.goal_pos - env.current_pos
        relative_distance = np.linalg.norm(goal_vector) / 1000.0  # 归一化
        
        # 目标方向（已在环境中计算）
        goal_direction = env.goal_dir / math.pi  # 归一化到[-1, 1]
        
        return np.array([relative_distance, goal_direction], dtype=np.float32)
    
    def process_state(self, env):
        """
        处理完整状态，返回LSTM模型需要的格式
        """
        self_state = self.extract_self_state(env)
        goal_state = self.extract_goal_state(env)
        obstacle_states = self.extract_dynamic_obstacle_states(env)
        
        # 添加到历史
        current_full_state = {
            'self': self_state,
            'goal': goal_state,
            'obstacles': obstacle_states
        }
        self.state_history.append(current_full_state)
        
        return current_full_state
    
    def get_lstm_input(self):
        """
        获取LSTM的输入格式
        如果历史不足，用零填充
        """
        if len(self.state_history) == 0:
            return None
        
        # 获取最新状态
        latest = self.state_history[-1]
        
        # 确保obstacle_states有固定数量的障碍物
        # 如果不足，用零填充；如果超过，截断
        max_obstacles = 10  # 假设最多10个障碍物
        obstacle_states = latest['obstacles']
        
        if len(obstacle_states) < max_obstacles:
            # 填充零
            padding = np.zeros((max_obstacles - len(obstacle_states), 5))
            obstacle_states = np.vstack([obstacle_states, padding])
        else:
            # 截断
            obstacle_states = obstacle_states[:max_obstacles]
        
        return {
            'self_state': latest['self'],
            'goal_state': latest['goal'],
            'obstacle_states': obstacle_states
        }


class LSTMFlightWrapper:
    """
    封装Flight环境，添加LSTM状态处理
    """
    def __init__(self, flight_env, history_length=5):
        self.env = flight_env
        self.preprocessor = LSTMStatePreprocessor(history_length)
        
        # LSTM模型的状态维度
        self.self_state_dim = 4   # [pos_x, pos_y, direction, velocity]
        self.goal_state_dim = 2   # [distance, direction]
        self.obstacle_state_dim = 5  # [rel_x, rel_y, vx, vy, radius]
        self.max_obstacles = 10
        
    def reset(self, scenario):
        """重置环境"""
        self.preprocessor.reset()
        original_state = self.env.reset(scenario)
        
        # 处理第一个状态
        self.preprocessor.process_state(self.env)
        
        return self.get_lstm_state()
    
    def step(self, action):
        """执行动作"""
        original_state, reward, done = self.env.step(action)
        
        # 处理新状态
        self.preprocessor.process_state(self.env)
        
        lstm_state = self.get_lstm_state()
        
        return lstm_state, reward, done
    
    def get_lstm_state(self):
        """获取LSTM格式的状态"""
        lstm_input = self.preprocessor.get_lstm_input()
        
        if lstm_input is None:
            # 返回零状态
            return {
                'self_state': np.zeros(self.self_state_dim),
                'goal_state': np.zeros(self.goal_state_dim),
                'obstacle_states': np.zeros((self.max_obstacles, self.obstacle_state_dim))
            }
        
        return lstm_input
    
    def render(self, *args, **kwargs):
        """渲染环境"""
        return self.env.render(*args, **kwargs)


def create_lstm_ac_model(self_state_dim, goal_state_dim, obstacle_state_dim, 
                        human_num, action_dim, use_attention=False):
    """
    创建LSTM-AC模型的工厂函数
    """
    from lstm_model import LSTMRL, AttentionLSTMRL
    
    if use_attention:
        model = AttentionLSTMRL(
            state_dim=self_state_dim,
            goal_dim=goal_state_dim,
            obstacle_state_dim=obstacle_state_dim,
            human_num=human_num,
            action_dim=action_dim,
            lstm_hidden_dim=50,
            attention_dim=100
        )
    else:
        model = LSTMRL(
            state_dim=self_state_dim,
            goal_dim=goal_state_dim,
            obstacle_state_dim=obstacle_state_dim,
            human_num=human_num,
            action_dim=action_dim,
            lstm_hidden_dim=50,
            use_lstm=True
        )
    
    return model


# 使用示例
if __name__ == "__main__":
    print("LSTM集成模块测试")
    print("=" * 50)
    
    # 模拟环境
    class MockEnv:
        def __init__(self):
            self.current_pos = np.array([100.0, 100.0])
            self.current_dir = 0.5
            self.goal_pos = np.array([600.0, 600.0])
            self.goal_dir = 0.3
            
            # 模拟动态障碍物
            class MockObstacle:
                def __init__(self, pos, velocity, direction, radius):
                    self.pos = pos
                    self.velocity = velocity
                    self.direction = direction
                    self.radius = radius
            
            self.dynamic_obstacles = [
                MockObstacle(np.array([200.0, 200.0]), 2.0, 0.5, 30.0),
                MockObstacle(np.array([400.0, 300.0]), 1.5, 1.0, 25.0),
                MockObstacle(np.array([300.0, 500.0]), 1.8, -0.5, 35.0),
            ]
            
            class MockAction:
                linear_velocity = 5.0
            
            self.action = MockAction()
    
    # 测试预处理器
    mock_env = MockEnv()
    preprocessor = LSTMStatePreprocessor(history_length=5)
    
    print("\n测试状态提取:")
    self_state = preprocessor.extract_self_state(mock_env)
    print(f"自身状态: {self_state}")
    
    goal_state = preprocessor.extract_goal_state(mock_env)
    print(f"目标状态: {goal_state}")
    
    obstacle_states = preprocessor.extract_dynamic_obstacle_states(mock_env)
    print(f"障碍物状态形状: {obstacle_states.shape}")
    print(f"第一个障碍物状态: {obstacle_states[0]}")
    
    print("\n测试LSTM输入格式:")
    preprocessor.process_state(mock_env)
    lstm_input = preprocessor.get_lstm_input()
    print(f"自身状态维度: {lstm_input['self_state'].shape}")
    print(f"目标状态维度: {lstm_input['goal_state'].shape}")
    print(f"障碍物状态维度: {lstm_input['obstacle_states'].shape}")
    
    print("\n集成测试完成!")
