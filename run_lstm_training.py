"""
完整的LSTM训练脚本
结合动态障碍物环境进行LSTM-RL训练
"""

import numpy as np
import tensorflow as tf
import math
from collections import deque

# 导入环境和模型
from Env.flight_with_dynamic_obstacles import Flight, DynamicObstacle, Scenairo
from Algo.lstm_flight_integration import LSTMFlightWrapper
from Algo.lstm_model import LSTMRL, AttentionLSTMRL


def create_test_scenario():
    """创建测试场景"""
    # 创建动态障碍物
    dynamic_obstacles = [
        DynamicObstacle([150, 500], 30, 2.0, None, [50, 50, 650, 650], 0.03),
        DynamicObstacle([550, 350], 25, 1.5, None, [50, 50, 650, 650], 0.05),
        DynamicObstacle([300, 100], 35, 1.8, None, [50, 50, 650, 650], 0.02),
        DynamicObstacle([400, 400], 28, 2.2, None, [50, 50, 650, 650], 0.04),
        DynamicObstacle([200, 300], 22, 1.7, None, [50, 50, 650, 650], 0.03),
    ]
    
    scenario = Scenairo(
        init_pos=[100, 100],
        init_dir=45,
        goal_pos=[600, 600],
        goal_dir=0,
        circle_obstacles=[[300, 300, 40], [450, 200, 35]],
        dynamic_obstacles=dynamic_obstacles
    )
    
    return scenario


class SimpleLSTMAgent:
    """简单的LSTM智能体，用于测试"""
    def __init__(self, model, action_dim):
        self.model = model
        self.action_dim = action_dim
        self.sess = None
        
        # 创建计算图
        self.build_graph()
    
    def build_graph(self):
        """构建TensorFlow计算图"""
        # 重置图
        tf.reset_default_graph()
        
        # 输入占位符
        self.self_state_ph = tf.placeholder(tf.float32, [None, 4], name='self_state')
        self.goal_state_ph = tf.placeholder(tf.float32, [None, 2], name='goal_state')
        self.obstacle_states_ph = tf.placeholder(tf.float32, [None, 10, 5], name='obstacle_states')
        
        # 构建网络
        self.action_logits = self.model.build_network(
            self.self_state_ph,
            self.goal_state_ph,
            self.obstacle_states_ph,
            scope='policy_network'
        )
        
        # 动作分布（用于离散动作）
        self.action_probs = tf.nn.softmax(self.action_logits)
        
        # 价值网络（用于AC算法）
        self.value = self.model.build_value_network(
            self.self_state_ph,
            self.goal_state_ph,
            self.obstacle_states_ph,
            scope='value_network'
        )
        
        # 用于训练的占位符
        self.action_ph = tf.placeholder(tf.int32, [None], name='action')
        self.advantage_ph = tf.placeholder(tf.float32, [None], name='advantage')
        self.target_value_ph = tf.placeholder(tf.float32, [None], name='target_value')
        
        # Actor损失（策略梯度）
        action_one_hot = tf.one_hot(self.action_ph, self.action_dim)
        action_log_prob = tf.reduce_sum(action_one_hot * tf.log(self.action_probs + 1e-10), axis=1)
        self.actor_loss = -tf.reduce_mean(action_log_prob * self.advantage_ph)
        
        # Critic损失（价值函数）
        self.critic_loss = tf.reduce_mean(tf.square(self.value - self.target_value_ph))
        
        # 总损失
        self.total_loss = self.actor_loss + 0.5 * self.critic_loss
        
        # 优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = self.optimizer.minimize(self.total_loss)
        
        # 初始化会话
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def select_action(self, lstm_state, deterministic=False):
        """选择动作"""
        feed_dict = {
            self.self_state_ph: [lstm_state['self_state']],
            self.goal_state_ph: [lstm_state['goal_state']],
            self.obstacle_states_ph: [lstm_state['obstacle_states']]
        }
        
        action_probs = self.sess.run(self.action_probs, feed_dict=feed_dict)[0]
        
        if deterministic:
            action = np.argmax(action_probs)
        else:
            action = np.random.choice(self.action_dim, p=action_probs)
        
        return action
    
    def get_value(self, lstm_state):
        """获取状态价值"""
        feed_dict = {
            self.self_state_ph: [lstm_state['self_state']],
            self.goal_state_ph: [lstm_state['goal_state']],
            self.obstacle_states_ph: [lstm_state['obstacle_states']]
        }
        
        value = self.sess.run(self.value, feed_dict=feed_dict)[0][0]
        return value
    
    def train(self, batch_data):
        """训练智能体"""
        states, actions, advantages, target_values = batch_data
        
        self_states = np.array([s['self_state'] for s in states])
        goal_states = np.array([s['goal_state'] for s in states])
        obstacle_states = np.array([s['obstacle_states'] for s in states])
        
        feed_dict = {
            self.self_state_ph: self_states,
            self.goal_state_ph: goal_states,
            self.obstacle_states_ph: obstacle_states,
            self.action_ph: actions,
            self.advantage_ph: advantages,
            self.target_value_ph: target_values
        }
        
        _, actor_loss, critic_loss = self.sess.run(
            [self.train_op, self.actor_loss, self.critic_loss],
            feed_dict=feed_dict
        )
        
        return actor_loss, critic_loss


def test_lstm_model():
    """测试LSTM模型基本功能"""
    print("=" * 50)
    print("测试LSTM模型")
    print("=" * 50)
    
    # 1. 创建环境
    base_env = Flight()
    lstm_env = LSTMFlightWrapper(base_env, history_length=5)
    
    # 2. 创建模型
    model = LSTMRL(
        state_dim=4,
        goal_dim=2,
        obstacle_state_dim=5,
        human_num=10,
        action_dim=base_env.action.n_actions,
        lstm_hidden_dim=50,
        use_lstm=True
    )
    
    print(f"环境动作维度: {base_env.action.n_actions}")
    
    # 3. 创建场景并重置
    scenario = create_test_scenario()
    print(f"场景: {scenario}")
    
    lstm_state = lstm_env.reset(scenario)
    
    print(f"LSTM状态:")
    print(f"  自身状态: {lstm_state['self_state'].shape}")
    print(f"  目标状态: {lstm_state['goal_state'].shape}")
    print(f"  障碍物状态: {lstm_state['obstacle_states'].shape}")
    
    # 4. 创建智能体
    agent = SimpleLSTMAgent(model, base_env.action.n_actions)
    
    # 5. 测试几步
    print("\n开始测试运行...")
    for step in range(10):
        action = agent.select_action(lstm_state)
        value = agent.get_value(lstm_state)
        
        next_lstm_state, reward, done = lstm_env.step(action)
        
        print(f"步骤 {step}: 动作={action}, 价值={value:.3f}, 奖励={reward:.3f}, 完成={done}")
        
        if done:
            print(f"任务完成! 结果: {['超时', '失败', '边界', '成功'][lstm_env.env.result]}")
            break
        
        lstm_state = next_lstm_state
    
    print("\n测试完成!")


def simple_training_demo():
    """简单的训练演示"""
    print("\n" + "=" * 50)
    print("LSTM训练演示")
    print("=" * 50)
    
    # 创建环境
    base_env = Flight()
    lstm_env = LSTMFlightWrapper(base_env, history_length=5)
    
    # 创建模型
    model = LSTMRL(
        state_dim=4,
        goal_dim=2,
        obstacle_state_dim=5,
        human_num=10,
        action_dim=base_env.action.n_actions,
        lstm_hidden_dim=50,
        use_lstm=True
    )
    
    # 创建智能体
    agent = SimpleLSTMAgent(model, base_env.action.n_actions)
    
    # 训练参数
    num_episodes = 5
    max_steps = 100
    gamma = 0.99
    
    print(f"训练参数:")
    print(f"  Episodes: {num_episodes}")
    print(f"  最大步数: {max_steps}")
    print(f"  折扣因子: {gamma}")
    
    # 训练循环
    for episode in range(num_episodes):
        scenario = create_test_scenario()
        lstm_state = lstm_env.reset(scenario)
        
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(lstm_state)
            
            # 执行动作
            next_lstm_state, reward, done = lstm_env.step(action)
            
            # 存储经验
            episode_states.append(lstm_state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            if step % 20 == 0:
                print(f"  步骤 {step}: 奖励={reward:.3f}")
            
            if done:
                result_names = ['超时', '失败', '边界', '成功']
                print(f"  完成! 结果: {result_names[lstm_env.env.result]}, 总步数: {step + 1}")
                break
            
            lstm_state = next_lstm_state
        
        # 计算优势和目标价值
        advantages = []
        target_values = []
        
        running_add = 0
        for r in reversed(episode_rewards):
            running_add = running_add * gamma + r
            target_values.insert(0, running_add)
        
        target_values = np.array(target_values)
        
        # 计算优势
        values = [agent.get_value(s) for s in episode_states]
        advantages = target_values - np.array(values)
        
        # 训练
        batch_data = (episode_states, episode_actions, advantages, target_values)
        actor_loss, critic_loss = agent.train(batch_data)
        
        print(f"  Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
        print(f"  平均奖励: {np.mean(episode_rewards):.3f}")
    
    print("\n训练演示完成!")
    
    # 测试训练后的智能体
    print("\n测试训练后的智能体...")
    scenario = create_test_scenario()
    lstm_state = lstm_env.reset(scenario)
    
    total_reward = 0
    for step in range(max_steps):
        action = agent.select_action(lstm_state, deterministic=True)
        next_lstm_state, reward, done = lstm_env.step(action)
        total_reward += reward
        
        # 尝试渲染
        try:
            lstm_env.render(sleep_time=0.05, show_trace=True, show_pos=True)
        except:
            pass
        
        if done:
            result_names = ['超时', '失败', '边界', '成功']
            print(f"测试结果: {result_names[lstm_env.env.result]}")
            print(f"总步数: {step + 1}, 总奖励: {total_reward:.3f}")
            break
        
        lstm_state = next_lstm_state


def main():
    """主函数"""
    import sys
    
    print("LSTM-RL训练脚本")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # 完整训练模式
        simple_training_demo()
    else:
        # 测试模式
        test_lstm_model()
    
    print("\n所有测试完成!")


if __name__ == "__main__":
    main()
