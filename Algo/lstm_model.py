"""
LSTM-based RL Model for Dynamic Obstacle Avoidance
基于LSTM的强化学习模型，用于动态障碍物避障
参考：CrowdNav项目的LSTM-RL实现
"""

import tensorflow as tf
import numpy as np


class LSTMStateEncoder(object):
    """
    LSTM编码器，用于处理动态障碍物的时序信息
    类似于CrowdNav中的处理方式
    """
    def __init__(self, obstacle_state_dim, human_num, lstm_hidden_dim=50):
        """
        obstacle_state_dim: 每个障碍物的状态维度 (例如: [relative_x, relative_y, vx, vy, radius])
        human_num: 动态障碍物的数量
        lstm_hidden_dim: LSTM隐藏层维度
        """
        self.obstacle_state_dim = obstacle_state_dim
        self.human_num = human_num
        self.lstm_hidden_dim = lstm_hidden_dim
        
    def build(self, obstacle_states, scope='lstm_encoder'):
        """
        obstacle_states: [batch_size, human_num, obstacle_state_dim]
        返回: [batch_size, lstm_hidden_dim]
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # 重塑输入以适应LSTM
            # [batch_size, human_num, obstacle_state_dim]
            
            # 创建LSTM cell
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_dim, name='lstm_cell')
            
            # 将每个障碍物视为一个时间步
            # outputs: [batch_size, human_num, lstm_hidden_dim]
            # states: (c_state, h_state)
            outputs, states = tf.nn.dynamic_rnn(
                lstm_cell,
                obstacle_states,
                dtype=tf.float32,
                scope='lstm_rnn'
            )
            
            # 使用最后一个时间步的输出
            last_output = outputs[:, -1, :]  # [batch_size, lstm_hidden_dim]
            
            return last_output


class MultiHumanStateEncoder(object):
    """
    多障碍物状态编码器
    参考CrowdNav的multi_human_rl.py
    """
    def __init__(self, obstacle_state_dim, human_num, mlp_dims=[150, 100, 100]):
        """
        使用MLP对每个障碍物进行编码，然后聚合
        """
        self.obstacle_state_dim = obstacle_state_dim
        self.human_num = human_num
        self.mlp_dims = mlp_dims
        
    def build(self, obstacle_states, scope='multi_human_encoder'):
        """
        obstacle_states: [batch_size, human_num, obstacle_state_dim]
        返回: [batch_size, mlp_dims[-1]]
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(obstacle_states)[0]
            
            # 重塑为 [batch_size * human_num, obstacle_state_dim]
            reshaped = tf.reshape(obstacle_states, [-1, self.obstacle_state_dim])
            
            # 对每个障碍物应用相同的MLP
            mlp_output = reshaped
            for i, dim in enumerate(self.mlp_dims):
                mlp_output = tf.layers.dense(
                    mlp_output,
                    dim,
                    activation=tf.nn.relu,
                    name=f'mlp_layer_{i}'
                )
            
            # 重塑回 [batch_size, human_num, mlp_dims[-1]]
            mlp_output = tf.reshape(mlp_output, [batch_size, self.human_num, self.mlp_dims[-1]])
            
            # 使用max pooling聚合所有障碍物的信息
            pooled = tf.reduce_max(mlp_output, axis=1)  # [batch_size, mlp_dims[-1]]
            
            return pooled


class LSTMRL(object):
    """
    完整的LSTM-RL模型
    结合自身状态、目标信息和动态障碍物的时序信息
    """
    def __init__(self, 
                 state_dim,           # 自身状态维度 (位置、速度等)
                 goal_dim,            # 目标信息维度
                 obstacle_state_dim,  # 单个障碍物状态维度
                 human_num,           # 动态障碍物数量
                 action_dim,          # 动作维度
                 lstm_hidden_dim=50,
                 mlp_dims=[150, 100],
                 use_lstm=True):
        """
        use_lstm: 是否使用LSTM编码障碍物，False则使用MLP
        """
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.obstacle_state_dim = obstacle_state_dim
        self.human_num = human_num
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp_dims = mlp_dims
        self.use_lstm = use_lstm
        
        # 创建编码器
        if use_lstm:
            self.obstacle_encoder = LSTMStateEncoder(
                obstacle_state_dim, human_num, lstm_hidden_dim
            )
        else:
            self.obstacle_encoder = MultiHumanStateEncoder(
                obstacle_state_dim, human_num, mlp_dims
            )
    
    def build_network(self, 
                      self_state,        # [batch_size, state_dim]
                      goal_state,        # [batch_size, goal_dim]
                      obstacle_states,   # [batch_size, human_num, obstacle_state_dim]
                      scope='lstm_rl'):
        """
        构建完整的网络
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # 1. 编码动态障碍物信息
            obstacle_encoding = self.obstacle_encoder.build(obstacle_states)
            
            # 2. 拼接所有信息
            # [batch_size, state_dim + goal_dim + lstm_hidden_dim/mlp_dims[-1]]
            joint_state = tf.concat([self_state, goal_state, obstacle_encoding], axis=1)
            
            # 3. 通过全连接层处理
            fc1 = tf.layers.dense(joint_state, 150, activation=tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, 100, activation=tf.nn.relu, name='fc2')
            
            # 4. 输出动作
            action_logits = tf.layers.dense(fc2, self.action_dim, name='action_output')
            
            return action_logits
    
    def build_value_network(self,
                           self_state,
                           goal_state,
                           obstacle_states,
                           scope='value_network'):
        """
        构建价值网络（用于Actor-Critic）
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # 编码障碍物
            obstacle_encoding = self.obstacle_encoder.build(obstacle_states)
            
            # 拼接状态
            joint_state = tf.concat([self_state, goal_state, obstacle_encoding], axis=1)
            
            # 全连接层
            fc1 = tf.layers.dense(joint_state, 150, activation=tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, 100, activation=tf.nn.relu, name='fc2')
            fc3 = tf.layers.dense(fc2, 100, activation=tf.nn.relu, name='fc3')
            
            # 输出状态价值
            value = tf.layers.dense(fc3, 1, name='value_output')
            
            return value


class AttentionLSTMRL(object):
    """
    带注意力机制的LSTM-RL模型
    参考CrowdNav中的注意力机制实现
    """
    def __init__(self,
                 state_dim,
                 goal_dim,
                 obstacle_state_dim,
                 human_num,
                 action_dim,
                 lstm_hidden_dim=50,
                 attention_dim=100):
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.obstacle_state_dim = obstacle_state_dim
        self.human_num = human_num
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.attention_dim = attention_dim
    
    def attention_mechanism(self, 
                           query,           # [batch_size, query_dim]
                           keys_values,     # [batch_size, human_num, key_value_dim]
                           scope='attention'):
        """
        实现注意力机制
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(keys_values)[0]
            
            # 扩展query以匹配keys
            query_expanded = tf.expand_dims(query, 1)  # [batch_size, 1, query_dim]
            query_expanded = tf.tile(query_expanded, [1, self.human_num, 1])  # [batch_size, human_num, query_dim]
            
            # 拼接query和keys
            attention_input = tf.concat([query_expanded, keys_values], axis=2)
            
            # 计算注意力分数
            attention_mlp = tf.layers.dense(
                tf.reshape(attention_input, [-1, tf.shape(attention_input)[2]]),
                self.attention_dim,
                activation=tf.nn.relu,
                name='attention_mlp'
            )
            
            attention_scores = tf.layers.dense(
                attention_mlp,
                1,
                name='attention_scores'
            )
            
            # 重塑并归一化
            attention_scores = tf.reshape(attention_scores, [batch_size, self.human_num])
            attention_weights = tf.nn.softmax(attention_scores, axis=1)  # [batch_size, human_num]
            
            # 应用注意力权重
            attention_weights_expanded = tf.expand_dims(attention_weights, 2)  # [batch_size, human_num, 1]
            attended_values = tf.reduce_sum(keys_values * attention_weights_expanded, axis=1)  # [batch_size, key_value_dim]
            
            return attended_values, attention_weights
    
    def build_network(self,
                     self_state,
                     goal_state,
                     obstacle_states,
                     scope='attention_lstm_rl'):
        """
        构建带注意力的网络
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # 1. 使用LSTM编码每个障碍物
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_dim)
            lstm_outputs, _ = tf.nn.dynamic_rnn(
                lstm_cell,
                obstacle_states,
                dtype=tf.float32
            )  # [batch_size, human_num, lstm_hidden_dim]
            
            # 2. 创建query（自身状态和目标信息）
            query = tf.concat([self_state, goal_state], axis=1)
            
            # 3. 应用注意力机制
            attended_obstacle_info, attention_weights = self.attention_mechanism(
                query, lstm_outputs
            )
            
            # 4. 拼接所有信息
            joint_state = tf.concat([self_state, goal_state, attended_obstacle_info], axis=1)
            
            # 5. 全连接层
            fc1 = tf.layers.dense(joint_state, 150, activation=tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, 100, activation=tf.nn.relu, name='fc2')
            
            # 6. 输出动作
            action_logits = tf.layers.dense(fc2, self.action_dim, name='action_output')
            
            return action_logits, attention_weights


# 使用示例和测试
if __name__ == "__main__":
    print("LSTM-RL模型定义完成")
    print("支持的模型类型:")
    print("1. LSTMRL - 基础LSTM编码器")
    print("2. MultiHumanStateEncoder - MLP编码器")
    print("3. AttentionLSTMRL - 带注意力机制的LSTM")
    
    # 简单的维度验证
    with tf.Session() as sess:
        # 测试LSTM编码器
        batch_size = 4
        human_num = 5
        obstacle_state_dim = 5
        
        test_input = tf.placeholder(tf.float32, [batch_size, human_num, obstacle_state_dim])
        
        lstm_encoder = LSTMStateEncoder(obstacle_state_dim, human_num, lstm_hidden_dim=50)
        output = lstm_encoder.build(test_input)
        
        sess.run(tf.global_variables_initializer())
        
        dummy_data = np.random.randn(batch_size, human_num, obstacle_state_dim)
        result = sess.run(output, feed_dict={test_input: dummy_data})
        
        print(f"\n测试成功!")
        print(f"输入维度: {dummy_data.shape}")
        print(f"输出维度: {result.shape}")
