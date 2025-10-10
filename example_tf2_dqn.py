#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TensorFlow 2.x 示例: DQN网络
展示了如何用TF2风格编写强化学习网络
"""

import tensorflow as tf
import numpy as np


class DQNNetwork(tf.keras.Model):
    """
    DQN网络 - TensorFlow 2.x版本
    使用Keras API构建
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super(DQNNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 构建网络层
        self.layers_list = []
        
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers_list.append(
                tf.keras.layers.Dense(
                    hidden_dim, 
                    activation='relu',
                    name=f'hidden_{i}'
                )
            )
        
        # 输出层
        self.output_layer = tf.keras.layers.Dense(
            action_dim,
            name='output'
        )
    
    def call(self, state, training=False):
        """
        前向传播
        Args:
            state: [batch_size, state_dim]
            training: 是否训练模式
        Returns:
            q_values: [batch_size, action_dim]
        """
        x = state
        
        # 通过隐藏层
        for layer in self.layers_list:
            x = layer(x)
        
        # 输出Q值
        q_values = self.output_layer(x)
        
        return q_values


class DQNAgent:
    """
    DQN智能体 - TensorFlow 2.x版本
    """
    
    def __init__(self, state_dim, action_dim, 
                 learning_rate=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # 创建Q网络
        self.q_network = DQNNetwork(state_dim, action_dim)
        
        # 创建目标网络
        self.target_network = DQNNetwork(state_dim, action_dim)
        
        # 优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        # 损失函数
        self.loss_fn = tf.keras.losses.MeanSquaredError()
    
    @tf.function
    def predict(self, state):
        """
        预测Q值
        Args:
            state: [batch_size, state_dim]
        Returns:
            q_values: [batch_size, action_dim]
        """
        return self.q_network(state, training=False)
    
    def select_action(self, state, epsilon=0.1):
        """
        选择动作 (epsilon-greedy)
        Args:
            state: [state_dim]
            epsilon: 探索率
        Returns:
            action: int
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        # 扩展维度
        state_batch = tf.expand_dims(state, 0)
        
        # 预测Q值
        q_values = self.predict(state_batch)
        
        # 选择最大Q值对应的动作
        action = tf.argmax(q_values[0]).numpy()
        
        return action
    
    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """
        训练一步
        Args:
            states: [batch_size, state_dim]
            actions: [batch_size]
            rewards: [batch_size]
            next_states: [batch_size, state_dim]
            dones: [batch_size]
        Returns:
            loss: scalar
        """
        with tf.GradientTape() as tape:
            # 当前Q值
            current_q = self.q_network(states, training=True)
            action_indices = tf.stack(
                [tf.range(tf.shape(actions)[0]), actions], 
                axis=1
            )
            current_q_values = tf.gather_nd(current_q, action_indices)
            
            # 目标Q值
            next_q = self.target_network(next_states, training=False)
            max_next_q = tf.reduce_max(next_q, axis=1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q
            
            # 计算损失
            loss = self.loss_fn(target_q_values, current_q_values)
        
        # 反向传播
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )
        
        return loss
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def save(self, path):
        """保存模型"""
        self.q_network.save_weights(path)
    
    def load(self, path):
        """加载模型"""
        self.q_network.load_weights(path)


# 使用示例
if __name__ == '__main__':
    print("TensorFlow 2.x DQN Example")
    print("=" * 60)
    
    # 创建智能体
    agent = DQNAgent(state_dim=4, action_dim=2)
    
    # 模拟训练数据
    batch_size = 32
    states = tf.random.normal([batch_size, 4])
    actions = tf.random.uniform([batch_size], maxval=2, dtype=tf.int32)
    rewards = tf.random.normal([batch_size])
    next_states = tf.random.normal([batch_size, 4])
    dones = tf.cast(tf.random.uniform([batch_size]) > 0.9, tf.float32)
    
    # 训练一步
    loss = agent.train_step(states, actions, rewards, next_states, dones)
    print(f"Training loss: {loss.numpy():.4f}")
    
    # 选择动作
    state = tf.random.normal([4])
    action = agent.select_action(state)
    print(f"Selected action: {action}")
    
    # 保存模型
    agent.save('dqn_model')
    print("Model saved")
    
    print("=" * 60)
    print("✓ Example completed successfully!")
