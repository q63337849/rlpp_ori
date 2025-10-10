"""
CNN模型 - TensorFlow 2.x版本
包含1D卷积的深度学习模型
"""
import numpy as np
import tensorflow as tf
from .util import model


class StateModel(tf.keras.Model):
    """CNN状态处理模型"""

    def __init__(self, eval_model: bool, size_splits, n_filters=5):
        """
        Args:
            eval_model: 是否为评估模型（用于区分eval和target）
            size_splits: [obs_dim, goal_dim] 观测和目标的维度分割
            n_filters: 卷积核数量
        """
        super().__init__()
        self.d_obs, self.d_goal = size_splits
        self.eval_model = eval_model
        self.n_filters = n_filters

        # 卷积层
        self.conv1 = tf.keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=5,
            strides=2,
            activation='relu',
            kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
            bias_initializer=tf.keras.initializers.Constant(0.1),
            trainable=eval_model
        )

        # 池化层
        self.pool1 = tf.keras.layers.MaxPooling1D(
            pool_size=2,
            strides=2
        )

        # 展平层
        self.flatten = tf.keras.layers.Flatten()

    def call(self, obs, goal):
        """
        前向传播
        Args:
            obs: 观测 [batch, d_obs] 或 [batch, d_obs, 1]
            goal: 目标 [batch, d_goal]
        Returns:
            拼接后的特征
        """
        # 确保obs是3D: [batch, length, channels]
        if len(obs.shape) == 2:
            obs = tf.expand_dims(obs, axis=-1)

        # 卷积 -> 池化 -> 展平
        x = self.conv1(obs)
        x = self.pool1(x)
        x = self.flatten(x)

        # 拼接展平后的特征和目标
        output = tf.concat([x, goal], axis=1)
        return output

    def process_state(self, state):
        """
        处理输入状态，分离obs和goal
        Args:
            state: tensor或numpy array [batch, d_obs + d_goal]
        Returns:
            obs, goal: 分离后的tensor
        """
        # 确保输入是tensor
        if not isinstance(state, tf.Tensor):
            state = tf.convert_to_tensor(state, dtype=tf.float32)

        # 使用TensorFlow切片代替numpy split
        obs_state = state[:, :self.d_obs]
        goal_state = state[:, self.d_obs:]

        # CNN需要添加channel维度
        obs_state = tf.expand_dims(obs_state, axis=-1)

        return obs_state, goal_state


@model
class ActorModel(tf.keras.Model):
    """Actor模型 - 连续动作"""

    def __init__(self, special_info, eval_model: bool):
        """
        Args:
            special_info: 包含size_splits, n_filters, actor_n_layers的字典
            eval_model: 是否为评估模型
        """
        super().__init__()
        self.eval_model = eval_model
        self.special_info = special_info

        # 状态处理层（CNN）
        self.state = StateModel(
            eval_model,
            special_info['size_splits'],
            special_info['n_filters']
        )

        # Actor特定层
        self.actor_layers = []
        for n_layer in special_info['actor_n_layers']:
            self.actor_layers.append(
                tf.keras.layers.Dense(
                    n_layer,
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
                    bias_initializer=tf.keras.initializers.Constant(0.1),
                    trainable=eval_model
                )
            )

        # 输出层（连续动作，使用tanh）
        self.output_layer = tf.keras.layers.Dense(
            1,
            activation='tanh',
            kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
            bias_initializer=tf.keras.initializers.Constant(0.1),
            trainable=eval_model
        )

    def call(self, inputs):
        """
        前向传播
        Args:
            inputs: 可以是state数组或(obs, goal)元组
        Returns:
            动作输出
        """
        if isinstance(inputs, tuple):
            obs, goal = inputs
        else:
            obs, goal = self.state.process_state(inputs)

        # 状态编码
        layer = self.state(obs, goal)

        # Actor层
        for actor_layer in self.actor_layers:
            layer = actor_layer(layer)

        # 输出动作
        output = self.output_layer(layer)
        return output

    def __call__(self, state):
        """兼容旧接口"""
        if isinstance(state, dict):
            obs = state.get('obs', None)
            goal = state.get('goal', None)
            if obs is not None and goal is not None:
                return self.call((obs, goal))
        return self.call(state)


@model
class CriticModel(tf.keras.Model):
    """Critic模型 - Q值估计"""

    def __init__(self, action, special_info: dict, eval_model: bool):
        """
        Args:
            action: Actor模型输出（或占位符，TF2中不使用）
            special_info: 包含size_splits, n_filters, critic_n_layers的字典
            eval_model: 是否为评估模型
        """
        super().__init__()
        self.eval_model = eval_model
        self.special_info = special_info
        self.actor_action = action  # 保存引用，兼容性

        # 状态处理层（CNN）
        self.state = StateModel(
            eval_model,
            special_info['size_splits'],
            special_info['n_filters']
        )

        # Critic特定层
        self.critic_layers = []
        for n_layer in special_info['critic_n_layers']:
            self.critic_layers.append(
                tf.keras.layers.Dense(
                    n_layer,
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
                    bias_initializer=tf.keras.initializers.Constant(0.1),
                    trainable=eval_model
                )
            )

        # 输出层（Q值）
        self.output_layer = tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
            bias_initializer=tf.keras.initializers.Constant(0.1),
            trainable=eval_model
        )

    def call(self, inputs, action=None):
        """
        前向传播
        Args:
            inputs: state数组或(obs, goal)元组
            action: 动作（如果提供）
        Returns:
            Q值输出
        """
        if isinstance(inputs, tuple) and len(inputs) == 2:
            obs, goal = inputs
        else:
            obs, goal = self.state.process_state(inputs)

        # 状态编码
        state_features = self.state(obs, goal)

        # 拼接动作
        if action is not None:
            layer = tf.concat([state_features, action], axis=1)
        else:
            layer = state_features

        # Critic层
        for critic_layer in self.critic_layers:
            layer = critic_layer(layer)

        # 输出Q值
        output = self.output_layer(layer)
        return output

    def __call__(self, state, action=None):
        """兼容旧接口"""
        if isinstance(state, dict):
            obs = state.get('obs', None)
            goal = state.get('goal', None)
            if obs is not None and goal is not None:
                return self.call((obs, goal), action)
        return self.call(state, action)


@model
class ActionModel(tf.keras.Model):
    """Action模型 - 离散动作（策略网络）"""

    def __init__(self, n_actions, special_info: dict, eval_model: bool):
        """
        Args:
            n_actions: 动作数量
            special_info: 包含size_splits, n_filters, actor_n_layers的字典
            eval_model: 是否为评估模型
        """
        super().__init__()
        self.n_actions = n_actions
        self.eval_model = eval_model
        self.special_info = special_info

        # 状态处理层（CNN）
        self.state = StateModel(
            eval_model,
            special_info['size_splits'],
            special_info['n_filters']
        )

        # Actor特定层
        self.actor_layers = []
        for n_layer in special_info['actor_n_layers']:
            self.actor_layers.append(
                tf.keras.layers.Dense(
                    n_layer,
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
                    bias_initializer=tf.keras.initializers.Constant(0.1),
                    trainable=eval_model
                )
            )

        # 输出层（动作概率分布）
        self.output_layer = tf.keras.layers.Dense(
            n_actions,
            activation='softmax',
            kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
            bias_initializer=tf.keras.initializers.Constant(0.1),
            trainable=eval_model
        )

    def call(self, inputs):
        """
        前向传播
        Args:
            inputs: state数组或(obs, goal)元组
        Returns:
            动作概率分布
        """
        if isinstance(inputs, tuple):
            obs, goal = inputs
        else:
            obs, goal = self.state.process_state(inputs)

        # 状态编码
        layer = self.state(obs, goal)

        # Actor层
        for actor_layer in self.actor_layers:
            layer = actor_layer(layer)

        # 输出动作概率
        output = self.output_layer(layer)
        return output

    def __call__(self, state):
        """兼容旧接口"""
        if isinstance(state, dict):
            obs = state.get('obs', None)
            goal = state.get('goal', None)
            if obs is not None and goal is not None:
                return self.call((obs, goal))
        return self.call(state)


@model
class ValueModel(tf.keras.Model):
    """Value模型 - 状态价值或Q值"""

    def __init__(self, n_actions, special_info: dict, eval_model: bool):
        """
        Args:
            n_actions: 输出维度（多个动作的Q值或状态价值）
            special_info: 包含size_splits, n_filters, critic_n_layers的字典
            eval_model: 是否为评估模型
        """
        super().__init__()
        self.n_actions = n_actions
        self.eval_model = eval_model
        self.special_info = special_info

        # 状态处理层（CNN）
        self.state = StateModel(
            eval_model,
            special_info['size_splits'],
            special_info['n_filters']
        )

        # Critic特定层
        self.critic_layers = []
        for n_layer in special_info['critic_n_layers']:
            self.critic_layers.append(
                tf.keras.layers.Dense(
                    n_layer,
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
                    bias_initializer=tf.keras.initializers.Constant(0.1),
                    trainable=eval_model
                )
            )

        # 输出层
        self.output_layer = tf.keras.layers.Dense(
            n_actions,
            kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
            bias_initializer=tf.keras.initializers.Constant(0.1),
            trainable=eval_model
        )

    def call(self, inputs):
        """
        前向传播
        Args:
            inputs: state数组或(obs, goal)元组
        Returns:
            价值输出
        """
        if isinstance(inputs, tuple):
            obs, goal = inputs
        else:
            obs, goal = self.state.process_state(inputs)

        # 状态编码
        layer = self.state(obs, goal)

        # Critic层
        for critic_layer in self.critic_layers:
            layer = critic_layer(layer)

        # 输出价值
        output = self.output_layer(layer)
        return output

    def __call__(self, state):
        """兼容旧接口"""
        if isinstance(state, dict):
            obs = state.get('obs', None)
            goal = state.get('goal', None)
            if obs is not None and goal is not None:
                return self.call((obs, goal))
        return self.call(state)