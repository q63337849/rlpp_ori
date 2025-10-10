"""
CNN2模型 - TensorFlow 2.x版本
包含2层卷积的深度学习模型
"""
import numpy as np
import tensorflow as tf
from .util import model


class StateModel(tf.keras.Model):
    """CNN2状态处理模型 - 2层卷积"""

    def __init__(self, eval_model: bool, size_splits, n_filters=5):
        """
        Args:
            eval_model: 是否为评估模型（用于区分eval和target）
            size_splits: [obs_dim, goal_dim] 观测和目标的维度分割
            n_filters: 第一层卷积核数量
        """
        super().__init__()
        self.d_obs, self.d_goal = size_splits
        self.eval_model = eval_model
        self.n_filters = n_filters

        # 第一层卷积
        self.conv1 = tf.keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=3,
            strides=1,
            activation='relu',
            kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
            bias_initializer=tf.keras.initializers.Constant(0.1),
            trainable=eval_model
        )

        # 第一层池化
        self.pool1 = tf.keras.layers.MaxPooling1D(
            pool_size=2,
            strides=2
        )

        # 第二层卷积
        self.conv2 = tf.keras.layers.Conv1D(
            filters=2 * n_filters,  # 第二层使用2倍的卷积核
            kernel_size=3,
            strides=1,
            activation='relu',
            kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
            bias_initializer=tf.keras.initializers.Constant(0.1),
            trainable=eval_model
        )

        # 第二层池化
        self.pool2 = tf.keras.layers.MaxPooling1D(
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

        # 第一层卷积 -> 池化
        x = self.conv1(obs)
        x = self.pool1(x)

        # 第二层卷积 -> 池化
        x = self.conv2(x)
        x = self.pool2(x)

        # 展平
        x = self.flatten(x)

        # 拼接展平后的特征和目标
        output = tf.concat([x, goal], axis=1)
        return output

    def process_state(self, state):
        """
        处理输入状态，分离obs和goal
        Args:
            state: numpy array [batch, d_obs + d_goal]
        Returns:
            obs, goal: 分离后的tensor
        """
        obs_state, goal_state = np.split(
            state,
            indices_or_sections=[self.d_obs],
            axis=1
        )
        # CNN需要添加channel维度
        obs_state = np.expand_dims(obs_state, axis=-1)

        return (
            tf.convert_to_tensor(obs_state, dtype=tf.float32),
            tf.convert_to_tensor(goal_state, dtype=tf.float32)
        )


# ActorModel, CriticModel, ActionModel, ValueModel
# 与cnn.py完全相同，只是使用上面的StateModel

@model
class ActorModel(tf.keras.Model):
    """Actor模型 - 连续动作"""

    def __init__(self, special_info, eval_model: bool):
        super().__init__()
        self.eval_model = eval_model
        self.special_info = special_info

        self.state = StateModel(
            eval_model,
            special_info['size_splits'],
            special_info['n_filters']
        )

        self.actor_layers = []
        for n_layer in special_info['actor_n_layers']:
            self.actor_layers.append(
                tf.keras.layers.Dense(
                    n_layer, activation='relu',
                    kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
                    bias_initializer=tf.keras.initializers.Constant(0.1),
                    trainable=eval_model
                )
            )

        self.output_layer = tf.keras.layers.Dense(
            1, activation='tanh',
            kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
            bias_initializer=tf.keras.initializers.Constant(0.1),
            trainable=eval_model
        )

    def call(self, inputs):
        if isinstance(inputs, tuple):
            obs, goal = inputs
        else:
            obs, goal = self.state.process_state(inputs)

        layer = self.state(obs, goal)
        for actor_layer in self.actor_layers:
            layer = actor_layer(layer)
        output = self.output_layer(layer)
        return output

    def __call__(self, state):
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
        super().__init__()
        self.eval_model = eval_model
        self.special_info = special_info
        self.actor_action = action

        self.state = StateModel(
            eval_model,
            special_info['size_splits'],
            special_info['n_filters']
        )

        self.critic_layers = []
        for n_layer in special_info['critic_n_layers']:
            self.critic_layers.append(
                tf.keras.layers.Dense(
                    n_layer, activation='relu',
                    kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
                    bias_initializer=tf.keras.initializers.Constant(0.1),
                    trainable=eval_model
                )
            )

        self.output_layer = tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
            bias_initializer=tf.keras.initializers.Constant(0.1),
            trainable=eval_model
        )

    def call(self, inputs, action=None):
        if isinstance(inputs, tuple) and len(inputs) == 2:
            obs, goal = inputs
        else:
            obs, goal = self.state.process_state(inputs)

        state_features = self.state(obs, goal)

        if action is not None:
            layer = tf.concat([state_features, action], axis=1)
        else:
            layer = state_features

        for critic_layer in self.critic_layers:
            layer = critic_layer(layer)
        output = self.output_layer(layer)
        return output

    def __call__(self, state, action=None):
        if isinstance(state, dict):
            obs = state.get('obs', None)
            goal = state.get('goal', None)
            if obs is not None and goal is not None:
                return self.call((obs, goal), action)
        return self.call(state, action)


@model
class ActionModel(tf.keras.Model):
    """Action模型 - 离散动作"""

    def __init__(self, n_actions, special_info: dict, eval_model: bool):
        super().__init__()
        self.n_actions = n_actions
        self.eval_model = eval_model
        self.special_info = special_info

        self.state = StateModel(
            eval_model,
            special_info['size_splits'],
            special_info['n_filters']
        )

        self.actor_layers = []
        for n_layer in special_info['actor_n_layers']:
            self.actor_layers.append(
                tf.keras.layers.Dense(
                    n_layer, activation='relu',
                    kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
                    bias_initializer=tf.keras.initializers.Constant(0.1),
                    trainable=eval_model
                )
            )

        self.output_layer = tf.keras.layers.Dense(
            n_actions, activation='softmax',
            kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
            bias_initializer=tf.keras.initializers.Constant(0.1),
            trainable=eval_model
        )

    def call(self, inputs):
        if isinstance(inputs, tuple):
            obs, goal = inputs
        else:
            obs, goal = self.state.process_state(inputs)

        layer = self.state(obs, goal)
        for actor_layer in self.actor_layers:
            layer = actor_layer(layer)
        output = self.output_layer(layer)
        return output

    def __call__(self, state):
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
        super().__init__()
        self.n_actions = n_actions
        self.eval_model = eval_model
        self.special_info = special_info

        self.state = StateModel(
            eval_model,
            special_info['size_splits'],
            special_info['n_filters']
        )

        self.critic_layers = []
        for n_layer in special_info['critic_n_layers']:
            self.critic_layers.append(
                tf.keras.layers.Dense(
                    n_layer, activation='relu',
                    kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
                    bias_initializer=tf.keras.initializers.Constant(0.1),
                    trainable=eval_model
                )
            )

        self.output_layer = tf.keras.layers.Dense(
            n_actions,
            kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
            bias_initializer=tf.keras.initializers.Constant(0.1),
            trainable=eval_model
        )

    def call(self, inputs):
        if isinstance(inputs, tuple):
            obs, goal = inputs
        else:
            obs, goal = self.state.process_state(inputs)

        layer = self.state(obs, goal)
        for critic_layer in self.critic_layers:
            layer = critic_layer(layer)
        output = self.output_layer(layer)
        return output

    def __call__(self, state):
        if isinstance(state, dict):
            obs = state.get('obs', None)
            goal = state.get('goal', None)
            if obs is not None and goal is not None:
                return self.call((obs, goal))
        return self.call(state)