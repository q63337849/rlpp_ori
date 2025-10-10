"""
MLP模型 - TensorFlow 2.x版本
完全重写为Keras风格
"""
import numpy as np
import tensorflow as tf
from .util import model


class StateModel(tf.keras.Model):
    """状态处理模型"""

    def __init__(self, eval_model: bool, size_splits, n_layers):
        """
        Args:
            eval_model: 是否为评估模型（用于区分eval和target）
            size_splits: [obs_dim, goal_dim] 观测和目标的维度分割
            n_layers: 隐藏层节点数列表
        """
        super().__init__()
        self.d_obs, self.d_goal = size_splits
        self.eval_model = eval_model

        # 构建MLP层
        self.dense_layers = []
        for n_layer in n_layers:
            self.dense_layers.append(
                tf.keras.layers.Dense(
                    n_layer,
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
                    bias_initializer=tf.keras.initializers.Constant(0.1),
                    trainable=eval_model
                )
            )

    def call(self, obs, goal):
        """
        前向传播
        Args:
            obs: 观测 [batch, d_obs]
            goal: 目标 [batch, d_goal]
        Returns:
            拼接后的特征 [batch, feature_dim + d_goal]
        """
        layer = obs
        for dense_layer in self.dense_layers:
            layer = dense_layer(layer)
        # 拼接处理后的观测和原始目标
        output = tf.concat([layer, goal], axis=1)
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
        return (
            tf.convert_to_tensor(obs_state, dtype=tf.float32),
            tf.convert_to_tensor(goal_state, dtype=tf.float32)
        )


@model
class ActorModel(tf.keras.Model):
    """Actor模型 - 连续动作"""

    def __init__(self, special_info, eval_model: bool):
        """
        Args:
            special_info: 包含size_splits, state_n_layers, actor_n_layers的字典
            eval_model: 是否为评估模型
        """
        super().__init__()
        self.eval_model = eval_model
        self.special_info = special_info

        # 状态处理层
        self.state = StateModel(
            eval_model,
            special_info['size_splits'],
            special_info['state_n_layers']
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
            # 如果是字典形式（旧TF1.x格式）
            obs = state.get('obs', state.get(self.state.obs, None))
            goal = state.get('goal', state.get(self.state.goal, None))
            if obs is not None and goal is not None:
                return self.call((obs, goal))
        return self.call(state)


@model
class CriticModel(tf.keras.Model):
    """Critic模型 - Q值估计"""

    def __init__(self, action, special_info: dict, eval_model: bool):
        """
        Args:
            action: Actor模型输出（或占位符）
            special_info: 包含size_splits, state_n_layers, critic_n_layers的字典
            eval_model: 是否为评估模型
        """
        super().__init__()
        self.eval_model = eval_model
        self.special_info = special_info
        self.actor_action = action  # 保存引用，但在TF2中不使用

        # 状态处理层
        self.state = StateModel(
            eval_model,
            special_info['size_splits'],
            special_info['state_n_layers']
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
            special_info: 包含size_splits, state_n_layers, actor_n_layers的字典
            eval_model: 是否为评估模型
        """
        super().__init__()
        self.n_actions = n_actions
        self.eval_model = eval_model
        self.special_info = special_info

        # 状态处理层
        self.state = StateModel(
            eval_model,
            special_info['size_splits'],
            special_info['state_n_layers']
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
            special_info: 包含size_splits, state_n_layers, critic_n_layers的字典
            eval_model: 是否为评估模型
        """
        super().__init__()
        self.n_actions = n_actions
        self.eval_model = eval_model
        self.special_info = special_info

        # 状态处理层
        self.state = StateModel(
            eval_model,
            special_info['size_splits'],
            special_info['state_n_layers']
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