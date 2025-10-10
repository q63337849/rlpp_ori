"""
ActionValue Supervisor - TensorFlow 2.x版本
完全重写为Eager Execution模式
"""
import tensorflow as tf
import pickle
import numpy as np


class ActionValueSupervisor:
    model_names = ['ActionModel', 'ValueModel']

    def __init__(self,
                 n_actions,
                 ActionModel,
                 action_special_info,
                 ValueModel,
                 value_special_info,
                 actor_lr=0.001,
                 critic_lr=0.01):
        """
        Action-Value监督器

        Args:
            n_actions: 动作数量
            ActionModel: Action网络类
            action_special_info: Action特殊信息
            ValueModel: Value网络类
            value_special_info: Value特殊信息
            actor_lr: Actor学习率
            critic_lr: Critic学习率
        """
        self.n_actions = n_actions
        self.ActionModel = ActionModel
        self.action_special_info = action_special_info
        self.ValueModel = ValueModel
        self.value_special_info = value_special_info
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.build = False

        # 模型实例（延迟初始化）
        self.Actor = None
        self.Critic = None
        self.TargetActor = None
        self.TargetCritic = None

        # 优化器
        self.actor_optimizer = None
        self.critic_optimizer = None

    def action(self, state):
        """
        获取Actor的动作输出

        Args:
            state: 状态

        Returns:
            动作输出
        """
        if not self.build:
            raise RuntimeError("模型未构建，请先调用 build_model()")

        if not isinstance(state, dict):
            output = self.Actor(state)
        else:
            output = self.Actor(**state)

        return output.numpy() if hasattr(output, 'numpy') else output

    def value(self, state):
        """
        获取Critic的价值输出

        Args:
            state: 状态

        Returns:
            价值输出
        """
        if not self.build:
            raise RuntimeError("模型未构建，请先调用 build_model()")

        if not isinstance(state, dict):
            output = self.Critic(state)
        else:
            output = self.Critic(**state)

        return output.numpy() if hasattr(output, 'numpy') else output

    @tf.function
    def _train_critic_step(self, state_tensors, value_target):
        """
        Critic训练步骤（使用tf.function加速）

        Args:
            state_tensors: 状态张量
            value_target: 目标价值

        Returns:
            td_error: TD误差
            critic_loss: Critic损失
        """
        with tf.GradientTape() as tape:
            # 前向传播
            if isinstance(state_tensors, dict):
                critic_output = self.Critic(**state_tensors)
            else:
                critic_output = self.Critic(state_tensors)

            # 计算TD误差和损失
            td_error = value_target - critic_output
            critic_loss = tf.reduce_mean(tf.square(td_error))

        # 反向传播
        gradients = tape.gradient(critic_loss, self.Critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(gradients, self.Critic.trainable_variables)
        )

        return td_error, critic_loss

    def train_critic(self, state, value_target):
        """
        训练Critic

        Args:
            state: 状态
            value_target: 目标价值

        Returns:
            critic_loss: Critic损失
            td_error: TD误差
        """
        if not self.build:
            raise RuntimeError("模型未构建，请先调用 build_model()")

        # 转换为tensor
        if not isinstance(state, dict):
            state_tensors = tf.convert_to_tensor(state, dtype=tf.float32)
        else:
            state_tensors = {k: tf.convert_to_tensor(v, dtype=tf.float32)
                             for k, v in state.items()}

        value_target = tf.convert_to_tensor(value_target, dtype=tf.float32)

        # 训练
        td_error, critic_loss = self._train_critic_step(state_tensors, value_target)

        return critic_loss.numpy(), td_error.numpy()

    @tf.function
    def _train_actor_step(self, state_tensors, action, decayed_td_error):
        """
        Actor训练步骤（使用tf.function加速）

        Args:
            state_tensors: 状态张量
            action: 动作
            decayed_td_error: 衰减的TD误差

        Returns:
            actor_loss: Actor损失
        """
        with tf.GradientTape() as tape:
            # 前向传播
            if isinstance(state_tensors, dict):
                actor_output = self.Actor(**state_tensors)
            else:
                actor_output = self.Actor(state_tensors)

            # 确保action是正确的shape和类型
            action = tf.cast(action, tf.int32)

            # 处理不同维度的action输入
            if len(action.shape) == 0:  # 标量 []
                action = tf.expand_dims(action, axis=0)  # 变成 [1]
            elif len(action.shape) == 2:  # [batch_size, 1]
                action = tf.squeeze(action, axis=-1)  # 变成 [batch_size]
            # 如果已经是 [batch_size]，则不需要改变

            # 确保是1D
            action = tf.reshape(action, [-1])

            # 计算Actor损失（策略梯度）
            # 假设actor_output是概率分布，action是动作索引
            batch_size = tf.shape(actor_output)[0]
            indices = tf.stack([tf.range(batch_size), action], axis=1)
            action_prob = tf.gather_nd(actor_output, indices)

            # 负对数似然加权TD误差
            actor_loss = -tf.reduce_mean(
                tf.math.log(action_prob + 1e-10) * tf.stop_gradient(decayed_td_error)
            )

        # 反向传播
        gradients = tape.gradient(actor_loss, self.Actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(gradients, self.Actor.trainable_variables)
        )

        return actor_loss

    def train_actor(self, state, action, decayed_td_error):
        """
        训练Actor

        Args:
            state: 状态
            action: 动作
            decayed_td_error: 衰减的TD误差

        Returns:
            actor_loss: Actor损失
            neg_actor_loss: 负Actor损失（为了兼容旧接口）
        """
        if not self.build:
            raise RuntimeError("模型未构建，请先调用 build_model()")

        # 转换为tensor
        if not isinstance(state, dict):
            state_tensors = tf.convert_to_tensor(state, dtype=tf.float32)
        else:
            state_tensors = {k: tf.convert_to_tensor(v, dtype=tf.float32)
                             for k, v in state.items()}

        action = tf.convert_to_tensor(action, dtype=tf.int32)
        decayed_td_error = tf.convert_to_tensor(decayed_td_error, dtype=tf.float32)

        # 训练
        actor_loss = self._train_actor_step(state_tensors, action, decayed_td_error)

        return actor_loss.numpy(), -actor_loss.numpy()

    def _build_model(self):
        """内部构建模型"""
        if self.build:
            print("警告：模型已构建，跳过重复构建")
            return

        self.build = True

        # 创建模型实例
        self.Actor = self.ActionModel(
            self.n_actions,
            self.action_special_info,
            eval_model=True
        )

        # Critic输出状态价值（不是动作价值）
        self.Critic = self.ValueModel(
            1,  # 状态价值是标量
            self.value_special_info,
            eval_model=True
        )

        # 创建Target模型
        self.TargetActor = self.ActionModel(
            self.n_actions,
            self.action_special_info,
            eval_model=False
        )

        self.TargetCritic = self.ValueModel(
            1,
            self.value_special_info,
            eval_model=False
        )

        # 创建优化器
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

    def build_model(self):
        """构建模型"""
        self._build_model()

    def load_model(self, path):
        """
        加载模型权重

        Args:
            path: 模型文件路径
        """
        if not self.build:
            self._build_model()

        with open(path, 'rb') as fp:
            c_weights, a_weights = pickle.load(fp)

        # 加载Critic权重
        for var, weight in zip(self.Critic.trainable_variables, c_weights):
            var.assign(weight)

        # 加载Actor权重
        for var, weight in zip(self.Actor.trainable_variables, a_weights):
            var.assign(weight)

    def save_model(self, path):
        """
        保存模型权重

        Args:
            path: 保存路径
        """
        if not self.build:
            raise RuntimeError("模型未构建，无法保存")

        # 获取权重
        c_weights = [var.numpy() for var in self.Critic.trainable_variables]
        a_weights = [var.numpy() for var in self.Actor.trainable_variables]

        # 保存
        with open(path, 'wb') as fp:
            pickle.dump((c_weights, a_weights), fp)