"""
ActorCritic Supervisor - TensorFlow 2.x版本
完全重写为Eager Execution模式
"""
import tensorflow as tf
import pickle
import numpy as np


class ActorCriticSupervisor:
    model_names = ['ActorModel', 'CriticModel']

    def __init__(self,
                 ActorModel,
                 actor_special_info,
                 CriticModel,
                 critic_special_info,
                 actor_lr=0.001,
                 critic_lr=0.01):
        """
        Actor-Critic监督器

        Args:
            ActorModel: Actor网络类
            actor_special_info: Actor特殊信息
            CriticModel: Critic网络类
            critic_special_info: Critic特殊信息
            actor_lr: Actor学习率
            critic_lr: Critic学习率
        """
        self.ActorModel = ActorModel
        self.actor_special_info = actor_special_info
        self.CriticModel = CriticModel
        self.critic_special_info = critic_special_info
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
            state: 状态，格式需要匹配Actor模型的输入

        Returns:
            动作输出
        """
        if not self.build:
            raise RuntimeError("模型未构建，请先调用 build_model()")

        # 确保state是tensor
        if not isinstance(state, dict):
            # 如果state不是字典，调用Actor的__call__方法
            output = self.Actor(state)
        else:
            # 如果是字典形式（包含obs, goal等）
            output = self.Actor(**state)

        return output.numpy() if hasattr(output, 'numpy') else output

    def target_action(self, state):
        """
        获取Target Actor的动作输出

        Args:
            state: 状态

        Returns:
            目标动作输出
        """
        if not self.build:
            raise RuntimeError("模型未构建，请先调用 build_model()")

        if not isinstance(state, dict):
            output = self.TargetActor(state)
        else:
            output = self.TargetActor(**state)

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

        # 先获取Actor的动作
        if not isinstance(state, dict):
            action = self.Actor(state)
            # Critic需要state和action
            value = self.Critic(state, action)
        else:
            action = self.Actor(**state)
            # 将state字典和action合并
            value = self.Critic(action=action, **state)

        return value.numpy() if hasattr(value, 'numpy') else value

    def target_value(self, state):
        """
        获取Target Critic的价值输出

        Args:
            state: 状态

        Returns:
            目标价值输出
        """
        if not self.build:
            raise RuntimeError("模型未构建，请先调用 build_model()")

        # 先获取Target Actor的动作
        if not isinstance(state, dict):
            action = self.TargetActor(state)
            value = self.TargetCritic(state, action)
        else:
            action = self.TargetActor(**state)
            value = self.TargetCritic(action=action, **state)

        return value.numpy() if hasattr(value, 'numpy') else value

    @tf.function
    def _train_critic_step(self, state_tensors, action, value_target):
        """
        Critic训练步骤（使用tf.function加速）

        Args:
            state_tensors: 状态张量（字典或单个tensor）
            action: 动作
            value_target: 目标价值

        Returns:
            critic_loss: Critic损失
        """
        with tf.GradientTape() as tape:
            # 前向传播
            if isinstance(state_tensors, dict):
                critic_output = self.Critic(action=action, **state_tensors)
            else:
                critic_output = self.Critic(state_tensors, action)

            # 计算损失
            critic_loss = tf.reduce_mean(tf.square(value_target - critic_output))

        # 反向传播
        gradients = tape.gradient(critic_loss, self.Critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(gradients, self.Critic.trainable_variables)
        )

        return critic_loss

    def train_critic(self, state, action, value_target):
        """
        训练Critic

        Args:
            state: 状态
            action: 动作
            value_target: 目标价值

        Returns:
            critic_loss: Critic损失
            placeholder: 占位符（为了兼容旧接口）
        """
        if not self.build:
            raise RuntimeError("模型未构建，请先调用 build_model()")

        # 转换为tensor
        if not isinstance(state, dict):
            state_tensors = tf.convert_to_tensor(state, dtype=tf.float32)
        else:
            state_tensors = {k: tf.convert_to_tensor(v, dtype=tf.float32)
                             for k, v in state.items()}

        action = tf.convert_to_tensor(action, dtype=tf.float32)
        value_target = tf.convert_to_tensor(value_target, dtype=tf.float32)

        # 训练
        critic_loss = self._train_critic_step(state_tensors, action, value_target)

        return critic_loss.numpy(), 0.0

    @tf.function
    def _train_actor_step(self, state_tensors):
        """
        Actor训练步骤（使用tf.function加速）

        Args:
            state_tensors: 状态张量

        Returns:
            actor_loss: Actor损失
        """
        with tf.GradientTape() as tape:
            # 前向传播
            if isinstance(state_tensors, dict):
                action = self.Actor(**state_tensors)
                critic_output = self.Critic(action=action, **state_tensors)
            else:
                action = self.Actor(state_tensors)
                critic_output = self.Critic(state_tensors, action)

            # Actor损失：最大化Q值（最小化负Q值）
            actor_loss = -tf.reduce_mean(critic_output)

        # 反向传播
        gradients = tape.gradient(actor_loss, self.Actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(gradients, self.Actor.trainable_variables)
        )

        return actor_loss

    def train_actor(self, state):
        """
        训练Actor

        Args:
            state: 状态

        Returns:
            actor_loss: Actor损失
            placeholder: 占位符（为了兼容旧接口）
        """
        if not self.build:
            raise RuntimeError("模型未构建，请先调用 build_model()")

        # 转换为tensor
        if not isinstance(state, dict):
            state_tensors = tf.convert_to_tensor(state, dtype=tf.float32)
        else:
            state_tensors = {k: tf.convert_to_tensor(v, dtype=tf.float32)
                             for k, v in state.items()}

        # 训练
        actor_loss = self._train_actor_step(state_tensors)

        return actor_loss.numpy(), 0.0

    def build_model(self):
        """构建模型并同步权重"""
        self._build_model()
        self.synchronize_weights()

    def _build_model(self):
        """内部构建模型"""
        if self.build:
            print("警告：模型已构建，跳过重复构建")
            return

        self.build = True

        # 创建模型实例
        self.Actor = self.ActorModel(self.actor_special_info, eval_model=True)
        self.Critic = self.CriticModel(self.Actor, self.critic_special_info, eval_model=True)

        # 创建Target模型
        self.TargetActor = self.ActorModel(self.actor_special_info, eval_model=False)
        self.TargetCritic = self.CriticModel(self.TargetActor, self.critic_special_info, eval_model=False)

        # 创建优化器
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

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

        # 同步到Target网络
        self.synchronize_weights()

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

    def synchronize_weights(self, tau=1.0):
        """
        同步目标网络权重

        Args:
            tau: 软更新参数（1.0表示完全复制，0.0表示不更新）
        """
        if not self.build:
            raise RuntimeError("模型未构建，无法同步权重")

        # 更新Target Critic
        for target_var, source_var in zip(self.TargetCritic.trainable_variables,
                                          self.Critic.trainable_variables):
            target_var.assign(tau * source_var + (1.0 - tau) * target_var)

        # 更新Target Actor
        for target_var, source_var in zip(self.TargetActor.trainable_variables,
                                          self.Actor.trainable_variables):
            target_var.assign(tau * source_var + (1.0 - tau) * target_var)