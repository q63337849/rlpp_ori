"""
Value Supervisor - TensorFlow 2.x版本
完全重写为Eager Execution模式
"""
import tensorflow as tf
import pickle
import numpy as np


class ValueSupervisor:
    model_names = ['ValueModel']  # 依赖的模型名

    def __init__(self,
                 n_actions,
                 ValueModel,
                 value_special_info,
                 critic_lr=0.001):
        """
        Value监督器

        Args:
            n_actions: 动作数量
            ValueModel: Value网络类
            value_special_info: Value特殊信息
            critic_lr: Critic学习率
        """
        self.ValueModel = ValueModel
        self.n_actions = n_actions
        self.value_special_info = value_special_info
        self.critic_lr = critic_lr
        self.build = False

        # 模型实例（延迟初始化）
        self.Value = None
        self.TargetValue = None

        # 优化器
        self.optimizer = None

    def _build_model(self):
        """内部构建模型"""
        if self.build:
            print("警告：模型已构建，跳过重复构建")
            return

        self.build = True

        # 创建Value模型
        self.Value = self.ValueModel(
            self.n_actions,
            self.value_special_info,
            eval_model=True
        )

        # 创建Target Value模型
        self.TargetValue = self.ValueModel(
            self.n_actions,
            self.value_special_info,
            eval_model=False
        )

        # 创建优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

    def build_model(self):
        """构建模型并同步权重"""
        self._build_model()
        self.synchronize_weights()

    def load_model(self, path):
        """
        加载模型权重

        Args:
            path: 模型文件路径
        """
        if not self.build:
            self._build_model()

        with open(path, 'rb') as fp:
            weights = pickle.load(fp)

        # 加载权重
        for var, weight in zip(self.Value.trainable_variables, weights):
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
        weights = [var.numpy() for var in self.Value.trainable_variables]

        # 保存
        with open(path, 'wb') as fp:
            pickle.dump(weights, fp)

    @tf.function
    def _train_step(self, state_tensors, value_target):
        """
        训练步骤（使用tf.function加速）

        Args:
            state_tensors: 状态张量
            value_target: 目标价值

        Returns:
            loss: 损失值
        """
        with tf.GradientTape() as tape:
            # 前向传播
            if isinstance(state_tensors, dict):
                value_output = self.Value(**state_tensors)
            else:
                value_output = self.Value(state_tensors)

            # 计算损失
            loss = tf.reduce_mean(tf.square(value_target - value_output))

        # 反向传播
        gradients = tape.gradient(loss, self.Value.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.Value.trainable_variables)
        )

        return loss

    def train(self, state, value_target):
        """
        训练Value网络

        Args:
            state: 状态
            value_target: 目标价值

        Returns:
            loss: 损失值
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

        value_target = tf.convert_to_tensor(value_target, dtype=tf.float32)

        # 训练
        loss = self._train_step(state_tensors, value_target)

        return loss.numpy(), 0.0

    def value(self, state):
        """
        获取Value网络的价值输出

        Args:
            state: 状态

        Returns:
            价值输出
        """
        if not self.build:
            raise RuntimeError("模型未构建，请先调用 build_model()")

        # 前向传播
        if not isinstance(state, dict):
            output = self.Value(state)
        else:
            output = self.Value(**state)

        return output.numpy() if hasattr(output, 'numpy') else output

    def target_value(self, state):
        """
        获取Target Value网络的价值输出

        Args:
            state: 状态

        Returns:
            目标价值输出
        """
        if not self.build:
            raise RuntimeError("模型未构建，请先调用 build_model()")

        # 前向传播
        if not isinstance(state, dict):
            output = self.TargetValue(state)
        else:
            output = self.TargetValue(**state)

        return output.numpy() if hasattr(output, 'numpy') else output

    def synchronize_weights(self):
        """同步目标网络权重（完全复制）"""
        if not self.build:
            raise RuntimeError("模型未构建，无法同步权重")

        # 完全复制权重到Target网络
        for target_var, source_var in zip(self.TargetValue.trainable_variables,
                                          self.Value.trainable_variables):
            target_var.assign(source_var)