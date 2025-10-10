import tensorflow as tf
import numpy as np


class AC:
    off_policy = False
    def __init__(self, model, decay_rate=0.95, gamma=0.9, tau=0.05, sigma=0.1, **kwargs):
        self.model = model
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.current_decay = 1.
        self.off_policy = False

    def sample(self, state):
        """TF 2.x 版本"""
        action_probs = self.model.action(np.expand_dims(state, axis=0))

        # TF 2.x: 如果是 Tensor 直接调用 .numpy()
        if isinstance(action_probs, tf.Tensor):
            action_probs = action_probs.numpy()

        action_probs = np.squeeze(action_probs, axis=0)
        return np.random.choice(np.arange(len(action_probs)), p=action_probs)

    def predict(self, state):
        """TF 2.x 版本"""
        action_probs = self.model.action(np.expand_dims(state, axis=0))

        # TF 2.x: 如果是 Tensor 直接调用 .numpy()
        if isinstance(action_probs, tf.Tensor):
            action_probs = action_probs.numpy()

        return np.argmax(action_probs[0])

    def learn(self, *exp):
        """TF 2.x 版本"""
        s, a, r, s_, d = exp
        s, s_ = np.expand_dims(s, axis=0), np.expand_dims(s_, axis=0)

        # 获取 value
        value_s_ = self.model.value(s_)
        if isinstance(value_s_, tf.Tensor):
            value_s_ = value_s_.numpy()

        target_value = r + (not d) * self.gamma * value_s_

        # 训练 critic
        critic_loss, td_error = self.model.train_critic(s, target_value)

        # 转换 td_error
        if isinstance(td_error, tf.Tensor):
            td_error = td_error.numpy()

        # 训练 actor
        actor_loss, actor_metric = self.model.train_actor(s, a, td_error * self.current_decay)

        self.current_decay *= self.decay_rate
        return critic_loss, actor_loss

    def reset(self):
        self.current_decay = 1.