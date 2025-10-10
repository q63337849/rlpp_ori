# ===================================
# Migrated from TensorFlow 1.x to 2.x
# Original file: D:\codetest\rlpp_ori\Algo\LSTM_AC_Policy.py
# Migration may not be complete. 
# Please review TODO comments.
# ===================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class LSTMActorCritic(nn.Module):
    """
    结合LSTM和Actor-Critic的网络结构
    用于动态环境下的路径规划
    """

    def __init__(self, state_dim, obstacle_dim, action_dim,
                 lstm_hidden_dim=128, mlp_hidden_dims=[128, 64]):
        """
        Args:
            state_dim: 机器人状态维度 (位置、速度、目标等)
            obstacle_dim: 单个障碍物状态维度 (位置、速度等)
            action_dim: 动作空间维度
            lstm_hidden_dim: LSTM隐藏层维度
            mlp_hidden_dims: MLP隐藏层维度列表
        """
        super(LSTMActorCritic, self).__init__()

        self.state_dim = state_dim
        self.obstacle_dim = obstacle_dim
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # LSTM层用于编码动态障碍物序列
        self.lstm = nn.LSTM(
            input_size=obstacle_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Actor网络 (策略网络)
        actor_input_dim = state_dim + lstm_hidden_dim
        self.actor_layers = self._build_mlp(
            actor_input_dim,
            mlp_hidden_dims,
            action_dim
        )

        # Critic网络 (价值网络)
        self.critic_layers = self._build_mlp(
            actor_input_dim,
            mlp_hidden_dims,
            1  # 输出单个价值
        )

        # 初始化权重
        self._initialize_weights()

    def _build_mlp(self, input_dim, hidden_dims, output_dim):
        """构建多层感知机"""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, robot_state, obstacles_state, hidden_state=None):
        """
        前向传播
        Args:
            robot_state: [batch_size, state_dim] 机器人状态
            obstacles_state: [batch_size, num_obstacles, obstacle_dim] 障碍物序列
            hidden_state: LSTM隐藏状态 (h, c) 或 None
        Returns:
            action_probs: [batch_size, action_dim] 动作概率分布
            value: [batch_size, 1] 状态价值
            hidden_state: 新的LSTM隐藏状态
        """
        batch_size = robot_state.size(0)

        # 通过LSTM编码障碍物序列
        if obstacles_state.size(1) > 0:  # 如果有障碍物
            lstm_out, hidden_state = self.lstm(obstacles_state, hidden_state)
            # 取最后一个时间步的输出
            obstacles_encoding = lstm_out[:, -1, :]
        else:  # 如果没有障碍物
            obstacles_encoding = torch.zeros(
                batch_size, self.lstm_hidden_dim,
                device=robot_state.device
            )
            hidden_state = None

        # 拼接机器人状态和障碍物编码
        combined_state = torch.cat([robot_state, obstacles_encoding], dim=1)

        # Actor输出动作概率
        action_logits = self.actor_layers(combined_state)
        action_probs = F.softmax(action_logits, dim=-1)

        # Critic输出状态价值
        value = self.critic_layers(combined_state)

        return action_probs, value, hidden_state

    def get_action(self, robot_state, obstacles_state, hidden_state=None,
                   deterministic=False):
        """
        获取动作
        Args:
            robot_state: [batch_size, state_dim]
            obstacles_state: [batch_size, num_obstacles, obstacle_dim]
            hidden_state: LSTM隐藏状态
            deterministic: 是否使用确定性策略
        Returns:
            action: 选择的动作
            action_log_prob: 动作的对数概率
            value: 状态价值
            hidden_state: 新的隐藏状态
        """
        action_probs, value, hidden_state = self.forward(
            robot_state, obstacles_state, hidden_state
        )

        if deterministic:
            # 选择概率最大的动作
            action = torch.argmax(action_probs, dim=-1)
            action_log_prob = torch.log(
                action_probs.gather(1, action.unsqueeze(1))
            ).squeeze(1)
        else:
            # 从概率分布中采样
            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)

        return action, action_log_prob, value.squeeze(1), hidden_state

    def evaluate_actions(self, robot_state, obstacles_state, actions,
                         hidden_state=None):
        """
        评估动作
        Args:
            robot_state: [batch_size, state_dim]
            obstacles_state: [batch_size, num_obstacles, obstacle_dim]
            actions: [batch_size] 执行的动作
            hidden_state: LSTM隐藏状态
        Returns:
            action_log_probs: 动作的对数概率
            values: 状态价值
            entropy: 策略熵
        """
        action_probs, values, _ = self.forward(
            robot_state, obstacles_state, hidden_state
        )

        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return action_log_probs, values.squeeze(1), entropy


class LSTMAC_Agent:
    """
    LSTM-AC智能体,用于训练和决策
    """

    def __init__(self, state_dim, obstacle_dim, action_dim,
                 lstm_hidden_dim=128, mlp_hidden_dims=[128, 64],
                 lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, value_loss_coef=0.5,
                 entropy_coef=0.01, max_grad_norm=0.5,
                 device='cuda'):
        """
        Args:
            state_dim: 状态维度
            obstacle_dim: 障碍物维度
            action_dim: 动作维度
            lstm_hidden_dim: LSTM隐藏维度
            mlp_hidden_dims: MLP隐藏维度
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            clip_epsilon: PPO裁剪参数
            value_loss_coef: 价值损失系数
            entropy_coef: 熵损失系数
            max_grad_norm: 梯度裁剪
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 创建网络
        self.policy = LSTMActorCritic(
            state_dim, obstacle_dim, action_dim,
            lstm_hidden_dim, mlp_hidden_dims
        ).to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # 存储经验
        self.reset_buffers()

    def reset_buffers(self):
        """重置经验缓冲区"""
        self.states = []
        self.obstacles = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def select_action(self, robot_state, obstacles_state,
                      hidden_state=None, deterministic=False):
        """
        选择动作
        Args:
            robot_state: numpy array [state_dim]
            obstacles_state: numpy array [num_obstacles, obstacle_dim]
            hidden_state: LSTM隐藏状态
            deterministic: 是否确定性
        Returns:
            action: int
            hidden_state: 新的隐藏状态
        """
        # 转换为tensor
        robot_state_tensor = torch.FloatTensor(robot_state).unsqueeze(0).to(self.device)
        obstacles_state_tensor = torch.FloatTensor(obstacles_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value, hidden_state = self.policy.get_action(
                robot_state_tensor, obstacles_state_tensor,
                hidden_state, deterministic
            )

        # 存储经验
        if not deterministic:
            self.states.append(robot_state)
            self.obstacles.append(obstacles_state)
            self.actions.append(action.item())
            self.log_probs.append(log_prob.item())
            self.values.append(value.item())

        return action.item(), hidden_state

    def store_transition(self, reward, done):
        """存储转移"""
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, next_value):
        """
        计算广义优势估计(GAE)
        Args:
            next_value: 下一个状态的价值
        Returns:
            advantages: 优势函数
            returns: 回报
        """
        values = self.values + [next_value]
        advantages = []
        gae = 0

        for t in reversed(range(len(self.rewards))):
            delta = (self.rewards[t] +
                     self.gamma * values[t + 1] * (1 - self.dones[t]) -
                     values[t])
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.values)]

        return advantages, returns

    def update(self, next_robot_state, next_obstacles_state,
               hidden_state=None, epochs=10, batch_size=64):
        """
        更新策略
        Args:
            next_robot_state: 下一个机器人状态
            next_obstacles_state: 下一个障碍物状态
            hidden_state: 隐藏状态
            epochs: 更新轮数
            batch_size: 批次大小
        Returns:
            loss_info: 损失信息字典
        """
        if len(self.states) == 0:
            return {}

        # 计算下一个状态的价值
        with torch.no_grad():
            next_robot_state_tensor = torch.FloatTensor(next_robot_state).unsqueeze(0).to(self.device)
            next_obstacles_state_tensor = torch.FloatTensor(next_obstacles_state).unsqueeze(0).to(self.device)
            _, next_value, _ = self.policy(
                next_robot_state_tensor,
                next_obstacles_state_tensor,
                hidden_state
            )
            next_value = next_value.item()

        # 计算GAE
        advantages, returns = self.compute_gae(next_value)

        # 转换为tensor
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        obstacles_tensor = torch.FloatTensor(np.array(self.obstacles)).to(self.device)
        actions_tensor = torch.LongTensor(self.actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # 标准化优势
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std() + 1e-8
        )

        # 训练多个epochs
        dataset_size = len(self.states)
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0

        for _ in range(epochs):
            # 随机打乱数据
            indices = torch.randperm(dataset_size)

            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                # 获取批次数据
                batch_states = states_tensor[batch_indices]
                batch_obstacles = obstacles_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # 评估动作
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_obstacles, batch_actions
                )

                # 计算比率
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # PPO裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = F.mse_loss(values, batch_returns)

                # 熵损失
                entropy_loss = -entropy.mean()

                # 总损失
                loss = (policy_loss +
                        self.value_loss_coef * value_loss +
                        self.entropy_coef * entropy_loss)

                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()

                # 记录损失
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                update_count += 1

        # 清空缓冲区
        self.reset_buffers()

        # 返回损失信息
        return {
            'total_loss': total_loss / update_count,
            'policy_loss': total_policy_loss / update_count,
            'value_loss': total_value_loss / update_count,
            'entropy': total_entropy / update_count
        }

    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])