# ===================================
# Migrated from TensorFlow 1.x to 2.x
# Original file: D:\codetest\rlpp_ori\auto_migrate.py
# Migration may not be complete. 
# Please review TODO comments.
# ===================================

"""
自动迁移TensorFlow 1.x代码到2.x
使用方法: python auto_migrate.py --input_dir ./your_project --output_dir ./your_project_v2
"""

import os
import re
import argparse
import shutil
from pathlib import Path


class TFMigrator:
    """TensorFlow 1.x → 2.x 代码迁移工具"""
    
    def __init__(self):
        # 定义替换规则
        self.replacements = [
            # Session相关
            (r'tf\.Session\(\)', 'tf.compat.v1.Session()  # TODO: Remove session, use eager execution'),
            (r'sess\.run\((.*?)\)', r'# TODO: Convert to eager execution\n\1'),
            (r'tf\.placeholder\((.*?)\)', r'# TODO: Remove placeholder, use direct input\ntf.Variable(\1)'),
            
            # 优化器
            (r'tf\.train\.AdamOptimizer', 'tf.keras.optimizers.Adam'),
            (r'tf\.train\.GradientDescentOptimizer', 'tf.keras.optimizers.SGD'),
            (r'tf\.train\.RMSPropOptimizer', 'tf.keras.optimizers.RMSprop'),
            (r'tf\.train\.AdagradOptimizer', 'tf.keras.optimizers.Adagrad'),
            
            # 层
            (r'tf\.layers\.dense', 'tf.keras.layers.Dense'),
            (r'tf\.layers\.conv2d', 'tf.keras.layers.Conv2D'),
            (r'tf\.layers\.conv1d', 'tf.keras.layers.Conv1D'),
            (r'tf\.layers\.max_pooling2d', 'tf.keras.layers.MaxPooling2D'),
            (r'tf\.layers\.average_pooling2d', 'tf.keras.layers.AveragePooling2D'),
            (r'tf\.layers\.dropout', 'tf.keras.layers.Dropout'),
            (r'tf\.layers\.batch_normalization', 'tf.keras.layers.BatchNormalization'),
            (r'tf\.layers\.flatten', 'tf.keras.layers.Flatten'),
            
            # 损失函数
            (r'tf\.losses\.mean_squared_error', 'tf.keras.losses.MSE'),
            (r'tf\.losses\.absolute_difference', 'tf.keras.losses.MAE'),
            (r'tf\.nn\.softmax_cross_entropy_with_logits_v2', 'tf.nn.softmax_cross_entropy_with_logits'),
            
            # 初始化器
            (r'tf\.glorot_uniform_initializer', 'tf.keras.initializers.GlorotUniform'),
            (r'tf\.glorot_normal_initializer', 'tf.keras.initializers.GlorotNormal'),
            (r'tf\.random_normal_initializer', 'tf.keras.initializers.RandomNormal'),
            (r'tf\.random_uniform_initializer', 'tf.keras.initializers.RandomUniform'),
            (r'tf\.truncated_normal_initializer', 'tf.keras.initializers.TruncatedNormal'),
            
            # 其他常用函数
            (r'tf\.contrib', 'tf.compat.v1.contrib  # TODO: Find TF2 equivalent'),
            (r'tf\.get_variable', 'tf.Variable  # TODO: Adjust variable creation'),
            (r'tf\.variable_scope', '# TODO: Replace with class-based approach'),
            (r'tf\.global_variables_initializer\(\)', '# Not needed in TF2 eager mode'),
        ]
    
    def migrate_file(self, file_path, output_path):
        """迁移单个文件"""
        print(f"Migrating: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 添加迁移注释
            migrated_content = f"""# ===================================
# Migrated from TensorFlow 1.x to 2.x
# Original file: {file_path}
# Migration may not be complete. 
# Please review TODO comments.
# ===================================

"""
            
            # 应用所有替换规则
            migrated_content += content
            for pattern, replacement in self.replacements:
                migrated_content = re.sub(pattern, replacement, migrated_content)
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 写入迁移后的文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(migrated_content)
            
            print(f"✓ Migrated to: {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ Error migrating {file_path}: {e}")
            return False
    
    def migrate_directory(self, input_dir, output_dir):
        """迁移整个目录"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # 统计
        total_files = 0
        migrated_files = 0
        skipped_files = 0
        
        # 遍历所有Python文件
        for py_file in input_path.rglob('*.py'):
            total_files += 1
            
            # 计算相对路径
            rel_path = py_file.relative_to(input_path)
            out_file = output_path / rel_path
            
            # 迁移文件
            if self.migrate_file(str(py_file), str(out_file)):
                migrated_files += 1
            else:
                skipped_files += 1
        
        # 复制非Python文件
        for item in input_path.rglob('*'):
            if item.is_file() and item.suffix != '.py':
                rel_path = item.relative_to(input_path)
                out_item = output_path / rel_path
                os.makedirs(os.path.dirname(out_item), exist_ok=True)
                shutil.copy2(str(item), str(out_item))
        
        # 打印统计信息
        print("\n" + "=" * 60)
        print("Migration Summary")
        print("=" * 60)
        print(f"Total Python files: {total_files}")
        print(f"Successfully migrated: {migrated_files}")
        print(f"Skipped: {skipped_files}")
        print("=" * 60)
        
        # 生成迁移报告
        self.generate_report(output_dir)
    
    def generate_report(self, output_dir):
        """生成迁移报告"""
        report_path = Path(output_dir) / 'MIGRATION_REPORT.md'
        
        report = """# TensorFlow Migration Report

## Migration Status

This directory contains code migrated from TensorFlow 1.x to 2.x.

## Important Notes

1. **Review TODO Comments**: Search for `# TODO:` comments in the migrated files
2. **Test Thoroughly**: Run all tests to ensure functionality is preserved
3. **Manual Fixes Required**: Some patterns cannot be automatically converted

## Common Manual Fixes

### 1. Remove Session Usage

**Before (TF 1.x):**
```python
sess = tf.compat.v1.Session()  # TODO: Remove session, use eager execution
result = # TODO: Convert to eager execution
output, feed_dict={input_ph: data}
```

**After (TF 2.x):**
```python
# Direct execution in eager mode
result = model(data)
```

### 2. Replace Placeholders

**Before (TF 1.x):**
```python
x = # TODO: Remove placeholder, use direct input
tf.Variable(tf.float32, [None, 784])
y = model(x)
```

**After (TF 2.x):**
```python
@tf.function
def forward(x):
    return model(x)

y = forward(x_data)
```

### 3. Use GradientTape for Training

**Before (TF 1.x):**
```python
optimizer = tf.keras.optimizers.Adam(0.001)
train_op = optimizer.minimize(loss)
# TODO: Convert to eager execution
train_op
```

**After (TF 2.x):**
```python
optimizer = tf.keras.optimizers.Adam(0.001)
with tf.GradientTape() as tape:
    loss = compute_loss()
gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

## Next Steps

1. Review all migrated files
2. Fix TODO comments
3. Run tests: `python -m pytest tests/`
4. Update documentation
5. Commit changes

## Resources

- [TensorFlow Migration Guide](https://www.tensorflow.org/guide/migrate)
- [Effective TensorFlow 2](https://www.tensorflow.org/guide/effective_tf2)

---
*Generated by auto_migrate.py*
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✓ Migration report saved to: {report_path}")


def create_test_script(output_dir):
    """创建测试脚本"""
    test_script = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"
测试迁移后的TensorFlow 2.x环境
\"\"\"

import sys
import tensorflow as tf
import numpy as np

def test_basic():
    \"\"\"基础功能测试\"\"\"
    print("Testing TensorFlow 2.x basic functionality...")
    
    # 测试eager execution
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    z = tf.matmul(x, y)
    
    print(f"✓ Eager execution works")
    print(f"  Result shape: {z.shape}")
    
    return True

def test_model():
    \"\"\"模型测试\"\"\"
    print("\\nTesting model creation and training...")
    
    # 创建简单模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    
    # 编译
    model.compile(optimizer='adam', loss='mse')
    
    # 训练
    x_train = np.random.random((100, 5))
    y_train = np.random.random((100, 1))
    
    history = model.fit(x_train, y_train, epochs=2, verbose=0)
    
    print(f"✓ Model training works")
    print(f"  Final loss: {history.history['loss'][-1]:.4f}")
    
    return True

def test_gradient_tape():
    \"\"\"GradientTape测试\"\"\"
    print("\\nTesting GradientTape...")
    
    x = tf.Variable(3.0)
    
    with tf.GradientTape() as tape:
        y = x ** 2
    
    dy_dx = tape.gradient(y, x)
    
    print(f"✓ GradientTape works")
    print(f"  d(x²)/dx at x=3: {dy_dx.numpy()}")
    
    return True

def test_gpu():
    \"\"\"GPU测试\"\"\"
    print("\\nTesting GPU availability...")
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"✓ Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("⚠ No GPU found, using CPU")
    
    return True

def main():
    print("=" * 60)
    print("TensorFlow 2.x Environment Test")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {sys.version}")
    print("=" * 60)
    
    tests = [
        test_basic,
        test_model,
        test_gradient_tape,
        test_gpu
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test failed: {e}")
            results.append(False)
    
    print("\\n" + "=" * 60)
    if all(results):
        print("🎉 All tests passed!")
        print("Your TensorFlow 2.x environment is ready.")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return all(results)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
"""
    
    test_path = Path(output_dir) / 'test_tf2_environment.py'
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    # 添加执行权限
    os.chmod(test_path, 0o755)
    
    print(f"✓ Test script created: {test_path}")


def create_requirements(output_dir):
    """创建新的requirements文件"""
    requirements = """# TensorFlow 2.x Environment Requirements
# Python 3.10+

# Deep Learning Frameworks
tensorflow==2.15.0
# tensorflow-gpu==2.15.0  # Uncomment for GPU support
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0

# Scientific Computing
numpy==1.24.3
scipy==1.11.1
pandas==2.0.3

# Visualization
matplotlib==3.7.2
seaborn==0.12.2

# Machine Learning
scikit-learn==1.3.0

# Reinforcement Learning
gym==0.26.2
# stable-baselines3==2.1.0  # Optional

# Image Processing
opencv-python==4.8.0.74
Pillow==10.0.0

# Utilities
tqdm==4.66.1
pyyaml==6.0.1
tensorboard==2.15.0

# Development
pytest==7.4.0
black==23.7.0
flake8==6.1.0

# Optional: Additional RL Libraries
# gymnasium==0.29.1
# stable-baselines3==2.1.0
# ray[rllib]==2.7.0
"""
    
    req_path = Path(output_dir) / 'requirements_tf2.txt'
    with open(req_path, 'w', encoding='utf-8') as f:
        f.write(requirements)
    
    print(f"✓ Requirements file created: {req_path}")


def create_example_model(output_dir):
    """创建TF2风格的示例模型"""
    example = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"
TensorFlow 2.x 示例: DQN网络
展示了如何用TF2风格编写强化学习网络
\"\"\"

import tensorflow as tf
import numpy as np


class DQNNetwork(tf.keras.Model):
    \"\"\"
    DQN网络 - TensorFlow 2.x版本
    使用Keras API构建
    \"\"\"
    
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
        \"\"\"
        前向传播
        Args:
            state: [batch_size, state_dim]
            training: 是否训练模式
        Returns:
            q_values: [batch_size, action_dim]
        \"\"\"
        x = state
        
        # 通过隐藏层
        for layer in self.layers_list:
            x = layer(x)
        
        # 输出Q值
        q_values = self.output_layer(x)
        
        return q_values


class DQNAgent:
    \"\"\"
    DQN智能体 - TensorFlow 2.x版本
    \"\"\"
    
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
        \"\"\"
        预测Q值
        Args:
            state: [batch_size, state_dim]
        Returns:
            q_values: [batch_size, action_dim]
        \"\"\"
        return self.q_network(state, training=False)
    
    def select_action(self, state, epsilon=0.1):
        \"\"\"
        选择动作 (epsilon-greedy)
        Args:
            state: [state_dim]
            epsilon: 探索率
        Returns:
            action: int
        \"\"\"
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
        \"\"\"
        训练一步
        Args:
            states: [batch_size, state_dim]
            actions: [batch_size]
            rewards: [batch_size]
            next_states: [batch_size, state_dim]
            dones: [batch_size]
        Returns:
            loss: scalar
        \"\"\"
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
        \"\"\"更新目标网络\"\"\"
        self.target_network.set_weights(self.q_network.get_weights())
    
    def save(self, path):
        \"\"\"保存模型\"\"\"
        self.q_network.save_weights(path)
    
    def load(self, path):
        \"\"\"加载模型\"\"\"
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
"""
    
    example_path = Path(output_dir) / 'example_tf2_dqn.py'
    with open(example_path, 'w', encoding='utf-8') as f:
        f.write(example)
    
    os.chmod(example_path, 0o755)
    
    print(f"✓ Example model created: {example_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate TensorFlow 1.x code to 2.x'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory with TF 1.x code'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for migrated code'
    )
    parser.add_argument(
        '--create_extras',
        action='store_true',
        help='Create extra files (test script, requirements, example)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TensorFlow 1.x → 2.x Migration Tool")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 执行迁移
    migrator = TFMigrator()
    migrator.migrate_directory(args.input_dir, args.output_dir)
    
    # 创建额外文件
    if args.create_extras:
        print("\nCreating extra files...")
        create_test_script(args.output_dir)
        create_requirements(args.output_dir)
        create_example_model(args.output_dir)
    
    print("\n" + "=" * 60)
    print("✓ Migration completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the migrated code in:", args.output_dir)
    print("2. Check MIGRATION_REPORT.md for details")
    print("3. Fix all TODO comments")
    print("4. Run tests: python test_tf2_environment.py")
    print("5. Install new dependencies: pip install -r requirements_tf2.txt")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
