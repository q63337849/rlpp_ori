

"""对比测试三个CNN模型"""
import sys

sys.path.insert(0, r"D:\codetest\rlpp2")

import numpy as np
import tensorflow as tf
import time

# 配置
special_info = {
    "size_splits": [100, 2],
    "n_filters": 8,
    "actor_n_layers": [64, 32],
    "critic_n_layers": [64, 64]
}

# 测试数据
batch_size = 32
state = np.random.random((batch_size, 102)).astype(np.float32)

print("=" * 70)
print("CNN模型性能对比")
print("=" * 70)

models = ["cnn", "cnn2", "cnn3"]

for model_name in models:
    print(f"\n测试 {model_name}.py ...")

    try:
        # 动态导入
        if model_name == "cnn":
            from Model.cnn import ActorModel
        elif model_name == "cnn2":
            from Model.cnn2 import ActorModel
        else:
            from Model.cnn3 import ActorModel

        # 创建模型
        actor = ActorModel(special_info, eval_model=True)

        # 预热
        _ = actor(state)

        # 性能测试
        start_time = time.time()
        for _ in range(100):
            output = actor(state)
        elapsed_time = time.time() - start_time

        # 统计信息
        params = sum([tf.size(v).numpy() for v in actor.trainable_variables])

        print(f"  ✓ 模型创建成功")
        print(f"  ✓ 输出形状: {output.shape}")
        print(f"  ✓ 参数量: {params:,}")
        print(f"  ✓ 100次前向传播耗时: {elapsed_time:.3f}s")
        print(f"  ✓ 平均每次: {elapsed_time / 100 * 1000:.2f}ms")

    except Exception as e:
        print(f"  ✗ 错误: {e}")

print("\n" + "=" * 70)
print("对比完成!")
print("=" * 70)


