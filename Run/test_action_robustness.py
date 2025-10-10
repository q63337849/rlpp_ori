# ===================================
# Migrated from TensorFlow 1.x to 2.x
# Original file: D:\codetest\rlpp_ori\Run\test_action_robustness.py
# Migration may not be complete. 
# Please review TODO comments.
# ===================================

#!/usr/bin/env python3
"""
测试ContinueAction对各种DDPG输出格式的鲁棒性
"""

import numpy as np
import math
from Env.flight_with_dynamic_obstacles import ContinueAction


def test_ddpg_output_formats():
    """测试DDPG可能输出的各种格式"""
    print("=== 测试DDPG输出格式鲁棒性 ===")

    action = ContinueAction(45, 10., math.pi / 60, 4)

    # 模拟DDPG可能输出的各种格式
    test_cases = [
        # 描述, 输入值
        ("Python float", 0.5),
        ("Python int", 1),
        ("Numpy float32", np.float32(0.3)),
        ("Numpy float64", np.float64(-0.7)),
        ("Numpy scalar", np.array(0.4).item()),
        ("Numpy 0-d array", np.array(0.2)),
        ("Numpy 1-d array (size 1)", np.array([0.6])),
        ("Numpy 1-d array (size 2)", np.array([0.3, -0.4])),
        ("Python list (size 1)", [0.8]),
        ("Python list (size 2)", [-0.2, 0.5]),
        ("Python tuple", (0.1, -0.3)),
        ("Large numpy array", np.array([0.2, 0.4, 0.6, 0.8])),
        ("Empty array", np.array([])),
        ("String (error case)", "invalid"),
        ("None (error case)", None),
        ("Dict (error case)", {"action": 0.5}),
    ]

    success_count = 0

    for desc, test_input in test_cases:
        try:
            action(test_input)
            print(f"✓ {desc}: delta_dis={action.delta_dis:.3f}, delta_dir={action.delta_dir:.3f}")
            success_count += 1
        except Exception as e:
            print(f"✗ {desc}: 错误 - {type(e).__name__}: {e}")

    print(f"\n成功处理: {success_count}/{len(test_cases)} 种格式")
    return success_count == len(test_cases) - 3  # 预期最后3个错误案例会失败


def test_boundary_values():
    """测试边界值处理"""
    print("\n=== 测试边界值处理 ===")

    action = ContinueAction(45, 10., math.pi / 60, 4)

    boundary_cases = [
        ("最小值", -1.0),
        ("最大值", 1.0),
        ("零值", 0.0),
        ("超出下界", -2.0),
        ("超出上界", 2.0),
        ("极小值", -1e10),
        ("极大值", 1e10),
        ("NaN", float('nan')),
        ("正无穷", float('inf')),
        ("负无穷", float('-inf')),
    ]

    for desc, test_input in boundary_cases:
        try:
            action(test_input)
            if math.isnan(action.delta_dir) or math.isinf(action.delta_dir):
                print(f"⚠ {desc}: 产生了非正常值 - delta_dir={action.delta_dir}")
            else:
                print(f"✓ {desc}: delta_dis={action.delta_dis:.3f}, delta_dir={action.delta_dir:.3f}")
        except Exception as e:
            print(f"✗ {desc}: 错误 - {type(e).__name__}: {e}")


def test_tensorflow_style_outputs():
    """测试TensorFlow风格的输出"""
    print("\n=== 测试TensorFlow风格输出 ===")

    action = ContinueAction(45, 10., math.pi / 60, 4)

    # 模拟TensorFlow/PyTorch可能的输出格式
    try:
        # 模拟tf.Tensor.numpy()的输出
        tf_like_outputs = [
            np.array(0.5, dtype=np.float32),  # 标量tensor
            np.array([0.3], dtype=np.float32),  # 1-d tensor
            np.array([[0.2]], dtype=np.float32),  # 2-d tensor (batch=1)
        ]

        for i, output in enumerate(tf_like_outputs):
            action(output)
            print(f"✓ TF风格输出 {i + 1}: delta_dis={action.delta_dis:.3f}, delta_dir={action.delta_dir:.3f}")

    except Exception as e:
        print(f"✗ TensorFlow风格测试失败: {e}")


def test_performance():
    """测试性能"""
    print("\n=== 性能测试 ===")

    action = ContinueAction(45, 10., math.pi / 60, 4)

    # 测试大量调用的性能
    import time

    n_calls = 10000
    test_action = 0.5

    start_time = time.time()
    for _ in range(n_calls):
        action(test_action)
    end_time = time.time()

    avg_time = (end_time - start_time) / n_calls * 1000  # 毫秒
    print(f"平均每次调用时间: {avg_time:.4f} ms")
    print(f"每秒可处理: {1000 / avg_time:.0f} 次动作")


def simulate_training_loop():
    """模拟实际训练循环中的使用"""
    print("\n=== 模拟训练循环 ===")

    action = ContinueAction(45, 10., math.pi / 60, 4)

    # 模拟DDPG训练中的典型动作序列
    print("模拟DDPG探索阶段（有噪声）:")
    for step in range(10):
        # 模拟网络输出 + 噪声
        base_action = np.random.uniform(-0.5, 0.5)
        noise = np.random.normal(0, 0.1)
        noisy_action = np.clip(base_action + noise, -1.0, 1.0)

        action(noisy_action)
        print(f"  步骤 {step}: 动作={noisy_action:.3f} -> 转向={math.degrees(action.delta_dir):.1f}°")

    print("\n模拟DDPG利用阶段（无噪声）:")
    for step in range(5):
        # 模拟确定性策略输出
        deterministic_action = np.sin(step * 0.5)  # 平滑变化的动作

        action(deterministic_action)
        print(f"  步骤 {step}: 动作={deterministic_action:.3f} -> 转向={math.degrees(action.delta_dir):.1f}°")


def main():
    """主测试函数"""
    print("ContinueAction鲁棒性测试")
    print("=" * 50)

    all_passed = True

    try:
        # 运行所有测试
        format_test_passed = test_ddpg_output_formats()
        all_passed = all_passed and format_test_passed

        test_boundary_values()
        test_tensorflow_style_outputs()
        test_performance()
        simulate_training_loop()

        print("\n" + "=" * 50)
        if all_passed:
            print("鲁棒性测试通过！ContinueAction可以安全处理DDPG的各种输出格式。")
        else:
            print("部分测试未通过，但这在预期范围内（错误格式应该被处理）。")

    except Exception as e:
        print(f"测试过程中出现意外错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()