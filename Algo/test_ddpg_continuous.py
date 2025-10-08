#!/usr/bin/env python3
"""
测试DDPG连续动作与动态障碍物的兼容性
"""

import numpy as np
import math
from Env.flight_with_dynamic_obstacles import Flight, Scenairo, ContinueAction, create_default_dynamic_obstacles


def test_continue_action():
    """测试ContinueAction类的不同输入格式"""
    print("=== 测试ContinueAction类 ===")

    action = ContinueAction(45, 10., math.pi / 60, 4)

    # 测试不同的输入格式
    test_cases = [
        ("单个浮点数", 0.5),
        ("单个负浮点数", -0.3),
        ("numpy浮点数", np.float32(0.8)),
        ("长度为1的数组", [0.2]),
        ("长度为2的数组", [0.5, -0.7]),
        ("numpy数组", np.array([0.3, 0.6]))
    ]

    for desc, test_input in test_cases:
        try:
            action(test_input)
            print(f"✓ {desc}: delta_dis={action.delta_dis:.3f}, delta_dir={action.delta_dir:.3f}")
        except Exception as e:
            print(f"✗ {desc}: 错误 - {e}")

    print(f"动作维度 n_actions: {action.n_actions}")
    print(f"动作边界: [-{action.action_bound}, {action.action_bound}]")


def test_ddpg_environment():
    """测试DDPG环境的完整流程"""
    print("\n=== 测试DDPG环境 ===")

    # 创建连续动作环境
    action = ContinueAction(45, 10., math.pi / 60, 4)
    env = Flight(action=action)

    # 创建测试场景
    dynamic_obstacles = create_default_dynamic_obstacles()
    scenario = Scenairo(
        init_pos=[100, 100],
        init_dir=0,
        goal_pos=[600, 600],
        goal_dir=90,
        circle_obstacles=[[300, 300, 40], [500, 200, 35]],
        dynamic_obstacles=dynamic_obstacles
    )

    print(f"环境动作维度: {env.action.n_actions}")
    print(f"环境状态维度: {env.d_states}")

    # 重置环境
    state = env.reset(scenario)
    print(f"初始状态形状: {state.shape}")

    # 测试几个连续动作
    test_actions = [
        0.0,  # 直行
        0.5,  # 右转
        -0.5,  # 左转
        [0.8, 0.3],  # 快速+右转
        [-0.2, -0.8]  # 慢速+左转
    ]

    for i, action_input in enumerate(test_actions):
        try:
            next_state, reward, done = env.step(action_input)
            print(f"步骤 {i + 1}: 动作={action_input}, 奖励={reward:.3f}, 完成={done}")
            print(f"  位置: ({env.current_pos[0]:.1f}, {env.current_pos[1]:.1f})")

            if done:
                result_names = ['超时', '失败', '边界', '成功']
                print(f"  结果: {result_names[env.result]}")
                break

        except Exception as e:
            print(f"步骤 {i + 1} 错误: {e}")
            break


def test_ddpg_simulation():
    """模拟DDPG算法的动作选择"""
    print("\n=== 模拟DDPG算法 ===")

    # 创建环境
    action = ContinueAction(45, 10., math.pi / 60, 4)
    env = Flight(action=action)

    # 创建简单场景
    scenario = Scenairo(
        init_pos=[50, 350],
        init_dir=0,
        goal_pos=[650, 350],
        goal_dir=0,
        circle_obstacles=[[350, 350, 60]],
        dynamic_obstacles=create_default_dynamic_obstacles()
    )

    state = env.reset(scenario)

    # 模拟DDPG的动作选择过程
    for step in range(50):
        # 模拟DDPG网络输出（随机策略用于测试）
        if step < 10:
            # 开始阶段：保持直行
            action_output = 0.0
        elif step < 20:
            # 遇到障碍：右转
            action_output = 0.6
        elif step < 30:
            # 绕过障碍：左转回正
            action_output = -0.4
        else:
            # 后期：随机微调
            action_output = np.random.uniform(-0.3, 0.3)

        # 添加exploration noise（模拟DDPG的噪声）
        noise = np.random.normal(0, 0.1)
        noisy_action = np.clip(action_output + noise, -1.0, 1.0)

        next_state, reward, done = env.step(noisy_action)

        if step % 10 == 0:
            print(f"步骤 {step}: 动作={noisy_action:.3f}, 奖励={reward:.3f}")
            print(f"  位置: ({env.current_pos[0]:.1f}, {env.current_pos[1]:.1f})")

        if done:
            result_names = ['超时', '失败', '边界', '成功']
            print(f"任务完成于步骤 {step}: {result_names[env.result]}")
            break

        state = next_state


def visual_test():
    """可视化测试（如果可用）"""
    print("\n=== 可视化测试 ===")

    try:
        # 创建环境
        action = ContinueAction(45, 10., math.pi / 60, 4)
        env = Flight(action=action)

        # 创建场景
        scenario = Scenairo(
            init_pos=[100, 100],
            init_dir=45,
            goal_pos=[600, 600],
            goal_dir=0,
            circle_obstacles=[[300, 300, 40]],
            dynamic_obstacles=create_default_dynamic_obstacles()
        )

        state = env.reset(scenario)

        print("开始可视化演示（如果窗口可用）...")

        # 简单的导航策略
        for step in range(100):
            # 计算到目标的角度
            goal_vector = env.goal_pos - env.current_pos
            goal_angle = math.atan2(goal_vector[1], goal_vector[0])
            current_angle = env.current_dir

            # 角度差
            angle_diff = goal_angle - current_angle

            # 归一化角度差到[-pi, pi]
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            elif angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            # 转换为动作（简单比例控制）
            action_output = np.clip(angle_diff / math.pi, -1.0, 1.0)

            next_state, reward, done = env.step(action_output)

            # 尝试渲染（如果可用）
            try:
                env.render(sleep_time=0.05, show_trace=True, show_pos=True)
            except:
                pass  # 如果无法渲染则跳过

            if done:
                result_names = ['超时', '失败', '边界', '成功']
                print(f"可视化测试完成: {result_names[env.result]}")
                break

    except Exception as e:
        print(f"可视化测试跳过: {e}")


def main():
    """主测试函数"""
    print("DDPG连续动作与动态障碍物兼容性测试")
    print("=" * 50)

    try:
        test_continue_action()
        test_ddpg_environment()
        test_ddpg_simulation()
        visual_test()

        print("\n" + "=" * 50)
        print("所有测试完成！")
        print("ContinueAction类已准备好与DDPG算法配合使用")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()