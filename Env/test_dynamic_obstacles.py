# ===================================
# Migrated from TensorFlow 1.x to 2.x
# Original file: D:\codetest\rlpp_ori\Env\test_dynamic_obstacles.py
# Migration may not be complete. 
# Please review TODO comments.
# ===================================

#!/usr/bin/env python3

import numpy as np
import math
import time
from flight_with_dynamic_obstacles import Flight, Scenairo, DynamicObstacle, create_default_dynamic_obstacles


def create_test_scenario_1():
    """测试场景1：基础动态障碍物测试"""
    dynamic_obstacles = create_default_dynamic_obstacles()

    scenario = Scenairo(
        init_pos=[50, 50],  # 起始位置：左下角
        init_dir=45,  # 起始方向：45度
        goal_pos=[650, 650],  # 目标位置：右上角
        goal_dir=0,  # 目标方向：0度
        circle_obstacles=[[400, 400, 50]],  # 一个静态圆形障碍物
        dynamic_obstacles=dynamic_obstacles
    )
    return scenario


def create_test_scenario_2():
    """测试场景2：复杂动态环境"""
    # 创建更多动态障碍物
    dynamic_obstacles = [
        # 快速移动的小障碍物
        DynamicObstacle([100, 300], 20, 3.0, math.radians(0), [50, 50, 650, 650]),
        # 慢速移动的大障碍物
        DynamicObstacle([500, 400], 40, 1.0, math.radians(135), [50, 50, 650, 650]),
        # 圆周运动效果（通过反弹实现）
        DynamicObstacle([350, 200], 25, 2.5, math.radians(60), [200, 150, 500, 300]),
        # 垂直运动
        DynamicObstacle([200, 600], 30, 2.0, math.radians(-90), [50, 50, 650, 650]),
        # 对角线运动
        DynamicObstacle([600, 100], 35, 1.8, math.radians(225), [50, 50, 650, 650])
    ]

    scenario = Scenairo(
        init_pos=[100, 100],
        init_dir=0,
        goal_pos=[600, 600],
        goal_dir=90,
        circle_obstacles=[[300, 300, 40], [500, 150, 30]],  # 静态障碍物
        dynamic_obstacles=dynamic_obstacles
    )
    return scenario


def create_test_scenario_3():
    """测试场景3：狭窄通道中的动态障碍物"""
    # 创建在狭窄空间中移动的障碍物
    dynamic_obstacles = [
        # 在通道中来回移动的障碍物
        DynamicObstacle([320, 200], 25, 1.5, math.radians(90), [300, 150, 400, 550]),
        DynamicObstacle([380, 400], 25, 1.5, math.radians(-90), [300, 150, 400, 550]),
    ]

    # 创建狭窄通道（使用线障碍物）
    line_obstacles = [
        [300, 150, 300, 550],  # 左墙
        [400, 150, 400, 550],  # 右墙
        [300, 150, 400, 150],  # 上墙
        [300, 550, 400, 550],  # 下墙
    ]

    scenario = Scenairo(
        init_pos=[50, 350],
        init_dir=0,
        goal_pos=[650, 350],
        goal_dir=0,
        line_obstacles=line_obstacles,
        dynamic_obstacles=dynamic_obstacles
    )
    return scenario


def manual_control_test(env, scenario):
    """手动控制测试 - 使用键盘控制（模拟）"""
    print("\n=== 手动控制测试 ===")
    print("无人机将按预设路径飞行，观察动态障碍物的行为")

    env.reset(scenario)

    # 预设的动作序列（模拟智能路径规划）
    actions = [
        # 向目标方向前进
        12, 12, 12, 12, 12,  # 直行
        8, 8, 8,  # 左转
        12, 12, 12, 12,  # 直行
        16, 16, 16,  # 右转
        12, 12, 12, 12, 12,  # 直行
        8, 8,  # 小左转
        12, 12, 12, 12, 12, 12, 12, 12  # 长直行
    ]

    for i, action in enumerate(actions):
        if action >= env.action.n_actions:
            action = action % env.action.n_actions

        state, reward, done = env.step(action)
        env.render(sleep_time=0.1, show_trace=True, show_pos=True, show_arrow=True)

        print(f"Step {i + 1}: Action={action}, Reward={reward:.3f}, Done={done}")

        if done:
            print(f"任务完成！结果: {env.result}")
            break

    return i + 1


def random_action_test(env, scenario, max_steps=150):
    """随机动作测试"""
    print("\n=== 随机动作测试 ===")
    print("无人机将执行随机动作，测试在动态环境中的行为")

    env.reset(scenario)

    for step in range(max_steps):
        action = np.random.randint(0, env.action.n_actions)
        state, reward, done = env.step(action)
        env.render(sleep_time=0.05, show_trace=True, show_pos=True)

        if step % 20 == 0:
            print(f"Step {step}: 当前位置=({env.current_pos[0]:.1f}, {env.current_pos[1]:.1f})")

        if done:
            print(f"任务结束于步骤 {step}, 结果: {env.result}")
            break

    return step + 1


def obstacle_behavior_test(env, scenario):
    """动态障碍物行为测试"""
    print("\n=== 动态障碍物行为测试 ===")
    print("观察动态障碍物的运动模式和边界反弹")

    env.reset(scenario)

    # 记录障碍物的初始位置
    initial_positions = []
    for obs in env.dynamic_obstacles:
        initial_positions.append(obs.pos.copy())
        print(f"障碍物初始位置: ({obs.pos[0]:.1f}, {obs.pos[1]:.1f}), 方向: {math.degrees(obs.direction):.1f}°")

    # 让障碍物运动一段时间
    for step in range(100):
        # 不移动无人机，只更新障碍物
        env.update_dynamic_obstacles()
        env.render(sleep_time=0.08, show_trace=False)

        if step % 25 == 0:
            print(f"\n--- 步骤 {step} ---")
            for i, obs in enumerate(env.dynamic_obstacles):
                print(
                    f"障碍物 {i + 1}: 位置=({obs.pos[0]:.1f}, {obs.pos[1]:.1f}), 方向={math.degrees(obs.direction):.1f}°")

    print("\n障碍物行为测试完成")


def performance_test(env, scenario):
    """性能测试"""
    print("\n=== 性能测试 ===")
    print("测试带动态障碍物的系统性能")

    start_time = time.time()
    step_count = 0

    env.reset(scenario)

    for step in range(200):
        action = np.random.randint(0, env.action.n_actions)
        state, reward, done = env.step(action)

        # 每隔10步渲染一次以提高性能
        if step % 10 == 0:
            env.render(sleep_time=0.01, show_trace=True)

        step_count += 1

        if done:
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    steps_per_second = step_count / elapsed_time if elapsed_time > 0 else 0

    print(f"性能测试结果:")
    print(f"  总步数: {step_count}")
    print(f"  总时间: {elapsed_time:.2f}秒")
    print(f"  平均步数/秒: {steps_per_second:.1f}")


def main():
    """主测试函数"""
    print("动态障碍物功能测试程序")
    print("=" * 50)

    # 创建环境
    env = Flight()

    # 测试不同场景
    scenarios = [
        ("基础动态障碍物测试", create_test_scenario_1()),
        ("复杂动态环境测试", create_test_scenario_2()),
        ("狭窄通道测试", create_test_scenario_3())
    ]

    for scenario_name, scenario in scenarios:
        print(f"\n{'=' * 20} {scenario_name} {'=' * 20}")
        print(f"动态障碍物数量: {len(scenario.dynamic_obstacles)}")
        print(f"静态圆形障碍物: {len(scenario.circle_obstacles) if scenario.circle_obstacles else 0}")
        print(f"静态线障碍物: {len(scenario.line_obstacles) if scenario.line_obstacles else 0}")

        try:
            # 运行不同类型的测试
            manual_control_test(env, scenario)
            time.sleep(1)  # 短暂暂停

            random_action_test(env, scenario, max_steps=100)
            time.sleep(1)

            # 只对第一个场景进行详细的障碍物行为测试
            if scenario_name == "基础动态障碍物测试":
                obstacle_behavior_test(env, scenario)
                performance_test(env, scenario)

        except KeyboardInterrupt:
            print("\n测试被用户中断")
            break
        except Exception as e:
            print(f"测试出现错误: {e}")
            continue

    print("\n所有测试完成！")


if __name__ == "__main__":
    main()