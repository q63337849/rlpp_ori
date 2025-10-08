"""
快速LSTM测试脚本 - 验证集成是否正常工作
"""

import numpy as np
import sys
sys.path.append('D:\\codetest\\rlpp_ori')

from Env.flight_with_dynamic_obstacles import Flight, DynamicObstacle, Scenairo
from Algo.lstm_flight_integration import LSTMFlightWrapper


def quick_test():
    """快速测试LSTM环境封装"""
    print("快速LSTM测试")
    print("=" * 50)
    
    # 1. 创建基础环境
    print("1. 创建基础环境...")
    base_env = Flight()
    print(f"   动作数量: {base_env.action.n_actions}")
    
    # 2. 封装为LSTM环境
    print("\n2. 封装为LSTM环境...")
    lstm_env = LSTMFlightWrapper(base_env, history_length=5)
    print(f"   历史长度: 5")
    print(f"   最大障碍物数: {lstm_env.max_obstacles}")
    
    # 3. 创建简单场景
    print("\n3. 创建测试场景...")
    dynamic_obstacles = [
        DynamicObstacle([200, 300], 30, 2.0, None, [50, 50, 650, 650], 0.03),
        DynamicObstacle([400, 400], 25, 1.5, None, [50, 50, 650, 650], 0.04),
        DynamicObstacle([300, 200], 35, 1.8, None, [50, 50, 650, 650], 0.02),
    ]
    
    scenario = Scenairo(
        init_pos=[100, 100],
        init_dir=0,
        goal_pos=[600, 600],
        goal_dir=90,
        circle_obstacles=[[300, 300, 40]],
        dynamic_obstacles=dynamic_obstacles
    )
    print(f"   起点: {scenario.init_pos}")
    print(f"   终点: {scenario.goal_pos}")
    print(f"   静态障碍物: 1个")
    print(f"   动态障碍物: {len(dynamic_obstacles)}个")
    
    # 4. 重置环境
    print("\n4. 重置环境...")
    lstm_state = lstm_env.reset(scenario)
    
    print(f"   自身状态维度: {lstm_state['self_state'].shape}")
    print(f"   目标状态维度: {lstm_state['goal_state'].shape}")
    print(f"   障碍物状态维度: {lstm_state['obstacle_states'].shape}")
    
    print(f"\n   自身状态值: {lstm_state['self_state']}")
    print(f"   目标状态值: {lstm_state['goal_state']}")
    print(f"   障碍物状态(前3个):")
    for i in range(min(3, len(dynamic_obstacles))):
        print(f"     障碍物 {i+1}: {lstm_state['obstacle_states'][i]}")
    
    # 5. 运行几步
    print("\n5. 运行测试步骤...")
    for step in range(10):
        # 随机选择动作
        action = np.random.randint(0, base_env.action.n_actions)
        
        # 执行动作
        next_lstm_state, reward, done = lstm_env.step(action)
        
        print(f"   步骤 {step}: 动作={action}, 奖励={reward:.3f}, 完成={done}")
        
        if done:
            result_names = ['超时', '失败', '边界', '成功']
            print(f"   任务结束: {result_names[lstm_env.env.result]}")
            break
        
        lstm_state = next_lstm_state
    
    print("\n✓ 快速测试完成!")
    return True


def test_state_dimensions():
    """测试状态维度是否正确"""
    print("\n" + "=" * 50)
    print("状态维度测试")
    print("=" * 50)
    
    base_env = Flight()
    lstm_env = LSTMFlightWrapper(base_env)
    
    # 创建场景
    scenario = Scenairo(
        init_pos=[100, 100],
        init_dir=0,
        goal_pos=[600, 600],
        goal_dir=0,
        circle_obstacles=None,
        dynamic_obstacles=[
            DynamicObstacle([300, 300], 30, 2.0, None, [50, 50, 650, 650], 0.03)
        ]
    )
    
    lstm_state = lstm_env.reset(scenario)
    
    # 验证维度
    tests = [
        ("自身状态", lstm_state['self_state'].shape, (4,)),
        ("目标状态", lstm_state['goal_state'].shape, (2,)),
        ("障碍物状态", lstm_state['obstacle_states'].shape, (10, 5))
    ]
    
    all_passed = True
    for name, actual, expected in tests:
        passed = actual == expected
        all_passed = all_passed and passed
        status = "✓" if passed else "✗"
        print(f"{status} {name}: {actual} (期望: {expected})")
    
    if all_passed:
        print("\n✓ 所有维度测试通过!")
    else:
        print("\n✗ 部分测试失败")
    
    return all_passed


def test_multiple_scenarios():
    """测试多个场景"""
    print("\n" + "=" * 50)
    print("多场景测试")
    print("=" * 50)
    
    base_env = Flight()
    lstm_env = LSTMFlightWrapper(base_env)
    
    # 不同数量的动态障碍物
    obstacle_counts = [0, 1, 3, 5, 10, 15]
    
    for count in obstacle_counts:
        # 创建指定数量的障碍物
        dynamic_obstacles = [
            DynamicObstacle(
                [np.random.randint(100, 600), np.random.randint(100, 600)],
                np.random.randint(20, 40),
                np.random.uniform(1.0, 3.0),
                None,
                [50, 50, 650, 650],
                0.03
            )
            for _ in range(count)
        ]
        
        scenario = Scenairo(
            init_pos=[100, 100],
            init_dir=0,
            goal_pos=[600, 600],
            goal_dir=0,
            dynamic_obstacles=dynamic_obstacles
        )
        
        try:
            lstm_state = lstm_env.reset(scenario)
            # 测试几步
            for _ in range(3):
                action = np.random.randint(0, base_env.action.n_actions)
                lstm_state, reward, done = lstm_env.step(action)
                if done:
                    break
            
            print(f"✓ {count}个动态障碍物: 成功")
        except Exception as e:
            print(f"✗ {count}个动态障碍物: 失败 - {e}")
            return False
    
    print("\n✓ 多场景测试通过!")
    return True


def main():
    """主函数"""
    print("LSTM集成验证测试")
    print("=" * 50)
    print()
    
    try:
        # 运行所有测试
        test1 = quick_test()
        test2 = test_state_dimensions()
        test3 = test_multiple_scenarios()
        
        print("\n" + "=" * 50)
        print("测试总结")
        print("=" * 50)
        print(f"快速测试: {'✓ 通过' if test1 else '✗ 失败'}")
        print(f"维度测试: {'✓ 通过' if test2 else '✗ 失败'}")
        print(f"多场景测试: {'✓ 通过' if test3 else '✗ 失败'}")
        
        if test1 and test2 and test3:
            print("\n🎉 所有测试通过! LSTM集成成功!")
            print("\n下一步:")
            print("1. 运行 'python run_lstm_training_complete.py' 进行基础测试")
            print("2. 运行 'python run_lstm_training_complete.py train' 进行训练演示")
        else:
            print("\n❌ 部分测试失败，请检查配置")
    
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
