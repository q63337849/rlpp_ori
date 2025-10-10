import tensorflow as tf

# 强制启用 TensorFlow 2.x Eager Execution
if hasattr(tf, 'config') and hasattr(tf.config, 'run_functions_eagerly'):
    tf.config.run_functions_eagerly(True)

# 禁用 TF 1.x 兼容模式（如果之前启用了）
# 注释掉任何 tf.compat.v1.disable_eager_execution() 调用

print(f"TensorFlow版本: {tf.__version__}")
print(f"Eager Execution: {tf.executing_eagerly()}")

import common
import options
import sys
from Run.data import DataPath, DataArgs
import Algo
import Scene  # 使用原始Scene模块
from Run.exec import Train, Predict, Exec  # 添加 Exec
import numpy as np
import math


# 导入动态障碍物相关类
from Env.flight_with_dynamic_obstacles import Flight, DynamicObstacle, Scenairo

__agent = None


def default_agent(env, algo_t=options.dqn, model_t=options.M_CNN, force_new=False):
    """简化使用难度，设计得不够好"""
    global __agent
    if __agent is None or force_new:  # ← 添加 force_new 参数
        algo = Algo.AlgoDispatch(algo_t, model_t)
        supervisor = algo(buffer_size=500)

        # 根据算法类型处理动作空间
        if algo_t.upper() == 'DDPG':
            action_dim = 1
            model = supervisor(env.d_states, action_dim, critic_lr=0.001)
        else:
            model = supervisor(env.d_states, env.action.n_actions, critic_lr=0.001)

        __agent = model(size_splits=env.d_states_detail, actor_n_layers=(20, 10),
                        critic_n_layers=(20, 20, 20), n_filters=5, state_n_layers=(20,))
    return __agent

def create_dynamic_obstacles_by_type(dynamic_type='basic'):
    """根据类型创建动态障碍物 - 支持随机方向运动"""
    if dynamic_type == 'basic':
        return [
            DynamicObstacle([150, 500], 30, 2.0, None, [50, 50, 650, 650], 0.03),
            DynamicObstacle([550, 350], 25, 1.5, None, [50, 50, 650, 650], 0.05),
            DynamicObstacle([300, 100], 35, 1.8, None, [50, 50, 650, 650], 0.02)
        ]
    elif dynamic_type == 'fast':
        return [
            DynamicObstacle([200, 200], 25, 4.0, None, [50, 50, 650, 650], 0.08),
            DynamicObstacle([400, 400], 30, 3.5, None, [50, 50, 650, 650], 0.06),
            DynamicObstacle([500, 200], 20, 5.0, None, [50, 50, 650, 650], 0.10)
        ]
    elif dynamic_type == 'slow_large':
        return [
            DynamicObstacle([150, 300], 50, 1.0, None, [50, 50, 650, 650], 0.02),
            DynamicObstacle([450, 500], 45, 0.8, None, [50, 50, 650, 650], 0.01),
            DynamicObstacle([350, 150], 40, 1.2, None, [50, 50, 650, 650], 0.03)
        ]
    elif dynamic_type == 'mixed':
        return [
            DynamicObstacle([100, 100], 20, 3.0, None, [50, 50, 650, 650], 0.07),  # 快小
            DynamicObstacle([300, 400], 40, 1.5, None, [50, 50, 650, 650], 0.03),  # 慢大
            DynamicObstacle([550, 250], 25, 2.5, None, [50, 50, 650, 650], 0.05),  # 中等
            DynamicObstacle([450, 550], 35, 1.8, None, [50, 50, 650, 650], 0.04),  # 慢大
            DynamicObstacle([200, 500], 15, 4.0, None, [50, 50, 650, 650], 0.09)  # 快小
        ]
    elif dynamic_type == 'chaotic':
        # 使用混乱模式
        from Env.flight_with_dynamic_obstacles import create_chaotic_dynamic_obstacles
        return create_chaotic_dynamic_obstacles()
    elif dynamic_type == 'random':
        # 完全随机生成
        from Env.flight_with_dynamic_obstacles import create_random_dynamic_obstacles
        return create_random_dynamic_obstacles(num_obstacles=np.random.randint(3, 7))
    else:
        print(f"未知的动态障碍物类型: {dynamic_type}，使用basic类型")
        return create_dynamic_obstacles_by_type('basic')


def convert_to_dynamic_scenarios(original_scenarios, dynamic_type='basic', seed=None):
    """将原始场景转换为带动态障碍物的场景"""
    dynamic_scenarios = []

    print(f"正在为 {len(original_scenarios)} 个场景添加动态障碍物")

    # 如果提供了种子，设置随机种子以获得可重复的结果
    if seed is not None:
        np.random.seed(seed)
        print(f"  使用随机种子: {seed}")

    for i, orig_scenario in enumerate(original_scenarios):
        # 为每个场景创建独立的动态障碍物
        # 使用场景索引作为局部种子，确保可重复性
        if seed is not None:
            np.random.seed(seed + i)

        enhanced_scenario = Scenairo(
            init_pos=orig_scenario.init_pos,
            init_dir=orig_scenario.init_dir,
            goal_pos=orig_scenario.goal_pos,
            goal_dir=orig_scenario.goal_dir,
            circle_obstacles=orig_scenario.circle_obstacles,
            line_obstacles=getattr(orig_scenario, 'line_obstacles', None),
            dynamic_obstacles=create_dynamic_obstacles_by_type(dynamic_type)
        )
        dynamic_scenarios.append(enhanced_scenario)

    return dynamic_scenarios


def training(env, agent, paths, train_scenes, predict_scenes):
    """训练函数 - 使用带进度条的 Exec 类"""
    print("\n=== 动态障碍物环境训练开始 ===")
    print(f"训练场景数量: {len(train_scenes)}")
    print(f"预测场景数量: {len(predict_scenes)}")

    if train_scenes and hasattr(train_scenes[0], 'dynamic_obstacles'):
        print(f"每个场景的动态障碍物数量: {len(train_scenes[0].dynamic_obstacles)}")

    print("\n🔧 构建模型...")
    if hasattr(agent, 'build_model'):
        agent.build_model()
    elif hasattr(agent, 'model') and hasattr(agent.model, 'build_model'):
        agent.model.build_model()
    else:
        print("  ⚠️ 找不到 build_model 方法，尝试继续...")

    for i in range(args.nth_rounds):
        print(f'\n{"=" * 70}')
        print(f'  训练大轮次: {i + 1}/{args.nth_rounds}')
        print(f'{"=" * 70}')

        # 训练
        train_exec = Exec(
            env=env,
            agent=agent,
            scenes=train_scenes,
            paths=paths,
            train=True,
            episodes=args.episodes,
            max_n_step=args.max_n_step
        )
        train_exec()



        # 测试A：训练后立即测试
        print(f'\n🔬 测试A：训练后立即测试（不保存不加载，使用内存中的模型）...')

        # 测试模型输出
        print("\n检查模型输出...")
        import tensorflow as tf
        test_obs = np.zeros(env.d_states, dtype=np.float32)
        action_probs = agent.model.action(np.expand_dims(test_obs, axis=0))

        if isinstance(action_probs, tf.Tensor):
            action_probs = action_probs.numpy()

        max_prob = np.max(action_probs)
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
        print(f"  训练后 - MaxProb: {max_prob:.4f}, Entropy: {entropy:.4f}")

        test_before_save = Exec(
            env=env,
            agent=agent,
            scenes=train_scenes,
            paths=paths,
            train=False,
            episodes=1,
            max_n_step=args.max_n_step
        )
        test_before_save()

        # 保存模型
        print(f'\n💾 保存模型...')
        save_path = paths.model_save_file()
        print(f"  保存到: {save_path}")
        agent.model.save_model(save_path)

        # 检查保存的文件
        import os
        if os.path.exists(save_path):
            print(f"  ✓ 文件已保存，大小: {os.path.getsize(save_path) / 1024:.2f} KB")

        # 测试B：加载模型后测试
        print(f'\n🔬 测试B：加载模型后测试...')
        agent.model.load_model(save_path)

        # 再次测试模型输出
        action_probs = agent.model.action(np.expand_dims(test_obs, axis=0))
        if isinstance(action_probs, tf.Tensor):
            action_probs = action_probs.numpy()

        max_prob_after = np.max(action_probs)
        entropy_after = -np.sum(action_probs * np.log(action_probs + 1e-10))
        print(f"  加载后 - MaxProb: {max_prob_after:.4f}, Entropy: {entropy_after:.4f}")

        if abs(max_prob - max_prob_after) > 0.1:
            print("  ⚠️ 警告：加载前后模型输出差异很大！保存/加载有问题！")
        else:
            print("  ✓ 模型加载正常")

        test_after_load = Exec(
            env=env,
            agent=agent,
            scenes=train_scenes,
            paths=paths,
            train=False,
            episodes=1,
            max_n_step=args.max_n_step
        )
        test_after_load()

        # 在测试场景上测试
        print(f'\n📊 在测试场景上测试...')
        predict_exec = Exec(
            env=env,
            agent=agent,
            scenes=predict_scenes,
            paths=paths,
            train=False,
            episodes=1,
            max_n_step=args.max_n_step
        )
        predict_exec()

    print("\n✅ 训练结束")


def prediction(env, agent, paths, predict_scenes):
    """预测函数 - TensorFlow 2.x 版本"""
    print("\n=== 动态障碍物环境预测开始 ===")

    # 检查模型文件
    model_path = paths.model_load_file()
    print(f"\n尝试加载模型: {model_path}")

    import os
    import tensorflow as tf

    if not os.path.exists(model_path):
        print("✗ 模型文件不存在！")
        return

    print("✓ 模型文件存在")
    print(f"  文件大小: {os.path.getsize(model_path) / 1024:.2f} KB")

    # === 改动 1：加载前先构建（或用假前向触发变量创建） ===
    print("\n🔧 构建推理模型...")
    try:
        import numpy as np
        # 优先调用显式的 build 接口
        if hasattr(agent, 'build_model'):
            agent.build_model()
        elif hasattr(agent, 'model') and hasattr(agent.model, 'build_model'):
            agent.model.build_model()
        else:
            # 兜底：用一次假前向触发变量创建
            dummy_state = np.zeros((1,) + tuple(env.d_states), dtype=np.float32)
            _ = agent.model.action(dummy_state)
        print("✓ 模型结构已构建")
    except Exception as e:
        print(f"  ⚠️ 构建模型时出错：{e}，尝试用一次假前向触发变量创建")
        try:
            import numpy as np
            dummy_state = np.zeros((1,) + tuple(env.d_states), dtype=np.float32)
            _ = agent.model.action(dummy_state)
            print("✓ 通过假前向完成变量创建")
        except Exception as ee:
            print(f"  ✗ 假前向失败：{ee}")

    # 加载模型（通过 agent.model，不是 agent）
    print("\n📦 加载模型权重...")
    try:
        # 正确的调用方式
        if hasattr(agent, 'model') and hasattr(agent.model, 'load_model'):
            agent.model.load_model(model_path)
        elif hasattr(agent, 'load_model'):
            agent.load_model(model_path)
        else:
            raise AttributeError("找不到 load_model 接口（请检查 agent/model 实现）")
        print("✓ 模型加载完成")
    except Exception as e:
        print(f"✗ 加载模型失败: {e}")
        traceback.print_exc()
        return

    # 检查模型变量
    print("\n🔍 检查模型权重...")
    try:
        variables = []
        if hasattr(agent, 'model') and hasattr(agent.model, 'variables'):
            variables = agent.model.variables
        elif hasattr(agent, 'variables'):
            variables = agent.variables

        print(f"  总共有 {len(variables)} 个变量")
        zero_like = True
        for v in variables[:5]:
            v_np = v.numpy() if hasattr(v, 'numpy') else np.array(v)
            if np.any(v_np != 0):
                zero_like = False
                break
        if zero_like:
            print("  ⚠️ 警告：变量看起来像未初始化或全 0")
        else:
            print("  ✓ 变量非零，加载看起来正常")
    except Exception as e:
        print(f"  无法检查变量：{e}")

    # 快速前向检查输出分布
    print("\n🔬 测试模型输出...")
    try:
        import numpy as np
        dummy_state = np.zeros((1,) + tuple(env.d_states), dtype=np.float32)

        if hasattr(agent, 'model') and hasattr(agent.model, 'action'):
            action_probs = agent.model.action(dummy_state)
        elif hasattr(agent, 'action'):
            action_probs = agent.action(dummy_state)
        else:
            raise AttributeError("没有发现 action 接口以执行一次前向")

        # 转换为 numpy
        if isinstance(action_probs, tf.Tensor):
            action_probs = action_probs.numpy()

        # 确保是 2D 数组
        if len(action_probs.shape) == 1:
            action_probs = np.expand_dims(action_probs, axis=0)

        max_prob = np.max(action_probs)
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
        top_3_actions = np.argsort(action_probs[0])[-3:][::-1]
        top_3_probs = action_probs[0][top_3_actions]

        print(f"  MaxProb: {max_prob:.4f}, Entropy: {entropy:.4f}")
        print(f"  Top 3 actions: {top_3_actions}, probs: {top_3_probs}")

        if max_prob < 0.2:
            print("  ⚠️ 警告：模型输出接近均匀分布（MaxProb < 0.2）")
            print("  这可能意味着模型没有正确训练或加载！")
        elif entropy > 2.0:
            print("  ⚠️ 警告：熵值过高（Entropy > 2.0）")
        else:
            print("  ✓ 模型输出看起来正常")

    except Exception as e:
        print(f"  ⚠️ 测试模型输出时出错: {e}")
        import traceback
        traceback.print_exc()

    # 步骤1：在训练场景上测试
    print("\n" + "=" * 70)
    print("  🔍 步骤1：在训练场景上测试")
    print("=" * 70)

    train_scenes, train_info = build_train_scenes(args)
    if train_info and 'independent_circle' in train_info:
        print(f"\n加载训练场景...")
        print(f"independent_circle: task_size: {train_info['independent_circle']['task_size']}, "
              f"obstacles_size: {train_info['independent_circle']['obstacles_size']}")
    if train_info and 'obs_id' in train_info:
        print(f"obs_id: {train_info['obs_id']}, task_id: {train_info.get('task_id')}, "
              f"n_scenarios: {train_info.get('n_scenarios')}, percentage: {train_info.get('percentage')}")
    print(f"正在为 {len(train_scenes)} 个场景添加动态障碍物")
    print(f"  使用随机种子: {getattr(args, 'nth_times', -1)}")

    # 与预测场景保持相同的动态障碍物设定
    dyns = create_dynamic_obstacles_by_type(getattr(args, "dynamic_obstacle_type", "basic"))
    for sc in train_scenes:
        sc.dynamic_obstacles = dyns

    print(f"训练场景数量: {len(train_scenes)}")

    test_on_train_exec = Exec(
        env=env,
        agent=agent,
        scenes=train_scenes,
        paths=paths,
        train=False,
        episodes=1,
        max_n_step=args.max_n_step
    )
    test_on_train_exec()

    # 步骤2：在测试场景上测试
    print("\n" + "=" * 70)
    print("  📊 步骤2：在测试场景上测试")
    print("=" * 70)
    print(f"测试场景数量: {len(predict_scenes)}")

    if predict_scenes and hasattr(predict_scenes[0], 'dynamic_obstacles'):
        print(f"每个场景的动态障碍物数量: {len(predict_scenes[0].dynamic_obstacles)}")

    predict_exec = Exec(
        env=env,
        agent=agent,
        scenes=predict_scenes,
        paths=paths,
        train=False,
        episodes=1,
        max_n_step=args.max_n_step
    )
    predict_exec()

    print("\n✅ 预测结束")



def run(args):
    """主运行函数 - 支持动态障碍物"""
    print(f"\n动态障碍物类型: {args.dynamic_obstacle_type}")
    print(f"障碍物类型: {args.obstacle_type}")
    print(f"算法类型: {args.algo_t}")

    # 使用原始的Scene模块加载场景
    if args.obstacle_type == options.O_circle:
        train_task = Scene.circle_train_task4x200
        test_task = Scene.circle_test_task4x100
    else:
        train_task = Scene.line_train_task4x200
        test_task = Scene.line_test_task4x100

    scene = Scene.ScenarioLoader()
    print("\n加载原始训练场景: ")
    original_train_scenes = scene.load_scene(**train_task, percentage=args.percentage)
    print("\n加载原始预测场景: ")
    original_predict_scenes = scene.load_scene(**test_task)

    # 转换为带动态障碍物的场景- 使用固定种子
    print(f"\n为场景添加动态障碍物 (类型: {args.dynamic_obstacle_type})...")

    # 使用 nth_times 作为种子，确保相同编号的实验使用相同的动态障碍物
    seed = args.nth_times if args.nth_times >= 0 else 42
    print(f"使用种子 {seed} 生成动态障碍物")

    train_scenes = convert_to_dynamic_scenarios(original_train_scenes, args.dynamic_obstacle_type, seed=seed)
    predict_scenes = convert_to_dynamic_scenarios(original_predict_scenes, args.dynamic_obstacle_type, seed=seed + 10000)

    print(f"增强后训练场景数量: {len(train_scenes)}")
    print(f"增强后预测场景数量: {len(predict_scenes)}")

    # 根据算法类型创建环境
    if args.algo_t.upper() == 'DDPG':
        print("使用连续动作空间（DDPG算法）")
        from Env.flight_with_dynamic_obstacles import ContinueAction
        action = ContinueAction(45, 10., math.pi / 60, 4)
        env = Flight(action=action)
    else:
        print("使用离散动作空间（DQN等算法）")
        from Env.flight_with_dynamic_obstacles import DiscreteAction
        action = DiscreteAction(45, 10., math.pi / 60, 4, 5)
        env = Flight(action=action)

    # 测试模式下强制创建新的 agent
    if not args.train:
        agent = default_agent(env, args.algo_t, args.model_t, force_new=True)  # ← 添加 force_new=True
    else:
        agent = default_agent(env, args.algo_t, args.model_t)

    paths = DataPath(args)

    if args.train:
        training(env, agent, paths, train_scenes, predict_scenes)
    else:
        prediction(env, agent, paths, predict_scenes)


def demo_single_scenario():
    """演示单个动态障碍物场景"""
    print("\n=== 单场景演示模式 ===")

    # 创建简单的测试场景
    dynamic_obstacles = create_dynamic_obstacles_by_type('basic')
    scenario = Scenairo(
        init_pos=[100, 100],
        init_dir=45,
        goal_pos=[600, 600],
        goal_dir=0,
        circle_obstacles=[[300, 300, 40], [450, 200, 35]],
        dynamic_obstacles=dynamic_obstacles
    )

    env = Flight()
    state = env.reset(scenario)

    print("开始演示...")
    print(f"起始位置: {scenario.init_pos}")
    print(f"目标位置: {scenario.goal_pos}")
    print(f"动态障碍物数量: {len(scenario.dynamic_obstacles)}")

    # 简单的路径规划演示
    for step in range(100):
        # 使用简单的策略：主要向目标前进，偶尔转向
        if step % 10 == 0:
            action = np.random.choice([8, 12, 16])  # 左转、直行、右转
        else:
            action = 12  # 主要直行

        if action >= env.action.n_actions:
            action = action % env.action.n_actions

        state, reward, done = env.step(action)
        env.render(sleep_time=0.1, show_trace=True, show_pos=True, show_arrow=True)

        if step % 20 == 0:
            print(f"步骤 {step}: 位置=({env.current_pos[0]:.1f}, {env.current_pos[1]:.1f}), 奖励={reward:.3f}")

        if done:
            print(f"任务完成于步骤 {step}!")
            result_names = ['超时', '失败', '边界碰撞', '成功']
            print(f"结果: {result_names[env.result]}")
            break

    input("按回车键结束演示...")


if __name__ == "__main__":
    # 动态障碍物相关参数
    args = DataArgs(train=False, nth_times=25)  # 设置为False进行预测测试 True,False
    args.max_n_step = 200

    # 指定具体的算法和模型
    args.algo_t = 'AC'  # 或 'DDPG', 'DQN', 等
    args.model_t = 'cnn'  # 或 'mlp', 'cnn', 等
    args.env_t = 'flight'

    # 动态障碍物类型选择 - 新增随机运动模式
    # 可选: 'basic', 'fast', 'slow_large', 'mixed', 'chaotic', 'random'
    args.dynamic_obstacle_type = 'basic'  # 使用混乱模式测试随机运动

    """for prediction"""
    args.draw_rate = 0.1

    """for training"""
    args.episodes = 1
    args.nth_rounds = 1
    args.percentage = 0.3

    print("动态障碍物环境配置:")
    print(args.__dict__)
    print(f"\n动态障碍物运动模式说明:")
    print(f"- basic: 基础随机运动，适中的方向变化")
    print(f"- fast: 快速随机运动，频繁的方向变化")
    print(f"- slow_large: 慢速大型障碍物，较少方向变化")
    print(f"- mixed: 混合速度和大小，不同变化频率")
    print(f"- chaotic: 混乱模式，高度不可预测")
    print(f"- random: 完全随机生成的障碍物")
    print(f"\n当前使用模式: {args.dynamic_obstacle_type}")

    # 选择运行模式
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        # 演示模式
        demo_single_scenario()
    else:
        # 正常训练/预测模式
        run(args)