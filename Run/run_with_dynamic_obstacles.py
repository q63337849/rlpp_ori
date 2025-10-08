import common
import options
import sys
from Run.data import DataPath, DataArgs
import Algo
import Scene  # 使用原始Scene模块
from Run.exec import Train, Predict
import numpy as np
import math

# 导入动态障碍物相关类
from Env.flight_with_dynamic_obstacles import Flight, DynamicObstacle, Scenairo


__agent = None


def default_agent(env, algo_t=options.dqn, model_t=options.M_CNN):
    """简化使用难度，设计得不够好"""
    global __agent
    if __agent is None:
        algo = Algo.AlgoDispatch(algo_t, model_t)
        supervisor = algo(buffer_size=500)

        # 根据算法类型处理动作空间
        if algo_t.upper() == 'DDPG':
            # DDPG使用连续动作，通常是1维输出
            action_dim = 1  # DDPG通常输出单一连续值
            model = supervisor(env.d_states, action_dim, critic_lr=0.001)
        else:
            # DQN等使用离散动作
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
            DynamicObstacle([200, 500], 15, 4.0, None, [50, 50, 650, 650], 0.09)   # 快小
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


def convert_to_dynamic_scenarios(original_scenarios, dynamic_type='basic'):
    """将原始场景转换为带动态障碍物的场景"""
    dynamic_scenarios = []
    dynamic_obstacles = create_dynamic_obstacles_by_type(dynamic_type)

    print(f"正在为 {len(original_scenarios)} 个场景添加 {len(dynamic_obstacles)} 个动态障碍物")

    for i, orig_scenario in enumerate(original_scenarios):
        # 使用我们的Scenairo类创建新场景
        enhanced_scenario = Scenairo(
            init_pos=orig_scenario.init_pos,
            init_dir=orig_scenario.init_dir,
            goal_pos=orig_scenario.goal_pos,
            goal_dir=orig_scenario.goal_dir,
            circle_obstacles=orig_scenario.circle_obstacles,
            line_obstacles=getattr(orig_scenario, 'line_obstacles', None),
            dynamic_obstacles=create_dynamic_obstacles_by_type(dynamic_type)  # 每个场景独立的动态障碍物
        )
        dynamic_scenarios.append(enhanced_scenario)

    return dynamic_scenarios


def training(env, agent, paths, train_scenes, predict_scenes):
    print("\n=== 动态障碍物环境训练开始 ===")
    print(f"训练场景数量: {len(train_scenes)}")
    print(f"预测场景数量: {len(predict_scenes)}")

    # 检查场景中的动态障碍物
    if train_scenes and hasattr(train_scenes[0], 'dynamic_obstacles'):
        print(f"每个场景的动态障碍物数量: {len(train_scenes[0].dynamic_obstacles)}")

    train = Train(draw=False, episodes=args.episodes, max_n_step=args.max_n_step)
    predict = Predict(draw=common.use_windows, draw_rate=0.01, max_n_step=args.max_n_step, save=True)
    agent.build_model()

    for i in range(args.nth_rounds):
        print(f'当前训练轮次: {i+1}/{args.nth_rounds}')
        train(env, agent, train_scenes, paths)
        agent.save_model(paths.model_save_file())
        predict(env, agent, predict_scenes, paths)
    print("=== 训练结束 ===")


def prediction(env, agent, paths, predict_scenes):
    print("\n=== 动态障碍物环境预测开始 ===")
    print(f"预测场景数量: {len(predict_scenes)}")

    # 检查场景中的动态障碍物
    if predict_scenes and hasattr(predict_scenes[0], 'dynamic_obstacles'):
        print(f"每个场景的动态障碍物数量: {len(predict_scenes[0].dynamic_obstacles)}")

    predict = Predict(draw=common.use_windows, draw_rate=args.draw_rate, max_n_step=args.max_n_step)

    # 使用最新训练的模型
    agent.load_model(paths.model_load_file())
    predict(env, agent, predict_scenes, paths)
    print("=== 预测结束 ===")


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

    # 转换为带动态障碍物的场景
    print(f"\n为场景添加动态障碍物 (类型: {args.dynamic_obstacle_type})...")
    train_scenes = convert_to_dynamic_scenarios(original_train_scenes, args.dynamic_obstacle_type)
    predict_scenes = convert_to_dynamic_scenarios(original_predict_scenes, args.dynamic_obstacle_type)

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
    args = DataArgs(train=False, nth_times=-1)  # 设置为False进行预测测试 True,False
    args.max_n_step = 200

    # 指定具体的算法和模型
    args.algo_t = 'AC'        # 或 'DDPG', 'DQN', 等
    args.model_t = 'cnn'       # 或 'mlp', 'cnn', 等
    args.env_t = 'flight'

    # 动态障碍物类型选择 - 新增随机运动模式
    # 可选: 'basic', 'fast', 'slow_large', 'mixed', 'chaotic', 'random'
    args.dynamic_obstacle_type = 'chaotic'  # 使用混乱模式测试随机运动

    """for prediction"""
    args.draw_rate = 0.1

    """for training"""
    args.episodes = 15
    args.nth_rounds = 5
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