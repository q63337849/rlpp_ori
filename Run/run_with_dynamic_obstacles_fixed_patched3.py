
import os
import math
import traceback
import numpy as np
import tensorflow as tf

print(f"TensorFlow: {tf.__version__}")
try:
    print(f"Eager: {tf.executing_eagerly()}")
except Exception:
    pass

# ==== Project imports ====
import common
import options
from Run.data import DataPath, DataArgs
from Run.exec import Exec  # progress-bar executor
import Algo
import Scene  # original scene loader

# Use the dynamic-obstacle flight env from rlpp_ori-style projects
from Env.flight_with_dynamic_obstacles import (
    Flight, DynamicObstacle, Scenairo,  # Scenairo name as in project
    DiscreteAction, ContinueAction
)

# ---- Agent import fallback + thin predict agent ----
RLAgent = None
try:
    from Algo.agent import Agent as RLAgent
except Exception:
    try:
        from Algo.Agent import Agent as RLAgent
    except Exception:
        try:
            from Agent import Agent as RLAgent
        except Exception:
            RLAgent = None

class _ThinPredictAgent:
    def __init__(self, action, model):
        self.action = action
        self.model = model

__agent = None

# ---------- Robust dummy-state builder ----------
def _dummy_from_env(env):
    # Prefer env.get_state()
    try:
        s = env.get_state()
        s = np.asarray(s, dtype=np.float32)
        if s.ndim >= 1:
            return s[None, ...]
    except Exception:
        pass
    # Fallback to d_states
    ds = getattr(env, "d_states", None)
    if isinstance(ds, int):
        return np.zeros((1, ds), np.float32)
    if isinstance(ds, (tuple, list)):
        return np.zeros((1, *tuple(ds)), np.float32)
    # Last resort
    return np.zeros((1, 84, 84, 3), np.float32)

# ---------- Agent factory ----------
def default_agent(env, algo_t=options.dqn, model_t=options.M_CNN, force_new=False, for_train=False):
    global __agent
    if __agent is None or force_new:
        algo = Algo.AlgoDispatch(algo_t, model_t)
        supervisor = algo(buffer_size=500)
        if algo_t.upper() == "DDPG":
            action_dim = 1
            model = supervisor(env.d_states, action_dim, critic_lr=0.001)
        else:
            model = supervisor(env.d_states, env.action.n_actions, critic_lr=0.001)

        if RLAgent is not None:
            __agent = RLAgent(
                env.action, model,
                batch_size=64, score_offset=-1.0, discount=0.98,
                critic_n_layers=(20, 20, 20), n_filters=5, state_n_layers=(20,)
            )
        else:
            if for_train:
                print("⚠️ 未找到 RLAgent 类，训练不可用；仅创建推理适配器。")
            __agent = _ThinPredictAgent(env.action, model)
    return __agent

# ---------- Dynamic obstacles ----------
def create_dynamic_obstacles_by_type(dynamic_type='basic'):
    if dynamic_type == 'basic':
        return [
            DynamicObstacle([150, 500], 30, 2.0, None, [50, 50, 650, 650], 0.03),
            DynamicObstacle([550, 350], 25, 1.5, None, [50, 50, 650, 650], 0.05),
            DynamicObstacle([300, 100], 35, 1.8, None, [50, 50, 650, 650], 0.02),
        ]
    elif dynamic_type == 'fast':
        return [
            DynamicObstacle([200, 200], 25, 4.0, None, [50, 50, 650, 650], 0.08),
            DynamicObstacle([400, 400], 30, 3.5, None, [50, 50, 650, 650], 0.07),
            DynamicObstacle([600, 600], 20, 3.8, None, [50, 50, 650, 650], 0.09),
        ]
    elif dynamic_type == 'slow_large':
        return [
            DynamicObstacle([250, 550], 45, 1.0, None, [50, 50, 650, 650], 0.01),
            DynamicObstacle([500, 250], 50, 0.8, None, [50, 50, 650, 650], 0.01),
        ]
    elif dynamic_type == 'mixed':
        return [
            DynamicObstacle([200, 500], 30, 2.0, None, [50, 50, 650, 650], 0.03),
            DynamicObstacle([500, 200], 20, 3.5, None, [50, 50, 650, 650], 0.07),
            DynamicObstacle([350, 350], 40, 1.2, None, [50, 50, 650, 650], 0.02),
        ]
    elif dynamic_type == 'chaotic':
        return [
            DynamicObstacle([180, 520], 30, 3.0, None, [50, 50, 650, 650], 0.15),
            DynamicObstacle([520, 180], 30, 3.0, None, [50, 50, 650, 650], 0.15),
            DynamicObstacle([350, 350], 30, 3.0, None, [50, 50, 650, 650], 0.15),
        ]
    elif dynamic_type == 'random':
        rng = np.random.default_rng(42)
        dyns = []
        for _ in range(3):
            cx, cy = rng.integers(100, 600, size=2)
            r = int(rng.integers(20, 45))
            v = float(rng.uniform(0.8, 3.5))
            p = float(rng.uniform(0.01, 0.12))
            dyns.append(DynamicObstacle([int(cx), int(cy)], r, v, None, [50, 50, 650, 650], p))
        return dyns
    else:
        return create_dynamic_obstacles_by_type('basic')

def convert_to_dynamic_scenarios(original_scenarios, dynamic_type='basic', seed=None):
    dyn_scenarios = []
    if seed is not None:
        np.random.seed(seed)
        print(f"  使用随机种子: {seed}")
    dyns = create_dynamic_obstacles_by_type(dynamic_type)
    for sc in original_scenarios:
        dyn_scenarios.append(
            Scenairo(
                init_pos=sc.init_pos,
                init_dir=sc.init_dir,
                goal_pos=sc.goal_pos,
                goal_dir=sc.goal_dir,
                circle_obstacles=sc.circle_obstacles,
                line_obstacles=getattr(sc, "line_obstacles", None),
                dynamic_obstacles=dyns,
            )
        )
    return dyn_scenarios

# ---------- Train / Predict ----------
def training(env, agent, paths, train_scenes, predict_scenes):
    print("\n=== 动态障碍物环境训练开始 ===")
    print(f"训练场景数量: {len(train_scenes)}")
    print(f"预测场景数量: {len(predict_scenes)}")
    if train_scenes and hasattr(train_scenes[0], 'dynamic_obstacles'):
        print(f"每个场景的动态障碍物数量: {len(train_scenes[0].dynamic_obstacles)}")

    print("\n🔧 构建模型...")
    try:
        if hasattr(agent, "build_model"):
            agent.build_model()
        elif hasattr(agent, "model") and hasattr(agent.model, "build_model"):
            agent.model.build_model()
        else:
            _ = agent.model.action(_dummy_from_env(env))
        print("✓ 模型结构已构建")
    except Exception as e:
        print(f"  ⚠️ 构建模型失败：{e}")

    train_exec = Exec(env=env, agent=agent, scenes=train_scenes, paths=paths,
                      train=True, episodes=paths.args.episodes, max_n_step=paths.args.max_n_step)
    train_exec()

    print("\n=== 训练后完整预测（全集） ===")
    predict_exec = Exec(env=env, agent=agent, scenes=predict_scenes, paths=paths,
                        train=False, episodes=1, max_n_step=paths.args.max_n_step)
    predict_exec()
    print("\n✅ 训练结束")

def prediction(env, agent, paths, predict_scenes):
    print("\n=== 动态障碍物环境预测开始 ===")

    model_path = str(paths.model_load_file())  # ✅ 取真实字符串路径
    print(f"\n尝试加载模型: {model_path}")
    if not os.path.exists(model_path):
        print("✗ 模型文件不存在！")
        return
    print("✓ 模型文件存在")
    print(f"  文件大小: {os.path.getsize(model_path) / 1024:.2f} KB")

    print("\n🔧 构建推理模型...")
    try:
        if hasattr(agent, "build_model"):
            agent.build_model()
        elif hasattr(agent, "model") and hasattr(agent.model, "build_model"):
            agent.model.build_model()
        else:
            _ = agent.model.action(_dummy_from_env(env))
        print("✓ 模型结构已构建")
    except Exception as e:
        print(f"  ⚠️ 构建模型失败：{e}，尝试假前向")
        try:
            _ = agent.model.action(_dummy_from_env(env))
            print("✓ 通过假前向完成变量创建")
        except Exception as ee:
            print(f"  ✗ 假前向失败：{ee}")
            return

    print("\n📦 加载模型权重...")
    try:
        if hasattr(agent, "model") and hasattr(agent.model, "load_model"):
            agent.model.load_model(model_path)
        elif hasattr(agent, "load_model"):
            agent.load_model(model_path)
        else:
            raise AttributeError("找不到 load_model 接口")
        print("✓ 模型加载完成")
    except Exception as e:
        print(f"✗ 加载模型失败: {e}")
        traceback.print_exc()
        return

    print("\n🔬 测试模型输出...")
    try:
        out = agent.model.action(_dummy_from_env(env)) if hasattr(agent, "model") else None
        if isinstance(out, tf.Tensor):
            out = out.numpy()
        if out is not None and out.ndim == 1:
            out = np.expand_dims(out, 0)
        if out is not None:
            max_prob = float(np.max(out))
            entropy  = float(-np.sum(out * np.log(out + 1e-10)))
            top3     = np.argsort(out[0])[-3:][::-1]
            top3p    = out[0][top3]
            print(f"  MaxProb: {max_prob:.4f}, Entropy: {entropy:.4f}")
            print(f"  Top 3 actions: {top3}, probs: {top3p}")
            if max_prob < 0.2:
                print("  ⚠️ 模型输出接近均匀分布（MaxProb < 0.2）")
    except Exception as e:
        print(f"  ⚠️ 前向检查失败：{e}")

    print("\n" + "=" * 70)
    print("  🔍 步骤1：在训练场景上测试")
    print("=" * 70)
    obstacle_type = getattr(paths.args, "obstacle_type", options.O_circle)
    if obstacle_type == options.O_circle:
        train_task = Scene.circle_train_task4x200
    else:
        train_task = Scene.line_train_task4x200

    scene_loader = Scene.ScenarioLoader()
    train_original = scene_loader.load_scene(**train_task, percentage=getattr(paths.args, "percentage", 0.3))
    dyns = convert_to_dynamic_scenarios(
        train_original,
        getattr(paths.args, "dynamic_obstacle_type", "basic"),
        seed=getattr(paths.args, "nth_times", -1)
    )
    print(f"训练场景数量: {len(dyns)}")
    test_on_train_exec = Exec(env=env, agent=agent, scenes=dyns, paths=paths,
                              train=False, episodes=1, max_n_step=paths.args.max_n_step)
    test_on_train_exec()

    print("\n" + "=" * 70)
    print("  📊 步骤2：在测试场景上测试")
    print("=" * 70)
    predict_exec = Exec(env=env, agent=agent, scenes=predict_scenes, paths=paths,
                        train=False, episodes=1, max_n_step=paths.args.max_n_step)
    predict_exec()

    print("\n✅ 预测结束")

def run(args):
    print("\n===== 动态障碍物实验配置 =====")
    print(vars(args))

    if args.obstacle_type == options.O_circle:
        train_task = Scene.circle_train_task4x200
        test_task  = Scene.circle_test_task4x100
    else:
        train_task = Scene.line_train_task4x200
        test_task  = Scene.line_test_task4x100

    loader = Scene.ScenarioLoader()
    print("\n加载原始训练场景: ")
    original_train = loader.load_scene(**train_task, percentage=args.percentage)
    print("\n加载原始预测场景: ")
    original_test  = loader.load_scene(**test_task)

    print(f"\n为场景添加动态障碍物 (类型: {args.dynamic_obstacle_type})...")
    seed = args.nth_times if args.nth_times >= 0 else 42
    print(f"使用种子 {seed} 生成动态障碍物")
    train_scenes   = convert_to_dynamic_scenarios(original_train, args.dynamic_obstacle_type, seed=seed)
    predict_scenes = convert_to_dynamic_scenarios(original_test,  args.dynamic_obstacle_type, seed=seed + 10000)

    print(f"增强后训练场景数量: {len(train_scenes)}")
    print(f"增强后预测场景数量: {len(predict_scenes)}")

    if args.algo_t.upper() == "DDPG":
        print("使用连续动作空间（DDPG算法）")
        action = ContinueAction(45, 10., math.pi/60, 4)
    else:
        print("使用离散动作空间（DQN/AC 等算法）")
        action = DiscreteAction(45, 10., math.pi/60, 4, 5)
    env = Flight(action=action)

    agent = default_agent(env, args.algo_t, args.model_t, force_new=not args.train, for_train=args.train)

    paths = DataPath(args)

    # ✅ Ensure model directory exists using the true string path
    try:
        model_dir = os.path.dirname(str(paths.model_load_file()))
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
    except Exception as e:
        print(f"⚠️ 创建模型目录失败: {e}")

    if args.train:
        training(env, agent, paths, train_scenes, predict_scenes)
    else:
        prediction(env, agent, paths, predict_scenes)

if __name__ == "__main__":
    args = DataArgs(train=False, nth_times=25)
    args.max_n_step = 200
    args.algo_t = 'AC'
    args.model_t = 'cnn'
    args.env_t = 'flight'
    args.obstacle_type = options.O_circle
    args.model_t_obstacle = 'C'
    args.dynamic_obstacle_type = 'basic'
    args.draw_rate = 0.1
    args.episodes = 1
    args.nth_rounds = 1
    args.percentage = 0.3

    print("\n可用动态障碍物模式：basic / fast / slow_large / mixed / chaotic / random")
    print(f"当前使用模式: {args.dynamic_obstacle_type}")

    run(args)
