
import os
import glob
import math
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
import Scene

# Env with dynamic obstacles
from Env.flight_with_dynamic_obstacles import (
    Flight, DynamicObstacle, Scenairo,
    DiscreteAction, ContinueAction
)

# ---------------- Small helpers ----------------
def _to_numpy(x):
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    return np.asarray(x, dtype=np.float32)

def _ensure_batch(x):
    x = _to_numpy(x)
    if x.ndim == 1:
        x = x[None, :]
    return x

def _as_str(p):  # DataPath 的路径对象转真实字符串
    try:
        return str(p)
    except Exception:
        return f"{p}"

def _weight_paths(paths):
    """返回常用的保存/加载路径: ckpt prefix / h5 / saved_model_dir"""
    prefix = _as_str(paths.model_save_file())  # 例如 .../shared_data/model_0
    h5     = prefix + ".h5"
    sm_dir = prefix + "_saved_model"
    return prefix, h5, sm_dir

# ---------------- Callable model adapter ----------------
class _CallableModelAdapter:
    """Wrap a bare callable into an object with .action(), .load_model(), .variables."""
    def __init__(self, fn):
        self._fn = fn
        self._vars = []
    def action(self, x):
        # Try call signatures; if all fail, return dummy distribution
        x = _ensure_batch(x)
        try:
            y = self._fn(x)
        except TypeError:
            try:
                y = self._fn()
            except Exception:
                # fallback uniform with size inferred from input last dim if possible
                n = x.shape[-1] if x.ndim >= 2 else 5
                y = np.ones((x.shape[0], int(n)), dtype=np.float32) / float(n)
                return y
        except Exception:
            n = x.shape[-1] if x.ndim >= 2 else 5
            y = np.ones((x.shape[0], int(n)), dtype=np.float32) / float(n)
            return y
        y = _to_numpy(y)
        if y.ndim == 1:
            y = y[None, :]
        # Normalize to a distribution
        row_sum = np.sum(np.abs(y), axis=1, keepdims=True) + 1e-8
        y = y / row_sum
        return y
    def load_model(self, path):
        print("⚠️ 当前模型为 callable 适配器，没有可加载的权重接口，跳过加载。")
    @property
    def variables(self):
        return self._vars
    def build_model(self):
        return
    def save_weights(self, path):
        print("⚠️ callable 适配器不支持 save_weights，跳过。")

# ---------------- Agent import (with fallbacks) ----------------
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

# ---------------- Exec-compatible thin agent ----------------
class _ThinPredictAgent:
    off_policy = False
    def __init__(self, action, model, env=None):
        self.action = action
        self.model = model
        self.env = env

    def build_model(self):
        if hasattr(self.model, "build_model"):
            self.model.build_model()
        else:
            try:
                dummy = np.zeros((1, getattr(self.action, 'n_actions', 1)), dtype=np.float32)
                _ = self.model.action(dummy)
            except Exception:
                pass

    def load_model(self, path):
        if hasattr(self.model, "load_model"):
            return self.model.load_model(path)
        print("⚠️ 当前模型无 load_model 接口，跳过加载。")

    def save_weights(self, *args, **kwargs):
        if hasattr(self.model, "save_weights"):
            return self.model.save_weights(*args, **kwargs)
        print("⚠️ 当前模型无 save_weights 接口，跳过保存。")

    def _probs(self, obs):
        obs = _ensure_batch(obs)
        try:
            probs = self.model.action(obs)
            probs = _to_numpy(probs)
        except Exception as e:
            # Fallback uniform distribution
            n = getattr(self.action, 'n_actions', 5)
            probs = np.ones((obs.shape[0], n), dtype=np.float32) / float(n)
            return probs
        if probs.ndim == 1:
            probs = probs[None, :]
        # If model outputs logits, softmax it
        if np.any(probs < 0) or np.any(probs > 1.0):
            e = np.exp(probs - np.max(probs, axis=1, keepdims=True))
            probs = e / (np.sum(e, axis=1, keepdims=True) + 1e-8)
        # Clip & renorm
        probs = np.clip(probs, 1e-6, 1.0)
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs

    def predict(self, obs):
        probs = self._probs(obs)
        act = int(np.argmax(probs[0]))
        return act

    def sample(self, obs):
        probs = self._probs(obs)[0]
        n = getattr(self.action, 'n_actions', len(probs))
        if len(probs) != n:
            probs = np.resize(probs, n)
            probs = np.clip(probs, 1e-6, 1.0)
            probs = probs / np.sum(probs)
        act = int(np.random.choice(np.arange(n), p=probs))
        return act

    def reset(self):
        return

    def learn(self, *args, **kwargs):
        return 0.0, 0.0

__agent = None

# ---------------- Dummy from env ----------------
def _dummy_from_env(env):
    try:
        s = env.get_state()
        s = _to_numpy(s)
        if s.ndim >= 1:
            return s[None, ...]
    except Exception:
        pass
    ds = getattr(env, "d_states", None)
    if isinstance(ds, int):
        return np.zeros((1, ds), np.float32)
    if isinstance(ds, (tuple, list)):
        return np.zeros((1, *tuple(ds)), np.float32)
    return np.zeros((1, 84, 84, 3), np.float32)

# ---------------- Agent factory ----------------
def default_agent(env, algo_t=options.dqn, model_t=options.M_CNN, force_new=False, for_train=False):
    global __agent
    if __agent is None or force_new:
        algo = Algo.AlgoDispatch(allo_t=algo_t, model_t=model_t) if hasattr(Algo, "AlgoDispatch") else Algo.AlgoDispatch(algo_t, model_t)  # support different signatures
        supervisor = algo(buffer_size=500)
        if algo_t.upper() == "DDPG":
            action_dim = 1
            model = supervisor(env.d_states, action_dim, critic_lr=0.001)
        else:
            model = supervisor(env.d_states, env.action.n_actions, critic_lr=0.001)
        if not hasattr(model, "action"):
            print("ℹ️ 检测到 supervisor 返回的是可调用对象，使用 _CallableModelAdapter 适配。")
            model = _CallableModelAdapter(model)
        if RLAgent is not None:
            __agent = RLAgent(
                env.action, model,
                batch_size=64, score_offset=-1.0, discount=0.98,
                critic_n_layers=(20, 20, 20), n_filters=5, state_n_layers=(20,)
            )
        else:
            if for_train:
                print("⚠️ 未找到 RLAgent 类，训练不可用；仅创建推理适配器。")
            __agent = _ThinPredictAgent(env.action, model, env=env)
    return __agent

# ---------------- Dynamic obstacles ----------------
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

# ---------------- Save/Load full weights ----------------
def save_full_weights(agent, paths):
    """尽可能把‘完整权重’落盘（优先Keras H5；其次TF Checkpoint；最后SavedModel）。"""
    prefix, h5, sm_dir = _weight_paths(paths)
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    model = getattr(agent, "model", agent)

    # 1) Keras save_weights (最易用)
    try:
        if hasattr(model, "save_weights"):
            model.save_weights(h5)
            print(f"✅ 已保存 Keras 权重文件: {h5}")
            return True
    except Exception as e:
        print(f"⚠️ 保存 H5 失败，尝试 TF Checkpoint。原因: {e}")

    # 2) TF Checkpoint
    try:
        ckpt = tf.train.Checkpoint(model=model)
        path = ckpt.save(prefix)
        print(f"✅ 已保存 TF Checkpoint 前缀: {path}")
        return True
    except Exception as e:
        print(f"⚠️ 保存 Checkpoint 失败，尝试 SavedModel。原因: {e}")

    # 3) SavedModel
    try:
        if hasattr(model, "save"):
            model.save(sm_dir)  # tf.keras.Model.save(dir)
            print(f"✅ 已保存 SavedModel 目录: {sm_dir}")
            return True
    except Exception as e:
        print(f"⚠️ 保存 SavedModel 失败。原因: {e}")

    print("❌ 未能保存任何可用的权重格式，请检查模型类型/接口。")
    return False

def load_full_weights(agent, paths):
    """稳健加载：优先 H5；否则 Checkpoint；否则 SavedModel；最后回退项目自带 load_model。"""
    prefix, h5, sm_dir = _weight_paths(paths)
    model = getattr(agent, "model", agent)

    # 先创建变量图（避免 restore 未建图失败）
    try:
        if hasattr(model, "build_model"):
            model.build_model()
        else:
            dummy = np.zeros((1, getattr(getattr(agent, "action", None), "n_actions", 5)), np.float32)
            if hasattr(model, "action"):
                _ = model.action(dummy)
    except Exception:
        pass

    # 1) H5
    if os.path.exists(h5):
        try:
            if hasattr(model, "load_weights"):
                model.load_weights(h5)
                print(f"✅ 已加载 Keras 权重: {h5}")
                return True
        except Exception as e:
            print(f"⚠️ 加载 H5 失败，尝试 TF Checkpoint。原因: {e}")

    # 2) TF Checkpoint：检测配套的 index+data
    ckpt_index = prefix + ".index"
    data_files = glob.glob(prefix + ".data-*")
    if os.path.exists(ckpt_index) and data_files:
        try:
            ckpt = tf.train.Checkpoint(model=model)
            ckpt.restore(prefix).expect_partial()
            print(f"✅ 已加载 TF Checkpoint: {prefix} （检测到 {os.path.basename(ckpt_index)} 和 {len(data_files)} 个 data shard）")
            return True
        except Exception as e:
            print(f"⚠️ 加载 Checkpoint 失败，尝试 SavedModel。原因: {e}")

    # 3) SavedModel
    if os.path.isdir(sm_dir):
        try:
            loaded = tf.keras.models.load_model(sm_dir)
            if hasattr(model, "set_weights") and hasattr(loaded, "get_weights"):
                model.set_weights(loaded.get_weights())
                print(f"✅ 已从 SavedModel 复制权重: {sm_dir}")
                return True
            else:
                print("⚠️ SavedModel 已加载，但当前 model 不支持 set_weights，跳过复制。")
        except Exception as e:
            print(f"⚠️ 加载 SavedModel 失败。原因: {e}")

    # 4) 项目自带 load_model（最后兜底）
    try:
        if hasattr(model, "load_model"):
            model.load_model(_as_str(paths.model_load_file()))
            print("ℹ️ 已使用项目自带的 load_model()。")
            return True
    except Exception as e:
        print(f"⚠️ 项目自带 load_model 也失败。原因: {e}")

    print("❌ 未找到可用的权重文件（H5/Checkpoint/SavedModel），或加载失败。")
    return False

# ---------------- Train / Predict ----------------
def training(env, agent, paths, train_scenes, predict_scenes, args):
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
                      train=True, episodes=args.episodes, max_n_step=args.max_n_step)
    train_exec()

    # ✅ 训练后：保存完整权重（H5/Checkpoint/SavedModel 任一成功即可）
    print("\n💾 正在保存完整权重 ...")
    ok = save_full_weights(agent, paths)
    if not ok:
        print("❗建议检查：模型是否为 Keras/能通过 tf.train.Checkpoint 包装；若均不是，请在自定义模型类中实现 save/load。")

    # 训练后全量预测（验证保存是否可加载）
    print("\n=== 训练后完整预测（全集） ===")
    # 尝试重新加载一次（验证保存正确性）
    load_full_weights(agent, paths)

    predict_exec = Exec(env=env, agent=agent, scenes=predict_scenes, paths=paths,
                        train=False, episodes=1, max_n_step=args.max_n_step)
    predict_exec()
    print("\n✅ 训练结束")

def prediction(env, agent, paths, predict_scenes, args):
    print("\n=== 动态障碍物环境预测开始 ===")
    prefix, h5, sm_dir = _weight_paths(paths)

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

    # 📦 加载：H5 → Checkpoint → SavedModel → 项目自带 load_model
    print("\n📦 加载模型权重...")
    loaded = load_full_weights(agent, paths)
    if not loaded:
        base_dir = os.path.dirname(prefix)
        print("目录列举：", base_dir)
        try:
            for f in os.listdir(base_dir):
                print(" -", f)
        except Exception:
            pass

    # 简单前向检查
    print("\n🔬 测试模型输出...")
    try:
        n = getattr(env.action, 'n_actions', 5)
        dummy = np.zeros((1, n), np.float32)
        out = None
        if hasattr(agent, "_probs"):
            out = agent._probs(dummy)
        elif hasattr(agent, "model") and hasattr(agent.model, "action"):
            out = agent.model.action(dummy)

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

    # 在训练集上测试
    print("\n" + "=" * 70)
    print("  🔍 步骤1：在训练场景上测试")
    print("=" * 70)
    obstacle_type = getattr(args, "obstacle_type", options.O_circle)
    if obstacle_type == options.O_circle:
        train_task = Scene.circle_train_task4x200
    else:
        train_task = Scene.line_train_task4x200

    scene_loader = Scene.ScenarioLoader()
    train_original = scene_loader.load_scene(**train_task, percentage=getattr(args, "percentage", 0.3))
    dyns = convert_to_dynamic_scenarios(
        train_original,
        getattr(args, "dynamic_obstacle_type", "basic"),
        seed=getattr(args, "nth_times", -1)
    )
    print(f"训练场景数量: {len(dyns)}")
    test_on_train_exec = Exec(env=env, agent=agent, scenes=dyns, paths=paths,
                              train=False, episodes=1, max_n_step=args.max_n_step)
    test_on_train_exec()

    # 在测试集上测试
    print("\n" + "=" * 70)
    print("  📊 步骤2：在测试场景上测试")
    print("=" * 70)
    predict_exec = Exec(env=env, agent=agent, scenes=predict_scenes, paths=paths,
                        train=False, episodes=1, max_n_step=args.max_n_step)
    predict_exec()

    print("\n✅ 预测结束")

# ---------------- Orchestrator ----------------
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
    try:
        model_dir = os.path.dirname(_as_str(paths.model_load_file()))
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
    except Exception as e:
        print(f"⚠️ 创建模型目录失败: {e}")

    if args.train:
        training(env, agent, paths, train_scenes, predict_scenes, args)
    else:
        prediction(env, agent, paths, predict_scenes, args)

if __name__ == "__main__":
    args = DataArgs(train=True, nth_times=26)
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
