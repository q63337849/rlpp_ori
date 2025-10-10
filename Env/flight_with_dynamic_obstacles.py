# ===================================
# Migrated from TensorFlow 1.x to 2.x
# Original file: D:\codetest\rlpp_ori\Env\flight_with_dynamic_obstacles.py
# Migration may not be complete. 
# Please review TODO comments.
# ===================================

""" 修复版：实时路径规划环境（带动态障碍物）
修复了动态障碍物显示为椭圆的问题
"""

import numpy as np
import math
from Geometry.geometry import Geometry, Trajectory
from Draw.draw import Canvas


class DynamicObstacle:
    """动态障碍物类 - 支持随机方向运动"""
    def __init__(self, init_pos, radius, velocity, direction=None, bounds=None, change_direction_probability=0.02):
        """
        init_pos: 初始位置 [x, y]
        radius: 半径
        velocity: 速度 (像素/步)
        direction: 初始方向 (弧度)，如果为None则随机生成
        bounds: 边界 [x_min, y_min, x_max, y_max]，用于反弹
        change_direction_probability: 每步随机改变方向的概率
        """
        self.init_pos = np.array(init_pos, dtype=np.float64)
        self.pos = self.init_pos.copy()
        self.radius = radius
        self.velocity = velocity

        # 如果没有指定方向，随机生成
        if direction is None:
            self.direction = np.random.uniform(0, 2 * math.pi)
            self.init_direction = self.direction
        else:
            self.direction = direction
            self.init_direction = direction

        self.bounds = bounds if bounds is not None else [0, 0, 700, 700]
        self.change_direction_probability = change_direction_probability
        self.canvas_item = None  # 用于跟踪canvas上的绘制项

        # 随机运动参数
        self.direction_change_timer = 0
        self.direction_change_interval = np.random.randint(20, 60)  # 随机间隔改变方向

    def update(self):
        """更新障碍物位置 - 支持随机方向变化"""
        # 随机改变方向
        if np.random.random() < self.change_direction_probability:
            # 随机选择新方向
            self.direction = np.random.uniform(0, 2 * math.pi)

        # 定期改变方向
        self.direction_change_timer += 1
        if self.direction_change_timer >= self.direction_change_interval:
            self.direction = np.random.uniform(0, 2 * math.pi)
            self.direction_change_timer = 0
            self.direction_change_interval = np.random.randint(20, 60)  # 重新随机间隔

        # 计算新位置
        new_pos = self.pos + self.velocity * np.array([np.cos(self.direction), np.sin(self.direction)])

        # 边界碰撞检测和随机反弹
        direction_changed = False
        if new_pos[0] - self.radius <= self.bounds[0] or new_pos[0] + self.radius >= self.bounds[2]:
            # 水平边界碰撞 - 随机反弹角度
            base_angle = math.pi - self.direction  # 基础反射角
            random_variation = np.random.uniform(-math.pi/4, math.pi/4)  # 添加随机变化
            self.direction = base_angle + random_variation
            direction_changed = True

        if new_pos[1] - self.radius <= self.bounds[1] or new_pos[1] + self.radius >= self.bounds[3]:
            # 垂直边界碰撞 - 随机反弹角度
            base_angle = -self.direction  # 基础反射角
            random_variation = np.random.uniform(-math.pi/4, math.pi/4)  # 添加随机变化
            self.direction = base_angle + random_variation
            direction_changed = True

        # 如果方向改变了，重置定期改变计时器
        if direction_changed:
            self.direction_change_timer = 0
            self.direction_change_interval = np.random.randint(30, 80)

        # 更新位置
        self.pos = self.pos + self.velocity * np.array([np.cos(self.direction), np.sin(self.direction)])

        # 确保在边界内
        self.pos[0] = np.clip(self.pos[0], self.bounds[0] + self.radius, self.bounds[2] - self.radius)
        self.pos[1] = np.clip(self.pos[1], self.bounds[1] + self.radius, self.bounds[3] - self.radius)

    def reset(self):
        """重置到初始位置"""
        self.pos = self.init_pos.copy()
        self.direction = self.init_direction
        self.canvas_item = None
        self.direction_change_timer = 0
        self.direction_change_interval = np.random.randint(20, 60)

    def get_obstacle_data(self):
        """返回障碍物数据 [x, y, radius]"""
        return np.array([self.pos[0], self.pos[1], self.radius])


class Scenairo:
    region_size = np.array([700, 700])
    center_pos = region_size // 2
    # 任务起点与终点之间的最大直线距离，用于归一化
    max_straight_line_dis = math.hypot(650 - 50, 650 - 50)

    def __init__(self, init_pos, init_dir, goal_pos, goal_dir, circle_obstacles=None, line_obstacles=None, dynamic_obstacles=None):
        """
        pos: 位置，(x,y)
        dir：方向，
        dynamic_obstacles: 动态障碍物列表
        """
        self.init_pos, self.init_dir = init_pos, init_dir
        self.goal_pos, self.goal_dir = goal_pos, goal_dir
        self.circle_obstacles, self.line_obstacles = circle_obstacles, line_obstacles
        self.dynamic_obstacles = dynamic_obstacles if dynamic_obstacles is not None else []

    def __str__(self):
        return "init_p: {}, goal_pos: {}, init_dir: {}, dynamic_obs: {}".format(
            self.init_pos, self.goal_pos, self.init_dir, len(self.dynamic_obstacles))


class Action:
    """Map agent's actions to control inputs"""

    def __init__(self,
                 max_turning_angle,  # 转弯角度
                 min_turning_radius,  # 转弯半径
                 max_angular_velocity  # 最大转弯角速度
                 ):
        """不保存目标状态"""
        self.max_turning_angle = math.radians(max_turning_angle)
        self.max_angular_velocity = max_angular_velocity
        self.linear_velocity = self.max_angular_velocity * min_turning_radius  # math.pi/6
        self.basic_time_step = self.max_turning_angle / self.max_angular_velocity
        self.basic_distance_step = self.linear_velocity * self.basic_time_step  # 固定的路程步长

    def __call__(self, action):
        raise NotImplementedError


class DiscreteAction(Action):
    def __init__(self,
                 max_turning_angle,  # 转弯角度
                 min_turning_radius,  # 转弯半径
                 max_angular_velocity,  # 最大转弯角速度
                 n_granularity=4, n_actions=5,
                 ):
        super().__init__(max_turning_angle, min_turning_radius, max_angular_velocity)
        # 动作设计：已知动作个数和max_angle
        self.n_granularity = n_granularity
        self.max_time_step = self.basic_time_step * self.n_granularity
        self.max_distance_step = self.basic_distance_step * self.n_granularity
        assert n_actions % 2 == 1
        self.n_actions = n_actions
        self.actions = np.linspace(-self.max_turning_angle, self.max_turning_angle, self.n_actions)
        self.forward_action = self.n_actions // 2
        self.n_angle_actions = self.n_actions
        self.n_actions *= self.n_granularity  # 4个粒度等级

    def __call__(self, action):
        dis_step_times, angle_action = divmod(action, self.n_angle_actions)
        self.delta_dis = self.basic_distance_step * (dis_step_times + 1)
        self.delta_dir = self.actions[angle_action]


class ContinueAction(Action):
    def __init__(self,
                 max_turning_angle,  # 转弯角度
                 min_turning_radius,  # 转弯半径
                 max_angular_velocity,  # 最大转弯角速度
                 n_granularity=4
                 ):
        super().__init__(max_turning_angle, min_turning_radius, max_angular_velocity)
        # 动作设计：已知动作个数和max_angle
        self.n_granularity = n_granularity
        self.max_time_step = self.basic_time_step * self.n_granularity
        self.max_distance_step = self.basic_distance_step * self.n_granularity

        # 为DDPG算法添加必要的属性
        self.n_actions = 2  # 连续动作维度：[distance_multiplier, angle_multiplier]
        self.action_bound = 1.0  # 动作边界 [-1, 1]

    def __call__(self, action):
        """
        处理DDPG的连续动作输入
        action可以是：
        1. 单个浮点数 (DDPG输出) - 将被转换为角度控制
        2. 长度为2的数组 [distance_factor, angle_factor]
        """
        # 更安全的类型检查
        try:
            # 尝试转换为numpy数组以统一处理
            action_array = np.asarray(action)

            if action_array.ndim == 0 or (action_array.ndim == 1 and action_array.size == 1):
                # 单个值的情况 (包括numpy标量)
                angle_factor = float(action_array.flat[0])
                distance_factor = 0.5  # 固定中等速度
            elif action_array.ndim == 1 and action_array.size >= 2:
                # 数组且长度>=2的情况
                distance_factor, angle_factor = float(action_array[0]), float(action_array[1])
            else:
                # 其他情况，使用默认值
                distance_factor, angle_factor = 0.5, 0.0

        except (TypeError, ValueError, IndexError):
            # 如果转换失败，尝试直接处理
            try:
                # 尝试作为单个数值处理
                angle_factor = float(action)
                distance_factor = 0.5
            except (TypeError, ValueError):
                # 最后的备用方案
                distance_factor, angle_factor = 0.5, 0.0
                print(f"Warning: 无法解析动作 {action}，使用默认值")

        # 将[-1,1]范围的输入转换为实际的控制量
        # distance_factor: [-1,1] -> [1, n_granularity]的距离倍数
        distance_multiplier = (distance_factor + 1) / 2  # [-1,1] -> [0,1]
        distance_multiplier = distance_multiplier * (self.n_granularity - 1) + 1  # [0,1] -> [1,n]

        self.delta_dis = self.basic_distance_step * distance_multiplier

        # angle_factor: [-1,1] -> [-max_turning_angle, max_turning_angle]
        self.delta_dir = angle_factor * self.max_turning_angle


class BLPAction(Action):
    def __init__(self,
                 max_turning_angle,  # 转弯角度
                 min_turning_radius,  # 转弯半径
                 max_angular_velocity,  # 最大转弯角速度
                 ):
        super().__init__(max_turning_angle, min_turning_radius, max_angular_velocity)

    def __call__(self, action):
        delta_dir, delta_t = action
        self.delta_dir = delta_dir
        self.delta_dis = self.linear_velocity * delta_t


class Kinematic:
    def __init__(self):
        """不保存目标状态"""
        self.turning_angle_precision = 1 / 45  # 1/45 rad = 1.2732395447351628°

        # 状态量
        self.init_pos, self.init_dir = None, None
        self.goal_pos, self.goal_dir = None, None
        self.last_pos, self.last_dir = None, None
        self.current_pos, self.current_dir = None, None

        self.traj = Trajectory()

    def init(self, init_pos, init_dir, goal_pos, goal_dir):
        """设置初始状态，初始状态需要保存"""
        self.init_pos, self.init_dir = np.array(init_pos, dtype=np.float64), math.radians(init_dir)
        self.goal_pos, self.goal_dir = np.array(goal_pos, dtype=np.float64), math.radians(goal_dir)
        self.current_pos, self.current_dir = self.init_pos, self.init_dir

    def transition(self, delta_dir, delta_dis):
        """time_step; yaw_rate; speed"""
        if abs(delta_dir) < self.turning_angle_precision:  # 近似认为向前直线飞行
            delta_pos = np.array([delta_dis * np.cos(self.current_dir), delta_dis * np.sin(self.current_dir)])
            self.traj.line(self.current_pos, delta_pos)
        else:
            radius = delta_dis / delta_dir  # 有向半径，此处不能取绝对值，利用半径的正负号来统一左转和右转
            delta_x = radius * (np.sin(delta_dir - self.current_dir) + np.sin(self.current_dir))  # -delta_dir抵消右转角度为负
            delta_y = radius * (np.cos(delta_dir - self.current_dir) - np.cos(self.current_dir))
            delta_pos = np.array([delta_x, delta_y])
            self.traj.arc(self.current_pos, self.current_dir, radius, delta_dir)

        self.last_pos = self.current_pos
        self.current_pos = self.current_pos + delta_pos  # Any in-place operations are prohibited, don't write: self.current_pos += delta_pos
        self.last_dir = self.current_dir
        self.current_dir -= delta_dir  # python数据类型，此处为非原址操作


class Flight(Kinematic):
    timeout, failure, border, success, n_outcomes = 0, 1, 2, 3, 4

    def __init__(self,
                 action=None,  # 支持传入自定义action
                 max_turning_angle=45,
                 min_turning_radius=10.,
                 max_angular_velocity=math.pi / 60,
                 max_detect_range=120.,  # 探测范围
                 detect_angle_interval=5,  # 距离传感器间的角度间隔
                 max_detect_angle=90,  # 最大探测角度
                 safe_dis=0.,  # 安全距离
                 canvas_size=None,  # 画布大小，默认是场景区域大小
                 add_noise=False,  # 是否对距离传感器施加噪声
                 use_border=True  # 是否将场景边界视为障碍
                 ):
        super().__init__()
        self.region_width, self.region_height = Scenairo.region_size
        self.canvas = Canvas(Scenairo.region_size if canvas_size is None else canvas_size, Scenairo.region_size)
        self.redraw = None

        self.double_pi = 2 * math.pi

        # 设置动作空间
        if action is not None:
            self.action = action
        else:
            # 默认使用离散动作空间
            self.action = DiscreteAction(max_turning_angle, min_turning_radius, max_angular_velocity, 4, 5)

        # 目标方位相关
        self.last_raw_goal_abs_dir = None  # 记录地面坐标系下目标相对于机体的方向
        self.compensate_goal_abs_dir = None  # 跨越180°线会产生不连续的角度值，通过补偿将其变成连续值
        # 目标距离相关
        self.straight_line_dis = None
        # 障碍距离相关
        div, mod = divmod(max_detect_angle, detect_angle_interval)
        assert mod == 0
        self.max_detect_angle = math.radians(max_detect_angle)
        self.detect_angle_interval = math.radians(detect_angle_interval)
        self.safe_dis = safe_dis
        self.add_noise = add_noise
        self.max_detect_range = max_detect_range
        self.no_obstacle_dis = 2 * self.max_detect_range  # 探测方向上没有障碍时的距离，需要区别于障碍位于max_detect_range处

        # 决策信息：障碍距离+目标位置
        self.obs_distances, self.goal_dis, self.goal_dir = None, None, None
        self.last_goal_dis, self.last_goal_abs_dir = None, None  # 需要加绝对值

        self.d_obstacles = 2 * div + 1  # 周围环境
        self.d_states_detail = (self.d_obstacles, 2)  # 状态构成
        self.d_states = sum(self.d_states_detail)  # 周围环境+目标方位
        self.terminal = np.zeros(self.d_states)  # 结束状态
        self.result = None
        self.residual_traj = None  # 当前位置到目标位置的剩余轨迹（直线）

        print("d_obstacles: ", self.d_obstacles)

        # 控制量
        self.delta_dis = None
        self.delta_dir = None
        self.delta_yaw_rate = None
        self.delta_time = None
        self.yaw_rate = None

        self.circle_obstacles = None
        self.line_obstacles = np.array([
            [0, 0, self.region_width, 0],
            [self.region_width, 0, self.region_width, self.region_height],
            [self.region_width, self.region_height, 0, self.region_height],
            [0, self.region_height, 0, 0]], dtype=np.float64) if use_border else None

        # 动态障碍物
        self.dynamic_obstacles = []
        self.dynamic_obstacle_items = []  # 跟踪动态障碍物的canvas项

    def set_obstacles(self, circle_obstacles, line_obstacles, dynamic_obstacles=None):
        if circle_obstacles is not None:
            self.circle_obstacles = np.array(circle_obstacles)
        if line_obstacles is not None:
            self.line_obstacles = np.array(line_obstacles) if self.line_obstacles is None else \
                np.concatenate((self.line_obstacles, line_obstacles))
        if dynamic_obstacles is not None:
            self.dynamic_obstacles = dynamic_obstacles
            self.dynamic_obstacle_items = []  # 重置canvas项列表

    def update_dynamic_obstacles(self):
        """更新所有动态障碍物的位置"""
        for obstacle in self.dynamic_obstacles:
            obstacle.update()

    def get_current_dynamic_obstacles(self):
        """获取当前所有动态障碍物的位置和半径"""
        if not self.dynamic_obstacles:
            return None
        return np.array([obs.get_obstacle_data() for obs in self.dynamic_obstacles])

    def clear_dynamic_obstacles_from_canvas(self):
        """清除canvas上的动态障碍物"""
        for item in self.dynamic_obstacle_items:
            if item is not None:
                self.canvas.canvas.delete(item)
        self.dynamic_obstacle_items = []

    def draw_dynamic_obstacles(self):
        """绘制动态障碍物 - 确保保持圆形"""
        self.clear_dynamic_obstacles_from_canvas()  # 先清除旧的

        for obstacle in self.dynamic_obstacles:
            # 计算绘制位置
            center = self.canvas.transform_abs_coord(obstacle.pos)
            radius = obstacle.radius  # 直接使用像素半径

            # 绘制正圆
            lowerleft = center - [radius, radius]
            topright = center + [radius, radius]

            item = self.canvas.canvas.create_oval(
                *lowerleft, *topright,
                fill='red',
                outline='darkred',
                width=2
            )
            self.dynamic_obstacle_items.append(item)

    def render_trajectory(self, points, show_scope=False, circle_scope=True, show_arrow=False, show_pos=True):
        """根据点列绘制轨迹"""
        for i in range(len(points) - 1):
            self.traj.line2(points[i], points[i + 1])
            self.current_pos = points[i + 1]
            delta_x, delta_y = points[i + 1] - points[i]
            self.current_dir = math.atan2(delta_y, delta_x)
            self.render(0., True, show_scope, circle_scope, show_arrow, show_pos)

    def render(self, sleep_time=0.01, show_trace=True, show_scope=False, circle_scope=True, show_arrow=False,
               show_pos=False):
        # call step before render
        if self.redraw:  # draw scenairo
            self.redraw = False
            self.canvas.create()  # create or reset canvas
            # draw start point and goal point
            radius = 8
            self.canvas.draw_oval(self.init_pos, radius, 'black', transform_radius=False)
            self.canvas.draw_oval(self.goal_pos, radius, 'yellow', transform_radius=False)
            # if show_arrow:
            self.canvas.draw_arrow(self.init_pos, self.init_dir, 8 * radius, 'black', transform_length=False)
            if circle_scope:
                self.canvas.draw_oval(self.init_pos, self.max_detect_range, 'pink', background=True)
            else:
                self.canvas.draw_sector(self.init_pos, self.max_detect_range, self.init_dir - math.pi / 2, math.pi,
                                        'pink', background=True)
            if self.circle_obstacles is not None:
                for obstacle in self.circle_obstacles:
                    self.canvas.draw_oval(obstacle[:2], obstacle[2], 'deepskyblue')
            if self.line_obstacles is not None:
                for obstacle in self.line_obstacles:
                    self.canvas.draw_line(obstacle[:2], obstacle[2:], 'deepskyblue')

        # 绘制动态障碍物 - 使用新的绘制方法
        if self.dynamic_obstacles:
            self.draw_dynamic_obstacles()

        if show_trace:  # draw trace
            if self.traj.traj_type == self.traj.ARC:
                self.canvas.draw_arc(*self.traj.traj, 'black')
            else:
                self.canvas.draw_line(*self.traj.traj, 'black')
            if show_pos:
                self.canvas.draw_oval(self.current_pos, 3, 'green', transform_radius=False)
            if show_arrow:
                self.canvas.draw_arrow(self.current_pos, self.current_dir, 20, 'blue', transform_length=False)
            if show_scope:
                if circle_scope:
                    self.canvas.draw_oval(self.current_pos, self.max_detect_range, 'pink', background=True)
                else:
                    self.canvas.draw_sector(self.current_pos, self.max_detect_range, self.current_dir - math.pi / 2,
                                            math.pi, 'pink', background=True)
        self.canvas.update(sleep_time)

    def reset(self, scenairo):
        self.redraw = True

        self.set_obstacles(scenairo.circle_obstacles, scenairo.line_obstacles, scenairo.dynamic_obstacles)
        self.init(scenairo.init_pos, scenairo.init_dir, scenairo.goal_pos, scenairo.goal_dir)

        # 重置动态障碍物
        for obstacle in self.dynamic_obstacles:
            obstacle.reset()

        self.last_raw_goal_abs_dir = 0.
        self.compensate_goal_abs_dir = 0.

        self.delta_dis = None
        self.yaw_rate = 0.
        self.delta_yaw_rate = None
        self.delta_time = None
        self.delta_dir = None
        self.result = self.timeout
        self.residual_traj = None
        self._state()

        self.straight_line_dis = Scenairo.max_straight_line_dis
        self.last_goal_dis, self.last_goal_abs_dir = self.goal_dis, abs(self.goal_dir)
        return self.state

    def step(self, action):
        # 更新动态障碍物位置
        self.update_dynamic_obstacles()

        self.action(action)
        self.delta_dir, self.delta_dis = self.action.delta_dir, self.action.delta_dis
        self.transition(self.delta_dir, self.delta_dis)

        self.delta_time = self.delta_dis / self.action.linear_velocity

        last_yaw_rate = self.yaw_rate
        self.yaw_rate = self.delta_dir / self.delta_time
        self.delta_yaw_rate = self.yaw_rate - last_yaw_rate

        if not self.is_safe():  # check cross
            self.result = self.failure
            return self.terminal, -10., True

        self._state()
        if Geometry.dist_p2seg(self.goal_pos, self.last_pos, self.current_pos) <= \
                4. * self.action.basic_distance_step:  # check goal      点到线段的距离
            self.result = self.success
            self.residual_traj = self.current_pos, self.goal_pos
            return self.terminal, 10., True

        # granularity reward
        reward = 0.1 * self.delta_time / self.action.max_time_step

        # greed reward
        if self.goal_dis < self.last_goal_dis:  # distance decrease
            reward += 0.2
        else:  # distance increase
            reward += -0.2
        goal_abs_dir = abs(self.goal_dir)
        if goal_abs_dir < self.last_goal_abs_dir:  # direction decrease
            reward += 0.2
        else:  # direction increase
            reward += -0.2

        # smoothness reward
        abs_delta_yaw_rate = abs(self.delta_yaw_rate)  # 不能使用角度，而应该使用角速率
        reward -= abs_delta_yaw_rate / self.action.max_angular_velocity * 0.05  # [0, 2*yaw_rate] -> [0, -0.1]

        self.last_goal_dis, self.last_goal_abs_dir = self.goal_dis, goal_abs_dir
        return self.state, reward, False

    def is_safe(self):
        # 检查静态圆形障碍物
        current_circle_obstacles = self.circle_obstacles
        # 添加动态障碍物到检查列表
        dynamic_obs_data = self.get_current_dynamic_obstacles()
        if dynamic_obs_data is not None:
            if current_circle_obstacles is not None:
                current_circle_obstacles = np.concatenate((current_circle_obstacles, dynamic_obs_data))
            else:
                current_circle_obstacles = dynamic_obs_data

        if current_circle_obstacles is None:
            circle_safe = True
        else:
            if self.traj.traj_type == self.traj.ARC:  # 当前轨迹为弧
                for circle in current_circle_obstacles:
                    center, radius = circle[:2], circle[2]
                    dis = Geometry.dist_p2arc(center, *self.traj.traj)
                    if dis < radius:
                        return False
                circle_safe = True
            else:
                for circle in current_circle_obstacles:
                    center, radius = circle[:2], circle[2]
                    dis = Geometry.dist_p2seg(center, *self.traj.traj)
                    if dis < radius:
                        return False
                circle_safe = True

        if self.line_obstacles is None:
            line_safe = True
        else:
            ob0 = self.line_obstacles[:, 0]
            ob1 = self.line_obstacles[:, 1]
            ob2 = self.line_obstacles[:, 2]
            ob3 = self.line_obstacles[:, 3]
            x_ab, y_ab = ob0 - ob2, ob1 - ob3
            x_dc, y_dc = self.current_pos[0] - self.last_pos[0], self.current_pos[1] - self.last_pos[1]
            x_ac, y_ac = ob0 - self.last_pos[0], ob1 - self.last_pos[1]
            det = x_ab * y_dc - x_dc * y_ab

            mask = np.abs(det) > 1e-2
            Det = det[mask]
            X_ac = x_ac[mask]
            Y_ac = y_ac[mask]
            k1 = (y_dc * X_ac - x_dc * Y_ac) / Det
            k2 = (-y_ab[mask] * X_ac + x_ab[mask] * Y_ac) / Det
            tem1 = np.logical_and(0 <= k1, k1 <= 1)
            tem2 = np.logical_and(0 <= k2, k2 <= 1)
            line_safe = not np.any(np.logical_and(tem1, tem2))

        return line_safe and circle_safe

    def _state(self):
        all_dis = []

        # 静态圆形障碍物
        current_circle_obstacles = self.circle_obstacles
        # 添加动态障碍物
        dynamic_obs_data = self.get_current_dynamic_obstacles()
        if dynamic_obs_data is not None:
            if current_circle_obstacles is not None:
                current_circle_obstacles = np.concatenate((current_circle_obstacles, dynamic_obs_data))
            else:
                current_circle_obstacles = dynamic_obs_data

        if current_circle_obstacles is not None:
            all_dis.append(self.get_circle_ob_dis_with_obstacles(current_circle_obstacles))

        if self.line_obstacles is not None:
            all_dis.append(self.get_line_ob_dis())

        if all_dis:
            self.obs_distances = np.min(all_dis, axis=0)
        else:
            self.obs_distances = np.full(self.d_obstacles, self.no_obstacle_dis)

        if self.add_noise:
            self.obs_distances *= 1 + np.random.normal(loc=0, scale=0.05, size=(self.d_obstacles,))

        self.goal_dir, self.goal_dis = self.get_goal_dir(), self.get_goal_dis()

    @property
    def state(self):
        """
        obs_distances: [0, 1] & {2}
        goal_dis: [0, 1]
        goal_dir: [-1, 1]
        """
        return np.concatenate((self.obs_distances / self.max_detect_range,
                               [self.goal_dis / self.straight_line_dis, self.goal_dir / math.pi]))

    def get_goal_dis(self):
        return Geometry.dist(self.goal_pos - self.current_pos)

    def get_goal_dir(self):
        """取值范围: (-360°,360°)，初始取值范围: [-180°,180°]"""
        x, y = self.goal_pos - self.current_pos  # 看作目标相对于机体移动
        raw_goal_abs_dir = np.arctan2(y, x)
        # 检测地面坐标系下的目标方向跨越180°线情况并进行相应补偿
        diff_raw_goal_abs_dir = raw_goal_abs_dir - self.last_raw_goal_abs_dir
        if diff_raw_goal_abs_dir < -math.pi:  # 逆时针跨过180°
            self.compensate_goal_abs_dir += self.double_pi
        elif diff_raw_goal_abs_dir > math.pi:  # 顺时针跨国180°
            self.compensate_goal_abs_dir -= self.double_pi
        self.last_raw_goal_abs_dir = raw_goal_abs_dir
        real_goal_abs_dir = raw_goal_abs_dir + self.compensate_goal_abs_dir  # 补偿
        # 机体坐标系下的目标方向，目标在机体右侧为正
        goal_rel_dir = self.current_dir - real_goal_abs_dir

        # 根据取值范围进行折叠
        if goal_rel_dir > math.pi:
            self.compensate_goal_abs_dir += self.double_pi
            goal_rel_dir -= self.double_pi
        if goal_rel_dir <= -math.pi:
            self.compensate_goal_abs_dir -= self.double_pi
            goal_rel_dir += self.double_pi
        return goal_rel_dir

    def get_circle_ob_dis_with_obstacles(self, obstacles):
        """使用指定的障碍物列表计算距离"""
        distance = np.full((self.d_obstacles, len(obstacles)), self.no_obstacle_dis)
        angle = np.arange(self.max_detect_angle, -self.max_detect_angle - self.detect_angle_interval / 2,
                          -self.detect_angle_interval)
        rad_angle = self.current_dir + angle
        s_angle = np.sin(rad_angle)
        c_angle = np.cos(rad_angle)

        x1 = self.current_pos[0] - obstacles[:, 0]
        y1 = self.current_pos[1] - obstacles[:, 1]

        coe1 = c_angle[:, np.newaxis] * x1[np.newaxis, :] + s_angle[:, np.newaxis] * y1[np.newaxis, :]

        square_coe1 = np.square(coe1)
        coe2 = np.square(x1) + np.square(y1) - np.square(obstacles[:, 2])

        coe = square_coe1 - coe2  # 37x7
        mask = coe >= 0

        dis = -np.sqrt(coe[mask]) - coe1[mask]
        dis[np.logical_or(dis > self.max_detect_range, dis < 0)] = self.no_obstacle_dis
        distance[mask] = dis
        return np.min(distance, axis=1)

    def get_circle_ob_dis(self):
        """原始方法，保持向后兼容"""
        if self.circle_obstacles is None:
            return np.full(self.d_obstacles, self.no_obstacle_dis)
        return self.get_circle_ob_dis_with_obstacles(self.circle_obstacles)

    def get_line_ob_dis(self):
        """线障碍物距离检测"""
        distance = np.full((self.d_obstacles, len(self.line_obstacles)), self.no_obstacle_dis)
        angle = np.arange(self.max_detect_angle, -self.max_detect_angle - self.detect_angle_interval / 2,
                          -self.detect_angle_interval)
        rad_angle = self.current_dir + angle
        c_angle = np.cos(rad_angle)
        c_angle2 = c_angle[:, np.newaxis]
        s_angle = np.sin(rad_angle)
        s_angle2 = s_angle[:, np.newaxis]

        ob0 = self.line_obstacles[:, 0]
        ob1 = self.line_obstacles[:, 1]
        ob2 = self.line_obstacles[:, 2]
        ob3 = self.line_obstacles[:, 3]

        diff1 = ob2 - ob0
        diff1 = diff1[np.newaxis, :]
        diff2 = ob1 - ob3
        diff2 = diff2[np.newaxis, :]
        coe2 = diff1 * s_angle2 + diff2 * c_angle2
        diff1 = ob1 - self.current_pos[1]
        diff1 = diff1[np.newaxis, :]
        diff2 = self.current_pos[0] - ob0
        diff2 = diff2[np.newaxis, :]
        coe1 = diff1 * c_angle2 + diff2 * s_angle2

        mask1 = np.abs(coe2) > 1e-2
        rate_list = coe1[mask1] / coe2[mask1]
        mask2 = np.logical_and(0 <= rate_list, rate_list <= 1)
        mask1[mask1] = mask2  # new mask1

        rate_list = rate_list[mask2]  # new list

        indices = np.nonzero(mask1)
        indice_x = indices[0]
        indice_y = indices[1]

        coe1 = ob2[indice_y] * rate_list + (1 - rate_list) * ob0[indice_y] - self.current_pos[0]
        coe1 = coe1.astype(np.int32)
        coe2 = ob3[indice_y] * rate_list + (1 - rate_list) * ob1[indice_y] - self.current_pos[1]
        coe2 = coe2.astype(np.int32)

        tst1 = coe1 * c_angle[indice_x]
        tst2 = coe2 * s_angle[indice_x]

        mask3 = np.logical_and(tst1 >= 0, tst2 >= 0)
        dis = np.sqrt(coe1[mask3] ** 2 + coe2[mask3] ** 2)

        mask4 = dis < self.max_detect_range

        mask3[mask3] = mask4
        mask1[mask1] = mask3

        distance[mask1] = dis[mask4]

        return np.min(distance, axis=1)


def create_default_dynamic_obstacles():
    """创建3个默认的动态障碍物 - 使用随机方向"""
    obstacles = [
        # 障碍物1：完全随机方向
        DynamicObstacle(
            init_pos=[150, 500],
            radius=30,
            velocity=2.0,
            direction=None,  # 随机方向
            bounds=[50, 50, 650, 650],
            change_direction_probability=0.03  # 3%概率每步改变方向
        ),
        # 障碍物2：完全随机方向，变化更频繁
        DynamicObstacle(
            init_pos=[550, 350],
            radius=25,
            velocity=1.5,
            direction=None,  # 随机方向
            bounds=[50, 50, 650, 650],
            change_direction_probability=0.05  # 5%概率每步改变方向
        ),
        # 障碍物3：完全随机方向，较稳定
        DynamicObstacle(
            init_pos=[300, 100],
            radius=35,
            velocity=1.8,
            direction=None,  # 随机方向
            bounds=[50, 50, 650, 650],
            change_direction_probability=0.02  # 2%概率每步改变方向
        )
    ]
    return obstacles


def create_random_dynamic_obstacles(num_obstacles=5, region_bounds=[50, 50, 650, 650]):
    """创建指定数量的完全随机动态障碍物"""
    obstacles = []

    for i in range(num_obstacles):
        # 随机位置（避免太靠近边界）
        margin = 50
        pos_x = np.random.uniform(region_bounds[0] + margin, region_bounds[2] - margin)
        pos_y = np.random.uniform(region_bounds[1] + margin, region_bounds[3] - margin)

        # 随机属性
        radius = np.random.uniform(20, 40)  # 随机半径
        velocity = np.random.uniform(1.0, 3.0)  # 随机速度
        change_prob = np.random.uniform(0.01, 0.06)  # 随机方向变化概率

        obstacle = DynamicObstacle(
            init_pos=[pos_x, pos_y],
            radius=radius,
            velocity=velocity,
            direction=None,  # 随机方向
            bounds=region_bounds,
            change_direction_probability=change_prob
        )
        obstacles.append(obstacle)

    return obstacles


def create_chaotic_dynamic_obstacles():
    """创建混乱模式的动态障碍物 - 高度随机和不可预测"""
    obstacles = [
        # 快速随机变向的小障碍物
        DynamicObstacle(
            init_pos=[200, 200],
            radius=20,
            velocity=3.5,
            direction=None,
            bounds=[50, 50, 650, 650],
            change_direction_probability=0.08  # 8%概率改变方向
        ),
        # 中等速度的中型障碍物
        DynamicObstacle(
            init_pos=[400, 400],
            radius=30,
            velocity=2.2,
            direction=None,
            bounds=[50, 50, 650, 650],
            change_direction_probability=0.04
        ),
        # 慢速大型障碍物
        DynamicObstacle(
            init_pos=[500, 200],
            radius=45,
            velocity=1.2,
            direction=None,
            bounds=[50, 50, 650, 650],
            change_direction_probability=0.03
        ),
        # 超快速小障碍物
        DynamicObstacle(
            init_pos=[150, 550],
            radius=15,
            velocity=4.0,
            direction=None,
            bounds=[50, 50, 650, 650],
            change_direction_probability=0.10  # 10%概率改变方向
        ),
        # 中速随机障碍物
        DynamicObstacle(
            init_pos=[600, 300],
            radius=35,
            velocity=2.8,
            direction=None,
            bounds=[50, 50, 650, 650],
            change_direction_probability=0.06
        )
    ]
    return obstacles