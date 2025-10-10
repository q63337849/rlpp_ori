# ===================================
# Migrated from TensorFlow 1.x to 2.x
# Original file: D:\codetest\rlpp_ori\Run\scene_adapter.py
# Migration may not be complete. 
# Please review TODO comments.
# ===================================

"""
Scene模块适配器 - 为动态障碍物功能提供兼容性
由于原始Scene模块可能不支持dynamic_obstacles参数，这个适配器提供了兼容性解决方案
"""

import numpy as np
import math
from Env.flight_with_dynamic_obstacles import DynamicObstacle


class Scenario:
    """扩展的场景类，支持动态障碍物"""

    def __init__(self, init_pos, init_dir, goal_pos, goal_dir,
                 circle_obstacles=None, line_obstacles=None, dynamic_obstacles=None):
        self.init_pos = init_pos
        self.init_dir = init_dir
        self.goal_pos = goal_pos
        self.goal_dir = goal_dir
        self.circle_obstacles = circle_obstacles
        self.line_obstacles = line_obstacles
        self.dynamic_obstacles = dynamic_obstacles if dynamic_obstacles is not None else []

    def __str__(self):
        return f"Scenario(init={self.init_pos}, goal={self.goal_pos}, dynamic_obs={len(self.dynamic_obstacles)})"


class ScenarioLoader:
    """场景加载器适配器"""

    def __init__(self):
        # 尝试导入原始Scene模块
        try:
            import Scene
            self.original_scene = Scene
            self.has_original = True
            print("检测到原始Scene模块，使用适配模式")
        except ImportError:
            self.original_scene = None
            self.has_original = False
            print("未检测到原始Scene模块，使用内置场景生成")

    def load_scene(self, **kwargs):
        """加载场景，兼容原始Scene模块"""
        if self.has_original and hasattr(self.original_scene, 'ScenarioLoader'):
            # 使用原始场景加载器
            original_loader = self.original_scene.ScenarioLoader()
            original_scenarios = original_loader.load_scene(**kwargs)

            # 转换为支持动态障碍物的格式
            enhanced_scenarios = []
            for scenario in original_scenarios:
                enhanced_scenario = Scenario(
                    init_pos=scenario.init_pos,
                    init_dir=scenario.init_dir,
                    goal_pos=scenario.goal_pos,
                    goal_dir=scenario.goal_dir,
                    circle_obstacles=getattr(scenario, 'circle_obstacles', None),
                    line_obstacles=getattr(scenario, 'line_obstacles', None),
                    dynamic_obstacles=[]  # 原始场景没有动态障碍物
                )
                enhanced_scenarios.append(enhanced_scenario)
            return enhanced_scenarios
        else:
            # 使用内置场景生成
            return self._generate_builtin_scenarios(**kwargs)

    def _generate_builtin_scenarios(self, percentage=1.0, **kwargs):
        """生成内置场景"""
        print(f"生成内置场景，使用比例: {percentage}")

        # 基础训练场景
        base_scenarios = [
            # 简单场景
            Scenario([50, 50], 45, [650, 650], 0, [[300, 300, 50]]),
            Scenario([100, 100], 0, [600, 600], 90, [[200, 400, 40], [500, 200, 45]]),
            Scenario([50, 350], 0, [650, 350], 0, [[350, 350, 60]]),

            # 复杂场景
            Scenario([100, 500], -45, [600, 100], 180, [[250, 300, 35], [450, 400, 40], [350, 150, 30]]),
            Scenario([500, 50], 135, [100, 600], 45, [[300, 250, 50], [400, 450, 35]]),

            # 狭窄通道场景
            Scenario([50, 200], 0, [650, 500], 45, [[200, 200, 30], [400, 350, 40], [500, 150, 35]]),
            Scenario([300, 50], 90, [400, 650], 0, [[150, 300, 45], [550, 400, 30]]),

            # 复杂导航场景
            Scenario([100, 100], 30, [600, 550], -30,
                     [[200, 200, 40], [400, 300, 35], [500, 450, 30], [350, 150, 25]]),
            Scenario([550, 550], -120, [150, 150], 60,
                     [[300, 400, 45], [250, 250, 30], [450, 200, 35]]),
            Scenario([50, 400], 15, [600, 200], -45,
                     [[200, 350, 40], [400, 250, 35], [500, 400, 30]])
        ]

        # 根据百分比选择场景
        num_scenarios = max(1, int(len(base_scenarios) * percentage))
        selected_scenarios = base_scenarios[:num_scenarios]

        print(f"生成了 {len(selected_scenarios)} 个内置场景")
        return selected_scenarios


# 任务定义，兼容原始代码结构
circle_train_task4x200 = {
    'percentage': 1.0,
    'task_type': 'circle_train'
}

circle_test_task4x100 = {
    'percentage': 0.5,
    'task_type': 'circle_test'
}

line_train_task4x200 = {
    'percentage': 1.0,
    'task_type': 'line_train'
}

line_test_task4x100 = {
    'percentage': 0.5,
    'task_type': 'line_test'
}


def create_dynamic_obstacle_variants():
    """创建不同类型的动态障碍物组合"""
    variants = {}

    # 基础动态障碍物
    variants['basic'] = lambda: [
        DynamicObstacle([150, 500], 30, 2.0, math.radians(-45), [50, 50, 650, 650]),
        DynamicObstacle([550, 350], 25, 1.5, math.radians(180), [50, 50, 650, 650]),
        DynamicObstacle([300, 100], 35, 1.8, math.radians(90), [50, 50, 650, 650])
    ]

    # 快速动态障碍物
    variants['fast'] = lambda: [
        DynamicObstacle([200, 200], 25, 4.0, math.radians(45), [50, 50, 650, 650]),
        DynamicObstacle([400, 400], 30, 3.5, math.radians(135), [50, 50, 650, 650]),
        DynamicObstacle([500, 200], 20, 5.0, math.radians(-90), [50, 50, 650, 650])
    ]

    # 慢速大障碍物
    variants['slow_large'] = lambda: [
        DynamicObstacle([150, 300], 50, 1.0, math.radians(0), [50, 50, 650, 650]),
        DynamicObstacle([450, 500], 45, 0.8, math.radians(180), [50, 50, 650, 650]),
        DynamicObstacle([350, 150], 40, 1.2, math.radians(90), [50, 50, 650, 650])
    ]

    # 混合速度
    variants['mixed'] = lambda: [
        DynamicObstacle([100, 100], 20, 3.0, math.radians(45), [50, 50, 650, 650]),
        DynamicObstacle([300, 400], 40, 1.5, math.radians(180), [50, 50, 650, 650]),
        DynamicObstacle([550, 250], 25, 2.5, math.radians(-60), [50, 50, 650, 650]),
        DynamicObstacle([450, 550], 35, 1.8, math.radians(225), [50, 50, 650, 650]),
        DynamicObstacle([200, 500], 15, 4.0, math.radians(0), [50, 50, 650, 650])
    ]

    return variants


def enhance_scenarios_with_dynamic_obstacles(scenarios, dynamic_type='basic'):
    """为现有场景添加动态障碍物"""
    variants = create_dynamic_obstacle_variants()
    dynamic_creator = variants.get(dynamic_type, variants['basic'])

    enhanced_scenarios = []
    for scenario in scenarios:
        # 创建新的场景，添加动态障碍物
        enhanced_scenario = Scenario(
            init_pos=scenario.init_pos,
            init_dir=scenario.init_dir,
            goal_pos=scenario.goal_pos,
            goal_dir=scenario.goal_dir,
            circle_obstacles=scenario.circle_obstacles,
            line_obstacles=getattr(scenario, 'line_obstacles', None),
            dynamic_obstacles=dynamic_creator()
        )
        enhanced_scenarios.append(enhanced_scenario)

    return enhanced_scenarios


# 使用示例和测试
if __name__ == "__main__":
    print("测试场景适配器...")

    # 测试场景加载
    loader = ScenarioLoader()
    scenarios = loader.load_scene(percentage=0.5)

    print(f"加载了 {len(scenarios)} 个场景")
    for i, scenario in enumerate(scenarios[:3]):  # 只显示前3个
        print(f"场景 {i + 1}: {scenario}")

    # 测试动态障碍物增强
    enhanced_scenarios = enhance_scenarios_with_dynamic_obstacles(scenarios, 'basic')
    print(f"\n增强后场景数量: {len(enhanced_scenarios)}")

    for i, scenario in enumerate(enhanced_scenarios[:3]):
        print(f"增强场景 {i + 1}: 动态障碍物数量 = {len(scenario.dynamic_obstacles)}")