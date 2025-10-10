# ===================================
# Migrated from TensorFlow 1.x to 2.x
# Original file: D:\codetest\rlpp_ori\Algo\__init__.py
# Migration may not be complete.
# Please review TODO comments.
# ===================================

"""

"""
import Supervisor as sp
import options
from .agent import Agent
from .supervisor import AlgoDispatch



class AlgoDispatch:
    """算法分派
    首先选取algorithm/supervisor/model的类型，其次配置algorithm所需参数，然后配置supervisor所需参数，最后配置model所需参数
    信息提供的顺序：basic info => algo info => supervisor info => model info
    """
    import_algo_code = 'from .{0} import {0}'
    supervisor_map = {
        options.dqn: sp.S_VALUE,
        options.dqn2: sp.S_VALUE,
        options.ddqn: sp.S_VALUE,
        options.ddpg: sp.S_ACTOR_CRITIC,
        options.ac: sp.S_ACTION_VALUE
    }

    def __init__(self, algo_name, model_name):
        """basic info"""
        self.algo_name = algo_name
        self.model_name = model_name
        self.algo = self.import_algo()
        self.algo_kwargs = {}  # ← 添加：初始化
        self.supervisor_kwargs = {}  # ← 添加：初始化
        self.sv_dispatch = sp.SupervisorDispatch(
            self.supervisor_map[algo_name],
            model_name,
            self.build_algo
        )

    def __call__(self, **algo_kwargs):
        """algo info"""
        self.algo_kwargs = algo_kwargs
        return self.sv_dispatch

    def build_algo(self, supervisor):
        """callback for SupervisorDispatch"""
        # ← 修复：先检查类属性或创建临时实例来判断
        try:
            # 方法1：检查类属性
            is_off_policy = getattr(self.algo, 'off_policy', False)
        except:
            # 方法2：创建临时实例检查
            try:
                temp_instance = self.algo(supervisor, **self.algo_kwargs)
                is_off_policy = temp_instance.off_policy
            except:
                is_off_policy = False

        if is_off_policy:
            from .util import ReplayMemory
            buffer = ReplayMemory(
                buffer_size=self.algo_kwargs.get('buffer_size', 10000),
                action_dt=getattr(self.algo, 'action_type', None)
            )
            return self.algo(supervisor, buffer, **self.algo_kwargs)
        else:
            return self.algo(supervisor, **self.algo_kwargs)

    def import_algo(self):
        exec(self.import_algo_code.format(self.algo_name))
        return locals()[self.algo_name]