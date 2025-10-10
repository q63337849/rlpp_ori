# ===================================
# Migrated from TensorFlow 1.x to 2.x
# Original file: D:\codetest\rlpp_ori\options.py
# Migration may not be complete. 
# Please review TODO comments.
# ===================================

"""
全局参数选项
"""

# 算法
dqn, dqn2, ddqn, ddpg, ac = 'DQN', 'DQN2', 'DDQN', 'DDPG', 'AC'   # 算法名
# 模型
M_CNN, M_CNN2, M_CNN3, M_MLP, M_SHARED_CNN, M_SHARED_MLP = 'cnn', 'cnn2', 'cnn3', 'mlp', 'shared_cnn', 'shared_mlp'   # model_id
# 环境
flight = 'flight'
# 障碍
O_circle, O_line = 'C', 'L'


