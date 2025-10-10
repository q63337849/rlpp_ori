# ===================================
# Migrated from TensorFlow 1.x to 2.x
# Original file: D:\codetest\rlpp_ori\Supervisor\util.py
# Migration may not be complete. 
# Please review TODO comments.
# ===================================



class ModelInfo:
    """将字典转为属性"""
    def __init__(self, attr_dict):
        for name, value in attr_dict.items():
            setattr(self, name, value)
