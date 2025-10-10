"""
Model工具函数 - TensorFlow 2.x版本
重写@model和@shared装饰器以适配TF 2.x
"""
import tensorflow as tf


def model(cls):
    """
    Model装饰器 - TF 2.x版本
    用于标记模型类，在TF 2.x中主要用于保持接口兼容性
    """

    class ModelWrapper(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 在TF 2.x中，trainable_variables自动管理
            # 为了兼容旧代码，添加vars属性

        @property
        def vars(self):
            """兼容TF 1.x的vars属性"""
            if isinstance(self, tf.keras.Model):
                return self.trainable_variables
            else:
                # 如果不是Keras模型，收集所有层的变量
                variables = []
                for attr_name in dir(self):
                    attr = getattr(self, attr_name)
                    if isinstance(attr, tf.keras.layers.Layer):
                        variables.extend(attr.trainable_variables)
                    elif isinstance(attr, tf.Variable):
                        variables.append(attr)
                return variables

    # 保持原类名
    ModelWrapper.__name__ = cls.__name__
    ModelWrapper.__qualname__ = cls.__qualname__

    return ModelWrapper


def shared(cls):
    """
    Shared装饰器 - TF 2.x版本
    实现模型共享，使用单例模式
    """
    _instance = {}

    class SharedWrapper:
        def __new__(wrapper_cls, eval_model: bool, *args, **kwargs):
            """使用eval_model作为键区分eval和target模型"""
            if eval_model not in _instance:
                # 创建新实例
                instance = cls(eval_model, *args, **kwargs)
                _instance[eval_model] = instance
            return _instance[eval_model]

        @classmethod
        def clear_instances(wrapper_cls):
            """清除所有缓存的实例"""
            _instance.clear()

    # 保持原类名
    SharedWrapper.__name__ = cls.__name__
    SharedWrapper.__qualname__ = cls.__qualname__

    return SharedWrapper