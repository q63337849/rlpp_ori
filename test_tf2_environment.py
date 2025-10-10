#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试迁移后的TensorFlow 2.x环境
"""

import sys
import tensorflow as tf
import numpy as np

def test_basic():
    """基础功能测试"""
    print("Testing TensorFlow 2.x basic functionality...")
    
    # 测试eager execution
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    z = tf.matmul(x, y)
    
    print(f"✓ Eager execution works")
    print(f"  Result shape: {z.shape}")
    
    return True

def test_model():
    """模型测试"""
    print("\nTesting model creation and training...")
    
    # 创建简单模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    
    # 编译
    model.compile(optimizer='adam', loss='mse')
    
    # 训练
    x_train = np.random.random((100, 5))
    y_train = np.random.random((100, 1))
    
    history = model.fit(x_train, y_train, epochs=2, verbose=0)
    
    print(f"✓ Model training works")
    print(f"  Final loss: {history.history['loss'][-1]:.4f}")
    
    return True

def test_gradient_tape():
    """GradientTape测试"""
    print("\nTesting GradientTape...")
    
    x = tf.Variable(3.0)
    
    with tf.GradientTape() as tape:
        y = x ** 2
    
    dy_dx = tape.gradient(y, x)
    
    print(f"✓ GradientTape works")
    print(f"  d(x²)/dx at x=3: {dy_dx.numpy()}")
    
    return True

def test_gpu():
    """GPU测试"""
    print("\nTesting GPU availability...")
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"✓ Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("⚠ No GPU found, using CPU")
    
    return True

def main():
    print("=" * 60)
    print("TensorFlow 2.x Environment Test")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {sys.version}")
    print("=" * 60)
    
    tests = [
        test_basic,
        test_model,
        test_gradient_tape,
        test_gpu
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    if all(results):
        print("🎉 All tests passed!")
        print("Your TensorFlow 2.x environment is ready.")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return all(results)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
