# ===================================
# Migrated from TensorFlow 1.x to 2.x
# Original file: D:\codetest\rlpp_ori\start_tensorboard.py
# Migration may not be complete. 
# Please review TODO comments.
# ===================================

"""
TensorBoard启动和监控辅助脚本
自动查找最新的日志目录并启动TensorBoard
"""

import os
import subprocess
import sys
import glob
from datetime import datetime


def find_latest_log_dir():
    """查找最新的日志目录"""
    log_base = './logs'

    if not os.path.exists(log_base):
        print(f"日志目录不存在: {log_base}")
        return None

    # 查找所有lstm_rl_开头的目录
    log_dirs = glob.glob(os.path.join(log_base, 'lstm_rl_*'))

    if not log_dirs:
        print("未找到训练日志")
        return None

    # 按修改时间排序，返回最新的
    latest_dir = max(log_dirs, key=os.path.getmtime)
    return latest_dir


def list_available_logs():
    """列出所有可用的日志"""
    log_base = './logs'

    if not os.path.exists(log_base):
        print("日志目录不存在")
        return []

    log_dirs = glob.glob(os.path.join(log_base, 'lstm_rl_*'))

    if not log_dirs:
        print("未找到训练日志")
        return []

    print("\n可用的训练日志:")
    print("=" * 60)

    for i, log_dir in enumerate(sorted(log_dirs, key=os.path.getmtime, reverse=True)):
        mtime = os.path.getmtime(log_dir)
        time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

        # 计算目录大小
        total_size = 0
        for root, dirs, files in os.walk(log_dir):
            total_size += sum(os.path.getsize(os.path.join(root, f)) for f in files)
        size_mb = total_size / (1024 * 1024)

        print(f"{i + 1}. {os.path.basename(log_dir)}")
        print(f"   时间: {time_str}")
        print(f"   大小: {size_mb:.2f} MB")
        print()

    return log_dirs


def start_tensorboard(log_dir, port=6006):
    """启动TensorBoard"""
    print(f"\n{'=' * 60}")
    print(f"启动TensorBoard")
    print(f"{'=' * 60}")
    print(f"日志目录: {log_dir}")
    print(f"端口: {port}")
    print(f"{'=' * 60}")
    print(f"\n📊 TensorBoard启动中...")
    print(f"🌐 访问地址: http://localhost:{port}")
    print(f"⚠️  按 Ctrl+C 停止TensorBoard")
    print(f"{'=' * 60}\n")

    try:
        # 启动TensorBoard
        subprocess.run([
            'tensorboard',
            '--logdir', log_dir,
            '--port', str(port),
            '--bind_all'
        ])
    except KeyboardInterrupt:
        print("\n\n✓ TensorBoard已停止")
    except FileNotFoundError:
        print("\n❌ 错误: tensorboard未安装")
        print("请运行: pip install tensorboard")


def compare_runs(log_dirs):
    """对比多个训练运行"""
    if not log_dirs:
        print("没有可对比的日志")
        return

    print(f"\n{'=' * 60}")
    print(f"对比多个训练运行")
    print(f"{'=' * 60}")

    # 使用所有日志目录的父目录
    log_base = os.path.dirname(log_dirs[0])

    print(f"日志根目录: {log_base}")
    print(f"包含 {len(log_dirs)} 个训练运行")
    print(f"{'=' * 60}\n")

    start_tensorboard(log_base, port=6006)


def monitor_training():
    """实时监控训练"""
    print("\n" + "=" * 60)
    print("训练监控助手")
    print("=" * 60)

    # 列出所有日志
    log_dirs = list_available_logs()

    if not log_dirs:
        print("\n没有找到训练日志。")
        print("请先运行训练: python train_lstm_rl.py")
        return

    print("选择操作:")
    print("1. 监控最新训练")
    print("2. 对比所有训练")
    print("3. 选择特定训练")
    print("0. 退出")

    try:
        choice = input("\n请选择 (0-3): ").strip()

        if choice == '1':
            # 监控最新训练
            latest_dir = find_latest_log_dir()
            if latest_dir:
                start_tensorboard(latest_dir)

        elif choice == '2':
            # 对比所有训练
            compare_runs(log_dirs)

        elif choice == '3':
            # 选择特定训练
            print("\n选择要监控的训练 (输入编号):")
            idx = int(input("编号: ").strip()) - 1

            if 0 <= idx < len(log_dirs):
                sorted_dirs = sorted(log_dirs, key=os.path.getmtime, reverse=True)
                start_tensorboard(sorted_dirs[idx])
            else:
                print("无效的编号")

        elif choice == '0':
            print("退出")

        else:
            print("无效的选择")

    except KeyboardInterrupt:
        print("\n\n退出监控")
    except Exception as e:
        print(f"\n错误: {e}")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        if sys.argv[1] == 'latest':
            # 直接启动最新的
            latest_dir = find_latest_log_dir()
            if latest_dir:
                start_tensorboard(latest_dir)

        elif sys.argv[1] == 'compare':
            # 对比所有
            log_dirs = glob.glob('./logs/lstm_rl_*')
            compare_runs(log_dirs)

        elif sys.argv[1] == 'list':
            # 列出所有
            list_available_logs()

        else:
            print("用法:")
            print("  python start_tensorboard.py          # 交互式选择")
            print("  python start_tensorboard.py latest   # 监控最新训练")
            print("  python start_tensorboard.py compare  # 对比所有训练")
            print("  python start_tensorboard.py list     # 列出所有训练")
    else:
        # 交互式模式
        monitor_training()


if __name__ == "__main__":
    main()