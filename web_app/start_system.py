#!/usr/bin/env python3
"""
AI辅助肺肿瘤预测系统 - 一键启动脚本
"""

import os
import sys
import subprocess
import time
import signal

def check_node_installed():
    """检查Node.js是否安装"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Node.js版本: {result.stdout.strip()}")
            return True
        else:
            return False
    except FileNotFoundError:
        return False

def check_npm_installed():
    """检查npm是否安装"""
    # 尝试多个可能的npm路径
    npm_paths = [
        'npm',
        'npm.cmd',  # Windows
        os.path.join(os.environ.get('APPDATA', ''), 'npm', 'npm.cmd'),  # Windows npm
        os.path.join(os.environ.get('PROGRAMFILES', ''), 'nodejs', 'npm.cmd'),  # Windows nodejs
    ]
    
    for npm_path in npm_paths:
        try:
            result = subprocess.run([npm_path, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ npm版本: {result.stdout.strip()}")
                return True, npm_path
        except FileNotFoundError:
            continue
    
    # 如果直接路径找不到，尝试使用where命令查找
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run(['where', 'npm'], capture_output=True, text=True)
            if result.returncode == 0:
                npm_path = result.stdout.strip().split('\n')[0]
                result = subprocess.run([npm_path, '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✓ npm版本: {result.stdout.strip()}")
                    return True, npm_path
        else:  # Linux/Mac
            result = subprocess.run(['which', 'npm'], capture_output=True, text=True)
            if result.returncode == 0:
                npm_path = result.stdout.strip()
                result = subprocess.run([npm_path, '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✓ npm版本: {result.stdout.strip()}")
                    return True, npm_path
    except:
        pass
    
    return False, None

def install_frontend_dependencies(npm_path='npm'):
    """安装前端依赖"""
    print("正在安装前端依赖...")
    try:
        result = subprocess.run([npm_path, 'install'], cwd='frontend', capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 前端依赖安装成功")
            return True
        else:
            print(f"✗ 前端依赖安装失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ 安装失败: {e}")
        return False

def signal_handler(signum, frame):
    print("\n正在停止系统...")
    sys.exit(0)

def main():
    print("=" * 50)
    print("AI辅助肺肿瘤预测系统 - 一键启动")
    print("=" * 50)
    
    # 检查Node.js
    print("1. 检查Node.js...")
    if not check_node_installed():
        print("✗ Node.js未安装，请先安装Node.js")
        print("下载地址: https://nodejs.org/")
        print("安装完成后重新运行此脚本")
        sys.exit(1)
    
    # 检查npm
    print("2. 检查npm...")
    npm_installed, npm_path = check_npm_installed()
    if not npm_installed:
        print("✗ npm未安装或无法找到")
        print("请确保npm已正确安装并添加到系统PATH中")
        print("或者重新安装Node.js（通常包含npm）")
        sys.exit(1)
    
    # 检查前端目录
    if not os.path.exists('frontend'):
        print("✗ 前端目录不存在")
        sys.exit(1)
    
    # 检查package.json
    if not os.path.exists('frontend/package.json'):
        print("✗ package.json不存在")
        sys.exit(1)
    
    # 检查node_modules
    if not os.path.exists('frontend/node_modules'):
        print("3. 安装前端依赖...")
        if not install_frontend_dependencies(npm_path):
            sys.exit(1)
    else:
        print("✓ 前端依赖已存在")
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # 启动后端
    print("4. 启动后端服务...")
    try:
        backend = subprocess.Popen([sys.executable, 'app.py'])
        print("✓ 后端服务启动成功")
    except Exception as e:
        print(f"✗ 后端服务启动失败: {e}")
        sys.exit(1)
    
    time.sleep(3)
    
    # 启动前端
    print("5. 启动前端服务...")
    try:
        frontend = subprocess.Popen([npm_path, 'start'], cwd='frontend')
        print("✓ 前端服务启动成功")
    except Exception as e:
        print(f"✗ 前端服务启动失败: {e}")
        backend.terminate()
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("系统启动完成！")
    print("=" * 50)
    print("前端界面: http://localhost:3000")
    print("后端API:  http://localhost:5000")
    print("按 Ctrl+C 停止系统")
    print("=" * 50)
    
    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        backend.terminate()
        frontend.terminate()
        print("系统已停止")

if __name__ == '__main__':
    main() 