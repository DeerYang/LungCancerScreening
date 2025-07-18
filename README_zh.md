<div align="right">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</div>

# AI辅助肺癌筛查系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Flask-2.2+-000000.svg)](https://flask.palletsprojects.com/)

一个完整的AI辅助肺癌筛查开源系统。本项目包含基于**LUNA16全数据集**训练的三阶段深度学习模型（分割、分类、恶性程度分析）。代码库提供全部训练脚本、基于Flask的后端服务、以及带有交互式CT查看器的React前端应用。

---

## 🌟 功能亮点

*   **端到端流程**: 从原始CT扫描到生成诊断报告的完整工作流。
*   **三阶段AI分析**: 用于精准诊断的复杂流程。
    *   **图像分割**: 使用UNet模型在2D切片上识别候选结节。
    *   **结节分类**: 使用3D-CNN确认候选区域是否为真实结节。
    *   **恶性预测**: 使用微调的3D-CNN评估已确认结节的恶性风险。
*   **交互式CT查看器**: 用户友好的网页查看器，可逐层浏览CT切片并可视化AI检测到的、带有边界框的结节。
*   **全栈应用**: 强大的Flask后端通过RESTful API提供AI模型服务，由现代化的React + Ant Design前端调用。
*   **数据持久化**: 使用MySQL数据库持久化存储诊断结果和上传的CT文件。
*   **大模型智能助手**: 集成AI助手，帮助医生解读诊断报告并回答相关问题。
*   **完整的训练脚本**: 提供全部PyTorch训练代码，方便用户复现模型或进行二次研究。

## 🛠️ 技术栈

*   **后端**: Python, Flask, PyTorch, SQLAlchemy, PyMySQL, SimpleITK, NumPy, SciPy
*   **前端**: React, Ant Design, ECharts, Axios
*   **数据库**: MySQL
*   **AI模型**: UNet, 3D-CNN
*   **大语言模型**: 通义千问 (通过 Dashscope API)

## 🚀 快速开始

请按照以下步骤在本地计算机上设置并运行此项目。

### 1. 环境要求

*   **Python**: 3.9 或更高版本。
*   **Node.js**: 16.x 或更高版本。
*   **MySQL Server**: 5.7 或更高版本 (推荐 8.x)。
*   **Git**

### 2. 安装步骤

**步骤 1: 克隆仓库**
```bash
git clone https://github.com/YourUsername/LungCancerScreening.git
cd LungCancerScreening
```
*(请将 `YourUsername` 替换为您的GitHub用户名)*

**步骤 2: 配置后端**
```bash
# 创建并激活Python虚拟环境
python -m venv .venv
# Windows 用户请运行:
.venv\Scripts\activate
# macOS/Linux 用户请运行:
source .venv/bin/activate

# 安装Python依赖
pip install -r requirements.txt
```

**步骤 3: 配置数据库**
1.  登录到您的MySQL服务器。
2.  为项目创建数据库。
    ```sql
    CREATE DATABASE lung_cancer_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
    ```
3.  打开后端配置文件 `web_app/app.py`。
4.  找到并用您的MySQL凭据更新数据库连接字符串。
    ```python
    # 在 web_app/app.py 文件中
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://您的用户名:您的密码@localhost/lung_cancer_db'
    ```

**步骤 4: 配置前端**
```bash
# 进入前端目录
cd web_app/frontend

# 安装npm依赖 (此过程可能需要几分钟)
npm install
```

**步骤 5: 配置API密钥 (聊天机器人功能必需)**

集成的AI智能助手由阿里云通义千问大模型驱动。要启用此功能，您需要配置您的API Key。

1.  首先，请从阿里云百炼控制台获取您的API Key: [https://bailian.console.aliyun.com/](https://bailian.console.aliyun.com/?tab=api#/api)。

2.  然后，使用以下任一方法进行配置（SDK将按此顺序查找密钥）。

    ---

    #### 方法一：环境变量（推荐）
    设置一个名为 `DASHSCOPE_API_KEY` 的环境变量，其值为您的密钥。
    *   **在 Windows (PowerShell) 中:**
        ```powershell
        $env:DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"
        ```
    *   **在 macOS/Linux 中:**
        ```bash
        export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"
        ```
        *(为使其永久生效，可将此行添加到您终端的启动文件中，如 `~/.bashrc` 或 `~/.zshrc`。)*

    ---
    
    #### 方法二：凭证文件
    在您的用户主目录下创建一个配置文件 `~/.dashscope/api_key`，并将您的API Key单独粘贴进去。
    *(在Windows上, `~` 通常指 `C:\Users\您的用户名`。)*

    ---

    #### 方法三：直接修改代码（不推荐）
    作为最后的手段，您可以直接修改 `web_app/app.py` 文件中的代码行 `dashscope.api_key = ...`。**警告：此方法不安全，有将密钥泄露到版本控制系统中的风险。**


### 3. 运行项目

我们提供了一个便捷脚本来同时启动后端和前端服务。

在项目的**根目录** (`LungCancerScreening/`) 下，运行：
```bash
python start_system.py
```

系统启动后，即可访问：
*   **前端 URL**: [http://localhost:3000](http://localhost:3000)
*   **后端 API**: [http://localhost:5000](http://localhost:5000)

## 🧠 训练自己的模型

如果您希望自己训练模型，可以使用我们提供的训练脚本。

### 1. 数据集准备

1.  从官网下载LUNA16数据集: [LUNA16 Grand Challenge](https://luna16.grand-challenge.org/Download/)。
2.  将 `subset0` 到 `subset9` 文件夹放入 `data-unversioned/data/` 目录。
3.  将 `annotations.csv` 和 `candidates.csv` 文件放入 `data/` 目录。

您的数据目录结构应如下所示：
```
LungCancerScreening/
├── data/
│   ├── annotations.csv
│   └── candidates.csv
└── data-unversioned/
    └── data/
        ├── subset0/
        ├── subset1/
        └── ...
```

### 2. 训练流程

对每个模型（分割、分类、恶性预测），都遵循两步流程：**缓存数据**和**开始训练**。

1.  **分割模型**:
    ```bash
    # 第1步: 缓存分割任务所需的数据
    python prep_seg_cache.py
    # 第2步: 开始训练
    python segmentTraining.py --epochs 10 --batch-size 16
    ```
2.  **结节分类模型**:
    ```bash
    # 第1步: 缓存分类任务所需的数据
    python prepcache.py
    # 第2步: 开始训练
    python training.py --epochs 5 --batch-size 8 --balanced
    ```
3.  **恶性预测模型**:
    ```bash
    # 第1步: 缓存恶性分析任务所需的数据
    python TumorPrepCache.py
    # 第2步: 开始训练 (基于分类模型进行微调)
    python TumorTraining.py --epochs 10 --batch-size 8 --finetune "path/to/your/best_classification_model.state"
    ```
    *(注意：请根据需要调整训练参数，如轮次、批大小等。)*

### 3. 模型评估

训练完成后，您可以使用 `model_eval.py` 来评估整个流程的性能。
```bash
python model_eval.py --run-validation \
    --segmentation-path "path/to/your/best_segmentation_model.state" \
    --classification-path "path/to/your/best_classification_model.state" \
    --malignancy-path "path/to/your/best_malignancy_model.state"
```
您也可以通过提供 `series_uid` 来评估单个CT扫描：
```bash
python model_eval.py a.b.c.123  # 请替换为有效的 series_uid
```

## 🤝 贡献

欢迎各种贡献、问题和功能请求！请随时查看[问题页面](https://github.com/YourUsername/LungCancerScreening/issues)。

## 📜 许可证

本项目采用MIT许可证。详情请参阅[LICENSE](LICENSE)文件。
