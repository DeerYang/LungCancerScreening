# AI-Assisted Lung Cancer Screening System / AI辅助肺癌筛查系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Flask-2.2+-000000.svg)](https://flask.palletsprojects.com/)

An end-to-end open-source system for AI-assisted lung cancer screening. This project features a 3-stage deep learning pipeline (Segmentation, Classification, Malignancy Analysis) trained on the full **LUNA16 dataset**. It includes all training scripts, a Flask backend, and a React frontend with an interactive CT viewer.
<br>
一个完整的AI辅助肺癌筛查开源系统。本项目包含基于**LUNA16全数据集**训练的三阶段深度学习模型（分割、分类、恶性程度分析）。代码库提供全部训练脚本、基于Flask的后端服务、以及带有交互式CT查看器的React前端应用。

---

## 🌟 Features / 功能亮点

*   **End-to-End Pipeline**: A complete workflow from raw CT scans to diagnostic reports.
    <br>**端到端流程**: 从原始CT扫描到生成诊断报告的完整工作流。

*   **3-Stage AI Analysis**: A sophisticated pipeline for accurate diagnostics.
    <br>**三阶段AI分析**: 用于精准诊断的复杂流程。
    *   **Segmentation**: UNet model to identify nodule candidates on 2D CT slices.
        <br>**图像分割**: 使用UNet模型在2D切片上识别候选结节。
    *   **Classification**: 3D-CNN to confirm if candidates are true nodules.
        <br>**结节分类**: 使用3D-CNN确认候选区域是否为真实结节。
    *   **Malignancy Prediction**: A fine-tuned 3D-CNN to assess the malignancy risk of confirmed nodules.
        <br>**恶性预测**: 使用微调的3D-CNN评估已确认结节的恶性风险。

*   **Interactive CT Viewer**: An intuitive web-based viewer to scroll through CT slices and visualize AI-detected nodules with bounding boxes.
    <br>**交互式CT查看器**: 用户友好的网页查看器，可逐层浏览CT切片并可视化AI检测到的、带有边界框的结节。

*   **Full-Stack Application**: A robust Flask backend serves the AI models via a RESTful API, consumed by a modern React + Ant Design frontend.
    <br>**全栈应用**: 强大的Flask后端通过RESTful API提供AI模型服务，由现代化的React + Ant Design前端调用。

*   **Data Persistence**: Diagnostic results and uploaded CT files are saved persistently using a MySQL database.
    <br>**数据持久化**: 使用MySQL数据库持久化存储诊断结果和上传的CT文件。

*   **LLM-Powered Chatbot**: An integrated AI assistant to help doctors interpret diagnostic reports and answer questions.
    <br>**大模型智能助手**: 集成AI助手，帮助医生解读诊断报告并回答相关问题。

*   **Complete Training Scripts**: The full PyTorch training pipeline is provided, allowing for model reproduction and further research.
    <br>**完整的训练脚本**: 提供全部PyTorch训练代码，方便用户复现模型或进行二次研究。

## 🛠️ Tech Stack / 技术栈

*   **Backend**: Python, Flask, PyTorch, SQLAlchemy, PyMySQL, SimpleITK, NumPy, SciPy
*   **Frontend**: React, Ant Design, ECharts, Axios
*   **Database**: MySQL
*   **AI Models**: UNet, 3D-CNN
*   **LLM**: Qwen (via Dashscope API)

## 🚀 Getting Started / 快速开始

Follow these steps to set up and run the project on your local machine.
<br>
请按照以下步骤在本地计算机上设置并运行此项目。

### 1. Prerequisites / 环境要求

*   **Python**: 3.9 or higher.
*   **Node.js**: 16.x or higher.
*   **MySQL Server**: 5.7 or higher (8.x recommended).
*   **Git**

### 2. Installation / 安装步骤

**Step 1: Clone the repository / 克隆仓库**
```bash
git clone https://github.com/YourUsername/LungCancerScreening.git
cd LungCancerScreening
```
*(Please replace `YourUsername` with your GitHub username. / 请将 `YourUsername` 替换为你的GitHub用户名)*

**Step 2: Set up the Python backend / 配置后端**
```bash
# Create and activate a Python virtual environment
# 创建并激活Python虚拟环境
python -m venv .venv
source .venv/bin/activate  # on Windows, use `.venv\Scripts\activate`

# Install Python dependencies
# 安装Python依赖
pip install -r requirements.txt
```

**Step 3: Set up the MySQL database / 配置数据库**
1.  Log in to your MySQL server. / 登录到你的MySQL服务器。
2.  Create the database for this project. / 为项目创建数据库。
    ```sql
    CREATE DATABASE lung_cancer_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
    ```
3.  Open the backend configuration file `web_app/app.py`. / 打开后端配置文件 `web_app/app.py`。
4.  Find and update the database connection string with your MySQL credentials. / 找到并用你的MySQL凭据更新数据库连接字符串。
    ```python
    # web_app/app.py
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://YOUR_USER:YOUR_PASSWORD@localhost/lung_cancer_db'
    ```

**Step 4: Set up the React frontend / 配置前端**
```bash
# Navigate to the frontend directory
# 进入前端目录
cd web_app/frontend

# Install npm dependencies (this might take a few minutes)
# 安装npm依赖 (这可能需要几分钟)
npm install
```

**Step 5: Configure API Key / 配置API密钥 (Required for Chatbot / 聊天机器人功能必需)**

The integrated AI Chatbot is powered by Alibaba Cloud's Dashscope service (通义千问). To enable this feature, you must configure your API Key.
<br>
集成的AI智能助手由阿里云通义千问大模型驱动。要启用此功能，您需要配置您的API Key。

1.  First, obtain your API Key from the Alibaba Cloud Bailian Console: [https://bailian.console.aliyun.com/](https://bailian.console.aliyun.com/?tab=api#/api).
    <br>首先，请从阿里云百炼控制台获取您的API Key。

2.  Configure the key using one of the following methods. The SDK will search for the key in this order.
    <br>然后，使用以下任一方法进行配置。SDK将按以下顺序查找密钥。

    ---

    #### Method 1: Environment Variable (Recommended for Servers & CI/CD)
    #### 方法一：环境变量（推荐用于服务器和持续集成环境）
    
    Set an environment variable named `DASHSCOPE_API_KEY`. This is the most flexible method for production and development environments.
    <br>
    设置一个名为 `DASHSCOPE_API_KEY` 的环境变量。这是用于生产和开发环境最灵活的方法。

    *   **On macOS/Linux:**
        ```bash
        export DASHSCOPE_API_KEY="sk-your-actual-api-key"
        ```
        *(To make this permanent, add the line to your shell's startup file, e.g., `~/.bashrc` or `~/.zshrc`.)*
        <br>
        *(为使其永久生效，可将此行添加到您终端的启动文件中，如 `~/.bashrc` 或 `~/.zshrc`。)*

    *   **On Windows (PowerShell):**
        ```powershell
        $env:DASHSCOPE_API_KEY="sk-your-actual-api-key"
        ```
        *(To set it permanently, use the System Properties control panel.)*
        <br>
        *(如需永久设置，请使用系统属性控制面板。)*

    ---

    #### Method 2: Credentials File (Recommended for Local Development)
    #### 方法二：凭证文件（推荐用于本地开发）
    
    Create a configuration file in your user's home directory. This is useful for keeping all your keys in one secure place on your local machine.
    <br>
    在您的用户主目录下创建一个配置文件。此方法便于在本地开发机上统一管理密钥。

    1.  Create a file at the following path: `~/.dashscope/api_key`.
        <br>在以下路径创建文件：`~/.dashscope/api_key`。
        *(On Windows, `~` usually refers to `C:\Users\YourUsername`.)*
        <br>
        *(在Windows上, `~` 通常指 `C:\Users\您的用户名`。)*

    2.  Open the file and paste **only your API key** into it. The file content should be just the key itself.
        <br>
        打开文件，将您的API Key**单独**粘贴进去。文件内容应仅包含密钥本身。
        
        Example file content (`~/.dashscope/api_key`):
        <br>
        文件内容示例 (`~/.dashscope/api_key`):
        ```
        sk-your-actual-api-key
        ```

    ---

    #### Method 3: Direct Code Modification (Not Recommended)
    #### 方法三：直接修改代码（不推荐）
    
    You can directly edit the source code as a last resort. **Warning: This is insecure, as it risks exposing your key in version control.**
    <br>
    作为最后的手段，您可以直接修改源代码。**警告：此方法不安全，有将密钥泄露到版本控制系统中的风险。**

    *   Open `web_app/app.py`. / 打开 `web_app/app.py` 文件。
    *   Find the following line and replace the placeholder with your actual key:
        <br>找到下面这行代码，并将占位符替换为您的真实密钥：
        ```python
        # In web_app/app.py
        dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', 'sk-your-actual-api-key')
        ```

    ---
    *(If you wish to use a different language model, you will need to modify the `chat_with_ai` function in `web_app/app.py` to call your preferred model's API.)*
    <br>
    *(如果您希望使用其他大语言模型，您需要修改 `web_app/app.py` 中的 `chat_with_ai` 函数，以调用您偏好的模型API。)*

### 3. Running the Application / 运行项目

A convenience script is provided to start both the backend and frontend servers simultaneously.
<br>
我们提供了一个便捷脚本来同时启动后端和前端服务。

From the project **directory** (`LungCancerScreening/web_app`), run:
<br>
在项目的**目录** (`LungCancerScreening/web_app`) 下，运行：
```bash
python start_system.py
```

The system will be available at:
<br>
系统启动后，即可访问：
*   **Frontend URL**: [http://localhost:3000](http://localhost:3000)
*   **Backend API**: [http://localhost:5000](http://localhost:5000)

## 🧠 Training Your Own Models / 训练自己的模型

If you wish to train the models yourself, you can use the provided training scripts.
<br>
如果你希望自己训练模型，可以使用我们提供的训练脚本。

### 1. Dataset Setup / 数据集准备

1.  Download the LUNA16 dataset from the official website: [LUNA16 Grand Challenge](https://luna16.grand-challenge.org/Download/).
    <br>从官网下载LUNA16数据集。
2.  Place the `subset0` through `subset9` folders inside the `data-unversioned/data/` directory. You can use any number of subsets based on your needs.
    <br>将 `subset0` 到 `subset9` 文件夹（可根据需求放置任意数量的子集）放入 `data-unversioned/data/` 目录。
3.  Place `annotations.csv` and `candidates.csv` (also from the LUNA16 challenge) into the `data/` directory.
    <br>将 `annotations.csv` 和 `candidates.csv` 文件放入 `data/` 目录。

Your data directory structure should look like this:
<br>
你的数据目录结构应如下所示：
```
LungCancerScreening/
├── data/
│   ├── annotations.csv
│   ├── annotations_with_malignancy.csv
│   └── candidates.csv
└── data-unversioned/
    └── data/
        ├── subset0/
        ├── subset1/
        └── ...
```

### 2. Training Workflow / 训练流程

For each model (segmentation, classification, malignancy), there is a two-step process: **caching** and **training**.
<br>
对每个模型（分割、分类、恶性预测），都遵循两步流程：**缓存数据**和**开始训练**。

1.  **Segmentation Model / 分割模型**:
    ```bash
    # Step 1: Cache the data for segmentation
    python prep_seg_cache.py

    # Step 2: Start training
    python segmentTraining.py --epochs 10 --batch-size 16
    ```
2.  **Classification Model / 结节分类模型**:
    ```bash
    # Step 1: Cache the data for classification
    python prepcache.py

    # Step 2: Start training
    python training.py --epochs 5 --batch-size 8 --balanced
    ```
3.  **Malignancy Model / 恶性预测模型**:
    ```bash
    # Step 1: Cache the data for malignancy analysis
    python TumorPrepCache.py

    # Step 2: Start training (fine-tuning from the classification model)
    python TumorTraining.py --epochs 10 --batch-size 8 --finetune "path/to/your/best_classification_model.state"
    ```
    *(Note: Adjust the training parameters like epochs, batch size, etc., as needed.)*
    <br>
    *(注意：请根据需要调整训练参数，如轮次、批大小等。)*

### 3. Model Evaluation / 模型评估

After training, you can evaluate the performance of your entire pipeline using `model_eval.py`. This script runs the full 3-stage analysis on the validation set and generates a confusion matrix to show the results.
<br>
训练完成后，您可以使用 `model_eval.py` 来评估整个流程的性能。该脚本会在验证集上运行完整的三阶段分析，并生成混淆矩阵来展示结果。

**Running the evaluation / 运行评估脚本:**
```bash
python model_eval.py --run-validation \
    --segmentation-path "path/to/your/best_segmentation_model.state" \
    --classification-path "path/to/your/best_classification_model.state" \
    --malignancy-path "path/to/your/best_malignancy_model.state"
```
Ensure you provide the correct paths to your trained models. If you are using the pre-trained models provided in this repository, you can use the default paths.
<br>请确保您提供了训练好的模型的正确路径。如果您使用本仓库提供的预训练模型，则可以直接使用默认路径。
The script will output a confusion matrix for each CT series in the validation set, followed by a total confusion matrix for the entire set.
<br>脚本会为验证集中的每个CT序列输出一个混淆矩阵，最后会输出一个总的混淆矩阵。
You can also evaluate a single CT scan by providing its series_uid:
<br>您也可以通过提供 series_uid 来评估单个CT扫描：
```bash
python model_eval.py a.b.c.123  # Replace with a valid series_uid
```

## 🤝 Contributing / 贡献

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/YourUsername/LungCancerScreening/issues).
<br>
欢迎各种贡献、问题和功能请求！请随时查看[问题页面](https://github.com/YourUsername/LungCancerScreening/issues)。

## 📜 License / 许可证

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
<br>
本项目采用MIT许可证。详情请参阅[LICENSE](LICENSE)文件。
