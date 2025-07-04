# AI辅助肺肿瘤预测系统

一个基于深度学习的肺肿瘤智能诊断系统，集成了三个AI模型和通义千问大模型，为医生提供专业的CT图像分析和诊断建议。

## 系统架构

### 三个核心AI模型

1. **分割模型 (Segmentation Model)**
   - 功能：对CT图像进行语义分割，识别可能的结节区域
   - 技术：基于UNet的深度学习模型
   - 输出：候选结节位置和分割置信度

2. **结节分类模型 (Nodule Classification Model)**
   - 功能：判断候选区域是否为真正的结节
   - 技术：基于3D CNN的LunaModel
   - 输出：结节概率和分类结果

3. **恶性程度分类模型 (Malignancy Classification Model)**
   - 功能：评估结节的恶性程度
   - 技术：基于3D CNN的LunaModel
   - 输出：恶性概率和最终诊断

### 系统模块

1. **医生用户界面**
   - 现代化的Web界面
   - 直观的CT图像上传和分析流程
   - 实时结果展示和交互

2. **AI模型核心**
   - 集成三个深度学习模型
   - 自动化的CT图像处理流程
   - 高精度的肿瘤预测

3. **聊天机器人模块**
   - 基于通义千问大模型
   - 专业的医疗咨询助手
   - 实时交互式帮助

## 技术栈

### 后端
- **Flask**: Web框架
- **PyTorch**: 深度学习框架
- **SimpleITK**: 医学图像处理
- **通义千问API**: 大语言模型
- **Socket.IO**: 实时通信

### 前端
- **React**: 前端框架
- **Ant Design**: UI组件库
- **ECharts**: 数据可视化
- **Axios**: HTTP客户端

## 安装和运行

### 环境要求

- Python 3.8+
- Node.js 16+
- CUDA (可选，用于GPU加速)

### 1. 克隆项目

```bash
git clone <repository-url>
cd LungCancerScreening
```

### 2. 设置环境变量

```bash
# 设置通义千问API密钥
export DASHSCOPE_API_KEY=your_api_key_here
```

### 3. 安装后端依赖

```bash
cd web_app
pip install -r requirements.txt
```

> **注意**：requirements.txt文件已整合到web_app目录下，包含了原始项目的所有依赖和Web应用的新增依赖。

### 4. 安装前端依赖

```bash
cd frontend
npm install
```

### 5. 启动系统

#### 方式一：使用启动脚本（推荐）

```bash
# 启动后端
python start_backend.py

# 新开终端，启动前端
python start_frontend.py
```

#### 方式二：手动启动

```bash
# 启动后端
cd web_app
python app.py

# 启动前端
cd web_app/frontend
npm start
```

### 6. 访问系统

- 前端界面：http://localhost:3000
- 后端API：http://localhost:5000

## 使用说明

### 1. 仪表板
- 查看系统状态和模型加载情况
- 监控诊断统计和性能指标
- 查看历史数据趋势

### 2. CT图像上传
- 支持MHD、DICOM、NIfTI等医学影像格式
- 拖拽上传或点击选择文件
- 实时显示分析进度
- 查看详细诊断结果

### 3. AI助手
- 与通义千问AI进行实时对话
- 获取系统使用指导
- 咨询医疗相关问题
- 快速问题模板

### 4. 历史记录
- 查看所有分析历史
- 搜索和筛选记录
- 下载分析报告
- 统计分析数据

## 模型文件

系统需要以下预训练模型文件：

```
data-unversioned/
├── seg/models/seg/seg_2025-06-30_16.17.31_none.best.state
├── nodule/models/nodule-model/cls_2025-06-29_15.30.21_nodule-comment.best.state
└── tumor/models/tumor_cls/seg_2025-07-01_07.47.10_finetune-depth2.best.state
```

如果模型文件不存在，系统将以模拟模式运行。

## API接口

### 健康检查
```
GET /api/health
```

### 上传CT图像
```
POST /api/upload
Content-Type: multipart/form-data
```

### AI聊天
```
POST /api/chat
Content-Type: application/json
{
  "message": "用户消息"
}
```

### 获取历史记录
```
GET /api/predictions
```

## 开发说明

### 项目结构
```
web_app/
├── app.py                 # Flask主应用
├── requirements.txt       # Python依赖
├── start_backend.py      # 后端启动脚本
├── start_frontend.py     # 前端启动脚本
├── frontend/             # React前端
│   ├── src/
│   │   ├── components/   # 组件
│   │   ├── pages/        # 页面
│   │   └── App.js        # 主应用
│   └── package.json      # Node.js依赖
└── README.md             # 说明文档
```

### 自定义配置

1. **模型路径配置**
   在 `app.py` 中修改 `model_paths` 字典

2. **API配置**
   设置环境变量 `DASHSCOPE_API_KEY`

3. **端口配置**
   修改 `app.py` 中的端口设置

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径
   - 确认PyTorch版本兼容性

2. **API调用失败**
   - 检查网络连接
   - 确认API密钥设置

3. **前端无法启动**
   - 检查Node.js版本
   - 重新安装依赖

### 日志查看

后端日志会显示在控制台，包括：
- 模型加载状态
- API调用记录
- 错误信息

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题，请联系开发团队。 