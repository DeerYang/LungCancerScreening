import os
import sys
import json
import io
import tempfile
from datetime import datetime, timedelta
import traceback
import random

# 将项目根目录添加到系统路径，以正确导入自定义模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from PIL import Image, ImageDraw

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import dashscope
from dashscope import Generation

from segmentModel import UNetWrapper
from TumorModel import LunaModel
from TumorDatasets import CandidateInfoTuple
from util.util import patientCoord2voxelCoord, voxelCoord2patientCoord

# --- 1. 应用初始化与配置 ---
app = Flask(__name__)
CORS(app)

# 文件上传与数据库配置
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # 将上传路径添加到Flask配置中
# 请替换为您的MySQL连接信息: 'mysql+pymysql://<user>:<password>@<host>/<dbname>'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/lung_cancer_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)  # 初始化数据库扩展

# 通义千问 API Key 配置
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', 'YOUR_DASHSCOPE_API_KEY_HERE')
if 'YOUR_DASHSCOPE_API_KEY' in dashscope.api_key:
    print("警告：通义千问API密钥未设置。")

# 内存缓存，用于在单次会话中快速访问CT数据以进行可视化
ct_data_cache = {}


# --- 2. 数据库模型定义 ---
class Diagnosis(db.Model):
    """
    定义诊断记录的数据库表结构。
    """
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path_mhd = db.Column(db.String(512))
    file_path_raw = db.Column(db.String(512))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    diagnosis = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float)
    result_json = db.Column(db.Text, nullable=False)  # 存储完整的预测结果JSON

    def to_dict(self):
        """将模型对象转换为字典，以便于API返回。"""
        try:
            result_data = json.loads(self.result_json)
            # 确保使用数据库中的权威信息覆盖JSON中的内容
            result_data.update({
                'id': self.id,
                'diagnosis': self.diagnosis,
                'confidence': self.confidence
            })
            return result_data
        except (json.JSONDecodeError, TypeError):
            return {"id": self.id, "error": "无法解析存储的诊断结果"}


# --- 3. 核心业务逻辑封装 ---
class TumorPredictionSystem:
    """
    封装AI模型加载和三阶段预测流程。
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        self.seg_model, self.cls_model, self.mal_model = None, None, None
        self.models_loaded = False
        self.model_paths = {
            'segmentation': '../data-unversioned/seg/models/seg/seg_2025-07-02_13.09.34_none.best.state',
            'classification': '../data-unversioned/nodule/models/nodule-model/best_2025-07-02_10.36.28_nodule-comment.best.state',
            'malignancy': '../data-unversioned/tumor/models/tumor_cls/seg_2025-07-02_14.29.48_finetune-depth2.best.state'
        }
        self.load_models()

    def load_models(self):
        """加载所有AI模型并设置为评估模式。"""
        try:
            if os.path.exists(self.model_paths['segmentation']):
                seg_dict = torch.load(self.model_paths['segmentation'], map_location=self.device)
                self.seg_model = UNetWrapper(in_channels=7, n_classes=1, depth=3, wf=4, padding=True, batch_norm=True,
                                             up_mode='upconv')
                self.seg_model.load_state_dict(seg_dict['model_state'])
                self.seg_model.eval().to(self.device)
                print("✓ 分割模型加载成功")

            if os.path.exists(self.model_paths['classification']):
                cls_dict = torch.load(self.model_paths['classification'], map_location=self.device)
                self.cls_model = LunaModel()
                self.cls_model.load_state_dict(cls_dict['model_state'])
                self.cls_model.eval().to(self.device)
                print("✓ 结节分类模型加载成功")

            if os.path.exists(self.model_paths['malignancy']):
                mal_dict = torch.load(self.model_paths['malignancy'], map_location=self.device)
                self.mal_model = LunaModel()
                self.mal_model.load_state_dict(mal_dict['model_state'])
                self.mal_model.eval().to(self.device)
                print("✓ 恶性分类模型加载成功")

            if self.seg_model and self.cls_model:
                self.models_loaded = True
                print("所有必需模型已加载，系统进入真实预测模式。")
            else:
                self.models_loaded = False
                print("必需模型未完全加载，系统将使用模拟模式。")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.models_loaded = False

    def process_ct_files(self, mhd_data, raw_data, mhd_filename, raw_filename):
        """完整的CT文件处理和三阶段预测流程。"""
        if not self.models_loaded:
            return {"error": "模型未加载，无法进行预测"}

        print(f"开始真实模型预测: {mhd_filename}")
        try:
            series_uid = mhd_filename.replace('.mhd', '')
            with tempfile.TemporaryDirectory() as temp_dir:
                mhd_path = os.path.join(temp_dir, mhd_filename)
                raw_path = os.path.join(temp_dir, raw_filename)
                with open(mhd_path, 'wb') as f: f.write(mhd_data)
                with open(raw_path, 'wb') as f: f.write(raw_data)

                ct_mhd = sitk.ReadImage(mhd_path)
                ct_hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

                # 缓存CT数据，供图像查看API使用
                ct_data_cache[series_uid] = {
                    'hu_a': ct_hu_a,
                    'origin_xyz': ct_mhd.GetOrigin(),
                    'vxSize_xyz': ct_mhd.GetSpacing(),
                    'direction_a': np.array(ct_mhd.GetDirection()).reshape(3, 3)
                }

                # --- 三阶段AI预测 ---
                # 1. 分割
                mask_a = self._segment_ct(ct_hu_a)
                # 2. 提取候选区域
                candidate_info_list = self._group_segmentation_output(series_uid, ct_mhd, ct_hu_a, mask_a)
                print(f"分割后找到 {len(candidate_info_list)} 个候选区域。")
                # 3. 分类
                classifications_list = self._classify_candidates(ct_mhd, ct_hu_a,
                                                                 candidate_info_list) if candidate_info_list else []

            # --- 格式化结果 ---
            nodules = []
            for prob, prob_mal, center_xyz, _ in classifications_list:
                if prob > 0.5:
                    malignancy_level = 'N/A'
                    if prob_mal is not None:
                        if prob_mal < 0.3:
                            malignancy_level = 'low'
                        elif prob_mal < 0.7:
                            malignancy_level = 'medium'
                        else:
                            malignancy_level = 'high'

                    center_irc = patientCoord2voxelCoord(center_xyz, ct_mhd.GetOrigin(), ct_mhd.GetSpacing(),
                                                         np.array(ct_mhd.GetDirection()).reshape(3, 3))
                    nodules.append({
                        "id": len(nodules) + 1,
                        "nodule_probability": round(prob, 4),
                        "malignancy_probability": round(prob_mal, 4) if prob_mal is not None else None,
                        "malignancy_level": malignancy_level,
                        "center_xyz": [round(c, 4) for c in center_xyz],
                        "center_irc": {"index": center_irc.index, "row": center_irc.row, "col": center_irc.col},
                        "diameter_mm": 10.0,  # 示例值
                    })

            overall_finding, most_concerning_nodule = "no_nodules_found", None
            if nodules:
                most_concerning_nodule = max(nodules, key=lambda x: x['malignancy_probability'] or -1)
                top_level = most_concerning_nodule['malignancy_level']
                if top_level == 'high':
                    overall_finding = "high_risk"
                elif top_level == 'medium':
                    overall_finding = "moderate_risk"
                elif top_level == 'low':
                    overall_finding = "low_risk"
                else:
                    overall_finding = "nodules_present_malignancy_unavailable"

            prediction_result = {
                "filename": mhd_filename, "timestamp": datetime.now().isoformat(),
                "total_slices": ct_hu_a.shape[0], "voxel_spacing": list(ct_mhd.GetSpacing()),
                "summary": {
                    "overall_finding": overall_finding, "nodule_count": len(nodules),
                    "most_concerning_nodule": most_concerning_nodule},
                "nodules": nodules,
                "support_info": {
                    "total_candidates": len(candidate_info_list), "note": "本次结果基于真实模型直接推理",
                    "malignancy_analysis_available": self.mal_model is not None}
            }
            return prediction_result

        except Exception as e:
            traceback.print_exc()
            return {"error": f"处理失败: {str(e)}"}

    def _segment_ct(self, ct_hu_a):
        """使用UNet模型对CT图像进行逐切片分割。"""
        print("开始分割CT...")
        with torch.no_grad():
            output_a = np.zeros_like(ct_hu_a, dtype=np.float32)
            context_slices = 3
            for slice_ndx in range(ct_hu_a.shape[0]):
                ct_t = torch.zeros((context_slices * 2 + 1, 512, 512))
                start_ndx, end_ndx = slice_ndx - context_slices, slice_ndx + context_slices + 1
                for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
                    context_ndx = max(0, min(context_ndx, ct_hu_a.shape[0] - 1))
                    ct_t[i] = torch.from_numpy(ct_hu_a[context_ndx].astype(np.float32))

                ct_t.clamp_(-1000, 1000)
                prediction_g = self.seg_model(ct_t.unsqueeze(0).to(self.device))
                output_a[slice_ndx] = prediction_g[0].cpu().numpy()

            mask_a = output_a > 0.5
            mask_a = ndimage.binary_erosion(mask_a, iterations=1)
        print("分割完成。")
        return mask_a

    def _group_segmentation_output(self, series_uid, ct_mhd, ct_hu_a, clean_a):
        """对分割掩码进行连通域分析，提取候选结节。"""
        origin_xyz, vxSize_xyz = ct_mhd.GetOrigin(), ct_mhd.GetSpacing()
        direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)
        candidateLabel_a, candidate_count = ndimage.label(clean_a)
        if candidate_count == 0: return []

        centerIrc_list = ndimage.center_of_mass(
            ct_hu_a.clip(-1000, 1000) + 1001,
            labels=candidateLabel_a,
            index=np.arange(1, candidate_count + 1)
        )
        if not isinstance(centerIrc_list, list): centerIrc_list = [centerIrc_list]

        candidateInfo_list = []
        for center_irc in centerIrc_list:
            center_xyz = voxelCoord2patientCoord(center_irc, origin_xyz, vxSize_xyz, direction_a)
            if np.all(np.isfinite(center_irc)) and np.all(np.isfinite(center_xyz)):
                candidateInfo_list.append(CandidateInfoTuple(False, False, False, 0.0, series_uid, center_xyz))
        return candidateInfo_list

    def _get_ct_chunk(self, ct_hu_a, center_xyz, origin_xyz, vxSize_xyz, direction_a):
        """从完整CT中提取一个3D候选块。"""
        width_irc = (32, 48, 48)
        center_irc = patientCoord2voxelCoord(center_xyz, origin_xyz, vxSize_xyz, direction_a)
        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])
            if start_ndx < 0: start_ndx, end_ndx = 0, int(width_irc[axis])
            if end_ndx > ct_hu_a.shape[axis]: end_ndx, start_ndx = ct_hu_a.shape[axis], int(
                ct_hu_a.shape[axis] - width_irc[axis])
            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = ct_hu_a[tuple(slice_list)].copy()
        ct_chunk.clip(-1000, 1000, out=ct_chunk)
        return torch.from_numpy(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    def _classify_candidates(self, ct_mhd, ct_hu_a, candidateInfo_list):
        """对所有候选块进行结节分类和恶性程度预测。"""
        print("开始分类候选区域...")
        origin_xyz, vxSize_xyz = ct_mhd.GetOrigin(), ct_mhd.GetSpacing()
        direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)
        classifications_list = []
        with torch.no_grad():
            for candidate in candidateInfo_list:
                input_g = self._get_ct_chunk(ct_hu_a, candidate.center_xyz, origin_xyz, vxSize_xyz, direction_a).to(
                    self.device)

                _, prob_nodule_g = self.cls_model(input_g)
                prob_nodule = prob_nodule_g[0, 1].item()

                prob_mal = None
                if self.mal_model:
                    _, prob_mal_g = self.mal_model(input_g)
                    prob_mal = prob_mal_g[0, 1].item()

                classifications_list.append((prob_nodule, prob_mal, candidate.center_xyz, None))
        print("分类完成。")
        return classifications_list


# 实例化预测系统
prediction_system = TumorPredictionSystem()


# --- 4. Flask API 路由 ---

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口，返回系统和模型加载状态。"""
    return jsonify({
        "status": "healthy",
        "models_loaded": prediction_system.models_loaded,
        "device": str(prediction_system.device),
        "timestamp": datetime.now().isoformat(),
        "mode": "production" if prediction_system.models_loaded else "simulation"
    })


@app.route('/api/models/status', methods=['GET'])
def get_models_status():
    """获取详细的模型文件存在状态。"""
    return jsonify({
        "models_loaded": prediction_system.models_loaded,
        "device": str(prediction_system.device),
        "available_models": {
            "segmentation": os.path.exists(prediction_system.model_paths['segmentation']),
            "classification": os.path.exists(prediction_system.model_paths['classification']),
            "malignancy": os.path.exists(prediction_system.model_paths['malignancy'])
        }
    })


@app.route('/api/upload', methods=['POST'])
def upload_ct():
    """处理CT文件上传、分析和结果持久化。"""
    try:
        if 'mhd_file' not in request.files or 'raw_file' not in request.files:
            return jsonify({"error": "请同时上传 .mhd 和 .raw 文件"}), 400
        mhd_file, raw_file = request.files['mhd_file'], request.files['raw_file']
        if not mhd_file.filename.endswith('.mhd') or not raw_file.filename.endswith('.raw'):
            return jsonify({"error": "文件类型错误"}), 400
        if mhd_file.filename.replace('.mhd', '') != raw_file.filename.replace('.raw', ''):
            return jsonify({"error": "文件名不匹配"}), 400

        mhd_data, raw_data = mhd_file.read(), raw_file.read()
        result = prediction_system.process_ct_files(mhd_data, raw_data, mhd_file.filename, raw_file.filename)
        if "error" in result: return jsonify(result), 500

        # 持久化文件和诊断记录
        upload_subdir_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{raw_file.filename.replace('.raw', '')}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_subdir_name)
        os.makedirs(upload_path, exist_ok=True)
        mhd_save_path = os.path.join(upload_path, mhd_file.filename)
        raw_save_path = os.path.join(upload_path, raw_file.filename)
        with open(mhd_save_path, 'wb') as f:
            f.write(mhd_data)
        with open(raw_save_path, 'wb') as f:
            f.write(raw_data)

        summary = result.get('summary', {})
        new_diagnosis = Diagnosis(
            filename=mhd_file.filename,
            file_path_mhd=mhd_save_path,
            file_path_raw=raw_save_path,
            diagnosis=summary.get('overall_finding', 'unknown'),
            confidence=summary.get('most_concerning_nodule', {}).get('malignancy_probability'),
            result_json=json.dumps(result)
        )
        db.session.add(new_diagnosis)
        db.session.commit()

        result['success'] = True
        return jsonify(result)

    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": f"上传或处理失败: {str(e)}"}), 500


@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """从数据库获取所有历史诊断记录。"""
    try:
        all_diagnoses = Diagnosis.query.order_by(Diagnosis.id.desc()).all()
        return jsonify({
            "predictions": [d.to_dict() for d in all_diagnoses],
            "total_count": len(all_diagnoses)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"获取历史记录失败: {str(e)}"}), 500


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """从数据库聚合统计数据。"""
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return jsonify({
        "today_diagnoses": Diagnosis.query.filter(Diagnosis.timestamp >= today_start).count(),
        "benign_count": Diagnosis.query.filter_by(diagnosis='low_risk').filter(
            Diagnosis.timestamp >= today_start).count(),
        "malignant_count": Diagnosis.query.filter_by(diagnosis='high_risk').filter(
            Diagnosis.timestamp >= today_start).count(),
        "pending_count": random.randint(0, 5),  # 模拟
        "model_accuracies": {"segmentation": 98.9, "classification": 65.6, "malignancy": 67.0},  # 硬编码
        "system_performance": {"cpu_usage": random.randint(20, 70), "memory_usage": random.randint(40, 80),
                               "gpu_usage": random.randint(10, 50)},  # 模拟
        "timestamp": datetime.now().isoformat(),
    })


@app.route('/api/diagnosis-trend', methods=['GET'])
def get_diagnosis_trend():
    """从数据库聚合过去7天的诊断趋势。"""
    trend_data = []
    today = datetime.utcnow().date()
    for i in range(7):
        date = today - timedelta(days=6 - i)
        start_of_day = datetime(date.year, date.month, date.day)
        end_of_day = start_of_day + timedelta(days=1)
        count = Diagnosis.query.filter(Diagnosis.timestamp >= start_of_day, Diagnosis.timestamp < end_of_day).count()
        trend_data.append({"date": date.strftime("%m-%d"), "count": count})
    return jsonify({"trend": trend_data})


@app.route('/api/diagnosis-distribution', methods=['GET'])
def get_diagnosis_distribution():
    """从数据库聚合所有诊断结果的分布。"""
    dist = {
        "良性结节": Diagnosis.query.filter_by(diagnosis='low_risk').count(),
        "恶性结节": Diagnosis.query.filter_by(diagnosis='high_risk').count(),
        "需要进一步检查": Diagnosis.query.filter_by(diagnosis='moderate_risk').count(),
        "正常": Diagnosis.query.filter_by(diagnosis='no_nodules_found').count(),
    }
    return jsonify({"distribution": [{"name": k, "value": v} for k, v in dist.items() if v > 0]})


@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    """
    与集成的大语言模型进行交互。
    可以接收一个可选的 predictionId，以便在提问时提供上下文。
    """
    try:
        data = request.get_json()
        user_message = data.get('message')
        prediction_id = data.get('predictionId')  # 从前端接收可选的ID

        if not user_message:
            return jsonify({"error": "消息内容不能为空"}), 400

        # --- 核心修改：构建上下文和提示词 ---

        # 1. 基础系统提示词 (System Prompt)
        system_prompt = (
            "你是一个专业的AI医疗助手，专门帮助医生解读肺部CT扫描的AI分析报告。"
            "你的回答应该专业、严谨、简洁，并始终强调最终诊断需要由执业医师做出。"
            "请使用中文回答。"
        )

        # 2. 如果有 predictionId，查询数据库并构建上下文
        context_prompt = ""
        if prediction_id:
            diagnosis_record = Diagnosis.query.get(prediction_id)
            if diagnosis_record:
                # 将诊断结果格式化为一段清晰的文本
                report_data = diagnosis_record.to_dict()
                summary = report_data.get('summary', {})
                nodules = report_data.get('nodules', [])

                context_prompt = f"""
以下是关于文件 '{report_data.get('filename')}' (ID: {prediction_id}) 的AI分析报告摘要，请基于此信息回答问题：

[报告上下文]
- 总体发现: {summary.get('overall_finding', 'N/A')}
- 检测到的结节数量: {summary.get('nodule_count', 0)}
- 最可疑结节的恶性风险等级: {summary.get('most_concerning_nodule', {}).get('malignancy_level', 'N/A')}
- 最可疑结节的恶性概率: {summary.get('most_concerning_nodule', {}).get('malignancy_probability', 'N/A')}
- 结节列表详情:
"""
                if nodules:
                    for nodule in nodules:
                        context_prompt += f"  - 结节ID {nodule['id']}: 结节可能性 {nodule['nodule_probability']:.2%}，恶性风险 '{nodule['malignancy_level']}'，恶性概率 {nodule['malignancy_probability'] if nodule['malignancy_probability'] is not None else 'N/A'}\n"
                else:
                    context_prompt += "  - 未发现结节。\n"
                context_prompt += "[报告上下文结束]\n\n"

        # 3. 组合最终的提示词
        # 我们使用一个简单的结构，将上下文和用户问题都放在用户角色中
        final_user_prompt = f"{context_prompt}用户问题: {user_message}"

        print("--- 向大模型发送的最终用户提示词 ---")
        print(final_user_prompt)
        print("---------------------------------")

        # --- 调用通义千问 API ---
        response = Generation.call(
            model='qwen-turbo',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': final_user_prompt}
            ],
            result_format='message'  # 设置为 message 格式
        )

        if response.status_code == 200:
            ai_response = response.output.choices[0]['message']['content']
            return jsonify({"response": ai_response})
        else:
            error_msg = f"通义千问API错误: {response.code} - {response.message}"
            print(error_msg)
            return jsonify({"error": error_msg}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"聊天功能异常: {str(e)}"}), 500


@app.route('/api/ct-slice/<series_uid>/<int:slice_ndx>', methods=['GET'])
def get_ct_slice(series_uid, slice_ndx):
    """
    根据 series_uid 从数据库找到文件路径，
    从文件中加载CT数据，然后动态生成并返回带标注的CT切片图像。
    """
    try:
        # --- 1. 从数据库查找对应的诊断记录以获取文件路径 ---
        # 使用 series_uid (不含.mhd) 来查找，这里假设 filename 存储的是 xxx.mhd
        diagnosis_record = Diagnosis.query.filter(Diagnosis.filename == f"{series_uid}.mhd").order_by(
            Diagnosis.id.desc()).first()

        if not diagnosis_record or not diagnosis_record.file_path_mhd:
            return jsonify({"error": "No record or file path found in database for this series."}), 404

        mhd_path = diagnosis_record.file_path_mhd

        # 检查文件是否存在
        if not os.path.exists(mhd_path):
            return jsonify({"error": f"CT file not found at path: {mhd_path}"}), 404

        # --- 2. 从文件加载CT数据 ---
        ct_mhd = sitk.ReadImage(mhd_path)
        hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        vxSize_xyz = ct_mhd.GetSpacing()

        if not (0 <= slice_ndx < hu_a.shape[0]):
            return jsonify({"error": "Slice index out of bounds."}), 400

        # --- 3. 生成图像 ---
        # 肺窗处理，将HU值映射到人眼最敏感的0-255灰度范围
        window_level, window_width = -600, 1500
        min_val, max_val = window_level - window_width // 2, window_level + window_width // 2
        display_slice = np.clip(hu_a[slice_ndx], min_val, max_val)
        display_slice = ((display_slice - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        img = Image.fromarray(display_slice, 'L').convert('RGB')
        draw = ImageDraw.Draw(img)

        # 绘制边界框，前端在请求时，会通过URL参数将当前切片上所有结节的坐标和风险等级传来
        # 利用Pillow库，实时地在图像上画出对应颜色和大小的矩形框
        nodules_json = request.args.get('nodules')
        if nodules_json:
            nodules_on_slice = json.loads(nodules_json)
            _, vx_size_row, vx_size_col = vxSize_xyz
            color_map = {'high': 'red', 'medium': 'orange', 'low': 'lime'}
            for nodule in nodules_on_slice:
                center_r, center_c = nodule['center_irc']['row'], nodule['center_irc']['col']
                radius_r_px = (nodule.get('diameter_mm', 10.0) / 2) / vx_size_row
                radius_c_px = (nodule.get('diameter_mm', 10.0) / 2) / vx_size_col
                box_color = color_map.get(nodule.get('malignancy_level', 'low'), 'cyan')
                x0, y0, x1, y1 = center_c - radius_c_px, center_r - radius_r_px, center_c + radius_c_px, center_r + radius_r_px
                draw.rectangle([x0, y0, x1, y1], outline=box_color, width=2)
                draw.text((x0, y0 - 12), f"ID:{nodule['id']}", fill=box_color)

        img_io = io.BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        return img_io, 200, {'Content-Type': 'image/png'}

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to generate slice image: {str(e)}"}), 500


# --- 5. 应用启动 ---
if __name__ == '__main__':
    print("启动AI辅助肺肿瘤预测系统后端...")
    with app.app_context():
        # 确保在应用启动时，数据库表已创建
        db.create_all()
        print("数据库表检查/创建完成。")
    app.run(host='0.0.0.0', port=5000, debug=True)