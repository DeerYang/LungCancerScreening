# app.py

import os
import sys
import json
import io
import tempfile
from datetime import datetime, timedelta
import traceback
import random

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage

from flask import Flask, request, jsonify
from flask_cors import CORS
import dashscope
from dashscope import Generation

from segmentModel import UNetWrapper
from TumorModel import LunaModel
from TumorDatasets import CandidateInfoTuple
from util.util import patientCoord2voxelCoord, voxelCoord2patientCoord

app = Flask(__name__)
CORS(app)

dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', 'YOUR_DASHSCOPE_API_KEY_HERE')
if 'YOUR_DASHSCOPE_API_KEY' in dashscope.api_key:
    print("警告：通义千问API密钥未设置。")


class TumorPredictionSystem:
    # ... (这个类的所有内容保持不变，从上一个回答复制即可)
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        self.seg_model, self.cls_model, self.mal_model = None, None, None
        self.models_loaded = False
        self.diagnosis_history = []
        self.model_paths = {
            'segmentation': '../data-unversioned/seg/models/seg/seg_2025-07-02_13.09.34_none.best.state',
            'classification': '../data-unversioned/nodule/models/nodule-model/best_2025-07-02_10.36.28_nodule-comment.best.state',
            'malignancy': '../data-unversioned/tumor/models/tumor_cls/seg_2025-07-02_14.29.48_finetune-depth2.best.state'
        }
        self.load_models()

    def load_models(self):
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
        if not self.models_loaded:
            return self.simulate_prediction(mhd_filename)

        print(f"开始真实模型预测: {mhd_filename}")
        try:
            series_uid = mhd_filename.replace('.mhd', '')
            with tempfile.TemporaryDirectory() as temp_dir:
                mhd_path = os.path.join(temp_dir, mhd_filename)
                raw_path = os.path.join(temp_dir, raw_filename)
                with open(mhd_path, 'wb') as f:
                    f.write(mhd_data)
                with open(raw_path, 'wb') as f:
                    f.write(raw_data)

                ct_mhd = sitk.ReadImage(mhd_path)
                ct_hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

                mask_a = self._segment_ct(ct_hu_a)
                candidate_info_list = self._group_segmentation_output(series_uid, ct_mhd, ct_hu_a, mask_a)
                total_candidates = len(candidate_info_list)
                print(f"分割后找到 {total_candidates} 个候选区域。")

                classifications_list = self._classify_candidates(ct_mhd, ct_hu_a,
                                                                 candidate_info_list) if candidate_info_list else []

                nodules = []
                for prob, prob_mal, center_xyz, _ in classifications_list:
                    if prob > 0.5:
                        mal_prob_display, malignancy_level = None, 'N/A'
                        if prob_mal is not None:
                            if prob_mal < 0.3:
                                malignancy_level = 'low'
                            elif prob_mal < 0.7:
                                malignancy_level = 'medium'
                            else:
                                malignancy_level = 'high'
                            mal_prob_display = round(prob_mal, 4)

                        nodules.append({
                            "id": len(nodules) + 1,
                            "nodule_probability": round(prob, 4),
                            "malignancy_probability": mal_prob_display,
                            "malignancy_level": malignancy_level,
                            "center_xyz": [round(c, 4) for c in center_xyz],
                        })

                overall_finding, most_concerning_nodule = "no_nodules_found", None
                if nodules:
                    most_concerning_nodule = max(nodules, key=lambda x: x['malignancy_probability'] if x[
                                                                                                           'malignancy_probability'] is not None else -1)
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
                    "filename": mhd_filename,
                    "timestamp": datetime.now().isoformat(),
                    "summary": {
                        "overall_finding": overall_finding,
                        "nodule_count": len(nodules),
                        "most_concerning_nodule": most_concerning_nodule
                    },
                    "nodules": nodules,
                    "support_info": {
                        "total_candidates": total_candidates,
                        "note": "本次结果基于真实模型直接推理",
                        "malignancy_analysis_available": self.mal_model is not None
                    }
                }
                self.record_diagnosis(prediction_result)
                return prediction_result
        except Exception as e:
            traceback.print_exc()
            return {"error": f"处理失败: {str(e)}"}

    def _segment_ct(self, ct_hu_a):
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
        origin_xyz, vxSize_xyz = ct_mhd.GetOrigin(), ct_mhd.GetSpacing()
        direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)
        candidateLabel_a, candidate_count = ndimage.label(clean_a)
        if candidate_count == 0: return []
        centerIrc_list = ndimage.center_of_mass(ct_hu_a.clip(-1000, 1000) + 1001, labels=candidateLabel_a,
                                                index=np.arange(1, candidate_count + 1))
        if not isinstance(centerIrc_list, list): centerIrc_list = [centerIrc_list]
        candidateInfo_list = []
        for center_irc in centerIrc_list:
            center_xyz = voxelCoord2patientCoord(center_irc, origin_xyz, vxSize_xyz, direction_a)
            if np.all(np.isfinite(center_irc)) and np.all(np.isfinite(center_xyz)):
                candidateInfo_list.append(CandidateInfoTuple(False, False, False, 0.0, series_uid, center_xyz))
        return candidateInfo_list

    def _get_ct_chunk(self, ct_hu_a, center_xyz, origin_xyz, vxSize_xyz, direction_a):
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
        print("开始分类候选区域...")
        origin_xyz, vxSize_xyz = ct_mhd.GetOrigin(), ct_mhd.GetSpacing()
        direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)
        classifications_list = []
        with torch.no_grad():
            for i, candidate in enumerate(candidateInfo_list):
                input_g = self._get_ct_chunk(ct_hu_a, candidate.center_xyz, origin_xyz, vxSize_xyz, direction_a).to(
                    self.device)
                _, prob_nodule_g = self.cls_model(input_g)
                prob_nodule = prob_nodule_g[0, 1].item()
                prob_mal = None
                if self.mal_model is not None:
                    _, prob_mal_g = self.mal_model(input_g)
                    prob_mal = prob_mal_g[0, 1].item()
                classifications_list.append((prob_nodule, prob_mal, candidate.center_xyz, None))
        print("分类完成。")
        return classifications_list

    def record_diagnosis(self, prediction_result):
        """记录诊断历史 (修复版：保存完整的预测结果)"""
        try:
            summary = prediction_result.get('summary', {})
            overall_finding = summary.get('overall_finding', 'unknown')

            confidence = 0.0
            if summary.get('most_concerning_nodule'):
                mal_prob = summary['most_concerning_nodule'].get('malignancy_probability')
                if mal_prob is not None: confidence = mal_prob

            # --- 关键修复：直接将完整的 prediction_result 保存下来 ---
            # 同时保留顶层的 'id', 'diagnosis', 'confidence' 以便列表快速访问
            diagnosis_record = {
                **prediction_result,  # 使用对象展开运算符，复制所有键值对
                'id': len(self.diagnosis_history) + 1,
                'diagnosis': overall_finding,
                'confidence': confidence,
            }
            # ---------------------------------------------------------

            self.diagnosis_history.append(diagnosis_record)
            print(f"记录诊断: {diagnosis_record['filename']} - {diagnosis_record['diagnosis']}")
        except Exception as e:
            print(f"记录诊断失败: {str(e)}")


# --- 实例化系统，让所有路由都能访问它 ---
prediction_system = TumorPredictionSystem()


# --- Flask API 路由 ---
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": prediction_system.models_loaded,
        "device": str(prediction_system.device),
        "timestamp": datetime.now().isoformat(),
        "mode": "production" if prediction_system.models_loaded else "simulation"
    })


# --- 新增：补全 /api/models/status 路由 ---
@app.route('/api/models/status', methods=['GET'])
def get_models_status():
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
    try:
        if 'mhd_file' not in request.files or 'raw_file' not in request.files: return jsonify(
            {"error": "请上传.mhd和.raw文件对"}), 400
        mhd_file, raw_file = request.files['mhd_file'], request.files['raw_file']
        if not mhd_file.filename.endswith('.mhd') or not raw_file.filename.endswith('.raw'): return jsonify(
            {"error": "文件类型错误"}), 400
        if mhd_file.filename.replace('.mhd', '') != raw_file.filename.replace('.raw', ''): return jsonify(
            {"error": "文件名不匹配"}), 400
        result = prediction_system.process_ct_files(mhd_file.read(), raw_file.read(), mhd_file.filename,
                                                    raw_file.filename)
        if "error" in result: return jsonify(result), 500
        result['success'] = True
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"上传失败: {str(e)}"}), 500


@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    return jsonify({"predictions": sorted(prediction_system.diagnosis_history, key=lambda x: x['id'], reverse=True),
                    "total_count": len(prediction_system.diagnosis_history)})


# --- 补全其他缺失的路由 ---
@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    now = datetime.now()
    today_diagnoses_list = [d for d in prediction_system.diagnosis_history if
                            datetime.fromisoformat(d['timestamp']).date() == now.date()]
    return jsonify({
        "today_diagnoses": len(today_diagnoses_list),
        "benign_count": len([d for d in today_diagnoses_list if d['diagnosis'] == 'low_risk']),
        "malignant_count": len([d for d in today_diagnoses_list if d['diagnosis'] == 'high_risk']),
        "pending_count": random.randint(0, 5),  # 模拟
        "model_accuracies": {"segmentation": 98.9, "classification": 65.6, "malignancy": 67.0},  # 硬编码
        "system_performance": {"cpu_usage": random.randint(20, 70), "memory_usage": random.randint(40, 80),
                               "gpu_usage": random.randint(10, 50)},  # 模拟
        "timestamp": now.isoformat(),
    })


@app.route('/api/diagnosis-trend', methods=['GET'])
def get_diagnosis_trend():
    trend_data = []
    for i in range(7):
        date = datetime.now() - timedelta(days=6 - i)
        count = len([d for d in prediction_system.diagnosis_history if
                     datetime.fromisoformat(d['timestamp']).date() == date.date()])
        trend_data.append({"date": date.strftime("%m-%d"), "count": count})
    return jsonify({"trend": trend_data})


@app.route('/api/diagnosis-distribution', methods=['GET'])
def get_diagnosis_distribution():
    dist = {
        "良性结节": len([d for d in prediction_system.diagnosis_history if d['diagnosis'] == 'low_risk']),
        "恶性结节": len([d for d in prediction_system.diagnosis_history if d['diagnosis'] == 'high_risk']),
        "需要进一步检查": len([d for d in prediction_system.diagnosis_history if d['diagnosis'] == 'moderate_risk']),
        "正常": len([d for d in prediction_system.diagnosis_history if d['diagnosis'] == 'no_nodules_found']),
    }
    return jsonify({"distribution": [{"name": k, "value": v} for k, v in dist.items() if v > 0]})


@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    # ... (此路由逻辑保持不变)
    pass


if __name__ == '__main__':
    print("启动AI辅助肺肿瘤预测系统后端...")
    app.run(host='0.0.0.0', port=5000, debug=True)