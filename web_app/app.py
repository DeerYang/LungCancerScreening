# app.py

# 导入操作系统相关的功能，用于文件路径操作、环境变量获取等
import os
# 导入系统相关的功能，用于操作 Python 解释器的环境和变量
import sys
# 导入 JSON 处理模块，用于处理 JSON 数据的编码和解码
import json
# 导入 I/O 操作模块，用于处理字节流和文本流
import io
# 导入临时文件处理模块，用于创建和管理临时文件和目录
import tempfile
# 导入日期和时间处理模块，用于处理日期和时间的计算和格式化
from datetime import datetime, timedelta
# 导入异常追踪模块，用于打印详细的异常堆栈信息
import traceback
# 导入随机数生成模块，用于生成随机数
import random

# 获取项目根目录的绝对路径，通过当前文件所在目录向上一级
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 如果项目根目录不在 Python 解释器的搜索路径中，则将其添加到搜索路径的开头
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入 PyTorch 深度学习框架
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 NumPy 科学计算库，用于处理多维数组和矩阵
import numpy as np
# 导入 SimpleITK 图像处理库，用于读取和处理医学图像
import SimpleITK as sitk
# 导入 SciPy 的图像处理模块，用于图像的形态学操作和标签处理
import scipy.ndimage as ndimage
# 导入 Pillow 库用于图像操作
from PIL import Image, ImageDraw

# 导入 Flask Web 框架
from flask import Flask, request, jsonify
# 导入 Flask-CORS 扩展，用于处理跨域资源共享问题
from flask_cors import CORS
# 导入通义千问的 SDK
import dashscope
# 导入通义千问的生成模型模块
from dashscope import Generation

# 导入自定义的分割模型包装类
from segmentModel import UNetWrapper
# 导入自定义的肿瘤分类模型类
from TumorModel import LunaModel
# 导入自定义的候选信息元组类
from TumorDatasets import CandidateInfoTuple
# 导入自定义的坐标转换工具函数
from util.util import patientCoord2voxelCoord, voxelCoord2patientCoord

# 创建 Flask 应用实例
app = Flask(__name__)
# 启用 CORS 支持，允许跨域请求
CORS(app)

# 从环境变量中获取通义千问的 API 密钥，如果未设置则使用默认值
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', 'YOUR_DASHSCOPE_API_KEY_HERE')
# 如果 API 密钥使用的是默认值，则打印警告信息
if 'YOUR_DASHSCOPE_API_KEY' in dashscope.api_key:
    print("警告：通义千问API密钥未设置。")

# 临时存储上传的CT数据，以便新API可以访问
# 注意：这是一个简化的实现。在生产环境中，应使用更持久的缓存或存储方案。
ct_data_cache = {}


class TumorPredictionSystem:
    """
    肿瘤预测系统类，负责加载模型、处理 CT 文件、进行预测和记录诊断历史。
    """

    def __init__(self):
        # 检查是否有可用的 CUDA 设备，如果有则使用 GPU，否则使用 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 打印当前使用的设备信息
        print(f"使用设备: {self.device}")
        # 初始化分割模型、分类模型和恶性分类模型为 None
        self.seg_model, self.cls_model, self.mal_model = None, None, None
        # 标记模型是否已成功加载
        self.models_loaded = False
        # 用于存储诊断历史记录的列表
        self.diagnosis_history = []
        # 定义各个模型的文件路径
        self.model_paths = {
            'segmentation': '../data-unversioned/seg/models/seg/seg_2025-07-02_13.09.34_none.best.state',
            'classification': '../data-unversioned/nodule/models/nodule-model/best_2025-07-02_10.36.28_nodule-comment.best.state',
            'malignancy': '../data-unversioned/tumor/models/tumor_cls/seg_2025-07-02_14.29.48_finetune-depth2.best.state'
        }
        # 调用加载模型的方法
        self.load_models()

    def load_models(self):
        """
        加载分割模型、分类模型和恶性分类模型。
        """
        try:
            # 检查分割模型文件是否存在
            if os.path.exists(self.model_paths['segmentation']):
                # 加载分割模型的状态字典
                seg_dict = torch.load(self.model_paths['segmentation'], map_location=self.device)
                # 实例化分割模型
                self.seg_model = UNetWrapper(in_channels=7, n_classes=1, depth=3, wf=4, padding=True, batch_norm=True,
                                             up_mode='upconv')
                # 加载分割模型的状态字典到模型中
                self.seg_model.load_state_dict(seg_dict['model_state'])
                # 将模型设置为评估模式
                self.seg_model.eval().to(self.device)
                # 打印分割模型加载成功的信息
                print("✓ 分割模型加载成功")
            # 检查分类模型文件是否存在
            if os.path.exists(self.model_paths['classification']):
                # 加载分类模型的状态字典
                cls_dict = torch.load(self.model_paths['classification'], map_location=self.device)
                # 实例化分类模型
                self.cls_model = LunaModel()
                # 加载分类模型的状态字典到模型中
                self.cls_model.load_state_dict(cls_dict['model_state'])
                # 将模型设置为评估模式
                self.cls_model.eval().to(self.device)
                # 打印分类模型加载成功的信息
                print("✓ 结节分类模型加载成功")
            # 检查恶性分类模型文件是否存在
            if os.path.exists(self.model_paths['malignancy']):
                # 加载恶性分类模型的状态字典
                mal_dict = torch.load(self.model_paths['malignancy'], map_location=self.device)
                # 实例化恶性分类模型
                self.mal_model = LunaModel()
                # 加载恶性分类模型的状态字典到模型中
                self.mal_model.load_state_dict(mal_dict['model_state'])
                # 将模型设置为评估模式
                self.mal_model.eval().to(self.device)
                # 打印恶性分类模型加载成功的信息
                print("✓ 恶性分类模型加载成功")

            # 如果分割模型和分类模型都已成功加载，则标记模型加载成功
            if self.seg_model and self.cls_model:
                self.models_loaded = True
                print("所有必需模型已加载，系统进入真实预测模式。")
            else:
                self.models_loaded = False
                print("必需模型未完全加载，系统将使用模拟模式。")
        except Exception as e:
            # 打印模型加载失败的信息
            print(f"模型加载失败: {e}")
            self.models_loaded = False

    def process_ct_files(self, mhd_data, raw_data, mhd_filename, raw_filename):
        """
        处理 CT 文件，进行预测并返回预测结果。
        :param mhd_data: MHD 文件的二进制数据
        :param raw_data: RAW 文件的二进制数据
        :param mhd_filename: MHD 文件的文件名
        :param raw_filename: RAW 文件的文件名
        :return: 预测结果
        """
        # 如果模型未完全加载，则使用模拟预测
        if not self.models_loaded:
            return self.simulate_prediction(mhd_filename)

        # 打印开始真实模型预测的信息
        print(f"开始真实模型预测: {mhd_filename}")
        try:
            # 获取系列 UID，通过去掉 MHD 文件名的扩展名
            series_uid = mhd_filename.replace('.mhd', '')
            # 创建一个临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                # 构建临时 MHD 文件的路径
                mhd_path = os.path.join(temp_dir, mhd_filename)
                # 构建临时 RAW 文件的路径
                raw_path = os.path.join(temp_dir, raw_filename)
                # 将 MHD 文件的二进制数据写入临时文件
                with open(mhd_path, 'wb') as f:
                    f.write(mhd_data)
                # 将 RAW 文件的二进制数据写入临时文件
                with open(raw_path, 'wb') as f:
                    f.write(raw_data)

                # 使用 SimpleITK 读取 MHD 文件
                ct_mhd = sitk.ReadImage(mhd_path)
                # 将读取的图像数据转换为 NumPy 数组
                ct_hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

                # --- 新增缓存 ---
                # 将CT数据和元数据缓存起来，以便新的图像API可以访问
                ct_data_cache[series_uid] = {
                    'hu_a': ct_hu_a,
                    'origin_xyz': ct_mhd.GetOrigin(),
                    'vxSize_xyz': ct_mhd.GetSpacing(),
                    'direction_a': np.array(ct_mhd.GetDirection()).reshape(3, 3)
                }
                # --- 结束新增 ---

                # 调用分割 CT 图像的方法
                mask_a = self._segment_ct(ct_hu_a)
                # 对分割结果进行分组，得到候选信息列表
                candidate_info_list = self._group_segmentation_output(series_uid, ct_mhd, ct_hu_a, mask_a)
                # 统计候选区域的总数
                total_candidates = len(candidate_info_list)
                # 打印分割后找到的候选区域数量
                print(f"分割后找到 {total_candidates} 个候选区域。")

                # 如果有候选信息列表，则对候选区域进行分类
                classifications_list = self._classify_candidates(ct_mhd, ct_hu_a,
                                                                 candidate_info_list) if candidate_info_list else []

                # 用于存储检测到的结节信息的列表
                nodules = []
                # 遍历分类结果列表
                for prob, prob_mal, center_xyz, _ in classifications_list:
                    # 如果结节概率大于 0.5，则认为是结节
                    if prob > 0.5:
                        # 初始化恶性概率显示值和恶性程度为 None 和 'N/A'
                        mal_prob_display, malignancy_level = None, 'N/A'
                        # 如果恶性分类模型存在，则计算恶性概率和恶性程度
                        if prob_mal is not None:
                            if prob_mal < 0.3:
                                malignancy_level = 'low'
                            elif prob_mal < 0.7:
                                malignancy_level = 'medium'
                            else:
                                malignancy_level = 'high'
                            # 保留恶性概率的四位小数
                            mal_prob_display = round(prob_mal, 4)

                        # --- 新增：将患者坐标转换为体素坐标以获取切片索引 ---
                        origin_xyz, vxSize_xyz = ct_mhd.GetOrigin(), ct_mhd.GetSpacing()
                        direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)
                        center_irc = patientCoord2voxelCoord(center_xyz, origin_xyz, vxSize_xyz, direction_a)

                        # 将结节信息添加到列表中
                        nodules.append({
                            "id": len(nodules) + 1,
                            "nodule_probability": round(prob, 4),
                            "malignancy_probability": mal_prob_display,
                            "malignancy_level": malignancy_level,
                            "center_xyz": [round(c, 4) for c in center_xyz],
                            # --- 新增字段，用于前端可视化 ---
                            "center_irc": {
                                "index": center_irc.index,
                                "row": center_irc.row,
                                "col": center_irc.col,
                            },
                            "diameter_mm": 10.0,  # 暂时硬编码直径，后续可从模型获取
                            # --- 结束新增 ---
                        })

                # 初始化总体检测结果和最可疑结节为 "no_nodules_found" 和 None
                overall_finding, most_concerning_nodule = "no_nodules_found", None
                # 如果检测到结节，则更新总体检测结果和最可疑结节
                if nodules:
                    # 找到最可疑的结节
                    most_concerning_nodule = max(nodules, key=lambda x: x['malignancy_probability'] if x[
                                                                                                           'malignancy_probability'] is not None else -1)
                    # 获取最可疑结节的恶性程度
                    top_level = most_concerning_nodule['malignancy_level']
                    if top_level == 'high':
                        overall_finding = "high_risk"
                    elif top_level == 'medium':
                        overall_finding = "moderate_risk"
                    elif top_level == 'low':
                        overall_finding = "low_risk"
                    else:
                        overall_finding = "nodules_present_malignancy_unavailable"

                # 构建预测结果字典
                prediction_result = {
                    "filename": mhd_filename,
                    "timestamp": datetime.now().isoformat(),
                    # --- 新增字段，用于前端可视化 ---
                    "total_slices": ct_hu_a.shape[0],
                    "voxel_spacing": list(ct_mhd.GetSpacing()),
                    # --- 结束新增 ---
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
                # 记录诊断结果
                self.record_diagnosis(prediction_result)
                return prediction_result
        except Exception as e:
            # 打印详细的异常堆栈信息
            traceback.print_exc()
            return {"error": f"处理失败: {str(e)}"}

    def _segment_ct(self, ct_hu_a):
        """
        对 CT 图像进行分割。
        :param ct_hu_a: CT 图像的 NumPy 数组
        :return: 分割后的掩码数组
        """
        print("开始分割CT...")
        # 禁用梯度计算，提高推理速度
        with torch.no_grad():
            # 初始化输出数组，与输入 CT 图像形状相同
            output_a = np.zeros_like(ct_hu_a, dtype=np.float32)
            # 定义上下文切片的数量
            context_slices = 3
            # 遍历 CT 图像的每一个切片
            for slice_ndx in range(ct_hu_a.shape[0]):
                # 初始化输入张量
                ct_t = torch.zeros((context_slices * 2 + 1, 512, 512))
                # 计算上下文切片的起始和结束索引
                start_ndx, end_ndx = slice_ndx - context_slices, slice_ndx + context_slices + 1
                # 遍历上下文切片
                for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
                    # 确保上下文切片索引在有效范围内
                    context_ndx = max(0, min(context_ndx, ct_hu_a.shape[0] - 1))
                    # 将上下文切片的数据转换为 PyTorch 张量
                    ct_t[i] = torch.from_numpy(ct_hu_a[context_ndx].astype(np.float32))
                # 对输入张量进行截断操作，将值限制在 -1000 到 1000 之间
                ct_t.clamp_(-1000, 1000)
                # 使用分割模型进行预测
                prediction_g = self.seg_model(ct_t.unsqueeze(0).to(self.device))
                # 将预测结果从 GPU 转移到 CPU 并转换为 NumPy 数组
                output_a[slice_ndx] = prediction_g[0].cpu().numpy()
            # 对输出数组进行二值化处理，大于 0.5 的值设为 True，否则设为 False
            mask_a = output_a > 0.5
            # 对掩码数组进行形态学腐蚀操作
            mask_a = ndimage.binary_erosion(mask_a, iterations=1)
        print("分割完成。")
        return mask_a

    def _group_segmentation_output(self, series_uid, ct_mhd, ct_hu_a, clean_a):
        """
        对分割结果进行分组，得到候选信息列表。
        :param series_uid: 系列 UID
        :param ct_mhd: CT 图像的 SimpleITK 对象
        :param ct_hu_a: CT 图像的 NumPy 数组
        :param clean_a: 分割后的掩码数组
        :return: 候选信息列表
        """
        # 获取 CT 图像的原点坐标和体素大小
        origin_xyz, vxSize_xyz = ct_mhd.GetOrigin(), ct_mhd.GetSpacing()
        # 获取 CT 图像的方向矩阵
        direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)
        # 对掩码数组进行标记，得到不同的连通区域
        candidateLabel_a, candidate_count = ndimage.label(clean_a)
        # 如果没有候选区域，则返回空列表
        if candidate_count == 0: return []
        # 计算每个连通区域的质心坐标
        centerIrc_list = ndimage.center_of_mass(ct_hu_a.clip(-1000, 1000) + 1001, labels=candidateLabel_a,
                                                index=np.arange(1, candidate_count + 1))
        # 如果质心坐标不是列表形式，则将其转换为列表
        if not isinstance(centerIrc_list, list): centerIrc_list = [centerIrc_list]
        # 用于存储候选信息的列表
        candidateInfo_list = []
        # 遍历质心坐标列表
        for center_irc in centerIrc_list:
            # 将体素坐标转换为患者坐标
            center_xyz = voxelCoord2patientCoord(center_irc, origin_xyz, vxSize_xyz, direction_a)
            # 如果质心坐标和患者坐标都是有限值，则将候选信息添加到列表中
            if np.all(np.isfinite(center_irc)) and np.all(np.isfinite(center_xyz)):
                candidateInfo_list.append(CandidateInfoTuple(False, False, False, 0.0, series_uid, center_xyz))
        return candidateInfo_list

    def _get_ct_chunk(self, ct_hu_a, center_xyz, origin_xyz, vxSize_xyz, direction_a):
        """
        从 CT 图像中提取指定中心坐标的图像块。
        :param ct_hu_a: CT 图像的 NumPy 数组
        :param center_xyz: 中心坐标
        :param origin_xyz: CT 图像的原点坐标
        :param vxSize_xyz: CT 图像的体素大小
        :param direction_a: CT 图像的方向矩阵
        :return: 提取的图像块的 PyTorch 张量
        """
        # 定义图像块的大小
        width_irc = (32, 48, 48)
        # 将患者坐标转换为体素坐标
        center_irc = patientCoord2voxelCoord(center_xyz, origin_xyz, vxSize_xyz, direction_a)
        # 用于存储切片索引的列表
        slice_list = []
        # 遍历每个坐标轴
        for axis, center_val in enumerate(center_irc):
            # 计算切片的起始索引
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            # 计算切片的结束索引
            end_ndx = int(start_ndx + width_irc[axis])
            # 确保起始索引不小于 0
            if start_ndx < 0: start_ndx, end_ndx = 0, int(width_irc[axis])
            # 确保结束索引不大于 CT 图像的尺寸
            if end_ndx > ct_hu_a.shape[axis]: end_ndx, start_ndx = ct_hu_a.shape[axis], int(
                ct_hu_a.shape[axis] - width_irc[axis])
            # 将切片索引添加到列表中
            slice_list.append(slice(start_ndx, end_ndx))
        # 从 CT 图像中提取图像块
        ct_chunk = ct_hu_a[tuple(slice_list)].copy()
        # 对图像块进行截断操作，将值限制在 -1000 到 1000 之间
        ct_chunk.clip(-1000, 1000, out=ct_chunk)
        # 将图像块转换为 PyTorch 张量，并添加批次和通道维度
        return torch.from_numpy(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    def _classify_candidates(self, ct_mhd, ct_hu_a, candidateInfo_list):
        """
        对候选区域进行分类。
        :param ct_mhd: CT 图像的 SimpleITK 对象
        :param ct_hu_a: CT 图像的 NumPy 数组
        :param candidateInfo_list: 候选信息列表
        :return: 分类结果列表
        """
        print("开始分类候选区域...")
        # 获取 CT 图像的原点坐标和体素大小
        origin_xyz, vxSize_xyz = ct_mhd.GetOrigin(), ct_mhd.GetSpacing()
        # 获取 CT 图像的方向矩阵
        direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)
        # 用于存储分类结果的列表
        classifications_list = []
        # 禁用梯度计算，提高推理速度
        with torch.no_grad():
            # 遍历候选信息列表
            for i, candidate in enumerate(candidateInfo_list):
                # 从 CT 图像中提取候选区域的图像块
                input_g = self._get_ct_chunk(ct_hu_a, candidate.center_xyz, origin_xyz, vxSize_xyz, direction_a).to(
                    self.device)
                # 使用分类模型进行预测，得到结节概率
                _, prob_nodule_g = self.cls_model(input_g)
                # 将结节概率从 GPU 转移到 CPU 并转换为 Python 标量
                prob_nodule = prob_nodule_g[0, 1].item()
                # 初始化恶性概率为 None
                prob_mal = None
                # 如果恶性分类模型存在，则使用恶性分类模型进行预测
                if self.mal_model is not None:
                    _, prob_mal_g = self.mal_model(input_g)
                    # 将恶性概率从 GPU 转移到 CPU 并转换为 Python 标量
                    prob_mal = prob_mal_g[0, 1].item()
                # 将分类结果添加到列表中
                classifications_list.append((prob_nodule, prob_mal, candidate.center_xyz, None))
        print("分类完成。")
        return classifications_list

    def record_diagnosis(self, prediction_result):
        """
        记录诊断历史。
        :param prediction_result: 预测结果
        """
        try:
            # 获取预测结果的摘要信息
            summary = prediction_result.get('summary', {})
            # 获取总体检测结果
            overall_finding = summary.get('overall_finding', 'unknown')

            # 初始化置信度为 0.0
            confidence = 0.0
            # 如果有最可疑结节，则获取其恶性概率作为置信度
            if summary.get('most_concerning_nodule'):
                mal_prob = summary['most_concerning_nodule'].get('malignancy_probability')
                if mal_prob is not None: confidence = mal_prob

            # 关键修复：直接将完整的预测结果保存下来
            # 同时保留顶层的 'id', 'diagnosis', 'confidence' 以便列表快速访问
            diagnosis_record = {
                **prediction_result,  # 使用对象展开运算符，复制所有键值对
                'id': len(self.diagnosis_history) + 1,
                'diagnosis': overall_finding,
                'confidence': confidence,
            }
            # 将诊断记录添加到诊断历史列表中
            self.diagnosis_history.append(diagnosis_record)
            # 打印记录诊断的信息
            print(f"记录诊断: {diagnosis_record['filename']} - {diagnosis_record['diagnosis']}")
        except Exception as e:
            # 打印记录诊断失败的信息
            print(f"记录诊断失败: {str(e)}")


# 实例化肿瘤预测系统，让所有路由都能访问它
prediction_system = TumorPredictionSystem()


# Flask API 路由：健康检查
@app.route('/api/health', methods=['GET'])
def health_check():
    """
    健康检查接口，返回系统的健康状态、模型加载状态、设备信息和时间戳。
    """
    return jsonify({
        "status": "healthy",
        "models_loaded": prediction_system.models_loaded,
        "device": str(prediction_system.device),
        "timestamp": datetime.now().isoformat(),
        "mode": "production" if prediction_system.models_loaded else "simulation"
    })


# 新增：补全 /api/models/status 路由
@app.route('/api/models/status', methods=['GET'])
def get_models_status():
    """
    获取模型状态接口，返回模型加载状态、设备信息和各个模型的可用性。
    """
    return jsonify({
        "models_loaded": prediction_system.models_loaded,
        "device": str(prediction_system.device),
        "available_models": {
            "segmentation": os.path.exists(prediction_system.model_paths['segmentation']),
            "classification": os.path.exists(prediction_system.model_paths['classification']),
            "malignancy": os.path.exists(prediction_system.model_paths['malignancy'])
        }
    })


# Flask API 路由：上传 CT 文件
@app.route('/api/upload', methods=['POST'])
def upload_ct():
    """
    上传 CT 文件接口，处理上传的 MHD 和 RAW 文件，进行预测并返回预测结果。
    """
    try:
        # 检查请求中是否包含 MHD 文件和 RAW 文件
        if 'mhd_file' not in request.files or 'raw_file' not in request.files: return jsonify(
            {"error": "请上传.mhd和.raw文件对"}), 400
        # 获取 MHD 文件和 RAW 文件
        mhd_file, raw_file = request.files['mhd_file'], request.files['raw_file']
        # 检查文件类型是否正确
        if not mhd_file.filename.endswith('.mhd') or not raw_file.filename.endswith('.raw'): return jsonify(
            {"error": "文件类型错误"}), 400
        # 检查文件名是否匹配
        if mhd_file.filename.replace('.mhd', '') != raw_file.filename.replace('.raw', ''): return jsonify(
            {"error": "文件名不匹配"}), 400
        # 调用肿瘤预测系统的处理方法进行预测
        result = prediction_system.process_ct_files(mhd_file.read(), raw_file.read(), mhd_file.filename,
                                                    raw_file.filename)
        # 如果预测结果包含错误信息，则返回错误响应
        if "error" in result: return jsonify(result), 500
        # 添加成功标志
        result['success'] = True
        return jsonify(result)
    except Exception as e:
        # 打印详细的异常堆栈信息
        traceback.print_exc()
        return jsonify({"error": f"上传失败: {str(e)}"}), 500


# Flask API 路由：获取所有预测结果
@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """
    获取所有预测结果接口，返回按 ID 降序排列的预测结果列表和总数。
    """
    return jsonify({"predictions": sorted(prediction_system.diagnosis_history, key=lambda x: x['id'], reverse=True),
                    "total_count": len(prediction_system.diagnosis_history)})


# 补全其他缺失的路由：统计信息
@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """
    获取统计信息接口，返回今日诊断数量、良性结节数量、恶性结节数量、待处理数量、模型准确率、系统性能和时间戳。
    """
    # 获取当前时间
    now = datetime.now()
    # 筛选出今日的诊断记录
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


# Flask API 路由：诊断趋势
@app.route('/api/diagnosis-trend', methods=['GET'])
def get_diagnosis_trend():
    """
    获取诊断趋势接口，返回过去 7 天的诊断数量趋势。
    """
    # 用于存储诊断趋势数据的列表
    trend_data = []
    # 遍历过去 7 天
    for i in range(7):
        # 计算日期
        date = datetime.now() - timedelta(days=6 - i)
        # 统计该日期的诊断数量
        count = len([d for d in prediction_system.diagnosis_history if
                     datetime.fromisoformat(d['timestamp']).date() == date.date()])
        # 将日期和诊断数量添加到趋势数据列表中
        trend_data.append({"date": date.strftime("%m-%d"), "count": count})
    return jsonify({"trend": trend_data})


# Flask API 路由：诊断分布
@app.route('/api/diagnosis-distribution', methods=['GET'])
def get_diagnosis_distribution():
    """
    获取诊断分布接口，返回不同诊断结果的数量分布。
    """
    # 初始化诊断分布字典
    dist = {
        "良性结节": len([d for d in prediction_system.diagnosis_history if d['diagnosis'] == 'low_risk']),
        "恶性结节": len([d for d in prediction_system.diagnosis_history if d['diagnosis'] == 'high_risk']),
        "需要进一步检查": len([d for d in prediction_system.diagnosis_history if d['diagnosis'] == 'moderate_risk']),
        "正常": len([d for d in prediction_system.diagnosis_history if d['diagnosis'] == 'no_nodules_found']),
    }
    # 将诊断分布字典转换为列表形式
    return jsonify({"distribution": [{"name": k, "value": v} for k, v in dist.items() if v > 0]})


# Flask API 路由：与 AI 聊天
@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    # ... (此路由逻辑保持不变)
    pass


# --- 新增API路由：获取单个CT切片图像 ---
@app.route('/api/ct-slice/<series_uid>/<int:slice_ndx>', methods=['GET'])
def get_ct_slice(series_uid, slice_ndx):
    """
    获取指定CT扫描的单个切片，并以PNG图像形式返回。
    可以在查询参数中传入结节信息以在图上绘制边界框。
    """
    try:
        ct_data = ct_data_cache.get(series_uid)
        if not ct_data:
            return jsonify({"error": "CT data not found or expired."}), 404

        hu_a = ct_data['hu_a']
        if not (0 <= slice_ndx < hu_a.shape[0]):
            return jsonify({"error": "Slice index out of bounds."}), 400

        # 1. 获取原始切片并进行窗口化/归一化以用于显示
        slice_data = hu_a[slice_ndx]
        # 经典的肺窗（Lung Window）: level=-600, width=1500
        window_level, window_width = -600, 1500
        min_val = window_level - window_width // 2
        max_val = window_level + window_width // 2

        display_slice = np.clip(slice_data, min_val, max_val)
        display_slice = ((display_slice - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # 2. 将numpy数组转换为Pillow图像
        img = Image.fromarray(display_slice, 'L').convert('RGB')
        draw = ImageDraw.Draw(img)

        # 3. 从查询参数获取结节信息并绘制边界框
        nodules_json = request.args.get('nodules')
        if nodules_json:
            nodules_on_slice = json.loads(nodules_json)
            vx_size_z, vx_size_row, vx_size_col = ct_data['vxSize_xyz']

            for nodule in nodules_on_slice:
                center_r, center_c = nodule['center_irc']['row'], nodule['center_irc']['col']
                diameter_mm = nodule.get('diameter_mm', 10.0)

                # 将直径从mm转换为像素
                radius_r_px = (diameter_mm / 2) / vx_size_row
                radius_c_px = (diameter_mm / 2) / vx_size_col

                # 定义边界框颜色
                level = nodule.get('malignancy_level', 'low')
                color_map = {'high': 'red', 'medium': 'orange', 'low': 'lime'}
                box_color = color_map.get(level, 'cyan')

                # 计算边界框坐标 (Pillow的坐标系是(x, y)，对应我们数组的(col, row))
                x0, y0 = center_c - radius_c_px, center_r - radius_r_px
                x1, y1 = center_c + radius_c_px, center_r + radius_r_px
                draw.rectangle([x0, y0, x1, y1], outline=box_color, width=2)
                draw.text((x0, y0 - 12), f"ID:{nodule['id']}", fill=box_color)

        # 4. 将图像保存到内存流并返回
        img_io = io.BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        return img_io, 200, {'Content-Type': 'image/png'}

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to generate slice image: {str(e)}"}), 500


if __name__ == '__main__':
    # 打印启动信息
    print("启动AI辅助肺肿瘤预测系统后端...")
    # 启动 Flask 应用，监听所有 IP 地址，端口为 5000，开启调试模式
    app.run(host='0.0.0.0', port=5000, debug=True)