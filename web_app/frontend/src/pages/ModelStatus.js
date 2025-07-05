import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Progress, Alert, Spin, Tag, Descriptions } from 'antd';
import { 
  CheckCircleOutlined, 
  ExclamationCircleOutlined,
  RobotOutlined,
  DatabaseOutlined,
  SettingOutlined,
  BookOutlined
} from '@ant-design/icons';
import axios from 'axios';

const ModelStatus = () => {
  const [modelStatus, setModelStatus] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  // --- 新增：固定的、基于真实测试的性能指标 ---
  const modelPerformanceBenchmarks = {
    segmentation: {
      accuracy: 98.9,
      description: "在13万+样本上，像素级分割的准确率",
    },
    classification: {
      accuracy: 65.6,
      description: "在154个结节样本上，判断是否为结节的准确率",
    },
    malignancy: {
      accuracy: 67.0,
      description: "在已确认的结节上，判断良恶性的准确率",
    },
  };
  // ------------------------------------------

  useEffect(() => {
    fetchModelStatus();
  }, []);

  const fetchModelStatus = async () => {
    try {
      const [modelResponse, healthResponse] = await Promise.all([
        axios.get('/api/models/status'),
        axios.get('/api/health')
      ]);

      setModelStatus(modelResponse.data);
      setHealthStatus(healthResponse.data);
    } catch (error) {
      console.error('获取模型状态失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const getModelStatusColor = (available) => available ? 'green' : 'red';
  const getModelStatusIcon = (available) => available ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />;

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>正在检查模型状态...</div>
      </div>
    );
  }

  return (
    <div>
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Alert
            message="AI模型状态中心"
            description={
              healthStatus?.models_loaded
                ? "所有必要的AI模型已成功加载，系统处于真实预测模式。"
                : "部分或全部必要模型未加载，系统当前可能无法进行真实预测。"
            }
            type={healthStatus?.models_loaded ? "success" : "warning"}
            showIcon
          />
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card><Statistic title="系统状态" value={healthStatus?.status === 'healthy' ? '正常' : '异常'} prefix={healthStatus?.status === 'healthy' ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />} valueStyle={{ color: healthStatus?.status === 'healthy' ? '#3f8600' : '#cf1322' }} /></Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card><Statistic title="运行模式" value={healthStatus?.mode === 'production' ? '真实预测' : '模拟'} prefix={<SettingOutlined />} valueStyle={{ color: healthStatus?.mode === 'production' ? '#3f8600' : '#faad14' }} /></Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card><Statistic title="计算设备" value={healthStatus?.device?.toUpperCase()} prefix={<DatabaseOutlined />} /></Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card><Statistic title="可用模型数" value={Object.values(modelStatus?.available_models || {}).filter(Boolean).length} suffix={`/ 3`} prefix={<RobotOutlined />} /></Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={12}>
          <Card title="模型加载状态">
            <Descriptions bordered column={1}>
              <Descriptions.Item label="分割模型 (必需)">
                <Tag color={getModelStatusColor(modelStatus?.available_models?.segmentation)} icon={getModelStatusIcon(modelStatus?.available_models?.segmentation)}>
                  {modelStatus?.available_models?.segmentation ? '已加载' : '未找到'}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="结节分类模型 (必需)">
                <Tag color={getModelStatusColor(modelStatus?.available_models?.classification)} icon={getModelStatusIcon(modelStatus?.available_models?.classification)}>
                  {modelStatus?.available_models?.classification ? '已加载' : '未找到'}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="恶性分类模型 (可选)">
                <Tag color={getModelStatusColor(modelStatus?.available_models?.malignancy)} icon={getModelStatusIcon(modelStatus?.available_models?.malignancy)}>
                  {modelStatus?.available_models?.malignancy ? '已加载' : '未找到'}
                </Tag>
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>

        <Col xs={24} lg={12}>
           <Card title="模型性能基准">
            <p style={{marginBottom: '16px', color: '#666'}}>以下为模型在标准测试集上的性能指标，代表了模型的理论能力。</p>
            <div>
                <span>分割模型准确率</span>
                <Progress percent={modelPerformanceBenchmarks.segmentation.accuracy} />
                <div style={{fontSize: 12, color: '#999'}}>{modelPerformanceBenchmarks.segmentation.description}</div>
            </div>
             <div style={{marginTop: '16px'}}>
                <span>结节分类模型准确率</span>
                <Progress percent={modelPerformanceBenchmarks.classification.accuracy} />
                 <div style={{fontSize: 12, color: '#999'}}>{modelPerformanceBenchmarks.classification.description}</div>
            </div>
             <div style={{marginTop: '16px'}}>
                <span>恶性分类模型准确率</span>
                <Progress percent={modelPerformanceBenchmarks.malignancy.accuracy} status="active" />
                 <div style={{fontSize: 12, color: '#999'}}>{modelPerformanceBenchmarks.malignancy.description}</div>
            </div>
          </Card>
        </Col>
      </Row>

      <Row style={{ marginTop: 16 }}>
        <Col span={24}>
            <Card title="关于模型性能">
                <Alert
                    message="如何理解性能指标？"
                    description="这些准确率是在特定的、标准化的数据集上通过交叉验证得出的，它们反映了模型在受控环境下的性能上限。在实际临床应用中，由于数据多样性和复杂性，实际表现可能会有差异。本系统旨在作为辅助诊断工具，最终诊断需由专业医师结合临床信息做出。"
                    type="info"
                    showIcon
                    icon={<BookOutlined />}
                />
            </Card>
        </Col>
      </Row>
    </div>
  );
};

export default ModelStatus;