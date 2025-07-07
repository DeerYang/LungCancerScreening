// src/pages/Upload.js

import React, { useState } from 'react';
import {
  Upload,
  Button,
  Card,
  Row,
  Col,
  Result,
  Descriptions,
  Tag,
  Progress,
  Alert,
  Spin,
  message,
  List,
  Space,
  Typography,
  Empty,
  Avatar
} from 'antd';
import {
  InboxOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  UploadOutlined,
  CloseCircleOutlined,
  HeartOutlined,
  RadarChartOutlined,
  WarningOutlined,
  EyeOutlined // 导入新图标
} from '@ant-design/icons';
import axios from 'axios';
import CTViewer from '../components/CTViewer'; // 假设 CTViewer.js 和 Upload.js 在同一目录

const { Title } = Typography;

const UploadPage = () => {
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [mhdFile, setMhdFile] = useState(null);
  const [rawFile, setRawFile] = useState(null);
  const [ctModalVisible, setCtModalVisible] = useState(false);

  const handleMhdFileChange = ({ file }) => {
    if (file.status === 'removed') {
      setMhdFile(null);
    } else {
      setMhdFile(file);
    }
  };

  const handleRawFileChange = ({ file }) => {
    if (file.status === 'removed') {
      setRawFile(null);
    } else {
      setRawFile(file);
    }
  };

  const resetUpload = () => {
    setMhdFile(null);
    setRawFile(null);
    setResult(null);
    setUploading(false);
    message.success('已重置，可以开始新的分析。');
  };

  const handleUpload = async () => {
    if (!mhdFile || !rawFile) {
      message.error('请同时选择 .mhd 和 .raw 文件');
      return;
    }
    if (mhdFile.name.replace('.mhd', '') !== rawFile.name.replace('.raw', '')) {
      message.error('文件名不匹配，请确保文件来自同一次CT扫描');
      return;
    }

    setUploading(true);
    setResult(null);

    const formData = new FormData();
    formData.append('mhd_file', mhdFile.originFileObj || mhdFile);
    formData.append('raw_file', rawFile.originFileObj || rawFile);

    try {
      const response = await axios.post('/api/upload', formData, {
        timeout: 600000, // 10分钟超时
      });
      if (response.data.success) {
        setResult(response.data);
        message.success('CT文件分析完成！');
      } else {
        throw new Error(response.data.error || '分析失败');
      }
    } catch (error) {
      console.error('上传失败:', error);
      setResult({ error: error.response?.data?.error || error.message || '上传或分析失败，请检查后端服务或文件。' });
      message.error(error.response?.data?.error || '上传失败，请重试');
    } finally {
      setUploading(false);
    }
  };

  const getOverallFindingInfo = (finding) => {
    switch(finding) {
      case 'high_risk':
        return { color: 'red', text: '高风险', icon: <ExclamationCircleOutlined /> };
      case 'moderate_risk':
        return { color: 'orange', text: '中等风险', icon: <WarningOutlined /> };
      case 'low_risk':
        return { color: 'green', text: '低风险', icon: <CheckCircleOutlined /> };
      case 'no_nodules_found':
        return { color: 'blue', text: '未发现结节', icon: <HeartOutlined /> };
      default:
        return { color: 'grey', text: '未知或分析不可用', icon: <CloseCircleOutlined /> };
    }
  };

  const getMalignancyColor = (level) => {
    switch (level) {
      case 'high': return 'red';
      case 'medium': return 'orange';
      case 'low': return 'green';
      default: return 'blue';
    }
  };

  const renderResult = () => {
    if (!result) return null;

    if (result.error) {
      return (
        <Result
          status="error"
          title="分析失败"
          subTitle={result.error}
          extra={<Button type="primary" key="retry" onClick={resetUpload}>重新上传</Button>}
        />
      );
    }

    const { summary, nodules, support_info } = result;
    const overallInfo = getOverallFindingInfo(summary.overall_finding);

    return (
      <div>
        <Result
          icon={overallInfo.icon}
          status={overallInfo.color === 'red' ? 'error' : (overallInfo.color === 'orange' ? 'warning' : 'success')}
          title={`分析完成：${overallInfo.text}`}
          subTitle={`文件名: ${result.filename} | 分析时间: ${new Date(result.timestamp).toLocaleString()}`}
          extra={<Button type="primary" onClick={resetUpload}>分析新图像</Button>}
        />

        <Row gutter={[16, 16]}>
          <Col xs={24} md={12}>
            <Card title="分析摘要" style={{ marginBottom: 16 }}>
              <Descriptions bordered column={1}>
                <Descriptions.Item label="总体发现">
                  <Tag color={overallInfo.color}>{overallInfo.text}</Tag>
                </Descriptions.Item>
                <Descriptions.Item label="检测到结节数量">
                  {summary.nodule_count}
                </Descriptions.Item>
                <Descriptions.Item label="候选区域总数">
                  {support_info.total_candidates}
                </Descriptions.Item>
                <Descriptions.Item label="恶性分析">
                  <Tag color={support_info.malignancy_analysis_available ? 'green' : 'red'}>
                    {support_info.malignancy_analysis_available ? '可用' : '不可用'}
                  </Tag>
                </Descriptions.Item>
              </Descriptions>
            </Card>

            <Card title="最可疑结节详情">
              {summary.most_concerning_nodule ? (
                <Descriptions bordered column={1}>
                  <Descriptions.Item label="结节ID">{summary.most_concerning_nodule.id}</Descriptions.Item>
                  <Descriptions.Item label="结节可能性">
                    <Progress percent={summary.most_concerning_nodule.nodule_probability * 100} size="small" />
                  </Descriptions.Item>
                  <Descriptions.Item label="恶性风险等级">
                    <Tag color={getMalignancyColor(summary.most_concerning_nodule.malignancy_level)}>
                      {summary.most_concerning_nodule.malignancy_level.toUpperCase()}
                    </Tag>
                  </Descriptions.Item>
                   <Descriptions.Item label="恶性概率">
                    <Progress percent={(summary.most_concerning_nodule.malignancy_probability || 0) * 100} size="small" status="exception" />
                  </Descriptions.Item>
                </Descriptions>
              ) : (
                <Empty description="未发现可疑结节" />
              )}
            </Card>
          </Col>

          <Col xs={24} md={12}>
            <Card
              title={`结节详细列表 (${summary.nodule_count}个)`}
              extra={
                <Button
                  icon={<EyeOutlined />}
                  onClick={() => setCtModalVisible(true)}
                  disabled={!result || !result.total_slices || summary.nodule_count === 0}
                >
                  显示CT图像
                </Button>
              }
            >
              <List
                itemLayout="horizontal"
                dataSource={nodules}
                locale={{ emptyText: <Empty description="未发现结节" /> }}
                renderItem={item => (
                  <List.Item>
                    <List.Item.Meta
                      avatar={<Avatar style={{ backgroundColor: getMalignancyColor(item.malignancy_level) }} icon={<RadarChartOutlined />} />}
                      title={`结节 #${item.id}`}
                      description={
                        <Space direction="vertical" size="small" style={{width: '100%'}}>
                          <div>结节概率: {(item.nodule_probability * 100).toFixed(2)}%</div>
                          <div>恶性概率: {item.malignancy_probability !== null ? `${(item.malignancy_probability * 100).toFixed(4)}%` : 'N/A'}</div>
                        </Space>
                      }
                    />
                     <Tag color={getMalignancyColor(item.malignancy_level)}>{item.malignancy_level.toUpperCase()}</Tag>
                  </List.Item>
                )}
              />
            </Card>
          </Col>
        </Row>

        <CTViewer
          visible={ctModalVisible}
          onCancel={() => setCtModalVisible(false)}
          result={result}
        />
      </div>
    );
  };


  return (
    <div style={{ padding: '24px', maxWidth: '1200px', margin: '0 auto' }}>
      <Title level={2}>CT文件上传与分析</Title>

      {!result && (
        <>
          <Alert
            message="文件格式要求"
            description="请上传DICOM格式的CT扫描文件对：.mhd文件（元数据）和.raw文件（体素数据）。这两个文件必须来自同一个CT扫描，文件名前缀必须相同。"
            type="info"
            showIcon
            style={{ marginBottom: '24px' }}
          />

          <div style={{ minHeight: '160px', marginBottom: '24px' }}>
            <Row gutter={16}>
                <Col span={12}>
                  <Upload.Dragger
                      accept=".mhd"
                      beforeUpload={() => false}
                      onChange={handleMhdFileChange}
                      maxCount={1}
                      fileList={mhdFile ? [mhdFile] : []}
                  >
                      <p className="ant-upload-drag-icon"><InboxOutlined /></p>
                      <p className="ant-upload-text">点击或拖拽 .mhd 文件到此区域</p>
                      <p className="ant-upload-hint">元数据文件</p>
                  </Upload.Dragger>
                </Col>
                <Col span={12}>
                  <Upload.Dragger
                      accept=".raw"
                      beforeUpload={() => false}
                      onChange={handleRawFileChange}
                      maxCount={1}
                      fileList={rawFile ? [rawFile] : []}
                  >
                      <p className="ant-upload-drag-icon"><InboxOutlined /></p>
                      <p className="ant-upload-text">点击或拖拽 .raw 文件到此区域</p>
                      <p className="ant-upload-hint">体素数据文件</p>
                  </Upload.Dragger>
                </Col>
            </Row>
          </div>

          <div style={{textAlign: 'center', marginTop: '48px'}}>
            <Button
                type="primary"
                size="large"
                onClick={handleUpload}
                loading={uploading}
                disabled={!mhdFile || !rawFile}
                icon={<UploadOutlined/>}
            >
              {uploading ? '正在分析中...' : '开始分析'}
            </Button>
          </div>

          {uploading && (
              <div style={{marginTop: '24px'}}>
                 <Spin spinning={true} size="large">
                    <Alert
                        message="分析进行中，请勿关闭页面"
                        description={
                            <div>
                                <p>AI模型正在进行三阶段分析，这可能需要几分钟时间。</p>
                                <ol>
                                    <li><strong>图像分割</strong>: 识别CT图像中的结节候选区域。</li>
                                    <li><strong>候选提取</strong>: 从分割结果中提取独立的候选结节。</li>
                                    <li><strong>分类评估</strong>: 判断每个结节的可能性及其恶性风险。</li>
                                </ol>
                            </div>
                        }
                        type="info"
                        showIcon
                    />
                 </Spin>
             </div>
          )}
        </>
      )}

      {result && renderResult()}
    </div>
  );
};

export default UploadPage;