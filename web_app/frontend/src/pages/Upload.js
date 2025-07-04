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
  Image,
  Divider,
  message,
  List,
  Space,
  Typography
} from 'antd';
import { 
  InboxOutlined, 
  FileImageOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  UploadOutlined,
  FileTextOutlined,
  CloseCircleOutlined
} from '@ant-design/icons';
import axios from 'axios';

const { Dragger } = Upload;
const { Title, Text } = Typography;

const UploadPage = () => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [mhdFile, setMhdFile] = useState(null);
  const [rawFile, setRawFile] = useState(null);

  const handleMhdFileChange = (info) => {
    const file = info.file;
    if (file.status === 'removed') {
      setMhdFile(null);
      return;
    }
    
    if (file.type !== '' && !file.name.endsWith('.mhd')) {
      message.error('请选择.mhd文件');
      return;
    }
    
    setMhdFile(file);
  };

  const handleRawFileChange = (info) => {
    const file = info.file;
    if (file.status === 'removed') {
      setRawFile(null);
      return;
    }
    
    if (file.type !== '' && !file.name.endsWith('.raw')) {
      message.error('请选择.raw文件');
      return;
    }
    
    setRawFile(file);
  };

  const resetUpload = () => {
    // 重置所有状态
    setMhdFile(null);
    setRawFile(null);
    setResult(null);
    setUploading(false);
    setUploadProgress(0);
    message.success('已重置，请重新选择文件');
  };

  const handleUpload = async () => {
    if (!mhdFile || !rawFile) {
      message.error('请选择.mhd和.raw文件对');
      return;
    }

    // 检查文件名是否匹配
    const mhdName = mhdFile.name.replace('.mhd', '');
    const rawName = rawFile.name.replace('.raw', '');
    
    if (mhdName !== rawName) {
      message.error('文件名不匹配，请确保.mhd和.raw文件来自同一个CT扫描');
      return;
    }

    setUploading(true);
    setUploadProgress(0);
    setResult(null);

    const formData = new FormData();
    // 使用文件对象本身，如果originFileObj不存在则使用文件对象
    const mhdFileObj = mhdFile.originFileObj || mhdFile;
    const rawFileObj = rawFile.originFileObj || rawFile;
    
    // 添加调试信息
    console.log('MHD文件对象:', mhdFileObj);
    console.log('RAW文件对象:', rawFileObj);
    console.log('MHD文件名:', mhdFileObj.name);
    console.log('RAW文件名:', rawFileObj.name);
    
    formData.append('mhd_file', mhdFileObj);
    formData.append('raw_file', rawFileObj);

    try {
      // 模拟上传进度
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + 10;
        });
      }, 200);

      const response = await axios.post('http://localhost:5000/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000, // 5分钟超时
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (response.data.success) {
        setResult(response.data);
        message.success('CT文件上传并分析完成！');
      } else {
        message.error(response.data.error || '分析失败');
      }
    } catch (error) {
      console.error('上传失败:', error);
      message.error(error.response?.data?.error || '上传失败，请重试');
    } finally {
      setUploading(false);
      setUploadProgress(0);
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

  const getMalignancyText = (level) => {
    switch (level) {
      case 'high': return '高风险';
      case 'medium': return '中等风险';
      case 'low': return '低风险';
      default: return '未知';
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
          extra={[
            <Button type="primary" key="retry" onClick={resetUpload}>
              重新上传
            </Button>,
          ]}
        />
      );
    }

    return (
      <div>
        <Row gutter={[16, 16]} style={{ marginBottom: '16px' }}>
          <Col span={24}>
            <Title level={3} style={{ margin: 0 }}>
              📊 分析结果
            </Title>
          </Col>
        </Row>
        
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <Card title="分析结果概览">
              <Row gutter={[16, 16]}>
                <Col span={8}>
                  <Card size="small">
                    <div style={{ textAlign: 'center' }}>
                      <FileImageOutlined style={{ fontSize: 24, color: '#1890ff' }} />
                      <div style={{ marginTop: 8 }}>
                        <div>候选区域</div>
                        <div style={{ fontSize: 20, fontWeight: 'bold' }}>
                          {result.total_candidates}
                        </div>
                      </div>
                    </div>
                  </Card>
                </Col>
                <Col span={8}>
                  <Card size="small">
                    <div style={{ textAlign: 'center' }}>
                      <CheckCircleOutlined style={{ fontSize: 24, color: '#52c41a' }} />
                      <div style={{ marginTop: 8 }}>
                        <div>检测到结节</div>
                        <div style={{ fontSize: 20, fontWeight: 'bold' }}>
                          {result.nodules_found}
                        </div>
                      </div>
                    </div>
                  </Card>
                </Col>
                <Col span={8}>
                  <Card size="small">
                    <div style={{ textAlign: 'center' }}>
                      <ExclamationCircleOutlined style={{ fontSize: 24, color: '#cf1322' }} />
                      <div style={{ marginTop: 8 }}>
                        <div>恶性概率</div>
                        <div style={{ fontSize: 20, fontWeight: 'bold' }}>
                          {(result.nodules && result.nodules.length > 0 ? result.nodules[0].malignancy_probability * 100 : 0).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  </Card>
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>

        <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
          <Col span={24}>
            <Card title="详细分析结果">
              <Descriptions bordered column={2}>
                <Descriptions.Item label="文件名">
                  {result.filename}
                </Descriptions.Item>
                <Descriptions.Item label="分析时间">
                  {new Date(result.timestamp).toLocaleString()}
                </Descriptions.Item>
                <Descriptions.Item label="分割置信度">
                  <Progress 
                    percent={result.segmentation_confidence * 100} 
                    size="small"
                    status="active"
                  />
                </Descriptions.Item>
                <Descriptions.Item label="是否为结节">
                  <Tag color={result.nodules && result.nodules.length > 0 ? 'green' : 'red'}>
                    {result.nodules && result.nodules.length > 0 ? '是' : '否'}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="诊断结果">
                  <Tag color={getMalignancyColor(result.nodules && result.nodules.length > 0 ? result.nodules[0].malignancy_level : 'unknown')}>
                    {result.nodules && result.nodules.length > 0 ? getMalignancyText(result.nodules[0].malignancy_level) : '未知'}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="诊断置信度">
                  <Tag color={result.nodules && result.nodules.length > 0 ? getMalignancyColor(result.nodules[0].malignancy_level) : 'blue'}>
                    {(result.nodules && result.nodules.length > 0 ? (result.nodules[0].malignancy_probability * 100).toFixed(1) : '0.0')}%
                  </Tag>
                </Descriptions.Item>
              </Descriptions>
            </Card>
          </Col>
        </Row>

        <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
          <Col span={24}>
            <Card title="医疗建议">
              <List
                dataSource={result.recommendations}
                renderItem={(rec, index) => (
                  <List.Item>
                    <List.Item.Meta
                      title={`建议 ${index + 1}`}
                      description={
                        <Space direction="vertical" size="small">
                          <div>
                            <Text>{rec}</Text>
                          </div>
                        </Space>
                      }
                    />
                  </List.Item>
                )}
              />
            </Card>
          </Col>
        </Row>

        <Row style={{ marginTop: 16 }}>
          <Col span={24}>
            <Button type="primary" onClick={resetUpload}>
              分析新图像
            </Button>
          </Col>
        </Row>
      </div>
    );
  };

  return (
    <div style={{ padding: '24px', maxWidth: '1200px', margin: '0 auto' }}>
      <Title level={2}>CT文件上传与分析</Title>
      
      {/* 上传区域 - 只在没有结果时显示 */}
      {!result && (
        <>
          <Alert
            message="文件格式要求"
            description="请上传DICOM格式的CT扫描文件对：.mhd文件（元数据）和.raw文件（体素数据）。这两个文件必须来自同一个CT扫描，文件名前缀必须相同。"
            type="info"
            showIcon
            style={{ marginBottom: '24px' }}
          />

          <Card title="选择CT文件" style={{ marginBottom: '24px' }}>
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              <div>
                <Text strong>1. 选择.mhd文件（元数据）：</Text>
                <Upload
                  accept=".mhd"
                  beforeUpload={() => false}
                  onChange={handleMhdFileChange}
                  showUploadList={true}
                  maxCount={1}
                >
                  <Button icon={<UploadOutlined />}>选择.mhd文件</Button>
                </Upload>
              </div>

              <div>
                <Text strong>2. 选择.raw文件（体素数据）：</Text>
                <Upload
                  accept=".raw"
                  beforeUpload={() => false}
                  onChange={handleRawFileChange}
                  showUploadList={true}
                  maxCount={1}
                >
                  <Button icon={<UploadOutlined />}>选择.raw文件</Button>
                </Upload>
              </div>

              <Button
                type="primary"
                size="large"
                onClick={handleUpload}
                loading={uploading}
                disabled={!mhdFile || !rawFile}
                icon={<FileTextOutlined />}
                style={{ marginTop: '16px' }}
              >
                开始分析
              </Button>
            </Space>
          </Card>

          {uploading && (
            <Card title="分析进度" style={{ marginBottom: '24px' }}>
              <Progress percent={uploadProgress} status="active" />
              <Text type="secondary">
                正在处理CT数据，包括分割、候选区域提取和分类分析...
              </Text>
            </Card>
          )}
        </>
      )}

      {/* 结果展示区域 - 只在有结果时显示 */}
      {result && renderResult()}
    </div>
  );
};

export default UploadPage; 