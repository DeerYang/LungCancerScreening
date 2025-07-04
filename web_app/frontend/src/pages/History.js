import React, { useState, useEffect } from 'react';
import { 
  Table, 
  Card, 
  Tag, 
  Button, 
  Space, 
  Modal, 
  Descriptions, 
  Progress,
  DatePicker,
  Input,
  Row,
  Col,
  Statistic,
  Alert
} from 'antd';
import { 
  EyeOutlined, 
  DownloadOutlined, 
  SearchOutlined,
  FileImageOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';
import axios from 'axios';
import dayjs from 'dayjs';

const { RangePicker } = DatePicker;
const { Search } = Input;

const HistoryPage = () => {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [searchText, setSearchText] = useState('');
  const [dateRange, setDateRange] = useState(null);

  useEffect(() => {
    fetchPredictions();
  }, []);

  const fetchPredictions = async () => {
    setLoading(true);
    try {
      const response = await axios.get('/api/predictions');
      setPredictions(response.data.predictions);
    } catch (error) {
      console.error('获取历史记录失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleViewDetails = (record) => {
    setSelectedRecord(record);
    setModalVisible(true);
  };

  const getDiagnosisColor = (diagnosis) => {
    switch (diagnosis) {
      case '良性':
        return 'green';
      case '恶性':
        return 'red';
      case '需要进一步检查':
        return 'orange';
      default:
        return 'blue';
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'green';
    if (confidence >= 0.6) return 'orange';
    return 'red';
  };

  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 80,
    },
    {
      title: '文件名',
      dataIndex: 'filename',
      key: 'filename',
      render: (text) => (
        <Space>
          <FileImageOutlined />
          {text}
        </Space>
      ),
    },
    {
      title: '诊断结果',
      dataIndex: 'diagnosis',
      key: 'diagnosis',
      render: (diagnosis) => (
        <Tag color={getDiagnosisColor(diagnosis)}>
          {diagnosis}
        </Tag>
      ),
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence) => (
        <Progress 
          percent={confidence * 100} 
          size="small" 
          status="active"
          strokeColor={getConfidenceColor(confidence)}
        />
      ),
    },
    {
      title: '分析时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp) => dayjs(timestamp).format('YYYY-MM-DD HH:mm:ss'),
      sorter: (a, b) => dayjs(a.timestamp).unix() - dayjs(b.timestamp).unix(),
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space size="middle">
          <Button 
            type="link" 
            icon={<EyeOutlined />}
            onClick={() => handleViewDetails(record)}
          >
            查看详情
          </Button>
          <Button 
            type="link" 
            icon={<DownloadOutlined />}
          >
            下载报告
          </Button>
        </Space>
      ),
    },
  ];

  const filteredPredictions = predictions.filter(prediction => {
    const matchesSearch = prediction.filename.toLowerCase().includes(searchText.toLowerCase()) ||
                         prediction.diagnosis.toLowerCase().includes(searchText.toLowerCase());
    
    const matchesDate = !dateRange || (
      dayjs(prediction.timestamp).isAfter(dateRange[0]) && 
      dayjs(prediction.timestamp).isBefore(dateRange[1])
    );
    
    return matchesSearch && matchesDate;
  });

  const statistics = {
    total: predictions.length,
    benign: predictions.filter(p => p.diagnosis === '良性').length,
    malignant: predictions.filter(p => p.diagnosis === '恶性').length,
    averageConfidence: predictions.length > 0 
      ? (predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length * 100).toFixed(1)
      : 0
  };

  return (
    <div>
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Alert
            message="历史记录"
            description="查看所有CT图像分析的历史记录，包括诊断结果、置信度和详细分析报告。"
            type="info"
            showIcon
          />
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="总分析数"
              value={statistics.total}
              prefix={<FileImageOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="良性诊断"
              value={statistics.benign}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="恶性诊断"
              value={statistics.malignant}
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="平均置信度"
              value={statistics.averageConfidence}
              suffix="%"
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col span={24}>
          <Card title="筛选和搜索">
            <Row gutter={[16, 16]}>
              <Col xs={24} md={12}>
                <Search
                  placeholder="搜索文件名或诊断结果"
                  allowClear
                  enterButton={<SearchOutlined />}
                  value={searchText}
                  onChange={(e) => setSearchText(e.target.value)}
                />
              </Col>
              <Col xs={24} md={12}>
                <RangePicker
                  placeholder={['开始日期', '结束日期']}
                  value={dateRange}
                  onChange={setDateRange}
                  style={{ width: '100%' }}
                />
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col span={24}>
          <Card title="分析历史记录">
            <Table
              columns={columns}
              dataSource={filteredPredictions}
              rowKey="id"
              loading={loading}
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) => 
                  `第 ${range[0]}-${range[1]} 条，共 ${total} 条记录`,
              }}
            />
          </Card>
        </Col>
      </Row>

      <Modal
        title="详细分析报告"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setModalVisible(false)}>
            关闭
          </Button>,
          <Button key="download" type="primary" icon={<DownloadOutlined />}>
            下载报告
          </Button>,
        ]}
        width={800}
      >
        {selectedRecord && (
          <Descriptions bordered column={2}>
            <Descriptions.Item label="记录ID" span={1}>
              {selectedRecord.id}
            </Descriptions.Item>
            <Descriptions.Item label="文件名" span={1}>
              {selectedRecord.filename}
            </Descriptions.Item>
            <Descriptions.Item label="诊断结果" span={1}>
              <Tag color={getDiagnosisColor(selectedRecord.diagnosis)}>
                {selectedRecord.diagnosis}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="置信度" span={1}>
              <Progress 
                percent={selectedRecord.confidence * 100} 
                size="small"
                strokeColor={getConfidenceColor(selectedRecord.confidence)}
              />
            </Descriptions.Item>
            <Descriptions.Item label="分析时间" span={2}>
              {dayjs(selectedRecord.timestamp).format('YYYY-MM-DD HH:mm:ss')}
            </Descriptions.Item>
            <Descriptions.Item label="医疗建议" span={2}>
              <ul>
                <li>建议进行定期随访</li>
                <li>考虑进行活检确认</li>
                <li>建议3个月后复查</li>
              </ul>
            </Descriptions.Item>
          </Descriptions>
        )}
      </Modal>
    </div>
  );
};

export default HistoryPage; 