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
  Input,
  Row,
  Col,
  Statistic,
  Alert,
  Empty,
  List,
  Avatar
} from 'antd';
import {
  EyeOutlined,
  DownloadOutlined,
  SearchOutlined,
  FileImageOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  WarningOutlined,
  HeartOutlined,
  RadarChartOutlined,
  CloseCircleOutlined
} from '@ant-design/icons';
import axios from 'axios';
import dayjs from 'dayjs';

const { Search } = Input;

const HistoryPage = () => {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [searchText, setSearchText] = useState('');

  useEffect(() => {
    fetchPredictions();
  }, []);

  const fetchPredictions = async () => {
    setLoading(true);
    try {
      const response = await axios.get('/api/predictions');
      // 后端返回的是最新的在前面，这里我们保持这个顺序
      setPredictions(response.data.predictions);
    } catch (error) {
      console.error('获取历史记录失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleViewDetails = async (record) => {
    // 假设历史记录列表只包含摘要，如果需要完整数据，可以再从后端获取
    // 在我们当前的设计中，历史记录应该已经包含了所有需要的信息
    setSelectedRecord(record);
    setModalVisible(true);
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
        return { color: 'grey', text: finding || '未知', icon: <CloseCircleOutlined /> };
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
    },
    {
      title: '总体诊断',
      dataIndex: 'diagnosis',
      key: 'diagnosis',
      render: (diagnosis) => {
        const info = getOverallFindingInfo(diagnosis);
        return <Tag color={info.color}>{info.text}</Tag>;
      },
    },
    {
      title: '最高恶性概率',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence) => (
        <Progress
          percent={(confidence || 0) * 100}
          size="small"
          status="exception"
          format={percent => `${percent.toFixed(2)}%`}
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
        </Space>
      ),
    },
  ];

  const filteredPredictions = predictions.filter(p =>
    p.filename.toLowerCase().includes(searchText.toLowerCase())
  );

  const statistics = {
    total: predictions.length,
    high_risk: predictions.filter(p => p.diagnosis === 'high_risk').length,
    low_risk: predictions.filter(p => p.diagnosis === 'low_risk').length,
    no_nodules: predictions.filter(p => p.diagnosis === 'no_nodules_found').length
  };

  return (
    <div>
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Alert
            message="诊断历史记录"
            description="这里记录了所有已完成的CT图像分析。您可以搜索并查看每一次分析的详细报告。"
            type="info"
            showIcon
          />
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic title="总分析数" value={statistics.total} prefix={<FileImageOutlined />} />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic title="高风险诊断" value={statistics.high_risk} prefix={<ExclamationCircleOutlined />} valueStyle={{ color: '#cf1322' }}/>
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic title="低风险诊断" value={statistics.low_risk} prefix={<CheckCircleOutlined />} valueStyle={{ color: '#52c41a' }} />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic title="未发现结节" value={statistics.no_nodules} prefix={<HeartOutlined />} valueStyle={{ color: '#1890ff' }} />
          </Card>
        </Col>
      </Row>

      <Card style={{ marginTop: 16 }}>
        <Search
          placeholder="按文件名搜索..."
          allowClear
          enterButton
          onSearch={setSearchText}
        />
      </Card>

      <Card title="分析历史列表" style={{ marginTop: 16 }}>
        <Table
          columns={columns}
          dataSource={filteredPredictions}
          rowKey="id"
          loading={loading}
          pagination={{ pageSize: 10 }}
        />
      </Card>

      <Modal
        title="详细分析报告"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setModalVisible(false)}>关闭</Button>,
          <Button key="download" type="primary" icon={<DownloadOutlined />}>下载PDF报告</Button>,
        ]}
        width={800}
      >
        {selectedRecord && (
          <Row gutter={[16, 16]}>
            <Col span={12}>
              <Card title="分析摘要" size="small">
                <Descriptions bordered column={1} size="small">
                  <Descriptions.Item label="总体发现">
                    <Tag color={getOverallFindingInfo(selectedRecord.diagnosis).color}>{getOverallFindingInfo(selectedRecord.diagnosis).text}</Tag>
                  </Descriptions.Item>
                   <Descriptions.Item label="检测到结节">{selectedRecord.summary.nodule_count}</Descriptions.Item>
                   <Descriptions.Item label="最高恶性概率">
                      <Progress percent={(selectedRecord.confidence || 0) * 100} size="small" status="exception" format={p => `${p.toFixed(2)}%`} />
                   </Descriptions.Item>
                </Descriptions>
              </Card>
            </Col>
            <Col span={12}>
                <Card title="最可疑结节" size="small">
                    {selectedRecord.summary.most_concerning_nodule ? (
                        <Descriptions bordered column={1} size="small">
                             <Descriptions.Item label="结节ID">{selectedRecord.summary.most_concerning_nodule.id}</Descriptions.Item>
                             <Descriptions.Item label="恶性风险">{selectedRecord.summary.most_concerning_nodule.malignancy_level.toUpperCase()}</Descriptions.Item>
                             <Descriptions.Item label="恶性概率">{(selectedRecord.summary.most_concerning_nodule.malignancy_probability * 100).toFixed(4)}%</Descriptions.Item>
                        </Descriptions>
                    ) : <Empty description="无" />}
                </Card>
            </Col>
             <Col span={24}>
                <Card title={`结节详细列表 (${selectedRecord.summary.nodule_count}个)`} size="small">
                    <List
                        dataSource={selectedRecord.nodules}
                        renderItem={item => (
                        <List.Item>
                            <List.Item.Meta
                            avatar={<Avatar style={{ backgroundColor: getMalignancyColor(item.malignancy_level) }} icon={<RadarChartOutlined />} />}
                            title={`结节 #${item.id}`}
                            description={`恶性概率: ${item.malignancy_probability !== null ? `${(item.malignancy_probability * 100).toFixed(4)}%` : 'N/A'}`}
                            />
                            <Tag color={getMalignancyColor(item.malignancy_level)}>{item.malignancy_level.toUpperCase()}</Tag>
                        </List.Item>
                        )}
                        locale={{ emptyText: <Empty description="未发现结节" /> }}
                    />
                </Card>
            </Col>
          </Row>
        )}
      </Modal>
    </div>
  );
};

export default HistoryPage;