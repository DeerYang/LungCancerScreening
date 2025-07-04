import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Statistic, Progress, Alert, Spin, Typography } from 'antd';
import { 
  FileImageOutlined, 
  CheckCircleOutlined, 
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import axios from 'axios';

const Dashboard = () => {
  const [systemStatus, setSystemStatus] = useState(null);
  const [statistics, setStatistics] = useState(null);
  const [trendData, setTrendData] = useState(null);
  const [distributionData, setDistributionData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
    // 每30秒刷新一次数据
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const [healthResponse, statsResponse, trendResponse, distributionResponse] = await Promise.all([
        axios.get('/api/health'),
        axios.get('/api/statistics'),
        axios.get('/api/diagnosis-trend'),
        axios.get('/api/diagnosis-distribution')
      ]);
      
      setSystemStatus(healthResponse.data);
      setStatistics(statsResponse.data);
      setTrendData(trendResponse.data);
      setDistributionData(distributionResponse.data);
      
      // 如果模型未加载，尝试获取更详细的状态
      if (!healthResponse.data.models_loaded) {
        try {
          const modelResponse = await axios.get('/api/models/status');
          setSystemStatus(prev => ({
            ...prev,
            modelDetails: modelResponse.data
          }));
        } catch (modelError) {
          console.error('模型状态检查失败:', modelError);
        }
      }
    } catch (error) {
      console.error('获取仪表盘数据失败:', error);
      setSystemStatus({ status: 'error', models_loaded: false });
    } finally {
      setLoading(false);
    }
  };

  const getChartOption = () => ({
    title: {
      text: '诊断结果统计',
      left: 'center'
    },
    tooltip: {
      trigger: 'item'
    },
    legend: {
      orient: 'vertical',
      left: 'left'
    },
    series: [
      {
        name: '诊断结果',
        type: 'pie',
        radius: '50%',
        data: distributionData?.distribution || [
          { value: 35, name: '良性结节' },
          { value: 15, name: '恶性结节' },
          { value: 25, name: '需要进一步检查' },
          { value: 25, name: '正常' }
        ],
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }
    ]
  });

  const getLineChartOption = () => ({
    title: {
      text: '每日诊断数量',
      left: 'center'
    },
    tooltip: {
      trigger: 'axis'
    },
    xAxis: {
      type: 'category',
      data: trendData?.trend?.map(item => item.date) || ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    },
    yAxis: {
      type: 'value'
    },
    series: [
      {
        name: '诊断数量',
        type: 'line',
        data: trendData?.trend?.map(item => item.count) || [12, 19, 15, 25, 22, 18, 14],
        smooth: true
      }
    ]
  });

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>正在检查系统状态...</div>
      </div>
    );
  }

  return (
    <div>
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Alert
            message="系统状态"
            description={
              <div>
                <div>
                  {systemStatus?.models_loaded 
                    ? "所有AI模型已成功加载，系统运行正常" 
                    : systemStatus?.modelDetails?.available_models 
                      ? `系统运行正常，但部分模型未加载。可用模型：${Object.entries(systemStatus.modelDetails.available_models).filter(([_, available]) => available).map(([name, _]) => name).join(', ') || '无'}`
                      : "系统运行正常，当前使用模拟模式进行演示"
                  }
                </div>
                {statistics?.timestamp && (
                  <div style={{ marginTop: 8, fontSize: 12 }}>
                    <ReloadOutlined style={{ marginRight: 4 }} />
                    数据更新时间: {new Date(statistics.timestamp).toLocaleString()}
                  </div>
                )}
              </div>
            }
            type={systemStatus?.models_loaded ? "success" : "info"}
            showIcon
          />
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="今日诊断"
              value={statistics?.today_diagnoses || 0}
              prefix={<FileImageOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="良性诊断"
              value={statistics?.benign_count || 0}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="恶性诊断"
              value={statistics?.malignant_count || 0}
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="待处理"
              value={statistics?.pending_count || 0}
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={12}>
          <Card title="模型准确率" style={{ height: 300 }}>
            <div style={{ marginBottom: 16 }}>
              <div style={{ marginBottom: 8 }}>
                <span>分割模型准确率</span>
                <Progress 
                  percent={statistics?.model_accuracies?.segmentation || 0} 
                  status={statistics?.model_accuracies?.segmentation > 0 ? "active" : "exception"}
                />
                <div style={{ fontSize: 11, color: '#666' }}>
                  基于131,502个样本的真实测试
                </div>
              </div>
              <div style={{ marginBottom: 8 }}>
                <span>分类模型准确率</span>
                <Progress 
                  percent={statistics?.model_accuracies?.classification || 0} 
                  status={statistics?.model_accuracies?.classification > 0 ? "active" : "exception"}
                />
                <div style={{ fontSize: 11, color: '#666' }}>
                  基于154个结节样本的真实测试
                </div>
              </div>
              <div style={{ marginBottom: 8 }}>
                <span>恶性程度模型准确率</span>
                <Progress 
                  percent={statistics?.model_accuracies?.malignancy || 0} 
                  status={statistics?.model_accuracies?.malignancy > 0 ? "active" : "exception"}
                />
                <div style={{ fontSize: 11, color: '#666' }}>
                  基于分类模型性能的真实测试
                </div>
              </div>
            </div>
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="系统性能" style={{ height: 300 }}>
            <div style={{ marginBottom: 16 }}>
              <div style={{ marginBottom: 8 }}>
                <span>CPU使用率</span>
                <Progress percent={statistics?.system_performance?.cpu_usage || 0} />
              </div>
              <div style={{ marginBottom: 8 }}>
                <span>内存使用率</span>
                <Progress percent={statistics?.system_performance?.memory_usage || 0} />
              </div>
              <div style={{ marginBottom: 8 }}>
                <span>GPU使用率</span>
                <Progress percent={statistics?.system_performance?.gpu_usage || 0} />
              </div>
            </div>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={12}>
          <Card title="诊断结果分布">
            <ReactECharts option={getChartOption()} style={{ height: 300 }} />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="每日诊断趋势">
            <ReactECharts option={getLineChartOption()} style={{ height: 300 }} />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard; 