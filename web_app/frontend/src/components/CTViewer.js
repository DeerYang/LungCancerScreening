import React, { useState, useEffect, useMemo } from 'react';
import { Modal, Slider, Spin, Row, Col, Typography, Tag, Card } from 'antd';
import axios from 'axios';
import { RadarChartOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

const CTViewer = ({ visible, onCancel, result }) => {
  const [currentSlice, setCurrentSlice] = useState(0);
  const [imageUrl, setImageUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // 当结果或可见性变化时，设置初始切片
  useEffect(() => {
    if (visible && result && result.summary.most_concerning_nodule) {
      setCurrentSlice(result.summary.most_concerning_nodule.center_irc.index);
    } else if (visible && result) {
      setCurrentSlice(Math.floor(result.total_slices / 2));
    }
  }, [visible, result]);

  // 找出当前切片上有哪些结节
  const nodulesOnCurrentSlice = useMemo(() => {
    if (!result || !result.nodules) return [];
    return result.nodules.filter(n => n.center_irc.index === currentSlice);
  }, [result, currentSlice]);

  // 当切片或结节列表变化时，获取新的图像URL
  useEffect(() => {
    if (!visible || !result) return;

    setLoading(true);
    setError(null);

    const series_uid = result.filename.replace('.mhd', '');

    // 将当前切片上的结节信息编码到URL查询参数中
    const nodulesParam = encodeURIComponent(JSON.stringify(nodulesOnCurrentSlice));

    const url = `/api/ct-slice/${series_uid}/${currentSlice}?nodules=${nodulesParam}`;

    let currentImageUrl = '';

    // 使用axios获取blob，然后创建对象URL，这样可以更好地处理加载和错误状态
    axios.get(url, { responseType: 'blob' })
      .then(response => {
        currentImageUrl = URL.createObjectURL(response.data);
        setImageUrl(currentImageUrl);
      })
      .catch(err => {
        console.error("Failed to load slice image:", err);
        setError('无法加载此切片图像。');
        setImageUrl('');
      })
      .finally(() => {
        setLoading(false);
      });

    // 清理上一个对象URL
    return () => {
      if (currentImageUrl) {
        URL.revokeObjectURL(currentImageUrl);
      }
    };
  }, [currentSlice, visible, result, nodulesOnCurrentSlice]); // 依赖 nodulesOnCurrentSlice 确保框被重绘

  const getMalignancyColor = (level) => {
      switch (level) {
          case 'high': return 'red';
          case 'medium': return 'orange';
          case 'low': return 'green';
          default: return 'blue';
      }
  };

  return (
    <Modal
      open={visible}
      onCancel={onCancel}
      title="CT 图像查看器"
      width="90vw"
      style={{ top: 20 }}
      footer={null} // 不显示默认的确定/取消按钮
    >
      {result && (
        <Row gutter={24}>
          <Col span={16}>
            <Card>
              <Title level={4}>切片 {currentSlice} / {result.total_slices - 1}</Title>
              <div style={{ position: 'relative', width: '100%', paddingBottom: '100%', backgroundColor: '#000' }}>
                {loading && <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}><Spin size="large" /></div>}
                {error && <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', color: 'red' }}>{error}</div>}
                {imageUrl && !loading && (
                  <img
                    src={imageUrl}
                    alt={`CT Slice ${currentSlice}`}
                    style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
                  />
                )}
              </div>
              <Slider
                value={currentSlice}
                min={0}
                max={result.total_slices - 1}
                onChange={setCurrentSlice}
                style={{ marginTop: 20 }}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card title="当前切片上的结节">
              {nodulesOnCurrentSlice.length > 0 ? (
                nodulesOnCurrentSlice.map(nodule => (
                  <Card key={nodule.id} size="small" style={{ marginBottom: 12 }}>
                    <Text strong><RadarChartOutlined style={{ color: getMalignancyColor(nodule.malignancy_level), marginRight: 8}}/>结节 ID: {nodule.id}</Text>
                    <div>
                      <Text>风险等级: </Text>
                      <Tag color={getMalignancyColor(nodule.malignancy_level)}>{nodule.malignancy_level.toUpperCase()}</Tag>
                    </div>
                    <Text type="secondary">坐标 (r, c): ({nodule.center_irc.row}, {nodule.center_irc.col})</Text>
                  </Card>
                ))
              ) : (
                <Text>当前切片未发现已识别的结节。</Text>
              )}
            </Card>
          </Col>
        </Row>
      )}
    </Modal>
  );
};

export default CTViewer;