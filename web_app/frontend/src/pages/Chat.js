import React, { useState, useRef, useEffect } from 'react';
import { 
  Card, 
  Input, 
  Button, 
  List, 
  Avatar, 
  Typography, 
  Space,
  Alert,
  Spin,
  Divider,
  Tag,
  Row,
  Col
} from 'antd';
import { 
  SendOutlined, 
  RobotOutlined, 
  UserOutlined,
  MessageOutlined,
  QuestionCircleOutlined
} from '@ant-design/icons';
import axios from 'axios';

const { TextArea } = Input;
const { Text } = Typography;

const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 检查连接状态
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await axios.get('/api/health');
        setConnected(response.status === 200);
      } catch (error) {
        setConnected(false);
      }
    };
    
    checkConnection();
    // 每30秒检查一次连接状态
    const interval = setInterval(checkConnection, 30000);
    
    return () => clearInterval(interval);
  }, []);

  // 添加欢迎消息
  useEffect(() => {
    setMessages([
      {
        id: 1,
        type: 'ai',
        content: '您好！我是AI医疗助手，专门帮助医生解答关于肺肿瘤诊断和AI辅助诊断系统的问题。请问有什么可以帮助您的吗？',
        timestamp: new Date().toLocaleString()
      }
    ]);
  }, []);

  const handleSend = async () => {
    if (!inputValue.trim() || loading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue,
      timestamp: new Date().toLocaleString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      const response = await axios.post('/api/chat', {
        message: inputValue
      });

      const aiMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: response.data.response,
        timestamp: new Date().toLocaleString()
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('聊天失败:', error);
      const errorMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: '抱歉，我现在无法回答您的问题，请稍后再试。',
        timestamp: new Date().toLocaleString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const quickQuestions = [
    "如何使用这个系统进行CT图像分析？",
    "三个AI模型分别有什么作用？",
    "如何解读诊断结果？",
    "系统的准确率如何？",
    "支持哪些图像格式？"
  ];

  const renderMessage = (message) => {
    const isAI = message.type === 'ai';
    
    return (
      <List.Item style={{ 
        padding: '12px 0',
        borderBottom: 'none'
      }}>
        <div style={{ 
          display: 'flex', 
          alignItems: 'flex-start',
          flexDirection: isAI ? 'row' : 'row-reverse',
          width: '100%'
        }}>
          <Avatar 
            icon={isAI ? <RobotOutlined /> : <UserOutlined />}
            style={{ 
              backgroundColor: isAI ? '#1890ff' : '#52c41a',
              margin: isAI ? '0 12px 0 0' : '0 0 0 12px'
            }}
          />
          <div style={{
            maxWidth: '70%',
            backgroundColor: isAI ? '#f0f2f5' : '#e6f7ff',
            padding: '12px 16px',
            borderRadius: '12px',
            position: 'relative'
          }}>
            <div style={{ marginBottom: 4 }}>
              <Text strong>{isAI ? 'AI助手' : '您'}</Text>
              <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
                {message.timestamp}
              </Text>
            </div>
            <div style={{ whiteSpace: 'pre-wrap' }}>
              {message.content}
            </div>
          </div>
        </div>
      </List.Item>
    );
  };

  return (
    <div>
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Alert
            message="AI医疗助手"
            description="基于通义千问大模型，专门为医生提供肺肿瘤诊断相关的专业咨询和系统使用指导。"
            type="info"
            showIcon
            icon={<MessageOutlined />}
          />
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={16}>
          <Card 
            title="AI医疗助手" 
            extra={
              <Tag color={connected ? 'green' : 'red'}>
                {connected ? '已连接' : '未连接'}
              </Tag>
            }
            style={{ height: 600 }}
            bodyStyle={{ padding: 0, height: 'calc(100% - 57px)' }}
          >
            <div style={{ 
              height: '100%', 
              display: 'flex', 
              flexDirection: 'column',
              padding: '16px'
            }}>
              <div style={{ 
                flex: 1, 
                overflowY: 'auto',
                marginBottom: 16
              }}>
                <List
                  dataSource={messages}
                  renderItem={renderMessage}
                  locale={{ emptyText: '暂无消息' }}
                />
                {loading && (
                  <div style={{ textAlign: 'center', padding: '20px' }}>
                    <Spin />
                    <div style={{ marginTop: 8 }}>AI正在思考中...</div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
              
              <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: 16 }}>
                <Space.Compact style={{ width: '100%' }}>
                  <TextArea
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="请输入您的问题..."
                    autoSize={{ minRows: 2, maxRows: 4 }}
                    disabled={loading}
                  />
                  <Button 
                    type="primary" 
                    icon={<SendOutlined />}
                    onClick={handleSend}
                    disabled={!inputValue.trim() || loading}
                    style={{ height: 'auto' }}
                  >
                    发送
                  </Button>
                </Space.Compact>
              </div>
            </div>
          </Card>
        </Col>

        <Col xs={24} lg={8}>
          <Card title="快速问题" icon={<QuestionCircleOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              {quickQuestions.map((question, index) => (
                <Button
                  key={index}
                  type="dashed"
                  block
                  onClick={() => setInputValue(question)}
                  style={{ textAlign: 'left', height: 'auto', whiteSpace: 'normal' }}
                >
                  {question}
                </Button>
              ))}
            </Space>
          </Card>

          <Card title="使用提示" style={{ marginTop: 16 }}>
            <ul style={{ paddingLeft: 16 }}>
              <li>您可以询问关于系统操作的问题</li>
              <li>可以咨询肺肿瘤诊断相关知识</li>
              <li>可以了解AI模型的原理和准确率</li>
              <li>可以获取医疗建议和指导</li>
            </ul>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default ChatPage; 