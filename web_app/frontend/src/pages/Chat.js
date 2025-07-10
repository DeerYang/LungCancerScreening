import React, { useState, useRef, useEffect } from 'react';
import { useSearchParams, Link } from 'react-router-dom'; // 引入 useSearchParams 和 Link
import {
  Card, Input, Button, List, Avatar, Typography,
  Space, Alert, Spin, Divider, Tag, Row, Col
} from 'antd';
import {
  SendOutlined, RobotOutlined, UserOutlined, MessageOutlined,
  QuestionCircleOutlined, InfoCircleOutlined, CloseCircleOutlined // 新增图标
} from '@ant-design/icons';
import axios from 'axios';

const { TextArea } = Input;
const { Text, Title } = Typography;

const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const messagesEndRef = useRef(null);

  // --- 核心修改：从URL获取当前讨论的诊断ID ---
  const [searchParams, setSearchParams] = useSearchParams();
  const predictionId = searchParams.get('id');
  const predictionFilename = searchParams.get('filename');

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
      } catch (error) { setConnected(false); }
    };
    checkConnection();
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  // 添加欢迎消息，根据是否有上下文调整内容
  useEffect(() => {
    const welcomeMessage = predictionId
      ? `您好！我是AI医疗助手。我们现在正在讨论关于文件 **${predictionFilename || `ID: ${predictionId}`}** 的分析报告。请问您有什么具体问题吗？`
      : '您好！我是AI医疗助手，专门帮助医生解答关于肺肿瘤诊断和AI辅助诊断系统的问题。请问有什么可以帮助您的吗？';

    setMessages([
      {
        id: 1,
        type: 'ai',
        content: welcomeMessage,
        timestamp: new Date().toLocaleString()
      }
    ]);
  }, [predictionId, predictionFilename]);

  const handleSend = async () => {
    if (!inputValue.trim() || loading) return;

    const userMessage = {
      id: Date.now(), type: 'user', content: inputValue,
      timestamp: new Date().toLocaleString()
    };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      // --- 核心修改：发送请求时带上 predictionId ---
      const response = await axios.post('/api/chat', {
        message: inputValue,
        predictionId: predictionId, // 将ID传给后端
      });

      const aiMessage = {
        id: Date.now() + 1, type: 'ai',
        content: response.data.response,
        timestamp: new Date().toLocaleString()
      };
      setMessages(prev => [...prev, aiMessage]);

    } catch (error) {
      console.error('聊天失败:', error);
      const errorMessage = {
        id: Date.now() + 1, type: 'ai',
        content: `抱歉，我现在无法回答您的问题。错误: ${error.response?.data?.error || error.message}`,
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
    predictionId ? `总结一下这份报告的主要发现。` : "如何使用这个系统？",
    predictionId ? `根据这份报告，下一步的临床建议是什么？` : "三个AI模型分别有什么作用？",
    predictionId ? `这个报告中“高风险”具体意味着什么？` : "如何解读诊断结果？",
    "系统的准确率如何？",
  ];

  const renderMessageContent = (content) => {
      // 简单的Markdown替换，将**text**替换为<strong>text</strong>
      const htmlContent = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      return <div dangerouslySetInnerHTML={{ __html: htmlContent }} />;
  };

  const renderMessage = (message) => {
    const isAI = message.type === 'ai';
    return (
      <List.Item style={{ padding: '12px 0', borderBottom: 'none' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', flexDirection: isAI ? 'row' : 'row-reverse', width: '100%' }}>
          <Avatar icon={isAI ? <RobotOutlined /> : <UserOutlined />} style={{ backgroundColor: isAI ? '#1890ff' : '#52c41a', margin: isAI ? '0 12px 0 0' : '0 0 0 12px' }}/>
          <div style={{ maxWidth: '70%', backgroundColor: isAI ? '#f0f2f5' : '#e6f7ff', padding: '12px 16px', borderRadius: '12px', position: 'relative' }}>
            <div style={{ marginBottom: 4 }}>
              <Text strong>{isAI ? 'AI助手' : '您'}</Text>
              <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>{message.timestamp}</Text>
            </div>
            <div style={{ whiteSpace: 'pre-wrap' }}>
              {renderMessageContent(message.content)}
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
            {predictionId && (
              <Alert
                message={
                  <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                    <span>当前正在讨论报告: <strong>{predictionFilename || `ID: ${predictionId}`}</strong></span>
                    <Link to="/chat">
                        <Button type="link" icon={<CloseCircleOutlined />} size="small">结束本次讨论</Button>
                    </Link>
                  </div>
                }
                type="info"
                showIcon
                icon={<InfoCircleOutlined />}
                style={{marginBottom: 16}}
              />
            )}
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={16}>
          <Card title={<Title level={4} style={{margin:0}}>AI医疗助手</Title>} extra={<Tag color={connected ? 'green' : 'red'}>{connected ? '已连接' : '未连接'}</Tag>} style={{ height: 600 }} bodyStyle={{ padding: 0, height: 'calc(100% - 57px)' }}>
            <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: '16px' }}>
              <div style={{ flex: 1, overflowY: 'auto', marginBottom: 16 }}>
                <List dataSource={messages} renderItem={renderMessage} locale={{ emptyText: '暂无消息' }} />
                {loading && (
                  <div style={{ textAlign: 'center', padding: '20px' }}><Spin /><div style={{ marginTop: 8 }}>AI正在思考中...</div></div>
                )}
                <div ref={messagesEndRef} />
              </div>
              <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: 16 }}>
                <Space.Compact style={{ width: '100%' }}>
                  <TextArea value={inputValue} onChange={(e) => setInputValue(e.target.value)} onKeyPress={handleKeyPress} placeholder="请输入您的问题..." autoSize={{ minRows: 2, maxRows: 4 }} disabled={loading} />
                  <Button type="primary" icon={<SendOutlined />} onClick={handleSend} disabled={!inputValue.trim() || loading} style={{ height: 'auto' }}>发送</Button>
                </Space.Compact>
              </div>
            </div>
          </Card>
        </Col>

        <Col xs={24} lg={8}>
          <Card title="快速提问" icon={<QuestionCircleOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              {quickQuestions.map((question, index) => (
                <Button key={index} type="dashed" block onClick={() => setInputValue(question)} style={{ textAlign: 'left', height: 'auto', whiteSpace: 'normal' }}>{question}</Button>
              ))}
            </Space>
          </Card>
          <Card title="使用提示" style={{ marginTop: 16 }}>
            <ul style={{ paddingLeft: 16 }}>
              <li>您可以询问关于系统操作的问题。</li>
              <li>可以咨询肺肿瘤诊断相关知识。</li>
              <li>从“历史记录”页面进入，可以针对特定报告提问。</li>
              <li>AI的建议仅供参考，不构成最终诊断。</li>
            </ul>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default ChatPage;