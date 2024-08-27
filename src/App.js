import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ChatWindow from './components/ChatWindow';
import InputArea from './components/InputArea';
import ScrollingText from './components/ScrollingText';
import Login from './components/Login';
import ChatHistory from './components/ChatHistory';
import axios from 'axios';

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [showLogin, setShowLogin] = useState(true);
  const [showChatHistory, setShowChatHistory] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);

  const healthTips = [
    "Health Tip- Drink at least 8 glasses of water daily.",
    "Health Tip- Exercise for 30 minutes a day, 5 days a week.",
    "Health Tip- Eat a balanced diet rich in fruits and vegetables.",
    "Health Tip- Get 7-9 hours of sleep each night.",
    "Health Tip- Practice mindfulness or meditation to reduce stress."
  ];

  useEffect(() => {
    checkAuthStatus();
  }, []);

  const checkAuthStatus = async () => {
    try {
      await axios.get('http://localhost:5000/protected', { withCredentials: true });
      setIsAuthenticated(true);
      // Remove the fetchChatHistory call from here
    } catch (error) {
      setIsAuthenticated(false);
    }
  };

  const handleSendMessage = async (text) => {
    setMessages(prev => [...prev, { text, sender: 'user' }]);
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:5000/chat', { question: text }, { withCredentials: true });
      setMessages(prev => [...prev, { text: response.data.answer, sender: 'bot' }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { text: 'Sorry, there was an error processing your request.', sender: 'bot' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewChat = () => setMessages([]);

  const handleLogout = async () => {
    try {
      await axios.get('http://localhost:5000/logout', { withCredentials: true });
      setIsAuthenticated(false);
      setMessages([]);
      console.log('Logged out successfully');
    } catch (error) {
      console.error('Error logging out:', error);
    }
  };

  const handleLogin = () => {
    setIsAuthenticated(true);
    setShowLogin(false);
    // Remove the fetchChatHistory call from here
    setMessages([]); // Clear messages on login
  };

  const handleSignup = () => {
    alert('Signup successful. Please log in.');
    setShowLogin(true);
  };

  const handleViewHistory = async () => {
    try {
      const response = await axios.get('http://localhost:5000/chat_history', { withCredentials: true });
      setChatHistory(response.data);
      setShowChatHistory(true);
    } catch (error) {
      console.error('Error fetching chat history:', error);
    }
  };

  const handleCloseChatHistory = () => {
    setShowChatHistory(false);
  };

  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} onSignup={handleSignup} showLogin={showLogin} setShowLogin={setShowLogin} />;
  }

  return (
    <div className="App">
      <Sidebar 
        onNewChat={handleNewChat} 
        onLogout={handleLogout} 
        onViewHistory={handleViewHistory}
      />
      <div className="main-content">
        <div className="scrolling-text-container">
          <ScrollingText tips={healthTips} />
        </div>
        <Header />
        <ChatWindow messages={messages} isLoading={isLoading} />
        <InputArea onSendMessage={handleSendMessage} isLoading={isLoading} />
      </div>
      {showChatHistory && (
        <ChatHistory history={chatHistory} onClose={handleCloseChatHistory} />
      )}
    </div>
  );
}

export default App;