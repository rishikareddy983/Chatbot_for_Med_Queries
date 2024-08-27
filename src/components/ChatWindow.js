import React, { useEffect, useRef } from 'react';

const ChatWindow = ({ messages, isLoading }) => {
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const initialGreeting = (
    <div className="initial-greeting">
      <h1><span className="user-name">Hello, USER</span></h1>
      <h2>How can I help you today?</h2>
      
    </div>
  );

  return (
    <div className="chat-window">
      {messages.length === 0 ? (
        initialGreeting
      ) : (
        messages.map((message, index) => (
          <div key={index} className={`message ${message.sender}`}>
            {message.text}
          </div>
        ))
      )}
      {isLoading && <div className="loading">Generating response</div>}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default ChatWindow;