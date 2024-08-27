import React from 'react';

function ChatHistory({ history, onClose }) {
  return (
    <div className="chat-history-modal">
      <div className="chat-history-content">
        <h2>Chat History</h2>
        <div className="chat-history-list">
          {history.map((entry, index) => (
            <div key={index} className="chat-history-entry">
              <p><strong>Question:</strong> {entry.question}</p>
              <p><strong>Answer:</strong> {entry.answer}</p>
            </div>
          ))}
        </div>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
}

export default ChatHistory;