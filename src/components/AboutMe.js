import React from 'react';

function AboutMe({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <h2>About This Project</h2>
        <p>This AI Medical Chatbot is powered by Mistral-7B, a state-of-the-art language model. It's designed to provide general medical information and assistance.</p>
        <p>Key Features:</p>
        <ul>
          <li>Powered by Mistral-7B</li>
          <li>Provides medical information and advice</li>
          <li>Uses RAG (Retrieval-Augmented Generation) for accurate responses</li>
          <li>Built with React and Flask</li>
        </ul>
        <p>Note: This chatbot is for informational purposes only and should not replace professional medical advice.</p>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
}

export default AboutMe;