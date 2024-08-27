import React, { useState } from 'react';
import axios from 'axios';

function Login({ onLogin, onSignup, showLogin, setShowLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      if (!showLogin) {
        // Signup
        await axios.post('http://localhost:5000/signup', { username, email, password });
        alert('Signup successful. Please log in.');
        setShowLogin(true);
        setUsername('');
        setPassword('');
        setEmail('');
      } else {
        // Login
        await axios.post('http://localhost:5000/login', { username, password }, { withCredentials: true });
        onLogin();
      }
    } catch (error) {
      console.error('Error:', error);
      alert(error.response?.data?.message || 'An error occurred');
    }
  };

  const styles = {
    loginContainer: {
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
      backgroundColor: '#f0f2f5',
    },
    loginBox: {
      backgroundColor: 'white',
      padding: '20px',
      borderRadius: '8px',
      boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
      width: '300px',
    },
    title: {
      textAlign: 'center',
      marginBottom: '20px',
      color: '#1877f2',
    },
    form: {
      display: 'flex',
      flexDirection: 'column',
    },
    input: {
      margin: '10px 0',
      padding: '10px',
      border: '1px solid #dddfe2',
      borderRadius: '6px',
      fontSize: '16px',
    },
    button: {
      backgroundColor: '#1877f2',
      color: 'white',
      border: 'none',
      padding: '10px',
      borderRadius: '6px',
      fontSize: '16px',
      fontWeight: 'bold',
      cursor: 'pointer',
      marginTop: '10px',
    },
    switchText: {
      textAlign: 'center',
      marginTop: '20px',
    },
    switchLink: {
      color: '#1877f2',
      cursor: 'pointer',
    },
  };

  return (
    <div style={styles.loginContainer}>
      <div style={styles.loginBox}>
        <h2 style={styles.title}>{showLogin ? 'Login' : 'Sign Up'}</h2>
        <form onSubmit={handleSubmit} style={styles.form}>
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            style={styles.input}
          />
          {!showLogin && (
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              style={styles.input}
            />
          )}
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            style={styles.input}
          />
          <button type="submit" style={styles.button}>
            {showLogin ? 'Login' : 'Sign Up'}
          </button>
        </form>
        <p style={styles.switchText}>
          {showLogin ? "Don't have an account?" : "Already have an account?"}
          <span style={styles.switchLink} onClick={() => setShowLogin(!showLogin)}>
            {showLogin ? ' Sign Up' : ' Login'}
          </span>
        </p>
      </div>
    </div>
  );
}

export default Login;