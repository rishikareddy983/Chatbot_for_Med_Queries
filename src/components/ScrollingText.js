import React, { useState, useEffect } from 'react';

const ScrollingText = ({ tips }) => {
  const [currentTipIndex, setCurrentTipIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTipIndex((prevIndex) => (prevIndex + 1) % tips.length);
    }, 5000); // Change tip every 5 seconds

    return () => clearInterval(interval);
  }, [tips]);

  return (
    <div className="scrolling-text">
      {tips[currentTipIndex]}
    </div>
  );
};

export default ScrollingText;