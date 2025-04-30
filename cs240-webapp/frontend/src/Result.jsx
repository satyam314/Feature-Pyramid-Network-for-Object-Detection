import { useState, useEffect } from 'react';

function Result() {
  const [imageUrl, setImageUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Function to check the status and get the result image
    const checkStatus = async () => {
      const res = await fetch('/api/status');
      const data = await res.json();

      // Debugging: Log the data status and result image
      console.log("Status data:", data);

      if (data.status === 'done' && data.result_image) {
        setIsLoading(false);
        // Construct the URL for the result image
        const resultImageUrl = `/results/${data.result_image}`;
        console.log("Result image URL:", resultImageUrl);  // Debugging
        setImageUrl(resultImageUrl);
        console.log("Result image URL 2:", resultImageUrl);  // Debugging
      } else {
        // If detection is not done yet, keep checking every 2 seconds
        setTimeout(checkStatus, 2000);
      }
    };

    checkStatus(); // Initial check for detection status
  }, []);

  return (
    <div style={{ textAlign: 'center', marginTop: '100px' }}>
      <h2>Detection Result</h2>
      {isLoading ? (
        <p>Loading detection result...</p>
      ) : (
        <div>
          {imageUrl ? (
            <img src={`http://127.0.0.1:5000${imageUrl}`} alt="Processed Result" width="100%" />
          ) : (
            <p>Image not found!</p>
          )}
        </div>
      )}
    </div>
  );
}

export default Result;
