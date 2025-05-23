import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ParticlesBackground from './ParticlesBackground';
import Loader from './Loader';

function Loading() {
  const navigate = useNavigate();

  useEffect(() => {
    const interval = setInterval(async () => {
      const res = await fetch('/api/status');
      const data = await res.json();

      if (data.status == 'done') {
        clearInterval(interval);
        navigate('/result');  // Redirect to result page after detection
      }
    }, 2000);  // Check every 2 seconds

    return () => clearInterval(interval);
  }, [navigate]);

  return (
    <>
      <ParticlesBackground />
      <div style={{ textAlign: 'center', marginTop: '100px' }}>
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '20px' }}>
          <Loader />
        </div>
        <h2>Processing your image...</h2>
        <p>Please wait while detection is running.</p>
      </div>
    </>
  );
}

export default Loading;
