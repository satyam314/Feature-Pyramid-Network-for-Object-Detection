import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Form from './Form';
import ParticlesBackground from './ParticlesBackground';

function ImageUploader() {
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [expanded, setExpanded] = useState(false); // for animation
  const navigate = useNavigate();

  useEffect(() => {
    setExpanded(!!previewUrl); // true if previewUrl is set
  }, [previewUrl]);

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setImage(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setImage(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleZoneClick = (e) => {
    if (e.target.id !== 'uploadButton') {
      document.getElementById('fileInput').click();
    }
  };

  const handleUpload = async () => {
    if (!image) {
      setUploadStatus('No image selected.');
      return;
    }

    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        navigate('/loading');
      } else {
        setUploadStatus('Upload failed.');
      }
    } catch (error) {
      setUploadStatus('Upload error: ' + error.message);
    }
  };

  return (
    <>
    <ParticlesBackground />
    <h1>Object Detection 
      using FPN</h1>
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onClick={handleZoneClick}
      style={{
        
        padding: expanded ? '30px' : '20px',
        textAlign: 'center',
        cursor: 'pointer',
        marginBottom: '10px',
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        transition: 'max-height 1s ease, padding 0.5s ease',
        maxHeight: expanded ? '700px' : '250px'
        
      }}
    >
      {previewUrl ? (
        <img
          src={previewUrl}
          alt="Preview"
          style={{
            maxWidth: '100%',
            maxHeight: '300px',
            marginBottom: '10px',
            opacity: expanded ? 1 : 0,
            transform: expanded ? 'scale(1)' : 'scale(0.95)',
            transition: 'opacity 0.5s ease, transform 0.5s ease',
          }}
        />
      ) : (
        <Form />
      )}

      <input
        type="file"
        id="fileInput"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />

      {image && (
        <button
          id="uploadButton"
          onClick={(e) => {
            e.stopPropagation();
            handleUpload();
          }}
          style={{
            marginTop: '10px',
            opacity: expanded ? 1 : 0,
            transform: expanded ? 'translateY(0)' : 'translateY(10px)',
            transition: 'opacity 0.5s ease, transform 0.5s ease',
          }}
        >
          Upload
        </button>
      )}

      {uploadStatus && <p style={{ marginTop: '10px' }}>{uploadStatus}</p>}
    </div>
    </>
  );
}

export default ImageUploader;
