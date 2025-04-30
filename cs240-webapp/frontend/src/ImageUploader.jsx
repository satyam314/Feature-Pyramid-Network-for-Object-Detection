import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function ImageUploader() {
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const navigate = useNavigate();

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
    // If the click target is NOT the upload button, open file selector
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
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onClick={handleZoneClick}
      style={{
        border: '2px dashed #ccc',
        padding: '20px',
        textAlign: 'center',
        cursor: 'pointer',
        marginBottom: '10px',
        position: 'relative',
      }}
    >
      {previewUrl ? (
        <img
          src={previewUrl}
          alt="Preview"
          style={{ maxWidth: '100%', maxHeight: '300px', marginBottom: '10px' }}
        />
      ) : (
        <p>Drag and drop an image here, or click to select</p>
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
            e.stopPropagation(); // <--- VERY IMPORTANT
            handleUpload();
          }}
          style={{ marginTop: '10px' }}
        >
          Upload
        </button>
      )}
      {uploadStatus && <p>{uploadStatus}</p>}
    </div>
  );
}

export default ImageUploader;
