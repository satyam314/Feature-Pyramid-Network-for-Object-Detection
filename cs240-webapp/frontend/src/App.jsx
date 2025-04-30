import ImageUploader from './ImageUploader'
import './App.css'
import { Routes, Route } from 'react-router-dom'
import Loading from './Loading'
import Result from './Result'  

function App() {
  return (
    <Routes>
      <Route path="/" element={<ImageUploader />} />
      <Route path="/loading" element={<Loading />} />
      <Route path="/result" element={<Result />} />   {/* 👈 New route */}
    </Routes>
  )
}

export default App

