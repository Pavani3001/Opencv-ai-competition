import { useRef, useState } from 'react'

export default function ImageUploader({ onUpload, isLoading }) {
  const fileInputRef = useRef(null)
  const [preview, setPreview] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [fileName, setFileName] = useState('')

  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp']
  const MAX_SIZE = 10 * 1024 * 1024 // 10 MB

  const handleFile = (file) => {
    if (!file) return

    if (!ALLOWED_TYPES.includes(file.type)) {
      alert(`Unsupported file type: ${file.type}\nAllowed: JPEG, PNG, WebP, BMP`)
      return
    }

    if (file.size > MAX_SIZE) {
      alert(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Maximum: 10 MB.`)
      return
    }

    setFileName(file.name)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(file)
    onUpload(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragActive(false)
    const file = e.dataTransfer.files?.[0]
    handleFile(file)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragActive(true)
  }

  const handleDragLeave = () => setDragActive(false)

  const handleClick = () => {
    if (!isLoading) fileInputRef.current?.click()
  }

  const handleChange = (e) => {
    const file = e.target.files?.[0]
    handleFile(file)
  }

  return (
    <div className="uploader-wrapper">
      <div
        className={`dropzone ${dragActive ? 'dropzone-active' : ''} ${isLoading ? 'dropzone-disabled' : ''}`}
        onClick={handleClick}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.webp,.bmp"
          onChange={handleChange}
          hidden
        />
        {preview ? (
          <div className="preview-container">
            <img src={preview} alt="Preview" className="preview-image" />
            <p className="preview-name">{fileName}</p>
          </div>
        ) : (
          <div className="dropzone-content">
            <div className="dropzone-icon">📷</div>
            <p className="dropzone-title">Drop an image here or click to upload</p>
            <p className="dropzone-subtitle">JPEG, PNG, WebP, or BMP · Max 10 MB</p>
          </div>
        )}
      </div>
    </div>
  )
}
