import { useState, useEffect } from 'react'
import './App.css'
import ImageUploader from './components/ImageUploader'
import AnalysisResult from './components/AnalysisResult'
import AgentTrace from './components/AgentTrace'
import Header from './components/Header'

const API_BASE = ''

function App() {
  const [analysisResult, setAnalysisResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [awsStatus, setAwsStatus] = useState(null)

  useEffect(() => {
    fetch(`${API_BASE}/api/aws-status`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => setAwsStatus(data))
      .catch(() => setAwsStatus({ storage_mode: 'Local Filesystem', aws_configured: false }))
  }, [])

  const handleUpload = async (file) => {
    setIsLoading(true)
    setError(null)
    setAnalysisResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || `Server error: ${response.status}`)
      }

      const data = await response.json()
      setAnalysisResult(data)
    } catch (err) {
      console.error('Analysis failed:', err)
      setError(err.message || 'Failed to analyze image. Is the backend running?')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app">
      <Header awsStatus={awsStatus} />
      <main className="main-content">
        <div className="upload-section">
          <ImageUploader onUpload={handleUpload} isLoading={isLoading} />
          {error && (
            <div className="error-banner">
              <span className="error-icon">⚠</span>
              <span>{error}</span>
            </div>
          )}
        </div>

        {isLoading && (
          <div className="loading-section">
            <div className="loading-spinner"></div>
            <p className="loading-text">Agent is analyzing the image...</p>
            <p className="loading-subtext">Running detection → assessing quality → deciding next steps</p>
          </div>
        )}

        {analysisResult && !isLoading && (
          <div className="results-container">
            <AnalysisResult data={analysisResult} awsStatus={awsStatus} />
            <AgentTrace trace={analysisResult.analysis?.trace || []} />
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>OpenCV AI Competition 2026 — Agentic Vision Award</p>
        <p className="footer-tech">OpenCV {analysisResult?.opencv_version || '5'} · YOLOv8n · FastAPI · React</p>
      </footer>
    </div>
  )
}

export default App
