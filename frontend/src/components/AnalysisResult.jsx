import { useState } from 'react'

export default function AnalysisResult({ data }) {
  const [showAnnotated, setShowAnnotated] = useState(true)
  const analysis = data.analysis || {}
  const detections = analysis.final_detections || []
  const comparison = analysis.comparison || {}

  return (
    <div className="result-panel">
      {/* Image display with toggle */}
      <div className="result-section">
        <div className="section-header">
          <h2>Detection Results</h2>
          <div className="image-toggle">
            <button
              className={`toggle-btn ${showAnnotated ? 'active' : ''}`}
              onClick={() => setShowAnnotated(true)}
            >
              Annotated
            </button>
            <button
              className={`toggle-btn ${!showAnnotated ? 'active' : ''}`}
              onClick={() => setShowAnnotated(false)}
            >
              Original
            </button>
          </div>
        </div>
        <div className="result-image-container">
          <img
            src={showAnnotated ? data.image_annotated : data.image_original}
            alt={showAnnotated ? 'Annotated image with detections' : 'Original uploaded image'}
            className="result-image"
          />
        </div>
      </div>

      {/* Scene summary */}
      <div className="result-section">
        <h2>Scene Summary</h2>
        <p className="scene-summary">{analysis.scene_summary || 'No summary available.'}</p>

        {analysis.needs_human_review && (
          <div className="human-review-banner">
            <span className="review-icon">🔍</span>
            <div>
              <strong>Human Review Requested</strong>
              <p>{analysis.human_review_reason}</p>
            </div>
          </div>
        )}

        <div className="stop-reason">
          <span className="stop-label">Agent Decision:</span>
          <span className="stop-value">{analysis.stop_reason}</span>
        </div>
      </div>

      {/* Detections table */}
      {detections.length > 0 && (
        <div className="result-section">
          <h2>Detected Objects ({detections.length})</h2>
          <div className="detections-table-wrapper">
            <table className="detections-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Object</th>
                  <th>Confidence</th>
                  <th>Position</th>
                  <th>Size</th>
                </tr>
              </thead>
              <tbody>
                {detections.map((det, i) => (
                  <tr key={i} className={det.confidence < 0.45 ? 'low-confidence' : ''}>
                    <td>{i + 1}</td>
                    <td className="det-label">{det.label}</td>
                    <td>
                      <div className="confidence-bar-cell">
                        <div
                          className="confidence-bar"
                          style={{
                            width: `${Math.round(det.confidence * 100)}%`,
                            backgroundColor: det.confidence > 0.65 ? 'var(--color-success)' : det.confidence > 0.45 ? 'var(--color-warning)' : 'var(--color-danger)',
                          }}
                        />
                        <span className="confidence-text">{(det.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td className="det-pos">({det.x}, {det.y})</td>
                    <td className="det-size">{det.width}×{det.height}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Comparison: initial vs final */}
      {comparison.initial_count !== undefined && (
        <div className="result-section">
          <h2>Initial vs. Final Comparison</h2>
          <div className="comparison-grid">
            <div className="comparison-card">
              <span className="comparison-label">Initial</span>
              <span className="comparison-value">{comparison.initial_count} objects</span>
              <span className="comparison-conf">{(comparison.initial_avg_confidence * 100).toFixed(1)}% avg</span>
            </div>
            <div className="comparison-arrow">→</div>
            <div className="comparison-card">
              <span className="comparison-label">Final</span>
              <span className="comparison-value">{comparison.final_count} objects</span>
              <span className="comparison-conf">{(comparison.final_avg_confidence * 100).toFixed(1)}% avg</span>
            </div>
            <div className={`comparison-verdict ${comparison.labels_changed ? 'changed' : 'unchanged'}`}>
              {comparison.labels_changed
                ? '🔄 Results changed after follow-up inspection'
                : '✅ Results confirmed — no change needed'}
            </div>
          </div>
        </div>
      )}

      {/* Image quality */}
      {analysis.image_quality && Object.keys(analysis.image_quality).length > 0 && (
        <div className="result-section">
          <h2>Image Quality Assessment</h2>
          <div className="quality-grid">
            <QualityMetric label="Brightness" value={analysis.image_quality.brightness} flag={analysis.image_quality.is_dark} flagText="Dark" />
            <QualityMetric label="Contrast" value={analysis.image_quality.contrast} flag={analysis.image_quality.is_low_contrast} flagText="Low" />
            <QualityMetric label="Sharpness" value={analysis.image_quality.sharpness} flag={analysis.image_quality.is_blurry} flagText="Blurry" />
          </div>
        </div>
      )}

      {/* Model limitations */}
      <div className="result-section limitations-section">
        <h2>Model Limitations</h2>
        <p className="limitations-text">{analysis.model_limitations}</p>
      </div>

      {/* Timing & Storage */}
      <div className="result-meta">
        <span>⏱️ Total time: {(analysis.total_duration_ms / 1000).toFixed(2)}s</span>
        <span>🔄 Steps: {analysis.trace?.length || 0}</span>
        <span>👁️ OpenCV {data.opencv_version}</span>
        {data.storage && (
          <span className="storage-badge">
            💾 {data.storage.location}
            {data.storage.cloud_url && (
              <a
                href={data.storage.cloud_url}
                target="_blank"
                rel="noreferrer"
                className="cloud-link"
              >
                (View S3 Artifact)
              </a>
            )}
          </span>
        )}
      </div>
    </div>
  )
}

function QualityMetric({ label, value, flag, flagText }) {
  return (
    <div className={`quality-card ${flag ? 'quality-flagged' : ''}`}>
      <span className="quality-label">{label}</span>
      <span className="quality-value">{typeof value === 'number' ? value.toFixed(1) : value}</span>
      {flag && <span className="quality-flag">⚠ {flagText}</span>}
    </div>
  )
}
