export default function AgentTrace({ trace }) {
  if (!trace || trace.length === 0) return null

  const getActionIcon = (action) => {
    const icons = {
      initial_detection: '🔍',
      quality_assessment: '📊',
      enhance_and_redetect: '✨',
      crop_and_inspect: '🔬',
      accept_result: '✅',
      request_human_review: '👤',
    }
    return icons[action] || '⚙️'
  }

  const getActionColor = (action) => {
    const colors = {
      initial_detection: 'var(--color-primary)',
      quality_assessment: 'var(--color-info)',
      enhance_and_redetect: 'var(--color-warning)',
      crop_and_inspect: 'var(--color-accent)',
      accept_result: 'var(--color-success)',
      request_human_review: 'var(--color-danger)',
    }
    return colors[action] || 'var(--color-muted)'
  }

  return (
    <div className="trace-panel">
      <div className="section-header">
        <h2>Agent Execution Trace</h2>
        <span className="trace-count">{trace.length} step{trace.length !== 1 ? 's' : ''}</span>
      </div>
      <p className="trace-description">
        Each step shows the agent's observation, reasoning, and decision.
        The agent dynamically chose this path based on the actual visual results.
      </p>

      <div className="trace-timeline">
        {trace.map((step, i) => (
          <div key={i} className="trace-step" style={{ '--step-color': getActionColor(step.action) }}>
            <div className="trace-connector">
              <div className="trace-dot">{getActionIcon(step.action)}</div>
              {i < trace.length - 1 && <div className="trace-line" />}
            </div>
            <div className="trace-content">
              <div className="trace-header">
                <span className="trace-step-number">Step {step.step_number}</span>
                <span className="trace-action-name">
                  {step.action.replace(/_/g, ' ')}
                </span>
                {step.duration_ms > 0 && (
                  <span className="trace-duration">{step.duration_ms.toFixed(0)}ms</span>
                )}
              </div>
              <div className="trace-reason">
                <strong>Reason:</strong> {step.reason}
              </div>
              {step.tool_input && (
                <div className="trace-tool-input">
                  <strong>Input:</strong> {step.tool_input}
                </div>
              )}
              <div className="trace-result">
                <strong>Result:</strong> {step.result_summary}
              </div>
              {step.detections_count > 0 && (
                <div className="trace-stats">
                  <span>{step.detections_count} detection{step.detections_count !== 1 ? 's' : ''}</span>
                  <span>·</span>
                  <span>{(step.avg_confidence * 100).toFixed(1)}% avg confidence</span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
