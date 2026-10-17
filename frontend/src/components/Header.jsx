export default function Header({ awsStatus }) {
  const isAwsConnected = awsStatus?.aws_configured
  const storageLabel = awsStatus?.storage_mode || 'AWS Ready'

  return (
    <header className="app-header">
      <div className="header-content">
        <div className="header-title-group">
          <h1 className="header-title">
            <span className="header-icon">👁️</span>
            Agentic Vision
          </h1>
          <p className="header-subtitle">
            Intelligent Scene Understanding with OpenCV 5
          </p>
        </div>
        <div className="header-badges">
          <span className="badge badge-opencv">OpenCV 5</span>
          <span className="badge badge-agent">Agentic Loop</span>
          <span
            className={`badge badge-aws ${isAwsConnected ? 'badge-aws-active' : ''}`}
            title={`Storage: ${storageLabel} (${awsStatus?.tier_mode || 'Free Tier'})`}
          >
            {isAwsConnected ? '☁️ AWS S3 Connected' : '☁️ AWS Architecture (Free Tier)'}
          </span>
        </div>
      </div>
    </header>
  )
}
