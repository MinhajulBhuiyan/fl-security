import { useState } from 'react'
import './Tooltip.css'

const Tooltip = ({ content, children }) => {
  const [isVisible, setIsVisible] = useState(false)

  const showTooltip = () => {
    setIsVisible(true)
  }

  const hideTooltip = () => {
    setIsVisible(false)
  }

  return (
    <div className="tooltip-wrapper" onMouseEnter={showTooltip} onMouseLeave={hideTooltip}>
      <div className="tooltip-trigger">
        {children}
      </div>
      {isVisible && (
        <div className="tooltip-content">
          <div className="tooltip-text">
            {content}
          </div>
        </div>
      )}
    </div>
  )
}

export default Tooltip