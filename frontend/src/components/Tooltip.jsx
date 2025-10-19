import { useState, useRef, useEffect } from 'react'
import './Tooltip.css'

const Tooltip = ({ content, children }) => {
  const [isVisible, setIsVisible] = useState(false)
  const [position, setPosition] = useState({ top: 0, left: 0 })
  const triggerRef = useRef(null)

  const showTooltip = () => {
    if (triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect()
      setPosition({
        top: rect.bottom + window.scrollY + 10,
        left: rect.left + window.scrollX + rect.width / 2 - 150
      })
    }
    setIsVisible(true)
  }

  const hideTooltip = () => {
    setIsVisible(false)
  }

  return (
    <div className="tooltip-wrapper" onMouseEnter={showTooltip} onMouseLeave={hideTooltip} ref={triggerRef}>
      <div className="tooltip-trigger">
        {children}
      </div>
      {isVisible && (
        <div className="tooltip-content" style={{
          top: `${position.top}px`,
          left: `${position.left}px`
        }}>
          <div className="tooltip-text">
            {content}
          </div>
        </div>
      )}
    </div>
  )
}

export default Tooltip