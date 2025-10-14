import { useState, useEffect } from 'react'
import ExperimentForm from './components/ExperimentForm'
import ResultsView from './components/ResultsView'
import Tooltip from './components/Tooltip'
import { ApiService } from './services/api'
import './App.css'

function App() {
  const [currentExperiment, setCurrentExperiment] = useState(null)
  const [results, setResults] = useState(null)
  const [experiments, setExperiments] = useState([])

  // Load existing experiments on component mount
  useEffect(() => {
    loadExperimentsList()
  }, [])

  const loadExperimentsList = async () => {
    try {
      const response = await ApiService.getExperimentsList()
      const experimentsData = response.experiments.map(exp => ({
        id: exp.exp_id,
        startTime: new Date(exp.created_at),
        status: exp.status === 'done' ? 'completed' : exp.status,
        progress: exp.progress
      }))
      setExperiments(experimentsData)
    } catch (error) {
      console.error('Failed to load experiments list:', error)
    }
  }

  const handleExperimentStart = (expId) => {
    setCurrentExperiment(expId)
    setResults(null)
    
    // Add to experiments history
    const newExp = {
      id: expId,
      startTime: new Date(),
      status: 'running'
    }
    setExperiments(prev => [newExp, ...prev.slice(0, 9)]) // Keep last 10
  }

  const handleResultsReady = (experimentResults) => {
    setResults(experimentResults)
    
    // Update experiment status in history
    setExperiments(prev => 
      prev.map(exp => 
        exp.id === currentExperiment 
          ? { ...exp, status: 'completed', endTime: new Date() }
          : exp
      )
    )
    
    // Refresh experiments list from backend
    loadExperimentsList()
  }

  const loadPreviousExperiment = async (expId) => {
    try {
      setCurrentExperiment(expId)
      setResults(null) // Clear results while loading
      const result = await ApiService.getExperimentResults(expId)
      setResults(result)
    } catch (error) {
      console.error('Failed to load experiment results:', error)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>Federated Learning Security Research</h1>
          <p>Label Flipping Attack Analysis & Defense Mechanisms</p>
        </div>
      </header>

      <main className="app-main">
        <div className="main-content">
          {/* Experiment Form */}
          <section className="experiment-section">
            <ExperimentForm 
              onExperimentStart={handleExperimentStart}
              onResultsReady={handleResultsReady}
            />
          </section>

          {/* Results and History Section */}
          {(results || experiments.length > 0) && (
            <div className={`results-and-history ${results ? 'has-results' : 'no-results'}`}>
              {/* Results Section */}
              {results && (
                <section className="results-section">
                  <ResultsView 
                    results={results} 
                    experimentId={currentExperiment}
                  />
                </section>
              )}

              {/* Experiment History Sidebar */}
              {experiments.length > 0 && (
                <aside className={`history-sidebar ${results ? 'sidebar-mode' : 'center-mode'}`}>
                  <h3>Recent Experiments 
                    <Tooltip content="View your experiment history. Click on any experiment to see its detailed results, performance charts, and configuration parameters.">
                      <span className="help-icon">?</span>
                    </Tooltip>
                  </h3>
                  <div className="experiments-list">
                    {experiments.map(exp => (
                      <div 
                        key={exp.id}
                        className={`experiment-item ${exp.id === currentExperiment ? 'active' : ''}`}
                        onClick={() => loadPreviousExperiment(exp.id)}
                      >
                        <div className="exp-header">
                          <span className="exp-id">{exp.id}</span>
                          <span className={`exp-status ${exp.status}`}>
                            <div className={`status-dot ${exp.status}`}></div>
                            {exp.status}
                          </span>
                        </div>
                        <div className="exp-time">
                          {exp.startTime.toLocaleTimeString()}
                        </div>
                      </div>
                    ))}
                  </div>
                </aside>
              )}
            </div>
          )}
        </div>
      </main>

      <footer className="app-footer">
        <div className="footer-content">
          <span className="footer-text">
            Federated Learning Security Research Platform
          </span>
          <a href="https://github.com/MinhajulBhuiyan/fl-security" target="_blank" rel="noopener noreferrer" className="footer-link">
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            Repository
          </a>
        </div>
      </footer>
    </div>
  )
}

export default App
