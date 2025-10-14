import React, { useState, useEffect } from 'react';
import { ApiService, EXPERIMENT_CONFIG } from '../services/api';
import Tooltip from './Tooltip';
import './ExperimentForm.css';

const ExperimentForm = ({ onExperimentStart, onResultsReady }) => {
  const [config, setConfig] = useState(EXPERIMENT_CONFIG.defaultConfig);
  const [isRunning, setIsRunning] = useState(false);
  const [currentExperiment, setCurrentExperiment] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [kwargsText, setKwargsText] = useState(JSON.stringify(EXPERIMENT_CONFIG.defaultConfig.kwargs, null, 2));

  // Handle form field changes
  const handleInputChange = (field, value) => {
    setConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };

  // Handle kwargs JSON input
  const handleKwargsChange = (text) => {
    setKwargsText(text);
    try {
      const parsed = JSON.parse(text);
      setConfig(prev => ({ ...prev, kwargs: parsed }));
      setError(null);
    } catch (e) {
      setError('Invalid JSON in kwargs field');
    }
  };

  // Validate form before submission
  const validateForm = () => {
    if (config.num_poisoned_workers < 0) {
      setError('Number of poisoned workers must be >= 0');
      return false;
    }
    if (config.workers_per_round < 1) {
      setError('Workers per round must be >= 1');
      return false;
    }
    try {
      JSON.parse(kwargsText);
    } catch (e) {
      setError('Invalid JSON in kwargs field');
      return false;
    }
    setError(null);
    return true;
  };

  // Start experiment
  const handleStartExperiment = async () => {
    if (!validateForm()) return;

    setIsRunning(true);
    setError(null);
    
    try {
      // Use real API
      const response = await ApiService.runExperiment(config);
      setCurrentExperiment(response.exp_id);
      setStatus('submitted');
      onExperimentStart && onExperimentStart(response.exp_id);
      
      // Start polling for status
      pollExperimentStatus(response.exp_id);
    } catch (err) {
      setError(err.message);
      setIsRunning(false);
    }
  };

  // Poll experiment status
  const pollExperimentStatus = async (expId) => {
    const pollInterval = setInterval(async () => {
      try {
        const statusResponse = await ApiService.getExperimentStatus(expId);
        setStatus(statusResponse.status);

        if (statusResponse.status === 'done') {
          clearInterval(pollInterval);
          setIsRunning(false);
          
          // Fetch results
          const results = await ApiService.getExperimentResults(expId);
          onResultsReady && onResultsReady(results);
          
        } else if (statusResponse.status === 'error') {
          clearInterval(pollInterval);
          setIsRunning(false);
          setError(statusResponse.message || 'Experiment failed');
        }
      } catch (err) {
        clearInterval(pollInterval);
        setIsRunning(false);
        setError('Failed to get experiment status');
      }
    }, 2000); // Poll every 2 seconds

    // Cleanup on component unmount
    return () => clearInterval(pollInterval);
  };

  // Cancel experiment
  const handleCancel = () => {
    setIsRunning(false);
    setCurrentExperiment(null);
    setStatus(null);
    setError(null);
  };

  // Reset form
  const handleReset = () => {
    setConfig(EXPERIMENT_CONFIG.defaultConfig);
    setKwargsText(JSON.stringify(EXPERIMENT_CONFIG.defaultConfig.kwargs, null, 2));
    setError(null);
  };

  return (
    <div className="experiment-form">
      <div className="form-header">
        <h2>Experiment Configuration
          <Tooltip content="Configure parameters for federated learning security experiments. Set up attack scenarios and worker selection strategies to test defense mechanisms against label flipping attacks.">
            <span className="help-icon">?</span>
          </Tooltip>
        </h2>
        <p>Configure federated learning parameters and attack settings</p>
      </div>

      <form onSubmit={(e) => { e.preventDefault(); handleStartExperiment(); }}>
        <div className="form-grid">
          {/* Number of Poisoned Workers */}
          <div className="form-group">
            <label htmlFor="poisoned-workers">
              Poisoned Clients
              <Tooltip content="Number of malicious participants that will perform label flipping attacks during training. These clients will deliberately mislabel their data to degrade model performance.">
                <span className="tooltip">?</span>
              </Tooltip>
            </label>
            <input
              id="poisoned-workers"
              type="number"
              min="0"
              max="50"
              value={config.num_poisoned_workers}
              onChange={(e) => handleInputChange('num_poisoned_workers', parseInt(e.target.value))}
              disabled={isRunning}
            />
          </div>

          {/* Replacement Method */}
          <div className="form-group">
            <label htmlFor="replacement-method">
              Attack Method
              <Tooltip content="Defines how malicious clients will flip labels. 'Random' assigns random incorrect labels, 'Class-specific' targets specific classes for maximum impact on model accuracy.">
                <span className="tooltip">?</span>
              </Tooltip>
            </label>
            <select
              id="replacement-method"
              value={config.replacement_method}
              onChange={(e) => handleInputChange('replacement_method', e.target.value)}
              disabled={isRunning}
            >
              {EXPERIMENT_CONFIG.replacementMethods.map(method => (
                <option key={method.value} value={method.value}>
                  {method.label}
                </option>
              ))}
            </select>
          </div>

          {/* Selection Strategy */}
          <div className="form-group">
            <label htmlFor="selection-strategy">
              Selection Strategy
              <Tooltip content="Determines how clients are chosen for each training round. 'Random' selects clients randomly, while other strategies may use performance metrics or other criteria.">
                <span className="tooltip">?</span>
              </Tooltip>
            </label>
            <select
              id="selection-strategy"
              value={config.selection_strategy}
              onChange={(e) => handleInputChange('selection_strategy', e.target.value)}
              disabled={isRunning}
            >
              {EXPERIMENT_CONFIG.selectionStrategies.map(strategy => (
                <option key={strategy.value} value={strategy.value}>
                  {strategy.label}
                </option>
              ))}
            </select>
          </div>

          {/* Workers Per Round */}
          <div className="form-group">
            <label htmlFor="workers-per-round">
              Workers Per Round
              <Tooltip content="Number of clients participating in each training round. Higher values increase training diversity but may include more poisoned workers.">
                <span className="tooltip">?</span>
              </Tooltip>
            </label>
            <input
              id="workers-per-round"
              type="number"
              min="1"
              max="50"
              value={config.workers_per_round}
              onChange={(e) => handleInputChange('workers_per_round', parseInt(e.target.value))}
              disabled={isRunning}
            />
          </div>

          {/* Quick Demo Mode */}
          <div className="form-group checkbox-group">
            <label htmlFor="quick-mode" className="checkbox-label">
              <input
                id="quick-mode"
                type="checkbox"
                checked={config.quick_mode}
                onChange={(e) => handleInputChange('quick_mode', e.target.checked)}
                disabled={isRunning}
              />
              Quick Mode
              <Tooltip content="Run a shorter experiment for testing purposes. Uses fewer training epochs and a smaller dataset subset for faster results.">
                <span className="tooltip">?</span>
              </Tooltip>
            </label>
          </div>
        </div>

        {/* Advanced Configuration */}
        <details className="advanced-config">
          <summary>Advanced Configuration</summary>
          <div className="form-group">
            <label htmlFor="kwargs">
              Strategy Parameters (JSON)
              <Tooltip content="Additional parameters for selection strategies in JSON format. These control fine-grained behavior like minimum learning rates, breakpoint conditions, or probability thresholds.">
                <span className="tooltip">?</span>
              </Tooltip>
            </label>
            <textarea
              id="kwargs"
              value={kwargsText}
              onChange={(e) => handleKwargsChange(e.target.value)}
              disabled={isRunning}
              rows="6"
              placeholder='{"NUM_WORKERS_PER_ROUND": 5}'
            />
          </div>
        </details>

        {/* Status Display */}
        {(currentExperiment || status) && (
          <div className="status-section">
            <div className="status-card">
              <div className="status-header">
                <h4>Experiment Status
                  <Tooltip content="Shows the current state of your experiment: Submitted (queued for processing), Running (actively training), Complete (finished successfully with results), or Failed (error occurred during execution).">
                    <span className="help-icon">?</span>
                  </Tooltip>
                </h4>
                {currentExperiment && (
                  <span className="experiment-id">{currentExperiment}</span>
                )}
              </div>
              <div className={`status-badge status-${status}`}>
                <span className="status-indicator"></span>
                {status === 'submitted' && 'Submitted'}
                {status === 'running' && 'Running'}
                {status === 'done' && 'Complete'}
                {status === 'error' && 'Failed'}
              </div>
              {status === 'running' && (
                <div className="progress-bar">
                  <div className="progress-fill"></div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="error-section">
            <div className="error-card">
              Error: {error}
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="form-actions">
          {!isRunning ? (
            <>
              <button type="submit" className="btn btn-primary">
                Start Experiment
              </button>
              <button type="button" onClick={handleReset} className="btn btn-secondary">
                Reset
              </button>
            </>
          ) : (
            <button type="button" onClick={handleCancel} className="btn btn-danger">
              Cancel
            </button>
          )}
        </div>
      </form>
    </div>
  );
};

export default ExperimentForm;