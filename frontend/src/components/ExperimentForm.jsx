import React, { useState, useEffect } from 'react';
import { ApiService, EXPERIMENT_CONFIG } from '../services/api';
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
    setConfig(prev => {
      const newConfig = {
        ...prev,
        [field]: value
      };
      
      // Sync workers_per_round with kwargs.NUM_WORKERS_PER_ROUND
      if (field === 'workers_per_round') {
        newConfig.kwargs = {
          ...prev.kwargs,
          NUM_WORKERS_PER_ROUND: value
        };
        setKwargsText(JSON.stringify(newConfig.kwargs, null, 2));
      }
      
      return newConfig;
    });
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
    
    // Prevent multiple submissions
    if (isRunning) {
      setError('An experiment is already running. Please wait for it to complete.');
      return;
    }

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
        <h2>Experiment Configuration</h2>
        <p>Configure federated learning parameters and attack settings</p>
      </div>

      <form onSubmit={(e) => { e.preventDefault(); handleStartExperiment(); }}>
        <div className="form-grid">
          {/* Dataset Selection */}
          <div className="form-group">
            <label htmlFor="dataset">Dataset</label>
            <select
              id="dataset"
              value={config.dataset}
              onChange={(e) => handleInputChange('dataset', e.target.value)}
              disabled={isRunning}
            >
              {EXPERIMENT_CONFIG.datasets.map(dataset => (
                <option key={dataset.value} value={dataset.value}>
                  {dataset.label} - {dataset.description}
                </option>
              ))}
            </select>
          </div>
          
          {/* Number of Poisoned Workers */}
          <div className="form-group">
            <label htmlFor="poisoned-workers">Poisoned Clients</label>
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
            <label htmlFor="replacement-method">Attack Method</label>
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

          {/* Selection Strategy - Moved to Advanced */}
          
          {/* Workers Per Round - Moved to Advanced */}
        </div>

        {/* Advanced Configuration */}
        <details className="advanced-config">
          <summary>Advanced Configuration</summary>
          
          {/* Defense Mechanism Toggle */}
          <div className="form-group defense-toggle">
            <label htmlFor="enable-defense">
              <input
                id="enable-defense"
                type="checkbox"
                checked={config.enable_defense || false}
                onChange={(e) => handleInputChange('enable_defense', e.target.checked)}
                disabled={isRunning}
              />
              Enable Defense Mechanism
            </label>
            <span className="defense-info">
              {config.enable_defense ? '✓ Defense Active' : 'No Defense'}
            </span>
          </div>

          {/* Defense Method Selection */}
          {config.enable_defense && (
            <div className="form-group">
              <label htmlFor="defense-method">Defense Method</label>
              <select
                id="defense-method"
                value={config.defense_method || 'byzantine_robust'}
                onChange={(e) => handleInputChange('defense_method', e.target.value)}
                disabled={isRunning}
              >
                <option value="byzantine_robust">Byzantine-Robust Aggregation</option>
                <option value="anomaly_detection">Anomaly Detection</option>
                <option value="gradient_clipping">Gradient Clipping</option>
                <option value="client_filtering">Client Filtering</option>
              </select>
            </div>
          )}

          {/* Selection Strategy */}
          <div className="form-group">
            <label htmlFor="selection-strategy">Worker Selection Strategy</label>
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
            <label htmlFor="workers-per-round">Workers Per Round</label>
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
          <div className="form-group">
            <label htmlFor="kwargs">Strategy Parameters (JSON)</label>
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
                <h4>Experiment Status</h4>
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
            <>
              <button type="submit" className="btn btn-primary" disabled>
                Experiment Running...
              </button>
              <button type="button" onClick={handleCancel} className="btn btn-danger">
                Cancel
              </button>
            </>
          )}
        </div>
      </form>
    </div>
  );
};

export default ExperimentForm;