import React, { useState, useMemo, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import Tooltip from './Tooltip';
import { ApiService } from '../services/api';
import './ResultsView.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  ChartTooltip,
  Legend,
  Filler
);

const ResultsView = ({ results, experimentId }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedImage, setSelectedImage] = useState(null);
  const [classificationResult, setClassificationResult] = useState(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [classificationError, setClassificationError] = useState(null);

  // Prepare chart data using useMemo for better performance
  const chartData = useMemo(() => {
    if (!results?.results?.epochs) return null;

    const { epochs, accuracy, loss } = results.results;
    
    return {
      accuracy: {
        labels: epochs,
        datasets: [
          {
            label: 'Model Accuracy (%)',
            data: accuracy,
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.08)',
            borderWidth: 2.5,
            fill: true,
            tension: 0.4,
            pointBackgroundColor: '#06b6d4',
            pointBorderColor: '#ffffff',
            pointBorderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 6,
          }
        ]
      },
      loss: {
        labels: epochs,
        datasets: [
          {
            label: 'Training Loss',
            data: loss,
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.08)',
            borderWidth: 2.5,
            fill: true,
            tension: 0.4,
            pointBackgroundColor: '#f59e0b',
            pointBorderColor: '#ffffff',
            pointBorderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 6,
          }
        ]
      },
      combined: {
        labels: epochs,
        datasets: [
          {
            label: 'Accuracy (%)',
            data: accuracy,
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.08)',
            borderWidth: 2,
            yAxisID: 'y',
            tension: 0.4,
            pointBackgroundColor: '#06b6d4',
            pointBorderColor: '#ffffff',
            pointBorderWidth: 1.5,
            pointRadius: 3,
          },
          {
            label: 'Loss',
            data: loss.map(l => l * 20), // Scale loss for dual axis
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.08)',
            borderWidth: 2,
            yAxisID: 'y1',
            tension: 0.4,
            pointBackgroundColor: '#f59e0b',
            pointBorderColor: '#ffffff',
            pointBorderWidth: 1.5,
            pointRadius: 3,
          }
        ]
      }
    };
  }, [results]);

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        align: 'center',
        labels: {
          font: {
            size: 13,
            family: 'system-ui',
            weight: '500'
          },
          color: '#475569',
          padding: 16,
          usePointStyle: true,
          pointStyle: 'circle',
          boxWidth: 8,
          boxHeight: 8
        }
      },
      title: {
        display: false
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(15, 23, 42, 0.95)',
        titleColor: '#f1f5f9',
        bodyColor: '#f1f5f9',
        borderColor: '#64748b',
        borderWidth: 1,
        padding: 12,
        titleFont: {
          size: 14,
          weight: 'bold'
        },
        bodyFont: {
          size: 13
        },
        displayColors: true,
        boxPadding: 8,
        callbacks: {
          labelColor: function(context) {
            return {
              borderColor: context.dataset.borderColor,
              backgroundColor: context.dataset.borderColor,
              borderRadius: 4,
              borderWidth: 0
            }
          }
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Epoch',
          font: {
            size: 12,
            weight: 'bold'
          },
          color: '#475569',
          padding: 10
        },
        grid: {
          color: 'rgba(226, 232, 240, 0.5)',
          drawBorder: false,
          lineWidth: 0.5
        },
        ticks: {
          color: '#64748b',
          font: {
            size: 11
          }
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Accuracy (%)',
          font: {
            size: 12,
            weight: 'bold'
          },
          color: '#475569',
          padding: 10
        },
        grid: {
          color: 'rgba(226, 232, 240, 0.5)',
          drawBorder: false,
          lineWidth: 0.5
        },
        ticks: {
          color: '#64748b',
          font: {
            size: 11
          }
        }
      }
    },
    elements: {
      point: {
        hoverBorderWidth: 3,
        radius: 4,
        hoverRadius: 6
      },
      line: {
        tension: 0.4
      }
    }
  };

  const lossChartOptions = {
    ...chartOptions,
    scales: {
      ...chartOptions.scales,
      y: {
        ...chartOptions.scales.y,
        title: {
          display: true,
          text: 'Loss',
          font: {
            size: 12,
            weight: 'bold'
          },
          color: '#475569',
          padding: 10
        }
      }
    }
  };

  const combinedChartOptions = {
    ...chartOptions,
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Epoch'
        }
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Accuracy (%)'
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: 'Loss (scaled)'
        },
        grid: {
          drawOnChartArea: false,
        },
      }
    }
  };

  // Download CSV results
  const downloadCSV = () => {
    if (!results?.raw_csv) return;
    
    const blob = new Blob([results.raw_csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `experiment_${experimentId}_results.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  // Download JSON results
  const downloadJSON = () => {
    const data = JSON.stringify(results, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `experiment_${experimentId}_results.json`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  // Image handling functions
  const handleImageSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setClassificationResult(null);
      setClassificationError(null);
    }
  };

  const handleClassifyImage = async () => {
    if (!selectedImage || !experimentId) return;

    setIsClassifying(true);
    setClassificationError(null);

    try {
      // Auto-detection: backend will determine the correct model architecture
      const result = await ApiService.classifyImage(experimentId, selectedImage);

      if (result.success) {
        setClassificationResult(result.result);
      } else {
        setClassificationError(result.error);
      }
    } catch (error) {
      setClassificationError(error.message);
    } finally {
      setIsClassifying(false);
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    setClassificationResult(null);
    setClassificationError(null);
  };

  if (!results) {
    return (
      <div className="results-view">
        <div className="no-results">
          <h3>No Results Yet</h3>
          <p>Start an experiment to see results here</p>
        </div>
      </div>
    );
  }

  const { epochs, accuracy, loss } = results.results;
  const finalAccuracy = accuracy[accuracy.length - 1];
  const finalLoss = loss[loss.length - 1];
  const maxAccuracy = Math.max(...accuracy);
  const accuracyDrop = maxAccuracy - finalAccuracy;

  return (
    <div className="results-view">
      <div className="results-header">
        <h2>Experiment Results</h2>
        {experimentId && (
          <span className="experiment-id">{experimentId}</span>
        )}
      </div>

      {/* Tab Navigation */}
      <div className="tab-nav">
        <button 
          className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button 
          className={`tab ${activeTab === 'charts' ? 'active' : ''}`}
          onClick={() => setActiveTab('charts')}
        >
          Charts
        </button>
        <button 
          className={`tab ${activeTab === 'workers' ? 'active' : ''}`}
          onClick={() => setActiveTab('workers')}
        >
          Workers
        </button>
        <button 
          className={`tab ${activeTab === 'data' ? 'active' : ''}`}
          onClick={() => setActiveTab('data')}
        >
          Data
        </button>
        <button 
          className={`tab ${activeTab === 'image-test' ? 'active' : ''}`}
          onClick={() => setActiveTab('image-test')}
        >
          Image Testing
        </button>
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'overview' && (
          <div className="overview-tab">
            <div className="metrics-grid">
              <div className="metric-card">
                <h4>Final Accuracy</h4>
                <span className="metric-value">{finalAccuracy.toFixed(2)}%</span>
              </div>
              <div className="metric-card">
                <h4>Final Loss</h4>
                <span className="metric-value">{finalLoss.toFixed(3)}</span>
              </div>
              <div className="metric-card">
                <h4>Peak Accuracy</h4>
                <span className="metric-value">{maxAccuracy.toFixed(2)}%</span>
              </div>
              <div className="metric-card">
                <h4>Training Epochs</h4>
                <span className="metric-value">{epochs.length}</span>
              </div>
            </div>
            
            <div className="summary-card">
              <h4>Analysis Summary</h4>
              <div className="summary-content">
                <p>
                  Training completed over {epochs.length} epochs with final accuracy of {finalAccuracy.toFixed(2)}% and loss of {finalLoss.toFixed(3)}.
                </p>
                {accuracyDrop > 2 && (
                  <p className="warning">
                    <strong>Attack Detected:</strong> Accuracy dropped {accuracyDrop.toFixed(2)}% from peak, indicating attack impact.
                  </p>
                )}
                {accuracyDrop <= 2 && (
                  <p className="success">
                    <strong>Stable Training:</strong> Consistent performance maintained throughout training.
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'charts' && (
          <div className="charts-tab">
            {chartData && (
              <>
                <div className="chart-container">
                  <h4>Model Accuracy</h4>
                  <div className="chart-wrapper">
                    <Line data={chartData.accuracy} options={chartOptions} />
                  </div>
                </div>
                
                <div className="chart-container">
                  <h4>Training Loss</h4>
                  <div className="chart-wrapper">
                    <Line data={chartData.loss} options={lossChartOptions} />
                  </div>
                </div>

                <div className="chart-container">
                  <h4>Combined View</h4>
                  <div className="chart-wrapper">
                    <Line data={chartData.combined} options={combinedChartOptions} />
                  </div>
                </div>
              </>
            )}
            {!chartData && (
              <div className="no-chart-data">
                <p>No chart data available</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'workers' && (
          <div className="workers-tab">
            <h4>Worker Selection Per Epoch</h4>
            {results.worker_selection ? (
              <div className="worker-selection-container">
                <div className="worker-selection-grid">
                  {results.worker_selection.map((workers, epoch) => (
                    <div key={epoch} className="epoch-workers">
                      <h5>Epoch {epoch + 1}</h5>
                      <div className="workers-list">
                        {workers.map(worker => (
                          <span 
                            key={worker} 
                            className={`worker-badge ${
                              results.poisoned_workers?.includes(worker) ? 'poisoned' : 'clean'
                            }`}
                            title={
                              results.poisoned_workers?.includes(worker) 
                                ? 'Poisoned Worker (Attacker)' 
                                : 'Clean Worker'
                            }
                          >
                            Worker {worker}
                            {results.poisoned_workers?.includes(worker) && ' [P]'}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <p>Worker selection data not available</p>
            )}
          </div>
        )}

        {activeTab === 'data' && (
          <div className="data-tab">
            <div className="download-section">
              <h4>Export Results</h4>
              <div className="download-buttons">
                <button onClick={downloadCSV} className="btn btn-secondary">
                  Download CSV
                </button>
                <button onClick={downloadJSON} className="btn btn-secondary">
                  Download JSON
                </button>
              </div>
            </div>

            {/* Experiment Configuration */}
            <div className="config-section">
              <h4>Experiment Configuration</h4>
              <div className="config-grid">
                <div className="config-item">
                  <span className="config-label">Poisoned Workers:</span>
                  <span className="config-value">{results.config?.num_poisoned_workers || 0}</span>
                </div>
                <div className="config-item">
                  <span className="config-label">Replacement Method:</span>
                  <span className="config-value">{results.config?.replacement_method || 'N/A'}</span>
                </div>
                <div className="config-item">
                  <span className="config-label">Selection Strategy:</span>
                  <span className="config-value">{results.config?.selection_strategy || 'N/A'}</span>
                </div>
                <div className="config-item">
                  <span className="config-label">Workers per Round:</span>
                  <span className="config-value">{results.config?.workers_per_round || 'N/A'}</span>
                </div>
                <div className="config-item">
                  <span className="config-label">Training Epochs:</span>
                  <span className="config-value">{results.metadata?.total_epochs || epochs?.length || 'N/A'}</span>
                </div>
                <div className="config-item">
                  <span className="config-label">Dataset:</span>
                  <span className="config-value">{results.config?.dataset || 'Fashion-MNIST'}</span>
                </div>
                <div className="config-item">
                  <span className="config-label">Duration:</span>
                  <span className="config-value">{results.duration || 'N/A'}</span>
                </div>
              </div>
            </div>
            
            <div className="raw-data-section">
              <h4>Raw Training Data</h4>
              <div className="data-table">
                <table>
                  <thead>
                    <tr>
                      <th>Epoch</th>
                      <th>Accuracy (%)</th>
                      <th>Loss</th>
                      <th>Acc. Change</th>
                      <th>Loss Change</th>
                    </tr>
                  </thead>
                  <tbody>
                    {epochs && accuracy && loss && epochs.map((epoch, i) => (
                      <tr key={epoch}>
                        <td>{epoch}</td>
                        <td>{accuracy[i] != null ? accuracy[i].toFixed(2) : 'N/A'}%</td>
                        <td>{loss[i] != null ? loss[i].toFixed(4) : 'N/A'}</td>
                        <td className={i > 0 && accuracy[i] != null && accuracy[i-1] != null ? (accuracy[i] > accuracy[i-1] ? 'positive' : 'negative') : ''}>
                          {i > 0 && accuracy[i] != null && accuracy[i-1] != null ? `${(accuracy[i] - accuracy[i-1]) > 0 ? '+' : ''}${(accuracy[i] - accuracy[i-1]).toFixed(2)}%` : '-'}
                        </td>
                        <td className={i > 0 && loss[i] != null && loss[i-1] != null ? (loss[i] < loss[i-1] ? 'positive' : 'negative') : ''}>
                          {i > 0 && loss[i] != null && loss[i-1] != null ? `${(loss[i] - loss[i-1]) > 0 ? '+' : ''}${(loss[i] - loss[i-1]).toFixed(4)}` : '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'image-test' && (
          <div className="image-test-tab">
            <div className="image-testing-container">
              <h4>Test This Experiment's Trained Model
                <Tooltip content="Upload images to see how THIS experiment's model classifies them. This demonstrates the real impact of attacks on model accuracy. Clean experiments (0 poisoned workers) will classify accurately, while poisoned experiments will show degraded performance.">
                  <span className="help-icon">?</span>
                </Tooltip>
              </h4>
              
              {/* Experiment Info */}
              <div className="model-info-section">
                <div className="info-card">
                  <h5>Using Model From:</h5>
                  <p className="experiment-id-display">{experimentId}</p>
                </div>
                <div className="info-card">
                  <h5>Attack Configuration:</h5>
                  <p><strong>Poisoned Workers:</strong> {results.config?.num_poisoned_workers || 0}</p>
                  <p><strong>Attack Method:</strong> {results.config?.replacement_method || 'N/A'}</p>
                </div>
                {results.config?.num_poisoned_workers === 0 && (
                  <div className="clean-model-badge">
                    ✓ Clean Model (No Attack)
                  </div>
                )}
                {results.config?.num_poisoned_workers > 0 && (
                  <div className="poisoned-model-badge">
                    ⚠ Poisoned Model ({results.config.num_poisoned_workers} attackers)
                  </div>
                )}
              </div>
              
              <div className="image-upload-section">
                <div className="upload-area">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageSelect}
                    id="image-upload"
                    style={{ display: 'none' }}
                  />
                  <label htmlFor="image-upload" className="upload-button">
                    {selectedImage ? 'Change Image' : 'Select Image'}
                  </label>
                  
                  {selectedImage && (
                    <div className="selected-image-info">
                      <p>Selected: {selectedImage.name}</p>
                      <button onClick={clearImage} className="clear-button">
                        Clear
                      </button>
                    </div>
                  )}
                </div>
                
                <button 
                  onClick={handleClassifyImage}
                  disabled={!selectedImage || isClassifying}
                  className="classify-button"
                >
                  {isClassifying ? 'Classifying with Experiment Model...' : 'Classify Image'}
                </button>
              </div>

              {classificationError && (
                <div className="classification-error">
                  <h5>Error</h5>
                  <p>{classificationError}</p>
                </div>
              )}

              {classificationResult && (
                <div className="classification-results">
                  <h5>Classification Result</h5>
                  
                  <div className="result-summary">
                    <div className="predicted-class">
                      <span className="label">Predicted:</span>
                      <span className="value">{classificationResult.class_names[classificationResult.predicted_class]}</span>
                      <span className="confidence">({(classificationResult.confidence * 100).toFixed(1)}% confidence)</span>
                    </div>
                    
                    <div className="dataset-info">
                      <span className="label">Model Type:</span>
                      <span className="value">{classificationResult.dataset_type === 'fashion_mnist' ? 'Fashion-MNIST (Digits/Clothing)' : 'CIFAR-10 (Objects)'}</span>
                    </div>
                  </div>

                  <div className="confidence-bars">
                    <h6>All Predictions:</h6>
                    {classificationResult.all_probabilities.map((prob, idx) => (
                      <div key={idx} className="confidence-bar">
                        <span className="class-name">{classificationResult.class_names[idx]}</span>
                        <div className="bar-container">
                          <div 
                            className={`bar ${idx === classificationResult.predicted_class ? 'predicted' : ''}`}
                            style={{ width: `${prob * 100}%` }}
                          ></div>
                        </div>
                        <span className="percentage">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>

                  {results.config?.num_poisoned_workers > 0 && (
                    <div className="attack-notice">
                      <div className="warning-icon">⚠️</div>
                      <div className="warning-text">
                        <strong>Attack Impact:</strong> This experiment used {results.config.num_poisoned_workers} poisoned workers 
                        with "{results.config.replacement_method}" attack. The model's predictions may be incorrect due to poisoning.
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultsView;