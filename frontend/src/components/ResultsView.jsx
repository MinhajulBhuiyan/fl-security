import React, { useState, useMemo } from 'react';
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
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderWidth: 3,
            fill: true,
            tension: 0.4,
            pointBackgroundColor: '#10b981',
            pointBorderColor: '#ffffff',
            pointBorderWidth: 2,
            pointRadius: 5,
            pointHoverRadius: 7,
          }
        ]
      },
      loss: {
        labels: epochs,
        datasets: [
          {
            label: 'Training Loss',
            data: loss,
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            borderWidth: 3,
            fill: true,
            tension: 0.4,
            pointBackgroundColor: '#ef4444',
            pointBorderColor: '#ffffff',
            pointBorderWidth: 2,
            pointRadius: 5,
            pointHoverRadius: 7,
          }
        ]
      },
      combined: {
        labels: epochs,
        datasets: [
          {
            label: 'Accuracy (%)',
            data: accuracy,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderWidth: 2,
            yAxisID: 'y',
            tension: 0.4,
          },
          {
            label: 'Loss',
            data: loss.map(l => l * 20), // Scale loss for dual axis
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            borderWidth: 2,
            yAxisID: 'y1',
            tension: 0.4,
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
        labels: {
          font: {
            size: 12,
            family: 'system-ui'
          }
        }
      },
      title: {
        display: false
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: '#374151',
        borderWidth: 1,
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
            size: 14,
            weight: 'bold'
          }
        },
        grid: {
          color: 'rgba(55, 65, 81, 0.3)'
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Accuracy (%)',
          font: {
            size: 14,
            weight: 'bold'
          }
        },
        grid: {
          color: 'rgba(55, 65, 81, 0.3)'
        }
      }
    },
    elements: {
      point: {
        hoverBorderWidth: 3
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
            size: 14,
            weight: 'bold'
          }
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
            <div className="workers-summary">
              <h4>Worker Participation Summary</h4>
              {results.poisoned_workers && results.poisoned_workers.length > 0 && (
                <div className="poisoned-workers-info">
                  <h5>Poisoned Workers (Attackers)</h5>
                  <div className="poisoned-workers-list">
                    {results.poisoned_workers.map(worker => (
                      <span key={worker} className="worker-badge poisoned">
                        Worker {worker}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
            
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
                  <span className="config-label">Quick Mode:</span>
                  <span className="config-value">{results.config?.quick_mode ? 'Yes' : 'No'}</span>
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
                    {epochs.map((epoch, i) => (
                      <tr key={epoch}>
                        <td>{epoch}</td>
                        <td>{accuracy[i].toFixed(2)}%</td>
                        <td>{loss[i].toFixed(4)}</td>
                        <td className={i > 0 ? (accuracy[i] > accuracy[i-1] ? 'positive' : 'negative') : ''}>
                          {i > 0 ? `${(accuracy[i] - accuracy[i-1]) > 0 ? '+' : ''}${(accuracy[i] - accuracy[i-1]).toFixed(2)}%` : '-'}
                        </td>
                        <td className={i > 0 ? (loss[i] < loss[i-1] ? 'positive' : 'negative') : ''}>
                          {i > 0 ? `${(loss[i] - loss[i-1]) > 0 ? '+' : ''}${(loss[i] - loss[i-1]).toFixed(4)}` : '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultsView;