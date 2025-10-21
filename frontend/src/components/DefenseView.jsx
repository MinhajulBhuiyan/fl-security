import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import './DefenseView.css';

const DefenseView = ({ baselineResults, defenseResults, config }) => {
  const [showComparison, setShowComparison] = useState(true);
  const [selectedMetric, setSelectedMetric] = useState('accuracy');
  const [maliciousStats, setMaliciousStats] = useState(null);

  useEffect(() => {
    if (defenseResults) {
      calculateMaliciousStats();
    }
  }, [defenseResults]);

  const calculateMaliciousStats = () => {
    if (!defenseResults?.worker_selection || !defenseResults?.poisoned_workers) {
      return;
    }

    const poisonedSet = new Set(defenseResults.poisoned_workers);
    const stats = {
      totalRounds: defenseResults.worker_selection.length,
      poisonedParticipation: 0,
      blockedCount: 0,
      influenceByRound: []
    };

    defenseResults.worker_selection.forEach((workers, round) => {
      const poisonedInRound = workers.filter(w => poisonedSet.has(w));
      stats.poisonedParticipation += poisonedInRound.length;
      stats.influenceByRound.push({
        round: round + 1,
        count: poisonedInRound.length,
        percentage: (poisonedInRound.length / workers.length) * 100
      });
    });

    setMaliciousStats(stats);
  };

  const getComparisonChartData = () => {
    if (!baselineResults || !defenseResults) return null;

    const metric = selectedMetric === 'accuracy' ? 'accuracy' : 'loss';
    
    return {
      labels: baselineResults.results.epochs,
      datasets: [
        {
          label: 'No Defense',
          data: baselineResults.results[metric],
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.1)',
          borderWidth: 2,
          tension: 0.3
        },
        {
          label: 'With Defense',
          data: defenseResults.results[metric],
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          borderWidth: 2,
          tension: 0.3
        }
      ]
    };
  };

  const getMaliciousInfluenceChartData = () => {
    if (!maliciousStats) return null;

    return {
      labels: maliciousStats.influenceByRound.map(r => `Round ${r.round}`),
      datasets: [
        {
          label: 'Malicious Participants (%)',
          data: maliciousStats.influenceByRound.map(r => r.percentage),
          borderColor: 'rgb(255, 159, 64)',
          backgroundColor: 'rgba(255, 159, 64, 0.1)',
          borderWidth: 2,
          fill: true
        }
      ]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          font: { size: 12 }
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false
      }
    },
    scales: {
      y: {
        beginAtZero: selectedMetric === 'loss',
        max: selectedMetric === 'accuracy' ? 100 : undefined,
        title: {
          display: true,
          text: selectedMetric === 'accuracy' ? 'Accuracy (%)' : 'Loss'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Epoch'
        }
      }
    }
  };

  const calculateDefenseImpact = () => {
    if (!baselineResults || !defenseResults) return null;

    const baselineFinal = baselineResults.results.accuracy[baselineResults.results.accuracy.length - 1];
    const defenseFinal = defenseResults.results.accuracy[defenseResults.results.accuracy.length - 1];
    const improvement = defenseFinal - baselineFinal;

    return {
      baselineFinal,
      defenseFinal,
      improvement,
      improvementPercent: ((improvement / baselineFinal) * 100).toFixed(2)
    };
  };

  const impact = calculateDefenseImpact();

  return (
    <div className="defense-view">
      <div className="defense-header">
        <h2>Defense Mechanism Analysis</h2>
        <div className="defense-controls">
          <button 
            className={`toggle-btn ${showComparison ? 'active' : ''}`}
            onClick={() => setShowComparison(!showComparison)}
          >
            {showComparison ? 'Hide' : 'Show'} Comparison
          </button>
          <select 
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            className="metric-select"
          >
            <option value="accuracy">Accuracy</option>
            <option value="loss">Loss</option>
          </select>
        </div>
      </div>

      {/* Defense Status & Impact Summary */}
      {impact && (
        <div className="defense-summary">
          <div className="summary-card baseline">
            <h4>No Defense</h4>
            <div className="metric-value">{impact.baselineFinal.toFixed(2)}%</div>
            <div className="metric-label">Final Accuracy</div>
          </div>
          <div className="summary-card defense">
            <h4>With Defense</h4>
            <div className="metric-value">{impact.defenseFinal.toFixed(2)}%</div>
            <div className="metric-label">Final Accuracy</div>
          </div>
          <div className={`summary-card impact ${impact.improvement > 0 ? 'positive' : 'negative'}`}>
            <h4>Defense Impact</h4>
            <div className="metric-value">
              {impact.improvement > 0 ? '+' : ''}{impact.improvement.toFixed(2)}%
            </div>
            <div className="metric-label">
              ({impact.improvement > 0 ? '+' : ''}{impact.improvementPercent}% improvement)
            </div>
            {impact.improvement > 0 && (
              <div className="defense-badge success">
                ✓ Attack Mitigated
              </div>
            )}
          </div>
        </div>
      )}

      {/* Comparison Chart */}
      {showComparison && baselineResults && defenseResults && (
        <div className="chart-container">
          <h3>Performance Comparison: {selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1)}</h3>
          <div className="chart-wrapper">
            <Line data={getComparisonChartData()} options={chartOptions} />
          </div>
        </div>
      )}

      {/* Malicious Participant Tracking */}
      {maliciousStats && (
        <div className="malicious-tracking">
          <h3>Malicious Participant Activity</h3>
          
          <div className="tracking-stats">
            <div className="stat-card">
              <div className="stat-value">{defenseResults.poisoned_workers.length}</div>
              <div className="stat-label">Total Malicious Clients</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{maliciousStats.totalRounds}</div>
              <div className="stat-label">Training Rounds</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{maliciousStats.poisonedParticipation}</div>
              <div className="stat-label">Malicious Participations</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {((maliciousStats.poisonedParticipation / (maliciousStats.totalRounds * config.workers_per_round)) * 100).toFixed(1)}%
              </div>
              <div className="stat-label">Participation Rate</div>
            </div>
          </div>

          <div className="chart-container">
            <h4>Malicious Influence per Round</h4>
            <div className="chart-wrapper">
              <Line data={getMaliciousInfluenceChartData()} options={{
                ...chartOptions,
                scales: {
                  y: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: 'Malicious Participants (%)' }
                  },
                  x: {
                    title: { display: true, text: 'Training Round' }
                  }
                }
              }} />
            </div>
          </div>

          {/* Malicious Workers List */}
          <div className="malicious-workers-list">
            <h4>Identified Malicious Workers</h4>
            <div className="workers-grid">
              {defenseResults.poisoned_workers.map(workerId => (
                <div key={workerId} className="worker-badge malicious">
                  Worker #{workerId}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Defense Configuration Info */}
      {config && (
        <div className="defense-config">
          <h4>Defense Configuration</h4>
          <div className="config-details">
            <div className="config-item">
              <span className="config-label">Defense Method:</span>
              <span className="config-value">{config.defense_method || 'Byzantine-Robust Aggregation'}</span>
            </div>
            <div className="config-item">
              <span className="config-label">Attack Method:</span>
              <span className="config-value">{config.replacement_method}</span>
            </div>
            <div className="config-item">
              <span className="config-label">Selection Strategy:</span>
              <span className="config-value">{config.selection_strategy}</span>
            </div>
            <div className="config-item">
              <span className="config-label">Poisoned Workers:</span>
              <span className="config-value">{config.num_poisoned_workers}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DefenseView;
