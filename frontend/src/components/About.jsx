import React from 'react';
import { motion } from 'framer-motion';
import './About.css';

function About() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.15 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5 }
    }
  };

  return (
    <div className="about-page">
      {/* Back Button */}
      <motion.div 
        className="back-button-container"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5 }}
      >
        <button 
          className="back-button"
          onClick={() => window.location.href = '/'}
          aria-label="Back to homepage"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back
        </button>
      </motion.div>

      {/* Hero */}
      <motion.div 
        className="about-hero"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <h1 className="about-title">
          <svg className="title-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
          </svg>
          Federated Learning Security Research
        </h1>
        <p className="about-subtitle">
          Investigating Privacy-Preserving Machine Learning & Adversarial Defense Mechanisms
        </p>
      </motion.div>

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="about-content"
      >
        {/* What is This Project */}
        <motion.section variants={itemVariants} className="about-section">
          <div className="section-header">
            <svg className="section-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <h2>What This Project Does</h2>
          </div>
          <div className="content-card">
            <p className="lead-text">
              This platform <strong>simulates federated learning systems under attack</strong> and tests defense mechanisms to protect collaborative AI training from poisoning attacks.
            </p>
            <div className="key-points">
              <div className="key-point">
                <span className="point-icon">🎯</span>
                <div>
                  <strong>Purpose:</strong> Research how malicious participants can corrupt federated models and develop defenses to prevent it
                </div>
              </div>
              <div className="key-point">
                <span className="point-icon">🔬</span>
                <div>
                  <strong>Method:</strong> Run controlled experiments with configurable attacks and defenses on real datasets (CIFAR-10, Fashion-MNIST)
                </div>
              </div>
              <div className="key-point">
                <span className="point-icon">📊</span>
                <div>
                  <strong>Output:</strong> Visualize attack impact, defense effectiveness, and model accuracy across training rounds
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Federated Learning Explained */}
        <motion.section variants={itemVariants} className="about-section">
          <div className="section-header">
            <svg className="section-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
            </svg>
            <h2>Federated Learning</h2>
          </div>
          <div className="content-card">
            <p>
              <strong>Distributed training</strong> where multiple clients (phones, hospitals, IoT devices) collaboratively train a shared model <strong>without sharing raw data</strong>.
            </p>
            <div className="fl-steps">
              <div className="fl-step">
                <div className="step-number">1</div>
                <div className="step-content">
                  <strong>Server sends</strong> initial model to clients
                </div>
              </div>
              <div className="fl-step">
                <div className="step-number">2</div>
                <div className="step-content">
                  <strong>Clients train</strong> locally on private data
                </div>
              </div>
              <div className="fl-step">
                <div className="step-number">3</div>
                <div className="step-content">
                  <strong>Clients send</strong> model updates (not data) to server
                </div>
              </div>
              <div className="fl-step">
                <div className="step-number">4</div>
                <div className="step-content">
                  <strong>Server aggregates</strong> updates using FedAvg
                </div>
              </div>
              <div className="fl-step">
                <div className="step-number">5</div>
                <div className="step-content">
                  <strong>Repeat</strong> until model converges
                </div>
              </div>
            </div>
            <div className="benefit-badge">
              ✓ Data stays private • Collaborative learning • Scalable
            </div>
          </div>
        </motion.section>

        {/* Label Flipping Attacks */}
        <motion.section variants={itemVariants} className="about-section">
          <div className="section-header">
            <svg className="section-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <h2>Label Flipping Attacks</h2>
          </div>
          <div className="content-card">
            <p><strong>Malicious clients</strong> intentionally corrupt training labels to poison the global model.</p>
            
            <div className="attack-examples">
              <div className="attack-card severe">
                <div className="attack-badge">High Impact</div>
                <h4>Replace 1 → 9</h4>
                <p>All class 1 images labeled as class 9</p>
                <p className="impact">Impact: 85% → 60% accuracy</p>
              </div>
              <div className="attack-card moderate">
                <div className="attack-badge">Medium Impact</div>
                <h4>Replace 0 → 2</h4>
                <p>Similar classes confused</p>
                <p className="impact">Impact: 85% → 72% accuracy</p>
              </div>
              <div className="attack-card baseline">
                <div className="attack-badge">Baseline</div>
                <h4>No Attack</h4>
                <p>Clean training</p>
                <p className="impact">Accuracy: ~85-92%</p>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Worker Selection Strategies */}
        <motion.section variants={itemVariants} className="about-section">
          <div className="section-header">
            <svg className="section-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
            <h2>Worker Selection Strategies</h2>
          </div>
          <div className="content-card">
            <p><strong>Controls which clients</strong> participate in each training round.</p>
            
            <div className="selection-grid">
              <div className="selection-card">
                <h4>Random Selection</h4>
                <p>Uniform sampling each round</p>
              </div>
              <div className="selection-card">
                <h4>Before Breakpoint</h4>
                <p>Favor early-round clients</p>
              </div>
              <div className="selection-card">
                <h4>After Breakpoint</h4>
                <p>Shift to later clients</p>
              </div>
              <div className="selection-card">
                <h4>Poisoner Probability</h4>
                <p>Bias toward/against attackers</p>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Datasets & Models */}
        <motion.section variants={itemVariants} className="about-section">
          <div className="section-header">
            <svg className="section-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
            </svg>
            <h2>Datasets & Models</h2>
          </div>
          <div className="content-card">
            <div className="dataset-grid">
              <div className="dataset-card">
                <div className="dataset-icon">🖼️</div>
                <h4>CIFAR-10</h4>
                <p>60K color images (32×32)</p>
                <p>10 classes: airplane, car, bird, cat, etc.</p>
              </div>
              <div className="dataset-card">
                <div className="dataset-icon">👕</div>
                <h4>Fashion-MNIST</h4>
                <p>70K grayscale images (28×28)</p>
                <p>10 classes: shirt, dress, shoes, etc.</p>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Defense Mechanisms */}
        <motion.section variants={itemVariants} className="about-section">
          <div className="section-header">
            <svg className="section-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
            <h2>Defense Mechanisms</h2>
          </div>
          <div className="content-card">
            <p><strong>Protect the model</strong> by filtering or adjusting malicious updates before aggregation.</p>
            
            <div className="defense-grid">
              <div className="defense-card">
                <div className="defense-icon">🛡️</div>
                <h4>Byzantine-Robust</h4>
                <p>Use median instead of mean to ignore outliers</p>
                <div className="defense-metric">70-90% mitigation</div>
              </div>
              <div className="defense-card">
                <div className="defense-icon">🔍</div>
                <h4>Anomaly Detection</h4>
                <p>Flag clients with abnormal gradient patterns</p>
                <div className="defense-metric">60-80% mitigation</div>
              </div>
              <div className="defense-card">
                <div className="defense-icon">✂️</div>
                <h4>Gradient Clipping</h4>
                <p>Limit update magnitude to reduce attack impact</p>
                <div className="defense-metric">40-60% mitigation</div>
              </div>
              <div className="defense-card">
                <div className="defense-icon">🚫</div>
                <h4>Client Filtering</h4>
                <p>Blacklist suspicious clients over time</p>
                <div className="defense-metric">50-70% mitigation</div>
              </div>
            </div>

            <div className="comparison-box">
              <div className="comparison-side no-defense">
                <h5>❌ Without Defense</h5>
                <ul>
                  <li>Accuracy drops to 60-65%</li>
                  <li>Model permanently damaged</li>
                  <li>Targeted misclassification</li>
                </ul>
              </div>
              <div className="comparison-side with-defense">
                <h5>✅ With Defense</h5>
                <ul>
                  <li>Accuracy stays 80-86%</li>
                  <li>Attack mitigated</li>
                  <li>Model remains robust</li>
                </ul>
              </div>
            </div>
          </div>
        </motion.section>

        {/* How to Use */}
        <motion.section variants={itemVariants} className="about-section">
          <div className="section-header">
            <svg className="section-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <h2>How to Use This Platform</h2>
          </div>
          <div className="content-card">
            <div className="usage-steps">
              <div className="usage-step">
                <div className="usage-number">1</div>
                <div>
                  <strong>Configure Experiment</strong>
                  <p>Set dataset, poisoned workers, attack method, selection strategy</p>
                </div>
              </div>
              <div className="usage-step">
                <div className="usage-number">2</div>
                <div>
                  <strong>Enable/Disable Defense</strong>
                  <p>Toggle defense and choose method (Byzantine-Robust, Anomaly Detection, etc.)</p>
                </div>
              </div>
              <div className="usage-step">
                <div className="usage-number">3</div>
                <div>
                  <strong>Run Experiment</strong>
                  <p>System trains federated model over multiple rounds</p>
                </div>
              </div>
              <div className="usage-step">
                <div className="usage-number">4</div>
                <div>
                  <strong>Analyze Results</strong>
                  <p>View accuracy charts, worker selections, and defense effectiveness</p>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Call to Action */}
        <motion.section 
          variants={itemVariants} 
          className="about-section cta-section"
        >
          <div className="cta-card">
            <h3>Ready to Start Experimenting?</h3>
            <p>Explore how attacks impact federated learning and test defense mechanisms</p>
            <button 
              className="cta-button"
              onClick={() => window.location.href = '/'}
            >
              Launch Experiments
              <svg className="button-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </button>
          </div>
        </motion.section>
      </motion.div>
    </div>
  );
}

export default About;