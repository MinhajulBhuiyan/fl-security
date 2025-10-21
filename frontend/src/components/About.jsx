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
                <svg className="point-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <div>
                  <strong>Purpose:</strong> Research how malicious participants can corrupt federated models and develop defenses to prevent it
                </div>
              </div>
              <div className="key-point">
                <svg className="point-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
                </svg>
                <div>
                  <strong>Method:</strong> Run controlled experiments with configurable attacks and defenses on real datasets (CIFAR-10, Fashion-MNIST)
                </div>
              </div>
              <div className="key-point">
                <svg className="point-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
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
              <svg className="badge-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
              </svg>
              Data stays private • Collaborative learning • Scalable
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
            <p className="section-intro">
              In label flipping attacks, malicious participants intentionally mislabel their training data to poison the global model. 
              This causes the model to learn incorrect patterns, leading to targeted misclassifications. The severity depends on which 
              classes are swapped and how many poisoned workers participate.
            </p>
            
            <div className="attack-grid">
              <div className="attack-card baseline">
                <div className="attack-header">
                  <h4>No Attack</h4>
                  <span className="severity-badge baseline">Baseline</span>
                </div>
                <p className="attack-description">
                  Clean training with all labels correctly assigned. This establishes the performance benchmark.
                </p>
                <div className="attack-example">
                  <strong>Example:</strong> CIFAR-10 airplane images correctly labeled as class 0, car images as class 1.
                </div>
                <div className="attack-impact">
                  <span className="impact-label">Expected Accuracy:</span>
                  <span className="impact-value">85-92%</span>
                </div>
              </div>

              <div className="attack-card severe">
                <div className="attack-header">
                  <h4>Replace 0 → 9</h4>
                  <span className="severity-badge severe">High Impact</span>
                </div>
                <p className="attack-description">
                  All instances of class 0 are mislabeled as class 9, creating confusion between dissimilar categories.
                </p>
                <div className="attack-example">
                  <strong>Example:</strong> Airplanes (0) systematically labeled as trucks (9). Model learns to associate airplane features with truck labels.
                </div>
                <div className="attack-impact">
                  <span className="impact-label">Accuracy Drop:</span>
                  <span className="impact-value">85% → 60-70%</span>
                </div>
              </div>

              <div className="attack-card critical">
                <div className="attack-header">
                  <h4>Replace 1 → 9</h4>
                  <span className="severity-badge critical">Critical</span>
                </div>
                <p className="attack-description">
                  Class 1 systematically mislabeled as class 9, causing severe confusion in vehicle classification.
                </p>
                <div className="attack-example">
                  <strong>Example:</strong> Cars (1) labeled as trucks (9). Creates backdoor where car images are consistently misclassified.
                </div>
                <div className="attack-impact">
                  <span className="impact-label">Accuracy Drop:</span>
                  <span className="impact-value">85% → 55-65%</span>
                </div>
              </div>

              <div className="attack-card moderate">
                <div className="attack-header">
                  <h4>Replace 0 → 2</h4>
                  <span className="severity-badge moderate">Moderate</span>
                </div>
                <p className="attack-description">
                  Visually similar classes swapped, exploiting feature similarity for subtle poisoning.
                </p>
                <div className="attack-example">
                  <strong>Example:</strong> Airplanes (0) labeled as birds (2). Both have wings and fly, making confusion more plausible.
                </div>
                <div className="attack-impact">
                  <span className="impact-label">Accuracy Drop:</span>
                  <span className="impact-value">85% → 70-80%</span>
                </div>
              </div>

              <div className="attack-card moderate">
                <div className="attack-header">
                  <h4>Replace 4 → 6</h4>
                  <span className="severity-badge moderate">Moderate</span>
                </div>
                <p className="attack-description">
                  Mid-range classes swapped to test model's ability to distinguish subtle features.
                </p>
                <div className="attack-example">
                  <strong>Example:</strong> Deer (4) labeled as frogs (6). Tests if model can differentiate animal types despite size differences.
                </div>
                <div className="attack-impact">
                  <span className="impact-label">Accuracy Drop:</span>
                  <span className="impact-value">85% → 72-82%</span>
                </div>
              </div>

              <div className="attack-card severe">
                <div className="attack-header">
                  <h4>Replace 5 → 3</h4>
                  <span className="severity-badge severe">High Impact</span>
                </div>
                <p className="attack-description">
                  Common pet categories confused, disrupting learned patterns for similar animals.
                </p>
                <div className="attack-example">
                  <strong>Example:</strong> Dogs (5) labeled as cats (3). Both are four-legged pets with similar features, causing high confusion.
                </div>
                <div className="attack-impact">
                  <span className="impact-label">Accuracy Drop:</span>
                  <span className="impact-value">85% → 58-68%</span>
                </div>
              </div>

              <div className="attack-card moderate">
                <div className="attack-header">
                  <h4>Replace 1 → 3</h4>
                  <span className="severity-badge moderate">Moderate</span>
                </div>
                <p className="attack-description">
                  Cross-category confusion between vehicles and animals to create unexpected misclassifications.
                </p>
                <div className="attack-example">
                  <strong>Example:</strong> Cars (1) labeled as cats (3). Tests model robustness against nonsensical label assignments.
                </div>
                <div className="attack-impact">
                  <span className="impact-label">Accuracy Drop:</span>
                  <span className="impact-value">85% → 68-78%</span>
                </div>
              </div>

              <div className="attack-card severe">
                <div className="attack-header">
                  <h4>Replace 6 → 0</h4>
                  <span className="severity-badge severe">High Impact</span>
                </div>
                <p className="attack-description">
                  Reverse mapping to test inverse poisoning effects and model resilience.
                </p>
                <div className="attack-example">
                  <strong>Example:</strong> Frogs (6) labeled as airplanes (0). Completely unrelated classes to maximize model confusion.
                </div>
                <div className="attack-impact">
                  <span className="impact-label">Accuracy Drop:</span>
                  <span className="impact-value">85% → 60-70%</span>
                </div>
              </div>
            </div>

            <div className="attack-summary">
              <h4>Key Insights</h4>
              <ul>
                <li><strong>Dissimilar classes</strong> (e.g., 0→9, 6→0) cause more severe damage than similar classes (e.g., 0→2)</li>
                <li><strong>Visual similarity</strong> makes attacks harder to detect but slightly less impactful</li>
                <li><strong>Multiple poisoners</strong> amplify the attack effect exponentially</li>
                <li><strong>Targeted backdoors</strong> can be created for specific misclassification pairs</li>
              </ul>
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
            <p className="section-intro">
              Worker selection strategies determine which subset of clients participates in each training round. 
              Different strategies affect convergence speed, fairness, and vulnerability to attacks. The server 
              selects a fixed number of workers per round from the total pool of available clients.
            </p>
            
            <div className="strategy-detailed-grid">
              <div className="strategy-detail-card">
                <div className="strategy-header">
                  <h4>Random Selection</h4>
                  <span className="strategy-type">Baseline Strategy</span>
                </div>
                <p className="strategy-description">
                  Each training round randomly samples workers with uniform probability. Every client has an equal 
                  chance of being selected regardless of previous participation.
                </p>
                <div className="strategy-mechanics">
                  <strong>How it works:</strong>
                  <p>From a pool of 100 clients, randomly select 10 different clients each round using uniform distribution.</p>
                </div>
                <div className="strategy-example">
                  <strong>Example:</strong> Round 1: clients [5, 23, 67, 12, 89, 34, 76, 91, 45, 8]. Round 2: clients [34, 56, 2, 78, 91, 45, 13, 67, 88, 29].
                </div>
                <div className="strategy-implications">
                  <strong>Implications:</strong>
                  <ul>
                    <li>Fair distribution ensures all clients contribute over time</li>
                    <li>Baseline for measuring other strategies</li>
                    <li>Poisoners have proportional chance of being selected</li>
                    <li>May take longer to converge due to data distribution variance</li>
                  </ul>
                </div>
              </div>

              <div className="strategy-detail-card">
                <div className="strategy-header">
                  <h4>Before Breakpoint</h4>
                  <span className="strategy-type">Early-Phase Focused</span>
                </div>
                <p className="strategy-description">
                  Prioritizes a stable subset of clients during initial training epochs to accelerate early convergence. 
                  After a breakpoint epoch, gradually shifts to broader client selection.
                </p>
                <div className="strategy-mechanics">
                  <strong>How it works:</strong>
                  <p>Define breakpoint at epoch 5. Epochs 1-5: select 80% from clients 0-40. Epochs 6+: uniform random selection from all clients.</p>
                </div>
                <div className="strategy-example">
                  <strong>Example:</strong> Epoch 3: 8 clients from [0-40], 2 from [41-100]. Epoch 7: all 10 clients from uniform [0-100].
                </div>
                <div className="strategy-implications">
                  <strong>Implications:</strong>
                  <ul>
                    <li>Faster initial convergence with consistent early participants</li>
                    <li>Risk: if early pool contains poisoners, attack impact is amplified</li>
                    <li>Model may overfit to early clients' data distribution</li>
                    <li>Useful when early clients have higher quality data</li>
                  </ul>
                </div>
              </div>

              <div className="strategy-detail-card">
                <div className="strategy-header">
                  <h4>After Breakpoint</h4>
                  <span className="strategy-type">Late-Phase Shift</span>
                </div>
                <p className="strategy-description">
                  Initially uses random selection, then shifts selection bias toward later-indexed clients after 
                  a specified breakpoint to introduce fresh data and reduce overfitting.
                </p>
                <div className="strategy-mechanics">
                  <strong>How it works:</strong>
                  <p>Breakpoint at epoch 10. Epochs 1-10: uniform random. Epochs 11+: prefer clients 50-100 with 70% probability, 30% from 0-49.</p>
                </div>
                <div className="strategy-example">
                  <strong>Example:</strong> Epoch 8: clients uniformly from [0-100]. Epoch 15: 7 clients from [50-100], 3 from [0-49].
                </div>
                <div className="strategy-implications">
                  <strong>Implications:</strong>
                  <ul>
                    <li>Introduces diversity in later training stages</li>
                    <li>Can correct for early overfitting patterns</li>
                    <li>Tests model robustness to distribution shift</li>
                    <li>May destabilize if late clients have poor data quality</li>
                  </ul>
                </div>
              </div>

              <div className="strategy-detail-card">
                <div className="strategy-header">
                  <h4>Poisoner Probability</h4>
                  <span className="strategy-type">Attack-Aware</span>
                </div>
                <p className="strategy-description">
                  Adjusts selection probabilities based on suspected maliciousness. Can be configured to either 
                  oversample poisoners (stress-test attacks) or undersample them (defense simulation).
                </p>
                <div className="strategy-mechanics">
                  <strong>How it works:</strong>
                  <p>Identify 5 suspected poisoners. Attack mode: select poisoners with 80% probability. Defense mode: select poisoners with 10% probability.</p>
                </div>
                <div className="strategy-example">
                  <strong>Example:</strong> Attack mode - 8 of 10 workers are poisoners each round. Defense mode - only 1 of 10 workers is a poisoner.
                </div>
                <div className="strategy-implications">
                  <strong>Implications:</strong>
                  <ul>
                    <li>Stress-test: measures maximum attack impact with frequent poisoner participation</li>
                    <li>Defense mode: simulates client filtering defense effectiveness</li>
                    <li>Enables controlled experiments on attack severity vs. poisoner frequency</li>
                    <li>Requires prior knowledge or detection mechanism to identify poisoners</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="strategy-comparison">
              <h4>Strategy Comparison</h4>
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>Strategy</th>
                    <th>Convergence Speed</th>
                    <th>Fairness</th>
                    <th>Attack Resilience</th>
                    <th>Use Case</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>Random Selection</strong></td>
                    <td>Moderate</td>
                    <td>High</td>
                    <td>Moderate</td>
                    <td>Baseline experiments</td>
                  </tr>
                  <tr>
                    <td><strong>Before Breakpoint</strong></td>
                    <td>Fast (early)</td>
                    <td>Low (early bias)</td>
                    <td>Low if early poisoners</td>
                    <td>Quick initial training</td>
                  </tr>
                  <tr>
                    <td><strong>After Breakpoint</strong></td>
                    <td>Variable</td>
                    <td>Moderate</td>
                    <td>Can improve late-stage</td>
                    <td>Distribution shift testing</td>
                  </tr>
                  <tr>
                    <td><strong>Poisoner Probability</strong></td>
                    <td>Depends on mode</td>
                    <td>Low (intentional bias)</td>
                    <td>High (defense) / Low (attack)</td>
                    <td>Attack impact analysis</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </motion.section>

        {/* Datasets & Models */}
        <motion.section variants={itemVariants} className="about-section">
          <div className="section-header">
            <svg className="section-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
            </svg>
            <h2>Datasets & Neural Network Models</h2>
          </div>
          <div className="content-card">
            <p className="section-intro">
              We use two standard image classification datasets to evaluate federated learning security. 
              Each dataset is paired with a convolutional neural network optimized for its specific characteristics.
            </p>
            
            <div className="dataset-detailed-grid">
              <div className="dataset-detail-card">
                <div className="dataset-header">
                  <h4>CIFAR-10 Dataset</h4>
                  <span className="dataset-badge">Color Images</span>
                </div>
                <div className="dataset-specs">
                  <div className="spec-item">
                    <span className="spec-label">Total Images:</span>
                    <span className="spec-value">60,000 (50K train + 10K test)</span>
                  </div>
                  <div className="spec-item">
                    <span className="spec-label">Resolution:</span>
                    <span className="spec-value">32×32 pixels RGB</span>
                  </div>
                  <div className="spec-item">
                    <span className="spec-label">Classes:</span>
                    <span className="spec-value">10 categories</span>
                  </div>
                </div>
                <div className="class-list">
                  <strong>Class Categories:</strong>
                  <ul>
                    <li><strong>0:</strong> Airplane</li>
                    <li><strong>1:</strong> Automobile</li>
                    <li><strong>2:</strong> Bird</li>
                    <li><strong>3:</strong> Cat</li>
                    <li><strong>4:</strong> Deer</li>
                    <li><strong>5:</strong> Dog</li>
                    <li><strong>6:</strong> Frog</li>
                    <li><strong>7:</strong> Horse</li>
                    <li><strong>8:</strong> Ship</li>
                    <li><strong>9:</strong> Truck</li>
                  </ul>
                </div>
                <div className="model-architecture">
                  <strong>CIFAR-10 CNN Architecture:</strong>
                  <ul>
                    <li><strong>Input Layer:</strong> 32×32×3 (RGB channels)</li>
                    <li><strong>Conv Block 1:</strong> 32 filters (3×3), ReLU, BatchNorm, MaxPool (2×2)</li>
                    <li><strong>Conv Block 2:</strong> 64 filters (3×3), ReLU, BatchNorm, MaxPool (2×2)</li>
                    <li><strong>Conv Block 3:</strong> 128 filters (3×3), ReLU, BatchNorm, MaxPool (2×2)</li>
                    <li><strong>Flatten:</strong> Convert to 1D vector</li>
                    <li><strong>Dense Layer:</strong> 256 units, ReLU, Dropout (0.5)</li>
                    <li><strong>Output Layer:</strong> 10 units, Softmax activation</li>
                  </ul>
                </div>
                <div className="model-performance">
                  <strong>Expected Performance:</strong>
                  <p>Clean model: 85-92% test accuracy. Under attack: 55-80% depending on attack severity.</p>
                </div>
              </div>

              <div className="dataset-detail-card">
                <div className="dataset-header">
                  <h4>Fashion-MNIST Dataset</h4>
                  <span className="dataset-badge">Grayscale Images</span>
                </div>
                <div className="dataset-specs">
                  <div className="spec-item">
                    <span className="spec-label">Total Images:</span>
                    <span className="spec-value">70,000 (60K train + 10K test)</span>
                  </div>
                  <div className="spec-item">
                    <span className="spec-label">Resolution:</span>
                    <span className="spec-value">28×28 pixels grayscale</span>
                  </div>
                  <div className="spec-item">
                    <span className="spec-label">Classes:</span>
                    <span className="spec-value">10 fashion categories</span>
                  </div>
                </div>
                <div className="class-list">
                  <strong>Class Categories:</strong>
                  <ul>
                    <li><strong>0:</strong> T-shirt/Top</li>
                    <li><strong>1:</strong> Trouser</li>
                    <li><strong>2:</strong> Pullover</li>
                    <li><strong>3:</strong> Dress</li>
                    <li><strong>4:</strong> Coat</li>
                    <li><strong>5:</strong> Sandal</li>
                    <li><strong>6:</strong> Shirt</li>
                    <li><strong>7:</strong> Sneaker</li>
                    <li><strong>8:</strong> Bag</li>
                    <li><strong>9:</strong> Ankle Boot</li>
                  </ul>
                </div>
                <div className="model-architecture">
                  <strong>Fashion-MNIST CNN Architecture:</strong>
                  <ul>
                    <li><strong>Input Layer:</strong> 28×28×1 (single grayscale channel)</li>
                    <li><strong>Conv Block 1:</strong> 32 filters (3×3), ReLU, BatchNorm, MaxPool (2×2)</li>
                    <li><strong>Conv Block 2:</strong> 64 filters (3×3), ReLU, BatchNorm, MaxPool (2×2)</li>
                    <li><strong>Flatten:</strong> Convert to 1D vector</li>
                    <li><strong>Dense Layer:</strong> 128 units, ReLU, Dropout (0.4)</li>
                    <li><strong>Output Layer:</strong> 10 units, Softmax activation</li>
                  </ul>
                </div>
                <div className="model-performance">
                  <strong>Expected Performance:</strong>
                  <p>Clean model: 88-93% test accuracy. Under attack: 60-85% depending on attack severity.</p>
                </div>
              </div>
            </div>

            <div className="dataset-rationale">
              <h4>Why These Datasets?</h4>
              <ul>
                <li><strong>Standard Benchmarks:</strong> Widely used in federated learning research for reproducibility</li>
                <li><strong>Balanced Classes:</strong> Each class has equal representation (6,000 samples in CIFAR-10, 7,000 in Fashion-MNIST)</li>
                <li><strong>Attack Visibility:</strong> Label flipping attacks show clear, measurable impact on accuracy</li>
                <li><strong>Computational Efficiency:</strong> Small image sizes enable rapid experimentation</li>
                <li><strong>Real-World Relevance:</strong> Object recognition (CIFAR-10) and fashion classification (Fashion-MNIST) mirror practical applications</li>
              </ul>
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
            <p className="section-intro">
              Our framework implements four defense strategies to protect federated learning from malicious updates. 
              Each defense uses different techniques to detect and mitigate poisoned gradients.
            </p>
            
            <div className="defense-detailed-grid">
              <div className="defense-detail-card">
                <div className="defense-header">
                  <h4>Byzantine-Robust Aggregation</h4>
                  <span className="defense-type">Statistical</span>
                </div>
                <div className="defense-description">
                  Uses coordinate-wise median instead of mean to aggregate client updates, making the system resilient 
                  to Byzantine failures where malicious clients send arbitrary values.
                </div>
                <div className="defense-mechanics">
                  <strong>How It Works:</strong>
                  <p>
                    Instead of averaging gradients (mean aggregation), compute the median value for each parameter 
                    independently. Outlier gradients from poisoned clients are automatically filtered out since median 
                    is robust to extreme values.
                  </p>
                  <div className="algorithm-steps">
                    <strong>Algorithm:</strong>
                    <ol>
                      <li>Collect gradient updates from all selected clients</li>
                      <li>For each model parameter position, sort all client values</li>
                      <li>Select median value (middle element after sorting)</li>
                      <li>Apply median gradient to global model</li>
                    </ol>
                  </div>
                </div>
                <div className="defense-example">
                  <strong>Example Scenario:</strong>
                  <p>
                    Parameter update values: [0.02, 0.03, <strong>0.91</strong>, 0.025, 0.028] (poisoned value in bold)
                  </p>
                  <p>
                    <strong>Without defense:</strong> Mean = 0.2026 (corrupted by outlier)<br/>
                    <strong>With Byzantine-Robust:</strong> Median = 0.028 (outlier ignored)
                  </p>
                </div>
                <div className="defense-effectiveness">
                  <strong>Mitigation Effectiveness:</strong> 70-90% attack mitigation
                </div>
                <div className="defense-implementation">
                  <strong>Implementation:</strong> <code>federated_learning/utils/aggregation.py</code>
                </div>
              </div>

              <div className="defense-detail-card">
                <div className="defense-header">
                  <h4>Anomaly Detection</h4>
                  <span className="defense-type">Statistical</span>
                </div>
                <div className="defense-description">
                  Analyzes gradient distributions to identify statistically anomalous updates that deviate significantly 
                  from the expected pattern, flagging potentially malicious clients.
                </div>
                <div className="defense-mechanics">
                  <strong>How It Works:</strong>
                  <p>
                    Compute statistical measures (mean, standard deviation, Z-scores) across all client gradients. 
                    Updates with Z-scores exceeding a threshold (typically 2-3 standard deviations) are considered 
                    anomalous and excluded from aggregation.
                  </p>
                  <div className="algorithm-steps">
                    <strong>Algorithm:</strong>
                    <ol>
                      <li>Calculate mean gradient vector across all clients</li>
                      <li>Compute standard deviation for each parameter</li>
                      <li>Calculate Z-score for each client's gradient</li>
                      <li>Flag clients with Z-score &gt; threshold (e.g., 2.5σ)</li>
                      <li>Aggregate only non-anomalous updates</li>
                    </ol>
                  </div>
                </div>
                <div className="defense-example">
                  <strong>Example Scenario:</strong>
                  <p>
                    10 clients submit updates. Client 3's gradient has Z-score = 3.8 (exceeds threshold 2.5).
                  </p>
                  <p>
                    <strong>Action:</strong> Client 3's update is flagged as anomalous and excluded. 
                    Aggregate remaining 9 clients using standard mean.
                  </p>
                </div>
                <div className="defense-effectiveness">
                  <strong>Mitigation Effectiveness:</strong> 60-85% attack mitigation
                </div>
                <div className="defense-implementation">
                  <strong>Implementation:</strong> <code>federated_learning/utils/anomaly_detection.py</code>
                </div>
              </div>

              <div className="defense-detail-card">
                <div className="defense-header">
                  <h4>Gradient Clipping</h4>
                  <span className="defense-type">Norm-Based</span>
                </div>
                <div className="defense-description">
                  Limits the L2 norm of gradient updates to a maximum threshold, preventing malicious clients from 
                  injecting extremely large gradients that could destabilize model training.
                </div>
                <div className="defense-mechanics">
                  <strong>How It Works:</strong>
                  <p>
                    Calculate the L2 norm (magnitude) of each client's gradient vector. If the norm exceeds a predefined 
                    threshold, scale the gradient down proportionally to meet the limit while preserving direction.
                  </p>
                  <div className="algorithm-steps">
                    <strong>Algorithm:</strong>
                    <ol>
                      <li>Compute L2 norm: ||gradient|| = √(Σ g²)</li>
                      <li>If ||gradient|| &gt; threshold:</li>
                      <li>&nbsp;&nbsp;&nbsp;&nbsp;gradient = gradient × (threshold / ||gradient||)</li>
                      <li>Otherwise, keep gradient unchanged</li>
                      <li>Aggregate clipped gradients using mean</li>
                    </ol>
                  </div>
                </div>
                <div className="defense-example">
                  <strong>Example Scenario:</strong>
                  <p>
                    Gradient vector: [5.2, 8.7, -3.4, 12.1], L2 norm = 15.3, Threshold = 5.0
                  </p>
                  <p>
                    <strong>Action:</strong> Scale factor = 5.0 / 15.3 = 0.327<br/>
                    <strong>Clipped gradient:</strong> [1.70, 2.84, -1.11, 3.96] (norm = 5.0)
                  </p>
                </div>
                <div className="defense-effectiveness">
                  <strong>Mitigation Effectiveness:</strong> 50-75% attack mitigation
                </div>
                <div className="defense-implementation">
                  <strong>Implementation:</strong> <code>federated_learning/utils/gradient_clipping.py</code>
                </div>
              </div>

              <div className="defense-detail-card">
                <div className="defense-header">
                  <h4>Client Filtering</h4>
                  <span className="defense-type">Reputation-Based</span>
                </div>
                <div className="defense-description">
                  Maintains reputation scores for each client based on historical update quality. Clients with low 
                  reputation (consistently suspicious updates) are excluded from future training rounds.
                </div>
                <div className="defense-mechanics">
                  <strong>How It Works:</strong>
                  <p>
                    Track each client's contribution quality over time using metrics like gradient similarity to 
                    consensus, validation accuracy impact, and consistency. Assign reputation scores (0-1) and 
                    exclude clients below threshold.
                  </p>
                  <div className="algorithm-steps">
                    <strong>Algorithm:</strong>
                    <ol>
                      <li>Initialize all clients with reputation = 1.0</li>
                      <li>After each round, evaluate update quality (cosine similarity to consensus)</li>
                      <li>Update reputation: R_new = 0.9 × R_old + 0.1 × quality_score</li>
                      <li>Filter: Only select clients with reputation &gt; 0.5</li>
                      <li>Aggregate updates from high-reputation clients only</li>
                    </ol>
                  </div>
                </div>
                <div className="defense-example">
                  <strong>Example Scenario:</strong>
                  <p>
                    Round 5: Client 7 submits suspicious update (low similarity to consensus).
                  </p>
                  <p>
                    <strong>Reputation decay:</strong> R = 0.9 × 0.85 + 0.1 × 0.2 = 0.785<br/>
                    <strong>Round 6-8:</strong> Continued suspicious behavior drops R to 0.45<br/>
                    <strong>Round 9+:</strong> Client 7 excluded (R &lt; 0.5 threshold)
                  </p>
                </div>
                <div className="defense-effectiveness">
                  <strong>Mitigation Effectiveness:</strong> 65-80% attack mitigation (improves over time)
                </div>
                <div className="defense-implementation">
                  <strong>Implementation:</strong> <code>federated_learning/utils/client_filtering.py</code>
                </div>
              </div>
            </div>

            <div className="defense-comparison">
              <h4>Defense Mechanism Comparison</h4>
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>Defense</th>
                    <th>Detection Method</th>
                    <th>Computational Cost</th>
                    <th>Attack Types</th>
                    <th>Best Use Case</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>Byzantine-Robust</strong></td>
                    <td>Median aggregation</td>
                    <td>Low (sorting only)</td>
                    <td>Extreme outliers</td>
                    <td>High poisoning rates</td>
                  </tr>
                  <tr>
                    <td><strong>Anomaly Detection</strong></td>
                    <td>Z-score analysis</td>
                    <td>Medium (statistics)</td>
                    <td>Statistical deviations</td>
                    <td>Moderate attacks</td>
                  </tr>
                  <tr>
                    <td><strong>Gradient Clipping</strong></td>
                    <td>L2 norm limiting</td>
                    <td>Low (norm calculation)</td>
                    <td>Large gradients</td>
                    <td>Gradient explosion</td>
                  </tr>
                  <tr>
                    <td><strong>Client Filtering</strong></td>
                    <td>Reputation tracking</td>
                    <td>Medium (history tracking)</td>
                    <td>Persistent attackers</td>
                    <td>Long-term defense</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="defense-tradeoffs">
              <h4>Key Considerations</h4>
              <ul>
                <li><strong>No Single Solution:</strong> Each defense has strengths and weaknesses; combining multiple defenses often yields best results</li>
                <li><strong>Attack Adaptation:</strong> Sophisticated attackers may adapt to known defenses; defense diversity is critical</li>
                <li><strong>Honest Client Impact:</strong> Some defenses (e.g., gradient clipping) may slightly reduce convergence speed for honest clients</li>
                <li><strong>Computational Overhead:</strong> Defense mechanisms add processing time; balance security needs with efficiency</li>
                <li><strong>Threshold Tuning:</strong> Defense effectiveness depends on proper threshold calibration for your specific use case</li>
              </ul>
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