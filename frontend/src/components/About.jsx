import React from 'react';
import { motion } from 'framer-motion';
import './About.css';

function About() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        duration: 0.5
      }
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

      {/* Hero Section */}
      <motion.div 
        className="about-hero"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <motion.h1 
          className="about-title"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <svg className="title-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
          </svg>
          Federated Learning Security Research
        </motion.h1>
        <motion.p 
          className="about-subtitle"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          Exploring Privacy-Preserving Machine Learning & Defense Against Adversarial Attacks
        </motion.p>
      </motion.div>

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="about-content"
      >
        {/* Project Purpose */}
        <motion.section variants={itemVariants} className="about-section">
          <div className="section-header">
            <svg className="section-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <h2>Project Purpose</h2>
          </div>
          <div className="content-card">
            <p>
              This research platform investigates the <strong>security vulnerabilities</strong> in Federated Learning systems and develops <strong>defense mechanisms</strong> against adversarial attacks. Our goal is to ensure that machine learning models can be trained collaboratively across distributed devices while maintaining <strong>robustness</strong>, <strong>privacy</strong>, and <strong>accuracy</strong>.
            </p>
            <p>
              Real-world applications include secure healthcare diagnostics, financial fraud detection, and privacy-preserving mobile AI systems.
            </p>
          </div>
        </motion.section>

        {/* What is Federated Learning */}
        <motion.section variants={itemVariants} className="about-section">
          <div className="section-header">
            <svg className="section-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
            </svg>
            <h2>What is Federated Learning?</h2>
          </div>
          <div className="content-card">
            <p>
              Federated Learning is a <strong>distributed machine learning</strong> approach where multiple clients (devices, hospitals, organizations) collaboratively train a shared model <strong>without exchanging their raw data</strong>.
            </p>
            <div className="highlight-box">
              <h4>
                <svg className="inline-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                How It Works
              </h4>
              <ol className="workflow-list">
                <li><strong>Step 1:</strong> Server sends initial model to all clients</li>
                <li><strong>Step 2:</strong> Each client trains the model locally on their private data</li>
                <li><strong>Step 3:</strong> Clients send only model updates (gradients/weights) to server</li>
                <li><strong>Step 4:</strong> Server aggregates updates using FedAvg algorithm</li>
                <li><strong>Step 5:</strong> Process repeats for multiple rounds until convergence</li>
              </ol>
            </div>
            <div className="benefit-box">
              <svg className="inline-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              <strong>Key Benefit:</strong> Data stays on device, preserving privacy while enabling collaborative learning
            </div>
          </div>
        </motion.section>

        {/* Label Flipping Attacks */}
        <motion.section variants={itemVariants} className="about-section">
          <div className="section-header">
            <svg className="section-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <h2>Label Flipping Attack Methods</h2>
          </div>
          <div className="content-card">
            <p>
              In a <strong>Label Flipping Attack</strong>, malicious clients intentionally corrupt their training labels to poison the global model. This causes the model to misclassify specific targets, reducing overall accuracy and creating targeted vulnerabilities.
            </p>
            
            <div className="attack-strategies">
              <h4>
                <svg className="inline-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Attack Strategies We Test
              </h4>
              
              <div className="strategy-grid">
                <div className="strategy-card baseline">
                  <div className="strategy-badge">Baseline</div>
                  <h5>No Attack</h5>
                  <p className="strategy-desc">Clean training with correct labels</p>
                  <div className="strategy-details">
                    <p><strong>Example:</strong> All images labeled correctly (e.g., cat → cat, dog → dog)</p>
                    <p><strong>Expected Outcome:</strong> Model achieves ~85-92% accuracy on test set</p>
                    <p><strong>Purpose:</strong> Establishes performance baseline for comparison</p>
                  </div>
                </div>

                <div className="strategy-card severe">
                  <div className="strategy-badge">Severe</div>
                  <h5>Replace 0 → 9</h5>
                  <p className="strategy-desc">All instances of class 0 mislabeled as class 9</p>
                  <div className="strategy-details">
                    <p><strong>Example:</strong> CIFAR-10: All airplanes (0) labeled as trucks (9)</p>
                    <p><strong>Expected Outcome:</strong> Model confuses class 0 and 9, accuracy drops to ~60-70%</p>
                    <p><strong>Impact:</strong> Targeted misclassification between specific classes</p>
                  </div>
                </div>

                <div className="strategy-card critical">
                  <div className="strategy-badge">Critical</div>
                  <h5>Replace 1 → 9</h5>
                  <p className="strategy-desc">Class 1 systematically mislabeled as class 9</p>
                  <div className="strategy-details">
                    <p><strong>Example:</strong> CIFAR-10: All cars (1) labeled as trucks (9)</p>
                    <p><strong>Expected Outcome:</strong> Severe confusion between vehicles, accuracy ~55-65%</p>
                    <p><strong>Impact:</strong> Creates targeted backdoor for class 1 misclassification</p>
                  </div>
                </div>

                <div className="strategy-card moderate">
                  <div className="strategy-badge">Moderate</div>
                  <h5>Replace 0 → 2</h5>
                  <p className="strategy-desc">Class 0 confused with visually similar class 2</p>
                  <div className="strategy-details">
                    <p><strong>Example:</strong> CIFAR-10: Airplanes (0) labeled as birds (2)</p>
                    <p><strong>Expected Outcome:</strong> Moderate accuracy drop to ~70-80%</p>
                    <p><strong>Impact:</strong> Exploits visual similarity for subtle poisoning</p>
                  </div>
                </div>

                <div className="strategy-card moderate">
                  <div className="strategy-badge">Moderate</div>
                  <h5>Replace 4 → 6</h5>
                  <p className="strategy-desc">Visually similar classes swapped</p>
                  <div className="strategy-details">
                    <p><strong>Example:</strong> CIFAR-10: Deer (4) labeled as frogs (6)</p>
                    <p><strong>Expected Outcome:</strong> Localized confusion, accuracy ~72-82%</p>
                    <p><strong>Impact:</strong> Tests model's ability to distinguish subtle features</p>
                  </div>
                </div>

                <div className="strategy-card severe">
                  <div className="strategy-badge">Severe</div>
                  <h5>Replace 5 → 3</h5>
                  <p className="strategy-desc">Mid-range classes intentionally confused</p>
                  <div className="strategy-details">
                    <p><strong>Example:</strong> CIFAR-10: Dogs (5) labeled as cats (3)</p>
                    <p><strong>Expected Outcome:</strong> High confusion between pets, accuracy ~58-68%</p>
                    <p><strong>Impact:</strong> Disrupts learned patterns for similar animal classes</p>
                  </div>
                </div>

                <div className="strategy-card moderate">
                  <div className="strategy-badge">Moderate</div>
                  <h5>Replace 1 → 3</h5>
                  <p className="strategy-desc">Adjacent class confusion attack</p>
                  <div className="strategy-details">
                    <p><strong>Example:</strong> CIFAR-10: Cars (1) labeled as cats (3)</p>
                    <p><strong>Expected Outcome:</strong> Cross-category confusion, accuracy ~68-78%</p>
                    <p><strong>Impact:</strong> Creates unexpected misclassifications between unrelated objects</p>
                  </div>
                </div>

                <div className="strategy-card severe">
                  <div className="strategy-badge">Severe</div>
                  <h5>Replace 6 → 0</h5>
                  <p className="strategy-desc">Reverse mapping to initial class</p>
                  <div className="strategy-details">
                    <p><strong>Example:</strong> CIFAR-10: Frogs (6) labeled as airplanes (0)</p>
                    <p><strong>Expected Outcome:</strong> Severe model degradation, accuracy ~60-70%</p>
                    <p><strong>Impact:</strong> Tests inverse poisoning effects and model resilience</p>
                  </div>
                </div>
              </div>

              <div className="attack-summary">
                <h4>
                  <svg className="inline-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  Key Observations
                </h4>
                <ul>
                  <li><strong>Severity varies:</strong> Attacks on dissimilar classes (e.g., 0→9) cause more damage than similar classes (e.g., 0→2)</li>
                  <li><strong>Targeted impact:</strong> Each strategy affects specific class pairs differently</li>
                  <li><strong>Detection challenge:</strong> Some attacks are harder to detect due to label plausibility</li>
                  <li><strong>Defense evaluation:</strong> These strategies help us test the effectiveness of our defense mechanisms</li>
                </ul>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Worker Selection Strategy */}
        <motion.section variants={itemVariants} className="about-section">
          <div className="section-header">
            <svg className="section-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7h18M3 12h18M3 17h18" />
            </svg>
            <h2>Worker Selection Strategies</h2>
          </div>
          <div className="content-card">
            <p>
              The server selects a subset of clients (workers) each round to participate in training. Different selection strategies can influence resilience to attacks and fairness. Below are the main strategies implemented and tested in our platform.
            </p>

            <div className="strategy-grid two-row">
              <div className="strategy-row">
                <div className="strategy-card baseline">
                  <div className="strategy-badge">Baseline</div>
                  <h5>Random Selection</h5>
                  <p className="strategy-desc">Uniform random sampling of clients each round.</p>
                  <div className="strategy-details">
                    <p><strong>Example:</strong> From 100 devices, 10 are picked at random each round.</p>
                    <p><strong>Expected Outcome:</strong> Fair distribution of participation; baseline accuracy and variance measured across runs.</p>
                    <p><strong>Purpose / Impact:</strong> Provides an unbiased control to compare other selection strategies and their effect on attack surface.</p>
                  </div>
                </div>

                <div className="strategy-card moderate">
                  <div className="strategy-badge">Moderate</div>
                  <h5>Before Breakpoint</h5>
                  <p className="strategy-desc">Favor a stable subset of clients during early training epochs.</p>
                  <div className="strategy-details">
                    <p><strong>Example:</strong> Epochs 1–5 draw 80% of workers from the first 40 devices.</p>
                    <p><strong>Expected Outcome:</strong> Faster initial convergence but increased risk if early pool contains attackers.</p>
                    <p><strong>Purpose / Impact:</strong> Tests how concentrated early participation affects learning and attacker influence.</p>
                  </div>
                </div>
              </div>

              <div className="strategy-row">
                <div className="strategy-card moderate">
                  <div className="strategy-badge">Moderate</div>
                  <h5>After Breakpoint</h5>
                  <p className="strategy-desc">Shift selection bias to later or fresh clients after a breakpoint.</p>
                  <div className="strategy-details">
                    <p><strong>Example:</strong> After epoch 10 prefer clients indexed 50–100 to introduce fresh data.</p>
                    <p><strong>Expected Outcome:</strong> Increased diversity in later training; can reduce overfitting to early participants.</p>
                    <p><strong>Purpose / Impact:</strong> Evaluates robustness to stale or localized data and how late-stage diversity affects attack propagation.</p>
                  </div>
                </div>

                <div className="strategy-card severe">
                  <div className="strategy-badge">Severe</div>
                  <h5>Poisoner Probability</h5>
                  <p className="strategy-desc">Skew selection probabilities according to suspected maliciousness.</p>
                  <div className="strategy-details">
                    <p><strong>Example:</strong> If 5 participants are flagged, include them with 80% chance (to stress-test attacks) or exclude them with 90% chance (to harden the system).</p>
                    <p><strong>Expected Outcome:</strong> Allows measuring attack impact when attackers are oversampled, or defense effectiveness when they're down-weighted.</p>
                    <p><strong>Purpose / Impact:</strong> Directly tests defense policies and the trade-off between robustness and data diversity.</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="attack-summary">
              <h4>
                <svg className="inline-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19l2-2 4 4" />
                </svg>
                Selection Strategy Notes
              </h4>
              <ul>
                <li><strong>Trade-offs:</strong> Random selection is fair but can let attackers slip in; biased strategies can accelerate learning or test specific adversarial setups.</li>
                <li><strong>Defense interplay:</strong> Client selection interacts strongly with defenses (e.g., excluding suspected poisoners improves robustness but may reduce data diversity).</li>
                <li><strong>Reproducibility:</strong> Use fixed seeds when comparing strategies to ensure consistent experiments.</li>
              </ul>
            </div>
          </div>
        </motion.section>

        {/* Models & Datasets */}
        <motion.section variants={itemVariants} className="about-section">
          <div className="section-header">
            <svg className="section-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
            </svg>
            <h2>Neural Network Models & Datasets</h2>
          </div>
          <div className="content-card">
            <div className="models-grid">
              <div className="model-card">
                <h4>
                  <svg className="inline-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  CIFAR-10 CNN
                </h4>
                <div className="model-details">
                  <p><strong>Dataset:</strong> 60,000 color images (32×32 pixels)</p>
                  <p><strong>Classes:</strong> 10 categories - airplane, car, bird, cat, deer, dog, frog, horse, ship, truck</p>
                  <p><strong>Architecture:</strong> Convolutional Neural Network with multiple conv layers, pooling, and dropout</p>
                  <p><strong>Use Case:</strong> Object recognition in natural images</p>
                </div>
              </div>

              <div className="model-card">
                <h4>
                  <svg className="inline-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z" />
                  </svg>
                  Fashion-MNIST CNN
                </h4>
                <div className="model-details">
                  <p><strong>Dataset:</strong> 70,000 grayscale images (28×28 pixels)</p>
                  <p><strong>Classes:</strong> 10 fashion items - T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot</p>
                  <p><strong>Architecture:</strong> Lightweight CNN optimized for grayscale fashion items</p>
                  <p><strong>Use Case:</strong> Fashion item classification and e-commerce applications</p>
                </div>
              </div>
            </div>

            <div className="performance-note">
              <svg className="inline-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <strong>Evaluation Metric:</strong> We measure model accuracy before and after attacks to quantify the impact of label flipping on model performance and test the effectiveness of defense mechanisms.
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
            <p>Our platform implements and evaluates multiple defense strategies to mitigate adversarial attacks:</p>
            <ul className="defense-list">
              <li><strong>Byzantine-Robust Aggregation:</strong> Filters out malicious updates using statistical methods</li>
              <li><strong>Anomaly Detection:</strong> Identifies suspicious client behavior patterns</li>
              <li><strong>Gradient Clipping:</strong> Limits the impact of extreme model updates</li>
              <li><strong>Client Selection:</strong> Prioritizes trustworthy participants based on historical performance</li>
            </ul>
          </div>
        </motion.section>

        {/* Call to Action */}
        <motion.section 
          variants={itemVariants} 
          className="about-section cta-section"
        >
          <div className="cta-card">
            <h3>
              <svg className="inline-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Ready to Explore?
            </h3>
            <p>Run experiments, analyze attack impacts, and contribute to making Federated Learning more secure!</p>
            <button 
              className="cta-button"
              onClick={() => window.location.href = '/'}
            >
              Start Experimenting
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