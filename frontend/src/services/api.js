// API service for federated learning experiments

const API_BASE_URL = 'http://localhost:8000/api'; // Backend API base URL

export class ApiService {
  static async runExperiment(config) {
    try {
      const response = await fetch(`${API_BASE_URL}/run`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to start experiment');
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to run experiment:', error);
      throw error;
    }
  }

  static async getExperimentStatus(expId) {
    try {
      const response = await fetch(`${API_BASE_URL}/status/${expId}`);
      
      if (!response.ok) {
        throw new Error('Failed to get experiment status');
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to get experiment status:', error);
      throw error;
    }
  }

  static async getExperimentResults(expId) {
    try {
      const response = await fetch(`${API_BASE_URL}/results/${expId}`);
      
      if (!response.ok) {
        throw new Error('Failed to get experiment results');
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to get experiment results:', error);
      throw error;
    }
  }

  static async getExperimentsList() {
    try {
      const response = await fetch(`${API_BASE_URL}/experiments`);
      
      if (!response.ok) {
        throw new Error('Failed to get experiments list');
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to get experiments list:', error);
      throw error;
    }
  }

  // Mock functions for development - remove when backend is ready
  static async mockRunExperiment(config) {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    return {
      exp_id: `exp_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`
    };
  }

  static async mockGetStatus(expId) {
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Simulate experiment progress
    const random = Math.random();
    if (random > 0.7) {
      return { status: 'done' };
    } else if (random > 0.6) {
      return { status: 'error', message: 'Experiment failed due to invalid parameters' };
    } else {
      return { status: 'running' };
    }
  }

  static async mockGetResults(expId) {
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Generate mock results
    const epochs = Array.from({ length: 10 }, (_, i) => i + 1);
    const accuracy = epochs.map(e => 70 + Math.random() * 20 + (e * 0.5));
    const loss = epochs.map(e => Math.max(0.1, 2.0 - (e * 0.15) + (Math.random() - 0.5) * 0.3));
    
    return {
      results: {
        epochs,
        accuracy,
        loss,
        per_class_precision: epochs.map(() => Array.from({ length: 10 }, () => Math.random())),
        per_class_recall: epochs.map(() => Array.from({ length: 10 }, () => Math.random()))
      },
      worker_selection: epochs.map(() => Array.from({ length: 5 }, () => Math.floor(Math.random() * 50))),
      raw_csv: "epoch,accuracy,loss\n" + epochs.map((e, i) => `${e},${accuracy[i].toFixed(2)},${loss[i].toFixed(3)}`).join('\n')
    };
  }
}

// Configuration for available options
export const EXPERIMENT_CONFIG = {
  replacementMethods: [
    { value: 'replace_1_with_9', label: 'Replace 1 → 9' },
    { value: 'replace_0_with_2', label: 'Replace 0 → 2' },
    { value: 'replace_4_with_6', label: 'Replace 4 → 6' },
    { value: 'replace_5_with_3', label: 'Replace 5 → 3' },
    { value: 'replace_1_with_3', label: 'Replace 1 → 3' },
    { value: 'replace_6_with_0', label: 'Replace 6 → 0' },
    { value: 'default_no_change', label: 'No Attack (Baseline)' }
  ],
  
  selectionStrategies: [
    { value: 'RandomSelectionStrategy', label: 'Random Selection' },
    { value: 'BeforeBreakpoint', label: 'Before Breakpoint' },
    { value: 'AfterBreakpoint', label: 'After Breakpoint' },
    { value: 'PoisonerProbability', label: 'Poisoner Probability' }
  ],
  
  defaultConfig: {
    num_poisoned_workers: 0,
    replacement_method: 'replace_1_with_9',
    selection_strategy: 'RandomSelectionStrategy',
    workers_per_round: 5,
    quick_mode: true,
    kwargs: {
      NUM_WORKERS_PER_ROUND: 5
    }
  }
};