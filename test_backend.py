#!/usr/bin/env python3
"""
Test script to verify the CSV loading functionality
"""

import requests
import json

API_BASE_URL = "http://localhost:8000/api"

def test_experiments_list():
    """Test fetching experiments list"""
    try:
        response = requests.get(f"{API_BASE_URL}/experiments")
        if response.status_code == 200:
            data = response.json()
            print(f"Found {data['total']} experiments:")
            for exp in data['experiments']:
                print(f"  {exp['exp_id']} - {exp['status']} - {exp['created_at']}")
            return True
        else:
            print(f"Failed to fetch experiments: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error fetching experiments: {e}")
        return False

def test_experiment_results(exp_id):
    """Test fetching specific experiment results"""
    try:
        response = requests.get(f"{API_BASE_URL}/results/{exp_id}")
        if response.status_code == 200:
            data = response.json()
            print(f"Results for {exp_id}:")
            if 'results' in data:
                results = data['results']
                print(f"  Epochs: {len(results.get('epochs', []))}")
                print(f"  Accuracy range: {min(results.get('accuracy', [0])):.2f} - {max(results.get('accuracy', [0])):.2f}")
                print(f"  Loss range: {min(results.get('loss', [0])):.2f} - {max(results.get('loss', [0])):.2f}")
            print(f"  Worker selections: {len(data.get('worker_selection', []))}")
            return True
        else:
            print(f"Failed to fetch results for {exp_id}: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error fetching results for {exp_id}: {e}")
        return False

if __name__ == "__main__":
    print("Testing Backend CSV Loading Functionality")
    print("=" * 50)
    
    # Test experiments list
    if test_experiments_list():
        print("✅ Experiments list test passed")
    else:
        print("❌ Experiments list test failed")
    
    print()
    
    # Test specific experiment results
    exp_id = "exp_20251014_142238_10ada887"
    if test_experiment_results(exp_id):
        print(f"✅ Results test for {exp_id} passed")
    else:
        print(f"❌ Results test for {exp_id} failed")