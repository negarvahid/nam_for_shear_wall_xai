import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def find_latest_results_dir():
    """Find the most recent results directory."""
    dirs = [d for d in os.listdir('.') if d.startswith('nam_results_')]
    if not dirs:
        raise ValueError("No results directory found!")
    return sorted(dirs)[-1]

def plot_top_models(results, n_top=3):
    """Plot and compare the top N models."""
    # Sort results by test R² score, handling both old and new format
    def get_r2(result):
        metrics = result['metrics']
        if 'r2' in metrics:  # New format
            return metrics['r2']
        elif 'test_metrics' in metrics:  # Old format
            return metrics['test_metrics']['r2']
        else:
            raise ValueError("Unknown metrics format in results")
            
    sorted_results = sorted(results, key=get_r2, reverse=True)[:n_top]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))
    gs = plt.GridSpec(2, 3)
    
    # Helper function to get metric value
    def get_metric(result, metric_name):
        metrics = result['metrics']
        if metric_name in metrics:  # New format
            return metrics[metric_name]
        elif 'test_metrics' in metrics:  # Old format
            return metrics['test_metrics'][metric_name]
        else:
            raise ValueError(f"Cannot find metric {metric_name}")
    
    # Plot 1: R² comparison
    ax1 = fig.add_subplot(gs[0, :])
    model_names = [f"Model {i+1}" for i in range(n_top)]
    test_r2 = [get_metric(r, 'r2') for r in sorted_results]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1.bar(x, test_r2, width, label='R² Score')
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² Score Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.legend()
    
    # Plot 2: RMSE comparison
    ax2 = fig.add_subplot(gs[1, 0])
    test_rmse = [get_metric(r, 'rmse') for r in sorted_results]
    
    ax2.bar(x, test_rmse, width, label='RMSE')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.legend()
    
    # Plot 3: MAE comparison
    ax3 = fig.add_subplot(gs[1, 1])
    test_mae = [get_metric(r, 'mae') for r in sorted_results]
    
    ax3.bar(x, test_mae, width, label='MAE')
    ax3.set_ylabel('MAE')
    ax3.set_title('MAE Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names)
    ax3.legend()
    
    # Configuration table
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    # Create configuration table data
    table_data = []
    headers = ['Parameter', 'Model 1', 'Model 2', 'Model 3']
    table_data.append(headers)
    
    # Handle both old and new parameter names
    old_to_new = {
        'hidden_units': 'hidden_layers',
        'dropout_rate': 'dropout_rate',
        'learning_rate': 'learning_rate',
        'batch_size': 'batch_size'
    }
    
    params_to_show = ['hidden_layers', 'learning_rate', 'batch_size']
    for param in params_to_show:
        row = [param]
        for result in sorted_results:
            # Try both new and old parameter names
            value = result['params'].get(param, result['params'].get(old_to_new.get(param, param), 'N/A'))
            if param in ['hidden_layers', 'hidden_units']:
                value = str(value)
            row.append(value)
        table_data.append(row)
    
    # Add metrics to table
    metrics_to_show = [('R²', 'r2'), ('RMSE', 'rmse'), ('MAE', 'mae')]
    for metric_name, metric_key in metrics_to_show:
        row = [metric_name]
        for result in sorted_results:
            value = get_metric(result, metric_key)
            row.append(f"{value:.4f}")
        table_data.append(row)
    
    table = ax4.table(cellText=table_data,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('top_models_comparison.png')
    plt.close()
    
    return sorted_results

def load_and_analyze_results():
    # Find the latest results directory
    results_dir = find_latest_results_dir()
    print(f"\nLoading results from: {results_dir}")
    
    # Load all results
    with open(f'{results_dir}/all_results.json', 'r') as f:
        results = json.load(f)
    
    # Plot and get top models
    top_models = plot_top_models(results)
    
    # Print detailed configurations
    print("\nDetailed Configurations of Top Models:")
    for i, result in enumerate(top_models):
        print(f"\nModel {i+1}:")
        print("Parameters:")
        print(json.dumps(result['params'], indent=2))
        print("\nMetrics:")
        metrics = result['metrics']
        if 'r2' in metrics:  # New format
            print(f"R²: {metrics['r2']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"MAE: {metrics['mae']:.4f}")
            if 'explained_variance' in metrics:
                print(f"Explained Variance: {metrics['explained_variance']:.4f}")
        else:  # Old format
            print(f"R²: {metrics['test_metrics']['r2']:.4f}")
            print(f"RMSE: {metrics['test_metrics']['rmse']:.4f}")
            print(f"MAE: {metrics['test_metrics']['mae']:.4f}")
            if 'explained_variance' in metrics['test_metrics']:
                print(f"Explained Variance: {metrics['test_metrics']['explained_variance']:.4f}")
    
    # Print winning architecture
    winner = top_models[0]
    print("\n🏆 WINNING ARCHITECTURE 🏆")
    print("=======================")
    print(json.dumps(winner['params'], indent=2))
    print("\nBest Model Performance:")
    metrics = winner['metrics']
    if 'r2' in metrics:  # New format
        print(f"R²: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        if 'explained_variance' in metrics:
            print(f"Explained Variance: {metrics['explained_variance']:.4f}")
    else:  # Old format
        print(f"R²: {metrics['test_metrics']['r2']:.4f}")
        print(f"RMSE: {metrics['test_metrics']['rmse']:.4f}")
        print(f"MAE: {metrics['test_metrics']['mae']:.4f}")
        if 'explained_variance' in metrics['test_metrics']:
            print(f"Explained Variance: {metrics['test_metrics']['explained_variance']:.4f}")
    
    print("\nVisualization saved as 'top_models_comparison.png'")

if __name__ == "__main__":
    load_and_analyze_results() 