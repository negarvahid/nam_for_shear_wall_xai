import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import json
from datetime import datetime
import os

class FeatureNet(nn.Module):
    """Single feature subnetwork for NAM."""
    def __init__(self, hidden_units=[64, 32]):
        super().__init__()
        layers = []
        prev_units = 1
        
        for units in hidden_units:
            layers.extend([
                nn.Linear(prev_units, units),
                nn.ReLU(),
                nn.BatchNorm1d(units),
                nn.Dropout(0.2)
            ])
            prev_units = units
        
        layers.append(nn.Linear(prev_units, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x.unsqueeze(1))

class NAM(nn.Module):
    """Neural Additive Model."""
    def __init__(self, num_features, hidden_units=[64, 32], dropout_rate=0.2):
        super().__init__()
        self.feature_nets = nn.ModuleList([
            FeatureNet(hidden_units) for _ in range(num_features)
        ])
        
    def forward(self, x):
        return sum(net(x[:, i]) for i, net in enumerate(self.feature_nets))

class ShearWallDataset(Dataset):
    """Dataset class for shear wall data."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.to_numpy() if hasattr(X, 'to_numpy') else X)
        self.y = torch.FloatTensor(y.to_numpy() if hasattr(y, 'to_numpy') else y).reshape(-1, 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NAMAnalyzer:
    """Class for analyzing shear wall data using Neural Additive Models."""
    
    def __init__(self, data_dict, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the analyzer with data and setup the model.
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary containing X, y, and metadata from load_and_process_data()
        device : str
            Device to run the model on ('cuda' or 'cpu')
        """
        self.X = data_dict['X']
        self.y = data_dict['y'].values.reshape(-1) if data_dict['y'] is not None else None  # Handle None case
        self.feature_names = data_dict['feature_names']
        self.device = device
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.model = None  # Initialize model as None
        
    def prepare_data(self, test_size=0.2, batch_size=32):
        """Part 1: Data preparation and normalization."""
        # Convert data to numeric, replacing '-' with NaN (using newer pandas API)
        X_numeric = self.X.replace({'-': None}).astype(float)
        
        # Fill missing values with median of each column
        X_filled = X_numeric.fillna(X_numeric.median())
        
        # Normalize inputs to [-1, 1]
        X_scaled = self.scaler.fit_transform(X_filled)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.y, test_size=test_size, random_state=42
        )
        
        # Create data loaders
        train_dataset = ShearWallDataset(X_train, y_train)
        test_dataset = ShearWallDataset(X_test, y_test)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Store for later use
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
    def train_model(self, epochs=100, lr=0.001):
        """Part 1: Model training."""
        # Initialize model
        self.model = NAM(num_features=len(self.feature_names)).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(self.train_loader):.4f}')
        
        # Evaluate performance
        self.model.eval()
        with torch.no_grad():
            y_pred = []
            y_true = []
            
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                output = self.model(X_batch)
                y_pred.extend(output.cpu().numpy())
                y_true.extend(y_batch.numpy())
            
            y_pred = np.array(y_pred).reshape(-1)
            y_true = np.array(y_true).reshape(-1)
            
            # Remove any NaN values
            mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
            y_pred = y_pred[mask]
            y_true = y_true[mask]
            
            # Calculate metrics
            self.metrics = {
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'explained_variance': explained_variance_score(y_true, y_pred)
            }
            
            # Only calculate MSLE if we have positive values
            min_val = min(np.min(y_true), np.min(y_pred))
            if min_val > 0:
                self.metrics['msle'] = mean_squared_error(
                    np.log1p(y_true), 
                    np.log1p(y_pred)
                )
            
            print("\nTest Set Performance:")
            print(f"RMSE: {self.metrics['rmse']:.4f}")
            print(f"MAE: {self.metrics['mae']:.4f}")
            print(f"R²: {self.metrics['r2']:.4f}")
            print(f"Explained Variance: {self.metrics['explained_variance']:.4f}")
            if 'msle' in self.metrics:
                print(f"MSLE: {self.metrics['msle']:.4f}")
            
            # Plot residuals
            plt.figure(figsize=(12, 4))
            
            # Scatter plot of predictions vs actual
            plt.subplot(1, 2, 1)
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('Actual NCDE')
            plt.ylabel('Predicted NCDE')
            plt.title('Predictions vs Actual')
            
            # Residual plot
            residuals = y_pred - y_true
            plt.subplot(1, 2, 2)
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted NCDE')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            
            plt.tight_layout()
            plt.show()
    
    def visualize_features(self, features_to_plot=['hw', 'ρsh', 'M/Vlw'], num_points=100):
        """Part 2: Feature visualization."""
        self.model.eval()
        plt.figure(figsize=(15, 5))
        
        for i, feature in enumerate(features_to_plot, 1):
            feature_idx = self.feature_names.index(feature)
            
            # Generate points across the feature range
            x_range = np.linspace(-1, 1, num_points)
            X_dummy = np.zeros((num_points, len(self.feature_names)))
            X_dummy[:, feature_idx] = x_range
            
            # Get predictions
            with torch.no_grad():
                X_dummy_tensor = torch.FloatTensor(X_dummy).to(self.device)
                contribution = self.model.feature_nets[feature_idx](X_dummy_tensor[:, feature_idx])
                
            # Convert normalized x values back to original scale
            x_original = self.scaler.inverse_transform(X_dummy)[:, feature_idx]
            
            plt.subplot(1, 3, i)
            plt.plot(x_original, contribution.cpu().numpy())
            plt.title(f'Feature Function: {feature}')
            plt.xlabel(f'{feature} value')
            plt.ylabel('Contribution to NCDE')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def interpret_functions(self, features_to_analyze=['hw', 'ρsh', 'M/Vlw']):
        """Part 3: Function interpretation."""
        self.model.eval()
        interpretations = {}
        
        for feature in features_to_analyze:
            feature_idx = self.feature_names.index(feature)
            
            # Generate points across the feature range
            x_range = np.linspace(-1, 1, 100)
            X_dummy = np.zeros((100, len(self.feature_names)))
            X_dummy[:, feature_idx] = x_range
            
            # Get predictions
            with torch.no_grad():
                X_dummy_tensor = torch.FloatTensor(X_dummy).to(self.device)
                contribution = self.model.feature_nets[feature_idx](X_dummy_tensor[:, feature_idx])
                contribution = contribution.cpu().numpy().flatten()
            
            # Analyze function properties
            is_monotonic = np.all(np.diff(contribution) > 0) or np.all(np.diff(contribution) < 0)
            is_linear = np.allclose(np.diff(contribution, 2), 0, atol=1e-2)
            effect = 'positive' if contribution[-1] > contribution[0] else 'negative'
            
            interpretations[feature] = {
                'monotonic': is_monotonic,
                'linear': is_linear,
                'effect': effect,
                'range': np.ptp(contribution)
            }
            
            print(f"\nAnalysis for {feature}:")
            print(f"- Function is {'monotonic' if is_monotonic else 'non-monotonic'}")
            print(f"- Function is {'linear' if is_linear else 'non-linear'}")
            print(f"- Overall {effect} effect on NCDE")
            print(f"- Contribution range: {interpretations[feature]['range']:.4f}")
        
        return interpretations
    
    def analyze_feature_importance(self):
        """Part 4: Feature importance analysis."""
        self.model.eval()
        importance_scores = []
        
        # Calculate importance based on function range
        for i, feature in enumerate(self.feature_names):
            x_range = np.linspace(-1, 1, 100)
            X_dummy = np.zeros((100, len(self.feature_names)))
            X_dummy[:, i] = x_range
            
            with torch.no_grad():
                X_dummy_tensor = torch.FloatTensor(X_dummy).to(self.device)
                contribution = self.model.feature_nets[i](X_dummy_tensor[:, i])
                importance = np.ptp(contribution.cpu().numpy())
                importance_scores.append(importance)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title('Feature Importance Based on NAM Function Range')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
        print("\nTop 5 Most Important Features:")
        print(importance_df.head())
    
    def train_and_evaluate(self, model, epochs, lr, early_stopping_patience=5):
        """Train a model and return its performance metrics."""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        patience_counter = 0
        training_losses = []
        val_losses = []
        train_metrics_history = []
        test_metrics_history = []
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_loss = 0
            train_predictions = []
            train_targets = []
            
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                train_predictions.extend(output.detach().cpu().numpy())
                train_targets.extend(y_batch.cpu().numpy())
            
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            training_losses.append(avg_epoch_loss)
            
            # Calculate training metrics
            train_pred = np.array(train_predictions).reshape(-1)
            train_true = np.array(train_targets).reshape(-1)
            train_metrics = {
                'rmse': np.sqrt(mean_squared_error(train_true, train_pred)),
                'mae': mean_absolute_error(train_true, train_pred),
                'r2': r2_score(train_true, train_pred),
                'explained_variance': explained_variance_score(train_true, train_pred)
            }
            train_metrics_history.append(train_metrics)
            
            # Validation phase
            model.eval()
            val_loss = 0
            test_predictions = []
            test_targets = []
            
            with torch.no_grad():
                for X_batch, y_batch in self.test_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    output = model(X_batch)
                    val_loss += criterion(output, y_batch).item()
                    test_predictions.extend(output.cpu().numpy())
                    test_targets.extend(y_batch.cpu().numpy())
            
            val_loss /= len(self.test_loader)
            val_losses.append(val_loss)
            
            # Calculate test metrics
            test_pred = np.array(test_predictions).reshape(-1)
            test_true = np.array(test_targets).reshape(-1)
            test_metrics = {
                'rmse': np.sqrt(mean_squared_error(test_true, test_pred)),
                'mae': mean_absolute_error(test_true, test_pred),
                'r2': r2_score(test_true, test_pred),
                'explained_variance': explained_variance_score(test_true, test_pred)
            }
            test_metrics_history.append(test_metrics)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}]')
                print(f'Training - Loss: {avg_epoch_loss:.4f}, RMSE: {train_metrics["rmse"]:.4f}, '
                      f'MAE: {train_metrics["mae"]:.4f}, R²: {train_metrics["r2"]:.4f}')
                print(f'Testing  - Loss: {val_loss:.4f}, RMSE: {test_metrics["rmse"]:.4f}, '
                      f'MAE: {test_metrics["mae"]:.4f}, R²: {test_metrics["r2"]:.4f}')
        
        # Get final metrics
        final_metrics = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'training_history': {
                'losses': training_losses,
                'val_losses': val_losses,
                'train_metrics_history': train_metrics_history,
                'test_metrics_history': test_metrics_history
            }
        }
            
        return final_metrics, model

    def grid_search(self, param_grid=None):
        """Perform grid search over hyperparameters."""
        if param_grid is None:
            param_grid = {
                'hidden_units': [
                    [32],
                    [64],
                    [128],
                    [32, 16],
                    [64, 32],
                    [128, 64],
                    [64, 32, 16],
                    [128, 64, 32]
                ],
                'dropout_rate': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.0005],
                'batch_size': [32, 64]
            }
        
        results = []
        self.best_metrics = {'test_metrics': {'r2': float('-inf')}}
        self.best_model = None
        self.model = None
        
        # Prepare data once with largest batch size
        max_batch_size = max(param_grid['batch_size'])
        self.prepare_data(batch_size=max_batch_size)
        
        # Create results directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = f'nam_results_{timestamp}'
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate all combinations
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in product(*param_grid.values())]
        
        for i, params in enumerate(param_combinations):
            print(f"\nTesting configuration {i+1}/{len(param_combinations)}:")
            print(json.dumps(params, indent=2))
            
            # Update batch size if needed
            if params['batch_size'] != self.train_loader.batch_size:
                self.prepare_data(batch_size=params['batch_size'])
            
            # Initialize and train model
            current_model = NAM(
                num_features=len(self.feature_names),
                hidden_units=params['hidden_units'],
                dropout_rate=params['dropout_rate']
            ).to(self.device)
            
            metrics, trained_model = self.train_and_evaluate(
                current_model, 
                epochs=100,
                lr=params['learning_rate']
            )
            
            # Save results
            result = {
                'params': params,
                'metrics': metrics
            }
            results.append(result)
            
            # Create model-specific directory
            model_dir = f"{results_dir}/model_{i+1}"
            os.makedirs(model_dir, exist_ok=True)
            
            # Save learning curves
            plt.figure(figsize=(15, 5))
            
            # Training/Validation Loss
            plt.subplot(1, 3, 1)
            plt.plot(metrics['training_history']['losses'], label='Training Loss')
            plt.plot(metrics['training_history']['val_losses'], label='Validation Loss')
            plt.title('Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # RMSE over time
            plt.subplot(1, 3, 2)
            train_rmse = [m['rmse'] for m in metrics['training_history']['train_metrics_history']]
            test_rmse = [m['rmse'] for m in metrics['training_history']['test_metrics_history']]
            plt.plot(train_rmse, label='Training RMSE')
            plt.plot(test_rmse, label='Testing RMSE')
            plt.title('RMSE over Time')
            plt.xlabel('Epoch')
            plt.ylabel('RMSE')
            plt.legend()
            
            # R² over time
            plt.subplot(1, 3, 3)
            train_r2 = [m['r2'] for m in metrics['training_history']['train_metrics_history']]
            test_r2 = [m['r2'] for m in metrics['training_history']['test_metrics_history']]
            plt.plot(train_r2, label='Training R²')
            plt.plot(test_r2, label='Testing R²')
            plt.title('R² over Time')
            plt.xlabel('Epoch')
            plt.ylabel('R²')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'{model_dir}/learning_curves.png')
            plt.close()
            
            # Save detailed metrics
            with open(f'{model_dir}/metrics.json', 'w') as f:
                json.dump({
                    'params': params,
                    'final_train_metrics': metrics['train_metrics'],
                    'final_test_metrics': metrics['test_metrics'],
                    'training_history': {
                        'losses': metrics['training_history']['losses'],
                        'val_losses': metrics['training_history']['val_losses'],
                        'train_metrics_history': metrics['training_history']['train_metrics_history'],
                        'test_metrics_history': metrics['training_history']['test_metrics_history']
                    }
                }, f, indent=2, default=str)
            
            # Update best model if needed
            if metrics['test_metrics']['r2'] > self.best_metrics['test_metrics']['r2']:
                self.best_metrics = metrics
                self.best_model = trained_model
                self.best_params = params
                self.model = trained_model
                
                # Save best model
                torch.save({
                    'model_state_dict': trained_model.state_dict(),
                    'params': params,
                    'metrics': metrics
                }, f'{results_dir}/best_model.pt')
            
            # Print current results
            print("\nFinal Results:")
            print("Training Metrics:")
            print(f"RMSE: {metrics['train_metrics']['rmse']:.4f}")
            print(f"MAE: {metrics['train_metrics']['mae']:.4f}")
            print(f"R²: {metrics['train_metrics']['r2']:.4f}")
            print(f"Explained Variance: {metrics['train_metrics']['explained_variance']:.4f}")
            print("\nTesting Metrics:")
            print(f"RMSE: {metrics['test_metrics']['rmse']:.4f}")
            print(f"MAE: {metrics['test_metrics']['mae']:.4f}")
            print(f"R²: {metrics['test_metrics']['r2']:.4f}")
            print(f"Explained Variance: {metrics['test_metrics']['explained_variance']:.4f}")
        
        # Save all results
        with open(f'{results_dir}/all_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary plots
        self.plot_hyperparameter_effects(results, results_dir)
        
        # Set the best model as the current model
        self.model = self.best_model
        return results
        
    def plot_hyperparameter_effects(self, results, results_dir):
        """Plot the effects of different hyperparameters on model performance."""
        # Extract data for plotting
        data = []
        for result in results:
            data.append({
                'hidden_layers': len(result['params']['hidden_units']),
                'total_units': sum(result['params']['hidden_units']),
                'dropout_rate': result['params']['dropout_rate'],
                'learning_rate': result['params']['learning_rate'],
                'batch_size': result['params']['batch_size'],
                'train_rmse': result['metrics']['train_metrics']['rmse'],
                'test_rmse': result['metrics']['test_metrics']['rmse'],
                'train_r2': result['metrics']['train_metrics']['r2'],
                'test_r2': result['metrics']['test_metrics']['r2']
            })
        
        df = pd.DataFrame(data)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Plot 1: Effect of network size
        sns.scatterplot(data=df, x='total_units', y='test_r2', 
                       hue='hidden_layers', ax=axes[0,0])
        axes[0,0].set_title('Effect of Network Size on R²')
        axes[0,0].set_xlabel('Total Hidden Units')
        axes[0,0].set_ylabel('Test R²')
        
        # Plot 2: Effect of dropout rate
        sns.boxplot(data=df, x='dropout_rate', y='test_r2', ax=axes[0,1])
        axes[0,1].set_title('Effect of Dropout Rate on R²')
        
        # Plot 3: Effect of learning rate
        sns.boxplot(data=df, x='learning_rate', y='test_r2', ax=axes[1,0])
        axes[1,0].set_title('Effect of Learning Rate on R²')
        
        # Plot 4: Effect of batch size
        sns.boxplot(data=df, x='batch_size', y='test_r2', ax=axes[1,1])
        axes[1,1].set_title('Effect of Batch Size on R²')
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/hyperparameter_effects.png')
        plt.close()

    def analyze_best_model(self):
        """Analyze and visualize the best model's performance."""
        if self.best_model is None:
            raise ValueError("No best model available. Run grid_search first.")
        
        # Ensure we're using the best model
        self.model = self.best_model
        self.model.eval()
        
        print("\nBest Model Configuration:")
        print(json.dumps(self.best_params, indent=2))
        print("\nBest Model Performance:")
        print("\nTraining Metrics:")
        print(f"RMSE: {self.best_metrics['train_metrics']['rmse']:.4f}")
        print(f"MAE: {self.best_metrics['train_metrics']['mae']:.4f}")
        print(f"R²: {self.best_metrics['train_metrics']['r2']:.4f}")
        print(f"Explained Variance: {self.best_metrics['train_metrics']['explained_variance']:.4f}")
        print("\nTesting Metrics:")
        print(f"RMSE: {self.best_metrics['test_metrics']['rmse']:.4f}")
        print(f"MAE: {self.best_metrics['test_metrics']['mae']:.4f}")
        print(f"R²: {self.best_metrics['test_metrics']['r2']:.4f}")
        print(f"Explained Variance: {self.best_metrics['test_metrics']['explained_variance']:.4f}")
        
        # Now run the visualization methods
        self.visualize_features()
        self.interpret_functions()
        self.analyze_feature_importance()
        
    def plot_top_models(self, results, n_top=3):
        """Plot and compare the top N models based on test R² score.
        
        Parameters:
        -----------
        results : list
            List of dictionaries containing model results from grid_search
        n_top : int
            Number of top models to display (default: 3)
        """
        # Sort results by test R² score
        sorted_results = sorted(results, 
                              key=lambda x: x['metrics']['test_metrics']['r2'],
                              reverse=True)[:n_top]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(2, 3)
        
        # Plot 1: R² comparison
        ax1 = fig.add_subplot(gs[0, :])
        model_names = [f"Model {i+1}" for i in range(n_top)]
        train_r2 = [r['metrics']['train_metrics']['r2'] for r in sorted_results]
        test_r2 = [r['metrics']['test_metrics']['r2'] for r in sorted_results]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax1.bar(x - width/2, train_r2, width, label='Training R²')
        ax1.bar(x + width/2, test_r2, width, label='Testing R²')
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Score Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names)
        ax1.legend()
        
        # Plot 2: RMSE comparison
        ax2 = fig.add_subplot(gs[1, 0])
        train_rmse = [r['metrics']['train_metrics']['rmse'] for r in sorted_results]
        test_rmse = [r['metrics']['test_metrics']['rmse'] for r in sorted_results]
        
        ax2.bar(x - width/2, train_rmse, width, label='Training RMSE')
        ax2.bar(x + width/2, test_rmse, width, label='Testing RMSE')
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names)
        ax2.legend()
        
        # Plot 3: MAE comparison
        ax3 = fig.add_subplot(gs[1, 1])
        train_mae = [r['metrics']['train_metrics']['mae'] for r in sorted_results]
        test_mae = [r['metrics']['test_metrics']['mae'] for r in sorted_results]
        
        ax3.bar(x - width/2, train_mae, width, label='Training MAE')
        ax3.bar(x + width/2, test_mae, width, label='Testing MAE')
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
        
        params_to_show = ['hidden_units', 'dropout_rate', 'learning_rate', 'batch_size']
        for param in params_to_show:
            row = [param]
            for result in sorted_results:
                value = result['params'][param]
                if param == 'hidden_units':
                    value = str(value)
                row.append(value)
            table_data.append(row)
        
        # Add metrics to table
        metrics_to_show = [('Test R²', 'r2'), ('Test RMSE', 'rmse'), ('Test MAE', 'mae')]
        for metric_name, metric_key in metrics_to_show:
            row = [metric_name]
            for result in sorted_results:
                value = result['metrics']['test_metrics'][metric_key]
                row.append(f"{value:.4f}")
            table_data.append(row)
        
        table = ax4.table(cellText=table_data,
                         loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed configurations
        print("\nDetailed Configurations of Top Models:")
        for i, result in enumerate(sorted_results):
            print(f"\nModel {i+1}:")
            print("Parameters:")
            print(json.dumps(result['params'], indent=2))
            print("\nTest Metrics:")
            print(f"R²: {result['metrics']['test_metrics']['r2']:.4f}")
            print(f"RMSE: {result['metrics']['test_metrics']['rmse']:.4f}")
            print(f"MAE: {result['metrics']['test_metrics']['mae']:.4f}") 