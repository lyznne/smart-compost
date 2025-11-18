"""
SMART COMPOST - MODEL PROJECT.

---  Visualization module
---   visu.py

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 3 Dec 2024
* email   - emuthiani26@gmail.com


                                    Copyright (c) 2024      - enos.vercel.app


"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from sklearn.manifold import TSNE

def plot_training_history(history, save_dir="results/plots"):
    """Plot training and validation loss curves."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/training_history.png")
    plt.close()

    # Add learning rate plot if available
    if 'learning_rates' in history:
        plt.figure(figsize=(10, 5))
        plt.plot(history['learning_rates'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(f"{save_dir}/learning_rate.png")
        plt.close()

def plot_predictions_vs_actual(y_true, y_pred, save_dir="results/plots", feature_names=None):
    """Scatter plot of predictions vs actual values."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Default feature names if none provided
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(y_true.shape[1])]

    # Global plot with all features
    plt.figure(figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, y_true.shape[1]))

    for i in range(y_true.shape[1]):
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, label=feature_names[i], color=colors[i])

    plt.plot([min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())],
             [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())],
             'k--', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/predictions_vs_actual.png")
    plt.close()

    # Individual plots for each feature
    for i in range(y_true.shape[1]):
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.7, color=colors[i])
        plt.plot([y_true[:, i].min(), y_true[:, i].max()],
                 [y_true[:, i].min(), y_true[:, i].max()],
                 'k--', label='Perfect Prediction')
        plt.xlabel(f'Actual {feature_names[i]}')
        plt.ylabel(f'Predicted {feature_names[i]}')
        plt.title(f'{feature_names[i]} - Predictions vs Actual')
        plt.grid(True, alpha=0.3)

        # Add metrics to the plot
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        plt.text(0.05, 0.95, f'MSE: {mse:.4f}\nR²: {r2:.4f}',
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.savefig(f"{save_dir}/pred_vs_actual_{feature_names[i].replace(' ', '_').lower()}.png")
        plt.close()

def plot_residuals(y_true, y_pred, save_dir="results/plots", feature_names=None):
    """Plot residuals for error analysis."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Default feature names if none provided
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(y_true.shape[1])]

    # Calculate residuals
    residuals = y_true - y_pred

    # Combined residual plot
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, y_true.shape[1]))

    for i in range(y_true.shape[1]):
        plt.scatter(y_pred[:, i], residuals[:, i], alpha=0.6, label=feature_names[i], color=colors[i])

    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/residuals.png")
    plt.close()

    # Individual residual plots
    for i in range(y_true.shape[1]):
        plt.figure(figsize=(10, 6))

        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred[:, i], residuals[:, i], alpha=0.7, color=colors[i])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel(f'Predicted {feature_names[i]}')
        plt.ylabel('Residuals')
        plt.title(f'Residuals vs Predicted - {feature_names[i]}')
        plt.grid(True, alpha=0.3)

        # Histogram of residuals
        plt.subplot(1, 2, 2)
        plt.hist(residuals[:, i], bins=20, alpha=0.7, color=colors[i])
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.title(f'Residual Distribution - {feature_names[i]}')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/residuals_{feature_names[i].replace(' ', '_').lower()}.png")
        plt.close()

def plot_correlation_matrix(data, feature_names=None, save_dir="results/plots"):
    """
    Plot correlation matrix of data features.

    Args:
        data: Either a pandas DataFrame or numpy array with features
        feature_names: List of feature names if data is numpy array
        save_dir: Directory to save the plot
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame if numpy array
    if isinstance(data, np.ndarray):
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(data.shape[1])]
        data = pd.DataFrame(data, columns=feature_names)

    # Calculate correlation matrix
    corr_matrix = data.corr()

    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5)

    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/correlation_matrix.png")
    plt.close()

def plot_feature_importance(model, feature_names=None, save_dir="results/plots"):
    """
    Plot feature importance for models that support feature_importances_ attribute.

    Args:
        model: Model with feature_importances_ attribute
        feature_names: List of feature names
        save_dir: Directory to save the plot
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        print("Model does not support feature importance visualization")
        return

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)

    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(len(importances))]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Ranking')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_importance.png")
    plt.close()

def plot_hidden_state_tsne(hidden_states, labels=None, save_dir="results/plots"):
    """
    Visualize LSTM hidden states using t-SNE dimensionality reduction.

    Args:
        hidden_states: Hidden state tensor from LSTM model
        labels: Optional labels for coloring points
        save_dir: Directory to save the plot
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Reshape hidden states if needed
    if len(hidden_states.shape) > 2:
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    hidden_states_2d = tsne.fit_transform(hidden_states)

    plt.figure(figsize=(10, 8))

    if labels is not None:
        # Color by labels if provided
        scatter = plt.scatter(hidden_states_2d[:, 0], hidden_states_2d[:, 1],
                   c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Label Value')
    else:
        plt.scatter(hidden_states_2d[:, 0], hidden_states_2d[:, 1], alpha=0.7)

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Hidden States')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/hidden_states_tsne.png")
    plt.close()

def plot_prediction_time_series(y_true, y_pred, timestamps=None, feature_names=None, save_dir="results/plots"):
    """
    Plot time series of predictions vs actual values.

    Args:
        y_true: True target values
        y_pred: Predicted values
        timestamps: Optional time axis values
        feature_names: Names of the features
        save_dir: Directory to save the plot
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(y_true.shape[1])]

    if timestamps is None:
        timestamps = np.arange(len(y_true))

    # Plot time series for each feature
    for i in range(y_true.shape[1]):
        plt.figure(figsize=(14, 6))
        plt.plot(timestamps, y_true[:, i], 'b-', label='Actual', linewidth=2)
        plt.plot(timestamps, y_pred[:, i], 'r--', label='Predicted', linewidth=2)

        plt.xlabel('Time')
        plt.ylabel(feature_names[i])
        plt.title(f'Time Series Prediction - {feature_names[i]}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Calculate metrics for this feature
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])

        # Add metrics annotation
        plt.text(0.05, 0.95, f'MSE: {mse:.4f}\nR²: {r2:.4f}',
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f"{save_dir}/time_series_{feature_names[i].replace(' ', '_').lower()}.png")
        plt.close()

def save_model_architecture(model, save_path="results/model_architecture.txt"):
    """Save model architecture to text file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(str(model))

        # Add parameter count information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        f.write(f"\n\nTotal parameters: {total_params:,}")
        f.write(f"\nTrainable parameters: {trainable_params:,}")
        f.write(f"\nNon-trainable parameters: {total_params - trainable_params:,}")

def plot_feature_distributions(data, feature_names=None, save_dir="results/plots"):
    """
    Plot distributions of input features.

    Args:
        data: Input data, numpy array or pandas DataFrame
        feature_names: Names of features
        save_dir: Directory to save plots
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame if numpy array
    if isinstance(data, np.ndarray):
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(data.shape[1])]
        data = pd.DataFrame(data, columns=feature_names)

    # Create distribution plots for each feature
    for column in data.columns:
        plt.figure(figsize=(10, 6))

        # Histogram with KDE
        sns.histplot(data[column], kde=True)

        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        # Add statistics
        mean_val = data[column].mean()
        median_val = data[column].median()
        std_val = data[column].std()

        plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='g', linestyle='-.', label=f'Median: {median_val:.2f}')

        plt.text(0.05, 0.95,
                 f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd Dev: {std_val:.2f}',
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f"{save_dir}/distribution_{column.replace(' ', '_').lower()}.png")
        plt.close()

    # Create a combined boxplot
    plt.figure(figsize=(max(8, len(data.columns) * 0.8), 6))
    sns.boxplot(data=data)
    plt.title('Feature Distributions - Boxplot')
    plt.xticks(rotation=45 if len(data.columns) > 6 else 0)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/features_boxplot.png")
    plt.close()
