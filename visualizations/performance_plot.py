import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import time

def plot_imputation_results(results, save_path=None):
    """Plot comparison of different imputation methods across missing rates"""
    metrics = ['MSE', 'MAE', 'R2', 'RMSE']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics):
        data_to_plot = []
        for method in results:
            for rate in results[method]:
                data_to_plot.append({
                    'Method': method,
                    'Missing Rate': rate,
                    metric: results[method][rate][metric]
                })
        
        df = pd.DataFrame(data_to_plot)
        sns.lineplot(
            data=df,
            x='Missing Rate',
            y=metric,
            hue='Method',
            marker='o',
            ax=axes[idx]
        )
        axes[idx].set_title(f'{metric} vs Missing Rate')
        axes[idx].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}_metrics.png')
    plt.show()

def plot_training_progress(loss_history, save_path=None):
    """Plot training loss over epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history['generator'], label='Generator Loss')
    plt.plot(loss_history['discriminator'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(f'{save_path}_training.png')
    plt.show()

def plot_cgr_correlation_heatmap(original_data, imputed_data, mask, save_path=None):
    """Plot heatmap of CGR correlations"""
    cgr = original_data[:, -1]
    feature_correlations = []
    
    for i in range(original_data.shape[1]-1):
        missing_idx = ~mask[:, i]
        if np.any(missing_idx):
            orig_corr = np.corrcoef(original_data[missing_idx, i], cgr[missing_idx])[0,1]
            imp_corr = np.corrcoef(imputed_data[missing_idx, i], cgr[missing_idx])[0,1]
            feature_correlations.append({
                'Feature': f'Feature {i}',
                'Original': orig_corr,
                'Imputed': imp_corr
            })
    
    df = pd.DataFrame(feature_correlations)
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        df[['Original', 'Imputed']].T,
        annot=True,
        cmap='coolwarm',
        center=0,
        xticklabels=df['Feature']
    )
    plt.title('CGR Correlation Comparison')
    if save_path:
        plt.savefig(f'{save_path}_cgr_corr.png')
    plt.show()

def plot_feature_distributions(original_data, imputed_data, mask, save_path=None):
    """Plot distribution comparison of original vs imputed values"""
    n_features = original_data.shape[1] - 1  # Excluding CGR
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.ravel()
    
    for i in range(n_features):
        missing_idx = ~mask[:, i]
        
        sns.kdeplot(
            data=original_data[missing_idx, i],
            label='Original',
            ax=axes[i]
        )
        sns.kdeplot(
            data=imputed_data[missing_idx, i],
            label='Imputed',
            ax=axes[i]
        )
        
        axes[i].set_title(f'Feature {i} Distribution')
        axes[i].legend()
        axes[i].grid(True)
    
    # Remove empty subplots
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}_distributions.png')
    plt.show()

def plot_temporal_patterns(original_data, imputed_data, mask, window_size=24, save_path=None):
    """Plot temporal patterns and CGR relationships"""
    # Select a subset of features for visualization
    n_features = min(4, original_data.shape[1] - 1)
    cgr = original_data[:, -1]
    
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 4*n_features))
    if n_features == 1:
        axes = [axes]
    
    time_index = np.arange(len(cgr))
    
    for i in range(n_features):
        missing_idx = ~mask[:, i]
        
        # Plot original values
        axes[i].scatter(
            time_index[~missing_idx],
            original_data[~missing_idx, i],
            c='blue',
            label='Original',
            alpha=0.6
        )
        
        # Plot imputed values
        axes[i].scatter(
            time_index[missing_idx],
            imputed_data[missing_idx, i],
            c='red',
            label='Imputed',
            alpha=0.6
        )
        
        # Plot CGR trend
        ax2 = axes[i].twinx()
        ax2.plot(time_index, cgr, 'g--', label='CGR', alpha=0.3)
        
        axes[i].set_title(f'Feature {i} Temporal Pattern')
        axes[i].legend(loc='upper left')
        ax2.legend(loc='upper right')
        axes[i].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}_temporal.png')
    plt.show()

def create_results_summary(results, save_path=None):
    """Create and save detailed results summary"""
    summary_data = []
    
    for method in results:
        for rate in results[method]:
            metrics = results[method][rate]
            summary_data.append({
                'Method': method,
                'Missing Rate': rate,
                'MSE': metrics['MSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2'],
                'RMSE': metrics['RMSE'],
                'Execution Time': metrics.get('execution_time', np.nan)
            })
    
    df = pd.DataFrame(summary_data)
    
    if save_path:
        df.to_csv(f'{save_path}_summary.csv', index=False)
    
    return df

def plot_execution_times(results, save_path=None):
    """Plot execution times comparison"""
    execution_times = []
    
    for method in results:
        for rate in results[method]:
            execution_times.append({
                'Method': method,
                'Missing Rate': rate,
                'Time (s)': results[method][rate].get('execution_time', np.nan)
            })
    
    df = pd.DataFrame(execution_times)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Method', y='Time (s)', hue='Missing Rate')
    plt.xticks(rotation=45)
    plt.title('Execution Time Comparison')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'{save_path}_times.png')
    plt.show()


#2.Imputation performance plots
# Read the results
df = pd.read_excel('imputation_data.xlsx')

# Set up the plotting style for publication quality
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Define a color palette
colors = list(mcolors.TABLEAU_COLORS.values())

def create_performance_heatmap():
    """Create heatmap showing relative performance"""
    metrics = ['MSE', 'MAE', 'R2', 'RMSE']
    pivot_dfs = []
    
    for metric in metrics:
        pivot_df = df.pivot(index='Method', columns='Missing Rate', values=metric)
        # Normalize values
        if metric != 'R2':  # Lower is better
            pivot_df = (pivot_df.max().max() - pivot_df) / (pivot_df.max().max() - pivot_df.min().min())
        else:  # Higher is better
            pivot_df = (pivot_df - pivot_df.min().min()) / (pivot_df.max().max() - pivot_df.min().min())
        pivot_dfs.append(pivot_df)
    
    avg_performance = sum(pivot_dfs) / len(pivot_dfs)
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(avg_performance, cmap='YlOrRd')
    plt.colorbar(im)
    
    # Add labels
    plt.xticks(np.arange(len(avg_performance.columns)), avg_performance.columns, rotation=45)
    plt.yticks(np.arange(len(avg_performance.index)), avg_performance.index)
    
    # Add text annotations
    for i in range(len(avg_performance.index)):
        for j in range(len(avg_performance.columns)):
            text = plt.text(j, i, f'{avg_performance.iloc[i, j]:.2f}',
                          ha='center', va='center')
    
    #plt.title('Average Relative Performance Across All Metrics')
    plt.tight_layout()
    plt.savefig('D://From One drive//Paper 001//Paper 1//DLPIMjan2stoutput//performance_heatmap.tiff',dpi=600)
    plt.show()
    plt.close()

def create_execution_time_plot():
    """Create execution time comparison plot"""
    plt.figure(figsize=(12, 6))
    pivot_df = df.pivot(index='Method', columns='Missing Rate', values='execution_time')
    
    im = plt.imshow(pivot_df, cmap='viridis')
    plt.colorbar(im, label='Execution Time (seconds)')
    
    plt.xticks(np.arange(len(pivot_df.columns)), pivot_df.columns, rotation=45)
    plt.yticks(np.arange(len(pivot_df.index)), pivot_df.index)
    
    # Add text annotations
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            text = plt.text(j, i, f'{pivot_df.iloc[i, j]:.1f}',
                          ha='center', va='center')
    
    #plt.title('Execution Time by Method and Missing Rate')
    plt.tight_layout()
    plt.savefig('D://From One drive//Paper 001//Paper 1//DLPIMjan2stoutput//execution_time.tiff',dpi=600)
    plt.show()
    plt.close()

def create_missing_rate_impact_plot():
    """Create line plots showing impact of missing rate on MAE and R²"""
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))  # Changed to 2x1 layout
    metrics = ['MAE', 'R2']  # Changed to only MAE and R2
    
    for i, metric in enumerate(metrics):
        for j, method in enumerate(df['Method'].unique()):
            method_data = df[df['Method'] == method]
            axes[i].plot(method_data['Missing Rate'], 
                        method_data[metric], 
                        'o-', 
                        label=method,
                        color=colors[j % len(colors)])
            
        #axes[i].set_title(f'Impact of Missing Rate on {metric}')
        axes[i].set_xlabel('Missing Rate')
        axes[i].set_ylabel(metric)
        axes[i].grid(True, alpha=0.3)
        
        if i == len(metrics)-1:  # Show legend on last plot (R2)
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('D://From One drive//Paper 001//Paper 1//DLPIMjan2stoutput//missing_rate_impact.tiff', 
                dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

def create_mae_plot():
    plt.figure(figsize=(8, 4))
    
    for j, method in enumerate(df['Method'].unique()):
        method_data = df[df['Method'] == method]
        plt.plot(method_data['Missing Rate'], 
                method_data['MAE'], 
                'o-', 
                label=method,
                color=colors[j % len(colors)])
        
    plt.xlabel('Missing Rate')
    plt.ylabel('MAE')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('D://From One drive//Paper 001//Paper 1//DLPIMjan2stoutput//missing_rate_impact_mae.tiff', 
                dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

def create_r2_plot():
    plt.figure(figsize=(8, 4))
    
    for j, method in enumerate(df['Method'].unique()):
        method_data = df[df['Method'] == method]
        plt.plot(method_data['Missing Rate'], 
                method_data['R2'], 
                'o-', 
                label=method,
                color=colors[j % len(colors)])
        
    plt.xlabel('Missing Rate')
    plt.ylabel('R²')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('D://From One drive//Paper 001//Paper 1//DLPIMjan2stoutput//missing_rate_impact_r2.tiff', 
                dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
    
# Call the function
create_missing_rate_impact_plot()
def create_statistical_summary():
    """Create statistical summary of results"""
    summary = pd.DataFrame()
    
    for metric in ['MSE', 'MAE', 'R2', 'RMSE']:
        metric_summary = df.groupby('Method')[metric].agg(['mean', 'std', 'min', 'max'])
        summary[f'{metric}_mean'] = metric_summary['mean']
        summary[f'{metric}_std'] = metric_summary['std']
    
    summary.to_excel('D://From One drive//Paper 001//Paper 1//DLPIMjan2stoutput//statistical_summary.xlsx')
    return summary

def generate_all_plots():
    """Generate all visualizations"""
    print("Generating plots...")
    #create_main_comparison_plot()
    create_performance_heatmap()
    create_execution_time_plot()
    create_mae_plot()
    create_r2_plot()
   # create_method_ranking_plot()
    create_missing_rate_impact_plot()
    #create_missing_rate_impact_plot()
    summary = create_statistical_summary()
    print("\nPlots generated successfully!")
    print("\nStatistical Summary:")
    print(summary)

# Run the analysis
generate_all_plots()


#Plot evaluation metrics for all algorithms in a compact grouped bar chart with error bars
def plot_evaluation_metrics():
    """
    Plot evaluation metrics for all algorithms in a compact grouped bar chart with error bars
    """
    # Read the Excel file
    results_df = pd.read_excel('data.xlsx')
    
    # Convert Missing Rate to numeric by removing '%' and converting to float
    results_df['Missing Rate'] = results_df['Missing Rate'].str.rstrip('%').astype(float) / 100
    
    # Get unique methods
    methods = ['KNN', 'MICE', 'EM', 'MissForest', 'SoftImpute', 'Autoencoder', 
               'GAIN', 'DEGAIN', 'BRITS', 'DLPIM' ]
    
    # Create figure with adjusted size
    plt.figure(figsize=(12, 7))  # Increased height for better label visibility
    
    # Set font properties
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    
    # Set width of bars and positions
    bar_width = 0.25
    r1 = np.arange(len(methods))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Calculate mean and std values
    r2_values = []
    mse_values = []
    mae_values = []
    r2_std = []
    mse_std = []
    mae_std = []
    
    for method in methods:
        method_data = results_df[results_df['Method'] == method]
        r2_values.append(method_data['R2'].mean())
        mse_values.append(method_data['MSE'].mean())
        mae_values.append(method_data['MAE'].mean())
        r2_std.append(method_data['R2'].std())
        mse_std.append(method_data['MSE'].std())
        mae_std.append(method_data['MAE'].std())
    
    # Create bars with error bars
    plt.bar(r1, r2_values, width=bar_width, label='R²', color='limegreen', 
            yerr=r2_std, capsize=3, error_kw={'elinewidth': 1, 'capthick': 1})
    plt.bar(r2, mse_values, width=bar_width, label='MSE', color='red',
            yerr=mse_std, capsize=3, error_kw={'elinewidth': 1, 'capthick': 1})
    plt.bar(r3, mae_values, width=bar_width, label='MAE', color='dodgerblue',
            yerr=mae_std, capsize=3, error_kw={'elinewidth': 1, 'capthick': 1})
    
    # Add labels
    plt.xlabel('Methods', fontsize=10, fontfamily='Times New Roman')
    plt.ylabel('Values', fontsize=10, fontfamily='Times New Roman')
    
    # Add xticks
    plt.xticks([r + bar_width for r in range(len(methods))], 
               methods, 
               rotation=45, 
               ha='right',
               fontfamily='Times New Roman',
               fontsize=10)
    
    # Set y-axis limits with some padding for labels
    plt.ylim(0.0, 3.5)  # Increased upper limit for label visibility
    
    # Add value labels with adjusted positions
    label_padding = 0.1  # Padding for label placement
    for i in range(len(methods)):
        # R² labels
        plt.text(r1[i], r2_values[i] + r2_std[i] + label_padding, 
                f'{r2_values[i]:.3f}', ha='center', va='bottom', 
                fontsize=8, fontfamily='Times New Roman')
        
        # MSE labels
        plt.text(r2[i], mse_values[i] + mse_std[i] + label_padding, 
                f'{mse_values[i]:.3f}', ha='center', va='bottom', 
                fontsize=8, fontfamily='Times New Roman')
        
        # MAE labels
        plt.text(r3[i], mae_values[i] + mae_std[i] + label_padding, 
                f'{mae_values[i]:.3f}', ha='center', va='bottom', 
                fontsize=8, fontfamily='Times New Roman')
    
    # Create legend with adjusted position
    plt.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('evaluation_metrics_comparison.tiff', 
                dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

def generate_all_visualizations():
    print("Generating visualizations...")
    plot_evaluation_metrics()
    print("✓ Generated evaluation metrics comparison")

if __name__ == "__main__":
    generate_all_visualizations()


# Time execution plot , operformance_heatmap ,main_comparison_plot, method_ranking_plot, 

# Read the results
df = pd.read_excel('imputation_results.xlsx')

# Set up the plotting style for publication quality
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Define a color palette
colors = list(mcolors.TABLEAU_COLORS.values())

def create_main_comparison_plot():
    """Create the main comparison plot with all metrics"""
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig)
    metrics = ['MSE', 'MAE', 'R2', 'RMSE']
    
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[i//2, i%2])
        
        # Get unique methods and missing rates
        methods = df['Method'].unique()
        missing_rates = df['Missing Rate'].unique()
        x = np.arange(len(missing_rates))
        width = 0.8 / len(methods)
        
        # Plot bars for each method
        for j, method in enumerate(methods):
            method_data = df[df['Method'] == method]
            ax.bar(x + j*width - width*len(methods)/2, 
                  method_data[metric], 
                  width, 
                  label=method,
                  color=colors[j % len(colors)])
        
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(missing_rates)
        ax.set_xlabel('Missing Rate')
        ax.set_ylabel(metric)
        if metric in ['MSE', 'RMSE']:
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        if i == len(metrics)-1:  # Only show legend on last plot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('main_comparison.tiff',dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

def create_performance_heatmap():
    """Create heatmap showing relative performance"""
    metrics = ['MSE', 'MAE', 'R2', 'RMSE']
    pivot_dfs = []
    
    for metric in metrics:
        pivot_df = df.pivot(index='Method', columns='Missing Rate', values=metric)
        # Normalize values
        if metric != 'R2':  # Lower is better
            pivot_df = (pivot_df.max().max() - pivot_df) / (pivot_df.max().max() - pivot_df.min().min())
        else:  # Higher is better
            pivot_df = (pivot_df - pivot_df.min().min()) / (pivot_df.max().max() - pivot_df.min().min())
        pivot_dfs.append(pivot_df)
    
    avg_performance = sum(pivot_dfs) / len(pivot_dfs)
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(avg_performance, cmap='YlOrRd')
    plt.colorbar(im)
    
    # Add labels
    plt.xticks(np.arange(len(avg_performance.columns)), avg_performance.columns, rotation=45)
    plt.yticks(np.arange(len(avg_performance.index)), avg_performance.index)
    
    # Add text annotations
    for i in range(len(avg_performance.index)):
        for j in range(len(avg_performance.columns)):
            text = plt.text(j, i, f'{avg_performance.iloc[i, j]:.2f}',
                          ha='center', va='center')
    
    #plt.title('Average Relative Performance Across All Metrics')
    plt.tight_layout()
    plt.savefig('performance_heatmap.tiff',dpi=600)
    plt.show()
    plt.close()

def create_execution_time_plot():
    """Create execution time comparison plot"""
    plt.figure(figsize=(12, 6))
    pivot_df = df.pivot(index='Method', columns='Missing Rate', values='execution_time')
    
    im = plt.imshow(pivot_df, cmap='viridis')
    plt.colorbar(im, label='Execution Time (seconds)')
    
    plt.xticks(np.arange(len(pivot_df.columns)), pivot_df.columns, rotation=45)
    plt.yticks(np.arange(len(pivot_df.index)), pivot_df.index)
    
    # Add text annotations
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            text = plt.text(j, i, f'{pivot_df.iloc[i, j]:.1f}',
                          ha='center', va='center')
    
    #plt.title('Execution Time by Method and Missing Rate')
    plt.tight_layout()
    plt.savefig('execution_time.tiff',dpi=600)
    plt.show()
    plt.close()

def create_method_ranking_plot():
    """Create method ranking plot based on average performance"""
    metrics = ['MSE', 'MAE', 'RMSE']  # Exclude R2 as it's scaled differently
    method_scores = pd.DataFrame()
    
    for metric in metrics:
        scores = df.groupby('Method')[metric].mean()
        if metric in ['MSE', 'RMSE', 'MAE']:  # Lower is better
            scores = (scores.max() - scores) / (scores.max() - scores.min())
        method_scores[metric] = scores
    
    method_scores['Average'] = method_scores.mean(axis=1)
    method_scores = method_scores.sort_values('Average', ascending=True)
    
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(method_scores.index))
    plt.barh(y_pos, method_scores['Average'], color=colors[:len(method_scores)])
    plt.yticks(y_pos, method_scores.index)
    plt.xlabel('Normalized Average Score')
    plt.title('Overall Method Performance Ranking')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('method_ranking.tiff',dpi=600)
    plt.show()
    plt.close()

def create_missing_rate_impact_plot():
    """Create line plots showing impact of missing rate on performance"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    metrics = ['MSE', 'MAE', 'R2', 'RMSE']
    
    for i, metric in enumerate(metrics):
        for j, method in enumerate(df['Method'].unique()):
            method_data = df[df['Method'] == method]
            axes[i].plot(method_data['Missing Rate'], 
                        method_data[metric], 
                        'o-', 
                        label=method,
                        color=colors[j % len(colors)])
            
        axes[i].set_title(f'Impact of Missing Rate on {metric}')
        axes[i].set_xlabel('Missing Rate')
        axes[i].set_ylabel(metric)
        if metric in ['MSE', 'RMSE']:
            axes[i].set_yscale('log')
        axes[i].grid(True, alpha=0.3)
        
        if i == len(metrics)-1:  # Only show legend on last plot
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('missing_rate_impact.tiff', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

def create_statistical_summary():
    """Create statistical summary of results"""
    summary = pd.DataFrame()
    
    for metric in ['MSE', 'MAE', 'R2', 'RMSE']:
        metric_summary = df.groupby('Method')[metric].agg(['mean', 'std', 'min', 'max'])
        summary[f'{metric}_mean'] = metric_summary['mean']
        summary[f'{metric}_std'] = metric_summary['std']
    
    summary.to_excel('statistical_summary.xlsx')
    return summary

def generate_all_plots():
    """Generate all visualizations"""
    print("Generating plots...")
    create_main_comparison_plot()
    create_performance_heatmap()
    create_execution_time_plot()
    create_method_ranking_plot()
    create_missing_rate_impact_plot()
    summary = create_statistical_summary()
    print("\nPlots generated successfully!")
    print("\nStatistical Summary:")
    print(summary)

# Run the analysis
generate_all_plots()