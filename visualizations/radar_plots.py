import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Create a radar chart comparing imputation methods across different metrics

def create_radar_chart(results_df):
    """
    Create a radar chart comparing imputation methods across different metrics
    
    Parameters:
    results_df: DataFrame with columns ['Method', 'MSE', 'MAE', 'R2', 'RMSE']
    """
    # Calculate mean performance for each method
    metrics = ['MSE', 'MAE', 'R2', 'RMSE']
    method_means = results_df.groupby('Method')[metrics].mean()
    
    # Normalize metrics to 0-1 scale
    normalized_scores = method_means.copy()
    for metric in metrics:
        if metric != 'R2':  # Lower is better for these metrics
            normalized_scores[metric] = (method_means[metric].max() - method_means[metric]) / \
                                     (method_means[metric].max() - method_means[metric].min())
        else:  # Higher is better for R2
            normalized_scores[metric] = (method_means[metric] - method_means[metric].min()) / \
                                     (method_means[metric].max() - method_means[metric].min())
    
    # Set up the angles for each metric
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    
    # Close the plot by appending the first value
    angles = np.concatenate((angles, [angles[0]]))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot for each method
    colors = plt.cm.Set2(np.linspace(0, 1, len(method_means)))
    for idx, (method, scores) in enumerate(normalized_scores.iterrows()):
        values = scores.values
        values = np.concatenate((values, [values[0]]))  # Close the polygon
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    #plt.title("Imputation Methods Comparison\n(Higher values indicate better performance)", pad=20)
    
    # Add grid
    ax.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('radar_chart.tiff', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

def create_dual_radar_chart(results_df, highlight_method='DLPIM'):
    """
    Create two radar charts: one comparing all methods and one highlighting a specific method
    
    Parameters:
    results_df: DataFrame with performance metrics
    highlight_method: Method to highlight in the second chart
    """
    fig = plt.figure(figsize=(20, 10))
    
    # Calculate mean performance for each method
    metrics = ['MSE', 'MAE', 'R2', 'RMSE']
    method_means = results_df.groupby('Method')[metrics].mean()
    
    # Normalize metrics
    normalized_scores = method_means.copy()
    for metric in metrics:
        if metric != 'R2':
            normalized_scores[metric] = (method_means[metric].max() - method_means[metric]) / \
                                     (method_means[metric].max() - method_means[metric].min())
        else:
            normalized_scores[metric] = (method_means[metric] - method_means[metric].min()) / \
                                     (method_means[metric].max() - method_means[metric].min())
    
    # Set up angles
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    # First subplot - All methods
    ax1 = plt.subplot(121, projection='polar')
    colors = plt.cm.Set2(np.linspace(0, 1, len(method_means)))
    
    for idx, (method, scores) in enumerate(normalized_scores.iterrows()):
        values = scores.values
        values = np.concatenate((values, [values[0]]))
        ax1.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx])
        ax1.fill(angles, values, alpha=0.25, color=colors[idx])
    
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics)
    #ax1.set_title("All Methods Comparison")
    ax1.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Second subplot - Highlighted method vs average
    ax2 = plt.subplot(122, projection='polar')
    
    # Calculate average of other methods
    other_methods = normalized_scores.drop(highlight_method)
    avg_others = other_methods.mean()
    
    # Plot average of others
    values_avg = np.concatenate((avg_others.values, [avg_others.values[0]]))
    ax2.plot(angles, values_avg, 'o-', linewidth=2, label='Average of Others', 
             color='gray', alpha=0.5)
    ax2.fill(angles, values_avg, alpha=0.25, color='gray')
    
    # Plot highlighted method
    values_highlight = np.concatenate((normalized_scores.loc[highlight_method].values, 
                                     [normalized_scores.loc[highlight_method].values[0]]))
    ax2.plot(angles, values_highlight, 'o-', linewidth=2, label=highlight_method, 
             color='red')
    ax2.fill(angles, values_highlight, alpha=0.25, color='red')
    
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics)
    #ax2.set_title(f"{highlight_method} vs Average")
    ax2.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    #plt.suptitle("Imputation Methods Performance Comparison\n(Higher values indicate better performance)", y=1.05)
    
    plt.tight_layout()
    plt.savefig('dual_radar_chart.tiff', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

# Example usage
if __name__ == "__main__":
    # Load your results
    results_df = pd.read_excel('data.xlsx')
    
    # Create single radar chart
    create_radar_chart(results_df)
    print("✓ Generated radar chart")
    
    # Create dual radar chart with DLPIM highlighted
    create_dual_radar_chart(results_df, highlight_method='DLPIM')
    print("✓ Generated dual radar chart")
