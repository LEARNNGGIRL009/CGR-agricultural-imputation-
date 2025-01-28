import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# raw data analysis 

# Set style for publication quality
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11
})

# Output directory
OUTPUT_DIR = 'directory_path'

# Read the data
df = pd.read_csv('Nutrients.csv')

def plot_data_distributions():
    """Plot distribution of each nutrient"""
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.columns, 1):
        plt.subplot(3, 3, i)
        sns.histplot(data=df, x=column, kde=True)
        plt.title(f'{column} Distribution')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}nutrient_distributions.tiff', 
                dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_boxplots():
    """Create boxplots to show nutrient distributions and outliers"""
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df)
    plt.xticks(rotation=45)
    plt.title('Nutrient Distributions and Outliers')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}nutrient_boxplots.tiff', 
                dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_correlation_heatmap():
    """Create correlation heatmap"""
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f')
    plt.title('Nutrient Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}correlation_heatmap.tiff', 
                dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_pairplot():
    """Create pairplot for nutrient relationships"""
    pair_plot = sns.pairplot(df, diag_kind='kde')
    plt.suptitle('Nutrient Pairwise Relationships', y=1.02)
    pair_plot.savefig(f'{OUTPUT_DIR}nutrient_pairplot.tiff', 
                      dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_time_series():
    """Plot time series for each nutrient"""
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.columns, 1):
        plt.subplot(3, 3, i)
        plt.plot(df.index, df[column])
        plt.title(f'{column} Over Time')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}nutrient_timeseries.tiff', 
                dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

def create_correlation_table():
    """Create and save correlation table"""
    correlation_matrix = df.corr()
    correlation_matrix.to_excel(f'{OUTPUT_DIR}correlation_table.xlsx')
    return correlation_matrix

def save_summary_statistics():
    """Save summary statistics to Excel"""
    summary_stats = df.describe()
    missing_stats = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
    
    with pd.ExcelWriter(f'{OUTPUT_DIR}summary_statistics.xlsx') as writer:
        summary_stats.to_excel(writer, sheet_name='Summary Statistics')
        missing_stats.to_excel(writer, sheet_name='Missing Values')
    
    return summary_stats, missing_stats

def main():
    print("Starting data quality analysis...")
    print("Data Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    
    # Generate and save all visualizations
    print("\nGenerating visualizations...")
    plot_data_distributions()
    print("✓ Distribution plots saved")
    
    plot_boxplots()
    print("✓ Boxplots saved")
    
    plot_correlation_heatmap()
    print("✓ Correlation heatmap saved")
    
    plot_pairplot()
    print("✓ Pairplot saved")
    
    plot_time_series()
    print("✓ Time series plots saved")
    
    # Save statistical information
    correlation_matrix = create_correlation_table()
    print("✓ Correlation table saved")
    
    summary_stats, missing_stats = save_summary_statistics()
    print("✓ Summary statistics saved")
    
    print("\nAll visualizations and statistics have been saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()