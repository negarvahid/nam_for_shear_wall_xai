import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

# Define the feature names and their full descriptions from the readme
FEATURE_DICT = {
    'lw': 'Length',
    'hw': 'Height',
    'tw': 'Thickness',
    'f′c': 'Concrete compressive strength',
    'fyt': 'Transverse web reinforcement yield strength',
    'fysh': 'Transverse boundary reinforcement yield strength',
    'fyl': 'Vertical web reinforcement yield strength',
    'fybl': 'Vertical boundary reinforcement yield strength',
    'ρt': 'Transverse web reinforcement ratio',
    'ρsh': 'Transverse boundary reinforcement ratio',
    'ρl': 'Vertical web reinforcement ratio',
    'ρbl': 'Vertical boundary reinforcement ratio',
    'P/(Agf′c)': 'Axial Load Ratio',
    'b0': 'Boundary element depth',
    'db': 'Boundary element length',
    's/db': 'Hoop spacing / Boundary element length',
    'AR': 'Aspect ratio',
    'M/Vlw': 'Shear span ratio'
}

def sanitize_filename(filename):
    """Convert a string into a valid filename by removing invalid characters."""
    # Replace invalid characters with underscores
    filename = re.sub(r'[/\\?%*:|"<>()\'{}]', '_', filename)
    # Replace Greek letters and special characters
    filename = filename.replace('ρ', 'rho')
    filename = filename.replace('′', '_prime_')
    # Remove multiple underscores
    filename = re.sub(r'_+', '_', filename)
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    return filename

def analyze_and_transform_skewed_features(data, features=['fysh', 'ρsh'], threshold=1.0, high_skew_threshold=3.0):
    """
    Analyze and transform skewed features using various transformations.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe containing the features
    features : list
        List of features to analyze and transform
    threshold : float
        Skewness threshold above which to apply transformation
    high_skew_threshold : float
        Threshold above which to apply more aggressive transformations
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with transformed features
    dict
        Dictionary containing transformation info
    """
    from scipy import stats
    from sklearn.preprocessing import QuantileTransformer
    transformed_data = data.copy()
    transformations = {}
    
    # Create directory for plots
    os.makedirs('feature_analysis', exist_ok=True)
    
    for feature in features:
        # Calculate skewness
        original_skew = data[feature].skew()
        
        # Create plots for different transformations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Original distribution
        sns.histplot(data=data, x=feature, ax=axes[0])
        axes[0].set_title(f'Original Distribution\nSkewness: {original_skew:.2f}')
        
        # Apply transformations if skewed
        if abs(original_skew) > threshold:
            # Prepare data
            min_value = data[feature].min()
            max_value = data[feature].max()
            range_value = max_value - min_value
            offset = -min_value + range_value * 0.01
            
            # Try different transformations
            transformations_to_try = []
            
            # 1. Log transformation
            log_data = np.log1p(data[feature] + offset)
            log_skew = log_data.skew()
            transformations_to_try.append(('log1p', log_data, log_skew))
            
            # 2. Power transformation (x^0.2)
            power_data = np.power(data[feature] + offset, 0.2)
            power_skew = power_data.skew()
            transformations_to_try.append(('power', power_data, power_skew))
            
            # 3. Rank (percentile) transformation
            rank_data = data[feature].rank(pct=True)
            rank_skew = rank_data.skew()
            transformations_to_try.append(('rank', rank_data, rank_skew))
            
            # Plot all transformations
            for i, (name, trans_data, skew) in enumerate(transformations_to_try, start=1):
                sns.histplot(data=trans_data, ax=axes[i])
                axes[i].set_title(f'{name.title()} Transform\nSkewness: {skew:.2f}')
            
            # Choose the best transformation (lowest absolute skewness)
            best_transform = min(transformations_to_try, key=lambda x: abs(x[2]))
            transform_type, transformed_values, new_skew = best_transform
            
            # Apply the best transformation
            transformed_data[feature] = transformed_values
            
            if transform_type == 'log1p':
                transform_params = {'offset': offset}
            elif transform_type == 'power':
                transform_params = {'power': 0.2, 'offset': offset}
            elif transform_type == 'rank':
                transform_params = {}
            
            # Store transformation info
            transformations[feature] = {
                'type': transform_type,
                'params': transform_params,
                'original_skew': original_skew,
                'transformed_skew': new_skew,
                'tried_transforms': {name: skew for name, _, skew in transformations_to_try}
            }
            
            print(f"\n{feature}:")
            print("  Original skewness:", original_skew)
            print("  Transformation results:")
            for name, _, skew in transformations_to_try:
                print(f"    {name}: {skew:.2f}")
            print(f"  Selected: {transform_type} (skewness: {new_skew:.2f})")
        else:
            transformations[feature] = {
                'type': 'none',
                'original_skew': original_skew
            }
        
        plt.tight_layout()
        plt.savefig(f'feature_analysis/{feature}_transformations.png')
        plt.close()
    
    return transformed_data, transformations

def load_and_process_data(excel_path="Database_ShearWall.xlsx", verbose=True, test_size=0.2, random_state=42):
    """
    Load and process the shear wall database.
    
    Parameters:
    -----------
    excel_path : str, default="Database_ShearWall.xlsx"
        Path to the Excel file containing the database
    verbose : bool, default=True
        Whether to print information about the data
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing:
        - X_train, X_test: Training/testing features (normalized to [-1, 1])
        - y_train, y_test: Training/testing targets (normalized to [-1, 1])
        - feature_names: List of feature names
        - x_scaler: Fitted MinMaxScaler for features
        - y_scaler: Fitted MinMaxScaler for target
        - transformations: Dictionary containing transformation info
    """
    # Input validation
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    # List of features in order
    feature_names = list(FEATURE_DICT.keys())
    output_name = 'NCDE'  # The target variable
    
    try:
        # Read the database, skipping the first 3 rows which are headers
        df = pd.read_excel(excel_path, skiprows=3)
        
        if verbose:
            print("Original dataframe shape:", df.shape)
        
        # Remove the metadata columns (first 3 columns) and the last 3 columns
        df_processed = df.iloc[:, 3:-3]
        
        # Validate number of columns
        expected_cols = len(feature_names) + 1  # features + NCDE
        if df_processed.shape[1] != expected_cols:
            raise ValueError(f"Expected {expected_cols} columns but got {df_processed.shape[1]}")
        
        # Extract features and target
        X = df_processed.iloc[:, :18]
        y = df_processed.iloc[:, 18:19]  # NCDE column
        
        # Rename columns for clarity
        X.columns = feature_names
        y.columns = [output_name]
        
        # Convert data to numeric, replacing '-' with NaN
        X = X.replace({'-': None}).astype(float)
        y = y.astype(float)
        
        # Print information about missing values if verbose
        if verbose:
            print("\nMissing values in features:")
            missing_values = X.isnull().sum()
            for feature, count in missing_values.items():
                if count > 0:
                    print(f"{feature:10} : {count} ({count/len(X)*100:.1f}%)")
        
        # Validate target variable
        if y.isnull().any().any():
            raise ValueError("Target variable (NCDE) contains missing values")
            
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")
    
    # Analyze and transform skewed features
    skewed_features = ['fysh', 'ρsh']
    X_transformed, transformations = analyze_and_transform_skewed_features(X, features=skewed_features)
    
    if verbose:
        print("\nFeature Transformations:")
        for feature, info in transformations.items():
            if info['type'] == 'log1p':
                print(f"{feature}:")
                print(f"  Original skewness: {info['original_skew']:.2f}")
                print(f"  After log transform: {info['transformed_skew']:.2f}")
    
    # Analyze ρsh distribution before any transformations
    print("\nAnalyzing ρsh distribution before transformation:")
    rho_sh_stats = analyze_feature_distribution(X, 'ρsh')
    
    # Split the transformed data
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=test_size, random_state=random_state
    )
    
    # Compute medians from training data only
    train_medians = X_train.median()
    
    # Fill missing values in both sets using training medians
    X_train_filled = X_train.fillna(train_medians)
    X_test_filled = X_test.fillna(train_medians)
    
    # Initialize and fit scalers on training data only
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # Fit scalers on training data and transform both sets
    X_train_normalized = pd.DataFrame(
        x_scaler.fit_transform(X_train_filled),
        columns=X_train_filled.columns,
        index=X_train_filled.index
    )
    
    X_test_normalized = pd.DataFrame(
        x_scaler.transform(X_test_filled),
        columns=X_test_filled.columns,
        index=X_test_filled.index
    )
    
    y_train_normalized = pd.DataFrame(
        y_scaler.fit_transform(y_train),
        columns=y_train.columns,
        index=y_train.index
    )
    
    y_test_normalized = pd.DataFrame(
        y_scaler.transform(y_test),
        columns=y_test.columns,
        index=y_test.index
    )
    
    if verbose:
        print("\nFeature Dictionary (Abbreviation -> Full Name):")
        for abbr, full_name in FEATURE_DICT.items():
            print(f"{abbr:10} : {full_name}")
            
        print("\nData shapes:")
        print("Training set (X_train):", X_train_normalized.shape)
        print("Testing set (X_test):", X_test_normalized.shape)
        print("Training labels (y_train):", y_train_normalized.shape)
        print("Testing labels (y_test):", y_test_normalized.shape)
        
        print("\nFeature ranges after normalization (training set):")
        for col in X_train_normalized.columns:
            print(f"{col:10} : [{X_train_normalized[col].min():.2f}, {X_train_normalized[col].max():.2f}]")
        
        print("\nTarget range after normalization (training set):")
        print(f"NCDE      : [{y_train_normalized[output_name].min():.2f}, {y_train_normalized[output_name].max():.2f}]")
        
        print("\nFirst few rows of normalized training data:")
        print(X_train_normalized.head())
        print("\nFirst few rows of normalized training labels (NCDE):")
        print(y_train_normalized.head())
    
    return {
        'X_train': X_train_normalized,
        'X_test': X_test_normalized,
        'y_train': y_train_normalized,
        'y_test': y_test_normalized,
        'feature_names': feature_names,
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
        'transformations': transformations
    }

def analyze_features(data_dict, save_plots=True, output_dir="plots"):
    """
    Perform comprehensive analysis of features including distributions,
    correlations, and basic statistics.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing X, y, and metadata from load_and_process_data()
    save_plots : bool, default=True
        Whether to save plots to files
    output_dir : str, default="plots"
        Directory to save plots if save_plots is True
    """
    # Extract X and y from the data dictionary
    X = pd.concat([data_dict['X_train'], data_dict['X_test']])
    y = pd.concat([data_dict['y_train'], data_dict['y_test']])
    feature_names = data_dict['feature_names']
    
    # Create output directory if it doesn't exist
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert any string values to numeric, replacing non-numeric with NaN
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Basic statistics
    stats_df = pd.DataFrame({
        'mean': X.mean(),
        'std': X.std(),
        'min': X.min(),
        'max': X.max(),
        'skew': X.skew(),
        'kurtosis': X.kurtosis()
    })
    print("\nFeature Statistics:")
    print(stats_df)
    
    # Create box plots for all features
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=X)
    plt.xticks(rotation=45, ha='right')
    plt.title('Box Plots of All Features')
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/feature_boxplots.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create statistical summary plots
    for feature in feature_names:
        plt.figure(figsize=(10, 6))
        
        # Create bar plot for statistics
        stats_values = [
            stats_df.loc[feature, 'mean'],
            stats_df.loc[feature, 'std'],
            stats_df.loc[feature, 'min'],
            stats_df.loc[feature, 'max']
        ]
        stats_labels = ['Mean', 'Std Dev', 'Min', 'Max']
        
        # Create bar plot
        bars = plt.bar(stats_labels, stats_values)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.title(f'Statistical Summary of {feature}\n({FEATURE_DICT[feature]})')
        plt.ylabel('Value')
        
        if save_plots:
            safe_filename = sanitize_filename(feature)
            plt.savefig(f"{output_dir}/stats_summary_{safe_filename}.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    # Create a normalized parallel coordinates plot
    plt.figure(figsize=(15, 8))
    # Normalize the data for better visualization
    X_norm = (X - X.min()) / (X.max() - X.min())
    pd.plotting.parallel_coordinates(
        pd.concat([X_norm, pd.Series(np.zeros(len(X)), name='group')], axis=1),
        'group',
        alpha=0.1
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Normalized Parallel Coordinates Plot of All Features')
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/parallel_coordinates.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create statistical overview heatmap
    # Normalize statistics for better visualization
    stats_for_heatmap = stats_df.copy()
    for col in stats_for_heatmap.columns:
        stats_for_heatmap[col] = (stats_for_heatmap[col] - stats_for_heatmap[col].min()) / \
                                (stats_for_heatmap[col].max() - stats_for_heatmap[col].min())
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(stats_for_heatmap.T, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('Statistical Measures Heatmap (Normalized)')
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/statistics_heatmap.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Individual Feature Distributions
    for feature in feature_names:
        plt.figure(figsize=(10, 6))
        sns.histplot(X[feature].dropna(), kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(FEATURE_DICT[feature])
        if save_plots:
            safe_filename = sanitize_filename(feature)
            plt.savefig(f"{output_dir}/distribution_{safe_filename}.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    # NCDE Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(y['NCDE'], kde=True)
    plt.title('Distribution of NCDE (Output)')
    plt.xlabel('NCDE')
    if save_plots:
        plt.savefig(f"{output_dir}/distribution_NCDE_output.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Correlations
    # Features vs NCDE
    feature_ncde_corr = pd.DataFrame(index=feature_names, columns=['correlation'])
    for feature in feature_names:
        # Drop any NaN values for correlation calculation
        valid_data = pd.concat([X[feature], y['NCDE']], axis=1).dropna()
        if not valid_data.empty:
            feature_ncde_corr.loc[feature, 'correlation'] = stats.pearsonr(
                valid_data[feature], 
                valid_data['NCDE']
            )[0]
    
    # Plot feature-NCDE correlations
    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_ncde_corr.index, y='correlation', data=feature_ncde_corr)
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Correlations with NCDE')
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/correlations_features_vs_NCDE.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Inter-feature correlations heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Inter-feature Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    if save_plots:
        plt.savefig(f"{output_dir}/correlations_heatmap.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save statistics to CSV
    if save_plots:
        stats_df.to_csv(f"{output_dir}/feature_statistics.csv")
        feature_ncde_corr.to_csv(f"{output_dir}/feature_ncde_correlations.csv")
    
    # Identify highly correlated features
    corr_matrix = X.corr()
    high_corr_threshold = 0.7
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr = abs(corr_matrix.iloc[i, j])
            if corr > high_corr_threshold and not np.isnan(corr):
                high_corr_pairs.append((
                    feature_names[i],
                    feature_names[j],
                    corr
                ))
    
    if high_corr_pairs:
        print("\nHighly correlated feature pairs (|r| > 0.7):")
        for f1, f2, corr in high_corr_pairs:
            print(f"{f1} - {f2}: {corr:.3f}")
    
    return {
        'statistics': stats_df,
        'feature_ncde_correlations': feature_ncde_corr,
        'correlation_matrix': corr_matrix,
        'high_correlation_pairs': high_corr_pairs
    }

def analyze_feature_distribution(data, feature_name):
    """Analyze the distribution of a specific feature in detail."""
    # Get values and remove NaN values
    values = data[feature_name].dropna()
    
    # Basic statistics
    stats_dict = {
        'count': len(values),
        'mean': values.mean(),
        'median': values.median(),
        'std': values.std(),
        'min': values.min(),
        'max': values.max(),
        'skew': values.skew(),
        'kurtosis': values.kurtosis(),
        'zeros': (values == 0).sum(),
        'unique_values': len(values.unique()),
        'missing_values': data[feature_name].isna().sum()
    }
    
    # Calculate percentiles
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    for p in percentiles:
        stats_dict[f'p{p}'] = np.nanpercentile(values, p)
    
    # Print results
    print(f"\nDetailed Analysis of {feature_name}:")
    print(f"Count: {stats_dict['count']}")
    print(f"Missing values: {stats_dict['missing_values']}")
    print(f"Unique values: {stats_dict['unique_values']}")
    print(f"Zeros: {stats_dict['zeros']} ({stats_dict['zeros']/stats_dict['count']*100:.1f}%)")
    print(f"\nCentral Tendency:")
    print(f"Mean: {stats_dict['mean']:.6f}")
    print(f"Median: {stats_dict['median']:.6f}")
    print(f"Std Dev: {stats_dict['std']:.6f}")
    print(f"\nShape:")
    print(f"Skewness: {stats_dict['skew']:.2f}")
    print(f"Kurtosis: {stats_dict['kurtosis']:.2f}")
    print(f"\nRange:")
    print(f"Min: {stats_dict['min']:.6f}")
    print(f"Max: {stats_dict['max']:.6f}")
    
    print("\nPercentiles:")
    for p in percentiles:
        print(f"p{p}: {stats_dict[f'p{p}']:.6f}")
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Histogram with density
    plt.subplot(131)
    sns.histplot(data=values, kde=True)
    plt.title(f'Distribution of {feature_name}')
    plt.xlabel(feature_name)
    
    # Box plot
    plt.subplot(132)
    sns.boxplot(y=values)
    plt.title(f'Box Plot of {feature_name}')
    
    # Q-Q plot
    plt.subplot(133)
    from scipy import stats as scipy_stats
    scipy_stats.probplot(values, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {feature_name}')
    
    plt.tight_layout()
    plt.savefig(f'feature_analysis/{feature_name}_detailed_analysis.png')
    plt.close()
    
    return stats_dict

# If the script is run directly, show the example usage
if __name__ == "__main__":
    # Get the data in the new format
    data_dict = load_and_process_data(verbose=True)
    
    # Run the feature analysis
    analysis = analyze_features(data_dict) 