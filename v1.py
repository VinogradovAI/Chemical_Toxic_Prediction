import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from typing import Tuple

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Function to load training and test data from CSV files
def load_data(train_file: str, test_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    return train_df, test_df

# Function to check and clean the raw data
def raw_data_check_and_refresh(df: pd.DataFrame) -> pd.DataFrame:
    """Main info of dataset"""
    print(df.info())
    print(df.describe())
    print(df.head())

    # Check for missing values
    if df.isnull().sum().sum() == 0:
        print('No missing values')
    else:
        print('Missing values found')
        print(df.isnull().sum())
        df.dropna(inplace=True)
        print('Missing values deleted')

    # Check for duplicates in formulas
    if df['SMILES'].duplicated().sum() == 0:
        print('No duplicates in SMILES')
    else:
        print('Duplicates in SMILES found')
        df.drop_duplicates(subset='SMILES', keep='first', inplace=True)
        print('Duplicates in SMILES deleted')

    # Visualize the distribution of activity
    sns.histplot(df['activity'], kde=True)
    plt.title('Распределение коэффициента токсичности')
    plt.show()

    # Check if activities are positive
    if df['activity'].min() >= 0:
        print('All activities are positive')
    else:
        print('Negative activities found')
        df = df[df['activity'] >= 0]
        print('Negative activities deleted')

    return df

# Function to preprocess the data and generate molecular descriptors
def preprocess_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:



    df['Mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    df['MolecularWeight'] = df['Mol'].apply(Descriptors.MolWt)
    df['LogP'] = df['Mol'].apply(Descriptors.MolLogP)
    df['NumHDonors'] = df['Mol'].apply(Descriptors.NumHDonors)
    df['NumHAcceptors'] = df['Mol'].apply(Descriptors.NumHAcceptors)
    df['TPSA'] = df['Mol'].apply(Descriptors.TPSA)
    df['NumRotatableBonds'] = df['Mol'].apply(Descriptors.NumRotatableBonds)
    df['NumAromaticRings'] = df['Mol'].apply(lambda mol: Descriptors.NumAromaticRings(mol) if mol else 0)
    df['NumAliphaticRings'] = df['Mol'].apply(lambda mol: Descriptors.NumAliphaticRings(mol) if mol else 0)

    nitro_smarts = Chem.MolFromSmarts('[NX3](=O)[O-]')
    df['contains_nitro'] = df['Mol'].apply(lambda mol: mol.HasSubstructMatch(nitro_smarts) if mol else 0)
    nitroso_smarts = Chem.MolFromSmarts('[NX2]=O')
    df['contains_nitroso'] = df['Mol'].apply(lambda mol: mol.HasSubstructMatch(nitroso_smarts) if mol else 0)
    clororganic_smarts = Chem.MolFromSmarts('Cl')
    df['contains_clororganic'] = df['Mol'].apply(lambda mol: mol.HasSubstructMatch(clororganic_smarts) if mol else 0)
    tiol_smarts = Chem.MolFromSmarts('[SH]')
    df['contains_tiol'] = df['Mol'].apply(lambda mol: mol.HasSubstructMatch(tiol_smarts) if mol else 0)
    ciand_smarts = Chem.MolFromSmarts('[C,c]#N')
    df['contains_ciand'] = df['Mol'].apply(lambda mol: mol.HasSubstructMatch(ciand_smarts) if mol else 0)
    toxicophore_smarts = Chem.MolFromSmarts('c1ccccc1')
    df['contains_toxicophore'] = df['Mol'].apply(lambda mol: mol.HasSubstructMatch(toxicophore_smarts) if mol else 0)


    if is_train:
        final_df = df[['activity',
                       'MolecularWeight',
                       'LogP',
                       'NumHDonors',
                       'NumHAcceptors',
                       'TPSA',
                       'NumRotatableBonds',
                       'NumAromaticRings',
                       'NumAliphaticRings',
                       'contains_nitro',
                       'contains_toxicophore',
                       'contains_nitroso',
                       'contains_clororganic',
                       'contains_tiol',
                       'contains_ciand']]
    else:
        final_df = df[['MolecularWeight',
                       'LogP',
                       'NumHDonors',
                       'NumHAcceptors',
                       'TPSA',
                       'NumRotatableBonds',
                       'NumAromaticRings',
                       'NumAliphaticRings',
                       'contains_nitro',
                       'contains_toxicophore',
                       'contains_nitroso',
                       'contains_clororganic',
                       'contains_tiol',
                       'contains_ciand']]

    return final_df

# Function to visualize the data
def visualize_data(df: pd.DataFrame) -> None:
    sns.histplot(df['activity'], kde=True)
    plt.title('Distribution of Toxicity Coefficient')
    plt.show()
    sns.boxplot(x=df['activity'])
    plt.title('Boxplot of Toxicity Coefficient')
    plt.show()
    correlation_matrix = df[['activity',
                             'MolecularWeight',
                             'LogP',
                             'NumHDonors',
                             'NumHAcceptors',
                             'TPSA',
                             'NumRotatableBonds',
                             'NumAromaticRings',
                             'NumAliphaticRings',
                             'contains_nitro',
                             'contains_toxicophore',
                             'contains_nitroso',
                             'contains_clororganic',
                             'contains_tiol',
                             'contains_ciand'
                             ]].corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.title('Correlation Matrix')
    plt.show()

# Function to train a Random Forest model
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=100, random_state=42, verbose=0)
    model.fit(X_train, y_train)
    return model

# Function to optimize the Random Forest model using Grid Search
def optimize_with_grid_search(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    param_grid = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [None, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2, 3],
        'bootstrap': [True, False]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)
    print(f'Best parameters (Grid Search): {grid_search.best_params_}')
    return grid_search.best_estimator_

# Function to evaluate the model
def evaluate_model(model: RandomForestRegressor, X_test: np.ndarray, y_test: np.ndarray, label: str = 'Model') -> Tuple[np.ndarray, float, float, float]:
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'{label} - MAE: {mae}, RMSE: {rmse}, R2: {r2}')
    return y_pred, mae, rmse, r2


# Function to visualize the results of the model
def visualize_results(y_test: np.ndarray, y_pred_initial: np.ndarray, y_pred_grid: np.ndarray) -> None:
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_initial, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Initial Random Forest Model')
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_grid, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Grid Search Optimization')
    plt.tight_layout()
    plt.show()

# Function to visualize the feature importance
def visualize_feature_importance(model: RandomForestRegressor, X: pd.DataFrame, title: str = 'Feature Importance') -> None:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = X.columns[indices[:10]]
    plt.figure(figsize=(10, 6))
    plt.title(title)
    sns.barplot(x=importances[indices[:10]], y=top_features)
    plt.show()

# # Function to set up thresholds and filter important features
# def filter_important_features(model: RandomForestRegressor, X: pd.DataFrame, min_threshold: float = 0.2, max_threshold: float = 0.8) -> pd.DataFrame:
#     importances = model.feature_importances_
#     important_indices = np.where((importances >= min_threshold) & (importances <= max_threshold))[0]
#     important_features = X.columns[important_indices]
#     return X[important_features]

# Main function to execute the workflow
def main() -> None:
    train_file = './data/train.csv'
    test_file = './data/test_only_smiles.csv'
    train_df, test_df = load_data(train_file, test_file)

    # Check and refresh data
    train_df = raw_data_check_and_refresh(train_df)

    # Preprocess training data
    train_processed_df = preprocess_data(train_df, is_train=True)
    visualize_data(train_processed_df)


    # Preprocess test data
    test_processed_df = preprocess_data(test_df, is_train=False)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_processed_df.drop(columns=['activity']))
    X_test_scaled = scaler.transform(test_processed_df)
    y = train_df['activity'].dropna().values

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y, test_size=0.2, random_state=42)

    # Train initial model
    initial_model = train_random_forest(X_train, y_train)
    y_pred_initial, mae_initial, rmse_initial, r2_initial = evaluate_model(initial_model, X_val, y_val, label='Initial Model')

    # Optimize model with grid search
    grid_search_model = optimize_with_grid_search(X_train, y_train)
    y_pred_grid, mae_grid, rmse_grid, r2_grid = evaluate_model(grid_search_model, X_val, y_val, label='Grid Search Optimization')

    # Visualize results and feature importance
    visualize_results(y_val, y_pred_initial, y_pred_grid)
    visualize_feature_importance(grid_search_model,
                                 train_processed_df.drop(columns=['activity']),
                                 title='Feature Importance (Grid Search Optimization)')

    # Filter important features based on thresholds
    # important_features_df = filter_important_features(grid_search_model, train_processed_df.drop(columns=['activity']))

    # Evaluate on test dataset
    print("\n--- Evaluation on Test Dataset ---")
    y_pred_test = grid_search_model.predict(X_test_scaled)
    print("Predictions on test dataset:", y_pred_test)

    # Save predictions to CSV file
    test_df['activity'] = y_pred_test
    test_df.to_csv('./data/test_predictions.csv', index=False)

if __name__ == "__main__":
    main()