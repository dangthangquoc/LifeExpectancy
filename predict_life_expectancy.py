# Import necessary libraries for data handling, preprocessing, modeling, and evaluation
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport  # For generating interactive data profiling report
from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting data and hyperparameter tuning
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # For encoding categorical variables and scaling features
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.linear_model import LinearRegression, Ridge  # Linear models for regression
from sklearn.ensemble import RandomForestRegressor  # Non-linear ensemble model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score  # Metrics for evaluation
from statsmodels.stats.outliers_influence import variance_inflation_factor  # For multicollinearity analysis
from xgboost import XGBRegressor  # Gradient boosting model
import matplotlib.pyplot as plt  # For plotting validation predictions
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Load the training and test datasets
train_df = pd.read_csv('train.csv')  # Training data with features and target
test_df = pd.read_csv('test.csv')  # Test data for predictions

# Generate an interactive report to explore data interactions
profile = ProfileReport(train_df, title="My train report")  # Create a profiling report for train data
profile.to_file("train_report.html")  # Save the report as HTML
print("Interactive report generated: train_report.html")  # IMPORTANT: open the file in a browser

# Identify unknown countries in test set (not present in train set)
unknown_countries = set(test_df['Country']) - set(train_df['Country'])
print(f"Unknown countries in test.csv: {unknown_countries}")  # Print the list of unseen countries

# Step 1: Drop irrelevant columns based on initial analysis (keep 'ID' in test_df for submission)
columns_to_drop = ['AdultMortality-Male', 'AdultMortality-Female', 'Thinness1-19years', 'Polio', 'SLS', 'Year', 'Population', 'Schooling']
train_df = train_df.drop(columns=columns_to_drop + ['ID'])  # Drop columns from train data, including 'ID'
test_df = test_df.drop(columns=columns_to_drop)  # Drop columns from test data, keep 'ID'

# Separate features (X) and target (y) for training data
X = train_df.drop(columns=['TARGET_LifeExpectancy'])  # Features for training
y = train_df['TARGET_LifeExpectancy']  # Target variable
X_test = test_df.drop(columns=['ID'])  # Features for test data (keep 'ID' for submission)

# Step 2: Handle missing values in numerical features using median imputation
numerical_cols = [col for col in X.columns if col not in ['Country', 'Status']]  # Identify numerical columns
imputer = SimpleImputer(strategy='median')  # Use median imputation to handle missing values
X[numerical_cols] = imputer.fit_transform(X[numerical_cols])  # Impute missing values in train data
X_test[numerical_cols] = imputer.transform(X_test[numerical_cols])  # Apply same imputation to test data

# Step 3: Encode categorical variable 'Country' using OneHotEncoder
# 'Status' is already binary (0/1), so only encode 'Country'
ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore', min_frequency=20)  # Configure encoder
country_encoded = ohe.fit_transform(X[['Country']])  # Encode 'Country' in train data
country_encoded_df = pd.DataFrame(country_encoded, columns=ohe.get_feature_names_out(['Country']))  # Convert to DataFrame
X = X.drop(columns=['Country'])  # Drop original 'Country' column
X = pd.concat([X, country_encoded_df], axis=1)  # Add encoded columns to train data

# Apply the same encoding to test data
country_encoded_test = ohe.transform(X_test[['Country']])  # Encode 'Country' in test data
country_encoded_test_df = pd.DataFrame(country_encoded_test, columns=ohe.get_feature_names_out(['Country']))  # Convert to DataFrame
X_test = X_test.drop(columns=['Country'])  # Drop original 'Country' column
X_test = pd.concat([X_test, country_encoded_test_df], axis=1)  # Add encoded columns to test data

# Align test set columns with train set (fill missing columns with 0)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Step 4: Add interaction features to capture non-linear relationships
X['HIV_Under5LS'] = X['HIV-AIDS'] * X['Under5LS']  # Interaction between HIV-AIDS and Under5LS
X_test['HIV_Under5LS'] = X_test['HIV-AIDS'] * X_test['Under5LS']  # Apply to test data
X['HIV_Income'] = X['HIV-AIDS'] * X['IncomeCompositionOfResources']  # Interaction between HIV-AIDS and Income
X_test['HIV_Income'] = X_test['HIV-AIDS'] * X_test['IncomeCompositionOfResources']  # Apply to test data
X['AdultMortality_Income'] = X['AdultMortality'] * X['IncomeCompositionOfResources']  # Interaction between AdultMortality and Income
X_test['AdultMortality_Income'] = X_test['AdultMortality'] * X_test['IncomeCompositionOfResources']  # Apply to test data
numerical_cols.extend(['HIV_Under5LS', 'HIV_Income', 'AdultMortality_Income'])  # Add new features to numerical columns list

# Step 5: Check skewness of numerical features and transform highly skewed ones
skewness = X[numerical_cols].skew()  # Calculate skewness for numerical features
print("Skewness of numerical features:\n", skewness)  # Display skewness values
skewness_threshold = 1.5  # Define threshold for skewness correction
skewed_cols = skewness[abs(skewness) > skewness_threshold].index.tolist()  # Identify features with high skewness
print(f"Features with |skewness| > {skewness_threshold}: {skewed_cols}")  # Print skewed features

# Apply log1p transformation to skewed features to normalize distributions
for col in skewed_cols:
    X[col] = np.log1p(X[col].clip(lower=0))  # Clip to avoid negative values, then apply log1p
    X_test[col] = np.log1p(X_test[col].clip(lower=0))  # Apply same transformation to test data

# Step 6: Check multicollinearity using Variance Inflation Factor (VIF)
def calculate_vif(df):
    """Calculate VIF for each feature to identify multicollinearity."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

numerical_X = X[numerical_cols]  # Subset numerical features for VIF calculation
vif_data = calculate_vif(numerical_X)  # Calculate VIF
print("\nVIF Data:\n", vif_data)  # Display VIF results

# Manually drop features with high VIF to reduce multicollinearity
high_vif_cols = ['GDP', 'Diphtheria']  # Features with high VIF identified from analysis
print(f"Dropping columns for VIF reduction: {high_vif_cols}")  # Notify of dropped columns
X = X.drop(columns=high_vif_cols)  # Drop from train data
X_test = X_test.drop(columns=high_vif_cols)  # Drop from test data
numerical_cols = [col for col in numerical_cols if col not in high_vif_cols]  # Update numerical columns list

# Step 7: Cap outliers using IQR method to reduce their impact
def cap_outliers(series):
    """Cap outliers using the IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return series.clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

def detect_outliers(series):
    """Detect the ratio of outliers in a series using the IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    outliers = series[(series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)]
    return len(outliers) / len(series)

# Identify features to check for outliers
outlier_cols = ['AdultMortality', 'Measles', 'HIV-AIDS']
if 'IncomeCompositionOfResources' in X.columns:
    outlier_ratio = detect_outliers(X['IncomeCompositionOfResources'])  # Calculate outlier ratio
    print(f"IncomeCompositionOfResources outlier ratio: {outlier_ratio:.2%}")  # Display ratio
    if outlier_ratio > 0.05:  # If outlier ratio exceeds 5%, include for capping
        outlier_cols.append('IncomeCompositionOfResources')

# Cap outliers in selected columns
for col in outlier_cols:
    if col in X.columns:
        X[col] = cap_outliers(X[col])  # Cap outliers in train data
        X_test[col] = cap_outliers(X_test[col])  # Cap outliers in test data

# Step 8: Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=X['Status'])

# Scale numerical features using StandardScaler
scaler = StandardScaler()  # Initialize scaler
X_train_scaled = X_train.copy()  # Copy train data
X_val_scaled = X_val.copy()  # Copy validation data
X_test_scaled = X_test.copy()  # Copy test data
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])  # Scale train numerical features
X_val_scaled[numerical_cols] = scaler.transform(X_val[numerical_cols])  # Scale validation numerical features
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])  # Scale test numerical features

# Step 9: Evaluate models using MAE, RMSE, R², and Accuracy
def evaluate_model(model, X_train, X_val, y_train, y_val):
    """Evaluate a model using MAE, RMSE, R², and Accuracy metrics."""
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_val)  # Predict on validation set
    # Handle NaN predictions by replacing with mean of target
    if np.isnan(y_pred).any():
        print(f"Warning: NaN in y_pred for {model}")
        y_pred = np.where(np.isnan(y_pred), y_val.mean(), y_pred)
    # Calculate regression metrics
    mae = mean_absolute_error(y_val, y_pred)  # Mean Absolute Error
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))  # Root Mean Squared Error
    r2 = r2_score(y_val, y_pred)  # R² Score
    # Bin life expectancy into ranges for classification metrics
    bins = [0, 60, 70, 80, 90, 100]
    labels = [0, 1, 2, 3, 4]
    y_val_binned = pd.cut(y_val, bins=bins, labels=labels, include_lowest=True)  # Bin actual values
    y_pred_binned = pd.cut(y_pred, bins=bins, labels=labels, include_lowest=True)  # Bin predicted values
    # Handle NaN in binned predictions
    if pd.isna(y_pred_binned).any():
        print("Warning: NaN in y_pred_binned")
        y_pred_binned = y_pred_binned.fillna(y_val_binned.mode()[0])
    accuracy = accuracy_score(y_val_binned, y_pred_binned)  # Calculate Accuracy
    return mae, rmse, r2, accuracy  # Return metrics

# Define models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(random_state=42, n_jobs=-1)
}

# Evaluate each model and store results
results = {}
for name, model in models.items():
    mae, rmse, r2, accuracy = evaluate_model(model, X_train_scaled, X_val_scaled, y_train, y_val)  # Evaluate model
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'Accuracy': accuracy}  # Store results
    print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}, Accuracy: {accuracy:.2%}")  # Print results

# Step 10: Feature Importance and Selection using Random Forest
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)  # Initialize Random Forest for feature importance
rf_model.fit(X_train_scaled, y_train)  # Fit model on scaled training data
feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf_model.feature_importances_})  # Create DataFrame of feature importances
print("\nFeature Importance:\n", feature_importance.sort_values(by='Importance', ascending=False))  # Display sorted feature importances

# Select features with importance above threshold
important_cols = feature_importance[feature_importance['Importance'] > 0.0005]['Feature'].tolist()
print(f"Selected features (importance > 0.0005): {important_cols}")  # Print selected features
# Subset data to include only important features
X_train_scaled = X_train_scaled[important_cols]
X_val_scaled = X_val_scaled[important_cols]
X_test_scaled = X_test_scaled[important_cols]
numerical_cols = [col for col in numerical_cols if col in important_cols]  # Update numerical columns list

# Step 11: Hyperparameter Tuning for selected models
# Tune Ridge Regression
ridge_grid = GridSearchCV(Ridge(random_state=42),
                          {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
                          cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)  # Grid search for best alpha
ridge_grid.fit(X_train_scaled, y_train)  # Fit grid search
best_ridge = ridge_grid.best_estimator_  # Get best Ridge model
mae, rmse, r2, accuracy = evaluate_model(best_ridge, X_train_scaled, X_val_scaled, y_train, y_val)  # Evaluate tuned model
results['Ridge Regression (Tuned)'] = {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'Accuracy': accuracy}  # Store results
print(f"Ridge Regression (Tuned) - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}, Accuracy: {accuracy:.2%}")  # Print results

# Tune Random Forest
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),
                       {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2], 'max_features': [None, 'sqrt', 'log2']},
                       cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)  # Grid search for Random Forest
rf_grid.fit(X_train_scaled, y_train)  # Fit grid search
best_rf = rf_grid.best_estimator_  # Get best Random Forest model
mae, rmse, r2, accuracy = evaluate_model(best_rf, X_train_scaled, X_val_scaled, y_train, y_val)  # Evaluate tuned model
results['Random Forest (Tuned)'] = {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'Accuracy': accuracy}  # Store results
print(f"Random Forest (Tuned) - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}, Accuracy: {accuracy:.2%}")  # Print results

# Tune XGBoost
xgb_grid = GridSearchCV(XGBRegressor(random_state=42, n_jobs=-1),
                        {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.3], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0], 'min_child_weight': [1, 3]},
                        cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)  # Grid search for XGBoost
xgb_grid.fit(X_train_scaled, y_train)  # Fit grid search
best_xgb = xgb_grid.best_estimator_  # Get best XGBoost model
mae, rmse, r2, accuracy = evaluate_model(best_xgb, X_train_scaled, X_val_scaled, y_train, y_val)  # Evaluate tuned model
results['XGBoost (Tuned)'] = {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'Accuracy': accuracy}  # Store results
print(f"XGBoost (Tuned) - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}, Accuracy: {accuracy:.2%}")  # Print results

# Step 12: Select the best model based on MAE
best_model_name = min(results, key=lambda x: results[x]['MAE'])  # Identify model with lowest MAE
# Map model names to their instances
best_model = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(random_state=42, n_jobs=-1),
    'Ridge Regression (Tuned)': best_ridge,
    'Random Forest (Tuned)': best_rf,
    'XGBoost (Tuned)': best_xgb
}[best_model_name]

# Print best model and its metrics
print(f"\nBest Model: {best_model_name}")
print(f"MAE: {results[best_model_name]['MAE']:.2f}, RMSE: {results[best_model_name]['RMSE']:.2f}, R²: {results[best_model_name]['R²']:.2f}, Accuracy: {results[best_model_name]['Accuracy']:.2%}")

# Step 13: Final training and prediction on test set
best_model.fit(X_train_scaled, y_train)  # Train best model on full training data
test_predictions = best_model.predict(X_test_scaled)  # Predict on test set
# Handle NaN predictions by replacing with mean of training target
if np.isnan(test_predictions).any():
    print("Warning: NaN in test_predictions")
    test_predictions = np.where(np.isnan(test_predictions), y_train.mean(), test_predictions)
test_predictions = np.round(test_predictions, 1)  # Round predictions to 1 decimal place
# Create submission file with ID and predictions
output = pd.DataFrame({'ID': test_df['ID'], 'TARGET_LifeExpectancy': test_predictions})
output.to_csv('COSC2753_A1_Predictions_s3977877.csv', index=False)  # Save predictions
print("\nPredictions saved to 'COSC2753_A1_Predictions_s3977877.csv'")

# Plot validation predictions to visualize model performance
y_val_pred = best_model.predict(X_val_scaled)  # Predict on validation set
# Handle NaN predictions
if np.isnan(y_val_pred).any():
    print("Warning: NaN in y_val_pred")
    y_val_pred = np.where(np.isnan(y_val_pred), y_val.mean(), y_val_pred)
plt.scatter(y_val, y_val_pred, alpha=0.5)  # Scatter plot of actual vs predicted
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)  # Diagonal line
plt.xlabel('Actual Life Expectancy')  # X-axis label
plt.ylabel('Predicted Life Expectancy')  # Y-axis label
plt.title(f'{best_model_name} Validation Predictions')  # Plot title
plt.show()  # Display plot
