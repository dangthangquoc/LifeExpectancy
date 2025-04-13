## Prerequisites
Before running the code, ensure you have the following:

- **Python Version**: 3.12
- **Required Libraries**:
  - `pandas` (for data manipulation)
  - `numpy` (for numerical operations)
  - `scikit-learn` (for machine learning models and preprocessing)
  - `xgboost` (for XGBoost model)
  - `matplotlib` (for plotting)
  - `statsmodels` (for VIF calculation)

## Installation
1. **Install Required Libraries**:
   Install the necessary Python packages using `pip`:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib statsmodels ydata-profiling
   ```

2. **Verify Installation**:
   Ensure all libraries are installed by running:
   ```bash
   python -c "import pandas, numpy, sklearn, xgboost, matplotlib, statsmodels, ydata-profiling; print('All libraries installed successfully!')"
   ```

## Project Structure
- `train.csv`: Training dataset.
- `test.csv`: Test dataset.
- `predict_life_expectancy.py`: Main script containing the full pipeline.
- `Predictions.csv`: Output file with test set predictions (generated after running the script).
- `readme.txt`: This file.

## Running the Code
Follow these steps to run the code and generate predictions:

1. **Prepare the Data**:
   - Ensure `train.csv` and `test.csv` are in the same directory as `predict_life_expectancy.py`.
   - The script expects `train.csv` to have a `TARGET_LifeExpectancy` column and both files to have an `ID` column.

2. **Run the Script**:
   Execute the script:
   ```bash
   python predict_life_expectancy.py
   ```

3. **Expected Runtime**:
   - The script takes approximately 5 minutes to run on a standard machine (8-core CPU, 16GB RAM), primarily due to GridSearchCV for hyperparameter tuning.

## Output
- **Console Output**:
  - Preprocessing details (e.g., unknown countries, skewness, VIF, outliers ratio).
  - Model performance metrics (MAE, RMSE, R², Accuracy) for each model.
  - Feature importance from Random Forest.
  - Confirmation of `Predictions.csv` generation.
- **Files Generated**:
  - `predictions.csv`: Contains 867 rows with columns `ID` (1–867) and `TARGET_LifeExpectancy`.
- **Plots**:
  - A scatter plot of validation predictions (actual vs. predicted life expectancy) for the best model (Random Forest).

## Verifying the Output
To ensure the output is correct:
1. **Check `Predictions.csv`**:
   - Expected format: Two columns (`ID`, `TARGET_LifeExpectancy`).
   - 867 rows, no NaN values.
2. **Validate Metrics**:
   - Random Forest (best model) should have MAE = 2.16, R² = 0.91, Accuracy = 83.86%.
3. **Inspect the Scatter Plot**:
   - The plot should show predicted vs. actual life expectancy, with points clustering around the diagonal (y=x), indicating good prediction accuracy.


