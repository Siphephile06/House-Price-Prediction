# House Price Prediction using Machine Learning in Python

Predicting house prices is a key challenge in the real estate industry, helping buyers, sellers, and investors make informed decisions [2]. By using machine learning algorithms, we can estimate the sale price of a house based on various features such as location, size, and other physical characteristics [1, 2]. 

This project uses the **Ames Housing Dataset**, which contains information from the Ames Assessor's Office on residential properties sold in Ames, Iowa, from 2006 to 2010 [1]. The dataset contains information for more than 2,900 properties, with a data dictionary outlining more than 75 descriptive variables—ranging from nominal, ordinal, discrete, to continuous features [1].

This codebase implements a robust machine learning pipeline including exploratory data analysis, data cleaning, one-hot encoding for categorical variables, standard scaling, data splitting, and modeling with Support Vector Regressors (SVR), Random Forest Regressors, and Linear Regression [4, 28, 31, 33].

---

## Table of Contents
1. [Key Features](#key-features)
2. [Technologies Used](#technologies-used)
3. [Installation Requirements](#installation-requirements)
4. [Code Structure & Pipeline](#code-structure--pipeline)
5. [Basic Usage Example](#basic-usage-example)
6. [Configuration Options](#configuration-options)
7. [Contributing Guidelines](#contributing-guidelines)
8. [License Information](#license-information)

---

## Key Features
- **Exploratory Data Analysis (EDA):** Complete tools to visualize numeric correlations with Heatmaps and plot distribution graphs of categorical features [5, 6].
- **Data Cleaning:** Automated sequence index (`Id`) dropping, target imputation of missing values using training averages, and record-level null trimming [4, 16].
- **Data Processing (One-Hot Encoding):** Safe, dense transformation of nominal text columns into binary features, including safety boundaries for unknown categories during testing [4, 28].
- **Feature Scaling:** Z-score normalization for features to bring them onto the same scale, which is essential for distance-sensitive models like SVR to function properly [33].
- **Regression Analysis:** Implementation of multiple regression architectures including Linear Regression, non-linear Support Vector Regression (SVR), and Random Forest Regressors to minimize predictive loss [6, 8, 28].

---

## Technologies Used
* **Python** (Core Programming Language)
* **Pandas** (Data Manipulation & Structured Wrangling)
* **Scikit-learn** (Machine Learning, Split Validation, Scaling, & Encoding Transformers)
* **Seaborn** (Statistical Data Visualization)
* **Matplotlib** (Graph Plotting and Image Rendering Engine)

---

## Installation Requirements

This project relies on a specific set of pinned library versions to guarantee performance and reproducibility. You can install all requirements by setting up a virtual environment and installing via pip:

```bash
# Create a virtual environment
python -m venv env

# Activate the virtual environment
# On Windows:
source env/Scripts/activate
# On macOS/Linux:
source env/bin/activate

# Install the required dependencies
pip install -r requirements.txt
```

### `requirements.txt`
```text
cloudpickle==3.1.2
contourpy==1.3.3
cycler==0.12.1
et_xmlfile==2.0.0
fonttools==4.63.0
joblib==1.6.0
kiwisolver==1.5.1
matplotlib==3.11.1
narwhals==2.25.0
numpy==2.4.6
openpyxl==3.1.5
packaging==26.3
pandas==3.0.5
pillow==12.3.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
scikit-learn==1.9.0
scipy==1.17.1
seaborn==0.13.2
six==1.17.0
threadpoolctl==3.6.0
tzdata==2026.3
```

-----

## Code Structure & Pipeline

The pipeline follows a clean, structured workflow to transform raw house data into robust predictions [4, 31, 33]:

```
 ┌──────────────────────┐
 │ Load excel dataset   │
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ Exploratory Analysis │ ──► Saves Correlation Heatmaps & Bar Charts
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ Data Cleaning & Null │ ──► Drops "Id", Imputes missing target values
 │ Imputation           │
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ Categorical Encoding │ ──► Fits OneHotEncoder (Dense, Ignore Unknowns)
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ Train-Test Splitting │ ──► Partitions 80% Train / 20% Validation
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ Feature Scaling      │ ──► Z-score standardization (StandardScaler)
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ Model Training       │ ──► Fits SVR, Random Forest, & Linear Regression
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ Performance Metrics  │ ──► Evaluates validation accuracy using MAPE
 └──────────────────────┘
```

-----

## Basic Usage Example

Below is the complete, working implementation of the housing prediction script. This script loads your spreadsheet, creates visualization outputs, processes features, normalizes them, and executes multiple model architectures:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

dataset = pd.read_excel("HousePricePrediction.xlsx")

object_cols = dataset.select_dtypes(include=["object"]).columns

int_cols = dataset.select_dtypes(include=["int64"]).columns

fl_cols = dataset.select_dtypes(include=["float64"]).columns

numerical_dataset = dataset.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(12, 6))
sns.heatmap(numerical_dataset.corr(),
            cmap="BrBG",
            fmt=".2f",
            linewidths=2,
            annot=True)
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")

unique_values = []
for col in object_cols:
    unique_values.append(dataset[col].unique().size)
plt.figure(figsize=(10, 6))
plt.title("No. Unique values of Categorical Features")
plt.xticks(rotation=90)
sns.barplot(x=object_cols, y=unique_values)
plt.savefig("bar_chart.png")

plt.figure(figsize=(18, 36))
plt.title("Categorical Features: Distribution")
plt.xticks(rotation=90)
index = 1

for col in object_cols:
    y = dataset[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=60)
    sns.barplot(x=list(y.index), y=y)
    index += 1
plt.savefig("fuller_bar_chart.png")

dataset.drop(["Id"],
             axis=1,
             inplace=True)

dataset["SalePrice"] = dataset["SalePrice"].fillna(
    dataset["SalePrice"].mean()
)

new_dataset = dataset.dropna()

new_dataset.isnull().sum()

s = (new_dataset.select_dtypes(exclude=["number"]).columns)

OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

X = df_final.drop(["SalePrice"], axis=1)
Y = df_final["SalePrice"]

X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=0
)

# 1. Instantiate the scaler
scaler = StandardScaler()

# 2. Fit to training features and transform both sets
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# 3. Train SVR on the SCALED data
model_SVR = svm.SVR()
model_SVR.fit(X_train_scaled, Y_train)
Y_pred = model_SVR.predict(X_valid_scaled)

model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)

mean_absolute_percentage_error(Y_valid, Y_pred)

model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)

print(mean_absolute_percentage_error(Y_valid, Y_pred))
```

-----

## Configuration Options

To customize the execution of your pipeline, you can adjust several parameters in the script:

- **`OneHotEncoder` Configuration:**
  - `sparse_output=False`: Keeps output as a dense NumPy array, allowing easy conversion to a standard Pandas DataFrame for debugging and visualization.
  - `handle_unknown='ignore'`: Ignores novel categories in test data by encoding them as all `0`s, protecting the model from runtime crashes.
- **`StandardScaler` Feature Standardization:**
  - Essential for distance-based estimators like `svm.SVR` [33]. It centers each continuous variable to a mean of 0 and scales it to unit variance, preventing high-magnitude columns like `LotArea` from dominating predictions [33].
- **`RandomForestRegressor` Configuration:**
  - `n_estimators`: The number of decision trees. Setting this to `10` runs extremely fast, but increasing it to `100` or `200` typically yields much more stable predictions and lower variance.
- **Validation Split Configuration:**
  - `train_size` & `test_size`: Standardized at `0.8` (80%) and `0.2` (20%) to balance learning data and valuation reliability [31, 32].
  - `random_state`: Integer seed value (e.g. `0`) to guarantee deterministic shuffling so your splits remain identical across runs [34].

-----

## Contributing Guidelines

Contributions to this repository are always welcome! To keep the codebase clean:
1. **Fork** the repository to your own GitHub account.
2. Create a local feature branch: `git checkout -b feature/amazing-feature`.
3. Format and clean your code files before committing.
4. Keep commit messages concise and descriptive (e.g., `git commit -m 'Implement standard scaling for SVR'`).
5. Push to your feature branch and submit a **Pull Request**.

-----

## License Information

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software for personal, academic, or commercial purposes.
