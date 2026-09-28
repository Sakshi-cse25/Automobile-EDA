# Exploratory Data Analysis on Automobile Dataset

## Project Overview
This project performs Exploratory Data Analysis (EDA) on the **UCI Automobile Dataset**.

The analysis focuses on:
- Understanding the structure of the dataset
- Checking missing values and duplicate records
- Cleaning and preparing the data
- Descriptive statistics
- Univariate, bivariate and multivariate analysis
- Identifying outliers
- Visualizing relationships between automobile features
- Extracting key findings and limitations

## Dataset
**Source:** UCI Machine Learning Repository – Automobile Dataset

Dataset page:
https://archive.ics.uci.edu/dataset/10/automobile

The original dataset contains automobile information such as:
- Manufacturer
- Fuel type
- Body style
- Engine size
- Horsepower
- Curb weight
- Mileage
- Price

## Files

- `automobile_eda.py` – Complete Python EDA code
- `imports-85.data` – Raw UCI Automobile dataset
- `cleaned_automobile_dataset.csv` – Cleaned dataset (optional)

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Analysis Performed

### Data Cleaning
- Converted numerical columns to appropriate numeric types
- Removed duplicate records
- Removed records with missing price
- Median imputation for numerical missing values
- Mode imputation for categorical missing values
- Standardized categorical labels
- Checked for impossible values

### Exploratory Analysis
The project includes:
1. Car price distribution
2. Number of cars by manufacturer
3. Fuel type distribution
4. Price by fuel type
5. Engine size vs price
6. Horsepower vs price
7. Correlation heatmap
8. Average price by manufacturer

## Key Findings

- Car prices are concentrated in the lower-to-middle price range, with fewer high-priced vehicles.
- Gasoline vehicles are more common than diesel vehicles in the dataset.
- Engine size generally has a positive relationship with price.
- Horsepower generally has a positive relationship with price.
- Average car prices vary across manufacturers.
- Some high-priced observations are present and were retained because they may represent genuine vehicles.

## Limitations

- The dataset contains a relatively small number of records.
- It represents automobiles from a historical period.
- Some categories contain fewer observations.
- Correlation does not imply causation.

## How to Run

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn
```

Place `imports-85.data` in the same folder as `automobile_eda.py`, then run:

```bash
python automobile_eda.py
```

## Author

**Sakshi Kumari**  
Registration No.: 241302220  
B.Tech Computer Science and Engineering
