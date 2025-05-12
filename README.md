# Housing Price Statistical Analysis and Regression Modeling

This project performs a statistical analysis on a housing dataset to understand the characteristics of its variables and to build a simple linear regression model to predict house prices based on area. The analysis includes descriptive statistics, data visualization, hypothesis testing, and an interpretation of the regression model's performance.

## Project Overview

The project is divided into several key parts:
1.  **Exploratory Data Analysis (EDA):**
    *   Describing the nature of each variable (numerical, categorical, ordinal).
    *   Calculating main statistics (mean, median, mode, standard deviation, IQR, quartiles, range).
    *   Visualizing data distributions using histograms and pie charts for binary categorical variables.
2.  **Sample Size Calculation & Comparison:**
    *   Calculating the minimum sample size needed for a given confidence level and margin of error.
    *   Creating a sample dataset and comparing its mean with the population mean to check if it falls within the defined confidence interval.
3.  **Hypothesis Testing:**
    *   Performing Chi-Square tests to determine if there's an association between house price categories (above/below median) and features like the number of bathrooms and the presence of guestrooms.
4.  **Linear Regression Modeling:**
    *   Building a simple linear regression model to predict `price` based on `area`.
    *   Evaluating the model's performance using R-squared and interpreting the coefficients.
5.  **Findings & Investment Strategy:**
    *   Summarizing key findings from the analysis and regression model.
    *   Proposing a basic investment strategy based on the insights.

## Technologies Used

*   **Python:** The core programming language.
*   **Pandas:**
    *   Used for data manipulation, loading the CSV dataset, and creating DataFrames.
    *   Essential for data cleaning, transformation, and descriptive statistics (`.describe()`, `.mean()`, `.median()`, `.value_counts()`).
*   **NumPy:**
    *   Used for numerical operations, particularly in calculations involving standard deviation and confidence intervals.
*   **Matplotlib & Seaborn:**
    *   Used for creating visualizations such as histograms, pie charts, and scatter plots with regression lines (`plt.hist()`, `plt.pie()`, `sns.distplot()`, `plt.scatter()`).
*   **SciPy (scipy.stats):**
    *   Used for statistical functions, including:
        *   Calculating Z-scores (`st.norm.ppf`).
        *   Performing Shapiro-Wilk test for normality (`st.shapiro`).
        *   Performing Levene's test for homogeneity of variances (`st.levene`).
        *   Performing Chi-Square test for independence (`st.chi2_contingency`).
*   **Scikit-learn (sklearn.linear_model, sklearn.model_selection):**
    *   Used for building and evaluating the linear regression model (`LinearRegression`, `train_test_split`).
*   **Statsmodels:**
    *   Used for a more detailed statistical summary of the linear regression model, providing R-squared, coefficients, p-values, and other diagnostic information (`sm.OLS`).
*   **Jupyter Notebook / Google Colab:** The interactive development environment for conducting the analysis and presenting results.

## Dataset

The dataset used is `Housing3.csv`, loaded from a GitHub repository. It contains the following features for houses:
*   `price`: Numerical (Price of the house)
*   `area`: Numerical (Area of the house in sq. ft.)
*   `bedrooms`: Ordinal/Numerical (Number of bedrooms)
*   `bathrooms`: Ordinal/Numerical (Number of bathrooms)
*   `stories`: Ordinal/Numerical (Number of stories)
*   `mainroad`: Categorical (Binary: yes/no - Connected to main road)
*   `guestroom`: Categorical (Binary: yes/no - Has a guest room)
*   `basement`: Categorical (Binary: yes/no - Has a basement)
*   `hotwaterheating`: Categorical (Binary: yes/no - Has hot water heating)
*   `airconditioning`: Categorical (Binary: yes/no - Has air conditioning)
*   `parking`: Ordinal/Numerical (Number of parking spots)
*   `prefarea`: Categorical (Binary: yes/no - Located in a preferred area)
*   `furnishingstatus`: Categorical (unfurnished, semi-furnished, furnished)

## Analysis Steps & Key Findings

### Part 1: Full Dataset Analysis
*   **Variable Description & Statistics:** Each variable was identified by its nature (numerical, categorical, ordinal).
    *   **Numerical Variables (`price`, `area`):** Mean, median, mode, standard deviation, IQR, quartiles, and range were calculated. Histograms showed that `price` is right-skewed.
    *   **Ordinal/Count Variables (`bedrooms`, `bathrooms`, `stories`, `parking`):** Frequencies, mode, median, and range were considered. Histograms were generated.
    *   **Binary Categorical Variables (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`):** Frequencies were calculated and visualized using pie charts.
    *   **Multi-class Categorical Variable (`furnishingstatus`):** Frequencies were calculated and visualized using a pie chart.

### Part 2: Sample Analysis
*   **Minimum Sample Size:** Calculated to be 254 for a 95% confidence level and a 3% margin of error from a population of 333.
*   **Population vs. Sample Mean:**
    *   A random sample of the calculated size was drawn.
    *   The means of numerical variables in the sample were compared to the population means.
    *   Confidence intervals for the sample means were calculated. It was found that the population means for `price`, `area`, `bedrooms`, `bathrooms`, `stories`, and `parking` all fell within their respective 95% confidence intervals of the sample means.
    *   A hypothesis test (Z-test for means, assuming normality for large samples via CLT, or non-parametric if distributions are very skewed) confirmed that the sample mean for `area` was not significantly different from the population mean, suggesting the sample is representative in this regard.

### Part 3: Hypothesis Testing
*   **Normality & Variance Checks:**
    *   Shapiro-Wilk tests indicated that `price` and `bathrooms` (treated as numerical for this test) are not normally distributed.
    *   Levene's test showed that the variances of `price` and `bathrooms` are significantly different.
*   **Chi-Square Test for Independence:**
    1.  **Price Category vs. Guestroom:**
        *   H0: Price category is independent of the presence of guestrooms.
        *   H1: There is an association between price category and guestrooms.
        *   **Result:** p-value << 0.05. Rejected H0. There's a strong association.
    2.  **Price Category vs. Number of Bathrooms:**
        *   H0: Price category is independent of the number of bathrooms.
        *   H1: There is an association between price category and the number of bathrooms.
        *   **Result:** p-value << 0.05. Rejected H0. There's a strong association.

### Part 4: Linear Regression Modeling
*   **Model:** A simple linear regression model was built to predict `price` (dependent variable) using `area` (independent variable).
*   **Training and Testing:** The dataset was split into training (70%) and testing (30%) sets.
*   **Model Coefficients:**
    *   Intercept: ~2,476,063
    *   Coefficient for `area`: ~464.17
    *   This implies that for each unit increase in area, the price is expected to increase by approximately 464.17 units, starting from a base price of ~2,476,063.
*   **Model Evaluation (OLS Summary & R-squared):**
    *   **R-squared:** Approximately 0.318 (or 31.8%). This indicates that about 31.8% of the variability in house prices can be explained by the area of the house.
    *   **P-values:** Both the intercept and the `area` coefficient were statistically significant (P>|t| ≈ 0.000).
    *   **Distribution of `price`:** A distribution plot (distplot) of `price` showed right skewness.
*   **Actual vs. Predicted Values:** A comparison showed discrepancies, aligning with the relatively low R-squared value, indicating that `area` alone is not a comprehensive predictor of `price`.

## Key Findings & Interpretation (from Regression)
*   The R-squared value of ~32% suggests that `area` is a significant predictor of `price`, but it doesn't capture the full picture. Other factors heavily influence housing prices.
*   The model coefficients are statistically significant, confirming a positive relationship between area and price.
*   The scatter plot and actual vs. predicted values highlight the model's limitations due to its simplicity.

## Proposed Business Strategy
Based on the analysis, particularly the regression insights:
1.  **Invest in Large Properties with High-Value Features:** Since larger properties generally command higher prices, focus on these. However, given that `area` alone isn't the sole driver, prioritize large properties that *also* possess other high-value features (e.g., prime location, modern amenities, number of bathrooms, guestrooms – as indicated by the Chi-Square tests).
2.  **Target Under-Valued Large Properties in Growing Neighborhoods:** Identify larger properties in areas with growth potential where current prices might not fully reflect their size due to other missing local amenities or market maturity. These could offer higher appreciation.
3.  **Diversify to Spread Risk:** The low R-squared of the simple model underscores that relying solely on `area` is risky. A diversified portfolio across different property types, locations, and price points (considering features beyond just area) is recommended.

## Setup and Environment
1.  **Python:** Python 3.x.
2.  **Required Libraries:** Pandas, NumPy, Matplotlib, Seaborn, SciPy, Scikit-learn, Statsmodels. These can be installed via pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
    ```
3.  **Data Source:** The `Housing3.csv` file is loaded directly from a GitHub URL within the notebook.

## Usage
1.  Ensure all prerequisites and libraries are installed.
2.  Open `Statitics Report Group 10.ipynb` in a Jupyter Notebook or Google Colab environment.
3.  Execute the cells sequentially to reproduce the analysis, visualizations, and model building.

## Future Work / Potential Improvements
*   **Multivariate Regression:** Develop a multiple linear regression model incorporating other significant variables (e.g., `bathrooms`, `bedrooms`, `stories`, categorical features like `prefarea`, `airconditioning`) to improve price prediction accuracy.
*   **Feature Engineering:** Create new features (e.g., price per sq. ft., interaction terms).
*   **Handle Skewness:** Apply transformations (e.g., log transformation) to the `price` variable to address its right-skewness, potentially improving model performance and assumption satisfaction for linear regression.
*   **Advanced Modeling:** Explore non-linear models or machine learning algorithms for price prediction.
*   **Cross-validation:** Implement cross-validation techniques for more robust model evaluation.
