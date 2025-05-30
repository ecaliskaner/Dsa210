# Weather Impact on Istanbul Traffic Analysis

## Motivation
Understanding the relationship between weather conditions and traffic patterns in Istanbul is crucial for:
- Urban planning and traffic management
- Helping commuters make informed travel decisions
- Improving emergency response during adverse weather conditions
- Contributing to smart city initiatives

## Data Sources
1. Weather Data (Weather1.csv):
   - Source: [(https://meteostat.net/en/station/17060?t=2025-04-11/2025-04-18&utm_source=chatgpt.com#google_vignette)]
   - Contains daily weather measurements including:
     - Temperature (min, max, average)
     - Precipitation
     - Wind speed and direction
     - Atmospheric pressure

2. Traffic Data (traffic_index.csv):
   - Source: [https://data.ibb.gov.tr/en/dataset/istanbul-trafik-indeksi/resource/ba47eacb-a4e1-441c-ae51-0e622d4a18e2]
   - Daily traffic index measurements for Istanbul
   - Includes minimum, maximum, and average traffic indices

## Data Analysis Pipeline

### 1. Data Preparation
- DateTime format standardization
- Missing value handling:
  - Interpolation for traffic data
  - Mean imputation for wind speed and pressure
  - Zero imputation for precipitation
- Data merging and cleaning

### 2. Exploratory Data Analysis (EDA)
- Descriptive statistics for all weather parameters
- Distribution analysis
- Correlation analysis between weather factors and traffic
- Time series visualization
- Box plots for categorical analysis

### 3. Statistical Analysis
- T-tests comparing traffic patterns:
  - Rainy vs non-rainy days
  - Different temperature categories
  - Different wind speed categories
- Pearson correlation analysis
- Statistical significance testing

### 4. Machine Learning Implementation
- Linear Regression model with:
  - Weather parameters as features
  - Interaction terms between rain and other factors
  - Feature importance analysis
- Model evaluation using:
  - Mean Squared Error
  - R-squared score
  - Residual analysis

## Key Findings
1. Rain Impact:
   - Significant increase in traffic during rainy days (+4.88 points)
   - Stronger effect in colder temperatures
   - Consistent across all wind conditions

2. Temperature Effects:
   - Inverse relationship with traffic
   - Higher traffic in colder weather
   - Very Cold: 29.06 average traffic
   - Very Warm: 26.29 average traffic

3. Wind Speed Impact:
   - Slight positive correlation with traffic
   - Strongest effect when combined with rain
   - Very High Wind + Rain: highest traffic index (33.70)

4. Combined Effects:
   - Weather factors explain about 5% of traffic variation
   - Most important factors:
     1. Average temperature
     2. Minimum temperature
     3. Rain occurrence
     4. Precipitation amount

## Limitations and Future Work
1. Current Limitations:
   - Missing data in several weather parameters
   - Daily aggregation might miss hourly patterns
   - Limited to one city's data
   - No consideration of special events or holidays

2. Future Improvements:
   - Include hourly data analysis
   - Add more weather parameters (humidity, visibility)
   - Incorporate holiday and event data
   - Implement more advanced ML models:
     - Time series forecasting
     - Non-linear models
     - Neural networks



## AI Assistance Disclosure
This project utilized AI assistance for:
- Code documentation, structure and help for coding
- README organization
- Statistical analysis guidance
- Visualization suggestions
  
## Project Timeline
- March 10: Project proposal and initial setup
- April 18: Data collection and EDA
- May 23: Machine learning implementation
- May 30: Final submission and documentation

## Acknowledgments
- Data sources
- Academic advisors
- Tools and libraries used

## Dependencies
Required Python packages are listed in requirements.txt:
- pandas
- numpy
- seaborn
- matplotlib
- scipy




