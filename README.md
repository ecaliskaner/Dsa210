# Impact of Weather on Traffic in Istanbul

## Motivation  
Traffic congestion is a major issue in Istanbul, impacting daily life, commute times, and city efficiency. Weather conditions are often cited as a factor affecting traffic, but the exact relationship remains unclear. This project aims to analyze whether bad weather increases traffic by slowing down vehicles or if good weather leads to more congestion as people go out more. Understanding these trends could help city planners, businesses, and commuters make better decisions.

## Data Sources  
- **Weather Data**: [Meteostat Istanbul Weather Data](https://meteostat.net/en/station/17060?t=2025-04-11/2025-04-18&utm_source=chatgpt.com#google_vignette)  
- **Traffic Data**: [Istanbul Traffic Index (IBB)](https://data.ibb.gov.tr/en/dataset/istanbul-trafik-indeksi/resource/ba47eacb-a4e1-441c-ae51-0e622d4a18e2)
- **For Organizing**: [ChatGPT](https://chatgpt.com)
- **Help for Coding**:[Cursor](https://www.cursor.com/)

## Data Collection & Processing  
- Weather data (temperature, precipitation, wind speed) will be collected daily/hourly.  
- Traffic index data from IBB will provide congestion levels over time.  
- The datasets will be merged by timestamp for correlation analysis.  
- Additional events (e.g., football matches) may be included to assess their impact.

## Analysis Plan  
### Exploratory Data Analysis (EDA)  
- Data cleaning, visualization, and identifying patterns.  
- Checking seasonal trends and extreme weather effects.  

### Statistical Analysis  
- Correlation analysis between weather and traffic congestion.  
- Hypothesis testing to examine the effect of different weather conditions.  

### Machine Learning Models  
- Predictive models using regression or classification.  
- Time series forecasting to predict congestion based on weather conditions.  

## Tools & Libraries  
- **Programming:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib  
- **Database & Querying:** Groby  

## Expected Outcomes  
- Insights into how different weather conditions impact traffic.  
- Identification of patterns affecting peak congestion times.  
- Possible applications for traffic prediction and urban planning.  

## Challenges & Limitations  
- Incomplete or inconsistent data.  
- Other factors like roadwork, accidents, or protests affecting traffic.  
- Behavioral effects (people avoiding travel in extreme weather).  

---

