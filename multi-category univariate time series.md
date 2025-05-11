**EDA**
data quality checks - 
1. data consistency 
2. missing values 
3. duplicates 
4. acf - plot acf category wise, wherever no data is present take is a 0 for demand 
5. checks for staionarity - catwegory wise ADF test 
6. checks for seaonality - category wise ACF plot on de-trended series
7. outliers - break the data into actegories and then compute following based on presence of trends, seasonality, stationarity (refer png) z-score, inter quartile range,
 
   
**Pre-processing**
missing value imputation - using linear interpoltion, weighted moving average 
make time stationary - log return tarnsformation, differencing of order -1,2, 

**Feature enginerring**

**Model Training**

**Model Validation**

**Model deployment**f


