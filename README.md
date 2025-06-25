Project:  Module 11: Practical Application 2
===

# Author: Rana Obulam

Dataset information
---
The provided dataset contains information on 426K cars to ensure the speed of processing. Your goal is to understand what factors make a car more or less expensive. As a result of your analysis, you should provide clear recommendations to your client—a used car dealership—as to what consumers value in a used car.

DataSet info: Dataset has 426880 records and 18 columns.

Data Source: [https://github.com/OBULAMRANA/accept_coupon/blob/main/coupons.csv](https://github.com/OBULAMRANA/M11-PA-2/blob/main/vehicles.csv.zip)

Python Code: [https://github.com/OBULAMRANA/M11-PA-2/blob/main/M11_PA_2.ipynb](https://github.com/OBULAMRANA/M11-PA-2/blob/main/M11_PA_2.ipynb)

### Data Understanding & Cleaning
---
Many columns are having missing data.
```
df.isna().mean().round(4)*100
0
id	0.00
region	0.00
price	0.00
year	0.28
manufacturer	4.13
model	1.24
condition	40.79
cylinders	41.62
fuel	0.71
odometer	1.03
title_status	1.93
transmission	0.60
VIN	37.73
drive	30.59
size	71.77
type	21.75
paint_color	30.50
state	0.00

dtype: float64
```
Replaced NaN and INF values in price column with mean of price
```
has_nan = df['price'].isna().any().sum()
has_inf = np.isinf(df['price']).any().sum()
df_cleaned = df[~np.isinf(df['price'])]
has_inf = np.isinf(df_cleaned['price']).any().sum()
print(has_nan)
0
print(has_inf)    
0

```
Checked for duplicated.No duplicated fournd
```
sum(df.duplicated())
     
0
```
Boxplot - PriceLog
---
![image](https://github.com/user-attachments/assets/d939f13c-f0cb-4eac-901f-cfe2bc5dbc8d)



Histogram Plot - Price log
---
![image](https://github.com/user-attachments/assets/61c77df8-0953-4625-a875-06ed12a6368e)

Lost data with Z-SCORE (< 1%) vs IRQ (98%)
---
```
df_zscore = df[np.abs(stats.zscore(df['price'])) < 0.5].copy()
     

df_zscore.shape
     
(426849, 19)

zscore_data_lost = 1 - (df_zscore.shape[0]/df.shape[0])
print("We lost {:.6%} of the data by the z-score method" .format(zscore_data_lost))
     
We lost 0.007262% of the data by the z-score method

df_zscore['price'].describe()
     
price
count	426849.00
mean	17552.14
std	20667.53
min	0.00
25%	5900.00
50%	13950.00
75%	26455.00
max	5000000.00
```

```
irq_data_lost = 1 - (df_irq.shape[0]/df.shape[0])
print("We lost {:.2%} of the data by the IRQ method" .format(irq_data_lost))
     
We lost 98.08% of the data by the IRQ method


df_irq['price'].describe()
     
price
count	8177.00
mean	3088930.26
std	87973256.90
min	57400.00
25%	61000.00
50%	67995.00
75%	77999.00
max	3736928711.00

```
Violin plot - title_status vs price log
---
![image](https://github.com/user-attachments/assets/070554cb-b558-4365-a19c-8897e1b64b00)


Violin plot - condition vs price log
---
![image](https://github.com/user-attachments/assets/fc14714d-a9c9-4d11-93e8-4231e5be213b)


Conclusion & Recommendation
---

Out of all the columns gien in dataset, columns named 'region', 'manufacturer', 'model', 'drive', 'size', 'type', 'paint_color' & 'state' as too many categorical variables that will be overfitting the model and hence ignored those column data. If we apply One-Hot encoding on those columns, its going to generate 100s of columns which will overload the model to process.

Column named 'id', 'VIN', 'fuel', & 'cylinders' does not play a role on car price and hence dropped from the DataSet

Column named 'odometer', 'title_status' & 'condition' is going to play a role as per domain knowledge that I have and hence considered for modeling. Applied one-hot encoding on catorigical data like 'title_status' & 'condition' and applied three different models like Linear-Regression, Losso-Regression and Ridge-Regression and all these three models gives almost ~0 on Train_R2_Score and Test_R2_Score. This shows the the models are under-fit and hence not able to predict actual price of the model

This under-fitting could be due to the 'most_frequent' imputer method used to fillin missing data for columns 'title_status' & 'condition'.

Violin plot on 'title_status' VS PriceLog & 'condition' VS PriceLog shows that the dependency is almost close to mean of the given data and hence the model. This data suggests that the model might be able to predict actual value given the data considered for models designed.

Overall the suggestions would be that the 'odometer', 'title_status' & 'condition' is going to play a role on price of a car!





