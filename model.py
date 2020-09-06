import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import sklearn


df = pd.read_csv('salary_data_cleaned.csv')

print(df.columns)


#choose relevant columns
df_model = df[['Average Salary','Rating','Size','Type of ownership','Industry','Sector','Revenue', 'Hourly','Employer Provided','Job State','Job at HQ State',
               'Company Age','Python_yn','Spark_,yn','AWS_yn','Excel_yn']]
df_model['Number of Competitors'] = df.apply(lambda x: len(x.Competitors.split(',')) if x.Competitors != -1 else 0, axis = 1)

print(df_model.head())

#get dummy data
df_dum = pd.get_dummies(df_model)
print(df_dum.head())

#train test split
from sklearn.model_selection import train_test_split, cross_val_score

x = df_dum.drop('Average Salary', axis=1)
y = df_dum['Average Salary'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#multiple linear regression
#statsmodel
import statsmodels.api as sm

x_sm = x = sm.add_constant(x)
model = sm.OLS(y,x_sm)
summ_sm = model.fit().summary()
print(summ_sm)

#sklearn
from sklearn.linear_model import LinearRegression, Lasso
lm = LinearRegression()
lm.fit (x_train, y_train)
c_v_s = cross_val_score(lm, x_train, y_train, scoring = 'neg_mean_absolute_error', cv=3)
print(c_v_s)

#lasso regressions
lm_l = Lasso()
c_v_l = np.mean(cross_val_score(lm, x_train, y_train, scoring = 'neg_mean_absolute_error', cv=3))
print(c_v_l)

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha = (i/100))
    error.append(np.mean(cross_val_score(lm, x_train, y_train, scoring = 'neg_mean_absolute_error', cv=3)))

# random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
c_v_r = cross_val_score(rf, x_train, y_train, scoring = 'neg_mean_absolute_error', cv=3)
print(np.mean(c_v_r))

# tune models using gridsearchcv
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV (rf, parameters, scoring = 'neg_mean_absolute_error', cv=3)
gs.fit(x_train,y_train)
gs.best_score_
print(gs.best_score_)
print(gs.best_score_)
print(gs.best_estimator_)
print(gs.best_estimator_)

# test esembles
tprod_rf = gs.best_estimator_.predict(x_test)

from sklearn.metrics import mean_absolute_error
m_a_e = mean_absolute_error(y_test, tprod_rf)

