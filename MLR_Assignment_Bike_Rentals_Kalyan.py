#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[75]:


#Reading the dataset
bikesdata = pd.read_csv("C://Users//kalya//Desktop//Kalyan//Upgrad AIML//Linear Regression//day.csv")


# In[76]:


bikesdata.head()


# In[77]:


#Checking the structure of the data set
bikesdata.shape


# In[78]:


#Checking if any missing values
bikesdata.info()


# In[79]:


bikesdata.describe()


# In[80]:


#Drop the first column instant and dteday as they are not useful to our analysis anyway
bikesdata.drop(['instant','dteday'], axis=1, inplace=True)
bikesdata.head()


# In[81]:


#Visualising the data
sns.pairplot(bikesdata)
plt.show()


# In[82]:


#visualizing the numeric variables of the dataset using pairplot 
sns.pairplot(bikesdata, vars=["temp", "hum",'casual','windspeed','registered','atemp','cnt'])
plt.show()


# ### As we can see from the graph above, there is strong correlation between the count variable and temperature, humidity and windspeed. We shall explore these correlation with different variables further. 

# In[83]:


#mapping categorical variables with their subcategories to help with visualization analysis 
bikesdata['season']=bikesdata.season.map({1: 'spring', 2: 'summer',3:'fall', 4:'winter' })
bikesdata['mnth']=bikesdata.mnth.map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'June',7:'July',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
bikesdata['weathersit']=bikesdata.weathersit.map({1: 'Clear',2:'Mist + Cloudy',3:'Light Snow',4:'Snow + Fog'})
bikesdata['weekday']=bikesdata.weekday.map({0:'Sun',1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat'})

bikesdata.head()


# In[84]:


#visualizing the categorical variables of the dataset using boxplot 
plt.figure(figsize=(20, 12))
plt.subplot(2, 4, 1)
sns.boxplot(x='season', y='cnt', data=bikesdata)
plt.subplot(2, 4, 2)
sns.boxplot(x='mnth', y='cnt', data=bikesdata)
plt.subplot(2, 4, 3)
sns.boxplot(x='weekday', y='cnt', data=bikesdata)
plt.subplot(2, 4, 4)
sns.boxplot(x='weathersit', y='cnt', data=bikesdata)
plt.subplot(2, 4, 5)
sns.boxplot(x='yr', y='cnt', data=bikesdata)
plt.subplot(2, 4, 6)
sns.boxplot(x='workingday', y='cnt', data=bikesdata)
plt.subplot(2, 4, 7)
sns.boxplot(x='holiday', y='cnt', data=bikesdata)
plt.show()


# Some of the observations from the plots above are as follows:¶
# People are more likely to rent bikes in the summer and the fall season
# 
# Bike rental rates are the most in September and October
# 
# Saturday, Wednesday and Thursday are the days where more bikes are rented
# 
# Most bike rentals take place in the clear weather
# 
# More bikes were rented in 2019
# 
# There is no big discernable difference in bike rental rates depending on whether it's a working day or not
# 
# Bike rental rates are higher on holidays

# In[85]:


#Now, let us try to conduct a linear regression model as we can see in the above plots about the correlation between varuables
bikesdata.shape


# In[86]:


bikesdata.drop(['atemp', 'casual', 'registered'], axis=1, inplace=True)


# In[87]:


bikesdata.shape


# In[88]:


bikesdata.head()


# In[89]:


bikesdata.describe()


# In[90]:


bikesdata.info()


# In[91]:


bikesdata.isnull().sum()


# In[92]:


#creating dummy variables for the variables of month, weekday, weathersit, seasons
month = pd.get_dummies(bikesdata.mnth, drop_first=True)
weekday = pd.get_dummies(bikesdata.weekday, drop_first=True)
weathersit = pd.get_dummies(bikesdata.weathersit, drop_first=True)
season = pd.get_dummies(bikesdata.season, drop_first=True)


# In[93]:


#adding the dummy variables to bikesdata
bikesdata = pd.concat([bikesdata,month, weekday, weathersit, season], axis=1)
bikesdata.head(5)


# In[94]:


bikesdata.columns


# In[95]:


# dropping the original columns season,mnth,weekday,weathersit as we have created the dummies for it
bikesdata.drop(['season','mnth','weekday','weathersit'], axis = 1, inplace = True)
bikesdata.head()


# In[96]:


bikesdata.columns


# In[97]:


#Convert the true/false values to 1 and 0
varlist = [ 'Aug', 'Dec', 'Feb', 'Jan', 'July', 'June', 'Mar', 'May', 'Nov', 'Oct', 'Sep',
       'Mon', 'Sat', 'Sun', 'Thu', 'Tue', 'Wed', 'Light Snow', 'Mist + Cloudy',
       'spring', 'summer', 'winter' ]
bikesdata[varlist] = bikesdata[varlist].apply(lambda x: x.map({True:1, False:0}))
bikesdata[varlist].head()


# In[98]:


#checking the shape, info of the dataset and also checking the correlation of variables in a heatmap 
bikesdata.shape


# In[99]:


bikesdata.info()


# In[100]:


#making a heatmap to showcase correlation between the new variables 
plt.figure(figsize=(20, 12))
sns.heatmap(bikesdata.corr(), cmap='YlGnBu', annot=True)
plt.title('Correlation between variables in the dataset after data preparation is done')
plt.show()


# ### Preparing the data for training the model

# In[101]:


#splitting the dataset into train and test sets
df_train, df_test = train_test_split(bikesdata, train_size=0.7, random_state=100)


# In[102]:


#checking the shape of the training dataset
df_train.shape


# In[103]:


#checking the shape of the test dataset
df_test.shape


# ### Scaling of the vaiables

# In[104]:


#we have to rescale the variables like hum, temp, windspeed, cnt as they have large values as compared to the other variables of the dataset
#we have to normalize these values using the scaler.fit_transform() 
scaler = MinMaxScaler()
scaler_var = ['hum', 'windspeed', 'temp', 'cnt']
df_train[scaler_var] = scaler.fit_transform(df_train[scaler_var])


# In[105]:


#checking the normalized values of the train set after performing scaling 
df_train.describe()


# #### As we can see the max amount is 1 for all the variables. So, we are good with the scaling

# In[106]:


# checking the correlation coefficients to see which variables are highly correlated post data preparation and rescaling

plt.figure(figsize = (26, 13))
sns.heatmap(df_train.corr(), cmap="YlGnBu", annot=True)
plt.title('Heatmap to check correlation after data preparation and rescaling')
plt.show()


# #### The count has a heavy correlation of 0.65 with temperature variable. Lets observe the variance using pairplots

# In[107]:


#checking for correlation between count and temp using a pairplot
plt.figure(figsize=[6,6])
plt.scatter(df_train.temp, df_train.cnt)
plt.title('Correlation between count vs temp')
plt.show()


# ### Train the model

# In[108]:


#building our first model using the variable temp
#preparing the variables for model building 

y_train = df_train.pop('cnt')
X_train = df_train


# In[109]:


#checking the variables
y_train.head(5)


# In[110]:


X_train.head(5)


# In[111]:


#add a constant (intercept)
X_train_sm = sm.add_constant(X_train['temp'])

#create our first model
lr = sm.OLS(y_train, X_train_sm)

#fit
lr_model = lr.fit()

#params
lr_model.params


# In[112]:


lr_model.summary()


# In[113]:


#visualising our data with a scatter plot and the fitted regression line to see the best fit line
plt.scatter(X_train_sm.iloc[:, 1], y_train)
plt.plot(X_train_sm.iloc[:, 1], 0.1690 + 0.6409*X_train_sm.iloc[:, 1], 'r')
plt.title('Fitted regression line as explained by temp')
plt.show()


# In[114]:


#adding another variable thus performing multiple regression 
#adding variable yr and checking to see if it improves the R-squared

X_train_sm = X_train[['temp', 'yr']]
X_train_sm = sm.add_constant(X_train_sm)

#create second model
lr = sm.OLS(y_train, X_train_sm)

#fit
lr_model = lr.fit()

#params
lr_model.params


# In[115]:


#checking summary with temp and yr as selected variables
lr_model.summary()


# In[116]:


#adding all variables and finding out the R-squared values
#checking all the variables in our dataset
bikesdata.columns


# In[117]:


y_train


# In[118]:


#building model with all variables
X_train_sm = sm.add_constant(X_train)

#create third model
lr = sm.OLS(y_train, X_train_sm)

#fit
lr_model = lr.fit()

#params
lr_model.params


# In[119]:


#checking summary with all the variables
lr_model.summary()


# In[120]:


#checking the number of columns in our dataset
len(bikesdata.columns)


# In[121]:


#creating the RFE object
lm = LinearRegression()
lm.fit(X_train, y_train)

#setting feature selection variables to 15
rfe = RFE(lm, n_features_to_select = 15) 

#fitting rfe ofject on our training dataset
rfe = rfe.fit(X_train, y_train)


# In[122]:


#checking the elements selected and the ones rejected in a list after rfe
list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[123]:


#getting the selected feature variables in one one variable
true_rfe = X_train.columns[rfe.support_]


# In[124]:


#checking the values of true_rfe
true_rfe


# In[125]:


len(true_rfe)


# In[126]:


#building model using selected RFE variables
#creating training set with RFE selected variables
X_train_rfe = X_train[true_rfe]


# In[127]:


#adding constant to training variable
X_train_rfe = sm.add_constant(X_train_rfe)

#creating first training model with rfe selected variables
lr = sm.OLS(y_train, X_train_rfe)

#fit
lr_model = lr.fit()

#params
lr_model.params


# In[128]:


#summary of model
lr_model.summary()


# In[129]:


#checking the VIF of the model 

#dropping the constant variables from the dataset
X_train_rfe = X_train_rfe.drop(['const'], axis = 1)


# In[130]:


#calculating the VIF of the model
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[131]:


#workingday variable can be dropped due to high VIF
X_train_new_1 = X_train_rfe.drop(['hum'], axis = 1)


# In[132]:


#adding constant to training variable
X_train_lr1 = sm.add_constant(X_train_new_1)

#creating first training model with rfe selected variables
lr = sm.OLS(y_train, X_train_lr1)

#fit
lr_model = lr.fit()

#summary
lr_model.summary()


# In[133]:


#checking the VIF of the model 

#dropping the constant variables from the dataset
X_train_lr1 = X_train_lr1.drop(['const'], axis = 1, inplace=True)


# In[135]:


#calculating the VIF of the model
vif = pd.DataFrame()
X = X_train_new_1
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[136]:


#Temp variable can be dropped due to high VIF
X_train_new_1 = X_train_rfe.drop(['temp'], axis = 1)


# In[137]:


#adding constant to training variable
X_train_lr1 = sm.add_constant(X_train_new_1)

#creating first training model with rfe selected variables
lr = sm.OLS(y_train, X_train_lr1)

#fit
lr_model = lr.fit()

#summary
lr_model.summary()


# In[140]:


#checking the VIF of the model 

#dropping the constant variables from the dataset
X_train_lr1 = X_train_lr1.drop(['const'], axis = 1)


# In[139]:


#calculating the VIF of the model
vif = pd.DataFrame()
X = X_train_new_1
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[141]:


#hum variable can be dropped due to low VIF and high p-value
X_train_new_2 = X_train_lr1.drop(['hum'], axis = 1)


# In[142]:


#adding constant to training variable
X_train_lr2 = sm.add_constant(X_train_new_2)

#creating first training model with rfe selected variables
lr = sm.OLS(y_train, X_train_lr2)

#fit
lr_model = lr.fit()

#summary
lr_model.summary()


# In[143]:


#checking the VIF of the model 

#dropping the constant variables from the dataset
X_train_lr2 = X_train_lr2.drop(['const'], axis = 1)


# In[144]:


#calculating the VIF of the model
vif = pd.DataFrame()
X = X_train_new_2
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[145]:


#hum variable can be dropped due to a high VIF
X_train_new_3 = X_train_lr2.drop(['windspeed'], axis = 1)


# In[146]:


#adding constant to training variable
X_train_lr3 = sm.add_constant(X_train_new_3)

#creating first training model with rfe selected variables
lr = sm.OLS(y_train, X_train_lr3)

#fit
lr_model = lr.fit()

#summary
lr_model.summary()


# In[147]:


#checking the VIF of the model 

#dropping the constant variables from the dataset
X_train_lr3 = X_train_lr3.drop(['const'], axis = 1)


# In[148]:


#calculating the VIF of the model
vif = pd.DataFrame()
X = X_train_new_3
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# ### For all the variables, VIF is <5. Also, P-Valye < 0.05 or almost zero. R-Squared value is 0.76. So, we are good with the model

# ### Next step : Residual analysis

# In[149]:


X_train_lr3 = sm.add_constant(X_train_lr3)
#X_train_lm5.columns
X_train_lr3


# In[150]:


#getting the y_train_pred for residual analysis
y_train_pred = lr_model.predict(X_train_lr3)


# In[151]:


y_train_pred


# In[152]:


res = y_train - y_train_pred
#distribution of the error terms shown here; distribution should be centered around 0 and should be a normal distribution
sns.distplot(res)
plt.title('Error distribution')
plt.show()


# In[153]:


#perform standardization (MinMax scaling) on test set
#checking the variables to be standardized
scaler_var


# In[154]:


#perform scaling on test data set
#train dataset is to be fit, test dataset is not fit but simply transformed
#test dataset is transformed based on fit of train dataset
df_test[scaler_var] = scaler.transform(df_test[scaler_var])


# In[155]:


df_test.describe()


# In[156]:


#building test model using the variable temp
#preparing the variables for model building 

y_test = df_test.pop('cnt')
X_test = df_test


# In[157]:


#checking the values
y_test.head(5)


# In[158]:


X_test.head(5)


# In[160]:


#Printing feature variables

X_train_lr3.columns


# In[162]:


#dropping constant
X_train_lr3.drop(columns= 'const', inplace = True)


# In[163]:


#creating new X_test dataset based on the feature variables using RFE
X_test_new = X_test[X_train_lr3.columns]

#adding a constant variable
X_test_new = sm.add_constant(X_test_new)


# In[164]:


#making predictions
y_pred = lr_model.predict(X_test_new)


# In[167]:


#build a scatter plot to observe relationship between the dependent and the feature variables

sns.pairplot(bikesdata, y_vars=X_train_lr3.columns, x_vars='cnt')
plt.figure(figsize = (16, 12))
plt.show()


# ### We can now validate the assumptions of linear regression in the model:
# #### As we can see, temperature has a linear relationship with the dependent variable (cnt).
# ##### As we have observed earlier every variable in our chosen model has a VIF<5 which ensures that there is no mulitcollinearity.
# 
# ##### The error distribution as observed above is normal (ie concentrated around 0) which is another assumption of linear regression.
# 
# ##### Step 5: Prediction and evaluation of the test set

# In[168]:


#r2 score of the test set
r2_test = r2_score(y_true=y_test, y_pred=y_pred)
print('r2 score on the test set is', r2_test)


# In[169]:


#r2 score of the training set
r2_train = r2_score(y_true=y_train, y_pred= y_train_pred)
print('r2 score on the train set is', r2_train)


# In[170]:


# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
sns.regplot(x=y_test, y=y_pred, ci=52, fit_reg=True, line_kws={"color": "red"})
plt.scatter(y_test, y_pred)
fig.suptitle('y_test vs y_pred', fontsize = 16)               
plt.xlabel('y_test', fontsize = 14)                          
plt.ylabel('y_pred', fontsize = 14) 
plt.show()


# In[171]:


lr_model.summary()


# Equation of the best fitted line is:
# cnt = 0.2468×yr - 0.06060Xholiday + 0.0486×workingday - 0.0883×Jan - 0.0086×July + 0.0765×Sep + 0.0527×Sat - 0.3220×Light Snow - 0.0868xMist+Cloudy - 0.2844xsprint - 0.0580xsummer - 0.0795xwinter

# In[172]:


#finding out the mean squared error 

train_mse = (mean_squared_error(y_true=y_train, y_pred=y_train_pred))
test_mse = (mean_squared_error(y_true=y_test, y_pred=y_pred))
print('Mean squared error of the train set is', train_mse)
print('Mean squared error of the test set is', test_mse)

