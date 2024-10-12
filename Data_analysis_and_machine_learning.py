# %% [markdown]
# # Understanding the Loan Data Set
# ## Step 1: Importing Necessary Libraries For Data Exploration

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import mean_absolute_error, accuracy_score


# %% [markdown]
# ## Step 2: Importing The Dataset and Looking at its dimentions and values

# %% [markdown]
# **So the Raw csv file contains 167608 enteries and 35 column. For sure we need some "Data Cleaning"**

# %%
file_path = 'C:\\Users\\aa695\\Downloads\\lc_large.csv'

data_set = pd.read_csv(file_path)
data_set.shape


# %%
data_set.describe()

# %% [markdown]
# Looking for Missing Values is the first step for **"Data Cleaning"**. So,

# %%
data_set.isna().sum()

# %% [markdown]
#     Thats alot of missing data right there. This could be due to inefficiency of the data entry procedure, but we are not going to talk about it. So, Now we will be Dropping Unnecessary Columns and Removing the Missing Values.

# %%
cleaned_data = data_set.dropna()

cleaned_data.isna().sum()



# %% [markdown]
# Now the Data Looks Clean and Okay to work with as it has no missing value. Before analyzing the data we must check the data type and name of each columns so that we won't face any error in the feature. 

# %%
cleaned_data.dtypes

# %% [markdown]
# As the Data Seems to have improper Names and Data Types for most of the columns. So we will begin by **Renamming The Columns** for better understanding of what eacg solumn says

# %%
df_cleaned = cleaned_data.rename(columns={'loan_status': 'Loan_Status', 'loan_amnt': 'Loan_amount','issue_d': 'Issue_Date' , 'term': 'Time_period',
       'total_pymnt':'Recieved_Payment', 'title': 'Lending_Reason',  
        'verification_status':'Income_Verification'}).copy()

df_cleaned.head(5)

# %% [markdown]
# See! How good the table looks with proper headings, Now its time to **Correct the Data Types** of the columns

# %%
df_cleaned["Issue_Date"] = pd.to_datetime(df_cleaned["Issue_Date"])
df_cleaned['Year'] = df_cleaned['Issue_Date'].dt.year


df_cleaned["earliest_cr_line"] = pd.to_datetime(df_cleaned["earliest_cr_line"])

df_cleaned.head(5)

# %% [markdown]
# ## Step 3: Exploring Relationship Between Variables
# Since the Data is cleaned and good to work with, it's time to find relationships and distribution of data in our table using **Visualizations**.
# 
# Plotting Features Distribution

# %% [markdown]
# *On Analysis, we found that the bank released the most number of loans back in 2016 of worth $ 57.5 Million*

# %%
plot = df_cleaned["Year"].value_counts().plot(kind= 'bar', title='Top Year for Lending most ammount of Loans', xlabel='Years', ylabel= ' Number Of Loans Issued' )


# %% [markdown]
# *After Analysing the Data we come to the conclusion that the bank released over $141 Million worth of loans from 2016 to 2018 but only recieved $100 Millions of the Loan back, creating a backlog of $41 Million*

# %% [markdown]
# **Amount of Loan Released on average**

# %%
df_cleaned['Loan_amount'].plot(kind = 'hist'  , title= 'Loan realeases')

# %%
Total_loan_amount= df_cleaned['Loan_amount'].sum()
f"${Total_loan_amount}"

# %%
Total_loan_recieved= df_cleaned['Recieved_Payment'].sum()
f"${Total_loan_recieved:.2f}"

# %%
import seaborn as sns

# %% [markdown]
# **Number Of Loans Released By The Bank Over the Period**

# %%
sns.lineplot(data = df_cleaned, x = 'Issue_Date', y= 'Loan_amount')

# %% [markdown]
# **Number Of Loans Paid Back to The Bank Over the Period**

# %%
sns.lineplot(data = df_cleaned, x = 'Issue_Date', y= 'Recieved_Payment')

# %% [markdown]
# *Clearly There is a decline in the amount of Loan Paid Back. Therefore, the back needs a system that is based on **Machine Learning** to release further loans.*

# %%
sns.pairplot(data = df_cleaned.head(10000), vars = ['Issue_Date','Loan_amount', 
        'Recieved_Payment', 
        'open_acc', 'dti'], hue= 'Loan_Status')

# %% [markdown]
# *In Order to perform further Analysis, it is must to know the **Corellations**, So,*

# %%
corr = df_cleaned[['Loan_amount',   'int_rate',
       'installment', 
       'annual_inc']].dropna().corr()
corr

# %% [markdown]
# *Ignoring the **Diagonal Enteries**, we can look for the values arround the diagonal*

# %%
sns.heatmap(corr, annot= True)

# %% [markdown]
# ### Step 4: Answering Some Questions
# 
# How Likely are you to get a loan?
# 
# *Creating a **Random Forest Classifier** as it's the best Model for Training Data which is categorical of nature, So*

# %% [markdown]
# *First We need to convert the **Categorical Data** to Numerical Figures so that computer can understant the underlying pattern in our data and make accurate predictions*

# %%
df_cleaned['is_train'] = np.random.uniform(0,1, len(df_cleaned)) >= 0.7
df_cleaned['Loan_Status_num'] = pd.factorize(df_cleaned['Loan_Status'])[0].copy() 
df_cleaned['grade_num'] = pd.factorize(df_cleaned['grade'])[0].copy() 
df_cleaned['Lending_Reason_num'] = pd.factorize(df_cleaned['Lending_Reason'])[0].copy()
df_cleaned['home_ownership_num'] = pd.factorize(df_cleaned['home_ownership'])[0].copy()
df_cleaned['Time_period_num']  = df_cleaned['Time_period'].str.extract(r'(\d+)').astype(int) 

df_cleaned.head(5)

# %% [markdown]
# *For Testing Purpose and In order to know the **Accuracy** of our model, we will split the data as Test and Train*

# %%
train, test = df_cleaned[df_cleaned['is_train']== True],df_cleaned[df_cleaned['is_train'] == False] 
print(len(test), len(train))


# %% [markdown]
# *So The **Train Data has 7093 rows** which is 70% while the **Test Data Has 3066 rows** which is 30% of the entire data*

# %%
features = [ 'Loan_amount',  'Time_period_num', 'int_rate',
       'installment', 'Recieved_Payment', 'Lending_Reason_num', 'home_ownership_num',
       'annual_inc',  'pub_rec',
       'revol_bal', 'revol_util', 'tot_coll_amt', 'tot_cur_bal', 'total_acc',
       'open_acc', 'total_rev_hi_lim',  'is_train']
X = train[features]
Y = train['grade_num']

from sklearn.ensemble import RandomForestClassifier


Loan_model = RandomForestClassifier(random_state = 0)

Loan_model.fit(X,Y)


# %%

prediction_Y = Loan_model.predict(test[features])
prediction_Y


# %% [markdown]
# *As we can see that the prediction is in the form of array of the numerical figures. In order to understant what these predictions are, we must revert the change using **factorize***

# %%
original_grades = pd.factorize(df_cleaned['grade'])[1]
original_grades

# %% [markdown]
# *We will map the "prediction_grades" to the "original_grades"*

# %%
prediction_grades = original_grades[Loan_model.predict(test[features])]
prediction_grades

# %%
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test['grade_num'], prediction_Y)
accuracy_per = accuracy*100
print(f" The Prediction is upto {accuracy_per:0.2f} % Accurate")


