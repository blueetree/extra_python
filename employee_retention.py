import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import seaborn as sns

##############################
# Load Data
##############################
Dir_PATH = ''
File_PATH = Dir_PATH + 'employee_retention_data.csv'
df = pd.read_csv(File_PATH)

##############################
# Quick Look
##############################
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
for column in df.columns:
    uniques = sorted(df[column].unique())
    print('{0:20s} {1:5d}\t'.format(column, len(uniques)), uniques[:5])
# Remove outliers: the max of seniority is 99
print(df[df['seniority'] > 50])
df = df[df['seniority'] < 50]
# check uniqueness
duplicateRowsDF = df[df.duplicated(['employee_id', 'company_id'])]
print(duplicateRowsDF)
# convert date
df['join_date'] = pd.to_datetime(df['join_date'], errors='coerce')
df['quit_date'] = pd.to_datetime(df['quit_date'], errors='coerce')

##############################
# Create new table
##############################
temp_df = df.copy()
temp_df['join_date'] = temp_df['join_date'].dt.date
temp_df['quit_date'] = temp_df['quit_date'].dt.date
# create date list
dates = pd.date_range(start='2011/01/24', end='2015/12/13')
temp = {'date' : dates}
temp = pd.DataFrame(data=temp)
temp['key'] = 0
# create company list
unique_companies = temp_df.company_id.unique()
temp1 = {'company_id':unique_companies}
temp1 = pd.DataFrame(data=temp1)
temp1['key'] = 0
# create cartesian product of date and company -> table merged
merged = pd.merge(temp, temp1,on='key')
merged.drop('key', 1, inplace=True)
# count number of join
join_by_day = temp_df.groupby(['company_id','join_date'])['company_id','join_date'].sum()
join_by_day.columns = ['number_quit']
join_by_day.reset_index(inplace=True)
join_by_day.columns = ['company_id','date','number_join']
join_by_day['date'] = pd.to_datetime(join_by_day['date'])
# count number of quit
quit_by_day = temp_df.groupby(['company_id','quit_date'])[['company_id','quit_date']].sum()
quit_by_day.columns = ['number_quit']
quit_by_day.reset_index(inplace=True)
quit_by_day.columns = ['company_id','date','number_quit']
quit_by_day['date'] = pd.to_datetime(quit_by_day['date'])
# join table merged with join and quit table -> daily change
test = pd.merge(merged,quit_by_day,left_on=['company_id','date'],right_on=['company_id','date'],how='left').fillna(0)
test = pd.merge(test,join_by_day,left_on=['company_id','date'],right_on=['company_id','date'],how='left').fillna(0)
test['employee_count'] = test['number_join']-test['number_quit']
test.drop(['number_quit','number_join'],1, inplace=True)
# add with the last row
employee_count_table = test.groupby(['date', 'company_id']).sum().groupby(['company_id']).cumsum()
employee_count_table.reset_index(inplace=True)

# import datetime
# st = datetime.date(2011, 1, 24)
# end = datetime.date(2015, 12, 13)
# step = datetime.timedelta(days=1)
# date_list = []
# while st < end:
#     date_list.append(st)
#     st += step
# headcount_df = pd.DataFrame(columns=['day', 'employee_headcount', 'company_id'])
# headcount = 0
# company_ids = temp_df['company_id'].unique()
# # the first row
# for c in company_ids:
#     for j in temp_df['join_date']:
#         if j == datetime.date(2011, 1, 24):
#             headcount += 1
#     headcount_df = headcount_df.append({'day': datetime.date(2011, 1, 24),
#                                         'employee_headcount': headcount, 'company_id': c}, ignore_index=True)
#     headcount = 0
# # following rows
# for d in date_list[1:]:
#     for c in company_ids:
#         headcount += len(temp_df[(temp_df['join_date'] == d) & (temp_df['company_id'] == c)])
#         headcount -= len(temp_df[(temp_df['quit_date'] == d) & (temp_df['company_id'] == c)])
#         day = d - datetime.timedelta(days=1)
#         ori = headcount_df[(headcount_df['day'] == day) & (headcount_df['company_id'] == c)]['employee_headcount']
#         total = int(ori.values) + headcount
#         print(d, c, total)
#         headcount_df = headcount_df.append({'day': d,
#                                             'employee_headcount': total,
#                                             'company_id': c}, ignore_index=True)
#         headcount = 0
# headcount_df.to_csv('headcount.csv', index=False)

##############################
# EDA
##############################
# Explore the distributions of length of employment
length = pd.to_datetime(df['quit_date'])-pd.to_datetime(df['join_date'])
length1 = length.apply(lambda x:float(x.days))
plt.set_xlabel('Employment Length')
plt.set_ylabel('Count')
plt.hist(length1[np.isfinite(length1)],bins=100)
plt.axvline(x=500,color='red')
plt.axvline(x=396,color='orange')
plt.axvline(x=250,color='yellow')
plt.show()
df['length'] = length1
df['length'] = df['length'].fillna(-1)

# headcount_df = pd.read_csv('headcount.csv')
df['quit'] = df['quit_date'].notnull()*1
print(df['quit'].value_counts())
# sns.countplot(x='quit', data=df, palette='hls')

count_no_quit = len(df[df['quit']==0])
count_quit = len(df[df['quit']==1])
pct_of_no_quit = count_no_quit/(count_no_quit+count_quit)
print("percentage of stay is", pct_of_no_quit*100)
pct_of_quit = count_quit/(count_no_quit+count_quit)
print("percentage of quit", pct_of_quit*100)
print(df.groupby('quit').mean())
# OBSERVATIONS
# The average salary of employees who stay is higher than that of the employees who didn't
pd.crosstab(df['company_id'], df['quit']).plot(kind='bar')
plt.title('Quit Frequency for Company')
plt.xlabel('Company')
plt.ylabel('Frequency of quit')

table = pd.crosstab(df['company_id'], df['quit'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Company vs Employment')
plt.xlabel('Company ID')
plt.ylabel('Proportion of Employment')
plt.show()

# datetime -> for EDA only
# df['join_month'] = df['join_date'].dt.month
# df['quit_month'] = df['quit_date'].dt.month
# df['join_quarter'] = df['join_date'].dt.quarter
# df['quit_quarter'] = df['quit_date'].dt.quarter
# join_month = df.groupby(['join_month']).size()
# join_month.index = join_month.index.astype('int', copy=False)
# quit_month= df.groupby(['quit_month']).size()
# quit_month.index = quit_month.index.astype('int', copy=False)
# join_quarter = df.groupby(['join_quarter']).size()
# join_quarter.index = join_quarter.index.astype('int', copy=False)
# quit_quarter = df.groupby(['quit_quarter']).size()
# quit_quarter.index = quit_quarter.index.astype('int', copy=False)
# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(100, 100))
# ax[0, 0].plot(join_month, 'r')
# ax[0, 0].set_title('join_month')
# ax[1, 0].plot(quit_month, 'b')
# ax[1, 0].set_title('quit_month')
# ax[0, 1].plot(join_quarter, 'g')
# ax[0, 1].set_title('join_quarter')
# ax[1, 1].plot(quit_quarter, 'k')
# ax[1, 1].set_title('quit_quarter')
# plt.show()

# convert type
feature_numeric = list(df.columns)
feature_numeric.remove('dept')
feature_category = 'dept'
# convert text data to numeric
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
x_ohe = encoder.fit_transform(df[feature_category].values.reshape(-1, 1))
columnames = map(list, encoder.categories_)
flat_names = [item for sublist in columnames for item in sublist]
df = pd.concat([df[feature_numeric], pd.DataFrame(x_ohe, columns=flat_names)], axis=1).reindex()
# get_dummies approach
# df_dept = pd.get_dummies(df['dept'])
# df = pd.concat([df, df_dept], axis=1)
# df.head()

# add new features
df['salary/seniority'] = df['salary'] / df['seniority']

#Using Pearson Correlation
corr = df.corr()
sns.heatmap(corr)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax1 = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# feature importance
X = df[['employee_id', 'company_id', 'seniority', 'salary', 'join_month',
        'quit_month', 'join_quarter', 'quit_quarter', 'customer_service',
        'data_science', 'design', 'engineer', 'marketing', 'sales', 'salary/seniority']]
y = df[['quit']]

print(X.isnull().sum())
print(y.isnull().sum())

X = X.fillna(0)

from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)

print("Feature ranking:")
#
for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))

# data segmentation ideas
# split length
def is_employed(row):
    if row['length'] < 396 and row['quit_date'] == row['quit_date']:
        val = 1
    else:
        val = 0
    return val
df['employed'] = df.apply(is_employed,axis=1)
# split salary
df['salary_percent'] = pd.qcut(df['salary'], 50, labels=False)
# salary_percent = df.groupby(['salary_percent'])['employed'].sum()
percentiles = df.groupby('salary_percent')['employed'].agg(['sum','count'])
percentiles['percent'] = percentiles['sum']/percentiles['count']
plt.plot(percentiles.index,percentiles['percent'])
plt.set_xlabel('Salary Quantile')
plt.set_ylabel('Percentage who Quit Early')
plt.show()
# split seniority
