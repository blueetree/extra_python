import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import utils
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

##############################
# Load Data
##############################
Dir_PATH = ''
File_PATH = Dir_PATH + 'housing.csv'
df = pd.read_csv(File_PATH)

##############################
# Quick Look
##############################
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

##############################
# Data Conversion
##############################
feature_numeric = list(df.columns[:9])
feature_category = df.columns[9]
# convert text data to numeric
def ohe(df, feature_numeric, feature_category, num):
    '''
    Convert categorical features to numerical using OneHotEncoding
    :param df: DataFrame
    :param feature_numeric: Array. List of Column names
    :param feature_category: Array. List of Column names
    :param num: Int. Number of category columns
    :return: DataFrame
    '''
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    x_ohe = encoder.fit_transform(df[feature_category].values.reshape(-num, num))
    columnames = map(list, encoder.categories_)
    flat_names = [item for sublist in columnames for item in sublist]
    df = pd.concat([df[feature_numeric], pd.DataFrame(x_ohe, columns=flat_names)], axis=1).reindex()
    return df
df = ohe(df, feature_numeric, feature_category, 1)

##############################
# EDA + Feature Engineering
##############################
explore_df = df.copy()
target_column = "median_house_value"
# correlation plot
corr = explore_df.corr()
fig=plt.figure(figsize=(13,11))
g1 = sns.heatmap(corr, cmap='coolwarm', vmin=0, vmax=1)
g1.set_xticklabels(g1.get_xticklabels(), rotation=70, fontsize=8)
g1.set_yticklabels(g1.get_yticklabels(), rotation=15, fontsize=8)
plt.title("Correlation Plot")
# plt.savefig("Corr_Features.png", dpi=200)
# plt.show()
# print(corr[target_column].sort_values(ascending = False))


# experimenting the attribute combination
explore_df['rooms_per_household'] = explore_df["total_rooms"]/explore_df["households"]
explore_df['bedrooms_per_rooms'] = explore_df["total_bedrooms"]/explore_df["total_rooms"]
explore_df['population_per_household'] = explore_df['population']/explore_df['households']
corr = explore_df.corr()
# print(corr[target_column].sort_values(ascending = False))


# check features with high correlation with the target
# with hist
sns.set(font_scale=1.2)
j1=sns.jointplot(explore_df[target_column], explore_df['median_income'], kind='hex', color="#4CB391")
j1.set_axis_labels(target_column, 'Median_income', fontsize=15)
# plt.savefig('name.png', dpi=200)
# with KDE
j2 = sns.jointplot(explore_df[target_column], explore_df['<1H OCEAN'], kind='kde', color='orange')
j2.set_axis_labels(target_column, '<1H OCEAN', fontsize=15)
# plt.savefig('name.png', dpi=200)

df = explore_df

##############################
# Data Splitting
##############################
# split the data first to avoid contamination
X=df.drop([target_column], axis=1)
Y=df[target_column]
bins = np.linspace(14999, 500001, 5)
Y_binned = np.digitize(Y, bins)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=30, stratify=Y_binned)
print ("train feature shape: ", X_train.shape)
print ("test feature shape: ", X_test.shape)

###############################################################
# Test functions before Pipeline: Cleaning + Scaling + PCA
###############################################################
# data cleaning
imputer = Imputer(strategy = "median")
X_train = pd.DataFrame(imputer.fit_transform(X_train))
y_train = y_train.reset_index(drop=True)
train = pd.concat([X_train, y_train], axis=1)
# data scaling
scaler = StandardScaler()
scaler.fit(train)
feature_scaled = scaler.transform(train)

# Apply PCA
pca = PCA(n_components=4)
pca.fit(feature_scaled)
feature_scaled_pca = pca.transform(feature_scaled)
print("shape of the scaled and 'PCA'ed features: ", np.shape(feature_scaled_pca))

# Let's see the variance to see out of the
# 4 components which are contributing most
feat_var = np.var(feature_scaled_pca, axis=0)
feat_var_rat = feat_var/(np.sum(feat_var))
print ("Variance Ratio of the 4 Principal Components Ananlysis: ", feat_var_rat)

# scatter plot: different level of continuous target
y_label = pd.cut(y_train, 3, labels=[0, 1, 2])
target_list = y_label.tolist()
print (type(target_list))
feature_scaled_pca_X0 = feature_scaled_pca[:, 0]
feature_scaled_pca_X1 = feature_scaled_pca[:, 1]
feature_scaled_pca_X2 = feature_scaled_pca[:, 2]
feature_scaled_pca_X3 = feature_scaled_pca[:, 3]
labels = target_list
colordict = {0:'brown', 1:'darkslategray', 2:'blue'}
piclabel = {0:'low', 1:'medium', 2:'high'}
markers = {0:'o', 1:'*', 2:'X'}
alphas = {0:0.3, 1:0.4, 2:0.5}

fig = plt.figure(figsize=(12, 7))
plt.subplot(1,2,1)
for l in np.unique(labels):
    ix = np.where(labels==l)
    plt.scatter(feature_scaled_pca_X0[ix], feature_scaled_pca_X1[ix], c=colordict[l],
                label=piclabel[l], s=40, marker=markers[l], alpha=alphas[l])
plt.xlabel("First Principal Component", fontsize=15)
plt.ylabel("Second Principal Component", fontsize=15)
plt.legend(fontsize=15)
plt.subplot(1,2,2)
for l1 in np.unique(labels):
    ix1 = np.where(labels==l1)
    plt.scatter(feature_scaled_pca_X2[ix1], feature_scaled_pca_X3[ix1], c=colordict[l1],
               label=piclabel[l1], s=40, marker=markers[l1], alpha=alphas[l1])
plt.xlabel("Third Principal Component", fontsize=15)
plt.ylabel("Fourth Principal Component", fontsize=15)
plt.legend(fontsize=15)
# plt.savefig('PCAs.png', dpi=200)
# plt.show()

#################################
# Pipeline + Tuning: GridSearchCV
#################################
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
pipe_steps = [('imputer', Imputer(strategy = "median")), ('scaler', StandardScaler()),
              ('pca', PCA()), ('SupVM', SVC(kernel='rbf'))]
check_params= {
    'pca__n_components': [2],
    'SupVM__C': [0.1, 0.5, 1, 10,30, 40, 50, 75, 100, 500, 1000],
    'SupVM__gamma' : [0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50]
}
pipeline = Pipeline(pipe_steps)
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
print ("Start Fitting Training Data")
# for cv in tqdm(range(4,6))
for cv in range(4,6):
    create_grid = GridSearchCV(pipeline, param_grid=check_params, cv=cv)
    create_grid.fit(X_train, y_train)
    print ("score for %d fold CV := %3.2f" %(cv, create_grid.score(X_test, y_test)))
    print ("!!!!!!!! Best-Fit Parameters From Training Data !!!!!!!!!!!!!!")
    print (create_grid.best_params_)
print ("out of the loop")
print ("grid best params: ", create_grid.best_params_)