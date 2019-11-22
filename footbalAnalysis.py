#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:15:37 2019

@author: nageshsinghchauhan
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import seaborn as sns
from scipy.stats import poisson
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv("/Users/nageshsinghchauhan/Downloads/ML/kaggle/football/results.csv")
data.nunique()

#set date to datetime
data = data.astype({'date':'datetime64[ns]'})
#to check for null values in each column
data.isnull().sum() 

#What are the most tournament played ?
tournament = data['tournament'].value_counts()
tournament = tournament[:15]
plt.figure(figsize = (15,10))
sns.set_style("whitegrid")
ax = sns.barplot(y=tournament.index, x=tournament.values, palette="Reds_r", orient='h')
ax.set_ylabel('Tournaments', size=16)
ax.set_xlabel('Number of tournament', size=16)
ax.set_title("TOP 10 TYPE OF MATCH TOURNAMENTS", fontsize=18)

#What are the teams with the best goal average ?
#create two dataframe for the home and away teams
home = data[['home_team','home_score']].rename(columns = {'home_team': 'team', 'home_score': 'score'})
away = data[['away_team','away_score']].rename(columns = {'away_team': 'team', 'away_score': 'score'})
team_score = home.append(away).reset_index(drop = True)
country_info = team_score.groupby('team')['score'].agg(['sum', 'count', 'mean']).reset_index()
country_info = country_info.rename(columns = {'sum':'num_goals','count':'num_matches','mean':'goal_avg'})

plt.figure(figsize=(10,7))
sns.set_style("whitegrid")
plot_data = country_info.sort_values(by = 'goal_avg',ascending = False)[:10]
ax = sns.barplot(x="team",y="goal_avg", data=plot_data,palette="Blues_r")
plt.xlabel("Teams")## From United Kingdom users : 
plt.ylabel("Goal average")
plt.title("Teams with maximum goal average")

#What are the teams who played the most matches ?
plt.figure(figsize=(10,7))
sns.set_style("whitegrid")
plot_data1 = country_info.sort_values(by = 'num_matches',ascending = False)[:10]
ax = sns.barplot(x="team",y="num_matches", data=plot_data1,palette="Blues_r")
plt.xlabel("Teams")## From United Kingdom users : 
plt.ylabel("Number of matches")
plt.title("Teams with maximum matches played")

#What are the teams who scored the most ?
plt.figure(figsize=(10,7))
sns.set_style("whitegrid")
plot_data2 = country_info.sort_values(by = 'num_goals',ascending = False)[:10]
ax = sns.barplot(x="team",y="num_goals", data=plot_data2,palette="Blues_r")
plt.xlabel("Teams")## From United Kingdom users : 
plt.ylabel("Number of goals")
plt.title("Teams with maximum goals scored")


##Get correlation between different variables
corr = data.corr(method='kendall')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
data.columns


#Add winner coloumn
def winner(row):
    if row['home_score'] > row['away_score']:
        return 'winner'
    if row["home_score"] == row["away_score"]:
        return 'Draw'
    if row["home_score"] < row["away_score"]:
        return 'lose'

data['Winner'] = data.apply(lambda row:winner(row), axis = 1)
data['neutral'] = [1 if x == 'False' else 0 for x in data['neutral']]

"""
X = data.iloc[:,1:9]
y = data.iloc[:,-1]

#Converting categorial data
X = pd.get_dummies(X, prefix=['home_team', 'away_team', 'tournament',
       'city', 'country',], drop_first=True)

y = pd.get_dummies(y, prefix = ['Winner'], drop_first = True)

#splitting the dataset into test and training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.2, random_state= 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#kernel Support Vector machine model
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',random_state = 0)
classifier = classifier.fit(X_train,y_train)

#prediction
y_pred = classifier.predict(X_test)

from sklearn import metrics

print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


"""