# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:15:17 2020

@author: Devang  Mehrotra
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#database
dataset = pd.read_csv('data1.csv')

#find all null values
dataset.isnull().sum()

#drop null values
dataset.dropna(inplace=True)

#univariate analysis
dataset['Gender'].value_counts()

#compute avg age for each gender(both colmns in a db)
dataset.groupby('gender')['age'].mean()

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 2:3])
X[:,2:3] = imputer.transform(X[:, 2:3])


imputer1 = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value='C')
imputer1 = imputer1.fit(X[:, 6:])
X[:,6:] = imputer1.transform(X[:, 6:])

# Taking care of missing data
df['string column name'].fillna(df['string column name'].mode().values[0], inplace = True)
df['numeric column name'].fillna(df['numeric column name'].mean(), inplace = True)
df['column name'].fillna(0, inplace = True)

#encoding categorical data
dataset['sex']=dataset['sex'].replace(['female','male','other'],[0,1,2])
dataset['smoking']=dataset['smoking'].replace(['never','quit','yes'],[0,1,2])
dataset['working']=dataset['working'].replace(['home','never','stopped','travel critical','travel non critical'],[1,0,2,3,4])
dataset['income']=dataset['income'].replace(['blank','gov','high','med','low'],[0,4,3,2,1])
dataset['blood_type']=dataset['blood_type'].replace(['unknown','abn','abp','an','ap','bn','bp','on','op'],[0,1,2,3,4,5,6,7,8])
dataset['insurance']=dataset['insurance'].replace(['no','yes'],[0,1])

#encoding categorical data
def convert_words_to_int(word):
        word_dict={"one":1,"two":2}
        return word_dict[word]
X["expereince"]=X["expereince"].apply(lamda x: convert_words_to_int(x))

#encoding categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0,4,5,9])],remainder='passthrough')
X= np.array(ct.fit_transform(X), dtype=np.float)

#Avoiding the dummy variable trap
X=X[:,[0,1,3,4,5,6,8,9,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27]]

#selecting best features
#apply SelectKBest class to extract top 10 best features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
fs = SelectKBest(score_func=chi2, k='all')
fs.fit(X_train, y_train)
X_fs = fs.transform(X_train)
X_fs = fs.transform(X_train)
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
    
#selecting best features
import statsmodels.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
#50 here is number of rows in X,change that acc
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
#remove umimp indep variables
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#select rown with p>0.05
    
#selecting best features 
X_train=X_train[:,[0,1,3,4]]
X_test=X_test[:,[0,1,3,4]]

#defining X and y
X1=dataset.iloc[:,[0,1,2,3,4,5,6,9,10,12,18,19,20,21,26,32]].values
y1=dataset.iloc[:,33].values
X2=dataset.iloc[:,0:33].values
y2=dataset.iloc[:,34].values


#split into training and test set
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.25,random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.25,random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting
y_pred1 = regressor1.predict(X_test1)
y_pred2 = regressor2.predict(X_test2)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc)

#Accuracy(dont use this)
print(classifier.score(X_train,y_train))
print(classifier.score(X_test,y_test))

#Save into csv file
kaggle_data = pd.DataFrame({'PassengerId':dataset2.PassengerId, 'Survived':y_pred}).set_index('PassengerId')
kaggle_data.to_csv('sub1.csv')

#manual input
new_input=np.array([gender,age,height,weight,income,smoking,alcohol,contacts,totalpeople,working,masks,symptoms,contactsinfected,asthma,lung,healthworker])
new_input1=new_input.reshape(1,-1)
new_output = regressor.predict(new_input1)
print("\nRisk of getting covid-19", new_output,"%")
OR
regressor.predict([[gender,age,height]])

#if used feature scaling, then manual input-
new_output=sc_x.inverse_transform(regressor.predict(sc_x.transform(np.array([6.5]))))

#Creating a pickle file
import pickle
filename = 'finalized_model'
pickle.dump(regressor, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))

#better way for pickle
import pickle
pickle_out=open("classifier.pkl","wb")
pickle.dump(classifier,pickle_out)
pickle_out.close()

#Accuracy using pickle file
result = loaded_model.score(X_test, y_test)
print("Accuracy ",result*100)

#diagram for regression
#Visualising the Training Set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the Test Set results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#diagram for classification
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
