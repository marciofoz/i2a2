''' 
   from https://medium.com/analytics-vidhya/random-forest-on-titanic-dataset-88327a014b4d
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'
sns.set(style='whitegrid', palette='Set2', font_scale=1.2)

local_zip = 'titanic.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('titanic')
zip_ref.close()

train_data = pd.read_csv('titanic/train.csv')
test_data = pd.read_csv('titanic/test.csv')

print('\nNull Values in Training \n{}'.format(train_data.isnull().sum()))
print('Null Values in Testing \n{}'.format(test_data.isnull().sum()))
print('\nDuplicated values in train {}'.format(train_data.duplicated().sum()))
print('Duplicated values in test {}'.format(test_data.duplicated().sum()))
print('Embarkation per ports \n{}'.format(train_data['Embarked'].value_counts()))
# since the most common port is Southampton the chances are that the missing one is from there
train_data['Embarked'].fillna(value='S', inplace=True)
print('\nEmbarkation per ports after filling \n{}'.format(train_data['Embarked'].value_counts()))

train_data['Fare'].fillna(value=train_data.Fare.mean(), inplace=True)
test_data['Fare'].fillna(value=test_data.Fare.mean(), inplace=True)
train_data['Age'].fillna(value=train_data.Age.mean(), inplace=True)
test_data['Age'].fillna(value=test_data.Age.mean(), inplace=True)

print('\nNull Values in Training \n{}'.format(train_data.isnull().sum()))
print('\nNull Values in Testing \n{}'.format(test_data.isnull().sum()))

categories = {"female": 1, "male": 0}
train_data['Sex']= train_data['Sex'].map(categories)
test_data['Sex']= test_data['Sex'].map(categories)

categories = {"S": 1, "C": 2, "Q": 3}
train_data['Embarked']= train_data['Embarked'].map(categories)
test_data['Embarked']= test_data['Embarked'].map(categories)

train_data = train_data.drop(['Cabin', 'Name','Ticket','PassengerId'], axis=1)
test_data = test_data.drop(['Cabin','Name','Ticket','PassengerId'], axis=1)

y = train_data['Survived']
train_data = train_data.drop('Survived', axis=1)  # Dropping label to normalize

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_data)
scaled_test = scaler.transform(test_data)

scaled_train = pd.DataFrame(scaled_train, columns=train_data.columns, index=train_data.index)
scaled_test = pd.DataFrame(scaled_test, columns=test_data.columns, index=test_data.index)
scaled_train.head()

X_train, X_test, y_train, y_test = train_test_split(scaled_train, y, test_size=0.2)

# First Random Forest
clf = RandomForestClassifier(n_estimators=1000,random_state=42)
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

feature_imp = pd.Series(clf.feature_importances_, index=scaled_train.columns).sort_values(ascending=False)

y_pred = clf.predict(X_test)
print("1st Accuracy (all features): {}".format(metrics.accuracy_score(y_test, y_pred)))

plt.figure(figsize=(10,6))
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.tight_layout()

# Removing less important features
new_train = scaled_train.drop(['Parch','Embarked'], axis=1)
new_test = scaled_test.drop(['Parch','Embarked'], axis=1)

# Second Random Forest
X_train, X_test, y_train, y_test = train_test_split(new_train, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=1000,random_state=42)
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("2nd Accuracy (without Parch/Embarked): {}".format(metrics.accuracy_score(y_test, y_pred)))

#print(classification_report(y_test,y_pred))

