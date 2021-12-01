import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('teleCust1000t.csv')
print(df.head())

df['custcat'].value_counts()
print(df.columns)

X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender',
        'reside']].values
print(X[0:5])

y = df['custcat'].values
print(y[0:5])

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# Create a Gaussian Classifier
model = GaussianNB()


# Train the model using the training sets
model.fit(X_train, y_train)

yhat = model.predict(X_test)
print(yhat[0:5])

print("Train set Accuracy: ", metrics.accuracy_score(y_train, model.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))