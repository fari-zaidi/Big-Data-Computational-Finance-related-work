import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LassoCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')


# Load the dataset
df = pd.read_csv('MLF_GP1_CreditScore.csv')


cat_col = [col for col in df.columns if df[col].dtype == 'object']
print('Categorical columns:', cat_col)

#converting categorical data column
def convert_categorical_to_numerical(data, col):
    LE = LabelEncoder()

    # encoding the categorical columns in the DataFrame using LabelEncoder
    for col in cat_col:
        if data[col].dtype == 'object':
            data[col] = LE.fit_transform(df[col])

    return data

cat_free_data = convert_categorical_to_numerical(df, cat_col)

# Define the input and target variables
X = df.drop(['InvGrd'], axis=1)
y = df['InvGrd']
print(X.head(4))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("---------------------------------------------------------------")
print("-----------------Linear Regression Ridge(L1)-------------------")
print("---------------------------------------------------------------")
# Linear Regression with Ridge Regularization
reg_ridge = Ridge(alpha=1.0, fit_intercept=True, normalize=True)
reg_ridge.fit(X_train, y_train)
ridge_prd = reg_ridge.predict(X_test)
ridge_prd_binary = [1 if pred >= 0.5 else 0 for pred in ridge_prd]
ridge_acc = accuracy_score(y_test, ridge_prd_binary)
print("Ridge Regularization Accuracy: %.2f%%" % ((ridge_acc)*100))
print("---------------------------------------------------------------")
reg_ridge = RidgeClassifier(alpha=0.5, solver='auto', random_state=42)
reg_ridge.fit(X_train, y_train)
y_pred = reg_ridge.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred)) 
print("---------------------------------------------------------------")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# print("")
# print("")
# print("Classification Report:")
# print(classification_report(y_test, ridge_prd_binary)) 
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, ridge_prd_binary))


print("---------------------------------------------------------------")
print("---------------------------------------------------------------")
print("----------------Linear Regression Lasso(L2)--------------------")
print("---------------------------------------------------------------")
# Linear Regression with Lasso Regularization
reg_lasso = Lasso(alpha=1.0, fit_intercept=True, normalize=True)
reg_lasso.fit(X_train, y_train)
lasso_pred = reg_lasso.predict(X_test)
lasso_pred_binary = [1 if pred >= 0.5 else 0 for pred in lasso_pred]
lasso_acc = accuracy_score(y_test, lasso_pred_binary)
print("Lasso Regularization Accuracy: %.2f%%" % ((lasso_acc)*100))
print("---------------------------------------------------------------")
reg_lasso = LassoCV(cv=5, random_state=42)
reg_lasso.fit(X_train, y_train)
y_pred = reg_lasso.predict(X_test).round()

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("---------------------------------------------------------------")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("---------------------------------------------------------------")
print("---------------Logistic Regression Ridge(L1)-------------------")
print("---------------------------------------------------------------")

# Logistic Regression with Ridge Regularization
log_ridge = LogisticRegression(penalty='l2', solver='lbfgs')
log_ridge.fit(X_train, y_train)
ridge_log_pred = log_ridge.predict(X_test)
ridge_log_acc = accuracy_score(y_test, ridge_log_pred)
print("Logistic Regression with Ridge Regularization Accuracy:%.2f%%" % ((ridge_log_acc)*100))
print("---------------------------------------------------------------")
log_ridge = LogisticRegression(penalty='l2', C=1, solver='lbfgs', random_state=42)
log_ridge.fit(X_train, y_train)
y_pred = log_ridge.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("---------------------------------------------------------------")
print("---------------------------------------------------------------")
print("---------------Logistic Regression Lasso(L2)-------------------")
print("---------------------------------------------------------------")
# Logistic Regression with Lasso Regularization
log_lasso = LogisticRegression(penalty='l1', solver='saga') #'liblinear'
log_lasso.fit(X_train, y_train)
lasso_log_pred = log_lasso.predict(X_test)
lasso_log_acc = accuracy_score(y_test, lasso_log_pred)
print("Logistic Regression with Lasso Regularization Accuracy:%.2f%%" % ((lasso_log_acc)*100))
print("---------------------------------------------------------------")
log_lasso = LogisticRegression(penalty='l1', C=1, solver='saga', random_state=42)
log_lasso.fit(X_train, y_train)
y_pred = log_lasso.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("---------------------------------------------------------------")


print("---------------------------------------------------------------")
print("----------------Neural Network Classifier----------------------")
print("---------------------------------------------------------------")
# Neural Network Classifier
clasf = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', max_iter=1000, random_state=42)
clasf.fit(X_train, y_train)
nn_prd = clasf.predict(X_test)
nn_acc = accuracy_score(y_test, nn_prd)
print("Neural Network Accuracy: %.2f%%" % ((nn_acc)*100))
print("---------------------------------------------------------------")
mlp_clasf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, alpha=0.0001,
                    solver='adam', activation='relu', random_state=42)
mlp_clasf.fit(X_train, y_train)
y_pred = mlp_clasf.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("---------------------------------------------------------------")

#cumulative comparision of classifier
plt.figure(figsize=(9, 5))
plt.title('Classifiers Comparision based on given problem statement')
models = [' Linear(Ridge)', 'Linear(Lasso)', ' Logistic(Ridge)', 'Logistic(Lasso)', 'Neural Netwoork']
res =[ridge_acc, lasso_acc, ridge_log_acc, lasso_log_acc, nn_acc]
plt.bar(models, res)

