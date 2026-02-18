from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle 

iris = load_iris()

#Features (input data)
X = iris.data

#Target (output data)
y = iris.target

#Feature Names
feature_names = iris.feature_names

target_names = iris.target_names

print("Features: ", feature_names)
print("Target Names: ", target_names)
print("First five samples: ", X[:5])

#! Data preprocessing step 1 checkfor missing incomplete or duplicate data

data = pd.DataFrame(iris.data, columns=feature_names)
print("\nMissing value per feature: ", data.isnull().sum())

print("Number of duplicate rows: ", data.duplicated().sum())

data.drop_duplicates(inplace=True)



print("\nMissing value per feature: ", data.isnull().sum())

print("Number of duplicate rows: ", data.duplicated().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

scalar=StandardScaler()

X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled=scalar.transform(X_test)

print("First 5 rows of scaled training data\n", X_train_scaled[:5])


#Step 3 Train and evaluate

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

with open("model_and_scalar.pkl", "wb") as f:
    pickle.dump({"model": model,"scalar":scalar}, f)

print("\n Model and Scalar have been saved to model and scalar")
y_pred = model.predict(X_test_scaled)

accuracy= accuracy_score(y_test, y_pred)

print(f"Accuracy: ", accuracy)

print("Classification Report:\n", classification_report(y_test,y_pred, target_names=target_names))

cm= confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names,yticklabels=target_names)
plt.xlabel("predicted")
plt.ylabel("True")
plt.title("Confusion matrix - iris project")
plt.show()
