import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
				accuracy_score)
import matplotlib.pyplot as plt


# Load the data
df = pd.read_csv("train.csv")
X = df[["Retention Time", "218nm Area", "250nm Area", "260nm Area", "330nm Area", "350nm Area"]]
y = df["ID"]
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2) 

# Create and train a logit model
logit_model = LogisticRegression()
logit_model.fit(X_train, y_train)

# Make predictions for the model
y_pred = logit_model.predict(X_test)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Display confusion matrix
disp = ConfusionMatrixDisplay(cm, display_labels=logit_model.classes_)
disp.plot()
plt.show()

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('{:.2%}'.format(accuracy))
