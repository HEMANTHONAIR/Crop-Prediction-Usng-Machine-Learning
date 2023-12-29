import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score, accuracy_score, roc_curve

# Load datasets
df1 = pd.read_csv("test_crop.csv")
df2 = pd.read_csv("crop .csv")

# Data preprocessing
df1.label = df1.label.map({"wheat": 0, "Sugarcane": 1})
df2.label = df2.label.map({"rice": 2, "wheat": 0, "Sugarcane": 1, "Mung Bean": 3, "Tea": 4, "millet": 5, "maize": 6,
                           "Lentil": 7, "Jute": 8, "Coffee": 9, "Cotton": 10, "Ground Nut": 11, "Peas": 12, "Rubber": 13,
                           "Tobacco": 14, "Kidney Beans": 15, "Moth Beans": 16, "Coconut": 17, "Black gram": 18,
                           "Adzuki Beans": 19, "Pigeon Peas": 20, "Chickpea": 21, "banana": 22, "grapes": 23, "apple": 24,
                           "mango": 25, "muskmelon": 26, "orange": 27, "papaya": 28})

# Visualization
sns.set(font_scale=5)
sns.countplot(x=df1['temperature'], hue=df1['label'])

# Data splitting
X_test = df1.drop('label', axis=1)
Y_test = df1['label']
X_train = df2.drop('label', axis=1)
Y_train = df2['label']

# Logistic Regression
lr = LogisticRegression(random_state=0)
lr.fit(X_train, Y_train)

# Cross-validation
cross_val_scores = cross_val_score(lr, X_test, Y_test)
mean_cross_val_score = cross_val_scores.mean() * 100

# Model predictions
Y_prediction = lr.predict(X_test)

# Model evaluation
confusion_mat = confusion_matrix(Y_test, Y_prediction)
f1_score_val = f1_score(Y_test, Y_prediction, average="macro")
recall_val = recall_score(Y_test, Y_prediction, average="macro")
precision_val = precision_score(Y_test, Y_prediction, average="weighted", zero_division=1)
accuracy_val = accuracy_score(Y_prediction, Y_test)

# ROC Curve
fpr, tpr, threshold = roc_curve(Y_test, Y_prediction)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# Save model
with open('sample_model.pK1', 'wb') as file:
    pickle.dump(lr, file)

# Specific Prediction
m = [[33.6, 76.8, 8.37, 90.2]]
Y_predict = lr.predict(m)
