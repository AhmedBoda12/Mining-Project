import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import pydotplus
from sklearn import tree
from IPython.display import Image
from numpy.random import random
import matplotlib.pyplot as plt
import seaborn as sns

# Load our spam dataset 

spam_df = pd.read_csv('spam.csv', encoding="ISO-8859-1")

spam_df = spam_df[['v1', 'v2']]
spam_df.rename(columns={'v1': 'spam', 'v2': 'text'}, inplace=True)

# Assuming we have 'text' column for email content and 'spam' column for labels
X = spam_df['text']
y = spam_df['spam']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a bag-of-words representation
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Model fitting and evaluation with varying max depths
maxdepths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]

trainAcc = np.zeros(len(maxdepths))
testAcc = np.zeros(len(maxdepths))

index = 0
for depth in maxdepths:
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train_bow, y_train)
    y_pred_train = clf.predict(X_train_bow)
    y_pred_test = clf.predict(X_test_bow)
    trainAcc[index] = accuracy_score(y_train, y_pred_train)
    testAcc[index] = accuracy_score(y_test, y_pred_test)
    index += 1

# Plot of training and testing accuracies
plt.figure(figsize=(10, 6))
plt.plot(maxdepths, trainAcc, 'ro-', maxdepths, testAcc, 'bv--')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy vs. Max Depth')
plt.grid(True)

# Save plot as an image
plt.savefig("decision_tree_accuracy_plot.png")
plt.show()

# Model fitting and evaluation with varying regularization strengths
Cs = [0.01, 0.1, 1, 10, 100]

trainAcc = np.zeros(len(Cs))
testAcc = np.zeros(len(Cs))

index = 0
for C in Cs:
    clf = LogisticRegression(solver='liblinear', penalty='l1', C=C)
    clf.fit(X_train_bow, y_train)
    y_pred_train = clf.predict(X_train_bow)
    y_pred_test = clf.predict(X_test_bow)
    trainAcc[index] = accuracy_score(y_train, y_pred_train)
    testAcc[index] = accuracy_score(y_test, y_pred_test)
    index += 1

# Plot of training and testing accuracies
plt.figure(figsize=(10, 6))
plt.plot(Cs, trainAcc, 'ro-', Cs, testAcc, 'bv--')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Accuracy')
plt.xscale('log')  # Use logarithmic scale for better visualization
plt.title('Training and Test Accuracy vs. Regularization Strength (C)')
plt.grid(True)

# Save plot as an image
plt.savefig("logreg_accuracy_plot.png")
plt.show()

# Initialize and train the decision tree classifier
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train_bow, y_train)

# Make predictions
y_pred_tree = tree_model.predict(X_test_bow)

# Evaluate accuracy
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree Accuracy: {accuracy_tree:.2f}")

print("Decision Tree Classification Report")
print(classification_report(y_test, tree_model.predict(X_test_bow)))

print("Decision Tree Confusion Matrix")
print(confusion_matrix(y_test, tree_model.predict(X_test_bow)))

dt_conf = confusion_matrix(y_test, tree_model.predict(X_test_bow))

# Plot confusion matrix for Decision Tree
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(dt_conf, annot=True, fmt='g', cmap='Blues',
            xticklabels=['Non-Spam', 'Spam'],
            yticklabels=['Non-Spam', 'Spam'])
plt.ylabel('Actual', fontsize=13)
plt.xlabel('Predicted', fontsize=13)
plt.title('Decision Tree Confusion Matrix', fontsize=15)
plt.savefig('Decision Tree Confusion Matrix.png')

# Initialize and train the logistic regression model

logreg_model = LogisticRegression(solver='liblinear', penalty='l1')
logreg_model.fit(X_train_bow, y_train)

# Make predictions
y_pred_logreg = logreg_model.predict(X_test_bow)

# Evaluate accuracy
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {accuracy_logreg:.2f}")

print("Logistic Regression Classification Report")
print(classification_report(y_test, logreg_model.predict(X_test_bow)))

print("Logistic Regression Confusion Matrix")
print(confusion_matrix(y_test, logreg_model.predict(X_test_bow)))

lr_conf = confusion_matrix(y_test, logreg_model.predict(X_test_bow))

# Plot confusion matrix for Logistic Regression
plt.subplot(1, 2, 2)
sns.heatmap(lr_conf, annot=True, fmt='g', cmap='Greens',
            xticklabels=['Non-Spam', 'Spam'],
            yticklabels=['Non-Spam', 'Spam'])
plt.ylabel('Actual', fontsize=13)
plt.xlabel('Predicted', fontsize=13)
plt.title('Logistic Regression Confusion Matrix', fontsize=15)
plt.tight_layout()
plt.show()

# Generate decision tree visualization
dot_data = tree.export_graphviz(tree_model, feature_names=vectorizer.get_feature_names_out(),
                                class_names=['not spam', 'spam'], filled=True, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("decision_tree.png")
Image(graph.create_png())


# test example
new_email = str(input("Enter the mail: "))


new_email_bow = vectorizer.transform([new_email])

tree_prob = tree_model.predict(new_email_bow)
log_prob = logreg_model.predict(new_email_bow)

if tree_prob == "spam":
    print("The new email is classified by decision tree as spam.")
else:
    print("The new email is classified by decision tree as not spam.")


if log_prob == "spam":
    print("The new email is classified by logistics regression as spam.")
else:
    print("The new email is classified by logistics regression as not spam.")

