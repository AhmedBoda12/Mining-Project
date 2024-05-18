import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pydotplus
from sklearn import tree
from IPython.display import Image

# Load our spam dataset 
spam_df = pd.read_csv('spam.csv', encoding="ISO-8859-1", nrows=5500)
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

# Initialize and train the decision tree classifier
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train_bow, y_train)

# Make predictions
y_pred_tree = tree_model.predict(X_test_bow)

# Evaluate accuracy
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree Accuracy: {accuracy_tree:.2f}")

# Initialize and train the logistic regression model
logreg_model = LogisticRegression(solver='liblinear', penalty='l1')
logreg_model.fit(X_train_bow, y_train)

# Make predictions
y_pred_logreg = logreg_model.predict(X_test_bow)

# Evaluate accuracy
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {accuracy_logreg:.2f}")

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
