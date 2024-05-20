import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import time

# Step 1: Read the CSV file into a pandas DataFrame
file_path = './breast-cancer.csv'
data = pd.read_csv(file_path)

# Step 2.1: Data cleaning - remove rows with any empty cells
cleaned_data = data.dropna()

# Step 2.2: Split the dataset into training set (80%) and testing set (20%)
train_data, test_data = train_test_split(cleaned_data, test_size=0.2, random_state=42)

# Separate features and target variable from training and testing data
X_train = train_data.drop(columns=['id', 'diagnosis'])
y_train = train_data['diagnosis']
X_test = test_data.drop(columns=['id', 'diagnosis'])
y_test = test_data['diagnosis']

# Step 3.1: Train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Track the training time
start_time = time.time()
clf.fit(X_train, y_train)
end_time = time.time()

training_time = end_time - start_time
print(f"Decision Tree Classifier: Training time: {training_time:.4f} seconds")

# Step 3.2: Draw the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=['M', 'B'], rounded=True)
plt.show()

# Step 3.3: Evaluate the trained model using the testing data
y_pred = clf.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred, pos_label='M')  # Sensitivity for 'M' (Malignant)
specificity = recall_score(y_test, y_pred, pos_label='B')  # Specificity for 'B' (Benign)

print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Recall for M): {sensitivity:.4f}")
print(f"Specificity (Recall for B): {specificity:.4f}")

# Step 3.4: Visualize the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=['M', 'B'])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['M', 'B'])
disp.plot(cmap=plt.cm.Blues)
plt.show()


# Step 4: Train the SVM Classifier with RBF kernel
svm_clf = SVC(kernel='rbf', random_state=42)

# Track the training time
start_time = time.time()
svm_clf.fit(X_train, y_train)
end_time = time.time()

training_time = end_time - start_time
print(f"SVM Training time: {training_time:.4f} seconds")

# Step 4.1: Evaluate the SVM model using the testing data
y_pred_svm = svm_clf.predict(X_test)

# Calculate performance metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
sensitivity_svm = recall_score(y_test, y_pred_svm, pos_label='M')  # Sensitivity for 'M' (Malignant)
specificity_svm = recall_score(y_test, y_pred_svm, pos_label='B')  # Specificity for 'B' (Benign)

print(f"SVM Accuracy: {accuracy_svm:.4f}")
print(f"SVM Sensitivity (Recall for M): {sensitivity_svm:.4f}")
print(f"SVM Specificity (Recall for B): {specificity_svm:.4f}")

# Step 4.2: Visualize the confusion matrix for SVM
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm, labels=['M', 'B'])
disp_svm = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_svm, display_labels=['M', 'B'])
disp_svm.plot(cmap=plt.cm.Blues)
plt.show()

# Step 6: Find feature importances using Random Forest
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
importances = rf_clf.feature_importances_

# Get feature importances in descending order
sorted_indices = importances.argsort()[::-1]

# Visualize the top two columns (features) in x-y coordinate system
top_two_features = X_train.columns[sorted_indices[:2]]
plt.figure(figsize=(10, 6))
plt.scatter(X_train[top_two_features[0]], X_train[top_two_features[1]], c=y_train.map({'M': 1, 'B': 0}), cmap='viridis')
plt.xlabel(top_two_features[0])
plt.ylabel(top_two_features[1])
plt.title("Top Two Features")
plt.colorbar(label='Diagnosis (M=1, B=0)')
plt.show()

def retrain_decision_tree(X_train, X_test, y_train, y_test, features_to_keep):
    # Retrain Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    
    # Track the training time
    start_time = time.time()
    clf.fit(X_train[features_to_keep], y_train)
    end_time = time.time()

    training_time = end_time - start_time

    # Evaluate the model
    y_pred = clf.predict(X_test[features_to_keep])

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, pos_label='M')  # Sensitivity for 'M' (Malignant)
    specificity = recall_score(y_test, y_pred, pos_label='B')  # Specificity for 'B' (Benign)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall for M): {sensitivity:.4f}")
    print(f"Specificity (Recall for B): {specificity:.4f}")
    print(f"Training time: {training_time:.4f} seconds")

    # Draw the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(clf, filled=True, feature_names=features_to_keep, class_names=['M', 'B'], rounded=True)
    plt.show()

    # Visualize the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['M', 'B'])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['M', 'B'])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    return clf

# Remove the feature with the lowest importance and retrain the model
features_to_keep = X_train.columns[sorted_indices[:-1]]
print("Retraining Decision Tree after removing the least important feature...")
retrain_decision_tree(X_train, X_test, y_train, y_test, features_to_keep)

# Remove the four features with the lowest importances and retrain the model
features_to_keep = X_train.columns[sorted_indices[:-4]]
print("Retraining Decision Tree after removing the four least important features...")
retrain_decision_tree(X_train, X_test, y_train, y_test, features_to_keep)

# Remove the ten features with the lowest importances and retrain the model
features_to_keep = X_train.columns[sorted_indices[:-10]]
print("Retraining Decision Tree after removing the ten least important features...")
retrain_decision_tree(X_train, X_test, y_train, y_test, features_to_keep)

'''
Question 7

Both the Initial Decision Tree and the Decision 
Tree after removing the four least important features, as
well as the SVM model, achieved the highest accuracy 
with 0.9474

7.1 - Removing least important features didn't speed up training 
times. First Decision Tree took 0.0050 seconds to be trained, and Second,
Third, and Fourth 0,0090, 0.0090, 0.0060 respectively 

7.2 - No, performance was better when not removing any features

7.3 - It might be a good idea to remove less important features (especially
with a large dataset), but it's important to evaluate how important is the 
feature you are going to remove

'''