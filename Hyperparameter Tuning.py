from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_dt = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
grid_dt.fit(X_final, y)

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

grid_svc = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_svc.fit(X_final, y)

param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],     # Regularization strength
    'penalty': ['l1', 'l2'],         # Type of regularization
    'solver': ['liblinear'],          # liblinear supports both l1 and l2
    'max_iter': [100, 200, 500]       # Iterations for model convergence
}

# Grid Search
grid_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, scoring='accuracy')
grid_lr.fit(X_final, y)

print("\n Model Comparison Summary")
print(f"Decision Tree Best Accuracy: {grid_dt.best_score_:.4f} with params: {grid_dt.best_params_}")
print(f"SVC Best Accuracy: {grid_svc.best_score_:.4f} with params: {grid_svc.best_params_}")
print(f"Logistic Regression Best Accuracy: {grid_lr.best_score_:.4f} with params: {grid_lr.best_params_}")