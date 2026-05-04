def check_overfitting(model, X, y, m=10, s=0.2):
    train_acc_list = []
    test_acc_list = []

    for i in range(m):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=s, random_state=i)

        # Train model
        model.fit(X_train, y_train)

        # Training accuracy
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)

        # Testing accuracy
        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

    # Return average accuracies
    return np.mean(train_acc_list), np.mean(test_acc_list)
print("\nOverfitting Check :\n")

for name, model in models.items():
    train_acc, test_acc = check_overfitting(model, X_final, y, m=10, s=0.2)
    print(f"{name}  Train = {train_acc:.4f} | Test = {test_acc:.4f}")
