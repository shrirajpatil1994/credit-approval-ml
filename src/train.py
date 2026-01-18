import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data():
    """
    Create and return the credit approval dataset.
    """
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "income": [50000, 60000, 70000, 80000, 90000],
        "approved": [0, 0, 1, 1, 1]
    })

    X = df[["age", "income"]]
    y = df["approved"]

    return X, y


def train_model(X, y, config):
    """
    Split data and train a logistic regression model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"]
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


if __name__ == "__main__":
    config={
        "test_size": 0.4,
        "random_state": 42
    }
    X, y = load_data()
    model, X_test, y_test = train_model(X, y, config)
    accuracy = evaluate_model(model, X_test, y_test)

    print(f"Test accuracy: {accuracy}")
