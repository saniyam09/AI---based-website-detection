"""Train and compare phishing-URL detection models.

This script loads a CSV dataset, extracts phishing-related features, splits the
sample into training and testing sets, trains several classifiers, chooses the
best model, and saves it with joblib for use by the Flask app.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from feature_extraction import FEATURE_COLUMNS, extract_features

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"
DATASET_PATH = DATA_DIR / "phishing_dataset.csv"
MODEL_PATH = MODEL_DIR / "phishing_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def generate_dataset(path: Path) -> pd.DataFrame:
    """Create a synthetic but realistic phishing dataset if none exists."""
    legitimate_urls = [
        "https://www.google.com",
        "https://www.microsoft.com",
        "https://www.amazon.com",
        "https://www.github.com",
        "https://www.paypal.com",
        "https://www.dropbox.com",
        "https://www.linkedin.com",
        "https://www.netflix.com",
        "https://www.coursera.org",
        "https://www.apple.com",
        "https://www.facebook.com",
        "https://www.salesforce.com",
        "https://www.bankofamerica.com",
        "https://www.chase.com",
        "https://www.wellsfargo.com",
        "https://www.ebay.com",
        "https://www.wordpress.com",
        "https://www.instagram.com",
        "https://www.reddit.com",
        "https://www.github.io",
        "https://www.icloud.com",
        "https://www.nasa.gov",
        "https://www.mozilla.org",
        "https://www.wordpress.org",
    ]

    phishing_urls = [
        "http://secure-login-update-account.net",
        "http://verify-bank-account-login.com",
        "http://confirm-password-update-secure.net",
        "http://www-login-secure-bank-confirmation.xyz",
        "https://login.account.verify-update.com",
        "http://banking-update-verify-login.info",
        "http://secure-confirmation-account-password.net",
        "http://paypal-security-verify-update.com",
        "http://account-login-confirm-security.net",
        "http://update-password-confirm-your-account.xyz",
        "http://secure-login-portal-account.com",
        "http://verify-your-bank-account-credential.xyz",
        "http://account-verify-login-update.com",
        "http://confirm-secure-password-bank.net",
        "https://login.verify.account.secure-update.org",
        "http://www.paypal-login-verify-update.net",
        "http://accountconfirm-loginsecure.icu",
        "http://secure-bank-login-update-check.online",
        "http://confirm-account.verify-login.net",
        "http://admin-login-password-update.work",
        "https://www.paypal.com@evil.com",
        "http://192.168.1.10/login.php",
        "http://123.45.67.89/login_verify.php",
        "http://banking-login-update@secure.net",
        "http://confirm-account-update-password.info",
        "http://verify-login-account-billing.com",
        "http://hidden-secure-login-portal.net",
        "http://account-update-password-lock.xyz",
        "http://secure-confirmation-login-verify.com",
        "https://www.mysite-login-update.tk",
        "http://verify-user-account-deactivation.net",
        "http://account-password-secure-login.org",
        "http://www.bank-update-confirmation-login.com",
        "http://reset-password-verify-account-now.xyz",
        "http://login-secure-verify-bank.info",
        "http://bank-login-update-password.net",
        "http://secure-login-account-confirmation.com",
        "http://warning-account-update-confirmation.xyz",
        "http://login.verify.smartbank-update.net",
        "http://update-login-security-account.biz",
        "http://secure-login-verify-portal.org",
    ]

    rows = []
    for url in legitimate_urls:
        rows.append({"url": url, "label": "legitimate"})
    for url in phishing_urls:
        rows.append({"url": url, "label": "phishing"})

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the dataset and ensure it exists."""
    if not path.exists():
        return generate_dataset(path)
    return pd.read_csv(path)


def prepare_features(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create feature matrix and target labels for training."""
    dataset = dataset.copy()
    dataset["label"] = dataset["label"].astype(str).str.strip().str.lower()
    dataset = dataset.dropna(subset=["url", "label"])

    feature_rows = []
    for url in dataset["url"]:
        feature_rows.append(extract_features(url))

    X = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    y = dataset["label"].map({"legitimate": 0, "phishing": 1})
    return X, y


def train_models() -> dict:
    """Train and compare multiple ML classifiers."""
    dataset = load_dataset(DATASET_PATH)
    X, y = prepare_features(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        }
        trained_models[name] = model

    best_model_name = max(results, key=lambda model_name: results[model_name]["accuracy"])
    best_model = trained_models[best_model_name]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    metadata = {
        "best_model": best_model_name,
        "best_accuracy": results[best_model_name]["accuracy"],
        "metrics": results,
        "feature_columns": FEATURE_COLUMNS,
    }

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return {"best_model": best_model_name, "results": results}


def main() -> None:
    """Entry point for training the phishing detection model."""
    summary = train_models()
    print("Model training finished.")
    print("Best model:", summary["best_model"])
    for model_name, metrics in summary["results"].items():
        print(f"{model_name}: accuracy={metrics['accuracy']}, precision={metrics['precision']}, recall={metrics['recall']}, f1={metrics['f1_score']}")


if __name__ == "__main__":
    main()
