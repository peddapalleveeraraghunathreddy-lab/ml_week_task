
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
INPUT_PATH = BASE_DIR / "data" / "new_customers.csv"
OUTPUT_PATH = BASE_DIR / "predictions.csv"

THRESHOLD = 0.50

# ==========================================
# 2. SERVICE COLUMNS
# ==========================================

service_columns = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

def create_features(data):

    data = data.copy()

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    data["tenure_group"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["New", "Medium", "Long", "Very Long"]
    )

    data["total_services"] = (
        data[service_columns] == "Yes"
    ).sum(axis=1)

    data["avg_spend_per_month"] = np.where(
        data["tenure"] > 0,
        data["TotalCharges"] / data["tenure"],
        data["MonthlyCharges"]
    )

    data["has_family"] = (
        (data["Partner"] == "Yes") |
        (data["Dependents"] == "Yes")
    ).astype(int)

    data["is_long_contract"] = (
        data["Contract"].isin(["One year", "Two year"])
    ).astype(int)

    return data

# ==========================================
# 4. MAIN PREDICTION FUNCTION
# ==========================================

def main():

    # Check model file
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Check that churn_model.joblib is inside the models folder."
        )

    # Check input CSV
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_PATH}\n"
            "Check that new_customers.csv is inside the data folder."
        )

    # Load trained model
    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully!")

    # Load new customer data
    new_customers = pd.read_csv(INPUT_PATH)

    print("Customer data loaded successfully!")
    print("Number of customers:", len(new_customers))

    # Required original input columns
    required_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges"
    ]

    # Validate input columns
    missing_columns = [
        col for col in required_columns
        if col not in new_customers.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Feature engineering
    new_customers = create_features(new_customers)

    # Generate churn probabilities
    probabilities = model.predict_proba(
        new_customers
    )[:, 1]

    # Predict churn using threshold
    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    # Add prediction results
    new_customers["Churn_Probability"] = probabilities

    new_customers["Predicted_Churn"] = predictions

    new_customers["Churn_Status"] = (
        new_customers["Predicted_Churn"].map({
            0: "No Churn",
            1: "Churn"
        })
    )

    # Save predictions
    new_customers.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nPredictions generated successfully!")

    print("\nPrediction Results:")

    print(
        new_customers[
            [
                "Churn_Probability",
                "Predicted_Churn",
                "Churn_Status"
            ]
        ]
    )

    print("\nPredictions saved at:")
    print(OUTPUT_PATH)


# ==========================================
# 5. RUN PROGRAM
# ==========================================

if __name__ == "__main__":
    main()
