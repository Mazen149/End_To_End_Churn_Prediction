import pandas as pd
from .CustomerData import CustomerData

def predict_new(data: CustomerData, preprocessor, model):
    # to DF
    df = pd.DataFrame([data.model_dump()])

    # transform
    X_processed = preprocessor.transform(df)

    # predict
    y_pred = model.predict(X_processed)
    y_prob = model.predict_proba(X_processed)

    return {
        "churn_prediction": bool(y_pred[0]),
        "churn_probability": float(y_prob[0][1]),
    }