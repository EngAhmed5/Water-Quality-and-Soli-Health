import joblib
import pandas as pd
import numpy as np
from configs import DATAPATH
from preprocessing import remove_unnecessary_columns, feature_engineering

def test_single_row():
    # 1. Load the saved components
    print("Loading models and transformers...")
    try:

        pipeline_data = joblib.load("Saved_models/SVC.pkl")

        # Extract your components
        model = pipeline_data['model']
        scaler = pipeline_data['scaler']
        encoder = pipeline_data['encoder']   
    except FileNotFoundError as e:
        print(f"Error: Could not find saved models. Ensure main.py has been run. {e}")
        return

    # 2. Load the original data and grab the first row
    print("Loading original data...")
    df_raw = pd.read_csv(DATAPATH)
    
    # Take the first row as a DataFrame (to keep column names)
    sample_row = df_raw.iloc[[0]].copy()
    print("\n--- Original Raw Data (First Row) ---")
    print(sample_row)

    # 3. Preprocessing (Must match the steps in main.py)
    # Step A: Remove unnecessary columns
    cols_to_remove = ['FID', 'Shape', 'FID_1', 'Area', 'Block', 'Profile', 'Long', 'Lat', 'Cluster', 'Cluster_Name']
    sample_processed = remove_unnecessary_columns(sample_row, cols_to_remove)
    
    # Step B: Feature Engineering (Adds SOC, C_N_Ratio, ESP)
    sample_processed = feature_engineering(sample_processed)

    # 4. Encoding & Scaling
    # Identify categorical and numerical columns (matching preprocessing.py logic)
    num_features = sample_processed.select_dtypes(include=['number']).columns.tolist()
    cat_features = sample_processed.select_dtypes(include=['object', 'category']).columns.tolist()

    # Apply loaded Label Encoder to categorical columns
    for cat in cat_features:
        sample_processed[cat] = encoder.transform(sample_processed[cat])

    # Apply loaded Scaler to numerical columns
    sample_processed[num_features] = scaler.transform(sample_processed[num_features])

    # 5. Prediction
    print("\n--- Processed Data (Input to Model) ---")
    print(sample_processed)
    
    prediction = model.predict(sample_processed)
    
    # Map the numeric prediction back to the name
    cluster_map = {
        2: "High Agricultural Performance Cluster",
        1: "Balanced Agricultural Productivity Cluster",
        0: "Low Agricultural Viability Cluster"
    }
    
    result_name = cluster_map.get(prediction[0], "Unknown Cluster")

    print("\n" + "="*30)
    print(f"PREDICTION RESULT")
    print("="*30)
    print(f"Cluster ID: {prediction[0]}")
    print(f"Cluster Name: {result_name}")
    print("="*30)

if __name__ == "__main__":
    test_single_row()