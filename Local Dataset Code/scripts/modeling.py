from configs import * 
from load_explore_data import *
from preprocessing import * 
from clustring import * 


def train(model, model_name, x_train, y_train, x_val, y_val):
    model.fit(x_train, y_train)
    model_train_score = model.score(x_train, y_train) * 100
    model_val_score = model.score(x_val, y_val) * 100
    print(f"{model_name} score on Training data: {model_train_score:.2f}%")
    print(f"{model_name} score on Validation data: {model_val_score:.2f}%")
    return model , model_name 

def class_report(model, x_val, y_val):
    y_pred = model.predict(x_val)
    print(classification_report(y_val, y_pred , target_names=['Low Agricultural Viability Cluster','Balanced Agricultural Productivity Cluster','High Agricultural Performance Cluster']))

def plot_confusion_matrix(model, x_val, y_val, model_name):
    y_pred = model.predict(x_val)
    disp = ConfusionMatrixDisplay.from_predictions(y_val, y_pred, display_labels=['Low Agricultural Viability Cluster','Balanced Agricultural Productivity Cluster','High Agricultural Performance Cluster'], cmap=plt.cm.Blues, normalize=None, xticks_rotation=45)
    disp.ax_.set_title(f'Confusion Matrix for {model_name}')
    plt.show()

def save_model_pipeline(model, model_name, scaler, encoder, save_dir=SAVED_DIR):
    """
    Saves the trained model, scaler, and encoder into a single file.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Replace spaces with underscores for a clean filename
    safe_model_name = model_name.replace(" ", "_")
    file_path = os.path.join(save_dir, f"{safe_model_name}.pkl")
    
    # Bundle the components into a single dictionary
    pipeline_data = {
        'model': model,
        'scaler': scaler,
        'encoder': encoder
    }
    
    # Save to disk
    joblib.dump(pipeline_data, file_path)
    print(f"Saved {model_name} pipeline to: {file_path}")
