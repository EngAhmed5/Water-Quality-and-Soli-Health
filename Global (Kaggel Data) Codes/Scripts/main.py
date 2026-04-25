from configs import *
from load_explore_data import *
from preprocessing import *
from clustring import *
from modeling import *


def main():
    print("="*60)
    print("           STEP 1: LOADING & EXPLORING DATA")
    print("="*60)
    original_data = load_data(DATAPATH)
    explore_data(original_data)
    
    print("           STEP 3: ELBOW Method")
    cluster_features = original_data.select_dtypes(include=['number']).columns.tolist()
    print(f"Features used for clustering: {cluster_features}")
    
    elbow_method(original_data, cluster_features)
    
    print("\n" + "="*60)
    print("           STEP 4: CLUSTERING (Target Generation)")
    print("="*60)
    
    data, kmeans_model, Kmeans_score, centroids = clustering(original_data , cluster_features , k= 3 )
    
    print(f"\nClustering Completed! Training Silhouette Score: {Kmeans_score:.4f}")
    print("\n--- CENTROIDS ---")
    print(centroids)
    
    data = map_clusters(data)
    
    print("\n" + "="*60)
    print("           STEP 5: PREPARE X & y FOR SUPERVISED MODELS")
    print("="*60)
    
    x, y , x_train, x_val, y_train, y_val = split_data(data)
    
    print(f"x_train shape: {x_train.shape} | x_val shape: {x_val.shape}")
    print(f"Target classes: {sorted(y_train.unique())}")
    
    print("\n" + "="*60)
    print("           STEP 6:Preprocessing ")
    print("="*60)
    
    x_train , x_val , encoder , scaler = processing_data(x_train,x_val)
    
    print("\n" + "="*60)
    print("           STEP 7: APPLYING Impulse ")
    print("="*60)
    
    
    print("\n" + "="*60)
    print("           STEP 8: MODELING & EVALUATION ")
    print("="*60)
    
    # Model 1: Logistic Regression
    print("\n>>> Training Logistic Regression ...")
    lr = LogisticRegression(C= 0.01  ,random_state=RANDOM_STATE) # C = 0.01 for better performance and prevent the overfitting 
    lr_model , lr_model_name =train(lr, "Logistic Regression", x_train, y_train, x_val, y_val)
    class_report(lr_model, x_val, y_val)
    plot_confusion_matrix(lr_model, x_val, y_val, lr_model_name)
    
    # Model 2: SVM
    print("\n>>> Training SVC ...")
    svc = SVC(random_state=RANDOM_STATE , C= 0.1) # C = 0.01 for better performance and prevent the overfitting
    svc_model , svc_model_name =train(svc, "SVC", x_train, y_train, x_val, y_val)
    class_report(svc_model, x_val, y_val)
    plot_confusion_matrix(svc_model, x_val, y_val, svc_model_name)
    
    # Model 3: Random Forest
    print("\n>>> Training Random Forest Classifier ...")
    rf = RandomForestClassifier(random_state=RANDOM_STATE , n_estimators=2 )    # n_estimators=2 for prevent the overfitting
    rf_model , rf_model_name = train(rf, "Random Forest", x_train, y_train, x_val, y_val)
    class_report(rf_model, x_val, y_val)
    plot_confusion_matrix(rf, x_val, y_val, rf_model_name)
    
        # Model 3: KNN
    print("\n>>> Training KNN Classifier ...")
    knn = KNeighborsClassifier()
    knn_model , knn_model_name = train(knn, "KNN", x_train, y_train, x_val, y_val)
    class_report(knn_model, x_val, y_val)
    plot_confusion_matrix(knn_model, x_val, y_val, knn_model_name)
    
    print("\n" + "="*60)
    print("                 PIPELINE FINISHED")
    print("="*60)


if __name__ == "__main__":
    main()