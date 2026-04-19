from configs import * 

def split_data(df):
    x = df.drop(columns=['Cluster','Cluster_Name'])
    y = df['Cluster']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=RANDOM_STATE , shuffle=True)
    return x, y , x_train, x_test, y_train, y_test


def processing_data(x_train , x_val):
    scaler = StandardScaler()
    encoder = LabelEncoder()
    
    num_features = x_train.select_dtypes(include=['number']).columns.tolist()
    cat_features = x_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1. Process Categorical Features
    if cat_features:
        
        for cat in cat_features:
            x_train[cat] = encoder.fit_transform(x_train[cat])
            x_val[cat] = encoder.transform(x_val[cat])
        
    
    # 2. Process Numerical Features
    if num_features:
        x_train[num_features] = scaler.fit_transform(x_train[num_features])
        x_val[num_features] = scaler.transform(x_val[num_features])
    
    return x_train , x_val , encoder , scaler


def apply_sharp_impulse(data, level=0.5, type_name='gaussian', random_state=42): # --------- updated put to review ----------------
    np.random.seed(random_state)
    X_array = data.to_numpy()
    
    if type_name == 'gaussian':
        gaussian = np.random.normal(loc=0.0, scale=level, size=X_array.shape)
        X_array = X_array + gaussian
        
    elif type_name == 'impulse':
        X_array = X_array.copy()
        mask = np.random.rand(*X_array.shape) < level
        
        spikes = np.random.choice([-3.0, 3.0], size=X_array.shape)
        X_array[mask] = spikes[mask]
    else:
        raise ValueError(" must be either 'gaussian' or 'impulse'")
        
    X_df = pd.DataFrame(X_array, columns=data.columns, index=data.index)
    return X_df