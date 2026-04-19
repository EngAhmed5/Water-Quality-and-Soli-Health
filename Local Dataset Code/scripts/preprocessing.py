from configs import * 


def remove_unnecessary_columns(df, columns_to_remove): ##----------------- updated put to review ----------------
    df = df.drop(columns=columns_to_remove, errors='ignore')
    return df

def feature_engineering(df):
   # 1. Soil Organic Carbon (SOC)
    df['SOC'] = (df['OM'] / 1.724).round(2)

    #2. Carbon-to-Nitrogen Ratio (C:N)
    df['N_percentage'] = df['N'] / 10000
    df['C_N_Ratio'] = (df['SOC'] / df['N_percentage']).round(2)

    # 3. Exchangeable Sodium Percentage (ESP)
    df['ESP'] = ((100 * df['SAR']) / (df['SAR'] + 1)).round(2)

    # Drop Unnecessary Columns
    df = remove_unnecessary_columns(df, ['N_percentage'])  

    return df

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
