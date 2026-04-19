from configs import * 


def load_data(path):
    df = pd.read_csv(path)
    return df

def explore_data(df):
    print("\n" + "="*50 + " BASIC INFO " + "="*50)
    print(df.head())
    
    print("\n" + "="*50 + " SHAPE " + "="*50)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n" + "="*50 + " DATA TYPES " + "="*50)
    print(df.dtypes)
    
    print("\n" + "="*50 + " MISSING VALUES " + "="*50)
    missing = pd.DataFrame({
        "Null Count": df.isnull().sum(),
        "Null %": df.isnull().mean() * 100
    })
    print(missing[missing["Null Count"] > 0])
    
    print("\n" + "="*50 + " STATISTICS (NUMERICAL) " + "="*50)
    print(df.describe())
    
    print("\n" + "="*50 + " CATEGORICAL SUMMARY " + "="*50)
    cat_cols = df.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        print(f"\nColumn: {col}")
        print(df[col].value_counts().head(10))
    
    print("\n" + "="*50 + " CARDINALITY " + "="*50)
    cardinality = df.nunique().sort_values(ascending=False)
    print(cardinality)



def detect_outliers_iqr(df, threshold = 1.5):

    outlier_indices = {}
    columns =df.select_dtypes('number').columns.to_list()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75) 
        IQR = Q3 - Q1  
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        outlier_indices[col] = outliers
    
    return outlier_indices



if __name__ == "__main__":
    df = load_data(DATAPATH)
    
    explore_data(df)
    
    outliers_iqr = detect_outliers_iqr(df)
    print("\n" + "="*50 + " Outlier Counts " + "="*50)
    for col, indices in outliers_iqr.items():
        print(f"{col}: {len(indices)} outliers detected")
    