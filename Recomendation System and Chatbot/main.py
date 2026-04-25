from generate_report import generate_soil_report
from chatbot import start_agri_chat

if __name__ == "__main__":
    
    # ----- this parameters will passed after the model predictions ----- #
    # In a real application, these would come from the soil analysis model's output, but here we are hardcoding them for testing purposes.
    # Sample Inputs
    
    mock_cluster = "Balanced Agricultural Productivity Cluster"
    mock_features = {
        'pH': 7.8,
        'EC': 2.5,
        'SAR': 4.1,
        'CaCO3': 8.5,
        'Gypsum': 1.2,
        'OM': 1.8,
        'N': 1500,
        'P': 25,
        'K': 350,
        'SOC': 1.04,
        'C_N_Ratio': 6.93,
        'ESP': 8.04
    }
    
    print("Generating plain-text report...\n" + "="*50)
    report = generate_soil_report(mock_cluster, mock_features)
    print(report)
    start_agri_chat(initial_report=report) # for end chatbot write 'exit' or 'quit' or 'bye'