from config import client

def generate_soil_report(cluster_name, features_dict):
    """
    Generates a strict, plain-text expert agronomist report using Llama-3.3-70b via Groq.
    Formatted specifically for chat interfaces (no markdown, no symbols).
    """
    formatted_features = "\n".join([f"- {key}: {value}" for key, value in features_dict.items()])
    
    prompt = f"""
You are a highly experienced Agronomist, Soil Scientist, and Agricultural Consultant specializing in the agricultural landscape of Egypt.

Your task is to analyze a soil sample's data and generate a detailed, professional agricultural report to guide local farmers and agricultural engineers.

### INPUT DATA ###
- Predicted Soil Quality Cluster: {cluster_name}
- Soil Features (Lab Results):
{formatted_features}

### REPORT REQUIREMENTS ###
Based on the input data above, please generate a comprehensive report following this exact structure:

1. **Executive Summary & Cluster Interpretation**: 
   Briefly explain what the assigned soil cluster means and provide a high-level summary of the soil's current state.

2. **Detailed Soil Profile Analysis**: 
   Analyze the specific features provided (e.g., pH, Organic Matter (OM/SOC), Macronutrients (N, P, K), Salinity (EC), and Sodium hazards (SAR/ESP)). Clearly highlight the soil's strengths and any critical limiting factors.

3. **Crop Recommendations for Egypt**: 
   Suggest specific, economically viable crops suited for this exact soil profile and Egypt's climate. Break this down into Winter Crops, Summer Crops, and Orchards/Trees.

4. **Optimal Irrigation Strategy**: 
   Recommend the best irrigation methods (e.g., drip, sprinkler, or controlled surface). Consider the soil's salinity risks, drainage needs, and Egypt's critical need for water conservation.

5. **Soil Management & Amendment Guidelines**: 
   Provide actionable recommendations to improve or maintain this soil (e.g., specific fertilizers, gypsum application, organic compost, or salt leaching).

### TONE & STYLE ###
- Professional, scientific, objective, and highly actionable.
- Format with clear Markdown headings, bullet points, and bold text for readability.
- Tailor every point to the specific lab results provided. Do not use conversational filler (like "Sure, I can help with that"). Start directly with the report.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert Egyptian Agricultural Consultant. You output ONLY plain, unformatted text reports. You NEVER use Markdown symbols (no #, no *, no \")."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=2048
        )
        
        report_text = response.choices[0].message.content.strip()
        clean_text = report_text.replace('*', '').replace('#', '').replace('"', '').replace('`', '')
        
        return clean_text
        
    except Exception as e:
        return f"An error occurred while generating the report: {e}"

# for testing the report generation standalone:
# if __name__ == "__main__":
    
#     # ----- this parameters will passed after the model predictions ----- #
    
#     mock_cluster = "Balanced Agricultural Productivity Cluster"
#     mock_features = {
#         'pH': 7.8,
#         'EC': 2.5,
#         'SAR': 4.1,
#         'CaCO3': 8.5,
#         'Gypsum': 1.2,
#         'OM': 1.8,
#         'N': 1500,
#         'P': 25,
#         'K': 350,
#         'SOC': 1.04,
#         'C_N_Ratio': 6.93,
#         'ESP': 8.04
#     }
    
#     print("Generating plain-text report...\n" + "="*50)
#     report = generate_soil_report(mock_cluster, mock_features)
#     print(report)