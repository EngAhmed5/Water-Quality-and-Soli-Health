from config import client

def start_agri_chat(initial_report=""):
    """
    Starts an interactive, terminal-based chatbot strictly focused on Agriculture.
    It takes the generated soil report as the initial context.
    """
    
    print("\n" + "="*60)
    print("🌾 Welcome to the AI Agronomist Chatbot! 🌾")
    print("Ask me anything about soil, crops, fertilizers, or water.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("="*60 + "\n")

    system_instruction = """You are an expert Egyptian Agricultural Consultant. 
    Your expertise is STRICTLY limited to soil quality, crop recommendations, water management, irrigation, and agricultural practices.
    
    RULES:
    1. If the user asks about ANY topic outside agriculture (e.g., coding, politics, history, general trivia), politely decline and state that you are an agricultural assistant.
    2. NEVER use Markdown symbols like asterisks (*), hashes (#), or quotes ("). Output clean, plain text only.
    3. Keep your answers concise, practical, and easy for a farmer or agricultural engineer to understand.
    4. Base your advice on the Egyptian agricultural context.
    """

    messages =[
        {"role": "system", "content": system_instruction}
    ]

    if initial_report:
        messages.append({
            "role": "assistant", 
            "content": f"I have analyzed the soil sample and generated this report: \n\n{initial_report}\n\nHow can I further assist you with this field?"
        })
        print("AI Agronomist: Hello! I have reviewed your soil report. Do you have any specific questions about the recommended crops, fertilizers, or irrigation?")
    else:
        print("AI Agronomist: Hello! How can I assist you today with your farm, soil, or crops?")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower().strip() in['exit', 'quit', 'bye']:
            print("\nAI Agronomist: Goodbye! Wishing you a bountiful harvest. 🌱\n")
            break
        
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.5, 
                max_tokens=1024
            )
            
            reply = response.choices[0].message.content.strip()
            clean_reply = reply.replace('*', '').replace('#', '').replace('"', '').replace('`', '')
            
            print(f"\nAI Agronomist: {clean_reply}")
            messages.append({"role": "assistant", "content": clean_reply})

        except Exception as e:
            print(f"\nSystem Error: {e}")
            messages.pop() 

# for testing the chatbot standalone: but it needs the report as context.

# if __name__ == "__main__":
#     start_agri_chat(report)