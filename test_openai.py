import os
import yaml
from openai import AzureOpenAI

#JP Test Issue - needed to add this to access environement variable
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def load_config():
    with open('ragtest/settings.yaml', 'r') as file:
        return yaml.safe_load(file)

def test_chat():
    config = load_config()
    llm_config = config['llm']

    api_key=os.getenv("GRAPHRAG_API_KEY")
    print(f"api_key: {api_key}")
    api_version=llm_config['api_version']
    print(f"api_version: {api_version}")
    azure_endpoint=llm_config['api_base']      
    print(f"azure_endpoint: {azure_endpoint}")    
    model = llm_config['deployment_name']
    print(f"model: {model}")
    
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a short joke about programming."}
        ]
    )
    
    print(response.choices[0].message.content)

if __name__ == "__main__":
    test_chat()
