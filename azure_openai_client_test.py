from azure_openai_client import AzureOpenAIClient

azure_client = AzureOpenAIClient(deployment_name="gpt-4o-mini")

system_text = "You are a helpful assistant."
user_text = "Tell me a short joke about programming."

response_text = azure_client.send_prompt(system_text, user_text)

print(response_text)