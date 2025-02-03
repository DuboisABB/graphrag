# azure_openai_client.py

import os
from openai import AzureOpenAI

import yaml

def load_config():
    with open('ragtest/settings.yaml', 'r') as file:
        return yaml.safe_load(file)


class AzureOpenAIClient:
    _instances = {}

    def __new__(cls, deployment_name="o1-preview", *args, **kwargs):
        if deployment_name not in cls._instances:
            instance = super(AzureOpenAIClient, cls).__new__(cls)
            cls._instances[deployment_name] = instance
            instance._initialized = False
            instance._deployment_name = deployment_name
        return cls._instances[deployment_name]

    def __init__(self, deployment_name="o1-preview"):
        if self._initialized:
            return

        #self.api_key = os.getenv("AZURE_OPENAI_KEY")
        #self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        config = load_config()
        llm_config = config['llm']
        self.api_key=os.getenv("GRAPHRAG_API_KEY")
        print(f"api_key: {self.api_key}")
        self.api_version=llm_config['api_version']
        print(f"api_version: {self.api_version}")
        self.azure_endpoint=llm_config['api_base']      
        print(f"azure_endpoint: {self.azure_endpoint}")

        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
        )
        self.deployment_name = self._deployment_name  # Use the deployment_name from __new__
        self._initialized = True

    def get_client(self):
        return self.client

    def get_deployment_name(self):
        return self.deployment_name

    def send_prompt(self, system_text, user_text):
        if self.deployment_name == "o1-preview":
            combined_text = f"System Role: {system_text}\n\nUser Role: {user_text}"
            messages = [{"role": "user", "content": combined_text}]
        else:
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]

        args = {"model": self.deployment_name, "messages": messages}

        if self.deployment_name != "o1-preview":
            args["temperature"] = 0.0

        print(f"Sending prompt to Azure OpenAI using {self.deployment_name}...")
        response = self.client.chat.completions.create(**args)
        return response.choices[0].message.content
