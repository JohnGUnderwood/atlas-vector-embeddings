import pymongo
import requests
from os import getenv,environ
import atexit

class MongoDBConnection:
    def __init__(self):
        self.url = getenv("MDBCONNSTR")
        self.db_name = getenv("MDB_DB",default="test")
        try:
            self.client = pymongo.MongoClient(self.url)
            self.client.admin.command('ping')
            try:
                self.db = self.client.get_database(self.db_name)
            except Exception as e:
                raise Exception("Failed to connect to {}. {}".format(self.db_name,e))
        except Exception as e:
            raise Exception("Failed to connect to MongoDB. {}".format(self.db_name,e))
        atexit.register(self.close)

    def get_database(self):
        return self.db
    
    def get_session(self):
        return self.client.start_session()

    def close(self):
        self.client.close()

class Embeddings():
    def __init__(self):
        self.provider = getenv("PROVIDER")
        self.api_key = getenv("EMBEDDING_API_KEY",None)
        # Embedding services. Default to using Azure OpenAI.
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.model = getenv("EMBEDDING_MODEL","text-embedding-ada-002")
            self.dimensions = getenv("EMBEDDING_DIMENSIONS",1536)
        elif self.provider == "vectorservice":
            import requests
        elif self.provider == "mistral":
            from mistralai.client import MistralClient
            self.client = MistralClient(api_key=self.api_key)
            self.model = getenv("EMBEDDING_MODEL","mistral-embed")
            self.dimensions = getenv("EMBEDDING_DIMENSIONS",1024)
        elif self.provider == "azure_openai":
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version="2023-12-01-preview",
                azure_endpoint=getenv("OPENAIENDPOINT")
            )
            self.model = getenv("OPENAIDEPLOYMENT")
            self.dimensions = getenv("EMBEDDING_DIMENSIONS",1536)
        elif self.provider == "fireworks":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.fireworks.ai/inference/v1"
            )
            self.model = getenv("EMBEDDING_MODEL","nomic-ai/nomic-embed-text-v1.5")
            self.dimensions = getenv("EMBEDDING_DIMENSIONS",768)
        elif self.provider == "nomic":
            from nomic import embed as nomic_embed
            environ["NOMIC_API_KEY"] = self.api_key
            self.model = getenv("EMBEDDING_MODEL","nomic-embed-text-v1.5")
            self.dimensions = getenv("EMBEDDING_DIMENSIONS",768)
        else:
            print("No valid provider specified. Defaulting to Azure OpenAI and OPENAIAPIKEY variable.")
            self.provider = "azure_openai"
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=getenv("OPENAIAPIKEY"),
                api_version="2023-12-01-preview",
                azure_endpoint=getenv("OPENAIENDPOINT")
            )
            self.model = getenv("OPENAIDEPLOYMENT")
            self.dimensions = getenv("EMBEDDING_DIMENSIONS",1536)
            
        self.dimensions = int(self.dimensions)
        print("Using provider: ",self.provider)
        print("Using model: ",self.model)
        print("Using dimensions: ",self.dimensions) 
    

    def get_dimensions(self):
        return self.dimensions

    def get_embedding_VectorService(embed_text):
        response = requests.get(getenv("VECTOR_SERVICE_URL"), params={"text":embed_text }, headers={"accept": "application/json"})
        vector_embedding = response.json()
        return vector_embedding

    # Function to get embeddings from OpenAI
    def get_embedding_OpenAI(self,text):
        text = text.replace("\n", " ")
        if(self.dimensions):
            vector_embedding = self.client.embeddings.create(input = [text], model=self.model, dimensions=self.dimensions).data[0].embedding
        else:
            vector_embedding = self.client.embeddings.create(input = [text], model=self.model,).data[0].embedding
        return vector_embedding

    # Function to get embeddings from Azure OpenAI
    def get_embedding_Azure_OpenAI(self,text):
        text = text.replace("\n", " ")
        vector_embedding = self.client.embeddings.create(input = [text], model=self.model).data[0].embedding
        return vector_embedding

    # Function to get embeddings from Mistral
    def get_embedding_Mistral(self,text):
        vector_embedding = self.client.embeddings(model=self.model, input=[text]).data[0].embedding
        return vector_embedding

    # Function to get embeddings from Fireworks.ai
    def get_embedding_Fireworks(self,text):
        text = text.replace("\n", " ")
        vector_embedding = self.client.embeddings.create(
            model=self.model,
            input=f"search document: {text}",
            dimensions=self.dimensions
        ).data[0].embedding
        return vector_embedding

    def get_embedding_Nomic(self,text):
        vector_embedding = nomic_embed.text(
            texts=[text],
            model=self.model,
            task_type="search_document",
            dimensionality=self.dimensions
        )['embeddings']
        return vector_embedding
    
    # Providing multiple embedding services depending on config
    def get_embedding(self,text):
        if self.provider == "openai":
            return self.get_embedding_OpenAI(text)
        elif self.provider == "vectorservice":
            return self.get_embedding_VectorService(text)
        elif self.provider == "mistral":
            return self.get_embedding_Mistral(text)
        elif self.provider == "azure_openai":
            return self.get_embedding_Azure_OpenAI(text)
        elif self.provider == "fireworks":
            return self.get_embedding_Fireworks(text)
        elif self.provider == "nomic":
            return self.get_embedding_Nomic(text)