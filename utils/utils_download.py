from zenml.steps import step
from zenml.pipelines import pipeline
from zenml.client import Client
from typing import List, Dict, Any, Tuple, Annotated,Optional
from pydantic import BaseModel,ValidationError, model_validator,Field
from datasets import load_dataset
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import warnings
import os
import wandb

class Util_loaders(BaseModel):
    dataset_name : str = Field(..., description="Name of the dataset to load")
    model_name : str = Field(..., description="Name of the model to load (often same as tokenizer_name)")
    tokenizer_name : str = Field(..., description="Name of the tokenizer to load (often same as model_name)")
    download_path : Optional[str] = Field(None, description="Optional path to download these datasets, models, and tokenizers.")

    @model_validator(mode='after')
    def check_model_tokenizer_names(self):

        if self.model_name != self.tokenizer_name:
            warnings.warn(
                f"Warning: model_name ('{self.model_name}') and tokenizer_name ('{self.tokenizer_name}') are different. "
                "It's generally recommended for them to be the same when using pre-trained models."
            )
        if self.download_path:
            try:
                os.makedirs(self.download_path, exist_ok=True)
                print(f"Creating download directory: {os.path.abspath(self.download_path)}")
            except OSError as e:
                print(f"Error creating download directory '{self.download_path}': {e}")
                self.download_path = None

        return self
    
    def load_data(self):
        """Loads the dataset specified by dataset_name."""

        try:
            ds = load_dataset(self.dataset_name)
            print(f"Successfully downloaded dataset: {self.dataset_name}")
            if self.download_path != None:
                print("saving dataset disk locally")
                ds.save_to_disk(self.download_path)
            return ds
        except Exception as e:
            print(f"Error downloading dataset '{self.dataset_name}': {e}")
            return None
        
    def load_model(self):
        """Loads the model specified by model_name."""

        try:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.download_path)
            print(f"Successfully downloaded model: {self.model_name}")
            return model
        except Exception as e:
            print(f"Error loading model '{self.model_name}': {e}")
            return None
        
    def load_tokenizer(self):
        """Loads the tokenizer specified by tokenizer_name."""

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, cache_dir=self.download_path)
            print(f"Successfully downloaded tokenizer: {self.tokenizer_name}")
            return tokenizer
        except Exception as e:
            print(f"Error loading tokenizer '{self.tokenizer_name}': {e}")
            return None

if __name__ == '__main__':
    check1 = Util_loaders(dataset_name = "stanfordnlp/imdb", model_name = "google-bert/bert-base-uncased", tokenizer_name = "google-bert/bert-base-uncased", download_path = 'D:/Hugging_face')
    ds= check1.load_data()
    dm = check1.load_model()
    dt = check1.load_tokenizer()
    print(dm)
    print(dt)