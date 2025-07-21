from typing import List, Dict, Tuple, Annotated, Optional
from pydantic import BaseModel,ValidationError, model_validator,Field
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer,DataCollatorWithPadding,get_scheduler
import os
from utils_download import Util_loaders
from itertools import islice

class Utilsinfo:
    def __init__(self, model_name:str):
        self.dataset = load_dataset("arrow", data_files={'train': r"D:\Hugging_face\train\data-00000-of-00001.arrow", 'test': r"D:\Hugging_face\test\data-00000-of-00001.arrow", 'unsupervised': r"D:\Hugging_face\unsupervised\data-00000-of-00001.arrow"})
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def check_tokenizer_outputs(self, text: List[str]):
        return self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    
    def get_prediction(self, tokenized_inputs):
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)

        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=-1)
        predicted_score = torch.softmax(logits, dim=-1)

        return predicted_class_idx,predicted_score

    def __tokenize_function(self,datasetdict):
        token_text = self.tokenizer(datasetdict['text'], truncation=True)

        return token_text
    
    def dataset_to_token_mapper(self):
        token_dataset = self.dataset.map(self.__tokenize_function, batched = True)

        return token_dataset
class Data_Process:

    @staticmethod
    def remove_column(tokenized_datasets):
        remove_data_col = {split_name: dataset.remove_columns(['text']) for split_name, dataset in tokenized_datasets.items()}

        return remove_data_col
    
    @staticmethod
    def rename_column(tokenized_datasets):
        rename_data_col = {split_name: dataset.rename_column("label", "labels") for split_name, dataset in tokenized_datasets.items()}

        return rename_data_col

    @staticmethod
    def dynamic_padding_and_batching(tokenized_datasets, tokenizer):
        pad_data = DataCollatorWithPadding(tokenizer=tokenizer)

        train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=8, collate_fn=pad_data)
        eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=8, collate_fn=pad_data)
        unsupervised_dataloader = DataLoader(tokenized_datasets['unsupervised'], batch_size=16, collate_fn=pad_data)

        return [train_dataloader, eval_dataloader, unsupervised_dataloader]
    
if __name__ == '__main__':
    init_utilsinfo = Utilsinfo("google-bert/bert-base-uncased")
    print(init_utilsinfo.model)
    print(init_utilsinfo.tokenizer)
    input = init_utilsinfo.check_tokenizer_outputs(["The movie was very good", "the movie was so bad" ,"so my friends were right when they said to dont watch this movie"])
    print(input)
    pred_cla, pred_sc = init_utilsinfo.get_prediction(input)
    print(pred_cla)
    print(pred_sc)
    print(init_utilsinfo.dataset)
    token_dataset = init_utilsinfo.dataset_to_token_mapper()
    print(token_dataset['train'][0])

    data_process = Data_Process.remove_column(token_dataset)
    data_rename_and_removed_col = Data_Process.rename_column(data_process)
    print(data_rename_and_removed_col)
    train_processed_dataset,test_processed_dataset,unsupervised_processed_dataset = Data_Process.dynamic_padding_and_batching(data_rename_and_removed_col, init_utilsinfo.tokenizer)
    first_2 = islice(train_processed_dataset, 2)
    print(train_processed_dataset.batch_size)
    for batch in first_2:
        print(batch)