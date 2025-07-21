import os
import sys
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.join(script_dir, '..')
# sys.path.insert(0, os.path.abspath(project_root))


from zenml.steps import step
from zenml.pipelines import pipeline
from zenml.client import Client
from typing import List, Dict, Any, Tuple, Annotated,Optional
from datasets import load_dataset,Dataset, DatasetDict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer,BertForSequenceClassification
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
import warnings
from itertools import islice
from utils.utils_download import Util_loaders
# from utils.utils_info import Utilsinfo


@step(enable_cache = False)
def ingest_data() -> DatasetDict:
    init_util = Util_loaders(dataset_name = "stanfordnlp/imdb", model_name = "google-bert/bert-base-uncased", tokenizer_name = "google-bert/bert-base-uncased", download_path = 'D:/Hugging_face')
    model_verify_tokenizer_name = init_util.check_model_tokenizer_names()
    in_data_from_source = init_util.load_data()

    return in_data_from_source

@step(enable_cache = True)
def load_model() -> BertForSequenceClassification:
    init_util = Util_loaders(dataset_name = "stanfordnlp/imdb", model_name = "google-bert/bert-base-uncased", tokenizer_name = "google-bert/bert-base-uncased", download_path = 'D:/Hugging_face')
    model = init_util.load_model()
    return model
    
@step(enable_cache = True)
def load_tokenizer() -> PreTrainedTokenizerFast:
    init_util = Util_loaders(dataset_name = "stanfordnlp/imdb", model_name = "google-bert/bert-base-uncased", tokenizer_name = "google-bert/bert-base-uncased", download_path = 'D:/Hugging_face')
    tokenizer = init_util.load_tokenizer()
    return tokenizer

# @step
# def process_load_dataset() -> DatasetDict:
#     init_info = Utilsinfo('None')
#     local_dataset = init_info.dataset
#     return local_dataset

@pipeline
def requirements_pipeline():
    data = ingest_data()
    model = load_model()
    tokenizer = load_tokenizer()
    return data,model,tokenizer

if __name__ ==  '__main__':
    print("starting_pipeline\n")
    requirements_pipeline()