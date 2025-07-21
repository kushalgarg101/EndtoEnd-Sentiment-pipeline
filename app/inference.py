import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from utils_info import Data_Process

class Infer:
    def __init__(self, input):
        self.input = input
        self.__model_path = r"D:\End_to_end_sentiment\app\weights"
        self.tokenize = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.tuned_model = AutoModelForSequenceClassification.from_pretrained(self.__model_path)

    def model_outputs(self):
        self.tokenizer_text = self.tokenize(self.input, padding = True, truncation = True, return_tensors="pt")
        outs = self.tuned_model(**self.tokenizer_text)
        logits = outs.logits
        probabs = torch.sigmoid(logits).detach().numpy()
        labels = ["Positive" if x[1] > 0.5 else "Negative" for x in probabs]
        
        return labels
    
if __name__ == "__main__":
    init_infer = Infer(["this movie was so bad", "this movie was one which i enjoyed most out of previous ones"])
    print(init_infer.model_outputs())