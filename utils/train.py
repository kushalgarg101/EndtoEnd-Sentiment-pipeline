from datasets import load_dataset
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,get_scheduler
import os
from utils_download import Util_loaders
from utils_info import Utilsinfo,Data_Process
from torch.optim import AdamW
from tqdm.auto import tqdm
import evaluate
from itertools import islice
import wandb

def train(model, num_epochs, lr, train_dataloader, test_dataloader):
    num_training_steps = num_epochs * 30

    with wandb.init(
        project="sentiment-classifier",
        notes="subset_check_fixed_table_logging",
        config={
            "epochs": num_epochs,
            "learning_rate": lr,
            "batch_size": train_dataloader.batch_size,
            "num_training_steps": num_training_steps,
        }
    ) as run:
        optimizer = AdamW(model.parameters(), lr=run.config.learning_rate)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=run.config.num_training_steps,
        )

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        accuracy_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")

        global_step = 0
        for epoch in range(run.config.epochs):
            train_table = wandb.Table(columns=["step", "loss", "prediction", "label", "correct"])
            eval_table  = wandb.Table(columns=["epoch", "label", "prediction", "correct"])

            model.train()
            train_iter = islice(train_dataloader, 30)
            for batch in train_iter:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss

                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                labels = batch["labels"]

                for p, l in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                    train_table.add_data(global_step, loss.item(), p, l, bool(p == l))

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                run.log({"loss": loss.item(), "step": global_step})
                global_step += 1
            
            run.log({"train_predictions_epoch_" + str(epoch): train_table})

            model.eval()
            eval_predictions = []
            eval_references  = []
            eval_iter = islice(test_dataloader, 30)
            with torch.no_grad():
                for batch in eval_iter:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)
                    labels = batch["labels"]

                    for p, l in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                        eval_table.add_data(epoch, l, p, bool(p == l))

                    eval_predictions.extend(preds.cpu().tolist())
                    eval_references.extend(labels.cpu().tolist())

            results_accuracy  = accuracy_metric.compute(predictions=eval_predictions, references=eval_references)
            results_precision = precision_metric.compute(predictions=eval_predictions, references=eval_references, average="binary", pos_label=1)
            results_recall    = recall_metric.compute(predictions=eval_predictions, references=eval_references, average="binary", pos_label=1)

            run.log({
                "epoch": epoch,
                "eval_accuracy":  results_accuracy["accuracy"],
                "eval_precision": results_precision["precision"],
                "eval_recall":    results_recall["recall"],
                "eval_predictions_epoch" + str(epoch): eval_table
            })

if __name__ == '__main__':
    print("Starting training script...")
    init_utilsinfo = Utilsinfo("google-bert/bert-base-uncased")
    token_dataset = init_utilsinfo.dataset_to_token_mapper()
    data_process = Data_Process.remove_column(token_dataset)
    data_rename_and_removed_col = Data_Process.rename_column(data_process)
    train_processed_dataset, test_processed_dataset, _ = Data_Process.dynamic_padding_and_batching(
        data_rename_and_removed_col, init_utilsinfo.tokenizer
    )
    
    train(
        model=init_utilsinfo.model,
        num_epochs=5,
        lr=5e-5,
        train_dataloader=train_processed_dataset,
        test_dataloader=test_processed_dataset
    )