import os
import time
import gc
import argparse
import numpy as np
import copy
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertForMaskedLM,
)

import evaluate

from data_extractor import DataExtractor
from data_extractor import TorchDataset


def str2bool(v):
    """
    Argument Parse helper function for boolean values
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def generate_data_for_plot(learning_curve_y, learning_curve_x):
    learning_curve = list(zip(learning_curve_y, learning_curve_x))
    file = open('./progress.csv', 'w+', newline='')
    header = 'y,x\n'
    with file:
        mywriter = csv.writer(file)
        mywriter.writerow(header.strip().split(","))
        mywriter.writerows(learning_curve)
    print(f"Successfully generated csv for learning curve!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, default=2)
    parser.add_argument("--save", type=str, default="./model1_best_base.pt")
    args = parser.parse_args()

    # Data and Tokenization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    batch_size = 4
    train_dataset = TorchDataset(
        file_name="./data/diverse.triplets.train.tsv",
        queries_path="./data/diverse.queries.all.tsv",
        passages_path="./data/diverse.passages.all.tsv",
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_dataset = TorchDataset(
        file_name="./data/diverse.triplets.dev.tsv",
        queries_path="./data/diverse.queries.all.tsv",
        passages_path="./data/diverse.passages.all.tsv",
    )
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # Model Training and Evaluation
    NUM_EPOCHS = 1
    LEARNING_RATE = 0.00003

    # load model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    if args.exp == 3:
        # model.load_state_dict(torch.load(model_path+"model1_best.pt"))
        model_frozen = copy.deepcopy(model)
        for param in model_frozen.distilbert.parameters():
            param.requires_grad = False
        model = model_frozen

    model.to(device)
    model.train()
    if args.exp < 3:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif args.exp == 3:
        optimizer = torch.optim.Adam(model.distilbert.parameters(), lr=LEARNING_RATE)

    def evaluate(inputs, model, tokenizer, labels):
        encodings = tokenizer(
            inputs,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        ids, masks = encodings["input_ids"], encodings["attention_mask"]
        outputs = model(ids.to(device), masks.to(device), labels=labels.to(device))

        return outputs

    dataloader = train_dataloader
    N = len(dataloader)
    lowest_loss = float("inf")
    start = time.time()
    learning_curve_y = []
    learning_curve_x = []

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for i, (queries, pos_docs, neg_docs) in enumerate(dataloader):
            if args.exp != 1:
                optimizer.zero_grad()  # set gradient to zero

                queries = list(queries) * 2  # 2*B
                docs = list(pos_docs) + list(neg_docs)

                labels = torch.cat([torch.ones(len(pos_docs)),
                                    torch.zeros(len(neg_docs))]).long().to(device)  # 2*batch,

                outputs = evaluate(
                    inputs=list(zip(queries, docs)),
                    model=model,
                    tokenizer=tokenizer,
                    labels=labels,
                )

                loss = outputs.loss
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss)

                if i % 10 == 0:
                    elapsed_time = time.time() - start
                    remaining_time = elapsed_time * (1 / (i + 1) * N - 1)
                    print(
                        f"{i}: remaining time: {remaining_time:.1f} | est. epoch loss: {epoch_loss / (i + 1):.4f}"
                    )

            if i % 10 == 0:
                with torch.no_grad():
                    correct = total = 0
                    val_start = time.time()
                    for dq, dp, dn in dev_dataloader:
                        queries = list(dq) * 2  # 2*B
                        docs = list(dp) + list(dn)
                        labels = torch.cat([torch.ones(len(dp)),
                                            torch.zeros(len(dn))]).long().to(device)
                        outputs = evaluate(
                            inputs=list(zip(queries, docs)),
                            model=model,
                            tokenizer=tokenizer,
                            labels=labels
                        )

                        predicted_classes = outputs.logits.argmax(dim=-1)
                        correct += (labels == predicted_classes).sum()

                        total += len(labels)
                        if time.time() - val_start > 15:
                            break
                    print(f"{i}: est. validation accuracy: {correct / total:.4f}")
                    learning_curve_y.append(correct / total)
                    learning_curve_x.append(i * batch_size)  # epoch normally

            if (epoch_loss / (i + 1)) < lowest_loss:
                if args.exp == 1:
                    torch.save(model.state_dict(), "model1_best_pretrain.pt")
                elif args.exp == 2:
                    torch.save(model.state_dict(), "model1_best_base.pt")
                elif args.exp == 3:
                    torch.save(model.state_dict(), "model1_best_freeze.pt")
                lowest_loss = epoch_loss / (i + 1)

        print(f"loss for epoch {epoch} is {epoch_loss}")
        generate_data_for_plot(learning_curve_y, learning_curve_x)


if __name__ == "__main__":
    main()
