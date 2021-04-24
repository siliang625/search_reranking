import os
import time
import gc
import argparse
import numpy as np
import numpy as np
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
    parser.add_argument("--exp", type=int, default=7)
    parser.add_argument("--save", type=str, default="./model2_best_diverse_mean_maskedLM.pt")
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
    model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

    # if args.exp == 1:
    #     pass
    # elif args.exp == 2:
    #
    # elif args.exp == 3:

    if args.exp == 7:
        # For Experiment7: average
        model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
        triplet_loss = nn.TripletMarginLoss(margin=1.0)

    elif args.exp == 6:
        # For Experiment6: base + cosine
        triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1 - F.cosine_similarity(x, y, dim=-1),
            margin=1.0,
        )

    elif args.exp == 5:
        # For Experiment5: base + margin = 0.1
        triplet_loss = nn.TripletMarginLoss(margin=0.1)

    elif args.exp == 4:
        # For Experiment4: base
        triplet_loss = nn.TripletMarginLoss(margin=1.0)

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.distilbert.parameters(), lr=LEARNING_RATE)

    def evaluate(inputs, model, tokenizer):
        encodings = tokenizer(
            inputs,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        ids, masks = encodings["input_ids"], encodings["attention_mask"]
        outputs = model.distilbert(ids.to(device), masks.to(device))
        if args.exp < 7:
            # Experiment: using the first index of the last layers
            outputs_hidden = outputs.last_hidden_state[:, 0]
        else:
            # Averaging last layers
            outputs_hidden = outputs.last_hidden_state.mean(dim=1)

        return outputs_hidden.view(3, len(queries), -1)

    dataloader = train_dataloader
    N = len(dataloader)
    lowest_loss = float("inf")
    start = time.time()
    learning_curve_y = []
    learning_curve_x = []

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for i, (queries, pos_docs, neg_docs) in enumerate(dataloader):
            # readability
            # train()
            # evaluate()
            # print()
            optimizer.zero_grad()  # set gradient to zero
            anchors, positives, negatives = evaluate(
                inputs=list(queries + pos_docs + neg_docs),
                model=model,
                tokenizer=tokenizer,
            )

            loss = triplet_loss(anchors, positives, negatives)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss)

            if i % 10 == 0:
                elapsed_time = time.time() - start
                remaining_time = elapsed_time * (1 / (i + 1) * N - 1)
                print(
                    f"{i}: remaining time: {remaining_time:.1f} | est. epoch loss: {epoch_loss / (i + 1):.4f}"
                )

            if i % 100 == 0:
                with torch.no_grad():
                    correct = total = 0
                    val_start = time.time()
                    for dq, dp, dn in dev_dataloader:
                        anchors, positives, negatives = evaluate(
                            inputs=list(dq + dp + dn),
                            model=model,
                            tokenizer=tokenizer,
                        )
                        if args.exp == 6:
                            # cosine distance
                            pos_dist = 1 - F.cosine_similarity(anchors, positives, dim=-1)
                            neg_dist = 1 - F.cosine_similarity(anchors, negatives, dim=-1)
                        else:
                            # using l2 norm
                            pos_dist = (anchors - positives).norm(dim=-1)  # B distances
                            neg_dist = (anchors - negatives).norm(dim=-1)  # B distances

                        correct += float((pos_dist < neg_dist).sum())
                        total += len(dq)
                        if time.time() - val_start > 15:
                            break
                    print(f"{i}: est. validation accuracy: {correct / total:.4f}")
                    learning_curve_y.append(correct / total)
                    learning_curve_x.append(i * batch_size)  # epoch normally

            if (epoch_loss / (i + 1)) < lowest_loss:
                if args.exp == 4:
                    torch.save(model.state_dict(), "model2_best_diverse_base.pt")
                elif args.exp == 5:
                    torch.save(model.state_dict(), "model2_best_diverse_margin.pt")
                elif args.exp == 6:
                    torch.save(model.state_dict(), "model2_best_diverse_cosine.pt")
                elif args.exp == 7:
                    torch.save(model.state_dict(), "model2_best_diverse_mean_maskedLM.pt")

                lowest_loss = epoch_loss / (i + 1)

        print(f"loss for epoch {epoch} is {epoch_loss}")

        generate_data_for_plot(learning_curve_y, learning_curve_x)


if __name__ == "__main__":
    main()
