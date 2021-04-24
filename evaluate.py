import os
import time
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertForMaskedLM,
)

from data_extractor import TorchDataset


def main():
    batch_size = 4

    dev_dataset = TorchDataset(
        file_name="./data/diverse.triplets.dev.tsv",
        queries_path="./data/diverse.queries.all.tsv",
        passages_path="./data/diverse.passages.all.tsv",
    )
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TorchDataset(
        file_name="./data/diverse.triplets.test.tsv",
        queries_path="./data/diverse.queries.all.tsv",
        passages_path="./data/diverse.passages.all.tsv",
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # load model
    # DistilBertForSequenceClassification
    # DistilBertForMaskedLM
    model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
    model.load_state_dict(torch.load("demo_model.pt", map_location=device))
    model.to(device)

    model.eval()

    data_loader = dev_dataloader
    N = len(data_loader)

    correct, total = 0, 0
    start = time.time()
    with torch.no_grad():
        for i, (queries, pos_docs, neg_docs) in enumerate(data_loader):

            inputs = list(queries) + list(pos_docs) + list(neg_docs)

            encodings = tokenizer(
                inputs,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            ids, masks = encodings["input_ids"], encodings["attention_mask"]

            ids = ids.to(device)  # (3B, MAXLEN)
            masks = masks.to(device)  # (3B, MAXLEN)

            # TODO: could add more layers after distilbert!
            outputs = model.distilbert(ids, masks)
            outputs_hidden = outputs.last_hidden_state.mean(dim=1)
            anchors, positives, negatives = outputs_hidden.view(3, len(queries), -1)

            # compute 2 distance: positive_doc to query, negative_doc to query using l2 distance
            pos_dist = (anchors - positives).norm(dim=-1)  # B distances
            neg_dist = (anchors - negatives).norm(dim=-1)  # B distances

            # pos_dist = 1 - F.cosine_similarity(anchors, positives, dim=-1)  # B distances
            # neg_dist = 1 - F.cosine_similarity(anchors, negatives, dim=-1)  # B distances

            correct += (pos_dist < neg_dist).sum()
            total += len(queries)

            if i % 10 == 0:
                remaining_time = (time.time() - start) / (i + 1) * N - (time.time() - start)
                print(
                    f"remaining time: {remaining_time:.2f} | est. accuracy: {correct / total:.4f}"
                )

        print(f"accuracy {correct / total}")


if __name__ == "__main__":
    main()