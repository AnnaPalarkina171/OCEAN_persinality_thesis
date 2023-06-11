from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from torch.optim import AdamW
from torch.utils import data
from sklearn import metrics
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import logging
import random
import torch
import sys
import os
  
def seed_everything(seed_value=42):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    return seed_value

def encoder(labels, texts, cur_tokenizer, cur_device):
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(cur_device)
    encoding = cur_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=maxl,
    ).to(cur_device)
    return labels_tensor, encoding

def create_ids_mask(dataset):
    texts = dataset.post_text.to_list()
    labels = dataset[prediction_column].to_list()
    labels_tensor, encoding = encoder(labels, texts, tokenizer, device)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    return data.TensorDataset(input_ids, attention_mask, labels_tensor) 

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    arg = parser.add_argument
    arg("--model", "-m", help="Path to a BERT model", default='xlm-roberta-base') #required=True)
    arg("--tokenizer", "-t", help="Path to a BERT model", default='xlm-roberta-base') #required=True)
    arg("--predict", "-p", help="Prediction column", required=True)
    arg("--lr",  type=float, help="Learning Rate", default=4e-5)
    arg("--dropout",  type=float, help="Dropout", default=0.3)
    arg("--epochs", "-e", type=int, help="Number of epochs", default=15)
    arg("--maxl", type=int, help="Max length", default=100)
    arg("--minl", type=int, help="Min length", default=5)
    arg("--bsize", "-b", type=int, help="Batch size", default=128)
    arg("--seed", "-s", type=int, help="Random seed", default=42)
    arg("--warmup_steps", "-ws", type=int, help="Warmup steps", default=1000)


    args = parser.parse_args()
    _ = seed_everything(args.seed)
    logger.info(f"Training with seed {args.seed}...")
    model_name = args.model
    logger.info(f"Model name {model_name}...")
    tokenizer_name = args.tokenizer
    logger.info(f"Tokenizer name {tokenizer_name}...")
    prediction_column = args.predict
    logger.info(f"Prediction column {prediction_column}...")
    LR = args.lr
    logger.info(f"Learning rate {LR}...")
    dropout = args.dropout
    logger.info(f"Dropout  {dropout}...")
    epochs = args.epochs
    logger.info(f"Epochs {epochs}...")
    maxl = args.maxl
    logger.info(f"Max length {maxl}...")
    minl = args.minl
    logger.info(f"Min length {minl}...")
    bsize = args.bsize
    logger.info(f"Batch size {bsize}...")
    warmup_steps = args.warmup_steps
    logger.info(f"Warmup steps {warmup_steps}...")
    freeze = False
    logger.info(f"Freeze {freeze}...")



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dataset = pd.read_csv('../3_classes_light_preprocess.csv')
    dataset['lengths'] = dataset.post_text.apply(lambda x: len(str(x).split()))
    df = dataset.query('lengths > @minl')[['post_text', prediction_column]]

    train_data, valid_data = train_test_split(df, test_size=0.3, stratify=df[prediction_column], random_state=args.seed)
    valid_data, test_data = train_test_split(valid_data, test_size=0.5, stratify=valid_data[prediction_column], random_state=args.seed)

    logger.info(f"Train dataset with oversampling shape: {train_data.shape}...") 
    logger.info(f"Valid dataset shape: {valid_data.shape}...")
    logger.info(f"Test dataset shape: {test_data.shape}...")

    num_classes = train_data[prediction_column].nunique()
    logger.info(f"We have {num_classes} classes")

    logger.info(f"Tokenizing with max length {maxl}...")
    train_dataset = create_ids_mask(train_data)
    train_iter = data.DataLoader(train_dataset, batch_size=bsize, shuffle=True) 

    dev_dataset = create_ids_mask(valid_data)
    dev_iter = data.DataLoader(dev_dataset, batch_size=bsize, shuffle=False)

    test_dataset = create_ids_mask(test_data)
    test_iter = data.DataLoader(test_dataset, batch_size=bsize, shuffle=False)
    logger.info("Tokenizing finished.")

    # bert = AutoModel.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)
    # model = Classifier(n_classes=num_classes, bert=bert).to(device)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True).to(device)

    weights = [len(x) / len(train_data) for i, x in train_data.groupby(prediction_column)]
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))

    optimizer = AdamW(model.parameters(), lr=LR)

    total_steps = len(train_dataset) * epochs
    #warmup_steps = int(total_steps * 0.1)
    warmup_steps = warmup_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps)

    if freeze:
        logger.info("Freezing the model, training only the classifier on top")
        for param in model.base_model.parameters():
            param.requires_grad = False

    plot_train_losses, plot_dev_losses = [], []

    logger.info(f"Training with batch size {bsize} for {epochs} epochs...")

    for epoch in range(epochs):
        logger.info(f"-------------------------- Epoch: {epoch} --------------------------")
        # train
        model.train()
        train_losses = 0
        y_true_train, y_pred_train = [], []
        for input_ids, attention_mask, labels_tensor in train_iter:
            optimizer.zero_grad()

            outputs = model(input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True).logits

            loss = loss_fn(outputs, labels_tensor)
            train_losses += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            batch_predictions = torch.log_softmax(outputs, dim=1).argmax(dim=1)
            y_true_train += labels_tensor.tolist()
            y_pred_train += batch_predictions.tolist()
        accuracy_train = accuracy_score(y_true=y_true_train, y_pred=y_pred_train)
        f1_train = f1_score(y_true=y_true_train, y_pred=y_pred_train, average="macro")
        train_loss = train_losses / len(train_iter)
        plot_train_losses.append(train_loss)

        logger.info(
            f"Train loss: {train_loss:.4f}, Train accuracy: {accuracy_train:.4f}, Train F1: {f1_train:.4f}"
        )

        # validation
        model.eval()
        val_losses = 0
        y_true_val, y_pred_val = [], []
        with torch.no_grad():
            for input_ids, attention_mask, labels_tensor in dev_iter:

                outputs = model(input_ids=input_ids,
                attention_mask=attention_mask).logits

                loss = loss_fn(outputs, labels_tensor)
                val_losses += loss.item()
                batch_predictions = torch.log_softmax(outputs, dim=1).argmax(dim=1)
                y_true_val += labels_tensor.tolist()
                y_pred_val += batch_predictions.tolist()
        accuracy_dev = accuracy_score(y_true=y_true_val, y_pred=y_pred_val)
        f1_dev = f1_score(y_true=y_true_val, y_pred=y_pred_val, average="macro")
        val_loss = val_losses / len(dev_iter)
        plot_dev_losses.append(val_loss)
        logger.info(
            f"Dev loss: {val_loss:.4f}, Validation accuracy: {accuracy_dev:.4f}, Validation F1: {f1_dev:.4f}"
        )
        logger.info(
            f"{classification_report(y_true_val, y_pred_val)}"
        )

    print(f"Train losses: ", plot_train_losses)
    print("Validation losses:", plot_dev_losses)

    torch.save(model.state_dict(),f'3_classes_light_preprocess_0520_{prediction_column}.bin')
    


    # test
    model.eval()
    y_true_test, y_pred_test = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels_tensor in test_iter:
            outputs = model(input_ids=input_ids,
                attention_mask=attention_mask).logits
            batch_predictions = torch.log_softmax(outputs, dim=1).argmax(dim=1)
            y_true_test += labels_tensor.tolist()
            y_pred_test += batch_predictions.tolist()
    accuracy_test = accuracy_score(y_true=y_true_test, y_pred=y_pred_test)
    f1_test = f1_score(y_true=y_true_test, y_pred=y_pred_test, average="macro")

    logger.info(
            f"Test accuracy: {accuracy_test:.4f}, Test F1: {f1_test:.4f}"
        )
    logger.info(
            f"{classification_report(y_true_test, y_pred_test)}"
        )
    
    logger.info('Done!')
