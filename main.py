import argparse
import copy
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils.training_utils import device
from utils.preprocessing_utils import create_labels_mapping, create_vocab
from dataset.ag_news_dataset import AGNewsDataset, collate_func
from model.model import RNNTextClassifier


def train(model, criterion, optimizer, train_loader: DataLoader):
    # sign the "train" label
    model.train()

    # create a variable named "losses" to store the training losses
    losses = []
    for batch in tqdm(train_loader):
        texts = batch["texts"].to(device)
        labels = batch["labels"].to(device)

        # the result of prediction in this step
        current_prediction = model(texts)

        # criterion function: input[batch_size, num_labels], output[batch_size]
        current_loss = criterion(current_prediction, labels)
        losses.append(current_loss)

        # back propagation
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()

    # calculate and print the training loss for this epoch
    train_loss = torch.tensor(losses).mean()
    print(f"Train Loss : {train_loss:.3f}")


def validate(model, criterion, dev_loader: DataLoader) -> float:
    # sign the "validate" label
    model.eval()

    # init "all_prediction" and "all_labels" to store the all results of prediction and labels in dev data
    all_labels = []
    all_predictions = []

    # init the variable "losses" to store all loss in dev data
    losses = []

    # the validation step doesn't need gradient descent
    with torch.no_grad():
        for batch in dev_loader:
            # get the data of samples in each batch
            texts = batch["texts"].to(device)
            labels = batch["labels"].to(device)

            predictions = model(texts)

            loss = criterion(predictions, labels)
            losses.append(loss.item())

            all_labels.append(labels)
            # predictions are the shape as [batch_size, 4]
            all_predictions.append(predictions.argmax(dim=-1))

    # all_labels and all_predictions are list of many tensors, so need to concatenate them to a tensor
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)

    valid_loss = torch.tensor(losses).mean()

    # accuracy_score function is not compatible with tensor, so we need to change them to numpy
    valid_acc = accuracy_score(
        y_true=all_labels.detach().cpu().numpy(),
        y_pred=all_predictions.detach().cpu().numpy()
    )

    print(f"Valid Loss : {valid_loss:.3f}")
    print(f"Valid Acc  : {valid_acc:.3f}")

    return valid_loss


def test(model, test_loader: DataLoader, labels_mapping: dict):

    # init "all_prediction" and "all_labels" to store the all results of prediction and labels in dev data
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            texts = batch["texts"].to(device)
            labels = batch["labels"].to(device)

            predictions = model(texts)

            all_labels.append(labels)
            # predictions are the shape as [batch_size, 4]
            all_predictions.append(predictions.argmax(dim=-1))

    all_labels = torch.cat(all_labels).detach().cpu().numpy()
    all_predictions = torch.cat(all_predictions).detach().cpu().numpy()

    # show the results of testing
    test_acc = accuracy_score(
        y_true=all_labels,
        y_pred=all_predictions,
    )

    print(f"Test Acc   : {test_acc:.3f}")

    print("\nClassification Report : ")
    print(classification_report(all_labels, all_predictions, target_names=labels_mapping.keys()))

    print("\nConfusion Matrix : ")
    print(confusion_matrix(all_labels, all_predictions))


def predict(model, text: str, tokenizer: Callable, vocab: Vocab, labels_mapping: dict):

    # get the tokens from the text
    tokens = tokenizer(text)

    # change the textual tokens to their indexes
    indexes = vocab(tokens)

    # Because the model only accept the mini-batch data, we need to change the shape of indexes
    temp_input = torch.tensor([indexes]).to(device)

    with torch.no_grad():
        prediction = model(temp_input)

    # the reason that add ".item()" : get the data from a tensor
    prediction_index = prediction[0].argmax(dim=0).item()

    prediction_label = {
        index: label for label, index in labels_mapping.items()
    }[prediction_index]

    print(f"\ntext: {text}")
    print(f"prediction: {prediction_label}")


def main(args: argparse.Namespace):
    # 1. Load the data
    train_dataset = AGNewsDataset("text_data/train.csv")
    dev_dataset = AGNewsDataset("text_data/dev.csv")
    test_dataset = AGNewsDataset("text_data/test.csv")

    # 2. Init the tokenizer [tokenizer -> input: text:str, output: tokens:list(str)]
    tokenizer = get_tokenizer("basic_english")

    # 3. Init the vocabulary [vocab -> input: tokens:list(str), output: indexes:list(int)]
    vocab = create_vocab(texts=train_dataset.texts, tokenizer=tokenizer, min_freq=args.min_freq)
    print(vocab)
    exit()


    # 4. Create the label mapping [labels_mapping -> input: textual labels:list(str), output: labels_indexes:list(int)]
    labels_mapping = create_labels_mapping(labels=train_dataset.labels)

    # 5. Construct the model
    model = RNNTextClassifier(
        vocab_len=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        labels_len=len(labels_mapping)
    )
    model.to(device=device)

    # 6. Choose the loss function
    criterion = nn.CrossEntropyLoss()

    # 7. Choose the optimizer
    optimizer = Adam(
        params=model.parameters(),
        lr=args.lr,
    )

    collate_fc = lambda samples: collate_func(
        samples=samples,
        tokenizer=tokenizer,
        vocab=vocab,
        max_len=args.max_len,
        labels_mapping=labels_mapping,
    )

    # 8. Construct the dataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fc
    )

    # dev and test loader don't need to be shuffle
    dev_loader = DataLoader(
        dataset=dev_dataset,
        batch_size=args.dev_batch_size,
        shuffle=False,
        collate_fn=collate_fc
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=collate_fc
    )

    # 9. Start to train, develop, test the model
    best_model = None
    min_validate_loss = float("inf")

    for i in range(args.epoch_num):
        print(f"Epoch: {i + 1}")
        train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
        )

        validate_loss = validate(
            model=model,
            criterion=criterion,
            dev_loader=dev_loader,
        )

        print()
        # Update and store the best model
        if validate_loss < min_validate_loss:
            min_validate_loss = validate_loss
            # assign the memory and copy the whole model
            best_model = copy.deepcopy(model)

    best_model.to(device=device)
    test(
        model=best_model,
        test_loader=test_loader,
        labels_mapping=labels_mapping,
    )

    predict(
        model=best_model,
        text="Roddick to Lead U.S. Against Belarus (AP),AP - Andy Roddick and the rest of the U.S. Davis Cup team figure it's about time the country reclaimed the championship.",
        tokenizer=tokenizer,
        vocab=vocab,
        labels_mapping=labels_mapping,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data parameters
    parser.add_argument("--min_freq", type=int, default=3, help="Minimum frequency of words added to vocab")
    parser.add_argument("--max_len", type=int, default=25, help="Max length of sentences be input to the dataLoader")
    # model parameters
    parser.add_argument("--embedding_dim", type=int, default=100, help="the dimension of the word embedding")
    parser.add_argument("--hidden_dim", type=int, default=50, help="the dimension of the vectors from hidden layer")
    # process parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=64, help="The size of training batch")
    parser.add_argument("--dev_batch_size", type=int, default=32, help="The size of validation batch")
    parser.add_argument("--test_batch_size", type=int, default=32, help="The size of testing batch")
    parser.add_argument("--epoch_num", type=int, default=10, help="The number of the training epochs")

    args = parser.parse_args()
    main(args)
