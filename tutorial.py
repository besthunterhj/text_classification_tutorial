import csv
import re
from typing import Dict
import torch
from torch.optim import Adam
from collections import Counter
from collections.abc import Callable
from torchtext.vocab import Vocab
from torchtext.vocab import vocab as vc
from torchtext.data import get_tokenizer
from torch import nn
from torch.utils.data import Dataset, DataLoader


# 1. Create a class named AGNewsDataset to read the train, valid, test data
class AGNewsDataset(Dataset):
    def __init__(self, fp: str):
        """
        Init the dataset and read from the file
        :param fp: the target file that will be read
        """
        self.texts, self.labels = self.read(fp)

    def __len__(self):
        """
        Get the length of the dataset
        :return: the length(int)
        """
        return len(self.labels)

    def __getitem__(self, index: int):
        """
        Get the target tuple(text, label) from the dataset by searching correspond index
        :param index: the index of the list(text and label)
        :return: target text and label
        """
        current_text = self.texts[index]
        current_label = self.labels[index]
        return current_text, current_label

    @classmethod
    def read(cls, fp: str):
        """
        Read the data from dataset and return text_list and label_list
        :param fp: the path of the target file
        :return: text_list and label_list
        """
        text_data = []  # as X
        labels = []  # as Y

        with open(fp) as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                news_category, news_title, news_body = line

                # build the data_x
                text = news_title + " " + news_body

                # remove the meaningless char
                text = re.sub("&[A-Za-z]{1,2};", "", text)
                text = re.sub(" #[0-9]{2};", "", text)
                text = text.replace("\\", " ")
                text = re.sub("[A-Za-z]{1,10}=[A-Za-z0-9]{1,10} ", "", text)
                text = re.sub("[A-Za-z]{1,10}=\"{1, 2}[\\w]\"{1,2}", "", text)
                text = re.sub("[A-Za-z]{1,10}=[\"]{1,2}http[s]{0,1}://([\w.]+/?)\S*", "", text)
                text = re.sub("target=[\s\w/]{1,50}\"\"", "", text)

                text_data.append(text)

                # build the data_y
                labels.append(news_category)

        return text_data, labels


# 2. Init the tokenizer from torchtext(finished in the "main" block)


# 3. Create the vocabulary to map the token to its index correspondingly
def create_vocab(texts: list, tokenizer: Callable, min_freq: int, unknown_token: str = "<unk>",
                 unknown_index: int = 0) -> Vocab:
    # 3.1 Get all tokens from texts read from dataset
    all_tokens = [
        token
        for text in texts
        for token in tokenizer(text)
    ]

    # 3.2 Make a counter to count the tokens
    tokens_counter = Counter(all_tokens)

    # 3.3 Init the tokens_dist and sort the tokens
    tokens_dict = dict(
        sorted(
            tokens_counter.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    # 3.4 Init the vocabulary
    vocab = vc(
        ordered_dict=tokens_dict,
        min_freq=min_freq
    )

    # 3.5 In order to handle the unknown token, insert the <unk> token to the vocabulary
    vocab.insert_token(
        token=unknown_token,
        index=unknown_index
    )

    # Set the default token as <unk> to handle the out-of-vocabulary (OOV) error
    vocab.set_default_index(index=unknown_index)

    return vocab


# 4. Create the label mapping to map the label to its index correspondingly
def create_labels_mapping(labels: list) -> Dict[str, int]:
    labels = list(set(labels))
    # sort the labels by their first character
    sorted(labels, reverse=True)
    labels_mapping = {
        textual_label: index
        for index, textual_label in enumerate(labels)
    }

    return labels_mapping


# 5. Init a class named RNNTextClassifier which will be used to finish the classification task
class RNNTextClassifier(nn.Module):
    # 5.1 Finish the construction of the model
    def __init__(self, vocab_len: int, embedding_dim: int, hidden_dim: int, labels_len: int):
        super(RNNTextClassifier, self).__init__()
        # Init the attributes of the model(In other word, they are the hyper-parameters)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Init the layer of the RNN model
        self.embedding = nn.Embedding(vocab_len, self.embedding_dim)
        # Read the document of pytorch to finish this block
        self.rnn = nn.RNN(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            # nonlinearity='relu'
        )
        self.linear = nn.Linear(
            self.hidden_dim, labels_len
        )

    # 5.2 Init the process of the model
    def forward(self, x: torch.tensor):
        """

        :param x: the input of the model, its size must be [batch_size, text_len]
        :return:
        """
        # the size of the output of embedding function is [batch_size, text_len, embedding_dim]
        embeddings = self.embedding(x)

        # last_hidden is the final state of the rnn
        _, last_hidden = self.rnn(embeddings)

        # remove the first dimension of last_hidden[1, batch_size, hidden_dim] -> last_hidden[batch_size, hidden_dim]
        last_hidden = last_hidden.squeeze(dim=0)

        # Let the size of output feature is same as the len of labels by Linear Layer -> y_hat[batch_size, labels_len]
        y_hat = self.linear(last_hidden)

        return y_hat


def pad(token_indexes: list):
    ...


def collate_func(samples, tokenizer: Callable, vocab: Vocab, labels_mapping: dict):
    # 0. Separate texts and labels from the parameter samples(the element of samples is tuple)
    texts, labels = list(zip(*samples))

    # 1. Texts:
    #   + tokenization
    #   + covert the tokens to indexes
    #   + padding
    #   + convert to a tensor

    # 2. Labels:
    #   + convert the textual labels to integer label
    #   + convert to a tensor


if __name__ == "__main__":
    # 1. Load the data
    train_dataset = AGNewsDataset("text_data/train.csv")
    dev_dataset = AGNewsDataset("text_data/dev.csv")
    test_dataset = AGNewsDataset("text_data/test.csv")

    # 2. Init the tokenizer
    tokenizer = get_tokenizer("basic_english")

    # 3. Init the vocabulary
    vocab = create_vocab(train_dataset.texts, tokenizer, 5)

    # 4. Create the label mapping
    labels_mapping = create_labels_mapping(train_dataset.labels)

    # 5. Construct the model
    embedding_dim = 100
    hidden_dim = 50
    model = RNNTextClassifier(
        vocab_len=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        labels_len=len(labels_mapping)
    )

    # 6. Choose the loss function
    criterion = nn.CrossEntropyLoss()

    # 7. Choose the optimizer
    lr = 1e-3
    optimizer = Adam(
        params=model.parameters(),
        lr=lr,
    )

    # 8. Construct the dataloader
    batch_size = 64
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_func
    )

    # dev and test loader don't need to be shuffle
    dev_loader = DataLoader(
        dataset=dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_func
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_func
    )