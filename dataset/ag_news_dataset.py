import re
import csv
from typing import List, Tuple
from collections.abc import Callable

import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab


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
    def read(cls, fp: str) -> Tuple[list, list]:
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
                news_category = line[0]
                text = ""
                for i in range(1, len(line)):
                    if i > 1:
                        text = text + " "
                    text = text + line[i]
                # news_category, news_title, news_body = line
                # news_category, text = line

                # build the data_x
                # text = news_title + " " + news_body

                # remove the meaningless char
                text = re.sub("&[A-Za-z]{1,2};", "", text)
                text = re.sub(" #[0-9]{2};", "", text)
                text = re.sub("[A-Za-z]{1,10}=[A-Za-z0-9]{1,10} ", "", text)
                text = re.sub("[A-Za-z]{1,10}=\"{1, 2}[\\w]\"{1,2}", "", text)
                text = re.sub("[A-Za-z]{1,10}=[\"]{1,2}http[s]{0,1}://([\w.]+/?)\S*", "", text)
                text = re.sub("target=[\s\w/]{1,50}\"\"", "", text)

                text_data.append(text)

                # build the data_y
                labels.append(news_category)

        return text_data, labels

        # with open(fp) as file:
        #     csv_reader = csv.reader(file)
        #     for line in csv_reader:
        #         news_category, news_title, news_body = line
        #
        #         # build the data_x
        #         text = news_title + " " + news_body
        #
        #         # remove the meaningless char
        #         text = re.sub("&[A-Za-z]{1,2};", "", text)
        #         text = re.sub(" #[0-9]{2};", "", text)
        #         text = re.sub("[A-Za-z]{1,10}=[A-Za-z0-9]{1,10} ", "", text)
        #         text = re.sub("[A-Za-z]{1,10}=\"{1, 2}[\\w]\"{1,2}", "", text)
        #         text = re.sub("[A-Za-z]{1,10}=[\"]{1,2}http[s]{0,1}://([\w.]+/?)\S*", "", text)
        #         text = re.sub("target=[\s\w/]{1,50}\"\"", "", text)
        #
        #         text_data.append(text)
        #
        #         # build the data_y
        #         labels.append(news_category)
        #
        # return text_data, labels


def pad(token_indexes: List[int], max_len: int, default_padding_val: int = 0) -> List[int]:
    """
    Normalize the length of current list of token_indexes, so it can change to part of tensor
    :param default_padding_val: the default index of the character <pad>
    :param token_indexes: the indexes for tokens of current text
    :param max_len: the maximum length of normalization
    :return: List[int], a sequence which be normalized for its length
    """

    if len(token_indexes) > max_len:
        return token_indexes[:max_len]

    else:
        padded_token_indexes = token_indexes.copy()
        for i in range(max_len - len(token_indexes)):
            padded_token_indexes.append(default_padding_val)

        return padded_token_indexes


def collate_func(samples: List[Tuple[str, str]], tokenizer: Callable, vocab: Vocab, labels_mapping: dict,
                 max_len: int) -> dict:
    """
     zip(*parameter): the "parameter" must be a list of tuples, and this function is separate it to two list which
     consists of the parameter[i][0] and parameter[i][-1]
    """
    texts, labels = list(zip(*samples))

    texts = torch.tensor(list(
        map(
            lambda current_text: pad(vocab(tokenizer(current_text)), max_len=max_len),
            texts
        )
    ))

    labels = torch.tensor(list(
        map(
            lambda current_label: labels_mapping[current_label],
            labels
        )
    ))

    # Recommend to return the dictionary
    return {
        "texts": texts,
        "labels": labels
    }
