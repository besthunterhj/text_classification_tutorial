import csv
import re
import torch
import numpy
from torchtext.vocab import vocab as vc
from torchtext.data import get_tokenizer
from torch import nn
from torch.utils.data import Dataset


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


# def create_vocab():
    


if __name__ == "__main__":
    input()
# train_dataset = AGNewsDataset("text_data/train.csv")
# valid_dataset = AGNewsDataset("text_data/dev.csv")
# test_dataset = AGNewsDataset("text_data/test.csv")

# 2. Init the tokenizer
tokenizer = get_tokenizer("basic_english")

# 3. Init the vocabulary


