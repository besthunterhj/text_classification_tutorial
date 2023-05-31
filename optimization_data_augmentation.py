from typing import Set, Dict, List, Tuple

import en_core_web_lg
from torchtext.data import get_tokenizer
from dataset.ag_news_dataset import AGNewsDataset
from augmenty.token.replace import create_wordnet_synonym_augmenter_v1
from augmentation_genetic_algorithm import roulette_wheel_selection, crossover, mutation


# divide the training dataset according to the labels
def divide_training_dataset(training_dataset: AGNewsDataset) -> Tuple[Dict, List]:

    """
    return divided_dataset -> {'label1': [...], 'label2': [...], ..., }
    """

    label_set = set(training_dataset.labels)
    divided_dataset = {}

    # init the keys of "divided_dataset"
    for item in label_set:
        divided_dataset[item] = []

    # fill the values according to the news from training data
    for item in training_dataset:

        # judge if the label of current sample exists in "label_set"
        try:
            # append to the current label list from "divided_dataset"
            divided_dataset[item[-1]].append(item[0])

        except KeyError:
            print("current label doesn't exist in the label set")
            raise KeyError

    return divided_dataset, list(label_set)


if __name__ == '__main__':

    training_dataset = AGNewsDataset(
        fp="./text_data/train.csv"
    )

    tokenizer = get_tokenizer("basic_english")
    english_synonym_augmenter = create_wordnet_synonym_augmenter_v1(
        level=0.3,
        lang='en'
    )
    spacy_tool = en_core_web_lg.load()

    # select the data examples according to the labels (selection algorithm)
    divided_dataset, label_set = divide_training_dataset(training_dataset=training_dataset)

    # !!! only one of the type , remain 3 types data
    for i in range(len(divided_dataset.keys())):

        # the process of genetic algorithm
        selected_samples = roulette_wheel_selection(population=divided_dataset[label_set[i]], selection_num=20000)
        crossover_samples = crossover(selected_samples=selected_samples, crossover_prob=0.99, tokenizer=tokenizer)
        mutation_samples = mutation(
            crossovered_samples=crossover_samples,
            mutation_prob=0.1,
            augmenter=english_synonym_augmenter,
            spacy_obj=spacy_tool
        )

        # write the generated data to external file
        if '/' in label_set[i]:
            file_name = f'{label_set[i][:3]}_augmentation.csv'
        else:
            file_name = f'{label_set[i]}_augmentation.csv'
        with open(file_name, 'w', encoding='utf-8') as file:
            for item in mutation_samples:
                # current_news_text = '"' + item + '"'
                file.write(label_set[i])
                file.write(',')
                file.write(item)
                file.write('\n')


    print("Done!")