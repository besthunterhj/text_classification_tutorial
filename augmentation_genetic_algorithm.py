from typing import List
# from optimization_data_augmentation import
import augmenty
import nltk
import textstat
import numpy as np

"""适配值函数由计算当前样本文本的 gunning fog index 实现"""


# 选择算子： 轮盘赌算法
def roulette_wheel_selection(population: List[str], selection_num: int) -> List[str]:
    """
    the selection algorithm for select the samples which is chosen to join in the augmentation
    :param population: the sub-training set of current label
    :param selection_num: the number of the samples that will be chosen
    :return: a list of selected samples, the number is decided by the parameter "selection_num"
    """

    # init the selected samples list
    selected_samples = []

    # compute the adapting scores (gunning fog index) of the population
    adapting_scores = []
    for item in population:
        current_score = textstat.gunning_fog(item)
        adapting_scores.append(current_score)

    # compute the probabilities of selection
    population_selection_probs = []
    adapting_scores_sum = sum(adapting_scores)
    for item in adapting_scores:
        current_prob = item / adapting_scores_sum
        population_selection_probs.append(current_prob)

    # begin to select
    for i in range(selection_num):
        # define the "m" (accumulation variable) and "r" (random number in 0-1)
        accumulation = 0
        random_num = np.random.rand()

        # accumulate the probabilities and judge the selected individual in each round
        for j in range(len(population_selection_probs)):
            accumulation += population_selection_probs[j]

            # judge whether select current training example
            # !! note: permit for the repeating sampling
            if random_num <= accumulation:
                selected_samples.append(population[j])
                break

    return selected_samples


# 交配算子
def crossover(selected_samples: List[str], crossover_prob: float, tokenizer) -> List[str]:

    """
    the crossover algorithm for constructing various pseudo training samples
    :param selected_samples: the subset of training samples that have been selected by the selection algorithm
    :param crossover_prob: the probability that decides current sample whether engages the crossover
    :param tokenizer: the tokenizer for tokenize the input training samples
    :return: a list of crossovered samples
    """

    # init a list to store the crossovered samples and original samples that do not engage the crossover
    returned_samples = []

    # get the list of crossover engaged samples
    crossover_engaged_samples = []
    for item in selected_samples:
        temp_prob = np.random.rand()

        # judge whether engages the crossover
        if temp_prob <= crossover_prob:
            crossover_engaged_samples.append(item)
        else:
            # the samples that do not engage the crossover directly flow to next algorithm
            returned_samples.append(item)

    # let the number of selected_samples and judge the type according to the remainder
    remainder = len(crossover_engaged_samples) % 2

    # init an integer "last_symbol" to indicate the (index + 1) of the last sample that join the crossover
    if remainder > 0:
        last_symbol = len(crossover_engaged_samples) - 1
    else:
        last_symbol = len(crossover_engaged_samples)

    for i in range(0, last_symbol, 2):
        current_first_sample = crossover_engaged_samples[i]
        current_second_sample = crossover_engaged_samples[i + 1]

        # tokenize
        tokenized_first_sample = tokenizer(current_first_sample)
        tokenized_second_sample = tokenizer(current_second_sample)

        # handle the difference between the lengths of sample 1 and sample 2
        shorter_length = min(len(tokenized_first_sample), len(tokenized_second_sample))

        # randomly select the crossover head
        # the first and last tokens are always punctuation, so it's better to narrow the range
        candidate_head_indices = list(range(2, shorter_length - 2))
        crossover_head = np.random.choice(candidate_head_indices)

        # begin to crossover for current two samples
        first_crossovered = tokenized_first_sample[:crossover_head] + tokenized_second_sample[crossover_head:]
        second_crossovered = tokenized_second_sample[:crossover_head] + tokenized_first_sample[crossover_head:]

        # merge each crossovered sample which has been tokenized, so that the following mutation can work
        first_crossovered_samples = " ".join(first_crossovered)
        second_crossovered_samples = " ".join(second_crossovered)

        # append the 2 crossovered samples to the list
        returned_samples.append(first_crossovered_samples.strip())
        returned_samples.append(second_crossovered_samples.strip())

    return returned_samples


# 变异算子（基于同义词替换）
def mutation(crossovered_samples: List[str], mutation_prob: float, augmenter, spacy_obj) -> List[str]:

    # init a list to store mutated samples and non-mutated samples
    after_mutation_samples = []
    # init the mutation engaged list
    mutation_engaged_samples = []

    # get the mutation engaged samples randomly
    for item in crossovered_samples:
        temp_prob = np.random.rand()

        # judge whether engages the mutation
        if temp_prob <= mutation_prob:
            mutation_engaged_samples.append(item)
        else:
            # append and return the samples that do not engage the mutation directly
            after_mutation_samples.append(item)

    # the process of mutation
    if len(mutation_engaged_samples) > 0:

        # mutation: synonym replacement
        mutated_samples = augmenty.texts(
            texts=mutation_engaged_samples,
            augmenter=augmenter,
            nlp=spacy_obj
        )

        # append the items from "mutated_samples"
        for sample in mutated_samples:
            after_mutation_samples.append(sample)

    return after_mutation_samples


if __name__ == '__main__':

    test_text1 = 'HP iPod Photo?,"So now that Apple has officially rolled out the new iPod Photo, we still have one nagging question: will HP follow suit and come out with an HP iPod Photo?"'
    test_text2 = 'Legendary bandit buried in India,"The funeral of India\'s most notorious bandit, Veerappan, takes place  at a village in southern Tamil Nadu state."'

    test_list1 = nltk.tokenize.word_tokenize(test_text1)
    test_list2 = nltk.tokenize.word_tokenize(test_text2)

    # test = test_list1[:5] + test_list2[5:]
    # test2 = test_list2[:5] + test_list1[5:]
    #
    print(test_text2)
    print(test_list2)

    sent1 = " ".join(test_list2).strip()
    print(sent1)

    # print(test_list2)
    # print(test)
    # print(test2)
    # print(test_list1)
    # print(test_list2)
    # print(list(range(1, 10)))
    exit()


    print(nltk.tokenize.word_tokenize(test_text1))