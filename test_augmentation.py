import augmenty
import en_core_web_lg
from augmenty.token.replace import create_wordnet_synonym_augmenter_v1
from nltk.corpus import wordnet


if __name__ == '__main__':

    # print(wordnet.synsets('car'))
    # exit()

    nlp = en_core_web_lg.load()
    docs = nlp.pipe(
        [
            "Augmentation is a wonderful tool for obtaining higher performance on limited data.",
            "You can also use it to see how robust your model is to changes.",
            "Legendary bandit buried in India, 'The funeral of India's most notorious bandit, Veerappan, takes place  at a village in southern Tamil Nadu state.'"
        ]
    )

    english_synonym_augmenter = create_wordnet_synonym_augmenter_v1(
        level=0.5,
        lang='en'
    )
    # exit()

    augmented_docs = augmenty.docs(docs, augmenter=english_synonym_augmenter, nlp=nlp)

    for doc in augmented_docs:
        print(doc)

    # augmenters = augmenty.augmenters()
    # help(augmenters["char_replace_random_v1"])