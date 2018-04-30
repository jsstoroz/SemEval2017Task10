from collections import Counter
import os
import pickle
import spacy

nlp = spacy.load('en', disable=['ner'])

training_files = './scienceie2017_train/train2/'


def flatten(l):
    return [item for sublist in l for item in sublist]


def add_files(path):
    all_text = list()
    for file in list(os.listdir(path)):
        if file.endswith(".txt"):
            t_file = open(os.path.join(path, file), "rU", encoding='utf-8')
            text = [token.text for token in nlp(t_file.readline().strip())]
            all_text.append(text)
    return all_text

all_doc_text = flatten(add_files(training_files))

VOCAB = Counter(all_doc_text)
with open('vocab.pkl', 'wb') as f:
    pickle.dump(VOCAB, f)