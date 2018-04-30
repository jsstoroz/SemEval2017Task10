import json
import spacy
from collections import defaultdict

nlp = spacy.load('en', disable=['ner'])

def readGO(filename="ontlist.txt"):
    go_words = defaultdict(int)
    with open(filename, 'r') as f:
        for line in f:
            words = [token.text for token in nlp(line)]
            for word in words:
                go_words[str(word)] += 1
    return go_words


if __name__ == '__main__':
    print('Reading ontology...')

    go_set = readGO()
    with open('readGO.json', 'w') as ont_file:
        json.dump(go_set, ont_file)
