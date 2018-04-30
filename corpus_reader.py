import os

import spacy
import pickle

nlp = spacy.load('en')


class ScienceIECorpus:
    def __init__(self):
        self.docs = []

    def add_files(self, path):
        i = 0
        for file in list(os.listdir(path)):
            if file.endswith(".txt"):
                self.docs.append(ScienceFile(os.path.join(path, file)))


class ScienceFile:
    def __init__(self, filename):
        t_file = open(filename, "rU", encoding='utf-8')
        a_file = open(filename.replace(".txt",".ann"), "rU", encoding='utf-8')
        self.filename = filename
        self.text = nlp(t_file.readline().strip())
        # self.sentences = list(self.text.sents)
        self.words = [Token(word=token.text, pos=token.tag_, lemma=token.lemma,
                            tags={"Process": "O", "Task": "O", "Material": "O"}) for token in self.text]
        #self.entities is a list of named entities in the document in the form [(ID, tag, (start, end), text), ...]
        self.entities = []
        #self.relations is a list of relations in the documents in the form [(ID, Rel, (Arg 1:E1, Arg2:E2)), ...]
        self.relations = []

        for line in a_file:
            line = line.strip().split("\t")
            e_id = line[0]

            if e_id.startswith("T"):
                e_text = line[2]

                if len(line) == 3:
                    entity1 = line[1].split(" ")
                    if len(entity1) == 3:
                        e_tag, start, end = entity1
                    else:
                        e_tag, start, _, end = entity1

                start = int(start)
                end = int(end)
                entity = Entity(e_id=e_id,e_tag=e_tag,start=start,end=end,e_text=e_text)
                self.entities.append(entity)

                prev_text = nlp("".join([str(self.text)[c] for c in range(start)]))
                ne_text = [token.text for token in nlp(e_text.strip())]

                word_idx = len([token.text for token in prev_text])
                if len(ne_text) > 1:
                    for i in range(len(ne_text)):
                        #assign word tag if it's an named entity
                        if i == 0:
                            self.words[word_idx+i].tags[e_tag] = "B"
                        elif i == len(ne_text) - 1:
                            try:
                                self.words[word_idx+i].tags[e_tag] = "L"
                            except IndexError:
                                #because '30kV.' didn't tokenize correctly - got '30kV.' instead of '30kV, .'
                                self.words[word_idx + i - 1].tags[e_tag] = "L"
                        else:
                            self.words[word_idx+i].tags[e_tag] = "I"
                else:
                    self.words[word_idx].tags[e_tag] = "U"

            elif e_id.startswith("R"):
                rel_info = line[1].split(" ")
                #[(ID, Rel, (Arg1, Arg2)), ...]
                relation = Relation(id=line[0], reltype=rel_info[0],
                                    arg1=rel_info[1][5:], arg2=rel_info[2][5:])
                self.relations.append(relation)
            else:
                #[(ID, Rel, (Arg1, Arg2)), ...]
                rel_info = line[1].split(" ")
                relation = Relation(id=line[0], reltype=rel_info[0],
                                    arg1=rel_info[1], arg2=rel_info[2])
                self.relations.append(relation)


class Token:
    def __init__(self, word, pos, lemma, tags):
        self.word = word
        self.pos = pos
        self.lemma = lemma
        self.tags = tags


class Entity:
    def __init__(self, e_id, e_tag, start, end, e_text):
        self.e_id = e_id
        self.e_tag = e_tag
        self.start = start
        self.end = end
        self.e_text = e_text


class Relation:
    def __init__(self, id, reltype, arg1, arg2):
        self.id = id
        self.reltype = reltype
        self.arg1 = arg1
        self.arg2 = arg2


if __name__ == '__main__':

    training_files = './scienceie2017_train/train2/'
    dev_files = './scienceie2017_dev/dev/'
    test_files = './semeval_articles_test/semeval_articles_test'

    print('Loading training files...')
    train_corpus = ScienceIECorpus()
    train_corpus.add_files(training_files)
    train_sents = [doc.words for doc in train_corpus.docs]
    with open('training-data-2.pkl', 'wb') as train_file:
        pickle.dump(train_sents, train_file)

    print('Loading dev files...')
    dev_corpus = ScienceIECorpus()
    dev_corpus.add_files(dev_files)
    dev_sents = [doc.words for doc in dev_corpus.docs]
    with open('dev-data-2.pkl', 'wb') as dev_file:
        pickle.dump(dev_sents, dev_file)

    print('Loading test files...')
    test_corpus = ScienceIECorpus()
    test_corpus.add_files(test_files)
    test_sents = [doc.words for doc in test_corpus.docs]
    with open('test-data-2.pkl', 'wb') as test_file:
        pickle.dump(test_sents, test_file)
