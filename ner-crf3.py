import json
import pickle
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import spacy

nlp = spacy.load('en', disable=['ner'])

TRAIN_FILE = "training-data-2.pkl"
DEV_FILE = "dev-data-2.pkl"
TEST_FILE = "test-data-2.pkl"

with open("readGO.json", "r") as f:
    GO_SET = json.load(f)
with open("vocab.pkl", "rb") as f:
    VOCAB = pickle.load(f)


def isInGO(word):
    return word in GO_SET


def isHapax(word):
    return VOCAB[word] == 1


def contains_non_alnum(word):
    for char in word:
        if not char.isalnum():
            return True
    return False


class CRF:
    def word2features(self, sent, i):
        word = sent[i].word
        postag = sent[i].pos
        # lemma = sent[i].lemma

        features = {
            'bias': 1.0,

            'word[-4:]': word[-4:],
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word[:2]': word[:2],
            'word[:3]': word[:3],
            'word[:4]': word[:4],
            # 'wordlen': len(word),
            'word.containsnonalnum': contains_non_alnum(word),
            'word.isnumber': word.isdigit(),
            'word.isalpha()': word.isalpha(),
            'word.isalnum()': word.isalnum(),
            'word.lower()': word.lower(),
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'postag': postag,
            # 'postag[:2]': postag[:2],
            # 'isLong': len(word) > 6
            'isHapax': isHapax(word),
            'inGO': isInGO(word),
            'word.len>=2': len(word) >= 2,
            'word.len>=3': len(word) >= 3,
            'word.len>=4': len(word) >= 4
            # 'lemma': lemma
        }

        if i > 0:
            word1 = sent[i - 1].word
            postag1 = sent[i - 1].pos
            # lemma1 = sent[i-1].lemma

            features.update({
                '-1word[-4:]': word1[-4:],
                '-1word[-3:]': word1[-3:],
                '-1word[-2:]': word1[-2:],
                '-1word[:2]': word1[:2],
                '-1word[:3]': word1[:3],
                '-1word[:4]': word1[:4],
                # '-1wordlen': len(word1),
                '-1word.containsnonalnum': contains_non_alnum(word1),
                '-1word.isalnum()': word1.isalnum(),
                '-1word.isnumber': word1.isdigit(),
                '-1word.isalpha()': word1.isalpha(),
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                # '-1:postag[:2]': postag1[:2],
                '-1word.len>=2': len(word1) >= 2,
                '-1word.len>=3': len(word1) >= 3,
                '-1word.len>=4': len(word1) >= 4,
                # '-1isLong': len(word1) > 6
                '-1isHapax': isHapax(word1),
                '-1inGO': isInGO(word1),
                # '-1lemma': lemma1
            })
        else:
            features['BOD'] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1].word
            postag1 = sent[i + 1].pos
            # lemma1 = sent[i+1].lemma

            features.update({
                '+1word[-4:]': word1[-4:],
                '+1word[-3:]': word1[-3:],
                '+1word[-2:]': word1[-2:],
                '+1word[:2]': word1[:2],
                '+1word[:3]': word1[:3],
                '+1word[:4]': word1[:4],
                # '+1wordlen': len(word1),
                '+1word.containsnonalnum': contains_non_alnum(word1),
                '+1word.isalnum()': word1.isalnum(),
                '+1word.isnumber': word1.isdigit(),
                '+1word.isalpha()': word1.isalpha(),
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                # '+1:postag[:2]': postag1[:2],
                '+1word.len>=2': len(word1) >= 2,
                '+1word.len>=3': len(word1) >= 3,
                '+1word.len>=4': len(word1) >= 4,
                # '+1isLong': len(word1) > 6
                '+1isHapax': isHapax(word1),
                '+1inGO': isInGO(word1),
                # '+1lemma': lemma1
            })
        else:
            features['EOD'] = True

        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent, entity_type):
        tags = [token.tags[entity_type] for token in sent]
        return tags

    def sent2tokens(self, sent):
        return [token.word for token in sent]

    def train(self, entity_type, train=TRAIN_FILE, test=DEV_FILE):
        print("Training " + entity_type + "CRF...")
        train_file = open(train, "rb")
        test_file = open(test, "rb")

        train_sents = pickle.load(train_file)
        X_train = [self.sent2features(sent) for sent in train_sents]
        y_train = [self.sent2labels(sent, entity_type) for sent in train_sents]

        test_sents = pickle.load(test_file)
        X_test = [self.sent2features(sent) for sent in test_sents]
        y_test = [self.sent2labels(sent, entity_type) for sent in test_sents]

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

        crf.fit(X_train, y_train)

        labels = list(crf.classes_)
        labels.remove('O')
        # print(labels)

        y_pred = crf.predict(X_test)
        print("F1 Score:")
        print(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))

        # group B, I, L, U results
        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )

        print(metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, digits=3
        ))

if __name__ == "__main__":

    ENTITY_TYPES = ["Material", "Process", "Task"]
    CRF = CRF()

    for entity in ENTITY_TYPES:
        CRF.train(entity, test=TEST_FILE)
