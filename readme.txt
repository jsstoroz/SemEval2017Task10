Jennifer Storozum
Information Extraction Final Project
SemEval 2017 ScienceIE task (Task 10)

===TO RUN===
Make sure all pickle and json auxiliary files are in the same directory
readGO.json
training-data-2.pkl
dev-data-2.pkl
test-data-2.pkl
vocab.pkl

$ python ner-crf3.py 

===File List===
File List:
readGO.json
training-data-2.pkl
dev-data-2.pkl
test-data-2.pkl
vocab.pkl
ner-crf3.py
vocab.py
corpus_reader.py
readont.py
ontlist.txt
readme.txt
scienceie2017_dev.zip
scienceie2017_train.zip
semeval_articles_test.zip

===Presentation Slides===

SemEval 10_ Science IE.pdf


===Task Description===
Extracting Keyphrases and 
Relations from Scientific Texts

Subtask (B): Classification of identified keyphrases

In this task, each keyphrase needs to be labelled by one of three types: (i) PROCESS, (ii) TASK, and (iii) MATERIAL.
PROCESS
Keyphrases relating to some scientific model, algorithm or process should be labelled by PROCESS.
TASK
Keyphrases those denote the application, end goal, problem, task should be labelled by TASK.
MATERIAL
MATERIAL keyphrases identify the resources used in the paper.

The data consists of:
* .ann files: standoff annotation files, each line represents an annotation. Format: ID<tab>label<space>start-offset<space>end-offset<tab>surface-form
The offsets represent character offsets based on the .txt files. Note that the evaluation script ignores the IDs and the surface forms and only judges based on the character offsets.
* .txt files: text corresponding to the standoff annotation files
* .xml files: full publications from ScienceDirect in .xml format. Note that the text contained in the .txt files are paragraphs from the .xml files. These files are *not needed* for participating in the challenge. They are included because some teams might want to use them as additional background information. 

## References:
* SemEval task: https://scienceie.github.io/
* .ann format: http://brat.nlplab.org/standoff.html

===Results===
Dev Data:

See presentation slides

Test Data: 

Material F1 .33
Process F1 .31
Task F1 .12