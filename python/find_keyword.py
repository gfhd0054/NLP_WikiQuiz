import csv
import nltk
import re

from nltk import Tree
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.chunk import RegexpParser
from nltk.stem.wordnet import WordNetLemmatizer

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

def loadFile(filename):
    f = open(filename, "r", encoding="utf-8")
    rdr = csv.reader(f)
    loadedTopic = []
    loadedSent = []
    for i, line in enumerate(rdr):
        if i == 0:
            continue
        loadedTopic.append(line[1].lower())
        loadedSent.append(line[2])
    return loadedTopic, loadedSent

def findNER(sent):
    ner = predictor.predict(
        sentence = sent
    )
    nerWord = ner["words"]
    nerTag = ner["tags"]
    nerDict = {}
    WORD = ""
    TAG = ""
    for w, t in zip(nerWord, nerTag):
        if t != "O":
            if t[2:] == TAG:
                WORD = "{} {}".format(WORD, w)
            elif len(WORD) > 0:
                nerDict[WORD] = TAG
            else:
                WORD = w
                TAG = t[2:]
        else:
            if len(WORD) > 0:
                nerDict[WORD] = TAG
                WORD = ""
                TAG = ""
            else:
                continue
    return nerDict

def makeNERList(sentElem):
    tokens = sent_tokenize(sentElem)
    nerDict = {}
    for token in tokens:
        nerDict.update(findNER(token))
    return nerDict

def lemmatize(pair):
    lemmatizer = WordNetLemmatizer()
    noun = ["NN", "NNS", "NNP", "NNPS"]
    verb = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    adjective = ["JJ", "JJR", "JJS"]
    if pair[1] in noun:
        return lemmatizer.lemmatize(pair[0], pos="n")
    elif pair[1] in verb:
        return lemmatizer.lemmatize(pair[0], pos="v")
    elif pair[1] in adjective:
        return lemmatizer.lemmatize(pair[0], pos="a")
    else:
        return pair

def extractSemantic(sent):
    stopWords = set(stopwords.words('english'))
    context = word_tokenize(sent)
    tag = pos_tag(context)
    context = [(w.lower(), p) for w, p in tag if w not in stopWords and w.isalpha()]
    context = [lemmatize(item) for item in context]
    context = [w.lower() for w in context if type(w) == str]
    fdist = nltk.FreqDist(w for w in context)
    return list(fdist.keys())[:10]

loadedTopic, loadedSent = loadFile("input.csv")
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")

def flatten(inputlist):
    outputList = []
    for l in inputlist:
        if type(l) == list:
            outputList += l
        else:
            outputList.append(l)
    return outputList

def findNP(sent):
    grammar = r"""
    NP: { <DT>? <JJ>* <NN.*>+ }
    XNP: { <CC|,> <NP> }
    CNP: { <NP> <XNP>+ }
    """
    chunk_parse = RegexpParser(grammar)
    token = word_tokenize(sent)
    postag = pos_tag(token)
    chunk = chunk_parse.parse(postag)
    leaves = [subtree.leaves() for subtree in chunk if type(subtree) == Tree and subtree.label() in ["NP", "CNP"]]
    output = []
    for leaf in leaves:
        np = ""
        for w, t in leaf:
            if len(np) > 0:
                np = "{} {}".format(np, w)
            else:
                np = w
        output.append(np)
    return output

output = open("output_temp.csv", "w", encoding="utf-8")
wr = csv.writer(output)
for i, sent in enumerate(loadedSent):
    print (i+1)
    nerList = list(makeNERList(sent).items())
    nerKeys = [w for w, t in nerList]
    sentList = sent_tokenize(sent)
    np = []
    for s in sentList:
        leaves = findNP(s)
        print (leaves)
        np += leaves
    wr.writerow([i+1, np, nerList, sent])
