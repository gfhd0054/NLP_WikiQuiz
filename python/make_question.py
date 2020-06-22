# Interim Report Version
# TODO: Exclude parenthesized phrases

from nltk.tag import pos_tag
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import nltk

from sentence_keyword_set import input_list

import wikipedia

from operator import itemgetter

be_verbs = ["am", "is", "are", "was", "were"]
auxiliary_verbs = ["do", "does", "did", "have", "had"]
modal_verbs = ["can", "must", "will", "may", "should", "might"]

time_words = ["second", "minute", "hour",
              "day", "month", "year", "time", "date"]
place_words = ["room", "building", "place"]

# random_page = wikipedia.random(pages=1)
# random_page = wikipedia.random(pages=1); print("Target page: " + random_page); wikipedia.summary(random_page, sentences=1)

# print("Categories: " + str(wikipedia.page(random_page).categories))
# print(pos_tag(word_tokenize(wikipedia.summary(random_page, sentences=1))))


def noun_to_interrogative(word, subject=False):
    '''
    Return an interrogative word to represent a noun.
    :param word: Keyword to be asked
    :type word: string
    :param subject: Set true if the keyword is the subject of original sentence
    :type subject: boolean
    :return: Interrogative corresponding to param word
    :rtype: string
    '''
    if len(word_tokenize(word)) > 1:
        word = word_tokenize(word)[-1]
    if word in nltk.corpus.names.words():
        return "Who"
    if subject:
        return "What"
    for synset in wn.synsets(word):
        for time_word in time_words:
            if time_word in synset.definition():
                return "When"
        for place_word in place_words:
            if place_word in synset.definition():
                return "Where"
    return "What"


def make_question(sentence, keyword):
    '''
    Create a sentence from the original sentence, asking for the keyword as its
    answer

    The model assumed that the target sentence is in a simple structure, not a 
    combination of multiple sentences.

    :param sentence: A declarative sentence to be converted as a question.
    :type sentence: string
    :param keyword: Keyword of param sentence.
    :type keyword: string
    :return: Interrogative sentence converted from param sentence.
    :rtype: string
    '''
    tokenized = word_tokenize(sentence)
    if len(word_tokenize(keyword)) > 1:
        word_index = tokenized.index(word_tokenize(keyword)[0])
        for part in word_tokenize(keyword):
            tokenized.remove(part)
        tokenized.insert(word_index, keyword)

    upenn_set = pos_tag(tokenized)
    universal_set = pos_tag(tokenized, tagset='universal')
    upenn_pos = list(map(itemgetter(1), upenn_set))
    universal_pos = list(map(itemgetter(1), universal_set))
    pos_set = list(
        map((lambda x, y, z: [x, y, z]), tokenized, upenn_pos, universal_pos))
    wnl = WordNetLemmatizer()

    verb_index = universal_pos.index("VERB")

    if pos_set[tokenized.index(keyword)][2] == "NOUN":
        # lower the first letter if it is not a proper noun
        if tokenized[0] != keyword and pos_set[0][1] != "NNP" and pos_set[0][1] != "NNPS":
            tokenized[0] = tokenized[0].lower()

        if tokenized.index(keyword) < verb_index: # Keyword is the subject of the sentence.
            interrogative = noun_to_interrogative(keyword, subject=True)
        else: # Examine the verb and rearrange the sentence accordingly to make a question.
            interrogative = noun_to_interrogative(keyword)
            if (pos_set[verb_index][1] == "MD" or pos_set[verb_index][0] in be_verbs) and pos_set[verb_index + 1][2] == "VERB":
                tokenized.insert(0, tokenized[verb_index])
                del tokenized[verb_index + 1]
                pos_set.insert(0, pos_set[verb_index])
                del pos_set[verb_index + 1]
            elif tokenized[verb_index] in be_verbs:
                tokenized.insert(0, tokenized[verb_index])
                pos_set.insert(0, pos_set[verb_index])
                del tokenized[verb_index + 1]
                del pos_set[verb_index + 1]
            elif pos_set[verb_index][1] == "VBD":
                lem_verb = wnl.lemmatize(tokenized[verb_index], "v")
                tokenized[verb_index] = lem_verb
                pos_set[verb_index] = [lem_verb, "VB", "VERB"]
                tokenized.insert(0, "did")
                pos_set.insert(0, ["did", "MD", "AUX"])
            elif pos_set[verb_index][1] == "VBZ":
                lem_verb = wnl.lemmatize(tokenized[verb_index], "v")
                tokenized[verb_index] = lem_verb
                pos_set[verb_index] = [lem_verb, "VB", "VERB"]
                tokenized.insert(0, "does")
                pos_set.insert(0, ["does", "MD", "AUX"])
            else:
                tokenized.insert(0, "do")
                pos_set.insert(0, ["do", "MD", "AUX"])

        # Remove the keyword and any words decorating the keyword.
        for i in reversed(range(tokenized.index(keyword))):
            if pos_set[i][2] == "DET" or pos_set[i][2] == "ADP" or pos_set[i][1] == "PRP$":
                del tokenized[i]
                del pos_set[i]
            else:
                break
        tokenized.remove(keyword)
        tokenized.insert(0, interrogative)
        return " ".join(tokenized)[:-2] + "?"


for sentence, keyword in input_list:
    print(make_question(sentence, keyword))
# print(make_question('Enrique Colla was an Argentine football player.', 'Enrique Colla'))
