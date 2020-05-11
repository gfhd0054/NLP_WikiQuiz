import wikipedia as wiki
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree


def diet(sent):
    '''
    Do diet for give sent. If the sentence has blanks at the both end, trim it.
    :param sent: String before diet
    :return: String after diet
    '''
    start = 0
    end = len(sent)
    for i in range(len(sent)):
        if sent[i] != ' ':
            break
        else:
            start = i + 1
    for i in range(len(sent)-1, -1, -1):
        if sent[i] != ' ':
            break
        else:
            end = i
    if end <= start:
        return ""
    else:
        return sent[start:end]


def tokenize(sent, token = ' '):
    '''
    Tokenize function for user-defined token
    :param sent: String
    :param token: mainly character, or string
    :return: list of tokens tokenized by given token.
    '''
    t = []
    pos = 0
    for i in range(len(sent)):
        if sent[i] == token:
            t.append(sent[pos:i])
            pos = i + 1
    t.append(sent[pos:])
    return t


def tree_to_str(elem):
    if type(elem) == Tree:
        words = [tree_to_str(word) for word in elem]
        return " ".join(words)
    else:
        return elem[0]


def tree_to_list(tree):
    '''
    Convert tree of phrases to list
    :param tree: nltk.tree.Tree
    :return: list of phrases
    '''
    chunks = []
    if tree.label() == 'S':
        for i in tree:
            if type(i) == Tree:
                phrase = tree_to_str(i)
                chunks.append((phrase, i.label()))
            else:
                chunks.append(i)
    return chunks


def get_S(sent):
    '''
    Get main subject of the sentence
    :param sent: raw sentence
    :return: list of phrases, consisting of main subject
    '''
    words = chunk_sent(sent)
    for i in range(len(words)):
        if words[i][1] == 'VP':
            return words[:i]
    return []


def get_V(sent):
    '''
    Get main verb of the sentence
    :param sent: raw sentence
    :return: list of phrases, consisting of main verb
    '''
    words = chunk_sent(sent)
    for i in range(len(words)):
        if words[i][1] == 'VP':
            return words[i]
    return []


def get_O(sent):
    '''
    Get main subject of the sentence
    :param sent: raw sentence
    :return: list of phrases, consisting of object, or whole explanation.
    '''
    words = chunk_sent(sent)
    for i in range(len(words)):
        if words[i][1] == 'VP':
            return words[i+1:]
    return []


def get_lined_ups(sent):
    '''
    Check whether the sentence is containing lined elements.
    :param phrs: list of phrases and conjuction, prepositions, and punctuations.
    :return: List of lined up elements. If don't exist, return empty list
    '''
    elements = " ".join([phr[0] for phr in get_O(sent)])
    # Combine the prepositions
    grammar1 = r"""
        INNP : {<IN|TO><NP>}
        NP : {<NP><INNP>+}
    """
    elements = chunk_sent(elements)
    # parser
    cp1 = nltk.RegexpParser(grammar1)
    t = cp1.parse(elements)
    template = tree_to_list(t)
    # Check for lined up
    grammar2 = r"""
        ELEM : {<NP><,>}
        ARRAY : {<ELEM>*<ELEM|NP><CC><NP>}
    """
    cp2 = nltk.RegexpParser(grammar2)
    t = cp2.parse(template)
    tokens = []
    for i in tree_to_list(t):
        if i[1] == 'ARRAY':
            chunk = chunk_sent(i[0])
            t = cp1.parse(chunk)
            for j in tree_to_list(t):
                if j[1] == 'NP':
                    tokens.append(j[0])
    if len(tokens) == 0:
        return []
    return [diet(token) for token in tokens]


def subsent_tokenize(sent):
    '''
    Tokenize the sentence according to comma, only if the token is not just arrangement.
    :param sent: raw sentence
    :return: list of tokenized sentences
    '''
    chunks = tokenize(sent, token=',')
    result = []
    tmpl = []
    for chunk in chunks:
        tag = tag_subsent(chunk)
        if len(tmpl) == 0:
            tmpl.append(chunk)
            continue
        if tag == 'Simp':
            tmpl.append(chunk)
        else:
            result.append(", ".join(tmpl))
            tmpl = []
    if len(tmpl) != 0:
        result.append(", ".join(tmpl))
    return result


def tag_subsent(sub):
    '''
    According to the semantics of sub, tag it whether it is conjunction phrase, sentence, or simple word.
    :param sub: sentence(potential) tokenized by ','.
    :return: type of the subsentence. (Conj, Sent, Simp)
    '''
    chunk = chunk_sent(sub)
    # Glue NP
    grammar1 = r"""
            INNP : {<IN|TO><NP>}
                   {<INNP>*}
            NP : {<NP><INNP>}
            SENT : {<NP><VP>}
            SIMP : {<CC><NP>}
        """
    # parser
    cp1 = nltk.RegexpParser(grammar1)
    t = cp1.parse(chunk)
    template = tree_to_list(t)
    if len(template) == 1 and template[0][1] == 'SIMP':
        return 'Simp'
    for temp in template:
        if temp[1] == 'SENT':
            return 'Sent'
    if template[0][1] == 'INNP':
        return 'Conj'


def chunk_sent(sent):
    '''
    Segregate chunks of the sentence
    :param sent: list of tagged words
    :return: list of tagged chunks
    '''
    # Define chunk grammar
    grammar = r"""
        NP : {<DT|PP\$>?<JJ|VBG|VBN>*<N.*>+}
             {<NNP>+}
        VP : {<V.*><TO><VB>}
             {<V.*><VBG>}
             {<V.*><JJ>}
             {<VB|VBD|VBP|VBZ>}
             {<MD><VP>}
    """
    sent = pos_tag(word_tokenize(sent))
    # parser
    cp = nltk.RegexpParser(grammar)
    t = cp.parse(sent)
    return tree_to_list(t)


def purify_sent(sent):
    '''
    Discard parentheses of sentence
    :param sent: raw sentence
    :return: sentence without parentheses
    '''
    # todo!
    return None


def lemmatize_sent(sent):
    '''
    Lemmatize the whole words in sentence, and return it.
    :param sent: raw POS-tagged sentence.
    :return: Lemmatized sentence.
    '''
    return [lemmatize_word(word) for word in pos_tag(word_tokenize(sent))]


def lemmatize_word(entry):
    '''
    Lemmatize the word according to its POS.
    Ex. entry = (geese, NNS) -> return (goose, NN)
    :param entry: tuple that contains word and POS.
    :return: Lemmatized entry, according to its POS.
    '''
    wnl = WordNetLemmatizer()
    nouns = ['NN', 'NNS']
    verbs = ['VB', 'VBD', 'VBP', 'VBZ', 'VBG']
    adjs = ['VBN', 'JJ', 'JJR', 'JJS', 'JJT'] # todo: How to deal with the VBN?
    advs = ['RB', 'RBR', 'RBT']
    if entry[1] in nouns:
        return wnl.lemmatize(entry[0], pos='n'), 'NN'
    elif entry[1] in verbs:
        lem = wnl.lemmatize(entry[0], pos='v')
        if lem == entry[0]:
            return entry
        else:
            return wnl.lemmatize(entry[0], pos='v'), 'VB'
    elif entry[1] in adjs:
        return wnl.lemmatize(entry[0], pos='a'), 'JJ'
    elif entry[1] in advs:
        return wnl.lemmatize(entry[0], pos='r'), 'RB'
    else:
        return entry


def main():
    sent_set = [
        "Prevention and treatment involve maintaining a healthy diet, regular physical exercise, a normal body weight, and avoiding use of tobacco.",
        "Serious long-term complications include cardiovascular disease, stroke, chronic kidney disease, foot ulcers, damage to the nerves, damage to the eyes and cognitive impairment.",
        "In addition to the known ones above, they include blurred vision, headache, fatigue, slow healing of cuts, and itchy skin.",
        "It was widely used by professional writers, in offices, and for business correspondence in private homes.",
        "Newton originally divided the spectrum into six named colors: red, orange, yellow, green, blue, and violet.",
        "The human eye is relatively insensitive to indigo's frequencies, and some people who have otherwise-good vision cannot distinguish indigo from blue and violet.",
        "Recommended measures to prevent infection include frequent hand washing, maintaining physical distance from others (especially from those with symptoms), quarantine, covering coughs, and keeping unwashed hands away from the face.",
        "Management involves the treatment of symptoms, supportive care, isolation, and experimental measures."
    ]
    debug = True

    if debug:
        for test_sent in sent_set:
            print(test_sent)
            S = " ".join([token[0] for token in get_S(test_sent)])
            print(S, " : ", end='')
            for item in get_lined_ups(test_sent):
                print(item, end=', ')
            print()
            print(subsent_tokenize(test_sent))
            print()
        return

    fileids = ["Diabetes", "Abraham Lincoln", "Typewriter", "Visible spectrum", "Coronavirus disease 2019"]
    for fileid in fileids:
        first_sent = wiki.summary(fileid, sentences=1)
        print(chunk_sent(first_sent))


main()
