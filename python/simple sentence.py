from nltk.tag import pos_tag
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize
import nltk
import csv




#data = [6,"baseball",[('the bottom', 'NP'), ('doubleheaders', 'NP'), ('nine players', 'NP'), ('seven innings', 'NP'), ('a game', 'NP'), ('team—bats', 'NP'), ('a baseball game', 'NP'), ('an inning', 'NP'), ('turns', 'NP'), ('Little League', 'NP'), ('the field', 'NP'), ('offense', 'NP'), ('nine innings', 'NP'), ('batting and baserunning', 'NP'), ('the high school level', 'NP'), ('every inning', 'NP'), ('each team', 'NP'), ('bat', 'NP'), ('the home team—bats', 'NP'), ('college and minor leagues', 'NP'), ('a pair', 'NP'), ('two teams', 'NP')],
# "A baseball game is played between two teams, each composed of nine players, that take turns playing offense (batting and baserunning) and defense (pitching and fielding). A pair of turns, one at bat and one in the field, by each team constitutes an inning. A game consists of nine innings (seven innings at the high school level and in doubleheaders in college and minor leagues, and six innings at the Little League level). One team—customarily the visiting team—bats in the top, or first half, of every inning. The other team—customarily the home team—bats in the bottom, or second half, of every inning."]

data = [99,"euthanasia",[('intractable suffering', 'NP'), ('Netherlands', 'LOC'), ('a life', 'NP'), ('the term', 'NP'), ('the request', 'NP'), ('British House of Lords', 'ORG'), ('the practice', 'NP'), ('a doctor', 'NP'), ('the british house', 'NP'), ('a deliberate intervention', 'NP'), ('Belgium', 'LOC'), ('termination', 'NP'), ('different euthanasia laws', 'NP'), ('medical ethics defines', 'NP'), ('a patient', 'NP'), ('pain and suffering.different countries', 'NP'), ('life', 'NP'), ('definition', 'NP'), ('euthanasia', 'NP'), ('the concept', 'NP'), ('assisted suicide and termination', 'NP'), ('Dutch', 'NP'), ('the express intention', 'NP'), ('request', 'NP'), ('select committee', 'NP')],
        "Euthanasia is the practice of intentionally ending a life to relieve pain and suffering.Different countries have different euthanasia laws. The British House of Lords select committee on medical ethics defines euthanasia as ""a deliberate intervention undertaken with the express intention of ending a life, to relieve intractable suffering"". In the Netherlands and Belgium, euthanasia is understood as ""termination of life by a doctor at the request of a patient"". The Dutch law, however, does not use the term 'euthanasia' but includes the concept under the broader definition of ""assisted suicide and termination of life on request."]


def remove_this_term(data):
    result =[]
    this_terms = ['this', 'that','these','those']
    for word,tag in data[2]:
        no_this_term = True
        for t in this_terms:
            if(word.find(t)>0):
                no_this_term = False
        if(no_this_term):
            result.append((word,tag))
    data[2] = result
    return data

def remove_parenthesis(data):
    s = data[3]
    start = s.find('(')
    while (start != -1):
        end = s.find(')')
        for i in range(s[start+1:end].count('(')):
            end += s[end+1:].find(')')+1
        length = end - start + 1
        s = s[:start] + s[end+1:]
        start = s.find('(')
    data[3] = s
    s = s.lower()
    keywords =[]
    for (word, tag) in data[2]:
        if s.find(word.lower())>0:
            keywords.append((word,tag))
    data[2] = keywords
    return data

def make_simple(data):
    sentences = sent_tokenize(data[3])
    result =[]
    for s in sentences:
        divide_idx =[]
        conj =[]
        tagged = pos_tag(word_tokenize(s))
        for i, (word, tag) in enumerate(tagged):
            if tag =='CC' and word!= 'and':
                conj.append(i)
        if(len(conj)==0):
            result.append(s)
            continue
        conj.append(len(tagged)-1)
        for i in range(len(conj)-1):
            Noun_exist =False
            for (word, tag) in tagged[conj[i]:conj[i+1]]:
                if(tag[0] == 'N'):
                    Noun_exist = True
                if(tag[0] == 'V'):
                    if(Noun_exist):
                        divide_idx.append(conj[i])
                    else:
                        break
        divide_idx.append(len(tagged)-1)
        length = 0
        prev_end =0
        for i, (word, tag) in enumerate(tagged):
            if( i in divide_idx):
                result.append(s[prev_end:length])
                prev_end = length + len(word)+1
            length += len(word)
            if( word.isalpha()):
                length += 1
    return_val = []
    for s in result:
        keywords =[]
        for (word, tag) in data[2]:
            if(s.lower().find(word.lower())>0):
                keywords.append((word,tag))
        if(len(keywords)>0):
            return_val.append([data[0],data[1],keywords,s])
    return return_val


data = remove_this_term(data)
data = remove_parenthesis(data)
data = make_simple(data)
print(data)
