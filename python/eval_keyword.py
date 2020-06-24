import nltk
import csv

f = open("../csv/keywordlist.csv", 'r', encoding='utf-8')
rdr = csv.reader(f)
i = 0
tag_counter = {'LOC' : 0, 'MISC' : 0, 'NP' : 0, 'ORG' : 0, 'PER' : 0}
grade_cnt = 0
length = [0] * 11
length_C = [0] * 11
for line in rdr:
    if i == 0:
        i += 1
        continue
    tag_counter[line[1]] += 1
    if line[2] != '':
        grade_cnt += 1
    l_index = len(line[0])//5
    length[l_index] += 1
    if line[0].isupper():
        length_C[l_index] += 1
    i += 1

length_L = [0] * 11
for i in range(11):
    length_L[i] = length[i] - length_C[i]

print(tag_counter)
print(grade_cnt)
print(length)
print(length_C)
print(length_L)
print('total : {}'.format(i))

f.close()