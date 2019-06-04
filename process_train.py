
# coding: utf-8

# %pdb on
import json
import re
import numpy as np
import copy
from tqdm import tqdm 
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from redis import StrictRedis



db = StrictRedis(host='localhost', port=6379, db=0)


with open('./hotpot_train_v1.1.json', 'r') as fin:
    train_set = json.load(fin)
print('Finish Reading! len = ', len(train_set))




# with open('./hotpot_train_v1.1_small.json', 'w') as fout:
#    json.dump(train_set[:5000], fout)




from hotpot_evaluate_v1 import normalize_answer, f1_score
from fuzzywuzzy import fuzz, process as fuzzy_process

def fuzzy_retrive(entity, pool):
    if len(pool) > 100:
        # fullwiki, exact match
        # TODO: test ``entity (annotation)'' and find the most like one
        if pool.get(entity):
            return entity
        else:
            return None
    else:
        # distractor mode or use link in original wiki, no need to consider ``entity (annotation)''
        pool = pool if isinstance(pool, list) else pool.keys()
        f1max, ret = 0, None
        for t in pool:
            f1, precision, recall = f1_score(entity, t)
            if f1 > f1max:
                f1max, ret = f1, t
        return ret

def find_near_matches(w, sentence):
    ret = []
    max_ratio = 0
    t = 0
    for word in sentence.split():
        while sentence[t] != word[0]:
            t += 1
        score = (fuzz.ratio(w, word) + fuzz.partial_ratio(w, word)) / 2
        if score > max_ratio:
            max_ratio = score
            ret = [(t, t + len(word))]
        elif score == max_ratio:
            ret.append((t, t + len(word)))
        else:
            pass
        t += len(word)
    return ret if max_ratio > 85 else []     

def dp(a, b): # a source, b long text
    f, start = np.zeros((len(a), len(b))), np.zeros((len(a), len(b)), dtype = np.int)
    for j in range(len(b)):
        f[0, j] = int(a[0] != b[j])
        if j > 0 and b[j - 1].isalnum():
            f[0, j] += 10
        start[0, j] = j
    for i in range(1, len(a)):        
        for j in range(len(b)):
            # (0, i-1) + del(i) ~ (start[j], j)
            f[i, j] = f[i - 1, j] + 1
            start[i, j] = start[i - 1, j]
            if j == 0:
                continue
            if f[i, j] > f[i - 1, j - 1] + int(a[i] != b[j]):
                f[i, j] = f[i - 1, j - 1] + int(a[i] != b[j])
                start[i, j] = start[i-1, j - 1]

            if f[i, j] > f[i, j - 1] + 0.5:
                f[i, j] = f[i, j - 1] + 0.5
                start[i, j] = start[i, j - 1]
#     print(f[len(a) - 1])
    r = np.argmin(f[len(a) - 1])
    ret = [start[len(a) - 1, r], r + 1]
#     print(b[ret[0]:ret[1]])
    score = f[len(a) - 1, r] / len(a)
    return (ret, score)

def fuzzy_find(entities, sentence):
    ret = []
    for entity in entities:
        item = re.sub(r' \(.*?\)$', '', entity).strip()
        if item == '':
            item = entity
            print(item)
        r, score = dp(item, sentence)
        if score < 0.5:
            matched = sentence[r[0]: r[1]].lower()
            final_word = item.split()[-1]
            # from end
            retry = False
            while fuzz.partial_ratio(final_word.lower(), matched) < 80:
                retry = True
                end = len(item) - len(final_word)
                while end > 0 and item[end - 1].isspace():
                    end -= 1
                if end == 0:
                    retry = False
                    score = 1
                    break
                item = item[:end]
                final_word = item.split()[-1]
            if retry:
#                 print(entity + ' ### ' + sentence[r[0]: r[1]] + ' ### ' + item)
                r, score = dp(item, sentence)
                score += 0.1

            if score >= 0.5:
#                 print(entity + ' ### ' + sentence[r[0]: r[1]] + ' ### ' + item)
                continue
            del final_word
            # from start
            retry = False
            first_word = item.split()[0]
            while fuzz.partial_ratio(first_word.lower(), matched) < 80:
                retry = True
                start = len(first_word)
                while start < len(item) and item[start].isspace():
                    start += 1
                if start == len(item):
                    retry = False
                    score = 1
                    break
                item = item[start:]
                first_word = item.split()[0]
            if retry:
#                 print(entity + ' ### ' + sentence[r[0]: r[1]] + ' ### ' + item)
                r, score = dp(item, sentence)
                score = max(score, 1 - ((r[1] - r[0]) / len(entity)))
                score += 0.1
#             if score > 0.5:
#                 print(entity + ' ### ' + sentence[r[0]: r[1]] + ' ### ' + item)
            if score < 0.5:
                if item.isdigit() and sentence[r[0]: r[1]] != item:
                    continue
                ret.append((entity, sentence[r[0]: r[1]], int(r[0]), int(r[1]), score))
    non_intersection = []
    for i in range(len(ret)):
        ok = True
        for j in range(len(ret)):
            if j != i:
                if not (ret[i][2] >= ret[j][3] or ret[j][2] >= ret[i][3]) and ret[j][4] < ret[i][4]:
                    ok = False
                    break
                if ret[i][4] > 0.2 and ret[j][4] < 0.1 and not ret[i][1][0].isupper() and len(ret[i][1].split()) <= 3:
                    ok = False
#                    print(ret[i])
                    break
        if ok:
            non_intersection.append(ret[i][:4])
    return non_intersection

# print(dp('Skiffle', 'Die Rh\u00f6ner S\u00e4uw\u00e4ntzt are a Skif, dm , fle-Bluesband from Eichenzell-L\u00fctter in Hessen, Germany.'))
# def fuzzy_find(entities, sentence):
#     items = fuzzy_process.extract(sentence, entities, scorer=fuzz.partial_token_set_ratio)
#     items = [x for x, y in items if y > 85]
#     items_matched = []
#     for item in items:
#         positions = []
#         for w in re.split('[\s,.?!]', item):
#             r = find_near_matches(w, sentence)
#             if len(r) > 0:
#                 # assume by default sorted by starts
#                 positions.append(r)
#         # To find an interval, which length is minimized
#         print(item, positions)
#         assert len(positions) > 0
#         min_len, s_min, e_min = len(sentence), -1, -1
#         while s_min < 0:
#             if len(positions) == 1:
#                 s_min, e_min = positions[0][0]
#                 break
#             for s0, e0 in positions[0]:
#                 for s_1, e_1 in positions[-1]:
#                     if s_1 <= e0:
#                         continue
#                     if e_1 - s0 >= min_len:
#                         break
#                     ok = True
# #                 last = e0
# #                 for k in range(1, len(positions) - 1):
# #                     ok = False
# #                     for s_k, e_k in positions[k]:
# #                         if last < s_k and e_k < s_1:
# #                             last = e_k
# #                             ok = True
# #                             break
# #                     if not ok:
# #                         break
#                     if ok:
#                         min_len, s_min, e_min = e_1 - s0, s0, e_1
#             if min_len > 2 * len(item): # invalid, too long
#                 positions.pop()
#                 s_min, e_min = -1, -1
#         items_matched.append(sentence[s_min: e_min])
#     return list(zip(items, items_matched))   
print(list(fuzzy_find(['Miami Gardens, Florida', 'WSCV', 'Hard Rock Stadium'], r"Hard Rock Stadium is a multipurpose football stadium located in Miami Gardens, a city north of Miami. It is the home stadium of the Miami Dolphins of the National Football League (NFL).")))
# print(fuzzy_find(["19 Kids and Counting", "nine girls and 10 boys"], r" A spin-off show of \"19 kids ande counting\", it features the Duggar family: Jill Dillard, Jessa Seewald, sixteen of their seventeen siblings, and parents Jim Bob and Michelle Duggar."))
# print(fuzzy_retrive('Joshua Aaron Charles', ['Jawahar Navodaya Vidyalaya Kanpur', 'Dead Poets Society', 'Josh Charles', 'Aaron1', 'josh charles']))


# construct cognitive graph in training data    
from utils import judge_question_type
def find_fact_content(bundle, title, sen_num):
    for x in bundle['context']:
        if x[0] == title:
            return x[1][sen_num]
test = copy.deepcopy(train_set)
for bundle in tqdm(test):
    entities = set([title for title, sen_num in bundle['supporting_facts']])
    bundle['Q_edge'] = fuzzy_find(entities, bundle['question'])
    question_type = judge_question_type(bundle['question'])
    for fact in bundle['supporting_facts']:
        try:
            title, sen_num = fact
            pool = set()
            for i in range(sen_num + 1):
                name = 'edges:###{}###{}'.format(i, title)
                tmp = set([x.decode().split('###')[0] for x in db.lrange(name, 0, -1)])
                pool |= tmp
            pool &= entities
            stripped = [re.sub(r' \(.*?\)$', '', x) for x in pool] + ['yes', 'no']
            if bundle['answer'] not in stripped:
                if fuzz.ratio(re.sub(r'\(.*?\)$', '', title), bundle['answer']) > 80:
                    pool.add(title)
                else:
                    pool.add(bundle['answer'])
            if bundle['answer'] == 'yes' or bundle['answer'] == 'no' \
                    or (question_type > 0 and bundle['type'] == 'comparison'):
                pool.add(title)
            r = fuzzy_find(pool, find_fact_content(bundle, title, sen_num))
            fact.append(r)
        except IndexError as e: 
            print(bundle['_id'])
with open('./hotpot_train_v1.1_refined3.json', 'w') as fout:
    json.dump(test, fout)

