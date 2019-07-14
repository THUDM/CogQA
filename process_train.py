
# coding: utf-8

# %pdb on
import json
import re
import numpy as np
import copy
from tqdm import tqdm 
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from redis import StrictRedis
from utils import fuzzy_find


db = StrictRedis(host='localhost', port=6379, db=0)


with open('./hotpot_train_v1.1.json', 'r') as fin:
    train_set = json.load(fin)
print('Finish Reading! len = ', len(train_set))


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

print(list(fuzzy_find(['Miami Gardens, Florida', 'WSCV', 'Hard Rock Stadium'], r"Hard Rock Stadium is a multipurpose football stadium located in Miami Gardens, a city north of Miami. It is the home stadium of the Miami Dolphins of the National Football League (NFL).")))


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
with open('./hotpot_train_v1.1_refined.json', 'w') as fout:
    json.dump(test, fout)

