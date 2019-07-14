import re
import json
from tqdm import tqdm, trange
import pdb
import random
from collections import namedtuple
import numpy as np
import copy
import traceback
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.optimization import BertAdam
from model import BertForMultiHopQuestionAnswering, CognitiveGNN
from utils import warmup_linear, find_start_end_after_tokenized, find_start_end_before_tokenized, bundle_part_to_batch, fuzzy_retrieve, WindowMean, fuzz



class Bundle(object):
    """The structure to contain all data for training. 
    
    A flexible class. The properties are defined in FIELDS and dynamically added by capturing variables with the same names at runtime.
    """
    pass

FIELDS = ['ids', 'hop_start_weights', 'hop_end_weights', 'ans_start_weights', 'ans_end_weights', 'segment_ids', 'sep_positions',
     'additional_nodes', 'adj', 'answer_id', 'question_type', '_id']


# Judge question type with interrogative words
GENERAL_WD = ['is', 'are', 'am', 'was', 'were', 'have', 'has', 'had', 'can', 'could', 
              'shall', 'will', 'should', 'would', 'do', 'does', 'did', 'may', 'might', 'must', 'ought', 'need', 'dare']
GENERAL_WD += [x.capitalize() for x in GENERAL_WD]
GENERAL_WD = re.compile(' |'.join(GENERAL_WD))
def judge_question_type(q : str, G = GENERAL_WD) -> int:
    if q.find(' or ') >= 0:
        return 2 
    elif G.match(q):
        return 1
    else:
        return 0

def improve_question_type_and_answer(data, e2i):
    '''Improve the result of the judgement of question type in training data with other information.
    
    If the question is a special question(type 0), answer_id is the index of final answer node. Otherwise answer_ids are
    the indices of two compared nodes and the result of comparison(0 / 1).
    This part is not very important to the overall results, but avoids Runtime Errors in rare cases.
    
    Args:
        data (Json): Refined distractor-setting samples.
        e2i (dict): entity2index dict.
    
    Returns:
        (int, int or (int, int, 0 / 1), string): question_type, answer_id and answer_entity.
    '''
    question_type = judge_question_type(data['question'])
    # fix judgement by answer
    if data['answer'] == 'yes' or data['answer'] == 'no':
        question_type = 1
        answer_entity = data['answer']
    else:
        # check whether the answer can be extracted as a span
        answer_entity = fuzzy_retrieve(data['answer'], e2i, 'distractor', 80)
        if answer_entity is None:
            raise ValueError('Cannot find answer: {}'.format(data['answer']))
    
    if question_type == 0:
        answer_id = e2i[answer_entity]
    elif len(data['Q_edge']) != 2:
        if question_type == 1:
            raise ValueError('There must be 2 entities in "Q_edge" for type 1 question.')
        elif question_type == 2: # Judgement error, should be type 0
            question_type = 0
            answer_id = e2i[answer_entity]
    else:
        answer_id = [e2i[data['Q_edge'][0][0]], e2i[data['Q_edge'][1][0]]] # compared nodes
        if question_type == 1:
            answer_id.append(int(data['answer'] == 'yes'))
        elif question_type == 2:
            if data['answer'] == data['Q_edge'][0][1]:
                answer_id.append(0)
            elif data['answer'] == data['Q_edge'][1][1]:
                answer_id.append(1)
            else: # cannot exactly match an option
                score = (fuzz.partial_ratio(data['answer'], data['Q_edge'][0][1]), fuzz.partial_ratio(data['answer'], data['Q_edge'][1][1]))
                if score[0] < 50 and score[1] < 50:
                    raise ValueError('There is no exact match in selecting question. answer: {}'.format(data['answer']))
                else:
                    answer_id.append(0 if score[0] > score[1] else 1)
    return question_type, answer_id, answer_entity

def convert_question_to_samples_bundle(tokenizer, data: 'Json refined', neg = 2):
    '''Make training samples.
    
    Convert distractor-setting samples(question + 10 paragraphs + answer + supporting facts) to bundles.
    
    Args:
        tokenizer (BertTokenizer): BERT Tokenizer to transform sentences to a list of word pieces.
        data (Json): Refined distractor-setting samples with gold-only cognitive graphs. 
        neg (int, optional): Defaults to 2. Negative answer nodes to add in every sample.
    
    Raises:
        ValueError: Invalid question type. 

    Returns:
        Bundle: A bundle containing 10 separate samples(including gold and negative samples).
    '''

    context = dict(data['context']) # all the entities in 10 paragraphs
    gold_sentences_set = dict([((para, sen), edges) for para, sen, edges in data['supporting_facts']]) 
    e2i, i2e = {}, [] # entity2index, index2entity
    for entity, sens in context.items():
        e2i[entity] = len(i2e)
        i2e.append(entity)
    clues = [[]] * len(i2e) # pre-extracted clues

    ids, hop_start_weights, hop_end_weights, ans_start_weights, ans_end_weights, segment_ids, sep_positions, additional_nodes = [], [], [], [], [], [], [], []
    tokenized_question = ['[CLS]'] + tokenizer.tokenize(data['question']) + ['[SEP]']

    # Extract clues for entities in the gold-only cogntive graph
    for entity_x, sen, edges in data['supporting_facts']:
        for entity_y, _, _, _ in edges:
            if entity_y not in e2i: # entity y must be the answer
                assert data['answer'] == entity_y
                e2i[entity_y] = len(i2e)
                i2e.append(entity_y)
                clues.append([])
            if entity_x != entity_y:
                y = e2i[entity_y]
                clues[y] = clues[y] + tokenizer.tokenize(context[entity_x][sen]) + ['[SEP]']
    
    question_type, answer_id, answer_entity = improve_question_type_and_answer(data, e2i)
    
    # Construct training samples
    for entity, para in context.items():
        num_hop, num_ans = 0, 0
        tokenized_all = tokenized_question + clues[e2i[entity]]
        if len(tokenized_all) > 512: # BERT-base accepts at most 512 tokens
            tokenized_all = tokenized_all[:512]
            print('CLUES TOO LONG, id: {}'.format(data['_id']))
        # initialize a sample for ``entity''
        sep_position = [] 
        segment_id = [0] * len(tokenized_all)
        hop_start_weight = [0] * len(tokenized_all)
        hop_end_weight = [0] * len(tokenized_all)
        ans_start_weight = [0] * len(tokenized_all)
        ans_end_weight = [0] * len(tokenized_all)

        for sen_num, sen in enumerate(para):
            tokenized_sen = tokenizer.tokenize(sen) + ['[SEP]']
            if len(tokenized_all) + len(tokenized_sen) > 512 or sen_num > 15:
                break
            tokenized_all += tokenized_sen
            segment_id += [sen_num + 1] * len(tokenized_sen)
            sep_position.append(len(tokenized_all) - 1)
            hs_weight = [0] * len(tokenized_sen)
            he_weight = [0] * len(tokenized_sen)
            as_weight = [0] * len(tokenized_sen)
            ae_weight = [0] * len(tokenized_sen)
            if (entity, sen_num) in gold_sentences_set:
                edges = gold_sentences_set[(entity, sen_num)]
                intervals = find_start_end_after_tokenized(tokenizer, tokenized_sen,
                    [matched  for _, matched, _, _ in edges])
                for j, (l, r) in enumerate(intervals):
                    if edges[j][0] == answer_entity or question_type > 0: # successive node edges[j][0] is answer node
                        as_weight[l] = ae_weight[r] = 1
                        num_ans += 1
                    else: # edges[j][0] is next-hop node
                        hs_weight[l] = he_weight[r] = 1
                        num_hop += 1
            hop_start_weight += hs_weight
            hop_end_weight += he_weight
            ans_start_weight += as_weight
            ans_end_weight += ae_weight
            
        assert len(tokenized_all) <= 512
        # if entity is a negative node, train negative threshold at [CLS] 
        if 1 not in hop_start_weight:
            hop_start_weight[0] = 0.1
        if 1 not in ans_start_weight:
            ans_start_weight[0] = 0.1

        ids.append(tokenizer.convert_tokens_to_ids(tokenized_all))
        sep_positions.append(sep_position)
        segment_ids.append(segment_id)
        hop_start_weights.append(hop_start_weight)
        hop_end_weights.append(hop_end_weight)
        ans_start_weights.append(ans_start_weight)
        ans_end_weights.append(ans_end_weight)

    # Construct negative answer nodes for task #2(answer node prediction)
    n = len(context)
    edges_in_bundle = []
    if question_type == 0:
        # find all edges and prepare forbidden set(containing answer) for negative sampling
        forbidden = set([])
        for para, sen, edges in data['supporting_facts']:
            for x, matched, l, r in edges:
                edges_in_bundle.append((e2i[para], e2i[x]))
                if x == answer_entity:
                    forbidden.add((para, sen))
        if answer_entity not in context and answer_entity in e2i:
            n += 1
            tokenized_all = tokenized_question + clues[e2i[answer_entity]]
            if len(tokenized_all) > 512:
                tokenized_all = tokenized_all[:512]
                print('ANSWER TOO LONG! id: {}'.format(data['_id']))
            additional_nodes.append(tokenizer.convert_tokens_to_ids(tokenized_all))

        for i in range(neg):
            # build negative answer node n+i
            father_para = random.choice(list(context.keys()))
            father_sen = random.randrange(len(context[father_para]))
            if (father_para, father_sen) in forbidden:
                father_para = random.choice(list(context.keys()))
                father_sen = random.randrange(len(context[father_para]))
            if (father_para, father_sen) in forbidden:
                neg -= 1
                continue
            tokenized_all = tokenized_question + tokenizer.tokenize(context[father_para][father_sen]) + ['[SEP]']
            if len(tokenized_all) > 512:
                tokenized_all = tokenized_all[:512]
                print('NEG TOO LONG! id: {}'.format(data['_id']))
            additional_nodes.append(tokenizer.convert_tokens_to_ids(tokenized_all))
            edges_in_bundle.append((e2i[father_para], n))
            n += 1

    if question_type >= 1:
        for para, sen, edges in data['supporting_facts']:
            for x, matched, l, r in edges:
                if e2i[para] < n and  e2i[x] < n:
                    edges_in_bundle.append((e2i[para], e2i[x]))
                    
    assert n == len(additional_nodes) + len(context)
    adj = torch.eye(n) * 2
    for x, y in edges_in_bundle:
        adj[x, y] = 1
    adj /= torch.sum(adj, dim=0, keepdim=True)

    _id = data['_id']
    ret = Bundle()
    for field in FIELDS:
        setattr(ret, field, eval(field))
    return ret
    
def homebrew_data_loader(bundles, mode : 'bundle or tensors' = 'tensors', batch_size = 8):
    '''Return a generator like DataLoader in pytorch
    
    Different data are fed in task #1 and #2. In task #1, steps for different entities are decoupled into 10 samples
    and can be randomly shuffled. But in task #2, inputs must be whole graphs. 
    
    Args:
        bundles (list): List of bundles for questions.
        mode (string, optional): Defaults to 'tensors'. 'tensors' represents dataloader for task #1,
            'bundle' represents dataloader for task #2.
        batch_size (int, optional): Defaults to 8. 
    
    Raises:
        ValueError: Invalid mode
    
    Returns:
        (int, Generator): number of batches and a generator to generate batches.
    '''

    if mode == 'bundle':
        random.shuffle(bundles)
        def gen():
            for bundle in bundles:
                yield bundle
        return len(bundles), gen()
    elif mode == 'tensors':
        all_bundle = Bundle()
        for field in FIELDS[:7]:
            t = []
            setattr(all_bundle, field, t)
            for bundle in bundles:
                t.extend(getattr(bundle, field))
        n = len(t)
        # random shuffle
        orders = np.random.permutation(n)
        for field in FIELDS[:7]:
            t = getattr(all_bundle, field)
            setattr(all_bundle, field, [t[x] for x in orders])
        
        num_batch = (n - 1) // batch_size + 1
        def gen():
            for batch_num in range(num_batch):
                l, r = batch_num * batch_size, min((batch_num + 1) * batch_size, n)
                yield bundle_part_to_batch(all_bundle, l, r)
        return num_batch, gen()
    else:
        raise ValueError('mode must be "bundle" or "tensors"!')
        