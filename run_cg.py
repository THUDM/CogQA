import re
import json
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
from model import BertForMultiHopQuestionAnswering, CognitiveGraph
from torch.optim import Adam
from tqdm import tqdm, trange
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import pdb
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.optimization import BertAdam
from utils import warmup_linear, find_start_end_after_tokenized, find_start_end_before_tokenized, bundle_part_to_batch, judge_question_type, fuzzy_retrieve, WindowMean, fuzz
import random
from collections import namedtuple
import numpy as np
import copy
import traceback
# import inspect
# from gpu_mem_track import  MemTracker
# frame = inspect.currentframe()          # define a frame to track
# gpu_tracker = MemTracker(frame, device=6) 

FIELDS = ['ids', 'hop_start_weights', 'hop_end_weights', 'ans_start_weights', 'ans_end_weights', 'segment_ids', 'sep_positions',
     'additional_nodes', 'adj', 'answer_id', 'question_type', '_id']

class Bundle(object):
    pass



def convert_question_to_samples_bundle(tokenizer, data: 'Json refined', neg = 2):
    context = dict(data['context'])
    gold_sentences_set = dict([((para, sen), edges) for para, sen, edges in data['supporting_facts']]) 
    e2i = {}
    i2e = []
    for entity, sens in context.items():
        assert not entity in e2i
        e2i[entity] = len(i2e)
        i2e.append(entity)
    prev = [[]] * len(i2e) 

    ids, hop_start_weights, hop_end_weights, ans_start_weights, ans_end_weights, segment_ids, sep_positions, additional_nodes = [], [], [], [], [], [], [], []
    tokenized_question = ['[CLS]'] + tokenizer.tokenize(data['question']) + ['[SEP]']
    for title_x, sen, edges in data['supporting_facts']: # TODO: match previous sentence 
        for title_y, matched, l, r in edges:
            if title_y not in e2i: # answer
                assert data['answer'] == title_y
                e2i[title_y] = len(i2e)
                i2e.append(title_y)
                prev.append([])
            if title_x != title_y:
                y = e2i[title_y]
                prev[y] = prev[y] + tokenizer.tokenize(context[title_x][sen]) + ['[SEP]']
    question_type = judge_question_type(data['question'])

    # fix by answer:
    if data['answer'] == 'yes' or data['answer'] == 'no':
        question_type = 1
        answer_entity = data['answer']
    else:
        # find answer entity
        answer_entity = fuzzy_retrieve(data['answer'], e2i, 'distractor', 80)
        if answer_entity is None:
            raise ValueError('Cannot find answer: {}'.format(data['answer']))
    
    if question_type == 0:
        answer_id = e2i[answer_entity]
    elif len(data['Q_edge']) != 2:
        if question_type == 1:
            raise ValueError('There must be 2 entities in "Q_edge" for type 1 question.')
        elif question_type == 2:
            # print('Convert type 2 question to 0.\n Question:{}'.format(data['question']))
            question_type = 0
            answer_id = e2i[answer_entity]
    else:
        answer_id = [e2i[data['Q_edge'][0][0]], e2i[data['Q_edge'][1][0]]]
        if question_type == 1:
            answer_id.append(int(data['answer'] == 'yes'))
        elif question_type == 2:
            if data['answer'] == data['Q_edge'][0][1]:
                answer_id.append(0)
            elif data['answer'] == data['Q_edge'][1][1]:
                answer_id.append(1)
            else:
                score = (fuzz.partial_ratio(data['answer'], data['Q_edge'][0][1]), fuzz.partial_ratio(data['answer'], data['Q_edge'][1][1]))
                if score[0] < 50 and score[1] < 50:
                    raise ValueError('There is no exact match in selecting question. answer: {}'.format(data['answer']))
                else:
                    # print('Resolve type 1 or 2 question: {}\n answer: {}'.format(data['question'], data['answer']))
                    answer_id.append(0 if score[0] > score[1] else 1)
        else:
            pass
    
    for entity, sens in context.items():
        num_hop, num_ans = 0, 0
        tokenized_all = tokenized_question + prev[e2i[entity]]
        if len(tokenized_all) > 512:
            tokenized_all = tokenized_all[:512]
            print('PREV TOO LONG, id: {}'.format(data['_id']))
        segment_id = [0] * len(tokenized_all)
        sep_position = [] 
        hop_start_weight = [0] * len(tokenized_all)
        hop_end_weight = [0] * len(tokenized_all)
        ans_start_weight = [0] * len(tokenized_all)
        ans_end_weight = [0] * len(tokenized_all)

        for sen_num, sen in enumerate(sens):
            tokenized_sen = tokenizer.tokenize(sen) + ['[SEP]']
            if len(tokenized_all) + len(tokenized_sen) > 512 or sen_num > 15:
                break
            # if sen_num > 10:
            #     raise ValueError('Too many sentences in context: {}'.format(sens))
            tokenized_all += tokenized_sen
            segment_id += [sen_num + 1] * len(tokenized_sen)
            sep_position.append(len(tokenized_all) - 1)
            hs_weight = [0] * len(tokenized_sen)
            he_weight = [0] * len(tokenized_sen)
            as_weight = [0] * len(tokenized_sen)
            ae_weight = [0] * len(tokenized_sen)
            if (entity, sen_num) in gold_sentences_set:
                tmp = gold_sentences_set[(entity, sen_num)]
                intervals = find_start_end_after_tokenized(tokenizer, tokenized_sen,
                    [matched  for _, matched, _, _ in tmp])
                for j, (l, r) in enumerate(intervals):
                    if tmp[j][0] == answer_entity or question_type > 0:
                        as_weight[l] = ae_weight[r] = 1
                        num_ans += 1
                    else:
                        hs_weight[l] = he_weight[r] = 1
                        num_hop += 1
            hop_start_weight += hs_weight
            hop_end_weight += he_weight
            ans_start_weight += as_weight
            ans_end_weight += ae_weight
            
        assert len(tokenized_all) <= 512
        # for i in range(len(start_weight)):
        #     start_weight[i] /= max(num_spans, 1)
        #     end_weight[i] /= max(num_spans, 1)
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
            tokenized_all = tokenized_question + prev[e2i[answer_entity]]
            if len(tokenized_all) > 512:
                tokenized_all = tokenized_all[:512]
                print('ANSWER TOO LONG! id: {}'.format(data['_id']))
            additional_nodes.append(tokenizer.convert_tokens_to_ids(tokenized_all))
        for i in range(neg):
            # build negative node n+i
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
            edges_in_bundle.append((e2i[father_para], n + i))
        n += neg
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
        

def train(bundles, model, device, batch_size = 4, num_epoch = 1, mode = 'tensors', model_cg = None, gradient_accumulation_steps = 1, lr = 1e-4):
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    num_batch, dataloader = homebrew_data_loader(bundles, mode = mode, batch_size=batch_size)
    num_steps = num_batch * num_epoch
    global_step = 0
    opt = BertAdam(optimizer_grouped_parameters, lr = lr, warmup = 0.1, t_total=num_steps)
    if mode == 'bundle':
        opt_cg = Adam(model_cg.parameters(), lr=1e-4) # TODO hyperparam
        model_cg.to(device)
        model_cg.train()
        warmed = False

    model.to(device)
    model.train()
    for epoch in trange(num_epoch, desc = 'Epoch'):
        ans_mean, hop_mean = WindowMean(), WindowMean()
        if mode == 'bundle':
            final_mean = WindowMean()
            opt_cg.zero_grad()
        opt.zero_grad()
        tqdm_obj = tqdm(dataloader, total = num_batch)
        for step, batch in enumerate(tqdm_obj):
            # torch.cuda.empty_cache()
            # gpu_tracker.track()
            try:
                if mode == 'tensors':
                    batch = tuple(t.to(device) for t in batch)
                    hop_loss, ans_loss, pooled_output = model(*batch)
                    hop_loss, ans_loss = hop_loss.mean(), ans_loss.mean()
                    pooled_output.detach()
                    loss = ans_loss + hop_loss
                elif mode == 'bundle':
                    hop_loss, ans_loss, final_loss = model_cg(batch, model, device)
                    hop_loss, ans_loss = hop_loss.mean(), ans_loss.mean()
                    loss = ans_loss + hop_loss + 0.2 * final_loss
                # torch.cuda.empty_cache()
                # gpu_tracker.track()
                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = lr * warmup_linear(global_step/num_steps, warmup = 0.1)
                    for param_group in opt.param_groups:
                        param_group['lr'] = lr_this_step
                    global_step += 1
                    if mode == 'bundle':
                        opt_cg.step()
                        opt_cg.zero_grad()
                        final_mean_loss = final_mean.update(final_loss.item())
                        tqdm_obj.set_description('ans_loss: {:.2f}, hop_loss: {:.2f}, final_loss: {:.2f}'.format(
                            ans_mean.update(ans_loss.item()), hop_mean.update(hop_loss.item()), final_mean_loss))
                        if final_mean_loss < 0.9 and step > 100:
                            warmed = True
                        if warmed:
                            opt.step()
                        opt.zero_grad()
                    else:
                        opt.step()
                        opt.zero_grad()
                        tqdm_obj.set_description('ans_loss: {:.2f}, hop_loss: {:.2f}'.format(
                            ans_mean.update(ans_loss.item()), hop_mean.update(hop_loss.item())))
                    if step % 1000 == 0:
                        print('')
                        output_model_file = './models/bert-base-uncased.bin.tmp'
                        saved_dict = {'bert-params' : model.module.state_dict()}
                        saved_dict['cg-params'] = model_cg.state_dict()
                        torch.save(saved_dict, output_model_file)
            except Exception as err:
                traceback.print_exc()
                if mode == 'bundle':   
                    print(batch._id) 
    return (model, model_cg)


def main(output_model_file = './models/bert-base-uncased.bin', load = False, mode = 'tensors', batch_size = 12, lr = 1e-4):
    BERT_MODEL = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
    with open('./hotpot_train_v1.1_refined3.json' ,'r') as fin:
        dataset = json.load(fin)
    bundles = []
    for data in tqdm(dataset):
        try:
            bundles.append(convert_question_to_samples_bundle(tokenizer, data))
        except Exception as err:
            # traceback.print_exc()
            # print(data['question'])
            pass
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    if load:
        print('Loading model from {}'.format(output_model_file))
        model_state_dict = torch.load(output_model_file)
        model = BertForMultiHopQuestionAnswering.from_pretrained(BERT_MODEL, state_dict=model_state_dict['bert-params'])
        model_cg = CognitiveGraph(model.config.hidden_size)
        #model_cg.load_state_dict(model_state_dict['cg-params'])

    else:
        model = BertForMultiHopQuestionAnswering.from_pretrained(BERT_MODEL,
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
        model_cg = CognitiveGraph(model.config.hidden_size)

    print('Start Training... on {} GPUs'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model, device_ids = range(torch.cuda.device_count()))
    model, model_cg = train(bundles, model, device, batch_size=batch_size, model_cg=model_cg, mode = mode, lr=lr)
    print('Saving model to {}'.format(output_model_file))
    saved_dict = {'bert-params' : model.module.state_dict()}
    saved_dict['cg-params'] = model_cg.state_dict()
    torch.save(saved_dict, output_model_file)

import fire
if __name__ == "__main__":
    fire.Fire(main)
