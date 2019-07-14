import re
import json
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
from model import BertForMultiHopQuestionAnswering, CognitiveGNN
from torch.optim import Adam
from tqdm import tqdm, trange
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import pdb
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.optimization import BertAdam
from utils import warmup_linear, find_start_end_after_tokenized, find_start_end_before_tokenized, bundle_part_to_batch, judge_question_type, fuzzy_retrieve, WindowMean, get_context_fullwiki
import random
from collections import namedtuple
import numpy as np
import copy
# from line_profiler import LineProfiler

def cognitive_graph_propagate(tokenizer, data: 'Json eval(Context as pool)', model1, model2, device, setting:'distractor / fullwiki' = 'fullwiki', max_new_nodes = 5):
    """Answer the question in ``data'' by trained CogQA model.
    
    Args:
        tokenizer (Tokenizer): Word-Piece tokenizer.
        data (Json): Unrefined.
        model1 (nn.Module): System 1 model.
        model2 (nn.Module): System 2 model.
        device (torch.device): Selected device.
        setting (string, optional): 'distractor / fullwiki'. Defaults to 'fullwiki'.
        max_new_nodes (int, optional): Maximum number of new nodes in cognitive graph. Defaults to 5.
    
    Returns:
        tuple: (gold_ret, ans_ret, graph_ret, ans_nodes_ret)
    """
    context = dict(data['context'])
    e2i = dict([(entity, id) for id, entity in enumerate(context.keys())])
    n = len(context)
    i2e = [''] * n
    for k, v in e2i.items():
        i2e[v] = k  
    prev = [[] for i in range(n)] # elements: (title, sen_num)
    queue = range(n) 
    semantics = [None] * n

    tokenized_question = ['[CLS]'] + tokenizer.tokenize(data['question']) + ['[SEP]']

    def construct_infer_batch(queue):
        """Construct next batch (frontier nodes to visit).
        
        Args:
            queue (list): A queue containing frontier nodes.
        
        Returns:
            tuple: A batch of inputs
        """
        ids, sep_positions, segment_ids, tokenized_alls, B_starts = [], [], [], [], []
        max_length, max_seps, num_samples = 0, 0, len(queue)
        for x in queue:
            tokenized_all = copy.copy(tokenized_question)
            for title, sen_num in prev[x]:
                tokenized_all += tokenizer.tokenize(context[title][sen_num]) + ['[SEP]']
            if len(tokenized_all) > 512:
                tokenized_all = tokenized_all[:512]
                print('PREV TOO LONG, id: {}'.format(data['_id']))
            segment_id = [0] * len(tokenized_all)
            sep_position = [] 
            B_starts.append(len(tokenized_all))
            for sen_num, sen in enumerate(context[i2e[x]]):
                tokenized_sen = tokenizer.tokenize(sen) + ['[SEP]']
                if len(tokenized_all) + len(tokenized_sen) > 512 or sen_num > 15:
                    break
                tokenized_all += tokenized_sen
                segment_id += [sen_num + 1] * len(tokenized_sen)
                sep_position.append(len(tokenized_all) - 1)
            max_length = max(max_length, len(tokenized_all))
            max_seps = max(max_seps, len(sep_position))
            tokenized_alls.append(tokenized_all)
            ids.append(tokenizer.convert_tokens_to_ids(tokenized_all))
            sep_positions.append(sep_position)
            segment_ids.append(segment_id)

        ids_tensor = torch.zeros((num_samples, max_length), dtype = torch.long, device = device)
        sep_positions_tensor = torch.zeros((num_samples, max_seps), dtype = torch.long, device = device)
        segment_ids_tensor = torch.zeros((num_samples, max_length), dtype = torch.long, device = device)
        input_mask = torch.zeros((num_samples, max_length), dtype = torch.long, device = device)
        B_starts = torch.tensor(B_starts, dtype = torch.long, device = device)
        for i in range(num_samples):
            length = len(ids[i])
            ids_tensor[i, :length] = torch.tensor(ids[i], dtype = torch.long)
            sep_positions_tensor[i, :len(sep_positions[i])] = torch.tensor(sep_positions[i], dtype = torch.long)
            segment_ids_tensor[i, :length] = torch.tensor(segment_ids[i], dtype = torch.long)
            input_mask[i, :length] = 1
        return ids_tensor, segment_ids_tensor, input_mask, sep_positions_tensor, tokenized_alls, B_starts
    
    gold_ret, ans_nodes = set([]), set([])
    allow_limit = [0, 0]
    while len(queue) > 0:
        # visit all nodes in the frontier queue
        ids, segment_ids, input_mask, sep_positions, tokenized_alls, B_starts = construct_infer_batch(queue)
        hop_preds, ans_preds, semantics_preds, no_ans_logits = model1(ids, segment_ids, input_mask, sep_positions,
            None, None, None, None, 
            B_starts, allow_limit)  
        new_queue = []
        for i, x in enumerate(queue):
            semantics[x] = semantics_preds[i]
            # for hop spans
            for k in range(hop_preds.size()[1]):
                l, r, j = hop_preds[i, k]
                j = j.item()
                if l == 0:
                    break
                gold_ret.add((i2e[x], j)) # supporting facts
                orig_text = context[i2e[x]][j]
                pred_slice = tokenized_alls[i][l : r + 1]
                l, r = find_start_end_before_tokenized(orig_text, [pred_slice])[0]
                if l == r == 0:
                    continue    
                recovered_matched = orig_text[l: r]
                pool = context if setting == 'distractor' else (i2e[x], j)
                matched = fuzzy_retrieve(recovered_matched, pool, setting)    
                if matched is not None:
                    if setting == 'fullwiki' and matched not in e2i and n < 10 + max_new_nodes:
                        context_new = get_context_fullwiki(matched)
                        if len(context_new) > 0: # cannot resovle redirection
                            # create new nodes in the cognitive graph
                            context[matched] = context_new
                            prev.append([])
                            semantics.append(None)
                            e2i[matched] = n
                            i2e.append(matched)
                            n += 1
                    if matched in e2i and e2i[matched] != x:
                        y = e2i[matched]
                        if y not in new_queue and (i2e[x], j) not in prev[y]:
                            # new edge means new clues! update the successor as frontier nodes.
                            new_queue.append(y)
                            prev[y].append(((i2e[x], j)))
            # for ans spans
            for k in range(ans_preds.size()[1]):
                l, r, j = ans_preds[i, k]
                j = j.item()
                if l == 0:
                    break
                gold_ret.add((i2e[x], j))
                orig_text = context[i2e[x]][j]
                pred_slice = tokenized_alls[i][l : r + 1]
                l, r = find_start_end_before_tokenized(orig_text, [pred_slice])[0]
                if l == r == 0:
                    continue    
                recovered_matched = orig_text[l: r]
                matched = fuzzy_retrieve(recovered_matched, context, 'distractor', threshold=70)
                if matched is not None:
                    y = e2i[matched]
                    ans_nodes.add(y)
                    if (i2e[x], j) not in prev[y]:
                        prev[y].append(((i2e[x], j)))
                elif n < 10 + max_new_nodes:
                    context[recovered_matched] = []
                    e2i[recovered_matched] = n
                    i2e.append(recovered_matched)
                    new_queue.append(n)
                    ans_nodes.add(n)
                    prev.append([(i2e[x], j)])
                    semantics.append(None)
                    n += 1
        if len(new_queue) == 0 and len(ans_nodes) == 0 and allow_limit[1] < 0.1: # must find one answer
            # ``allow'' is an offset of negative threshold. 
            # If no ans span is valid, make the minimal gap between negative threshold and probability of ans spans -0.1, and try again.
            prob, pos_in_queue = torch.min(no_ans_logits, dim = 0)
            new_queue.append(queue[pos_in_queue])
            allow_limit[1] = prob.item() + 0.1
        queue = new_queue

    question_type = judge_question_type(data['question'])

    if n == 0:
        return set([]), 'yes', [], []
    if n == 1 and question_type > 0:
        ans_ret = 'yes' if question_type == 1 else i2e[0]
        return [(i2e[0], 0)], ans_ret, [], []
    # GCN || CompareNets
    semantics = torch.stack(semantics)
    if question_type == 0:
        adj = torch.eye(n, device = device) * 2
        for x in range(n):
            for title, sen_num in prev[x]:
                adj[e2i[title], x] = 1
        adj /= torch.sum(adj, dim=0, keepdim=True)
        pred = model2.gcn(adj, semantics)
        for x in range(n):
            if x not in ans_nodes:
                pred[x] -= 10000.
        ans_ret = i2e[torch.argmax(pred).item()]
    else:
        # Take the most golden paragraphs as x,y
        gold_num = torch.zeros(n)
        for title, sen_num in gold_ret:
            gold_num[e2i[title]] += 1
        x, y = gold_num.topk(2)[1].tolist()
        diff_sem = semantics[x] - semantics[y]
        classifier = model2.both_net if question_type == 1 else model2.select_net
        pred = int(torch.sigmoid(classifier(diff_sem)).item() > 0.5)
        ans_ret = ['no', 'yes'][pred] if question_type == 1 else [i2e[x], i2e[y]][pred] 
    
    ans_ret = re.sub(r' \(.*?\)$', '', ans_ret)

    graph_ret = []
    for x in range(n):
        for title, sen_num in prev[x]:
            graph_ret.append('({}, {}) --> {}'.format(title, sen_num, i2e[x]))    

    ans_nodes_ret = [i2e[x] for x in ans_nodes]
    return gold_ret, ans_ret, graph_ret, ans_nodes_ret

def main(BERT_MODEL='bert-base-uncased', model_file='./models/bert-base-uncased.bin', data_file='./hotpot_dev_distractor_v1.json', max_new_nodes=5):
    setting = 'distractor' if data_file.find('distractor') >= 0 else 'fullwiki'
    with open(data_file, 'r') as fin:
        dataset = json.load(fin)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    print('Loading model from {}'.format(model_file))
    model_state_dict = torch.load(model_file)
    model1 = BertForMultiHopQuestionAnswering.from_pretrained(BERT_MODEL, state_dict=model_state_dict['params1'])
    model2 = CognitiveGNN(model1.config.hidden_size)
    model2.load_state_dict(model_state_dict['params2'])
    sp, answer, graphs = {}, {}, {}
    print('Start Training... on {} GPUs'.format(torch.cuda.device_count()))
    model1 = torch.nn.DataParallel(model1, device_ids = range(torch.cuda.device_count()))
    model1.to(device).eval()
    model2.to(device).eval()

    with torch.no_grad():
        for data in tqdm(dataset):
            gold, ans, graph_ret, ans_nodes = cognitive_graph_propagate(tokenizer, data, model1, model2, device, setting = setting, max_new_nodes=max_new_nodes)
            sp[data['_id']] = list(gold)
            answer[data['_id']] = ans
            graphs[data['_id']] = graph_ret + ['answer_nodes: ' + ', '.join(ans_nodes)]
    pred_file = data_file.replace('.json', '_pred.json')
    with open(pred_file, 'w') as fout:
        json.dump({'answer': answer, 'sp': sp, 'graphs': graphs}, fout)
    
import fire
if __name__ == "__main__":
    fire.Fire(main)