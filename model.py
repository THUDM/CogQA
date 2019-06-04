from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel, BertLayerNorm, gelu, BertEncoder, BertPooler
import torch
from torch import nn
from utils import fuzzy_find, find_start_end_after_tokenized, find_start_end_before_tokenized, bundle_part_to_batch
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
import re
import pdb

class MLP(nn.Module):
    def __init__(self, input_sizes, dropout_prob = 0.2, bias = False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(input_sizes)):
            self.layers.append(nn.Linear(input_sizes[i - 1], input_sizes[i], bias=bias))
        self.norm_layers = nn.ModuleList()
        if len(input_sizes) > 2:
            for i in range(1, len(input_sizes) - 1):
                self.norm_layers.append(nn.LayerNorm(input_sizes[i]))
        self.drop_out = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.drop_out(x))
            if i < len(self.layers) - 1:
                x = gelu(x)
                if len(self.norm_layers):
                    x = self.norm_layers[i](x)
        return x

class GCN(nn.Module):
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.05)

    def __init__(self, input_size):
        super(GCN, self).__init__()
        self.diffusion = nn.Linear(input_size, input_size, bias=False)
        self.retained = nn.Linear(input_size, input_size, bias=False)
        # self.predict = nn.Linear(input_size, 1, bias=False)
        self.predict = MLP(input_sizes = (input_size, input_size, 1))
        self.apply(self.init_weights)

    def forward(self, A, x):
        layer1_diffusion = A.t().mm(gelu(self.diffusion(x)))
        x = gelu(self.retained(x) + layer1_diffusion)
        layer2_diffusion = A.t().mm(gelu(self.diffusion(x)))
        x = gelu(self.retained(x) + layer2_diffusion)
        return self.predict(x).squeeze(-1)

                
            
        

class BertEmbeddingsPlus(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config, max_sentence_type = 30):
        super(BertEmbeddingsPlus, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.sentence_type_embeddings = nn.Embedding(max_sentence_type, config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(( token_type_ids > 0).long())
        sentence_type_embeddings = self.sentence_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings + sentence_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertModelPlus(BertModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddingsPlus(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_hidden=-4):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1]
        # pooled_output = self.pooler(sequence_output)
        encoded_layers, hidden_layers = encoded_layers[-1], encoded_layers[output_hidden]
        return encoded_layers, hidden_layers

class BertForMultiHopQuestionAnswering(PreTrainedBertModel):
    def __init__(self, config):
        super(BertForMultiHopQuestionAnswering, self).__init__(config)
        self.bert = BertModelPlus(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 4)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sep_positions = None, 
                hop_start_weights=None, hop_end_weights=None, ans_start_weights=None, ans_end_weights=None, B_starts=None, allow_limit = (0, 0)):
        batch_size = input_ids.size()[0]
        device = input_ids.get_device() if input_ids.is_cuda else torch.device('cpu')
        # pdb.set_trace()
        sequence_output, hidden_output = self.bert(input_ids, token_type_ids, attention_mask)
        semantics = hidden_output[:, 0]
        # Some shapes: sequence_output [batch_size, max_length, hidden_size], pooled_output [batch_size, hidden_size]
        if sep_positions is None:
            return semantics # Only semantics, used in bundle forward
        else:
            max_sep = sep_positions.size()[-1]
        if max_sep == 0:
            empty = torch.zeros(batch_size, 0, dtype = torch.long, device = device)
            return empty, empty, semantics, empty # Only semantics, used in eval, ``empty'' is wrong but simple


        # Predict spans
        logits = self.qa_outputs(sequence_output)
        hop_start_logits, hop_end_logits, ans_start_logits, ans_end_logits = logits.split(1, dim=-1)
        hop_start_logits = hop_start_logits.squeeze(-1)
        hop_end_logits = hop_end_logits.squeeze(-1)
        ans_start_logits = ans_start_logits.squeeze(-1)
        ans_end_logits = ans_end_logits.squeeze(-1) # Shape: [batch_size, max_length]

        if hop_start_weights is not None: # Train mode
            lgsf = torch.nn.LogSoftmax(dim = 1) # If there is no targeted span in sentence, start_weights = end_weights = 0(vec)
            hop_start_loss = - torch.sum(hop_start_weights * lgsf(hop_start_logits), dim = -1)
            hop_end_loss = - torch.sum(hop_end_weights * lgsf(hop_end_logits), dim = -1)
            ans_start_loss = - torch.sum(ans_start_weights * lgsf(ans_start_logits), dim = -1)
            ans_end_loss = - torch.sum(ans_end_weights * lgsf(ans_end_logits), dim = -1)
            hop_loss = torch.mean((hop_start_loss + hop_end_loss)) / 2
            ans_loss = torch.mean((ans_start_loss + ans_end_loss)) / 2
        else:
            K_hop, K_ans = 3, 1
            hop_preds = torch.zeros(batch_size, K_hop, 3, dtype = torch.long, device = device) # (start, end, sen_num)
            ans_preds = torch.zeros(batch_size, K_ans, 3, dtype = torch.long, device = device)
            ans_start_gap = torch.zeros(batch_size, device = device)
            for u,(start_logits, end_logits, preds, K, allow) in enumerate(
                ((hop_start_logits, hop_end_logits, hop_preds, K_hop, allow_limit[0]), 
                (ans_start_logits, ans_end_logits, ans_preds, K_ans, allow_limit[1]))):
                for i in range(batch_size):
                    if sep_positions[i, 0] > 0:
                        values, indices = start_logits[i, B_starts[i]:].topk(K)
                        for k, index in enumerate(indices):
                            if values[k] <= start_logits[i, 0] - allow: # not golden
                                if u == 1:
                                    ans_start_gap[i] = start_logits[i, 0] - values[k]
                                break
                            start = index + B_starts[i]
                            # find ending
                            for j, ending in enumerate(sep_positions[i]):
                                if ending > start or ending <= 0:
                                    break
                            if ending <= start:
                                break
                            ending = min(ending, start + 10)
                            end = torch.argmax(end_logits[i, start:ending]) + start
                            preds[i, k, 0] = start
                            preds[i, k, 1] = end
                            preds[i, k, 2] = j
        return (hop_loss, ans_loss, semantics) if hop_start_weights is not None else (hop_preds, ans_preds, semantics, ans_start_gap)
     

class CognitiveGraph(nn.Module):
    def __init__(self, hidden_size):
        super(CognitiveGraph, self).__init__()
        self.gcn = GCN(hidden_size)
        self.both_net = MLP((hidden_size, hidden_size, 1))
        self.select_net = MLP((hidden_size, hidden_size, 1))
        # TODO: Re-initialization

    def forward(self, bundle, model, device):
        batch = bundle_part_to_batch(bundle)
        batch = tuple(t.to(device) for t in batch)
        hop_loss, ans_loss, semantics = model(*batch) # Shape of semantics: [num_para, hidden_size]
        # pdb.set_trace()
        num_additional_nodes = len(bundle.additional_nodes)

        if num_additional_nodes > 0:
            max_length_additional = max([len(x) for x in bundle.additional_nodes])
            ids = torch.zeros((num_additional_nodes, max_length_additional), dtype = torch.long, device = device)
            segment_ids = torch.zeros((num_additional_nodes, max_length_additional), dtype = torch.long, device = device)
            input_mask = torch.zeros((num_additional_nodes, max_length_additional), dtype = torch.long, device = device)
            for i in range(num_additional_nodes):
                length = len(bundle.additional_nodes[i])
                ids[i, :length] = torch.tensor(bundle.additional_nodes[i], dtype = torch.long)
                input_mask[i, :length] = 1
            additional_semantics = model(ids, segment_ids, input_mask)

            semantics = torch.cat((semantics, additional_semantics), dim = 0)

        assert semantics.size()[0] == bundle.adj.size()[0]
        
        if bundle.question_type == 0: # Wh-
            pred = self.gcn(bundle.adj.to(device), semantics)
            ce = torch.nn.CrossEntropyLoss()
            final_loss = ce(pred.unsqueeze(0), torch.tensor([bundle.answer_id], dtype = torch.long, device = device))
        else:
            x, y, ans = bundle.answer_id
            ans = torch.tensor(ans, dtype = torch.float, device = device)
            diff_sem = semantics[x] - semantics[y]
            classifier = self.both_net if bundle.question_type == 1 else self.select_net
            final_loss = 0.2 * torch.nn.functional.binary_cross_entropy_with_logits(classifier(diff_sem).squeeze(-1), ans.to(device))
            # print(ans_loss)
        return hop_loss, ans_loss, final_loss

    

if __name__ == "__main__":
    BERT_MODEL = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
    orig_text =  ''.join(["Theatre Centre is a UK-based theatre company touring new plays for young audiences aged 4 to 18, founded in 1953 by Brian Way, the company has developed plays by writers including which British writer, dub poet and Rastafarian?",
                    " It is the largest urban not-for-profit theatre company in the country and the largest in Western Canada, with productions taking place at the 650-seat Stanley Industrial Alliance Stage, the 440-seat Granville Island Stage, the 250-seat Goldcorp Stage at the BMO Theatre Centre, and on tour around the province."
    ])
    # orig_text = 'Brian \"Boosh\" Boucher (pronounced \"Boo-shay\") (born January 2, 1977) is a retired American professional ice hockey goaltender, who played 13 seasons in the National Hockey League (NHL) for the Philadelphia Flyers, Phoenix Coyotes, Calgary Flames, Chicago Blackhawks, Columbus Blue Jackets, San Jose Sharks, and Carolina Hurricanes.'
    tokenized_text = tokenizer.tokenize(orig_text)
    print(len(tokenized_text))
    # results = find_start_end_after_tokenized(tokenizer, tokenized_text, ['Boosh\" Boucher', 'January 2, 1977', 'Philadelphia Flyers'])
    # tokenized_spans = []
    # for start, end in results:
    #     print(tokenized_text[start: end + 1])
    #     tokenized_spans.append(tokenized_text[start: end + 1])
    # print('Before...')
    # results = find_start_end_before_tokenized(orig_text, tokenized_spans)
    # for start, end in results:
    #     print(orig_text[start:end])













# class BertForMultiHopQuestionAnswering(PreTrainedBertModel):
#     def __init__(self, config):
#         super(BertForMultiHopQuestionAnswering, self).__init__(config)
#         self.bert = BertModel(config)
#         self.qa_outputs = nn.Linear(config.hidden_size, 2)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.gold_classifier = nn.Linear(config.hidden_size, 1)
#         self.ans_classifier = nn.Linear(config.hidden_size, 3)
#         self.apply(self.init_bert_weights)

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, gold_label = None, ans_label = None, start_weight=None, end_weight=None):
#         sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         # Judge whether sentence B is gold sentence or not
#         pooled_output = self.dropout(pooled_output)
#         pooled_logits = self.gold_classifier(pooled_output) # [batch_size, 1]
#         loss_bce = torch.nn.BCEWithLogitsLoss()
#         loss_ce = torch.nn.CrossEntropyLoss()
#         if gold_label is not None and ans_label is not None:
#             gold_loss = loss_bce(pooled_logits.squeeze(-1), gold_label)
#             pooled_logits2 = self.ans_classifier(pooled_output) * gold_label.unsqueeze(1) # If sentence[i] is not gold sentence, ans_label[i] == 0
#             ans_loss = loss_ce(pooled_logits2, ans_label)
        
#         # Predict spans
#         logits = self.qa_outputs(sequence_output)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)

#         if start_weight is not None and end_weight is not None:
#             # # If we are on multi-GPU, split add a dimension
#             # if len(start.size()) > 1:
#             #     start = start.squeeze(-1)
#             # if len(end.size()) > 1:
#             #     end = end.squeeze(-1)
#             # # sometimes the start/end positions are outside our model inputs, we ignore these terms
#             # ignored_index = start_logits.size(1)
#             # start.clamp_(0, ignored_index)
#             # end.clamp_(0, ignored_index)

#             lgsf = torch.nn.LogSoftmax(dim = 1) # If ans_label[i] == 0, start_weight = end_weight = 0(vec)
#             start_loss = - torch.sum(start_weight * lgsf(start_logits), dim = -1)
#             end_loss = - torch.sum(end_weight * lgsf(end_logits), dim = -1)
#             span_loss = (start_loss + end_loss) / 2
#             return torch.mean(span_loss), torch.mean(gold_loss), torch.mean(ans_loss)
#         else:
#             return pooled_logits.squeeze(-1), start_logits, end_logits

