#! -*- coding:utf-8 -*-


import json
from tqdm import tqdm
import codecs

rel_set = set()


train_data = []
dev_data = []
test_data = []

def get_triples(json_line):
    txt = ' '.join(json_line['token'])
    head = json_line['h']['name']
    tail = json_line['t']['name']
    relation = json_line['relation']
    new_line = {}
    new_line['text'] = txt.lstrip('\"').strip('\r\n').rstrip('\"')
    new_line['triple_list']=[[head, relation, tail]]
    return new_line
    

with open('semeval_train.txt') as f:
    for l in tqdm(f):
        a = json.loads(l)
        line = get_triples(a)
        if line['triple_list'][0][1] == 'Other':
            continue
        # line = {
        #         'text': a['sentText'].lstrip('\"').strip('\r\n').rstrip('\"'),
        #         'triple_list': [(i['em1Text'], i['label'], i['em2Text']) for i in a['relationMentions'] if i['label'] != 'None']
        #        }
        if not line['triple_list']:
            continue
        train_data.append(line)


with open('semeval_val.txt') as f:
    for l in tqdm(f):
        a = json.loads(l)
        line = get_triples(a)
        if line['triple_list'][0][1] == 'Other':
            continue
        # line = {
        #         'text': a['sentText'].lstrip('\"').strip('\r\n').rstrip('\"'),
        #         'triple_list': [(i['em1Text'], i['label'], i['em2Text']) for i in a['relationMentions'] if i['label'] != 'None']
        #        }
        if not line['triple_list']:
            continue
        dev_data.append(line)

cnt = 0
with open('semeval_test.txt') as f:
    for l in tqdm(f):
        a = json.loads(l)
        line = get_triples(a)
        if line['triple_list'][0][1] == 'Other':
            continue
        # line = {
        #         'text': a['sentText'].lstrip('\"').strip('\r\n').rstrip('\"'),
        #         'triple_list': [(i['em1Text'], i['label'], i['em2Text']) for i in a['relationMentions'] if i['label'] != 'None']
        #        }
        if not line['triple_list']:
            continue
        test_data.append(line)

with open('semeval_rel2id.json') as f:
    rel2id = json.load(f)
    for rel in rel2id.keys():
        rel_set.add(rel)
        
id2predicate = {i:j for i,j in enumerate(sorted(rel_set))}
predicate2id = {j:i for i,j in id2predicate.items()}


with codecs.open('rel2id.json', 'w', encoding='utf-8') as f:
    json.dump([id2predicate, predicate2id], f, indent=4, ensure_ascii=False)


with codecs.open('train_triples.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)


with codecs.open('dev_triples.json', 'w', encoding='utf-8') as f:
    json.dump(dev_data, f, indent=4, ensure_ascii=False)


with codecs.open('test_triples.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)
