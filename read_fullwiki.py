import bz2
import json
from tqdm import tqdm
import os
from itertools import chain
import re
import pickle as pkl
from urllib.parse import unquote
HTML_SYM = re.compile(r'<(.*?)>')
EDGE_XY = re.compile(r'<a href="(.*?)">(.*?)</a>')

def get_edges(sentence):
    ret = EDGE_XY.findall(sentence)
    return [unquote(x) + '###' + y for x, y in ret]

from redis import StrictRedis
db = StrictRedis(host='localhost', port=6379, db=0)

WIKI_PATH = './enwiki-20171001-pages-meta-current-withlinks-abstracts/'
files = [os.path.join(WIKI_PATH, dirname, filename) 
         for dirname in os.listdir(WIKI_PATH) 
         for filename in os.listdir(os.path.join(WIKI_PATH, dirname)) 
        ]

for filename in tqdm(files):
    if os.path.isfile(filename):
        with bz2.open(filename, 'rb') as fin:
            for line in fin:
                page = json.loads(line)
                if len(page['text']) >= 1:
                    db.delete(page['title'])
                    db.rpush(page['title'], *(page['text']))
                    for i, sentence in enumerate(page['text_with_links']):
                        t = get_edges(sentence)   
                        if len(t) > 0:
                            name = 'edges:###'+ str(i) + '###' + page['title']
                            db.delete(name)
                            db.rpush(name, *t)
                        
