from config import MODEL, TEST
import pickle
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(MODEL)
def tokenize_function(examples):
    t = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    return t

def get_dataset_dict(dataset):
    with open(dataset) as f_in:
        next(f_in)
        d = dict()
        d['year_cont'] = list()
        d['year'] = list()
        d['month'] = list()
        d['day'] = list()
        d['weekday'] = list()
        d['day_of_year'] = list()
        d['text'] = list()
        for l in f_in:
            yc,y,m,day,w,dy,t= l.rstrip().split('\t')
            d['year_cont'].append(yc)
            d['year'].append(int(y))
            d['month'].append(int(m))
            d['day'].append(int(day))
            d['weekday'].append(int(w))
            d['day_of_year'].append(int(dy))
            d['text'].append(t)
    return d

train_dataset = Dataset.from_dict(get_dataset_dict('train_huggingface_format.csv')).map(tokenize_function, batched=True).shuffle(seed=42)
eval_dataset_full = Dataset.from_dict(get_dataset_dict('dev-0_huggingface_format.csv')).map(tokenize_function, batched=True)
eval_dataset_small = eval_dataset_full.shuffle(seed=42).select(range(2000))
test_dataset_A = Dataset.from_dict(get_dataset_dict('test-A_huggingface_format.csv')).map(tokenize_function, batched=True)

if TEST:
    train_dataset = train_dataset.select(range(500))
    eval_dataset_full = eval_dataset_full.select(range(400))
    eval_dataset_small = eval_dataset_small.select(range(50))
    test_dataset_A = test_dataset_A.select(range(200))

scalers = dict()
scalers['year'] = MinMaxScaler().fit(np.array(train_dataset['year']).reshape(-1,1))

def add_scaled(example):
    for factor in ('year',):
        example[factor + '_scaled'] = scalers[factor].transform(np.array(example[factor]).reshape(-1,1)).reshape(1,-1)[0].item()
    return example

train_dataset = train_dataset.map(add_scaled)
eval_dataset_full = eval_dataset_full.map(add_scaled)
eval_dataset_small = eval_dataset_small.map(add_scaled)
test_dataset_A = test_dataset_A.map(add_scaled)


with open('train_dataset.pickle','wb') as f_p:
    pickle.dump(train_dataset, f_p)

with open('eval_dataset_small.pickle','wb') as f_p:
    pickle.dump(eval_dataset_small, f_p)

with open('eval_dataset_full.pickle','wb') as f_p:
    pickle.dump(eval_dataset_full, f_p)

with open('test_dataset_A.pickle','wb') as f_p:
    pickle.dump(test_dataset_A, f_p)

with open('scalers.pickle','wb') as f_p:
    pickle.dump(scalers, f_p)
