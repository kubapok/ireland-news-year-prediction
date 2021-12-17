import pickle
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

#with open('train_dataset.pickle','rb') as f_p:
#    train_dataset = pickle.load(f_p)
#
#with open('eval_dataset_small.pickle','rb') as f_p:
#    eval_dataset_small = pickle.load(f_p)
#
#with open('eval_dataset_full.pickle','rb') as f_p:
#    eval_dataset_full = pickle.load(f_p)
#
#with open('test_dataset_A.pickle','rb') as f_p:
#    test_dataset_A = pickle.load(f_p)

with open('dev-0_huggingface_format.csv','r') as f_p:
    eval_dataset_full = f_p.readlines()

with open('test-A_huggingface_format.csv','r') as f_p:
    test_dataset = f_p.readlines()

device = 'cuda'
model = AutoModelForSequenceClassification.from_pretrained('./roberta_year_prediction/epoch_best')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model.eval()
model.to(device)

with open('scalers.pickle', 'rb') as f_scaler:
    scalers = pickle.load(f_scaler)

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
def predict(dataset, out_f):
    outputs = []

    for sample in tqdm(dataset[1:]):
        y, t = sample.split('\t')
        t = t.rstrip()

        t = tokenizer(t, padding="max_length", truncation=True, max_length=512, return_tensors='pt').to('cuda')

        outputs.extend(model(**t).logits.tolist())
        outputs_transformed = scalers['year'].inverse_transform(outputs)

    with open(out_f,'w') as f_out:

        for o in outputs_transformed:
            f_out.write(str(o[0]) + '\n')

predict(eval_dataset_full, '../dev-0/out.tsv')
predict(test_dataset, '../test-A/out.tsv')
