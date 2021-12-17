import pickle
import torch
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

with open('eval_dataset_full.pickle','rb') as f_p:
    eval_dataset_full = pickle.load(f_p)

with open('test_dataset_A.pickle','rb') as f_p:
    test_dataset = pickle.load(f_p)

device = 'cuda'
model = AutoModelForSequenceClassification.from_pretrained('./roberta_year_prediction/epoch_best')
model.eval()
model.to(device)

with open('scalers.pickle', 'rb') as f_scaler:
    scalers = pickle.load(f_scaler)

def predict(dataset, out_f):
    eval_dataloader = DataLoader(dataset, batch_size=1)
    outputs = []

    progress_bar = tqdm(range(len(eval_dataloader)))

    for batch in eval_dataloader:
        batch['input_ids'] = torch.stack(batch['input_ids']).permute(1,0).to(device)
        batch['attention_mask'] = torch.stack(batch['attention_mask']).permute(1,0).to(device)
        batch['labels'] = batch['year_scaled'].to(device).float()

        batch['labels'].to(device)
        batch['input_ids'].to(device)
        batch['attention_mask'].to(device)

        for c in set(batch.keys()) - {'input_ids', 'attention_mask', 'labels'}:
            del batch[c]
        outputs.extend(model(**batch).logits.tolist())
        progress_bar.update(1)
    outputs_transformed = scalers['year'].inverse_transform(outputs)

    with open(out_f,'w') as f_out:

        for o in outputs_transformed:
            f_out.write(str(o[0]) + '\n')

predict(eval_dataset_full, '../dev-0/out.tsv')
predict(test_dataset, '../test-A/out.tsv')
