from config import MODEL, TEST
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer, RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm

BATCH_SIZE = 4
EARLY_STOPPING = 3
WARMUP_STEPS = 10_000
LR=1e-6

STEPS_EVAL = 5_000
if TEST:
    STEPS_EVAL = 100
    WARMUP_STEPS = 10


with open('train_dataset.pickle','rb') as f_p:
    train_dataset = pickle.load(f_p)

with open('eval_dataset_small.pickle','rb') as f_p:
    eval_dataset_small = pickle.load(f_p)

with open('eval_dataset_full.pickle','rb') as f_p:
    eval_dataset_full = pickle.load(f_p)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
eval_dataloader_small = DataLoader(eval_dataset_small, batch_size=BATCH_SIZE)
eval_dataloader_full = DataLoader(eval_dataset_full, batch_size=BATCH_SIZE)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)
optimizer = AdamW(model.parameters(), lr=LR)


num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=num_training_steps
)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


progress_bar = tqdm(range(num_training_steps))
model.train()

model.train()
model.to(device)

def transform_batch(batch):
        batch['input_ids'] = torch.stack(batch['input_ids']).permute(1,0).to(device)
        batch['attention_mask'] = torch.stack(batch['attention_mask']).permute(1,0).to(device)
        batch['labels'] = batch['year_scaled'].to(device).float()

        batch['labels'].to(device)
        batch['input_ids'].to(device)
        batch['attention_mask'].to(device)

        for c in set(batch.keys()) - {'input_ids', 'attention_mask', 'labels'}:
            del batch[c]
        return batch


def eval(full = False):
    model.eval()
    eval_loss = 0.0
    dataloader = eval_dataloader_full if full else eval_dataloader_small
    for i, batch in enumerate(eval_dataloader_small):
        batch = transform_batch(batch)
        outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.item()
    print(f'epoch {epoch} eval loss: {eval_loss /  i }')
    model.train()
    return eval_loss


best_eval_loss = 9999
epochs_without_progress = 0
for epoch in range(num_epochs):
    train_loss = 0.0
    for i, batch in enumerate(train_dataloader):
        batch = transform_batch(batch)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        train_loss += loss.item()
        progress_bar.update(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if i % STEPS_EVAL == 0 and i > 1 :
            print(f' epoch {epoch} train loss: {train_loss /  STEPS_EVAL }', end = '\t\t')
            train_loss = 0.0
            eval(full = False)

    model.save_pretrained(f'roberta_year_prediction/epoch_{epoch}')
    eval_loss =  eval(full=True)

    if eval_loss < best_eval_loss:
        model.save_pretrained(f'roberta_year_prediction/epoch_best')
        best_eval_loss = eval_loss
    else:
        epochs_without_progress += 1

    if epochs_without_progress > EARLY_STOPPING:
        break
