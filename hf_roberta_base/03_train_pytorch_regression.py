from config import *
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer, RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
#from transformers import AdamW
from torch.optim import Adam
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm


if TEST:
    STEPS_EVAL = 10
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
optimizer = Adam(model.parameters(), lr=LR)


num_training_steps = NUM_EPOCHS * len(train_dataloader)
#lr_scheduler = get_scheduler(
#    "linear",
#    optimizer=optimizer,
#    num_warmup_steps=WARMUP_STEPS,
#    num_training_steps=num_training_steps
#)


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
    with torch.no_grad():
        eval_loss = 0.0
        dataloader = eval_dataloader_full if full else eval_dataloader_small
        items_passed = 0 
        for i, batch in enumerate(dataloader):
            items_passed += len(batch)
            batch = transform_batch(batch)
            labels = batch['labels']
            del batch['labels']
            outputs = model(**batch)
            o = soft_clip(outputs['logits']).squeeze()
            loss = criterion(o, labels)
            eval_loss += loss.item()
        eval_loss = (eval_loss / items_passed)
        print(f'eval loss full={full}: {eval_loss:.5f}', end = '\n')
    model.train()
    return eval_loss

criterion = torch.nn.MSELoss(reduction='sum').to(device)

lrelu = torch.nn.LeakyReLU(0.1)
def soft_clip(t):
    t = lrelu(t)
    t = -lrelu(-t + 1 ) + 1
    return t

best_eval_loss = 9999
epochs_without_progress = 0
for epoch in range(NUM_EPOCHS):
    train_loss = 0.0
    items_passed = 0 
    for i, batch in enumerate(train_dataloader):
        items_passed += len(batch)
        batch = transform_batch(batch)
        labels = batch['labels']
        del batch['labels']
        outputs = model(**batch)
        o = soft_clip(outputs['logits']).squeeze()
        loss = criterion(o, labels)
        loss.backward()
        train_loss += loss.item()
        progress_bar.update(1)

        optimizer.step()
        #lr_scheduler.step()
        optimizer.zero_grad()
        model.zero_grad()

        if i % STEPS_EVAL == 0 and i > 1 :
            print(f' epoch {epoch} train loss: {(train_loss /  items_passed):.5f}', end = '\t')
            items_passed = 0
            train_loss = 0.0
            eval(full = False)

    eval_loss =  eval(full=True)
    model.save_pretrained(f'roberta_year_prediction/epoch_{epoch}_loss{eval_loss:.5f}')
    model.save_pretrained(f'roberta_year_prediction/epoch_last')

    if eval_loss < best_eval_loss:
        model.save_pretrained(f'roberta_year_prediction/epoch_best')
        print('\nsaving best model')
        best_eval_loss = eval_loss
    else:
        epochs_without_progress += 1
        print(f'epochs_witohut_progress: {epochs_without_progress}')

    if epochs_without_progress > EARLY_STOPPING:
        print('early stopping')
        break

    print(f'best_eval_loss: {best_eval_loss:5f}', end = '\n')
