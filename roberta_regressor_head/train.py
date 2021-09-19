import os
import torch
import random
import copy
from fairseq.models.roberta import RobertaModel, RobertaHubInterface
from fairseq import hub_utils
from fairseq.data.data_utils import collate_tokens
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler


EVAL_OFTEN = True
EVAL_EVERY = 3000
BATCH_SIZE = 25
model_type = 'base' # base or large



roberta = torch.hub.load('pytorch/fairseq', f'roberta.{model_type}')
roberta.cuda()
device='cuda'


# LOAD DATA
train_in = [l.rstrip('\n') for l in open('../train/in.tsv',newline='\n').readlines()] # shuffled
dev_in = [l.rstrip('\n') for l in open('../dev-0/in.tsv',newline='\n').readlines()] # shuffled

train_year = [float(l.rstrip('\n')) for l in open('../train/expected.tsv',newline='\n').readlines()]
dev_year = [float(l.rstrip('\n')) for l in open('../dev-0/expected.tsv',newline='\n').readlines()]

dev_in_not_shuffled = copy.deepcopy(dev_in) # not shuffled
test_in = [l.rstrip('\n') for l in open('../test-A/in.tsv',newline='\n').readlines()] # not shuffled

# SHUFFLE DATA
c = list(zip(train_in,train_year))
random.shuffle(c)
train_in, train_year = zip(*c) 
c = list(zip(dev_in,dev_year))
random.shuffle(c)
dev_in, dev_year = zip(*c) 

# SCALE DATA
scaler = MinMaxScaler()
train_year_scaled = scaler.fit_transform(np.array(train_year).reshape(-1,1))
dev_year_scaled = scaler.transform(np.array(dev_year).reshape(-1,1))


class RegressorHead(torch.nn.Module):
    def __init__(self):
        super(RegressorHead, self).__init__()
        in_dim = 768 if model_type == 'base' else 1024
        self.linear = torch.nn.Linear(in_dim, 1)
        self.m = torch.nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.linear(x)
        x = self.m(x)
        x = - self.m(-x + 1 ) +1
        return x 

def get_features_and_year(dataset_in,dataset_y):
    for i in tqdm(range(0,len(dataset_in), BATCH_SIZE)):
        batch_of_text = dataset_in[i:i+BATCH_SIZE]
        
        batch = collate_tokens([roberta.encode(p)[:512] for p in batch_of_text], pad_idx=1)
        features = roberta.extract_features(batch).mean(1)
        years = torch.FloatTensor(dataset_y[i:i+BATCH_SIZE]).to(device)

        yield features, years

def eval_dev(short=False):
    criterion_eval = torch.nn.MSELoss(reduction='sum')
    roberta.eval()
    regressor_head.eval()

    loss = 0.0
    loss_clipped = 0.0
    loss_scaled = 0.0

    if short:
        dataset_in = dev_in[:1000]
        dataset_years = dev_year_scaled[:1000]
    else:
        dataset_in = dev_in
        dataset_years = dev_year_scaled

    predictions_sum = 0
    for batch, year in tqdm(get_features_and_year(dataset_in, dataset_years)):

        predictions_sum += year.shape[0]
        x = regressor_head(batch.to(device))
        x_clipped = torch.clamp(x,0.0,1.0)

        original_x = torch.FloatTensor(scaler.inverse_transform(x.detach().cpu().numpy().reshape(1,-1)))
        original_x_clipped = torch.FloatTensor(scaler.inverse_transform(x_clipped.detach().cpu().numpy().reshape(1,-1)))
        original_year =  torch.FloatTensor(scaler.inverse_transform(year.detach().cpu().numpy().reshape(1,-1)))

        loss_scaled += criterion_eval(x, year).item()
        loss += criterion_eval(original_x, original_year).item()
        loss_clipped += criterion_eval(original_x_clipped, original_year).item()

    print('valid loss scaled: ' + str(np.sqrt(loss_scaled/predictions_sum)))
    print('valid loss: ' + str(np.sqrt(loss/predictions_sum)))
    print('valid loss clipped: ' + str(np.sqrt(loss_clipped/predictions_sum)))



def train_one_epoch():
    roberta.train()
    regressor_head.train()
    loss_value=0.0
    iteration = 0
    for batch, year in get_features_and_year(train_in,train_year_scaled):
        iteration +=1
        roberta.zero_grad()
        regressor_head.zero_grad()

        predictions = regressor_head(batch.to(device))

        loss = criterion(predictions, year)
        loss_value += loss.item()
        loss.backward()
        optimizer.step()

        roberta.zero_grad()
        regressor_head.zero_grad()


        if EVAL_OFTEN and (iteration > 1) and (iteration % EVAL_EVERY == 1):
            print('train loss: ' + str(np.sqrt(loss_value / (EVAL_EVERY*BATCH_SIZE))))
            eval_dev(True)
            roberta.train()
            regressor_head.train()
            loss_value = 0.0


def predict(dataset='dev'):
    if dataset=='dev':
        f_out_path = '../dev-0/out.tsv'
        dataset_in_not_shuffled = dev_in_not_shuffled
    elif dataset=='test':
        f_out_path = '../test-A/out.tsv'
        dataset_in_not_shuffled = test_in
    roberta.eval()
    regressor_head.eval()
    f_out = open(f_out_path,'w')
    for batch, year in tqdm(get_features_and_year(dataset_in_not_shuffled, dev_year_scaled)):
        x = regressor_head(batch)
        x_clipped = torch.clamp(x,0.0,1.0)
        original_x_clipped =  scaler.inverse_transform(x_clipped.detach().cpu().numpy().reshape(1,-1))
        for y in original_x_clipped[0]:
            f_out.write(str(y) + '\n')
    f_out.close()


regressor_head = RegressorHead().to(device)

optimizer = torch.optim.Adam(list(roberta.parameters()) + list(regressor_head.parameters()), lr=1e-6)
criterion = torch.nn.MSELoss(reduction='sum').to(device)


for i in range(100):
    print('epoch ' + str(i))
    train_one_epoch()

    print(f'epoch {i} done, EVALUATION ON FULL DEV:')
    eval_dev()
    print('evaluation done')
    predict('dev')
    predict('test')

    torch.save(roberta.state_dict(),'checkpoints/roberta_to_regressor' + str(i) + '.pt')
    torch.save(regressor_head.state_dict(),'checkpoints/regressor_head' + str(i) + '.pt')


roberta.load_state_dict(torch.load('checkpoints/roberta_to_regressor1.pt'))
regressor_head.load_state_dict(torch.load('checkpoints/regressor_head1.pt'))
predict('dev')
predict('test')
