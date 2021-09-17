import numpy as np

train_exp_f = open('../train/expected.tsv')
train_mean = np.mean([float(a.rstrip()) for a in train_exp_f])
train_mean_line = str(train_mean) + '\n'


def pred(in_f_path, out_f_path):
    f_in = open(in_f_path, 'r')
    f_out = open(out_f_path, 'w')

    for l in f_in:
        f_out.write(train_mean_line)

    f_in.close()
    f_out.close()

pred('../dev-0/in.tsv', '../dev-0/out.tsv')
pred('../test-A/in.tsv', '../test-A/out.tsv')
