import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, AdamW, \
    get_linear_schedule_with_warmup
from consort.bertclassification.data import data_load, colloate_fn, guide_load
from consort.bertclassification.models.bert_model import BERT
from sklearn.metrics import classification_report, roc_auc_score
from consort.bertclassification.config import Config
from argparse import ArgumentParser

p = '/efs/lanj3/20220109_220339/'
files = sorted(os.listdir(p))
num = []
for f in files:
    path = os.path.join(p, f)
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', default=path+"/test.csv",
                        help='Path to the input file')
    parser.add_argument('-o', '--output', default=path, help='Path to the output file')
    parser.add_argument('-m', '--model', default=path+"/best.mdl", help='Path to the model file')
    parser.add_argument('-b', '--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--gpu', action='store_true', default=1, help='Use GPU')
    parser.add_argument('-d', '--device', type=int, default=0,
                        help='GPU device index (for multi-GPU machines)')
    args = parser.parse_args()

    map_location = 'cuda:{}'.format(args.device) if args.gpu else 'cpu'
    if args.gpu:
        torch.cuda.set_device(args.device)

    # load saved weights, config, vocabs and valid patterns
    state = torch.load(args.model, map_location=map_location)

    config = state['config']
    # config['threshold_guide'] = 1
    if type(config) is dict:
        config = Config.from_dict(config)
    # config = Config.from_json_file('models/config.json')
    test, column = data_load([args.input], config)

    model = BERT(config)
    model.load_bert(config.bert_model_name)
    model.load_state_dict(state['model'])

    if args.gpu:
        model.cuda(args.device)

    batch_num = len(test) // args.batch_size + (len(test) % args.batch_size)

    all_result = []
    embedding = []
    targets = []
    progress = tqdm.tqdm(total=batch_num, ncols=75)
    for param in model.parameters():
        param.requires_grad = False

    bert_layer = list(model.children())[-1]
    # fc_layer = list(model.children())[1]
    # sig_layer = nn.Sigmoid()


    for batch in DataLoader(test, batch_size=config.eval_batch_size, shuffle=False,
                            collate_fn=colloate_fn):
        model.eval()
        output = bert_layer(batch.text_ids, attention_mask=batch.attention_mask_text)[1]
        # output = fc_layer(output)
        # result = sig_layer(output)
        embedding.extend(output.tolist())

        if config.guide:
            guidance = guide_load(config)
            sim, combine, result = model(batch, guidance)
        else:
            result = model(batch)

        target = batch.labels
        if config.use_gpu:
            result = result.cpu().data
            target = target.cpu().data
        if config.guide:
            result = np.where(result.numpy() > config.threshold_guide, 1, 0)
        else:
            result = np.where(result.numpy() > 0.5, 1, 0)
        #
        targets.extend(target.tolist())
        all_result.extend(result.tolist())

        progress.update(1)
    progress.close()

    # print(targets)
    # print(all_result)
    # column = ['10', '11a', '11b', '12a', '12b', '3a', '3b', '4a', '4b', '5', '6a',
    #           '6b', '7a', '7b', '8a', '8b', '9']
    labels = []
    for inst in all_result:
        label = []
        for j in range(len(inst)):
            if inst[j] == 1:
                label.append(column[0][j])
        if len(label) == 0:
            label.append('0')
        labels.append(label)
    report = classification_report(all_result, targets, target_names=column[0], digits=4)
    print(report)
    # print(o)
    num.append(roc_auc_score(all_result, targets, average = "micro"))
    # result = json.dumps({'epoch': 0, 'train_loss': 0, 'test': report})
    # best = json.dumps({'best epoch': 0})
    # log_file = args.input[:-8]+'log_1.txt'
    # with open(log_file, 'a', encoding='utf-8') as w:
    #     w.write(result + '\n')
    #     w.write(best + '\n')

    file = pd.read_csv(args.input, header=0)
    file['Prediction'] = labels
    # file['embedding'] = embedding
    # file.to_csv(args.input, index=False)
print(sum(num)/len(num))