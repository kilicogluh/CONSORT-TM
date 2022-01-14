import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, recall_score
from sklearn.metrics import classification_report
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from models.bert_model import BERT
from config import Config
from data import colloate_fn, data_load, data_aug, data_snorkel
import torch
from torch.utils.data import DataLoader
import tqdm
import time, os, json
from sklearn.model_selection import GroupKFold, KFold


def evaluate(model, Loss, data, config, batch_num, epoch, name, list_name, guidance=None, log_dir=None):
    progress = tqdm.tqdm(total=batch_num, ncols=75, desc='{} {}'.format(name, epoch))
    pred_result = []
    target_result = []
    running_loss = 0.0
    for batch in DataLoader(data, batch_size=config.eval_batch_size, shuffle=False,
                            collate_fn=colloate_fn):
        model.eval()
        target = batch.labels
        pred = model(batch)
        loss = Loss(pred, target)
        running_loss += loss.item()
        if config.use_gpu:
            target = target.cpu().data
            pred = pred.cpu().data

        pred = np.where(pred.numpy() > 0.5, 1, 0)
        target_result.extend(target.tolist())
        pred_result.extend(pred.tolist())
        progress.update(1)
    progress.close()
    column = list_name[0]
    f1 = round(f1_score(target_result, pred_result, average="micro", zero_division=0), 4)
    recall = round(recall_score(target_result, pred_result, average="micro", zero_division=0), 4)
    report = classification_report(target_result, pred_result, target_names=column, digits=4, zero_division=0)
    print(report)
    return f1, recall, report, running_loss


def train_bert(config):

    train, list_name = data_load(config.train_file, config)
    test = data_load(config.test_file, config)
    all = train + test
    groups = [i.PMCID for i in all]
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_home_dir = os.path.join(config.log_path, timestamp)
    os.mkdir(log_home_dir)
    kf = GroupKFold(n_splits=5)

    for train_index, test_index in kf.split(all, groups=groups):

        train = [all[i] for i in train_index]
        test = [all[i] for i in test_index]
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_dir = os.path.join(log_home_dir, timestamp)
        os.mkdir(log_dir)
        log_file = os.path.join(log_dir, 'log.txt')
        best_model = os.path.join(log_dir, 'best.mdl')
        with open(log_file, 'w', encoding='utf-8') as w:
            w.write(json.dumps(config.to_dict()) + '\n')

        if config.augmentation:
            train = data_aug(config, train, log_dir)
            # train = train + data_snorkel(config)
            print('INFO: Augmentation finished.')

        use_gpu = config.use_gpu
        if use_gpu and config.gpu_device >= 0:
            torch.cuda.set_device(config.gpu_device)

        model = BERT(config)
        model.load_bert(config.bert_model_name)
        batch_num = len(train) // config.batch_size
        total_steps = batch_num * config.max_epoch
        # dev_batch_num = len(dev) // config.eval_batch_size + (len(dev) % config.eval_batch_size != 0)
        test_batch_num = len(test) // config.eval_batch_size + (len(test) % config.eval_batch_size != 0)
        if use_gpu:
            model.cuda()
        param_groups = [
            {
                'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
                'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() if not n.startswith('bert')],
                'lr': config.learning_rate, 'weight_decay': config.weight_decay
            },
        ]

        optimizer = AdamW(params=param_groups, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
        )
        Loss = torch.nn.BCELoss()
        best_loss, best_epoch = 100, 0

        for epoch in range(config.max_epoch):
            running_loss = 0.0
            progress = tqdm.tqdm(total=batch_num, ncols=75, desc='Train {}'.format(epoch))
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(DataLoader(
                    train, batch_size=config.batch_size,
                    shuffle=True, collate_fn=colloate_fn)):
                optimizer.zero_grad()
                model.train()
                prediction = model(batch)
                loss = Loss(prediction, batch.labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
                progress.update(1)

            progress.close()
            print('INFO: Training Loss is ', round(running_loss/len(train), 4))

            if running_loss/len(train) < best_loss:
                best_loss = running_loss/len(train)
                best_epoch = epoch
                f1_test, recall_test, report_test, test_loss = evaluate(model, Loss, test, config, test_batch_num,
                                                                            epoch, 'TEST', list_name, None, log_dir)
                result = json.dumps({'epoch': epoch, 'train_loss': best_loss, 'test': report_test})
                torch.save(dict(model=model.state_dict(), config=config.to_dict()), best_model)

            else:
                result = json.dumps({'epoch': epoch, 'train_loss': running_loss/len(train)})

            with open(log_file, 'a', encoding='utf-8') as w:
                w.write(result + '\n')
            print('INFO: Log file: ', log_file)
        model.train()
        best = json.dumps({'best epoch': best_epoch})
        with open(log_file, 'a', encoding='utf-8') as w:
            w.write(best + '\n')


if __name__ == '__main__':
    config = Config.from_json_file('models/config.json')
    train_bert(config)
