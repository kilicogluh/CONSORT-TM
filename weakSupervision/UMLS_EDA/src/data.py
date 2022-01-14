import torch
from collections import namedtuple
from transformers import BertTokenizer
from collections import Counter
import ast, math, random
import os
import data_aug.UMLS_EDA.augment4class as augment4class
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd


batch_fields = ['text_ids', 's_id', 'attention_mask_text', 'labels', 'index']
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))

instance_fields = ['text', 'text_ids', 'attention_mask_text',
                   'labels', 't_labels', 'PMCID', 's_id', 'index']
Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))


def process_data_for_bert(tokenizer, data, max_length=128):
    instances = []
    c = 0
    for i in data:
        text_ids = tokenizer.encode(i[0],
                                    add_special_tokens=True,
                                    truncation=True,
                                    max_length=max_length)

        pad_num = max_length - len(text_ids)
        attn_mask = [1] * len(text_ids) + [0] * pad_num
        text_ids = text_ids + [0] * pad_num
        labels = list(i[1])
        instance = Instance(
            text=i[0],
            text_ids=text_ids,
            attention_mask_text=attn_mask,
            labels=labels,
            t_labels=i[2],
            PMCID=i[3],
            s_id=i[4],
            index=c
        )
        instances.append(instance)
        c += 1
    return instances


def colloate_fn(batch, gpu=True):
    batch_text_idxs = []
    batch_attention_masks_text = []
    batch_labels = []
    batch_index = []
    batch_s_id = []
    for inst in batch:
        batch_text_idxs.append(inst.text_ids)
        batch_attention_masks_text.append(inst.attention_mask_text)
        batch_labels.append(inst.labels)
        batch_index.append(inst.index)
        batch_s_id.append(inst.s_id)
    if gpu:
        batch_text_idxs = torch.cuda.LongTensor(batch_text_idxs)
        batch_attention_masks_text = torch.cuda.FloatTensor(batch_attention_masks_text)
        batch_labels = torch.cuda.FloatTensor(batch_labels)
        batch_index = torch.cuda.LongTensor(batch_index)
        batch_s_id = torch.cuda.IntTensor(batch_s_id)
    else:
        batch_text_idxs = torch.LongTensor(batch_text_idxs)
        batch_attention_masks_text = torch.FloatTensor(batch_attention_masks_text)
        batch_labels = torch.FloatTensor(batch_labels)
        batch_index = torch.LongTensor(batch_index)
        batch_s_id = torch.IntTensor(batch_s_id)
    return Batch(
        text_ids=batch_text_idxs,
        s_id=batch_s_id,
        attention_mask_text=batch_attention_masks_text,
        labels=batch_labels,
        index=batch_index
    )


def convert(x):
    label = x.split(', ')
    label = [i.strip() for i in label]
    return label


def unique(section_header):
    headers = []
    for i in section_header:
        if i not in headers:
            headers.append(i)
    return headers


def data_load(path, config):
    data = []
    concat_file = pd.DataFrame()
    for i in path:
        file = pd.read_csv(i, header=0)
        if 'snorkel' in i.lower():
            file['CONSORT_item'] = file['CONSORT_Item'].apply(lambda x: ast.literal_eval(x))
            concat_file = concat_file.append(file[['PMID', 'text', 'CONSORT_item']])
        else:
            file['CONSORT_item'] = file['CONSORT_Item'].apply(lambda x: ast.literal_eval(x))
            concat_file = concat_file.append(file[['PMCID', 'sentence_id', 'text', 'CONSORT_item', 'section']])

    # concat_file = concat_file.drop_duplicates(subset='text')
    label_file = pd.read_csv('/efs/CONSORT/CONSORT-TM/data/Methods_train.csv', header=0)
    label_file = label_file.append(pd.read_csv('/efs/CONSORT/CONSORT-TM/data/Methods_test.csv', header=0))
    label_file['CONSORT_item'] = label_file['CONSORT_Item'].apply(lambda x: ast.literal_eval(x))
    labels = list(set([i for j in label_file['CONSORT_item'] for i in j]))

    labels.sort()
    label = MultiLabelBinarizer()
    list_name = [labels[1:]]
    label.fit(list_name)
    for i in concat_file.iterrows():
        # x = i[1].text
        x = i[1].section + ' ' + i[1].text
        y = label.transform([i[1].CONSORT_item])
        section = i[1].section
        data.append([x, y[0], i[1].CONSORT_item, i[1].PMCID, int(i[1].sentence_id[1:]), section])
        # data.append([x, y[0], i[1].CONSORT_item, i[1].PMID, 0])
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name, do_lower_case=True)
    data = process_data_for_bert(tokenizer, data)
    return data[:10], list_name


def data_aug(config, train, log_dir=None, aug_file=None):
    countFrequency = Counter([j for i in train for j in i.t_labels if j != '0'])
    augLabel = dict()
    for i in countFrequency:
        if countFrequency[i] < config.threshold:
            augLabel[i] = [math.ceil(config.threshold/countFrequency[i])-1, config.threshold-countFrequency[i]]

    augData = []
    methods = []
    augtofile = []
    for i in augLabel:
        augsample = [augment4class.eda(j.text, alpha_sr=0.2, alpha_ri=0.2, alpha_rs=0.2, p_rd=0.2, alpha_umls=0.5,
                                       num_aug=augLabel[i][0])[:-1] for j in train if i in j.t_labels]
        texts = [j for k in augsample for j in k]
        labels = [[j.labels] * augLabel[i][0] for j in train if i in j.t_labels]
        labels = [j for k in labels for j in k]
        t_labels = [[j.t_labels] * augLabel[i][0] for j in train if i in j.t_labels]
        t_labels = [j for k in t_labels for j in k]
        PMCID = [[j.PMCID] * augLabel[i][0] for j in train if i in j.t_labels]
        PMCID = [j for k in PMCID for j in k]
        sid = [[j.s_id] * augLabel[i][0] for j in train if i in j.t_labels]
        sid = [j for k in sid for j in k]
        index = random.sample(range(len(texts)), augLabel[i][1]) if augLabel[i][1] < len(texts) else range(len(texts))
        for j in index:
            augData.append([texts[j][0], labels[j], t_labels[j], PMCID[j], sid[j]])
            methods.append(texts[j][1])
            augtofile.append([PMCID[j], sid[j], texts[j][0], t_labels[j], 1, texts[j][1]])
    # save training file
    traintofile = [[t.PMCID, t.s_id, t.text, t.t_labels, 0, ''] for t in train]
    file = pd.DataFrame(traintofile+augtofile, columns=['PMCID', 'sentence_id', 'text', 'CONSORT_item', 'augment', 'methods'])
    file = file.sort_values(by=['PMCID', 'sentence_id'],ascending=True)
    file.to_csv(os.path.join(log_dir, 'aug.csv'), index=False)

    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name, do_lower_case=True)
    augData = process_data_for_bert(tokenizer, augData)
    return train+augData


def data_snorkel(config):
    path = '/efs/CONSORT2/data/classification_data/train_SNORKEL_500instancesPerLabel_06082021.csv'
    file = pd.read_csv(path, header=0)
    file['CONSORT_item'] = file['CONSORT_item'].apply(lambda x: ast.literal_eval(x))
    file = file[['PMID', 'text', 'CONSORT_item']]
    data = []
    label = MultiLabelBinarizer()
    list_name = [['10', '11a', '11b', '12a', '12b', '3a', '3b', '4a', '4b', '5', '6a',
                  '6b', '7a', '7b', '8a', '8b', '9']]
    label.fit(list_name)

    for i in file.iterrows():
        x = i[1].text
        y = label.transform([i[1].CONSORT_item])
        data.append([x, y[0], i[1].CONSORT_item, i[1].PMID, None])
        # data.append([x, y[0], i[1].CONSORT_item, i[1].PMID, 0])
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name, do_lower_case=True)
    data = process_data_for_bert(tokenizer, data)
    return data


def load_umlsaug(path, config):
    train = pd.read_csv(path+'/aug.csv', header=0)
    train['CONSORT_item'] = train['CONSORT_item'].apply(lambda x: ast.literal_eval(x))
    test = pd.read_csv(path+'/test.csv', header=0)
    test['CONSORT_item'] = test['CONSORT_Item'].apply(lambda x: ast.literal_eval(x))
    train_data = []
    test_data = []

    label = MultiLabelBinarizer()
    list_name = [['10', '11a', '11b', '12a', '12b', '3a', '3b', '4a', '4b', '5', '6a',
                  '6b', '7a', '7b', '8a', '8b', '9']]
    label.fit(list_name)

    for i in train.iterrows():
        x = i[1].text
        y = label.transform([i[1].CONSORT_item])
        train_data.append([x, y[0], i[1].CONSORT_item, i[1].PMCID, i[1].sentence_id])

    for i in test.iterrows():
        x = i[1].section + ' ' + i[1].text
        y = label.transform([i[1].CONSORT_item])
        test_data.append([x, y[0], i[1].CONSORT_item, i[1].PMCID, i[1].sentence_id])

    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name, do_lower_case=True)
    train_data = process_data_for_bert(tokenizer, train_data)
    test_data = process_data_for_bert(tokenizer, test_data)

    return train_data, test_data
