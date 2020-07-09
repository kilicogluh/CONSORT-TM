import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from sklearn.metrics import roc_curve, auc, f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from simpletransformers.classification import MultiLabelClassificationModel
import random
import itertools
from ast import literal_eval
random.seed(1)
from IPython.core.interactiveshell import InteractiveShell
import argparse
InteractiveShell.ast_node_interactivity = "all"


parser = argparse.ArgumentParser(description='Multi Label Classificarion')
parser.add_argument('--data_path', help='path to .csv consisting of sentences', default = "data/Methods_train.csv")
parser.add_argument('--model_path', help='path to directory consisting of BERT model', default="models/BioBert/")
parser.add_argument('--model_save_path', help='path to directory consisting of BERT model', required=True)
parser.add_argument('--text_column', help='column name containing sentences', default="text")


args = vars(parser.parse_args())

data_path = args["data_path"]
model_path = args["model_path"]
text_col = args["text_column"]
model_save_path = args["model_save_path"]

if model_save_path == None:
    print("Please enter path to save the model in train mode")
    exit(0)

def explode_rows(id_, df):
    df[id_] = df[id_].apply(literal_eval)
    t = df[id_].apply(pd.Series).reset_index().melt(id_vars='index').dropna()[['index', 'value']].set_index('index')
    new_df = pd.merge(t,df.loc[:, df.columns != id_], left_index=True, right_index=True).rename(columns={"value":id_})
    return new_df
    
def preprocess_data(df, text_col="sentence_text", label_col="CONSORT_Item"):
    def one_hot_encoding(df, label_col):
        return df[label_col].str.get_dummies().add_prefix(label_col + "_")        
    one_hot_df = one_hot_encoding(df, label_col)
    num_labels = set(one_hot_df.columns.values)
    one_hot_df["labels"] = list(zip(*[one_hot_df[col] for col in one_hot_df]))
    df = pd.concat([df["PMCID"], df["sentence_id"], df["top_section"], df[label_col], df[text_col], one_hot_df], axis=1)
    df = df.rename(columns={text_col:'text'})
    df["text"] = df["text"].str.lower()
    df["labels"] = df["labels"].apply(lambda x: list(x))
    df = df.groupby(["PMCID", "sentence_id", "text", "top_section"]).agg(
        {"labels" : lambda x: [sum(y) for y in zip(*x)],
         label_col : lambda x: list(x)
        }
    ).reset_index()
    df["n_labels"] = df["labels"].apply(lambda x: sum(x))
    # For top_section
    one_hot_df = one_hot_encoding(df, "top_section")
    df["top_section"] = list(zip(*[one_hot_df[col] for col in one_hot_df]))
    df["top_section"] = df["top_section"].apply(lambda x: list(x))
    return df, len(num_labels)


train_df_linh = pd.read_csv(data_path)
train_df_linh["CONSORT_Item"] = train_df_linh["CONSORT_Item"].apply(lambda x: literal_eval(x))
train_df_linh["labels"] = train_df_linh["labels"].apply(lambda x: literal_eval(x))
train_df_linh["text"] = train_df_linh[["text", "section"]].apply(lambda x: x["section"] + " " + x["text"], axis=1)

labels_train = []
for l in list(itertools.chain(train_df_linh["CONSORT_Item"])):
    labels_train.extend([k for k in l])
print(f"train labels {set(labels_train)}")

consort_items = sorted(set(labels_train))
id2index = {}
for i, item in enumerate(consort_items):
    id2index[i] = item
n_labels = len(id2index)
print("Label to Index Dictionary" , id2index)


config = {'reprocess_input_data': True,
          'fp16':False,
          "evaluate_during_training": False,
          "output_dir": model_save_path, 
          'train_batch_size': 4,
          'gradient_accumulation_steps':16,
          'learning_rate': 3e-5, 
          'num_train_epochs': 30,
          "overwrite_output_dir": True,
          "do_lower_case": True,
          'max_seq_length': 512, 
          'threshold':0.4}
model = MultiLabelClassificationModel('bert', model_path, num_labels=n_labels, args=config)
model.train_model(train_df_linh)
