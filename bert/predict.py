import pandas as pd
from simpletransformers.classification import MultiLabelClassificationModel
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi Label Classification')
    parser.add_argument('--data_path', help='path to .csv consisting of sentences', required=True)
    parser.add_argument('--model_path', help='path to directory consisting of BERT model', required=True)
    parser.add_argument('--text_column', help='column name containing sentences', default="sentence_text")
    args = vars(parser.parse_args())

    # Reading Data
    data = pd.read_csv(args["data_path"], engine="python")

    # Load labels dictionary
    # TODO - store pickle of labels for each model
    labels_dict = {0: '0',
                   1: '10',
                   2: '11a',
                   3: '11b',
                   4: '12a',
                   5: '12b',
                   6: '3a',
                   7: '3b',
                   8: '4a',
                   9: '4b',
                   10: '5',
                   11: '6a',
                   12: '6b',
                   13: '7a',
                   14: '7b',
                   15: '8a',
                   16: '8b',
                   17: '9'}

    # Loading Model
    config = {'reprocess_input_data': True,
              'fp16': False,
              "evaluate_during_training": False,
              'train_batch_size': 4,
              'gradient_accumulation_steps': 16,
              'learning_rate': 3e-5,
              'num_train_epochs': 30,
              'n_top_sections': 29, 
              "overwrite_output_dir": True,
              "do_lower_case": True,
              'max_seq_length': 512}
    
    if "top_section" not in data.columns:
        config["n_top_sections"] = 0
        top_section_data = [1] * data.shape[0]
    else:
        top_section_data = data["top_section"].tolist()
        
    model = MultiLabelClassificationModel('bert', args["model_path"], num_labels=len(labels_dict), args=config)

    predictions = model.predict(data[args["text_column"]].tolist(), top_section_data)

    predicted_labels = []
    for l in predictions[0]:
        preds = list(set([labels_dict[i] for i in range(len(l)) if l[i]]))
        if len(preds) == 0:
            preds = ["0"]
        if len(preds) > 1 and "0" in preds:
            preds.remove("0")
        predicted_labels.append(preds)

    data["predictions"] = predicted_labels
    data.to_csv(args["data_path"], index=False)
