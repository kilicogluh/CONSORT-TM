import copy
import json
import os

from transformers import BertConfig


class Config(object):
    def __init__(self, **kwargs):
        # bert
        self.bert_model_name = kwargs.pop('bert_model_name', 'bert-large-cased')
        self.bert_cache_dir = kwargs.pop('bert_cache_dir', None)

        # files
        self.train_file = kwargs.pop('train_file', None)
        self.dev_file = kwargs.pop('dev_file', None)
        self.test_file = kwargs.pop('test_file', None)
        self.valid_pattern_path = kwargs.pop('valid_pattern_path', None)
        self.log_path = kwargs.pop('log_path', None)
        # training
        self.accumulate_step = kwargs.pop('accumulate_step', 1)
        self.batch_size = kwargs.pop('batch_size', 8)
        self.eval_batch_size = kwargs.pop('eval_batch_size', 4)
        self.max_epoch = kwargs.pop('max_epoch', 50)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.bert_learning_rate = kwargs.pop('bert_learning_rate', 1e-5)
        self.weight_decay = kwargs.pop('weight_decay', 0)
        self.bert_weight_decay = kwargs.pop('bert_weight_decay', 0)

        self.augmentation = kwargs.pop('augmentation', 0)
        self.threshold = kwargs.pop('threshold', 100)

        # others
        self.use_gpu = kwargs.pop('use_gpu', True)
        self.gpu_device = kwargs.pop('gpu_device', -1)

    @classmethod
    def from_dict(cls, json_object):
        config = cls()
        for k, v in json_object.items():
            setattr(config, k, v)
        return config

    @classmethod
    def from_json_file(cls, path):
        with open(path, 'r', encoding='utf-8') as r:
            return cls.from_dict(json.load(r))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def save_config(self, path):
        """Save a configuration object to a file.
        :param path (str): path to the output file or its parent directory.
        """
        if os.path.isdir(path):
            path = os.path.join(path, 'config.json')
        print('Save config to {}'.format(path))
        with open(path, 'w', encoding='utf-8') as w:
            w.write(json.dumps(self.to_dict(), indent=2,
                               sort_keys=True))
    @property
    def bert_config(self):
        return BertConfig.from_pretrained(self.bert_model_name, output_all_encoded_layers=False)