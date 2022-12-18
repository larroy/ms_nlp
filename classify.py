# NOTE: This is only a starter template. Wherever additional changes are required, please feel free modify/update.

# Changes:
# Add Exploratory data analysis routines, see attached notebook "eda.ipynb"
# Fix type annotations, replace with classes from typing module. Fix incorrect annotations.
# Workaround bug in torchtext
# https://github.com/pytorch/text/blob/v0.8.1/torchtext/data/field.py#L291 while calling
# build_vocab, it never builds a vocabulary because it checks that the value is the same field,
# although the attribute fields contains a tuple, so the check is always false.
#
# X_train.fields
# Out[2]:
# {'Subject': ('Subject', <torchtext.data.field.Field at 0x7f37f52f1450>),
#  'Body': ('Body', <torchtext.data.field.Field at 0x7f37f52f1450>),
#  'Date': ('Date', <torchtext.data.field.Field at 0x7f37f52f1450>),
#  'Label': ('Label', <torchtext.data.field.Field at 0x7f37f52f1510>)}
#
#                sources += [getattr(arg, name) for name, field in
#                            arg.fields.items() if field is self]
#
# I added a workaround to remove the tuple and add only the field, so the vocabulary is propperly
# built.
#




import pickle
import string
import json
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchtext as tt
from torchtext.data import Field

from eda import run_eda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: Feel free to improve the model
# Switch to softmax activation
class EmailClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_class):
        super(EmailClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            )
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, text):
        embeddings = self.embedding(text)
        hidden_out = self.lstm(embeddings)
        output = self.fc(hidden_out)
        return output


# Step #0: Load data
def load_data(path: str) -> List:
    """Load Pickle files"""
    with open(path, 'rb') as f:
        data_list = pickle.load(f)
    return data_list


# Step #1: Analyse data
def analyse_data(data: list) -> None:
    """Analyse data files.
    See accompanying jupyter notebook "eda.ipynb"""
    #run_eda(pd.DataFrame(data))
    return None


# Step #2: Define data fields
def data_fields() -> Tuple:
    # Type of fields on Data
    TEXTFIELD: Field = tt.data.Field(sequential=True,
                                     init_token='<sos>',
                                     eos_token='<eos>',
                                     lower=True,
                                     tokenize=tt.data.utils.get_tokenizer("basic_english"), )
    LABELFIELD: Field = tt.data.Field(sequential=False,
                                      use_vocab=False,
                                      unk_token=None,
                                      is_target=True)

    fields = {'Subject': ('Subject', TEXTFIELD), 'Body': ('Body', TEXTFIELD),
              'Date': ('Date', TEXTFIELD), 'Label': ('Label', LABELFIELD)}

    return fields, TEXTFIELD, LABELFIELD


# Step #2: Clean data
def data_clean(data: List, fields: Dict) -> List:
    """A data cleaning routine."""

    clean_data = []
    for curr_data in data:
        curr_data["Subject"] = curr_data["Subject"].translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        curr_data["Body"] = curr_data["Body"].translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        clean_data.append(tt.data.Example.fromJSON(json.dumps(curr_data), fields))

    return clean_data


def fields_bug_workaround(fields: Dict) -> Dict:
    res = {}
    for k,v in fields.items():
        res[k] = v[1]
    return res

# Step #2: Prepare data
def data_prepare(data: list, fields: dict, val_percent: int) -> List:
    """A data preparation routine."""

    clean_train, clean_val = tt.data.Dataset(data, fields).split(split_ratio=val_percent)

    return clean_train, clean_val


# Step #3: Extract features
def extract_features(X_train: List, X_valid: List, TEXTFIELD: tt.data.Field,
                     LABELFIELD: tt.data.Field) -> List:
    # TODO: Batch using subject + body
    train_iter, val_iter = [], []
    if X_train:
        X_train.fields = fields_bug_workaround(X_train.fields)
        TEXTFIELD.build_vocab(X_train)
        LABELFIELD.build_vocab(X_train)
        train_iter = tt.data.BucketIterator(X_train, batch_size=32, sort_key=lambda x: len(x.subject),
                                                   device=device, sort=True, sort_within_batch=True)

    if X_valid:
        X_valid.fields = fields_bug_workaround(X_valid.fields)
        val_iter = tt.data.BucketIterator(X_valid, batch_size=32, sort_key=lambda x: len(x.subject),
                                                 device=device, sort=True, sort_within_batch=True)

    return train_iter, val_iter, TEXTFIELD, LABELFIELD


# Step #4: Train model
def train_model(classification_model: EmailClassifier, train_iter: tt.data.BucketIterator,
                val_iter: tt.data.BucketIterator) -> EmailClassifier:
    """Create a training loop"""

    train_iter.create_batches()
    for batch in train_iter.batches:
        for data_point in batch:
            x = data_point.body
            y = data_point.label

    return classification_model


# Step #5: Stand-alone Test data & Compute metrics
def compute_metrics(classification_model: EmailClassifier, test_data: list) -> None:
    return None


def main(train_path: str, test_path: str) -> None:
    ### Perform the following steps and complete the code

    ### Step #0: Load data
    train_data = load_data(train_path)

    ### Step #1: Analyse data
    analyse_data(train_data)

    ### Step #2: Clean and prepare data
    fields, TEXTFIELD, LABELFIELD = data_fields()
    train_data = data_clean(train_data, fields)

    # In the EDA step we have seen that the dataset is not balanced, also 50% train / validation split seems quite big
    train_ds, val_ds = data_prepare(train_data, fields, val_percent=0.5)

    ### Step #3: Extract features
    train_iter, val_iter, TEXTFIELD, LABELFIELD = extract_features(train_ds, val_ds, TEXTFIELD, LABELFIELD)
    # Vocab size is not 5546 after the fix. Before it was 4 only special tokens, see comments in Changelog.
    vocab_size = len(TEXTFIELD.vocab.stoi)

    ### Step #4: Train model
    classification_model = EmailClassifier(vocab_size=vocab_size, embed_size=128, hidden_size=128, num_class=4)
    classification_model = train_model(classification_model, train_iter, val_iter)

    ### Step #5: Stand-alone Test data & Compute metrics
    test_data = load_data(test_path)
    analyse_data(test_data)
    test_data = data_clean(test_data, fields)
    compute_metrics(classification_model, test_data)

    return 0


if __name__ == "__main__":
    train_path = "./agnews_combined_train.pkl"
    test_path = "./agnews_combined_test.pkl"
    main(train_path, test_path)
