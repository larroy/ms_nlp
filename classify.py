# NOTE: This is only a starter template. Wherever additional changes are required, please feel free modify/update.

import pickle
import string
import json
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchtext as tt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Feel free to improve the model
class EmailClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_class):
        super(EmailClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size = embed_size,
                            hidden_size = hidden_size,
                            )
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, text):
        embeddings = self.embedding(text)
        hidden_out = self.lstm(embeddings)
        output = self.fc(hidden_out)
        return output


# Step #0: Load data
def load_data(path: str) -> list:
    """Load Pickle files"""

    with open(path, 'rb') as f:
        data_list = pickle.load(f)
    return data_list

# Step #1: Analyse data
def analyse_data(data: list) -> None:
    """Analyse data files"""
    return None

# Step #2: Define data fields
def data_fields() -> dict:
    # Type of fields on Data
    TEXTFIELD = tt.legacy.data.Field(sequential = True,
        init_token = '<sos>',
        eos_token = '<eos>',
        lower = True,
        tokenize = tt.legacy.data.utils.get_tokenizer("basic_english"),)
    LABELFIELD = tt.legacy.data.Field(sequential = False,
        use_vocab = False,
        unk_token = None,
        is_target = True)

    fields = {'Subject': ('subject', TEXTFIELD), 'Body': ('body', TEXTFIELD), 
                'Date': ('date', TEXTFIELD), 'Label': ('label', LABELFIELD)}

    return fields, TEXTFIELD, LABELFIELD

# Step #2: Clean data
def data_clean(data: list, fields: dict) -> list:
    """A data cleaning routine."""
    
    clean_data = []
    for curr_data in data:
        curr_data["Subject"] = curr_data["Subject"].translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        curr_data["Body"] = curr_data["Body"].translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        clean_data.append(tt.legacy.data.Example.fromJSON(json.dumps(curr_data), fields))        

    return clean_data

# Step #2: Prepare data
def data_prepare(data: list, fields: dict, val_percent: int) -> list:
    """A data preparation routine."""

    clean_train, clean_val = tt.legacy.data.Dataset(data, fields).split(split_ratio = val_percent)

    return clean_train, clean_val

# Step #3: Extract features
def extract_features(X_train: list, X_valid: list, TEXTFIELD: tt.legacy.data.Field, LABELFIELD: tt.legacy.data.Field) -> list:
    train_iter, val_iter = [], []
    if X_train:
        TEXTFIELD.build_vocab(X_train)  
        LABELFIELD.build_vocab(X_train)
        train_iter = tt.legacy.data.BucketIterator(X_train, batch_size=32, sort_key=lambda x: len(x.subject),
                            device=device, sort=True, sort_within_batch=True)
    
    if X_valid:
        val_iter = tt.legacy.data.BucketIterator(X_valid, batch_size=32, sort_key=lambda x: len(x.subject),
                            device=device, sort=True, sort_within_batch=True)
    
    return train_iter, val_iter, TEXTFIELD, LABELFIELD

# Step #4: Train model
def train_model(classification_model: EmailClassifier, train_iter: tt.legacy.data.BucketIterator, val_iter: tt.legacy.data.BucketIterator) -> EmailClassifier:
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
    train_ds, val_ds = data_prepare(train_data, fields, val_percent = 0.5)

    ### Step #3: Extract features
    train_iter, val_iter, TEXTFIELD, LABELFIELD = extract_features(train_ds, val_ds, TEXTFIELD, LABELFIELD)
    vocab_size = len(TEXTFIELD.vocab.stoi)
    
    ### Step #4: Train model
    classification_model = EmailClassifier(vocab_size = vocab_size, embed_size = 128, hidden_size = 128, num_class = 4)
    classification_model = train_model(classification_model, train_iter, val_iter)

    ### Step #5: Stand-alone Test data & Compute metrics
    test_data = load_data(test_path)
    analyse_data(test_data)
    test_data = data_clean(test_data, fields)
    compute_metrics(classification_model, test_data)
    

    return 0

if __name__ == "__main__":
    train_path = "./agnews_combined_train.pkl"
    test_path = "./agnews_combined_train.pkl"
    main(train_path, test_path)