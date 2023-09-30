import os
import os.path
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import lr_scheduler, AdamW, SGD
from collections import defaultdict

import logging
logging.basicConfig(level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

# read the pandas file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("train.csv",encoding="unicode_escape")

data.sample(6)
data['sentiment'].value_counts().tolist()

# configurations
train_maxlen = 140
batch_size = 8
epochs = 10
bert_model = 'bert-base-uncased'
learning_rate = 2e-5
map_dict={'neutral': 0, 'positive': 1, 'negative': 2}

class Tokenize_dataset:
  """
  This class tokenizes the dataset using bert tokenizer
  """

  def __init__(self, text, targets, tokenizer, max_len):
    self.text = text
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.targets = targets

  def __len__(self):
    return len(self.targets)

  def __getitem__(self, item):
    text = str(self.text[item])
    targets = self.targets[item]
    """
    Using encode_plus instead of encode as it helps to provide additional information that we need
    """
    inputs = self.tokenizer.encode_plus(
        str(text),
        add_special_tokens = True,
        max_length = self.max_len,
        pad_to_max_length = True,
        truncation = True
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long)
    }
  
# loss function
# def loss_function(outputs, targets):
# 	"""
# 	This function defines the loss function we use in the model which since is multiclass is crossentropy
# 	"""
# 	return nn.CrossEntropyLoss()(outputs, targets)


# training function
def train_function(data_loader, model, optimizer, device, loss_function, scheduler, n_examples):
    """
    Function defines the training that we will happen over the entire dataset
    """
    # model = model.train()
    model.train()
    losses = []
    correct_predictions = 0

    """
    looping over the entire training dataset
    """
#   with torch.no_grad():
    for d in data_loader:
        input_ids = d["ids"].to(device)
        attention_mask = d["mask"].to(device)
        targets = d["targets"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        _, preds = torch.max(outputs, dim=1)
        loss = loss_function(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        
        # Gradient Descent
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.zero_grad()
        # Backward prop
        loss.backward()
        optimizer.step()
    scheduler.step()
    print("Optimizer lr=%.4f" % (optimizer.param_groups[0]["lr"]))
    
    return correct_predictions.double() / n_examples, np.mean(losses)

# evaluation function
def eval_function(data_loader, model, device, loss_function, n_examples):
    # model = model.eval()
    model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["ids"].to(device)
            attention_mask = d["mask"].to(device)
            targets = d["targets"].to(device)
            
            # Get model ouptuts
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            print("outputs", outputs);
            _, preds = torch.max(outputs, dim=1)
            print(preds, targets, "prediction vs target in eval");
            loss = loss_function(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    print(correct_predictions, n_examples, "predictions and examples")
    print(losses, np.mean(losses), np.sum(losses), "losses, mean loss and sum of losses")
                
    return correct_predictions.double() / n_examples, np.mean(losses)

# complete custom model from pretrained bert
class CompleteModel(nn.Module):
  """
  The model architecture is defined here which is a fully connected layer + normalization on top of a BERT model
  """

  def __init__(self, bert, classes):
    super(CompleteModel, self).__init__()
    self.bert = BertModel.from_pretrained(bert)
    self.drop = nn.Dropout(p=0.25)
    self.out = nn.Linear(self.bert.config.hidden_size, classes) # Number of output classes = 3, positive, negative and N(none)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
    output = self.drop(pooled_output)
    return self.out(output)
  
def plot_graphs(history):
   # Plot training and validation accuracy
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')

    # Graph chars
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1]);
  
def load_train_valid_data():
    training_set_path = "train.csv"
    validation_set_path = 'test.csv'
    df_train = pd.read_csv(training_set_path, encoding="unicode_escape")
    df_valid = pd.read_csv(validation_set_path, encoding="unicode_escape")

    df_train = df_train.sample(frac=0.1)
    df_valid = df_valid.sample(frac=0.1)

    df_train['target'] = df_train['sentiment'].map(map_dict)
    df_train['target'] = df_train['target']
    df_valid['target'] = df_valid['sentiment'].map(map_dict)
    df_valid['target'] = df_valid['target']
    df_valid = df_valid.dropna(subset=['target'])
    df_train = df_train.dropna(subset=['target'])
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    return df_train, df_valid
def run():
    df_train, df_valid = load_train_valid_data()

    tokenizer = BertTokenizer.from_pretrained(bert_model)
    train_dataset = Tokenize_dataset(
        text = df_train['text'].values,
        targets = df_train['target'].values,
        tokenizer = tokenizer,
        max_len = train_maxlen
    )
    class_counts = []
    for i in range(3):
        class_counts.append(df_train[df_train['target']==i].shape[0])
    print(f"Class Counts: {class_counts}")
    num_samples = sum(class_counts)
    print(num_samples)
    labels = df_train['target'].values
    class_weights = []
    for i in range(len(class_counts)):
      if class_counts[i] != 0:
          class_weights.append(num_samples/class_counts[i])
      else:
          class_weights.append(0)
    weights = [class_weights[labels[i]] for i in range(int(num_samples))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        # sampler = sampler
    )
    valid_dataset = Tokenize_dataset(
        text = df_valid['text'].values,
        targets = df_valid['target'].values,
        tokenizer = tokenizer,
        max_len = train_maxlen
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = batch_size,
        shuffle = True
    )

    device=torch.device("cpu")

    if torch.cuda.is_available():
       device = torch.device("cuda:0")
    # if torch.has_mps:
    #    device = torch.device("mps")
    
    loss_function = nn.CrossEntropyLoss().to(device)
    print(f"Device: {device}")
    model = CompleteModel(bert_model, 3).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_data_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(epochs):
        # Show details 
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)
        model.train()
        train_losses = []
        train_correct_predictions = 0

        """
        looping over the entire training dataset
        """
    #   with torch.no_grad():
        for d in train_data_loader:
            input_ids = d["ids"].to(device)
            attention_mask = d["mask"].to(device)
            targets = d["targets"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
            loss = loss_function(outputs, targets)
            train_correct_predictions += torch.sum(preds == targets)
            train_losses.append(loss.item())
            
            # Gradient Descent
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.zero_grad()
            # Backward prop
            loss.backward()
            optimizer.step()
            scheduler.step()
        print("Optimizer lr=%.4f" % (optimizer.param_groups[0]["lr"]))
        train_acc = train_correct_predictions.double() / len(df_train.index)
        train_loss = np.mean(train_losses)
        print(f"Train loss {train_loss} accuracy {train_acc}")
        print("validating")

        model.eval()
        val_losses = []
        val_correct_predictions = 0
        
        with torch.no_grad():
            for d in valid_data_loader:
                input_ids = d["ids"].to(device)
                attention_mask = d["mask"].to(device)
                targets = d["targets"].to(device)
                
                # Get model ouptuts
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # print("outputs of eval", outputs);
                _, preds = torch.max(outputs, dim=1)
                print(preds, targets, "prediction vs target in eval");
                loss = loss_function(outputs, targets)
                val_correct_predictions += torch.sum(preds == targets)
                val_losses.append(loss.item())

        val_acc =val_correct_predictions.double() / len(df_valid.index)
        val_loss = np.mean(val_losses)    
        print(f"Val loss {val_loss} accuracy {val_acc}")
        print("epoch end, beginning next")
    
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        # If we beat prev performance
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

    plot_graphs(history)
    print("The end")

def get_predictions():
    review_text = "This is ridiculuous."
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    encoded_review = tokenizer.encode_plus(
        str(review_text),
        add_special_tokens = True,
        max_length = 140,
        pad_to_max_length = True,
        truncation = True
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_ids = torch.tensor(encoded_review['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(encoded_review['attention_mask']).unsqueeze(0)
    token_type_ids = torch.tensor(encoded_review["token_type_ids"]).unsqueeze(0)

    model = CompleteModel(bert_model, 3).to(device)
    model.load_state_dict(torch.load("best_model_state.bin"), strict=False)
    model.eval()
# predict with the model
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    _, preds = torch.max(output, dim=1)
    # find key in map_dict whose value matches preds.item()
    predicted_sentiment = [i for i in map_dict if map_dict[i]==preds.item()]
    
    print(f'Review text: {review_text}')
    print(f'Sentiment  : {predicted_sentiment}')

if __name__ == "__main__":
    run()
    # to run predictions, use the following function
    # get_predictions()
