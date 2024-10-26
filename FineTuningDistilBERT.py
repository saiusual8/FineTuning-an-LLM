# -*- coding: utf-8 -*-


pip install transformers datasets torch

from datasets import load_dataset

#Load the IMDB dataset
dataset = load_dataset('imdb')

#Split the dataset into training and test sets
train_data = dataset['train']
test_data = dataset['test']


print(train_data[0])

from transformers import DistilBertTokenizer

#Load DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

#Tokenize the datase. This tokenizer helps to preprocess the text data so it can be fed into the model.
#Tokenization will convert the text into input IDs and attention masks required by DistilBERT

def tokenize_data(example):
  return tokenizer(example['text'], padding='max_length', truncation=True, max_length = 512)

#Apply the tokenization function to the dataset
train_data = train_data.map(tokenize_data, batched = True)
test_data = test_data.map(tokenize_data, batched = True)

#Set the format of the dataset to Pytorch tensors
train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

from transformers import DistilBertForSequenceClassification

#Load DistilBERT model with a classification head
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

from transformers import TrainingArguments

#Define training Arguments

training_args = TrainingArguments(
    output_dir='./results',        #Output Directory
    eval_strategy='steps',        #Evaluate every epoch
    learning_rate=2e-5,            #Learning Rate
    per_device_train_batch_size=16,#Training batch size
    per_device_eval_batch_size=16, #Evaluation batch size
    num_train_epochs=3,            #Number of training epochs
    weight_decay=0.01,             #Weight decay for regularization
    load_best_model_at_end=True,   #Automatically load the best model at the end.
    greater_is_better=False,       #We want the loss to be minimized
    evaluation_strategy='steps',    #Evaluation strategy starts at each steps
    save_strategy='steps',          #Save strategy is set as Steps
    eval_steps=500,
    save_steps=500,
)

#Set a lower learning rate if necessary
training_args.learning_rate = 1e-5 #Trying a smaller learning rate

from transformers import Trainer

#Initialize the trainer
trainer = Trainer(
    model=model,                 #The Model to be fine-tuned
    args=training_args,          #Training arguments
    train_dataset=train_data,    #Training dataset
    eval_dataset=test_data,      #Evaluation dataset
)

#start fine-tuning\


import os
os.environ['WANDB_MODE'] = 'disabled'
trainer.train()

