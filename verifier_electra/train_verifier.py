#! /usr/bin/env python

import argparse
import os
import sys

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
import time
import datetime

from datasets import load_dataset
from transformers import ElectraTokenizer
from keras.preprocessing.sequence import pad_sequences
from transformers import ElectraForSequenceClassification, AdamW, ElectraConfig
from transformers import get_linear_schedule_with_warmup

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Specify the initial learning rate')
parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Specify the AdamW loss epsilon')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
parser.add_argument('--n_epochs', type=int, default=1, help='Specify the number of epochs to train for')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--reverse', type=int, default=0, help='Specify the whether the electra is reversed')
parser.add_argument('--save_path', type=str, help='Load path to which trained model will be saved')


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def _find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1
    print("Didn't find match, return <no answer>")
    return -1,0


def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Set the seed value all over the place to make this reproducible.
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Choose device
    device = get_default_device()

    train_data = load_dataset('squad_v2', split='train')
    print(train_data[1953])

    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator', do_lower_case=True)

    input_ids = []
    token_type_ids = []
    labels = []
    count = 0
    too_long = 0
    for ex in train_data:
        count+=1
        # if count==17:
        #     break
        question, passage = ex["question"], ex["context"]
        if args.reverse==0:
            combo = question + " [SEP] " + passage
        elif args.reverse==1:
            combo = passage + " [SEP] " + question
        inp_ids = tokenizer.encode(combo)
        if len(inp_ids) > 512:
            too_long+=1
            continue
        tok_type_ids = [0 if i<= inp_ids.index(102) else 1 for i in range(len(inp_ids))]  # Indicates whether part of sentence A or B -> 102 is Id of [SEP] token
        if len(ex["answers"]["text"])==0:
            lab = 0
        else:
            lab = 1
        input_ids.append(inp_ids)
        token_type_ids.append(tok_type_ids)
        labels.append(lab)

        # Balance the training dataset (the original set has twice as many on-topic as off-topic)
        if lab==0:
            input_ids.append(inp_ids)
            token_type_ids.append(tok_type_ids)
            labels.append(lab)

    print("Number of questions with answers:")
    print(sum(labels))
    print("Number of questions without answers")
    print(len(labels)-sum(labels))
    print("Greater than 512 tokens in length:")
    print(too_long)
    # Pad all sequences with 0
    max_len = max([len(sen) for sen in input_ids])
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    token_type_ids = pad_sequences(token_type_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    # Create attention masks
    attention_masks = []
    for sen in input_ids:
        att_mask = [int(token_id > 0) for token_id in sen]
        attention_masks.append(att_mask)
    #  Convert to torch tensors
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    token_type_ids = torch.tensor(token_type_ids)
    token_type_ids = token_type_ids.long().to(device)
    attention_masks = torch.tensor(attention_masks)
    attention_masks = attention_masks.long().to(device)
    labels = torch.tensor(labels)
    labels = labels.long().to(device)
    # Create the DataLoader for training set.
    train_data = TensorDataset(input_ids, token_type_ids, attention_masks, labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    model = ElectraForSequenceClassification.from_pretrained(
        "google/electra-base-discriminator", # Use the 12-layer Electra model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    model.to(device)

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                    lr = args.learning_rate,
                    eps = args.adam_epsilon
                    )

    loss_values = []

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * args.n_epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    for epoch in range(args.n_epochs):
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.n_epochs))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()      
    # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_tok_typ_ids = batch[1].to(device)
            b_att_msks = batch[2].to(device)
            b_labels = batch[3].to(device)
            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=b_tok_typ_ids, attention_mask=b_att_msks, labels=b_labels)
            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            loss = outputs[0]
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)            
        
        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
    # Save the model to a file
    file_path = args.save_path+'electra_verifier_seed'+str(args.seed)+'.pt'
    torch.save(model, file_path)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)