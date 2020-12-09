#! /usr/bin/env python

import torch
import numpy as np

from transformers import ElectraTokenizer
from transformers import ElectraForSequenceClassification

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

from datasets import load_dataset
from models import ElectraQA

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1
    # Coudln't find the sub-list
    return 0,0

def hellinger(p,q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2.0)

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()
seed="2"
model_path = "/home/alta/relevance/vr311/phd/question_answering/playing_with_squad/combo_electra/seed"+seed+"/electra_comboQA_seed"+seed+".pt"
model = torch.load(model_path, map_location=device)
model.eval().to(device)

tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator', do_lower_case=True)

# prompt = "In what country is Normandy located"
# #prompt = "When were the Normans in Normandy"
# response = "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (Norman comes from Norseman) raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries."

#prompt = "What name is given to any prime number larger than 2"
#response = "Hence, 6 is not prime. The image at the right illustrates that 12 is not prime: 12 = 3 · 4. No even number greater than 2 is prime because by definition, any such number n has at least three distinct divisors, namely 1, 2, and n. This implies that n is not prime. Accordingly, the term odd prime refers to any prime number greater than 2. Similarly, when written in the usual decimal system, all prime numbers larger than 5 end in 1, 3, 7, or 9, since even numbers are multiples of 2 and numbers ending in 0 or 5 are multiples of 5."

dev_data = load_dataset('squad_v2', split='validation')

stop_words = stopwords.words('english')

hell_saliency_true_all = []
hell_uni_true_all = []
count=0
for ex in dev_data:
    count+=1
    print('\n'+str(count))
    if count==13:
        break
    prompt, response = ex["question"], ex["context"]
    combo = prompt + " [SEP] " + response
    answers = ex["answers"]["text"]
    if len(answers)==0:
        # Ignore the negative examples for now
        continue
    pr_resp = tokenizer.encode(combo, add_special_tokens=True)
    if len(pr_resp) > 512:
        # Ignore long ones for now
        continue
    pr_resp_pt = torch.tensor(pr_resp).to(device)

    embedding_matrix = model.electra.embeddings.word_embeddings
    embedded = torch.tensor(embedding_matrix(pr_resp_pt), requires_grad=True)
    # print(embedded)
    # print(embedded.size())

    start_logits, end_logits, verification_logit = model.saliency(torch.unsqueeze(embedded, 0))

    verification_logit.backward()

    saliency_max = torch.norm(embedded.grad.data.abs(), dim=1)
    saliency_max = saliency_max.detach().cpu().numpy()
    # We don't care about the first and last tokens
    saliency_max = saliency_max[1:-1]

    # Extract only the response words
    words = tokenizer.tokenize(combo)
    sep = words.index("[SEP]")
    saliency_max = saliency_max[sep+1:]
    resp_words = words[sep+1:]
    prompt_words = words[:sep]

    # Identify content words in prompt and then de-weight the saliency values for these words in response
    content_words = [w for w in prompt_words if w.casefold() not in stop_words]
    idx_to_remove = [i for i, w in enumerate(resp_words) if w in content_words]
    for idx in idx_to_remove:
        saliency_max[idx] = 0.0

    # Normalise values
    saliency_max = saliency_max / np.sum(saliency_max)

    # Get Hellinger distance values

    # Get Hellinger distance with true labels

    # Construct true distribution
    true_dist = [0.0] * len(resp_words)
    for ans in answers:
        ans_tok = tokenizer.tokenize(ans)
        s_idx, e_idx = find_sub_list(ans_tok, resp_words)
        for idx in range(s_idx, e_idx+1):
            true_dist[idx] = 1.0

    # true_idx = resp_words.index("france")
    # true_dist[true_idx] = 1.0

    true_dist = np.asarray(true_dist)

    true_dist = true_dist / np.sum(true_dist)

    uni_dist = [1.0] * len(resp_words)
    uni_dist = np.asarray(uni_dist)
    uni_dist = uni_dist / np.sum(uni_dist)

    hell_saliency_true = hellinger(true_dist, saliency_max)
    hell_saliency_true_all.append(hell_saliency_true)
    print("Saliency Hellinger:")
    print(hell_saliency_true)
    hell_uni_true = hellinger(true_dist, uni_dist)
    hell_uni_true_all.append(hell_uni_true)
    print("Uniform Hellinger")
    print(hell_uni_true)

hell_saliency_true_all = np.array(hell_saliency_true_all)
hell_uni_true_all = np.array(hell_uni_true_all)

print("Saliency:")

r1 = np.mean(hell_saliency_true_all) 
print("Mean: ", r1) 
  
r2 = np.std(hell_saliency_true_all) 
print("std: ", r2) 
  
r3 = np.var(hell_saliency_true_all) 
print("variance: ", r3)

print("Uniform:")

r1 = np.mean(hell_uni_true_all) 
print("Mean: ", r1) 
  
r2 = np.std(hell_uni_true_all) 
print("std: ", r2) 
  
r3 = np.var(hell_uni_true_all) 
print("variance: ", r3)

# # print the predicted answer to the question:
# start_logits = start_logits.detach().cpu().numpy()
# end_logits = end_logits.detach().cpu().numpy()
# print("Start position: ")
# print(np.argmax(start_logits))
# print("End position: ")
# print(np.argmax(end_logits))