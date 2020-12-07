#! /usr/bin/env python

import torch
import numpy as np

from transformers import ElectraTokenizer
from transformers import ElectraForSequenceClassification

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

from models import ElectraQA

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

prompt = "In what country is Normandy located"
#prompt = "When were the Normans in Normandy"
response = "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (Norman comes from Norseman) raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries."

#prompt = "What name is given to any prime number larger than 2"
#response = "Hence, 6 is not prime. The image at the right illustrates that 12 is not prime: 12 = 3 · 4. No even number greater than 2 is prime because by definition, any such number n has at least three distinct divisors, namely 1, 2, and n. This implies that n is not prime. Accordingly, the term odd prime refers to any prime number greater than 2. Similarly, when written in the usual decimal system, all prime numbers larger than 5 end in 1, 3, 7, or 9, since even numbers are multiples of 2 and numbers ending in 0 or 5 are multiples of 5."

combo = prompt + " [SEP] " + response

pr_resp = tokenizer.encode(combo, add_special_tokens=True)
pr_resp_pt = torch.tensor(pr_resp).to(device)

embedding_matrix = model.electra.embeddings.word_embeddings
embedded = torch.tensor(embedding_matrix(pr_resp_pt), requires_grad=True)
print(embedded)
print(embedded.size())

start_logits, end_logits, verification_logit = model.saliency(torch.unsqueeze(embedded, 0))

#print(rel_logit)
verification_logit.backward()
#torch.sum(start_logits).backward()
#torch.sum(end_logits).backward()

saliency_max = torch.norm(embedded.grad.data.abs(), dim=1)
saliency_max = saliency_max.detach().cpu().numpy()
# We don't care about the first and last tokens
saliency_max = saliency_max[1:-1]

# Extract only the response words
words = tokenizer.tokenize(combo)
sep = words.index("[SEP]")
saliency_max = saliency_max[sep+1:]
resp_words = words[sep+1:]
print(resp_words)

# Normalise values using softmax
saliency_max = saliency_max/sum(saliency_max)

"""
# Plot a bar chart
M = len(words)
xx = np.linspace(0, M, M)
print(len(words))
print(saliency_max.shape)
plt.figure(figsize=(40,60))
plt.barh(xx, list(saliency_max)[::-1])
plt.yticks(xx, labels=np.flip(words), fontsize=40)
plt.xticks(fontsize=40)
plt.ylabel('Response + Prompt')
plt.title('Salient words identification')
plt.ylim([-2, M+2])
plt.savefig('./saliency.png')
plt.close()
"""
# saliency_all = saliency_all.detach().cpu().numpy()
# saliency_all = saliency_all[1:-1, :]
"""
# Save the gradient values so that we can average them across 15 seeds
np.savetxt("/home/alta/relevance/vr311/phd/question_answering/playing_with_squad/combo_electra/saliency/verification_seed"+seed+".txt", saliency_all)
with open("/home/alta/relevance/vr311/phd/question_answering/playing_with_squad/combo_electra/saliency/words.txt", "w") as output:
    for row in words:
        output.write(str(row) + '\n')
                                                                                     

# Try and plot a heatmap
saliency_all = saliency_all.detach().cpu().numpy()
saliency_all = saliency_all[1:-1, :]

M = len(words)
plt.figure(figsize=(80, 120))
plt.imshow(saliency_all)
xx = np.linspace(0, M, M) * 0.985
plt.axes().set_aspect(aspect=20)
plt.yticks(xx, labels=words, fontsize=80)
plt.colorbar()

plt.savefig('./heatmap.png')
plt.close()
"""

# print the predicted answer to the question:
start_logits = start_logits.detach().cpu().numpy()
end_logits = end_logits.detach().cpu().numpy()
print("Start position: ")
print(np.argmax(start_logits))
print("End position: ")
print(np.argmax(end_logits))