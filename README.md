# Training and evaluation of machine comprehension models on SQuAD 2.0

## Dependencies
Need Python 3.5 at least

### To install
pip install torch <br />
pip install transformers <br />
pip install numpy <br />
pip install Keras <br />
pip install datasets <br />

## Models

The models of interest are in the directories named `naive_electra`, `naive_bert`, `combo_electra`, `combo_bert`. <br />
The `naive` ELECTRA and BERT models take the pre-trained transformer and attach a question-answering head on top. <br />
The `combo` ELECTRA and BERT models intriduce an additional verifier model that acts as a binary classifier to identify unanswerable questions. <br />

Each of these directories contain a training and test script. These scripts can be run directly using appropriate CPU / GPU machines (batching is supported). To clarify, the following directory structure clearly outlines the training and test scripts of interest in each model directory:

    .
    ├── ...
    ├── combo_bert                                      
    │   ├── models.py          
    │   ├── test_qa.py
    │   └── train_qa.py
    ├── combo_electra                                      
    │   ├── models.py          
    │   ├── test_qa__all.py
    │   └── train_qa.py
    ├── naive_bert                              
    │   ├── test.py
    │   └── train.py
    ├── naive_electra                              
    │   ├── test_all.py
    │   └── train.py
    └── ...
