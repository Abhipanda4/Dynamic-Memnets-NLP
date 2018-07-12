from glob import glob
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import re
import numpy as np


def pad_collate(batch):
    max_context_sen = float('-inf')
    max_context = float('-inf')
    max_question = float('-inf')
    for lol in batch:
        context, question, _ = lol
        max_context = max(max_context, len(context))
        max_question = max(max_question, len(question))
        for sen in context:
            max_context_sen = max(max_context_sen, len(sen))
    # max_context = min(max_context, 100)
    for i, lol in enumerate(batch):
        context, question, answer = lol
        context_ = np.zeros((max_context, max_context_sen))
        for j, sen in enumerate(context):
            context_[j] = np.pad(sen, (0,max_context_sen - len(sen)), 'constant', constant_values=0)
        question = np.pad(question, (0,max_question - len(question)), 'constant', constant_values=0)
        batch[i] = (context_, question, answer)
    return default_collate(batch)


# spit ratio P/Q
P=1;
Q=10;

class BabiDataset(Dataset):
    def __init__(self, task_id, mode='train'):
        raw_train, raw_test = get_raw_babi(task_id);
        self.mode = mode
        self.VOCAB={'<PAD>': 0, '<EOS>': 1}
        self.IVOCAB={0: '<PAD>', 1:  '<EOS>'}
        self.train = self.get_indexed_qa(raw_train)
        self.test = self.get_indexed_qa(raw_test)
        self.valid = [self.train[i][int(-P*len(self.train[i])/Q):] for i in range(3)]
        self.train = [self.train[i][:int((Q-P) * len(self.train[i])/Q)] for i in range(3)]

    def __len__(self):
        if self.mode == 'train':
            return len(self.train[0])
        elif self.mode == 'valid':
            return len(self.valid[0])
        elif self.mode == 'test':
            return len(self.test[0])

    def __getitem__(self, index):
        if self.mode == 'train':
            contexts, questions, answers = self.train
        elif self.mode == 'valid':
            contexts, questions, answers = self.valid
        elif self.mode == 'test':
            contexts, questions, answers = self.test
        return contexts[index], questions[index], answers[index]

    # add token to dict and map it to a number
    def build_vocab(self,token):
        if token not in self.VOCAB:
            next_idx = len(self.VOCAB);
            self.VOCAB[token] = next_idx;
            self.IVOCAB[next_idx] = token;


    def get_indexed_qa(self, raw_data):
        unindexed = get_unindexed_qa(raw_data) # list of dicts
        contexts=[]
        questions=[]
        answers=[]
        for task in unindexed:
            context = [c.lower().split() + ['<EOS>'] for c in task["C"]]
            for con in context:
                for token in con:
                    self.build_vocab(token)
            context = [[self.VOCAB[token] for token in con] for con in context]
            question = task["Q"].lower().split()+['<EOS>']
            for token in question:
                self.build_vocab(token)
            question = [self.VOCAB[token] for token in question]

            answer=task["A"].lower();
            self.build_vocab(answer);
            answer = self.VOCAB[answer]

            contexts.append(context)
            questions.append(question)
            answers.append(answer)
        return (contexts, questions, answers)


# read test and train file for a particular task_id
def get_raw_babi(task_id):
    paths = glob('data/en-10k/qa{}_*'.format(task_id))
    for path in paths:
        if 'train' in path:
            with open(path, 'r') as fp:
                train = fp.read()
        elif 'test' in path:
            with open(path, 'r') as fp:
                test = fp.read()
    return train, test


def get_unindexed_qa(raw_data):
    tasks=[]
    task=None
    babi=raw_data.strip().split('\n')
    for i, line in enumerate(babi):
        idx =  int(line[0:line.find(' ')])
        if idx == 1:
            task={"C": [], "Q": "", "A": "", "S": []}
            cnt=0
            imap={}
        line=line.strip().replace('.', ' . ')
        line=line[line.find(' ')+1:]
        if line.find('?') == -1:
            # get context
            task["C"].append(line)
            imap[idx]=cnt;
            cnt+=1;
        else:
            # find question mark
            qm=line.find('?')
            AS=line[qm+1:].split('\t')
            task["Q"] = line[:qm]
            # get answer
            task["A"] = AS[1].strip()
            # get supporting facts
            task["S"] = []
            for val in AS[2].split():
                task["S"].append(imap[int(val.strip())])
            # append a task(dict of context C, question Q, answer A, Supporting fact S)
            tasks.append(task)
    return tasks

if __name__ == '__main__':
    dset_train = BabiDataset(20, 'train')
    train_loader = DataLoader(dset_train, batch_size=2, shuffle=True, collate_fn=pad_collate)
    for batch_idx, data in enumerate(train_loader):
        contexts, questions, answers = data
        # print(questions)
        break
