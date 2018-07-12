from new_loader import BabiDataset, pad_collate
from episodic_memory import EpisodicMemory, MemoryUpdateNetwork
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader

import argparse

def position_encoding(embedded_sentence):
    '''
    embedded_sentence.size() -> (#batch, #sentence, #token, #embedding)
    l.size() -> (#sentence, #embedding)
    output.size() -> (#batch, #sentence, #embedding)
    '''
    _, _, slen, elen = embedded_sentence.size()

    l = [[(1 - s/(slen-1)) - (e/(elen-1)) * (1 - 2*s/(slen-1)) for e in range(elen)] for s in range(slen)]
    l = torch.FloatTensor(l)
    l = l.unsqueeze(0) # for #batch
    l = l.unsqueeze(1) # for #sen
    l = l.expand_as(embedded_sentence)
    weighted = embedded_sentence * Variable(l)
    return torch.sum(weighted, dim=2).squeeze(2) # sum with tokens

class QuestionModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(QuestionModule, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, questions, word_embedding):
        '''
        questions.size() -> (#batch, #token)
        word_embedding() -> (#batch, #token, #embedding)
        gru() -> (1, #batch, #hidden)
        '''
        questions = word_embedding(questions)
        _, questions = self.gru(questions)
        questions = questions.transpose(0, 1)
        return questions

class InputModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(InputModule, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        for name, param in self.gru.state_dict().items():
            if 'weight' in name: init.xavier_normal(param)
        self.dropout = nn.Dropout(0.1)

    def forward(self, contexts, word_embedding):
        '''
        contexts.size() -> (#batch, #sentence, #token)
        word_embedding() -> (#batch, #sentence x #token, #embedding)
        position_encoding() -> (#batch, #sentence, #embedding)
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        '''
        batch_num, sen_num, token_num = contexts.size()

        contexts = contexts.view(batch_num, -1)
        contexts = word_embedding(contexts)

        contexts = contexts.view(batch_num, sen_num, token_num, -1)
        contexts = position_encoding(contexts)
        contexts = self.dropout(contexts)

        h0 = Variable(torch.zeros(2, batch_num, self.hidden_size))
        facts, hdn = self.gru(contexts, h0)
        facts = facts[:, :, :hidden_size] + facts[:, :, hidden_size:]
        return facts

class AnswerModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(AnswerModule, self).__init__()
        self.z = nn.Linear(2 * hidden_size, vocab_size)
        init.xavier_normal(self.z.state_dict()['weight'])
        self.dropout = nn.Dropout(0.1)

    def forward(self, M, questions):
        M = self.dropout(M)
        concat = torch.cat([M, questions], dim=2).squeeze(1)
        z = self.z(concat)
        return z

class DMNPlus(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_hop=3, qa=None, update_rule=1, architecture=1):
        super(DMNPlus, self).__init__()
        self.num_hop = num_hop
        self.qa = qa
        self.word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0, sparse=True)
        init.uniform(self.word_embedding.state_dict()['weight'], a=-(3**0.5), b=3**0.5)
        self.criterion = nn.CrossEntropyLoss(size_average=False)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.update_rule = update_rule

        self.input_module = InputModule(vocab_size, hidden_size)
        self.question_module = QuestionModule(vocab_size, hidden_size)
        self.memory = EpisodicMemory(hidden_size, arch=architecture)
        self.update_net = MemoryUpdateNetwork(hidden_size, hidden_size)
        self.answer_module = AnswerModule(vocab_size, hidden_size)

    def forward(self, contexts, questions):
        '''
        contexts.size() -> (#batch, #sentence, #token) -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #token) -> (#batch, 1, #hidden)
        '''
        facts = self.input_module(contexts, self.word_embedding)
        questions = self.question_module(questions, self.word_embedding)
        M = questions
        if self.update_rule == 1:
            for hop in range(self.num_hop):
                new_M = self.memory(facts, questions, M)
                M = self.gru(new_M, M.transpose(0, 1))[1]
                M = M.transpose(0, 1)
        elif self.update_rule == 2:
            for hop in range(self.num_hop):
                new_M = self.memory(facts, questions, M)
                M = self.update_net(new_M, M, questions)
        preds = self.answer_module(M, questions)
        return preds

    def interpret_indexed_tensor(self, var):
        if len(var.size()) == 3:
            # var -> n x #sen x #token
            for n, sentences in enumerate(var):
                for i, sentence in enumerate(sentences):
                    s = ' '.join([self.qa.IVOCAB[elem.data[0]] for elem in sentence])
                    print('' + str(n) + 'th of batch, ' + str(i) + 'th sentence, ' + str(s) + '')
        elif len(var.size()) == 2:
            # var -> n x #token
            for n, sentence in enumerate(var):
                s = ' '.join([self.qa.IVOCAB[elem.data[0]] for elem in sentence])
                print(str(n) + 'th of batch, ' + str(s) + '')
        elif len(var.size()) == 1:
            # var -> n (one token per batch)
            for n, token in enumerate(var):
                s = self.qa.IVOCAB[token.data[0]]
                print('' + str(n) + 'th of batch, ' + str(s) + '')

    def get_loss(self, contexts, questions, targets):
        output = self.forward(contexts, questions)
        loss = self.criterion(output, targets)
        reg_loss = 0
        for param in self.parameters():
            reg_loss += 0.001 * torch.sum(param * param)
        preds = F.softmax(output)
        _, pred_ids = torch.max(preds, dim=1)
        corrects = (pred_ids.data == answers.data)
        acc = torch.mean(corrects.float())
        return loss + reg_loss, acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--update_rule', type=int, default=1)
    parser.add_argument('--architecture', type=int, default=1)
    args = parser.parse_args()
    for run in range(10):
        for task_id in range(1, 21):
            dset = BabiDataset(task_id)
            vocab_size = len(dset.QA.VOCAB)
            hidden_size = 80

            model = DMNPlus(hidden_size, vocab_size, num_hop=3, qa=dset.QA, update_rule=args.update_rule, architecture=args.architecture)
            early_stopping_cnt = 0
            early_stopping_flag = False
            best_acc = 0
            optim = torch.optim.Adam(model.parameters(), lr=0.005)


            for epoch in range(256):
                dset.set_mode('train')
                train_loader = DataLoader(
                    dset, batch_size=100, shuffle=True, collate_fn=pad_collate
                )

                model.train()
                if not early_stopping_flag:
                    total_acc = 0
                    cnt = 0
                    for batch_idx, data in enumerate(train_loader):
                        optim.zero_grad()
                        contexts, questions, answers = data
                        batch_size = contexts.size()[0]
                        contexts = Variable(contexts.long())
                        questions = Variable(questions.long())
                        answers = Variable(answers)

                        loss, acc = model.get_loss(contexts, questions, answers)
                        loss.backward()
                        total_acc += acc * batch_size
                        cnt += batch_size

                        if batch_idx % 20 == 0:
                            print('[Task ' + str(task_id) + ', Epoch ' + str(epoch) + '] [Training] loss : ' + str(loss.data[0].item()) + ', acc : ' + str((total_acc / cnt).item()) + ', batch_idx : ' + str(batch_idx) + '')
                        optim.step()

                    dset.set_mode('valid')
                    valid_loader = DataLoader(
                        dset, batch_size=100, shuffle=False, collate_fn=pad_collate
                    )

                    model.eval()
                    total_acc = 0
                    cnt = 0
                    for batch_idx, data in enumerate(valid_loader):
                        contexts, questions, answers = data
                        batch_size = contexts.size()[0]
                        contexts = Variable(contexts.long())
                        questions = Variable(questions.long())
                        answers = Variable(answers)

                        _, acc = model.get_loss(contexts, questions, answers)
                        total_acc += acc * batch_size
                        cnt += batch_size

                    total_acc = total_acc / cnt
                    if total_acc > best_acc:
                        best_acc = total_acc
                        best_state = model.state_dict()
                        early_stopping_cnt = 0
                    else:
                        early_stopping_cnt += 1
                        if early_stopping_cnt > 20:
                            early_stopping_flag = True

                    print('[Run ' + str(run) + ', Task ' + str(task_id) + ', Epoch ' + str(epoch) + '] [Validate] Accuracy : ' + str(total_acc.item()) + '')
                    with open('log.txt', 'a') as fp:
                        fp.write('[Run ' + str(run) + ', Task ' + str(task_id) + ', Epoch ' + str(epoch) + '] [Validate] Accuracy : ' + str(total_acc.item()) + '' + '\n')
                    if total_acc == 1.0:
                        break
                else:
                    print('[Run ' + str(run) + ', Task ' + str(task_id) + '] Early Stopping at Epoch ' + str(epoch) + ', Valid Accuracy : ' + str(best_acc.item()) + '')
                    break

            dset.set_mode('test')
            test_loader = DataLoader(
                dset, batch_size=100, shuffle=False, collate_fn=pad_collate
            )
            test_acc = 0
            cnt = 0

            for batch_idx, data in enumerate(test_loader):
                contexts, questions, answers = data
                batch_size = contexts.size()[0]
                contexts = Variable(contexts.long())
                questions = Variable(questions.long())
                answers = Variable(answers)

                model.load_state_dict(best_state)
                _, acc = model.get_loss(contexts, questions, answers)
                test_acc += acc * batch_size
                cnt += batch_size
            print('[Run ' + str(run) + ', Task ' + str(task_id) + ', Epoch ' + str(epoch) + '] [Test] Accuracy : ' + str((test_acc / cnt).item()) + '')
            os.makedirs('models', exist_ok=True)
            with open('models/task' + str(task_id) + '_epoch' + str(epoch) + '_run' + str(run) + '_acc' + str(test_acc/cnt) + '.pth', 'wb') as fp:
                torch.save(model.state_dict(), fp)
            with open('log.txt', 'a') as fp:
                fp.write('[Run ' + str(run) + ', Task ' + str(task_id) + ', Epoch ' + str(epoch) + '] [Test] Accuracy : ' + str(total_acc.item()) + '' + '\n')
