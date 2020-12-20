import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.nn.utils import clip_grad_norm_

file=open("data/five","rb")
five=pickle.load(file) 
file=open("data/seven","rb")
seven=pickle.load(file) 

five_seq = []
for i, poet in enumerate(five):
    for j, seq in enumerate(poet):
        five_seq.append(seq)
seven_seq = []
for i, poet in enumerate(seven):
    for j, seq in enumerate(poet):
        seven_seq.append(seq)

class Dictionary(object):

    def __init__(self):
        self.word2idx = {} 
        self.idx2word = {} 
        self.idx = 0
    
    def add_word(self, word):
        if not word in self.word2idx: 
            self.word2idx[word] = self.idx 
            self.idx2word[self.idx] = word 
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx) 
    
class Corpus(object):
    
    def __init__(self):
        self.dictionary = Dictionary()
 
    def get_dict(self, seq_list):
        for seq in (seq_list):
            words = ['<bos>']+ [word for word in seq] + ['<eos>']
            for word in words:
                self.dictionary.add_word(word)  
        return self.dictionary.word2idx, self.dictionary.idx2word
    
    def get_idx_seq(self, seq_list):
        idx_seq = []
        for seq in (seq_list):
            idx_seq.append([self.dictionary.word2idx[word] for word in ['<bos>']+[word1 for word1 in seq]+['<eos>']])
        return idx_seq
    

corpus = Corpus()
word2idx, idx2word = corpus.get_dict(five_seq+seven_seq)
idx_five_seq = corpus.get_idx_seq(five_seq)
idx_seven_seq = corpus.get_idx_seq(seven_seq)
np.random.shuffle(idx_five_seq)
np.random.shuffle(idx_seven_seq)

embed_size = 64    
hidden_size = 256
num_layers = 1     
num_epochs = 30     
batch_size = 1024   
learning_rate = 0.002 
vocab_size = len(corpus.dictionary)


x = [seq[:-1] for seq in idx_five_seq]
y = [seq[1:] for seq in idx_five_seq]
x_train = torch.tensor(x[:550000], dtype = torch.long)
y_train = torch.tensor(y[:550000], dtype = torch.long)
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle=True)
x_test = torch.tensor(x[550000:], dtype = torch.long)
y_test = torch.tensor(y[550000:], dtype = torch.long)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset,batch_size = 1024)



class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Linear(vocab_size, embed_size,bias = False)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        #self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, state):
        x = self.embed(x)
        out,(h,c) = self.lstm(x,state)
        #out,(h,c) = self.lstm2(out)
        out = out.reshape(out.size(0)*out.size(1),out.size(2))
        out = self.linear(out) #(batch_size*sequence_length, hidden_size)->(batch_size*sequence_length, vacab_size)
        
        return out,(h,c)
    
    
model = RNNLM(vocab_size, embed_size, hidden_size, num_layers)
model.cuda()
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

import time

epoch_losses = []
val_losses = []
#train
for epoch in range(num_epochs):
    start = time.time()
    model.train()
    epoch_loss = 0.0
    for inputs, targets in train_loader:
        #inputs = inputs.cuda()
        #inputs = torch.nn.functional.one_hot(inputs,vocab_size)
        #inputs = inputs.float()
        inputs = torch.zeros(inputs.shape[0], inputs.shape[1], len(word2idx)).scatter_(dim=-1,
                           index=torch.unsqueeze(inputs,-1),
                               value=1).cuda()
        targets = targets.cuda()
        outputs,states = model(inputs,None)
        optimizer.zero_grad()
        loss = criterion(outputs,targets.reshape(-1))
        loss.backward()
        clip_grad_norm_(model.parameters(),0.5)
        optimizer.step()
        epoch_loss += loss.item()*1024
    epoch_loss = epoch_loss / len(train_dataset)
    
    model.eval()
    val_loss = 0.
    with torch.no_grad():
        for i,(inputs, targets) in enumerate(test_loader):
            inputs = torch.zeros(inputs.shape[0], inputs.shape[1], len(word2idx)).scatter_(dim=-1,
                       index=torch.unsqueeze(inputs,-1),
                       value=1)
            inputs = inputs.cuda()
            targets = targets.cuda()
            with torch.no_grad():
                outputs, hidden = model(inputs,None)
            loss = criterion(outputs,targets.reshape(-1))
            val_loss += loss.item()*inputs.shape[0]

    val_loss = val_loss / len(test_dataset)
    
    if len(val_losses) == 0 or val_loss < min(val_losses):
        print("best model, val loss: ", val_loss)
        torch.save(model.state_dict(), "model/lm5_best_myself.th")

    end = time.time()
    print(end-start)
    print('train loss %f'%epoch_loss)
    print('val loss%f'%val_loss)
    epoch_losses.append(epoch_loss)  
    val_losses.append(val_loss)
