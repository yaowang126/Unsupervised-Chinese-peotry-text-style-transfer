import pickle
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

file=open(r"F:/Columbia/DL_syst_perf/project/five","rb")
five=pickle.load(file) 
file=open(r"F:/Columbia/DL_syst_perf/project/seven","rb")
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
batch_size = 128   
learning_rate = 0.002 
vocab_size = len(corpus.dictionary)

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
model.load_state_dict(torch.load('model/lm7_best_2.th'))

softmax = nn.Softmax(dim = 1)

seven_generator = []
for i in range (100):  
    with torch.no_grad():
        example = []
        example.append(0)
        bos = torch.zeros(len(word2idx))
        bos[0] = 1.0
        bos = bos.unsqueeze(0)
        bos = bos.unsqueeze(0)
        bos = bos.cuda()
        state = (torch.zeros(num_layers, 1, hidden_size).cuda(),
                     torch.zeros(num_layers, 1, hidden_size).cuda())
        idx = 0
        for i in range(17):
            outputs, state = model(bos,state)
            idx = torch.multinomial(softmax(outputs), 1,replacement=True).item()
            bos = torch.zeros(len(word2idx))
            bos[idx] = 1.0
            bos = bos.unsqueeze(0)
            bos = bos.unsqueeze(0)
            bos = bos.cuda()
            example.append(idx)
    seven_generator.append([idx2word[word_idx] for word_idx in example])
seven_generator = [''.join(seq) for seq in seven_generator]

file = open('output/sampled_seven_sentences.txt','w',encoding='UTF-8');
for i, sentence in enumerate(seven_generator):
    file.write(str(sentence))
    file.write('\n')
file.close()