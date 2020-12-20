import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
#word2idx['mask'] = len(word2idx)+1
#idx2word[len(word2idx)-1] = 'mask'
idx_five_seq = corpus.get_idx_seq(five_seq)
idx_seven_seq = corpus.get_idx_seq(seven_seq)
np.random.shuffle(idx_five_seq)
np.random.shuffle(idx_seven_seq)

embed_size = 64    
hidden_size = 512
n_layers = 1     
num_epochs = 2   
batch_size = 256  
learning_rate = 5e-5
vocab_size = len(word2idx)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

def gumbel_softmax_soft(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return y

softmax2 = nn.Softmax(dim=2)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers):
        super(Encoder,self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size,embed_size) 
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, batch_first=True)
        
    def forward(self, x):
        x = self.embed(x)
        output,(hidden,cell) = self.lstm(x)

        return output, hidden,cell
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers):
        super(Decoder,self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size,embed_size) 
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, batch_first=True)

        
    def forward(self, x, hidden,cell):
        #x (batch_size)->(batch_size,seq_len = 1)
        x = x.unsqueeze(1)
        x = self.embed(x)
        output, (hidden, cell) = self.lstm(x, (hidden,cell))      #output(batch_size,seq_len =1,hidden_size) 
        
       
        return output, hidden, cell
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.linear_out = nn.Linear(2 * hidden_size, hidden_size)
    
    def forward(self, encoder_output, decoder_output):
        # encoder_output: [batch_size, 13, hidden_size]
        # decoder_output: [batch_size, 1, hidden_size]
        batch_size = encoder_output.shape[0]
        output_len = decoder_output.shape[1]
        input_len = encoder_output.shape[1]
        encoder_output_trans = encoder_output.transpose(1, 2)# [batch_size, hidden_size, 13]
        attn = torch.bmm(decoder_output, encoder_output_trans)  # [batch_size, 1, 13]
        attn = softmax2(attn)   
        
        context = torch.bmm(attn, encoder_output)   # [batch_size, 1, hidden_size]
        output = torch.cat((context, decoder_output), dim=2)  # [batch_size, 1, 2*hidden_size]
        output = torch.tanh(self.linear_out(output)) # [batch_size, 1, hidden_size]
    
        return output, attn
    
import random

logsoftmax = nn.LogSoftmax(dim = 1)
crossentropy_reduce = nn.CrossEntropyLoss(reduction='none')
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder,attention,max_len):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.max_len = max_len
        self.linear = nn.Linear(self.encoder.hidden_size,self.encoder.vocab_size) # we should not output 'mask' 
        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, x, gumbel = 0, targets = 0, teacher_force = 0.5):
        #x: (batch size,seq_len)
        #trg: (batch size,seq_len)
        batch_size = x.shape[0]
        max_len = self.max_len
        vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(max_len,batch_size,vocab_size)
        outputs = outputs.cuda()
        inputss = torch.zeros(max_len+1,batch_size,vocab_size)
        inputss = inputss.cuda()
        probs = torch.zeros(max_len,batch_size)
        encoder_output ,hidden,cell = self.encoder(x)
        # inputs -> batch_size * <bos>
        inputs = torch.zeros(batch_size,vocab_size)
        inputs[:,0] = 1 
        inputs_idx = torch.argmax(inputs, dim = 1,keepdim = False)
        inputs_idx = inputs_idx.cuda()
        inputs_idx = inputs_idx.long() #input_idx->(batch_size)
        inputs = inputs.cuda()
        inputss[0] = inputs        

        if gumbel > 0:
            for t in range(max_len):
                output, hidden, cell = self.decoder(inputs_idx, hidden, cell)
                output, _ = self.attention(encoder_output,hidden.transpose(0,1))#(batch_size,seq_len=1,hidden_size)
                output = self.linear(output)
                output = output.squeeze(1) # output (batch_size, seq_len=1, vocab_size)->(batch_size,vocab_size)
                outputs[t] = output
                inputs_grad = gumbel_softmax(output, gumbel)
                inputs_idx = torch.argmax(inputs_grad,dim=1,keepdim=False)
                inputs_idx = inputs_idx.cuda()
                inputs_idx = inputs_idx.long()
                inputss[t+1] = inputs_grad
                #prob = torch.sum(torch.mul(logsoftmax(output),inputs),dim=1)
                prob = crossentropy_reduce(output,inputs_idx)
                probs[t] = prob
        elif gumbel == 0:
            for t in range(max_len):
                output, hidden,cell = self.decoder(inputs_idx, hidden,cell)
                output, _ = self.attention(encoder_output,hidden.transpose(0,1))
                output = self.linear(output)
                output = output.squeeze(1)
                outputs[t] = output
                inputs_idx = torch.argmax(output,dim=1,keepdim=False)
                inputs_idx = inputs_idx.cuda()
                inputs_idx = inputs_idx.long()
                use_teacher_force = random.random() <= teacher_force 
                inputs_idx = (targets[:,t] if use_teacher_force else inputs_idx)
        
            
        outputs = outputs.transpose(0,1)
        inputss = inputss.transpose(0,1)
        probs = probs.transpose(0,1)
        
        return outputs, inputss, probs
    
enc57 = Encoder(vocab_size, embed_size, hidden_size, n_layers)
dec57 = Decoder(vocab_size, embed_size, hidden_size, n_layers)
enc75 = Encoder(vocab_size, embed_size, hidden_size, n_layers)
dec75 = Decoder(vocab_size, embed_size, hidden_size, n_layers)
atten57 = Attention(hidden_size)
atten75 = Attention(hidden_size)
model57 = Seq2Seq(enc57, dec57,atten57,17)
model75 = Seq2Seq(enc75, dec75,atten75,13)
model57.cuda()
model75.cuda()
model57.load_state_dict(torch.load('model/model57_att_t=0.1_cross%kl=100.pt'))

test = idx_five_seq[round(np.random.rand()*len(idx_five_seq))-1]
test = torch.tensor(test, dtype = torch.long,requires_grad=False)
test = test.unsqueeze(0).cuda()
output57, tryonehot57, tryprob57 = model57(test,gumbel=1)
output7s = []
for i in range(10):
    inputs_grad57 = gumbel_softmax(output57,0.1 )
    output7 = torch.argmax(inputs_grad57,dim=2,keepdim=False)
    output7 = output7.squeeze(0).cpu().numpy()
    trans7 = ''.join([idx2word[i] for i in output7])
    output7s.append(trans7)


file = open('output/sampled_five2seven.txt','w',encoding='UTF-8')
test = test.squeeze(0).cpu().numpy()
file.write('The input five sentence is')
file.write('\t')
file.write(''.join([idx2word[i] for i in test]))
file.write('\n')
file.write('The input seven sentence can be')
file.write('\n')
for i, sentence in enumerate(output7s):
    file.write(str(sentence))
    file.write('\n')
file.close()
