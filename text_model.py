"""Class for text data."""
import string
import numpy as np
import torch
import json
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn as nn
import torchtext

def get_text_encoder(opt,texts_to_build_vocab, word_embed_dim, lstm_hidden_dim):
    if opt.text_encoder == 'LSTM':
        return TextLSTMModel(texts_to_build_vocab, word_embed_dim, lstm_hidden_dim)
    elif opt.text_encoder == 'BIGRU':
        return TextBIGRUModel(texts_to_build_vocab, word_embed_dim, lstm_hidden_dim)

class SimpleVocab(object):

    def __init__(self):
        super(SimpleVocab, self).__init__()
        self.word2id = {}
        self.wordcount = {}
        self.word2id['<UNK>'] = 0
        self.word2id['<AND>'] = 1
        self.word2id['<BOS>'] = 2
        self.word2id['<EOS>'] = 3
        self.wordcount['<UNK>'] = 9e9
        self.wordcount['<AND>'] = 9e9
        self.wordcount['<BOS>'] = 9e9
        self.wordcount['<EOS>'] = 9e9

    def tokenize_text(self,text):   
        text = text.encode('ascii', 'ignore').decode('ascii')
        trans=str.maketrans({key: None for key in string.punctuation})
        tokens = str(text).lower().translate(trans).strip().split()
        return tokens
    
    def add_text_to_vocab(self,text):   
        tokens = self.tokenize_text(text)
        for token in tokens:
            if not token in self.word2id:
                self.word2id[token] = len(self.word2id)
                self.wordcount[token] = 0
            self.wordcount[token] += 1

    def threshold_rare_words(self, wordcount_threshold=3):  
        for w in self.word2id:
            if self.wordcount[w] < wordcount_threshold:
                self.word2id[w] = 0

    def encode_text(self, text):               
        tokens = self.tokenize_text(text)
        x = [self.word2id.get(t, 0) for t in tokens]          
        return x
    
    def get_size(self):               
        return len(self.word2id)

class TextLSTMModel(torch.nn.Module):

    def __init__(self,
                 texts_to_build_vocab = None,
                 word_embed_dim = 512,
                 lstm_hidden_dim = 512):

        super(TextLSTMModel, self).__init__()

        self.vocab = SimpleVocab()
        if texts_to_build_vocab != None:
            for text in texts_to_build_vocab:
                self.vocab.add_text_to_vocab(text)
        else:
            vocab_data = json.load(open("simplevocab.json"))
            self.vocab.word2id = vocab_data['word2id']
            self.vocab.wordcount = vocab_data['wordcount']

        vocab_size = self.vocab.get_size()
        self.word_embed_dim = word_embed_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.embedding_layer = nn.Embedding(vocab_size, word_embed_dim)
        self.lstm = nn.LSTM(word_embed_dim, lstm_hidden_dim)

        word2idx = self.vocab.word2id
        self.init_weights('glove', word2idx, word_embed_dim)
    
    def init_weights(self, wemb_type, word2idx, word_dim):
        if wemb_type.lower() == 'random_init':
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText()
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe(cache = '/opt/data/private/kevin/data/vocab/.vector_cache')#change path
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embedding_layer.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))


    def forward(self, x):
        """ input x: list of strings"""
        if type(x) is list:
            if type(x[0]) is str or type(x[0]) is unicode:
                x = [self.vocab.encode_text(text) for text in x]

        assert type(x) is list
        assert type(x[0]) is list
        assert type(x[0][0]) is int
        return self.forward_encoded_texts(x)

    def forward_encoded_texts(self, texts):
        # to tensor
        lengths = [len(t) for t in texts]
        itexts = torch.zeros((np.max(lengths), len(texts))).long()
        for i in range(len(texts)):
            itexts[:lengths[i], i] = torch.tensor(texts[i]) # shape(length,batch)

        # embed words
        itexts = torch.autograd.Variable(itexts).cuda()     # shape(length,batch)
        etexts = self.embedding_layer(itexts)               # shape(length,batch,dim)         

        # lstm
        lstm_output, _ = self.forward_lstm_(etexts)      #lstm_output shape(length,batch,hidden_num*directions)
        
        wrd = lstm_output.permute(1,0,2).contiguous()
        cap_len = torch.Tensor(lengths).cuda()

        max_len = int(cap_len.max())
        mask = torch.arange(max_len).expand(cap_len.size(0), max_len).to(cap_len.device)
        mask = (mask < cap_len.long().unsqueeze(1)).unsqueeze(-1)#N,L,1
        stc = torch.max(wrd.masked_fill(mask==0,0),dim=1)[0]
        return wrd, stc, cap_len


    def forward_lstm_(self, etexts):
        batch_size = etexts.shape[1]
        first_hidden = (torch.zeros(1, batch_size, self.lstm_hidden_dim),
                        torch.zeros(1, batch_size, self.lstm_hidden_dim))
        first_hidden = (first_hidden[0].cuda(), first_hidden[1].cuda())
        #first_hidden = (first_hidden[0], first_hidden[1])
        lstm_output, last_hidden = self.lstm(etexts, first_hidden)
        return lstm_output, last_hidden

class TextBIGRUModel(torch.nn.Module):
    def __init__(self,
                 texts_to_build_vocab = None,
                 word_embed_dim = 512,
                 lstm_hidden_dim = 512):

        super(TextBIGRUModel, self).__init__()

        self.vocab = SimpleVocab()
        if texts_to_build_vocab != None:
            for text in texts_to_build_vocab:
                self.vocab.add_text_to_vocab(text)
        else:
            vocab_data = json.load(open("simplevocab.json"))
            self.vocab.word2id = vocab_data['word2id']
            self.vocab.wordcount = vocab_data['wordcount']

        vocab_size = self.vocab.get_size()
        self.word_embed_dim = word_embed_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.embedding_layer = torch.nn.Embedding(vocab_size, word_embed_dim)
        self.GRU = torch.nn.GRU(word_embed_dim, lstm_hidden_dim,num_layers=1, batch_first=True, bidirectional=True)
        self.init_weights()

    def init_weights(self):
        self.embedding_layer.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        """ input x: list of strings"""
        if type(x) is list:
            if type(x[0]) is str or type(x[0]) is unicode:
                x = [self.vocab.encode_text(text) for text in x]

        assert type(x) is list
        assert type(x[0]) is list
        assert type(x[0][0]) is int
        return self.forward_encoded_texts(x)

    def forward_encoded_texts(self, texts):

        lengths = [len(t) for t in texts]
        itexts = torch.zeros(len(texts),(np.max(lengths))).long()
        for i in range(len(texts)):
            itexts[i,:lengths[i]] = torch.tensor(texts[i]) # shape(length,batch)

        # embed words
        itexts = torch.autograd.Variable(itexts).cuda()     # shape(length,batch)
        x_emb = self.embedding_layer(itexts)               # shape(length,batch,

        self.GRU.flatten_parameters()
        packed = pack_padded_sequence(x_emb, lengths, batch_first=True,enforce_sorted=False)
        lengths = torch.Tensor(lengths).cuda()

        # Forward propagate RNN
        out, _ = self.GRU(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] + cap_emb[:, :, cap_emb.size(2) // 2:]) / 2

        wrd = cap_emb.contiguous()
        cap_len = lengths
        max_len = int(cap_len.max())
        mask = torch.arange(max_len).expand(cap_len.size(0), max_len).to(cap_len.device)
        mask = (mask < cap_len.long().unsqueeze(1)).unsqueeze(-1)#N,L,1
        stc = torch.max(wrd.masked_fill(mask==0,0),dim=1)[0]

        return wrd, stc, cap_len
