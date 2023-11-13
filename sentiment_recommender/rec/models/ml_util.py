import torch, torchtext
from torch import nn, functional as F
import numpy as np

from .stop_words import SW, SPEC_CHARS

sentiment = ['sadness', 'anger', 'love', 'surprise', 'fear', 'happy']

def stopwords(text):
  text = [word.lower() for word in text.split() if word.lower() not in SW] # removing the stop words and lowercasing the selected words
  return " ".join(text) 

def spec_char_clean(text):
  for char in SPEC_CHARS:
    text = text.replace(char, '')
  return text

def number_removal(text):
  if type(text) == int:
    return text
  else:
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def clean_input(text):
  return number_removal(spec_char_clean(stopwords(text)))

class Sentences(torch.utils.data.Dataset):
  def __init__(self, fn):
    lengths = []
    convert = { u: n for n, u in enumerate(fn['Emotion'].unique()) }
    fn['Emotion'] = fn['Emotion'].apply(lambda u: convert[u])               # 12 unique words should be assigned integers starting from 0
    tokenizer = torchtext.data.utils.get_tokenizer('spacy', 'en_core_web_sm') # tokenizer using spaCy
    for i in range(len(fn['Text'])):
      lengths.append(len(tokenizer(fn['Text'].iat[i].strip())))                   # store the number of tokens in each sentence to beused in get item
    string = ' '.join([fn['Text'].iat[i].strip() 
    for i in range(len(fn['Text']))])                  # combine everything into one single string
    toks = tokenizer(string)                                                # tokenize the single string

    self.vocab = torchtext.vocab.build_vocab_from_iterator([toks])
    self.sentiment = fn['Emotion'].values
    self.text = fn['Text'].values
    self.length = lengths
    self.toks = torch.LongTensor([self.vocab[tok] for tok in toks])

  def __len__(self):
      return len(self.length)

  def __getitem__(self, i):
      sum = 0
      for x in range(i):
        sum += self.length[x]
      return (self.sentiment[i], self.toks[sum: sum + self.length[i]])          # return the sentiment and related tokns for a specific tweet
  
  def input_to_tensor(self, string):
      in_toks = string.split(" ")
      return torch.LongTensor([self.vocab[i] for i in in_toks])

class SentenceModel(nn.Module):                                                 # takes in a sentence, and outputs predicted sentiment
  def __init__(self, vocab_size, embedding_dim, lstm_dim, 
                   n_cats, n_layers = 2, drop_prob = 0.5):
    super().__init__()                                                      #constructor for parent class
    self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)          #use word embeddings 
    self.lstm = torch.nn.LSTM(embedding_dim, lstm_dim, n_layers,
                              dropout=drop_prob, batch_first=True)          #LSTM layer
    self.linear = nn.Linear(lstm_dim, n_cats)
    nn.init.xavier_uniform_(self.embedding.weight.data)
    nn.init.xavier_uniform_(self.linear.weight.data)
        
  def forward(self, text):
    emb = self.embedding(text)
    lstm_out, _ = self.lstm(emb)
    out = self.linear(lstm_out)
    return torch.mean(out, dim=1)                                           # certain dimensions required so take mean to reduce

def get_weights(tensor=None, model=None): #tensor should have shape ([x, y, z, ...]) (1 dim), NOT ([[x, y, z, ...]]), the function itself unsqueezes the input tensor for you
  tensor = tensor.unsqueeze(0)
  #device = torch.device('cuda:0')
  #model.to(device)
  m = nn.ReLU()
  s = nn.Softmax(dim=1)
  if not tensor.numel():
    return tensor
  with torch.no_grad():
    update_tensor = model(tensor)
    relud_logged = torch.log(m(update_tensor))
    return s(relud_logged).squeeze()

def normalize(p): # makes sure the distribution probability list adds to 1
  if sum(p) != 1.0:
    p = np.asarray(p)*(1.0/sum(p))
  return p
