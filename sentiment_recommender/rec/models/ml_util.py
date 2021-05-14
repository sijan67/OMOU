import torch, torchtext
from torch import nn, optim, functional as F
import pandas as pd, csv
import pdb
import random
import numpy as np
import os
import io
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
text = pd.read_csv(os.path.join(BASE_DIR,r'datasets\Emotion_final.csv'))

sentiment = ['sadness', 'anger', 'love', 'surprise', 'fear', 'happy']

def remove_punctuation(text): #function for removing punctuation
    import string 
    translator = str.maketrans('', '', string.punctuation) #replace the punctuations with no space
    return text.translate(translator)

class Sentences(torch.utils.data.Dataset):
    def __init__(self, fn):
        lengths = []
        convert = { u: n for n, u in enumerate(fn['Emotion'].unique()) }
        fn['Emotion'] = fn['Emotion'].apply(lambda u: convert[u])               # 12 unique words should be assigned integers starting from 0
        tokenizer = torchtext.data.utils.get_tokenizer('spacy', 'en_core_web_sm')# tokenizer using spaCy
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
        string = remove_punctuation(string)
        in_toks = string.split(" ")
        return torch.LongTensor([self.vocab[i] for i in in_toks])

ds_full = Sentences(text)
n_train = int(0.8 * len(ds_full))
n_test = len(ds_full) - n_train
rng = torch.Generator().manual_seed(291)
ds_train, ds_test = torch.utils.data.random_split(ds_full, [n_train, n_test], rng)

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

model = SentenceModel(len(ds_full.vocab)+2, 64, 64, len(text.Emotion.unique()))
#device = torch.device('cuda:0')
#model.to(device);
#crit = nn.CrossEntropyLoss().to(device)
#opt = optim.SGD(model.parameters(), lr=0.1)
#sched = optim.lr_scheduler.StepLR(opt, 1, gamma=1)

# Model class must be defined somewhere
model.load_state_dict(torch.load(os.path.join(BASE_DIR,r"datasets\sentence_model_state_dict_copy.pth"), map_location=torch.device('cpu')))
model.eval()

ldr = torch.utils.data.DataLoader(ds_test)

def get_weights(tensor=None, model=None): #tensor should have shape ([x, y, z, ...]) (1 dim), NOT ([[x, y, z, ...]]), the function itself unsqueezes the input tensor for you
  tensor = tensor.unsqueeze(0)
  #device = torch.device('cuda:0')
  #model.to(device)
  m = nn.ReLU()
  s = nn.Softmax(dim=1)
  with torch.no_grad():
    update_tensor = model(tensor)
    print(update_tensor)
    relud_logged = torch.log(m(update_tensor))
    print(relud_logged)
    print("Tensor converted to text: " + ' '.join([ds_full.vocab.itos[x] for x in tensor.squeeze()]))
    return s(relud_logged).squeeze()
  
  # this function calculates the appropriate probability distribution for every track based on input "tensor" and our model

tracks = pd.read_csv(os.path.join(BASE_DIR,r"datasets/tracks.csv"))
tracks.drop(inplace=True, columns=['duration_ms','key', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'tempo', 'time_signature', 'loudness'])
tracks.sort_values(by=['popularity'], ascending=False)

new_df = tracks[tracks.popularity >=65]

def normalize(p): # makes sure the distribution probability list adds to 1
    if sum(p) != 1.0:
        p = np.asarray(p)*(1.0/sum(p))
    return p

#user input is x -> we have to convert user input string into tensor "x"
#so input would be "input = get_weights(x, model)"

POPULARITY_WEIGHT = 1.5 # can be changed 
NUM_TRACK_OPTIONS = 25 # can be changed

def music_model(input_string):
  id_tracks = new_df["id"].values.tolist()
  scores_id = []
  seq = ds_full.input_to_tensor(input_string)
  input = get_weights(seq, model)
  for track in new_df.itertuples(index=False):
      score = POPULARITY_WEIGHT*(track.popularity/100) + input[0].item()*(1-track.valence) + input[1].item()*track.energy + input[2].item()*((track.energy + track.danceability)/2) + input[3].item()*(1-track.energy) + input[4].item()*track.valence + input[5].item()*((track.danceability + track.energy)/2)
      scores_id.append([score, track.id])

  distribution = []
  top_track_ids = []
  s = sorted(scores_id, reverse=True)
  s = s[0:NUM_TRACK_OPTIONS]

  for score, id in s:
    distribution.append(score)
    top_track_ids.append(id)
  id = np.random.choice(top_track_ids, p=normalize(distribution)) # picks a random track across a distribution generated by the scores
  print("Distribution: {0}".format(input))
  print("TRACK LINK: https://open.spotify.com/track/" + id)
  actual_track = new_df[new_df['id']==id]
  return [pd.Series.to_string(actual_track['artists'])]
