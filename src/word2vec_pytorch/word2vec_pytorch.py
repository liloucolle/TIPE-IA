# word2vec_skipgram.py
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from text_utils.text_utils import tokenisation, frequence_mots, vocabulaire, creation_paires
from content.echantillon_texte import test_simple


# Tokenize each line into list of words
tokens = [tokenisation(line) for line in test_simple]


# ---------------------------
# 2) Build vocab and indices
# ---------------------------
min_count = 1
word_freq = frequence_mots(tokens)
vocab, stoi, itos = vocabulaire(word_freq)
V = len(vocab)

print(f"Vocab size: {V}")


# Creation des paires
pairs = creation_paires(tokens, stoi, taille_fenetre=1)
print(f"Training pairs: {len(pairs)}")

# Unigram table for negative sampling (power=0.75 is Word2Vec classic)
# Build a tensor of unigram probabilities
unigram = torch.tensor([word_freq[itos[i]] for i in range(V)], dtype=torch.float)
unigram = unigram.pow(0.75)
unigram = unigram / unigram.sum()


# ---------------------------
# 4) Pytorch-formatted Datasets
# ---------------------------
class SkipGramSoftmaxDS(Dataset):
    """Each item: (center, context) for full-softmax training."""
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        return self.pairs[i]

class SkipGramNegSampleDS(Dataset):
    """Each item: center, pos, negatives (indices)."""
    def __init__(self, pairs, V, unigram_probs, K=5):
        self.pairs = pairs
        self.V = V
        self.K = K
        # cumulative distribution for torch.multinomial
        self.unigram = unigram_probs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        c, pos = self.pairs[i]
        # sample K negatives (avoid pos if desired)
        negs = torch.multinomial(self.unigram, self.K, replacement=True)
        return c, pos, negs


# ---------------------------
# 5) Model (two embedding tables)
# ---------------------------
class SkipGram(nn.Module):
    """
    in_embed: W  (V x N) — center embeddings (stored row-wise for convenience)
              each row W[w, :] is the center-word embedding for Vw
    out_embed: W' (V x N) — context embeddings (stored row-wise for convenience)
              each row W'[w, :] is the context-word embedding for Uw
    """
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, emb_dim)
        self.out_embed = nn.Embedding(vocab_size, emb_dim)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize like Word2Vec: small uniform
        initrange = 0.5 / self.in_embed.embedding_dim
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed.weight.data.zero_()  # often zero or small init

    def forward_softmax(self, centers):
        """
        centers: (B,) int64 → returns logits over vocab: (B, V)
        logits = h @ W'^T, where h = v_{w_I}
        """
        h = self.in_embed(centers)                      # (B, N)
        logits = h @ self.out_embed.weight.t()          # (B, V)
        return logits

    def forward_neg(self, centers, positives, negatives):
        """
        centers:   (B,)
        positives: (B,)
        negatives: (B, K)
        Returns the logistic loss terms for negative sampling.
        """
        B = centers.size(0)
        h = self.in_embed(centers)                      # (B, N)

        # Positive score: u_{w_O}^T v_{w_I}
        pos_vecs = self.out_embed(positives)            # (B, N)
        pos_score = torch.sum(h * pos_vecs, dim=1)      # (B,)

        # Negative scores
        neg_vecs = self.out_embed(negatives)            # (B, K, N)
        # Dot (B,1,N) · (B,K,N) → (B,K)
        neg_score = torch.bmm(neg_vecs, h.unsqueeze(2)).squeeze(2)  # (B, K)

        # Loss: -log σ(pos) - sum log σ(-neg)
        loss_pos = F.logsigmoid(pos_score).sum()
        loss_neg = F.logsigmoid(-neg_score).sum()
        loss = -(loss_pos + loss_neg) / B
        return loss

    def get_embeddings(self):
        """Return the learned word vectors (choose input table, common practice)."""
        return self.in_embed.weight.detach().cpu()


# ---------------------------
# 6) Training utilities
# ---------------------------
def train_softmax(model, ds, epochs=5, batch_size=128, lr=1e-3, device="cpu"):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()  # includes softmax

    for ep in range(1, epochs+1):
        total = 0.0
        n = 0
        for centers, targets in dl:
            centers = centers.to(device)
            targets = targets.to(device)

            logits = model.forward_softmax(centers)  # (B, V)
            loss = criterion(logits, targets)        # -log P(w_O|w_I)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * centers.size(0)
            n += centers.size(0)
        print(f"[softmax] epoch {ep}: loss = {total/n:.4f}")

def train_neg(model, ds, epochs=5, batch_size=1024, lr=2.5e-3, device="cpu"):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        total = 0.0
        n = 0
        for centers, positives, negatives in dl:
            centers   = centers.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            loss = model.forward_neg(centers, positives, negatives)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * centers.size(0)
            n += centers.size(0)
        print(f"[neg] epoch {ep}: loss = {total/n:.4f}")


# ---------------------------
# 7) Choose mode and train
# ---------------------------
emb_dim = 100 # embedding dimension
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SkipGram(V, emb_dim)

mode = "neg"        # "softmax" or "neg"
if mode == "softmax":
    ds = SkipGramSoftmaxDS(pairs)
    train_softmax(model, ds, epochs=100, batch_size=128, lr=0.001, device=device)
else:
    ds = SkipGramNegSampleDS(pairs, V, unigram_probs=unigram, K=5)
    train_neg(model, ds, epochs=100, batch_size=1024, lr=0.01, device=device)

# ---------------------------
# 8) Use the embeddings
# ---------------------------
emb = model.get_embeddings()   # (V, emb_dim)

# helper: cosine similarity
def most_similar(word, topk=5):
    if word not in stoi:
        return []
    widx = stoi[word]
    wvec = emb[widx]                               # (N,)
    sims = F.cosine_similarity(emb, wvec.unsqueeze(0))  # (V,)
    vals, idxs = torch.topk(sims, k=topk+1)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        if i == widx:  # skip itself
            continue
        out.append((itos[i], float(v)))
        if len(out) == topk:
            break
    return out

print("\nNearest neighbors:")
for q in ["fox", "dog", "learning", "quick"]:
    print(q, "→", most_similar(q, topk=5))
