# word2vec_skipgram.py
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from text_utils.text_utils import tokenisation, frequence_mots, vocabulaire, creation_paires
from content.echantillon_texte import texte_king_and_queen_2 as texte


# Tokeniser chaque ligne en liste de mots
tokens = tokenisation(texte)

# ---------------------------
# 2) Construction du vocabulaire et des indices
# ---------------------------
min_count = 1
word_freq = frequence_mots(tokens)
vocab, stoi, itos = vocabulaire(word_freq)
V = len(vocab)

print(f"Taille du vocabulaire: {V}")


# Création des paires
pairs = creation_paires(tokens, stoi, taille_fenetre=1)
print(f"Paires d'entraînement: {len(pairs)}")

# Table unigramme pour l'échantillonnage négatif (puissance=0.75 est classique pour Word2Vec)
# Construction d'un tenseur de probabilités unigrammes
unigram = torch.tensor([word_freq[itos[i]] for i in range(V)], dtype=torch.float)
unigram = unigram.pow(0.75)
unigram = unigram / unigram.sum()


# ---------------------------
# 4) Datasets formatés pour PyTorch
# ---------------------------
class SkipGramSoftmaxDS(Dataset):
    """Chaque élément: (centre, contexte) pour l'entraînement softmax complet."""
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        return self.pairs[i]

class SkipGramNegSampleDS(Dataset):
    """Chaque élément: centre, positif, négatifs (indices)."""
    def __init__(self, pairs, V, unigram_probs, K=5):
        self.pairs = pairs
        self.V = V
        self.K = K
        # distribution cumulative pour torch.multinomial
        self.unigram = unigram_probs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        c, pos = self.pairs[i]
        # échantillonner K négatifs (éviter pos si désiré)
        negs = torch.multinomial(self.unigram, self.K, replacement=True)
        return c, pos, negs


# ---------------------------
# 5) Modèle (deux matrices d'embeddings)
# ---------------------------
class SkipGram(nn.Module):
    """
    in_embed: W  (V x N) — embeddings centraux (stockés ligne par ligne pour commodité)
              chaque ligne W[w, :] est l'embedding du mot-centre pour Vw
    out_embed: W' (V x N) — embeddings contexte (stockés ligne par ligne pour commodité)
              chaque ligne W'[w, :] est l'embedding du mot-contexte pour Uw
    """
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, emb_dim)
        self.out_embed = nn.Embedding(vocab_size, emb_dim)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialiser comme Word2Vec: uniforme petit
        initrange = 0.5 / self.in_embed.embedding_dim
        # initialise les poids de W avec des valeurs uniformes dans [-initrange, initrange]
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        # initialise les poids de W' à zéro
        self.out_embed.weight.data.zero_()  # souvent zéro ou petite initialisation

    def forward_softmax(self, centers):
        """
        centers: indices des mots centres (B,)
        Retourne les logits pour le softmax complet sur le vocabulaire.
        """
        # Recuperation des embeddings des mots centres
        h = self.in_embed(centers)                      # (B, N)
        # Les logits représentent les scores de similarité non normalisés
        # logits = h x W'^T 
        logits = h @ self.out_embed.weight.t()          # (B, V)
        return logits

    def forward_neg(self, centers, positives, negatives):
        """
        centers:   (B,)
        positives: (B,)
        negatives: (B, K)
        Retourne les termes de perte logistique pour l'échantillonnage négatif.
        """
        B = centers.size(0)
        h = self.in_embed(centers)                      # (B, N)

        # Score positif: u_{w_O}^T v_{w_I}
        pos_vecs = self.out_embed(positives)            # (B, N)
        pos_score = torch.sum(h * pos_vecs, dim=1)      # (B,)

        # Scores négatifs
        neg_vecs = self.out_embed(negatives)            # (B, K, N)
        # Produit scalaire (B,1,N) · (B,K,N) → (B,K)
        neg_score = torch.bmm(neg_vecs, h.unsqueeze(2)).squeeze(2)  # (B, K)

        # Perte: -log σ(pos) - sum log σ(-neg)
        loss_pos = F.logsigmoid(pos_score).sum()
        loss_neg = F.logsigmoid(-neg_score).sum()
        loss = -(loss_pos + loss_neg) / B
        return loss

    def get_embeddings(self):
        """Retourne les vecteurs de mots appris (choisir table d'entrée, pratique commune)."""
        return self.in_embed.weight.detach().cpu()


# ---------------------------
# 6) Utilitaires d'entraînement
# ---------------------------
def train_softmax(model, ds, epochs=5, batch_size=128, lr=1e-3, device="cpu"):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()  # inclut softmax

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
        print(f"[softmax] époque {ep}: perte = {total/n:.4f}")

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
        print(f"[neg] époque {ep}: perte = {total/n:.4f}")


# ---------------------------
# 7) Choisir le mode et entraîner
# ---------------------------
emb_dim = 100 # dimension des embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SkipGram(V, emb_dim)

mode = "softmax"        # "softmax" ou "neg"
if mode == "softmax":
    ds = SkipGramSoftmaxDS(pairs)
    train_softmax(model, ds, epochs=100, batch_size=128, lr=0.01, device=device)
else:
    ds = SkipGramNegSampleDS(pairs, V, unigram_probs=unigram, K=5)
    train_neg(model, ds, epochs=100, batch_size=1024, lr=0.01, device=device)

# ---------------------------
# 8) Utiliser les embeddings
# ---------------------------
emb = model.get_embeddings()   # (V, emb_dim)

# fonction auxiliaire: similarité cosinus
def most_similar(word, topk=5):
    if word not in stoi:
        return []
    widx = stoi[word]
    wvec = emb[widx]                               # (N,)
    sims = F.cosine_similarity(emb, wvec.unsqueeze(0))  # (V,)
    vals, idxs = torch.topk(sims, k=topk+1)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        if i == widx:  # ignorer soi-même
            continue
        out.append((itos[i], float(v)))
        if len(out) == topk:
            break
    return out

# fonction auxiliaire: similarité cosinus
def most_similar_to_vector(vector, topk=5):
    sims = F.cosine_similarity(emb, vector.unsqueeze(0))  # (V,)
    vals, idxs = torch.topk(sims, k=topk+1)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        out.append((itos[i], float(v)))
        if len(out) == topk:
            break
    return out

computed = emb[stoi["king"]] - emb[stoi["man"]] + emb[stoi["woman"]]
print(most_similar_to_vector(computed, topk=1))

print("\nVoisins les plus proches:")
for q in ["king", "queen", "man", "woman"]:
    print(q, "→", most_similar(q, topk=5))
