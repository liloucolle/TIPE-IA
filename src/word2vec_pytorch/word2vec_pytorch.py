import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from text_utils.text_utils import tokenisation, frequence_mots, vocabulaire, creation_paires
from content.echantillon_texte import texte_100 as texte
from embedding_utils.embedding_utils import mots_similaires_torch, most_similar_to_vector_torch



torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
# Tokeniser chaque ligne en liste de mots
tokens = tokenisation(texte)
"""
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
"""

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
def train_softmax(model, ds, epochs=5, batch_size=128, lr=0.01, device="cpu"):
    print(f"Training with softmax with {device}")

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    model.to(device)
    #opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()  # inclut softmax

    # Récupérer les paires pour l'évaluation
    pairs = ds.pairs

    for ep in range(1, epochs+1):
        model.train()  # Mode entraînement
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
        
        # Afficher la perte d'entraînement et la perte d'évaluation        
        # La version pytorch est la moyenne des pertes au fil des pas, où les poids changent après chaque échantillon
        # La perte de reporting perte_tot est une évaluation a posteriori sur tout le dataset avec des poids fixes
        # Tous les 50 echantillons on calcule la perte de la meme facon que perte_tot_torch        
        train_loss = total/n
        if ep % 50 == 0 or ep == 1:  # Évaluation tous les 50 époques
            eval_loss = perte_tot_torch(model, pairs, device=device)
            #print(f"[softmax] époque {ep}: perte train = {train_loss:.4f}, perte eval = {eval_loss:.4f}")
        #else:
           # print(f"[softmax] époque {ep}: perte train = {train_loss:.4f}")

def train_neg(model, ds, epochs=5, batch_size=1024, lr=0.01, device="cpu"):
    print(f"Training with negative sampling with {device}")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

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
        #print(f"[neg] époque {ep}: perte = {total/n:.4f}")


@torch.no_grad()
def perte_tot_torch(model, pairs, device="cpu"):
    """
    Calcule la perte moyenne -log P(w_O|w_I) sur tout le dataset, sans autograd.
    Équivalent PyTorch de la fonction perte_tot NumPy.
    """
    model.eval()
    total = 0.0
    for c, pos in pairs:
        c = torch.tensor([c], device=device, dtype=torch.long)
        logits = model.forward_softmax(c)  # (1, V)
        logp = torch.log_softmax(logits, dim=1)
        total += -logp[0, pos].item()
    return total / max(1, len(pairs))


def deep_learning_pytorch(texte, dimension=100, taux_apprentissage=0.01, window_size=2, n_epochs=10, mode="softmax", batch_size=128):
        
    tokens = tokenisation(texte)
    
    # Edge case: empty corpus
    if len(tokens) == 0 or all(len(t) == 0 for t in tokens):
        return np.empty((0, dimension)), 0.0
    
    freq = frequence_mots(tokens)
    vocab, stoi, itos = vocabulaire(freq)
    V = len(vocab)
    
    pairs = creation_paires(tokens, stoi, taille_fenetre=window_size)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SkipGram(V, dimension)
    

    if mode == "softmax":
        ds = SkipGramSoftmaxDS(pairs)
        train_softmax(model, ds, epochs=n_epochs, batch_size=batch_size, lr=taux_apprentissage, device=device)
    else:  # neg
        unigram = torch.tensor([freq[itos[i]] for i in range(V)], dtype=torch.float)
        unigram = unigram.pow(0.75)
        unigram /= unigram.sum()
        ds = SkipGramNegSampleDS(pairs, V, unigram_probs=unigram, K=5)
        train_neg(model, ds, epochs=n_epochs, batch_size=batch_size, lr=taux_apprentissage, device=device)
    
    emb = model.get_embeddings().numpy()
    return emb


# ---------------------------
# 7) Choisir le mode et entraîner
# ---------------------------



"""
# ---------------------------   
# 8) Évaluation finale de la perte
# ---------------------------
print("\n" + "="*50)
print("ÉVALUATION FINALE DE LA PERTE")
print("="*50)

# Calculer la perte finale sur tout le dataset
perte_finale = perte_tot_torch(model, pairs, device=device)
print(f"Perte finale PyTorch: {perte_finale:.4f}")

# ---------------------------   
# 9) Utiliser les embeddings
# ---------------------------
emb = model.get_embeddings()   # (V, emb_dim)

# Test de la fonction mots_similaires
print("\n" + "="*50)
print("TEST DE SIMILARITES")
print("="*50)

for mot in ["king", "queen", "man", "woman"]:
    if mot in stoi:
        print(f"\nMots les plus similaires à '{mot}':")
        similaires = mots_similaires_torch(mot, emb, stoi, top_k=10)
        for i, (mot_sim, sim) in enumerate(similaires, 1):
            print(f"  {i}. {mot_sim}: {sim:.4f}")
    else:
        print(f"\nLe mot '{mot}' n'est pas dans le vocabulaire.")

# Test d'analogie utilisant les fonctions consolidées
computed = emb[stoi["king"]] - emb[stoi["man"]] + emb[stoi["woman"]]
print("Test d'analogie avec most_similar_to_vector:")
print(most_similar_to_vector_torch(computed, emb, stoi, topk=1))

# Test d'analogie avancé
print("\nTest d'analogie avancé:")
if "king" in stoi and "man" in stoi and "woman" in stoi:
    vector_analogie = emb[stoi["king"]] - emb[stoi["man"]] + emb[stoi["woman"]]
    exclude_words = {"king", "man", "woman"}
    
    # Obtenir plus de résultats pour compenser les exclusions
    all_results = most_similar_to_vector_torch(vector_analogie, emb, stoi, topk=20)
    
    # Filtrer les mots à exclure
    result = [(mot, sim) for mot, sim in all_results if mot not in exclude_words][:10]
    
    print(f"Analogie: king - man + woman ≈ ?")
    print(f"Top 10 résultats:")
    for i, (mot, sim) in enumerate(result, 1):
        marker = " ✓" if mot == "queen" else ""
        print(f"{i:2d}. {mot:<15} (similarité: {sim:.4f}){marker}")
    
    # Chercher le rang de "queen" dans les résultats
    queen_rank = None
    for i, (mot, sim) in enumerate(result, 1):
        if mot == "queen":
            queen_rank = i
            break
    
    if queen_rank:
        print(f"\n🎯 'queen' trouvée au rang {queen_rank} !")
    else:
        print(f"\n❌ 'queen' non trouvée dans le top 10")
else:
    print("❌ Certains mots de l'analogie ne sont pas dans le vocabulaire")


"""