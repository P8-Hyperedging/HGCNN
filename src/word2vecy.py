import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from data.data import (
    load_postgres_business_list_data,
    load_postgres_review_data
)
from data.n_preprocessing import build_hypergraph_incidence_matrix


# =====================================================
# CUDA / DEVICE
# =====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

print("Using device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =====================================================
# CACHE
# =====================================================

_WORD2VEC_CACHE = {
    "embeddings": None,
    "word_to_ix": None
}


def get_word2vec():
    global _WORD2VEC_CACHE

    if _WORD2VEC_CACHE["embeddings"] is None:
        embeddings, word_to_ix = load_word2vec()

        _WORD2VEC_CACHE["embeddings"] = embeddings.to(DEVICE)
        _WORD2VEC_CACHE["word_to_ix"] = word_to_ix

    return _WORD2VEC_CACHE["embeddings"], _WORD2VEC_CACHE["word_to_ix"]


# =====================================================
# TEXT PROCESSING
# =====================================================

def generate_tokens(corpus):
    return [sentence.lower().split() for sentence in corpus]


def build_vocabulary(tokens):
    words = [word for sentence in tokens for word in sentence]

    vocab = sorted(list(set(words)))

    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for word, i in word_to_ix.items()}

    return vocab, word_to_ix, ix_to_word


def generate_training_data(tokens, window_size=2):
    pairs = []

    for sentence in tokens:
        for i, target in enumerate(sentence):
            for j in range(-window_size, window_size + 1):
                if j == 0:
                    continue

                ctx_index = i + j

                if 0 <= ctx_index < len(sentence):
                    context = sentence[ctx_index]
                    pairs.append((target, context))

    return pairs


# =====================================================
# DATASET
# =====================================================

class Word2VecDataset(Dataset):
    def __init__(self, pairs, word_to_ix):
        self.targets = [word_to_ix[t] for t, c in pairs]
        self.contexts = [word_to_ix[c] for t, c in pairs]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.targets[idx], self.contexts[idx]


# =====================================================
# MODEL
# =====================================================

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.linear(x)
        return x


# =====================================================
# TRAIN
# =====================================================

def train_word2vec(
    corpus,
    embedding_dim=128,
    epochs=30,
    batch_size=4096,
    lr=0.001,
    window_size=2
):
    print("Tokenizing...")
    tokens = generate_tokens(corpus)

    print("Building vocabulary...")
    vocab, word_to_ix, ix_to_word = build_vocabulary(tokens)

    print("Generating training pairs...")
    pairs = generate_training_data(tokens, window_size)

    print("Vocabulary size:", len(vocab))
    print("Training pairs:", len(pairs))

    dataset = Word2VecDataset(pairs, word_to_ix)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model = Word2Vec(len(vocab), embedding_dim).to(DEVICE)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for targets, contexts in loader:
            targets = targets.to(DEVICE, non_blocking=True)
            contexts = contexts.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                output = model(targets)
                loss = loss_function(output, contexts)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if epoch%50==0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    embeddings = model.embeddings.weight.detach().cpu()

    return embeddings, word_to_ix


# =====================================================
# SAVE / LOAD
# =====================================================

def save_word2vec(embeddings, word_to_ix, path="word2vec.pt"):
    torch.save({
        "embeddings": embeddings,
        "word_to_ix": word_to_ix
    }, path)


def load_word2vec(path="word2vec.pt"):
    data = torch.load(path, map_location="cpu")
    return data["embeddings"], data["word_to_ix"]


# =====================================================
# INFERENCE
# =====================================================

def business_to_vec(name):
    embeddings, word_to_ix = get_word2vec()

    tokens = name.lower().split()

    vecs = [
        embeddings[word_to_ix[w]]
        for w in tokens
        if w in word_to_ix
    ]

    if len(vecs) == 0:
        return torch.zeros(
            embeddings.shape[1],
            device=DEVICE
        )

    return torch.mean(torch.stack(vecs), dim=0)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("Loading review graph...")
    H, business_ids, business_to_idx = build_hypergraph_incidence_matrix(
        load_postgres_review_data()
    )

    print("H shape:", H.shape)

    print("Loading businesses...")
    businesses = load_postgres_business_list_data(business_ids)

    corpus = [b.name for b in businesses if b.name]

    embeddings, word_to_ix = train_word2vec(
        corpus=corpus,
        embedding_dim=512,
        epochs=500,
        batch_size=10000,
        lr=0.001
    )

    save_word2vec(embeddings, word_to_ix)

    print("Model trained on CUDA and saved ✔️")