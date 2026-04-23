import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def generate_tokens(corpus):
    return [sentence.lower().split() for sentence in corpus]


def build_vocabulary(tokens):
    words = [word for sentence in tokens for word in sentence]
    vocab = list(set(words))
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for word, i in word_to_ix.items()}
    return vocab, word_to_ix, ix_to_word


def generate_training_data(tokens, window_size=2):
    data = []
    for sentence in tokens:
        for i, word in enumerate(sentence):
            for j in range(-window_size, window_size + 1):
                if j != 0 and 0 <= i + j < len(sentence):
                    data.append((word, sentence[i + j]))
    return data


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.linear(x)
        return x


def train_word2vec(corpus, embedding_dim=10, epochs=2000):
    tokens = generate_tokens(corpus)
    vocab, word_to_ix, ix_to_word = build_vocabulary(tokens)
    data = generate_training_data(tokens)

    model = Word2Vec(len(vocab), embedding_dim).to(DEVICE)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        total_loss = 0

        for target, context in data:
            target_ix = torch.tensor([word_to_ix[target]], device=DEVICE)
            context_ix = torch.tensor([word_to_ix[context]], device=DEVICE)

            optimizer.zero_grad()
            output = model(target_ix)
            loss = loss_function(output, context_ix)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    embeddings = model.embeddings.weight.detach().cpu()
    return embeddings, word_to_ix


def save_word2vec(embeddings, word_to_ix, path="word2vec.pt"):
    torch.save({"embeddings": embeddings, "word_to_ix": word_to_ix}, path)


def load_word2vec(path="word2vec.pt"):
    data = torch.load(path, map_location="cpu")
    return data["embeddings"], data["word_to_ix"]


def business_to_vec(name):
    embeddings, word_to_ix = get_word2vec()

    tokens = name.lower().split()

    vecs = [
        embeddings[word_to_ix[w]]
        for w in tokens if w in word_to_ix
    ]

    if len(vecs) == 0:
        return torch.zeros(embeddings.shape[1], device=DEVICE)

    return torch.mean(torch.stack(vecs), dim=0)