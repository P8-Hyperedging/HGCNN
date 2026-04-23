import torch

_WORD2VEC_CACHE = {"embeddings": None, "word_to_ix": None}

def get_word2vec(load_word2vec):
    if _WORD2VEC_CACHE["embeddings"] is None:
        emb, w2i = load_word2vec()
        _WORD2VEC_CACHE["embeddings"] = emb
        _WORD2VEC_CACHE["word_to_ix"] = w2i
    return _WORD2VEC_CACHE["embeddings"], _WORD2VEC_CACHE["word_to_ix"]


def business_to_vec(name, load_word2vec):
    embeddings, word_to_ix = get_word2vec(load_word2vec)

    tokens = name.lower().split()
    vecs = [embeddings[word_to_ix[w]] for w in tokens if w in word_to_ix]

    if len(vecs) == 0:
        return torch.zeros(embeddings.shape[1])

    return torch.mean(torch.stack(vecs), dim=0)