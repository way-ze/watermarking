from collections import defaultdict, Counter
from nltk.util import ngrams 
import torch

def pLM(word1, word2, tokenizer, output):
    # word 1 and word 2 can also be whole sentences, expandable.
    
    word1tok = tokenizer.convert_tokens_to_ids(word1)
    # word2tok = tokenizer.convert_tokens_to_ids(word2)
    
    # be careful of the order between word1tok and word2tok!!!!
    #with torch.inference_mode():
    #    output = model(torch.tensor([[word2tok]]))
    
    next_token_logits = torch.softmax(output.logits[0, -1, :], -1)
    prob = next_token_logits[word1tok]

    return prob

def getWordDict(corpus, tokenizer):
    word_freqs = defaultdict(int)
    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1
    return word_freqs

def getNgramDict(corpus, tokenizer, n_gram=2):
    bigram_freqs = Counter()
    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        bigram_freqs += Counter(ngrams(new_words, n_gram))
    return defaultdict(int, dict(bigram_freqs))

## Deprecated method, to just find pLM directly from tokenizer
def pHat(word1, word2, word_dict, n_gram_dict):
    # word_dict = get_word_dict(corpus)
    if (not word1 in word_dict) or (not word2 in word_dict):
        prob = 0
        return prob
    assert word1 in word_dict, "word1 is not in corpus"
    assert word2 in word_dict, "word2 is not in corpus"
    # n_gram_dict = get_n_gram_dict(corpus)
    prob = n_gram_dict[(word1, word2)] / word_dict[word2]
    assert prob >= 0 and prob <= 1, "probability is not between 0 and 1"
    return prob
