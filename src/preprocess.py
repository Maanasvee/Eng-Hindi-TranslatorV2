import re, pickle, os, glob, csv
from collections import Counter

MAX_LEN  = 50
MIN_FREQ = 2
PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"

def clean_english(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s'.,!?]", "", text)
    return re.sub(r"\s+", " ", text)

def clean_hindi(text):
    text = str(text).strip()
    text = re.sub(r"[^\u0900-\u097F0-9\s।.,!?]", "", text)
    return re.sub(r"\s+", " ", text)

def build_vocab(sentences, min_freq=MIN_FREQ):
    counter = Counter()
    for s in sentences: counter.update(s.split())
    vocab = {PAD:0, SOS:1, EOS:2, UNK:3}
    for w,f in counter.items():
        if f >= min_freq: vocab[w] = len(vocab)
    return vocab

def load_data(filepath):
    pairs = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            en = clean_english(row.get("english_sentence") or row.get("English") or "")
            hi = clean_hindi(row.get("hindi_sentence") or row.get("Hindi") or "")
            if en and hi and len(en.split())<=MAX_LEN and len(hi.split())<=MAX_LEN:
                pairs.append((en, hi))
    return pairs

def encode(sentence, vocab):
    return [vocab[SOS]] + [vocab.get(t, vocab[UNK]) for t in sentence.split()] + [vocab[EOS]]

if __name__ == "__main__":
    files = glob.glob("../data/*.csv")
    assert files, "No CSV in data/ — run download_data.py first!"
    all_pairs = []
    for f in files:
        p = load_data(f)
        print(f"Loaded {len(p)} pairs from {f}")
        all_pairs.extend(p)
    print(f"Total pairs: {len(all_pairs)}")
    en_vocab = build_vocab([p[0] for p in all_pairs])
    hi_vocab = build_vocab([p[1] for p in all_pairs])
    print(f"EN vocab: {len(en_vocab)} | HI vocab: {len(hi_vocab)}")
    with open("../data/pairs.pkl",    "wb") as f: pickle.dump(all_pairs, f)
    with open("../data/en_vocab.pkl", "wb") as f: pickle.dump(en_vocab,  f)
    with open("../data/hi_vocab.pkl", "wb") as f: pickle.dump(hi_vocab,  f)
    print("Saved vocabs and pairs!")