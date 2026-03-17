import torch, torch.nn as nn, pickle, random, time, os, gc
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import Encoder, Decoder, Seq2Seq
from preprocess import encode

EMBED_DIM  = 128
HIDDEN_DIM = 256
N_LAYERS   = 1
DROPOUT    = 0.3
BATCH_SIZE = 64
N_EPOCHS   = 10
CLIP       = 1.0
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TranslationDataset(Dataset):
    def __init__(self, pairs, en_vocab, hi_vocab):
        self.data = [(encode(en,en_vocab), encode(hi,hi_vocab)) for en,hi in pairs]
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def collate_fn(batch):
    src,tgt = zip(*batch)
    src = pad_sequence([torch.tensor(s) for s in src], batch_first=True, padding_value=0)
    tgt = pad_sequence([torch.tensor(t) for t in tgt], batch_first=True, padding_value=0)
    return src, tgt

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0
    for src,tgt in loader:
        src,tgt = src.to(DEVICE), tgt.to(DEVICE)
        optimizer.zero_grad()
        out = model(src,tgt)
        loss = criterion(out[:,1:].reshape(-1,out.size(-1)), tgt[:,1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        total += loss.item()
    return total/len(loader)

def eval_epoch(model, loader, criterion):
    model.eval()
    total = 0
    with torch.no_grad():
        for src,tgt in loader:
            src,tgt = src.to(DEVICE), tgt.to(DEVICE)
            out = model(src,tgt,0)
            total += criterion(out[:,1:].reshape(-1,out.size(-1)), tgt[:,1:].reshape(-1)).item()
    return total/len(loader)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Using device: {DEVICE}")
    with open("../data/pairs.pkl","rb")    as f: pairs    = pickle.load(f)
    with open("../data/en_vocab.pkl","rb") as f: en_vocab = pickle.load(f)
    with open("../data/hi_vocab.pkl","rb") as f: hi_vocab = pickle.load(f)
    random.shuffle(pairs)
    pairs = pairs[:30000]
    split = int(0.9*len(pairs))
    train_dl = DataLoader(TranslationDataset(pairs[:split],en_vocab,hi_vocab),
                          BATCH_SIZE,shuffle=True,collate_fn=collate_fn)
    val_dl   = DataLoader(TranslationDataset(pairs[split:],en_vocab,hi_vocab),
                          BATCH_SIZE,shuffle=False,collate_fn=collate_fn)
    print(f"Train: {split} | Val: {len(pairs)-split}")
    enc   = Encoder(len(en_vocab),EMBED_DIM,HIDDEN_DIM,N_LAYERS,DROPOUT).to(DEVICE)
    dec   = Decoder(len(hi_vocab),EMBED_DIM,HIDDEN_DIM,N_LAYERS,DROPOUT).to(DEVICE)
    model = Seq2Seq(enc,dec,DEVICE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=2,factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    best_val  = float("inf")
    os.makedirs("../checkpoints", exist_ok=True)
    for epoch in range(1, N_EPOCHS+1):
        t0  = time.time()
        tr  = train_epoch(model,train_dl,optimizer,criterion)
        val = eval_epoch(model,val_dl,criterion)
        scheduler.step(val)
        print(f"Epoch {epoch:02d}/{N_EPOCHS} | Train: {tr:.4f} | Val: {val:.4f} | Time: {time.time()-t0:.1f}s")
        if val < best_val:
            best_val = val
            torch.save({"model_state":model.state_dict(),"en_vocab":en_vocab,
                        "hi_vocab":hi_vocab,"embed_dim":EMBED_DIM,
                        "hidden_dim":HIDDEN_DIM,"n_layers":N_LAYERS,
                        "dropout":DROPOUT}, "../checkpoints/best_model.pt")
            print(f"  ✓ Checkpoint saved (val: {best_val:.4f})")
    print("Training complete!")