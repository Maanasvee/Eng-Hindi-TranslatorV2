import torch, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import clean_english, encode, UNK
from model import Encoder, Decoder, Seq2Seq

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 60

def load_model(checkpoint_path):
    ckpt     = torch.load(checkpoint_path, map_location=DEVICE)
    en_vocab = ckpt["en_vocab"]
    hi_vocab = ckpt["hi_vocab"]
    hi_inv   = {v:k for k,v in hi_vocab.items()}
    enc   = Encoder(len(en_vocab),ckpt["embed_dim"],ckpt["hidden_dim"],ckpt["n_layers"],ckpt["dropout"])
    dec   = Decoder(len(hi_vocab),ckpt["embed_dim"],ckpt["hidden_dim"],ckpt["n_layers"],ckpt["dropout"])
    model = Seq2Seq(enc,dec,DEVICE).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, en_vocab, hi_vocab, hi_inv

def translate_en_to_hi(text, model, en_vocab, hi_vocab, hi_inv):
    tokens = encode(clean_english(text), en_vocab)
    src    = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        enc_out, hidden = model.encoder(src)
        dec_input = torch.tensor([hi_vocab["<sos>"]]).to(DEVICE)
        result = []
        for _ in range(MAX_LEN):
            pred, hidden, _ = model.decoder(dec_input, hidden, enc_out)
            top = pred.argmax(1).item()
            if top == hi_vocab["<eos>"]: break
            if top not in (hi_vocab["<pad>"], hi_vocab["<sos>"]):
                result.append(hi_inv.get(top,""))
            dec_input = torch.tensor([top]).to(DEVICE)
    return " ".join(result)

if __name__ == "__main__":
    ckpt = os.path.join(os.path.dirname(__file__), "../checkpoints/best_model.pt")
    model, en_vocab, hi_vocab, hi_inv = load_model(ckpt)
    for t in ["Hello how are you", "I love learning languages", "What is your name"]:
        print(f"EN: {t}")
        print(f"HI: {translate_en_to_hi(t,model,en_vocab,hi_vocab,hi_inv)}\n")