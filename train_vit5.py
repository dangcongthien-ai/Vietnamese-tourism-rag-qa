import os, json, warnings, math
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
import faiss
from bert_score import score as bertscore

from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    AutoModel, AutoTokenizer as SBTokenizer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup
)
from transformers.utils import logging as hf_logging

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*symlinks.*")
warnings.filterwarnings("ignore", message=".*flash attention.*")

PROJECT = "Vietnamese_QA"
RUN_NAME = "ViT5-RAG-training"

SBERT_DIR = "sbert_model"
FAISS_INDEX_PATH = "contexts.index"
CONTEXTS_JSON_PATH = "contexts.json"

TRAIN_JSON = "./datasets/train_vietnam_tourism.json"
VALID_JSON = "./datasets/valid_vietnam_tourism.json"

VIT5_NAME = "VietAI/vit5-base"

TOP_K = 5
ENC_MAX_LEN = 512
DEC_MAX_LEN = 128

BATCH_SIZE = 8
EPOCHS = 2
LR = 2e-5
WEIGHT_DECAY = 1e-2
WARMUP_RATIO = 0.1
LOG_INTERVAL = 10
SEED = 42

SAVE_DIR = "vit5_qa_model"

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

# DATA UTILS
def load_squad(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    qs, ctxs, ans = [], [], []
    for art in data["data"]:
        for para in art["paragraphs"]:
            c = para["context"]
            for qa in para["qas"]:
                qs.append(qa["question"])
                ctxs.append(c)
                a = qa["answers"][0]["text"] if qa.get("answers") else ""
                ans.append(a)
    return qs, ctxs, ans

@torch.no_grad()
def sbert_embed_mean(texts, sbert_tok, sbert_model, max_len=256):
    enc = sbert_tok(
        texts, padding="longest", truncation=True, max_length=max_len,
        return_tensors="pt", return_token_type_ids=False
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    last = sbert_model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().astype(np.float32)

def build_rag_inputs(questions, gold_contexts, answers,
                     sbert_tok, sbert_model, index, kb_contexts, k=TOP_K):
    inputs, targets = [], []
    BS = 64
    for i in tqdm(range(0, len(questions), BS), desc="Retrieving", ncols=100):
        batch_q = questions[i:i+BS]
        qvecs = sbert_embed_mean(batch_q, sbert_tok, sbert_model, max_len=256)
        D, I = index.search(qvecs, k)
        for j in range(len(batch_q)):
            top_ctxs = [kb_contexts[idx] for idx in I[j]]
            gold = gold_contexts[i + j]
            if gold not in top_ctxs:
                top_ctxs[-1] = gold
            marked = [f"[CTX{t+1}] {c}" for t, c in enumerate(top_ctxs)]
            combined = " ".join(marked)
            inputs.append(f"question: {batch_q[j]} context: {combined}")
            targets.append(answers[i + j])
    return inputs, targets

# DATASET
class QADataset(Dataset):
    def __init__(self, tokenized_inputs, tokenized_labels):
        self.inputs = tokenized_inputs
        self.labels = tokenized_labels
    def __len__(self): return len(self.inputs["input_ids"])
    def __getitem__(self, i):
        item = {k: self.inputs[k][i] for k in self.inputs.keys()}
        item["labels"] = self.labels["input_ids"][i]
        return item

# MAIN
def main():
    set_seed(SEED)
    print(f"Device: {device} | CUDA: {torch.cuda.is_available()} | "
          f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

    wandb.init(project=PROJECT, name=RUN_NAME, config={
        "vit5": VIT5_NAME, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "lr": LR, "enc_max_len": ENC_MAX_LEN, "dec_max_len": DEC_MAX_LEN,
        "top_k": TOP_K
    })

    # Load retriever artifacts
    sbert_tok = SBTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    sbert_model = AutoModel.from_pretrained(SBERT_DIR).to(device).eval()
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CONTEXTS_JSON_PATH, "r", encoding="utf-8") as f:
        kb_contexts = json.load(f)
    print(f"Loaded SBERT+FAISS | KB contexts: {len(kb_contexts)}", flush=True)

    # Load SQuAD
    tr_q, tr_gold_ctx, tr_ans = load_squad(TRAIN_JSON)
    va_q, va_gold_ctx, va_ans = load_squad(VALID_JSON)
    print(f"Train QAs: {len(tr_q)} | Valid QAs: {len(va_q)}", flush=True)

    # Build RAG inputs
    tr_inputs, tr_targets = build_rag_inputs(tr_q, tr_gold_ctx, tr_ans, sbert_tok, sbert_model, index, kb_contexts, k=TOP_K)
    va_inputs, va_targets = build_rag_inputs(va_q, va_gold_ctx, va_ans, sbert_tok, sbert_model, index, kb_contexts, k=TOP_K)

    # Load ViT5
    tok = AutoTokenizer.from_pretrained(VIT5_NAME, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(VIT5_NAME, use_safetensors=True).to(device)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tok.pad_token_id

    # Tokenize
    tr_enc = tok(tr_inputs, truncation=True, max_length=ENC_MAX_LEN)
    tr_lab = tok(text_target=tr_targets, truncation=True, max_length=DEC_MAX_LEN)
    va_enc = tok(va_inputs, truncation=True, max_length=ENC_MAX_LEN)
    va_lab = tok(text_target=va_targets, truncation=True, max_length=DEC_MAX_LEN)

    train_ds = QADataset(tr_enc, tr_lab)
    valid_ds = QADataset(va_enc, va_lab)
    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    # Optim, Scheduler, AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    warmup = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # TRAIN
    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss = 0.0
        pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"Epoch {epoch}", ncols=100)
        for b, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(**batch)
                loss = out.loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            run_loss += loss.item()
            global_step += 1
            avg = run_loss / b
            pbar.set_postfix(loss=f"{avg:.4f}")
            if (b % LOG_INTERVAL == 0) or (b == len(train_loader)):
                wandb.log({"train_loss": avg, "epoch": epoch, "step": global_step})

        # Validation loss
        model.eval()
        val_loss_sum, val_count = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validating", ncols=100):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    out = model(**batch)
                bs = batch["input_ids"].size(0)
                val_loss_sum += out.loss.item() * bs
                val_count += bs
        val_loss = val_loss_sum / max(1, val_count)

        # BERTScore (val)
        preds = []
        refs = va_targets
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Generating (val)", ncols=100):
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device)
                }
                gen_ids = model.generate(**inputs, max_new_tokens=DEC_MAX_LEN)
                preds.extend(tok.batch_decode(gen_ids, skip_special_tokens=True))
        refs = refs[:len(preds)]
        P, R, F1 = bertscore(
            preds, refs, lang="vi",
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            batch_size=32, verbose=False
        )
        val_f1 = float(F1.mean().item())

        print(f"Epoch {epoch}: val_loss={val_loss:.4f} | val_BERTScore(F1)={val_f1:.4f}", flush=True)
        wandb.log({"val_loss": val_loss, "val_BERTScore": val_f1, "epoch": epoch, "step": global_step})

        # Save checkpoint mỗi epoch (safetensors)
        os.makedirs(SAVE_DIR, exist_ok=True)
        model.save_pretrained(SAVE_DIR, safe_serialization=True)
        tok.save_pretrained(SAVE_DIR)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "saved_at": datetime.now().isoformat()
        }, os.path.join(SAVE_DIR, f"training_state_epoch{epoch}.pt"))
        print(f"Saved epoch {epoch} to ./{SAVE_DIR}", flush=True)

    print(f"Training finished. Model saved to ./{SAVE_DIR}", flush=True)
    wandb.finish()

if __name__ == "__main__":
    main()