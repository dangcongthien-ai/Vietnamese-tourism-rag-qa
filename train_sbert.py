import os, json, math, warnings
import numpy as np
import torch
from tqdm import tqdm
import wandb
import faiss
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*symlinks.*")
warnings.filterwarnings("ignore", message=".*flash attention.*")

PROJECT = "Vietnamese_QA"
RUN_NAME = "SBERT-training"

MODEL_NAME = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
TOKENIZER_BACKBONE = "vinai/phobert-base"

TRAIN_JSON = "./datasets/train_vietnam_tourism.json"

MAX_LEN = 192
BATCH_SIZE = 16
ACC_STEPS = 4
EPOCHS = 3
LR = 5e-6
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
CLIP_NORM = 1.0
TAU = 0.05

CTX_EMB_BATCH = 64
SEED = 42

SAVE_SBERT_DIR = "sbert_model"
FAISS_INDEX_PATH = "contexts.index"
CONTEXTS_JSON_PATH = "contexts.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_squad_indexed(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    q2i, c2i = {}, {}
    questions, contexts = [], []
    pq, pc = [], []
    for art in data["data"]:
        for para in art["paragraphs"]:
            ctx = para["context"]
            if ctx not in c2i:
                c2i[ctx] = len(contexts)
                contexts.append(ctx)
            cidx = c2i[ctx]
            for qa in para["qas"]:
                q = qa["question"]
                if q not in q2i:
                    q2i[q] = len(questions)
                    questions.append(q)
                qidx = q2i[q]
                pq.append(qidx)
                pc.append(cidx)
    return questions, contexts, torch.tensor(pq, dtype=torch.long), torch.tensor(pc, dtype=torch.long)

def tokenize_all(tokenizer, texts, max_len=MAX_LEN):
    enc = tokenizer(
        texts, padding="max_length", truncation=True, max_length=max_len,
        return_tensors="pt", return_token_type_ids=False
    )
    return enc["input_ids"].contiguous(), enc["attention_mask"].contiguous()

def gather_rows(t2d, idx_cpu):
    return t2d.index_select(0, idx_cpu)

@torch.no_grad()
def encode_ctx_batch(model, tok_ids, attn, start, end):
    ids = tok_ids[start:end].to(device, non_blocking=True)
    msk = attn[start:end].to(device, non_blocking=True)
    last = model(input_ids=ids, attention_mask=msk).last_hidden_state
    mask = msk.unsqueeze(-1).float()
    emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb

def mean_pool_embeddings(model, input_ids, attention_mask):
    last = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb

def main():
    set_seed(SEED)
    print(f"Device: {device} | CUDA: {torch.cuda.is_available()} | "
          f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

    wandb.init(project=PROJECT, name=RUN_NAME, config={
        "model": MODEL_NAME, "batch_size": BATCH_SIZE, "acc_steps": ACC_STEPS,
        "epochs": EPOCHS, "lr": LR, "max_len": MAX_LEN, "tau": TAU
    })

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_BACKBONE, use_fast=False)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    questions, contexts, pair_q_idx, pair_ctx_idx = load_squad_indexed(TRAIN_JSON)
    print(f"Unique questions: {len(questions)} | Unique contexts: {len(contexts)} | Pairs: {len(pair_q_idx)}", flush=True)

    print("Pre-tokenizing questions & contexts ...", flush=True)
    q_ids, q_attn = tokenize_all(tokenizer, questions, MAX_LEN)
    c_ids, c_attn = tokenize_all(tokenizer, contexts, MAX_LEN)

    num_batches = math.ceil(len(pair_q_idx) / BATCH_SIZE)
    total_steps = EPOCHS * math.ceil(num_batches / ACC_STEPS)
    warmup_steps = int(WARMUP_RATIO * total_steps)
    print(f"num_batches/epoch: {num_batches} | batch_size: {BATCH_SIZE} | "
          f"accumulation: {ACC_STEPS} | optimizer_steps: {total_steps}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Train
    model.train()
    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(len(pair_q_idx))
        pair_q_idx = pair_q_idx[perm]
        pair_ctx_idx = pair_ctx_idx[perm]

        total_loss = 0.0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}", ncols=100)
        for b in pbar:
            s, e = b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, len(pair_q_idx))
            bq = pair_q_idx[s:e]
            bc = pair_ctx_idx[s:e]

            qi = gather_rows(q_ids,  bq).to(device, non_blocking=True)
            qm = gather_rows(q_attn, bq).to(device, non_blocking=True)
            ci = gather_rows(c_ids,  bc).to(device, non_blocking=True)
            cm = gather_rows(c_attn, bc).to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                q_emb = mean_pool_embeddings(model, qi, qm)
                c_emb = mean_pool_embeddings(model, ci, cm)
                logits = torch.matmul(q_emb, c_emb.T) / TAU
                labels = torch.arange(logits.size(0), device=device)
                loss = 0.5 * (
                    torch.nn.functional.cross_entropy(logits, labels) +
                    torch.nn.functional.cross_entropy(logits.T, labels)
                )

            loss = loss / ACC_STEPS
            scaler.scale(loss).backward()

            if ((b + 1) % ACC_STEPS == 0) or ((b + 1) == num_batches):
                if CLIP_NORM is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            total_loss += loss.item() * ACC_STEPS
            avg = total_loss / (b + 1)
            pbar.set_postfix(loss=f"{avg:.4f}")
            if (b + 1) % 50 == 0 or (b + 1) == num_batches:
                wandb.log({"train_loss": avg, "epoch": epoch, "global_step": global_step})

        print(f"Epoch {epoch} - Loss: {avg:.4f}", flush=True)
        wandb.log({"epoch": epoch, "avg_train_loss": avg})

    # Save SBERT
    os.makedirs(SAVE_SBERT_DIR, exist_ok=True)
    model.save_pretrained(SAVE_SBERT_DIR)
    tokenizer.save_pretrained(SAVE_SBERT_DIR)
    print(f"Saved SBERT to ./{SAVE_SBERT_DIR}", flush=True)

    # Build FAISS
    model.eval()
    torch.set_grad_enabled(False)
    ctx_embs = []
    print("Encoding contexts for FAISS index...", flush=True)
    for i in tqdm(range(0, len(contexts), CTX_EMB_BATCH), ncols=100):
        emb = encode_ctx_batch(model, c_ids, c_attn, i, i + CTX_EMB_BATCH)
        ctx_embs.append(emb.cpu().numpy())
    ctx_embs = np.vstack(ctx_embs).astype(np.float32)
    dim = ctx_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(ctx_embs)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CONTEXTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(contexts, f, ensure_ascii=False)
    print(f"Saved FAISS index to ./{FAISS_INDEX_PATH} and contexts to ./{CONTEXTS_JSON_PATH}", flush=True)

    wandb.finish()

if __name__ == "__main__":
    main()