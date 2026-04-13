import os, json, warnings
import numpy as np
import torch
import faiss
import gradio as gr
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from transformers.utils import logging as hf_logging

# Quiet setup
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*symlinks.*")
warnings.filterwarnings("ignore", message=".*flash attention.*")

# Paths
SBERT_DIR = "sbert_model"
FAISS_INDEX_PATH = "contexts.index"
CONTEXTS_JSON_PATH = "contexts.json"
VIT5_DIR = "vit5_qa_model"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Load retriever (SBERT)
sbert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
sbert_model = AutoModel.from_pretrained(SBERT_DIR).to(device).eval()

# Load FAISS index & contexts
index = faiss.read_index(FAISS_INDEX_PATH)
with open(CONTEXTS_JSON_PATH, "r", encoding="utf-8") as f:
    KB_CONTEXTS = json.load(f)

# Load generator (ViT5)
t5_tokenizer = AutoTokenizer.from_pretrained(VIT5_DIR, use_fast=True)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(VIT5_DIR).to(device).eval()
if t5_tokenizer.pad_token is None:
    t5_tokenizer.pad_token = t5_tokenizer.eos_token
t5_model.config.pad_token_id = t5_tokenizer.pad_token_id
if t5_model.config.decoder_start_token_id is None:
    t5_model.config.decoder_start_token_id = t5_tokenizer.pad_token_id

# Hyperparameters
SBERT_MAX_LEN = 192
VIT5_ENC_MAX_LEN = 512
VIT5_DEC_MAX_LEN = 128

# SBERT encode
def _sbert_embed(texts):
    enc = sbert_tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=SBERT_MAX_LEN,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = sbert_model(**enc).last_hidden_state[:, 0, :]
        vec = torch.nn.functional.normalize(out, p=2, dim=1)
    return vec.cpu().numpy().astype(np.float32)

# Retrieve top-k
def retrieve_contexts(question, top_k=5):
    qvec = _sbert_embed([question])
    D, I = index.search(qvec, top_k)
    ctxs = [KB_CONTEXTS[i] for i in I[0]]
    scores = D[0].tolist()
    return ctxs, scores

# Generate answer
def generate_answer(question, k, max_new_tokens, temperature, num_beams):
    ctxs, scores = retrieve_contexts(question, top_k=int(k))
    marked_ctxs = [f"[CTX{i+1}] {c}" for i, c in enumerate(ctxs)]
    combined = " ".join(marked_ctxs)
    prompt = f"question: {question.strip()} context: {combined.strip()}"

    early_stop = True if num_beams > 1 else False
    do_sample = True if temperature > 0.0 else False
    temp = float(temperature) if do_sample else 1.0

    inputs = t5_tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=VIT5_ENC_MAX_LEN
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = t5_model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=temp,
            num_beams=int(num_beams),
            top_p=0.95 if do_sample else None,
            early_stopping=early_stop,
        )
    answer = t5_tokenizer.decode(out_ids[0], skip_special_tokens=True)

    ctx_preview_lines = []
    for i, (c, s) in enumerate(zip(ctxs, scores), 1):
        short = " ".join(c.strip().split())[:400]
        ctx_preview_lines.append(f"Top {i} | score={s:.3f}: {short}...")

    return answer, "\n\n".join(ctx_preview_lines)

# Gradio UI
with gr.Blocks(css=".gradio-container {max-width: 1200px; margin: auto;}") as demo:
    gr.Markdown("## Vietnam Tourism QA — Retrieval (SBERT + FAISS) → Generation (ViT5)")
    gr.Markdown("Nhập **câu hỏi du lịch tiếng Việt**, hệ thống sẽ truy xuất **top-k đoạn phù hợp** và sinh **câu trả lời tự nhiên** bằng ViT5.")

    with gr.Row():
        with gr.Column(scale=1):
            q_in = gr.Textbox(label="Câu hỏi", lines=2, placeholder="Ví dụ: Đặc sản nên thử ở Hội An là gì?")
            with gr.Row():
                k_in = gr.Slider(1, 10, value=5, step=1, label="Top-k retrieval (k)")
                beams_in = gr.Slider(1, 8, value=4, step=1, label="num_beams (độ chính xác cao hơn khi >1)")
            with gr.Row():
                temp_in = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="temperature (0 = trả lời cố định, >0 = sáng tạo)")
                max_new_in = gr.Slider(16, 256, value=128, step=8, label="max_new_tokens (ViT5)")
            run_btn = gr.Button("Truy xuất & Trả lời", variant="primary")

        with gr.Column(scale=1):
            ans_out = gr.Textbox(label="Câu trả lời sinh ra", lines=3)
            ctxs_out = gr.Textbox(label="Top-k đoạn văn (rút gọn)", lines=14)

    run_btn.click(
        fn=generate_answer,
        inputs=[q_in, k_in, max_new_in, temp_in, beams_in],
        outputs=[ans_out, ctxs_out],
    )

if __name__ == "__main__":
    demo.launch()
