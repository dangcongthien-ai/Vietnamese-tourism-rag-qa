# Vietnamese Tourism RAG QA

Project hỏi đáp tiếng Việt về du lịch Việt Nam theo hướng RAG:

- `SBERT/PhoBERT` để truy xuất context
- `FAISS` để tìm top-k đoạn liên quan
- `ViT5` để sinh câu trả lời
- `Gradio` để chạy demo

## Dataset

Nguồn dữ liệu:

- https://www.kaggle.com/datasets/vuonglsts/vietnam-tourism-v2

Các file dữ liệu script đang dùng:

- `datasets/train_vietnam_tourism.json`
- `datasets/valid_vietnam_tourism.json`

## Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch transformers sentencepiece gradio faiss-cpu wandb bert-score tqdm numpy
```

## Cách chạy

Huấn luyện retriever:

```powershell
python train_sbert.py
```

Huấn luyện generator:

```powershell
python train_vit5.py
```

Chạy demo:

```powershell
python demo_gradio.py
```

## Output

Sau khi train:

- `sbert_model/`
- `contexts.index`
- `contexts.json`
- `vit5_qa_model/`

## File chính

- [train_sbert.py](/d:/1_Documents/Nam_3_HK1/Natural_Language_Processing/Final_Project/train_sbert.py)
- [train_vit5.py](/d:/1_Documents/Nam_3_HK1/Natural_Language_Processing/Final_Project/train_vit5.py)
- [demo_gradio.py](/d:/1_Documents/Nam_3_HK1/Natural_Language_Processing/Final_Project/demo_gradio.py)

## Hạn chế

Project hiện tại chưa tốt do dataset còn thiếu, chưa đủ đa dạng và chưa đủ lớn.

- Chất lượng retrieval và generation chưa ổn định.
- Kết quả hiện phù hợp mức demo/học thuật hơn là production.
