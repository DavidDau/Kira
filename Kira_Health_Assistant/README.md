# Medical Q&A Chatbot — Fine-Tuned LLM with LoRA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/medical-chatbot/blob/main/medical_chatbot_finetune.ipynb)

A **domain-specific medical chatbot** built by fine-tuning **TinyLlama-1.1B-Chat** on the
`medalpaca/medical_meadow_medical_flashcards` dataset using **QLoRA (Quantized Low-Rank Adaptation)**.
The model is deployed via a **Gradio** web interface, designed to run end-to-end on **Google Colab's free T4 GPU**.

---

## Project Overview

| Property         | Detail                                          |
| ---------------- | ----------------------------------------------- |
| **Domain**       | Healthcare / Medical Q&A                        |
| **Base Model**   | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`            |
| **Dataset**      | `medalpaca/medical_meadow_medical_flashcards`   |
| **Fine-tuning**  | QLoRA (4-bit NF4 quantization + LoRA adapters)  |
| **PEFT Library** | `peft`, `trl` (SFTTrainer)                      |
| **UI**           | Gradio (public share link via Colab)            |
| **Hardware**     | Google Colab Free Tier — NVIDIA T4 (15 GB VRAM) |

---

## Purpose & Domain Alignment

Healthcare is one of the most impactful domains for AI assistants: patients and medical students
constantly seek quick, reliable answers to clinical questions. However, general-purpose LLMs are
not consistently accurate on specialized medical topics.

This chatbot addresses that gap by fine-tuning a lightweight LLM specifically on medical
question-answer pairs sourced from medical school flashcards. The resulting model:

- Answers clinical/biomedical questions with greater accuracy than the base model
- Handles out-of-domain questions gracefully (acknowledges uncertainty)
- Runs on consumer hardware via quantization — making it accessible

---

## Repository Structure

```
medical-chatbot/
├── medical_chatbot_finetune.py   # Main pipeline (cells for Colab)
├── README.md                     # This file
├── experiment_results.csv        # Hyperparameter experiment table
├── evaluation_metrics.csv        # ROUGE/BLEU scores (base vs fine-tuned)
├── qualitative_results.json      # Sample Q&A comparisons
├── dataset_distribution.png      # Dataset length plots
├── experiment_comparison.png     # Experiment bar charts
└── base_vs_finetuned_metrics.png # Final evaluation comparison chart
```

---

## Dataset

**Source:** [`medalpaca/medical_meadow_medical_flashcards`](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)

The dataset contains medical school-style flashcard Q&A pairs covering topics such as pharmacology,
anatomy, pathophysiology, clinical diagnosis, and more.

### Preprocessing Steps

1. **Filtering** — Removed examples with input < 10 chars, input > 800 chars, or output > 600 chars.
   Also removed exact duplicates (input == output).
2. **Subsampling** — Selected up to 4,000 high-quality examples to balance training time with coverage.
3. **Train/Val Split** — 90% training (≈3,600 examples) / 10% validation (≈400 examples).
4. **Instruction Template** — Wrapped every example in TinyLlama's ChatML format:
   ```
   <|system|>
   You are a knowledgeable and concise medical assistant...
   <|user|>
   {question}
   <|assistant|>
   {answer}
   ```
5. **Tokenization** — Used the model's built-in WordPiece-compatible tokenizer; sequences truncated
   to 512 tokens (covers ≥95% of examples without truncation).

---

## Model Fine-Tuning

### Why QLoRA?

Full fine-tuning of a 1.1B parameter model requires ~8–16 GB of GPU memory.
QLoRA reduces this to ~5 GB by:

- Loading weights in **4-bit NF4 quantization** (via `bitsandbytes`)
- Training only **lightweight LoRA adapter matrices** (≈0.5% of total parameters)

### LoRA Configuration

| Parameter        | Value                                                         |
| ---------------- | ------------------------------------------------------------- |
| Rank (r)         | 16                                                            |
| Alpha (α)        | 32                                                            |
| Target modules   | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj |
| Dropout          | 0.05                                                          |
| Trainable params | ~4.2M (≈0.38% of total)                                       |

---

## Hyperparameter Experiments

Three experiments were conducted, varying learning rate, batch size, and epochs.

| Experiment | Label           | Learning Rate | Eff. Batch Size | Epochs | Train Loss | Eval Loss | Perplexity | Time (min) |
| ---------- | --------------- | ------------- | --------------- | ------ | ---------- | --------- | ---------- | ---------- |
| Exp 1      | Baseline        | 2e-4          | 16              | 1      | ~1.42      | ~1.55     | ~4.71      | ~18        |
| Exp 2      | Lower LR        | 1e-4          | 16              | 2      | ~1.31      | ~1.44     | ~4.22      | ~36        |
| **Exp 3**  | **Best Config** | **5e-5**      | **16**          | **2**  | **~1.27**  | **~1.38** | **~3.97**  | **~36**    |

**Key observations:**

- Lower learning rates (5e-5) with 2 training epochs gave the most stable convergence.
- The cosine LR scheduler with 5% warmup helped avoid loss spikes early in training.
- The 8-bit paged AdamW optimizer reduced GPU memory vs standard Adam by ~1.5 GB.
- Effective batch size of 16 (batch=2 × grad_accum=8 or batch=4 × grad_accum=4) was optimal for the T4.

---

## Evaluation Metrics

### Automatic Metrics (Base vs Fine-Tuned, 100 validation samples)

| Metric  | Base Model | Fine-Tuned | Δ (Improvement) |
| ------- | ---------- | ---------- | --------------- |
| ROUGE-1 | ~18.4      | ~31.7      | **+72%**        |
| ROUGE-2 | ~5.1       | ~11.3      | **+121%**       |
| ROUGE-L | ~14.2      | ~26.8      | **+89%**        |
| BLEU    | ~3.2       | ~7.9       | **+147%**       |

> _Exact numbers will vary slightly per run. These are representative results from a single seed._

### Qualitative Testing Examples

**Q: What are the classic symptoms of appendicitis?**

- **Base:** _"Appendicitis is an inflammation of the appendix, a small pouch in the abdomen..."_ _(generic, vague)_
- **Fine-Tuned:** _"Classic symptoms include right lower quadrant pain (McBurney's point), nausea, vomiting, fever, and rebound tenderness. Pain often begins periumbilically and migrates to the RLQ."_

**Q: How does metformin work in treating type 2 diabetes?**

- **Base:** _"Metformin is a medication used to treat diabetes. It helps lower blood sugar levels..."_ _(surface-level)_
- **Fine-Tuned:** _"Metformin works primarily by inhibiting hepatic gluconeogenesis via activation of AMPK, reducing hepatic glucose output. It also improves peripheral insulin sensitivity and delays intestinal glucose absorption."_

**Q: What is the best recipe for chocolate cake?** _(Out-of-domain)_

- **Fine-Tuned:** _"I'm a medical assistant specialized in clinical and biomedical topics. I'm not the right resource for cooking questions — I'd recommend a recipe site for that! Is there a medical question I can help with?"_

---

## How to Run

### Option 1: Google Colab (Recommended)

1. Click the **Open in Colab** badge at the top of this README.
2. Go to **Runtime → Change runtime type → T4 GPU**.
3. Run all cells in order (Ctrl+F9).
4. Cell 11 will output a **public Gradio link** — click it to use the chatbot.

### Option 2: Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/medical-chatbot.git
cd medical-chatbot

pip install transformers peft datasets trl accelerate bitsandbytes \
            evaluate rouge_score sacrebleu gradio matplotlib pandas numpy

# Run the full pipeline
python medical_chatbot_finetune.py
```

> **Note:** Local run requires a CUDA GPU with ≥8 GB VRAM. For CPU-only inference, change `device_map="auto"` to `device_map="cpu"` and remove `BitsAndBytesConfig` (uses more RAM).

---

## Sample Screenshots

| Dataset Analysis           | Experiment Results          | Base vs Fine-Tuned              |
| -------------------------- | --------------------------- | ------------------------------- |
| `dataset_distribution.png` | `experiment_comparison.png` | `base_vs_finetuned_metrics.png` |

---

## Disclaimer

This chatbot is intended **for educational and research purposes only**.
It should **not** be used as a substitute for professional medical advice, diagnosis, or treatment.
Always consult a qualified healthcare provider for medical decisions.

---

## License

MIT License — see `LICENSE` for details.
