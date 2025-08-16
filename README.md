# Hadith MCQ Kit — README

This guide walks you through the **full pipeline**: generating a Multiple-Choice Question dataset from hadith texts, then evaluating it with either **OpenAI** or a **local Hugging Face model**. It’s written to be friendly for first-time users on Windows/macOS/Linux.

---

## What’s in this kit?

* `hadith.py` – **Dataset → MCQs generator** (English/Arabic).
  Creates MCQs for: **Source**, **Chapter**, **Cloze**, **Authenticity**.
* `openaieval.py` – **Evaluator (OpenAI)**.
  Runs a model (e.g., `gpt-4o-mini`) over MCQs and reports accuracy.
* `hfeval.py` – **Evaluator (Hugging Face)**.
  Runs a **local** model (e.g., Qwen/Mistral) over MCQs and reports accuracy.

---

## 1) Requirements

### Python

* Python **3.9–3.11** is recommended.

### Python packages

Install these once:

```bash
pip install -U pandas scikit-learn "openai>=1.0.0" transformers accelerate sentencepiece safetensors einops tiktoken
```

> If you’ll use large Hugging Face models (7B+), also ensure **PyTorch** is installed properly:
>
> * **GPU (CUDA 12.1):** `pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
> * **GPU (CUDA 11.8):** `pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
> * **CPU only:** `pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

### OpenAI API key (required for `hadith.py` + `openaieval.py`)

Set your key as an environment variable:

* **Windows (PowerShell):**

  ```powershell
  $env:OPENAI_API_KEY="sk-yourkey"
  ```

  (For a permanent user-level var: `setx OPENAI_API_KEY "sk-yourkey"` and **restart** terminal.)

* **macOS/Linux (bash/zsh):**

  ```bash
  export OPENAI_API_KEY="sk-yourkey"
  ```

---

## 2) Project layout (suggested)

On your Desktop (or any folder):

```
your-folder/
  hadith.py
  openaieval.py
  hfeval.py
  data/
    hadithdatasetcsv.csv           # your input hadith data
    hadith_authenticity.csv        # (optional) authenticity labels
    arabicproverbs.csv             # (optional) Arabic proverbs
  results/
    # evaluation outputs will be saved here
```

You can change paths—just match them in the commands.

---

## 3) Input files: what `hadith.py` expects

### Required input CSV (e.g., `data/hadithdatasetcsv.csv`)

Columns (string):

* `text_en` – English matn / narration text
* `text_ar` – Arabic matn / narration text
* `source` – Collection/book (e.g., `Sahih Bukhari`)
* `chapter` – Chapter title (can be bilingual; generator handles it)
* `hadith_no` – ID/number inside the collection

> Rows missing any of these will be skipped.

### Optional: Authenticity CSV (`--auth`)

* Must include `text_en`, `text_ar`, `consensus_grade`
* Optional: `collection` (improves prompt wording)
* `consensus_grade` values like: **Sahih**, **Hasan**, **Daif**

### Optional: Arabic proverbs CSV (`--proverbs`)

* A simple 2-column file used to create “Not a Hadith” negatives.
* Expected columns:

  * `Column4` → English text
  * `Column1` → Arabic text
    The script renames them to `text_en`/`text_ar`.

---

## 4) Step-by-step: Generate MCQs → Evaluate

### A) Generate MCQs (`hadith.py`)

From your project folder:

```bash
python hadith.py \
  --input data/hadithdatasetcsv.csv \
  --output data/generated_hadith_questions.csv \
  --num 100 \
  --lang both \
  --auth data/hadith_authenticity.csv \
  --proverbs data/arabicproverbs.csv
```

**What it does**

* Cleans & deduplicates your hadith dataset.
* Creates MCQs for **Source**, **Chapter**, **Cloze**, **Authenticity** (both EN/AR if `--lang both`).
* Writes a CSV with columns like:
  `template, lang, prompt, choices, answer`

> Note: `choices` is a **stringified list** (e.g., `["A","B","C","D"]`). The evaluators handle this.

---

### B) Evaluate with OpenAI (`openaieval.py`)

**Simple run (recommended):**

```bash
python openaieval.py \
  --questions data/generated_hadith_questions.csv \
  --model gpt-4o-mini \
  --max_tokens 3 \
  --temperature 0 \
  --out results/gpt4o_results.csv \
  --show_confusions --show_mistakes
```

**What it does**

* Asks the model each MCQ with **numbered options**.
* Enforces **numeric-only** answers (1–n) and robustly maps free-text if the model disobeys.
* Normalizes text (case, punctuation, Arabic diacritics) before grading.
* Prints a **report** (overall + per template + per language).
* Saves `results/gpt4o_results.csv` with columns:

  * `template, lang, prompt, choices, correct_answer, model_answer, raw_model_reply, is_correct`

**Tips**

* For speed and consistency: keep `--max_tokens 3` and `--temperature 0`.
* `--show_confusions` and `--show_mistakes` print small diagnostics.
* You can test other OpenAI models by changing `--model` (e.g., `gpt-4o`).

---

### C) Evaluate with a local Hugging Face model (`hfeval.py`)

> **CPU users:** prefer **small models** (0.5B–1.5B) and `--dtype float32`.
> **GPU users:** install the right CUDA PyTorch and use `--dtype float16` or `--dtype bfloat16`.

**CPU-friendly example (fastest):**

```bash
python hfeval.py \
  --questions data/generated_hadith_questions.csv \
  --model_name Qwen/Qwen2-0.5B-Instruct \
  --dtype float32 \
  --use_chat_template \
  --temperature 0.2 --top_p 0.9 \
  --samples 5 \
  --out results/qwen_results.csv \
  --show_confusions --show_mistakes \
  --trust_remote_code
```

**GPU (7B model) example:**

```bash
python hfeval.py \
  --questions data/generated_hadith_questions.csv \
  --model_name Qwen/Qwen2-7B-Instruct \
  --dtype float16 \
  --use_chat_template \
  --temperature 0.2 --top_p 0.9 \
  --samples 5 \
  --out results/qwen_results.csv \
  --show_confusions --show_mistakes \
  --trust_remote_code
```

**What it does**

* Same evaluation logic as the OpenAI script (numeric-only answers, normalization, voting).
* `--samples K` enables **self-consistency** voting (K generations; pick majority).
* `--use_chat_template` uses the model’s chat template if available (often better).
* `--trust_remote_code` is required for some chat models (like Qwen) to load custom code.

**About downloads**

* First run will **download** model weights to your HF cache:

  * Windows: `C:\Users\<you>\.cache\huggingface\hub\`
  * macOS/Linux: `~/.cache/huggingface/hub/`
* Large models (7B+) are **many GB**. Use smaller models on CPU to save time/space.

---

## 5) Typical workflow (quick recap)

1. **Prepare input CSVs** in `data/` (see formats above).
2. **Generate MCQs**:

   ```bash
   python hadith.py --input data/hadithdatasetcsv.csv --output data/generated_hadith_questions.csv --num 100 --lang both --auth data/hadith_authenticity.csv --proverbs data/arabicproverbs.csv
   ```
3. **Evaluate (OpenAI)**:

   ```bash
   python openaieval.py --questions data/generated_hadith_questions.csv --model gpt-4o-mini --max_tokens 3 --temperature 0 --out results/gpt4o_results.csv
   ```

   **OR** **Evaluate (Hugging Face)**:

   ```bash
   python hfeval.py --questions data/generated_hadith_questions.csv --model_name Qwen/Qwen2-0.5B-Instruct --dtype float32 --use_chat_template --temperature 0.2 --top_p 0.9 --samples 5 --out results/qwen_results.csv --trust_remote_code
   ```
4. **Inspect results** in `results/*.csv` and the terminal report.

---

## 6) Interpreting results

Example terminal summary:

```
📊 Evaluation Report
==================================================
Overall Accuracy: 67.25% (269 / 400)

Accuracy by Template:
  Authenticity   : 64.00% (64 / 100)
  Chapter        : 78.00% (78 / 100)
  Cloze          : 72.00% (72 / 100)
  Source         : 55.00% (55 / 100)

Accuracy by Language:
  ar   : 68.50% (137 / 200)
  en   : 64.00% (128 / 200)
==================================================
Detailed results saved to: results/...
```

CSV columns:

* `template, lang, prompt, choices, correct_answer, model_answer, raw_model_reply, is_correct`

---





