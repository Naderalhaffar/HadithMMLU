# Hadith MCQ Kit — README (OpenAI + Gemini/DeepSeek)

This guide walks you through the **full pipeline**: generating a Multiple-Choice Question dataset from hadith texts, then evaluating it either with **OpenAI (API)** or with **Gemini/DeepSeek via chat** (copy raw replies and grade them offline). It’s written to be friendly for first-time users on Windows/macOS/Linux.

---

## What’s in this kit?

* `hadith.py` — **Dataset → MCQs generator** (English/Arabic).
  Creates MCQs for: **Source**, **Chapter**, **Cloze**, **Authenticity**.
* `openaieval.py` — **Evaluator (OpenAI API)**.
  Runs an OpenAI model (e.g., `gpt-4o-mini`) over MCQs and reports accuracy, plus charts.
* `chatgrade.py` — **(Optional) offline grader for chat models** (Gemini/DeepSeek).
  You paste model replies collected from the web UI; this script grades them with the **same normalizer** as `openaieval.py` and makes the same plots.

> If `chatgrade.py` isn’t in your repo yet, copy the tiny script below into a new file with that name.

---

## 1) Requirements

### Python

* Python **3.9–3.11** is recommended.

### Python packages

Install once:

```bash
pip install -U pandas scikit-learn "openai>=1.0.0" matplotlib tiktoken
```

### OpenAI API key (required for `openaieval.py`; optional for `hadith.py` if you use LLM distractors)

Set your key as an environment variable:

**Windows (PowerShell):**

```powershell
$env:OPENAI_API_KEY="sk-yourkey"
# optional permanent (restart terminal afterwards):
setx OPENAI_API_KEY "sk-yourkey"
```

**macOS/Linux (bash/zsh):**

```bash
export OPENAI_API_KEY="sk-yourkey"
```

---

## 2) Project layout (suggested)

```
your-folder/
  hadith.py
  openaieval.py
  chatgrade.py            # (optional) paste from this README
  data/
    hadithdatasetcsv.csv      # your input hadith data
    hadith_authenticity.csv   # (optional) authenticity labels
    arabicproverbs.csv        # (optional) proverbs for “Not a Hadith”
  results/
    # evaluation outputs and charts will be saved here
  figs/
    # put final figures here if you want LaTeX to \includegraphics them
```

You can change paths—just match them in the commands.

---

## 3) Input files: what `hadith.py` expects

### Required input CSV (e.g., `data/hadithdatasetcsv.csv`)

Columns (string):

* `text_en` — English matn / narration text
* `text_ar` — Arabic matn / narration text
* `source` — Collection/book (e.g., `Sahih Bukhari`)
* `chapter` — Chapter title (bilingual is fine)
* `hadith_no` — ID/number in the collection

> Rows missing any of these are skipped.

### Optional: Authenticity CSV (`--auth`)

* Must include `text_en`, `text_ar`, `consensus_grade` (e.g., **Sahih**, **Hasan**, **Daif**)
* Optional: `collection` (used in prompt wording)

### Optional: Arabic proverbs CSV (`--proverbs`)

* Two columns used to create “Not a Hadith” negatives:

  * `Column4` → English text
  * `Column1` → Arabic text
    (the script renames them to `text_en`/`text_ar`)

---

## 4) Step-by-step: Generate MCQs → Evaluate

### A) Generate MCQs (`hadith.py`)

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

### B) Evaluate with OpenAI (API) — `openaieval.py`

**Recommended run:**

```bash
python openaieval.py \
  --questions data/generated_hadith_questions.csv \
  --model gpt-4o-mini \
  --max_tokens 3 \
  --temperature 0 \
  --out results/gpt4o_results.csv
```

**What it does**

* Sends each MCQ with **numbered options** (1–4) and enforces **numeric-only** answers.
* If the model returns free-text, maps it to the closest option using a robust normalizer:

  * Unicode NFKC, punctuation unification, **Arabic diacritics stripped**, lowercasing, fuzzy match.
* Prints a **report** (overall + per template + per language).
* Saves `results/gpt4o_results.csv` with columns:
  `template, lang, prompt, choices, correct_answer, model_answer, raw_model_reply, is_correct`
* Exports bar charts:

  * `accuracy_by_template*.png`
  * `accuracy_by_language*.png`

**Tips**

* Keep `--max_tokens 3` and `--temperature 0` for consistent MCQ output.
* To run large Arabic-only batches (e.g., 4,000 items), just point `--questions` to the big CSV.

---

### C) Evaluate Gemini or DeepSeek (chat UI → offline grade)

Because we didn’t use their paid APIs, we **pasted** each question (prompt + options) into the web chat, captured the model’s **raw textual reply** (e.g., “4”, “Option C”, “Sahih”, “Bukhari”), and graded those replies **offline** against the gold CSV using the **same normalizer** as `openaieval.py`.

#### 1) Collect raw chat replies

Create a CSV (e.g., `results/gemini_raw.csv`) with the following columns per item:

* `template` — one of `Source|Chapter|Cloze|Authenticity`
* `lang` — `en` or `ar`
* `prompt` — the question text shown to the model
* `choices` — the stringified list of options, e.g. `["A","B","C","D"]`
* `answer` — the gold/correct option string (must match one of `choices`)
* `raw_model_reply` — exactly what the model replied in chat (any format)

> You can export the original `generated_hadith_questions.csv` and add a new `raw_model_reply` column while you’re chatting, or create a separate file with the same columns.

#### 2) Grade chat replies with `chatgrade.py`

Create `chatgrade.py` with the code below (copied from the same normalization logic used for API evaluation):

```python
import ast, re, difflib, unicodedata, argparse, pandas as pd
import matplotlib.pyplot as plt

AR_DIACRITICS = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u0653-\u065F\u06D6-\u06ED]')
def strip_ar_diacritics(s: str) -> str:
    return AR_DIACRITICS.sub('', s)

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('’', "'").replace('ʻ', "'").replace('`', "'")
    s = s.replace('–','-').replace('—','-').strip().lower()
    return strip_ar_diacritics(s)

def closest_choice(pred: str, choices: list) -> str:
    pred_n = normalize_text(pred)
    # exact and contains checks
    for c in choices:
        if normalize_text(c) == pred_n:
            return c
    for c in choices:
        cn = normalize_text(c)
        if pred_n in cn or cn in pred_n:
            return c
    # difflib fallback
    norm_map = {normalize_text(c): c for c in choices}
    best = difflib.get_close_matches(pred_n, list(norm_map.keys()), n=1, cutoff=0.0)
    return norm_map[best[0]] if best else choices[0]

def as_list(x):
    if isinstance(x, list): return x
    if isinstance(x, str):
        try: return ast.literal_eval(x)
        except Exception: return [x]
    return [str(x)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="CSV with template,lang,prompt,choices,answer,raw_model_reply")
    ap.add_argument("--out", required=True, help="Output graded CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.raw)
    rows = []
    for _, r in df.iterrows():
        choices = [str(c) for c in as_list(r["choices"])]
        gold = str(r["answer"])
        raw  = str(r["raw_model_reply"])
        mapped = closest_choice(raw, choices)
        correct = normalize_text(mapped) == normalize_text(gold)
        rows.append({
            "template": r["template"],
            "lang": r["lang"],
            "prompt": r["prompt"],
            "choices": r["choices"],
            "correct_answer": gold,
            "model_answer": mapped,
            "raw_model_reply": raw,
            "is_correct": int(correct)
        })

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False, encoding="utf-8-sig")

    print("\n📊 Chat Evaluation Report")
    print("="*60)
    overall = res["is_correct"].mean()*100 if len(res) else 0.0
    print(f"Overall accuracy: {overall:.2f}%  ({res['is_correct'].sum()} / {len(res)})")

    print("\nBy template:")
    for t, g in res.groupby("template"):
        acc = g["is_correct"].mean()*100
        print(f"  {t:<12} {acc:6.2f}%   (n={len(g)})")

    print("\nBy language:")
    for l, g in res.groupby("lang"):
        acc = g["is_correct"].mean()*100
        print(f"  {l:<2} {acc:10.2f}%   (n={len(g)})")

    # plots
    (res.groupby("template")["is_correct"].mean()*100).plot(kind="bar", edgecolor="black")
    plt.title("Accuracy by Template"); plt.ylabel("Accuracy (%)"); plt.xlabel("Template"); plt.tight_layout()
    plt.savefig(args.out.replace(".csv", "_accuracy_by_template.png")); plt.close()

    (res.groupby("lang")["is_correct"].mean()*100).plot(kind="bar", edgecolor="black")
    plt.title("Accuracy by Language"); plt.ylabel("Accuracy (%)"); plt.xlabel("Language"); plt.tight_layout()
    plt.savefig(args.out.replace(".csv", "_accuracy_by_language.png")); plt.close()

    print("\nSaved:")
    print(f"  - Per-item CSV: {args.out}")
    print(f"  - Charts: {args.out.replace('.csv','_accuracy_by_template.png')}, {args.out.replace('.csv','_accuracy_by_language.png')}\n")

if __name__ == "__main__":
    main()
```

**Run it (Gemini example):**

```bash
python chatgrade.py \
  --raw results/gemini_raw.csv \
  --out results/gemini_eval.csv
```

**Run it (DeepSeek example):**

```bash
python chatgrade.py \
  --raw results/deepseek_raw.csv \
  --out results/deepseek_eval.csv
```

You’ll get a terminal summary, a graded CSV, and PNG charts (the same style as the OpenAI API run).

---

## 5) Typical workflow (quick recap)

1. **Prepare input CSVs** in `data/` (see formats above).
2. **Generate MCQs**:

   ```bash
   python hadith.py \
     --input data/hadithdatasetcsv.csv \
     --output data/generated_hadith_questions.csv \
     --num 100 --lang both \
     --auth data/hadith_authenticity.csv \
     --proverbs data/arabicproverbs.csv
   ```
3. **Evaluate with OpenAI (API)**:

   ```bash
   python openaieval.py \
     --questions data/generated_hadith_questions.csv \
     --model gpt-4o-mini \
     --max_tokens 3 --temperature 0 \
     --out results/gpt4o_results.csv
   ```
4. **Evaluate Gemini/DeepSeek (chat → offline)**:

   * Paste each MCQ into chat, copy the model’s reply into a CSV (`raw_model_reply` column).
   * Grade it:

     ```bash
     python chatgrade.py --raw results/gemini_raw.csv --out results/gemini_eval.csv
     # or
     python chatgrade.py --raw results/deepseek_raw.csv --out results/deepseek_eval.csv
     ```
5. **Inspect results** in `results/*.csv` and the terminal report. Charts are saved alongside each CSV.

---

## 6) How grading works (important)

All evaluators (API and chat) use the **same** robust mapping from free text → option:

* Normalize both sides (Unicode NFKC, punctuation unification, **Arabic diacritics removed**, lowercasing).
* Try exact match and “contains” in both directions.
* Otherwise use a fuzzy fallback (difflib) to pick the closest option.

This ensures replies such as “Option C”, “3”, “Sahih”, or “Bukhari” are graded consistently.

---

## 7) Interpreting results

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

The per-item CSV includes:
`template, lang, prompt, choices, correct_answer, model_answer, raw_model_reply, is_correct`

---


