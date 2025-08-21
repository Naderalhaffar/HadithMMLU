import os
import ast
import argparse
import difflib
import re
import unicodedata
import pandas as pd
import openai
import matplotlib.pyplot as plt


openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------- Normalization helpers ----------
AR_DIACRITICS = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u0653-\u065F\u06D6-\u06ED]')
def strip_ar_diacritics(s: str) -> str:
    return AR_DIACRITICS.sub('', s)

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('’', "'").replace('ʻ', "'").replace('`', "'")
    s = s.replace('–','-').replace('—','-')
    s = s.strip()
    s = strip_ar_diacritics(s)
    return s.lower()

def closest_choice(pred: str, choices: list) -> str:
    pred_n = normalize_text(pred)
    # exact/contains checks
    for c in choices:
        if normalize_text(c) == pred_n:
            return c
    for c in choices:
        if pred_n in normalize_text(c) or normalize_text(c) in pred_n:
            return c
    # difflib fallback
    norm_map = {normalize_text(c): c for c in choices}
    best = difflib.get_close_matches(pred_n, list(norm_map.keys()), n=1, cutoff=0.0)
    return norm_map[best[0]] if best else choices[0]

# ---------- Task-specific instruction ----------
def task_instruction(template: str, lang: str) -> str:
    t = (template or '').strip().lower()
    # Bilingual hint helps Arabic prompts too
    if t == 'source':
        return ("Task: Choose the correct hadith collection (one of the options). "
                "المهمة: اختر اسم المجموعة الحديثية الصحيحة من الخيارات فقط.")
    if t == 'chapter':
        return ("Task: Choose the correct chapter title under which this hadith appears. "
                "المهمة: اختر عنوان الباب الصحيح لهذا الحديث من الخيارات فقط.")
    if t == 'cloze':
        return ("Task: Fill the blank by choosing the exact missing word from the options. "
                "المهمة: اختر الكلمة الناقصة الصحيحة من الخيارات فقط.")
    if t == 'authenticity':
        return ("Task: Choose the hadith authenticity grade. If the text is not a hadith, choose 'Not a Hadith'. "
                "المهمة: اختر درجة صحة الحديث، وإذا لم يكن نصًا حديثيًا فاختر 'ليس حديثاً'.")
    return "Task: Choose exactly one option. المهمة: اختر خيارًا واحدًا فقط."

def ask_model(prompt, choices, template, lang, model, max_tokens=3, temperature=0):
    # Ensure choices list
    if isinstance(choices, str):
        try:
            choices = ast.literal_eval(choices)
        except Exception:
            choices = [choices]
    choices = [str(c) for c in choices]

    # Numbered options
    options_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
    instr = task_instruction(template, lang)

    system = (
        "You are an MCQ solver. Return ONLY the index number (1-4). "
        "Do not explain."
    )
    user = f"{instr}\n\nQuestion:\n{prompt}\n\nOptions:\n{options_block}\n\nReturn ONLY one number (1-4)."

    resp = openai.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}],
        temperature=temperature,
        max_tokens=max_tokens  # small since we only want "1", "2", "3" or "4"
    )
    raw = resp.choices[0].message.content.strip()

    # Map numeric -> chosen string; else fallback to fuzzy
    m = re.match(r'^\s*([1-4])\s*$', raw)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(choices):
            return choices[idx], raw

    # If the model ignored instructions, map to closest choice
    mapped = closest_choice(raw, choices)
    return mapped, raw

def main():
    parser = argparse.ArgumentParser(description="Evaluate Hadith MCQs with OpenAI")
    parser.add_argument("--questions", required=True, help="CSV of generated questions")
    parser.add_argument("--out", required=True, help="Output CSV with model results")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--max_tokens", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    df = pd.read_csv(args.questions)
    rows = []
    for _, r in df.iterrows():
        template = r.get("template","")
        lang = r.get("lang","")
        prompt = r["prompt"]
        choices = r["choices"]
        gold = str(r["answer"])

        model_choice, raw_reply = ask_model(
            prompt, choices, template, lang,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )

        correct = normalize_text(model_choice) == normalize_text(gold)
        rows.append({
            "template": template,
            "lang": lang,
            "prompt": prompt,
            "choices": choices,
            "correct_answer": gold,
            "model_answer": model_choice,
            "raw_model_reply": raw_reply,
            "is_correct": int(correct)
        })

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False, encoding="utf-8-sig")

    # Report
    print("\n📊 Evaluation Report")
    print("=" * 50)
    overall = res["is_correct"].mean() * 100 if len(res) else 0.0
    print(f"Overall Accuracy: {overall:.2f}% ({res['is_correct'].sum()} / {len(res)})")

    print("\nAccuracy by Template:")
    for t, g in res.groupby("template"):
        acc = g["is_correct"].mean() * 100
        print(f"  {t:<15}: {acc:.2f}% ({g['is_correct'].sum()} / {len(g)})")

    print("\nAccuracy by Language:")
    for l, g in res.groupby("lang"):
        acc = g["is_correct"].mean() * 100
        print(f"  {l:<5}: {acc:.2f}% ({g['is_correct'].sum()} / {len(g)})")

    print("=" * 50)
    print(f"Detailed results saved to: {args.out}\n")

    template_acc = res.groupby("template")["is_correct"].mean() * 100
    template_acc.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Accuracy by Template")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Template")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("accuracy_by_template.png")
    plt.show()

    # Accuracy by language
    lang_acc = res.groupby("lang")["is_correct"].mean() * 100
    lang_acc.plot(kind="bar", color="lightgreen", edgecolor="black")
    plt.title("Accuracy by Language")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Language")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("accuracy_by_language.png")
    plt.show()
if __name__ == "__main__":
    main()
