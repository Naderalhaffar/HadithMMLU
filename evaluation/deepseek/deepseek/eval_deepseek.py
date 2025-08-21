
import os
import re
import ast
import math
import difflib
import unicodedata
import argparse
import pandas as pd
import matplotlib.pyplot as plt

AR_DIACRITICS = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u0653-\u065F\u06D6-\u06ED]')

def strip_ar_diacritics(s: str) -> str:
    return AR_DIACRITICS.sub('', s)

def normalize_text(s: str) -> str:
    if s is None:
        return ''
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('’', "'").replace('ʻ', "'").replace('`', "'")
    s = s.replace('–','-').replace('—','-')
    s = s.strip()
    s = strip_ar_diacritics(s)
    s = re.sub(r'\s+', ' ', s)
    return s.lower()

def closest_choice(pred: str, choices: list) -> str:
    if isinstance(choices, str):
        try:
            choices = ast.literal_eval(choices)
        except Exception:
            choices = [choices]
    choices = [str(c) for c in choices]

    pred_n = normalize_text(pred)
    alias = {
        "bukhari": "sahih bukhari",
        "muslim": "sahih muslim",
        "tirmidhi": "jami' al-tirmidhi",
        "nasai": "sunan an-nasa'i",
        "an nasai": "sunan an-nasa'i",
        "nasa'i": "sunan an-nasa'i",
        "abida'ud": "sunan abi da'ud",
        "abi da'ud": "sunan abi da'ud",
        "abu dawud": "sunan abi da'ud",
        "ibn majah": "sunan ibn majah",
        "صحيح البخاري": "صحيح البخاري",
        "صحيح مسلم": "صحيح مسلم",
        "جامع الترمذي": "جامع الترمذي",
        "سنن النسائي": "سنن النسائي",
        "سنن أبي داود": "سنن أبي داود",
        "سنن ابن ماجه": "سنن ابن ماجه",
    }
    if pred_n in alias:
        pred_n = normalize_text(alias[pred_n])

    for c in choices:
        if normalize_text(c) == pred_n:
            return c
    for c in choices:
        cn = normalize_text(c)
        if pred_n in cn or cn in pred_n:
            return c
    norm_map = {normalize_text(c): c for c in choices}
    best = difflib.get_close_matches(pred_n, list(norm_map.keys()), n=1, cutoff=0.0)
    return norm_map[best[0]] if best else choices[0]

def normalize_authenticity_label(s: str) -> str:
    n = normalize_text(s)
    if "not a hadith" in n or "not hadith" in n:
        return "Not a Hadith"
    if "sahih" in n:
        return "Sahih (Authentic)"
    if "daif" in n or "da'if" in n or "weak" in n:
        return "Da'if (Weak)"
    if "hasan" in n or "good" in n:
        return "Hasan (Good)"
    if "ليس حديث" in n or "ليس بحديث" in n:
        return "ليس بحديث"
    if "صحيح" in n:
        return "صحيح"
    if "ضعيف" in n:
        return "ضعيف"
    if "حسن" in n:
        return "حسن"
    return s

def parse_deepseek_text(raw: str):
    lines = [l.strip() for l in raw.splitlines()]

    def next_line():
        while lines:
            s = lines.pop(0).strip()
            if s != "":
                return s
        return None

    preds = []

    # 1) EN Source
    for _ in range(50):
        preds.append(next_line())

    # 2) EN Cloze
    for _ in range(50):
        preds.append(next_line())

    # 3) EN Chapter
    for _ in range(50):
        preds.append(next_line())

    # 4) EN Authenticity
    for _ in range(50):
        s = next_line()
        preds.append(s)

    # 5) AR Source (numbered)
    while lines and not re.match(r'^\s*\d+\.\s*', lines[0]):
        lines.pop(0)
    for _ in range(50):
        s = next_line()
        if s is None: break
        s = re.sub(r'^\s*\d+\.\s*', '', s)
        preds.append(s)

    # 6) AR Cloze (numbered)
    for _ in range(50):
        s = next_line()
        if s is None: break
        s = re.sub(r'^\s*\d+\.\s*', '', s)
        s = s.rstrip('،,')
        preds.append(s)

    # 7) AR Chapter (numbered, keep Arabic part if present)
    for _ in range(50):
        s = next_line()
        if s is None: break
        s = re.sub(r'^\s*\d+\.\s*', '', s)
        m = re.search(r'(كتاب.+)$', s)
        if m:
            s = m.group(1)
        preds.append(s)

    # 8) AR Authenticity
    while lines and not (("Authenticity" in lines[0]) or re.match(r'^\s*\d+\.\s*', lines[0])):
        lines.pop(0)
    if lines and "Authenticity" in lines[0]:
        lines.pop(0)
    for _ in range(50):
        s = next_line()
        if s is None: break
        s = re.sub(r'^\s*\d+\.\s*', '', s)
        preds.append(s)

    if len(preds) != 400:
        print(f"WARNING: Parsed {len(preds)} predictions (expected 400).")
    return preds

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z*z/(2*n)
    adj = z * ( (p*(1-p) + z*z/(4*n)) / n ) ** 0.5
    lower = (centre - adj) / denom
    upper = (centre + adj) / denom
    return max(0.0, lower)*100, min(1.0, upper)*100

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="deepseek_eval.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)

    if args.preds.lower().endswith(".csv"):
        p = pd.read_csv(args.preds)
        col = next((c for c in p.columns if c.lower() in ("pred","prediction","deepseek_pred","answer","model_answer")), None)
        if col is None:
            raise ValueError("CSV predictions must have a 'pred' column.")
        preds = p[col].tolist()
    else:
        with open(args.preds, "r", encoding="utf-8") as f:
            raw = f.read()
        preds = parse_deepseek_text(raw)

    n = min(len(preds), len(df))
    df = df.iloc[:n].copy()
    preds = preds[:n]

    cleaned = []
    for i, row in df.iterrows():
        t = str(row["template"]).strip().lower()
        pr = preds[i]
        if t == "authenticity":
            pr = normalize_authenticity_label(pr)
        cleaned.append(pr)

    rows = []
    for i, row in df.iterrows():
        chosen = closest_choice(cleaned[i], row["choices"])
        correct = normalize_text(chosen) == normalize_text(row["answer"])
        rows.append({
            "idx": i,
            "template": row["template"],
            "lang": row["lang"],
            "prompt": row["prompt"],
            "choices": row["choices"],
            "gold_answer": row["answer"],
            "deepseek_pred_raw": preds[i],
            "deepseek_mapped_choice": chosen,
            "is_correct": int(correct),
        })

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False, encoding="utf-8-sig")

    print("\n📊 DeepSeek Evaluation Report")
    print("="*60)
    overall = res["is_correct"].mean()*100 if len(res) else 0.0
    lo, hi = wilson_ci(res["is_correct"].sum(), len(res))
    print(f"Overall accuracy: {overall:.2f}%  (95% CI: {lo:.2f}–{hi:.2f}; n={len(res)})")

    print("\nBy template:")
    for t, grp in res.groupby("template"):
        acc = grp["is_correct"].mean()*100
        lo, hi = wilson_ci(grp["is_correct"].sum(), len(grp))
        print(f"  {t:<12} {acc:6.2f}%   (95% CI: {lo:.2f}–{hi:.2f}; n={len(grp)})")

    print("\nBy language:")
    for l, grp in res.groupby("lang"):
        acc = grp["is_correct"].mean()*100
        lo, hi = wilson_ci(grp["is_correct"].sum(), len(grp))
        print(f"  {l:<3}          {acc:6.2f}%   (95% CI: {lo:.2f}–{hi:.2f}; n={len(grp)})")

    # charts without specifying colors/styles
    tpl_acc = res.groupby("template")["is_correct"].mean()*100
    plt.figure()
    tpl_acc.plot(kind="bar")
    plt.title("DeepSeek Accuracy by Template")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Template")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("deepseek_accuracy_by_template.png")
    plt.close()

    lang_acc = res.groupby("lang")["is_correct"].mean()*100
    plt.figure()
    lang_acc.plot(kind="bar")
    plt.title("DeepSeek Accuracy by Language")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Language")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("deepseek_accuracy_by_language.png")
    plt.close()

    print("\nSaved:")
    print(f"  - Per-item CSV: {args.out}")
    print("  - Charts: deepseek_accuracy_by_template.png, deepseek_accuracy_by_language.png")

if __name__ == "__main__":
    main()
