
import os
import re
import ast
import math
import argparse
import difflib
import unicodedata
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
    """Return the best-matching element of choices to pred using normalization and difflib."""
    if isinstance(choices, str):
        try:
            choices = ast.literal_eval(choices)
        except Exception:
            choices = [choices]
    choices = [str(c) for c in choices]

    pred_n = normalize_text(pred)
    for c in choices:
        if normalize_text(c) == pred_n:
            return c
    for c in choices:
        cn = normalize_text(c)
        if pred_n in cn or cn in pred_n:
            return c
    alias_map = {
        "sahih al-bukhari": "sahih bukhari",
        "bukhari": "sahih bukhari",
        "sahih al-muslim": "sahih muslim",
        "muslim": "sahih muslim",
        "tirmidhi": "jami' al-tirmidhi",
        "jami at-tirmidhi": "jami' al-tirmidhi",
        "jami al-tirmidhi": "jami' al-tirmidhi",
        "abudawud": "sunan abi da'ud",
        "abu dawud": "sunan abi da'ud",
        "an-nasa'i": "sunan an-nasa'i",
        "nasa'i": "sunan an-nasa'i",
        "ibnmajah": "sunan ibn majah",
        "ibn majah": "sunan ibn majah",
        "صحيح البخاري": "صحيح البخاري",
        "صحيح مسلم": "صحيح مسلم",
        "جامع الترمذي": "جامع الترمذي",
        "سنن أبي داود": "سنن أبي داود",
        "سنن النسائي": "سنن النسائي",
        "سنن ابن ماجه": "سنن ابن ماجه",
    }
    if pred_n in alias_map:
        canon = alias_map[pred_n]
        for c in choices:
            if normalize_text(c) == normalize_text(canon):
                return c

    norm_map = {normalize_text(c): c for c in choices}
    best = difflib.get_close_matches(pred_n, list(norm_map.keys()), n=1, cutoff=0.0)
    return norm_map[best[0]] if best else choices[0]

def normalize_authenticity_label(s: str) -> str:
    n = normalize_text(s)
    if "not a hadith" in n or "not hadith" in n or "proverb" in n or "quote" in n or "bible" in n or "saying" in n:
        return "Not a Hadith"
    if "sahih" in n or "authentic" in n:
        return "Sahih (Authentic)"
    if "hasan" in n or "good" in n:
        return "Hasan (Good)"
    if "da'if" in n or "daif" in n or "weak" in n:
        return "Da'if (Weak)"
    if "ليس بحديث" in n or "ليس حديثا" in n or "ليس حديثاً" in n:
        return "ليس بحديث"
    if "صحيح" in n:
        return "صحيح"
    if "حسن" in n:
        return "حسن"
    if "ضعيف" in n:
        return "ضعيف"
    return s

HEAD_MARKERS = {
    "en": {
        "cloze": r"^Cloze",
        "chapter": r"^Book:\s*",
        "chapter_title": r"^Chapter Title:\s*(.+)$",
        "auth": r"^Authenticity Grade",
    },
    "ar": {
        "cloze": r"^إكمال الفراغ",
        "chapter": r"^تحديد عنوان الباب",
        "chapter_title": r"^عنوان الباب:\s*(.+)$",
        "auth": r"^درجة صحة الحديث",
    },
}

def parse_gemini_text(raw_text: str):
    """Return a list of 400 predictions aligned with dataset ordering:
       Source en(50) -> Cloze en(50) -> Chapter en(50) -> Auth en(50) -> Source ar(50) -> Cloze ar(50) -> Chapter ar(50) -> Auth ar(50).
    """
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    preds = []

    def consume(k):
        """Pop first element from lines."""
        if not lines:
            return None
        return lines.pop(0)

    for _ in range(50):
        preds.append(consume("src_en"))

    while lines and not re.match(HEAD_MARKERS["en"]["cloze"], lines[0], flags=re.I):

        if len(preds) >= 50:  
            break
        
        break
    if lines and re.match(HEAD_MARKERS["en"]["cloze"], lines[0], flags=re.I):
        lines.pop(0)  # drop the header
    for _ in range(50):
        preds.append(consume("cloze_en"))


    en_chapter_titles = []
    for _ in range(50):
      
        if lines and re.match(r"^Book:\s*", lines[0]):
            lines.pop(0)
        title = ""
        if lines and re.match(HEAD_MARKERS["en"]["chapter_title"], lines[0]):
            m = re.match(HEAD_MARKERS["en"]["chapter_title"], lines[0])
            title = m.group(1) if m else lines[0]
            lines.pop(0)
        else:
            title = consume("chapter_title_en") or ""
        en_chapter_titles.append(title)
    preds.extend(en_chapter_titles)

    
    while lines and not re.match(HEAD_MARKERS["en"]["auth"], lines[0]):
        if lines[0].startswith("صحيح") or lines[0].startswith("إكمال") or lines[0].startswith("الكتاب:"):
            break
        lines.pop(0)
    if lines and re.match(HEAD_MARKERS["en"]["auth"], lines[0]):
        lines.pop(0)
    for _ in range(50):
        preds.append(consume("auth_en"))

    for _ in range(50):
        preds.append(consume("src_ar"))

    while lines and not re.match(HEAD_MARKERS["ar"]["cloze"], lines[0]):
        lines.pop(0)
    if lines:
        lines.pop(0)  
    for _ in range(50):
        preds.append(consume("cloze_ar"))

    while lines and not re.match(HEAD_MARKERS["ar"]["chapter"], lines[0]):
        lines.pop(0)
    if lines:
        lines.pop(0)  
    ar_chapter_titles = []
    for _ in range(50):
        if lines and lines[0].startswith("الكتاب:"):
            lines.pop(0)
        title = ""
        if lines and re.match(HEAD_MARKERS["ar"]["chapter_title"], lines[0]):
            m = re.match(HEAD_MARKERS["ar"]["chapter_title"], lines[0])
            title = m.group(1) if m else lines[0]
            lines.pop(0)
        else:
            title = consume("chapter_title_ar") or ""
        ar_chapter_titles.append(title)
    preds.extend(ar_chapter_titles)

    while lines and not re.match(HEAD_MARKERS["ar"]["auth"], lines[0]):
        lines.pop(0)
    if lines:
        lines.pop(0)  
    for _ in range(50):
        preds.append(consume("auth_ar"))

    if len(preds) != 400:
        print(f"WARNING: Parsed {len(preds)} predictions, expected 400.")
    return preds

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z*z/(2*n)
    adj = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)
    lower = (centre - adj) / denom
    upper = (centre + adj) / denom
    return max(0.0, lower)*100, min(1.0, upper)*100

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to lll.csv with columns [template, lang, prompt, choices, answer]")
    ap.add_argument("--preds", required=True, help="Path to raw Gemini output text (as pasted) OR CSV with a 'pred' column aligned to rows")
    ap.add_argument("--out", default="gemini_eval.csv", help="Where to write per-item evaluation CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.data)

    preds = None
    if args.preds.lower().endswith(".csv"):
        dfp = pd.read_csv(args.preds)
        cand_cols = [c for c in dfp.columns if c.lower() in ("pred","prediction","gemini_pred","answer","model_answer")]
        if not cand_cols:
            raise ValueError("CSV predictions must contain a 'pred' (or similar) column.")
        preds = dfp[cand_cols[0]].tolist()
    else:
        with open(args.preds, "r", encoding="utf-8") as f:
            raw = f.read()
        preds = parse_gemini_text(raw)

    if len(preds) != len(df):
        print(f"WARNING: predictions count ({len(preds)}) != dataset rows ({len(df)}). We'll align by min length.")
    n = min(len(preds), len(df))
    df = df.iloc[:n].copy()
    preds = preds[:n]

    cleaned_preds = []
    for i, row in df.iterrows():
        template = str(row["template"]).strip().lower()
        lang = str(row["lang"]).strip().lower()
        pred = preds[i]

        if template == "chapter":
       
            pass
        elif template == "authenticity":
            pred = normalize_authenticity_label(pred)

        cleaned_preds.append(pred)

    out_rows = []
    for i, row in df.iterrows():
        choices = row["choices"]
        gold = str(row["answer"])
        pred = cleaned_preds[i]
        chosen = closest_choice(pred, choices)
        correct = normalize_text(chosen) == normalize_text(gold)

        out_rows.append({
            "idx": i,
            "template": row["template"],
            "lang": row["lang"],
            "prompt": row["prompt"],
            "choices": choices,
            "gold_answer": gold,
            "gemini_pred_raw": preds[i],
            "gemini_mapped_choice": chosen,
            "is_correct": int(correct)
        })

    res = pd.DataFrame(out_rows)
    res.to_csv(args.out, index=False, encoding="utf-8-sig")

    print("\\n📊 Gemini Evaluation Report")
    print("=" * 60)
    overall = res["is_correct"].mean() * 100 if len(res) else 0.0
    lo, hi = wilson_ci(res["is_correct"].sum(), len(res))
    print(f"Overall accuracy: {overall:.2f}%  (95% CI: {lo:.2f}–{hi:.2f}; n={len(res)})")

    print("\\nBy template:")
    for t, grp in res.groupby("template"):
        acc = grp["is_correct"].mean()*100
        lo, hi = wilson_ci(grp["is_correct"].sum(), len(grp))
        print(f"  {t:<12} {acc:6.2f}%   (95% CI: {lo:.2f}–{hi:.2f}; n={len(grp)})")

    print("\\nBy language:")
    for l, grp in res.groupby("lang"):
        acc = grp["is_correct"].mean()*100
        lo, hi = wilson_ci(grp["is_correct"].sum(), len(grp))
        print(f"  {l:<3}          {acc:6.2f}%   (95% CI: {lo:.2f}–{hi:.2f}; n={len(grp)})")


    tpl_acc = res.groupby("template")["is_correct"].mean() * 100
    plt.figure()
    tpl_acc.plot(kind="bar", edgecolor="black")
    plt.title("Gemini Accuracy by Template")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Template")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("gemini_accuracy_by_template.png")
    plt.close()

    lang_acc = res.groupby("lang")["is_correct"].mean() * 100
    plt.figure()
    lang_acc.plot(kind="bar", edgecolor="black")
    plt.title("Gemini Accuracy by Language")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Language")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("gemini_accuracy_by_language.png")
    plt.close()

    print("\\nSaved:")
    print(f"  - Per-item CSV: {args.out}")
    print("  - Charts: gemini_accuracy_by_template.png, gemini_accuracy_by_language.png")

if __name__ == "__main__":
    main()
