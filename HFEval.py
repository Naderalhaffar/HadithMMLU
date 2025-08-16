import os, ast, argparse, difflib, re, unicodedata, json
import torch, pandas as pd
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Normalization ---
AR_DIAC = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u0653-\u065F\u06D6-\u06ED]')
def strip_ar(s): return AR_DIAC.sub('', s or '')
def norm(s):
    if not isinstance(s, str): return ''
    s = unicodedata.normalize('NFKC', s).replace('’',"'").replace('ʻ',"'").replace('`',"'").replace('–','-').replace('—','-')
    return strip_ar(s.strip()).lower()

def closest_choice(pred, choices):
    pn = norm(pred)
    for c in choices:
        if norm(c) == pn or pn in norm(c) or norm(c) in pn: return c
    m = {norm(c):c for c in choices}
    mm = difflib.get_close_matches(pn, list(m.keys()), n=1, cutoff=0.0)
    return m[mm[0]] if mm else choices[0]

# --- Instruction ---
CANONICAL_SIX = ["Sahih Bukhari","Sahih Muslim","Sunan Abi Da'ud","Jami' al-Tirmidhi","Sunan an-Nasa'i","Sunan Ibn Majah"]
def instruction(template, lang):
    t = (template or '').strip().lower()
    if t == 'source':
        return ("Task: Choose the correct hadith collection (one of the options). "
                f"Only the six canonical collections are valid: {', '.join(CANONICAL_SIX)}. "
                "المهمة: اختر اسم المجموعة الحديثية الصحيحة من الخيارات فقط.")
    if t == 'chapter':
        return ("Task: Choose the correct chapter title. "
                "المهمة: اختر عنوان الباب الصحيح.")
    if t == 'cloze':
        return ("Task: Fill the blank by choosing the exact missing word. "
                "المهمة: اختر الكلمة الناقصة الصحيحة.")
    if t == 'authenticity':
        return ("Task: Choose the hadith authenticity grade. If not a hadith, choose 'Not a Hadith'. "
                "المهمة: اختر درجة صحة الحديث، وإن لم يكن حديثًا فاختر 'ليس حديثاً'.")
    return "Task: Choose exactly one option. المهمة: اختر خيارًا واحدًا فقط."

# --- HF helpers ---
def pick_dtype(d):
    d = (d or "auto").lower()
    if d in ("bfloat16","bf16"): return torch.bfloat16
    if d in ("float16","fp16","half"): return torch.float16
    if d in ("float32","fp32"): return torch.float32
    return torch.float16 if torch.cuda.is_available() else torch.float32

def load_model(model_name, dtype, trust=False):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=pick_dtype(dtype),
                                                 device_map="auto" if torch.cuda.is_available() else None,
                                                 trust_remote_code=trust)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return model, tok

def render(tok, sys, usr, use_chat):
    if use_chat and hasattr(tok, "apply_chat_template"):
        msgs = [{"role":"system","content":sys},{"role":"user","content":usr}]
        try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except: pass
    return (sys + "\n\n" + usr).strip()

def gen_once(model, tok, prompt, max_new_tokens, temperature, top_p, seed=None):
    if seed is not None:
        try: torch.manual_seed(seed)
        except: pass
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
    if torch.cuda.is_available(): inp = {k:v.to(model.device) for k,v in inp.items()}
    out = model.generate(**inp, max_new_tokens=max_new_tokens,
                         do_sample=temperature>0, temperature=float(temperature) if temperature>0 else None,
                         top_p=float(top_p) if temperature>0 else None,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# --- Ask ---
def ask(model, tok, prompt, choices, template, lang, max_new_tokens=6, temperature=0.2, top_p=0.9, samples=1, use_chat=True):
    if isinstance(choices, str):
        try: choices = ast.literal_eval(choices)
        except: choices = [choices]
    choices = [str(c) for c in choices]; n = len(choices) or 1
    opts = "\n".join(f"{i+1}. {c}" for i,c in enumerate(choices))
    sys = f"You are an MCQ solver. Return ONLY the index number (1-{n}). Do not explain."
    usr = f"{instruction(template,lang)}\n\nQuestion:\n{prompt}\n\nOptions:\n{opts}\n\nReturn ONLY one number (1-{n})."
    votes, raws = Counter(), []
    for k in range(max(1,samples)):
        raw = gen_once(model, tok, render(tok, sys, usr, use_chat), max_new_tokens, temperature, top_p, seed=k)
        m = re.search(rf'([1-{n}])', raw)
        ans = choices[int(m.group(1))-1] if m else closest_choice(raw, choices)
        votes[norm(ans)] += 1; raws.append(raw)
    chosen = {norm(c):c for c in choices}[votes.most_common(1)[0][0]]
    return chosen, json.dumps({"samples":raws}, ensure_ascii=False)

# --- Main ---
def main():
    p = argparse.ArgumentParser(description="Evaluate Hadith MCQs with a local HF model")
    p.add_argument("--questions", required=True); p.add_argument("--out", required=True)
    p.add_argument("--model_name", required=True); p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument("--temperature", type=float, default=0.2); p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=6); p.add_argument("--samples", type=int, default=1)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--show_confusions", action="store_true"); p.add_argument("--show_mistakes", action="store_true")
    args = p.parse_args()

    model, tok = load_model(args.model_name, args.dtype, args.trust_remote_code)
    df = pd.read_csv(args.questions); rows = []

    for _, r in df.iterrows():
        template, lang = r.get("template",""), r.get("lang","")
        choice, raw = ask(model, tok, r["prompt"], r["choices"], template, lang,
                          args.max_new_tokens, args.temperature, args.top_p, args.samples, args.use_chat_template)
        gold = str(r["answer"]); correct = int(norm(choice) == norm(gold))
        rows.append({"template":template,"lang":lang,"prompt":r["prompt"],"choices":r["choices"],
                     "correct_answer":gold,"model_answer":choice,"raw_model_reply":raw,"is_correct":correct})

    res = pd.DataFrame(rows); res.to_csv(args.out, index=False, encoding="utf-8-sig")

    print("\n📊 Evaluation Report"); print("="*50)
    print(f"Overall Accuracy: {res['is_correct'].mean()*100:.2f}% ({res['is_correct'].sum()} / {len(res)})")
    print("\nAccuracy by Template:")
    for t,g in res.groupby("template"): print(f"  {t:<15}: {g['is_correct'].mean()*100:.2f}% ({g['is_correct'].sum()} / {len(g)})")
    print("\nAccuracy by Language:")
    for l,g in res.groupby("lang"): print(f"  {l:<5}: {g['is_correct'].mean()*100:.2f}% ({g['is_correct'].sum()} / {len(g)})")
    if args.show_confusions:
        print("\nTop confusions by template:")
        for t,g in res.groupby("template"):
            wrong = g[g["is_correct"]==0]
            if not len(wrong): continue
            tbl = wrong.groupby(["correct_answer","model_answer"]).size().sort_values(ascending=False).head(5)
            print(f"\n[{t}]"); print(tbl)
    if args.show_mistakes:
        print("\nSample mistakes:"); print(res[res["is_correct"]==0][["template","lang","correct_answer","model_answer"]].head(10).to_string(index=False))
    print("="*50); print(f"Detailed results saved to: {args.out}\n")

if __name__ == "__main__":
    main()
