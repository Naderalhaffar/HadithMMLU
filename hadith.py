import os
import sys
import re
import json
import random
import string
import argparse

import pandas as pd
import openai
from sklearn.feature_extraction.text import TfidfVectorizer

openai.api_key = os.getenv("OPENAI_API_KEY")

# Arabic mappings for the six canonical sources
SRC_AR_MAP = {
    'Sahih Bukhari': 'صحيح البخاري',
    'Sahih Muslim': 'صحيح مسلم',
    "Sunan Abi Da'ud": 'سنن أبي داود',
    "Jami' al-Tirmidhi": 'جامع الترمذي',
    "Sunan an-Nasa'i": 'سنن النسائي',
    "Sunan Ibn Majah": 'سنن ابن ماجه',
}
IMPOSTER_SOURCE_AR = 'معجم الحديث الكبير'
ARABIC_WORD_RE = re.compile(r'[\u0600-\u06FF]+')

# Arabic labels for authenticity grades
GRADE_AR_MAP = {
    'Sahih': 'صحيح',
    'Hasan': 'حسن',
    'Daif':  'ضعيف'
}
NOT_HADITH_AR = 'ليس حديثاً'

SOURCE_ALIASES = {
    # Bukhari
    'bukhari': 'Sahih Bukhari', 'sahih bukhari': 'Sahih Bukhari',
    'al bukhari': 'Sahih Bukhari', 'al-bukhari': 'Sahih Bukhari',
    'البخاري': 'Sahih Bukhari', 'صحيح البخاري': 'Sahih Bukhari',
    # Muslim
    'muslim': 'Sahih Muslim', 'sahih muslim': 'Sahih Muslim',
    'مسلم': 'Sahih Muslim', 'صحيح مسلم': 'Sahih Muslim',
    # Abu Dawud
    'abu dawud': "Sunan Abi Da'ud", 'abi daud': "Sunan Abi Da'ud",
    'abu dawood': "Sunan Abi Da'ud", 'abo dawod': "Sunan Abi Da'ud",
    'أبي داود': "Sunan Abi Da'ud", 'سنن أبي داود': "Sunan Abi Da'ud", 'sunan abi dawud': "Sunan Abi Da'ud",
    # Tirmidhi
    'tirmidhi': "Jami' al-Tirmidhi",
    'al tirmidhi': "Jami' al-Tirmidhi",
    'al-tirmidhi': "Jami' al-Tirmidhi",
    'جامع الترمذي': "Jami' al-Tirmidhi",
    'الترمذي': "Jami' al-Tirmidhi",
    'jami` at-tirmidhi': "Jami' al-Tirmidhi",
    'jami at tirmidhi': "Jami' al-Tirmidhi",
    "jami' at-tirmidhi": "Jami' al-Tirmidhi",

    # Nasa'i
    'nasai': "Sunan an-Nasa'i", 'an nasai': "Sunan an-Nasa'i",
    'an-nasai': "Sunan an-Nasa'i", 'النسائي': "Sunan an-Nasa'i",
    'سنن النسائي': "Sunan an-Nasa'i",
    # Ibn Majah
    'ibn majah': "Sunan Ibn Majah", 'ibn maja': "Sunan Ibn Majah",
    'ابن ماجه': "Sunan Ibn Majah", 'سنن ابن ماجه': "Sunan Ibn Majah",
}

def _norm_source_key(s: str) -> str:
    if not isinstance(s, str): return ''
    s = s.strip().lower()
    s = s.replace('’', "'").replace('ʻ', "'").replace('`', "'")
    s = s.replace('–','-').replace('—','-')
    s = ' '.join(s.split())
    return s

def source_to_ar(value: str) -> str:
    """Map a collection/book name to Arabic, handling common variants."""
    if not isinstance(value, str) or not value.strip():
        return 'هذا الكتاب'
    raw = value.strip()
    # already Arabic?
    if raw in SRC_AR_MAP.values():
        return raw
    # canonical English -> Arabic
    if raw in SRC_AR_MAP:
        return SRC_AR_MAP[raw]
    # alias -> canonical -> Arabic
    alias = SOURCE_ALIASES.get(_norm_source_key(raw))
    if alias and alias in SRC_AR_MAP:
        return SRC_AR_MAP[alias]
    # fallback: show what we got
    return raw



def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate hadith MCQs: Source, Matn Cloze (via GPT), Chapter, and Authenticity'
    )
    parser.add_argument('-i','--input',   default=os.path.expanduser('~/Desktop/hadithdatasetcsv.csv'),
                        help='Input CSV (must have text_en, text_ar, source, chapter, hadith_no)')
    parser.add_argument('-o','--output',  default=os.path.expanduser('~/Desktop/generated_hadith_questions.csv'),
                        help='Output CSV of generated questions')
    parser.add_argument('-n','--num',     type=int, default=50,
                        help='Questions per template (default 50)')
    parser.add_argument('-l','--lang',    choices=['en','ar','both'], default='en',
                        help='Language of questions: en, ar, or both')
    parser.add_argument('--auth',         default=os.path.expanduser('~/Desktop/hadith_authenticity.csv'),
                        help='CSV with hadith authenticity grades (cols: text_en, text_ar, consensus_grade)')
    parser.add_argument('--proverbs',     default=os.path.expanduser('~/Desktop/arabicproverbs.csv'),
                        help='CSV of Arabic proverbs (cols: Column4=en, Column1=ar)')
    return parser.parse_args()

def normalize_text(s: str) -> str:
    s = s.lower()
    s = s.replace('’', "'").replace('—', '-').replace('–', '-')
    s = s.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', s).strip()

def load_csv(path):
    p = os.path.expanduser(path)
    if not os.path.exists(p):
        print(f"Error: file not found: {p}", file=sys.stderr); sys.exit(1)
    df = pd.read_csv(p, dtype=str)
    required = ['text_en','text_ar','source','chapter','hadith_no']
    for col in required:
        if col not in df.columns:
            print(f"Error: missing required column '{col}'", file=sys.stderr); sys.exit(1)
    df = df.dropna(subset=required)
    df['text_en'] = df['text_en'].str.strip()
    df['text_ar'] = df['text_ar'].str.strip()
    df = df[df['text_en'].astype(bool)].reset_index(drop=True)
    df['text_en_norm'] = df['text_en'].apply(normalize_text)
    return df

def load_auth_csv(path):
    p = os.path.expanduser(path)
    if not os.path.exists(p):
        print(f"Error: file not found: {p}", file=sys.stderr); sys.exit(1)
    df = pd.read_csv(p, dtype=str)
    return df.dropna(subset=['consensus_grade','text_en','text_ar'])

def load_proverbs_csv(path):
    p = os.path.expanduser(path)
    if not os.path.exists(p):
        print(f"Error: file not found: {p}", file=sys.stderr); sys.exit(1)
    df = pd.read_csv(p, dtype=str)
    df = df[['Column4','Column1']].dropna()
    return df.rename(columns={'Column4':'text_en','Column1':'text_ar'})

def make_mcq(correct, distractors):
    opts = list(distractors)
    if correct not in opts:
        opts.append(correct)
    random.shuffle(opts)
    return opts

def get_similar_words(term: str, n: int = 3, language: str = "English"):
    system = (
        "You are a helpful assistant that provides single-word synonyms or closely related "
        "words suitable as distractors in a multiple-choice question. "
        "Do NOT repeat the original word."
    )
    user = (
        f"Give me {n} {'Arabic' if language=='Arabic' else 'English'} synonyms or similar single words "
        f"for the word “{term}”. "
        "Respond with a JSON array of strings only, e.g. [\"...\", \"...\", \"...\"]."
    )
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": user}
        ],
        temperature=0.7,
        max_tokens=100
    )
    text = resp.choices[0].message.content.strip()
    try:
        words = json.loads(text)
        return [w.strip() for w in words if isinstance(w, str)][:n]
    except json.JSONDecodeError:
        found = re.findall(r'"([^"]+)"', text)
        return found[:n]


CANONICAL_EN = list(SRC_AR_MAP.keys()) + ['Mu‘jam al-Hadīth al-Kubrā']

def gen_source_en(df, n):
    qs = []
    sample = df.sample(min(n, len(df)), random_state=1)
    for _,r in sample.iterrows():
        txt, src = r['text_en'], r['source'].strip()
        others = [c for c in CANONICAL_EN if c != src]
        distractors = random.sample(others, 2) + ['Mu‘jam al-Hadīth al-Kubrā']
        qs.append({
            'template':'Source','lang':'en',
            'prompt': txt,
            'choices': make_mcq(src, distractors),
            'answer': src
        })
    return qs

def gen_cloze_en(df, n):
    vectorizer_en = TfidfVectorizer(token_pattern=r"\w+'?\w*", stop_words='english')
    vectorizer_en.fit(df['text_en'])

    qs = []
    sample = df.sample(min(n, len(df)), random_state=2).reset_index(drop=True)
    stops = set('the and a to of in for on is with that as his her by from'.split())

    for _, r in sample.iterrows():
        text = r['text_en']
        snippet = re.search(r'"([^"]+)"', text)
        snippet = snippet.group(1) if snippet else text

        words = re.findall(r"\w+'?\w*", snippet)
        candidates = [w for w in words
                      if w.lower() not in stops
                      and w.lower() in vectorizer_en.vocabulary_]

        if not candidates:
            continue

        idf_vals = {
            w: vectorizer_en.idf_[vectorizer_en.vocabulary_[w.lower()]]
            for w in candidates
        }
        gap = max(idf_vals, key=idf_vals.get)

        distractors = get_similar_words(gap, 3, language="English")
        if len(distractors) < 3:
            fallback = random.sample(
                [w for w in vectorizer_en.get_feature_names_out()
                 if w.lower() != gap.lower()],
                3 - len(distractors)
            )
            distractors += fallback

        prompt = re.sub(rf"\b{re.escape(gap)}\b", '____', text, count=1)
        qs.append({
            'template': 'Cloze', 'lang': 'en',
            'prompt': prompt,
            'choices': make_mcq(gap, distractors),
            'answer': gap
        })

    return qs


def gen_chapter_en(df, n):
    qs = []
    all_chaps = df['chapter'].unique().tolist()
    sample = df.sample(min(n, len(df)), random_state=3)
    for _,r in sample.iterrows():
        txt, chap = r['text_en'], r['chapter']
        wrongs = random.sample([c for c in all_chaps if c != chap], min(3, len(all_chaps)-1))
        prompt = f"Identify the chapter title under which this hadith appears in {r['source']}:\n\n{txt}"
        qs.append({
            'template':'Chapter','lang':'en',
            'prompt': prompt,
            'choices': make_mcq(chap, wrongs),
            'answer': chap
        })
    return qs

def gen_authenticity_en(auth_df, prov_df, n):
    qs = []
    half = n // 2
    has_collection = 'collection' in auth_df.columns

    coll = r['collection'] if has_collection and isinstance(r['collection'], str) and r['collection'].strip() else 'this collection'
    prompt = f"What is the authenticity grade of this hadith from {coll}?\n\n{snippet}"



    def strip_n(txt): return re.sub(r'^[^,]+,\s*','', txt or '').strip()
    auth_df['core_en'] = auth_df['text_en'].apply(strip_n)
    auth_df['wc_en'] = auth_df['core_en'].str.split().apply(len)
    good = auth_df[auth_df['wc_en'] >= 10]

    for _, r in good.sample(min(half, len(good))).iterrows():
         snippet = ' '.join(r['core_en'].split()[:25])
         coll = r['source'] if has_collection else 'this collection'
         prompt = f"What is the authenticity grade of this hadith from {coll}?\n\n{snippet}"
         qs.append({
             'template':'Authenticity','lang':'en',
             'prompt': prompt,
             'choices': ['Sahih','Hasan','Daif','Not a Hadith'],
             'answer': r['consensus_grade']
         })

    collections = [c for c in CANONICAL_EN]
    prov_df['wc_en'] = prov_df['text_en'].str.split().apply(len)
    pool = prov_df[prov_df['wc_en'] >= 10]
    if len(pool) < (n-half): pool = prov_df

    for _, r in pool.sample(n-half, random_state=6).iterrows():
        snippet = ' '.join(r['text_en'].split()[:25])
        fake_coll = random.choice(collections)
        prompt = f"What is the authenticity grade of this hadith from {fake_coll}?\n\n{snippet}"
        qs.append({
            'template':'Authenticity','lang':'en',
            'prompt': prompt,
            'choices': ['Sahih','Hasan','Daif','Not a Hadith'],
            'answer': 'Not a Hadith'
        })

    random.shuffle(qs)
    return qs


def gen_source_ar(df, n):
    qs = []
    valid = df['source'].str.strip().isin(SRC_AR_MAP)
    sample = df[valid].sample(min(n, valid.sum()), random_state=1)
    for _,r in sample.iterrows():
        txt = r['text_ar']
        src_ar = SRC_AR_MAP[r['source'].strip()]
        others = [SRC_AR_MAP[c] for c in SRC_AR_MAP if c != r['source'].strip()]
        distractors = random.sample(others, 2) + [IMPOSTER_SOURCE_AR]
        prompt = "من أي من الكتب الستة التالية ورد هذا الحديث؟\n\n" + txt
        qs.append({
            'template':'Source','lang':'ar',
            'prompt': prompt,
            'choices': make_mcq(src_ar, distractors),
            'answer': src_ar
        })
    return qs

def gen_cloze_ar(df, n):
    vectorizer_ar = TfidfVectorizer(
        token_pattern=ARABIC_WORD_RE.pattern,
        stop_words=['في','من','على','إلى','عن','و','أن','إن','الذي','التي','هذا','هذه','هو','هي']
    )
    vectorizer_ar.fit(df['text_ar'])

    qs = []
    sample = df.sample(min(n, len(df)), random_state=2).reset_index(drop=True)
    stops = set(['في','من','على','إلى','عن','و','أن','إن','الذي','التي','هذا','هذه','هو','هي'])

    for _, r in sample.iterrows():
        text = r['text_ar']
        snippet = re.search(r'"([^"]+)"', text)
        snippet = snippet.group(1) if snippet else text

        words = ARABIC_WORD_RE.findall(snippet)
        candidates = [w for w in words if w not in stops and w in vectorizer_ar.vocabulary_]

        if not candidates:
            continue

        idf_vals = {
            w: vectorizer_ar.idf_[vectorizer_ar.vocabulary_[w]]
            for w in candidates
        }
        gap = max(idf_vals, key=idf_vals.get)

        distractors = get_similar_words(gap, 3, language="Arabic")
        if len(distractors) < 3:
            fallback = random.sample(
                [w for w in vectorizer_ar.get_feature_names_out() if w != gap],
                3 - len(distractors)
            )
            distractors += fallback

        prompt = "استبدل (____) في الحديث الشريف بالكلمة الصحيحة الناقصة:\n\n" + re.sub(re.escape(gap), '____', text, count=1)

        qs.append({
            'template': 'Cloze', 'lang': 'ar',
            'prompt': prompt,
            'choices': make_mcq(gap, distractors),
            'answer': gap
        })

    return qs


def gen_chapter_ar(df, n):
    qs = []
    all_ar = []
    for chap in df['chapter'].unique():
        if '-' in chap:
            all_ar.append(chap.split('-',1)[1].strip())
    all_ar = list(dict.fromkeys(all_ar))
    sample = df.sample(min(n, len(df)), random_state=3)
    for _,r in sample.iterrows():
        parts = r['chapter'].split('-',1)
        if len(parts) < 2: continue
        chap_ar = parts[1].strip()
        wrongs = random.sample([c for c in all_ar if c != chap_ar], min(3, len(all_ar)-1))
        book_name = SRC_AR_MAP.get(r['source'].strip(), r['source'].strip())
        prompt = f"تحت أي باب في {book_name} ورد هذا الحديث؟\n\n" + r['text_ar']
        qs.append({
            'template':'Chapter','lang':'ar',
            'prompt': prompt,
            'choices': make_mcq(chap_ar, wrongs),
            'answer': chap_ar
        })
    return qs

def gen_authenticity_ar(auth_df, prov_df, n):
    qs = []
    half = n // 2

    def strip_narrator_ar(text):
        core = text or ''
        core = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', core)
        while '،' in core:
            prefix, rest = core.split('،',1)
            if re.match(r'^(?:حدثنا|أخبرنا|عن|روى)', prefix.strip()):
                core = rest.strip()
            else:
                break
        return core

    def normalize_grade_ar(g):
        if not isinstance(g, str): return None
        g0 = g.strip().lower()
        if 'صح' in g0 or 'ṣaḥ' in g0: return 'Sahih'
        if 'حسن' in g0 or 'ḥas' in g0 or 'hasan' in g0 or g0 == 'has': return 'Hasan'
        if 'ضعيف' in g0 or 'ضعف' in g0 or 'daʿīf' in g0 or 'daif' in g0: return 'Daif'
        return None

    auth_df['core_ar'] = auth_df['text_ar'].apply(strip_narrator_ar)
    auth_df['wc_ar'] = auth_df['core_ar'].str.split().apply(len)
    auth_df['grade_norm'] = auth_df['consensus_grade'].apply(normalize_grade_ar)
    good = auth_df[auth_df['grade_norm'].notnull() & (auth_df['wc_ar'] >= 10)]

    ar_collections = list(SRC_AR_MAP.values()) + [IMPOSTER_SOURCE_AR]
    has_collection = 'collection' in auth_df.columns


    for _, r in good.sample(min(half, len(good))).iterrows():
        snippet = ' '.join(r['core_ar'].split()[:25])
        if has_collection and isinstance(r['collection'], str) and r['collection'].strip():
            coll_ar = source_to_ar(r['collection'])
        else:
            coll_ar = 'هذا الكتاب'
        prompt = f"ما الدرجة الغالبة لهذا الحديث في {coll_ar}؟\n\n{snippet}"

        ar_grade = GRADE_AR_MAP[r['grade_norm']]
        qs.append({
            'template': 'Authenticity','lang':'ar',
            'prompt': prompt,
            'choices': [GRADE_AR_MAP['Sahih'], GRADE_AR_MAP['Hasan'], GRADE_AR_MAP['Daif'], NOT_HADITH_AR],
            'answer': ar_grade
        })

    prov_df['wc_ar'] = prov_df['text_ar'].str.split().apply(len)
    pool = prov_df[prov_df['wc_ar'] >= 10]
    if len(pool) < (n - half): pool = prov_df

    for _, r in pool.sample(n - half).iterrows():
        snippet = ' '.join(r['text_ar'].split()[:25])
        fake_coll_ar = random.choice(ar_collections)
        prompt = f"ما درجة صحة هذا الحديث في {fake_coll_ar}?\n\n{snippet}"
        qs.append({
            'template': 'Authenticity','lang':'ar',
            'prompt': prompt,
            'choices': [GRADE_AR_MAP['Sahih'], GRADE_AR_MAP['Hasan'], GRADE_AR_MAP['Daif'], NOT_HADITH_AR],
            'answer': NOT_HADITH_AR
        })

    random.shuffle(qs)
    return qs


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    df = load_csv(args.input)
    auth_df = load_auth_csv(args.auth)
    prov_df = load_proverbs_csv(args.proverbs)

    # filter out duplicates in source, chapter, hadith_no
    mask = df.groupby('text_en_norm')['source'].transform('nunique') == 1
    chap_mask = df.groupby('text_en_norm')['chapter'].transform('nunique') == 1
    num_mask = df.groupby('text_en_norm')['hadith_no'].transform('nunique') == 1
    df = df[mask & chap_mask & num_mask].reset_index(drop=True)

    all_q = []
    if args.lang in ('en','both'):
        all_q += gen_source_en(df, args.num)
        all_q += gen_cloze_en(df, args.num)
        all_q += gen_chapter_en(df, args.num)
        all_q += gen_authenticity_en(auth_df, prov_df, args.num)

    if args.lang in ('ar','both'):
        all_q += gen_source_ar(df, args.num)
        all_q += gen_cloze_ar(df, args.num)
        all_q += gen_chapter_ar(df, args.num)
        all_q += gen_authenticity_ar(auth_df, prov_df, args.num)

    out_path = os.path.expanduser(args.output)
    output_dir = os.path.dirname(out_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(all_q).to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f"Generated {len(all_q)} questions ({args.lang}) and saved to {out_path}")
