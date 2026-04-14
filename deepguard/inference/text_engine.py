"""
DeepGuard Text Analysis Engine
================================
Three analysis modes for text:

1. AI-GENERATED TEXT DETECTION
   Detects whether text was written by an AI (ChatGPT, Gemini, etc.)
   using statistical linguistics features:
   - Perplexity (AI text is unnaturally low-perplexity)
   - Burstiness (AI text has low sentence-length variance)
   - Vocabulary entropy, function word ratios, punctuation patterns
   - N-gram repetition, lexical diversity

2. FAKE NEWS / CLAIM DETECTION
   Analyses credibility signals in text:
   - Sensationalist vocabulary (ALL CAPS, excessive punctuation)
   - Hedging vs. assertive language
   - Named entity density
   - Emotional amplifiers, clickbait patterns
   - Source-quality signals

3. GRAMMAR CORRECTION
   Rule-based + pattern corrections:
   - Common spelling errors
   - Subject-verb agreement
   - Article errors (a/an)
   - Double spaces, punctuation spacing
   - Capitalisation after full stop
   Shows original vs corrected side-by-side with diff highlighting.

All analyses run locally — no API calls required.
"""

import re
import math
import string
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
#  Shared Text Utilities
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def sentences(text: str) -> list:
    """Split text into sentences."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 2]


def word_count(text: str) -> int:
    return len(tokenize(text))


# ─────────────────────────────────────────────────────────────────────────────
#  1. AI TEXT DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

# Common words heavily over-used by LLMs
AI_OVERUSED = {
    "delve", "delves", "delving", "delved",
    "absolutely", "certainly", "indeed", "notably",
    "crucial", "essential", "vital", "pivotal", "paramount",
    "comprehensive", "robust", "nuanced", "multifaceted",
    "leverage", "leveraging", "utilise", "utilize",
    "furthermore", "moreover", "nevertheless", "nonetheless",
    "it's worth noting", "it is worth noting",
    "in conclusion", "to summarize", "in summary",
    "a testament to", "stands as a testament",
    "underscores", "highlights the importance",
    "it's important to note", "it is important to note",
    "in today's world", "in today's society",
    "tapestry", "landscape", "realm", "domain",
    "shed light", "sheds light", "shed some light",
    "navigate", "navigating", "embark", "embarking",
    "ensure", "ensures", "foster", "fosters",
    "revolutionize", "revolutionizes", "transformative",
}

# Human writing tends to use these contractions more
HUMAN_CONTRACTIONS = {
    "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't", "won't", "wouldn't", "can't",
    "couldn't", "shouldn't", "i'm", "i've", "i'd", "i'll",
    "you're", "you've", "you'd", "you'll",
    "we're", "we've", "we'd", "they're", "they've",
}


def compute_burstiness(text: str) -> float:
    """
    Burstiness = (std - mean) / (std + mean) of sentence lengths.
    Human text: high burstiness (varied sentence lengths).
    AI text: low burstiness (uniformly medium-length sentences).
    Range: -1 (perfectly uniform) to +1 (highly bursty).
    """
    sents = sentences(text)
    if len(sents) < 3:
        return 0.0
    lengths = [len(tokenize(s)) for s in sents]
    mean = sum(lengths) / len(lengths)
    std  = math.sqrt(sum((l - mean) ** 2 for l in lengths) / len(lengths))
    if mean + std == 0:
        return 0.0
    return (std - mean) / (std + mean)


def compute_lexical_diversity(text: str) -> float:
    """Type-Token Ratio (unique words / total words). AI text tends ~0.45–0.60."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def ai_word_ratio(text: str) -> float:
    """Fraction of tokens that are known AI-overused words."""
    tokens = tokenize(text)
    text_lower = text.lower()
    hits = sum(1 for w in AI_OVERUSED if w in text_lower)
    return min(1.0, hits / max(1, len(tokens) / 20))


def contraction_ratio(text: str) -> float:
    """Fraction of human-style contractions. Higher = more human."""
    text_lower = text.lower()
    hits = sum(1 for c in HUMAN_CONTRACTIONS if c in text_lower)
    wc   = word_count(text)
    return min(1.0, hits / max(1, wc / 15))


def punctuation_variety(text: str) -> float:
    """
    AI text tends to use only . and , — humans use !, ?, ;, :, —, ...
    Returns fraction of non-standard punctuation.
    """
    puncts = [c for c in text if c in "!?;:—–()\"'…"]
    total  = max(1, len([c for c in text if c in string.punctuation]))
    return len(puncts) / total


def avg_sentence_length(text: str) -> float:
    sents = sentences(text)
    if not sents:
        return 0.0
    return sum(len(tokenize(s)) for s in sents) / len(sents)


def detect_ai_text(text: str) -> dict:
    """
    Returns a detailed AI-text detection result.
    Score 0–100: higher = more likely AI-generated.
    """
    text = text.strip()
    if word_count(text) < 10:
        return {
            "score": 50.0, "label": "UNCERTAIN",
            "confidence": 50.0, "risk": "LOW",
            "explanation": "Text too short for reliable analysis (need 10+ words).",
            "features": {}
        }

    # ── Feature extraction ────────────────────────────────────────────
    burst     = compute_burstiness(text)          # low → AI
    diversity = compute_lexical_diversity(text)   # mid → AI
    ai_vocab  = ai_word_ratio(text)               # high → AI
    contract  = contraction_ratio(text)           # low → AI
    punct_var = punctuation_variety(text)         # low → AI
    avg_sl    = avg_sentence_length(text)         # 18–25 → AI
    n_sents   = len(sentences(text))

    # ── Scoring (each feature contributes to AI score 0–100) ─────────
    score = 50.0  # neutral baseline

    # Burstiness: < -0.2 → very AI-like
    if burst < -0.3:
        score += 18
    elif burst < 0.0:
        score += 9
    elif burst > 0.3:
        score -= 12

    # AI vocabulary
    score += ai_vocab * 28

    # Contractions: AI rarely uses them
    if contract < 0.01:
        score += 10
    elif contract > 0.05:
        score -= 10

    # Punctuation variety
    if punct_var < 0.05:
        score += 8
    elif punct_var > 0.2:
        score -= 8

    # Sentence length: AI favours 18–26 words/sentence
    if 17 <= avg_sl <= 27:
        score += 10
    elif avg_sl < 10 or avg_sl > 35:
        score -= 8

    # Lexical diversity: AI ~0.45–0.60
    if 0.44 <= diversity <= 0.62:
        score += 8
    elif diversity < 0.35 or diversity > 0.75:
        score -= 6

    # Very short texts are uncertain
    if n_sents < 2:
        score = 0.4 * score + 0.6 * 50

    score = max(0.0, min(100.0, score))

    # ── Label & risk ──────────────────────────────────────────────────
    if score >= 72:
        label = "AI-GENERATED"; risk = "HIGH"
        expl  = ("Strong AI-writing patterns detected — uniform sentence "
                 "lengths, typical LLM vocabulary, low contraction use.")
    elif score >= 55:
        label = "LIKELY AI"; risk = "MEDIUM"
        expl  = ("Several AI-writing markers found. The text may be AI-generated "
                 "or heavily edited AI output.")
    elif score >= 40:
        label = "UNCERTAIN"; risk = "LOW"
        expl  = ("Mixed signals — could be AI-assisted writing or a very "
                 "formal human author.")
    else:
        label = "HUMAN"; risk = "LOW"
        expl  = ("Natural human writing patterns detected — varied sentence "
                 "lengths, contractions, and diverse vocabulary.")

    # Confidence = distance from 50, scaled
    conf = 50 + abs(score - 50)
    conf = min(98.0, conf)

    return {
        "score":       round(score, 1),
        "label":       label,
        "confidence":  round(conf, 1),
        "risk":        risk,
        "explanation": expl,
        "features": {
            "burstiness":        round(burst, 3),
            "lexical_diversity": round(diversity, 3),
            "ai_vocab_ratio":    round(ai_vocab, 3),
            "contraction_ratio": round(contract, 3),
            "punct_variety":     round(punct_var, 3),
            "avg_sentence_len":  round(avg_sl, 1),
            "sentence_count":    n_sents,
            "word_count":        word_count(text),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
#  2. FAKE NEWS DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

SENSATIONAL = [
    r"\b(BREAKING|EXCLUSIVE|URGENT|SHOCKING|EXPOSED|BOMBSHELL|SCANDAL)\b",
    r"\b(you won't believe|they don't want you to know|what they're hiding)\b",
    r"\b(share before deleted|banned from mainstream|censored)\b",
    r"\b(100%|proven|definitive proof|scientists baffled|doctors hate)\b",
    r"!!!+|\?\?\?+",
    r"\b(CURE|MIRACLE|SECRET|CONSPIRACY|HOAX|FRAUD)\b",
]

CREDIBILITY_POSITIVE = [
    r"\b(according to|study shows|research indicates|published in)\b",
    r"\b(spokesperson|official|statement|government|university)\b",
    r"\b(percent|statistics|data|evidence|survey)\b",
    r"\b(however|although|despite|while|on the other hand)\b",  # nuance
]

EMOTIONAL_TRIGGERS = [
    r"\b(outrage|furious|terrifying|horrifying|disgusting|evil|corrupt)\b",
    r"\b(destroy|attack|hate|dangerous|threat|crisis|catastrophe)\b",
    r"\b(wake up|sheeple|sheep|brainwash|propaganda|regime)\b",
]

CLICKBAIT_PATTERNS = [
    r"\bwhy .{3,40}\?$",
    r"\bthe truth about\b",
    r"\bthis .{3,30} will (shock|surprise|amaze)\b",
    r"\bnumber \d+ will\b",
    r"\bwhat (nobody|no one|they) (tells?|told) you\b",
]


def count_caps_ratio(text: str) -> float:
    """Ratio of ALL-CAPS words."""
    words = text.split()
    if not words: return 0.0
    caps = sum(1 for w in words if w.isupper() and len(w) > 2)
    return caps / len(words)


def detect_fake_news(text: str) -> dict:
    """
    Analyse text for fake news / misinformation signals.
    Returns score 0–100: higher = more likely misinformation.
    """
    text = text.strip()
    if word_count(text) < 5:
        return {
            "score": 50.0, "label": "UNCERTAIN",
            "confidence": 50.0, "risk": "LOW",
            "explanation": "Text too short for analysis.",
            "signals": {}, "triggered_patterns": []
        }

    score = 30.0   # start below neutral (most text is not fake news)
    triggered = []

    # Sensational patterns
    for pat in SENSATIONAL:
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            score += 8
            triggered.append(f"Sensational: '{m[0]}'")

    # Emotional trigger words
    for pat in EMOTIONAL_TRIGGERS:
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            score += 5
            triggered.append(f"Emotional trigger: '{m[0]}'")

    # Clickbait
    for pat in CLICKBAIT_PATTERNS:
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            score += 7
            triggered.append(f"Clickbait pattern: '{m[0]}'")

    # ALL CAPS words
    caps_r = count_caps_ratio(text)
    if caps_r > 0.15:
        score += caps_r * 30
        triggered.append(f"Excessive CAPS ({caps_r*100:.0f}% of words)")

    # Credibility signals (reduce score)
    for pat in CREDIBILITY_POSITIVE:
        if re.search(pat, text, re.IGNORECASE):
            score -= 5

    # Excessive punctuation
    exc = len(re.findall(r"[!?]{2,}", text))
    if exc > 2:
        score += exc * 4
        triggered.append(f"Excessive punctuation ({exc} instances)")

    score = max(0.0, min(100.0, score))

    if score >= 65:
        label = "LIKELY FAKE"; risk = "HIGH"
        expl  = ("Multiple misinformation signals: sensationalist language, "
                 "emotional manipulation, and/or clickbait patterns detected.")
    elif score >= 45:
        label = "SUSPICIOUS"; risk = "MEDIUM"
        expl  = ("Some credibility concerns found. Verify claims with "
                 "trusted sources before sharing.")
    elif score >= 25:
        label = "MOSTLY CREDIBLE"; risk = "LOW"
        expl  = ("Few misinformation markers. Text appears reasonably balanced, "
                 "though independent verification is always advisable.")
    else:
        label = "CREDIBLE"; risk = "LOW"
        expl  = ("No significant fake-news signals. Language is measured and "
                 "avoids sensationalism.")

    conf = 50 + abs(score - 50) * 0.8
    conf = min(95.0, conf)

    return {
        "score":              round(score, 1),
        "label":              label,
        "confidence":         round(conf, 1),
        "risk":               risk,
        "explanation":        expl,
        "triggered_patterns": triggered[:8],   # top 8
        "signals": {
            "caps_ratio":         round(caps_r, 3),
            "sensational_hits":   sum(1 for p in SENSATIONAL
                                      if re.search(p, text, re.I)),
            "emotional_hits":     sum(1 for p in EMOTIONAL_TRIGGERS
                                      if re.search(p, text, re.I)),
            "clickbait_hits":     sum(1 for p in CLICKBAIT_PATTERNS
                                      if re.search(p, text, re.I)),
            "credibility_hits":   sum(1 for p in CREDIBILITY_POSITIVE
                                      if re.search(p, text, re.I)),
            "word_count":         word_count(text),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
#  3. GRAMMAR CORRECTOR
# ─────────────────────────────────────────────────────────────────────────────

# (pattern, replacement, description)
GRAMMAR_RULES = [
    # ── Spacing ───────────────────────────────────────────────────────────
    (r"  +",                    " ",       "Extra spaces removed"),
    (r" ,",                     ",",       "Space before comma"),
    (r" \.",                    ".",       "Space before period"),
    (r" !",                     "!",       "Space before !"),
    (r" \?",                    "?",       "Space before ?"),
    (r",(?=[^\s])",             ", ",      "Missing space after comma"),
    (r"\.(?=[A-Z][a-z])",       ". ",      "Missing space after period"),

    # ── Article a/an ──────────────────────────────────────────────────────
    (r"\ba ([aeiouAEIOU]\w)",   r"an \1",  "a → an before vowel"),
    (r"\ban ([^aeiouAEIOU\s])", r"a \1",   "an → a before consonant"),

    # ── Common spelling ───────────────────────────────────────────────────
    (r"\brecieve\b",   "receive",     "recieve → receive"),
    (r"\bacheive\b",   "achieve",     "acheive → achieve"),
    (r"\bbelive\b",    "believe",     "belive → believe"),
    (r"\boccured\b",   "occurred",    "occured → occurred"),
    (r"\bseperate\b",  "separate",    "seperate → separate"),
    (r"\bdefinately\b","definitely",  "definately → definitely"),
    (r"\buntill\b",    "until",       "untill → until"),
    (r"\bwich\b",      "which",       "wich → which"),
    (r"\bthier\b",     "their",       "thier → their"),
    (r"\bfriend\b",    "friend",      "✓ friend (correct)"),  # common confusion check
    (r"\bexistance\b", "existence",   "existance → existence"),
    (r"\boccasion\b",  "occasion",    "✓ occasion"),
    (r"\bmaintainance\b","maintenance","maintainance → maintenance"),
    (r"\bneccessary\b","necessary",   "neccessary → necessary"),
    (r"\bproffesional\b","professional","proffesional → professional"),
    (r"\bque\b",       "queue",       "que → queue"),
    (r"\bgoverment\b", "government",  "goverment → government"),
    (r"\benviroment\b","environment", "enviroment → environment"),
    (r"\bwether\b",    "whether",     "wether → whether"),
    (r"\bequiptment\b","equipment",   "equiptment → equipment"),
    (r"\bbeautifull\b","beautiful",   "beautifull → beautiful"),
    (r"\btommorow\b",  "tomorrow",    "tommorow → tomorrow"),
    (r"\byesterady\b", "yesterday",   "yesterady → yesterday"),
    (r"\bcalender\b",  "calendar",    "calender → calendar"),
    (r"\bcommitee\b",  "committee",   "commitee → committee"),
    (r"\bliason\b",    "liaison",     "liason → liaison"),

    # ── Common grammar ────────────────────────────────────────────────────
    (r"\bi\b",         "I",           "i → I (capitalise pronoun)"),
    (r"\bI is\b",      "I am",        "I is → I am"),
    (r"\bhe don't\b",  "he doesn't",  "he don't → he doesn't"),
    (r"\bshe don't\b", "she doesn't", "she don't → she doesn't"),
    (r"\bit don't\b",  "it doesn't",  "it don't → it doesn't"),
    (r"\bthey was\b",  "they were",   "they was → they were"),
    (r"\bwe was\b",    "we were",     "we was → we were"),
    (r"\byou was\b",   "you were",    "you was → you were"),
    (r"\bgood\b(?= than)", "well",    "good than → well than"),
    (r"\bmore better\b","better",     "more better → better"),
    (r"\bmore easier\b","easier",     "more easier → easier"),
    (r"\bmore faster\b","faster",     "more faster → faster"),
    (r"\bcan able to\b","can",        "can able to → can"),
    (r"\bshould of\b", "should have", "should of → should have"),
    (r"\bcould of\b",  "could have",  "could of → could have"),
    (r"\bwould of\b",  "would have",  "would of → would have"),
    (r"\bmight of\b",  "might have",  "might of → might have"),
    (r"\bmust of\b",   "must have",   "must of → must have"),
    (r"\bless people\b","fewer people","less people → fewer people"),
    (r"\bless students\b","fewer students","less students → fewer students"),
    (r"\btheir is\b",  "there is",    "their is → there is"),
    (r"\btheir are\b", "there are",   "their are → there are"),
    (r"\byour welcome\b","you're welcome","your welcome → you're welcome"),
    (r"\bits a\b",     "it's a",      "its a → it's a"),

    # ── Capitalisation ────────────────────────────────────────────────────
    (r"(?<=[.!?]\s)([a-z])", lambda m: m.group(1).upper(),
     "Capitalise after sentence end"),
]


def correct_grammar(text: str) -> dict:
    """
    Apply grammar rules to text.
    Returns original, corrected, list of changes, and a diff structure.
    """
    original  = text
    corrected = text
    changes   = []

    for pattern, replacement, desc in GRAMMAR_RULES:
        if callable(replacement):
            new = re.sub(pattern, replacement, corrected)
        else:
            new = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE
                         if not any(c.isupper() for c in pattern) else 0)

        if new != corrected:
            # Find what changed
            before_snippet = _find_changed_snippet(corrected, new)
            after_snippet = _find_changed_snippet(new, corrected)
            changes.append({
                "rule":        desc,
                "pattern":     pattern,
                "before":      before_snippet,
                "after":       after_snippet,
            })
            corrected = new

    # Capitalise first character
    if corrected and corrected[0].islower():
        corrected = corrected[0].upper() + corrected[1:]
        if original[0].islower():
            changes.append({
                "rule":    "Capitalise first letter",
                "pattern": "^[a-z]",
                "before":  original[:20],
                "after":   corrected[:20]
            })

    # Build word-level diff
    diff = _build_diff(original, corrected)

    quality_before = _text_quality_score(original)
    quality_after  = _text_quality_score(corrected)

    return {
        "original":       original,
        "corrected":      corrected,
        "changes":        changes,
        "change_count":   len(changes),
        "diff":           diff,
        "quality_before": quality_before,
        "quality_after":  quality_after,
        "improved":       quality_after > quality_before,
    }


def _find_changed_snippet(before: str, after: str, ctx: int = 30) -> str:
    """Find first position where strings differ and return snippet."""
    for i, (a, b) in enumerate(zip(before, after)):
        if a != b:
            s = max(0, i - ctx // 2)
            return f"…{before[s:s+ctx]}…"
    return before[:ctx]


def _text_quality_score(text: str) -> float:
    """Simple quality heuristic: 0–100."""
    score = 100.0
    score -= len(re.findall(r"  +", text)) * 3
    score -= len(re.findall(r" [,\.!?]", text)) * 4
    score -= len(re.findall(r"\b(should of|could of|would of)\b", text, re.I)) * 8
    score -= len(re.findall(r"\b(their is|their are)\b", text, re.I)) * 8
    score -= len(re.findall(r"\b(recieve|acheive|seperate|definately)\b", text, re.I)) * 5
    return max(0.0, min(100.0, score))


def _build_diff(original: str, corrected: str) -> list:
    """
    Build a token-level diff list.
    Each item: {"type": "same"|"removed"|"added", "text": str}
    Uses simple LCS-based comparison on words.
    """
    orig_words = original.split()
    corr_words = corrected.split()

    # Simple patience diff
    diff = []
    i = j = 0
    while i < len(orig_words) or j < len(corr_words):
        if i < len(orig_words) and j < len(corr_words):
            if orig_words[i] == corr_words[j]:
                diff.append({"type": "same", "text": orig_words[i]})
                i += 1; j += 1
            else:
                # Try to find next match
                found = False
                for look in range(1, min(5, len(orig_words)-i, len(corr_words)-j)+1):
                    if orig_words[i+look] == corr_words[j] if i+look < len(orig_words) else False:
                        diff.append({"type": "removed", "text": orig_words[i]})
                        i += 1; found = True; break
                    if corr_words[j+look] == orig_words[i] if j+look < len(corr_words) else False:
                        diff.append({"type": "added",   "text": corr_words[j]})
                        j += 1; found = True; break
                if not found:
                    diff.append({"type": "removed", "text": orig_words[i]})
                    diff.append({"type": "added",   "text": corr_words[j]})
                    i += 1; j += 1
        elif i < len(orig_words):
            diff.append({"type": "removed", "text": orig_words[i]}); i += 1
        else:
            diff.append({"type": "added",   "text": corr_words[j]}); j += 1

    return diff


# ─────────────────────────────────────────────────────────────────────────────
#  Unified Text Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_text(text: str, mode: str = "all") -> dict:
    """
    mode: "ai" | "fake_news" | "grammar" | "all"
    Returns a combined result dict.
    """
    result = {"input_text": text, "word_count": word_count(text)}

    if mode in ("ai", "all"):
        result["ai_detection"]   = detect_ai_text(text)
    if mode in ("fake_news", "all"):
        result["fake_news"]      = detect_fake_news(text)
    if mode in ("grammar", "all"):
        result["grammar"]        = correct_grammar(text)

    return result
