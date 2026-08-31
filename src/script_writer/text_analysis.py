from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter
from typing import Any


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")
NUMBER_RE = re.compile(r"\b(?:\d+(?:[.,]\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.I)
FIRST_PERSON = {"i", "i'm", "i've", "me", "my", "mine", "we", "we're", "us", "our", "ours"}
SECOND_PERSON = {"you", "you're", "you've", "your", "yours"}
THIRD_PERSON = {"he", "she", "it", "they", "him", "her", "them", "his", "their", "theirs"}
TRANSITIONS = {
    "but", "however", "so", "because", "therefore", "then", "now", "instead",
    "although", "yet", "finally", "meanwhile", "for example", "the truth is",
}
SLANG = {"gonna", "wanna", "gotta", "kinda", "sorta", "yeah", "okay", "ok", "dude", "literally"}
EMOTIONAL = {
    "amazing", "angry", "awful", "brilliant", "cruel", "danger", "disaster", "fear",
    "hate", "horrible", "incredible", "love", "obscene", "pain", "shocking", "terrible",
    "worst", "best", "urgent", "threat",
}
IMPERATIVE_STARTERS = {
    "ask", "avoid", "build", "consider", "do", "don't", "follow", "get", "give", "imagine",
    "keep", "learn", "let", "look", "make", "remember", "save", "start", "stop", "take", "try",
    "use", "watch",
}


def clean_transcript(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def normalize_transcript(text: str) -> str:
    return " ".join(token.lower().replace("’", "'") for token in WORD_RE.findall(clean_transcript(text)))


def words(text: str) -> list[str]:
    return [token.replace("’", "'") for token in WORD_RE.findall(text)]


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def sentence_statistics(sentences: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(words(str(sentence.get("text", "")))) for sentence in sentences]
    lengths = [length for length in lengths if length > 0]
    if not lengths:
        return {
            "count": 0,
            "word_lengths": [],
            "mean_words": None,
            "median_words": None,
            "min_words": None,
            "max_words": None,
            "p90_words": None,
        }
    return {
        "count": len(lengths),
        "word_lengths": lengths,
        "mean_words": round(statistics.fmean(lengths), 3),
        "median_words": round(statistics.median(lengths), 3),
        "min_words": min(lengths),
        "max_words": max(lengths),
        "p90_words": percentile(lengths, 0.9),
    }


def linguistic_statistics(text: str, sentences: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [token.lower() for token in words(text)]
    token_count = len(tokens)
    counts = Counter(tokens)
    content_repeats = {
        token: count
        for token, count in counts.items()
        if count >= 3 and len(token) >= 4
    }
    sentence_count = max(1, len(sentences))
    question_count = sum("?" in str(sentence.get("text", "")) for sentence in sentences)
    imperative_count = 0
    for sentence in sentences:
        sentence_tokens = [token.lower() for token in words(str(sentence.get("text", "")))]
        if sentence_tokens and sentence_tokens[0] in IMPERATIVE_STARTERS:
            imperative_count += 1
    transitions: list[dict[str, Any]] = []
    lowered = text.lower()
    for transition in sorted(TRANSITIONS):
        count = len(re.findall(rf"\b{re.escape(transition)}\b", lowered))
        if count:
            transitions.append({"transition": transition, "count": count})
    sentence_stats = sentence_statistics(sentences)
    return {
        "sentence_length_distribution": sentence_stats,
        "vocabulary": {
            "token_count": token_count,
            "unique_token_count": len(counts),
            "type_token_ratio": round(len(counts) / token_count, 4) if token_count else 0,
            "average_token_characters": round(
                statistics.fmean(len(token) for token in tokens), 3
            ) if tokens else 0,
            "long_word_share": round(
                sum(len(token) >= 8 for token in tokens) / token_count, 4
            ) if token_count else 0,
        },
        "person_usage": {
            "first_person_count": sum(token in FIRST_PERSON for token in tokens),
            "second_person_count": sum(token in SECOND_PERSON for token in tokens),
            "third_person_count": sum(token in THIRD_PERSON for token in tokens),
        },
        "question_count": question_count,
        "question_density_per_sentence": round(question_count / sentence_count, 4),
        "imperative_candidate_count": imperative_count,
        "imperative_candidate_density": round(imperative_count / sentence_count, 4),
        "numerical_reference_count": len(NUMBER_RE.findall(text)),
        "emotional_lexicon_matches": sorted(token for token in counts if token in EMOTIONAL),
        "slang_or_informal_matches": sorted(token for token in counts if token in SLANG),
        "short_sentence_share": round(
            sum(length <= 8 for length in sentence_stats["word_lengths"])
            / sentence_stats["count"],
            4,
        ) if sentence_stats["count"] else 0,
        "repeated_content_tokens": content_repeats,
        "transition_markers": transitions,
    }


def sentence_cadence(sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for sentence in sentences:
        start = sentence.get("start")
        end = sentence.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end <= start:
            continue
        count = len(words(str(sentence.get("text", ""))))
        result.append(
            {
                "sentence_id": sentence.get("sentence_id"),
                "start_seconds": round(float(start), 4),
                "end_seconds": round(float(end), 4),
                "word_count": count,
                "words_per_second": round(count / (float(end) - float(start)), 3),
            }
        )
    return result
