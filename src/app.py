"""StudySnap AI: turn a lecture screenshot into simple revision notes."""

from __future__ import annotations

import json
import logging
import re
import tomllib
import unicodedata
from collections import Counter
from datetime import datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.errors import PyPdfError
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit.errors import StreamlitAuthError, StreamlitMissingAuthlibError

try:
    import pytesseract
except ImportError:
    pytesseract = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "output" / "users"
AUTH_CONFIG_PATH = PROJECT_ROOT / ".streamlit" / "secrets.toml"
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
LOGGER = logging.getLogger(__name__)
GENERIC_TERMS = {
    "write", "using", "create", "program", "practice", "list", "make",
    "example", "examples", "following", "given", "show", "used", "use",
    "learn", "learning", "question", "answer", "note", "notes", "class",
    "chapter", "exercise", "write program", "html program",
}
TECHNICAL_TERMS = [
    "machine learning", "deep learning", "neural network", "neural networks",
    "artificial intelligence", "data science", "database management system",
    "generative ai", "large language model", "prompt engineering",
    "full stack development", "database", "python", "html", "css",
    "javascript", "sql", "api", "rest api", "frontend", "backend",
    "react", "node.js", "statistics", "statistical", "probability",
    "probability distribution", "mean", "median", "mode", "variance",
    "standard deviation", "hypothesis testing", "correlation",
    "primary key", "foreign key", "algorithm",
    "data structure", "data structures", "loop", "function", "variable",
    "array", "dictionary", "class", "object", "table", "query",
    "normalization", "regression", "classification", "clustering", "dataset",
    "supervised learning", "unsupervised learning", "feature", "model",
    "tag", "attributes", "attribute", "element", "web page", "markup language",
]
TOPIC_LABELS = {
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "artificial intelligence": "Artificial Intelligence",
    "data science": "Data Science",
    "database management system": "Database Management System",
    "database": "Database",
    "python": "Python",
    "html": "HTML",
    "css": "CSS",
    "javascript": "JavaScript",
    "sql": "SQL",
    "neural network": "Neural Networks",
    "neural networks": "Neural Networks",
    "data structure": "Data Structures",
    "data structures": "Data Structures",
    "regression": "Regression",
    "classification": "Classification",
    "clustering": "Clustering",
}
SUBJECT_PATTERNS = [
    (
        "Full Stack Development",
        r"\bfull[\s-]*stack(?:\s+development)?\b",
    ),
    (
        "Generative AI",
        r"\bgenerative\s+(?:artificial\s+intelligence|ai)\b|\bgen\s*ai\b",
    ),
    (
        "Database Management Systems",
        r"\bdatabase\s+management\s+systems?\b|\bdbms\b",
    ),
    ("Data Science", r"\bdata\s+science\b"),
    ("Machine Learning", r"\bmachine\s+learning\b|\bml\b"),
    ("Artificial Intelligence", r"\bartificial\s+intelligence\b|\bai\b"),
    ("Statistics", r"\bstatistics?\b|\bstatistical\b"),
    ("Python", r"\bpython\b"),
    ("HTML", r"\bhtml\b|hypertext\s+markup\s+language"),
    ("CSS", r"\bcss\b|cascading\s+style\s+sheets?"),
    ("JavaScript", r"\bjavascript\b"),
    ("SQL", r"\bsql\b|structured\s+query\s+language"),
    ("Data Structures", r"\bdata\s+structures?\b"),
]
DEFINITION_PATTERN = re.compile(
    r"\b(?:is|are|means?|refers?\s+to|defined\s+as|known\s+as)\b",
    re.IGNORECASE,
)
FORMULA_PATTERN = re.compile(
    r"(?<![A-Za-z])"
    r"[A-Za-z][A-Za-z0-9_]*(?:\s*[=<>+\-*/^]\s*"
    r"[A-Za-z0-9_().%]+)+"
    r"(?:\s*[+\-*/^=<>]\s*[A-Za-z0-9_().%]+)*"
)
EXAMPLE_PATTERN = re.compile(
    r"\b(?:for example|e\.g\.|example\b|instance\b|such as)\b",
    re.IGNORECASE,
)
KEYWORD_NOUNS = {
    "algorithm", "array", "attribute", "code", "condition", "concept",
    "data", "database", "design", "element", "feature", "function",
    "input", "key", "language", "loop", "model", "network", "normalization",
    "object", "output", "process", "record", "result", "structure", "table",
    "tag", "value", "variable", "web", "page",
}
HTML_FOCUS_CONCEPTS = [
    (
        "HTML structure",
        r"<!doctype|<html\b|<head\b|<body\b|<title\b|html\s+(?:document\s+)?structure"
        r"|\btitle\b|\bbody\s+content\b",
    ),
    ("heading tags", r"<h[1-6]\b|\bheadings?\b"),
    ("tables", r"\btable(?:s)?\b|<(?:tr|td|th)\b"),
    ("ordered lists", r"ordered\s+lists?|<ol\b"),
    ("forms", r"\bforms?\b|<form\b"),
    ("paragraphs", r"<p\b|\bparagraphs?\b"),
    ("input fields", r"input\s+fields?|\b(?:password|email|dob|date of birth)\b|<input\b"),
    ("dropdowns", r"dropdowns?|\bselect(?:ion)?\b|<select\b"),
    ("radio buttons", r"radio\s+buttons?|type\s*=\s*[\"']?radio"),
    ("textarea", r"\btext\s*areas?\b|<textarea\b"),
]
HTML_INCIDENTAL_FOCUS = {
    "admission", "department", "employee", "emp", "salary", "joining date",
    "student", "marksheet", "mark sheet", "gender", "hobbies", "dob",
    "password", "email",
}
HTML_MARKUP_PATTERN = re.compile(
    r"\bhtml\b|<!doctype\b|\bweb\s?page\b|"
    r"<(?:html|head|body|title|h[1-6]|p|table|tr|td|th|form|ol|ul|li|"
    r"input|select|option|textarea)\b",
    re.IGNORECASE,
)
STOP_WORDS = sorted(
    {
        word
        for word in (
            set(TfidfVectorizer(stop_words="english").get_stop_words() or set())
            | GENERIC_TERMS
        )
        if " " not in word
    }
)


def clean_text(text: str) -> str:
    """Normalize OCR artifacts without removing meaningful numbered content."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace(r"\|", " ").replace("|", " ")
    cleaned_lines = []
    for line in normalized.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line or re.fullmatch(r"\d+\s*[).]?", line):
            continue
        line = re.sub(
            r"\b(?:[A-Z]\s+){1,}[A-Z]\b",
            lambda match: re.sub(r"\s+", "", match.group()),
            line,
        )
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        line = re.sub(r"([,.;:!?]){2,}", r"\1", line)
        line = re.sub(r"(?<!\w)[,;:!?](?!\w)", " ", line)
        line = re.sub(r"(?<!\w)[~`]+(?!\w)", " ", line)
        line = re.sub(r"\s+", " ", line).strip(" \t-•")
        if line and re.search(r"[A-Za-z0-9]", line):
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _otsu_threshold(image: Image.Image) -> int:
    """Find a useful binary threshold from a grayscale image histogram."""
    histogram = image.histogram()
    total_pixels = sum(histogram)
    weighted_total = sum(level * count for level, count in enumerate(histogram))
    background_weight = 0
    background_total = 0
    best_variance = -1.0
    threshold = 127

    for level, count in enumerate(histogram):
        background_weight += count
        if background_weight == 0:
            continue
        foreground_weight = total_pixels - background_weight
        if foreground_weight == 0:
            break
        background_total += level * count
        background_mean = background_total / background_weight
        foreground_mean = (weighted_total - background_total) / foreground_weight
        variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )
        if variance > best_variance:
            best_variance = variance
            threshold = level

    return threshold


def preprocess_image(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Prepare grayscale and thresholded variants for local OCR."""
    grayscale = ImageOps.grayscale(image)
    longest_side = max(grayscale.size)
    if longest_side < 1800:
        scale = min(3.0, 1800 / longest_side)
        grayscale = grayscale.resize(
            (round(grayscale.width * scale), round(grayscale.height * scale)),
            Image.Resampling.LANCZOS,
        )
    elif longest_side > 3000:
        scale = 3000 / longest_side
        grayscale = grayscale.resize(
            (round(grayscale.width * scale), round(grayscale.height * scale)),
            Image.Resampling.LANCZOS,
        )

    enhanced = ImageEnhance.Contrast(grayscale).enhance(1.7)
    denoised = enhanced.filter(ImageFilter.MedianFilter(size=3))
    threshold = _otsu_threshold(denoised)
    binary = denoised.point(lambda pixel: 255 if pixel > threshold else 0)
    return denoised, binary


def extract_ocr_text(image: Image.Image) -> str:
    """Recognize text using orientation correction and the strongest local variant."""
    if pytesseract is None:
        raise RuntimeError("The pytesseract Python package is not installed.")

    oriented_image = ImageOps.exif_transpose(image).convert("RGB")
    try:
        orientation_data = pytesseract.image_to_osd(
            oriented_image,
            config="--psm 0",
        )
        rotation_match = re.search(r"Rotate:\s*(\d+)", orientation_data)
        if rotation_match:
            rotation = int(rotation_match.group(1)) % 360
            if rotation:
                oriented_image = oriented_image.rotate(-rotation, expand=True)
    except pytesseract.TesseractError:
        # Orientation detection may fail on short notes; OCR can still proceed.
        pass

    variants = preprocess_image(oriented_image)
    best_image = variants[0]
    best_score = float("-inf")
    for candidate in variants:
        data = pytesseract.image_to_data(
            candidate,
            output_type=pytesseract.Output.DICT,
            config="--psm 3",
        )
        confidences = [
            float(confidence)
            for confidence, text in zip(data["conf"], data["text"])
            if text.strip() and float(confidence) >= 0
        ]
        if confidences:
            score = sum(confidences) / len(confidences) + len(confidences) * 0.05
            if score > best_score:
                best_image = candidate
                best_score = score

    return pytesseract.image_to_string(best_image, config="--psm 3")


def split_sentences(text: str) -> list[str]:
    """Split cleaned text at common sentence-ending punctuation."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        if sentence.strip()
    ]


def extract_keywords(text: str, limit: int = 8) -> list[str]:
    """Rank recognized subject terms and content-bearing terms from the source."""
    lowered = text.casefold()
    known_terms = [
        term for term in TECHNICAL_TERMS
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered)
    ]
    for subject, pattern in SUBJECT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE) and not any(
            subject.casefold() == term.casefold() for term in known_terms
        ):
            known_terms.append(subject)
    try:
        vectorizer = TfidfVectorizer(
            stop_words=STOP_WORDS,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]*\b",
        )
        scores = vectorizer.fit_transform([text]).toarray()[0]
        terms = vectorizer.get_feature_names_out()
        ranked_terms = sorted(zip(terms, scores), key=lambda item: (-item[1], item[0]))
        candidates = known_terms + [
            term for term, _ in ranked_terms
            if _useful_keyword(term)
        ]
        keywords = []
        seen = set()
        for term in candidates:
            normalized_term = term.casefold().strip()
            if normalized_term not in seen and _useful_keyword(normalized_term):
                keywords.append(term)
                seen.add(normalized_term)
            if len(keywords) >= limit:
                break
        return keywords
    except ValueError:
        words = re.findall(r"\b[a-zA-Z][a-zA-Z'-]{2,}\b", text.lower())
        return [
            word for word, _ in Counter(words).most_common(limit)
            if _useful_keyword(word)
        ][:limit]


def detect_subject(text: str) -> str:
    """Return the best-supported known subject, or leave it for manual entry."""
    matches = []
    for priority, (subject, pattern) in enumerate(SUBJECT_PATTERNS):
        found = list(re.finditer(pattern, text, re.IGNORECASE))
        if found:
            matches.append(
                (len(found), max(len(match.group()) for match in found), -priority, subject)
            )
    if not matches:
        return ""
    return max(matches)[3]


def detect_chapter(text: str) -> str:
    """Extract a numbered chapter, unit, module, or labelled topic when present."""
    match = re.search(
        r"(?im)^\s*((?:chapter|unit|module|lesson)\s+"
        r"(?:no\.?\s*)?[\w.-]+(?:\s*[:\-–]\s*[^\n]{1,100})?)\s*$",
        text,
    )
    if match:
        return " ".join(match.group(1).split()).strip(" :—-")
    return ""


def detect_title(text: str) -> str:
    """Prefer explicit title/topic headings or a concise source heading."""
    explicit = re.search(
        r"(?im)^\s*(?:title|topic)\s*:\s*([^\n]{2,100})",
        text,
    )
    if explicit:
        return explicit.group(1).strip(" \t:.-")
    for line in text.splitlines():
        heading = line.strip(" \t#*-•")
        if (
            1 < len(heading.split()) <= 12
            and len(heading) <= 100
            and not re.search(r"[.!?;]$", heading)
            and not re.match(
                r"(?i)^(?:chapter|unit|module|lesson)\b",
                heading,
            )
        ):
            return heading
    return ""


def extract_definitions(text: str) -> list[str]:
    """Keep only source sentences that use an explicit defining phrase."""
    return [
        sentence for sentence in _source_sentences(text)
        if DEFINITION_PATTERN.search(sentence)
        and not re.search(r"[=<>]", sentence)
    ][:8]


def extract_formulas(text: str) -> list[str]:
    """Extract equation-like source lines without synthesizing notation."""
    formulas = []
    for line in text.splitlines():
        candidate = line.strip(" \t-•")
        if (
            "=" in candidate
            and FORMULA_PATTERN.search(candidate)
            and candidate not in formulas
        ):
            formulas.append(candidate)
    return formulas[:10]


def extract_examples(text: str) -> list[str]:
    """Keep only source statements explicitly marked as examples."""
    return [
        sentence for sentence in _source_sentences(text)
        if EXAMPLE_PATTERN.search(sentence)
    ][:8]


def _useful_keyword(term: str) -> bool:
    """Keep known technical terms and words that resemble content-bearing nouns."""
    words = term.casefold().split()
    if term.casefold() in TECHNICAL_TERMS or any(
        term.casefold() == subject.casefold()
        for subject, _ in SUBJECT_PATTERNS
    ):
        return True
    if not words or any(word in GENERIC_TERMS for word in words):
        return False
    if len(words) > 1:
        return (
            words[0] in KEYWORD_NOUNS
            and words[1] in KEYWORD_NOUNS
        ) or (
            words[0].endswith(("tion", "sion", "ment", "ness", "ity", "ism", "ogy", "ics"))
            and words[1] in KEYWORD_NOUNS
        )
    word = words[0]
    return len(word) > 3 and (
        word.endswith(("tion", "sion", "ment", "ness", "ity", "ism", "ogy", "ics"))
        or word in {"data", "web", "code", "logic", "syntax", "output", "input"}
    )


def detect_topic(text: str) -> str:
    """Prefer a supported topic label or an explicit source heading."""
    lowered = text.casefold()
    found_topics = [
        (lowered.find(term), -len(term), label)
        for term, label in TOPIC_LABELS.items()
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered)
    ]
    if found_topics:
        return min(found_topics)[2]
    title = detect_title(text)
    if title:
        return title
    return ""


def summarize(text: str, sentence_limit: int = 2) -> str:
    """Select informative, distinct sentences for a concise extractive summary."""
    sentences = _source_sentences(text)
    if not sentences:
        return "No clear summary could be generated from the extracted text."
    keywords = extract_keywords(text, limit=12)
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (
            -sum(1 for keyword in keywords if keyword.casefold() in item[1].casefold()),
            abs(len(item[1].split()) - 18),
            item[0],
        ),
    )
    selected = sorted(ranked[:sentence_limit], key=lambda item: item[0])
    return " ".join(sentence for _, sentence in selected)


def make_important_questions(text: str) -> list[str]:
    """Generate five content-specific prompts, adapting to programming material."""
    sentences = _content_sentences(text) or [text.strip()]
    keywords = extract_keywords(text, limit=10)
    programming = (
        _study_type(text) in {"Programming", "Mixed"}
        and (
            _is_html_notes(text)
            or re.search(
                r"\b(?:code|program\w*|python|html|javascript|syntax|function\w*|"
                r"variable\w*|loop\w*)\b",
                text,
                re.IGNORECASE,
            )
            is not None
        )
    )
    if programming and _is_html_notes(text):
        questions = [question for _, _, question in _html_practice_tasks(text)]
        represented = set()
        for task_kind, _, _ in _html_practice_tasks(text):
            if task_kind == "webpage":
                represented.update({"HTML structure", "heading tags"})
            elif task_kind.endswith("table"):
                represented.add("tables")
            elif task_kind.endswith("form"):
                represented.update(
                    {"forms", "input fields", "dropdowns", "radio buttons", "textarea"}
                )
            elif task_kind == "ordered-list":
                represented.add("ordered lists")

        html_prompts = {
            "HTML structure": (
                "Write an HTML program with the document structure described in the notes."
            ),
            "heading tags": (
                "Write an HTML program that uses the heading tags discussed in the notes."
            ),
            "paragraphs": (
                "Write an HTML program that places the paragraph content described in the notes on a webpage."
            ),
            "tables": (
                "Write an HTML program to create a table using the table elements described in the notes."
            ),
            "ordered lists": (
                "Write an HTML program to display the listed items using an ordered list."
            ),
            "forms": (
                "Write an HTML program to create a form using the controls described in the notes."
            ),
            "input fields": (
                "Write an HTML program to add the input fields described in the notes to a form."
            ),
            "dropdowns": (
                "Write an HTML program to add the described choices to a dropdown in a form."
            ),
            "radio buttons": (
                "Write an HTML program to add the radio-button choices described in the notes."
            ),
            "textarea": (
                "Write an HTML program to add a textarea to the form described in the notes."
            ),
        }
        for concept, pattern in HTML_FOCUS_CONCEPTS:
            if (
                re.search(pattern, text, re.I)
                and concept not in represented
                and html_prompts[concept] not in questions
            ):
                questions.append(html_prompts[concept])
                represented.add(concept)
            if len(questions) == 5:
                break
        for sentence in sentences:
            if len(questions) == 5:
                break
            question = (
                "Write an HTML program that implements the requirement stated in this note: "
                f"{sentence}"
            )
            if question not in questions:
                questions.append(_trim_text(question, 240))
        fallback_index = 0
        while len(questions) < 5:
            concepts = [
                concept for concept, pattern in HTML_FOCUS_CONCEPTS
                if re.search(pattern, text, re.I)
            ]
            concept = (
                concepts[fallback_index % len(concepts)]
                if concepts else "HTML structure"
            )
            fallback_prompts = [
                f"Write an HTML program demonstrating {concept} as covered in the notes.",
                f"Build an HTML example that applies {concept} to the content described in the notes.",
                f"Create a small webpage using {concept} and state what it displays.",
            ]
            question = fallback_prompts[fallback_index % len(fallback_prompts)]
            if question in questions:
                question = f"{question[:-1]} (alternative {fallback_index + 1})."
            questions.append(question)
            fallback_index += 1
        return questions[:5]

    if programming:
        practice_sentences = [
            sentence for sentence in sentences
            if re.search(
                r"\b(?:write|create|make|build|design|implement|display|calculate)\b",
                sentence,
                re.I,
            )
        ]
        questions = [
            _trim_text(
                "Write a program that implements the requirement described in this note: "
                f"{sentence}",
                240,
            )
            for sentence in practice_sentences
        ]
        questions = list(dict.fromkeys(questions))
        excluded_terms = {"python", "html", "javascript", "program", "programming", "code", "notes"}
        for concept in keywords:
            if concept.casefold() in excluded_terms:
                continue
            question = (
                f"Write a short program using {concept} to demonstrate its role in the notes, "
                "and state the expected output."
            )
            if question not in questions:
                questions.append(question)
            if len(questions) == 5:
                break
        for sentence in sentences:
            if len(questions) == 5:
                break
            question = _trim_text(
                f"Write a small program that demonstrates the concept described here: {sentence}",
                240,
            )
            if question not in questions:
                questions.append(question)
        fallback_index = 0
        while len(questions) < 5:
            concepts = [
                concept for concept in keywords
                if concept.casefold() not in excluded_terms
            ]
            concept = (
                concepts[fallback_index % len(concepts)]
                if concepts else "the documented concept"
            )
            fallback_prompts = [
                f"Implement {concept} as a small working program, following the details in the notes.",
                f"Write another short program demonstrating {concept} from the notes and show its output.",
                f"Build a working example of {concept} using the information covered in the notes.",
            ]
            question = fallback_prompts[fallback_index % len(fallback_prompts)]
            if question in questions:
                question = f"{question[:-1]} (alternative {fallback_index + 1})."
            questions.append(question)
            fallback_index += 1
        return questions[:5]

    questions = []
    for index in range(5):
        sentence = sentences[index % len(sentences)]
        concept = next(
            (word for word in keywords if word.casefold() in sentence.casefold()),
            keywords[index % len(keywords)] if keywords else "the documented concept",
        )
        prompts = [
            f"Define {concept} using the information in this note: {sentence}",
            f"Explain the relationship described here: {sentence}",
            f"Describe one application of {concept} based on this note.",
            f"Compare {concept} with another concept mentioned in these notes.",
            f"Explain the process or property stated here: {sentence}",
        ]
        questions.append(_trim_text(prompts[index], 240))
    return questions


def _html_practice_tasks(text: str) -> list[tuple[str, str, str]]:
    """Recognize explicit HTML exercise prompts and retain their source order."""
    patterns = [
        (
            "webpage",
            r"\b(?:create|write|make|build|design)\b.{0,50}\bweb\s?page\b",
            "webpage creation",
            "Write an HTML program to create a webpage containing a title, heading, paragraph and body content.",
        ),
        (
            "employee-table",
            r"\b(?:create|write|make|build|display)\b.{0,50}\b(?:employee|emp)\b.{0,35}\btable\b"
            r"|\b(?:employee|emp)\b.{0,35}\btable\b",
            "employee tables",
            "Write an HTML program to create an employee table with Emp No., Name, Department, Salary and Joining Date.",
        ),
        (
            "student-registration-form",
            r"\bstudent\b.{0,30}\bregistration\b.{0,15}\bform\b|\bregistration\b.{0,15}\bform\b",
            "student registration forms",
            "Write an HTML program to create a Student Registration Form with Name, Password, Gender, Email, DOB, Hobbies and Submit/Reset buttons.",
        ),
        (
            "college-admission-form",
            r"\bcollege\b.{0,25}\badmission\b.{0,15}\bform\b|\badmission\b.{0,15}\bform\b",
            "college admission forms",
            "Write an HTML program to create a College Admission Form with a course dropdown, textarea, radio buttons and Submit button.",
        ),
        (
            "student-marksheet-table",
            r"\bstudent\b.{0,30}\bmarks?\s?sheet\b.{0,25}\btable\b|\bmarks?\s?sheet\b.{0,25}\btable\b",
            "student marksheet tables",
            "Write an HTML program to create a student marksheet table.",
        ),
        (
            "ordered-list",
            r"\b(?:five|5)\s+subjects?\b.{0,40}\bordered\s+list\b|\bordered\s+list\b.{0,40}\b(?:five|5)\s+subjects?\b",
            "ordered lists of five subjects",
            "Write an HTML program to display five subjects using an ordered list.",
        ),
    ]
    matches = []
    lowered = text.casefold()
    for kind, pattern, label, question in patterns:
        match = re.search(pattern, lowered, re.I)
        if match:
            matches.append((match.start(), kind, label, question))
    matches.sort(key=lambda item: item[0])
    return [(kind, label, question) for _, kind, label, question in matches]


def _is_html_notes(text: str) -> bool:
    """Recognize HTML notes by their subject name, markup, or webpage references."""
    return HTML_MARKUP_PATTERN.search(text) is not None


def make_short_answer_questions(keywords: list[str], topic: str) -> list[str]:
    """Create three short-answer prompts using extracted concepts."""
    terms = (keywords + [topic])[:3]
    while len(terms) < 3:
        terms.append(topic)
    return [
        f"Define {terms[0]} in one or two sentences.",
        f"State one use or application of {terms[1]}.",
        f"Explain the role of {terms[2]} in the topic covered.",
    ]


def make_mcqs(text: str, keywords: list[str]) -> list[dict[str, object]]:
    """Build cloze MCQs whose correct term appears in its source sentence."""
    if _is_html_notes(text):
        html_questions = _html_mcqs(text)
        if html_questions:
            return html_questions

    candidates = list(dict.fromkeys(keywords))
    for word in re.findall(r"\b[a-zA-Z][a-zA-Z'-]{3,}\b", text):
        normalized = word.casefold()
        if _useful_keyword(normalized) and normalized not in {
            item.casefold() for item in candidates
        }:
            candidates.append(word)
    candidates = candidates[:15]
    sentences = _content_sentences(text) or [text.strip()]
    matches = [
        (sentence, term)
        for sentence in sentences
        for term in sorted(candidates, key=lambda candidate: (-len(candidate), candidate.casefold()))
        if re.search(rf"\b{re.escape(term)}\b", sentence, re.IGNORECASE)
    ]
    questions = []

    for index in range(3):
        if not matches:
            source_statement = sentences[index % len(sentences)]
            options = _statement_options(sentences, source_statement, candidates)
            options = _position_answer(options, source_statement, index)
            questions.append(
                {
                    "question": "Which statement is supported by the extracted notes?",
                    "options": options,
                    "answer": options[0],
                }
            )
            continue
        sentence, answer = matches[index % len(matches)]
        distractors = [
            term for term in candidates
            if term.casefold() != answer.casefold()
        ][:2]
        options = [answer] + distractors
        if len(options) < 3:
            options = _statement_options(sentences, sentence, candidates)
            options = _position_answer(options, sentence, index)
            questions.append(
                {
                    "question": "Which statement is supported by the extracted notes?",
                    "options": options,
                    "answer": options[0],
                }
            )
            continue
        masked_sentence = re.sub(
            rf"\b{re.escape(answer)}\b", "_____", sentence, count=1, flags=re.IGNORECASE
        )
        options = _position_answer(options, answer, index)
        questions.append(
            {
                "question": f'Which term completes this statement from the notes: "{masked_sentence}"?',
                "options": options,
                "answer": answer,
            }
        )
    return questions


def _html_mcqs(text: str) -> list[dict[str, object]]:
    """Ask concept-specific HTML questions with independently verified answers."""
    concepts = [
        (
            "HTML structure",
            "Which HTML element contains the visible content of a webpage?",
            "<body>",
            ["<head>", "<title>"],
        ),
        (
            "heading tags",
            "Which HTML tag represents the largest heading?",
            "<h1>",
            ["<h6>", "<p>"],
        ),
        (
            "tables",
            "Which HTML tag defines a row in a table?",
            "<tr>",
            ["<td>", "<th>"],
        ),
        (
            "ordered lists",
            "Which HTML tag creates an ordered list?",
            "<ol>",
            ["<ul>", "<li>"],
        ),
        (
            "forms",
            "Which HTML element groups controls for submitting user input?",
            "<form>",
            ["<table>", "<ol>"],
        ),
        (
            "input fields",
            "Which HTML element is commonly used for a single-line input field?",
            "<input>",
            ["<textarea>", "<select>"],
        ),
        (
            "dropdowns",
            "Which HTML element creates a dropdown list?",
            "<select>",
            ["<textarea>", "<option>"],
        ),
        (
            "radio buttons",
            "Which input type lets a user select one option from a group?",
            'type="radio"',
            ['type="text"', 'type="password"'],
        ),
        (
            "textarea",
            "Which HTML element provides a multi-line text input?",
            "<textarea>",
            ["<input>", "<select>"],
        ),
    ]
    available = [
        item for item in concepts
        if any(item[0] == label and re.search(pattern, text, re.I)
               for label, pattern in HTML_FOCUS_CONCEPTS)
    ]
    if not available:
        return []
    return [
        {
            "question": available[index % len(available)][1],
            "options": _position_answer(
                [available[index % len(available)][2]]
                + available[index % len(available)][3],
                available[index % len(available)][2],
                index,
            ),
            "answer": available[index % len(available)][2],
        }
        for index in range(3)
    ]


def _position_answer(options: list[str], answer: str, index: int) -> list[str]:
    """Vary the correct option position while preserving the answer choices."""
    choices = [option for option in options if option.casefold() != answer.casefold()]
    choices.insert(index % (len(choices) + 1), answer)
    return choices


def _source_sentences(text: str) -> list[str]:
    """Split into distinct source statements while preserving their wording."""
    candidates = []
    seen = set()
    for line in text.splitlines():
        for sentence in split_sentences(line):
            sentence = sentence.strip(" \t-•")
            key = sentence.casefold()
            if (
                len(sentence.split()) >= 4
                and re.search(r"[A-Za-z]{3}", sentence)
                and key not in seen
            ):
                candidates.append(sentence)
                seen.add(key)
    return candidates


def _content_sentences(text: str) -> list[str]:
    """Drop OCR fragments and repeated sentences while keeping real list items."""
    candidates = []
    seen = set()
    for line in text.splitlines():
        pieces = split_sentences(line)
        for sentence in pieces:
            sentence = sentence.strip(" \t-•:;,.")
            key = sentence.casefold()
            if (
                len(sentence.split()) >= 4
                and re.search(r"[A-Za-z]{3}", sentence)
                and key not in seen
            ):
                candidates.append(sentence)
                seen.add(key)
    return candidates


def _important_points(text: str, keywords: list[str]) -> list[str]:
    """Rank useful source statements instead of copying OCR fragments in order."""
    sentences = _content_sentences(text)
    if not sentences:
        return ["No complete, meaningful points could be identified."]
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (
            -sum(1 for keyword in keywords if keyword.casefold() in item[1].casefold()),
            abs(len(item[1].split()) - 16),
            item[0],
        ),
    )
    selected = sorted(ranked[:5], key=lambda item: item[0])
    return [_trim_text(sentence, 180) for _, sentence in selected]


def _study_type(text: str) -> str:
    """Classify notes by matching a few transparent study-type indicators."""
    lowered = text.casefold()
    programming = len(re.findall(
        r"\b(?:code|program\w*|python|html|javascript|syntax|function\w*|"
        r"variable\w*|loop\w*)\b",
        lowered,
    ))
    if _is_html_notes(text):
        programming += 1
    practical = len(re.findall(
        r"\b(?:experiment\w*|procedure\w*|step\w*|calculat\w*|practical|"
        r"implement\w*|activit\w*|measurement\w*)\b",
        lowered,
    ))
    theory = len(re.findall(
        r"\b(?:definition\w*|defin\w*|theor\w*|concept\w*|propert\w*|"
        r"advantage\w*|explain\w*|principle\w*)\b",
        lowered,
    ))
    scores = {"Programming": programming, "Practical": practical, "Theory": theory}
    ranked_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if (
        ranked_scores[1][1] >= 2
        and ranked_scores[1][1] >= ranked_scores[0][1] * 0.4
    ):
        return "Mixed"
    return ranked_scores[0][0] if ranked_scores[0][1] else "Theory"


def _exam_focus(text: str, keywords: list[str]) -> list[str]:
    """Return three to five distinct revision concepts from the screenshot."""
    if _is_html_notes(text):
        focus = [
            label for label, pattern in HTML_FOCUS_CONCEPTS
            if re.search(pattern, text, re.I)
        ]
        incidental = {term.casefold() for term in HTML_INCIDENTAL_FOCUS}
        for term in keywords:
            if (
                term.casefold() not in incidental
                and term.casefold() not in {item.casefold() for item in focus}
            ):
                focus.append(term)
            if len(focus) == 5:
                break
        return focus[:5]

    focus = list(dict.fromkeys(keywords[:5]))
    for sentence in _content_sentences(text):
        if len(focus) >= 5:
            break
        terms = [
            term for term in extract_keywords(sentence, limit=3)
            if not any(term.casefold() == item.casefold() for item in focus)
        ]
        focus.extend(terms)
    for sentence in _content_sentences(text):
        if len(focus) >= 3:
            break
        point = _trim_text(sentence, 90)
        if point.casefold() not in {item.casefold() for item in focus}:
            focus.append(point)
    if len(focus) < 3:
        for sentence in _content_sentences(text):
            words = sentence.split()
            for start in range(1, len(words)):
                point = _trim_text(" ".join(words[start:start + 4]), 60)
                if point and point.casefold() not in {
                    item.casefold() for item in focus
                }:
                    focus.append(point)
                if len(focus) >= 3:
                    break
            if len(focus) >= 3:
                break
    if len(focus) < 3:
        source_terms = keywords or [
            word for word in re.findall(r"\b[a-zA-Z][a-zA-Z'-]{2,}\b", text)
            if word.casefold() not in STOP_WORDS and word.casefold() not in GENERIC_TERMS
        ]
        if source_terms:
            term = source_terms[0]
            fallback_topics = [
                f"Definition and purpose of {term}",
                f"Uses and applications of {term}",
                f"Examples of {source_terms[1]}" if len(source_terms) > 1
                else f"How {term} works",
            ]
            for point in fallback_topics:
                if point.casefold() not in {item.casefold() for item in focus}:
                    focus.append(point)
                if len(focus) >= 3:
                    break
    return focus[:5]


def _statement_options(
    sentences: list[str], answer: str, candidates: list[str]
) -> list[str]:
    """Use one source statement and clearly false, content-based distractors."""
    options = [answer]
    content_options = [
        clause.strip()
        for sentence in sentences
        for clause in re.split(r"[,;]|\band\b", sentence, flags=re.IGNORECASE)
        if len(clause.split()) >= 3
    ]
    for sentence in sentences + content_options:
        if sentence.casefold() == answer.casefold():
            continue
        candidate = f"The notes do not mention: {_trim_text(sentence, 100)}"
        if candidate.casefold() not in {item.casefold() for item in options}:
            options.append(candidate)
        if len(options) == 3:
            break
    for term in candidates:
        if len(options) == 3:
            break
        candidate = f"The notes do not mention {term}."
        if candidate.casefold() not in {item.casefold() for item in options}:
            options.append(candidate)
    while len(options) < 3:
        candidate = (
            f"The notes do not mention: {_trim_text(answer, 100)}"
            if len(options) == 1
            else f"The notes contradict: {_trim_text(answer, 100)}"
        )
        if candidate.casefold() in {item.casefold() for item in options}:
            candidate += " (false option)"
        options.append(candidate)
    return options


def _trim_text(text: str, limit: int) -> str:
    """Trim a generated prompt at a word boundary."""
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3].rsplit(' ', 1)[0]}..."


def build_notes_text(
    topic: str,
    title: str,
    subject: str,
    chapter: str,
    study_type: str,
    exam_focus: list[str],
    concepts: list[str],
    summary: str,
    keywords: list[str],
    important_points: list[str],
    definitions: list[str],
    formulas: list[str],
    examples: list[str],
    questions: list[str],
    short_questions: list[str],
    mcqs: list[dict[str, object]],
) -> str:
    """Format generated notes as a plain-text revision sheet."""
    sections = [
        "StudySnap AI - Smart Notes",
        "=" * 30,
        f"\nTitle\n{title or 'Not detected'}",
        f"\nSubject\n{subject or 'Not detected'}",
        f"\nChapter / Unit\n{chapter or 'Not detected'}",
        f"\nDetected Topic\n{topic}",
        f"\n📚 Study Type\n{study_type}",
        "\n🎯 Exam Focus\n" + "\n".join(f"- {item}" for item in exam_focus),
        f"\nShort Summary\n{summary}",
        "\nImportant Concepts\n"
        + ("\n".join(f"- {item}" for item in concepts) or "- No concepts identified"),
        "\nKey Definitions (from source)\n"
        + ("\n".join(f"- {item}" for item in definitions) or "- No explicit definitions found"),
        "\nImportant Keywords\n"
        + ("\n".join(f"- {word}" for word in keywords) or "- No keywords found"),
        "\nFormulas (from source)\n"
        + ("\n".join(f"- {item}" for item in formulas) or "- No formulas found"),
        "\nExamples (from source)\n"
        + ("\n".join(f"- {item}" for item in examples) or "- No explicit examples found"),
        "\nImportant Points\n"
        + ("\n".join(f"- {point}" for point in important_points) or "- No points found"),
        "\nQuick Revision\n"
        + (
            "- Revisit these concepts: " + ", ".join(concepts) + ".\n"
            if concepts else ""
        )
        + "- Review the important points above.\n"
        "- Cover the keywords and try to explain each one from memory.",
        "\nExam Mode - Important Questions\n"
        + "\n".join(f"{number}. {question}" for number, question in enumerate(questions, 1)),
        "\nExam Mode - Short Questions\n"
        + "\n".join(
            f"{number}. {question}"
            for number, question in enumerate(short_questions, 1)
        ),
        "\nExam Mode - MCQs\n"
        + "\n".join(
            f"{number}. {item['question']}\n"
            + "\n".join(
                f"   {letter}. {option}"
                for letter, option in zip("ABC", item["options"])
            )
            + f"\n   Answer: {item['answer']}"
            for number, item in enumerate(mcqs, 1)
        ),
    ]
    return "\n".join(sections) + "\n"


def load_study_history() -> list[dict[str, object]]:
    """Load locally saved study sessions, raising clear errors for invalid data."""
    history_path = _history_file_path()
    if not history_path.exists():
        return []
    with history_path.open(encoding="utf-8") as history_file:
        history = json.load(history_file)
    if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
        raise ValueError("Study history has an invalid format.")
    return history


def save_study_history(history: list[dict[str, object]]) -> None:
    """Atomically write local study history as UTF-8 JSON."""
    output_directory = _user_output_directory()
    history_path = _history_file_path()
    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_file = history_path.with_suffix(".json.tmp")
    temporary_file.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_file.replace(history_path)


def save_study_session(session: dict[str, object]) -> None:
    """Append a generated study session to the local history file."""
    history = load_study_history()
    history.append(session)
    save_study_history(history)


def update_study_session_metadata(
    session_id: str,
    title: str,
    subject: str,
    chapter: str,
) -> None:
    """Keep manually corrected title, subject, and chapter in saved history."""
    history = load_study_history()
    session = next(
        (item for item in history if item.get("id") == session_id),
        None,
    )
    if session is None:
        raise ValueError("The saved study session was not found.")
    updated_values = {
        "title": title,
        "subject": subject,
        "chapter": chapter,
    }
    if any(session.get(key) != value for key, value in updated_values.items()):
        session.update(updated_values)
        save_study_history(history)


def delete_study_session(session_id: str) -> None:
    """Delete one session by its stable local identifier."""
    history = load_study_history()
    updated_history = [
        session for session in history
        if session.get("id") != session_id
    ]
    if len(updated_history) == len(history):
        raise ValueError("The selected study session was not found.")
    save_study_history(updated_history)


def clear_study_history() -> None:
    """Clear every saved session while keeping the history file valid."""
    save_study_history([])


def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, int]:
    """Extract selectable text from every PDF page locally."""
    reader = PdfReader(BytesIO(pdf_bytes))
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("This PDF is password-protected.")
    page_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(page_text).strip(), len(page_text)


def _pdf_heading_topics(text: str) -> list[str]:
    """Extract concise section headings without treating prose as a topic."""
    topics = []
    seen = set()
    for line in text.splitlines():
        heading = re.sub(r"^\s*[-*•#\d.)]+\s*", "", line).strip(" \t:–—-")
        key = re.sub(r"\s+", " ", heading).casefold()
        if (
            not heading
            or key in seen
            or len(heading.split()) > 10
            or len(heading) > 90
            or re.search(r"[.!?]$", heading)
        ):
            continue
        is_requirement_id = re.match(r"(?i)^FR\s*\d+\s*[-:–—]\s*\w+", heading)
        is_named_heading = (
            re.match(r"(?i)^(?:chapter|unit|section|requirement)\b", heading)
            or re.match(r"(?i)^[A-Z][A-Za-z0-9/&(), –—-]{2,70}$", heading)
            and (
                heading.istitle()
                or heading.isupper()
                or len(heading.split()) <= 5
            )
        )
        if is_requirement_id or is_named_heading:
            topics.append(heading)
            seen.add(key)
    return topics


def generate_pdf_summary_points(text: str, limit: int = 8) -> list[str]:
    """Select up to eight distinct, source-grounded statements for PDF revision."""
    sentences = [
        sentence for sentence in _source_sentences(text)
        if len(sentence.split()) >= 6
        and re.search(
            r"\b(?:is|are|was|were|has|have|had|can|could|will|would|should|"
            r"must|shall|means?|provid\w*|allow\w*|enabl\w*|support\w*|"
            r"improv\w*|us\w*|stor\w*|contain\w*|includ\w*|requir\w*|"
            r"help\w*|describ\w*|defin\w*|perform\w*|display\w*|search\w*|"
            r"match\w*|represent\w*|relat\w*|interact\w*|design\w*|create\w*|"
            r"manag\w*|gather\w*|facilitat\w*)\b",
            sentence,
            re.IGNORECASE,
        )
    ]
    if not sentences:
        return []
    keywords = extract_keywords(text, limit=16)
    headings = _pdf_heading_topics(text)
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (
            -sum(
                keyword.casefold() in item[1].casefold()
                for keyword in keywords
            ),
            -sum(
                word.casefold() in item[1].casefold()
                for heading in headings
                for word in heading.split()
                if len(word) > 3
            ),
            -int(bool(re.search(
                r"\b(?:important|purpose|benefit|advantage|process|"
                r"relationship|requirement|result|therefore|because)\b",
                item[1],
                re.IGNORECASE,
            ))),
            abs(len(item[1].split()) - 20),
            item[0],
        ),
    )
    selected = sorted(ranked[:max(5, min(limit, 8))], key=lambda item: item[0])
    return [sentence for _, sentence in selected]


def _pdf_question_facts(text: str) -> list[tuple[str, str]]:
    """Pair complete factual PDF statements with their nearest section heading."""
    headings = {
        re.sub(r"\s+", " ", heading).casefold(): heading
        for heading in _pdf_heading_topics(text)
    }
    heading_lookup = set(headings)
    current_heading = ""
    facts = []
    seen = set()
    for line in text.splitlines():
        candidate = re.sub(r"^\s*[-*•#\d.)]+\s*", "", line).strip(" \t:–—-")
        heading_key = re.sub(r"\s+", " ", candidate).casefold()
        if heading_key in heading_lookup:
            current_heading = headings[heading_key]
            continue
        for sentence in split_sentences(line):
            sentence = sentence.strip(" \t-•")
            key = sentence.casefold()
            if (
                len(sentence.split()) < 6
                or len(sentence.split()) > 40
                or key in seen
                or not re.search(
                    r"\b(?:is|are|was|were|has|have|had|can|could|will|would|"
                    r"should|must|shall|means?|provid\w*|allow\w*|enabl\w*|"
                    r"support\w*|improv\w*|us\w*|stor\w*|contain\w*|includ\w*|"
                    r"requir\w*|help\w*|describ\w*|defin\w*|perform\w*|"
                    r"display\w*|search\w*|match\w*|represent\w*|relat\w*|"
                    r"interact\w*|design\w*|creat\w*|manag\w*|gather\w*)\b",
                    sentence,
                    re.IGNORECASE,
                )
            ):
                continue
            facts.append((current_heading, sentence))
            seen.add(key)
    return facts


def generate_pdf_questions(
    text: str,
) -> list[tuple[str, str, list[str], str]]:
    """Build two short answers, two descriptive questions, and one grounded MCQ."""
    facts = _pdf_question_facts(text)
    if not facts:
        return []

    questions: list[tuple[str, str, list[str], str]] = []
    used_prompts = set()
    source_statements = [fact for _, fact in facts]

    def add(kind: str, prompt: str) -> None:
        normalized = prompt.casefold()
        if normalized not in used_prompts:
            questions.append((kind, prompt, [], ""))
            used_prompts.add(normalized)

    topics = []
    for heading, fact in facts:
        topic = heading or next(
            (
                keyword for keyword in extract_keywords(fact, limit=8)
                if keyword.casefold() not in {"study notes", "programming"}
            ),
            "",
        )
        if topic and topic.casefold() not in {item.casefold() for item in topics}:
            topics.append(topic)

    for keyword in extract_keywords(text, limit=16):
        if (
            any(
                keyword.casefold() in fact.casefold()
                for fact in source_statements
            )
            and keyword.casefold() not in {item.casefold() for item in topics}
        ):
            topics.append(keyword)

    if not topics:
        topics = extract_keywords(text, limit=8)

    for topic in topics:
        if len([item for item in questions if item[0] == "Short Answer"]) == 2:
            break
        if re.search(r"\bscalability\b", topic, re.IGNORECASE):
            prompt = "What does the PDF state about the scalability requirement?"
        elif re.search(r"\bFR\s*\d+\s*[-:–—]", topic, re.IGNORECASE):
            prompt = f"What does {topic} require the system to do?"
        else:
            prompt = f"What does the PDF state about {topic}?"
        add("Short Answer", prompt)

    descriptive_prompts = []
    lowered = text.casefold()
    if (
        re.search(r"\busers?\b", lowered)
        and re.search(r"\bitem reports?\b", lowered)
        and re.search(r"\b(?:relationship|associated|linked|connect)\b", lowered)
    ):
        descriptive_prompts.append(
            "How does the PDF describe the relationship between users and item reports?"
        )
    if (
        re.search(r"\bitem reports?\b", lowered)
        and re.search(r"\bsearch(?:ing)?\b", lowered)
        and re.search(r"\b(?:improv\w*|help\w*|support\w*|facilitat\w*)\b", lowered)
    ):
        descriptive_prompts.append(
            "How do item reports improve searching, according to the PDF?"
        )
    if (
        re.search(r"\brequirements?\b", lowered)
        and re.search(r"\b(?:gathered|collected|elicited)\b", lowered)
        and re.search(r"\bdatabase\b", lowered)
        and re.search(r"\bUML\b", text, re.IGNORECASE)
    ):
        descriptive_prompts.append(
            "How do the gathered requirements support database and UML design?"
        )
    for prompt in descriptive_prompts:
        if len([item for item in questions if item[0] == "Descriptive"]) == 2:
            break
        add("Descriptive", prompt)

    for topic in topics:
        if len([item for item in questions if item[0] == "Descriptive"]) == 2:
            break
        add(
            "Descriptive",
            f"What key information does the PDF present about {topic}?",
        )

    short_facts = source_statements
    if len(short_facts) >= 4:
        mcq_topic = next(
            (
                topic for topic in topics
                if any(
                    topic.casefold() in fact.casefold()
                    for fact in short_facts
                )
            ),
            "",
        )
        correct_fact = next(
            (
                fact for fact in short_facts
                if mcq_topic and mcq_topic.casefold() in fact.casefold()
            ),
            short_facts[0],
        )
        if mcq_topic:
            prompt = (
                f"Which statement specifically describes {mcq_topic} "
                "according to the PDF?"
            )
        else:
            mcq_topic = next(
                (
                    keyword for keyword in extract_keywords(text, limit=12)
                    if keyword.casefold() in short_facts[0].casefold()
                ),
                "the main topic",
            )
            correct_fact = short_facts[0]
            prompt = (
                f"Which statement in the PDF describes {mcq_topic}?"
            )
        distractors = [
            fact for fact in short_facts
            if fact != correct_fact
            and mcq_topic.casefold() not in fact.casefold()
        ]
        if len(distractors) < 3:
            distractors = [
                fact for fact in short_facts
                if fact != correct_fact
            ]
        options = [correct_fact, *distractors[:3]]
        if len(options) == 4 and len(set(options)) == 4:
            questions.append(("MCQ", prompt, options, "A"))

    return questions[:5]


def generate_pdf_summary(text: str) -> str:
    """Generate a local extractive summary from PDF text."""
    if not text.strip():
        raise ValueError("This PDF does not contain readable text to summarize.")
    points = generate_pdf_summary_points(text)
    if not points:
        return "No complete, meaningful summary points could be identified."
    return "\n".join(f"• {point}" for point in points)


def generate_pdf_study_notes(text: str) -> dict[str, list[str]]:
    """Build concise study-note sections using only text extracted from a PDF."""
    if not text.strip():
        return {
            "Key Points": [],
            "Important Definitions": [],
            "Important Concepts": [],
            "Exam Focus": [],
        }

    keywords = extract_keywords(text, limit=10)
    key_points = _important_points(text, keywords)
    if key_points == ["No complete, meaningful points could be identified."]:
        key_points = []

    concepts = [
        keyword for keyword in keywords
        if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text, re.IGNORECASE)
    ][:8]
    exam_focus = []
    for focus in _pdf_heading_topics(text) + keywords:
        if (
            re.search(rf"(?<!\w){re.escape(focus)}(?!\w)", text, re.IGNORECASE)
            and focus.casefold() not in {item.casefold() for item in exam_focus}
        ):
            exam_focus.append(focus)
        if len(exam_focus) == 5:
            break

    return {
        "Key Points": key_points,
        "Important Definitions": extract_definitions(text),
        "Important Concepts": concepts,
        "Exam Focus": exam_focus,
    }


def clear_local_study_data() -> None:
    """Clear saved sessions, generated notes, and app-owned in-memory data."""
    clear_study_history()
    saved_notes = _user_output_directory() / "studysnap_notes.txt"
    if saved_notes.exists():
        saved_notes.unlink()
    for key in (
        "analysis",
        "analysis_image",
        "analysis_error",
        "ocr_error",
        "ocr_ready",
        "ocr_raw_text",
        "ocr_cleaned_text",
        "pdf_fingerprint",
        "pdf_text",
        "pdf_summary",
        "pdf_summary_error",
        "pdf_questions",
        "pdf_study_notes",
        "pdf_error",
        "pdf_page_count",
        "pdf_character_count",
        "history_open_id",
        "history_select_id",
        "history_message",
        "history_error",
        "confirm_clear_history",
    ):
        st.session_state.pop(key, None)
    st.session_state["upload_reset_version"] = (
        st.session_state.get("upload_reset_version", 0) + 1
    )


def _auth_configuration_status() -> tuple[bool, str]:
    """Validate native Streamlit OIDC setup without displaying credential values."""
    if not AUTH_CONFIG_PATH.is_file():
        return False, "The Google sign-in configuration file is missing."
    try:
        configuration = tomllib.loads(AUTH_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False, "The Google sign-in configuration file could not be read."

    auth = configuration.get("auth")
    required = (
        "redirect_uri",
        "cookie_secret",
        "client_id",
        "client_secret",
        "server_metadata_url",
    )
    if not isinstance(auth, dict) or any(
        not isinstance(auth.get(key), str) or not auth[key].strip()
        for key in required
    ):
        return False, "The Google sign-in configuration is incomplete."
    if any(
        "replace" in auth[key].casefold()
        or "your_" in auth[key].casefold()
        or auth[key].strip().startswith("<")
        for key in required
    ):
        return False, "Replace the example values with your Google OIDC credentials."
    return True, ""


def _redact_oidc_error_message(message: str) -> str:
    """Remove local credentials and OAuth tokens before logging auth errors."""
    try:
        auth = tomllib.loads(AUTH_CONFIG_PATH.read_text(encoding="utf-8")).get(
            "auth", {}
        )
    except (OSError, tomllib.TOMLDecodeError):
        auth = {}

    if isinstance(auth, dict):
        for key in ("client_id", "client_secret", "cookie_secret"):
            value = auth.get(key)
            if isinstance(value, str) and value:
                message = message.replace(value, "[REDACTED]")

    return re.sub(
        r"(?i)(\b(?:client_id|client_secret|cookie_secret|id_token|access_token|"
        r"refresh_token|authorization_code|code|state)\s*[=:]\s*)"
        r"([^&\s,;\"']+)",
        r"\1[REDACTED]",
        message,
    )


def _user_output_directory() -> Path:
    """Return this authenticated account's private local data directory."""
    configured_path = st.session_state.get("_user_output_directory")
    if not isinstance(configured_path, str) or not configured_path:
        raise RuntimeError("The authenticated account storage is not initialized.")
    return Path(configured_path)


def _history_file_path() -> Path:
    return _user_output_directory() / "study_history.json"


def _render_login_screen(auth_message: str, can_login: bool) -> None:
    """Show a standalone Google OIDC login screen without protected app content."""
    st.markdown(
        """
        <style>
        .login-shell {
            max-width: 680px;
            margin: 8vh auto 0;
            padding: clamp(1.6rem, 5vw, 3.2rem);
            text-align: center;
            background: linear-gradient(145deg, #ffffff 0%, #f4f6ff 100%);
            border: 1px solid #e2e8f3;
            border-radius: 26px;
            box-shadow: 0 22px 60px rgba(28, 45, 86, 0.12);
        }
        .login-logo {
            display: grid;
            width: 4rem;
            height: 4rem;
            place-items: center;
            margin: 0 auto 1rem;
            color: white;
            background: linear-gradient(135deg, #4258c8, #7955c8);
            border-radius: 19px;
            font-size: 2rem;
            box-shadow: 0 10px 24px rgba(75, 85, 190, 0.24);
        }
        .login-shell h1 { margin-bottom: 0.35rem; }
        .login-subtitle {
            color: #5968c5;
            font-size: 1.12rem;
            font-weight: 700;
            margin-bottom: 1.1rem;
        }
        .login-copy {
            max-width: 500px;
            margin: 0 auto 1.35rem;
            color: #66758d;
            line-height: 1.65;
        }
        @media (max-width: 600px) {
            .login-shell { margin-top: 3vh; border-radius: 20px; }
        }
        </style>
        <main class="login-shell">
          <div class="login-logo" aria-hidden="true">📚</div>
          <h1>StudySnap AI</h1>
          <div class="login-subtitle">AI-Powered Smart Study Assistant</div>
          <p class="login-copy">
            Sign in with Google to access your private study workspace.
            Your notes and study history are kept separate for your account.
          </p>
        </main>
        """,
        unsafe_allow_html=True,
    )
    login_columns = st.columns([1, 1.25, 1])
    center = login_columns[1]
    with center:
        if st.button(
            "Continue with Google",
            key="google_login_button",
            type="primary",
            use_container_width=True,
            disabled=not can_login,
        ):
            try:
                st.login()
            except (StreamlitAuthError, StreamlitMissingAuthlibError) as error:
                LOGGER.error(
                    "Google OIDC login failed (type=%s): %s",
                    type(error).__name__,
                    _redact_oidc_error_message(str(error)),
                )
                st.error(
                    "Google sign-in could not start. Verify the local OIDC "
                    "configuration and try again."
                )
    if not can_login:
        st.warning(
            f"{auth_message} Copy `.streamlit/secrets.example.toml` to "
            "`.streamlit/secrets.toml` and fill it with credentials from your "
            "Google Cloud OAuth client."
        )
    st.caption(
        "Privacy: authentication is handled by Google using OpenID Connect. "
        "StudySnap AI does not store your Google password."
    )


def build_markdown_notes(
    title: str,
    subject: str,
    chapter: str,
    summary: str,
    keywords: list[str],
    important_points: list[str],
    quick_revision: list[str],
    questions: list[str],
    short_questions: list[str],
    mcqs: list[dict[str, object]],
) -> str:
    """Build a clean Markdown revision sheet from generated source content."""
    sections = [
        "# StudySnap AI",
        f"## {title or 'Study Notes'}",
        f"**Subject:** {subject or 'Not specified'}  ",
        f"**Chapter/Topic:** {chapter or 'Not specified'}",
        "## Summary",
        summary or "No summary could be generated from the source content.",
        "## Important Keywords",
        *([f"- {word}" for word in keywords] or ["- No keywords found."]),
        "## Important Points",
        *([f"- {point}" for point in important_points] or ["- No points found."]),
        "## Quick Revision",
        *([f"- {item}" for item in quick_revision] or ["- Review the points above."]),
        "## Important Questions",
        *(
            [f"{number}. {question}" for number, question in enumerate(questions, 1)]
            or ["No questions generated."]
        ),
        "## Short Questions",
        *(
            [f"{number}. {question}" for number, question in enumerate(short_questions, 1)]
            or ["No short questions generated."]
        ),
        "## MCQs",
    ]
    if mcqs:
        for number, item in enumerate(mcqs, 1):
            sections.append(f"{number}. {item['question']}")
            sections.extend(
                f"   - {letter}. {option}"
                for letter, option in zip("ABC", item["options"])
            )
            sections.append(f"   - Answer: {item['answer']}")
    else:
        sections.append("No MCQs generated.")
    return "\n\n".join(sections) + "\n"


def build_pdf_notes(markdown_text: str) -> bytes:
    """Render a printable PDF locally with ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    from xml.sax.saxutils import escape

    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="StudySnap AI Notes",
        author="StudySnap AI",
    )
    base_styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "StudySnapTitle",
        parent=base_styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#263d78"),
        spaceAfter=16,
    )
    heading_style = ParagraphStyle(
        "StudySnapHeading",
        parent=base_styles["Heading2"],
        textColor=colors.HexColor("#354b91"),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "StudySnapBody",
        parent=base_styles["BodyText"],
        leading=14,
        spaceAfter=4,
    )
    story = []
    for line in markdown_text.splitlines():
        content = line.strip()
        if not content:
            story.append(Spacer(1, 4))
        elif content.startswith("# "):
            story.append(Paragraph(escape(content[2:]), title_style))
        elif content.startswith("## "):
            story.append(Paragraph(escape(content[3:]), heading_style))
        elif content.startswith("- "):
            story.append(Paragraph(f"&bull; {escape(content[2:])}", body_style))
        else:
            content = re.sub(r"^\d+\.\s*", "", content)
            content = re.sub(r"^\s*-\s*", "&bull; ", content)
            story.append(Paragraph(escape(content).replace("&amp;bull;", "&bull;"), body_style))
    document.build(story)
    return output.getvalue()


def make_flashcards(
    text: str,
    definitions: list[str],
    keywords: list[str],
) -> list[dict[str, str]]:
    """Build question-answer cards whose answers quote the uploaded study text."""
    cards = []
    seen_answers = set()
    for definition in definitions:
        term_match = re.match(
            r"(.{2,70}?)\s+(?:is|are|means?|refers?\s+to|defined\s+as|known\s+as)\b",
            definition,
            re.IGNORECASE,
        )
        term = term_match.group(1).strip(" .:-") if term_match else ""
        question = f"What does the source say about {term}?" if term else "What definition is given in the source?"
        cards.append({"question": question, "answer": definition})
        seen_answers.add(definition.casefold())

    for keyword in keywords:
        sentence = next(
            (
                source_sentence for source_sentence in _source_sentences(text)
                if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", source_sentence, re.I)
                and source_sentence.casefold() not in seen_answers
            ),
            None,
        )
        if sentence is None:
            continue
        cards.append(
            {
                "question": f"What do the notes state about {keyword}?",
                "answer": sentence,
            }
        )
        seen_answers.add(sentence.casefold())
    return cards


def _move_flashcard(index_key: str, card_count: int, direction: int) -> None:
    """Move the current flashcard pointer without changing card content."""
    current_index = st.session_state.get(index_key, 0)
    st.session_state[index_key] = (current_index + direction) % card_count


def _history_session_label(session: dict[str, object]) -> str:
    """Create a compact label for a saved session selector."""
    title = session.get("title") or session.get("topic") or "Study session"
    subject = session.get("subject") or "Subject not specified"
    timestamp = str(session.get("timestamp", ""))
    try:
        formatted_time = datetime.fromisoformat(timestamp).astimezone().strftime("%b %d, %Y %H:%M")
    except ValueError:
        formatted_time = "Date unavailable"
    return f"{formatted_time} · {subject} · {title}"


def _delete_selected_history(session_id: str) -> None:
    try:
        delete_study_session(session_id)
        analysis = st.session_state.get("analysis")
        if isinstance(analysis, dict) and analysis.get("history_id") == session_id:
            analysis["history_saved"] = False
        st.session_state["history_message"] = "Study session deleted."
        st.session_state["history_open_id"] = ""
        remaining = load_study_history()
        st.session_state["history_select_id"] = str(remaining[0]["id"]) if remaining else ""
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.session_state["history_error"] = f"Could not delete study session: {error}"


def _clear_all_history() -> None:
    try:
        clear_study_history()
        st.session_state["history_message"] = "Study history cleared."
        st.session_state["history_open_id"] = ""
        st.session_state["history_select_id"] = ""
    except OSError as error:
        st.session_state["history_error"] = f"Could not clear study history: {error}"


def render_study_dashboard_and_history() -> None:
    """Render statistics and history only from locally persisted study sessions."""
    st.markdown('<div id="study-history"></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="section-kicker">Your learning progress</div>', unsafe_allow_html=True)
    st.header("📊 Study Dashboard")
    try:
        history = load_study_history()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Could not load local study history: {error}")
        return
    if message := st.session_state.pop("history_message", None):
        st.success(message)
    if error_message := st.session_state.pop("history_error", None):
        st.error(error_message)

    subject_counts = Counter(
        str(session["subject"]).strip()
        for session in history
        if session.get("subject") and str(session["subject"]).strip()
    )
    question_count = sum(
        len(session.get("questions", []))
        + len(session.get("short_questions", []))
        + len(session.get("mcqs", []))
        for session in history
    )
    metric_columns = st.columns(4)
    for column, (label, value) in zip(
        metric_columns,
        (
            ("Study sessions", len(history)),
            ("Notes generated", len(history)),
            ("Questions generated", question_count),
            ("Subjects studied", len(subject_counts)),
        ),
    ):
        column.metric(label, value)
    most_studied = (
        sorted(subject_counts.items(), key=lambda item: (-item[1], item[0].casefold()))[0]
        if subject_counts else None
    )
    st.markdown(
        f"**Most studied subject:** {most_studied[0]} ({most_studied[1]} sessions)"
        if most_studied else "**Most studied subject:** No saved subject data yet."
    )
    if subject_counts:
        st.caption("Session counts by subject")
        st.bar_chart(dict(subject_counts))

    st.markdown("#### Recent study sessions")
    recent_sessions = sorted(
        history,
        key=lambda session: str(session.get("timestamp", "")),
        reverse=True,
    )[:5]
    if recent_sessions:
        for session in recent_sessions:
            st.markdown(f"- {_history_session_label(session)}")
    else:
        st.info("No study sessions saved yet. Generate Smart Notes to start your history.")

    st.markdown("#### Study History")
    st.caption("Open, review, or remove study sessions saved locally on this device.")
    if not history:
        st.caption("Saved sessions will appear here after notes are generated.")
        return
    sessions_by_id = {
        str(session["id"]): session
        for session in history
        if session.get("id")
    }
    if not sessions_by_id:
        st.error("The local history file contains no valid session identifiers.")
        return
    ids = list(sessions_by_id)
    current_selection = st.session_state.get("history_select_id", "")
    if current_selection not in ids:
        current_selection = ids[0]
    selected_id = st.selectbox(
        "Choose a saved session",
        ids,
        index=ids.index(current_selection),
        format_func=lambda session_id: _history_session_label(sessions_by_id[session_id]),
        key="history_select_id",
    )
    open_column, delete_column, clear_column = st.columns(3)
    with open_column:
        st.button(
            "Open saved session",
            key="history_open_button",
            on_click=lambda: st.session_state.update(history_open_id=selected_id),
        )
    with delete_column:
        st.button(
            "Delete selected session",
            key="history_delete_button",
            on_click=_delete_selected_history,
            args=(selected_id,),
        )
    with clear_column:
        st.checkbox("Confirm clear all", key="confirm_clear_history")
        st.button(
            "Clear all history",
            key="history_clear_button",
            disabled=not st.session_state.get("confirm_clear_history", False),
            on_click=_clear_all_history,
        )

    selected_session = sessions_by_id.get(st.session_state.get("history_open_id", ""))
    if selected_session:
        st.markdown("##### Saved session")
        st.write(f"**Title:** {selected_session.get('title') or 'Not specified'}")
        st.write(f"**Subject:** {selected_session.get('subject') or 'Not specified'}")
        st.write(
            f"**Chapter/Topic:** {selected_session.get('chapter') or selected_session.get('topic') or 'Not specified'}"
        )
        st.write(f"**Summary:** {selected_session.get('summary') or 'No summary saved.'}")
        for section, key in (
            ("Important Points", "important_points"),
            ("Important Questions", "questions"),
            ("Short Questions", "short_questions"),
        ):
            st.markdown(f"**{section}**")
            for item in selected_session.get(key, []):
                st.markdown(f"- {item}")
        st.markdown("**MCQs**")
        for number, item in enumerate(selected_session.get("mcqs", []), 1):
            st.markdown(f"{number}. {item.get('question', '')}")
            st.markdown(f"   Answer: {item.get('answer', '')}")


st.set_page_config(
    page_title="StudySnap AI",
    page_icon="📚",
    layout="wide",
)

auth_configured, auth_message = _auth_configuration_status()
if not auth_configured:
    _render_login_screen(auth_message, can_login=False)
    st.stop()

if not st.user.get("is_logged_in", False):
    _render_login_screen("", can_login=True)
    st.stop()

google_subject = st.user.get("sub")
if not isinstance(google_subject, str) or not google_subject.strip():
    st.error(
        "Google sign-in did not provide a usable account identifier. "
        "Please sign out and try again."
    )
    if st.button("Sign out", key="invalid_google_identity_logout"):
        st.logout()
    st.stop()

authenticated_subject = sha256(
    f"google:{google_subject}".encode("utf-8")
).hexdigest()
if st.session_state.get("_authenticated_subject") != authenticated_subject:
    st.session_state.clear()
    st.session_state["_authenticated_subject"] = authenticated_subject

st.session_state["_user_output_directory"] = str(OUTPUT_ROOT / authenticated_subject)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #172642;
        --muted: #66758d;
        --accent: #5369dc;
        --accent-deep: #7653c8;
        --accent-soft: #f0f2ff;
        --line: #e2e8f3;
        --surface: #ffffff;
    }
    html, body, [class*="st-"], [data-testid="stMarkdownContainer"] {
        font-family: "DM Sans", Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(104, 124, 229, 0.11), transparent 34rem),
            radial-gradient(ellipse at 95% 38%, rgba(133, 104, 207, 0.055), transparent 30rem),
            linear-gradient(145deg, #f5f7fc 0%, #fafbfe 52%, #f2f5fb 100%);
        color: var(--ink);
    }
    .main .block-container {
        max-width: 1280px;
        padding: 2.5rem 2.5rem 3.5rem;
    }
    h1, h2, h3, [data-testid="stHeader"] {
        color: var(--ink);
        letter-spacing: -0.025em;
    }
    h1, h2, h3 { line-height: 1.22; }
    h2 { font-weight: 750; }
    h3 { font-weight: 700; }
    [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] {
        margin-bottom: 0.2rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5f7ff 0%, #ffffff 42%, #f7f8fc 100%);
        border-right: 1px solid #e3e8f2;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #24365b;
        letter-spacing: -0.02em;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"],
    [data-testid="stSidebar"] [data-testid="stCheckbox"] {
        border-radius: 12px;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 2.7rem;
    }
    .sidebar-nav {
        display: grid;
        gap: 0.35rem;
        margin: 0.25rem 0 1rem;
    }
    .sidebar-nav a {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        padding: 0.62rem 0.75rem;
        color: #26395c !important;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid transparent;
        border-radius: 11px;
        font-size: 0.92rem;
        font-weight: 650;
        text-decoration: none !important;
        transition: background 140ms ease, border-color 140ms ease, transform 140ms ease;
    }
    .sidebar-nav a:hover {
        background: #eef1ff;
        border-color: #dce2fb;
        transform: translateX(2px);
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        line-height: 1.65;
    }
    [data-testid="stHorizontalBlock"] {
        gap: 1.15rem;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.7rem 2.85rem;
        border-radius: 28px;
        background:
            radial-gradient(circle at 88% 7%, rgba(169, 157, 255, 0.34), transparent 17rem),
            linear-gradient(120deg, #111e3d 0%, #294681 56%, #5c50a4 100%);
        border: 1px solid rgba(255, 255, 255, 0.16);
        box-shadow: 0 24px 58px rgba(25, 42, 91, 0.2);
        margin-bottom: 2rem;
    }
    .hero::after {
        content: "";
        position: absolute;
        width: 250px;
        height: 250px;
        border-radius: 50%;
        right: -65px;
        bottom: -175px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 0 0 28px rgba(255, 255, 255, 0.035),
                    0 0 0 58px rgba(255, 255, 255, 0.025);
    }
    .hero h1 {
        color: #ffffff;
        font-size: clamp(2.2rem, 4vw, 3.05rem);
        margin: 0 0 0.55rem;
        letter-spacing: -0.04em;
    }
    .hero .hero-subtitle {
        display: inline-block;
        color: #d9ddff;
        background: linear-gradient(90deg, #f4f6ff 0%, #bfc9ff 62%, #d9caff 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: clamp(1.1rem, 2.1vw, 1.38rem);
        font-weight: 700;
        margin: 0 0 0.7rem;
    }
    .hero .hero-description {
        color: #e2e8f7;
        max-width: 720px;
        font-size: 1.02rem;
        line-height: 1.7;
        margin: 0 0 1.35rem;
    }
    .workflow-grid {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem;
        max-width: 700px;
    }
    .workflow-step {
        color: #f6f7ff;
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 12px;
        padding: 0.55rem 0.78rem;
        font-size: 0.9rem;
        font-weight: 650;
        backdrop-filter: blur(8px);
    }
    .workflow-arrow {
        color: #c8ceff;
        font-weight: 700;
        padding: 0 0.1rem;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin: 1rem 0 2rem;
    }
    .feature-card {
        display: flex;
        min-height: 205px;
        flex-direction: column;
        align-items: flex-start;
        padding: 1.25rem;
        background: linear-gradient(150deg, #ffffff 0%, #f8f9ff 100%);
        border: 1px solid var(--line);
        border-top: 3px solid var(--accent);
        border-radius: 18px;
        box-shadow: 0 9px 25px rgba(28, 45, 86, 0.06);
        transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        border-color: #cbd3f4;
        box-shadow: 0 15px 32px rgba(33, 51, 105, 0.11);
    }
    .feature-icon {
        display: grid;
        width: 2.7rem;
        height: 2.7rem;
        place-items: center;
        margin-bottom: 0.8rem;
        background: #eef1ff;
        border-radius: 13px;
        font-size: 1.35rem;
    }
    .feature-title {
        color: #1b2d50;
        font-size: 1.05rem;
        font-weight: 750;
        margin-bottom: 0.4rem;
    }
    .feature-description {
        color: var(--muted);
        flex: 1;
        font-size: 0.9rem;
        line-height: 1.55;
        margin: 0 0 0.9rem;
    }
    .feature-action {
        color: #4e60cd !important;
        font-size: 0.88rem;
        font-weight: 750;
        text-decoration: none !important;
    }
    .feature-action:hover {
        color: #7653c8 !important;
        text-decoration: underline !important;
        text-underline-offset: 3px;
    }
    @media (max-width: 1050px) {
        .feature-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 560px) {
        .feature-grid { grid-template-columns: minmax(0, 1fr); }
        .feature-card { min-height: 0; }
    }
    .section-kicker {
        color: #5968c5;
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin: 0.15rem 0 0.3rem;
    }
    .upload-panel, .st-key-upload-panel, .info-card, .question-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 16px;
        box-shadow: 0 9px 25px rgba(28, 45, 86, 0.06);
        transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }
    .upload-panel {
        padding: 1.45rem 1.55rem 1.1rem;
        border: 1px solid #e0e5f2;
        border-top: 4px solid var(--accent);
        margin: 0.85rem 0 1.25rem;
        background: linear-gradient(145deg, #ffffff 0%, #fbfbff 100%);
    }
    .upload-subtitle {
        color: var(--muted);
        margin: 0 0 0.6rem;
    }
    .format-chip {
        display: inline-block;
        color: #4053b6;
        background: #eff1ff;
        border: 1px solid #dfe3ff;
        border-radius: 999px;
        padding: 0.28rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 0.8rem;
    }
    .helper-line {
        color: #6b7890;
        font-size: 0.86rem;
        margin: 0.45rem 0 0;
    }
    .info-card {
        min-height: 150px;
        padding: 1.25rem 1.2rem;
        height: 100%;
    }
    .info-card:hover, .question-card:hover {
        transform: translateY(-3px);
        border-color: #cbd3f4;
        box-shadow: 0 15px 32px rgba(33, 51, 105, 0.11);
    }
    .card-step {
        color: #707dd2;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        margin-bottom: 0.5rem;
    }
    .info-card .card-title {
        color: #1b2d50;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .info-card .card-copy {
        color: var(--muted);
        font-size: 0.91rem;
        line-height: 1.5;
        margin: 0;
    }
    .result-banner {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 1rem 1.25rem;
        margin: 0.3rem 0 0.4rem;
        color: #21346b;
        background: linear-gradient(105deg, #ffffff 0%, #eff1ff 100%);
        border: 1px solid #dfe4f7;
        border-left: 4px solid var(--accent);
        border-radius: 15px;
        box-shadow: 0 7px 20px rgba(28, 45, 86, 0.05);
        font-size: 1.08rem;
        font-weight: 750;
    }
    .exam-hero {
        padding: 1.25rem 1.45rem;
        margin: 0.4rem 0 0.5rem;
        border-radius: 18px;
        color: white;
        background: linear-gradient(115deg, #17284f 0%, #344b9a 62%, #6354ad 100%);
        box-shadow: 0 14px 30px rgba(38, 53, 117, 0.18);
    }
    .exam-hero h2 {
        color: white;
        margin: 0 0 0.25rem;
    }
    .exam-hero p {
        color: #e1e6ff;
        margin: 0;
    }
    .stat-card {
        min-height: 105px;
        padding: 1rem 1.1rem;
        background: linear-gradient(145deg, #ffffff 0%, #fafaff 100%);
        border: 1px solid var(--line);
        border-radius: 17px;
        box-shadow: 0 9px 24px rgba(28, 45, 86, 0.065);
    }
    .stat-value {
        color: #4e60cd;
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.15;
    }
    .stat-label {
        color: var(--muted);
        font-size: 0.9rem;
        font-weight: 650;
        margin-top: 0.2rem;
    }
    .question-label {
        display: inline-block;
        color: #4a5dc4;
        background: #eef0ff;
        border-radius: 7px;
        padding: 0.2rem 0.48rem;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }
    [data-testid="stMarkdownContainer"] code {
        color: #34499f;
        background: #eef1ff;
        border: 1px solid #e0e5ff;
        border-radius: 7px;
        padding: 0.22rem 0.5rem;
        margin: 0.12rem;
        font-size: 0.9em;
    }
    [data-testid="stImage"] img {
        border-radius: 11px;
    }
    [data-testid="stTabs"] [role="tab"] {
        border-radius: 9px 9px 0 0;
        font-weight: 650;
    }
    .footer {
        text-align: center;
        color: #66748b;
        border-top: 1px solid #e2e7f1;
        padding: 1.4rem 0 0.25rem;
        margin-top: 2.4rem;
        font-size: 0.84rem;
    }
    .footer strong { color: #34466d; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 11px;
        min-height: 3rem;
        font-weight: 700;
        border: 1px solid #d8def0;
        transition: transform 160ms ease, box-shadow 160ms ease,
                    border-color 160ms ease, background 160ms ease;
    }
    .stButton > button[kind="primary"] {
        color: #ffffff;
        border-color: transparent;
        background: linear-gradient(105deg, #4c63d8 0%, #654fc4 100%);
        box-shadow: 0 7px 18px rgba(79, 99, 216, 0.2);
    }
    .stButton > button[kind="primary"]:hover {
        color: #ffffff;
        background: linear-gradient(105deg, #4058cb 0%, #5943b7 100%);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 9px 23px rgba(28, 45, 86, 0.15);
        border-color: #bfc9eb;
    }
    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
        outline: 3px solid rgba(83, 104, 221, 0.35);
        outline-offset: 2px;
    }
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #fafbff 0%, #f1f3ff 100%);
        border: 1px dashed #a4afe3;
        border-radius: 16px;
        padding: 0.95rem;
        transition: border-color 160ms ease, background 160ms ease, box-shadow 160ms ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #586bd4;
        background: #f2f4ff;
        box-shadow: 0 8px 22px rgba(83, 104, 221, 0.08);
    }
    [data-testid="stFileUploader"] button {
        border-radius: 9px;
        font-weight: 700;
    }
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 12px;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #475873;
    }
    .pdf-upload-card {
        border-top: 3px solid var(--accent-deep);
    }
    .pdf-details-title {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 750;
        overflow-wrap: anywhere;
    }
    .pdf-details-meta {
        color: var(--muted);
        font-size: 0.88rem;
    }
    .st-key-pdf-action-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9ff 100%);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1.1rem;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.9rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-color: var(--line);
        border-radius: 17px;
        box-shadow: 0 7px 20px rgba(28, 45, 86, 0.05);
        transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #d0d8ef;
        box-shadow: 0 11px 27px rgba(28, 45, 86, 0.08);
    }
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.58;
    }
    [data-testid="stTextArea"] textarea {
        border-color: #dce3ef;
        border-radius: 11px;
        line-height: 1.55;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #8795e3;
        box-shadow: 0 0 0 3px rgba(83, 104, 221, 0.12);
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #4d60ce;
    }
    [data-testid="stExpander"] {
        border-color: var(--line);
        border-radius: 12px;
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
            animation-duration: 0.01ms !important;
        }
    }
    @media (max-width: 700px) {
        .main .block-container { padding: 1.35rem 1rem 2.5rem; }
        [data-testid="stSidebar"] { border-right: 0; }
        .hero { padding: 1.7rem 1.35rem; border-radius: 21px; }
        .hero .hero-description { font-size: 0.96rem; }
        .workflow-grid { align-items: flex-start; gap: 0.4rem; }
        .workflow-step { font-size: 0.83rem; }
        .workflow-arrow { display: none; }
        [data-testid="stHorizontalBlock"] { gap: 0.7rem; }
        .info-card { min-height: 0; }
        .upload-panel { padding: 1rem; }
        .result-banner { font-size: 0.98rem; }
        .pdf-action-card { padding: 0.85rem; }
        .stButton > button, .stDownloadButton > button { min-height: 2.8rem; }
    }
    @media (max-width: 420px) {
        .main .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
        .hero { padding: 1.4rem 1.1rem; }
        .workflow-step { padding: 0.45rem 0.6rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## 📚 StudySnap AI")
st.sidebar.caption("Your personal study workspace")
google_name = st.user.get("name")
google_email = st.user.get("email")
st.sidebar.markdown("### Signed in")
st.sidebar.write(google_name or "Student")
if google_email:
    st.sidebar.caption(google_email)
if st.sidebar.button("Sign out", key="google_logout_button", use_container_width=True):
    st.logout()
    st.stop()
st.sidebar.divider()
st.sidebar.markdown("### Navigation")
st.sidebar.markdown(
    """
    <nav class="sidebar-nav" aria-label="StudySnap AI navigation">
      <a href="#home"><span>🏠</span><span>Home</span></a>
      <a href="#pdf-study"><span>📄</span><span>PDF Study</span></a>
      <a href="#ocr-notes"><span>📸</span><span>OCR &amp; Notes</span></a>
      <a href="#flashcards"><span>🧠</span><span>Flashcards</span></a>
      <a href="#study-history"><span>📚</span><span>Study History</span></a>
      <a href="#settings"><span>⚙️</span><span>Settings</span></a>
    </nav>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown('<div id="settings"></div>', unsafe_allow_html=True)
st.sidebar.markdown("### ⚙️ Settings")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", key="dark_mode_enabled")
st.sidebar.caption("StudySnap AI keeps study processing on this device.")
confirm_clear_data = st.sidebar.checkbox(
    "Confirm clear study data",
    key=f"confirm_clear_data_{st.session_state.get('upload_reset_version', 0)}",
)

def _clear_study_data_callback() -> None:
    try:
        clear_local_study_data()
        st.session_state["study_data_cleared"] = True
        st.session_state.pop("clear_study_data_error", None)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.session_state["clear_study_data_error"] = (
            f"Could not clear local study data: {error}"
        )


st.sidebar.button(
    "🗑️ Clear Study Data",
    key=f"clear_study_data_button_{st.session_state.get('upload_reset_version', 0)}",
    disabled=not confirm_clear_data,
    on_click=_clear_study_data_callback,
    help="Deletes saved history and notes and clears the current study session.",
)
if st.session_state.pop("study_data_cleared", False):
    st.sidebar.success("Study data cleared successfully!")
if clear_error := st.session_state.pop("clear_study_data_error", None):
    st.sidebar.error(clear_error)

with st.sidebar.expander("ℹ️ About StudySnap AI"):
    st.write(
        "StudySnap AI helps students turn lecture screenshots and PDF notes "
        "into revision material using local OCR and text processing."
    )
    st.caption(
        "Includes Smart Notes, Important Points, Exam Mode, study history, "
        "PDF summaries, and revision tools."
    )

if dark_mode:
    st.markdown(
        """
        <style>
        :root {
            --ink: #e8edf8;
            --muted: #b3bfd2;
            --line: #35435b;
            --surface: #1b2639;
        }
        .stApp {
            background: linear-gradient(145deg, #101827 0%, #151e30 58%, #1b2038 100%);
            color: #e8edf8;
        }
        h1, h2, h3, h4, [data-testid="stHeader"],
        [data-testid="stMarkdownContainer"], label, p, li {
            color: #e8edf8;
        }
        .sidebar-nav a {
            color: #e8edf8 !important;
            background: rgba(39, 54, 83, 0.8);
        }
        .sidebar-nav a:hover {
            background: #314267;
            border-color: #50638d;
        }
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stMetric"],
        .info-card, .stat-card, .feature-card {
            background: #1b2639;
            border-color: #35435b;
        }
        .info-card .card-title, .footer strong { color: #e8edf8; }
        .info-card .card-copy, .helper-line, .footer { color: #b3bfd2; }
        .feature-title { color: #e8edf8; }
        .feature-description { color: #b3bfd2; }
        .feature-icon { background: #273653; }
        .feature-action { color: #aebaff !important; }
        [data-testid="stTextArea"] textarea,
        [data-testid="stTextInput"] input,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background: #182236;
            color: #e8edf8;
            border-color: #465571;
        }
        [data-testid="stFileUploader"] {
            background: #1b2639;
            border-color: #7788ca;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #172236 0%, #141e30 100%);
            border-color: #35435b;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
            color: #e8edf8;
        }
        .st-key-pdf-action-card {
            background: #1b2639;
            border-color: #35435b;
        }
        .pdf-details-title { color: #e8edf8; }
        .pdf-details-meta { color: #b3bfd2; }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] { color: #e8edf8; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div id="home"></div>
    <div class="hero">
      <h1>StudySnap AI</h1>
      <p class="hero-subtitle">AI-Powered Smart Study Assistant</p>
      <p class="hero-description">Upload your lecture notes or screenshots and instantly transform them into organized notes, important points and exam-focused questions.</p>
      <div class="workflow-grid">
        <span class="workflow-step">📸 Upload</span><span class="workflow-arrow">→</span>
        <span class="workflow-step">🔍 Extract</span><span class="workflow-arrow">→</span>
        <span class="workflow-step">🧠 Understand</span><span class="workflow-arrow">→</span>
        <span class="workflow-step">🎯 Revise</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">Your study workspace</div>', unsafe_allow_html=True)
st.subheader("Home Dashboard")
st.caption("Choose a study tool below or use the sidebar to jump to any section.")
feature_cards = [
    (
        "📄",
        "PDF Summary",
        "Turn a text-based PDF into a concise set of source-grounded revision points.",
        "#pdf-upload",
        "Go to PDF upload",
    ),
    (
        "❓",
        "Question Generator",
        "Create short-answer, descriptive, and multiple-choice questions from your PDF.",
        "#pdf-upload",
        "Generate PDF questions",
    ),
    (
        "📝",
        "Smart Study Notes",
        "Extract and organize key points from your lecture-note screenshot.",
        "#notes-upload",
        "Create study notes",
    ),
    (
        "🧠",
        "Flashcards",
        "Review source-backed concepts with reveal-answer and next/previous cards.",
        "#notes-upload",
        "Start flashcard workflow",
    ),
]
feature_cards_html = "".join(
    f'<article class="feature-card">'
    f'<div class="feature-icon" aria-hidden="true">{icon}</div>'
    f'<div class="feature-title">{title}</div>'
    f'<p class="feature-description">{description}</p>'
    f'<a class="feature-action" href="{anchor}">{action} →</a>'
    f'</article>'
    for icon, title, description, anchor, action in feature_cards
)
st.markdown(
    f'<div class="feature-grid">{feature_cards_html}</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">Get started</div>', unsafe_allow_html=True)
st.subheader("📸 Upload Your Notes")
st.caption(
    "Upload a screenshot to extract text, create Smart Notes, and practice with "
    "Exam Mode and flashcards."
)
st.markdown(
    '<div class="upload-subtitle">Drop your lecture screenshot here and let StudySnap AI do the rest.</div>'
    '<div class="format-chip">JPG &nbsp;•&nbsp; JPEG &nbsp;•&nbsp; PNG &nbsp;•&nbsp; WEBP</div>',
    unsafe_allow_html=True,
)
st.markdown('<div id="ocr-notes"></div><div id="notes-upload"></div>', unsafe_allow_html=True)
with st.container(border=True, key="upload-panel"):
    uploaded_file = st.file_uploader(
        "Choose a notes screenshot",
        type=SUPPORTED_IMAGE_TYPES,
        help="Supported formats: JPG, JPEG, PNG and WEBP.",
        key=f"notes_image_upload_{st.session_state.get('upload_reset_version', 0)}",
    )
st.markdown(
    '<p class="helper-line">Best results: clear screenshots with readable text. '
    'Your image is processed locally; no paid API or API key is used.</p>',
    unsafe_allow_html=True,
)

st.divider()
st.markdown('<div class="section-kicker">Additional study material</div>', unsafe_allow_html=True)
st.subheader("📄 Upload PDF Study Material")
st.caption("Add a text-based PDF and create a focused summary or study questions on this device.")
st.markdown('<div id="pdf-study"></div><div id="pdf-upload"></div>', unsafe_allow_html=True)
with st.container(border=True, key="pdf-upload-panel"):
    st.markdown(
        '<div class="upload-subtitle">Choose lecture notes, a chapter, or another course handout.</div>'
        '<div class="format-chip">PDF only &nbsp;•&nbsp; Local processing</div>',
        unsafe_allow_html=True,
    )
    pdf_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Text is extracted locally. Scanned image-only PDFs are not supported by this uploader.",
        key=f"pdf_study_upload_{st.session_state.get('upload_reset_version', 0)}",
    )
if pdf_file is not None:
    pdf_bytes = pdf_file.getvalue()
    pdf_fingerprint = sha256(pdf_bytes).hexdigest()
    if st.session_state.get("pdf_fingerprint") != pdf_fingerprint:
        st.session_state["pdf_fingerprint"] = pdf_fingerprint
        for key in (
            "pdf_text",
            "pdf_summary",
            "pdf_summary_error",
            "pdf_questions",
            "pdf_study_notes",
            "pdf_error",
            "pdf_page_count",
            "pdf_character_count",
        ):
            st.session_state.pop(key, None)
        with st.status("Reading PDF text locally...", expanded=False) as pdf_status:
            try:
                pdf_text, page_count = extract_pdf_text(pdf_bytes)
                st.session_state["pdf_page_count"] = page_count
                if not pdf_text:
                    st.session_state["pdf_error"] = (
                        "No selectable text was found. Scanned image-only PDFs are not supported."
                    )
                    pdf_status.update(
                        label="No selectable text found in this PDF.",
                        state="error",
                    )
                else:
                    st.session_state["pdf_text"] = pdf_text
                    st.session_state["pdf_character_count"] = len(pdf_text)
                    pdf_status.update(
                        label="PDF ready for study.",
                        state="complete",
                    )
            except (PyPdfError, OSError, ValueError):
                st.session_state["pdf_error"] = (
                    "Could not read this PDF. Check that it is valid, unprotected, "
                    "and contains selectable text."
                )
                pdf_status.update(
                    label="Could not read this PDF.",
                    state="error",
                )
    details_columns = st.columns([2, 1, 1])
    with details_columns[0]:
        st.markdown(
            f'<div class="pdf-details-title">📄 {pdf_file.name}</div>'
            '<div class="pdf-details-meta">Uploaded study material</div>',
            unsafe_allow_html=True,
        )
    with details_columns[1]:
        st.metric("File size", f"{pdf_file.size / (1024 * 1024):.2f} MB")
    with details_columns[2]:
        page_count = st.session_state.get("pdf_page_count")
        st.metric("Pages", page_count if page_count is not None else "—")
    if pdf_error := st.session_state.get("pdf_error"):
        st.warning(pdf_error)
    elif pdf_text := st.session_state.get("pdf_text"):
        st.success(
            f"PDF ready: {pdf_file.name} · "
            f"{st.session_state['pdf_page_count']} pages · "
            f"{st.session_state['pdf_character_count']} characters extracted."
        )
        with st.container(border=True, key="pdf-action-card"):
            st.markdown("**What would you like to do with this PDF?**")
            summary_column, questions_column = st.columns(2)
            with summary_column:
                if st.button("📝 Generate Summary", key="pdf_generate_summary"):
                    st.session_state.pop("pdf_summary_error", None)
                    with st.status("Preparing a local summary...", expanded=False) as summary_status:
                        try:
                            summary = generate_pdf_summary(pdf_text)
                            st.session_state["pdf_summary"] = summary
                            summary_status.update(
                                label="Local summary ready.",
                                state="complete",
                            )
                        except ValueError as error:
                            st.session_state["pdf_summary_error"] = str(error)
                            st.session_state.pop("pdf_summary", None)
                            summary_status.update(
                                label="Summary could not be generated.",
                                state="error",
                            )
            with questions_column:
                if st.button("❓ Generate Questions", key="pdf_generate_questions"):
                    with st.spinner("Preparing questions from your PDF..."):
                        st.session_state["pdf_questions"] = generate_pdf_questions(pdf_text)
            if st.button(
                "📘 Generate Study Notes",
                key="pdf_generate_study_notes",
                use_container_width=True,
            ):
                with st.spinner("Preparing study notes from your PDF..."):
                    st.session_state["pdf_study_notes"] = generate_pdf_study_notes(
                        pdf_text
                    )
        if summary_error := st.session_state.get("pdf_summary_error"):
            st.error(summary_error)
        if pdf_summary := st.session_state.get("pdf_summary"):
            st.markdown("#### PDF Summary")
            with st.container(border=True):
                st.write(pdf_summary)
        if pdf_questions := st.session_state.get("pdf_questions"):
            st.markdown("#### Questions from PDF")
            for number, (question_type, prompt, options, answer) in enumerate(
                pdf_questions,
                1,
            ):
                st.markdown(f"**{number}. {question_type}:** {prompt}")
                if question_type == "MCQ":
                    for option_number, option in enumerate(options):
                        option_label = chr(ord("A") + option_number)
                        st.markdown(f"- **{option_label}.** {option}")
                    correct_option = options[ord(answer) - ord("A")]
                    st.markdown(f"**Correct answer: {answer}.** {correct_option}")
            if len(pdf_questions) < 5:
                st.info(
                    "The PDF did not contain enough distinct, complete facts to "
                    "create all five questions without adding unsupported content."
                )
        if pdf_study_notes := st.session_state.get("pdf_study_notes"):
            st.markdown("#### Study Notes")
            with st.container(border=True):
                    for section_title, section_items in pdf_study_notes.items():
                        st.markdown(f"##### {section_title}")
                        if section_items:
                            for item in section_items:
                                st.markdown(f"- {item}")
                        else:
                            st.caption(
                                f"No clear {section_title.lower()} were identified "
                                "in the extracted PDF text."
                            )

if uploaded_file is None:
    st.markdown('<div id="flashcards"></div>', unsafe_allow_html=True)
    st.caption(
        "Flashcards become available after you upload a screenshot and generate "
        "Smart Notes."
    )
    st.markdown('<div class="section-kicker">A simple study workflow</div>', unsafe_allow_html=True)
    st.subheader("What you can do")
    how_columns = st.columns(4)
    feature_cards = [
        ("📸 Smart OCR", "Extract text from screenshots."),
        ("🧠 Smart Notes", "Convert raw text into useful revision notes."),
        ("🔑 Important Points", "Identify concepts and keywords."),
        ("🎯 Exam Mode", "Generate exam-focused questions and MCQs."),
    ]
    for column, (title, copy) in zip(how_columns, feature_cards):
        with column:
            st.markdown(
                f'<div class="info-card"><div class="card-title">{title}</div>'
                f'<p class="card-copy">{copy}</p></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="section-kicker" style="margin-top:1.7rem">From screenshot to study plan</div>',
        unsafe_allow_html=True,
    )
    st.subheader("How StudySnap AI Works")
    step_columns = st.columns(4)
    steps = [
        ("01", "📸", "Upload", "Choose a clear screenshot of your notes."),
        ("02", "🔍", "Extract", "OCR reads and cleans the text."),
        ("03", "🧠", "Organize", "Review the topic, focus, and key takeaways."),
        ("04", "🎯", "Revise", "Practice with exam-focused questions."),
    ]
    for column, (number, icon, title, copy) in zip(step_columns, steps):
        with column:
            st.markdown(
                f'<div class="info-card"><div class="card-step">{number}</div>'
                f'<div class="card-title">{icon} {title}</div>'
                f'<p class="card-copy">{copy}</p></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="footer"><strong>StudySnap AI</strong> · Smart Learning Assistant'
        '<br>Built for smarter exam preparation.</div>',
        unsafe_allow_html=True,
    )
    render_study_dashboard_and_history()
    st.stop()

image_bytes = uploaded_file.getvalue()
image_fingerprint = sha256(image_bytes).hexdigest()
if st.session_state.get("analysis_image") != image_fingerprint:
    for state_key in (
        "analysis",
        "analysis_error",
        "ocr_error",
        "ocr_ready",
        "ocr_raw_text",
        "ocr_cleaned_text",
    ):
        st.session_state.pop(state_key, None)
    st.session_state["analysis_image"] = image_fingerprint

try:
    with Image.open(BytesIO(image_bytes)) as opened_image:
        opened_image.load()
        uploaded_image = ImageOps.exif_transpose(opened_image).convert("RGB")
except (UnidentifiedImageError, OSError, ValueError):
    st.error(
        "This file is not a valid or readable image. Please upload a JPG, JPEG, PNG or WEBP image."
    )
    st.stop()

image_column, intro_column = st.columns([1, 1])
with image_column:
    st.subheader("Your screenshot")
    st.image(uploaded_image, width="stretch")
with intro_column:
    st.markdown('<div class="section-kicker">Step 1 · Review your upload</div>', unsafe_allow_html=True)
    st.subheader("Ready to study smarter?")
    st.write(
        "We’ll extract the text from this image, organize the main ideas, "
        "and prepare revision questions."
    )
    st.info("Select **Analyze screenshot** to extract text locally, then review it before generating notes.")

if st.button("✨ Analyze screenshot", type="primary", width="stretch"):
    st.session_state.pop("analysis", None)
    st.session_state.pop("analysis_error", None)
    st.session_state.pop("ocr_error", None)
    st.session_state["ocr_ready"] = False
    st.session_state["ocr_raw_text"] = ""
    st.session_state["ocr_cleaned_text"] = ""
    if pytesseract is None:
        st.session_state["ocr_error"] = (
            "The pytesseract Python package is missing. Install the project "
            "dependencies with `pip install -r requirements.txt`."
        )
    else:
        try:
            with st.status("Preprocessing the image and running local OCR...", expanded=False) as status:
                raw_text = extract_ocr_text(uploaded_image)
            cleaned_text = clean_text(raw_text)
            st.session_state["ocr_raw_text"] = raw_text
            st.session_state["ocr_cleaned_text"] = cleaned_text
            st.session_state["ocr_ready"] = True
            if cleaned_text:
                status.update(label="OCR complete. Review or edit the text below.", state="complete")
            else:
                status.update(label="No readable text detected. You can enter it below.", state="error")
                st.session_state["ocr_error"] = (
                    "OCR did not find readable text. Try a sharper, brighter screenshot "
                    "with larger text, or enter the text below."
                )
        except pytesseract.TesseractNotFoundError:
            st.session_state["ocr_error"] = (
                "The Tesseract OCR engine was not found. Install Tesseract locally "
                "and make sure it is available on your system PATH."
            )
        except pytesseract.TesseractError:
            st.session_state["ocr_error"] = (
                "Tesseract could not process this image. Try a clearer JPG, JPEG, PNG or WEBP image."
            )
        except (OSError, ValueError):
            st.session_state["ocr_error"] = (
                "The image could not be prepared for OCR. Try another JPG, JPEG, PNG or WEBP image."
            )

if error_message := st.session_state.get("ocr_error"):
    st.error(error_message)

if st.session_state.get("ocr_ready"):
    st.markdown('<div class="section-kicker">Step 2 · Review extracted text</div>', unsafe_allow_html=True)
    st.subheader("Extracted OCR Text")
    raw_tab, cleaned_tab = st.tabs(["Original OCR", "Cleaned text"])
    with raw_tab:
        st.text_area(
            "Text read from image",
            height=250,
            key="ocr_raw_text",
        )
    with cleaned_tab:
        st.text_area(
            "Edit this cleaned text before generating your study material",
            height=250,
            key="ocr_cleaned_text",
        )
    st.caption("The edited cleaned text is used to generate Smart Notes, Important Points and Exam Mode.")
    if st.button(
        "🧠 Generate Smart Notes",
        type="primary",
        width="stretch",
        disabled=not st.session_state.get("ocr_cleaned_text", "").strip(),
    ):
        cleaned_text = st.session_state["ocr_cleaned_text"].strip()
        raw_text = st.session_state.get("ocr_raw_text", "")
        keywords = extract_keywords(cleaned_text)
        important_points = _important_points(cleaned_text, keywords)
        subject = detect_subject(cleaned_text)
        chapter = detect_chapter(cleaned_text)
        title = detect_title(cleaned_text)
        topic = detect_topic(cleaned_text) or subject or "Study Notes"
        questions = make_important_questions(cleaned_text)
        short_questions = make_short_answer_questions(keywords, topic)
        mcqs = make_mcqs(cleaned_text, keywords)
        study_type = _study_type(cleaned_text)
        flashcards = make_flashcards(
            cleaned_text,
            extract_definitions(cleaned_text),
            keywords,
        )
        session_id = str(uuid4())
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        st.session_state["analysis"] = {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "source_text": cleaned_text,
            "topic": topic,
            "title": title or topic,
            "subject": subject,
            "chapter": chapter,
            "study_type": study_type,
            "exam_focus": _exam_focus(cleaned_text, keywords),
            "summary": summarize(cleaned_text),
            "keywords": keywords,
            "important_points": important_points,
            "definitions": extract_definitions(cleaned_text),
            "formulas": extract_formulas(cleaned_text),
            "examples": extract_examples(cleaned_text),
            "questions": questions,
            "short_questions": short_questions,
            "mcqs": mcqs,
            "flashcards": flashcards,
            "history_id": session_id,
            "timestamp": timestamp,
        }
        try:
            save_study_session(
                {
                    "id": session_id,
                    "timestamp": timestamp,
                    "subject": subject,
                    "chapter": chapter,
                    "topic": topic,
                    "title": title or topic,
                    "summary": st.session_state["analysis"]["summary"],
                    "important_points": important_points,
                    "questions": questions,
                    "short_questions": short_questions,
                    "mcqs": mcqs,
                    "keywords": keywords,
                    "definitions": st.session_state["analysis"]["definitions"],
                    "formulas": st.session_state["analysis"]["formulas"],
                    "examples": st.session_state["analysis"]["examples"],
                    "flashcards": flashcards,
                }
            )
            st.session_state["analysis"]["history_saved"] = True
            st.session_state.pop("history_error", None)
            st.session_state["history_message"] = "This study session was saved locally."
        except (OSError, ValueError, json.JSONDecodeError) as error:
            st.session_state["analysis"]["history_saved"] = False
            st.session_state["history_error"] = (
                f"Notes were generated, but the study session could not be saved: {error}"
            )

analysis = st.session_state.get("analysis")
if (
    analysis is None
    or analysis.get("source_text")
    != st.session_state.get("ocr_cleaned_text", "").strip()
):
    st.markdown('<div id="flashcards"></div>', unsafe_allow_html=True)
    st.info(
        "To use Flashcards, analyze a screenshot and generate Smart Notes first."
    )
    render_study_dashboard_and_history()
    st.stop()

st.divider()
st.markdown(
    '<div class="result-banner">✨ Your Study Material is Ready</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="section-kicker">A · Source material</div>', unsafe_allow_html=True)
st.header("🔍 Extracted Text")
image_column, text_column = st.columns([0.82, 1.18], gap="large")
with image_column:
    st.markdown("### 📸 Uploaded Notes")
    with st.container(border=True):
        st.image(uploaded_image, width="stretch")
with text_column:
    st.markdown("### Extracted Text")
    raw_tab, cleaned_tab = st.tabs(["Original OCR", "Cleaned text"])
    with raw_tab:
        st.text_area("Text read from image", analysis["raw_text"], height=250, disabled=True)
    with cleaned_tab:
        st.text_area("Ready-to-study text", analysis["cleaned_text"], height=250, disabled=True)

st.divider()
st.markdown('<div class="section-kicker">B · Your revision overview</div>', unsafe_allow_html=True)
st.header("🧠 Smart Notes")
with st.container(border=True):
    st.markdown("#### Study Details")
    title_column, subject_column, chapter_column = st.columns(3)
    with title_column:
        title_value = st.text_input(
            "Title",
            value=analysis["title"],
            key=f"smart_title_{image_fingerprint}",
        )
    with subject_column:
        subject_value = st.text_input(
            "Subject",
            value=analysis["subject"],
            placeholder="Enter subject if needed",
            key=f"smart_subject_{image_fingerprint}",
        )
    with chapter_column:
        chapter_value = st.text_input(
            "Chapter / Unit",
            value=analysis["chapter"],
            placeholder="Enter chapter if needed",
            key=f"smart_chapter_{image_fingerprint}",
        )
    if analysis.get("history_saved", True):
        try:
            update_study_session_metadata(
                analysis["history_id"],
                title_value,
                subject_value,
                chapter_value,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            st.session_state["history_error"] = (
                f"Could not update saved study details: {error}"
            )
    st.markdown("#### Topic")
    st.metric("Detected Topic", analysis["topic"])
    setting_columns = st.columns(2, gap="large")
    with setting_columns[0]:
        st.markdown("### 📚 Study Type")
        st.info(analysis["study_type"])
    with setting_columns[1]:
        st.markdown("### 🎯 Exam Focus")
        for focus_item in analysis["exam_focus"]:
            st.markdown(f"- {focus_item}")
    st.markdown("### Short Summary")
    st.write(analysis["summary"])
    st.markdown("### 🔑 Important Keywords")
    if analysis["keywords"]:
        st.write(" · ".join(f"`{word}`" for word in analysis["keywords"]))
    else:
        st.write("No keywords could be identified.")
    st.markdown("### Important Concepts")
    if analysis["keywords"]:
        st.write(" · ".join(analysis["keywords"]))
    else:
        st.write("No concepts could be identified from the uploaded content.")
    st.markdown("### Key Definitions")
    if analysis["definitions"]:
        for definition in analysis["definitions"]:
            st.markdown(f"- {definition}")
    else:
        st.write("No explicit definitions found in the uploaded content.")
    st.markdown("### Formulas")
    if analysis["formulas"]:
        for formula in analysis["formulas"]:
            st.code(formula)
    else:
        st.write("No formulas found in the uploaded content.")
    st.markdown("### Examples")
    if analysis["examples"]:
        for example in analysis["examples"]:
            st.markdown(f"- {example}")
    else:
        st.write("No explicit examples found in the uploaded content.")

st.divider()
st.markdown('<div class="section-kicker">C · Key takeaways</div>', unsafe_allow_html=True)
st.header("📌 Important Points")
with st.container(border=True):
    for number, point in enumerate(analysis["important_points"], 1):
        st.markdown(f"**{number}.** {point}")
    st.markdown("### Quick Revision")
    st.markdown("- Review the important points above.")
    st.markdown("- Cover each keyword and explain it from memory.")
    st.markdown("- Try the Exam Mode questions without looking at your notes.")

st.divider()
st.markdown('<div class="section-kicker">Last-minute study</div>', unsafe_allow_html=True)
st.header("⚡ Quick Revision")
if analysis["definitions"]:
    st.markdown("**Key definitions**")
    for definition in analysis["definitions"][:4]:
        st.markdown(f"- {definition}")
st.markdown("**Important concepts & keywords**")
revision_concepts = list(dict.fromkeys(analysis["keywords"] + analysis["exam_focus"]))[:8]
if revision_concepts:
    st.write(" · ".join(revision_concepts))
else:
    st.caption("No concepts or keywords were identified in the uploaded content.")
if analysis["formulas"]:
    st.markdown("**Formulas**")
    for formula in analysis["formulas"][:4]:
        st.code(formula)
st.markdown("**Important facts**")
for point in analysis["important_points"][:4]:
    st.markdown(f"- {point}")
st.markdown("**Important questions**")
for number, question in enumerate(analysis["questions"][:3], 1):
    st.markdown(f"{number}. {question}")

st.markdown('<div id="flashcards"></div>', unsafe_allow_html=True)
st.markdown("### 🗂️ Flashcard Mode")
st.caption("Review concepts from your notes; reveal the answer, then move between cards.")
flashcards = analysis["flashcards"]
if flashcards:
    flashcard_index_key = f"flashcard_index_{analysis['history_id']}"
    flashcard_index = st.session_state.get(flashcard_index_key, 0) % len(flashcards)
    st.session_state[flashcard_index_key] = flashcard_index
    active_card = flashcards[flashcard_index]
    st.caption(f"Card {flashcard_index + 1} of {len(flashcards)}")
    with st.container(border=True):
        st.markdown("**Question / Concept**")
        st.write(active_card["question"])
        answer_key = f"flashcard_answer_{analysis['history_id']}_{flashcard_index}"
        st.button(
            "Show Answer",
            key=f"flashcard_show_{analysis['history_id']}_{flashcard_index}",
            on_click=lambda key=answer_key: st.session_state.update({key: True}),
        )
        if st.session_state.get(answer_key):
            st.markdown("**Answer**")
            st.write(active_card["answer"])
    previous_column, next_column = st.columns(2)
    with previous_column:
        st.button(
            "← Previous",
            key=f"flashcard_previous_{analysis['history_id']}",
            on_click=_move_flashcard,
            args=(flashcard_index_key, len(flashcards), -1),
        )
    with next_column:
        st.button(
            "Next →",
            key=f"flashcard_next_{analysis['history_id']}",
            on_click=_move_flashcard,
            args=(flashcard_index_key, len(flashcards), 1),
        )
else:
    st.info("No source-backed flashcards could be created from this content.")

st.divider()
st.markdown(
    '<div class="exam-hero"><h2>🎯 Exam Mode</h2>'
    '<p>Focus on what matters most for your exam.</p></div>',
    unsafe_allow_html=True,
)
count_columns = st.columns(3)
for column, (value, title) in zip(
    count_columns,
    (("5", "Important Questions"), ("3", "Short Answers"), ("3", "MCQs")),
):
    with column:
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">{value}</div>'
            f'<div class="stat-label">{title}</div></div>',
            unsafe_allow_html=True,
        )
st.markdown("Work through each question before revealing an MCQ answer.")
st.subheader("🔥 Important Questions")
for number, question in enumerate(analysis["questions"], 1):
    with st.container(border=True):
        st.markdown(f'<span class="question-label">Q{number}</span>', unsafe_allow_html=True)
        st.markdown(question)

short_column, mcq_column = st.columns([0.9, 1.1])
with short_column:
    st.subheader("⚡ Short-Answer Questions")
    for number, question in enumerate(analysis["short_questions"], 1):
        with st.container(border=True):
            st.markdown(f'<span class="question-label">Q{number}</span>', unsafe_allow_html=True)
            st.markdown(question)
with mcq_column:
    st.subheader("🧩 Quick MCQ Challenge")
    for number, item in enumerate(analysis["mcqs"], 1):
        with st.container(border=True):
            st.markdown(f'<span class="question-label">MCQ {number}</span>', unsafe_allow_html=True)
            st.markdown(item["question"])
            for letter, option in zip("ABC", item["options"]):
                st.markdown(f"**{letter}.** {option}")
            st.markdown("**D.** None of the above")
            with st.expander(f"Show answer to MCQ {number}"):
                st.write(item["answer"])

notes_text = build_notes_text(
    topic=analysis["topic"],
    title=title_value,
    subject=subject_value,
    chapter=chapter_value,
    study_type=analysis["study_type"],
    exam_focus=analysis["exam_focus"],
    concepts=analysis["keywords"],
    summary=analysis["summary"],
    keywords=analysis["keywords"],
    important_points=analysis["important_points"],
    definitions=analysis["definitions"],
    formulas=analysis["formulas"],
    examples=analysis["examples"],
    questions=analysis["questions"],
    short_questions=analysis["short_questions"],
    mcqs=analysis["mcqs"],
)
quick_revision = []
quick_revision.extend(analysis["definitions"][:4])
if revision_concepts:
    quick_revision.append("Concepts and keywords: " + ", ".join(revision_concepts))
quick_revision.extend(analysis["formulas"][:4])
quick_revision.extend(analysis["important_points"][:4])
quick_revision.extend(analysis["questions"][:3])
markdown_text = build_markdown_notes(
    title=title_value,
    subject=subject_value,
    chapter=chapter_value or analysis["topic"],
    summary=analysis["summary"],
    keywords=analysis["keywords"],
    important_points=analysis["important_points"],
    quick_revision=quick_revision,
    questions=analysis["questions"],
    short_questions=analysis["short_questions"],
    mcqs=analysis["mcqs"],
)

st.markdown("### 📝 Notes Export")
save_column, text_column, markdown_column, pdf_column = st.columns(4)
with save_column:
    if st.button("💾 Save notes to output/", width="stretch"):
        try:
            output_directory = _user_output_directory()
            output_directory.mkdir(parents=True, exist_ok=True)
            notes_path = output_directory / "studysnap_notes.txt"
            notes_path.write_text(notes_text, encoding="utf-8")
            st.success(f"Notes saved to {notes_path.relative_to(PROJECT_ROOT)}")
        except OSError as error:
            st.error(f"Could not save the notes file: {error}")
with text_column:
    st.download_button(
        "⬇️ Download TXT",
        data=notes_text,
        file_name="studysnap_notes.txt",
        mime="text/plain",
        width="stretch",
    )
with markdown_column:
    st.download_button(
        "⬇️ Download Markdown",
        data=markdown_text,
        file_name="studysnap_notes.md",
        mime="text/markdown",
        width="stretch",
    )
with pdf_column:
    try:
        pdf_bytes = build_pdf_notes(markdown_text)
        st.download_button(
            "⬇️ Download PDF",
            data=pdf_bytes,
            file_name="studysnap_notes.pdf",
            mime="application/pdf",
            width="stretch",
        )
    except (ImportError, OSError, ValueError) as error:
        st.error(f"Could not prepare the local PDF export: {error}")

render_study_dashboard_and_history()
st.markdown(
    '<div class="footer"><strong>StudySnap AI</strong> · Smart Learning Assistant'
    '<br>Built for smarter exam preparation.</div>',
    unsafe_allow_html=True,
)
