from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


FACT_LINE = re.compile(r"^-\s*([^:]+):\s*(.+?)\s*$")
CAPITAL_LINE = re.compile(
    r"^-\s*국가:\s*(.+?)\s*\|\s*수도:\s*(.+?)\s*\|\s*별칭:\s*(.+?)\s*$"
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?요다])\s+|\n+")
QUESTION_STOPWORDS = {
    "나",
    "내",
    "너",
    "개인",
    "자료",
    "정보",
    "질문",
    "답변",
    "뭐",
    "뭐야",
    "무엇",
    "무엇인가요",
    "어떤",
    "어떻게",
    "알려줘",
    "알려주세요",
    "궁금해",
    "대해서",
}

PROFILE_FACT_ALIASES = (
    ("우편번호", ("우편번호",)),
    ("전체 주소", ("주소", "사는곳", "거주지", "어디살", "어디에살", "어디서살", "어디에서살")),
    ("보호자 휴대전화", ("보호자연락처", "보호자전화", "보호자휴대폰")),
    ("집전화번호", ("집전화",)),
    ("휴대전화", ("휴대전화", "휴대폰", "핸드폰", "전화번호", "연락처")),
    ("이메일", ("이메일", "메일주소", "메일")),
    ("생년월일", ("생년월일", "생일", "태어난날", "출생일")),
    ("학번", ("학번",)),
    ("과정", ("과정", "전공심화")),
    ("학과", ("학과", "전공")),
    ("학교", ("학교", "대학교", "대학어디", "어느대학")),
    ("학적 상태", ("학적상태", "재학상태", "재학중", "재학여부")),
    ("학년", ("몇학년", "현재학년")),
    ("반", ("몇반", "소속반")),
)

MODEL_QUERY_TERMS = (
    "모델",
    "llm",
    "ollama",
    "올라마",
    "오라마",
    "rag",
    "레그",
    "검색증강생성",
)

SOCIAL_MESSAGE_TOKENS = {
    "안녕",
    "안녕하세요",
    "반가워",
    "반갑습니다",
    "하이",
    "hello",
    "hi",
    "고마워",
    "고맙습니다",
    "감사합니다",
}

SENSITIVE_FACT_QUERIES = {
    "학번": ("학번",),
    "생년월일": ("생년월일", "생일", "태어난"),
    "이메일": ("이메일", "메일"),
    "휴대전화": ("휴대전화", "휴대폰", "핸드폰", "전화번호", "연락처"),
    "집전화번호": ("집전화",),
    "보호자 휴대전화": ("보호자",),
    "우편번호": ("우편번호",),
    "기본 주소": ("주소", "사는 곳", "거주지", "어디 살아", "어디에 살아", "어디서 살아", "어디에서 살아"),
    "상세 주소": ("주소", "사는 곳", "거주지", "어디 살아", "어디에 살아", "어디서 살아", "어디에서 살아"),
    "전체 주소": ("주소", "사는 곳", "거주지", "어디 살아", "어디에 살아", "어디서 살아", "어디에서 살아"),
}
SENSITIVE_PROFILE_FIELDS = frozenset(SENSITIVE_FACT_QUERIES)

GRADE_SUMMARY_QUESTION = re.compile(
    r"^\s*([1-4])\s*학년(?:\s*때)?\s*(?:전체\s*)?"
    r"(?:학점|성적|평점|gpa)(?:이|은|는|을|를|가)?\s*"
    r"(?:뭐야|어때|알려줘|알려주세요|궁금해|인가요|입니까)?\s*[?!.~]*\s*$",
    re.IGNORECASE,
)

ENGLISH_TRANSLATION_PREFIX = re.compile(
    r"^\s*영어(?:로)?\s*(?:번역|변환|바꿔)"
    r"(?:해줘|해주세요|해주겠어|해줄래|해|줘|주세요)?\s*[:：-]?\s*(.*?)\s*$",
    re.IGNORECASE,
)
ENGLISH_TRANSLATION_SUFFIX = re.compile(
    r"^\s*(.+?)\s*영어(?:로)?\s*(?:번역|변환|바꿔)"
    r"(?:해줘|해주세요|해주겠어|해줄래|해|줘|주세요)?\s*[?!.~]*\s*$",
    re.IGNORECASE,
)
TRANSLATION_REFERENCE_WORDS = {
    "",
    "이거",
    "이것",
    "이문장",
    "이내용",
    "위내용",
    "방금답변",
    "직전답변",
    "이답변",
    "답변",
}


def load_profile_facts(path: Path) -> dict[str, str]:
    facts: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = FACT_LINE.match(raw_line.strip())
        if match:
            facts[match.group(1).strip()] = match.group(2).strip()
    return facts


def _normalized(question: str) -> str:
    return re.sub(r"[\s!?.,~]+", "", question).lower()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    normalized_text = _normalized(text)
    return any(_normalized(term) in normalized_text for term in terms)


def _question_keywords(question: str) -> list[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9_]+", question.lower())
    keywords: list[str] = []
    endings = ("인가요", "입니까", "인가", "이야", "야", "은", "는", "이", "가", "을", "를")

    for token in tokens:
        if token in QUESTION_STOPWORDS:
            continue
        stem = token
        for ending in endings:
            if stem.endswith(ending) and len(stem) - len(ending) >= 2:
                stem = stem[: -len(ending)]
                break
        if stem and stem not in QUESTION_STOPWORDS:
            keywords.append(stem)
    return keywords


def context_supports_question(question: str, contexts: list[str]) -> bool:
    """Require at least one meaningful question keyword to occur in retrieved text."""
    corpus = "\n".join(contexts).lower().replace(" ", "")
    profile_field = requested_profile_field(question)
    if profile_field and _normalized(profile_field) in corpus:
        return True
    if is_model_info_question(question) and (
        "ollama등록모델명" in corpus or "모델형식" in corpus
    ):
        return True

    keywords = _question_keywords(question)
    if not keywords:
        return False
    return any(keyword.replace(" ", "") in corpus for keyword in keywords)


def has_meaningful_question(question: str) -> bool:
    return bool(_question_keywords(question))


def is_social_message(question: str) -> bool:
    """Recognize a short social message without storing a canned response."""
    tokens = re.findall(r"[가-힣A-Za-z]+", question.lower())
    return 0 < len(tokens) <= 3 and all(token in SOCIAL_MESSAGE_TOKENS for token in tokens)


def is_english_translation_request(question: str) -> bool:
    """Recognize an explicit request to translate text into English."""
    normalized_question = _normalized(question)
    return "영어" in normalized_question and any(
        term in normalized_question for term in ("번역", "변환", "바꿔")
    )


def extract_english_translation_text(question: str) -> str:
    """Extract inline source text; an empty result means use the prior answer."""
    if not is_english_translation_request(question):
        return ""

    match = ENGLISH_TRANSLATION_PREFIX.match(question)
    explicit_source = bool(match and match.group(1).strip())
    if match:
        candidate = match.group(1).strip()
    else:
        match = ENGLISH_TRANSLATION_SUFFIX.match(question)
        candidate = match.group(1).strip() if match else ""

    candidate = candidate.strip()
    # Explicitly quoted source is literal even when its text is '이거'.
    quoted = re.fullmatch(r'''["'“‘](.*)["'”’](?:을|를)?''', candidate, re.DOTALL)
    if quoted:
        return quoted.group(1)
    # A particle is removable only for a recognized reference, not arbitrary
    # source text: '가을' and '나는 너를' must never lose their final character.
    normalized_candidate = _normalized(candidate)
    if not explicit_source and (normalized_candidate in TRANSLATION_REFERENCE_WORDS or (
        candidate.endswith(("을", "를"))
        and _normalized(candidate[:-1]) in TRANSLATION_REFERENCE_WORDS
    )):
        return ""
    return candidate


def requested_grade_summary(question: str) -> str | None:
    """Return an explicitly requested whole-grade scope, never an answer value."""
    normalized_question = _normalized(question)
    grade_match = re.match(r"^([1-4])학년", normalized_question)
    if not grade_match or "1학기" in normalized_question or "2학기" in normalized_question:
        return None

    metric_match = re.search(r"학점|성적|평점|평균|gpa", normalized_question)
    if not metric_match:
        return None

    scope_text = normalized_question[grade_match.end() : metric_match.start()]
    if scope_text not in {"", "때", "전체", "총", "총합", "연간", "누적"}:
        return None

    request_text = normalized_question[metric_match.end() :]
    filler_terms = (
        "알려주세요",
        "알려줘",
        "알려줄래",
        "보여주세요",
        "보여줘",
        "어떻게되나요",
        "어떻게돼",
        "어때요",
        "어때",
        "몇점이야",
        "몇점",
        "얼마인가요",
        "얼마야",
        "궁금해",
        "뭐야",
        "인가요",
        "입니까",
        "이야",
        "이",
        "은",
        "는",
        "을",
        "를",
        "가",
    )
    for filler in filler_terms:
        request_text = request_text.replace(filler, "")
    return f"{grade_match.group(1)}학년" if not request_text else None


def context_explicitly_reports_missing(contexts: list[str]) -> bool:
    """Detect an explicit no-record fact in an already scope-matched document."""
    markers = (
        "기록 보유 여부: 없음",
        "성적표는 포함되지 않았다",
        "취득학점: 자료 없음",
        "평점평균(gpa): 자료 없음",
    )
    corpus = "\n".join(contexts).lower()
    return any(marker in corpus for marker in markers)


def _academic_search_key(value: str) -> str:
    return re.sub(r"[^가-힣a-z0-9+#]+", "", unicodedata.normalize("NFKC", value).lower())


def question_matches_course_document(question: str, document: str) -> bool:
    """Match a course name/alias in a fact document to the user's question."""
    normalized_question = _academic_search_key(question)
    values: list[str] = []
    for line in document.splitlines():
        if line.startswith("- 과목명:"):
            values.append(line.split(":", 1)[1].strip())
        elif line.startswith("- 과목명 검색 표기:"):
            values.extend(line.split(":", 1)[1].split(","))

    for value in values:
        normalized_value = _academic_search_key(value)
        # Preserve Roman/numeric suffix boundaries: JAVA I is not JAVA II,
        # C++ I is not C++ II, and course 1 is not course 11.
        pattern = re.escape(normalized_value) + r"(?![a-z0-9+#])"
        if normalized_value and re.search(pattern, normalized_question):
            return True
    return False


def focused_course_context(question: str, document: str) -> str:
    """Keep only the requested field from a retrieved course fact document."""
    normalized_question = question.lower()
    if "등급" in normalized_question or "성적" in normalized_question:
        requested_prefixes = ("- 과목명:", "- 등급:")
    elif "학점" in normalized_question:
        requested_prefixes = ("- 과목명:", "- 취득학점:")
    elif "평점" in normalized_question or "gpa" in normalized_question:
        requested_prefixes = ("- 과목명:", "- 평점:")
    else:
        return document

    selected = [
        line for line in document.splitlines() if line.startswith(requested_prefixes)
    ]
    return "\n".join(selected) if selected else document


def focused_course_fact_answer(question: str, document: str) -> str:
    """Render an exact course field without letting the LLM change its value."""
    facts = _facts_from_documents([document])
    course_name = facts.get("과목명", "")
    if not course_name:
        return ""
    normalized_question = _normalized(question)
    if "등급" in normalized_question or "성적" in normalized_question:
        value = facts.get("등급", "")
        return f"{course_name}의 성적은 {value}입니다." if value else ""
    if "학점" in normalized_question:
        value = facts.get("취득학점", "")
        return f"{course_name}은 {value}학점을 취득했습니다." if value else ""
    if "평점" in normalized_question or "gpa" in normalized_question:
        value = facts.get("평점", "")
        return f"{course_name}의 평점은 {value}입니다." if value else ""
    return ""


def is_profile_summary_question(question: str, person_name: str) -> bool:
    """Recognize a request to introduce the person named in the profile data."""
    normalized_question = _normalized(question)
    normalized_name = _normalized(person_name)
    if not normalized_name or normalized_name not in normalized_question:
        return False
    if requested_profile_field(question):
        return False
    remainder = normalized_question.replace(normalized_name, "", 1)
    filler_terms = (
        "에대해서",
        "에대해",
        "대해서",
        "대해",
        "설명해주세요",
        "설명해줘",
        "설명",
        "소개해주세요",
        "소개해줘",
        "소개해줄래",
        "소개",
        "알려주세요",
        "알려줘",
        "알려줄래",
        "말해주세요",
        "말해줘",
        "말해줄래",
        "어떤사람이야",
        "어떤사람",
        "누구인가요",
        "누구야",
        "누구",
        "좀",
        "는",
        "은",
        "이",
        "가",
    )
    for filler in filler_terms:
        remainder = remainder.replace(filler, "")
    return not remainder


def requested_profile_field(question: str) -> str | None:
    """Map natural Korean variants to a profile field, without storing its value."""
    normalized_question = _normalized(question)
    for field, aliases in PROFILE_FACT_ALIASES:
        if any(_normalized(alias) in normalized_question for alias in aliases):
            return field
    return None


def is_sensitive_profile_field(field: str | None) -> bool:
    return bool(field and field in SENSITIVE_PROFILE_FIELDS)


def is_model_info_question(question: str) -> bool:
    """Recognize questions about the chatbot model or RAG configuration."""
    normalized_question = _normalized(question)
    return any(_normalized(term) in normalized_question for term in MODEL_QUERY_TERMS)


def is_capital_question(question: str) -> bool:
    """Recognize questions that ask for a capital or a capital's country."""
    normalized_question = _normalized(question)
    return "수도" in normalized_question or "capital" in normalized_question


def focused_capital_answer(question: str, documents: list[str]) -> str:
    """Find a country/capital pair in retrieved RAG documents and render it."""
    normalized_question = _normalized(question)
    records: list[tuple[str, str, list[str]]] = []
    for document in documents:
        for raw_line in document.splitlines():
            match = CAPITAL_LINE.match(raw_line.strip())
            if not match:
                continue
            country = match.group(1).strip()
            capital = match.group(2).strip()
            aliases = [
                alias.strip()
                for alias in match.group(3).split(",")
                if alias.strip()
            ]
            records.append((country, capital, aliases))

    country_matches: list[tuple[int, str, str]] = []
    for country, capital, aliases in records:
        for name in [country, *aliases]:
            normalized_name = _normalized(name)
            if normalized_name and normalized_name in normalized_question:
                country_matches.append((len(normalized_name), country, capital))
    if country_matches:
        _, country, capital = max(country_matches, key=lambda item: item[0])
        return f"{country}의 수도는 {capital}입니다."

    capital_matches: list[tuple[int, str, str]] = []
    for country, capital, _ in records:
        normalized_capital = _normalized(capital)
        if normalized_capital and normalized_capital in normalized_question:
            capital_matches.append((len(normalized_capital), country, capital))
    if capital_matches:
        _, country, capital = max(capital_matches, key=lambda item: item[0])
        return f"{capital}는 {country}의 수도입니다."
    return ""


def _facts_from_documents(documents: list[str]) -> dict[str, str]:
    facts: dict[str, str] = {}
    for document in documents:
        for raw_line in document.splitlines():
            match = FACT_LINE.match(raw_line.strip())
            if match:
                facts[match.group(1).strip()] = match.group(2).strip()
    return facts


def focused_profile_fact_answer(question: str, documents: list[str]) -> str:
    """Render one explicitly requested profile fact retrieved from FAISS."""
    field = requested_profile_field(question)
    if not field:
        return ""
    facts = _facts_from_documents(documents)
    value = facts.get(field, "")
    if not value:
        return ""

    templates = {
        "전체 주소": "주소는 {value}입니다.",
        "우편번호": "우편번호는 {value}입니다.",
        "휴대전화": "휴대전화 번호는 {value}입니다.",
        "집전화번호": "집전화 번호는 {value}입니다.",
        "보호자 휴대전화": "보호자 휴대전화 번호는 {value}입니다.",
        "이메일": "이메일은 {value}입니다.",
        "생년월일": "생년월일은 {value}입니다.",
        "학번": "학번은 {value}입니다.",
        "학교": "학교는 {value}입니다.",
        "학과": "학과는 {value}입니다.",
        "학적 상태": "학적 상태는 {value}입니다.",
        "과정": "과정은 {value}입니다.",
        "학년": "현재 {value}입니다.",
        "반": "소속 반은 {value}입니다.",
    }
    return templates.get(field, "{value}입니다.").format(value=value)


def focused_model_answer(documents: list[str]) -> str:
    """Render model identity from the retrieved model-information document."""
    facts = _facts_from_documents(documents)
    model_name = facts.get("Ollama 등록 모델명", "")
    model_format = facts.get("모델 형식", "")
    execution = facts.get("실행 방식", "")
    structure = facts.get("구조", "")
    if not model_name:
        return ""

    answer = f"이 챗봇은 {execution or '로컬 Ollama'}에 등록된 {model_name}"
    if model_format:
        answer += f" ({model_format})"
    answer += " 모델을 사용합니다."
    if structure:
        answer += f" 답변은 {structure} 구조로 문서를 검색한 뒤 생성합니다."
    return answer


def focused_grade_summary_answer(question: str, documents: list[str]) -> str:
    """Render an annual grade summary from the retrieved grade document."""
    grade_scope = requested_grade_summary(question)
    if not grade_scope:
        return ""
    facts = _facts_from_documents(documents)
    if facts.get("성적 기록 보유 여부") == "없음":
        return ""

    credits = facts.get("연간 취득학점", "")
    first_semester = facts.get("1학기", "")
    second_semester = facts.get("2학기", "")
    normalized_question = _normalized(question)

    if "학점" in normalized_question and "성적" not in normalized_question:
        return f"{grade_scope} 연간 취득학점은 {credits}학점입니다." if credits else ""

    parts: list[str] = []
    if credits:
        parts.append(f"{grade_scope} 연간 취득학점은 {credits}학점입니다.")
    if first_semester:
        parts.append(f"1학기는 {first_semester}입니다.")
    if second_semester:
        parts.append(f"2학기는 {second_semester}입니다.")
    return " ".join(parts)


SEMESTER_FACT_QUERY = re.compile(
    r"^(?:내가|내|제)?(?P<year>\d{4})(?:학년도|년도|년)"
    r"(?P<semester>[12])학기(?:에(?:는|내가)?|의|때)?"
    r"(?P<metric>취득(?:한)?학점|이수(?:한)?학점|신청(?:한)?학점|학점|"
    r"평점평균|평균평점|평점|gpa|백분위(?:점수)?)"
    r"(?:은|는|이|가|을|를)?"
    r"(?:알려줘|알려주세요|알려줄래|보여줘|보여주세요|궁금해|뭐야|"
    r"몇(?:학점|점)?(?:이야|인가요|인지알려줘)?|얼마(?:야|인가요)?)?$"
)


def focused_semester_fact_answer(question: str, records_path: Path) -> str | None:
    """Return a recorded semester total for a specifically requested academic term."""
    normalized = re.sub(
        r"몇학점(?:을)?(?:취득|이수)했(?:어|나요|습니까)$", "취득학점", _normalized(question)
    )
    match = SEMESTER_FACT_QUERY.fullmatch(normalized)
    if match is None:
        return None

    year, semester, metric = match.group("year", "semester", "metric")
    records = json.loads(records_path.read_text(encoding="utf-8-sig"))
    record = next(
        (term for term in records["semester_summaries"]
         if str(term["academic_year"]) == year and str(term["semester"]) == semester),
        {},
    )
    if metric.startswith("신청"):
        field, label, unit = "applied_credits", "신청학점", "학점"
    elif metric.startswith("백분위"):
        field, label, unit = "percentile", "백분위 성적", "점"
    elif metric in {"평점평균", "평균평점", "평점", "gpa"}:
        field, label, unit = "gpa", "평점평균", "점"
    else:
        field, label, unit = "earned_credits", "취득학점", "학점"

    term = f"{year}학년도 {semester}학기"
    value = record.get(field)
    if value is None:
        return f"제공된 자료에 {term} {label} 기록이 없어요."
    return f"{term} {label}은 {str(value).removesuffix('.0')}{unit}입니다."

CREDIT_YEAR = re.compile(r"(?<!\d)(20\d{2})(?:학년도|년도|년)")
CREDIT_GRADE = re.compile(r"(?<!\d)([1-4])학년(?!도)")
CREDIT_TERM = re.compile(r"(?<!\d)([12])학기")
CREDIT_CATEGORY = re.compile(r"전공필수|전공선택|교양필수|교양선택|전공|교양")
CREDIT_FILLERS = re.compile(
    r"지금까지|현재까지|그동안|기록된|들었던|수강한|취득한|신청한|이수한|"
    r"알려주세요|알려줘요|알려줘|알려줄래|보여주세요|보여줘|확인해줘|"
    r"얼마인가요|얼마예요|얼마야|얼마|몇학점인지|몇학점|몇점이야|몇인가요|몇이야|몇|"
    r"땄나요|땄어|받았어|받은|들었어|들었나요|했어|했나요|있는지|있어|되는지|돼|"
    r"연간|전체|누적|총합|합계|모두|총|학점|취득|신청|이수|딴|과목|수강|"
    r"에내가|내가|나는|나의|내|제|그리고|랑|하고|까지|에서|동안|때|"
    r"인가요|이에요|이야|뭐야|궁금해|인지|은|는|이|가|을|를|에|의|과|와|도"
)


def focused_credit_answer(question: str, records_path: Path) -> str | None:
    """Answer common credit totals from the private transcript's recorded fields."""
    text = _academic_search_key(question)
    if "학점" not in text or re.search(r"평점|평균|gpa|백분위|등급|성적|영어|번역|졸업|장학금", text):
        return None

    records = json.loads(records_path.read_text(encoding="utf-8-sig"))
    course_aliases = []
    for course in records["courses"]:
        name = course["course_name"]
        aliases = {name, name.replace("(II)", "2").replace("(I)", "1"), course["course_code"]}
        for key in {_academic_search_key(alias) for alias in aliases}:
            if key and re.search(re.escape(key) + r"(?![a-z0-9+#])", text):
                course_aliases.append((key, course))

    course_key = max((key for key, _ in course_aliases), key=len, default="")
    remainder = text.replace(course_key, "", 1) if course_key else text
    years = list(CREDIT_YEAR.finditer(remainder))
    grades = list(CREDIT_GRADE.finditer(remainder))
    terms = list(CREDIT_TERM.finditer(remainder))
    categories = list(CREDIT_CATEGORY.finditer(remainder))
    through = any(remainder[match.end():].startswith("까지") for match in (*years, *grades, *terms))
    for pattern in (CREDIT_YEAR, CREDIT_GRADE, CREDIT_TERM, CREDIT_CATEGORY):
        remainder = pattern.sub("", remainder)
    if CREDIT_FILLERS.sub("", remainder):
        return None
    if any(len(matches) > 1 for matches in (years, grades, terms, categories)):
        return None

    year = int(years[0][1]) if years else None
    grade = f"{grades[0][1]}학년" if grades else None
    semester = int(terms[0][1]) if terms else None
    category = categories[0][0] if categories else None
    if grade:
        summaries = records_path.parent / "academic_facts"
        for summary in summaries.glob("*_year_summary.md"):
            facts = load_profile_facts(summary)
            if facts.get("학년") == grade:
                mapped_year = int(facts["학년도"])
                if year is not None and year != mapped_year:
                    return f"원본 자료에서 {grade} 기록은 {mapped_year}학년도에 해당해요."
                year = mapped_year
                break
        else:
            return f"원본 자료에는 {grade} 학점 기록이 없어요."
    if semester and year is None and not course_key:
        return f"{semester}학기의 학점을 확인하려면 학년도나 학년을 알려 주세요."

    if course_key:
        rows = [course for key, course in course_aliases if key == course_key]
    elif category:
        rows = [course for course in records["courses"] if course["category"].startswith(category)]
    elif year is None:
        rows = [records["cumulative"]]
    else:
        rows = records["semester_summaries"]

    if year is not None or semester:
        if through and year is not None:
            rows = [row for row in rows
                    if (row["academic_year"], row["semester"]) <= (year, semester or 2)]
        else:
            rows = [row for row in rows
                    if (year is None or row["academic_year"] == year)
                    and (semester is None or row["semester"] == semester)]

    period = " ".join(part for part in (
        grade or (f"{year}학년도" if year is not None else ""),
        f"{semester}학기" if semester else "",
    ) if part)
    if through and period:
        period += "까지 누적"
    scope = " ".join(part for part in (period, category or "") if part)
    if not rows:
        return f"원본 자료에 {scope or '해당 과목의'} 학점 기록이 없어요."
    if course_key:
        if len(rows) != 1:
            return "같은 과목이 여러 학기에 있어요. 확인할 학년이나 학기를 알려 주세요."
        scope = " ".join(part for part in (scope, rows[0]["course_name"]) if part)

    use_applied = "신청" in text
    use_earned = not use_applied or bool(re.search(r"취득|이수|딴|땄|받은|받았", text))
    answers = []
    if use_applied:
        answers.append(f"신청학점은 {sum(row['applied_credits'] for row in rows)}학점")
    if use_earned:
        answers.append(f"취득학점은 {sum(row['earned_credits'] for row in rows)}학점")
    return f"{scope or '누적'} {', '.join(answers)}입니다."


def focused_public_profile_context(documents: list[str]) -> str:
    """Keep only non-sensitive introduction fields from retrieved profile chunks."""
    allowed_fields = {
        "공개 소개",
        "이름",
        "학교",
        "소속",
        "학과",
        "학적 상태",
        "과정",
        "학년",
        "주야 구분",
        "정원 구분",
    }
    selected: list[str] = []
    for document in documents:
        for raw_line in document.splitlines():
            match = FACT_LINE.match(raw_line.strip())
            if match and match.group(1).strip() in allowed_fields:
                line = f"- {match.group(1).strip()}: {match.group(2).strip()}"
                if line not in selected:
                    selected.append(line)
    introduction = [line for line in selected if line.startswith("- 공개 소개:")]
    if introduction:
        return introduction[0].split(":", 1)[1].strip()
    return "\n".join(selected)


def protect_sensitive_answer(
    answer: str,
    question: str,
    facts: dict[str, str],
    allow_requested_sensitive: bool = True,
) -> str:
    """Remove sentences containing profile secrets that the question did not request."""
    q = _normalized(question)
    forbidden_values: list[str] = []

    for fact_name, query_terms in SENSITIVE_FACT_QUERIES.items():
        value = facts.get(fact_name)
        if value and (
            not allow_requested_sensitive or not _contains_any(q, query_terms)
        ):
            forbidden_values.append(value)

    safe_sentences = []
    for sentence in SENTENCE_SPLIT.split(answer.strip()):
        cleaned = sentence.strip()
        if cleaned and not any(value in cleaned for value in forbidden_values):
            safe_sentences.append(cleaned)

    if not safe_sentences:
        return "질문과 직접 관련된 정보가 확인되지 않았어요. 질문을 조금 더 구체적으로 입력해 주세요."
    safe_answer = " ".join(safe_sentences)
    return (
        safe_answer.replace("교수님 공유 ", "")
        .replace("교수님이 공유한 ", "")
        .replace("교수님께서 공유한 ", "")
    )
