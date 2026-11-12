"""Check explicit group references and scoped English causal wording.

These are controlled-language rules, not an unrestricted semantic classifier.
"""

import ast
import json
import re

from app.tool_contracts import ClaimIssue

_STRING = r'''(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\u201c[^\u201d]*\u201d|\u2018[^\u2019]*\u2019)'''
_VALUE = _STRING + r'|true\b|false\b|null\b|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w.])'
_REFERENCE = re.compile(r'(?P<attribute>' + _STRING + r'|[A-Za-z_]\w*)\s*[:=]\s*(?P<value>' + _VALUE + r')')
_QUOTED = re.compile(r'(?<!\w)' + _STRING + r'(?!\w)')
_ATTRIBUTE_CONTEXT = re.compile(
    r'(?=(?:(?i:\battribute)\s+(?P<after>[A-Za-z_]\w*)\b'
    r'|\b(?P<before>[A-Za-z_]\w*)\s+(?i:attribute)\b))'
)
_BARE_GROUP = re.compile(
    # Look ahead without consuming words: ignoring "The group" must not hide
    # the overlapping "group Martian" reference that starts at its second word.
    r'(?=(?:\b(?i:groups?)\s+(?:(?i:named|called)\s+)?(?P<after>[A-Z][\w/-]*|\[[^\n]*?[)\]])'
    r'|\b(?P<before>[A-Z][\w/-]*)\s+(?i:groups?)\b))'
)
_IDENTIFIERS = {
    "outcome", "prediction", "score", "base_rate", "selection_rate", "tpr", "fpr", "ppv", "auroc",
    "demographic_parity_difference", "equalized_odds_difference", "ppv_difference",
    "count", "p_value", "statistic", "severity", "substantial", "watch", "no_evidence_found", "unavailable",
}
_STATISTICAL_ANNOTATIONS = {"p", "P", "CI", "ci", "n", "N", "AUC", "AUROC", "TPR", "FPR", "PPV"}
_NUMBER_WORDS = dict(zip(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
     "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"],
    range(21), strict=True,
))
_WORD_COUNT = re.compile(
    r"\b(?P<number>" + "|".join(_NUMBER_WORDS) + r")\s+"
    r"(?:(?:eligible|reportable|suppressed|mapped|group|youngest|oldest|age|demographic|payer_code|race|gender)\s+){0,3}"
    r"(?:groups|attributes|rows|resamples)\b", re.IGNORECASE,
)
_GROUP_DETERMINERS = {
    "the", "this", "that", "each", "every", "any", "all", "no", "a", "these", "those", "some",
    "certain", "several", "many", "few", "both", "multiple", "various",
}
_GROUP_DESCRIPTORS = {
    "suppressed", "excluded", "included", "eligible", "ineligible", "reportable", "unreportable",
    "small", "smaller", "large", "larger", "demographic", "protected", "reference", "comparison",
    "available", "unavailable", "underrepresented", "minority", "majority",
}
_CAUSAL = re.compile(
    r"\b(?:causes?|caused(?: by)?|because of|due to|driven by|attributed to|results? from|"
    r"causal effects?|(?:proves?|confirms?|implies?|establish(?:es)?|demonstrates?) causation)\b",
    re.IGNORECASE,
)
_APPROVAL = re.compile(
    r"\b(?:proves? fairness|guarantees? fairness|is fair|legally compliant|clinically safe|validation (?:has )?passed)\b",
    re.IGNORECASE,
)
_NEGATED_PREFIX = re.compile(
    r"\b(?:not|cannot|can't|never|neither|no evidence(?: that)?)\s+(?:\w+\s+){0,5}$|\bno\s*$",
    re.IGNORECASE,
)
_INSUFFICIENCY = re.compile(
    r"\s+(?:both\s+)?(?:insufficient (?:evidence|data|sample|size|count|positive|group)|too few |"
    r"below_min_|(?:small|low) (?:sample|group|count|size|positive)|missing (?:score|data)|"
    r"undefined (?:rate|metric)|minimum (?:size|count|group))",
    re.IGNORECASE,
)
_DISPARITY_CONTEXT = re.compile(
    r"\b(?:gaps?|disparit(?:y|ies)|differences?|inequalit(?:y|ies)|"
    r"discriminat\w*|bias(?:ed)?|racis\w*|sexism|ageism|unfair\w*|"
    r"model|predict\w*|race|gender|payer|insurance|"
    r"groups?\s+(?:differ|show|have))\b",
    re.IGNORECASE,
)


def _string(value: str) -> str:
    if value.startswith('"'):
        return json.loads(value)
    if value.startswith("'"):
        return ast.literal_eval(value)
    return value[1:-1]


def check_group_mentions(text: str, groups: dict[str, list], path: str) -> tuple[str, list[ClaimIssue]]:
    """Validate quoted labels and attribute/value references, then mask their spans.

    Masking stops a verified age band or numeric label from becoming a numeric
    outcome claim. Typed references distinguish null, booleans, numbers and text.
    Bare labels can be verified when they exactly identify one trusted category;
    unknown or ambiguous names must be rewritten with explicit qualification.
    """
    issues = []
    spans = []
    identities = {name: {(type(v).__name__, v) for v in values} for name, values in groups.items()}

    def issue(message):
        issues.append(ClaimIssue(code="unsupported_group", message=f"{path}: {message}"))

    for match in _REFERENCE.finditer(text):
        attribute = match["attribute"]
        try:
            if not re.fullmatch(r"[A-Za-z_]\w*", attribute):
                attribute = _string(attribute)
            token = match["value"]
            value = _string(token) if re.fullmatch(_STRING, token) else json.loads(token)
        except (ValueError, SyntaxError):
            issue("Invalid group-reference string; use JSON-quoted labels.")
            spans.append(match.span())
            continue
        if attribute in _IDENTIFIERS and attribute not in groups:
            continue  # A metric annotation, still subject to numeric checks.
        if attribute in _STATISTICAL_ANNOTATIONS and attribute not in groups and type(value) in (int, float):
            continue  # p=... and CI: ... are numeric annotations, not group labels.
        if attribute not in groups or (type(value).__name__, value) not in identities[attribute]:
            issue("Group reference does not match the attribute and typed label in trusted evidence.")
        spans.append(match.span())

    def masked():
        characters = list(text)
        for start, end in spans:
            characters[start:end] = " " * (end - start)
        return "".join(characters)

    for match in _QUOTED.finditer(masked()):
        try:
            label = _string(match.group())
        except (ValueError, SyntaxError):
            issue("Invalid quoted label; use JSON quoting.")
            spans.append(match.span())
            continue
        matches = [name for name, values in identities.items() if ("str", label) in values]
        context = bool(re.search(r"\bgroups?\s+(?:(?:named|called)\s+)?$", text[:match.start()], re.IGNORECASE)
                       or re.match(r"\s+groups?\b", text[match.end():], re.IGNORECASE))
        if (label in groups or label in _IDENTIFIERS) and not context:
            pass  # A quoted column/metric identifier, not a group-value claim.
        elif not matches:
            issue("Quoted name is absent from trusted group labels and identifiers.")
        elif len(matches) > 1:
            # "For attribute race ... group \"\"" supplies an explicit scope.
            # Use only this sentence, exclude previously quoted data, and reject
            # conflicting attribute anchors rather than guessing the last one.
            sentence_prefix = re.split(r"(?<!\d)[.!?]|[.!?](?!\d)|[;\n]", masked()[:match.start()])[-1]
            anchors = {
                m["after"] or m["before"] for m in _ATTRIBUTE_CONTEXT.finditer(sentence_prefix)
                if m["after"] is not None or m["before"] in groups
            }
            if len(anchors) != 1 or not anchors.issubset(set(matches)):
                issue("Ambiguous group label " + json.dumps(label, ensure_ascii=False)
                      + " appears in " + ", ".join(matches) + "; qualify the intended attribute: "
                      + " or ".join(f"{name}: {json.dumps(label, ensure_ascii=False)}" for name in matches) + ".")
        spans.append(match.span())

    prose = masked()
    trusted_string_labels = {value for values in groups.values() for value in values if isinstance(value, str)}
    for match in _BARE_GROUP.finditer(prose):
        label = match["after"] or match["before"]
        if match["before"]:
            # Sentence-initial descriptions such as "Suppressed groups" do not
            # name a category. Retain rejection when a descriptor is itself a
            # trusted category label, since that usage is ambiguous.
            if label.casefold() in _GROUP_DETERMINERS:
                continue
            if label.casefold() in _NUMBER_WORDS and label not in trusted_string_labels:
                continue  # Normalize supported word-form counts below.
            descriptive = label.casefold() in _GROUP_DESCRIPTORS or label.casefold() in {name.casefold() for name in groups}
            if descriptive and label not in trusted_string_labels:
                continue
            if descriptive:
                issue(f"Ambiguous descriptive/category name {json.dumps(label)}; qualify its attribute explicitly.")
                continue
        matches = [name for name, values in identities.items() if ("str", label) in values]
        if len(matches) == 1:
            spans.append(match.span("after" if match["after"] else "before"))
            continue
        issue(f"Unquoted group-like name {json.dumps(label, ensure_ascii=False)}; "
              "quote exact group labels and include the attribute when ambiguous.")
    # Convert count phrases only after group-label masking, so a category called
    # "Six" is not changed into an outcome number. Numeric checks still run next.
    prose = _WORD_COUNT.sub(lambda m: str(_NUMBER_WORDS[m["number"].lower()]) + m.group()[len(m["number"]):], masked())
    return prose, issues


def _has_affirmative_claim(text: str, pattern: re.Pattern, *, allow_insufficiency: bool = False) -> bool:
    for clause in re.split(r"[;,\n]|\b(?:but|however|yet|whereas|although|and)\b", text, flags=re.IGNORECASE):
        for match in pattern.finditer(clause):
            prefix = clause[:match.start()]
            negated = bool(_NEGATED_PREFIX.search(prefix)) and not re.search(r"\bnot only\b", prefix, re.IGNORECASE)
            if negated:
                continue
            if match.group().lower().startswith("causal effect") and re.match(
                r"\s+(?:cannot|can not|must not)\s+be\s+(?:inferred|claimed|established|determined)",
                clause[match.end():], re.IGNORECASE,
            ):
                continue
            # Suppression/missingness may be explained by the computed policy.
            policy_context = re.search(r"\b(?:suppressed|suppression|excluded|unavailable|undefined|missing|omitted)\b", prefix, re.IGNORECASE)
            if (allow_insufficiency and policy_context and match.group().lower() in {"due to", "because of"}
                    and _INSUFFICIENCY.match(clause[match.end():])):
                continue
            if (allow_insufficiency and match.group().lower() in {"due to", "because of"}
                    and not _DISPARITY_CONTEXT.search(clause)):
                # Generic procedural explanations (e.g. bootstrap variability
                # due to random sampling) do not explain demographic gaps.
                # Other direct causal verbs are still screened unconditionally.
                continue
            return True
    return False


def has_causal_claim(text: str) -> bool:
    """Screen causal assertions with clause-local negation, not sentence-wide negation."""
    return _has_affirmative_claim(text, _CAUSAL, allow_insufficiency=True)


def has_unsupported_approval(text: str) -> bool:
    return _has_affirmative_claim(text, _APPROVAL)
