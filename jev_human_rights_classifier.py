"""
Jev Human Rights Text Classifier
================================

A research/triage classifier that decomposes human-rights-related text
classification into atomic Jev/System One questions, then applies explicit
Python rules over the returned probabilities.

This is intentionally NOT a legal-decision engine. The outputs are evidence
triage labels such as STRONG_MATCH, POSSIBLE_MATCH, and NOT_SUPPORTED.
A human analyst/lawyer should make any legal determination.

Frameworks implemented:
  1. UN Strategy and Plan of Action on Hate Speech (broad content definition)
  2. ICCPR Article 20(2) + Rabat Plan of Action threshold factors
  3. Direct and public incitement to commit genocide (Genocide Convention
     Article III(c), informed by ICTR/IRMCT jurisprudence)
  4. Technology-facilitated gender-based violence (UNFPA / UN Women)

Install:
    pip install typesafe-sdk python-dotenv

Environment (.env):
    TYPESAFE_API_KEY=...
    JEV_MODEL=jev-latest

Run:
    python jev_human_rights_classifier.py

The script asks for the message and optional contextual metadata. Context is
important for legal tests such as publicness, speaker influence, intent,
likelihood, coded language, and the Rabat threshold factors.
"""

from __future__ import annotations

import json
import os
import statistics
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Iterable

from dotenv import load_dotenv
from typesafe_sdk import Noul, TypeSafeClient


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

MODEL = os.getenv("JEV_MODEL", "jev-latest")

# These are operational triage thresholds, NOT legal thresholds.
# For high-stakes classifications, it is deliberately hard to reach a strong
# automated match and a substantial middle band is routed to review.
YES_THRESHOLD = 0.80
NO_THRESHOLD = 0.20

# If repeated Jev calls differ by more than this on an element, flag it as
# unstable for review even when the mean is high/low.
MAX_STABLE_SPREAD = 0.15

DEFAULT_REPEATS = 3


# ---------------------------------------------------------------------------
# Source / framework metadata
# ---------------------------------------------------------------------------

FRAMEWORKS: dict[str, dict[str, str]] = {
    "un_hate_speech": {
        "name": "UN Strategy and Plan of Action on Hate Speech",
        "source": "https://www.un.org/en/node/165070",
        "scope": (
            "Broad content-level definition; this is not itself a universal "
            "international legal definition of hate speech."
        ),
    },
    "iccp_article_20": {
        "name": "ICCPR Article 20(2) + Rabat Plan of Action",
        "source": "https://www.ohchr.org/",
        "scope": (
            "Advocacy of national, racial or religious hatred that constitutes "
            "incitement to discrimination, hostility or violence; Rabat factors "
            "are used to assess the threshold contextually."
        ),
    },
    "genocide_incitement": {
        "name": "Direct and public incitement to commit genocide",
        "source": "https://legal.un.org/avl/ha/cppcg/cppcg.html",
        "scope": (
            "Genocide Convention Article III(c), with directness/publicness and "
            "intent informed by ICTR/IRMCT jurisprudence."
        ),
    },
    "tfgbv": {
        "name": "Technology-facilitated gender-based violence",
        "source": "https://www.unfpa.org/TFGBV",
        "scope": (
            "Research/monitoring definition covering gender-based violence "
            "committed, assisted, aggravated or amplified through ICT/digital media."
        ),
    },
}


# ---------------------------------------------------------------------------
# Atomic Jev questions
# ---------------------------------------------------------------------------

# Design rule: one proposition per Noul. Jev returns P(True) for each.
# The state contains both the message and optional metadata. Questions explicitly
# tell the model not to assume missing context.
QUESTION_TEXT: dict[str, str] = {
    # Attribution / context-control questions
    "speaker_endorses_harmful_content": (
        "Based only on the supplied state, is the speaker presenting the harmful, "
        "hostile, discriminatory, or violent content as their own endorsed position, "
        "rather than neutrally quoting, reporting, documenting, or condemning it? "
        "Do not assume endorsement when context is missing."
    ),

    # ------------------------------------------------------------------
    # UN broad hate-speech content definition
    # ------------------------------------------------------------------
    "hs_identity_target": (
        "Does the message target or refer to a person or group on the basis of an "
        "identity factor such as religion, ethnicity, nationality, race, colour, "
        "descent, gender, or another identity characteristic?"
    ),
    "hs_attack_pejorative_discriminatory": (
        "Does the message attack the target or use pejorative or discriminatory "
        "language about the target?"
    ),

    # ------------------------------------------------------------------
    # ICCPR Article 20(2): core elements
    # ------------------------------------------------------------------
    "iccp_national_racial_religious_basis": (
        "Is the relevant target defined on a national, racial, or religious basis?"
    ),
    "iccp_advocacy_of_hatred": (
        "Does the message advocate or promote hatred toward that national, racial, "
        "or religious target, rather than merely describe or report hatred?"
    ),
    "iccp_intent_to_incite_audience": (
        "Based only on the supplied message and context, is there evidence that the "
        "speaker intends to incite the audience against the target group? Do not infer "
        "intent solely from offensive language when evidence is absent."
    ),
    "iccp_incites_discrimination": (
        "Does the message encourage the audience to discriminate against members of "
        "the target group, including denying rights, services, opportunities, or equal treatment?"
    ),
    "iccp_incites_hostility": (
        "Does the message encourage the audience toward active hostility against the target group?"
    ),
    "iccp_incites_violence": (
        "Does the message encourage the audience to commit violence against the target group?"
    ),

    # Rabat threshold factors. These are contextual indicators, not independent
    # criminal offences and not a numeric legal formula.
    "rabat_context_supports_harm": (
        "Does the supplied social or political context materially increase the risk "
        "that this expression could contribute to discrimination, hostility, or violence? "
        "If relevant context is absent, do not assume it."
    ),
    "rabat_speaker_influence": (
        "Does the supplied context indicate that the speaker has status, authority, "
        "credibility, or influence over the intended audience? If unknown, do not assume it."
    ),
    "rabat_content_form_inciting": (
        "Considering wording, tone, form, and any coded or culturally specific meaning "
        "provided in the state, is the expression framed in a strongly inciting manner?"
    ),
    "rabat_extent_significant": (
        "Does the supplied context indicate significant dissemination or reach, such as "
        "a public channel, mass audience, repeated distribution, or amplification? "
        "If reach is unknown, do not assume it."
    ),
    "rabat_likelihood_of_harm": (
        "Considering only the supplied context, is there a meaningful likelihood that "
        "the advocated discrimination, hostility, or violence could occur, including "
        "consideration of imminence where relevant?"
    ),

    # ------------------------------------------------------------------
    # Direct and public incitement to commit genocide
    # ------------------------------------------------------------------
    "gen_target_protected_group": (
        "Is the target a national, ethnical/ethnic, racial, or religious group, as such, "
        "within the protected-group categories of the Genocide Convention?"
    ),
    "gen_call_killing": (
        "Does the message urge or call on others to kill members of the protected group?"
    ),
    "gen_call_serious_bodily_or_mental_harm": (
        "Does the message urge or call on others to cause serious bodily or serious "
        "mental harm to members of the protected group?"
    ),
    "gen_call_destructive_conditions": (
        "Does the message urge or call on others to deliberately inflict conditions of "
        "life on the protected group calculated to bring about its physical destruction, "
        "in whole or in part?"
    ),
    "gen_call_prevent_births": (
        "Does the message urge or call on others to impose measures intended to prevent "
        "births within the protected group?"
    ),
    "gen_call_transfer_children": (
        "Does the message urge or call on others to forcibly transfer children of the "
        "protected group to another group?"
    ),
    "gen_direct": (
        "Considering the message's language and any cultural or contextual information "
        "supplied, would the intended audience understand it as a direct appeal to commit "
        "one or more genocidal acts, rather than a vague or merely indirect suggestion?"
    ),
    "gen_public": (
        "Based only on the supplied metadata/context, is the inciting communication public: "
        "addressed to the general public or an unselected audience in a public setting or "
        "through public/mass media, rather than only a private or selected group?"
    ),
    "gen_intent_to_incite": (
        "Based only on the supplied message and context, is there evidence that the speaker "
        "intends to incite others to commit the called-for genocidal act or acts?"
    ),
    "gen_intent_destroy_group": (
        "Based only on the supplied message and context, is there evidence of an intent to "
        "destroy the protected group, in whole or in part, as such?"
    ),

    # ------------------------------------------------------------------
    # Technology-facilitated gender-based violence (TFGBV)
    # ------------------------------------------------------------------
    "tfgbv_gender_basis": (
        "Is the harmful conduct directed at a person or group on the basis of gender, "
        "sex, gender identity/expression, or because the target is perceived as belonging "
        "to a gendered group?"
    ),
    "tfgbv_technology_mediated": (
        "Is the harmful conduct committed, assisted, aggravated, amplified, or enabled in "
        "part or fully through information and communication technology or digital media?"
    ),
    "tfgbv_harm_or_rights": (
        "Does the conduct cause, threaten, or create a meaningful likelihood of physical, "
        "sexual, psychological, social, political, or economic harm, or another infringement "
        "of the target's rights or freedoms?"
    ),

    # TFGBV subtypes. These are multi-label Nouls because more than one may apply.
    "tfgbv_threats": "Does the conduct include threats or intimidation directed at the target?",
    "tfgbv_gender_hate": (
        "Does the conduct include gender-based hate speech or misogynistic/misandrist abuse "
        "directed at a person or group because of gender?"
    ),
    "tfgbv_sexual_harassment": (
        "Does the conduct include unwanted sexualized harassment, sexual comments, sexual threats, "
        "or other sexualized abuse?"
    ),
    "tfgbv_doxxing": (
        "Does the conduct expose, threaten to expose, or weaponize private or identifying "
        "information about the target without a legitimate basis?"
    ),
    "tfgbv_stalking": (
        "Does the conduct involve stalking, persistent surveillance, monitoring, tracking, "
        "or unwanted repeated contact facilitated by technology?"
    ),
    "tfgbv_image_based_abuse": (
        "Does the conduct involve non-consensual creation, sharing, threatened sharing, or "
        "manipulation of intimate or sexual images, including synthetic/deepfake imagery?"
    ),
    "tfgbv_sextortion": (
        "Does the conduct involve coercion or blackmail using sexual or intimate information, "
        "images, or threats to disclose them?"
    ),
    "tfgbv_impersonation": (
        "Does the conduct involve harmful impersonation or a fake account used to target, "
        "harass, deceive, or damage the person?"
    ),
    "tfgbv_grooming": (
        "Does the conduct involve online grooming intended to facilitate sexual exploitation, "
        "sexual assault, or other gender-based abuse?"
    ),
    "tfgbv_hacking_or_tech_control": (
        "Does the conduct involve hacking, account/device compromise, spyware-like control, "
        "or misuse of digital access to monitor, coerce, or harm the target?"
    ),

    # ------------------------------------------------------------------
    # Cross-cutting research indicators (not standalone legal conclusions)
    # ------------------------------------------------------------------
    "indicator_dehumanization": (
        "Does the message dehumanize an identity group by portraying its members as animals, "
        "vermin, disease, contamination, objects, or less than human?"
    ),
    "indicator_collective_harm": (
        "Does the message advocate punishment, expulsion, deprivation, or violence against "
        "people collectively because they belong to an identity group?"
    ),
    "indicator_explicit_threat": (
        "Does the message contain an explicit threat of physical violence against a person or group?"
    ),
}


GENOCIDE_ACT_KEYS = [
    "gen_call_killing",
    "gen_call_serious_bodily_or_mental_harm",
    "gen_call_destructive_conditions",
    "gen_call_prevent_births",
    "gen_call_transfer_children",
]

TFGBV_SUBTYPE_KEYS = [
    "tfgbv_threats",
    "tfgbv_gender_hate",
    "tfgbv_sexual_harassment",
    "tfgbv_doxxing",
    "tfgbv_stalking",
    "tfgbv_image_based_abuse",
    "tfgbv_sextortion",
    "tfgbv_impersonation",
    "tfgbv_grooming",
    "tfgbv_hacking_or_tech_control",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ElementResult:
    probability: float
    stdev: float
    spread: float
    stable: bool
    status: str


@dataclass
class ClassificationResult:
    framework: str
    status: str
    score: float
    rationale: str
    required_elements: dict[str, float]
    supporting_elements: dict[str, float]
    unstable_elements: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def triage_status(probability: float) -> str:
    if probability >= YES_THRESHOLD:
        return "YES"
    if probability <= NO_THRESHOLD:
        return "NO"
    return "UNCERTAIN"


def mean_probability(elements: dict[str, ElementResult], key: str) -> float:
    return elements[key].probability


def minimum_probability(elements: dict[str, ElementResult], keys: Iterable[str]) -> float:
    values = [mean_probability(elements, key) for key in keys]
    return min(values) if values else 0.0


def maximum_probability(elements: dict[str, ElementResult], keys: Iterable[str]) -> float:
    values = [mean_probability(elements, key) for key in keys]
    return max(values) if values else 0.0


def unstable_from(elements: dict[str, ElementResult], keys: Iterable[str]) -> list[str]:
    return [key for key in keys if not elements[key].stable]


def gate_status(required: list[float], *, any_required: list[list[float]] | None = None) -> tuple[str, float]:
    """Return an operational evidence-triage status.

    required: all of these propositions are required.
    any_required: each inner list is an OR-group where at least one proposition is required.

    This is intentionally conservative:
      - STRONG_MATCH only when every required gate is >= YES_THRESHOLD.
      - NOT_SUPPORTED when an essential gate is <= NO_THRESHOLD, or every option
        in an OR-group is <= NO_THRESHOLD.
      - otherwise POSSIBLE_MATCH -> human review.
    """
    any_required = any_required or []

    essential_values = list(required)
    group_maxima = [max(group) for group in any_required if group]
    combined = essential_values + group_maxima
    score = min(combined) if combined else 0.0

    if required and any(value <= NO_THRESHOLD for value in required):
        return "NOT_SUPPORTED", score

    for group in any_required:
        if group and all(value <= NO_THRESHOLD for value in group):
            return "NOT_SUPPORTED", score

    if all(value >= YES_THRESHOLD for value in required) and all(
        any(value >= YES_THRESHOLD for value in group) for group in any_required
    ):
        return "STRONG_MATCH", score

    return "POSSIBLE_MATCH", score


# ---------------------------------------------------------------------------
# Jev calls + self-consistency
# ---------------------------------------------------------------------------

def build_questions() -> dict[str, Noul]:
    return {key: Noul(instructions=text) for key, text in QUESTION_TEXT.items()}


def run_once(client: TypeSafeClient, state: dict[str, Any], model: str) -> tuple[dict[str, float], dict[str, Any]]:
    # Add a throwaway uid so repeated runs are independent samples, matching the
    # approach used in TypeSafe's self-consistency cookbook.
    sampled_state = dict(state)
    sampled_state["sample_uid"] = uuid.uuid4().hex[:12]

    response = client.system_one(
        model=model,
        state=sampled_state,
        questions=build_questions(),
    )

    probabilities: dict[str, float] = {}
    for key in QUESTION_TEXT:
        answer = response.answers.get(key)
        if answer is None:
            raise RuntimeError(f"Jev response missing answer for question: {key}")
        probabilities[key] = clamp_probability(answer.noul)

    metadata = {
        "requested_model": model,
        "response_model": response.model,
        "input_tokens": getattr(response.usage, "input_tokens", None),
        "output_tokens": getattr(response.usage, "output_tokens", None),
    }
    return probabilities, metadata


def run_repeated(state: dict[str, Any], repeats: int, model: str) -> tuple[dict[str, ElementResult], list[dict[str, Any]]]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    runs: list[dict[str, float]] = []
    metadata: list[dict[str, Any]] = []

    with TypeSafeClient() as client:
        for _ in range(repeats):
            values, meta = run_once(client, state, model)
            runs.append(values)
            metadata.append(meta)

    elements: dict[str, ElementResult] = {}
    for key in QUESTION_TEXT:
        samples = [run[key] for run in runs]
        mean_value = statistics.mean(samples)
        stdev_value = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        spread_value = max(samples) - min(samples)
        elements[key] = ElementResult(
            probability=round(mean_value, 4),
            stdev=round(stdev_value, 4),
            spread=round(spread_value, 4),
            stable=spread_value <= MAX_STABLE_SPREAD,
            status=triage_status(mean_value),
        )

    return elements, metadata


# ---------------------------------------------------------------------------
# Framework classifiers
# ---------------------------------------------------------------------------

def classify_un_hate_speech(elements: dict[str, ElementResult]) -> ClassificationResult:
    core_keys = ["hs_identity_target", "hs_attack_pejorative_discriminatory"]
    content_status, content_score = gate_status(
        [mean_probability(elements, key) for key in core_keys]
    )

    endorsement = mean_probability(elements, "speaker_endorses_harmful_content")
    unstable = unstable_from(elements, [*core_keys, "speaker_endorses_harmful_content"])

    if content_status == "STRONG_MATCH" and endorsement >= YES_THRESHOLD:
        status = "STRONG_MATCH"
        rationale = (
            "Strong match to the UN broad hate-speech content definition, with strong "
            "evidence that the speaker endorses the content."
        )
        score = min(content_score, endorsement)
    elif content_status == "STRONG_MATCH" and endorsement <= NO_THRESHOLD:
        status = "CONTENT_MATCH_CONTEXT_CAVEAT"
        rationale = (
            "The text strongly contains identity-based attacking/pejorative content, but "
            "the supplied context suggests it is being quoted, reported, documented, or condemned "
            "rather than endorsed by the speaker."
        )
        score = content_score
    else:
        status = content_status
        rationale = (
            "Classification is based on the UN's broad content definition: identity-based targeting "
            "plus attacking, pejorative, or discriminatory language."
        )
        score = content_score

    return ClassificationResult(
        framework="un_hate_speech",
        status=status,
        score=round(score, 4),
        rationale=rationale,
        required_elements={key: mean_probability(elements, key) for key in core_keys},
        supporting_elements={
            "speaker_endorses_harmful_content": endorsement,
            "indicator_dehumanization": mean_probability(elements, "indicator_dehumanization"),
            "indicator_collective_harm": mean_probability(elements, "indicator_collective_harm"),
        },
        unstable_elements=unstable,
    )


def classify_iccp_article_20(elements: dict[str, ElementResult]) -> ClassificationResult:
    required_keys = [
        "iccp_national_racial_religious_basis",
        "iccp_advocacy_of_hatred",
        "iccp_intent_to_incite_audience",
    ]
    harm_keys = [
        "iccp_incites_discrimination",
        "iccp_incites_hostility",
        "iccp_incites_violence",
    ]

    status, score = gate_status(
        [mean_probability(elements, key) for key in required_keys],
        any_required=[[mean_probability(elements, key) for key in harm_keys]],
    )

    rabat_keys = [
        "rabat_context_supports_harm",
        "rabat_speaker_influence",
        "iccp_intent_to_incite_audience",  # intent is one Rabat factor
        "rabat_content_form_inciting",
        "rabat_extent_significant",
        "rabat_likelihood_of_harm",
    ]
    rabat_values = {key: mean_probability(elements, key) for key in rabat_keys}

    # We do not turn the Rabat factors into a fake legal arithmetic test.
    # Instead they are surfaced explicitly and can raise review priority.
    high_rabat_count = sum(value >= YES_THRESHOLD for value in rabat_values.values())
    low_rabat_count = sum(value <= NO_THRESHOLD for value in rabat_values.values())

    if status == "STRONG_MATCH" and high_rabat_count >= 4:
        final_status = "STRONG_MATCH_HIGH_REVIEW_PRIORITY"
        rationale = (
            "The core Article 20(2) elements strongly match and multiple Rabat threshold factors "
            "also indicate elevated contextual risk. This remains a triage result, not a legal finding."
        )
    elif status == "STRONG_MATCH":
        final_status = "STRONG_CORE_MATCH_REVIEW_CONTEXT"
        rationale = (
            "The core Article 20(2) elements strongly match, but the Rabat contextual threshold "
            "requires human assessment and may be incomplete from the supplied state."
        )
    elif status == "POSSIBLE_MATCH":
        final_status = "POSSIBLE_MATCH"
        rationale = (
            "Some Article 20(2) elements are present or uncertain. Review the individual elements "
            "and the six Rabat factors before drawing any legal conclusion."
        )
    else:
        final_status = "NOT_SUPPORTED"
        rationale = (
            "At least one essential Article 20(2) element is strongly unsupported by the supplied state."
        )

    relevant_keys = [*required_keys, *harm_keys, *rabat_keys]
    return ClassificationResult(
        framework="iccp_article_20",
        status=final_status,
        score=round(score, 4),
        rationale=rationale,
        required_elements={
            **{key: mean_probability(elements, key) for key in required_keys},
            "any_incitement_to_discrimination_hostility_or_violence": maximum_probability(elements, harm_keys),
        },
        supporting_elements={
            **{key: mean_probability(elements, key) for key in harm_keys},
            **rabat_values,
            "rabat_high_factor_count": float(high_rabat_count),
            "rabat_low_factor_count": float(low_rabat_count),
        },
        unstable_elements=unstable_from(elements, relevant_keys),
    )


def classify_genocide_incitement(elements: dict[str, ElementResult]) -> ClassificationResult:
    required_keys = [
        "gen_target_protected_group",
        "gen_direct",
        "gen_public",
        "gen_intent_to_incite",
        "gen_intent_destroy_group",
    ]

    action_values = [mean_probability(elements, key) for key in GENOCIDE_ACT_KEYS]
    status, score = gate_status(
        [mean_probability(elements, key) for key in required_keys],
        any_required=[action_values],
    )

    if status == "STRONG_MATCH":
        rationale = (
            "All operationally required elements strongly match: protected group, a call for at least "
            "one Convention genocidal act, directness, publicness, intent to incite, and genocidal intent. "
            "This is a strong triage match for specialist legal review, not a determination of criminal liability."
        )
    elif status == "POSSIBLE_MATCH":
        rationale = (
            "One or more incitement-to-genocide elements are uncertain. Directness can depend on linguistic/" 
            "cultural context, and publicness often requires platform/audience metadata. Human review is required."
        )
    else:
        rationale = (
            "At least one essential element is strongly unsupported, or none of the five Genocide Convention "
            "acts is strongly supported as the conduct being urged."
        )

    all_keys = [*required_keys, *GENOCIDE_ACT_KEYS]
    return ClassificationResult(
        framework="genocide_incitement",
        status=status,
        score=round(score, 4),
        rationale=rationale,
        required_elements={
            **{key: mean_probability(elements, key) for key in required_keys},
            "any_genocide_act_urged": maximum_probability(elements, GENOCIDE_ACT_KEYS),
        },
        supporting_elements={key: mean_probability(elements, key) for key in GENOCIDE_ACT_KEYS},
        unstable_elements=unstable_from(elements, all_keys),
    )


def classify_tfgbv(elements: dict[str, ElementResult]) -> ClassificationResult:
    required_keys = [
        "tfgbv_gender_basis",
        "tfgbv_technology_mediated",
        "tfgbv_harm_or_rights",
    ]
    status, score = gate_status([mean_probability(elements, key) for key in required_keys])

    subtypes = {
        key: mean_probability(elements, key)
        for key in TFGBV_SUBTYPE_KEYS
        if mean_probability(elements, key) >= NO_THRESHOLD
    }

    if status == "STRONG_MATCH":
        rationale = (
            "Strong match to the operational TFGBV definition: gender basis, technology mediation, "
            "and harm/rights impact are all strongly supported."
        )
    elif status == "POSSIBLE_MATCH":
        rationale = (
            "Some TFGBV elements are supported but one or more are uncertain. Review context and subtype signals."
        )
    else:
        rationale = (
            "At least one core TFGBV element is strongly unsupported by the supplied state."
        )

    return ClassificationResult(
        framework="tfgbv",
        status=status,
        score=round(score, 4),
        rationale=rationale,
        required_elements={key: mean_probability(elements, key) for key in required_keys},
        supporting_elements=subtypes,
        unstable_elements=unstable_from(elements, required_keys),
    )


# ---------------------------------------------------------------------------
# Output and validation
# ---------------------------------------------------------------------------

def validate_results(
    elements: dict[str, ElementResult],
    classifications: list[ClassificationResult],
) -> dict[str, Any]:
    unstable = sorted(key for key, value in elements.items() if not value.stable)
    uncertain = sorted(
        key for key, value in elements.items()
        if value.status == "UNCERTAIN"
    )

    # A simple validation policy: any strong legal-style triage result with
    # instability in a required element is downgraded for human review.
    downgraded: list[str] = []
    for result in classifications:
        if result.status.startswith("STRONG") and result.unstable_elements:
            result.status = "REVIEW_UNSTABLE"
            result.rationale += (
                " One or more relevant Jev probabilities were unstable across repeated runs, so the "
                "strong result has been downgraded to manual review."
            )
            downgraded.append(result.framework)

    return {
        "all_expected_questions_returned": set(elements) == set(QUESTION_TEXT),
        "unstable_element_count": len(unstable),
        "unstable_elements": unstable,
        "uncertain_element_count": len(uncertain),
        "uncertain_elements": uncertain,
        "downgraded_due_to_instability": downgraded,
    }


def analyse_message(state: dict[str, Any], repeats: int = DEFAULT_REPEATS, model: str = MODEL) -> dict[str, Any]:
    elements, api_metadata = run_repeated(state=state, repeats=repeats, model=model)

    classifications = [
        classify_un_hate_speech(elements),
        classify_iccp_article_20(elements),
        classify_genocide_incitement(elements),
        classify_tfgbv(elements),
    ]

    validation = validate_results(elements, classifications)

    return {
        "disclaimer": (
            "Research/triage output only. Jev probabilities and Python rules do not establish "
            "criminal liability, unlawfulness, intent, or a definitive human-rights/legal finding."
        ),
        "frameworks": FRAMEWORKS,
        "config": {
            "requested_model": model,
            "repeats": repeats,
            "yes_threshold": YES_THRESHOLD,
            "no_threshold": NO_THRESHOLD,
            "max_stable_spread": MAX_STABLE_SPREAD,
        },
        "api_runs": api_metadata,
        "state": state,
        "classifications": [asdict(result) for result in classifications],
        "element_results": {key: asdict(value) for key, value in elements.items()},
        "validation": validation,
    }


def ask_optional(prompt: str) -> str | None:
    value = input(prompt).strip()
    return value or None


def build_interactive_state() -> dict[str, Any]:
    print("\nJev Human Rights Text Classifier")
    print("--------------------------------")
    message = input("Message to classify: ").strip()
    if not message:
        raise ValueError("A message is required.")

    print("\nOptional context (press Enter to leave unknown).")
    platform = ask_optional("Platform/medium (e.g. Telegram public channel): ")
    publicness = ask_optional("Publicness (public/private/unknown): ")
    speaker_role = ask_optional("Speaker role/influence: ")
    audience = ask_optional("Audience/reach: ")
    context = ask_optional("Relevant social/political/cultural context: ")
    language_context = ask_optional("Language/slang/coded-language context: ")

    return {
        "message": message,
        "metadata": {
            "platform_or_medium": platform or "unknown",
            "publicness": publicness or "unknown",
            "speaker_role_or_influence": speaker_role or "unknown",
            "audience_or_reach": audience or "unknown",
            "social_political_cultural_context": context or "unknown",
            "language_or_coded_meaning_context": language_context or "unknown",
        },
    }


def print_summary(result: dict[str, Any]) -> None:
    print("\nClassification summary")
    print("======================")
    for item in result["classifications"]:
        print(
            f"{item['framework']:<24} {item['status']:<34} "
            f"score={item['score']:.3f}"
        )

    print("\nValidation")
    print("==========")
    validation = result["validation"]
    print(f"Unstable elements: {validation['unstable_element_count']}")
    print(f"Uncertain elements: {validation['uncertain_element_count']}")
    if validation["downgraded_due_to_instability"]:
        print(
            "Downgraded for instability: "
            + ", ".join(validation["downgraded_due_to_instability"])
        )

    print("\nReturned model versions")
    print("=======================")
    versions = [run["response_model"] for run in result["api_runs"]]
    print(", ".join(versions))


def main() -> None:
    if not os.getenv("TYPESAFE_API_KEY"):
        raise RuntimeError(
            "TYPESAFE_API_KEY is not set. Add it to your environment or .env file."
        )

    state = build_interactive_state()

    repeats_text = input(
        f"\nSelf-consistency runs [{DEFAULT_REPEATS}]: "
    ).strip()
    repeats = int(repeats_text) if repeats_text else DEFAULT_REPEATS

    result = analyse_message(state=state, repeats=repeats, model=MODEL)
    print_summary(result)

    save = input("\nSave full JSON result? [Y/n]: ").strip().lower()
    if save not in {"n", "no"}:
        output_path = "jev_human_rights_result.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
