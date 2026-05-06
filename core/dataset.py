"""
ProDropLens dataset module.

Defines the pilot, core, patching, and extended datasets. All minimal pairs
live here; notebooks should only consume these data structures.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# Enum definitions

class Person(str, Enum):
    S1 = "1s"   # I
    S2 = "2s"   # you, singular
    S3 = "3s"   # he/she/it
    P1 = "1p"   # we
    P2 = "2p"   # you, plural
    P3 = "3p"   # they


class Tense(str, Enum):
    SIMDIKI = "simdiki"   # present progressive, -iyor
    GECMIS = "gecmis"     # past, -di
    GELECEK = "gelecek"   # future, -ecek
    GENIS = "genis"       # aorist/simple present, -r


class Verb(str, Enum):
    GITMEK = "gitmek"
    GELMEK = "gelmek"
    YAPMAK = "yapmak"


# Data structures

@dataclass
class MinimalPair:
    """
    Overt-subject vs. pro-drop minimal pair.

    In each pair, only subject presence changes. The inflected verb form and
    context stay the same.
    """
    id: str
    layer: str                    # "pilot" | "core" | "extended"
    verb: Verb
    tense: Tense
    person: Person

    overt_text: str               # "Ben her gun okula gidiyorum."
    prodrop_text: str             # "Her gun okula gidiyorum."

    subject_token: str            # "Ben"
    target_form: str              # "gidiyorum", the correct target inflection
    contrast_form: str            # "gidiyor", the incorrect 3s contrast form

    notes: str = ""


@dataclass
class PatchingPair:
    """
    Clean/corrupted pair for activation patching.

    The clean sentence has subject-verb agreement. The corrupted sentence swaps
    the subject while keeping the original verb form, intentionally creating a
    grammatical mismatch.
    """
    id: str
    verb: Verb
    tense: Tense

    clean_person: Person
    corrupted_person: Person

    clean_text: str               # "Ben her gun okula gidiyorum."
    corrupted_text: str           # "Sen her gun okula gidiyorum."

    clean_subject: str            # "Ben"
    corrupted_subject: str        # "Sen"
    target_form: str              # "gidiyorum"
    contrast_form: str            # "gidiyorsun"


# Morphology table

# Verb inflection forms: VERB_FORMS[verb][tense][person] -> str
VERB_FORMS: dict[Verb, dict[Tense, dict[Person, str]]] = {
    Verb.GITMEK: {
        Tense.SIMDIKI: {
            Person.S1: "gidiyorum",
            Person.S2: "gidiyorsun",
            Person.S3: "gidiyor",
            Person.P1: "gidiyoruz",
            Person.P2: "gidiyorsunuz",
            Person.P3: "gidiyorlar",
        },
        Tense.GECMIS: {
            Person.S1: "gittim",
            Person.S2: "gittin",
            Person.S3: "gitti",
            Person.P1: "gittik",
            Person.P2: "gittiniz",
            Person.P3: "gittiler",
        },
        Tense.GELECEK: {
            Person.S1: "gideceğim",
            Person.S2: "gideceksin",
            Person.S3: "gidecek",
            Person.P1: "gideceğiz",
            Person.P2: "gideceksiniz",
            Person.P3: "gidecekler",
        },
        Tense.GENIS: {
            Person.S1: "giderim",
            Person.S2: "gidersin",
            Person.S3: "gider",
            Person.P1: "gideriz",
            Person.P2: "gidersiniz",
            Person.P3: "giderler",
        },
    },
    Verb.GELMEK: {
        Tense.SIMDIKI: {
            Person.S1: "geliyorum",
            Person.S2: "geliyorsun",
            Person.S3: "geliyor",
            Person.P1: "geliyoruz",
            Person.P2: "geliyorsunuz",
            Person.P3: "geliyorlar",
        },
        Tense.GECMIS: {
            Person.S1: "geldim",
            Person.S2: "geldin",
            Person.S3: "geldi",
            Person.P1: "geldik",
            Person.P2: "geldiniz",
            Person.P3: "geldiler",
        },
        Tense.GELECEK: {
            Person.S1: "geleceğim",
            Person.S2: "geleceksin",
            Person.S3: "gelecek",
            Person.P1: "geleceğiz",
            Person.P2: "geleceksiniz",
            Person.P3: "gelecekler",
        },
        Tense.GENIS: {
            Person.S1: "gelirim",
            Person.S2: "gelirsin",
            Person.S3: "gelir",
            Person.P1: "geliriz",
            Person.P2: "gelirsiniz",
            Person.P3: "gelirler",
        },
    },
    Verb.YAPMAK: {
        Tense.SIMDIKI: {
            Person.S1: "yapıyorum",
            Person.S2: "yapıyorsun",
            Person.S3: "yapıyor",
            Person.P1: "yapıyoruz",
            Person.P2: "yapıyorsunuz",
            Person.P3: "yapıyorlar",
        },
        Tense.GECMIS: {
            Person.S1: "yaptım",
            Person.S2: "yaptın",
            Person.S3: "yaptı",
            Person.P1: "yaptık",
            Person.P2: "yaptınız",
            Person.P3: "yaptılar",
        },
        Tense.GELECEK: {
            Person.S1: "yapacağım",
            Person.S2: "yapacaksın",
            Person.S3: "yapacak",
            Person.P1: "yapacağız",
            Person.P2: "yapacaksınız",
            Person.P3: "yapacaklar",
        },
        Tense.GENIS: {
            Person.S1: "yaparım",
            Person.S2: "yaparsın",
            Person.S3: "yapar",
            Person.P1: "yaparız",
            Person.P2: "yaparsınız",
            Person.P3: "yaparlar",
        },
    },
}

# Subject pronouns.
SUBJECT_PRONOUNS: dict[Person, str] = {
    Person.S1: "Ben",
    Person.S2: "Sen",
    Person.S3: "O",
    Person.P1: "Biz",
    Person.P2: "Siz",
    Person.P3: "Onlar",
}

# Tense-specific context words used by the sentence templates.
TENSE_CONTEXT: dict[Tense, str] = {
    Tense.SIMDIKI: "her gün okula",
    Tense.GECMIS: "dün okula",
    Tense.GELECEK: "yarın okula",
    Tense.GENIS: "her sabah okula",
}


def get_contrast_form(verb: Verb, tense: Tense) -> str:
    """Return the 3s form used as the logit-difference contrast."""
    return VERB_FORMS[verb][tense][Person.S3]


# Dataset builders

def build_pilot_set() -> list[MinimalPair]:
    """
    Layer 1: pilot set, 6 minimal pairs / 12 sentences.

    One verb (gitmek), one tense (present progressive), six persons, and both
    overt-subject and pro-drop variants. Used for phase 0 tokenizer discovery.
    """
    pairs = []
    verb = Verb.GITMEK
    tense = Tense.SIMDIKI
    ctx = TENSE_CONTEXT[tense]

    for person in Person:
        subj = SUBJECT_PRONOUNS[person]
        form = VERB_FORMS[verb][tense][person]
        contrast = get_contrast_form(verb, tense)

        pairs.append(MinimalPair(
            id=f"pilot_{verb.value}_{tense.value}_{person.value}",
            layer="pilot",
            verb=verb,
            tense=tense,
            person=person,
            overt_text=f"{subj} {ctx} {form}.",
            prodrop_text=f"{ctx.capitalize()} {form}.",
            subject_token=subj,
            target_form=form,
            contrast_form=contrast,
        ))

    return pairs


def build_core_set() -> list[MinimalPair]:
    """
    Layer 2: core set, 72 pairs.

    Three verbs, four tenses, six persons, and both overt-subject and pro-drop
    variants. Used for phases 1-3: Logit Lens, DLA, and attention analysis.
    """
    pairs = []

    for verb in Verb:
        for tense in Tense:
            ctx = TENSE_CONTEXT[tense]
            for person in Person:
                subj = SUBJECT_PRONOUNS[person]
                form = VERB_FORMS[verb][tense][person]
                contrast = get_contrast_form(verb, tense)

                pairs.append(MinimalPair(
                    id=f"core_{verb.value}_{tense.value}_{person.value}",
                    layer="core",
                    verb=verb,
                    tense=tense,
                    person=person,
                    overt_text=f"{subj} {ctx} {form}.",
                    prodrop_text=f"{ctx.capitalize()} {form}.",
                    subject_token=subj,
                    target_form=form,
                    contrast_form=contrast,
                ))

    return pairs


# Person swaps used for patching: (clean_person, corrupted_person).
PATCHING_PERSON_PAIRS: list[tuple[Person, Person]] = [
    (Person.S1, Person.S2),   # Ben -> Sen
    (Person.S1, Person.S3),   # Ben -> O
    (Person.S2, Person.S1),   # Sen -> Ben
    (Person.S3, Person.S1),   # O -> Ben
    (Person.P1, Person.P3),   # Biz -> Onlar
]

# Tenses included in the patching set.
PATCHING_TENSES: list[Tense] = [Tense.SIMDIKI, Tense.GECMIS]


def build_patching_set() -> list[PatchingPair]:
    """
    Layer 3: patching set, 30 pairs.

    Five person swaps, three verbs, and two tenses. Used for phase 4 activation
    patching. Corrupted sentences intentionally contain subject-verb agreement
    errors.
    """
    pairs = []

    for verb in Verb:
        for tense in PATCHING_TENSES:
            ctx = TENSE_CONTEXT[tense]
            for clean_p, corrupted_p in PATCHING_PERSON_PAIRS:
                clean_subj = SUBJECT_PRONOUNS[clean_p]
                corrupted_subj = SUBJECT_PRONOUNS[corrupted_p]
                clean_form = VERB_FORMS[verb][tense][clean_p]
                contrast_form = VERB_FORMS[verb][tense][corrupted_p]

                pairs.append(PatchingPair(
                    id=f"patch_{verb.value}_{tense.value}_{clean_p.value}_vs_{corrupted_p.value}",
                    verb=verb,
                    tense=tense,
                    clean_person=clean_p,
                    corrupted_person=corrupted_p,
                    clean_text=f"{clean_subj} {ctx} {clean_form}.",
                    corrupted_text=f"{corrupted_subj} {ctx} {clean_form}.",
                    clean_subject=clean_subj,
                    corrupted_subject=corrupted_subj,
                    target_form=clean_form,
                    contrast_form=contrast_form,
                ))

    return pairs


# Access helpers

def get_pilot() -> list[MinimalPair]:
    return build_pilot_set()


def get_core() -> list[MinimalPair]:
    return build_core_set()


def get_patching() -> list[PatchingPair]:
    return build_patching_set()


def get_all_minimal_pairs() -> list[MinimalPair]:
    """Return the combined pilot and core minimal-pair sets."""
    return build_pilot_set() + build_core_set()


def filter_pairs(
    pairs: list[MinimalPair],
    verb: Optional[Verb] = None,
    tense: Optional[Tense] = None,
    person: Optional[Person] = None,
) -> list[MinimalPair]:
    """Filter minimal pairs by verb, tense, and/or person."""
    result = pairs
    if verb is not None:
        result = [p for p in result if p.verb == verb]
    if tense is not None:
        result = [p for p in result if p.tense == tense]
    if person is not None:
        result = [p for p in result if p.person == person]
    return result
