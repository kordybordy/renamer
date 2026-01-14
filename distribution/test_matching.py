from distribution.models import DocumentMeta, FolderMeta
from distribution.scorer import (
    DEFAULT_STOPWORDS,
    extract_person_pairs,
    extract_surnames_from_folder,
    extract_tokens,
    normalize_text,
    score_document,
)


def make_doc(opposing_parties: list[str], filename: str) -> DocumentMeta:
    return DocumentMeta(
        source_path=f"/tmp/{filename}",
        file_name=filename,
        plaintiffs=[],
        defendants=[],
        opposing_parties=opposing_parties,
        case_numbers=[],
        letter_type="",
        raw_text_excerpt=None,
        extraction_source="filename",
    )


def make_folder(name: str) -> FolderMeta:
    return FolderMeta(
        folder_path=f"/cases/{name}",
        folder_name=name,
        normalized_name=normalize_text(name),
        tokens=extract_tokens(name, DEFAULT_STOPWORDS),
        surnames=extract_surnames_from_folder(name, DEFAULT_STOPWORDS),
        person_pairs=extract_person_pairs(name, DEFAULT_STOPWORDS),
        case_numbers=set(),
    )


def test_multi_party_full_match_ranked_first() -> None:
    doc = make_doc(
        ["Zygmuntowski Arkadiusz, Zygmuntowska Elżbieta"],
        "Zygmuntowski Arkadiusz, Zygmuntowska Elżbieta - Pozew.pdf",
    )
    correct = make_folder("ZYGMUNTOWSKI_ARKADIUSZ_ZYGMUNTOWSKA_ELZBIETA")
    wrong = make_folder("ZYGMUNTOWSKI_ARKADIUSZ")
    summary = score_document(doc, [wrong, correct], DEFAULT_STOPWORDS, top_k=2)
    assert summary.candidates[0].folder.folder_name == correct.folder_name


def test_surname_name_order_independent() -> None:
    doc = make_doc(["Kowalski Jan"], "Kowalski Jan - Pozew.pdf")
    correct = make_folder("KOWALSKI_JAN")
    wrong = make_folder("KOWALSKI_ANNA")
    summary = score_document(doc, [wrong, correct], DEFAULT_STOPWORDS, top_k=2)
    assert summary.candidates[0].folder.folder_name == correct.folder_name


def test_name_surname_order_flip_matches() -> None:
    doc = make_doc(["Grażyna Żmijewska"], "Grażyna Żmijewska - Pozew.pdf")
    correct = make_folder("ZMIJEWSKA_GRAZYNA")
    wrong = make_folder("ZMIJEWSKA_ANNA")
    summary = score_document(doc, [wrong, correct], DEFAULT_STOPWORDS, top_k=2)
    assert summary.candidates[0].folder.folder_name == correct.folder_name


def test_hyphenated_surname_matches() -> None:
    doc = make_doc(["Zwolińska-Gawlak Wanda"], "Zwolińska-Gawlak Wanda - Pozew.pdf")
    correct = make_folder("ZWOLINSKA - GAWLAK_WANDA")
    wrong = make_folder("ZWOLIŃSKA - GAWLAK_ANNA")
    summary = score_document(doc, [wrong, correct], DEFAULT_STOPWORDS, top_k=2)
    assert summary.candidates[0].folder.folder_name == correct.folder_name


def test_multi_party_prefers_more_people() -> None:
    doc = make_doc(
        ["Zmitrowicz Kamil, Zmitrowicz Tamara"],
        "Zmitrowicz Kamil, Zmitrowicz Tamara - Pozew.pdf",
    )
    correct = make_folder("ZMITROWICZ_KAMIL_ZMITROWICZ_TAMARA")
    wrong = make_folder("ZMITROWICZ_KAMIL")
    summary = score_document(doc, [wrong, correct], DEFAULT_STOPWORDS, top_k=2)
    assert summary.candidates[0].folder.folder_name == correct.folder_name


def test_diacritics_fold_matches_ascii_folder() -> None:
    doc = make_doc(["Żmijewska Grażyna"], "Żmijewska Grażyna - Pozew.pdf")
    correct = make_folder("ZMIJEWSKA_GRAZYNA")
    wrong = make_folder("KOWALSKA_GRAZYNA")
    summary = score_document(doc, [wrong, correct], DEFAULT_STOPWORDS, top_k=2)
    assert summary.candidates[0].folder.folder_name == correct.folder_name


def test_multi_party_family_matches_by_pairs() -> None:
    doc = make_doc(
        ["Dziewulska Zofia, Dziewulski Paweł"],
        "Dziewulska Zofia, Dziewulski Paweł - Pozew.pdf",
    )
    correct = make_folder("DZIEWULSKA_ZOFIA_DZIEWULSKI_PAWEL")
    wrong = make_folder("DZIEWULSKA_ZOFIA")
    summary = score_document(doc, [wrong, correct], DEFAULT_STOPWORDS, top_k=2)
    assert summary.candidates[0].folder.folder_name == correct.folder_name


def test_given_name_only_trap_is_penalized() -> None:
    doc = make_doc(["Żmijewska Grażyna"], "Żmijewska Grażyna - Pozew.pdf")
    correct = make_folder("ZMIJEWSKA_GRAZYNA")
    wrong = make_folder("GRAZYNA_KOWALSKA")
    summary = score_document(doc, [wrong, correct], DEFAULT_STOPWORDS, top_k=2)
    assert summary.candidates[0].folder.folder_name == correct.folder_name
