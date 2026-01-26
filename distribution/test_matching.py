from distribution.models import DocumentMeta, FolderMeta
from distribution.scorer import (
    DEFAULT_STOPWORDS,
    extract_person_pairs,
    extract_surnames_from_folder,
    extract_tokens,
    normalize_text,
    strip_folder_suffix,
    score_document,
)
from distribution.engine import DistributionConfig, DistributionEngine
from distribution.models import DistributionPlanItem


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
    match_name = strip_folder_suffix(name)
    return FolderMeta(
        folder_path=f"/cases/{name}",
        folder_name=name,
        match_name=match_name,
        normalized_name=normalize_text(match_name),
        tokens=extract_tokens(match_name, DEFAULT_STOPWORDS),
        surnames=extract_surnames_from_folder(match_name, DEFAULT_STOPWORDS),
        person_pairs=extract_person_pairs(match_name, DEFAULT_STOPWORDS),
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


def test_lslash_matches_ascii_folder() -> None:
    doc = make_doc(["Łukaszewski Jan"], "Łukaszewski Jan - Pozew.pdf")
    correct = make_folder("LUKASZEWSKI_JAN")
    wrong = make_folder("KOWALSKI_JAN")
    summary = score_document(doc, [wrong, correct], DEFAULT_STOPWORDS, top_k=2)
    assert summary.candidates[0].folder.folder_name == correct.folder_name


def test_folder_suffix_stripping_keeps_people_pairs() -> None:
    doc = make_doc(["Zieliński Franciszek"], "Zieliński Franciszek - Pozew.pdf")
    correct = make_folder("ZIELINSKI_FRANCISZEK XXVIII C 8703_25 (CC25MM)")
    wrong = make_folder("ZIELINSKI_FRANCISZEK_PAWEL")
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


def test_apply_plan_ask_selection_respected(tmp_path) -> None:
    input_dir = tmp_path / "input"
    case_root = tmp_path / "cases"
    input_dir.mkdir()
    case_root.mkdir()
    case_folder = case_root / "CASE_A"
    case_folder.mkdir()

    file_one = input_dir / "doc1.pdf"
    file_two = input_dir / "doc2.pdf"
    file_one.write_text("alpha")
    file_two.write_text("beta")

    plan = [
        DistributionPlanItem(
            source_pdf=str(file_one),
            chosen_folder=str(case_folder),
            candidates=[],
            decision="ASK",
            confidence=0.0,
            reason="Needs confirmation",
            dest_path=None,
        ),
        DistributionPlanItem(
            source_pdf=str(file_two),
            chosen_folder=None,
            candidates=[],
            decision="ASK",
            confidence=0.0,
            reason="Needs confirmation",
            dest_path=None,
        ),
    ]

    log_entries: list[str] = []
    engine = DistributionEngine(
        input_folder=str(input_dir),
        case_root=str(case_root),
        config=DistributionConfig(
            auto_threshold=70.0,
            gap_threshold=15.0,
            ai_threshold=0.7,
            top_k=15,
            stage2_k=80,
            candidate_pool_limit=200,
            fast_mode=True,
            stopwords=DEFAULT_STOPWORDS[:],
            unassigned_action="leave",
            unassigned_include_ask=False,
        ),
        ai_provider="openai",
        logger=log_entries.append,
    )

    audit_log = tmp_path / "audit.log"
    engine.apply_plan(plan, auto_only=False, audit_log_path=str(audit_log))

    assert (case_folder / "doc1.pdf").exists()
    assert not (case_folder / "doc2.pdf").exists()
    audit_contents = audit_log.read_text(encoding="utf-8")
    assert "doc2.pdf" in audit_contents
    assert "SKIPPED (unresolved_ask)" in audit_contents
