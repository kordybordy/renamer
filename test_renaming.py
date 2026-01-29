from app_text_utils import apply_meta_defaults, apply_party_order, build_filename, requirements_from_template


def test_render_filename_uses_defendant_and_letter_type() -> None:
    template = ["defendant", "letter_type"]
    requirements = requirements_from_template(template)
    raw_meta = {"defendant": "Katarzyna Obałek", "letter_type": "Pozew"}
    meta = apply_meta_defaults(raw_meta, requirements)
    meta = apply_party_order(
        meta,
        plaintiff_surname_first=True,
        defendant_surname_first=True,
    )
    filename = build_filename(meta, template)
    assert filename == "Obałek Katarzyna - Pozew.pdf"


def test_render_filename_uses_custom_element_value() -> None:
    template = ["defendant_s_surname_name", "letter_type"]
    requirements = requirements_from_template(
        template,
        {"defendant_s_surname_name": "Surname first for the defendant"},
    )
    raw_meta = {"defendant_s_surname_name": "Obałek Katarzyna", "letter_type": "Pozew"}
    meta = apply_meta_defaults(raw_meta, requirements)
    filename = build_filename(meta, template)
    assert filename == "Obałek Katarzyna - Pozew.pdf"
