import json

from config import FILENAME_RULES, DEFAULT_TEMPLATE_ELEMENTS


import os


def load_settings(gui):
    settings = gui.settings
    gui.input_edit.setText(settings.value("input_folder", ""))
    gui.output_edit.setText(settings.value("output_folder", ""))
    gui.distribution_input_edit.setText(
        settings.value("distribution_input_folder", settings.value("input_folder", ""))
    )
    gui.case_root_edit.setText(settings.value("case_root_folder", ""))
    default_order = FILENAME_RULES.get("surname_first", True)
    turbo_mode = settings.value("turbo_mode", False)
    gui.turbo_mode_checkbox.setChecked(str(turbo_mode).lower() == "true")
    plaintiff_order = settings.value("plaintiff_surname_first", default_order)
    defendant_order = settings.value("defendant_surname_first", default_order)
    gui.plaintiff_order_combo.setCurrentIndex(0 if str(plaintiff_order).lower() == "true" else 1)
    gui.defendant_order_combo.setCurrentIndex(0 if str(defendant_order).lower() == "true" else 1)
    saved_template = settings.value("template", [])
    if isinstance(saved_template, str):
        saved_template = json.loads(saved_template) if saved_template else []
    if saved_template:
        gui.template_list.clear()
        for element in saved_template:
            gui.add_template_item(element)

    saved_input = gui.input_edit.text()
    if saved_input and os.path.isdir(saved_input):
        gui.load_pdfs()


def save_settings(gui):
    gui.settings.setValue("input_folder", gui.input_edit.text())
    gui.settings.setValue("output_folder", gui.output_edit.text())
    gui.settings.setValue("distribution_input_folder", gui.distribution_input_edit.text())
    gui.settings.setValue("case_root_folder", gui.case_root_edit.text())
    gui.settings.setValue("template", gui.get_template_elements())
    gui.settings.setValue("turbo_mode", gui.turbo_mode_checkbox.isChecked())
    gui.settings.setValue(
        "plaintiff_surname_first", bool(gui.plaintiff_order_combo.currentData())
    )
    gui.settings.setValue(
        "defendant_surname_first", bool(gui.defendant_order_combo.currentData())
    )
