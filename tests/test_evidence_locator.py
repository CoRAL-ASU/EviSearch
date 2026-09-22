"""The evidence locator: which boxes a reviewer is shown for a value.

Each case below is one that went wrong on the real benchmark papers before it was handled - a Lancet decimal point,
an affiliation superscript, a thousands group, a value printed twice, a value that is not printed at all because it
was computed. The test PDF is built here, so every case is exact.
"""
import fitz
import pytest

from src.evisearch.services import evidence_locator as L


@pytest.fixture()
def doc():
    d = fitz.open()
    page = d.new_page(width=595, height=842)
    lines = [
        "Maha Hussain, MD1; Fred Saad, MD3; Karim Fizazi, MD, PhD4",
        "An international, open-label, randomised, phase 3 trial of darolutamide.",
        "Median overall survival was 76·6 months in the abiraterone group versus 45·7 months in the control group.",
        "In a sensitivity analysis the median was 76·6 months after censoring at crossover.",
        "Four treatment-related deaths occurred in the ADT plus D arm (n = 192).",
        "The expected number per 100 000 person-years is shown in the appendix.",
        "Metachronous disease: 108 (27·2%) versus 107 (27·2%).",
        "Events/N 34/152 in the high-volume subgroup.",
        "Darolutamide (n 5 497) Placebo (n 5 508) numbers at risk 193 103 43",
        "In this study, the E3805 study, patients received ADT alone or ADT plus docetaxel.",
    ]
    y = 60
    for text in lines:
        page.insert_text((40, y), text, fontsize=9)
        y += 40
    yield d
    d.close()


def _cell(doc, value, column, quote="", reasoning=()):
    return L.locate(doc, None, value, column, [{"pages": [1], "quote": quote, "origin": "reconciled"}], list(reasoning))


def test_numbers_are_read_as_printed():
    assert L.word_numbers("27·2%)") == ["27.2"]          # the Lancet decimal point
    assert L.word_numbers("(n=502)") == ["502"]
    assert L.word_numbers("34/152") == ["34", "152"]         # events/N
    assert L.word_numbers("45–66") == ["45", "66"]       # a range
    assert L.word_numbers("MD3;") == []                      # an affiliation mark is not the number 3
    assert L.word_numbers("000") == []                       # the tail of "100 000" is not the number 0
    assert L.word_numbers("E3805") == []                     # part of a name
    assert L.canon("68.0") == L.canon("68")                  # the same number, differently printed


def test_the_value_itself_is_boxed_not_its_sentence(doc):
    c = _cell(doc, "108 (27.2%)", "Mode of metastases - N (%) | Metachronous | Treatment")
    assert c["status"] == "value"
    assert c["regions"][0]["text"].startswith("108")
    assert len(c["regions"][0]["rects"]) == 1                # one box, not one per word


def test_a_value_printed_twice_is_boxed_where_the_quotation_is(doc):
    c = _cell(doc, "76.6", "Median OS (mo) | Overall | Treatment",
              quote="In a sensitivity analysis the median was 76·6 months after censoring at crossover.")
    assert c["status"] == "value"
    assert c["regions"][0]["occurrences"] == 2
    box = c["regions"][0]["rects"][0]
    assert 160 < box[1] < 190                                 # the second occurrence: the sensitivity line (baseline 180)


def test_a_small_integer_is_boxed_beside_its_column_word_and_never_in_a_superscript(doc):
    c = _cell(doc, "3", "Phase")
    assert c["status"] == "value" and c["regions"][0]["text"] == "3"
    assert 90 < c["regions"][0]["rects"][0][1] < 110          # the "phase 3" line, not "Fred Saad, MD3"


def test_zero_is_not_found_inside_a_thousands_separated_number(doc):
    c = _cell(doc, "0 (0%)", "Adverse Events - N (%) | Treatment-related Grade 5 | Treatment")
    assert all("100" not in r["text"] for r in c["regions"])


def test_a_computed_value_shows_the_numbers_it_was_computed_from(doc):
    c = _cell(doc, "4 (2.1%)", "Adverse Events - N (%) | Treatment-related Grade 5 | Treatment",
              quote="Four treatment-related deaths occurred in the ADT plus D arm",
              reasoning=["4/192 = 2.08% rounded to 2.1%"])
    assert c["status"] == "operand"
    assert [r["number"] for r in c["regions"]] == ["192"]


def test_numbers_merely_mentioned_in_the_reasoning_are_not_treated_as_operands(doc):
    c = _cell(doc, "100%", "Docetaxel administration - N (%) | Treatment",
              quote="patients received ADT alone or ADT plus docetaxel",
              reasoning=["docetaxel 75 mg per square meter every 21 days, so 100% received it"])
    assert c["status"] == "quote"                             # no arithmetic, so the passage, not stray numbers


def test_an_identifier_is_matched_as_a_whole_word(doc):
    c = _cell(doc, "E3805", "Trial Name")
    assert c["status"] == "value" and c["regions"][0]["text"].startswith("E3805")


def test_a_text_value_is_located_by_its_words(doc):
    c = _cell(doc, "ADT plus docetaxel", "Treatment Arm(s)")
    assert c["status"] == "value" and "docetaxel" in c["regions"][0]["text"].lower()


def test_a_value_on_no_cited_page_opens_the_page_without_a_box(doc):
    c = _cell(doc, "117 (20.8%)", "Region - N (%) | North America | Treatment")
    assert c["status"] == "page" and c["regions"] == []


def test_boxes_are_mapped_to_the_parser_chunks_that_contain_them(doc):
    parse = {"chunks": [{"id": "chunk-a", "type": "text",
                         "grounding": {"page": 0, "box": {"left": 0.0, "top": 0.3, "right": 1.0, "bottom": 0.4}}}]}
    c = L.locate(doc, parse, "108 (27.2%)", "Mode of metastases", [{"pages": [1], "quote": "", "origin": "r"}])
    assert c["regions"][0]["chunk_ids"] == ["chunk-a"]


def test_a_jco_equals_sign_and_a_numbers_at_risk_row_do_not_fuse_into_one_number(doc):
    """JCO encodes "=" as the glyph 5, so "n = 497" reads "n 5 497"; at-risk rows print columns a space apart."""
    pw = L.PageWords(doc[0])
    assert pw.positions("497") and pw.positions("508") and pw.positions("193") and pw.positions("103")
    assert not pw.positions("5497") and not pw.positions("193103")
    assert pw.positions("100000")                            # a genuine thousands group is still one number


def test_numbers_with_labels_are_still_located_as_numbers(doc):
    c = _cell(doc, "76.6 months in the abiraterone group; 45.7 months in the control group",
              "Median OS (mo) | Overall | Treatment")
    assert c["status"] == "value" and "76" in c["regions"][0]["text"]
    assert L.text_dominant("progression-free survival, prostate-specific antigen level at 7 months (0.2 ng/mL), "
                           "time to castration resistance and adverse event profile")
