"""Vision inference must read choices from the screenshot, not metadata."""

from PIL import Image

from lmms_eval.tasks.mmmu_pro_qwen3_official import utils_qwen3_official as utils


def test_vision_prompt_does_not_expose_option_metadata():
    doc = {"image": Image.new("RGB", (12, 12)),
           "options": "['hidden choice one', 'hidden choice two']", "answer": "B"}
    kwargs = {"vision_prompt": "Read the screenshot. Answer: $LETTER.",
              "post_prompt": "Only a letter.", "pre_prompt": "STANDARD"}
    assert utils.mmmu_pro_doc_to_text(doc, kwargs) == kwargs["vision_prompt"]
    doc["options"] = "not even parseable scoring metadata"
    assert utils.mmmu_pro_doc_to_text(doc, kwargs) == kwargs["vision_prompt"]
    images = utils.mmmu_pro_doc_to_visual(doc)
    assert len(images) == 1 and images[0].size == (12, 12)


def test_standard_still_receives_question_and_choices():
    doc = {"question": "Which is correct?", "options": "['Alpha', 'Beta']"}
    prompt = utils.mmmu_pro_doc_to_text(doc, {"post_prompt": "Answer directly."})
    assert prompt == "Which is correct?\nOptions:\nA. Alpha\nB. Beta\nAnswer directly."
