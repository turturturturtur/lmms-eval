import re

from lmms_eval.filters.extraction import ExtendedRegexFilter


def ai2d_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if lmms_eval_specific_kwargs["prompt_format"] == "mcq":
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"
    elif lmms_eval_specific_kwargs["prompt_format"] == "qa":
        options = "\n".join(choices)
        return f"{pre_prompt}{question}{options}{post_prompt}"
    elif lmms_eval_specific_kwargs["prompt_format"] == "mcq_xcomposer":
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = " ".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        return f"{pre_prompt}{question}\nContext: N/A\n{choices_str}{post_prompt}"
    else:
        raise ValueError(f"Unknown prompt format: {lmms_eval_specific_kwargs['prompt_format']}")


def ai2d_doc_to_visual(doc):
    image = doc.get("image")
    if image is None:
        raise ValueError(f"AI2D sample is missing image: question={doc.get('question', '')[:80]!r}")
    return [image.convert("RGB")]


def ai2d_doc_to_target(doc, model_specific_target_kwargs):
    if "options" not in doc or "answer" not in doc:
        raise KeyError(f"AI2D sample must contain options and answer fields, got keys={sorted(doc.keys())}")
    if model_specific_target_kwargs == "mcq":
        len_choices = len(doc["options"])
        options = [chr(ord("A") + i) for i in range(len_choices)]
        return options[int(doc["answer"])]
    elif model_specific_target_kwargs == "qa":
        return doc["options"][int(doc["answer"])]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            num_options = len(doc["options"])
            valid_letters = "".join(chr(ord("A") + i) for i in range(num_options))
            answer_patterns = [
                re.compile(rf"^\s*[\(\[]?\s*([{valid_letters}])\s*[\)\]\.]?\s*$", re.IGNORECASE),
                re.compile(rf"(?:answer|option|choice)(?:\s+is)?\s*[:：]?\s*[\(\[]?\s*([{valid_letters}])\s*[\)\]\.]?", re.IGNORECASE),
                re.compile(rf"^\s*([{valid_letters}])[\.\)]\s+", re.IGNORECASE),
                re.compile(rf"\(([{valid_letters}])\)", re.IGNORECASE),
            ]
            normalized_choices = {
                self.filter_ignores(choice.strip()): chr(ord("A") + i)
                for i, choice in enumerate(doc["options"])
                if isinstance(choice, str) and choice.strip()
            }

            # Process each response
            filtered = []
            for resp in r:
                extracted = None
                for pattern in answer_patterns:
                    match = pattern.search(resp)
                    if match:
                        extracted = match.group(1).upper()
                        break
                if extracted is None:
                    normalized_resp = self.filter_ignores(resp.strip())
                    for choice_text, letter in normalized_choices.items():
                        if choice_text and choice_text in normalized_resp:
                            extracted = letter
                            break
                filtered.append(extracted if extracted is not None else resp.strip())

            # Assuming we need the first response that matches or the original response
            filtered_resps.append(filtered[0])

        return filtered_resps
