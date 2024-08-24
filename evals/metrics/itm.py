import re
import numpy as np

def compute_contrastive_score_standard(predictions: list):
    predictions = np.array(predictions)
    correct = (predictions == 0).sum()  # First index is always the correct index in the ground truth
    return {"accuracy": (correct / len(predictions)) * 100, "num_samples": len(predictions)}


def compute_contrastive_score_itg(contrastive_scores: list):
    def text_correct(result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(result):
        return image_correct(result) and text_correct(result)

    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    for result in contrastive_scores:
        text_correct_count += 1 if text_correct(result) else 0
        image_correct_count += 1 if image_correct(result) else 0
        group_correct_count += 1 if group_correct(result) else 0

    denominator = len(contrastive_scores)
    return {
        "text_score": round(text_correct_count * 100 / denominator, 2),
        "image_score": round(image_correct_count * 100 / denominator, 2),
        "group_score": round(group_correct_count * 100 / denominator, 2),
    }


def compute_contrastive_score_itg_by_tag(score_data: list):
    tagged_score_data = {}
    unique_tags = set(score_data["tag"])

    for tag in unique_tags:
        mask = [i for i, t in enumerate(score_data["tag"]) if t == tag]
        tagged_score_data[tag] = {
            "c0_i0": score_data["c0_i0"][mask],
            "c0_i1": score_data["c0_i1"][mask],
            "c1_i0": score_data["c1_i0"][mask],
            "c1_i1": score_data["c1_i1"][mask],
        }

    tagged_scores = {}
    for tag, tag_res in tagged_score_data.items():
        tagged_scores[tag] = compute_contrastive_score_itg(tag_res)
        tagged_scores[tag]["num_samples"] = len(tag_res["c0_i0"])

    tagged_scores = dict(sorted(tagged_scores.items(), key=lambda x: x[1]["num_samples"], reverse=True))
    return tagged_scores


def compute_mllm_score_standard(results: dict):
    total_samples = len(results)
    success_count = sum(data["match"] for data in results.values())
    accuracy = (success_count / total_samples) * 100
    return {"accuracy": accuracy, "num_samples": total_samples}


def compute_mllm_score_itg(results: dict):
    total_samples = len(results)
    text_success_count = sum(data["text"][0]["match"] and data["text"][1]["match"] for data in results.values())
    image_success_count = sum(data["image"][0]["match"] and data["image"][1]["match"] for data in results.values())
    combined_success_count = sum(
        (data["text"][0]["match"] and data["text"][1]["match"])
        and (data["image"][0]["match"] and data["image"][1]["match"])
        for data in results.values()
    )

    text_success_rate = (text_success_count / total_samples) * 100
    image_success_rate = (image_success_count / total_samples) * 100
    combined_success_rate = (combined_success_count / total_samples) * 100

    curr_results = {
        "accuracy": {
            "text_score": text_success_rate,
            "image_score": image_success_rate,
            "group_score": combined_success_rate,
        },
        "num_samples": total_samples,
    }

    return curr_results


# calculate the score proposed in VALSE
def compute_valse_score(result):
    def cal_valse_score_pair(result):
        return result["c0_i0"] > result["c1_i0"]

    def cal_valse_score_pc(result):
        return result["c0_i0"] > 0.5

    def cal_valse_score_pf(result):
        return result["c1_i0"] < 0.5

    def cal_valse_score_acc(result):
        true_cnt = (result["c0_i0"] > 0.5).sum() + (result["c1_i0"] < 0.5).sum()
        return true_cnt / (len(result["c0_i0"]) * 2)

    def cal_score(list_correct):
        correct_cnt = list_correct.sum()
        denominator = len(list_correct)
        return correct_cnt / denominator

    return (
        cal_valse_score_acc(result),
        min(cal_score(cal_valse_score_pc(result)), cal_score(cal_valse_score_pf(result))),
        cal_score(cal_valse_score_pair(result)),
    )


def normalize_model_prediction(text: str):
    # Extract the text inside the brackets
    match = re.search(r"\((.*?)\)", text)
    if match:
        text = match.group(1)

    # Look for a pattern with a letter followed by a parenthesis
    match = re.search(r"([A-Za-z])\)", text)
    if match:
        # Return only the letter before the parenthesis
        text = match.group(1)

    # only keep before the first period or newline
    text = text.split("\n")[0]
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Convert multiple spaces to a single space and trim leading/trailing spaces
    text = re.sub(r"\s+", " ", text).strip()
    # make it lowercase
    text = text.lower()
    return text


def is_match_with_label(output, label):
    answer_phrase = f"ANSWER: {label}"
    normalized_output = normalize_model_prediction(output)
    normalized_correct_answer = normalize_model_prediction(label)
    normalized_answer_phrase = normalize_model_prediction(answer_phrase)

    string_match_phrase = normalized_correct_answer in normalized_output.split()
    match_phrase = re.search(re.escape(normalized_answer_phrase), normalized_output, re.IGNORECASE) is not None
    match_answer = normalized_correct_answer == normalized_output
    match = match_phrase or match_answer or string_match_phrase

    return match
