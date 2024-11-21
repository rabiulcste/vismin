import pandas as pd
from collections import defaultdict


VALID_CONTRASTIVE_KEYS = ["T0_I0", "T0_I1", "T1_I0", "T1_I1"]

def evaluate_relation(pred1, pred2, operator):
    if operator == ">":
        return pred1 > pred2
    elif operator == "<":
        return pred1 < pred2
    elif operator == "=":
        return pred1 == pred2
    else:
        raise ValueError(f"Invalid operator: {operator}")

def compute_contrastive_accuracy(contrastive_scores: list, ground_truth: list):
    category_correct_count = defaultdict(lambda: {"text": 0, "image": 0, "group": 0, "count": 0})

    for pred_dict, gt in zip(contrastive_scores, ground_truth):
        category = gt["category"]

        # Parse expected relationships from ground truth
        text_relations = gt["Text"].split(" & ")
        image_relations = gt["Image"].split(" & ")
        assert all(key in pred_dict for key in VALID_CONTRASTIVE_KEYS), f"All keys must be present in the prediction dictionary {pred_dict}"

        # Evaluate text relationships
        text_correct = all(
            evaluate_relation(pred_dict[relation.split(" ")[0]], pred_dict[relation.split(" ")[2]], relation.split(" ")[1])
            for relation in text_relations
        )

        # Evaluate image relationships
        image_correct = all(
            evaluate_relation(pred_dict[relation.split(" ")[0]], pred_dict[relation.split(" ")[2]], relation.split(" ")[1])
            for relation in image_relations
        )

        category_correct_count[category]["text"] += text_correct
        category_correct_count[category]["image"] += image_correct
        category_correct_count[category]["group"] += text_correct and image_correct
        category_correct_count[category]["count"] += 1

    full_correct_count = {}
    for category in category_correct_count: 
        denominator = category_correct_count[category]["count"]
        curr_text = round(category_correct_count[category]["text"] * 100 / denominator, 2)
        curr_image = round(category_correct_count[category]["image"] * 100 / denominator, 2)
        curr_group = round(category_correct_count[category]["group"] * 100 / denominator, 2)
        full_correct_count[category] = {
            "text": curr_text,
            "image": curr_image,
            "group": curr_group,
            "count": denominator,
        }

    return full_correct_count


def compute_vqa_accuracy(predictions: list, ground_truth: list):
    total_samples = len(predictions)
    text_success_count = sum(
        predictions[idx]["Text_Q0"] == ground_truth[idx]["Text_Q0"]
        and predictions[idx]["Text_Q1"] == ground_truth[idx]["Text_Q1"]
        for idx in range(total_samples)
    )
    image_success_count = sum(
        predictions[idx]["Image_Q0"] == ground_truth[idx]["Image_Q0"]
        and predictions[idx]["Image_Q1"] == ground_truth[idx]["Image_Q1"]
        for idx in range(total_samples)
    )
    combined_success_count = sum(
        (
            predictions[idx]["Text_Q0"] == ground_truth[idx]["Text_Q0"]
            and predictions[idx]["Text_Q1"] == ground_truth[idx]["Text_Q1"]
        )
        and (
            predictions[idx]["Image_Q0"] == ground_truth[idx]["Image_Q0"]
            and predictions[idx]["Image_Q1"] == ground_truth[idx]["Image_Q1"]
        )
        for idx in range(total_samples)
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
