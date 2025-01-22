from collections import defaultdict

import pandas as pd


def compute_contrastive_accuracy(contrastive_scores: list):
    split_correct_count = defaultdict(
        lambda: {"text": 0, "image": 0, "group": 0, "count": 0}
    )

    for pred_dict in contrastive_scores:
        split = pred_dict["category"]

        # Direct comparisons for text and image
        text_correct = (
            pred_dict["T0_I0"] > pred_dict["T1_I0"]
            and pred_dict["T1_I1"] > pred_dict["T0_I1"]
        )
        image_correct = (
            pred_dict["T0_I0"] > pred_dict["T0_I1"]
            and pred_dict["T1_I1"] > pred_dict["T1_I0"]
        )

        split_correct_count[split]["text"] += text_correct
        split_correct_count[split]["image"] += image_correct
        split_correct_count[split]["group"] += text_correct and image_correct
        split_correct_count[split]["count"] += 1

    full_correct_count = {}
    for split in split_correct_count:
        denominator = split_correct_count[split]["count"]
        curr_text = round(split_correct_count[split]["text"] * 100 / denominator, 2)
        curr_image = round(split_correct_count[split]["image"] * 100 / denominator, 2)
        curr_group = round(split_correct_count[split]["group"] * 100 / denominator, 2)
        full_correct_count[split] = {
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
