import argparse
import os
import evaluate
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Audio
from transformers import pipeline, AutoFeatureExtractor
import pandas as pd
import re

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def main(args):
    device = args.device
    model_id = args.model_id
    dataset_id = args.dataset
    text_column_name = args.text_column_name

    if os.path.isdir(dataset_id):
        dataset = load_from_disk(dataset_id)[args.split]
    else:
        dataset = load_dataset(
            dataset_id,
            args.name,
            split=args.split,
            # use_auth_token=True,
        )

    # load processor
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    sampling_rate = feature_extractor.sampling_rate
    print(f"Sampling rate: {sampling_rate}")

    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # load eval pipeline
    asr = pipeline("automatic-speech-recognition", model=model_id, device=device)

    # lowercase the text
    def lowercase(batch):
        batch[text_column_name] = batch[text_column_name].lower()
        return batch
    dataset = dataset.map(lowercase)

    # remove unwanted characters from the reference transcript
    def remove_special_characters(text, chars_to_ignore):
        chars_to_remove_regex = f"[{re.escape(chars_to_ignore)}]"
        text = re.sub(chars_to_remove_regex, "", text)
        return {text_column_name: text}

    if args.chars_to_ignore is not None:
        dataset = dataset.map(remove_special_characters, fn_kwargs={"chars_to_ignore": args.chars_to_ignore}, input_columns=[text_column_name])

    # replace unwanted characters with a space in the reference transcript
    def replace_whitespace(text, whitespace_chars):
        whitespace_chars_regex = f"[{re.escape(whitespace_chars)}]"
        text = re.sub(whitespace_chars_regex, " ", text)
        # remove multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return {text_column_name: text}

    if args.whitespace_chars is not None:
        dataset = dataset.map(replace_whitespace, fn_kwargs={"whitespace_chars": args.whitespace_chars}, input_columns=[text_column_name])

    predictions = []
    references = []
    durations = []
    paths = []
    class_values = [] if args.class_column_name else None

    # Check if class_column_name exists in the dataset
    if args.class_column_name and args.class_column_name not in dataset.column_names:
        print(f"Warning: Class column '{args.class_column_name}' not found in the dataset. Ignoring class-based analysis.")
        args.class_column_name = None

    for item in tqdm(dataset, desc='Decode Progress'):
        # use the "path" column if available, otherwise use the "audio" column
        path = item.get("path", None)
        if path is not None:
            audio_filepath = path
        else:
            audio_filepath = item["audio"]["path"]
        duration = item["audio"]["array"].shape[0] / item["audio"]["sampling_rate"]
        paths.append(audio_filepath)
        durations.append(duration)

        # run inference
        prediction = asr(item["audio"]["array"])
        pred_text = prediction["text"]

        # remove unwanted characters from the predicted transcript
        if args.chars_to_ignore is not None:
            pred_text = remove_special_characters(pred_text, args.chars_to_ignore)[text_column_name]

        # replace unwanted characters with a space in the predicted transcript
        if args.whitespace_chars is not None:
            pred_text = replace_whitespace(pred_text, args.whitespace_chars)[text_column_name]

        predictions.append(pred_text)
        references.append(item[text_column_name])
        
        # store class value if specified
        if args.class_column_name:
            class_values.append(item[args.class_column_name])

    # Calculate overall metrics
    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)
    cer = cer_metric.compute(references=references, predictions=predictions)
    cer = round(100 * cer, 2)
    print("\nOverall WER : ", wer)
    print("Overall CER : ", cer)

    # Calculate metrics by class if class_column_name is provided
    if args.class_column_name:
        print(f"\nMetrics by {args.class_column_name}:")
        class_df = pd.DataFrame({
            "reference": references,
            "prediction": predictions,
            "class": class_values
        })
        
        for class_value, group in class_df.groupby("class"):
            group_wer = wer_metric.compute(references=group["reference"].tolist(), 
                                          predictions=group["prediction"].tolist())
            group_cer = cer_metric.compute(references=group["reference"].tolist(), 
                                          predictions=group["prediction"].tolist())
            group_wer = round(100 * group_wer, 2)
            group_cer = round(100 * group_cer, 2)
            print(f"  {class_value} - WER: {group_wer}, CER: {group_cer} (samples: {len(group)})")

    # write to manifest
    if args.output is None:
        model_name = model_id.strip('/').split('/')[-1] if '/' in model_id else model_id
        dataset_name = dataset_id.strip('/').split('/')[-1] if '/' in dataset_id else dataset_id
        output = '-'.join([model_name, dataset_name, args.split]) + '.jsonl'
    else:
        output = args.output
    
    # Include class values in the output if available
    df_dict = {
        "audio_filepath": paths,
        "text": references,
        "pred_text": predictions,
        "duration": durations
    }
    if args.class_column_name:
        df_dict[args.class_column_name] = class_values
        
    df = pd.DataFrame(df_dict)
    df.to_json(output, orient='records', lines=True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="mozilla-foundation/common_voice_11_0",
        help="Dataset from huggingface to evaluate the model on. Example: mozilla-foundation/common_voice_11_0",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        help="Config of the dataset. Eg. 'hi' for the Hindi split of Common Voice",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=False,
        default="test",
        help="Split of the dataset. Eg. 'test'",
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        required=False,
        default="text",
        help="Name of the column containing the transcript",
    )
    parser.add_argument(
        "--class_column_name",
        type=str,
        required=False,
        default=None,
        help="Name of the column containing the class you'd like to calculate the metrics for. This could be 'gender', 'age', etc.",
    )
    parser.add_argument(
        "--chars_to_ignore",
        type=str,
        required=False,
        help="Characters to remove from the reference transcript",
    )
    parser.add_argument(
        "--whitespace_chars",
        type=str,
        required=False,
        help="Characters to replace with a space in both reference and predicted transcripts",
    )
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="Device to use for inference.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default=None,
        help="Path to NeMo manifest file to save the results",
    )

    args = parser.parse_args()
    main(args)
