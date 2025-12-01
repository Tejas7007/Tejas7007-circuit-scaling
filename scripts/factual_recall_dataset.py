#!/usr/bin/env python
import json
import os
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class FactualRecallExample:
    prompt: str
    subject: str
    correct_object: str
    distractors: List[str]

FACTUAL_FACTS = [
    ("The Eiffel Tower is located in", "Paris", ["London", "Berlin", "Rome"]),
    ("The capital of France is", "Paris", ["Berlin", "Madrid", "Rome"]),
    ("The capital of Japan is", "Tokyo", ["Seoul", "Beijing", "Osaka"]),
    ("The capital of India is", "New Delhi", ["Mumbai", "Kolkata", "Chennai"]),
    ("The Great Wall of China is located in", "China", ["Japan", "Korea", "Vietnam"]),
    ("Albert Einstein was born in", "Ulm", ["Berlin", "Vienna", "Zurich"]),
    ("Marie Curie was born in", "Warsaw", ["Paris", "London", "Prague"]),
    ("Isaac Newton was born in", "Woolsthorpe", ["London", "Cambridge", "Oxford"]),
    ("The tallest mountain in the world is", "Mount Everest",
     ["K2", "Kangchenjunga", "Makalu"]),
    ("The largest ocean on Earth is the", "Pacific Ocean",
     ["Atlantic Ocean", "Indian Ocean", "Arctic Ocean"]),
]

def build_examples() -> List[FactualRecallExample]:
    examples = []
    for prefix, correct, distractors in FACTUAL_FACTS:
        prompt = prefix.strip() + " "
        examples.append(FactualRecallExample(
            prompt=prompt,
            subject=prefix.strip(),
            correct_object=correct,
            distractors=distractors,
        ))
    return examples

def main(output_path: str = "data/factual_recall_prompts.jsonl") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    examples = build_examples()
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")
    print(f"Wrote {len(examples)} factual recall prompts to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str,
                        default="data/factual_recall_prompts.jsonl")
    args = parser.parse_args()
    main(args.output_path)

