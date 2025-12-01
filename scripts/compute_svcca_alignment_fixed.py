#!/usr/bin/env python3
"""
ioi_dataset.py

Generate IOI (Indirect Object Identification) style prompts for 
mechanistic interpretability experiments.

IOI prompts follow the pattern:
"When [Name1] and [Name2] went to the [place], [Name2] gave a [object] to"
Expected completion: [Name1]
"""

import random
from typing import List, Tuple

# Name lists
NAMES = [
    "Mary", "John", "Alice", "Bob", "Sarah", "Michael", "Emma", "James",
    "Lisa", "David", "Jennifer", "William", "Jessica", "Daniel", "Emily",
    "Christopher", "Ashley", "Matthew", "Amanda", "Andrew", "Stephanie",
    "Joshua", "Nicole", "Brandon", "Elizabeth", "Ryan", "Megan", "Justin",
    "Lauren", "Kevin", "Rachel", "Brian", "Samantha", "Tyler", "Katherine",
    "Jason", "Michelle", "Aaron", "Heather", "Adam", "Amber", "Nathan",
    "Brittany", "Zachary", "Rebecca", "Patrick", "Laura", "Sean", "Danielle"
]

# Places
PLACES = [
    "store", "park", "beach", "restaurant", "library", "museum", "cafe",
    "gym", "office", "school", "hospital", "airport", "station", "mall",
    "theater", "garden", "market", "hotel", "bank", "church"
]

# Objects
OBJECTS = [
    "drink", "book", "gift", "letter", "key", "phone", "bag", "ticket",
    "flower", "card", "pen", "watch", "ring", "hat", "bottle", "box",
    "file", "note", "map", "toy"
]

# Templates for IOI prompts
TEMPLATES = [
    "When {name1} and {name2} went to the {place}, {name2} gave a {object} to",
    "After {name1} and {name2} arrived at the {place}, {name2} handed the {object} to",
    "{name1} and {name2} were at the {place}. {name2} passed the {object} to",
    "At the {place}, {name1} met {name2}. Then {name2} gave a {object} to",
    "{name1} and {name2} visited the {place}. {name2} offered a {object} to",
]


def make_ioi_prompts(
    n_prompts: int,
    seed: int = 42,
    templates: List[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Generate n IOI-style prompts.

    Args:
        n_prompts: Number of prompts to generate
        seed: Random seed for reproducibility
        templates: Optional custom templates (uses default if None)

    Returns:
        Tuple of (prompts, expected_answers) where expected_answers[i]
        is the name that should be predicted for prompts[i]
    """
    random.seed(seed)
    
    if templates is None:
        templates = TEMPLATES

    prompts = []
    answers = []

    for _ in range(n_prompts):
        # Pick two different names
        name1, name2 = random.sample(NAMES, 2)
        place = random.choice(PLACES)
        obj = random.choice(OBJECTS)
        template = random.choice(templates)

        prompt = template.format(
            name1=name1,
            name2=name2,
            place=place,
            object=obj
        )
        prompts.append(prompt)
        answers.append(name1)  # The indirect object (first name) is the answer

    return prompts, answers


def make_ioi_dataset(n_prompts: int, seed: int = 42) -> dict:
    """
    Create a full IOI dataset dictionary.

    Args:
        n_prompts: Number of prompts
        seed: Random seed

    Returns:
        Dictionary with 'prompts', 'answers', and metadata
    """
    prompts, answers = make_ioi_prompts(n_prompts, seed)
    return {
        "prompts": prompts,
        "answers": answers,
        "n_prompts": n_prompts,
        "seed": seed,
    }


if __name__ == "__main__":
    # Test the prompt generation
    prompts, answers = make_ioi_prompts(5)
    print("Sample IOI prompts:")
    print("-" * 60)
    for p, a in zip(prompts, answers):
        print(f"Prompt: {p}")
        print(f"Expected: {a}")
        print()
