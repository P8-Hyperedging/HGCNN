import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from data.data import load_postgres_review_data, load_postgres_business_list_data
from data.n_preprocessing import get_business_id_mapping, group_reviews_by_user

def bar(count, total, width=40):
    filled = int(round(count / total * width))
    return "█" * filled + "░" * (width - filled)

def print_distribution(labels, title, class_names):
    labels = np.array(labels, dtype=int)
    total = len(labels)
    counts = np.bincount(labels, minlength=len(class_names))

    print(f"\n{'='*60}")
    print(f"  {title}  (n={total})")
    print(f"{'='*60}")
    print(f"  {'Class':<14} {'Count':>6}  {'%':>6}  Distribution")
    print(f"  {'-'*56}")
    for i, name in enumerate(class_names):
        c = counts[i]
        pct = c / total * 100
        print(f"  {name:<14} {c:>6}  {pct:>5.1f}%  {bar(c, total)}")
    print(f"  {'-'*56}")
    majority = counts.max() / total * 100
    print(f"  Majority-class baseline: {majority:.1f}%")
    print(f"  Random baseline:         {100/len(class_names):.1f}%")

if __name__ == "__main__":
    print("Loading reviews...")
    reviews = load_postgres_review_data()

    user_to_businesses = group_reviews_by_user(reviews)
    business_ids_set = {bid for businesses in user_to_businesses.values() for bid, _ in businesses}
    business_ids = sorted(business_ids_set)

    print(f"Loading {len(business_ids)} businesses...")
    businesses = load_postgres_business_list_data(business_ids)

    stars = [b.stars for b in businesses]

    # 9-class distribution (0.5-step)
    labels_9 = [round(s * 2) - 2 for s in stars]
    names_9  = ["1.0★", "1.5★", "2.0★", "2.5★", "3.0★", "3.5★", "4.0★", "4.5★", "5.0★"]
    print_distribution(labels_9, "9-CLASS DISTRIBUTION (0.5-step)", names_9)

    # 5-class distribution (full stars)
    labels_5 = [round(s) - 1 for s in stars]
    names_5  = ["1★", "2★", "3★", "4★", "5★"]
    print_distribution(labels_5, "5-CLASS DISTRIBUTION (full stars)", names_5)
