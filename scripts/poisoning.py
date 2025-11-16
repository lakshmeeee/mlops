# poison_label_swap.py
import argparse
import pandas as pd
import numpy as np

def poison_labels(df, label_col, fraction, seed=42):
    rng = np.random.RandomState(seed)
    df_poison = df.copy()
    n = len(df)
    k = int(np.round(n * fraction))

    idx = rng.choice(n, size=k, replace=False)
    classes = df[label_col].unique()

    for i in idx:
        original = df_poison.at[i, label_col]
        new_label = rng.choice(classes[classes != original])
        df_poison.at[i, label_col] = new_label

    return df_poison

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/data.csv")
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv("./data/data.csv")
    df_poison = poison_labels(df, "species", args.fraction, args.seed)

    out = f"./poisoned_data/data_poison_{int(args.fraction*100)}.csv"
    df_poison.to_csv(out, index=False)
    print(f"Saved label-swapped poisoned data to {out}")
