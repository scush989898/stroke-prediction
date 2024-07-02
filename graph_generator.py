import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def load_data(algorithm_name):
    summary_file = f"weka-results/metrics-summary/{algorithm_name}_summary.csv"
    confusion_matrix_file = (
        f"weka-results/confusion-matrix/{algorithm_name}_confusion_matrix.csv"
    )

    summary_df = pd.read_csv(summary_file)
    confusion_matrix_df = pd.read_csv(confusion_matrix_file, index_col=0)
    return summary_df, confusion_matrix_df


def process_summary_df(summary_df):
    summary_df["Valor"] = (
        summary_df["Valor"]
        .str.replace("%", "")
        .str.replace(",", ".")
        .astype(float)
    )
    return summary_df


algorithm_names = [
    "1-decision_table",
    "2-jrip",
    "3-one_r",
    "4-part",
    "5-zero_r",
    "6-decision_stump",
    "7-hoeffding_tree",
    "8-j48",
    "9-lmt",
    "10-random_forest",
    "11-random_tree",
    "12-rep_tree",
]

summaries = {}
confusion_matrices = {}

for algorithm in algorithm_names:
    summary_df, confusion_matrix_df = load_data(algorithm)
    summary_df = process_summary_df(summary_df)
    summaries[algorithm] = summary_df
    confusion_matrices[algorithm] = confusion_matrix_df

metrics = [
    metric.strip()
    for metric in summaries[algorithm_names[0]]["Métrica"].tolist()
]

for metric in metrics:
    plt.figure(figsize=(14, 8))
    values = []
    for algorithm in algorithm_names:
        metric_value = summaries[algorithm].loc[
            summaries[algorithm]["Métrica"].str.strip() == metric, "Valor"
        ]
        if not metric_value.empty:
            values.append(metric_value.values[0])
        else:
            print(
                f"""Métrica '{metric}' não encontrada
                para o algoritmo '{algorithm}'"""
            )
            values.append(0)
    sns.barplot(x=algorithm_names, y=values, palette="Blues_d")
    plt.title(f"{metric}")
    plt.ylabel("Value")
    plt.xlabel("Algorithm")
    plt.xticks(rotation=45)
    plt.show()

for algorithm, confusion_matrix_df in confusion_matrices.items():
    plt.figure(figsize=(10, 6))
    sns.heatmap(confusion_matrix_df, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Consusion Matrix - {algorithm}")
    plt.ylabel("Real")
    plt.xlabel("Predicted Value")
    plt.show()
