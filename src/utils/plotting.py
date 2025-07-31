import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
import re
import seaborn as sns
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_final_precision_bars_from_dict(precision_last: dict, save_path: Path, perc_success=0.02):
    label_to_color = {
        'LLM': lambda v: 'green' if v >= perc_success else 'red',
        'Expert': lambda v: '#DAA520' if v >= perc_success else 'red',
        'GPTree': lambda v: '#1f77b4' if v >= perc_success else 'red'
    }

    subsets = {
        'LLM': {},
        'LLM + Expert': {},
        'LLM + Expert + GPTree': {}
    }

    for k, (val, source) in precision_last.items():
        if source == 'LLM':
            subsets['LLM'][k] = (val, source)
        if source in ['LLM', 'Expert']:
            subsets['LLM + Expert'][k] = (val, source)
        subsets['LLM + Expert + GPTree'][k] = (val, source)

    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=False)
    titles = ["LLM", "LLM + Expert", "LLM + Expert + GPTree"]

    for i, (title_key, data) in enumerate(subsets.items()):
        items = sorted(data.items(), key=lambda x: x[1][0], reverse=True)
        vals, colors = [], []
        for _, (val, src) in items:
            vals.append(val)
            colors.append(label_to_color[src](val))

        indices = list(range(1, len(vals) + 1))
        axs[i].bar(indices, vals, color=colors)
        axs[i].axhline(y=perc_success, color='black', linewidth=2, linestyle='--')
        axs[i].set_title(f"Final Precision – {titles[i]}", fontsize=16)
        axs[i].set_ylim(0, 0.15)
        axs[i].set_yticks([0, 0.05, 0.1, 0.15])
        axs[i].set_ylabel("Precision Value", fontsize=12)
        axs[i].set_xticks(indices)
        axs[i].tick_params(axis='x', labelsize=9)
        axs[i].tick_params(axis='y', labelsize=10)
        axs[i].spines[['top', 'right']].set_visible(False)

    axs[2].set_xlabel("Question Index", fontsize=14)

    legend_patches = [
        mpatches.Patch(color='green', label='LLM ≥ 2%'),
        mpatches.Patch(color='#DAA520', label='Expert ≥ 2%'),
        mpatches.Patch(color='#1f77b4', label='GPTree ≥ 2%'),
        mpatches.Patch(color='red', label='< 2%')
    ]
    axs[2].legend(handles=legend_patches, loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300)
    logger.info(f"Saved precision bar plots to {save_path}")
    plt.close()

def plot_final_f1_bars_from_dict(f1_last: dict, save_path: Path, threshold: float = 0.5):
    """
    f1_last: dict mapping "Question (Source)" -> (f1_value, source_label)
    threshold: horizontal line and color cutoff on F1 score
    """
    label_to_color = {
        'LLM':     lambda v: 'green'    if v >= threshold else 'red',
        'Expert':  lambda v: '#DAA520' if v >= threshold else 'red',
        'GPTree':  lambda v: '#1f77b4' if v >= threshold else 'red',
    }

    subsets = {
        'LLM':                     {},
        'LLM + Expert':            {},
        'LLM + Expert + GPTree':   {}
    }

    # bucket each question into subsets
    for k, (val, src) in f1_last.items():
        if src == 'LLM':
            subsets['LLM'][k] = (val, src)
        if src in ('LLM', 'Expert'):
            subsets['LLM + Expert'][k] = (val, src)
        subsets['LLM + Expert + GPTree'][k] = (val, src)

    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=False)
    titles = ["LLM", "LLM + Expert", "LLM + Expert + GPTree"]

    for i, (title_key, data) in enumerate(subsets.items()):
        # sort by F1 descending
        items = sorted(data.items(), key=lambda x: x[1][0], reverse=True)

        vals   = [v for (_, (v, _)) in items]
        colors = [label_to_color[src](v) for (_, (v, src)) in items]
        indices = range(1, len(vals)+1)

        axs[i].bar(indices, vals, color=colors)
        axs[i].axhline(y=threshold, color='black', linewidth=2, linestyle='--')
        axs[i].set_title(f"Final F₁ Score – {titles[i]}", fontsize=16)
        axs[i].set_ylabel("F₁ Score", fontsize=12)
        axs[i].set_ylim(0, 1.0)
        axs[i].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axs[i].tick_params(axis='x', labelsize=9)
        axs[i].tick_params(axis='y', labelsize=10)
        axs[i].spines[['top', 'right']].set_visible(False)

    axs[2].set_xlabel("Question Index", fontsize=14)

    legend_patches = [
        mpatches.Patch(color='green',    label=f'LLM ≥ {threshold:.2f}'),
        mpatches.Patch(color='#DAA520',  label=f'Expert ≥ {threshold:.2f}'),
        mpatches.Patch(color='#1f77b4',  label=f'GPTree ≥ {threshold:.2f}'),
        mpatches.Patch(color='red',      label=f'< {threshold:.2f}')
    ]
    axs[2].legend(handles=legend_patches, loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300)
    logger.info(f"Saved F₁ bar plots to {save_path}")
    plt.close()


def plot_precision_histograms_from_dict(precision_last: dict, save_path: Path, perc_success=0.02):
    groups = {
        'LLM Questions': ['LLM'],
        'LLM + Expert Questions': ['LLM', 'Expert'],
        'LLM + Expert + GPTree Questions': ['LLM', 'Expert', 'GPTree']
    }

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for i, (title, sources) in enumerate(groups.items()):
        values = [v for (v, src) in precision_last.values() if src in sources]
        axs[i].hist(values, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axs[i].axvline(x=perc_success, color='red', linestyle='--', linewidth=2)
        axs[i].set_title(title, fontsize=15)
        axs[i].set_ylabel("Frequency", fontsize=13)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)

    axs[2].set_xlabel("Final Precision", fontsize=13)
    axs[2].tick_params(axis='x', labelsize=13)
    axs[2].tick_params(axis='y', labelsize=13)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300)
    logger.info(f"Saved precision histogram plots to {save_path}")
    plt.close()

    
def plot_precision_heatmaps(results_file_path: Path, save_path: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from matplotlib.backends.backend_pdf import PdfPages

    results_df = pd.read_csv(results_file_path)

    # Heatmap 1: Precision Rate
    fig1, ax1 = plt.subplots(figsize=(24, 12))
    precision_data = results_df.pivot(index='Score_Threshold', columns='Num_Questions', values='Precision_Rate')

    # Create custom mask where Score_Threshold > Num_Questions
    score_vals = precision_data.index.values[:, None]
    num_q_vals = precision_data.columns.values[None, :]
    mask1 = score_vals > num_q_vals  # shape: (num_score_thresholds, num_questions)

    sns.heatmap(
        precision_data,
        annot=True,
        cmap='viridis',
        fmt='.2f',
        linewidths=.5,
        linecolor='white',
        annot_kws={"size": 8},
        mask=mask1,
        ax=ax1,
        vmin=0,
        vmax=0.2
    )
    ax1.set_title('Grid Search Heatmap of Precision Rate', fontsize=24, pad=20)
    ax1.set_xlabel('Number of Questions Kept', fontsize=20)
    ax1.set_ylabel('Score Threshold', fontsize=20)
    ax1.tick_params(axis='x', labeltop=True, labelsize=12, rotation=90)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.xaxis.set_label_position('top')
    ax1.collections[0].colorbar.ax.tick_params(labelsize=14)

    # Heatmap 2: Predicted Successful Founders
    fig2, ax2 = plt.subplots(figsize=(24, 12))
    success_data = results_df.pivot(index='Score_Threshold', columns='Num_Questions', values='Predicted_Successful_Founders')

    # Reuse the same masking logic
    mask2 = score_vals > num_q_vals

    sns.heatmap(
        success_data,
        annot=True,
        cmap='Blues',
        fmt='.0f',
        linewidths=.5,
        linecolor='white',
        annot_kws={"size": 8},
        mask=mask2,
        ax=ax2
    )
    ax2.set_title('Grid Search Heatmap of Predicted Successful Founders', fontsize=24, pad=20)
    ax2.set_xlabel('Number of Questions Kept', fontsize=20)
    ax2.set_ylabel('Score Threshold', fontsize=20)
    ax2.tick_params(axis='x', labeltop=True, labelsize=12, rotation=90)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.xaxis.set_label_position('top')
    ax2.collections[0].colorbar.ax.tick_params(labelsize=14)

    # Save both to PDF
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig1, bbox_inches='tight')
        pdf.savefig(fig2, bbox_inches='tight')
    logger.info(f"Saved precision heatmaps to {save_path}")

    plt.close(fig1)
    plt.close(fig2)


def plot_combined_precision_heatmap_grid(
    result_files: dict[str, Path],
    save_path: Path,
    min_questions_to_plot: int = 1,
    max_questions_to_plot: int = 70
):
    fig, axs = plt.subplots(1, 3, figsize=(24, 8), sharey=True)

    for i, (label, file_path) in enumerate(result_files.items()):
        df = pd.read_csv(file_path)

        # Filter and pivot
        df = df[(df['Num_Questions'] >= min_questions_to_plot) & (df['Num_Questions'] <= max_questions_to_plot)]
        heatmap_data = df.pivot(index='Score_Threshold', columns='Num_Questions', values='Precision_Rate')

        # Create mask where score_threshold > num_questions
        mask = np.zeros_like(heatmap_data, dtype=bool)
        for row in range(heatmap_data.shape[0]):
            for col in range(heatmap_data.shape[1]):
                if heatmap_data.index[row] > heatmap_data.columns[col]:
                    mask[row, col] = True

        # Plot
        ax = axs[i]
        sns.heatmap(
            heatmap_data,
            ax=ax,
            annot=False,
            cmap='viridis',
            fmt='.2f',
            linewidths=.5,
            linecolor='white',
            mask=mask,
            vmin=0,
            vmax=0.2,
            cbar=i == 2,  # Only show colorbar on the last subplot
            cbar_kws={"ticks": [0, 0.2]} if i == 2 else None
        )

        # Labels
        ax.set_title(label, fontsize=18)
        ax.set_xlabel('Questions in Ensemble', fontsize=16)
        if i == 0:
            ax.set_ylabel('Score Threshold', fontsize=16)
        else:
            ax.set_ylabel('')
            ax.set_yticks([])

        # Ticks and formatting
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', labeltop=True, top=True, bottom=False, labelbottom=False, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        custom_xticks = [x for x in heatmap_data.columns if x in [5, 15, 25, 35, 45, 55, 65]]
        ax.set_xticks([heatmap_data.columns.get_loc(x) + 0.5 for x in custom_xticks])
        ax.set_xticklabels(custom_xticks, rotation=90)

        if i == 0:
            custom_yticks = [y for y in heatmap_data.index if y in [5, 15, 25, 35, 45, 55, 65]]
            ax.set_yticks([heatmap_data.index.get_loc(y) + 0.5 for y in custom_yticks])
            ax.set_yticklabels(custom_yticks)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300)
    print(f"✅ Combined heatmap saved to: {save_path}")
    plt.close()


def plot_per_file(df_list, expert_df_list, gptree_df_list, predictions_dir: Path, save_dir: Path):
    total = len(df_list) + len(expert_df_list) + len(gptree_df_list)
    fig, axes = plt.subplots(total, 1, figsize=(14, 5 * total))
    if total == 1:
        axes = [axes]

    excluded = {'Founder Index', 'Dataset', 'Success', 'SUCCESS_PROPORTION'}
    metric_cols = ['Index', 'Question', 'Pass Rate', 'Prec', 'TP', 'FP', 'TN', 'FN', 'Rec', 'F1', 'F0.5',
                   'Prec_Train', 'Prec_Validation', 'Prec_Test', 'Prec_Mean']

    all_lists = [
        (df_list, "LLM", 0.1),
        (expert_df_list, " Expert", 0.2),
        (gptree_df_list, " GPTree", 0.2)
    ]

    plot_idx = 0
    for group, suffix, y_max in all_lists:
        for file_name, df in group:
            founder_cols = [col for col in df.columns if col not in metric_cols]
            precision_results = compute_precision(df, excluded, founder_cols)

            if plot_idx == 0:
                summary_data = []
                for question, precisions in precision_results.items():
                    val_1000 = precisions[999] if len(precisions) > 999 else None
                    val_4000 = precisions[4000] if len(precisions) > 4000 else None
                    val_8000 = precisions[7950] if len(precisions) > 7950 else None
                    summary_data.append({
                        "Question": question,
                        "Precision@1000": round(val_1000, 3) if val_1000 else None,
                        "Precision@4000": round(val_4000, 3) if val_4000 else None,
                        "Precision@8000": round(val_8000, 3) if val_8000 else None,
                    })
                summary_df = pd.DataFrame(summary_data).sort_values(by="Precision@8000", ascending=False)
                print("\n=== Precision Summary for Question Set 1 (Ranked by @8000) ===")
                print(summary_df.to_string(index=False))

            ax = axes[plot_idx]
            for question, precisions in precision_results.items():
                ax.plot(range(1, len(precisions) + 1), precisions, label=question, alpha=0.9)

            ax.axhline(y=0.02, color='gray', linestyle='--', linewidth=1.5)
            match = re.search(r'set_(\d+)', file_name)
            set_num = match.group(1) if match else "?"
            ax.set_title(f"Question Set - {set_num}{suffix}", fontsize=20)
            ax.set_xlabel("Number of Founders", fontsize=18)
            ax.set_ylabel("Precision Rate", fontsize=18)
            ax.set_xlim(0, 9000)
            ax.set_ylim(0, y_max)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=16)
            ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.02, 0.5))

            plot_idx += 1

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    save_path = save_dir / "precision_curves_per_set.pdf"
    plt.savefig(save_path, format='pdf', dpi=300)
    logger.info(f"Saved plot to {save_path}")
    plt.close()


def plot_combined_precision(llm_df_list, expert_df_list, gptree_df_list, save_path: Path, perc_success=0.02):
    import pickle

    fig, axs = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

    cache_path = save_path.parent / "precision_cache.pkl"
    excluded = {'Founder Index', 'Dataset', 'Success', 'SUCCESS_PROPORTION'}
    metric_cols = ['Index', 'Question', 'Pass Rate', 'Prec', 'TP', 'FP', 'TN', 'FN', 'Rec', 'F1', 'F0.5',
                   'Prec_Train', 'Prec_Validation', 'Prec_Test', 'Prec_Mean']

    # Either load or compute + cache all precision results
    if cache_path.exists():
        logger.info(f"Loading cached precision data from {cache_path}")
        with open(cache_path, "rb") as f:
            precision_cache = pickle.load(f)
    else:
        logger.info(f"Computing precision data and saving to {cache_path}")
        precision_cache = {}
        for group in [llm_df_list, expert_df_list, gptree_df_list]:
            for file_name, df in group:
                founder_cols = [col for col in df.columns if col not in metric_cols]
                precision_cache[file_name] = compute_precision(df, excluded, founder_cols)
        with open(cache_path, "wb") as f:
            pickle.dump(precision_cache, f)

    def plot_lines(df_list, ax, mode):
        for file_name, _ in df_list:
            precision_results = precision_cache.get(file_name, {})
            for question, precisions in precision_results.items():
                if precisions:
                    smoothed = pd.Series(precisions).rolling(window=1, min_periods=1).mean()
                    val_7950 = precisions[7949] if len(precisions) > 7949 else 0
                    if mode == "expert":
                        color = '#DAA520'  # gold
                    elif mode == "gptree":
                        color = '#1f77b4'  # blue
                    else:
                        color = 'green' if val_7950 >= perc_success else 'red'
                    ax.plot(range(1, len(smoothed) + 1), smoothed, label=question, alpha=0.8, color=color)

    # Plot lines
    plot_lines(llm_df_list, axs[0], mode="llm")
    plot_lines(llm_df_list, axs[1], mode="llm"); plot_lines(expert_df_list, axs[1], mode="expert")
    plot_lines(llm_df_list, axs[2], mode="llm"); plot_lines(expert_df_list, axs[2], mode="expert"); plot_lines(gptree_df_list, axs[2], mode="gptree")

    # Format subplots
    for i, ax in enumerate(axs):
        ax.axhline(y=perc_success, color='black', linewidth=3)
        ax.set_title(["LLM Questions", "LLM + Expert Questions", "LLM + Expert + GPTree Questions"][i], fontsize=20)
        ax.set_ylabel("Precision Rate", fontsize=18)
        ax.set_xlim(0, 9000)
        ax.set_ylim(0, 0.2)
        ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=16)
        ax.grid(False)

    axs[2].set_xlabel("Number of Founders", fontsize=18)

    red_patch = mpatches.Patch(color='red', label='LLM < 2%')
    green_patch = mpatches.Patch(color='green', label='LLM ≥ 2%')
    gold_patch = mpatches.Patch(color='#DAA520', label='Expert')
    blue_patch = mpatches.Patch(color='#1f77b4', label='GPTree')
    axs[2].legend(handles=[green_patch, red_patch, gold_patch, blue_patch], fontsize=12, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300)
    logger.info(f"Saved combined precision comparison plot to {save_path}")
    plt.close()


def compute_precision(df: pd.DataFrame, excluded_rows: set[str], founder_columns: list[str]) -> dict[str, list[float]]:
    precision_results = {}
    for question in df['Question'].unique():
        if question in excluded_rows:
            continue

        precisions = []
        tp = fp = 0
        question_df = df[df['Question'] == question]

        for _, row in question_df.iterrows():
            for founder in founder_columns:
                pred = row[founder]
                actual = df.loc[df['Question'] == 'Success', founder].values[0]

                if pd.isna(pred) or pd.isna(actual):
                    continue

                try:
                    pred = int(pred)
                    actual = int(actual)
                except ValueError:
                    continue

                if pred == 1 and actual == 1:
                    tp += 1
                elif pred == 1 and actual == 0:
                    fp += 1

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                precisions.append(precision)

        precision_results[question] = precisions
    return precision_results

def plot_weighted_grid_heatmaps_grid(
    input_paths: list[Path],  # still passed as list but should contain only one file
    titles: list[str],
    save_path: Path,
    recall_threshold: float = 0.0,
    min_questions_to_plot: int = 1,
    max_questions_to_plot: int = 70,
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.patches as mpatches

    # Mapping from panel title to Weighting_Scheme label
    title_to_scheme = {
        "Original": "none",
        "Linear Weights": "linear",
        "Squared Weights": "squared",
        "Cubed Weights": "cubed",
        "6th Power": "power6",
        "20th Power": "power20",
    }

    assert len(input_paths) == 1, "Expected one input file containing all weighting schemes"
    df = pd.read_csv(input_paths[0])
    print(f"\n📂 Reading: {input_paths[0].name}")

    # Normalize scheme column
    df['Weighting_Scheme'] = df['Weighting_Scheme'].astype(str).str.strip().str.lower()
    df['Score_Threshold'] = df['Score_Threshold'].round(2)
    print("🧩 Available schemes in file:", df['Weighting_Scheme'].unique().tolist())

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=False)
    axes_flat = axes.flatten()

    for ax, title in zip(axes_flat, titles):
        scheme = title_to_scheme.get(title, "").strip().lower()
        print(f"\n🔍 Matching scheme: '{scheme}' for title: '{title}'")

        filtered = df[
            (df['Weighting_Scheme'] == scheme)
            & (df['Num_Questions'] >= min_questions_to_plot)
            & (df['Num_Questions'] <= max_questions_to_plot)
        ]

        print(f"✅ Found {len(filtered)} rows for scheme '{scheme}'")

        if filtered.empty:
            ax.set_title(f"{title} (No Data)", fontsize=16, pad=8)
            ax.axis("off")
            continue

        heat_precision = filtered.pivot_table(
            index='Score_Threshold',
            columns='Num_Questions',
            values='Precision_Rate',
            aggfunc='mean'
        ).sort_index()

        heat_recall = filtered.pivot_table(
            index='Score_Threshold',
            columns='Num_Questions',
            values='Recall',
            aggfunc='mean'
        ).sort_index()

        im = sns.heatmap(
            heat_precision, ax=ax, cmap="viridis",
            vmin=0, vmax=0.2, linewidths=0.5, linecolor='white',
            cbar=True, cbar_kws={"ticks": [0.0, 0.1, 0.2], "shrink": 0.6}
        )
        im.collections[0].colorbar.ax.tick_params(labelsize=12)

        for i, thr in enumerate(heat_recall.index):
            for j, nq in enumerate(heat_recall.columns):
                if heat_recall.loc[thr, nq] < recall_threshold:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))

        ax.plot(
            [-0.5, len(heat_precision.columns) - 0.5],
            [-0.5, len(heat_precision.index) - 0.5],
            color='red', linewidth=2
        )

        xticks = [idx + 0.5 for idx, col in enumerate(heat_precision.columns) if col % 5 == 0]
        xticklabels = [col for col in heat_precision.columns if col % 5 == 0]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=12)

        yticks = [idx + 0.5 for idx, thr in enumerate(heat_precision.index) if thr % 5 == 0]
        yticklabels = [thr for thr in heat_precision.index if thr % 5 == 0]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=12)

        ax.set_title(title, fontsize=16, pad=8)
        ax.set_xlabel('Num Questions', fontsize=14)
        if ax in [axes[0, 0], axes[1, 0]]:
            ax.set_ylabel('Score Threshold', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300)
    print(f"\n✅ Saved heatmap grid to: {save_path}")
    plt.close()




def plot_precision_recall_scatter_grid(
    input_path: Path,
    titles: list[str],
    save_path: Path,
    recall_threshold: float = 0.0,
    min_questions_to_plot: int = 1,
    max_questions_to_plot: int = 70,
):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    df = pd.read_csv(input_path)
    df['Score_Threshold'] = df['Score_Threshold'].round(2)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False, sharey=False)
    axes_flat = axes.flatten()

    line_colors = {
        0.02: 'blue',
        0.14: 'orange',
        0.20: 'red'
    }

    for ax, title in zip(axes_flat, titles):
        scheme_key = title.lower().replace(" ", "").replace("weights", "")
        filtered = df[
            (df['Weighting_Scheme'] == scheme_key)
            & (df['Num_Questions'] >= min_questions_to_plot)
            & (df['Num_Questions'] <= max_questions_to_plot)
        ]

        if filtered.empty:
            ax.set_title(f"{title} (No Data)", fontsize=16, pad=10)
            ax.axis("off")
            continue

        heat_precision = filtered.pivot_table(index='Score_Threshold', columns='Num_Questions', values='Precision_Rate')
        heat_recall = filtered.pivot_table(index='Score_Threshold', columns='Num_Questions', values='Recall')

        rec_vals, prec_vals, colors = [], [], []
        for thr in heat_precision.index:
            for nq in heat_precision.columns:
                p = heat_precision.loc[thr, nq]
                r = heat_recall.loc[thr, nq]
                if not pd.isna(p) and not pd.isna(r):
                    rec_vals.append(r)
                    prec_vals.append(p)
                    if r > 0.1:
                        colors.append('green')
                    elif r > 0.08:
                        colors.append('orange')
                    else:
                        colors.append('red')

        ax.scatter(rec_vals, prec_vals, c=colors, s=20, alpha=0.7)

        for y_val, col in line_colors.items():
            ax.axhline(y=y_val, color=col, linestyle='--', linewidth=2)

        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)

        ax.set_title(title, fontsize=18, pad=10)
        ax.set_xlabel('Recall', fontsize=16)
        ax.set_ylabel('Precision', fontsize=16)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 0.4)
        ax.set_xticks(np.arange(0, 0.51, 0.1))
        ax.set_yticks(np.arange(0, 0.41, 0.1))
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

    legend_elements = [
        Line2D([0], [0], color='blue', linestyle='--', lw=2, label='Random Chance (0.02)'),
        Line2D([0], [0], color='orange', linestyle='--', lw=2, label='Current Best (0.14)'),
        Line2D([0], [0], color='red', linestyle='--', lw=2, label='10× (0.20)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=14, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, format='pdf', dpi=300)
    print(f"✅ Saved scatter grid to {save_path}")
    plt.close()

def plot_precision_recall_f05_bars(results_df: pd.DataFrame, perc_success: float, figures_dir: Path, m: int, suffix: str):
    import matplotlib.pyplot as plt
    import numpy as np

    results_df_sorted = results_df.sort_values('Precision', ascending=False).reset_index(drop=True)
    results_df_sorted['Qnum'] = np.arange(1, len(results_df_sorted) + 1)

    fig, axes = plt.subplots(ncols=3, figsize=(18, 6))
    plt.subplots_adjust(left=0.1, wspace=0.4)

    # Precision
    ax = axes[0]
    ax.barh(results_df_sorted['Qnum'], results_df_sorted['Precision'], color='C0')
    ax.axvline(perc_success, color='gray', linestyle='--')
    ax.set_title('Precision (ranked)', fontsize=14)
    ax.set_xlabel('Precision', fontsize=12)
    ax.set_ylabel('Question #', fontsize=12)
    ax.invert_yaxis()

    # Recall
    ax = axes[1]
    ax.barh(results_df_sorted['Qnum'], results_df_sorted['Recall'], color='C2')
    ax.set_title('Recall (ranked)', fontsize=14)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_yticklabels([])
    ax.invert_yaxis()

    # F0.5
    ax = axes[2]
    ax.barh(results_df_sorted['Qnum'], results_df_sorted['F0.5'], color='C4')
    ax.set_title('F0.5 Score (ranked)', fontsize=14)
    ax.set_xlabel('F0.5', fontsize=12)
    ax.set_yticklabels([])
    ax.invert_yaxis()

    plt.tight_layout()
    fig_path = figures_dir / f"precision_recall_f05_barh_m{m}{suffix}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logging.info(f"Side-by-side figure saved to: {fig_path}")


def plot_precision_colored_bar_chart(results_df: pd.DataFrame, perc_success: float, alpha: float, figures_dir: Path, m: int, suffix: str):
    import matplotlib.pyplot as plt
    import numpy as np

    df_sorted = results_df.sort_values('Precision', ascending=False).reset_index(drop=True)
    precisions = df_sorted['Precision']
    significant = df_sorted['significant']
    above_thr = precisions > perc_success
    x = np.arange(len(df_sorted))
    labels = [f'Q{i+1}' for i in x]

    colors = [
        'darkgreen' if sig and thr else 'orange' if sig else 'lightblue' if thr else 'lightgray'
        for sig, thr in zip(significant, above_thr)
    ]

    plt.figure(figsize=(15, 8))
    plt.bar(x, precisions, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    plt.ylim(0, 0.5)
    plt.xticks(x, labels, rotation=45, ha='right', fontsize=8)
    plt.xlabel('Questions (ranked by Precision)')
    plt.ylabel('Precision')
    plt.title('Precision Scores by Question (Permutation Test Results)', fontsize=14, fontweight='bold')

    plt.axhline(perc_success, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({perc_success})')
    mean_prec = precisions.mean()
    plt.axhline(mean_prec, color='blue', linestyle=':', alpha=0.7, label=f'Mean Precision ({mean_prec:.3f})')

    legend_elems = [
        plt.Rectangle((0,0),1,1, facecolor='darkgreen', edgecolor='black', label='Significant & ≥ threshold'),
        plt.Rectangle((0,0),1,1, facecolor='orange', edgecolor='black', label='Significant only'),
        plt.Rectangle((0,0),1,1, facecolor='lightblue', edgecolor='black', label='≥ threshold only'),
        plt.Rectangle((0,0),1,1, facecolor='lightgray', edgecolor='black', label='Neither'),
    ]
    plt.legend(handles=legend_elems, loc='upper right', fontsize=10)
    plt.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    plt.tight_layout()

    stats = (
        f"Total questions: {len(df_sorted)}\n"
        f"Significant (α={alpha}): {significant.sum()}\n"
        f"Above {perc_success}: {above_thr.sum()}\n"
        f"Both: {(significant & above_thr).sum()}"
    )
    plt.gca().text(0.02, 0.98, stats, transform=plt.gca().transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig_path = figures_dir / f"precision_bar_chart_m{m}{suffix}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logging.info(f"Figure saved to: {fig_path}")

def plot_f1_histograms_from_dict(f1_last: dict, save_path: Path, threshold: float = 0.5):
    """
    Plot histograms of final F₁ scores for different question subsets.

    Parameters
    ----------
    f1_last : dict
        Mapping "Question (Source)" -> (f1_value, source_label).
    save_path : Path
        Where to write the PDF.
    threshold : float
        Vertical cutoff line (e.g. desired minimum F₁).
    """
    groups = {
        'LLM Questions': ['LLM'],
        'LLM + Expert Questions': ['LLM', 'Expert'],
        'LLM + Expert + GPTree Questions': ['LLM', 'Expert', 'GPTree']
    }

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for i, (title, sources) in enumerate(groups.items()):
        # collect F1 values for this subset
        values = [v for (v, src) in f1_last.values() if src in sources]

        axs[i].hist(values, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axs[i].axvline(x=threshold, color='red', linestyle='--', linewidth=2)

        axs[i].set_title(title, fontsize=15)
        axs[i].set_ylabel("Frequency", fontsize=13)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].tick_params(axis='y', labelsize=12)

    axs[2].set_xlabel("Final F\u2081 Score", fontsize=13)
    axs[2].tick_params(axis='x', labelsize=12)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300)
    logger.info(f"Saved F₁ histogram plots to {save_path}")
    plt.close()

import re
import matplotlib.pyplot as plt
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)



import re
import logging
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def compute_f1_local(
    df: pd.DataFrame,
    excluded_rows: set[str],
    founder_columns: list[str]
) -> dict[str, list[float]]:
    """
    For each question (excluding excluded_rows), compute F₁@k for k=1..N founders,
    where recall@k = TP(k) / (# actual positives among first k).
    Returns { question: [F1@1, F1@2, ..., F1@N] }.
    """
    # get true labels, in the SAME column order
    truth_series = (
        df
        .loc[df["Question"] == "Success", founder_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    ).iloc[0]

    f1_results: dict[str, list[float]] = {}

    for question, qdf in df.groupby("Question"):
        if question in excluded_rows:
            continue

        tp = fp = 0
        pos_so_far = 0
        f1_list: list[float] = []

        for k, col in enumerate(founder_columns, start=1):
            # read the one prediction for this question & this founder
            p = qdf[col].iat[0]
            if pd.isna(p):
                # no prediction => carry last F1 (or 0 if first)
                f1_list.append(f1_list[-1] if f1_list else 0.0)
                continue

            p_int = int(float(p))
            a_int = truth_series[col]

            # update counts
            if a_int == 1:
                pos_so_far += 1
            if p_int == 1:
                if a_int == 1:
                    tp += 1
                else:
                    fp += 1

            # compute precision and recall
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / pos_so_far     if pos_so_far > 0 else 0.0

            # compute F₁
            if (prec + rec) > 0:
                f1 = 2 * prec * rec / (prec + rec)
            else:
                f1 = 0.0

            f1_list.append(f1)

        f1_results[question] = f1_list

    return f1_results

def plot_f1_curves_per_set(
    df_list, expert_df_list, gptree_df_list,
    predictions_dir: Path, save_dir: Path
):
    """
    Plot F₁@k curves for each question set (LLM, Expert, GPTree),
    where recall@k = TP(k)/#actualPos(1..k), so the final point matches
    the true global F₁@N exactly.
    """
    total = len(df_list) + len(expert_df_list) + len(gptree_df_list)
    fig, axes = plt.subplots(total, 1, figsize=(14, 5 * total))
    if total == 1:
        axes = [axes]

    excluded = {
        "Founder Index", "Dataset", "Success",
        "SUCCESS_PROPORTION", "Trial Index"
    }
    metric_cols = [
        "Index","Question","Pass Rate","Prec","TP","FP","TN","FN",
        "Rec","F1","F0.5","Prec_Train","Prec_Validation","Prec_Test","Prec_Mean"
    ]
    groups = [
        (df_list,        "LLM"),
        (expert_df_list, "Expert"),
        (gptree_df_list, "GPTree"),
    ]

    first = True
    plot_idx = 0

    for group, suffix in groups:
        for file_name, df in group:
            founder_cols = [c for c in df.columns if c not in metric_cols]
            N = len(founder_cols)

            # compute local‐recall F₁@k
            f1_results = compute_f1_local(df, excluded, founder_cols)

            # one‐time summary (only on first subplot)
            if first:
                idxs = {"50":49, "100":99, "500":499}
                summary = []
                for q, f1s in f1_results.items():
                    row = {"Question": q}
                    for label, idx in idxs.items():
                        row[f"F1@{label}"] = (
                            round(f1s[idx], 3)
                            if len(f1s) > idx else None
                        )
                    row[f"F1@100%({N})"] = round(f1s[-1], 3)
                    summary.append(row)
                summary_df = pd.DataFrame(summary) \
                               .sort_values(by=f"F1@100%({N})", ascending=False)
                print(f"\n=== F1 Summary (full set of {N} trials) ===")
                print(summary_df.to_string(index=False))
                first = False

            ax = axes[plot_idx]
            for q, f1s in f1_results.items():
                ax.plot(range(1, N+1), f1s, label=q, alpha=0.7)

            ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.5)
            m = re.search(r"set_(\d+)", file_name)
            set_num = m.group(1) if m else "?"
            ax.set_title(f"Question Set {set_num} ({suffix})", fontsize=20)
            ax.set_xlabel("Number of Trials", fontsize=18)
            ax.set_ylabel("F₁ Score", fontsize=18)
            ax.set_xlim(1, N)
            ax.set_ylim(0, 1.0)
            ax.spines[["top","right"]].set_visible(False)
            ax.tick_params(labelsize=12)
            ax.legend(fontsize=7, loc="center left",
                      bbox_to_anchor=(1.02, 0.5))

            plot_idx += 1

    plt.tight_layout(rect=[0,0,0.88,1])
    save_path = save_dir / "f1_curves_per_set_localrecall.pdf"
    plt.savefig(save_path, format="pdf", dpi=300)
    logger.info(f"Saved F₁ curves per set (local recall) → {save_path}")
    plt.close()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

import pickle
import logging
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

def plot_combined_f1_curves(
    llm_df_list, expert_df_list, gptree_df_list,
    save_path: Path, threshold: float = 0.5
):
    """
    Plot combined LOCAL‐recall F₁‐score curves for:
      (1) LLM questions,
      (2) LLM + Expert questions,
      (3) LLM + Expert + GPTree questions.
    Curves switch green/red for LLM depending on final F₁ ≥ threshold.
    """
    excluded = {
        'Founder Index', 'Dataset', 'Success',
        'SUCCESS_PROPORTION', 'Trial Index'
    }
    metric_cols = [
        'Index','Question','Pass Rate','Prec','TP','FP','TN','FN',
        'Rec','F1','F0.5','Prec_Train','Prec_Validation','Prec_Test','Prec_Mean'
    ]

    # 1) build or load per‐file local‐F1@k cache
    cache_path = save_path.parent / "f1_cache.pkl"
    if cache_path.exists():
        logger.info(f"Loading cached F₁ data from {cache_path}")
        with open(cache_path, 'rb') as f:
            f1_cache = pickle.load(f)
    else:
        logger.info(f"Computing local‐recall F₁ curves and saving to {cache_path}")

        f1_cache = {}
        for group in (llm_df_list, expert_df_list, gptree_df_list):
            for fname, df in group:
                founders = [c for c in df.columns if c not in metric_cols]
                f1_cache[fname] = compute_f1_local(df, excluded, founders)
        with open(cache_path, 'wb') as f:
            pickle.dump(f1_cache, f)

    # 2) determine x‐axis max (largest number of founders across any set)
    max_trials = 0
    for group in (llm_df_list, expert_df_list, gptree_df_list):
        for _, df in group:
            n = len([c for c in df.columns if c not in metric_cols])
            max_trials = max(max_trials, n)

    # 3) plot
    fig, axs = plt.subplots(3, 1, figsize=(15,15), sharex=True)

    def _plot(group, ax, mode):
        for fname, df in group:
            f1_dict = f1_cache.get(fname, {})
            for q, f1s in f1_dict.items():
                # skip excluded questions
                if q in excluded:
                    continue
                color = (
                    '#DAA520' if mode == 'expert'
                    else '#1f77b4' if mode == 'gptree'
                    else ('green' if f1s[-1] >= threshold else 'red')
                )
                ax.plot(range(1, len(f1s)+1), f1s,
                        color=color, alpha=0.7)

    # LLM only
    _plot(llm_df_list,    axs[0], 'llm')
    # LLM + Expert
    _plot(llm_df_list,    axs[1], 'llm')
    _plot(expert_df_list, axs[1], 'expert')
    # LLM + Expert + GPTree
    _plot(llm_df_list,    axs[2], 'llm')
    _plot(expert_df_list, axs[2], 'expert')
    _plot(gptree_df_list, axs[2], 'gptree')

    titles = [
        "LLM Questions",
        "LLM + Expert Questions",
        "LLM + Expert + GPTree Questions"
    ]
    for i, ax in enumerate(axs):
        ax.axhline(threshold, color='black', linestyle='--', linewidth=2)
        ax.set_title(titles[i], fontsize=18)
        ax.set_xlim(1, max_trials)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("F₁ Score", fontsize=14)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=12)

    axs[2].set_xlabel("Number of Trials", fontsize=14)

    legend_patches = [
        plt.Line2D([],[], color='green', label=f'LLM ≥ {threshold:.2f}'),
        plt.Line2D([],[], color='red',   label=f'LLM < {threshold:.2f}'),
        plt.Line2D([],[], color='#DAA520', label='Expert'),
        plt.Line2D([],[], color='#1f77b4', label='GPTree'),
    ]
    axs[2].legend(handles=legend_patches, fontsize=10, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    logger.info(f"Saved combined F₁ curves to {save_path}")
    plt.close()

def plot_f1_colored_bar_chart(results_df: pd.DataFrame, f1_success_threshold: float, alpha: float, figures_dir: Path, m: int, suffix: str):
    import matplotlib.pyplot as plt
    import numpy as np
    import logging

    # Sort by F1 score
    df_sorted = results_df.sort_values('F1', ascending=False).reset_index(drop=True)
    f1_scores = df_sorted['F1']
    significant = df_sorted['significant']
    above_thr = f1_scores > f1_success_threshold
    x = np.arange(len(df_sorted))
    labels = [f'Q{i+1}' for i in x]

    # Color logic: both sig & above, only sig, only above, neither
    colors = [
        'darkgreen' if sig and thr
        else 'orange' if sig
        else 'lightblue' if thr
        else 'lightgray'
        for sig, thr in zip(significant, above_thr)
    ]

    plt.figure(figsize=(15, 8))
    plt.bar(x, f1_scores, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    plt.ylim(0, 1.0)  # F1 ranges from 0 to 1
    plt.xticks(x, labels, rotation=45, ha='right', fontsize=8)
    plt.xlabel('Questions (ranked by F1 score)')
    plt.ylabel('F1 score')
    plt.title('F1 Scores by Question (Permutation Test Results)', fontsize=14, fontweight='bold')

    # Threshold and mean lines
    plt.axhline(f1_success_threshold, color='red', linestyle='--', alpha=0.7,
                label=f'Threshold ({f1_success_threshold:.2f})')
    mean_f1 = f1_scores.mean()
    plt.axhline(mean_f1, color='blue', linestyle=':', alpha=0.7,
                label=f'Mean F1 ({mean_f1:.3f})')

    # Legend
    legend_elems = [
        plt.Rectangle((0,0),1,1, facecolor='darkgreen', edgecolor='black', label='Significant & ≥ threshold'),
        plt.Rectangle((0,0),1,1, facecolor='orange', edgecolor='black', label='Significant only'),
        plt.Rectangle((0,0),1,1, facecolor='lightblue', edgecolor='black', label='≥ threshold only'),
        plt.Rectangle((0,0),1,1, facecolor='lightgray', edgecolor='black', label='Neither'),
    ]
    plt.legend(handles=legend_elems, loc='upper right', fontsize=10)
    plt.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    plt.tight_layout()

    # Stats box
    stats = (
        f"Total questions: {len(df_sorted)}\n"
        f"Significant (α={alpha}): {significant.sum()}\n"
        f"Above {f1_success_threshold}: {above_thr.sum()}\n"
        f"Both: {(significant & above_thr).sum()}"
    )
    plt.gca().text(0.02, 0.98, stats, transform=plt.gca().transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Save
    fig_path = figures_dir / f"f1_bar_chart_m{m}_{suffix}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logging.info(f"Figure saved to: {fig_path}")

def plot_f1_heatmap(results_file_path: Path, save_path: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    # 1) Load
    results_df = pd.read_csv(results_file_path)

    # 2) Pivot for F1
    f1_data = results_df.pivot(
        index='Score_Threshold',
        columns='Num_Questions',
        values='F1_Score'
    )

    # 3) Build mask: Score_Threshold > Num_Questions
    score_vals = f1_data.index.values[:, None]
    num_q_vals = f1_data.columns.values[None, :]
    mask = score_vals > num_q_vals

    # 4) Plot
    plt.figure(figsize=(24, 12))
    ax = sns.heatmap(
        f1_data,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        vmin=0,
        vmax=1.0,
        linewidths=0.5,
        linecolor='white',
        mask=mask,
        cbar_kws={'shrink': 0.8, 'label': 'F1 Score'},
        annot_kws={'size': 8}
    )
    ax.set_title('Grid Search Heatmap of F₁ Score', fontsize=24, pad=20)
    ax.set_xlabel('Number of Questions Kept', fontsize=20)
    ax.set_ylabel('Score Threshold', fontsize=20)
    ax.tick_params(axis='x', labeltop=True, labelsize=12, rotation=90)
    ax.tick_params(axis='y', labelsize=12)
    ax.xaxis.set_label_position('top')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved F₁ heatmap to {save_path}")

def plot_metric_heatmap(
    results_file: Path,
    save_path: Path,
    metric: str,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str
):
    """
    Generic heatmap for any metric column in the grid-search CSV.
    """
    df = pd.read_csv(results_file)
    if metric not in df.columns:
        logger.error(f"Cannot plot heatmap for '{metric}': column not found in {results_file.name}")
        return

    data = df.pivot(
        index="Score_Threshold",
        columns="Num_Questions",
        values=metric
    )

    # mask out invalid cells where threshold > num_questions
    score_vals = data.index.values[:, None]
    num_q_vals = data.columns.values[None, :]
    mask = score_vals > num_q_vals

    plt.figure(figsize=(24, 12))
    ax = sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        mask=mask,
        cbar_kws={"shrink": 0.8, "label": metric},
        annot_kws={"size": 8}
    )
    ax.set_title(title, fontsize=24, pad=20)
    ax.set_xlabel("Number of Questions Kept", fontsize=20)
    ax.set_ylabel("Score Threshold", fontsize=20)
    ax.tick_params(axis="x", labeltop=True, labelsize=12, rotation=90)
    ax.tick_params(axis="y", labelsize=12)
    ax.xaxis.set_label_position("top")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved {metric} heatmap to {save_path}")


def plot_all_heatmaps(results_file: Path, figures_dir: Path, label: str):
    """
    Produce four heatmaps: Precision, Recall, F1, F0.5.
    """
    # Precision
    plot_metric_heatmap(
        results_file,
        figures_dir / f"precision_heatmap_{label}.pdf",
        metric="Precision_Rate",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        title="Grid Search Heatmap of Precision Rate"
    )

    # Recall
    plot_metric_heatmap(
        results_file,
        figures_dir / f"recall_heatmap_{label}.pdf",
        metric="Recall",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        title="Grid Search Heatmap of Recall"
    )

    # F₁
    plot_metric_heatmap(
        results_file,
        figures_dir / f"f1_heatmap_{label}.pdf",
        metric="F1_Score",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        title="Grid Search Heatmap of F₁ Score"
    )

    # F₀.₅
    plot_metric_heatmap(
        results_file,
        figures_dir / f"f0_5_heatmap_{label}.pdf",
        metric="F0.5_Score",            # always use this column name
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        title="Grid Search Heatmap of F₀.₅ Score"
    )

    plot_metric_heatmap(
        results_file,
        figures_dir / f"accuracy_heatmap_{label}.pdf",
        metric="Accuracy",
        cmap="cubehelix",
        vmin=0.0,
        vmax=1.0,
        title="Grid Search Heatmap of Accuracy"
    )

def plot_metric_heatmaps(fold_grid_dfs, metrics=("F1", "F0_5", "Prec", "Rec"), save_path=None):
    for metric in metrics:
        # Compute vmax across all folds
        vmax = max(df[metric].max() for df in fold_grid_dfs)

        num_folds = len(fold_grid_dfs)
        ncols = 3
        nrows = int(np.ceil(num_folds / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True)

        for i, df in enumerate(fold_grid_dfs):
            ax = axes.flat[i]
            pivot = df.pivot(index="thr", columns="n_q", values=metric)
            sns.heatmap(pivot, ax=ax, cmap="viridis", cbar=True, vmin=0.0, vmax=vmax)
            ax.set_title(f"{metric} – Fold {i+1}")
            ax.set_xlabel("n_q")
            ax.set_ylabel("thr")

            # Get best hyperparameter point (n_q, thr) from attrs
            if "best_combo" in df.attrs:
                n_q, thr = df.attrs["best_combo"]
                ax.plot(
                    [n_q - 0.5], [thr - 0.5],  # center in the cell
                    marker="o", color="red", markersize=6, markeredgecolor="black"
                )

        for j in range(i + 1, len(axes.flat)):
            axes.flat[j].axis("off")

        if save_path:
            metric_file = Path(save_path).with_stem(f"{Path(save_path).stem}_{metric.lower()}")
            plt.savefig(metric_file)  # reduce DPI if visual fidelity not critical
            # plt.close(fig)
        # else:
            # plt.show()

def plot_all_roc_curves(roc_curves, save_path=None):
    plt.figure(figsize=(6, 5))
    for entry in roc_curves:
        if entry["fpr"] is not None:
            plt.plot(entry["fpr"], entry["tpr"], label=f"Fold {entry['fold']} (AUC = {entry['auc']:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for All CV Folds")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    # plt.show()
    # plt.close()

def plot_all_pr_curves(pr_curves, save_path=None):
    plt.figure(figsize=(6, 5))
    for entry in pr_curves:
        if entry["precision"] is not None:
            plt.plot(entry["recall"], entry["precision"], label=f"Fold {entry['fold']} (AP = {entry['avg_prec']:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves for All CV Folds")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    # plt.show()
    # plt.close()

def plot_similarity_heatmap(sim_matrix, results_dir, suffix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cbar=True, xticklabels=False, yticklabels=False)
    plt.title("Question–Question Semantic Similarity")
    plt.savefig(results_dir / f"similarity_heatmap{suffix}.png")

def plot_confusion_matrix(y_true, y_pred, save_path):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Unsuccessful", "Successful"])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap="Blues", values_format='d', ax=ax, colorbar=False)

    # Set font sizes
    ax.set_xlabel("Predicted Label", fontsize=20)
    ax.set_ylabel("True Label", fontsize=20)
    ax.set_xticklabels(["Unsuccessful", "Successful"], fontsize=18)
    ax.set_yticklabels(["Unsuccessful", "Successful"], fontsize=18)
    
    # Remove title
    ax.set_title("")

    # Increase tick label size (for numbers inside the cells)
    for label in ax.texts:
        label.set_fontsize(18)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
