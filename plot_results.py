import json
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# ---------- Adjust this ----------
input_path = 'evaluation_summary.jsonl'  # <-- set your path here
plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)

fontsize = 22
log_scale_x = False
show_std = False

plt.rcParams.update({
    "font.family": "serif",
})

# --------- Color, style, marker configs from your code ---------
colors = {
    'avg_AAR_meta_auth': 'blue',
    'avg_AAR_rot_auth': 'green',
    'avg_AAR_aug_auth': 'black',
    'avg_AAP_meta_auth': 'blue',
    'avg_AAP_rot_auth': 'green',
    'avg_AAP_aug_auth': 'black',
    'avg_FAR_meta_auth': 'blue',
    'avg_FAR_rot_auth': 'green',
    'avg_FAR_aug_auth': 'black',
    'avg_FAR_user_meta_auth': 'blue',
    'avg_FAR_user_rot_auth': 'green',
    'avg_FAR_user_aug_auth': 'black',
    'avg_UFAR_user_meta_auth': 'blue',
    'avg_UFAR_user_rot_auth': 'green',
    'avg_UFAR_user_aug_auth': 'black',
    'avg_t_total_meta_auth': 'blue',
    'avg_t_total_rot_auth': 'green',
    'avg_t_total_aug_auth': 'black',
    'avg_t_apply_rotation': 'green',
    'avg_t_apply_augmentation': 'black',


}
line_styles = {
    'avg_AAR_meta_auth': '-',
    'avg_AAR_rot_auth': '-',
    'avg_AAR_aug_auth': '-',
    'avg_AAP_meta_auth': '-',
    'avg_AAP_rot_auth': '-',
    'avg_AAP_aug_auth': '-',
    'avg_FAR_meta_auth': '-',
    'avg_FAR_rot_auth': '-',
    'avg_FAR_aug_auth': '-',
    'avg_FAR_user_meta_auth': '-',
    'avg_FAR_user_rot_auth': '-',
    'avg_FAR_user_aug_auth': '-',
    'avg_UFAR_user_meta_auth': '-',
    'avg_UFAR_user_rot_auth': '-',
    'avg_UFAR_user_aug_auth': '-',
    'avg_t_total_meta_auth': '-',
    'avg_t_total_rot_auth': '-',
    'avg_t_total_aug_auth': '-',
    'avg_t_apply_rotation': '-',
    'avg_t_apply_augmentation': '-',
}
markers = {
    'avg_AAR_meta_auth': 'o',
    'avg_AAR_rot_auth': 'o',
    'avg_AAR_aug_auth': 'o',
    'avg_AAP_meta_auth': 'o',
    'avg_AAP_rot_auth': 'o',
    'avg_AAP_aug_auth': 'o',
    'avg_FAR_meta_auth': 'o',
    'avg_FAR_rot_auth': 'o',
    'avg_FAR_aug_auth': 'o',
    'avg_FAR_user_meta_auth': 'o',
    'avg_FAR_user_rot_auth': 'o',
    'avg_FAR_user_aug_auth': 'o',
    'avg_UFAR_user_meta_auth': 'o',
    'avg_UFAR_user_rot_auth': 'o',
    'avg_UFAR_user_aug_auth': 'o',
    'avg_t_total_meta_auth': 'o',
    'avg_t_total_rot_auth': 'o',
    'avg_t_total_aug_auth': 'o',
    'avg_t_apply_rotation': 'o',
    'avg_t_apply_augmentation': 'o',
}
pretty_labels = {
    'avg_AAR_meta_auth': r"${\rm Meta}$",
    'avg_AAR_rot_auth': r"${\rm Rot}$",
    'avg_AAR_aug_auth': r"${\rm Aug}$",
    'avg_AAP_meta_auth': r"${\rm Meta}$",
    'avg_AAP_rot_auth': r"${\rm Rot}$",
    'avg_AAP_aug_auth': r"${\rm Aug}$",
    'avg_FAR_meta_auth': r"${\rm Meta}$",
    'avg_FAR_rot_auth': r"${\rm Rot}$",
    'avg_FAR_aug_auth': r"${\rm Aug}$",
    'avg_FAR_user_meta_auth': r"${\rm Meta}$",
    'avg_FAR_user_rot_auth': r"${\rm Rot}$",
    'avg_FAR_user_aug_auth': r"${\rm Aug}$",
    'avg_UFAR_user_meta_auth': r"${\rm Meta}$",
    'avg_UFAR_user_rot_auth': r"${\rm Rot}$",
    'avg_UFAR_user_aug_auth': r"${\rm Aug}$",
    'avg_t_total_meta_auth': r"${\rm Meta}$",
    'avg_t_total_rot_auth': r"${\rm Rot}$",
    'avg_t_total_aug_auth': r"${\rm Aug}$",
    'avg_t_apply_rotation': r"${\rm Rot}$",
    'avg_t_apply_augmentation': r"${\rm Aug}$",
}

# ---------- Load data from JSONL ------------
data = []
with open(input_path, 'r') as f:
    for line in f:
        for match in re.finditer(r'\{.*?\}(?=\{|$)', line):
            data.append(json.loads(match.group()))

# ---------- Group by top_k -------------------
grouped_data = {}
for d in data:
    k = d.get('top_k')
    if k not in grouped_data:
        grouped_data[k] = []
    grouped_data[k].append(d)
top_k_values = sorted(grouped_data.keys())

# ------------ Utility functions --------------
def extract_means_stds(keys):
    means = {k: [] for k in keys}
    stds = {k: [] for k in keys}
    for k in top_k_values:
        group = grouped_data[k]
        for key in keys:
            vals = [d.get(key, np.nan) for d in group if key in d]
            means[key].append(np.mean(vals) if vals else np.nan)
            stds[key].append(np.std(vals) if vals else np.nan)
    return means, stds

def create_and_save_plot(variables, base_title, base_filename, y_label):
    means, stds = extract_means_stds(variables)
    for with_title in [True, False]:
        plt.figure(figsize=(10, 5))
        for var in variables:
            y = means[var]
            y_err = stds[var] if show_std else None
            plt.errorbar(
                top_k_values, y, yerr=y_err,
                label=pretty_labels.get(var, var),
                color=colors[var],
                linestyle=line_styles[var],
                marker=markers[var],
                capsize=5,
                fmt='-o'
            )
        if log_scale_x:
            plt.xscale('log')
            plt.xlabel('Top-k Index (log scale)', fontsize=fontsize)
        else:
            plt.xlabel('Top-k Index', fontsize=fontsize)
        plt.ylabel(y_label + (' ± Std' if show_std else ''), fontsize=fontsize)
        if with_title:
            plt.title(base_title, fontsize=fontsize+2)
        plt.legend(
            handlelength=4, fontsize=fontsize-2, handletextpad=1
        )
        plt.grid(True)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        suffix = "_no_title" if not with_title else ""
        filename = os.path.join(plot_dir, f"{base_filename}{suffix}.pdf")
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()


all_t_all_rot = []
all_t_all_aug = []

for k in top_k_values:
    group = grouped_data[k]
    for d in group:
        if 't_all_rot' in d:
            all_t_all_rot.extend(d['t_all_rot'])
        if 't_all_aug' in d:
            all_t_all_aug.extend(d['t_all_aug'])
print("aaa")
print(f"Global: avg_t_apply_rotation = {np.mean(all_t_all_rot):.3f} +/- {np.std(all_t_all_rot):.3f} s, avg_t_apply_augmentation = {np.mean(all_t_all_aug):.3f} +/- {np.std(all_t_all_aug):.3f} s")


    # times = [d.get('avg_t_total_meta_auth', np.nan) for d in group if 'avg_t_total_meta_auth' in d]
    # print(f"Top-k={k}: avg_t_total_meta_auth = {np.mean(times):.2f} s")


# --------- Plot #1: avg_AAR_xxx vs top_k
aar_keys = [
    'avg_AAR_meta_auth',
    'avg_AAR_rot_auth',
    'avg_AAR_aug_auth',
]
create_and_save_plot(
    aar_keys,
    'Mean of avg_AAR_* vs Top-k Index',
    'Mean_of_avg_AAR_vs_Top-k_Index',
    'Mean AAR',
)

# --------- Plot #2: avg_AAP_xxx vs top_k
aap_keys = [
    'avg_AAP_meta_auth',
    'avg_AAP_rot_auth',
    'avg_AAP_aug_auth'
]
create_and_save_plot(
    aap_keys,
    'Mean of avg_AAP_* vs Top-k Index',
    'Mean_of_avg_AAP_vs_Top-k_Index',
    'Mean AAP',
)

# --------- Plot #3: avg_FAR_xxx vs top_k
far_keys = [
    'avg_FAR_meta_auth',
    'avg_FAR_rot_auth',
    'avg_FAR_aug_auth'
]
create_and_save_plot(
    far_keys,
    'Mean of avg_FAR_* vs Top-k Index',
    'Mean_of_avg_FAR_vs_Top-k_Index',
    'Mean FAR',
)

# --------- Plot #4: avg_FAR_user_xxx vs top_k
far_user_keys = [
    'avg_FAR_user_meta_auth',
    'avg_FAR_user_rot_auth',
    'avg_FAR_user_aug_auth'
]
create_and_save_plot(
    far_user_keys,
    'Mean of avg_FAR_user_* vs Top-k Index',
    'Mean_of_avg_FAR_user_vs_Top-k_Index',
    'Mean FAR_user',
)   

# --------- Plot #5: avg_UFAR_user_xxx vs top_k
ufar_user_keys = [
    'avg_UFAR_user_meta_auth',
    'avg_UFAR_user_rot_auth',
    'avg_UFAR_user_aug_auth'
]
create_and_save_plot(
    ufar_user_keys,
    'Mean of avg_UFAR_user_* vs Top-k Index',
    'Mean_of_avg_UFAR_user_vs_Top-k_Index',
    'Mean UFAR_user',
)   

# --------- Plot #6: avg_t_total_xxx vs top_k
t_total_keys = [
    'avg_t_total_meta_auth',
    'avg_t_total_rot_auth',
    'avg_t_total_aug_auth'
]
create_and_save_plot(
    t_total_keys,
    'Mean of avg_t_total_* vs Top-k Index',
    'Mean_of_avg_t_total_vs_Top-k_Index',
    'Mean Time (s)',
) 

# --------- Plot #7: avg_t_apply_rotation/augmentation vs top_k
t_apply_keys = [
    'avg_t_apply_rotation',
    'avg_t_apply_augmentation'
]
create_and_save_plot(
    t_apply_keys,
    'Mean of avg_t_apply_rotation/augmentation vs Top-k Index',
    'Mean_of_avg_t_apply_rotation_augmentation_vs_Top-k_Index',
    'Mean Time (s)',
)



