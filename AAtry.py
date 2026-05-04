# # # import json
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import os
# # # import re

# # # input_path = 'time_db_creation.jsonl'  # <-- set your path here
# # # plot_dir = 'plots'
# # # os.makedirs(plot_dir, exist_ok=True)

# # # fontsize = 22
# # # log_scale_x = False
# # # show_std = False

# # # plt.rcParams.update({
# # #     "font.family": "serif",
# # # })


# # # colors = {
# # #     'time_meta_db_creation': 'blue',
# # #     'time_rot_db_creation': 'green',
# # #     'time_aug_db_creation': 'black',

# # # }
# # # line_styles = {
# # #     'time_meta_db_creation': '-',
# # #     'time_rot_db_creation': '-',
# # #     'time_aug_db_creation': '-',
# # # }
# # # markers = {
# # #     'time_meta_db_creation': 'o',
# # #     'time_rot_db_creation': 'o',
# # #     'time_aug_db_creation': 'o',
# # # }
# # # pretty_labels = {
# # #     'time_meta_db_creation': r"${\rm Meta}$",
# # #     'time_rot_db_creation': r"${\rm Rot}$",
# # #     'time_aug_db_creation': r"${\rm Aug}$",
# # # }

# # # # ---------- Load data from JSONL ------------
# # # data = []
# # # with open(input_path, 'r') as f:
# # #     for line in f:
# # #         for match in re.finditer(r'\{.*?\}(?=\{|$)', line):
# # #             data.append(json.loads(match.group()))

# # # # ---------- Group by top_k -------------------
# # # grouped_data = {}
# # # for d in data:
# # #     k = d.get('top_k')
# # #     if k not in grouped_data:
# # #         grouped_data[k] = []
# # #     grouped_data[k].append(d)
# # # top_k_values = sorted(grouped_data.keys())

# # # # ------------ Utility functions --------------
# # # def extract_means_stds(keys):
# # #     means = {k: [] for k in keys}
# # #     stds = {k: [] for k in keys}
# # #     for k in top_k_values:
# # #         group = grouped_data[k]
# # #         for key in keys:
# # #             vals = [d.get(key, np.nan) for d in group if key in d]
# # #             means[key].append(np.mean(vals) if vals else np.nan)
# # #             stds[key].append(np.std(vals) if vals else np.nan)
# # #     return means, stds

# # # def create_and_save_plot(variables, base_title, base_filename, y_label):
# # #     means, stds = extract_means_stds(variables)
# # #     for with_title in [True, False]:
# # #         plt.figure(figsize=(10, 5))
# # #         for var in variables:
# # #             y = means[var]
# # #             y_err = stds[var] if show_std else None
# # #             plt.errorbar(
# # #                 top_k_values, y, yerr=y_err,
# # #                 label=pretty_labels.get(var, var),
# # #                 color=colors[var],
# # #                 linestyle=line_styles[var],
# # #                 marker=markers[var],
# # #                 capsize=5,
# # #                 fmt='-o'
# # #             )
# # #         if log_scale_x:
# # #             plt.xscale('log')
# # #             plt.xlabel('Top-k Index (log scale)', fontsize=fontsize)
# # #         else:
# # #             plt.xlabel('Top-k Index', fontsize=fontsize)
# # #         plt.ylabel(y_label + (' ± Std' if show_std else ''), fontsize=fontsize)
# # #         if with_title:
# # #             plt.title(base_title, fontsize=fontsize+2)
# # #         plt.legend(
# # #             handlelength=4, fontsize=fontsize-2, handletextpad=1
# # #         )
# # #         plt.grid(True)
# # #         plt.tick_params(axis='both', which='major', labelsize=fontsize)
# # #         suffix = "_no_title" if not with_title else ""
# # #         filename = os.path.join(plot_dir, f"{base_filename}{suffix}.pdf")
# # #         plt.savefig(filename, format='pdf', bbox_inches='tight')
# # #         plt.close()


# # # aar_keys = [
# # #     'time_meta_db_creation',
# # #     'time_rot_db_creation',
# # #     'time_aug_db_creation',
# # # ]
# # # create_and_save_plot(
# # #     aar_keys,
# # #     'Time of creation vs Top-k Index',
# # #     'Time_of_creation_vs_Top-k_Index',
# # #     'Time (s)',
# # # )

# # import json
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import os
# # import re

# # display_tendency_curve = False # Set to False to turn off trend curves

# # input_path = 'time_db_creation.jsonl'  # <-- set your path here
# # plot_dir = 'plots'
# # os.makedirs(plot_dir, exist_ok=True)

# # fontsize = 22
# # log_scale_x = False
# # show_std = False

# # plt.rcParams.update({
# #     "font.family": "serif",
# # })

# # colors = {
# #     'time_meta_db_creation': 'blue',
# #     'time_rot_db_creation': 'green',
# #     'time_aug_db_creation': 'black',
# # }
# # line_styles = {
# #     'time_meta_db_creation': '-',
# #     'time_rot_db_creation': '-',
# #     'time_aug_db_creation': '-',
# # }
# # markers = {
# #     'time_meta_db_creation': 'o',
# #     'time_rot_db_creation': 'o',
# #     'time_aug_db_creation': 'o',
# # }
# # pretty_labels = {
# #     'time_meta_db_creation': r"${\rm Meta}$",
# #     'time_rot_db_creation': r"${\rm Rot}$",
# #     'time_aug_db_creation': r"${\rm Aug}$",
# # }

# # # ---------- Load data from JSONL ------------
# # data = []
# # with open(input_path, 'r') as f:
# #     for line in f:
# #         for match in re.finditer(r'\{.*?\}(?=\{|$)', line):
# #             data.append(json.loads(match.group()))

# # # ---------- Group by top_k -------------------
# # grouped_data = {}
# # for d in data:
# #     k = d.get('top_k')
# #     if k not in grouped_data:
# #         grouped_data[k] = []
# #     grouped_data[k].append(d)
# # top_k_values = sorted(grouped_data.keys())

# # # ------------ Utility functions --------------
# # def extract_means_stds(keys):
# #     means = {k: [] for k in keys}
# #     stds = {k: [] for k in keys}
# #     for k in top_k_values:
# #         group = grouped_data[k]
# #         for key in keys:
# #             vals = [d.get(key, np.nan) for d in group if key in d]
# #             means[key].append(np.mean(vals) if vals else np.nan)
# #             stds[key].append(np.std(vals) if vals else np.nan)
# #     return means, stds

# # def plot_tendency_curve(x, y, color, label):
# #     # Remove NaNs for fitting
# #     x_clean = np.array([xi for xi, yi in zip(x, y) if not np.isnan(yi)])
# #     y_clean = np.array([yi for yi in y if not np.isnan(yi)])
# #     if len(x_clean) > 1:
# #         # Degree of polynomial. Change (1 for linear, 2 for quadratic, etc) as you wish
# #         order = 2
# #         coeffs = np.polyfit(x_clean, y_clean, order)
# #         poly = np.poly1d(coeffs)
# #         x_smooth = np.linspace(min(x_clean), max(x_clean), 200)
# #         y_smooth = poly(x_smooth)
# #         plt.plot(
# #             x_smooth, y_smooth, color=color, linestyle='--',
# #             linewidth=2, label=f"{label} trend"
# #         )

# # # --- Compute average ratio (rot/aug) ---
# # ratios = []
# # for k in top_k_values:
# #     group = grouped_data[k]
# #     for d in group:
# #         rot = d.get('time_rot_db_creation')
# #         aug = d.get('time_aug_db_creation')
# #         if rot is not None and aug not in (None, 0):
# #             ratios.append(rot / aug)
# # if ratios:
# #     average_ratio = np.mean(ratios[-10:])
# #     print(f"Average ratio (rot/aug) = {average_ratio:.2f}")
# # else:
# #     print("No valid rot/aug pairs found!")

# # def create_and_save_plot(variables, base_title, base_filename, y_label):
# #     means, stds = extract_means_stds(variables)
# #     for with_title in [True, False]:
# #         plt.figure(figsize=(10, 5))
# #         for var in variables:
# #             y = means[var]
# #             y_err = stds[var] if show_std else None
# #             plt.errorbar(
# #                 top_k_values, y, yerr=y_err,
# #                 label=pretty_labels.get(var, var),
# #                 color=colors[var],
# #                 linestyle=line_styles[var],
# #                 marker=markers[var],
# #                 capsize=5,
# #                 fmt='-o'
# #             )
# #             if display_tendency_curve:
# #                 plot_tendency_curve(top_k_values, y, colors[var], pretty_labels.get(var, var))
# #         if log_scale_x:
# #             plt.xscale('log')
# #             plt.xlabel('Top-k Index (log scale)', fontsize=fontsize)
# #         else:
# #             plt.xlabel('Top-k Index', fontsize=fontsize)
# #         plt.ylabel(y_label + (' ± Std' if show_std else ''), fontsize=fontsize)
# #         if with_title:
# #             plt.title(base_title, fontsize=fontsize+2)
# #         plt.legend(
# #             handlelength=4, fontsize=fontsize-2, handletextpad=1
# #         )
# #         plt.grid(True)
# #         plt.tick_params(axis='both', which='major', labelsize=fontsize)
# #         suffix = "_no_title" if not with_title else ""
# #         filename = os.path.join(plot_dir, f"{base_filename}{suffix}.pdf")
# #         plt.savefig(filename, format='pdf', bbox_inches='tight')
# #         plt.close()

# # aar_keys = [
# #     'time_meta_db_creation',
# #     'time_rot_db_creation',
# #     'time_aug_db_creation',
# # ]
# # create_and_save_plot(
# #     aar_keys,
# #     'Time of creation vs Top-k Index',
# #     'Time_of_creation_vs_Top-k_Index',
# #     'Time (s)',
# # )


# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import re

# # --- Configuration ---
# display_tendency_curve = False
# input_path = 'time_db_creation.jsonl'
# plot_dir = 'plots'
# os.makedirs(plot_dir, exist_ok=True)

# fontsize = 22
# log_scale_x = False
# show_std = False

# NUM_INITIAL_ASYMPTOTE_POINTS = 30
# NUM_PLATEAU_ASYMPTOTE_POINTS = 30


# plt.rcParams.update({"font.family": "serif"})

# colors = {
#     'time_meta_db_creation': 'blue',
#     'time_rot_db_creation': 'green',
#     'time_aug_db_creation': 'black',
# }
# line_styles = {k: '-' for k in colors}
# markers = {k: 'o' for k in colors}
# pretty_labels = {
#     'time_meta_db_creation': r"${\rm Meta}$",
#     'time_rot_db_creation': r"${\rm Rot}$",
#     'time_aug_db_creation': r"${\rm Aug}$",
# }

# # --- Load data ---
# data = []
# with open(input_path, 'r') as f:
#     for line in f:
#         for match in re.finditer(r'\{.*?\}(?=\{|$)', line):
#             data.append(json.loads(match.group()))

# # --- Group by top_k ---
# grouped_data = {}
# for d in data:
#     k = d.get('top_k')
#     if k not in grouped_data:
#         grouped_data[k] = []
#     grouped_data[k].append(d)
# top_k_values = sorted(grouped_data.keys())

# # --- Extract means and stds ---
# def extract_means_stds(keys):
#     means = {k: [] for k in keys}
#     stds = {k: [] for k in keys}
#     for k in top_k_values:
#         group = grouped_data[k]
#         for key in keys:
#             vals = [d.get(key, np.nan) for d in group if key in d]
#             means[key].append(np.mean(vals) if vals else np.nan)
#             stds[key].append(np.std(vals) if vals else np.nan)
#     return means, stds

# # --- Identify asymptotes ---
# def get_asymptotes(y, num_initial=NUM_INITIAL_ASYMPTOTE_POINTS, num_final=NUM_PLATEAU_ASYMPTOTE_POINTS):
#     # Beginning asymptote: average of first `num_initial` points
#     begin_asymptote = np.nanmean(y[:num_initial])
#     # Plateau asymptote: average of last `num_final` points
#     plateau_asymptote = np.nanmean(y[-num_final:])
#     return begin_asymptote, plateau_asymptote

# # --- Compute ratios ---
# def compute_ratios(rot_data, aug_data, num_initial=NUM_INITIAL_ASYMPTOTE_POINTS, num_final=NUM_PLATEAU_ASYMPTOTE_POINTS):
#     # Beginning phase ratios
#     begin_ratios = []
#     for i in range(min(num_initial, len(rot_data))):
#         rot = rot_data[i]
#         aug = aug_data[i]
#         if rot and aug and aug != 0:
#             begin_ratios.append(rot / aug)
#     begin_avg = np.mean(begin_ratios) if begin_ratios else np.nan

#     # Plateau phase ratios
#     plateau_ratios = []
#     for i in range(max(0, len(rot_data) - num_final), len(rot_data)):
#         rot = rot_data[i]
#         aug = aug_data[i]
#         if rot and aug and aug != 0:
#             plateau_ratios.append(rot / aug)
#     plateau_avg = np.mean(plateau_ratios) if plateau_ratios else np.nan

#     return begin_avg, plateau_avg

# # --- Main plot function ---
# def create_and_save_plot(variables, base_title, base_filename, y_label):
#     means, stds = extract_means_stds(variables)
#     plt.figure(figsize=(10, 5))

#     # Plot curves
#     for var in variables:
#         y = means[var]
#         y_err = stds[var] if show_std else None
#         plt.errorbar(
#             top_k_values, y, yerr=y_err,
#             label=pretty_labels.get(var, var),
#             color=colors[var],
#             linestyle=line_styles[var],
#             marker=markers[var],
#             capsize=5,
#             fmt='-o'
#         )

#         # Get asymptotes
#         begin_asymptote, plateau_asymptote = get_asymptotes(y)
#         # Plot asymptotes
#         plt.axhline(begin_asymptote, color=colors[var], linestyle=':', alpha=0.5, label=f'{pretty_labels[var]} begin asymptote')
#         plt.axhline(plateau_asymptote, color=colors[var], linestyle='--', alpha=0.5, label=f'{pretty_labels[var]} plateau asymptote')

#     # Compute and print ratios
#     rot_means = means['time_rot_db_creation']
#     aug_means = means['time_aug_db_creation']
#     begin_avg_ratio, plateau_avg_ratio = compute_ratios(rot_means, aug_means)
#     print(f"Average Rot/Aug ratio (beginning phase): {begin_avg_ratio:.2f}")
#     print(f"Average Rot/Aug ratio (plateau phase): {plateau_avg_ratio:.2f}")

#     # Labels and legend
#     plt.xlabel('Top-k Index', fontsize=fontsize)
#     plt.ylabel(y_label + (' ± Std' if show_std else ''), fontsize=fontsize)
#     plt.title(base_title, fontsize=fontsize+2)
#     plt.legend(handlelength=4, fontsize=fontsize-2, handletextpad=1)
#     plt.grid(True)
#     plt.tick_params(axis='both', which='major', labelsize=fontsize)

#     # Save plot
#     filename = os.path.join(plot_dir, f"{base_filename}.pdf")
#     plt.savefig(filename, format='pdf', bbox_inches='tight')
#     plt.close()

# # --- Run plot ---
# aar_keys = ['time_meta_db_creation', 'time_rot_db_creation', 'time_aug_db_creation']
# create_and_save_plot(
#     aar_keys,
#     'Time of creation vs Top-k Index with Asymptotes',
#     'Time_of_creation_vs_Top-k_Index_with_Asymptotes',
#     'Time (s)',
# )

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# --- Configuration ---
display_tendency_curve = False
input_path = 'time_db_creation.jsonl'
plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)

fontsize = 22
log_scale_x = False
show_std = False
plot_asymptote = True  # Toggle asymptotes

NUM_INITIAL_ASYMPTOTE_POINTS = 30
NUM_PLATEAU_ASYMPTOTE_POINTS = 30

plt.rcParams.update({"font.family": "serif"})

colors = {
    'time_meta_db_creation': 'blue',
    'time_rot_db_creation': 'green',
    'time_aug_db_creation': 'black',
}
line_styles = {k: '-' for k in colors}
markers = {k: 'o' for k in colors}
pretty_labels = {
    'time_meta_db_creation': r"${\rm Meta}$",
    'time_rot_db_creation': r"${\rm Rot}$",
    'time_aug_db_creation': r"${\rm Aug}$",
}

# --- Load data ---
data = []
with open(input_path, 'r') as f:
    for line in f:
        for match in re.finditer(r'\{.*?\}(?=\{|$)', line):
            data.append(json.loads(match.group()))

# --- Group by top_k ---
grouped_data = {}
for d in data:
    k = d.get('top_k')
    if k not in grouped_data:
        grouped_data[k] = []
    grouped_data[k].append(d)
top_k_values = sorted(grouped_data.keys())

# --- Extract means and stds ---
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

# --- Asymptotes with Affine Fit for initial part ---
def get_asymptotes_affine(x, y, num_initial=NUM_INITIAL_ASYMPTOTE_POINTS, num_final=NUM_PLATEAU_ASYMPTOTE_POINTS):
    # Beginning asymptote: affine fit (linear regression) through first num_initial points
    x_init = np.array(x[:num_initial])
    y_init = np.array(y[:num_initial])
    mask_init = ~np.isnan(y_init)
    if mask_init.sum() >= 2:
        coeffs = np.polyfit(x_init[mask_init], y_init[mask_init], 1)  # degree=1 (affine)
        y_affine = np.polyval(coeffs, x_init)
    else:
        coeffs = [0, np.nan]
        y_affine = np.nan * np.ones_like(x_init)
    # Plateau asymptote: horizontal line
    y_plateau = np.nanmean(y[-num_final:])
    x_plateau = np.array(x[-num_final:])

    return x_init, y_affine, x_plateau, y_plateau

# --- Compute ratios ---
def compute_ratios(rot_data, aug_data, num_initial=NUM_INITIAL_ASYMPTOTE_POINTS, num_final=NUM_PLATEAU_ASYMPTOTE_POINTS):
    # Beginning phase ratios
    begin_ratios = []
    for i in range(min(num_initial, len(rot_data))):
        rot = rot_data[i]
        aug = aug_data[i]
        if rot and aug and aug != 0:
            begin_ratios.append(rot / aug)
    begin_avg = np.mean(begin_ratios) if begin_ratios else np.nan

    # Plateau phase ratios
    plateau_ratios = []
    for i in range(max(0, len(rot_data) - num_final), len(rot_data)):
        rot = rot_data[i]
        aug = aug_data[i]
        if rot and aug and aug != 0:
            plateau_ratios.append(rot / aug)
    plateau_avg = np.mean(plateau_ratios) if plateau_ratios else np.nan

    return begin_avg, plateau_avg

# --- Main plot function ---
def create_and_save_plot(variables, base_title, base_filename, y_label):
    means, stds = extract_means_stds(variables)
    plt.figure(figsize=(10, 5))

    plotted_handles = []
    plotted_labels = []

    for var in variables:
        y = means[var]
        y_err = stds[var] if show_std else None
        h = plt.errorbar(
            top_k_values, y, yerr=y_err,
            label=pretty_labels.get(var, var),
            color=colors[var],
            linestyle=line_styles[var],
            marker=markers[var],
            capsize=5,
            fmt='-o'
        )
        plotted_handles.append(h)
        plotted_labels.append(pretty_labels.get(var, var))

        # Get and plot asymptotes
        x_init, y_affine, x_plateau, y_plateau = get_asymptotes_affine(top_k_values, y)
        if plot_asymptote:
            # Initial affine asymptote line (no legend)
            plt.plot(
                x_init, y_affine,
                color=colors[var], linestyle=':', alpha=0.7, linewidth=2, label='_nolegend_'
            )
            # Plateau asymptote horizontal line (no legend)
            plt.plot(
                x_plateau, [y_plateau] * len(x_plateau),
                color=colors[var], linestyle='--', alpha=0.7, linewidth=2, label='_nolegend_'
    )
        # if plot_asymptote:
        #     # Initial affine asymptote line
        #     h1, = plt.plot(x_init, y_affine, color=colors[var], linestyle=':',
        #                    alpha=0.7, linewidth=2, label=f"{pretty_labels[var]} begin asymptote")
        #     # Plateau asymptote horizontal line over the plateau region
        #     h2, = plt.plot(x_plateau, [y_plateau] * len(x_plateau), color=colors[var], linestyle='--',
        #                    alpha=0.7, linewidth=2, label=f"{pretty_labels[var]} plateau asymptote")
        #     plotted_handles += [h1, h2]
        #     plotted_labels += [f"{pretty_labels[var]} begin asymptote", f"{pretty_labels[var]} plateau asymptote"]

    # Compute and print ratios
    rot_means = means['time_rot_db_creation']
    aug_means = means['time_aug_db_creation']
    begin_avg_ratio, plateau_avg_ratio = compute_ratios(rot_means, aug_means)
    print(f"Average Rot/Aug ratio (beginning phase): {begin_avg_ratio:.2f}")
    print(f"Average Rot/Aug ratio (plateau phase): {plateau_avg_ratio:.2f}")

    # Labels and legend
    plt.xlabel('Top-k Index', fontsize=fontsize)
    plt.ylabel(y_label + (' ± Std' if show_std else ''), fontsize=fontsize)
    # plt.title(base_title, fontsize=fontsize+2)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)

    # Remove duplicate labels
    from collections import OrderedDict
    legend_dict = OrderedDict()
    for h, l in zip(plotted_handles, plotted_labels):
        if l not in legend_dict:
            legend_dict[l] = h
    plt.legend(legend_dict.values(), legend_dict.keys(), handlelength=4, fontsize=fontsize-2, handletextpad=1)

    # Save plot
    filename = os.path.join(plot_dir, f"{base_filename}.pdf")
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

# --- Run plot ---
aar_keys = ['time_meta_db_creation', 'time_rot_db_creation', 'time_aug_db_creation']
create_and_save_plot(
    aar_keys,
    'Time of creation vs Top-k Index with Asymptotes',
    'Time_of_creation_vs_Top-k_Index_with_Asymptotes',
    'Time (s)',
)

ratios = []
for k in top_k_values:
    group = grouped_data[k]
    for d in group:
        rot = d.get('time_rot_db_creation')
        aug = d.get('time_aug_db_creation')
        if rot is not None and aug not in (None, 0):
            ratios.append(rot / aug)
if ratios:
    average_ratio = np.mean(ratios)
    print(f"Average ratio (rot/aug) = {average_ratio:.2f}")
else:
    print("No valid rot/aug pairs found!")