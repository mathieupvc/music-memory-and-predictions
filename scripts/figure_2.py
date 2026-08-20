from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import mne
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import ttest_1samp, ttest_ind
import scipy.stats
from statsmodels.regression.mixed_linear_model import MixedLM

from utils import select_parcel

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

ROOT = Path(__file__).resolve().parents[1]
print('ROOT:', ROOT)

#################################### Panel B-C, RMS contrasts ####################################

behavior_path = ROOT / "data" / "behavior.csv"
data_path = ROOT / "data" / "rms"
first_parcel_label = "advanced"
analysis_name = 'rms'
regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-lh-middletemporal',
               'ctx-lh-inferiortemporal', 'ctx-lh-temporalpole', 'ctx-lh-entorhinal', 'ctx-lh-parahippocampal',
               'Left-Hippocampus', 'Left-Amygdala',
               'ctx-rh-transversetemporal', 'ctx-rh-superiortemporal', 'ctx-rh-middletemporal',
               'ctx-rh-inferiortemporal', 'ctx-rh-temporalpole', 'ctx-rh-entorhinal', 'ctx-rh-parahippocampal',
               'Right-Hippocampus', 'Right-Amygdala']
remove_outliers=True

behavior_all = pd.read_csv(behavior_path)
path = Path(data_path)
subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))

conditions = ['novelty', 'recognition']

unique_parcellations = []
for i, subject_path in enumerate(tqdm(subject_paths)):
    rms_obj = np.load(subject_path, allow_pickle=True)
    parcellations = rms_obj['ch_parcellations']
    if first_parcel_label==True:
        parcellations = [[parcel[0]] for parcel in parcellations]
    elif first_parcel_label=="advanced":
        go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
        parcellations = select_parcel(parcellations, go_next)
    unique_parcellations += parcellations
unique_parcellations = np.unique(unique_parcellations)
regions_to_remove = ['Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter', 'CSF', 'Unknown']
unique_parcellations = unique_parcellations[np.isin(unique_parcellations, regions_to_remove, invert=True)]
unique_parcellations = sorted(unique_parcellations, key=lambda x: x[::-1], reverse=True)

if regions is not None:
    unique_parcellations = [region for region in unique_parcellations if region in regions]
rms_contrasts = [{region: [] for region in unique_parcellations} for c in range(len(conditions))]

for i, subject_path in enumerate(tqdm(subject_paths)):
    subject_name = Path(subject_path).name[:6]
    behavior = behavior_all[behavior_all['subject'] == subject_name]
    correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
    incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
    correct_recognition = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
    incorrect_recognition = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

    rms_obj = np.load(subject_path, allow_pickle=True)
    rms = rms_obj[analysis_name]
    parcellations = rms_obj['ch_parcellations']
    if first_parcel_label==True:
        parcellations = [[parcel[0]] for parcel in parcellations]
    elif first_parcel_label=="advanced":
        parcellations = select_parcel(parcellations, go_next)

    for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_recognition, incorrect_recognition]]):
        correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
        incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
        if (len(correct) > 0) & (len(incorrect) > 0):
            for c in range(rms.shape[1]):
                # Accumulate rms by region
                for region in unique_parcellations:
                    if any(ch_parcel in region for ch_parcel in parcellations[c]):
                        rms_contrasts[cond][region].append(np.mean(rms[correct, c], axis=0) - np.mean(rms[incorrect, c], axis=0))

# Calculate and save average rms for each condition and region
region_rms_per_condition = []
error_bars_per_condition = []
regions_per_condition = []
pvalue_per_condition = []
t_per_condition = []
for cond in range(len(conditions)):
    condition_rms = []
    condition_error_bars = []
    condition_regions = []
    condition_pvalue = []
    condition_t = []
    for region in unique_parcellations:
        region_rms = np.array(rms_contrasts[cond][region])
        condition_regions.append(region)
        mean_rms = np.mean(region_rms, axis=0)
        condition_rms.append(mean_rms)
        error_rms = np.std(region_rms, axis=0) / np.sqrt(region_rms.shape[0])
        condition_error_bars.append(error_rms)
        ttest = ttest_1samp(region_rms, 0)
        condition_pvalue.append(ttest.pvalue)
        condition_t.append(ttest.statistic)
    region_rms_per_condition.append(condition_rms)
    error_bars_per_condition.append(condition_error_bars)
    regions_per_condition.append(condition_regions)
    pvalue_per_condition.append(condition_pvalue)
    t_per_condition.append(condition_t)

# FDR correction
for c in range(len(pvalue_per_condition)):
    _, pval = mne.stats.fdr_correction(pvalue_per_condition[c], alpha=0.05, method='indep')
    pvalue_per_condition[c] = pval

new_names = {'ctx-lh-transversetemporal': 'L-TT', 'ctx-lh-superiortemporal': 'L-STG',
             'ctx-rh-transversetemporal': 'R-TT', 'ctx-rh-superiortemporal': 'R-STG',
             'ctx-lh-middletemporal': 'L-MTG', 'ctx-rh-middletemporal': 'R-MTG',
             'Left-Hippocampus': 'L-HPC', 'Left-Amygdala': 'L-AMY',
             'ctx-lh-parahippocampal': 'L-PHC', 'ctx-lh-temporalpole': 'L-TP',
             'ctx-lh-inferiortemporal': 'L-ITG', 'ctx-lh-entorhinal': 'L-EC',
             'Right-Hippocampus': 'R-HPC', 'Right-Amygdala': 'R-AMY',
             'ctx-rh-parahippocampal': 'R-PHC', 'ctx-rh-temporalpole': 'R-TP', 'ctx-rh-inferiortemporal': 'R-ITG',
             'ctx-rh-entorhinal': 'R-EC'}

cmap = list(plt.get_cmap('Set3').colors)
del cmap[8]
region_colors = {region: cmap[i % int(len(regions) / 2)] for i, region in enumerate(regions)}

fig, axs = plt.subplots(1, len(conditions), sharex=True, figsize=(10, 5))
# Plot each condition's data in a separate subplot
for i, (condition, regions, region_rms, error_bars, pvalues, tvalues) in enumerate(zip(conditions, regions_per_condition, region_rms_per_condition, error_bars_per_condition, pvalue_per_condition, t_per_condition)):
    sorted_id = np.argsort(region_rms)
    region_rms = np.array(region_rms)[sorted_id]
    regions = np.array(regions)[sorted_id]
    new_names_regions = [new_names[reg] for reg in regions]
    pvalues = np.array(pvalues)[sorted_id]
    colors = [region_colors[regions[r]] if p < 0.05 else 'w' for r, p in enumerate(pvalues)]
    axs[i].barh(new_names_regions, region_rms, xerr=None, capsize=5, color=colors, edgecolor='k', lw=1)
    axs[i].axvline(0, color='k', linestyle='--', linewidth=1.5)
    axs[i].tick_params(axis='y', labelsize=13)
    axs[i].tick_params(axis='x', labelsize=17)

    # Add individual sample points as scatter dots
    for j, region in enumerate(regions):
        samples = np.array(rms_contrasts[i][region])
        if remove_outliers:
            samples = samples[(np.abs(samples - samples.mean())) <= 2 * samples.std()]
        y_values = np.full(samples.shape, j) + np.random.uniform(-0.1, 0.1, samples.shape)  # Jitter for clarity
        axs[i].scatter(samples, y_values, color='black', alpha=0.7, s=2, label='Samples')

    axs[i].set_xlabel('Percent change of activity', fontsize=17)

plt.tight_layout()
fig.savefig(ROOT / "figures" / "figure_2" / "B_C_rms_contrasts.svg")


#################################### Panel B-C, behavior ####################################

# behavior = seeg_behavior_matrix(data_path, stim_path)
behavior_all_trials_path = ROOT / "data" / "behavior_all_trials.csv"
behavior = pd.read_csv(behavior_all_trials_path)
behavior = behavior[behavior['resp'] != 'different_responses']
behavior = behavior.dropna(subset='resp')
behavior['resp'] = behavior['resp'].astype(int)
behavior['resp'] = behavior['resp'] - 1  # set resp to 0 and 1
behavior["rep"] = behavior[["rep1", "rep2"]].idxmax(axis=1).str.extract("(\d+)").astype(int)

data = behavior.groupby(['subject', 'rep'], as_index=False).mean()
fig = plt.figure()
ax = sns.barplot(data=data, y='rep', x='resp', palette=['#ff877f', '#9c94f9'], orient='h')
ax = sns.swarmplot(data=data, y='rep', x='resp', color='black', size=4, orient='h')
ax.set_xlim(0, 1)
ax.set_xlabel('Proportion of "old" responses')
ax.set_ylabel('Presentation')

print('T-tests')
test = ttest_1samp(data[data['rep']==1]['resp'].to_numpy(), 0.5, alternative='less')
print('Novelty', test)
test = ttest_1samp(data[data['rep'] == 2]['resp'].to_numpy(), 0.5, alternative='greater')
print('Recognition', test)
test = ttest_1samp(data[data['rep'] == 2]['resp'].to_numpy() - data[data['rep'] == 1]['resp'].to_numpy(), 0, alternative='two-sided')
print('Recognition - Novelty', test)

plt.tight_layout()
fig.savefig(ROOT / "figures" / "figure_2" / "B_C_behavior.svg")


#################################### Panel D ####################################

behavior_path = ROOT / "data" / "behavior.csv"
data_path = ROOT / "data" / "rms"
first_parcel_label = "advanced"
analysis_name = 'rms'
regions_list = [['Left-Hippocampus', 'Right-Hippocampus'], ['Left-Amygdala', 'Right-Amygdala']]
region_titles = ["Hippocampus", "Amygdala"]

"""electrode-level test (FDR corrected) used to plot rms, gamma power or other scalar saved with the same format."""
behavior_all = pd.read_csv(behavior_path)
path = Path(data_path)
subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))
subject_names = []

conditions = ['novelty', 'retrieval']

fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

for r, regions in enumerate(regions_list):  # run the analysis for the hippocampus and the amygdala independently
    rms_contrasts = [[] for c in range(len(conditions))]
    pvalues = [[] for c in range(len(conditions))]
    subject_id = []  # used for random effects

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj[analysis_name]
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label=="advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            observed_diff = np.mean(rms[:, c][correct]) - np.mean(rms[:, c][incorrect])
                            rms_contrasts[cond].append(observed_diff)
                            pvalues[cond].append(ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=False, permutations=1000, random_state=None, alternative='greater').pvalue)  # testing of individual electrodes
                            if cond == 0:
                                subject_id.append(i)

    # Define subject colors
    subject_colors = plt.get_cmap('Set3').colors
    subject_colors = ['lightgray' for i in range(len(subject_colors)) if i != 1]  # all the same color

    # Create DataFrame
    data_dict = {
        'Novelty_retrieval': rms_contrasts[0],
        'Retrieval': rms_contrasts[1],
        'SubjectID': subject_id
    }
    model_data = pd.DataFrame(data_dict)

    # Compute mean and std for outlier detection
    mean_novelty = model_data['Novelty_retrieval'].mean()
    std_novelty = model_data['Novelty_retrieval'].std()
    mean_retrieval = model_data['Retrieval'].mean()
    std_retrieval = model_data['Retrieval'].std()

    # Mask to identify outliers
    outlier_mask = (
            (np.abs(model_data['Novelty_retrieval'] - mean_novelty) > 3 * std_novelty) |
            (np.abs(model_data['Retrieval'] - mean_retrieval) > 3 * std_retrieval)
    )

    # Data without outliers (for visualization & plotting regression)
    plot_data = model_data[~outlier_mask]

    # Assign colors after reordering subjects
    colors = [subject_colors[s] for s in plot_data['SubjectID']]

    # Plot scatter (without outliers)
    axs[r].scatter(plot_data['Novelty_retrieval'], plot_data['Retrieval'], color=colors, s=12, alpha=0.6)
    axs[r].set_xlabel('Novelty (% change of activity)', fontsize=17)
    if r == 0:
        axs[r].set_ylabel('Recognition (% change of activity)', fontsize=17)
    axs[r].tick_params(axis='y', labelsize=17)
    axs[r].tick_params(axis='x', labelsize=17)
    axs[r].axvline(0, ls='--', color='k')
    axs[r].axhline(0, ls='--', color='k')
    axs[r].set_xlim(-15, 60)  # manual limits that keep all data points
    axs[r].set_ylim(-15, 35)
    axs[r].set_title(region_titles[r], fontsize=17)

    # Fit mixed-effects model (using full dataset for stats)
    model_full = MixedLM.from_formula('Retrieval ~ Novelty_retrieval', groups='SubjectID', data=model_data)
    result_full = model_full.fit()
    p_value = result_full.pvalues['Novelty_retrieval']

    # Fit a new model for visualization (without outliers)
    model_plot = MixedLM.from_formula('Retrieval ~ Novelty_retrieval', groups='SubjectID', data=plot_data)
    result_plot = model_plot.fit()

    # Plot regression line (without outliers)
    x_vals = np.linspace(min(plot_data['Novelty_retrieval']), max(plot_data['Novelty_retrieval']), 100)
    y_vals = result_plot.params['Intercept'] + result_plot.params['Novelty_retrieval'] * x_vals
    axs[r].plot(x_vals, y_vals, color='k', linewidth=5, zorder=10)

    # Plot individual subject fits (without outliers)
    unique_subjects = np.unique(plot_data['SubjectID'])
    for subj in unique_subjects:
        mask = plot_data['SubjectID'] == subj
        x_subj = plot_data.loc[mask, 'Novelty_retrieval']
        y_subj = plot_data.loc[mask, 'Retrieval']
        if len(x_subj) > 1:  # Ensure enough points to fit
            slope, intercept = np.polyfit(x_subj, y_subj, 1)
            x_range = np.linspace(min(x_subj), max(x_subj), 100)
            axs[r].plot(x_range, slope * x_range + intercept, color=subject_colors[subj], lw=3, alpha=1)

plt.tight_layout()

fig.savefig(ROOT / "figures" / "figure_2" / "D.svg")


#################################### Panel E ####################################

behavior_path = ROOT / "data" / "behavior.csv"
data_path = ROOT / "data" / "rms_ma"
first_parcel_label = 'advanced'
analysis_name = 'ma_rms'
regions = [['Left-Hippocampus', 'Right-Hippocampus'], ['Left-Amygdala', 'Right-Amygdala']]
region_titles = ["Hippocampus", "Amygdala"]

behavior_all = pd.read_csv(behavior_path)
path = Path(data_path)
subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))
subject_names = []

conditions = ['novelty', 'retrieval']

rms_contrasts = [[[] for r in range(len(regions))] for c in range(len(conditions))]

for i, subject_path in enumerate(tqdm(subject_paths)):
    subject_name = Path(subject_path).name[:6]
    subject_names.append(subject_name)
    behavior = behavior_all[behavior_all['subject'] == subject_name]
    correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
    incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
    correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
    incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

    rms_obj = np.load(subject_path, allow_pickle=True)
    rms = rms_obj[analysis_name]
    parcellations = rms_obj['ch_parcellations']
    if first_parcel_label == True:
        parcellations = [[parcel[0]] for parcel in parcellations]
    elif first_parcel_label == "advanced":
        go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
        parcellations = select_parcel(parcellations, go_next)

    for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
        correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
        incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
        if (len(correct) > 0) & (len(incorrect) > 0):
            for c in range(rms.shape[1]):
                for r, region in enumerate(regions):
                    if any(ch_parcel in region for ch_parcel in parcellations[c]):
                        rms_contrasts[cond][r].append(np.mean(rms[correct, c, :], axis=0) - np.mean(rms[incorrect, c, :], axis=0))

fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
region_colors = ['#FF1100', '#3B2AF5']

# compute cluster stats
mask_list = [] # Store masks for test between conditions
mask_list_zero = [[] for _ in conditions]  # Store masks for condition vs. 0
for r in range(len(rms_contrasts[0])):
    diff = np.array(rms_contrasts[0][r]) - np.array(rms_contrasts[1][r])
    pval = 0.05  # arbitrary
    df = len(diff) - 1  # degrees of freedom denominator
    thresh = scipy.stats.t.ppf(1 - pval / 2, df=df)  # t distribution two-tailed
    # thresh = scipy.stats.t.ppf(1 - pval, df=df)  # t distribution one-tailed
    T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(diff,
                                                                     threshold=thresh,
                                                                     n_permutations=10000, tail=0,
                                                                     adjacency=None, n_jobs=-1)
    # Create a mask for significant clusters
    significant_mask = np.zeros_like(T_obs, dtype=bool)
    for j, cl in enumerate(clusters):
        if cluster_pv[j] <= 0.05:
            significant_mask[cl] = True
    mask_list.append(significant_mask)

# tests against 0
for r in range(len(rms_contrasts[0])):
    for cond in range(len(conditions)):
        diff = np.array(rms_contrasts[cond][r])
        pval = 0.05  # arbitrary
        df = len(diff) - 1  # degrees of freedom denominator
        thresh = scipy.stats.t.ppf(1 - pval / 2, df=df)  # t distribution two-tailed
        # thresh = scipy.stats.t.ppf(1 - pval, df=df)  # t distribution one-tailed
        T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(diff,
                                                                         threshold=thresh,
                                                                         n_permutations=10000, tail=0,
                                                                         adjacency=None, n_jobs=-1)
        # Create a mask for significant clusters
        significant_mask = np.zeros_like(T_obs, dtype=bool)
        for j, cl in enumerate(clusters):
            if cluster_pv[j] <= 0.05:
                significant_mask[cl] = True
        mask_list_zero[cond].append(significant_mask)

# plot
current = 0
for i, (condition, rms) in enumerate(zip(conditions, rms_contrasts)):
    for r, region_rms in enumerate(rms):
        avg_contrast = np.mean(region_rms, axis=0)
        sem_contrast = np.std(region_rms, axis=0) / np.sqrt(len(region_rms))
        times = np.linspace(0, 8, avg_contrast.shape[0])
        axs[r].plot(times, avg_contrast, color=region_colors[i], label=condition + str(regions[r]), lw=3)
        axs[r].fill_between(times, avg_contrast - sem_contrast, avg_contrast + sem_contrast, color=region_colors[i], alpha=0.5, lw=0)
        current += 1
for r in range(len(rms_contrasts[0])):
    axs[r].set_xlim(0, 8)
    axs[r].axhline(0, ls='--', c='k', zorder=0)
    axs[r].set_xlabel(f'Time (s)', fontsize=17)
    axs[r].tick_params(axis='y', labelsize=17)
    axs[r].tick_params(axis='x', labelsize=17)
    axs[r].set_title(region_titles[r], fontsize=17)
axs[0].set_ylabel(f'Percent change of activity', fontsize=17)
for r in range(len(rms_contrasts[0])):
    mask = mask_list[r]
    mask = np.ma.masked_where(~mask, -10 * np.ones_like(times))  # we mask what is not significant
    axs[r].plot(times, mask, 'k', linewidth=3)
    for cond in range(len(conditions)):
        mask = mask_list_zero[cond][r]
        mask = np.ma.masked_where(~mask, -(7+cond*1.5) * np.ones_like(times))  # we mask what is not significant
        axs[r].plot(times, mask, color=region_colors[cond], linewidth=3)

plt.tight_layout()

fig.savefig(ROOT / "figures" / "figure_2" / "E.svg")