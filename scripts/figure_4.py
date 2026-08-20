from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, pearsonr
import scipy.stats
from scipy.optimize import curve_fit
import statsmodels.api as sm
import mne

from utils import select_parcel

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

ROOT = Path(__file__).resolve().parents[1]
print('ROOT:', ROOT)


#################################### Panel B ####################################

behavior_path = ROOT / "data" / "behavior.csv"
condition = 'surprise_mean_polyrnn'

behavior = pd.read_csv(behavior_path)
behavior = behavior.dropna(subset=['correct'])

subjects = behavior['subject'].unique()
subject_colors = ['lightgray' for s in range(len(subjects))]

fig = plt.figure(figsize=(5, 5))

jitters = np.linspace(-0.05, 0.05, len(subjects))

# Plot scatter points and quadratic fit for each subject
for s, subject in enumerate(subjects):
    sub_df = behavior[behavior['subject'] == subject]
    x = sub_df[condition]
    y = sub_df['correct']

    # Scatter plot
    plt.scatter(x, y + jitters[s], color=subject_colors[s], s=0.5, label=subject)

    # Quadratic fit
    coeffs = np.polyfit(x, y, 2)  # Fit y = ax^2 + bx + c
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = np.polyval(coeffs, x_fit)
    plt.plot(x_fit, y_fit, color=subject_colors[s], alpha=1, lw=1.5)

# Average quadratic fit (on all data)
x_all = behavior[condition]
y_all = behavior['correct']
coeffs_avg = np.polyfit(x_all, y_all, 2)
x_fit_avg = np.linspace(x_all.min(), x_all.max(), 200)
y_fit_avg = np.polyval(coeffs_avg, x_fit_avg)
plt.plot(x_fit_avg, y_fit_avg, color='black', linewidth=6, label='Average Fit')

# Labeling and legend
plt.xlabel('Surprise', fontsize=17)
plt.ylabel('Correct', fontsize=17)
plt.gca().tick_params(axis='y', labelsize=17)
plt.gca().tick_params(axis='x', labelsize=17)
plt.tight_layout()

fig.savefig(ROOT / "figures" / "figure_4" / "B.svg")


#################################### Panel C ####################################

data_path = ROOT / "data" / "rms"
first_parcel_label = "advanced"
regions = ['ctx-lh-middletemporal', 'ctx-rh-middletemporal', 'Left-Hippocampus',
           'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal',
           'ctx-lh-entorhinal', 'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal',
           'ctx-rh-inferiortemporal']  # regions of temporal lobe that show a power contrast
similarity_path = ROOT / "data" / "similarity_separability" / "similarity.pkl"
distance_path = ROOT / "data" / "similarity_separability" / "separability.pkl"


behavior_all = pd.read_csv(behavior_path)
path = Path(data_path)
subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
subject_names = []

# Select electrodes

significant_electrodes = {}  # dict of signif electrodes by subject

for i, subject_path in enumerate(tqdm(subject_paths)):
    subject_name = Path(subject_path).name[:6]
    subject_names.append(subject_name)
    behavior = behavior_all[behavior_all['subject'] == subject_name]
    correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
    incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
    correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
    incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

    rms_obj = np.load(subject_path, allow_pickle=True)
    rms = rms_obj['rms']
    parcellations = rms_obj['ch_parcellations']
    if first_parcel_label == True:
        parcellations = [[parcel[0]] for parcel in parcellations]
    elif first_parcel_label == "advanced":
        go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
        parcellations = select_parcel(parcellations, go_next)

    significant_electrodes[subject_name] = []
    for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
        correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
        incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
        if (len(correct) > 0) & (len(incorrect) > 0):
            for c in range(rms.shape[1]):
                for region in regions:
                    if any(ch_parcel in region for ch_parcel in parcellations[c]):
                        significant_electrodes[subject_name].append(rms_obj['ch_names'][c])
    significant_electrodes[subject_name] = np.unique(significant_electrodes[subject_name]).tolist()  # keep each electrode only once


# Neural pattern analysis

# define the function that will be used for regressions
def run_regressions(representation_array):
    """representation_array contains intrastim correlation or interstim distance"""
    param_correct_list = []
    param_feature_list = []
    channel_positions = []
    subject_id_list = []
    for channel in range(representation_array.shape[1]):
        if ch_names[channel] in significant_electrodes[subject]:
            Y = np.array(correct_list)
            X = np.zeros((Y.shape[0], 1))
            X[:, 0] = representation_array[:, channel]
            X = (X - X.mean()) / X.std()
            X = sm.add_constant(X)

            ols = sm.Logit(Y, X).fit(disp=False)
            param_correct_list.append(ols.params[1])

            Y = X[:, 1]
            X = np.zeros((Y.shape[0], 1))
            X[:, 0] = feature_list
            X = (X - X.mean(axis=0)) / X.std(axis=0)
            X = sm.add_constant(X)
            ols = sm.OLS(Y, X).fit()
            param_feature_list.append(ols.params[1])

            channel_positions.append(subject_info['chs'][channel]['loc'][:3])
            subject_id_list.append(s)
    return param_correct_list, param_feature_list, channel_positions, subject_id_list


# Similarity

stim_features = pd.read_csv(behavior_path)
stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]

data = np.load(similarity_path, allow_pickle=True)

channel_positions_list = []
subject_id = []
regression_param_correct = []
regression_param_feature = []
regression_param_correct_subject_mean = []
regression_param_feature_subject_mean = []
p_ttest_array = []  # p value of the contrast correct vs incorrect

for s, subject in tqdm(enumerate(data["subjects"])):
    subject_stim_features = stim_features[stim_features['subject'] == subject]
    subject_info = data['info'][s]
    ch_names = subject_info['ch_names']
    indices_to_keep = []
    correct_list = []
    feature_list = []
    correlations_array = []
    stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
    similarity_matrix = data['similarity_matrices'][s]  # keep all channels

    first_trials_to_remove = 50
    for i, stim in enumerate(data['stim_names'][s]):
        if i > first_trials_to_remove:  # remove first trials
            if stim in subject_stim_features['stim_name'].tolist():
                if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                    if (stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb) & (stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb):  # check that both reps are in the data
                        indices_to_keep.append(i)
                        correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                        feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                        if 'rep1' in stim:
                            rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2'))
                        elif 'rep2' in stim:
                            rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                        else:
                            raise ValueError
                        correlations_array.append(similarity_matrix[i, rep_index, :])  # correlation of 2 reps

    correlations_array = np.array(correlations_array)

    p_ttest = np.zeros(correlations_array.shape[1])

    param_correct_list, param_feature_list, channel_positions, subject_id_list = run_regressions(correlations_array)
    regression_param_correct += param_correct_list
    regression_param_feature += param_feature_list
    channel_positions_list += channel_positions
    subject_id += subject_id_list

regression_param_correct = np.array(regression_param_correct)
regression_param_feature = np.array(regression_param_feature)
regression_param_correct_subject_mean = np.array(regression_param_correct_subject_mean)
regression_param_feature_subject_mean = np.array(regression_param_feature_subject_mean)

list_of_pvalues = []

alternative = ['greater', 'less']
for i, regression_param_array in enumerate([regression_param_correct, regression_param_feature]):
    # classical t-test (does not account for channel covariance within subjects)
    ttest = scipy.stats.ttest_1samp(regression_param_array, 0, alternative=alternative[i])
    print('ttest:', ttest)
    list_of_pvalues.append(ttest.pvalue)

intrastim_params = regression_param_correct.copy()
intrastim_feature_params = regression_param_feature.copy()


# Separability

stim_features = pd.read_csv(behavior_path)
stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]

data = np.load(distance_path, allow_pickle=True)

channel_positions_list = []
subject_id = []
regression_param_correct = []
regression_param_feature = []
regression_param_correct_subject_mean = []
regression_param_feature_subject_mean = []
for s, subject in tqdm(enumerate(data["subjects"])):
    subject_stim_features = stim_features[stim_features['subject'] == subject]
    subject_info = data['info'][s]
    ch_names = subject_info['ch_names']
    indices_to_keep = []
    correct_list = []
    feature_list = []
    stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
    similarity_matrix = data['distance_matrices'][s]  # keep all channels
    paired_stim = []
    for i, stim in enumerate(data['stim_names'][s]):
        if i > first_trials_to_remove:  # remove first trials
            if ('rep1' in stim) | ('rep2' in stim):
                if stim in subject_stim_features['stim_name'].tolist():
                    if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                        indices_to_keep.append(i)
                        correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                        feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                        if 'rep2' in stim:
                            if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1')))
                            else:
                                paired_stim.append('no_pair')
                        if 'rep1' in stim:
                            if stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb:
                                paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2')))
                            else:
                                paired_stim.append('no_pair')
    condition_matrix = similarity_matrix[indices_to_keep, :, :]  # select condition
    correlations_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
    for i in range(condition_matrix.shape[0]):
        current_corr = condition_matrix[i, :, :]
        mask = np.ones(condition_matrix.shape[1], dtype=bool)
        mask[indices_to_keep[i]] = False  # remove current stim
        mask[indices_to_keep[i]:] = False  # remove following trials
        correlations_array[i, :] = np.mean(current_corr[mask, :], axis=0)

    param_correct_list, param_feature_list, channel_positions, subject_id_list = run_regressions(correlations_array)
    regression_param_correct += param_correct_list
    regression_param_feature += param_feature_list
    channel_positions_list += channel_positions
    subject_id += subject_id_list

    # store subject average
    regression_param_correct_subject_mean.append(np.mean(param_correct_list))
    regression_param_feature_subject_mean.append(np.mean(param_feature_list))

regression_param_correct = np.array(regression_param_correct)
regression_param_feature = np.array(regression_param_feature)


alternative = ['greater', 'greater']
for i, regression_param_array in enumerate([regression_param_correct, regression_param_feature]):
    # classical t-test (does not account for channel covariance within subjects)
    ttest = scipy.stats.ttest_1samp(regression_param_array, 0, alternative=alternative[i])
    print('ttest:', ttest)
    list_of_pvalues.append(ttest.pvalue)

interstim_params = regression_param_correct

# boxplots
df = pd.DataFrame({'Similarity': intrastim_feature_params, 'Separability': regression_param_feature})
fig = plt.figure(figsize=(5, 5))
sns.violinplot(data=df, showmeans=True, showfliers=False, palette=['#FF928A', '#978EF5'], orient='v', inner=None)
sns.swarmplot(data=df, color=".25", size=2.5, orient='v')
plt.gca().axhline(0, ls='--', color='k')
plt.ylabel('\u03B2 (condition ~ surprise)', fontsize=17)
plt.tick_params(axis='y', labelsize=17)
plt.tick_params(axis='x', labelsize=17)

plt.scatter(0, np.mean(intrastim_feature_params), color='w', edgecolor='k', lw=2, s=75, zorder=10)
plt.scatter(1, np.mean(regression_param_feature), color='w', edgecolor='k', lw=2, s=75, zorder=10)

for i, p in enumerate([list_of_pvalues[1], list_of_pvalues[3]]):
    if p < 0.05:
        # Get the position to place the star
        x = i  # The x position corresponds to the box index
        y = df.iloc[:, i].max() + 0.15  # Slightly above the max value of the box
        plt.text(x, y, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')
plt.tight_layout()
fig.savefig(ROOT / "figures" / "figure_4" / "C.svg")



#################################### Panel D ####################################

remove_outliers = True
conditions = ['correct ~ similarity', 'correct ~ separability', 'similarity ~ surprise', 'separability ~ surprise']

# Electrode selection

path = Path(data_path)
subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
subject_names = []

unique_parcellations = []
for i, subject_path in enumerate(tqdm(subject_paths)):
    rms_obj = np.load(subject_path, allow_pickle=True)
    parcellations = rms_obj['ch_parcellations']
    if first_parcel_label == True:
        parcellations = [[parcel[0]] for parcel in parcellations]
    elif first_parcel_label == "advanced":
        go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
        parcellations = select_parcel(parcellations, go_next)
    unique_parcellations += parcellations
unique_parcellations = np.unique(unique_parcellations)
regions_to_remove = ['Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter', 'CSF', 'Unknown']
unique_parcellations = unique_parcellations[np.isin(unique_parcellations, regions_to_remove, invert=True)]
unique_parcellations = sorted(unique_parcellations, key=lambda x: x[::-1], reverse=True)

if regions is not None:
    unique_parcellations = [region for region in unique_parcellations if region in regions]
regression_parameters_dicts = [{region: [] for region in unique_parcellations} for c in range(len(conditions))]

significant_electrodes = {}  # dict of signif electrodes by subject

for i, subject_path in enumerate(tqdm(subject_paths)):
    subject_name = Path(subject_path).name[:6]
    subject_names.append(subject_name)

    rms_obj = np.load(subject_path, allow_pickle=True)
    rms = rms_obj['rms']
    parcellations = rms_obj['ch_parcellations']
    if first_parcel_label == True:
        parcellations = [[parcel[0]] for parcel in parcellations]
    elif first_parcel_label == "advanced":
        go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
        parcellations = select_parcel(parcellations, go_next)

    significant_electrodes[subject_name] = {}
    for c in range(rms.shape[1]):
        for region in regions:
            if any(ch_parcel in region for ch_parcel in parcellations[c]):
                significant_electrodes[subject_name][rms_obj['ch_names'][c]] = region


# Similarity

stim_features = pd.read_csv(behavior_path)
stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]

data = np.load(similarity_path, allow_pickle=True)

for s, subject in tqdm(enumerate(data["subjects"])):
    subject_stim_features = stim_features[stim_features['subject'] == subject]
    subject_info = data['info'][s]
    ch_names = subject_info['ch_names']
    indices_to_keep = []
    correct_list = []
    feature_list = []
    correlations_array = []
    stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
    similarity_matrix = data['similarity_matrices'][s]  # keep all channels

    first_trials_to_remove = 50
    for i, stim in enumerate(data['stim_names'][s]):
        if i > first_trials_to_remove:  # remove first trials
            if stim in subject_stim_features['stim_name'].tolist():
                if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                    if (stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb) & (stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb):  # check that both reps are in the data
                        indices_to_keep.append(i)
                        correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                        feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                        if 'rep1' in stim:
                            rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2'))
                        elif 'rep2' in stim:
                            rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                        else:
                            raise ValueError
                        correlations_array.append(similarity_matrix[i, rep_index, :])  # correlation of 2 reps

    correlations_array = np.array(correlations_array)

    for channel in range(correlations_array.shape[1]):
        if ch_names[channel] in significant_electrodes[subject].keys():
            Y = np.array(correct_list)
            X = np.zeros((Y.shape[0], 1))
            X[:, 0] = correlations_array[:, channel]
            X = (X - X.mean()) / X.std()
            X = sm.add_constant(X)

            ols = sm.Logit(Y, X).fit(disp=0)
            regression_parameters_dicts[0][significant_electrodes[subject][ch_names[channel]]].append(ols.params[1])

            Y = X[:, 1]
            X = np.zeros((Y.shape[0], 1))
            X[:, 0] = feature_list
            X = (X - X.mean(axis=0)) / X.std(axis=0)
            X = sm.add_constant(X)
            ols = sm.OLS(Y, X).fit()
            regression_parameters_dicts[2][significant_electrodes[subject][ch_names[channel]]].append(ols.params[1])


# Separability

stim_features = pd.read_csv(behavior_path)
stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]

data = np.load(distance_path, allow_pickle=True)

for s, subject in tqdm(enumerate(data["subjects"])):
    subject_stim_features = stim_features[stim_features['subject'] == subject]
    subject_info = data['info'][s]
    ch_names = subject_info['ch_names']
    indices_to_keep = []
    correct_list = []
    feature_list = []
    stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
    similarity_matrix = data['distance_matrices'][s]  # keep all channels
    paired_stim = []
    for i, stim in enumerate(data['stim_names'][s]):
        if i > first_trials_to_remove:  # remove first trials
            if ('rep1' in stim) | ('rep2' in stim):
                if stim in subject_stim_features['stim_name'].tolist():
                    if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                        indices_to_keep.append(i)
                        correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                        feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                        if 'rep2' in stim:
                            if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1')))
                            else:
                                paired_stim.append('no_pair')
                        if 'rep1' in stim:
                            if stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb:
                                paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2')))
                            else:
                                paired_stim.append('no_pair')

    condition_matrix = similarity_matrix[indices_to_keep, :, :]  # select condition
    correlations_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
    for i in range(condition_matrix.shape[0]):
        current_corr = condition_matrix[i, :, :]
        mask = np.ones(condition_matrix.shape[1], dtype=bool)
        mask[indices_to_keep[i]] = False  # remove current stim
        mask[indices_to_keep[i]:] = False  # remove following trials
        correlations_array[i, :] = np.mean(current_corr[mask, :], axis=0)

    for channel in range(correlations_array.shape[1]):
        if ch_names[channel] in significant_electrodes[subject].keys():
            Y = np.array(correct_list)
            X = np.zeros((Y.shape[0], 1))
            X[:, 0] = correlations_array[:, channel]
            X = (X - X.mean()) / X.std()
            X = sm.add_constant(X)

            ols = sm.Logit(Y, X).fit(disp=0)
            regression_parameters_dicts[1][significant_electrodes[subject][ch_names[channel]]].append(ols.params[1])

            Y = X[:, 1]
            X = np.zeros((Y.shape[0], 1))
            X[:, 0] = feature_list
            X = (X - X.mean(axis=0)) / X.std(axis=0)
            X = sm.add_constant(X)
            ols = sm.OLS(Y, X).fit()
            regression_parameters_dicts[3][significant_electrodes[subject][ch_names[channel]]].append(ols.params[1])

alternatives = ['greater', 'greater', 'less', 'greater']
# Calculate and save average rms for each condition and region
region_rms_per_condition = []
error_bars_per_condition = []
regions_per_condition = []
pvalue_per_condition = []
tvalue_per_condition = []
for cond in range(len(conditions)):
    condition_rms = []
    condition_error_bars = []
    condition_regions = []
    condition_pvalue = []
    condition_tvalue = []
    for region in unique_parcellations:
        region_rms = np.array(regression_parameters_dicts[cond][region])
        condition_regions.append(region)
        mean_rms = np.mean(region_rms, axis=0)
        condition_rms.append(mean_rms)
        error_rms = np.std(region_rms, axis=0) / np.sqrt(region_rms.shape[0])
        condition_error_bars.append(error_rms)
        ttest = ttest_1samp(region_rms, 0, alternative=alternatives[cond])
        condition_pvalue.append(ttest.pvalue)
        condition_tvalue.append(ttest.statistic)
    region_rms_per_condition.append(condition_rms)
    error_bars_per_condition.append(condition_error_bars)
    regions_per_condition.append(condition_regions)
    pvalue_per_condition.append(condition_pvalue)
    tvalue_per_condition.append(condition_tvalue)

# FDR correction
for c in range(len(pvalue_per_condition)):
    _, pval = mne.stats.fdr_correction(pvalue_per_condition[c], alpha=0.05, method='indep')
    pvalue_per_condition[c] = pval

new_names = {'ctx-lh-transversetemporal': 'L-TT', 'ctx-lh-superiortemporal': 'L-STG',
             'ctx-lh-middletemporal': 'L-MTG', 'ctx-lh-inferiortemporal': 'L-ITG',
             'ctx-lh-temporalpole': 'L-TP','ctx-lh-entorhinal': 'L-EC',
             'ctx-lh-parahippocampal': 'L-PHC',
             'Left-Hippocampus': 'L-HPC', 'Left-Amygdala': 'L-AMY',
             'ctx-rh-transversetemporal': 'R-TT', 'ctx-rh-superiortemporal': 'R-STG',
             'ctx-rh-middletemporal': 'R-MTG', 'ctx-rh-inferiortemporal': 'R-ITG',
             'ctx-rh-temporalpole': 'R-TP', 'ctx-rh-entorhinal': 'R-EC',
             'ctx-rh-parahippocampal': 'R-PHC',
             'Right-Hippocampus': 'R-HPC', 'Right-Amygdala': 'R-AMY'
             }

cmap = list(plt.get_cmap('Set3').colors)
del cmap[8]
region_colors = {region: cmap[i % int(len(new_names.keys()) / 2)] for i, region in enumerate(new_names.keys())}

fig, axs = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
# Plot each condition's data in a separate subplot
for i, (condition, regions, region_rms, error_bars, pvalues, tvalues) in enumerate(
        zip(conditions[2:4], regions_per_condition[2:4], region_rms_per_condition[2:4],
            error_bars_per_condition[2:4], pvalue_per_condition[2:4], tvalue_per_condition[2:4])):

    print("###################################", condition, "###################################")
    for r in range(len(regions)):
        print(regions[r] + ":", "p =", pvalues[r])

    if i==0:
        sorted_id = np.argsort(region_rms)[::-1]
    else:
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
        samples = np.array(regression_parameters_dicts[i+2][region])
        if remove_outliers:
            samples = samples[(np.abs(samples - samples.mean())) <= 2 * samples.std()]
        y_values = np.full(samples.shape, j) + np.random.uniform(-0.1, 0.1, samples.shape)  # Jitter for clarity
        axs[i].scatter(samples, y_values, color='black', alpha=0.7, s=2, label='Samples')
    axs[i].set_xlabel(f'\u03B2 ({condition})', fontsize=17)

plt.tight_layout()
fig.savefig(ROOT / "figures" / "figure_4" / "D.svg")


#################################### Panel E-F ####################################

# Electrode selection

behavior_all = pd.read_csv(behavior_path)
condition = 'surprise_mean_polyrnn'
path = Path(data_path)
subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
subject_names = []

significant_electrodes = {}  # dict of signif electrodes by subject

for i, subject_path in enumerate(tqdm(subject_paths)):
    subject_name = Path(subject_path).name[:6]
    subject_names.append(subject_name)
    behavior = behavior_all[behavior_all['subject'] == subject_name]
    correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
    incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
    correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
    incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

    rms_obj = np.load(subject_path, allow_pickle=True)
    rms = rms_obj['rms']
    parcellations = rms_obj['ch_parcellations']
    if first_parcel_label == True:
        parcellations = [[parcel[0]] for parcel in parcellations]
    elif first_parcel_label == "advanced":
        go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
        parcellations = select_parcel(parcellations, go_next)

    significant_electrodes[subject_name] = []
    for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
        correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
        incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
        if (len(correct) > 0) & (len(incorrect) > 0):
            for c in range(rms.shape[1]):
                for region in regions:
                    if any(ch_parcel in region for ch_parcel in parcellations[c]):
                        significant_electrodes[subject_name].append(rms_obj['ch_names'][c])
    significant_electrodes[subject_name] = np.unique(significant_electrodes[subject_name]).tolist()  # keep each electrode only once


# Similarity

stim_features = pd.read_csv(behavior_path)
stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]

data = np.load(similarity_path, allow_pickle=True)

channel_positions_list = []
subject_id = []
correct_full_list = []
feature_full_list = []
intrastim_distance_list = []
p_ttest_array = []  # p value of the contrast correct vs incorrect
signif_chan_list = []
for s, subject in tqdm(enumerate(data["subjects"])):
    subject_stim_features = stim_features[stim_features['subject'] == subject]
    subject_info = data['info'][s]
    ch_names = subject_info['ch_names']
    indices_to_keep = []
    correct_list = []
    feature_list = []
    correlations_array = []
    stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
    similarity_matrix = data['similarity_matrices'][s]  # keep all channels
    first_trials_to_remove = 50  # very sensitive to the first trials, then stabilizes (mostly for interstim distance, which makes sense)
    for i, stim in enumerate(data['stim_names'][s]):
        if i > first_trials_to_remove:  # remove first trials
            if stim in subject_stim_features['stim_name'].tolist():
                if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                    if (stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb) & (stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb):  # check that both reps are in the data
                        indices_to_keep.append(i)
                        correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                        feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                        if 'rep1' in stim:
                            rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2'))
                        elif 'rep2' in stim:
                            rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                        else:
                            raise ValueError
                        correlations_array.append(similarity_matrix[i, rep_index, :])  # correlation of 2 reps

    correlations_array = np.array(correlations_array)

    p_ttest_list = []
    for channel in range(correlations_array.shape[1]):
        if ch_names[channel] in significant_electrodes[subject]:
            Y = correlations_array[:, channel]
            X = np.zeros((Y.shape[0], 1))
            X[:, 0] = correct_list
            X = sm.add_constant(X)

            # independent t-test
            correct_corr = Y[X[:, 1] == 1]  # axis 0 is the intercept
            incorrect_corr = Y[X[:, 1] == 0]

            X = np.zeros((Y.shape[0], 1))
            X[:, 0] = feature_list
            X = sm.add_constant(X)
            ols = sm.OLS(Y, X).fit()
            param = ols.params[1]
            p_ttest = ols.pvalues[1]

            if (param < 0) & (p_ttest < 0.1):  # one-tailed test
                signif_chan_list.append(subject+'_'+ch_names[channel])
                correct_full_list += correct_list
                feature_full_list += feature_list
                intrastim_distance_list += ((Y - np.mean(Y)) / np.std(Y)).tolist()

                subject_id += [s for _ in range(len(correct_list))]

intrastim_distance_list = np.array(intrastim_distance_list)
correct_full_list = np.array(correct_full_list)
feature_full_list = np.array(feature_full_list)

fig, axs = plt.subplots(1, 1, figsize=(2, 2))
df = pd.DataFrame({'x': intrastim_distance_list, 'y': correct_full_list})
df = df.sort_values('x')
df['y_smooth'] = df['y'].rolling(window=30, min_periods=1, center=True).mean()
axs.scatter(df['x'], df['y_smooth'], alpha=1, s=0.5, label='rolling mean', color='lightgray')
axs.set_xlabel(f'Similarity', fontsize=12)
axs.set_ylabel('Correct', fontsize=12)
axs.tick_params(axis='y', labelsize=12)
axs.tick_params(axis='x', labelsize=12)
axs.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

print("correlation similarity and correct", pearsonr(intrastim_distance_list, correct_full_list, alternative='greater'))

num_bins = 10

slope, intercept = np.polyfit(intrastim_distance_list, correct_full_list, 1)
x_range = np.linspace(min(intrastim_distance_list), max(intrastim_distance_list), 100)
plt.plot(x_range, slope * x_range + intercept, color='k', label='linear fit', lw=2)

plt.tight_layout()
fig.savefig(ROOT / "figures" / "figure_4" / "E_inset.svg")

# correlation vs feature
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
df = pd.DataFrame({'x': feature_full_list, 'y': intrastim_distance_list})
df_avg = df.groupby('x', as_index=False)['y'].mean()
axs.scatter(df_avg['x'], df_avg['y'], alpha=1, s=20, label='average', color=plt.get_cmap("tab10")(1))
axs.set_ylabel(f'Similarity', fontsize=17)
axs.set_xlabel('Surprise', fontsize=17)
axs.tick_params(axis='y', labelsize=17)
axs.tick_params(axis='x', labelsize=17)
axs.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

x_range = np.linspace(min(feature_full_list), max(feature_full_list), 100)
slope, intercept = np.polyfit(feature_full_list, intrastim_distance_list, 1)
plt.plot(x_range, slope * x_range + intercept, color='k', label='linear fit', lw=3)

quadratic, slope, intercept = np.polyfit(feature_full_list, intrastim_distance_list, 2)

# exponential fit
def func(x, a, b, c):
    return a * np.exp(b * x) + c
popt, pcov = curve_fit(func, feature_full_list, intrastim_distance_list, maxfev = 10000)
# compute AIC
resid = intrastim_distance_list - func(feature_full_list, *popt)
N = len(intrastim_distance_list)
k = len(popt)
sigma2 = np.sum(resid ** 2) / N
logL = -0.5 * N * (np.log(2 * np.pi * sigma2) + 1)
aic_exponential = 2 * k - 2 * logL

plt.legend()

# compare linear and quadratic models' AIC
Y = intrastim_distance_list
X = np.zeros((Y.shape[0], 1))
X[:, 0] = feature_full_list
X = sm.add_constant(X)
ols = sm.OLS(Y, X).fit()
aic_linear = ols.aic
X = np.zeros((Y.shape[0], 2))
X[:, 0] = feature_full_list
X[:, 1] = feature_full_list**2
X = sm.add_constant(X)
ols = sm.OLS(Y, X).fit()
aic_quadratic = ols.aic

plt.tight_layout()
fig.savefig(ROOT / "figures" / "figure_4" / "E.svg")


# Separability

stim_features = pd.read_csv(behavior_path)
stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]

data = np.load(distance_path, allow_pickle=True)

channel_positions_list = []
subject_id = []
correct_full_list = []
feature_full_list = []
interstim_distance_list = []
signif_chan_list2 = []
for s, subject in tqdm(enumerate(data["subjects"])):
    subject_stim_features = stim_features[stim_features['subject'] == subject]
    subject_info = data['info'][s]
    ch_names = subject_info['ch_names']
    indices_to_keep = []
    correct_list = []
    feature_list = []
    stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
    similarity_matrix = data['distance_matrices'][s]  # keep all channels
    paired_stim = []
    for i, stim in enumerate(data['stim_names'][s]):
        if i > first_trials_to_remove:  # remove first trials
            if ('rep1' in stim) | ('rep2' in stim):
                if stim in subject_stim_features['stim_name'].tolist():
                    if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                        indices_to_keep.append(i)
                        correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                        feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                        if 'rep2' in stim:
                            if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1')))
                            else:
                                paired_stim.append('no_pair')
                        if 'rep1' in stim:
                            if stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb:
                                paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2')))
                            else:
                                paired_stim.append('no_pair')

    condition_matrix = similarity_matrix[indices_to_keep, :, :]  # select condition
    correlations_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
    for i in range(condition_matrix.shape[0]):
        current_corr = condition_matrix[i, :, :]
        mask = np.ones(condition_matrix.shape[1], dtype=bool)
        mask[indices_to_keep[i]:] = False  # remove current stim and following trials
        correlations_array[i, :] = np.mean(current_corr[mask, :], axis=0)

    p_ttest = np.zeros(correlations_array.shape[1])
    for channel in range(correlations_array.shape[1]):
        if ch_names[channel] in significant_electrodes[subject]:
            Y = correlations_array[:, channel]
            X = np.zeros((Y.shape[0], 1))
            X[:, 0] = correct_list
            X = sm.add_constant(X)

            # independent t-test
            correct_corr = Y[X[:, 1] == 1]  # axis 0 is the intercept
            incorrect_corr = Y[X[:, 1] == 0]

            X = np.zeros((Y.shape[0], 1))
            X[:, 0] = feature_list
            X = sm.add_constant(X)
            ols = sm.OLS(Y, X).fit()
            param = ols.params[1]
            p_ttest[channel] = ols.pvalues[1]

            if (param > 0) & (p_ttest[channel] < 0.1):  # one-tailed test
                signif_chan_list2.append(subject + '_' + ch_names[channel])
                correct_full_list += correct_list
                feature_full_list += feature_list
                interstim_distance_list += ((Y - np.mean(Y)) / np.std(Y)).tolist()

                subject_id += [s for _ in range(len(correct_list))]

interstim_distance_list = np.array(interstim_distance_list)
correct_full_list = np.array(correct_full_list)
feature_full_list = np.array(feature_full_list)

fig, axs = plt.subplots(1, 1, figsize=(2, 2))
df = pd.DataFrame({'x': interstim_distance_list, 'y': correct_full_list})
df = df.sort_values('x')
df['y_smooth'] = df['y'].rolling(window=30, min_periods=1, center=True).mean()
axs.scatter(df['x'], df['y_smooth'], alpha=1, s=0.5, label='rolling mean', color='lightgray')
axs.set_xlabel(f'Separability', fontsize=12)
axs.set_ylabel('Correct', fontsize=12)
axs.tick_params(axis='y', labelsize=12)
axs.tick_params(axis='x', labelsize=12)
axs.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

print("correlation separability and correct", pearsonr(interstim_distance_list, correct_full_list, alternative='greater'))

slope, intercept = np.polyfit(interstim_distance_list, correct_full_list, 1)
x_range = np.linspace(min(interstim_distance_list), max(interstim_distance_list), 100)
plt.plot(x_range, slope * x_range + intercept, color='k', label='linear fit', lw=2)

plt.tight_layout()
fig.savefig(ROOT / "figures" / "figure_4" / "F_inset.svg")

# interstim distance vs surprise
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
df = pd.DataFrame({'x': feature_full_list, 'y': interstim_distance_list})
df_avg = df.groupby('x', as_index=False)['y'].mean()
axs.scatter(df_avg['x'], df_avg['y'], alpha=1, s=20, label='average', color=plt.get_cmap("tab10")(1))
axs.set_ylabel(f'Separability', fontsize=17)
axs.set_xlabel('Surprise', fontsize=17)
axs.tick_params(axis='y', labelsize=17)
axs.tick_params(axis='x', labelsize=17)
axs.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

slope, intercept = np.polyfit(feature_full_list, interstim_distance_list, 1)
x_range = np.linspace(min(feature_full_list), max(feature_full_list), 100)
plt.plot(x_range, slope * x_range + intercept, color='k', label='linear fit', lw=3)

quadratic, slope, intercept = np.polyfit(feature_full_list, interstim_distance_list, 2)
x_range = np.linspace(min(feature_full_list), max(feature_full_list), 100)

# exponential fit
popt, pcov = curve_fit(func, feature_full_list, interstim_distance_list)
# compute AIC
resid = interstim_distance_list - func(feature_full_list, *popt)
N = len(interstim_distance_list)
k = len(popt)
sigma2 = np.sum(resid ** 2) / N
logL = -0.5 * N * (np.log(2 * np.pi * sigma2) + 1)
aic_exponential = 2 * k - 2 * logL

plt.legend()

# compare linear and quadratic models' AIC
Y = interstim_distance_list
X = np.zeros((Y.shape[0], 1))
X[:, 0] = feature_full_list
X = sm.add_constant(X)
ols = sm.OLS(Y, X).fit()
aic_linear = ols.aic
X = np.zeros((Y.shape[0], 2))
X[:, 0] = feature_full_list
X[:, 1] = feature_full_list ** 2
X = sm.add_constant(X)
ols = sm.OLS(Y, X).fit()
aic_quadratic = ols.aic
plt.tight_layout()

fig.savefig(ROOT / "figures" / "figure_4" / "F.svg")
