from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from scipy.optimize import curve_fit
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from utils import select_parcel, logistic_log_likelihood

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

ROOT = Path(__file__).resolve().parents[1]
print('ROOT:', ROOT)


behavior_path = ROOT / "data" / "behavior.csv"
condition = 'surprise_mean_polyrnn'
data_path = ROOT / "data" / "rms"
first_parcel_label = "advanced"
regions = ['ctx-lh-middletemporal', 'ctx-rh-middletemporal', 'Left-Hippocampus',
           'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal',
           'ctx-lh-entorhinal', 'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal',
           'ctx-rh-inferiortemporal']  # regions of temporal lobe that show a power contrast
similarity_path = ROOT / "data" / "similarity_separability" / "similarity.pkl"
distance_path = ROOT / "data" / "similarity_separability" / "separability.pkl"


# Electrode selection

behavior_all = pd.read_csv(behavior_path)
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


# Representation analysis

stim_features = pd.read_csv(behavior_path)
stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]

data = np.load(similarity_path, allow_pickle=True)
distance_data = np.load(distance_path, allow_pickle=True)

# exp function for fitting
def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c
# quadratic fit
def quadr_func(x, a, b, c):
    return a * x**2 + b * x + c
# linear fit
def linear_func(x, a, b):
    return a * x + b

func = exp_func

subject_id = []
correct_full_list = []
intrastim_distance_full_array = []
interstim_distance_full_array = []
coefs_intra = []
coefs_inter = []
saved_model = []
behavioral_models = []  # one logistic regression per subject, using surprise and surprise²
curve_param_intra = []  # parameters of the exponental fit
curve_param_inter = []
correlation_intra_inter = []  # check correlation between the two predictors
r2_intra = []  # r2 of exp fit
r2_inter = []
r2_intra_selected = []  # r2 of selected channels only
r2_inter_selected = []
logL_full_model = []
logL_intra = []
logL_inter = []
logL_null = []
logL_behavioral_model = []
p_ttest_array = []  # p value of the contrast correct vs incorrect
for s, subject in tqdm(enumerate(data["subjects"])):
    subject_stim_features = stim_features[stim_features['subject'] == subject]
    subject_info = data['info'][s]
    ch_names = subject_info['ch_names']
    indices_to_keep = []
    correct_list = []
    feature_list = []
    intrastim_distance_array = []
    stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
    similarity_matrix = data['similarity_matrices'][s]  # keep all channels
    distance_matrix = distance_data['distance_matrices'][s]  # keep all channels
    first_trials_to_remove = 50
    for i, stim in enumerate(data['stim_names'][s]):
        if i > first_trials_to_remove:
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
                        intrastim_distance_array.append(similarity_matrix[i, rep_index, :])  # correlation of 2 reps

    feature_list = np.array(feature_list)
    feature_z = (feature_list - feature_list.mean()) / feature_list.std()
    correct_list = np.array(correct_list)
    intrastim_distance_array = np.array(intrastim_distance_array)
    condition_matrix = distance_matrix[indices_to_keep, :, :]
    interstim_distance_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
    for i in range(condition_matrix.shape[0]):
        current_corr = condition_matrix[i, :, :]
        mask = np.ones(condition_matrix.shape[1], dtype=bool)
        mask[indices_to_keep[i]:] = False  # remove current stim and following trials
        interstim_distance_array[i, :] = np.mean(current_corr[mask, :], axis=0)

    for channel in range(intrastim_distance_array.shape[1]):
        if ch_names[channel] in significant_electrodes[subject]:
            Y = intrastim_distance_array[:, channel]
            X = np.zeros((Y.shape[0], 1))
            X[:, 0] = feature_list
            X = sm.add_constant(X)
            ols = sm.OLS(Y, X).fit()
            param_intra = ols.params[1]
            p_intra = ols.pvalues[1]

            Y = interstim_distance_array[:, channel]
            X = np.zeros((Y.shape[0], 1))
            X[:, 0] = feature_list
            X = sm.add_constant(X)
            ols = sm.OLS(Y, X).fit()
            param_inter = ols.params[1]
            p_inter = ols.pvalues[1]

            # fit exponentials
            intra_z = (intrastim_distance_array[:, channel] - np.mean(intrastim_distance_array[:, channel])) / np.std(intrastim_distance_array[:, channel])
            inter_z = (interstim_distance_array[:, channel] - np.mean(interstim_distance_array[:, channel])) / np.std(interstim_distance_array[:, channel])
            maxfev = 100000
            if func == exp_func:
                popt_intra, pcov_intra = curve_fit(exp_func, feature_z, intra_z, p0 = [1, 0, 0], bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), maxfev=maxfev)
                popt_inter, pcov_inter = curve_fit(exp_func, feature_z, inter_z, p0 = [1, 0, 0], bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), maxfev=maxfev)
            elif func == linear_func:
                popt_intra, pcov_intra = curve_fit(linear_func, feature_z, intra_z, maxfev=maxfev)
                popt_inter, pcov_inter = curve_fit(linear_func, feature_z, inter_z, maxfev=maxfev)
            predicted_intra = func(feature_z, *popt_intra)
            predicted_inter = func(feature_z, *popt_inter)
            r2_intra.append(1 - np.sum((predicted_intra - intra_z)**2) / np.sum((intra_z - intra_z.mean())**2))
            r2_inter.append(1 - np.sum((predicted_inter - inter_z)**2) / np.sum((inter_z - inter_z.mean())**2))

            r2_threshold = 0.05  # if either fit reaches the threshold, we keep the channel
            if ((r2_intra[-1] > r2_threshold) & (popt_intra[1] < 0)) | ((r2_inter[-1] > r2_threshold) & (popt_inter[1] > 0)):
                r2_intra_selected.append(r2_intra[-1])
                r2_inter_selected.append(r2_inter[-1])

                # Fit logistic regression
                X = np.zeros((predicted_intra.shape[0], 2))
                X[:, 0] = predicted_intra
                X[:, 1] = predicted_inter
                model = LogisticRegression(penalty=None)
                model.fit(X, correct_list)

                coefs_intra.append(model.coef_[0, 0])
                coefs_inter.append(model.coef_[0, 1])
                saved_model.append(model)
                curve_param_intra.append(popt_intra)
                curve_param_inter.append(popt_inter)
                subject_id.append(s)
                correlation_intra_inter.append(np.corrcoef(X[:, 0], X[:, 1])[0, 1])

                # store model logL
                proba = model.predict_proba(X)
                logL_full_model.append(logistic_log_likelihood(correct_list, proba[:, 1]))

                # control models with only one predictor or one intercept
                model = LogisticRegression(penalty=None)
                model.fit(X[:, 0][:, np.newaxis], correct_list)
                logL_intra.append(logistic_log_likelihood(correct_list, model.predict_proba(X[:, 0][:, np.newaxis])[:, 1]))
                model = LogisticRegression(penalty=None)
                model.fit(X[:, 1][:, np.newaxis], correct_list)
                logL_inter.append(logistic_log_likelihood(correct_list, model.predict_proba(X[:, 1][:, np.newaxis])[:, 1]))
                logL_null.append(logistic_log_likelihood(correct_list, np.sum(correct_list) / correct_list.shape[0]))

    # subject's behavioral model
    if (len(subject_id) > 0) & (subject_id[-1] == s):  # only if at least 1 channel of that subject was added
        X = np.zeros((feature_z.shape[0], 2))
        X[:, 0] = feature_z
        X[:, 1] = feature_z**2
        model = LogisticRegression(penalty=None)
        model.fit(X, correct_list)
        behavioral_models.append(model)
        logL_behavioral_model.append(logistic_log_likelihood(correct_list, model.predict_proba(X)[:, 1]))

# stat test
test_intra = ttest_1samp(coefs_intra, 0, alternative='greater')
test_inter = ttest_1samp(coefs_inter, 0, alternative='greater')
pval_intra = test_intra.pvalue
pval_inter = test_inter.pvalue
print("Test on logistic regression parameters")
print("Similarity: ", test_intra)
print("Separability: ", test_inter)


# boxplots
df = pd.DataFrame({'Similarity': coefs_intra, 'Separability': coefs_inter})
fig = plt.figure(figsize=(2.5, 2.5))
sns.violinplot(data=df, showmeans=True, showfliers=False, palette=['#FF928A', '#978EF5'], orient='v', inner=None, linewidth=1)
plt.gca().axhline(0, ls='--', color='k', lw=0.5)
plt.ylabel('Logistic \u03B2', fontsize=13)
plt.tick_params(axis='y', labelsize=13)
plt.tick_params(axis='x', labelsize=13)

plt.scatter(0, np.mean(coefs_intra), color='w', edgecolor='k', lw=1.5, s=30, zorder=10)
plt.scatter(1, np.mean(coefs_inter), color='w', edgecolor='k', lw=1.5, s=30, zorder=10)

# add stars
for i, p in enumerate([pval_intra, pval_inter]):
    if p < 0.05:
        # Get the position to place the star
        x = i  # The x position corresponds to the box index
        y = df.iloc[:, i].max() + 0.15  # Slightly above the max value of the box
        plt.text(x, y, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')
plt.tight_layout()

fig.savefig(ROOT / "figures" / "figure_5" / "B_inset.svg")


# curve parameters
curve_param_intra = np.array(curve_param_intra)
curve_param_inter = np.array(curve_param_inter)

# compare model logL
logL_full_model = np.array(logL_full_model)
logL_intra = np.array(logL_intra)
logL_inter = np.array(logL_inter)
logL_null = np.array(logL_null)

test_intra = ttest_1samp(logL_full_model - logL_intra, 0, alternative='greater')
print("ttest logL full model - similarity: ", test_intra)
test_inter = ttest_1samp(logL_full_model - logL_inter, 0, alternative='greater')
print("ttest logL full model - separability: ", test_inter)
test_null = ttest_1samp(logL_full_model - logL_null, 0, alternative='greater')
print("ttest logL full model - null: ", test_null)

# plot surprise - neural patterns relationships
simulated_surprise = np.linspace(feature_z.min(), feature_z.max(), 100)
curve_param_intra = [curve_param_intra[c, :] for c in range(curve_param_intra.shape[0])]
simulated_similarity = np.array([func(simulated_surprise, *popt) for popt in curve_param_intra])
curve_param_inter = [curve_param_inter[c, :] for c in range(curve_param_inter.shape[0])]
simulated_separability = np.array([func(simulated_surprise, *popt) for popt in curve_param_inter])
colors = ['#FF928A', '#978EF5']
fig, axes = plt.subplots(2, 1, figsize=(2.5, 5), sharex=True)
for ax, simulated in enumerate([simulated_similarity, simulated_separability]):
    axes[ax].plot(simulated_surprise, simulated.T, c=colors[ax], alpha=0.3, lw=0.5)
    mean = np.mean(simulated, axis=0)
    axes[ax].plot(simulated_surprise, mean, c=colors[ax], lw=4)
    axes[ax].set_ylim(mean.min(), mean.max())
    axes[ax].tick_params(axis='y', labelsize=17)
    axes[ax].tick_params(axis='x', labelsize=17)
    axes[ax].set_xlim(simulated_surprise.min(), simulated_surprise.max())
axes[1].set_xlabel('Surprise', fontsize=17)
axes[0].set_ylabel('Similarity', fontsize=17)
axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[1].set_ylabel('Separability', fontsize=17)
axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.tight_layout()

fig.savefig(ROOT / "figures" / "figure_5" / "A.svg")


# plot surprise -> P(correct)
proba_correct = np.zeros_like(simulated_similarity)
for c in range(proba_correct.shape[0]):  # channel models
    X = np.zeros((proba_correct.shape[1], 2))
    X[:, 0] = simulated_similarity[c, :]
    X[:, 1] = simulated_separability[c, :]
    proba_correct[c, :] = saved_model[c].predict_proba(X)[:, 1]
proba_correct_behavioral = np.zeros((len(behavioral_models), simulated_surprise.shape[0]))
for s in range(len(behavioral_models)):  # subject models (correct ~ surprise + surprise²)
    X = np.zeros((simulated_surprise.shape[0], 2))
    X[:, 0] = simulated_surprise
    X[:, 1] = simulated_surprise**2
    proba_correct_behavioral[s, :] = behavioral_models[s].predict_proba(X)[:, 1]

fig = plt.figure(figsize=(5, 5))
plt.plot(simulated_surprise, proba_correct.T, c='#fdb462', alpha=0.3, lw=0.5, zorder=0)
weights = logL_full_model - logL_full_model.min() - logL_full_model.max()  # translate to get positive values
weights = weights / weights.sum()
sorted_perf = np.argsort(logL_full_model)[::-1]
mean = np.mean(proba_correct, axis=0)  # average of all channels
behavioral_mean = np.mean(proba_correct_behavioral, axis=0)
plt.plot(simulated_surprise, mean, c='#fdb462', lw=4, label='Model average')
plt.plot(simulated_surprise, behavioral_mean, c='k', lw=4, label='Quadratic surprise model', zorder=1)
plt.xlabel('Surprise', fontsize=17)
plt.ylabel('Correct', fontsize=17)
plt.legend(fontsize=17)
plt.tick_params(axis='y', labelsize=17)
plt.tick_params(axis='x', labelsize=17)
plt.xlim(simulated_surprise.min(), simulated_surprise.max())
plt.tight_layout()

fig.savefig(ROOT / "figures" / "figure_5" / "B.svg")