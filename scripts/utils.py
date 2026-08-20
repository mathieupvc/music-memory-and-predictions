import numpy as np

def select_parcel(parcellations, go_next):
    """for each list of parcellations, select the first parcellation that is not in go_next."""
    new_parcellations = []
    for parcels in parcellations:
        added = False
        for parcel in parcels:
            if parcel not in go_next:
                new_parcellations.append([parcel])
                added = True
                break
        if not added:
            new_parcellations.append([parcels[0]])  # if all parcels were in go_next, add the first one
    assert len(parcellations) == len(new_parcellations)
    return new_parcellations


def permute_labels_and_compute_contrast(rms, correct_indices, incorrect_indices, n_permutations=1000):
    """Permute correct and incorrect labels across epochs and compute contrasts."""

    observed_diff = np.mean(rms[correct_indices]) - np.mean(rms[incorrect_indices])

    perm_diffs = np.zeros(n_permutations + 1)

    combined_indices = np.concatenate([correct_indices, incorrect_indices])
    n_correct = len(correct_indices)

    for i in range(n_permutations):
        # Shuffle the indices
        shuffled_indices = np.random.permutation(combined_indices)
        # Split shuffled data into 'correct' and 'incorrect' groups
        perm_correct_indices = shuffled_indices[:n_correct]
        perm_incorrect_indices = shuffled_indices[n_correct:]
        # Compute the contrast for the permutation
        perm_diffs[i] = np.mean(rms[perm_correct_indices]) - np.mean(rms[perm_incorrect_indices])
    perm_diffs[-1] = observed_diff

    return observed_diff, perm_diffs


def logistic_log_likelihood(y_true, y_pred):
    """Compute log likelihood for logistic regression (bernoulli distribution)"""
    return np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))