# music-memory-and-predictions
Code for the paper "Surprise gates two distinct mechanisms to support memorability in music" (https://doi.org/10.64898/2026.07.21.739807 )

# Installation
git clone https://github.com/mathieupvc/music-memory-and-predictions.git
cd music-memory-and-predictions
conda env create -f environment.yml

# Data
You can download the data here: TODO link to zenodo

# Use
conda activate memo_pred
python scripts/figure_2.py

# Data description
behavior.csv: only trials that have 2 presentations.
behavior_all_trials.csv: all trials, even ones that are presented only once.
rms: data for the neural activity contrast analysis
rms_ma: neural activity contrasts over time
similarity_separability: neural pattern similarity and separability

# Figure 1
For details on PolyRNN (panels C and D), see https://doi.org/10.1101/2024.11.27.625704

# Brain visualizations (not implemented yet)
Brain panels use a different MNE version and are thus available in the standalone notebook.
