Sleep Stage Classification from EEG 

This project is tasked with building a machine learning pipeline in order to classify 30-second 
EEG segments into their corresponding sleep stages using the Sleep-EDF dataset. We then 
compare a deep learning approach (1D CNN) with a Random Forest baseline using our 
engineered features. The goal of this project is to support sleep monitoring for the emerging 
consumer wearables market. The best model achieved 88.6% accuracy, and cross-validation 
confirmed via folding that results were significantly above chance. In accordance with 
recommendations based on my first project, attempts are made to handle potential data leakage 
and engineer relevant features.  





Repository Structure for Reproducibility  




data_ingestion/ 

●  artifact_screening.py – Identifies and removes the flatline segments, the amplitude 

outliers, and the spectral noise from raw EEG. 

● load_and_qc.py – Loads the raw EDF data, extracts the EEG channels, and then 
converts to 30-second epochs, finally it attaches the corresponding sleep stage labels. 

●  new_EDA.py – Performs the qualitative exploratory data analysis, including all the sleep 

stage distribution, spectral profile validation, and finally the ICA decomposition. 




feature_engineering/ 

●  generate_spectral_features.py – Extracts the spectral features (delta–gamma power, 

log-transforms, ratios … .) and then also adds transition-aware features like prev_stage. 





●  engineered_features/ – Contains per-subject .csv files with all of the extracted and 

transformed features from the sleep-only data. 





● reduced_features/ – Contains the feature-reduced datasets like (MI-50 and ANOVA-20), 
but pre-split into train/test sets for future modeling. 




feature_reduction/ 

● feature_reduction.py – Selects the top features using Mutual Information and the ANOVA 
F-test, storing the best-performing subsets for future modeling. 





models/ 

●  CNN_CV.py – Implements cross-validation for the CNN using MI-50 and ANOVA-20 
feature sets, includes shuffled-label control tests. Also includes all of the code for the 
actual CNN model, as well. Previous versions included visualizations to assist in making plots.  

● random_forest_baseline.py – Trains a Random Forest classifier on all of the same 
features and performs evaluation for some baseline comparisons. 

sleep_1d_cnn/ 

○  Contains the training plots and the model outputs for the original single-run CNN 

experiments. 

sleep_1d_cnn_with_cv/ 

○  Stores results and confusion matrices from CNN cross-validation experiments. 
The formats the files are saved in are not readable. I do not believe they will 
transfer to GitHub 





processed_eeg_data/ 

●  Contains all of the subject-level .csv files with cleaned, labeled and 30-second EEG 

epochs from both the Fpz-Cz and the Pz-Oz channels. 





sleep_only_data/ 

●  Same as above, but excludes all of that wake-stage data, leaving only sleep stages 

N1–N4 and REM for feature extraction. No waking data, only sleeping data. 





sleep-cassette/

●  Raw EDF files downloaded from the Sleep-EDF dataset via PhysioNet. These are all of 

the original unprocessed files. 









How to get the data: 
The raw EEG data is available from the Sleep-EDF Expanded dataset on PhysioNet. Download 
all of the “Sleep Cassette” recordings. 
https://physionet.org/content/sleep-edfx/1.0.0/#files-panel 

