# Towards Measuring Representational Similarity of Large Language Models

This is the code for our extended abstract at the UniReps Workshop at NeurIPS 2023: "Towards Measuring Representational Similarity of Large Language Models".

Our data is available on Zenodo: https://doi.org/10.5281/zenodo.8411089



## Reproducing results

### 1. Set up Python environment
```shell
conda create -n llmcomp python=3.10
conda active llmcomp
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio hydra-core ipykernel flake8 black huggingface-hub ipywidgets matplotlib seaborn numpy scipy pyarrow tokenizers datasets transformers
pip install . -e
```

### 2. (Optional) Comparing representations
Download the representations from [Zenodo](https://doi.org/10.5281/zenodo.8411089) and unzip them.
This will take some time.

Then run the following commands:
```shell
python compare_representations.py storage.results_subdir=PATH_TO_WHERE_RESULTS_WILL_BE_STORED measures=ot+is+tr storage.reps_subdir=PATH_TO_REPRESENTATIONS_FOR_WINOGRANDE
python compare_representations.py storage.results_subdir=PATH_TO_WHERE_RESULTS_WILL_BE_STORED measures=ot+is storage.reps_subdir=PATH_TO_REPRESENTATIONS_FOR_WINOGRANDE
python compare_representations.py storage.results_subdir=PATH_TO_WHERE_RESULTS_WILL_BE_STORED measures=ot+is+tr storage.reps_subdir=PATH_TO_REPRESENTATIONS_FOR_HUMANEVAL
python compare_representations.py storage.results_subdir=PATH_TO_WHERE_RESULTS_WILL_BE_STORED measures=ot+is storage.reps_subdir=PATH_TO_REPRESENTATIONS_FOR_HUMANEVAL
```
We recommend running multiple commands in parallel, e.g., from different tmux sessions.
Otherwise you may have to wait ~24h for the results.

### 3. Analyze similarity scores
If you have not run step 2, download the parquet files from [Zenodo](https://doi.org/10.5281/zenodo.8411089).

Use `plots.ipynb` to recreate the plots from the paper.

## License
Code is licensed under MIT license.
The data is available under CC-BY 4.0 license (see Zenodo).
