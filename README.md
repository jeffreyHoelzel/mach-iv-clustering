# MACH-IV Clustering
Using clustering algorithms to explore patterns in Machiavellianism (MACH-IV) personality data.

## Summary
This project applies four different clustering algorithms to the [MACH-IV dataset](https://openpsychometrics.org/_rawdata/) (a measure of Machiavellianism personality traits) in order to explore latent grouping patterns in the data. The repository contains code for preprocessing, applying clustering (K‑Means clustering, Gaussian Mixture Model, Spectral Clustering, and Hierarchical Agglomerative Clustering), and visualising the results. The goal is to identify whether distinct clusters of respondents emerge on the MACH-IV scale and to characterise those clusters in terms of personality traits and other variables.

## Architecture
```plaintext
├── cluster_analysis/          # high-level analysis code
├── data/                      # raw and processed datasets
├── docs/                      # all code for the website
├── kmeans/                    # code for k-means clustering
├── gmm/                       # code for GMM clustering
├── hierarchical/              # code for hierarchical clustering
├── spectral/                  # code for spectral clustering
├── requirements.txt           # Python dependencies
├── LICENSE                    # we use the MIT license
└── README.md                  # you are here!
```
## Dataset
- The primary dataset used is a subset of the MACH-IV scale containing 20 likert-style questions where 5=strongly agree and 1=strongly disagree, obtained from the [`Open-Source Psychometrics Project`](https://openpsychometrics.org/_rawdata/)
- The data directory includes the raw responses and various demographic or auxillary variables (i.e., age, gender, location, etc.).
- We specifically identified relationships between clustered Machiavellianism levels and the personality traits based off of the Ten Item Personality Inventory (TIPI).
- We store the pre-processed data in a CSV file located at [`data/MACH_data/data.cleaned.csv`](https://github.com/jeffreyHoelzel/mach-iv-clustering/tree/main/data/MACH_data).

## Getting Started
### Local Usage
1. Clone this repository using either SSH or HTTPS, or download this codebase as a .zip file extract everything locally.
2. Ensure you are in the root directory:
```bash
cd path/to/mach-iv-clustering
```
4. Create a Python virtual environment to install all the required dependencies in:
```bash
python -m venv venv              # creates a virutal environment named 'venv'
```
3. Activiate your Python virtual environment:
```bash
./venv/Scripts/activate          # for Windows Powershell
source venv/bin/activate         # for Mac or Linux
```
5. Install the dependencies:
```bash
pip install -r requirements.txt  # installs all dependencies in the requirements.txt file
```
6. Configuration
- Modify the global variables used throughout each algorithm at `<algorithm>/setup/config.py` (i.e., features, path to data, etc.).
- Modify any arguments necessary in each main script at `<algorithm>/run_<algorithm>.py`.
7. Simply run (a time-stamped artifacts folder will be generated in your current directory containing the program output):
```bash
python <algorithm>/run_<algorithm>.py
```
8. To run cluster analysis (a time-stamped artifacts folder will be generated in your current directory containing the program output):
```bash
 python cluster_analysis/analyze_clusters.py -i <algorithm>\<artifacts_folder>\data\<cluster_labels>.csv
```

### HPC Usage
1. Ensure you have access to an HPC. For this guide, we are assuming you are an NAU student with access to the Monsoon HPC. We are also assuming you have some basic understanding of the Linux command line and Monsoon.
2. Clone this repository using either SSH or HTTPS, or download this codebase as a .zip file extract everything locally.
3. SCP the dataset to your scratch directory:
```bash
scp path/to/mach-iv-clustering/MACH_data/data.cleaned.csv \
<NAU_ID>@monsoon.hpc.nau.edu:/scratch/<NAU_ID>      # ex. NAU ID: abc123
```
5. SCP the Bash shell script, .pyz files corresponding for the algorithm you want to run, and the `environment.yml` file to your home directory:
```bash
scp path/to/mach-iv-clustering/<algorithm>/hpc/run_<algorithm>_clustering.pyz \
path/to/mach-iv-clustering/<algorithm>/hpc/run_<algorithm>.sh \
path/to/mach-iv-clustering/environment.yml \
<NAU_ID>@monsoon.hpc.nau.edu:~/
```
6. Set up a Conda virtual environment your job can activate (update .sh script with environment name from .yml file):
```bash
module load anaconda3
conda env create -f environment.yml # creates a Conda environment named 'mach-iv-clustering'
```
7. Submit a new job (script takes care of dependency installation):
```bash
sbatch run_<algorithm>.sh
```
8. You can view the status of your job using the commands below. A time-stamped artifacts folder will be generated in your home directory containing the program output.
```bash
squeue --job <job_id>            # for GPU queue
jobstats -r                      # for any running jobs
```

9. Once the job is complete, you can view the results in your `scratch/<NAU_ID>/` directory and SCP the results back to your machine.
```bash
cd /scratch/$USER
```
