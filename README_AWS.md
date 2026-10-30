# BraTS Domain Generalization — AWS Migration

This bundle converts these two actual notebooks into runnable command-line programs:

- `final_report.ipynb` -> `run_final_report_aws.py` (baseline, MixStyle, KL, MMD, LOSO, consistency analysis and MMD analysis)
- `CSU_BraTS_experiments.ipynb` -> `run_csu_aws.py` (CSU-1, comparison, optional three seeds)

The `dg/` folder is the original code plus the CSU patch already integrated, so do **not** unzip `dg_csu_patch.zip` again into this bundle. Neither script needs Google Drive mounted or re-downloads the dataset by default. Existing original checkpoint names are preserved.

**Note:** A third notebook, `mid_sem.ipynb`, is visible in the screenshot but was not available here, so it has not been converted.

## Folder structure

```text
~/brats-dg/
├── dg/                        # already includes CSU integration
├── aws_paths.py               # dataset path relocation + split checks
├── run_final_report_aws.py    # original notebook converted to CLI
├── run_csu_aws.py             # CSU standalone notebook converted to CLI
├── requirements.txt
├── README_AWS.md
├── brats23_dg_png/            # COPY FROM GOOGLE DRIVE, includes index.csv, site_*/..., *.npz
│   ├── index.csv
│   ├── site_01/
│   └── ...
└── artifacts/                 # transfer from Drive to retain old checkpoints
    ├── checkpoints/
    └── results/
```

The raw BraTS2023 ZIP and mapping XLSX are not needed when your NPZ dataset already exists.

## 1. Install code on EC2 (Ubuntu example)

Use a **GPU instance** with an NVIDIA GPU, such as an existing g5 instance; ensure enough EBS storage for the dataset and artifacts.

```bash
ssh -i ~/Downloads/YOUR_AWS_KEY.pem ubuntu@YOUR_EC2_PUBLIC_IP
sudo apt-get update
sudo apt-get install -y unzip tmux python3-venv
mkdir -p ~/brats-dg
cd ~/brats-dg
```

Upload the downloaded `brats_dg_aws_migration.zip` from your *local* terminal:

```bash
scp -i ~/Downloads/YOUR_AWS_KEY.pem ~/Downloads/brats_dg_aws_migration.zip ubuntu@YOUR_EC2_PUBLIC_IP:~/brats-dg/
```

Back on EC2:

```bash
cd ~/brats-dg
unzip -o brats_dg_aws_migration.zip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# If torch / torchvision already work in your Deep Learning AMI, use that environment instead.
# Otherwise install GPU-enabled torch + torchvision using the official current selector:
# https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())'
nvidia-smi
```

Don't mix a CPU-only `torch` installation with a CUDA training expectation. If the virtual environment doesn't have CUDA-enabled PyTorch, install a matching pair from the official PyTorch installer before training.

## 2. Transfer the preprocessed dataset and old checkpoints

The Google Drive screenshot shows `brats23_dg_png` and `artifacts` folders. Transfer **the entire** `brats23_dg_png` directory (it has the `index.csv` plus `.npz` files within `site_*` directories). You can download them to your local machine with Google Drive then copy to EC2, e.g. from your Mac terminal:

```bash
scp -i ~/Downloads/YOUR_AWS_KEY.pem -r \
  ~/Downloads/brats23_dg_png ubuntu@YOUR_EC2_PUBLIC_IP:~/brats-dg/
scp -i ~/Downloads/YOUR_AWS_KEY.pem -r \
  ~/Downloads/artifacts ubuntu@YOUR_EC2_PUBLIC_IP:~/brats-dg/
```

If there are large files, you may instead transfer them via S3 with `aws s3 sync` on EC2 after uploading them to your own bucket. No credentials are embedded in the scripts.

The project folder also contains `reports`, `training_log.txt`, `best_resnet18.pt`, and a site mapping XLSX. They are not needed for ordinary training when NPZ already exists; archive/transfer them separately if you want a complete research copy.

### Fixing paths

**You do not need to manually edit `index.csv`.** `aws_paths.py` reads it and remaps the old Colab paths such as:

```text
/content/drive/MyDrive/18662/project/brats23_dg_png/site_18/ABC.npz
```

to the corresponding paths under `--data-root`, such as:

```text
/home/ubuntu/brats-dg/brats23_dg_png/site_18/ABC.npz
```

It validates *all* NPZ paths before starting, then recreates the original per-case site split via `dg.splits.build_main_split` and saves a split manifest in `artifacts/results/`. The source `index.csv` is never overwritten.

If your data directory is outside the project, pass the absolute root on each command:

```bash
python run_csu_aws.py --dry-run --data-root /mnt/data/brats23_dg_png
```

Or define a default location once in your shell:

```bash
export BRATS_DATA_ROOT=/mnt/data/brats23_dg_png
```

Optionally pass `--index /some/other/location/index.csv`.

## 3. Check files and GPU before running

```bash
cd ~/brats-dg
source .venv/bin/activate
python run_final_report_aws.py --stage data
python run_csu_aws.py --dry-run
python -c 'import torch; print("CUDA available:", torch.cuda.is_available())'
```

For a valid migration, site 18 should be the OOD test site and the case counts should match the original notebook/paper if exactly the same index and seeds are used. The split manifest is `artifacts/results/split_site18_seed18662.csv`.

## 4. Run CSU-1 (the next experiment)

```bash
python -u run_csu_aws.py
```

Defaults: site=18, seed=18662, ResNet18 (train layer2+FC), CSU p=0.75, alpha=0.1, cross-entropy only, max 15 epochs, early-stop patience 3. Trained checkpoint and result files:

```text
artifacts/checkpoints/best_CSU-L2-p0.75-a0.1-seed18662.pth
artifacts/results/CSU-L2-p0.75-a0.1-seed18662.json
artifacts/results/csu_site18_comparison.csv
artifacts/results/csu_vs_existing_site18.csv
```

The comparison CSV includes the original Baseline-L2 and MixStyle-L2 checkpoints **only if transferred and present**.

```bash
python -u run_csu_aws.py --eval-only   # use saved checkpoint only
python -u run_csu_aws.py --multiseed    # 18662, 18663, 18664
python -u run_csu_aws.py --p 0.5 --alpha 0.1
```

By default a matching checkpoint is reused if present; use `--force` to retrain it. No optimizer state is saved by the existing trainer, so resuming mid-epoch is not supported.

## 5. Run original final-report workflow (without notebook)

```bash
python run_final_report_aws.py --stage data
python -u run_final_report_aws.py --stage loso                  # baseline LOSO (8 sites)
python -u run_final_report_aws.py --stage train --group baseline
python -u run_final_report_aws.py --stage train --group mixstyle
python -u run_final_report_aws.py --stage train --group kl
python -u run_final_report_aws.py --stage train --group mmd
python -u run_final_report_aws.py --stage train --group optimal
python -u run_final_report_aws.py --stage report --group all    # all available checkpoints
python -u run_final_report_aws.py --stage consistency          # stochastic flip/KL checks
python -u run_final_report_aws.py --stage mmd-analysis         # feature alignment checks
```

`--stage all` runs train -> report -> consistency -> MMD analysis, but deliberately does **not** re-download, re-preprocess, or rerun expensive LOSO. Running all is lengthy. Each original notebook experiment configuration is represented; duplicate notebook cells and Jupyter display/magic calls were consolidated.

The original notebook also had raw data acquisition/preprocessing. If you ever need this again:

```bash
python -u run_final_report_aws.py --stage preprocess \
  --raw-zip /absolute/path/BraTS2023-GLI-TrainingData.zip \
  --mapping-xlsx /absolute/path/BraTS2023_2017_GLI_Mapping.xlsx
```

If an existing output directory is populated, the original preprocessing function skips work by design; do not overwrite your current preprocessed dataset inadvertently.

**Existing-code limitation:** The uploaded `dg.experiment.run_experiment` raises an error if KL and MMD are enabled simultaneously, and the notebooks don't run a combined KL+MMD experiment. Therefore the combined-loss results described in the PDF are not reproducible using only these uploaded source files. This migration does not invent that missing code.

## 6. Background execution (safe across SSH disconnection)

```bash
cd ~/brats-dg
source .venv/bin/activate
tmux new -s brats
# Now inside tmux:
python -u run_csu_aws.py 2>&1 | tee csu_training.log
```

Detach with `Ctrl+B`, release, then `D`. You can close SSH. Reconnect later:

```bash
tmux attach -t brats
# or view training without attaching
tail -n 80 ~/brats-dg/csu_training.log
```

Training continues **only while the EC2 instance is running**. Do not stop/terminate the machine while training. For the original models use another tmux session or replace the command with `python -u run_final_report_aws.py --stage train --group ...`.

## 7. Output and evaluation caveats

- `artifacts/checkpoints`: original `.pth` model weights
- `artifacts/results`: per-model experiment `.json`, comparison `.csv`, split manifest `.csv`
- No source file or old checkpoint is deleted by these scripts; `--force` can overwrite a matching checkpoint.
- Replicating exact past results additionally depends on PyTorch version/CUDA, source data and split seed.
- Site 18 was already studied while tuning methods: for a paper, add additional held-out sites and source-only hyperparameter selection instead of treating site18 as a fully untouched test set.
- The dataset class caches NPZ slices in CPU RAM. Choose an EC2 instance with adequate **system memory and storage**, not only GPU VRAM.
