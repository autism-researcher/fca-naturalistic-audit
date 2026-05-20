# Running the Waymo Experiment on Windows (via WSL2)

Step-by-step setup for `run_waymo_experiment.py` on Windows + WSL2.
The `waymo-open-dataset` package requires Linux and a pinned
TensorFlow version that does not install cleanly on native Windows.

---

## 1. Install WSL2 + Ubuntu (one-time)

PowerShell as Administrator:

```powershell
wsl --install -d Ubuntu-22.04
```

Reboot. After reboot, Ubuntu opens and asks for a Linux username/password.

```powershell
wsl --set-default-version 2
wsl --list --verbose
```

You should see `Ubuntu-22.04` with `VERSION 2`.

---

## 2. Install Miniconda inside WSL

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source ~/miniconda3/bin/activate
conda init bash
```

Close and reopen the Ubuntu terminal.

---

## 3. Create the Waymo conda environment

```bash
conda create -n waymo python=3.10 -y
conda activate waymo
pip install waymo-open-dataset-tf-2-12-0 numpy pandas scipy
python -c "from waymo_open_dataset.protos import scenario_pb2; print('Waymo SDK OK')"
```

---

## 4. Access your project folder from WSL

Windows path:
`C:\Users\AMT\OneDrive\Documents\Claude\Projects\New Paper 3\`

D: drive path:
`D:\New Paper3\paper3_pipeline\`

WSL view of D: drive:
`/mnt/d/New Paper3/paper3_pipeline/`

```bash
cd "/mnt/d/New Paper3/paper3_pipeline"
ls
```

You should see `src/`, `scripts/`, `data/`, etc.

---

## 5. (Optional) Tag the pipeline at Gate 2

```bash
cd "/mnt/d/New Paper3/paper3_pipeline"
git init
git add .
git commit -m "Pipeline frozen for pre-confirmatory Waymo run"
git tag pipeline-frozen-pre-confirmatory
```

Or pass `--skip-gate-check` for the pilot and document in `deviations_log.md`.

---

## 6. Run the pilot, then the full N=5000

```bash
conda activate waymo
cd "/mnt/d/New Paper3/paper3_pipeline"

# Pilot:
python scripts/run_waymo_experiment.py --n 100 --skip-gate-check

# Full run:
python scripts/run_waymo_experiment.py --n 5000 --skip-gate-check
```

If your data is somewhere else, add `--waymo-dir /path/to/shards`.

---

## Troubleshooting

**"Cannot import Waymo SDK"** — `conda activate waymo` first.

**"No TFRecord shards found"** — wrong `--waymo-dir` path. Check with `ls`.

**"Required tag pipeline-frozen-pre-confirmatory not found"** — see Section 5.

**Very slow extraction on /mnt/d/...** — expected. Mounted Windows
drives are slower than native WSL filesystem. The 30-60 GB of data
you have isn't worth re-copying; accept the slower I/O.

**Disconnected terminal** — use nohup to detach:
```bash
nohup python scripts/run_waymo_experiment.py --n 5000 --skip-gate-check > waymo_run.log 2>&1 &
disown
tail -f waymo_run.log
```
