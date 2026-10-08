# TSA digitisation pipeline: HPC deployment

Everything needed to run the Tables des Successions (TSA) OCR pipeline on a
fresh HPC account, one French departement at a time, with automatic storage
rotation so that limited scratch space is never overrun.

All of the scripts described here live in `scripts/hpc/`. They are designed to
be run **from the repository root** (the pipeline resolves model paths and
writes its log relative to the current working directory). The scripts `cd` to
the repo root themselves, so you can invoke them from anywhere.

---

## 1. Quickstart

```bash
# 1. Clone and enter the repo
git clone https://github.com/aureliusnoble/TSA.git
cd TSA
git checkout hpc            # until this branch is merged

# 2. Create and verify the conda environment
bash scripts/hpc/setup_env.sh

# 3. Put AWS credentials in place (the zips live in a private S3 bucket).
#    Either export them for this shell session:
export AWS_ACCESS_KEY_ID=...        # from the "keys" sheet of TSA_download_links_S3.xlsx
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=eu-west-2
#    ...or run `aws configure` to write ~/.aws/credentials.
#    NEVER commit these. No script in this repo contains them.

# 4. Process one departement
sbatch scripts/hpc/run_department.sbatch Paris      # on a SLURM cluster
# or, without SLURM:
bash   scripts/hpc/run_department.sh   Paris

# 5. Collect results: one compact archive per departement
ls results/                 # Paris.tar.gz, Nievre.tar.gz, ...
# download results/*.tar.gz to your machine (scp/rsync/sftp)
```

Each archive contains the pipeline's CSV tables (one CSV per page) plus a
`manifest.txt`. Departement names are accent-tolerant: `Nievre`, `Nièvre` and
`NIEVRE` all resolve to the same job.

---

## 2. Model weights (READ THIS before your first GPU job)

The production config `configs/recommended_grid_v2.yaml` references four models:

| role            | path                                 | in git on branch `hpc`? |
|-----------------|--------------------------------------|-------------------------|
| row extraction  | `models/rows_v7/model.pth`           | yes (~47 MB)            |
| column extract. | `models/cols_b/model.pth`            | yes (~47 MB)            |
| text-line       | `models/textlines_full/model.pth`    | **NO (~49 MB)**         |
| transcription   | `models/transcription_full/` (TrOCR) | **NO (~2.2 GB)**        |

Only `rows_v7` and `cols_b` are committed. The text-line and transcription
weights are excluded by the repo's `.gitignore` (`models/*`), so **a fresh
clone will be missing them** and inference will stop immediately with
`Model path ... does not exist`.

`setup_env.sh` checks all four paths and prints `OK` / `MISSING` for each, so
you will know before submitting a GPU job. Obtaining the two large models on
the cluster is an open item for the co-author (see section 9). Note: the
repo's `download_models.py` pulls a Google-Drive bundle and **replaces the
entire `models/` directory**, which would delete the committed `rows_v7` /
`cols_b` weights. Do not run it blindly.

---

## 3. How storage rotation works

Scratch is assumed to be limited, so `run_department.sh` keeps at most one
departement's images on disk at a time:

1. **Skip** if `results/<slug>.tar.gz` already exists.
2. **Download** the departement zip to `$TSA_SCRATCH/<slug>/` (via `aws s3 cp`,
   falling back to `curl -C -`).
3. **Unzip** into `$TSA_SCRATCH/<slug>/images/`.
4. **Write** a per-departement config with `make_config.py`.
5. **Run inference** (resumable, see section 4).
6. **Archive** only the CSV tables plus a manifest to `results/<slug>.tar.gz`.
7. **Optionally upload** the archive to S3 (`$TSA_RESULTS_S3`, skipped if unset).
8. **Delete** the raw zip and the extracted images, keeping only the archive.

Because the images are deleted in step 8 before the next departement is
downloaded, the sequential runner (`run_all.sh`) never holds more than one
departement's raw data at once.

### Disk high-water mark

The peak disk use for a departement is roughly `zip + extracted images`. The
zip and the extracted JPEGs are each about the "Size (GB)" figure in
`departments.csv` (JPEGs barely compress), so plan for **~2x the departement
size** in free scratch, plus a few MB for the CSV output. Largest first:

| departement      | size (GB) | pages  | peak scratch (GB) |
|------------------|-----------|--------|-------------------|
| Nord             | 58.1      | 39,914 | ~116              |
| Loire            | 56.7      | 17,060 | ~113              |
| Nièvre           | 33.6      | 24,527 | ~67               |
| Bouches-du-Rhone | 23.3      | 15,359 | ~47               |
| Saone-et-Loire   | 18.9      | 47,532 | ~38               |
| Deux-Sèvres      | 17.2      |  8,596 | ~34               |
| Creuse           | 17.1      | 10,403 | ~34               |
| Manche           | 12.0      | 25,799 | ~24               |
| Paris            |  9.0      |  8,812 | ~18               |

Full figures for all 20 departements are in `departments.csv`. If your scratch
quota is below ~120 GB, you can still run every departement **sequentially**
(the default): only one is on disk at a time. Running several SLURM jobs in
parallel multiplies the requirement by the number of concurrent jobs.

Point `TSA_SCRATCH` at the cluster's large scratch filesystem, e.g.:

```bash
export TSA_SCRATCH=/scratch/$USER/tsa
```

---

## 4. Resume semantics (safe to re-run)

Every step is idempotent, so a killed or timed-out job can simply be
resubmitted:

- **Completed departement** -> step 1 sees `results/<slug>.tar.gz` and exits
  immediately. Delete that archive to force a full re-run.
- **Download** -> `aws s3 cp` re-fetches atomically; the `curl` fallback uses
  `-C -` to resume a partial file. A marker file skips re-download once the zip
  is present and a second marker skips unzip once extraction finished.
- **Inference** -> resumes **per page**. `inference.py` scans the output
  directory (`output/pages/<commune>/<period>/*.csv`) via
  `Pipeline._get_processed_files()` and skips any input image whose CSV already
  exists (`_get_pending_images()`). So re-running continues exactly where it
  left off; already-transcribed pages are not redone. Pages that errored out
  previously (no CSV written) are retried.
- **Archive** -> re-tarring overwrites `results/<slug>.tar.gz` (written to a
  `.partial` file and moved into place, so an interrupted tar never looks
  complete).

---

## 5. Configuration

`make_config.py` derives a per-departement config from the single production
base config `configs/recommended_grid_v2.yaml`, overriding **only**
`directories.input`, `directories.output` and `directories.temp`. Model paths,
grid method (`regularised`), line split (`cells`), `line_input_size: 1500` and
the row/column normalisation stats are inherited unchanged.

```bash
python scripts/hpc/make_config.py \
  --department Paris \
  --input-dir  /scratch/$USER/tsa/Paris/images \
  --output-dir /scratch/$USER/tsa/Paris/output \
  --out        configs/generated/Paris.yaml
```

`run_department.sh` calls this for you; the generated YAMLs land in
`configs/generated/` (git-ignored). Model and layout paths stay repo-relative,
which is why inference must run from the repo root.

---

## 6. Running

```bash
# one departement, SLURM
sbatch scripts/hpc/run_department.sbatch Nievre

# one departement, no SLURM
bash scripts/hpc/run_department.sh Nievre

# every downloadable departement, sequentially, in priority order (no SLURM)
bash scripts/hpc/run_all.sh
```

`run_all.sh` iterates `departments.csv` in digitisation-priority order (Paris
first, then the fully-intact departements largest-first, then the incomplete
ones), skips any departement without an S3 zip, and does not abort the batch if
one departement fails. It also contains commented SLURM recipes for submitting
the whole set as independent jobs or as a strictly-sequential dependency chain.

> **SLURM resources are placeholders.** `run_department.sbatch` has `--partition`,
> `--gres`, `--mem` and `--time` set to guesses with loud comments. The target
> cluster's specifics are unknown; check `sinfo` / your cluster docs and edit
> them before relying on the job.

### Environment variables

| variable          | default           | meaning                                            |
|-------------------|-------------------|----------------------------------------------------|
| `TSA_SCRATCH`     | `<repo>/scratch`  | scratch root for downloads + extracted images      |
| `TSA_RESULTS_DIR` | `<repo>/results`  | where result archives are written                  |
| `TSA_RESULTS_S3`  | (unset)           | `s3://bucket/prefix` to also upload each archive to |
| `TSA_AWS_REGION`  | `eu-west-2`       | AWS region for the bucket                          |
| `TSA_ENV_NAME`    | `TSA`             | conda env name                                     |
| `TSA_KEEP_IMAGES` | `0`               | set `1` to keep the zip/images after archiving (debug) |
| AWS creds         | (your env/`~/.aws`) | read by the AWS CLI; never stored in-repo        |

---

## 7. Output

Each `results/<slug>.tar.gz` contains:

```
pages/<commune>/<period>/<page>.csv    # one CSV table per processed page
manifest.txt                           # departement, pages expected vs produced,
                                       # repo commit, source zip, timestamp
```

`pages/` mirrors the folder structure inside the source zip
(`Departement/Commune/Period/page.jpeg`; Paris has no commune level). The
manifest's `pages_expected` comes from the "Pages ready" column of
`departments.csv`; a mismatch with `csv_tables_produced` is flagged (it usually
means some pages failed or the zip was an incomplete copy).

---

## 8. Troubleshooting

- **GPU out of memory.** Lower the transcription batch size. In
  `configs/recommended_grid_v2.yaml` (or the generated per-departement YAML)
  reduce `models.transcription.batch_size` from `16` down (try 8, then 4).
  Separately, the text-line model runs at its trained input size
  `line_input_size: 1500`; this needs roughly **4x the GPU/CPU memory of the
  old 768px setting**. Do not lower it for throughput (it halves line-detection
  quality), but if 1500 will not fit at all, that is the first knob to discuss
  with the co-author. `transcription.precision: half` is already set.
- **`libGL.so.1: cannot open shared object file`.** The env pins
  `opencv-python-headless`, which has no libGL dependency, specifically to
  avoid this on display-less compute nodes. If you see it, you likely have a
  non-headless `opencv-python` installed; `pip uninstall opencv-python` inside
  the TSA env.
- **`conda: command not found`.** Install Miniforge/Miniconda, or
  `module load anaconda` (name varies by cluster), then re-run `setup_env.sh`.
- **`doc-ufcn` import error about `pkg_resources`.** Caused by setuptools >= 81;
  the env pins `setuptools<81`.
- **S3 `AccessDenied` / 403.** The bucket is private; make sure the AWS
  credentials (from the xlsx "keys" sheet) are exported or in `~/.aws`. `curl`
  of the https URL alone will not work for a private object.
- **Inference seems to redo pages.** It keys resume on the output CSV path
  `output/pages/<commune>/<period>/<page>.csv`. If you move or clear the
  per-departement `output/` directory, resume state is lost.

---

## 9. Ask co-author

Open questions that need the project owner's input:

- **Large model weights.** `textlines_full/model.pth` (~49 MB) and
  `transcription_full/` (~2.2 GB) are not in git. How should they reach the
  cluster: add to git/LFS, a separate S3 object, or a fixed
  `download_models.py` that does not clobber `rows_v7`/`cols_b`? (See section 2.)
- **S3 results upload.** Is there write access to the `tsatransferaurelius`
  bucket (or another) for uploading result archives via `TSA_RESULTS_S3`? What
  prefix should be used?
- **Credential / URL expiry.** Are the keys in the xlsx long-lived, or do the
  object URLs expire (presigned)? If presigned, `department_urls.csv` will need
  refreshing.
- **Bucket lifecycle.** Is there a lifecycle/expiry policy on the source zips we
  should be aware of before a long sequential run?
- **Departements not yet uploaded.** Priority-3 departements have missing pages
  in the current transfer ("await Dropbox upload" in the metadata). Confirm
  whether the S3 zips are the truncated copies or the complete re-uploads before
  treating their output as final.

### Metadata vs S3 discrepancies

Derived from `Transfer_Aurelius_metadata.xlsx` vs `TSA_download_links_S3.xlsx`:

- **In metadata but NOT on S3:** `Seine-Saint-Denis` (23 volumes, 248 pages
  present, 2,983 missing). The metadata notes it has **no zip archive**. It is
  listed in `departments.csv` with `in_s3_links=no` and is skipped by
  `run_all.sh`.
- **In S3 but NOT in metadata:** none (all 19 S3 zips match a metadata
  departement).
- **Name / accent mismatches** (handled automatically, but note for anyone
  editing the CSVs by hand):
  - `Deux-Sèvres` (links sheet) vs `Deux-Sevres` (metadata); zip is
    `Deux-Sevres.zip`.
  - `Nièvre` (both sheets) is served as `Nievre.zip` (de-accented object key).
  The result archives use the de-accented zip stem as their name
  (`Deux-Sevres.tar.gz`, `Nievre.tar.gz`).
- `Eure-et-Loire` is spelled with a trailing "e" in both sources (the actual
  departement is *Eure-et-Loir*). Kept as-is to match the S3 object name.

---

## 10. Files in this directory

| file                   | purpose                                                     |
|------------------------|-------------------------------------------------------------|
| `setup_env.sh`         | create + verify the `TSA` conda env; check model weights    |
| `environment.yaml`     | complete conda spec (superset of the root `environment.yaml`) |
| `make_config.py`       | write a per-departement config from the base config         |
| `run_department.sh`    | the rotation worker for one departement                     |
| `run_department.sbatch`| SLURM wrapper (placeholder resources, edit before use)      |
| `run_all.sh`           | sequential runner over all departements + SLURM recipes     |
| `department_urls.csv`  | departement -> S3 zip URL (19 rows; no credentials)         |
| `departments.csv`      | departement metadata joined to S3 links (20 rows)           |
| `README_HPC.md`        | this file                                                   |
