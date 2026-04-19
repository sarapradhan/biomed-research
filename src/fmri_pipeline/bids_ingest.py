from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import pandas as pd
from bids import BIDSLayout


@dataclass
class RunRecord:
    dataset: str
    site: str
    subject: str
    session: Optional[str]
    task: str
    run: Optional[str]
    bold_file: str
    brain_mask_file: str
    confounds_file: str
    tr: float


def _find_first_existing(layout: BIDSLayout, entities: Dict, suffixes: List[str], extensions: List[str]) -> Optional[str]:
    for suffix in suffixes:
        for ext in extensions:
            files = layout.get(**entities, suffix=suffix, extension=ext, return_type="filename")
            if files:
                return files[0]
    return None


def _infer_tr(layout: BIDSLayout, bold_file: str) -> float:
    try:
        metadata = layout.get_metadata(bold_file)
        if "RepetitionTime" in metadata:
            return float(metadata["RepetitionTime"])
    except Exception:
        pass
    return float(nib.load(bold_file).header.get_zooms()[3])


def build_participants_table(cfg: Dict, logger) -> pd.DataFrame:
    """Build participant/covariate table from participants.tsv + optional phenotypic TSV."""
    phenotypic = Path(cfg["paths"]["phenotypic_tsv"])
    rows: List[pd.DataFrame] = []

    for dataset_name, bids_root in cfg["paths"]["bids_roots"].items():
        p = Path(bids_root) / "participants.tsv"
        if p.exists():
            df = pd.read_csv(p, sep="\t")
            df["dataset"] = dataset_name
            rows.append(df)

    if not rows:
        raise FileNotFoundError("No participants.tsv found in configured BIDS roots")

    participants = pd.concat(rows, ignore_index=True)
    participants = participants.rename(columns={"participant_id": "subject_id"})

    if phenotypic.exists():
        pheno = pd.read_csv(phenotypic, sep=None, engine="python")
        if "subject_id" not in pheno.columns:
            if "participant_id" in pheno.columns:
                pheno = pheno.rename(columns={"participant_id": "subject_id"})
            elif "subject" in pheno.columns:
                pheno = pheno.rename(columns={"subject": "subject_id"})
        participants = participants.merge(pheno, on=[c for c in ["subject_id", "dataset"] if c in pheno.columns], how="left")

    if "diagnosis" not in participants.columns:
        logger.warning("No diagnosis column found. Group stats will fail unless diagnosis is provided.")

    return participants


def collect_runs(cfg: Dict, participants_df: pd.DataFrame, logger) -> pd.DataFrame:
    """Collect run-level records from fMRIPrep derivatives using BIDSLayout."""
    records: List[RunRecord] = []
    use_aroma = bool(cfg["bids"].get("use_aroma", True))
    target_space = cfg["bids"]["space"]

    for dataset_name, bids_root in cfg["paths"]["bids_roots"].items():
        derivatives_root = Path(cfg["paths"]["derivatives_root"])
        dataset_deriv = derivatives_root / dataset_name / "fmriprep"
        if not dataset_deriv.exists():
            logger.warning("Missing derivatives for %s at %s", dataset_name, str(dataset_deriv))
            continue

        layout = BIDSLayout(str(bids_root), derivatives=[str(dataset_deriv)], validate=False)
        dataset_task = cfg["bids"]["task_movie"] if dataset_name == "algonauts" else cfg["bids"]["task_rest"]

        bold_files = layout.get(
            scope="derivatives",
            task=dataset_task,
            suffix="bold",
            extension=[".nii", ".nii.gz"],
            space=target_space,
            desc="smoothAROMAnonaggr" if use_aroma else cfg["bids"].get("desc_preproc", "preproc"),
            return_type="filename",
        )

        if not bold_files and use_aroma:
            bold_files = layout.get(
                scope="derivatives",
                task=dataset_task,
                suffix="bold",
                extension=[".nii", ".nii.gz"],
                space=target_space,
                desc=cfg["bids"].get("desc_preproc", "preproc"),
                return_type="filename",
            )

        for bf in bold_files:
            entities = layout.parse_file_entities(bf)
            sub = entities.get("subject")
            ses = entities.get("session")
            run = entities.get("run")
            task = entities.get("task")

            confounds_file = _find_first_existing(
                layout,
                {
                    "scope": "derivatives",
                    "subject": sub,
                    "session": ses,
                    "task": task,
                    "run": run,
                    "desc": "confounds",
                },
                suffixes=["timeseries"],
                extensions=[".tsv"],
            )
            if confounds_file is None:
                confounds_file = _find_first_existing(
                    layout,
                    {
                        "scope": "derivatives",
                        "subject": sub,
                        "session": ses,
                        "task": task,
                        "run": run,
                    },
                    suffixes=["timeseries"],
                    extensions=[".tsv"],
                )

            brain_mask_file = _find_first_existing(
                layout,
                {
                    "scope": "derivatives",
                    "subject": sub,
                    "session": ses,
                    "task": task,
                    "run": run,
                    "space": target_space,
                    "desc": "brain",
                },
                suffixes=["mask"],
                extensions=[".nii", ".nii.gz"],
            )

            if not confounds_file or not brain_mask_file:
                logger.warning("Skipping %s due to missing confounds or mask", bf)
                continue

            site = dataset_name
            participant_row = participants_df[participants_df["subject_id"].astype(str).str.contains(sub)]
            if "site" in participant_row.columns and not participant_row["site"].dropna().empty:
                site = str(participant_row["site"].dropna().iloc[0])

            tr = _infer_tr(layout, bf)
            records.append(
                RunRecord(
                    dataset=dataset_name,
                    site=site,
                    subject=sub,
                    session=str(ses) if ses is not None else None,
                    task=str(task),
                    run=str(run) if run is not None else None,
                    bold_file=bf,
                    brain_mask_file=brain_mask_file,
                    confounds_file=confounds_file,
                    tr=tr,
                )
            )

    runs_df = pd.DataFrame([r.__dict__ for r in records])
    if runs_df.empty:
        raise RuntimeError("No runs found in configured derivatives.")

    tr_counts = runs_df.groupby("dataset")["tr"].nunique()
    for dataset, n_unique in tr_counts.items():
        if n_unique > 1:
            logger.warning("TR heterogeneity detected in %s: %d unique TR values", dataset, n_unique)

    participants_df = participants_df.copy()
    participants_df["subject"] = participants_df["subject_id"].astype(str).str.replace("sub-", "", regex=False)
    merged = runs_df.merge(participants_df, on=["dataset", "subject"], how="left", suffixes=("", "_participant"))
    return merged


def fmriprep_command_template(cfg: Dict) -> str:
    """Return reproducible fMRIPrep + ICA-AROMA docker command template."""
    return (
        "docker run --rm -ti "
        "-v {bids_root}:/data:ro -v {deriv_root}:/out -v {work_dir}:/work "
        "nipreps/fmriprep:latest /data /out participant "
        "--use-aroma --output-spaces {space} --fs-no-reconall "
        "--nthreads {nthreads} --omp-nthreads {omp_threads} --stop-on-first-crash -w /work"
    )
