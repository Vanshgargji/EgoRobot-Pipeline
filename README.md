# Labellerr-Aligned Egocentric Robotics Data Factory

Turn a raw first-person utensil video into segmented, audited, VLA-ready robotics training data.

This project was built for the Robotics Data Hackathon and shaped specifically around Labellerr's strengths: AI-assisted labeling, video annotation, dataset automation, quality validation, and feedback loops.

## Quick Review

If you are reviewing this project for the first time, open these files in order:

| Step | Open | Why it matters |
|---|---|---|
| 1 | `Labellerr_outputs_bartan1/phase7_output/PHASE7_MASTER_DASHBOARD.mp4` | Fastest visual proof of the full pipeline |
| 2 | `Labellerr_outputs_bartan1/phase6_output/phase6_dashboard.png` | Shows dataset quality and validation result |
| 3 | `Labellerr_outputs_bartan1/phase3_output/phase3_timeline_plot.png` | Shows detected action windows across the video |
| 4 | `Labellerr_outputs_bartan1/phase45_output/vla_ready_dataset.json` | Shows structured robotics-ready records |
| 5 | `Labellerr_outputs_bartan1/phase6_output/final_robot_dataset.jsonl` | Final validated dataset export |

## Project Snapshot

| Item | Result |
|---|---:|
| Input video | `bartan1.MP4` |
| Source duration | `600 seconds` |
| Source resolution | `1920x1080` |
| Source FPS | `30` |
| Analysis FPS | `2` |
| Processed frames | `1200` |
| Detected interaction events | `12` |
| Exported segment videos | `8` |
| VLA-ready records | `12` |
| Clean validation records | `12` |
| Flagged records | `0` |
| Demo video duration | `100 seconds` |

Full metrics are available in:

```text
Labellerr_outputs_bartan1/run_summary.json
```

## What We Built

We built a local egocentric video data pipeline that takes raw first-person footage and converts it into a structured dataset for robotics and embodied AI.

The pipeline performs:

1. Video ingestion and preprocessing
2. Frame-level hand/object motion feature extraction
3. Temporal action segmentation
4. Contact and separation event detection
5. VLA-style language/action record generation
6. Dataset validation and audit metadata creation
7. A director-cut demo video for easy inspection

The original notebook, `Labellerr.ipynb`, was Colab-oriented and referenced helper scripts that were not present as standalone files. To make the work reproducible locally, the same phase structure has been implemented in:

```text
run_labellerr_pipeline.py
```

## Why This Fits Labellerr

Labellerr helps teams create high-quality labeled data faster. This project applies that same idea to robotics video data.

| Labellerr area | How this project maps to it |
|---|---|
| Video annotation | Converts long egocentric footage into useful action clips |
| AI-assisted labeling | Generates frame-level motion and interaction labels automatically |
| Smart Feedback Loop | Adds audit metadata for review-ready records |
| Dataset automation | Exports structured JSON and JSONL files |
| Computer vision | Uses visual motion signals to detect hand/object activity |
| Robotics data | Produces VLA-style records for embodied AI workflows |

Positioning statement:

> A Labellerr-style Smart Feedback Loop for robotics data: raw egocentric videos go in, segmented action clips and validated VLA-ready JSONL records come out.

## Output Map

All generated outputs are saved inside:

```text
Labellerr_outputs_bartan1/
```

| Folder | Purpose |
|---|---|
| `phase1_output/` | Clean processed frames and preview video |
| `phase2_output/` | Frame-level features, motion plots, visual annotations |
| `phase3_output/` | Action segments, contact/separation events, timeline |
| `phase45_output/` | VLA-ready action records and per-clip JSON |
| `phase6_output/` | Final audited dataset and validation dashboard |
| `phase7_output/` | Final demo dashboard video |

## Phase Walkthrough

### Phase 1: Ingestion And Preprocessing

The raw video is sampled, resized, contrast-normalized, sharpened, and stored as a clean frame stream. This creates a consistent input layer for annotation and segmentation.

| Open this | Purpose |
|---|---|
| `Labellerr_outputs_bartan1/phase1_output/processed_preview.mp4` | Preview the cleaned video stream |
| `Labellerr_outputs_bartan1/phase1_output/processed_frames/` | Inspect extracted processed frames |
| `Labellerr_outputs_bartan1/phase1_output/phase1_meta.json` | View preprocessing metadata |

What to notice: the raw 10-minute video becomes a standardized frame dataset that later phases can process reliably.

### Phase 2: Zero-Shot Feature Extraction

The pipeline estimates hand/object activity using motion saliency. This avoids requiring a task-specific trained model and creates automatic frame-level labels.

| Open this | Purpose |
|---|---|
| `Labellerr_outputs_bartan1/phase2_output/velocity_plot.png` | Visualize motion intensity over time |
| `Labellerr_outputs_bartan1/phase2_output/phase2_features.jsonl` | See frame-level feature records |
| `Labellerr_outputs_bartan1/phase2_output/velocity_profiles/velocity_profile.csv` | Inspect velocity data |
| `Labellerr_outputs_bartan1/phase2_output/visualisations/` | View annotated sample frames |
| `Labellerr_outputs_bartan1/phase2_output/phase2_meta.json` | View phase metrics |

What to notice: this phase turns plain video frames into machine-readable annotation signals.

### Phase 3: Temporal Interaction Localization

The interaction signals are smoothed and segmented into useful action windows. Each window receives contact and separation timestamps.

| Open this | Purpose |
|---|---|
| `Labellerr_outputs_bartan1/phase3_output/phase3_timeline_plot.png` | See detected interaction windows |
| `Labellerr_outputs_bartan1/phase3_output/phase3_events.jsonl` | Inspect start/end event records |
| `Labellerr_outputs_bartan1/phase3_output/phase3_timeline.jsonl` | Inspect frame-by-frame timeline labels |
| `Labellerr_outputs_bartan1/phase3_output/segments/` | Watch extracted action clips |
| `Labellerr_outputs_bartan1/phase3_output/visualisations/` | View phase visual examples |
| `Labellerr_outputs_bartan1/phase3_output/phase3_meta.json` | View phase metrics |

What to notice: the system finds which parts of the long video are useful for robotics training.

### Phase 4+5: VLA Alignment

Detected action clips are converted into VLA-style records with language instructions, normalized hand motion, contact events, and embodiment metadata.

| Open this | Purpose |
|---|---|
| `Labellerr_outputs_bartan1/phase45_output/vla_ready_dataset.json` | Human-readable VLA dataset |
| `Labellerr_outputs_bartan1/phase45_output/phase45_vla.jsonl` | JSONL records for downstream training |
| `Labellerr_outputs_bartan1/phase45_output/clips/` | Per-clip annotation JSON files |
| `Labellerr_outputs_bartan1/phase45_output/phase45_timeline_plot.png` | VLA action timeline |
| `Labellerr_outputs_bartan1/phase45_output/phase45_meta.json` | View phase metrics |

Example record shape:

```json
{
  "video_id": "bartan1",
  "language_instruction": "Track the hand as it moves a utensil or container through the workspace.",
  "coordinate_space": "unit_cube_normalized",
  "embodiment": "Human-to-Humanoid-Unified"
}
```

What to notice: raw video has become structured robotics data with language, action, contact, and quality fields.

### Phase 6: Quality Validation

Every exported record is audited for simple schema and physics sanity. This creates review-ready metadata for a Labellerr-style feedback workflow.

| Open this | Purpose |
|---|---|
| `Labellerr_outputs_bartan1/phase6_output/phase6_dashboard.png` | Visual quality summary |
| `Labellerr_outputs_bartan1/phase6_output/final_robot_dataset.jsonl` | Final validated dataset |
| `Labellerr_outputs_bartan1/phase6_output/validation_report.json` | Validation report |
| `Labellerr_outputs_bartan1/phase6_output/phase6_meta.json` | View phase metrics |

What to notice: the pipeline does not only generate labels; it also checks whether records are clean enough to use.

### Phase 7: Director-Cut Demo

The final demo overlays the egocentric video with interaction state, progress score, velocity profile, and detected action windows.

| Open this | Purpose |
|---|---|
| `Labellerr_outputs_bartan1/phase7_output/PHASE7_MASTER_DASHBOARD.mp4` | Watch the full visual demo |
| `Labellerr_outputs_bartan1/phase7_output/phase7_meta.json` | View demo metadata |

What to notice: this is the fastest way to understand the complete pipeline without reading the raw JSON files first.

## Repository Files

| File | Purpose |
|---|---|
| `README.md` | Project explanation and interviewer walkthrough |
| `run_labellerr_pipeline.py` | Reproducible local pipeline runner |
| `Labellerr.ipynb` | Original notebook from the hackathon work |
| Robotics Data Hackathon problem statement PDF | Hackathon problem statement |
| `Venture Lab Hackathon.docx` | Notes and planning collected during the hackathon |
| `Labellerr AI.docx` | Company research and Labellerr alignment notes |
| `Labellerr_outputs_bartan1/` | Generated output artifacts |

Note: `bartan1.MP4` is ignored by git because it is larger than GitHub's normal file-size limit. Keep it in Drive, Git LFS, or another artifact store when reproducing the run.

## How To Reproduce

From this folder:

```powershell
python run_labellerr_pipeline.py --input bartan1.MP4 --output Labellerr_outputs_bartan1 --analysis-fps 2 --size 448
```

For a denser analysis run:

```powershell
python run_labellerr_pipeline.py --input bartan1.MP4 --output Labellerr_outputs_bartan1_highfps --analysis-fps 5 --size 448
```

## Internship-Relevant Skills Demonstrated

- Computer vision preprocessing
- Egocentric video analysis
- Automated annotation signal generation
- Action segmentation
- Dataset structuring with JSON and JSONL
- Quality validation and audit metadata
- Human review workflow thinking
- Robotics and embodied AI data preparation

## Next Improvements

With Labellerr mentorship or infrastructure, this can be extended further:

- Replace motion proxies with model-assisted hand pose and object segmentation.
- Add LabelGPT-style prompt-based object and action labels.
- Add a human review UI for low-confidence clips.
- Store accepted and rejected corrections as a feedback loop.
- Export to additional robotics formats such as EgoHumanoid-compatible schemas.
