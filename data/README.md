# Dataset Directory

This folder stores dataset manifests, metadata, and derived split files for the sign-language recognition pipeline.

## Intended structure

- `data/raw/`: raw video clips and associated metadata
- `data/processed/`: preprocessed frames, landmarks, or extracted features
- `data/manifests/`: JSON or CSV manifest files describing dataset samples

## Manifest contract

Each manifest record should contain:

- `sample_id`: unique sample identifier
- `signer_id`: signer identity used for signer-aware evaluation
- `class_id`: integer class index
- `class_name`: label name (for example, `hello` or `thank_you`)
- `video_path`: relative or absolute path to the video or sample file
- `num_frames`: number of frames in the sample
- `fps`: frame rate
- `duration`: sample duration in seconds

The splits should be signer-independent. A signer appearing in training must not appear in validation or test data.
