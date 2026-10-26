# Google Drive Data Upload Checklist for SignFlow Colab GPU Training

Follow this checklist to prepare and upload your dataset assets from your local machine to Google Drive before running the Colab training notebook.

---

## 1. Google Drive Directory Structure

Create a folder in your Google Drive named `SignFlow_Colab` (or place it at `/MyDrive/SignFlow_Colab`):

```
Google Drive
└── MyDrive/
    └── SignFlow_Colab/
        ├── archives/
        │   ├── asl_code.tar.gz            # Source code, configs, manifests (~3.5 MB)
        │   └── landmarks_wlasl100.tar.gz  # 100-class landmark archive (~350 MB, 2,038 files)
        ├── checkpoints/                   # Automatically populated by Colab during training
        └── outputs/                       # Automatically populated with evaluation reports
```

> [!TIP]
> If you wish to train on the entire 2,000-class dataset later, you can optionally create and upload `landmarks_all.tar.gz` (~3.2 GB, 21,083 files) into the `archives/` folder.

---

## 2. Generating the Upload Archives Locally

We have provided an automated packaging script in the repository: [`scripts/package_for_colab.py`](file:///f:/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/scripts/package_for_colab.py).

### Command to Package Code and WLASL-100 Landmarks:
Open your local terminal in the project directory and run:

```powershell
python scripts/package_for_colab.py --subset wlasl100 --out-dir dist
```

### Verification of Packaged Archives:
After running the command, check your local `dist/` folder:
- **`dist/asl_code.tar.gz`**: Size should be ~3.0 to ~4.0 MB. Contains `src/`, `configs/`, `scripts/`, `data/manifests/`, `pyproject.toml`, and requirements.
- **`dist/landmarks_wlasl100.tar.gz`**: Size should be ~300 to ~400 MB. Contains all **2,038 landmark `.npz` files** corresponding to the 100 WLASL classes across train, validation, and test splits.

---

## 3. Upload Steps to Google Drive

1. Open your web browser and go to [Google Drive](https://drive.google.com).
2. Create a new folder named **`SignFlow_Colab`**.
3. Inside `SignFlow_Colab`, create a subfolder named **`archives`**.
4. Drag and drop `dist/asl_code.tar.gz` and `dist/landmarks_wlasl100.tar.gz` into the `archives` folder.
5. Wait for the upload to finish (typically 1–3 minutes depending on your internet connection).
6. Verify file sizes in Google Drive:
   - `asl_code.tar.gz`: ~3.5 MB
   - `landmarks_wlasl100.tar.gz`: ~350 MB

---

## 4. Preflight Checklist Summary

- [ ] `SignFlow_Colab/archives/asl_code.tar.gz` uploaded and verified.
- [ ] `SignFlow_Colab/archives/landmarks_wlasl100.tar.gz` uploaded and verified.
- [ ] Colab Runtime set to **GPU (T4 or A100)** via **Runtime > Change runtime type**.
- [ ] Google Drive mount permissions granted when prompted by the notebook.
