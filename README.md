# Dataset Selection and Preprocessing
## Backup datasets (Alzheimer MRI datasets)

As backup options, we considered several MRI datasets for Alzheimer’s disease research, including [OASIS](https://www.kaggle.com/datasets/pulavendranselvaraj/oasis-dataset), [ADNI](https://adni.loni.usc.edu/data-samples/adni-data/neuroimaging/mri/mri-image-data-sets/) dataset – large Alzheimer’s imaging dataset (requires permission), and the [Alzheimer’s 4-class MRI](https://www.kaggle.com/datasets/preetpalsingh25/alzheimers-dataset-4-class-of-images) dataset. These datasets could potentially be combined to study domain shift by training on one dataset (e.g., ADNI) and evaluating on another (e.g., OASIS), since they were collected at different research institutions using different imaging protocols.

However, access restrictions (e.g., ADNI requires permission) and limited metadata about acquisition sites make them less convenient for controlled domain generalization experiments.


## Selected Dataset: BraTS
The [**Brain Tumor Segmentation (BraTS)**](https://www.synapse.org/Synapse:syn51514105) dataset contains multi-institution MRI scans of brain tumor patients.

One key advantage of BraTS is that each sample includes metadata describing the originating institution (collection site). This allows the dataset to be separated into subsets by hospital site, which naturally simulates domain shifts.

This property makes BraTS particularly suitable for domain generalization experiments, where models are trained on data from several hospitals and evaluated on a previously unseen hospital.

Our experimental design follows:

- Training domains: MRI scans from multiple hospitals

- Validation: held-out cases from the training domains

- Test domain: MRI scans from a completely unseen hospital

This setup allows us to study how well models generalize to new clinical environments.

# Data Format and Preprocessing
BraTS MRI scans are stored in NIfTI (`.nii.gz`) format, a common format for medical imaging that stores 3D volumetric data. Each MRI scan contains multiple axial slices of the brain.

Since most convolutional neural networks are designed for 2D images, we need to convert each MRI volume into individual 2D slices. Using the provided tumor segmentation masks, each slice is labeled as tumor or non-tumor, forming a binary classification dataset.

The processed slices are resized to 224×224 and stored as compressed .npz files (NumPy archives) on Google Drive. This format reduces storage overhead and allows faster loading during training.

## Preprocessing Pipeline
### Step 1: Read MRI volumes

The BraTS dataset is stored as compressed NIfTI files (.nii.gz).
Each file is loaded using the Nibabel library.

Example modalities:

```
BraTS-GLI-XXXX-t1n.nii.gz
BraTS-GLI-XXXX-t1c.nii.gz
BraTS-GLI-XXXX-t2w.nii.gz
BraTS-GLI-XXXX-t2f.nii.gz
BraTS-GLI-XXXX-seg.nii.gz
```

### Step 2: Convert 3D MRI to 2D slices

Each MRI volume is sliced along the axial axis:

```
240 × 240 × 155 → 155 slices
```

Each slice becomes a 2D image.

### Step 3: Assign slice labels

Using the segmentation mask:

```
segmentation > 0 → tumor slice
segmentation = 0 → normal slice
```

This produces labeled samples for classification.

### Step 4: Resize images

Each slice is resized to:

```
224 × 224
```

This resolution matches the expected input size for ResNet architectures.

### Step 5: Save processed dataset

The processed dataset is saved to Google Drive to avoid recomputing preprocessing.

Instead of saving thousands of PNG images, the slices are stored using the `.npz` format.

`.npz` is a compressed NumPy archive that stores multiple arrays in a single file.

Example contents of one .npz file:
```
images → array of MRI slices
labels → tumor / normal labels
site → hospital ID
case_id → patient identifier
```
We are saving the preprocessed data in this format since its an efficient compression and faster for loading compared to many small image files (`.png`). This format allows the dataset to be easily loaded during training while keeping the storage footprint manageable

# Model Backbone
We use pretrained `ResNet-18` as the baseline architecture. ResNet-18 is widely used in medical imaging due to its stable training behavior and moderate computational cost. Using pretrained ImageNet weights allows the model to leverage transfer learning, improving training efficiency when working with limited medical datasets.

This backbone is used both for the baseline model and for evaluating domain generalization methods such as MixStyle.

## Baseline for Domain Generalization Methods
ResNet-18 serves as a baseline architecture for evaluating domain generalization methods.

In our mid-semester experiments, we compare:

```
Baseline model:
ResNet18 trained with empirical risk minimization (ERM)

Domain generalization model:
ResNet18 + MixStyle
```

This comparison allows us to evaluate whether domain generalization techniques improve performance on unseen hospital domains.