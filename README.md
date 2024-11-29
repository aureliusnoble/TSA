# TSA Processing Pipeline

A comprehensive computer vision pipeline for the extraction and digitization of historical tabular data from French inheritance records (*Tables des Successions et Absences*). This system implements a multi-stage deep learning approach combining document layout analysis, table structure detection, and handwritten text recognition (HTR).

![Example Image](example.png)

## Table of Contents

1. [Technical Overview](#technical-overview)
2. [System Requirements](#system-requirements)
3. [Installation and Setup](#installation-and-setup)
    - [1. Repository Clone](#1-repository-clone)
    - [2. Environment Configuration](#2-environment-configuration)
    - [3. Model Download](#3-model-download)
4. [Data Organization](#data-organization)
    - [Input Directory Structure](#input-directory-structure)
    - [Classification Schema](#classification-schema)
5. [Pipeline Configuration](#pipeline-configuration)
6. [Running Inference](#running-inference)
    - [HPC Inference](#hpc-inference)
7. [Output Structure](#output-structure)
8. [Training](#training)
9. [Future Development](#future-development)
10. [Citation](#citation)

## Technical Overview

The TSA Processing Pipeline follows a sequential processing architecture with the following stages:

1. **Document Layout Analysis**
   - *Image Preprocessing & Binarization*: Preprocesses the image for better recognition results.
   - *Text Line Detection*: Identifies lines of text within the document using the DocUFCN model.
   - *Spatial Relationship Modeling*: Determines the structural components of the document, such as headers and footers.

2. **Table Structure Recognition**
   - *Row/Column Detection*: Utilises specialized DocUFCN models to detect table rows and columns.
   - *Grid Cell Extraction*: Extracts individual cells by analyzing polygon intersections.
   - *Cell Classification*: Categorises cells based on their spatial positions and predefined layout rules.

3. **Handwritten Text Recognition (HTR)**
   - *TrOCR-based Transformer Model*: Transcribes handwritten text within each table cell.
   - *Post-processing*: Applies domain-specific rules to refine transcribed text.

4. **Data Reconstruction**
   - *Table Reassembly*: Combines cell-level predictions to reconstruct complete table structures.
   - *Data Validation*: Ensures that the reconstructed data adheres to predefined schemas.
   - *Structured Output Generation*: Produces CSV files containing the digitized and structured data.



## System Requirements

### Hardware Prerequisites
- **GPU:** Any CUDA-capable GPU with a minimum of 8GB VRAM
- **System RAM:** 8GB or more.
- **Storage:** At least 10GB of free space for model weights and intermediate outputs.
- **OS:**: Linux/Windows. CUDA (the GPU library used in this project) is not available on MacOS.

### Software Dependencies
- **Conda:** A package and environment management system. [Download Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda is recommended for a lightweight installation).
- **CUDA 11.x+:** Required for GPU acceleration. Ensure compatibility with your GPU.
- **cuDNN:** Compatible with your CUDA version for deep learning operations.

## Installation and Setup

### 1. **Repository Clone**

To begin, you need to clone the TSA repository to your local machine. This involves copying the project files from GitHub to your computer.

**For All Operating Systems (Ubuntu, Windows):**

1. **Open Your Terminal or Command Prompt:**
   - **Ubuntu/Linux:** Press `Ctrl + Alt + T`.
   - **Windows:** Search for "Command Prompt" or "PowerShell" in the Start menu.

2. **Execute the Clone Command:**
   ```bash
   git clone https://github.com/aureliusnoble/TSA
   cd TSA
   ```

### 2. Environment Configuration

The project utilizes Conda to manage its dependencies, ensuring a consistent and isolated environment.

**Install Conda:**
- If you haven't installed Conda yet, download and install [Miniconda](https://docs.anaconda.com/miniconda/install/) suitable for your operating system.

**Create and Activate the Conda Environment:**
```bash
# Create the environment using the provided YAML file
conda env create -f environment.yaml

# Activate the newly created environment
conda activate TSA
```

**Verify CUDA Availability:**
```python
python -c "import torch; print(torch.cuda.is_available())"
```

This command checks if CUDA (GPU acceleration) is available. It should return True if your GPU is correctly set up.

**Troubleshooting:**
- If it returns False, ensure that CUDA and cuDNN are correctly installed and compatible with your GPU.
- Check your GPU drivers and consider reinstalling CUDA if necessary.

### 3. Model Download

The pipeline relies on pre-trained models for various stages. These models are not included in the repository, but can be downloaded by running.

**Download Pre-trained Models:**
```bash
python download_models.py
```

**Expected Model Structure:**
```
models/
├── transcription_nievre/          # TrOCR weights and configuration files
├── textlines_nievre/model.pth     # Line detection model weights
├── rows_nievre/model.pth          # Row detection model weights
└── columns_nievre/model.pth       # Column detection model weights
```

## Data Organization

### Input Directory Structure

Organize your input images hierarchically by Municipality and Year Range as follows:

```
inputs/
├── {Municipality}/
│   ├── {Year_Range}/
│   │   ├── {Municipality}_{Year_Range}_{Register_ID}_page{N}.jpg
│   │   └── ...
│   └── ...
└── ...
```

Example:
```
inputs/
├── Château-Chinon/
│   ├── 1857-1863/
│   │   ├── Château-Chinon_1857-1863_3Q5_699_page3.jpg
│   │   └── Château-Chinon_1857-1863_3Q5_699_page20.jpg
│   └── 1863-1869/
└── Brinon-sur-Beuvron/
    └── 1811-1819/
```

### Classification Schema

The pipeline uses two configuration files to interpret table structures within the documents. There are examples of these in `layout_classification/`

**Page Classification**
- Location: `layout_classification/page_classifications.csv`
- Purpose: Maps each document page to its corresponding table structure type.

Example:
```csv
File,Relative Path,Type
"Château-Chinon_1857-1863_3Q5_699_page3.jpg","Château-Chinon/1857-1863/Château-Chinon_1857-1863_3Q5_699_page3.jpg","2Ab"
"Brinon-sur-Beuvron_1811-1819_3Q2_407_page12.jpg","Brinon-sur-Beuvron/1811-1819/Brinon-sur-Beuvron_1811-1819_3Q2_407_page12.jpg","1A"
```

There is a full classification of the files from Nievre in `layout_classifications/page_classification_nievre.csv`.

**Table Layout Schema**
- Location: `layout_classification/table_layouts.csv`
- Purpose: Defines the expected column structures for each table type.

An example set of table formats is provided for the department of Nievre.

Example:
```csv
table_type,col_num,col_name,data_type
"1A",1,"Article Number","int"
"1A",2,"Surname","str"
"1A",3,"First Name","str"
...
"2Ab",1,"Article Number","int"
"2Ab",2,"Surname","str"
```

Flexibility: While the README mentions specific table types, the pipeline is designed to handle various layouts. Users can define new table structures by updating the table_layouts.csv file, ensuring adaptability to different document formats. All they have to do is define the expected columns in that format.

## Pipeline Configuration

Before running the pipeline, you need to specify model paths and processing parameters in a YAML configuration file.

**Example config.yaml:**
```yaml
models:
  # Path to transcription model directory (used for processor, tokenizer, and model)
  transcription: "models/transcription_nievre"

  # Paths to various model weights
  line_extraction: "models/textlines_nievre/model.pth"
  row_extraction: "models/rows_nievre/model.pth"
  column_extraction: "models/columns_nievre/model.pth"

directories:
  # Base directories for input/output
  input: "examples/inputs"
  output: "examples/outputs"

  # Configuration files
  page_classification: "classification/page_classifications.csv"
  table_guide: "classification/table_layouts.csv"

```

An example config is in `configs/example.yaml`.

## Running Inference

**Basic Execution:**
```bash
python inference.py --config configs/example.yaml
```

**Verbose Logging:**
```bash
python inference.py --config configs/example.yaml -v
```

Logs are saved to `logs/inference.py`. If you encounter issues please check here.

### HPC Inference

If you are running inference on an HPC you will need to submit a job script which tells the HPC to run the `inference.py` script on a GPU. For Slurm systems this can be achieved with `sbatch` or through a GUI (depending on the HPC). An example job submission script is included at `hpc_job.sh`.

Simply run `sbatch hpc_job.sh` in the terminal.

## Output Structure

After processing, the pipeline generates structured outputs organized as follows:

```
outputs/
├── {Municipality}/
│   └── {Year_Range}/
│       └── {Original_Image_Name}.csv
└── text_lines/
    └── temp/
```

**Example CSV Structure:**
```csv
filename,sub_file,row,column,y,x,w,h,transcription
Château-Chinon_1857-1863_3Q5_699_page3.jpg,col1_row1_page3.jpg,1,1,100,150,200,50,"John Doe"
```

## Future Development

Planned enhancements to the TSA Processing Pipeline include:

- Column Header Text Detection: Automate the assignment of column headers based on detected text.
- Include transcription model for all departments.
- Include download of original images for each department. 
- Include download of output database for each department.

## Training

A detailed guide on how these models were trained, and how you can train similar handwritten text recognition and layout analysis models is included [here](https://colab.research.google.com/drive/1_41qEuNzHDmUkkwpRorRiA2fLKcr3Qd6?usp=sharing). This is a detailed online workshop aimed at breaking down this project into small, simple steps for non-technical users.

## Citation

If you utilize this pipeline in your research, please cite the following:

```bibtex
@article{noble2024records,
  title={From Records to Riches: An Automated Pipeline for Digitising French Inheritance Records},
  author={Noble, Aurelius and Sutter, Noah},
  year={2024}
}
```