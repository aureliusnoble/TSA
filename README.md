# TSA Processing Pipeline

A comprehensive computer vision pipeline for the extraction and digitization of historical tabular data from French inheritance records (Tables des Successions et Absences). This system implements a multi-stage deep learning approach combining document layout analysis, table structure detection, and handwritten text recognition.

## Technical Overview

The pipeline implements a sequential processing architecture with the following stages:

1. **Document Layout Analysis**
   - Initial image preprocessing and binarization
   - Text line detection via DocUFCN
   - Spatial relationship modeling for structural component detection

2. **Table Structure Recognition**
   - Row/column detection using specialized DocUFCN models
   - Grid cell extraction through polygon intersection analysis
   - Cell classification based on spatial positioning and layout rules

3. **Handwritten Text Recognition (HTR)**
   - TrOCR-based transformer model for text transcription
   - Cell-level text recognition with contextual analysis
   - Post-processing with domain-specific rules

4. **Data Reconstruction**
   - Table structure reassembly from cell-level predictions
   - Data validation against predefined schemas
   - Structured output generation in CSV format

## System Requirements

### Hardware Prerequisites
- CUDA-capable GPU with minimum 8GB VRAM (Tesla T4 or better recommended)
- 16GB+ system RAM for batch processing
- 10GB+ storage for model weights and intermediate outputs

### Software Dependencies
- Conda or Miniconda (https://anaconda.org/anaconda/conda)
- CUDA 11.x+ and compatible cuDNN

## Installation and Setup

1. **Repository Clone**

Download the reposity and change to this working directory.
```bash
git clone https://github.com/aureliusnoble/TSA
cd TSA
```

2. **Environment Configuration**

To install the necessary dependencies, packages, and libraries please run.
```bash
# Create and activate conda environment
conda env create -f environment.yaml
conda activate TSA

# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

3. **Model Download**

The models are not downloaded automatically with the repo, but can be downloaded by running.
```bash
# Download pre-trained models
python download_models.py

# Expected model structure:
models/
├── transcription_nievre/          # TrOCR weights and config
├── textlines_nievre/model.pth     # Line detection weights
├── rows_nievre/model.pth          # Row detection weights
└── columns_nievre/model.pth       # Column detection weights
```

## Data Organization

### Input Directory Structure
The pipeline expects a hierarchical organization of input images:

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

The pipeline requires two configuration files for table structure interpretation:

1. **Page Classification** 

Examples in: `classification/page_classifications.csv`:
Maps each document page to its table structure type:

```csv
File,Relative Path,Type
"Château-Chinon_1857-1863_3Q5_699_page3.jpg","Château-Chinon/1857-1863/Château-Chinon_1857-1863_3Q5_699_page3.jpg","2Ab"
"Brinon-sur-Beuvron_1811-1819_3Q2_407_page12.jpg","Brinon-sur-Beuvron/1811-1819/Brinon-sur-Beuvron_1811-1819_3Q2_407_page12.jpg","1A"
```

2. **Table Layout Schema** 

Examples in: `classification/table_layouts.csv`:
Defines the expected column structure for each table type:

```csv
table_type,col_num,col_name,data_type
"1A",1,"Article Number","int"
"1A",2,"Surname","str"
"1A",3,"First Name","str"
...
"2Ab",1,"Article Number","int"
"2Ab",2,"Surname","str"
```

Currently supported table types:
- **Type 1A**: 11-column format (early 19th century)
- **Type 1B**: Modified 11-column layout
- **Type 2Aa**: 21-column extended format
- **Type 2Ab**: 22-column comprehensive format
- **Type 2B**: 22-column alternative format

## Pipeline Configuration

Create a YAML configuration file specifying model paths and processing parameters:

Example in: `configs/config.yaml`

```yaml

models:
  # Model paths
  transcription: "models/transcription_nievre"
  line_extraction: "models/textlines_nievre/model.pth"
  row_extraction: "models/rows_nievre/model.pth"
  column_extraction: "models/columns_nievre/model.pth"

directories:
  # I/O paths
  input: "examples/inputs"
  output: "examples/outputs"
  
  # Classification files
  page_classification: "classification/page_classifications.csv"
  table_guide: "classification/table_layouts.csv"
```

## Inference Execution

Run the pipeline:
```bash
# Basic execution
python inference.py --config configs/config.yaml
```

If you want more detailed output logs you can run
```bash
python inference.py --config configs/config.yaml -v
```
Logs are saved in `/logs`.

## Output Structure

The pipeline generates a structured output:

```
outputs/
├── {Municipality}/
│   └── {Year_Range}/
│       └── {Original_Image_Name}.csv
└── text_lines/
    └── temp/
```

Each output CSV contains:
- Structured table data with transcriptions, rows and column names
- Source image reference

## Future Development

Planned improvements:
- Column header text detection for column assignment
- General purpose models

## Citation

If you use this pipeline in your research, please cite:
```bibtex
Noble, Aurelius and Noah Sutter, 2024. From Records to Riches: An Automated Pipeline for Digitising French Inheritance Records.
```
