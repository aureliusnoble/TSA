# inference.py

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Union, Any
import cv2
import shutil
import torch
import yaml
from pydantic import BaseModel, Field
import textwrap
from rich.console import Console
from tqdm import tqdm
from transformers import (AutoTokenizer, TrOCRProcessor,
                          VisionEncoderDecoderModel)
from PIL import Image

import src.tsa_utils as tsa
from doc_ufcn.main import DocUFCN
import pandas as pd
from recognise import TextRecognizer

# Configure logging - only show warnings and errors in console, full logs in file
logging.basicConfig(
    level=logging.INFO,  # Set console logging to WARNING
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add file handler with more verbose logging
file_handler = logging.FileHandler('logs/inference.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Configure logger
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
console = Console()

def configure_logging(verbose: bool = False):
    """Configure comprehensive logging setup"""
    # First, get the root logger and reset its handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    # Set root logger to WARNING to suppress INFO messages from all sources
    root_logger.setLevel(logging.WARNING)
    
    # Configure our pipeline's logger
    pipeline_logger = logging.getLogger(__name__)
    pipeline_logger.setLevel(logging.INFO if not verbose else logging.DEBUG)
    
    # Add file handler for all logs
    file_handler = logging.FileHandler('logs/inference.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    pipeline_logger.addHandler(file_handler)
    
    # Add console handler if verbose
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        pipeline_logger.addHandler(console_handler)
    
    # Configure doc_ufcn logger
    doc_ufcn_logger = logging.getLogger("doc_ufcn")
    doc_ufcn_logger.setLevel(logging.WARNING)
    doc_ufcn_logger.propagate = False
    doc_ufcn_logger.handlers = [logging.NullHandler()]
    
    return pipeline_logger

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors"""
    pass

def validate_config_path(config_path: str) -> None:
    """
    Validate configuration file path with detailed error messages
    
    Args:
        config_path: Path to configuration file
        
    Raises:
        ConfigurationError: With detailed message about specific validation failure
    """
    path_obj = Path(config_path)
    
    if not path_obj.exists():
        raise ConfigurationError(
            f"Configuration file not found: '{config_path}'\n"
            "Please verify the file path and ensure the configuration file exists."
        )
        
    if not path_obj.is_file():
        raise ConfigurationError(
            f"Configuration path exists but is not a file: '{config_path}'\n"
            "The specified path must point to a valid YAML configuration file."
        )
        
    if not os.access(config_path, os.R_OK):
        raise ConfigurationError(
            f"Configuration file exists but is not readable: '{config_path}'\n"
            "Please check file permissions and ensure the process has read access."
        )

class ModelPaths(BaseModel):
    """Configuration for model paths"""
    transcription: Union[str, Dict[str, Any]] = Field(..., description="Path to transcription model or config dict")
    line_extraction: str = Field(..., description="Path to line extraction model")
    row_extraction: str = Field(..., description="Path to row extraction model")
    column_extraction: str = Field(..., description="Path to column extraction model")

class DirectoryPaths(BaseModel):
    """Configuration for directory paths"""
    input: str = Field(..., description="Input directory containing images")
    output: str = Field(..., description="Output directory for results")
    temp: Optional[str] = Field(None, description="Temporary directory for intermediate files")
    page_classification: str = Field(..., description="Path to page classification CSV")
    table_guide: str = Field(..., description="Path to table guide CSV")

class Config(BaseModel):
    """Main configuration class"""
    models: ModelPaths
    directories: DirectoryPaths
    device: Optional[str] = Field("cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = Field(1, ge=1)

class Pipeline:
    """Main pipeline class for document processing"""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config.device)
        
        # **Added Logging for Device Information**
        logger.info(f"Running on device: {self.device}")
        if self.device.type == 'cuda':
            try:
                gpu_name = torch.cuda.get_device_name(self.device)
                logger.info(f"GPU is enabled: {gpu_name}")
            except Exception as e:
                logger.warning(f"CUDA device detected but failed to get device name: {e}")
                logger.info("GPU is enabled")
        else:
            logger.info("GPU is not enabled")
        # **End of Added Logging**
        
        self._initialize_components()
        
    def _load_config(self, config_path: str) -> Config:
        """
        Load and validate configuration with comprehensive error checking
        
        Args:
            config_path (str): Path to YAML configuration file
            
        Returns:
            Config: Validated configuration object
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            PermissionError: If configuration file can't be accessed
            yaml.YAMLError: If configuration file has invalid YAML syntax
            ValueError: If configuration content is invalid
        """
        # Validate config file existence and accessibility
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            error_msg = f"Configuration file not found: {config_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        if not config_path_obj.is_file():
            error_msg = f"Configuration path exists but is not a file: {config_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if not os.access(config_path, os.R_OK):
            error_msg = f"Configuration file exists but is not readable: {config_path}"
            logger.error(error_msg)
            raise PermissionError(error_msg)
            
        try:
            # Attempt to load and parse YAML
            with open(config_path, 'r') as f:
                try:
                    config_dict = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    error_msg = f"Invalid YAML syntax in configuration file: {e}"
                    logger.error(error_msg)
                    raise yaml.YAMLError(error_msg) from e
                    
            if not isinstance(config_dict, dict):
                error_msg = "Configuration file must contain a YAML dictionary/object"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Validate required top-level sections
            required_sections = {'models', 'directories'}
            missing_sections = required_sections - set(config_dict.keys())
            if missing_sections:
                error_msg = f"Missing required configuration sections: {missing_sections}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Create and validate Config object
            try:
                config = Config(**config_dict)
            except Exception as e:
                error_msg = f"Invalid configuration content: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e
                
            # Validate model paths exist
            for model_type, path in config.models.dict().items():
                if model_type == 'transcription':
                    # Handle both string and dict formats for transcription
                    if isinstance(path, dict) and 'path' in path:
                        model_path = Path(path['path'])
                    else:
                        model_path = Path(path)
                else:
                    model_path = Path(path)
                    
                if not model_path.exists():
                    error_msg = f"Model path for {model_type} does not exist: {model_path}"
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                    
            # Validate directory paths exist or can be created
            for dir_type, path in config.directories.dict().items():
                if path is None:
                    continue  # Skip None values (e.g., optional temp directory)
                    
                dir_path = Path(path)
                if dir_type == 'input' and not dir_path.exists():
                    error_msg = f"Input directory does not exist: {path}"
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                elif dir_type in ('output', 'temp'):
                    try:
                        dir_path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        error_msg = f"Cannot create {dir_type} directory {path}: {e}"
                        logger.error(error_msg)
                        raise PermissionError(error_msg) from e
                        
            logger.info("Configuration loaded and validated successfully")
            return config
            
        except Exception as e:
            # Log the full exception stack trace for debugging
            logger.error(f"Configuration loading failed: {e}", exc_info=True)
            raise

    def _initialize_components(self):
        """Initialize all pipeline components"""
        try:
            logger.info("Initializing pipeline components...")
            # Initialize TSA utilities
            self.fp = tsa.FileProcessor()
            self.pc = tsa.PageClassification()
            self.ipp = tsa.ImagePreProcessor()
            self.ts = tsa.TableSegment()
            self.ld = tsa.LineDetection()
            self.htr = tsa.HandwrittenTextRecognition()
            self.pr = tsa.PageReconstruction()

            doc_ufcn_logger = logging.getLogger("doc_ufcn")
            doc_ufcn_logger.setLevel(logging.WARNING)  # Or logging.ERROR
            doc_ufcn_logger.propagate = False  # Prevent propagation to parent loggers
            doc_ufcn_logger.handlers = [logging.NullHandler()]
            
            # Initialize models
            self._initialize_models()
            
            logger.info("Pipeline components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def _initialize_models(self):
        """Initialize all ML models"""
        try:
            # Line extraction model
            self.line_model = DocUFCN(3, 768, self.device)
            self.line_model.load(
                Path(self.config.models.line_extraction),
                mean=[221, 221, 221],
                std=[80, 80, 80]
            )

            # Row/Column models
            self.row_model = DocUFCN(4, 768, self.device)
            self.row_model.load(
                Path(self.config.models.row_extraction),
                [228, 228, 228],
                [71, 71, 71]
            )

            self.col_model = DocUFCN(4, 768, self.device)
            self.col_model.load(
                Path(self.config.models.column_extraction),
                [229, 229, 229],
                [71, 71, 71]
            )

            # Initialize the TextRecognizer based on configuration
            # Handle both string path and dictionary configurations
            transcription_config = self.config.models.transcription
            if isinstance(transcription_config, str):
                # Legacy path-only format
                model_path = transcription_config
                recognizer_args = {"device": str(self.device)}  # Convert device to string
            else:
                # New dictionary format with parameters
                model_path = transcription_config.get("path")
                recognizer_args = {
                    "batch_size": transcription_config.get("batch_size", 20),
                    "precision": transcription_config.get("precision", "half"),
                    "beam_num": transcription_config.get("beam_num", 2),
                    "device": str(self.device)  # Convert device to string
                }
            
            # Initialize the text recognizer
            self.text_recognizer = TextRecognizer(model_path, **recognizer_args)
            logger.info(f"Initialized TextRecognizer with parameters: {recognizer_args}")

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename by removing spaces and special characters
        
        Args:
            filename (str): Original filename
            
        Returns:
            str: Sanitized filename with spaces and newlines replaced by underscores
        """
        # Replace spaces and newlines with underscores
        sanitized = filename.replace(' ', '_').replace('\n', '_')
        
        # Remove any consecutive underscores
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
            
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        return sanitized

    def _get_processed_files(self) -> Set[Tuple[str, str, str]]:
        """
        Scan output directory for already processed files
        
        Returns:
            Set[Tuple[str, str, str]]: Set of (municipality, year_range, filename) tuples
        """
        processed_files = set()
        output_dir = Path(self.config.directories.output)
        
        try:
            # Recursively scan output directory for CSV files
            for csv_path in output_dir.rglob('*.csv'):
                try:
                    # Get relative path components
                    rel_path = csv_path.relative_to(output_dir)
                    if len(rel_path.parts) >= 3:
                        municipality = rel_path.parts[1]
                        year_range = rel_path.parts[2]
                        # Remove .csv extension from filename
                        filename = csv_path.stem
                        if filename.endswith('.jpg'):
                            filename = filename[:-4]  # Remove .jpg suffix
                        processed_files.add((municipality, year_range, filename))
                except Exception as e:
                    logger.warning(f"Error processing path {csv_path}: {e}")
                    continue
            
            logger.info(f"Found {len(processed_files)} already processed files")
            return processed_files
            
        except Exception as e:
            logger.error(f"Error scanning output directory: {e}")
            return set()

    def _get_pending_images(self) -> Tuple[list[Path], int]:
        """
        Get list of images that need processing
        
        Returns:
            Tuple[list[Path], int]: List of paths to process and count of skipped files
        """
        input_dir = Path(self.config.directories.input)
        all_images = list(input_dir.rglob("*.jpg"))
        processed_files = self._get_processed_files()
        
        pending_images = []
        skipped_count = 0
        
        for image_path in all_images:
            try:
                # Get relative path components
                municipality = image_path.parent.parent.name
                year_range = image_path.parent.name
                filename = self._sanitize_filename(image_path.stem)
                
                # Check if file has already been processed
                if (municipality, year_range, filename) in processed_files:
                    logger.debug(f"Skipping already processed file: {image_path}")
                    skipped_count += 1
                else:
                    pending_images.append(image_path)
            except Exception as e:
                logger.warning(f"Error checking file {image_path}: {e}")
                # Include file in pending_images if there's an error checking its status
                pending_images.append(image_path)
        
        return pending_images, skipped_count

    def _create_output_structure(self, image_path: Path) -> Path:
        """
        Create and return the temporary directory structure for line images
        
        Args:
            image_path: Path to the image being processed
            
        Returns:
            Path: Path to the temporary directory for text line images
        """
        # Determine temp directory path - use configured path or default to output/text_lines
        if self.config.directories.temp:
            temp_base_dir = Path(self.config.directories.temp) / "text_lines"
            logger.debug(f"Using configured temp directory: {temp_base_dir}")
        else:
            temp_base_dir = Path(self.config.directories.output) / "text_lines"
            logger.debug(f"Using default temp directory in output path: {temp_base_dir}")
        
        # Clean up existing directory if it exists
        if os.path.exists(temp_base_dir):
            shutil.rmtree(temp_base_dir)  # Recursively delete the directory
            
        # Create temp directory
        temp_base_dir.mkdir(parents=True, exist_ok=True)
        
        return temp_base_dir

    def process_image(self, image_path: Path):
        """Process a single image through the pipeline"""
        try:
            # Load and preprocess image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Create output directory structure
            output_dir = self._create_output_structure(image_path)
            logger.debug(f"Created temporary directory: {output_dir}")

            # Binarization
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_binary = self.ipp.binarize(image_gray, "otsu")
            image_binary = cv2.cvtColor(image_binary, cv2.COLOR_GRAY2RGB)

            # Text line extraction
            polygons, pred, mask, overlap = self.line_model.predict(
                image_binary, raw_output=True, mask_output=True, overlap_output=True
            )
            
            # Extract text lines
            self.ld.extract_textlines(
                image, polygons, str(output_dir), image_path.name,
                self.line_model, padding=0, category=1
            )

            #Extract Headers
            self.ld.extract_textlines(
                image, polygons, str(output_dir), image_path.name,
                self.line_model, padding=0, category=2
            )

            # Page segmentation
            polygons_col, _, _, _ = self.col_model.predict(
                image, raw_output=True, mask_output=True, overlap_output=False
            )
            polygons_row, _, _, _ = self.row_model.predict(
                image, raw_output=True, mask_output=True, overlap_output=False
            )

            # Process grid cells
            height, width, _ = image.shape
            grid_cells = self._process_grid_cells(
                polygons_col, polygons_row, width, height
            )
            output_subdir = Path(output_dir) / image_path.stem
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Rename files and transcribe
            renamed_files = self.ts.assign_cell(str(output_subdir), grid_cells)
            transcriptions = self._process_transcriptions(
                renamed_files, str(output_subdir), image_path.name
            )

            return transcriptions

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise

    def _process_grid_cells(self, polygons_col, polygons_row, width, height):
        """Process grid cells from polygons"""
        polygons_col_unified = self.ts.combine_polygons(polygons_col, [2,3])
        polygons_row_unified = self.ts.combine_polygons(polygons_row, [2,3])

        bbox_col = self.ts.get_bounding_boxes(polygons_col_unified)
        bbox_row = self.ts.get_bounding_boxes(polygons_row_unified)

        bbox_row_adj = self.ts.adjust_bounding_boxes(bbox_row, 'row')
        bbox_col_adj = self.ts.adjust_bounding_boxes(bbox_col, 'column')

        # Adjust bounding boxes
        for i in range(len(bbox_row_adj)):
            x, y, w, h = bbox_row_adj[i]
            bbox_row_adj[i] = (0, y, w, h)

        for i in range(len(bbox_col_adj)):
            x, y, w, h = bbox_col_adj[i]
            bbox_col_adj[i] = (x, 0, w, h)

        bbox_row_adj = self.ts.add_rows(bbox_row_adj, width)
        return self.ts.find_grid_cells(bbox_row_adj, bbox_col_adj)

    def _process_transcriptions(self, renamed_files, directory, filename):
        """Process transcriptions for renamed files using the TextRecognizer"""
        df_pred = pd.DataFrame(columns=[
            'filename', 'sub_file', 'row', 'column',
            'x', 'y', 'w', 'h', 'transcription'
        ])

        for sub_image in renamed_files:
            # Parse file information
            column, row, y, x, w, h = self.htr.filename_parse(sub_image)
            
            # Load image path
            img_path = os.path.join(directory, sub_image)
            
            # Read image with OpenCV and convert to PIL for the recognizer
            cv_image = cv2.imread(img_path)
            if cv_image is not None:
                # Convert BGR to RGB (OpenCV loads as BGR, PIL expects RGB)
                cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(cv_image_rgb)
                
                # Get transcription using the new recognizer
                try:
                    generated_text = self.text_recognizer.recognise(pil_image)
                    logger.debug(f"Successfully transcribed {sub_image}")
                except Exception as e:
                    logger.error(f"Error transcribing {sub_image}: {e}")
                    generated_text = ""
            else:
                logger.warning(f"Could not read image {img_path}")
                generated_text = ""

            # Create DataFrame entry
            temp_df = pd.DataFrame({
                'filename': [filename],
                'sub_file': [sub_image],
                'row': [row],
                'column': [column],
                'header': '__header' in sub_image,
                'y': [y],
                'x': [x],
                'w': [w],
                'h': [h],
                'transcription': [generated_text]
            })

            df_pred = pd.concat([df_pred, temp_df], ignore_index=True)

        return df_pred

    def run(self):
        """Run the pipeline on all unprocessed images in input directory"""
        # Get list of pending images
        image_paths, skipped_count = self._get_pending_images()
        total_images = len(image_paths)
        
        logger.info(f"Found {total_images} images to process (skipped {skipped_count} already processed)")
        
        if total_images == 0:
            logger.info("No new images to process")
            return
        
        # Initialize progress tracking
        pbar = tqdm(
            total=total_images,
            desc="Processing images",
            unit="img",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        processed_count = 0
        error_count = 0
        
        for image_path in image_paths:
            try:
                municipality = image_path.parent.parent.name
                year = image_path.parent.name
                
                # Process image
                df_pred = self.process_image(image_path)
                
                # Reassemble and export
                # df_tables = pd.read_csv(self.config.directories.table_guide)
                # df_pages = pd.read_csv(self.config.directories.page_classification)
                
                df_headers = df_pred[df_pred['sub_file'].str.contains('__header')]

                df_pred_final = self.pr.reassemble_pages(
                    df_pred, image_path.name, df_headers
                )

                # Export with sanitized filename
                output_dir = Path(self.config.directories.output)
                sanitized_name = self._sanitize_filename(image_path.name)
                
                # Export transcription with sanitized filename
                self.pr.export_transcription(
                    df_pred_final,
                    str(output_dir),
                    municipality,
                    year,
                    sanitized_name
                )
                
                processed_count += 1
                pbar.update(1)

            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}", exc_info=True)
                error_count += 1
                continue
                
        pbar.close()
        
        # Log final statistics
        logger.info(
            f"Pipeline completed:\n"
            f"- Successfully processed: {processed_count}\n"
            f"- Errors encountered: {error_count}\n"
            f"- Previously processed (skipped): {skipped_count}\n"
            f"- Total files considered: {total_images + skipped_count}"
        )
        
        if error_count > 0:
            logger.warning(
                f"Pipeline completed with {error_count} errors. "
                "Check the log file for details."
            )

        # Cleanup temporary files if configured
        if self.config.directories.temp:
            temp_base_dir = Path(self.config.directories.temp) / "text_lines"
            if os.path.exists(temp_base_dir):
                try:
                    logger.info(f"Cleaning up temporary directory: {temp_base_dir}")
                    shutil.rmtree(temp_base_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary directory: {e}")


def main():
    """Enhanced main function with proper error handling"""
    parser = argparse.ArgumentParser(
        description='Document Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
            Configuration File Format:
            -------------------------
            The configuration must be a YAML file with the following structure:
            
            models:
              transcription:
                path: /path/to/transcription/model
                batch_size: 20  # Optional
                precision: "half"  # Optional: "full", "half", or "autocast"
                beam_num: 2  # Optional
              line_extraction: /path/to/line/model
              row_extraction: /path/to/row/model
              column_extraction: /path/to/column/model
            
            directories:
              input: /path/to/input/dir
              output: /path/to/output/dir
              temp: /path/to/temp/dir  # Optional, for intermediate processing files
              page_classification: /path/to/page/classification.csv
              table_guide: /path/to/table/guide.csv
            
            device: cuda  # Optional, defaults to cuda if available, else cpu
            batch_size: 1  # Optional, defaults to 1
        ''')
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true',
        help='Enable verbose logging output'
    )
    
    try:
        args = parser.parse_args()
        
        # Configure logging before anything else
        logger = configure_logging(args.verbose)
        logger.info("Starting OCR pipeline initialization...")
        
        # Validate configuration file path
        try:
            validate_config_path(args.config)
        except ConfigurationError as e:
            console.print(f"[red]Configuration Error:[/red] {str(e)}")
            logger.error(f"Configuration validation failed: {e}")
            sys.exit(1)
            
        # Initialize and run pipeline
        try:
            pipeline = Pipeline(args.config)
            logger.info("Starting pipeline execution...")
            pipeline.run()
            logger.info("Pipeline execution completed successfully")
            
        except yaml.YAMLError as e:
            error_msg = (
                f"Invalid YAML syntax in configuration file: {e}\n"
                "Please verify the configuration file contains valid YAML syntax."
            )
            console.print(f"[red]YAML Error:[/red] {error_msg}")
            logger.error(f"YAML parsing failed: {e}")
            sys.exit(1)
            
        except ValueError as e:
            error_msg = (
                f"Invalid configuration content: {e}\n"
                "Please verify all required fields are present and have valid values."
            )
            console.print(f"[red]Configuration Error:[/red] {error_msg}")
            logger.error(f"Configuration validation failed: {e}")
            sys.exit(1)
            
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            console.print(f"[red]Error:[/red] {error_msg}")
            logger.error("Pipeline failed with unexpected error", exc_info=True)
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline execution interrupted by user[/yellow]")
        logger.info("Pipeline execution interrupted by user")
        sys.exit(130)
        
if __name__ == "__main__":
    main()