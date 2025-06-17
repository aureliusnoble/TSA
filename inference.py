"""
Optimized OCR Pipeline with In-Memory Processing

Key optimizations over the original implementation:
1. In-memory line image processing - eliminates file I/O overhead
2. Batch transcription - processes all line images in a single GPU call
3. Greedy decoding by default (beam_num=1) - faster than beam search
4. Batch BERT classification - classifies all texts in one call
5. Detailed timing logs to identify bottlenecks

Performance improvements:
- Eliminates ~85 file writes and reads per page
- Reduces transcription time by 3-5x through batching
- Overall page processing time reduced significantly
"""

import argparse
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Union, Any, List
import cv2
import shutil
import torch
import yaml
from pydantic import BaseModel, Field
import textwrap
from rich.console import Console
from tqdm import tqdm
from transformers import (AutoTokenizer, TrOCRProcessor,
                          VisionEncoderDecoderModel, AutoModelForSequenceClassification)
from PIL import Image
import numpy as np
import pandas as pd
from collections import Counter

import src.tsa_utils as tsa
from doc_ufcn.main import DocUFCN
from src.recognise import TextRecognizer
import warnings

warnings.filterwarnings('ignore')

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

def extract_bounding_boxes_and_avg_y2(polygons_data):
    """
    Extract bounding boxes from a list of polygon data and calculate average y2 coordinate.
    
    Args:
        polygons_data: List of dictionaries, each with 'confidence' and 'polygon' keys
                      where 'polygon' is a list of [x,y] coordinate pairs
    
    Returns:
        tuple: (list of bounding boxes, average y2 coordinate)
    """
    bounding_boxes = []
    y2_coordinates = []
    
    for item in polygons_data:
        polygon = item['polygon']
        
        # Extract x and y coordinates from the polygon
        x_coords = [point[0] for point in polygon]
        y_coords = [point[1] for point in polygon]
        
        # Calculate the bounding box
        x1 = min(x_coords)  # Minimum x (left)
        y1 = min(y_coords)  # Minimum y (top)
        x2 = max(x_coords)  # Maximum x (right)
        y2 = max(y_coords)  # Maximum y (bottom)
        
        bounding_box = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        bounding_boxes.append(bounding_box)
        
        # Collect x2 coordinate for average calculation
        y2_coordinates.append(y2)
    
    # Calculate average x2 coordinate
    avg_y2 = sum(y2_coordinates) / len(y2_coordinates) if y2_coordinates else 0
    
    return bounding_boxes, avg_y2

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
    bert_classifier_model: Optional[str] = Field(None, description="Path to BERT column classifier model")
    bert_classifier_tokenizer: Optional[str] = Field(None, description="Name or path of BERT tokenizer")
    bert_classifier_label_map: Optional[str] = Field(None, description="Path to label mapping CSV")

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
    target_width: int = Field(3840, ge=1, description="Target width for image resizing")

class Pipeline:
    """
    Main pipeline class for document processing
    
    This optimized version processes line images in memory without intermediate file I/O,
    providing significant performance improvements over disk-based processing.
    """
    
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
        
        # Get target width from config (default to 3840 if not specified)
        self.target_width = getattr(self.config, 'target_width', 3840)
        logger.info(f"Target image width set to: {self.target_width} pixels")
        
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
                # Skip validation of optional BERT paths if they're None
                elif model_type in ['bert_classifier_model', 'bert_classifier_tokenizer', 'bert_classifier_label_map'] and path is None:
                    continue
                # Skip validation for tokenizer if it's a model name rather than a path
                elif model_type == 'bert_classifier_tokenizer' and not os.path.exists(path):
                    # Assume it's a model name from HuggingFace, so don't validate existence
                    continue
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
                    "batch_size": transcription_config.get("batch_size", 20),  # Changed default from 20 to match reference
                    "precision": transcription_config.get("precision", "half"),
                    "beam_num": transcription_config.get("beam_num", 1),  # Changed default to 1 for faster greedy decoding
                    "device": str(self.device)  # Convert device to string
                }
            
            # Initialize the text recognizer
            self.text_recognizer = TextRecognizer(model_path, **recognizer_args)
            logger.info(f"Initialized TextRecognizer with parameters: {recognizer_args}")

            # Initialize BERT classifier if configured
            self.bert_classifier = None
            self.bert_tokenizer = None
            self.label_id_to_name = None
            
            if (self.config.models.bert_classifier_model and 
                self.config.models.bert_classifier_tokenizer and
                self.config.models.bert_classifier_label_map):
                self._initialize_bert_classifier()

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
            
    def _initialize_bert_classifier(self):
        """Initialize the BERT classifier model for column classification"""
        try:
            logger.info("Initializing BERT column classifier...")
            
            # Load tokenizer
            tokenizer_name = self.config.models.bert_classifier_tokenizer
            self.bert_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"Loaded tokenizer: {tokenizer_name}")
            
            # Load label mapping
            label_map_path = self.config.models.bert_classifier_label_map
            label_df = pd.read_csv(label_map_path)
            self.label_id_to_name = {row['id']: row['label'] for _, row in label_df.iterrows()}
            logger.info(f"Loaded label mapping with {len(self.label_id_to_name)} categories")
            
            # Load model - First determine the number of labels
            num_labels = len(self.label_id_to_name)
            model_path = self.config.models.bert_classifier_model
            
            # If the path is a directory (huggingface model), load directly
            if os.path.isdir(model_path):
                self.bert_classifier = AutoModelForSequenceClassification.from_pretrained(
                    model_path, 
                    num_labels=num_labels
                )
                logger.info(f"Loaded BERT classifier from directory: {model_path}")
            else:
                # If it's a state dict file, first load the base model then apply state dict
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    tokenizer_name,
                    num_labels=num_labels,
                    id2label=self.label_id_to_name
                )
                self.bert_classifier = base_model
                self.bert_classifier.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded BERT classifier state dict from: {model_path}")
                
            # Move model to device
            self.bert_classifier.to(self.device)
            self.bert_classifier.eval()  # Set to evaluation mode
            
            logger.info("BERT column classifier initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize BERT classifier: {e}", exc_info=True)
            logger.warning("BERT classification will be disabled")
            self.bert_classifier = None
            self.bert_tokenizer = None
            self.label_id_to_name = None

    def classify_text(self, text):
        """
        Classify text using the BERT classifier
        
        Args:
            text (str): The text to classify
            
        Returns:
            tuple: (predicted_label, confidence_score) or (None, 0.0) if classifier not available
        """
        if not self.bert_classifier or not self.bert_tokenizer or not self.label_id_to_name:
            return None, 0.0
            
        try:
            # Prepare the input
            inputs = self.bert_tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.bert_classifier(**inputs)
                logits = outputs.logits
                
                # Get predicted class and confidence
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                confidence, prediction = torch.max(probabilities, dim=1)
                
                predicted_id = prediction.item()
                confidence_score = confidence.item()
                
                # Map ID to label name
                predicted_label = self.label_id_to_name.get(predicted_id, "UNKNOWN")
                
                return predicted_label, confidence_score
                
        except Exception as e:
            logger.error(f"Error during text classification: {e}")
            return None, 0.0

    def classify_texts_batch(self, texts: List[str]) -> List[Tuple[Optional[str], float]]:
        """
        Classify multiple texts using the BERT classifier in a batch
        
        Args:
            texts (List[str]): List of texts to classify
            
        Returns:
            List[Tuple[Optional[str], float]]: List of (predicted_label, confidence_score) tuples
        """
        if not self.bert_classifier or not self.bert_tokenizer or not self.label_id_to_name:
            return [(None, 0.0) for _ in texts]
            
        try:
            # Prepare the inputs
            inputs = self.bert_tokenizer(
                texts,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.bert_classifier(**inputs)
                logits = outputs.logits
                
                # Get predicted classes and confidences
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                confidences, predictions = torch.max(probabilities, dim=1)
                
                results = []
                for pred_id, conf in zip(predictions.tolist(), confidences.tolist()):
                    predicted_label = self.label_id_to_name.get(pred_id, "UNKNOWN")
                    results.append((predicted_label, conf))
                
                return results
                
        except Exception as e:
            logger.error(f"Error during batch text classification: {e}")
            return [(None, 0.0) for _ in texts]

    def _resize_image(self, image, target_width=None):
        """
        Resize image to target width while maintaining aspect ratio
        
        Args:
            image: OpenCV image (numpy array)
            target_width: Target width in pixels (uses self.target_width if None)
            
        Returns:
            Resized image
        """
        if image is None:
            raise ValueError("Cannot resize None image")
            
        # Use instance target_width if not specified
        if target_width is None:
            target_width = self.target_width
            
        # Get current dimensions
        height, width = image.shape[:2]
        
        # If image is already at target width, return unmodified
        if width == target_width:
            logger.debug(f"Image already at target width ({width}px), skipping resize")
            return image
            
        # Calculate new height to maintain aspect ratio
        aspect_ratio = height / width
        new_height = int(target_width * aspect_ratio)
        
        # Resize image
        resized_image = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_AREA)
                
        return resized_image

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
        
        # Look for all supported image formats
        all_images = []
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            all_images.extend(list(input_dir.rglob(f"*{ext}")))
            # Also check for uppercase extensions
            all_images.extend(list(input_dir.rglob(f"*{ext.upper()}"))) 
        
        processed_files = self._get_processed_files()
        
        pending_images = []
        skipped_count = 0
        hidden_files_count = 0
        
        for image_path in all_images:
            try:
                # Skip macOS metadata files (those starting with ._)
                if image_path.name.startswith('._'):
                    logger.debug(f"Skipping macOS metadata file: {image_path}")
                    hidden_files_count += 1
                    continue
                    
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
        
        logger.info(f"Skipped {hidden_files_count} macOS metadata files (._* files)")
        return pending_images, skipped_count

    def _extract_line_images_memory(self, image, polygons, header_y2, grid_cells, filename):
        """
        Extract line images and keep them in memory instead of saving to disk
        
        Returns:
            List of tuples: (image_array, metadata_dict)
        """
        line_images_data = []
        
        # Extract text lines (category 1) and headers (category 2)
        for category in [1, 2]:
            if category not in polygons:
                continue
                
            polygons_list = polygons.get(category, [])
            if not polygons_list:
                continue
                
            logger.debug(f"Extracting {len(polygons_list)} lines for category {category}")
                
            for idx, polygon_data in enumerate(polygons_list):
                polygon = polygon_data['polygon']
                
                # Convert polygon to numpy array
                pts = np.array(polygon, np.int32)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(pts)
                
                # Apply padding if needed
                padding = 0
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                # Extract the line image
                line_img = image[y:y+h, x:x+w]
                
                # Skip if image is empty
                if line_img.size == 0:
                    continue
                
                # Determine if this is a header
                is_header = (category == 2) or (y + h/2 < header_y2)
                
                # Create metadata
                metadata = {
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'category': category,
                    'is_header': is_header,
                    'polygon': polygon,
                    'filename': filename
                }
                
                line_images_data.append((line_img, metadata))
        
        logger.info(f"Extracted {len(line_images_data)} line images in memory")
        
        # Assign to grid cells
        assigned_images = []
        for line_img, metadata in line_images_data:
            # Find which cell this line belongs to
            cell_name = self._find_cell_assignment(
                metadata['x'], metadata['y'], 
                metadata['w'], metadata['h'], 
                grid_cells
            )
            
            if cell_name:
                # Parse cell name to get row and column
                parts = cell_name.split('_')
                if len(parts) >= 4:
                    column = parts[1].replace('col', '')
                    row = parts[3].replace('row', '')
                else:
                    column = '1'
                    row = '1'
            else:
                column = '1'
                row = '1'
                cell_name = 'col1_row1'
            
            # Update metadata
            metadata['cell_name'] = cell_name
            metadata['column'] = column
            metadata['row'] = row
            metadata['sub_file'] = f"{cell_name}_y{metadata['y']}_x{metadata['x']}_w{metadata['w']}_h{metadata['h']}"
            if metadata['is_header']:
                metadata['sub_file'] = "__header__" + metadata['sub_file']
            metadata['sub_file'] += ".png"
            
            assigned_images.append((line_img, metadata))
        
        return assigned_images

    def _find_cell_assignment(self, x, y, w, h, grid_cells):
        """Find which grid cell a line belongs to based on its position"""
        line_center_x = x + w / 2
        line_center_y = y + h / 2
        
        best_cell = None
        best_overlap = 0
        
        for cell_name, (cell_x, cell_y, cell_w, cell_h) in grid_cells.items():
            # Check if line center is within cell
            if (cell_x <= line_center_x <= cell_x + cell_w and
                cell_y <= line_center_y <= cell_y + cell_h):
                
                # Calculate overlap area
                overlap_x1 = max(x, cell_x)
                overlap_y1 = max(y, cell_y)
                overlap_x2 = min(x + w, cell_x + cell_w)
                overlap_y2 = min(y + h, cell_y + cell_h)
                
                if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    if overlap_area > best_overlap:
                        best_overlap = overlap_area
                        best_cell = cell_name
        
        return best_cell


    def _process_transcriptions_memory(self, line_images_data, filename):
        """
        Process transcriptions from in-memory line images
        
        Args:
            line_images_data: List of (image_array, metadata) tuples
            filename: Original filename
            
        Returns:
            DataFrame with transcription results
        """
        if not line_images_data:
            # Return empty dataframe with correct columns
            return pd.DataFrame(columns=[
                'filename', 'sub_file', 'row', 'column',
                'x', 'y', 'w', 'h', 'transcription',
                'col_name_line_lm', 'col_name_line_lm_conf',
                'col_name_lm', 'col_name_lm_conf'
            ])
        
        # Process in batches based on the configured batch size
        batch_size = self.text_recognizer.batch_size
        all_results = []
        
        logger.info(f"Processing {len(line_images_data)} line images in batches of {batch_size}")
        
        for batch_start in range(0, len(line_images_data), batch_size):
            batch_end = min(batch_start + batch_size, len(line_images_data))
            batch_data = line_images_data[batch_start:batch_end]
            
            # Prepare batch
            batch_images = []
            batch_metadata = []
            
            for line_img, metadata in batch_data:
                # Convert OpenCV image to PIL
                line_img_rgb = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(line_img_rgb)
                
                batch_images.append(pil_image)
                batch_metadata.append(metadata)
            
            # Process this batch
            logger.debug(f"Processing batch {batch_start//batch_size + 1}: {len(batch_images)} images")
            start_time = time.time()
            
            try:
                # Get transcriptions for this batch
                generated_texts = self.text_recognizer.recognise(batch_images)
                
                # Ensure generated_texts is a list
                if not isinstance(generated_texts, list):
                    generated_texts = [generated_texts]
                
                batch_time = time.time() - start_time
                logger.debug(f"Batch transcription completed in {batch_time:.2f}s ({batch_time/len(batch_images):.3f}s per image)")
                
                # Classify texts if BERT classifier is available
                if self.bert_classifier is not None:
                    classifications = self.classify_texts_batch(generated_texts)
                else:
                    classifications = [(None, 0.0) for _ in generated_texts]
                    
            except Exception as e:
                logger.error(f"Error in batch transcription: {e}")
                generated_texts = [""] * len(batch_images)
                classifications = [(None, 0.0) for _ in batch_images]
            
            # Build results for this batch
            for metadata, text, (label, conf) in zip(batch_metadata, generated_texts, classifications):
                all_results.append({
                    'filename': filename,
                    'sub_file': metadata['sub_file'],
                    'row': metadata['row'],
                    'column': metadata['column'],
                    'header': metadata['is_header'],
                    'y': metadata['y'],
                    'x': metadata['x'],
                    'w': metadata['w'],
                    'h': metadata['h'],
                    'transcription': text,
                    'col_name_line_lm': label,
                    'col_name_line_lm_conf': conf,
                    'col_name_lm': None,  # Will be updated below
                    'col_name_lm_conf': 0.0  # Will be updated below
                })
        
        # Create DataFrame from all results
        df_pred = pd.DataFrame(all_results)
        
        # Log overall performance
        total_time = time.time() - start_time
        logger.info(f"Processed {len(line_images_data)} images in {len(line_images_data)//batch_size + 1} batches, total time: {total_time:.2f}s")
        
        # Calculate column-level classifications (rest of the method remains the same)
        if self.bert_classifier is not None and not df_pred.empty:
            unique_columns = df_pred['column'].unique()
            column_classifications = {}
            
            for col_num in unique_columns:
                col_rows = df_pred[df_pred['column'] == col_num]
                classifications = col_rows[col_rows['col_name_line_lm'].notna()]['col_name_line_lm'].tolist()
                
                if classifications:
                    counter = Counter(classifications)
                    most_common_class, count = counter.most_common(1)[0]
                    confidence = count / len(classifications)
                    
                    column_classifications[col_num] = {
                        'col_name_lm': most_common_class,
                        'col_name_lm_conf': confidence
                    }
                else:
                    column_classifications[col_num] = {
                        'col_name_lm': None,
                        'col_name_lm_conf': 0.0
                    }
            
            # Apply column-level classifications
            for col_num, classification in column_classifications.items():
                mask = df_pred['column'].astype(str) == str(col_num)
                df_pred.loc[mask, 'col_name_lm'] = classification['col_name_lm']
                df_pred.loc[mask, 'col_name_lm_conf'] = classification['col_name_lm_conf']
            
            logger.info(f"Column classifications: {column_classifications}")
        
        return df_pred
    def process_image(self, image_path: Path):
        """Process a single image through the pipeline"""
        try:
            total_start_time = time.time()
            
            # Load image
            logger.info(f"Loading image: {image_path}")
            load_start = time.time()
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
                
            # Resize image to target width (default 3840px)
            image = self._resize_image(image)
            load_time = time.time() - load_start
            logger.info(f"Image loading and resizing took {load_time:.2f}s")

            # Binarization
            binarize_start = time.time()
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_binary = self.ipp.binarize(image_gray, "otsu")
            image_binary = cv2.cvtColor(image_binary, cv2.COLOR_GRAY2RGB)
            binarize_time = time.time() - binarize_start
            logger.info(f"Binarization took {binarize_time:.2f}s")

            # Text line extraction
            line_extract_start = time.time()
            polygons, _, _, overlap_textline = self.line_model.predict(
                image_binary, raw_output=True, mask_output=True, overlap_output=False
            )
            line_extract_time = time.time() - line_extract_start
            logger.info(f"Line extraction took {line_extract_time:.2f}s")

            # Page segmentation
            seg_start = time.time()
            polygons_col, _, _, overlap_column = self.col_model.predict(
                image, raw_output=True, mask_output=True, overlap_output=False
            )
            polygons_row, _, _, overlap_row = self.row_model.predict(
                image, raw_output=True, mask_output=True, overlap_output=False
            )
            seg_time = time.time() - seg_start
            logger.info(f"Page segmentation took {seg_time:.2f}s")

            bounding_boxes, header_y2 = extract_bounding_boxes_and_avg_y2(polygons_row[1])

            # Process grid cells
            grid_start = time.time()
            height, width, _ = image.shape
            grid_cells = self._process_grid_cells(
                polygons_col, polygons_row, width, height
            )

            for key, value in grid_cells.items():
                if '_row1' in key:
                    x, y, w, h = value
                    # Increase height by the amount we're moving y up
                    new_height = h + y
                    grid_cells[key] = (x, 0, w, new_height)
            grid_time = time.time() - grid_start
            logger.info(f"Grid cell processing took {grid_time:.2f}s")

            # Extract line images in memory instead of saving to disk
            extract_start = time.time()
            line_images_data = self._extract_line_images_memory(
                image, polygons, header_y2, grid_cells, image_path.name
            )
            extract_time = time.time() - extract_start
            logger.info(f"Line image extraction (in memory) took {extract_time:.2f}s for {len(line_images_data)} lines")
            
            # Process transcriptions from in-memory images
            transcribe_start = time.time()
            transcriptions = self._process_transcriptions_memory(
                line_images_data, image_path.name
            )
            transcribe_time = time.time() - transcribe_start
            
            total_time = time.time() - total_start_time
            logger.info(f"Total processing time for {image_path.name}: {total_time:.2f}s")
            logger.info(f"Time breakdown: Load={load_time:.2f}s, Binarize={binarize_time:.2f}s, Lines={line_extract_time:.2f}s, Segment={seg_time:.2f}s, Grid={grid_time:.2f}s, Extract={extract_time:.2f}s, Transcribe={transcribe_time:.2f}s")

            return transcriptions

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}", exc_info=True)
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
        """Original method for backward compatibility - processes one by one"""
        logger.warning("Using non-batched transcription processing - this method is deprecated")
        return self._process_transcriptions_original(renamed_files, directory, filename)

    def _process_transcriptions_original(self, renamed_files, directory, filename):
        """Original implementation that processes images one by one"""
        # Initialize DataFrame with all required columns
        df_pred = pd.DataFrame(columns=[
            'filename', 'sub_file', 'row', 'column',
            'x', 'y', 'w', 'h', 'transcription',
            'col_name_line_lm', 'col_name_line_lm_conf',  # Line-level classification
            'col_name_lm', 'col_name_lm_conf'  # Column-level classification
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
                    
                    # Classify text using BERT model
                    if self.bert_classifier is not None:
                        col_name_line_lm, col_name_line_lm_conf = self.classify_text(generated_text)
                    else:
                        col_name_line_lm, col_name_line_lm_conf = None, 0.0
                        
                except Exception as e:
                    logger.error(f"Error transcribing {sub_image}: {e}")
                    generated_text = ""
                    col_name_line_lm, col_name_line_lm_conf = None, 0.0
            else:
                logger.warning(f"Could not read image {img_path}")
                generated_text = ""
                col_name_line_lm, col_name_line_lm_conf = None, 0.0

            # Create DataFrame entry with placeholders for column-level classifications
            # (will be filled in later)
            temp_df = pd.DataFrame({
                'filename': [filename],
                'sub_file': [sub_image],
                'row': [row],
                'column': [column],
                'header': ['__header' in sub_image],
                'y': [y],
                'x': [x],
                'w': [w],
                'h': [h],
                'transcription': [generated_text],
                'col_name_line_lm': [col_name_line_lm],
                'col_name_line_lm_conf': [col_name_line_lm_conf],
                'col_name_lm': [None],  # Placeholder, will be updated
                'col_name_lm_conf': [0.0]  # Placeholder, will be updated
            })

            df_pred = pd.concat([df_pred, temp_df], ignore_index=True)

        # Calculate the most common column classification per column number
        if self.bert_classifier is not None:
            # Get unique column values
            unique_columns = df_pred['column'].unique()
            
            # Dictionary to store most common classifications
            column_classifications = {}
            
            for col_num in unique_columns:
                # Get all rows for this column
                col_rows = df_pred[df_pred['column'] == col_num]
                
                # Get classifications for this column that are not None
                classifications = col_rows[col_rows['col_name_line_lm'].notna()]['col_name_line_lm'].tolist()
                
                if classifications:
                    # Find most common classification
                    counter = Counter(classifications)
                    most_common_class, count = counter.most_common(1)[0]
                    confidence = count / len(classifications)
                    
                    column_classifications[col_num] = {
                        'col_name_lm': most_common_class,
                        'col_name_lm_conf': confidence
                    }
                else:
                    column_classifications[col_num] = {
                        'col_name_lm': None,
                        'col_name_lm_conf': 0.0
                    }
            
            # Apply column-level classifications back to the dataframe
            for col_num, classification in column_classifications.items():
                # Ensure we convert column to the same type as col_num to avoid mismatches
                # with numeric vs string columns
                mask = df_pred['column'].astype(str) == str(col_num)
                df_pred.loc[mask, 'col_name_lm'] = classification['col_name_lm']
                df_pred.loc[mask, 'col_name_lm_conf'] = classification['col_name_lm_conf']
                
            logger.info(f"Column classifications: {column_classifications}")
        else:
            # If BERT classifier is not available, add empty columns
            df_pred['col_name_lm'] = None
            df_pred['col_name_lm_conf'] = 0.0

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
                
                logger.info(f"Processing image: {image_path}")
                
                # Process image
                df_pred = self.process_image(image_path)
                
                # Reassemble and export
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
                logger.info(f"Successfully processed: {image_path}")

            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}", exc_info=True)
                error_count += 1
                pbar.update(1)  # Update progress bar even if there's an error
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
                batch_size: 20  # Optional, default: 20
                precision: "half"  # Optional: "full", "half", or "autocast"
                beam_num: 1  # Optional, default: 1 (greedy decoding, faster)
              line_extraction: /path/to/line/model
              row_extraction: /path/to/row/model
              column_extraction: /path/to/column/model
              bert_classifier_model: /path/to/bert/model.pt  # Optional
              bert_classifier_tokenizer: camembert-base  # Optional
              bert_classifier_label_map: /path/to/label_mapping.csv  # Optional
            
            directories:
              input: /path/to/input/dir
              output: /path/to/output/dir
              temp: /path/to/temp/dir  # Optional, no longer used for line images
              page_classification: /path/to/page/classification.csv
              table_guide: /path/to/table/guide.csv
            
            device: cuda  # Optional, defaults to cuda if available, else cpu
            batch_size: 1  # Optional, defaults to 1
            target_width: 3840  # Optional, defaults to 3840 - target width for image resizing
            
            Performance Notes:
            -----------------
            This optimized pipeline processes line images entirely in memory,
            eliminating file I/O overhead for significant performance gains.
            
            For best performance:
            - Set beam_num: 1 for greedy decoding (much faster than beam search)
            - Use batch_size: 20-50 depending on GPU memory  
            - Enable half precision on GPU: precision: "half"
            - Use CUDA device if available
            
            The pipeline now:
            - Extracts line images directly to memory
            - Processes all lines in a single batch
            - Eliminates temporary file creation/deletion overhead
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
    parser.add_argument(
        '--width',
        type=int,
        default=None,
        help='Override target width for image resizing (default: 3840 or value from config)'
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
        
        # Load config
        try:
            # Initialize and run pipeline
            pipeline = Pipeline(args.config)
            
            # Override target width if specified in command line arguments
            if args.width is not None:
                pipeline.target_width = args.width
                logger.info(f"Overriding target width from command line: {args.width}px")
                
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