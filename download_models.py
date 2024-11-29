import os
import shutil
import zipfile
import requests
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager
from tqdm import tqdm

@dataclass
class TransferConfig:
    """Configuration parameters for network transfer operations"""
    chunk_size: int = 1024 * 1024  # 1MB chunks for optimal network buffer utilization
    timeout: int = 60  # Extended timeout for large transfers
    verify_ssl: bool = True  # SSL verification flag

class ModelTransferManager:
    """
    Manages large-scale model transfer operations with atomic guarantees
    
    This class implements a complete pipeline for downloading and installing
    large model archives with proper resource management and atomic operations.
    """
    
    def __init__(self, config: Optional[TransferConfig] = None):
        self.config = config or TransferConfig()
    
    def _download_from_drive(self, file_id: str, destination: Path) -> None:
        """
        Implements resilient large-file transfer from Google Drive
        
        Args:
            file_id: Google Drive file identifier
            destination: Target path for downloaded content
        """
        direct_url = "https://drive.usercontent.google.com/download"
        params = {
            'id': file_id,
            'export': 'download',
            'confirm': 't'
        }
        
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        try:
            response = session.get(
                direct_url,
                params=params,
                stream=True,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )
            response.raise_for_status()
            
            file_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f:
                with tqdm(
                    total=file_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading model archive",
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                        size = f.write(chunk)
                        pbar.update(size)
                        
        except requests.RequestException as e:
            if destination.exists():
                destination.unlink()
            raise RuntimeError(f"Download failed: {str(e)}")
    
    def _extract_and_replace(self, archive_path: Path, target_dir: Path) -> None:
        """
        Performs atomic extraction and replacement of model directory
        
        Args:
            archive_path: Path to downloaded model archive
            target_dir: Target installation directory for models
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            print(f"\nExtracting to temporary location: {temp_path}")
            
            try:
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    # Validate archive structure
                    top_level_dirs = {
                        path.split('/')[0] 
                        for path in zip_ref.namelist() 
                        if '/' in path
                    }
                    if 'models' not in top_level_dirs:
                        raise ValueError("Invalid archive structure: missing 'models' directory")
                    
                    # Extract with progress tracking
                    total_files = len(zip_ref.filelist)
                    for idx, member in enumerate(zip_ref.filelist, 1):
                        zip_ref.extract(member, temp_path)
                        print(f"Extracting: {idx}/{total_files} files", end='\r')
                print("\nExtraction completed")
                
            except zipfile.BadZipFile:
                raise RuntimeError("Archive corruption detected")
            
            extracted_models = temp_path / "models"
            if not extracted_models.exists():
                raise RuntimeError("Extraction validation failed")
            
            # Implement atomic replacement
            print(f"Replacing existing models at {target_dir}")
            backup_dir = target_dir.with_name(f"{target_dir.name}_backup")
            
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
                
            if target_dir.exists():
                target_dir.rename(backup_dir)
                
            try:
                shutil.move(str(extracted_models), str(target_dir))
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
            except Exception as e:
                if backup_dir.exists():
                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    backup_dir.rename(target_dir)
                raise RuntimeError(f"Replacement failed: {str(e)}")
    
    def execute_pipeline(self, file_id: str, model_dir: Path = Path("models")) -> None:
        """
        Executes complete model acquisition and installation pipeline
        
        Args:
            file_id: Google Drive file identifier
            model_dir: Target installation directory (default: ./models)
        """
        download_path = Path("models_download.zip")
        
        try:
            print(f"Initiating model transfer pipeline")
            self._download_from_drive(file_id, download_path)
            self._extract_and_replace(download_path, model_dir)
            
            print(f"Cleaning up temporary files")
            download_path.unlink()
            
            print(f"Model installation completed successfully at {model_dir}")
            
        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            if download_path.exists():
                download_path.unlink()
            raise

if __name__ == "__main__":
    # Pipeline configuration
    config = TransferConfig(
        chunk_size=1024 * 1024,  # 1MB chunks
        timeout=120,  # 2-minute timeout
        verify_ssl=True
    )
    
    # Initialize manager
    manager = ModelTransferManager(config)
    
    # Execute pipeline with specific file ID
    try:
        manager.execute_pipeline(
            file_id="1EthRe0ondtlnlxjlzNFgZBg-Z70zazgg",
            model_dir=Path("models")
        )
    except Exception as e:
        print(f"Fatal error: {str(e)}")