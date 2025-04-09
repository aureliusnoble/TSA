import torch
import torch.cuda.amp
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

class TextRecognizer:
    """
    A streamlined OCR model for text recognition from images.
    """
    def __init__(self, 
                 model_path,
                 device=None,
                 batch_size=20,
                 precision='half',
                 beam_num=2):
        """
        Initialize the text recognizer.
        
        Args:
            model_path (str): Path to the transformer model
            device (str, optional): Device to use (cuda or cpu). Default: auto-select
            batch_size (int, optional): Batch size for processing. Default: 20
            precision (str, optional): Precision mode ('full', 'half', 'autocast'). Default: 'half'
            beam_num (int, optional): Number of beams for beam search. Default: 2
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.precision = precision
        self.beam_num = beam_num
        
        # Set device
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Adjust precision for CPU
        if self.device.startswith('cpu') and self.precision in ['half', 'autocast']:
            print("Warning: Half precision and autocast are only supported on CUDA devices. Using full precision.")
            self.precision = 'full'
            
        # Load model and processor
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        self.model = self._create_model()
        
    def _create_model(self):
        """Create and configure the model."""
        model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
        
        # Configure tokenizer settings
        model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        
        # Set beam search parameters
        model.config.max_length = 128
        model.config.no_repeat_ngram_size = 3
        model.config.num_beams = self.beam_num
        
        if self.beam_num == 1:
            model.config.length_penalty = 1
            model.config.early_stopping = False
        else:
            model.config.length_penalty = 2
            model.config.early_stopping = True
        
        # Move model to device and set precision
        model.to(self.device)
        if self.precision == 'half' and not self.device.startswith('cpu'):
            model.half()
            
        # Configure image size
        if isinstance(model.config.encoder.image_size, int):
            self.processor.current_processor.size = model.config.encoder.image_size
        elif isinstance(model.config.encoder.image_size, list):
            self.processor.current_processor.size = model.config.encoder.image_size[::-1]
        else:
            self.processor.current_processor.size = model.config.encoder.image_size
            
        return model
    
    def recognise(self, images):
        """
        Recognize text in the given images.
        
        Args:
            images: A single PIL image or a list of PIL images
            
        Returns:
            A string with recognized text if a single image is provided,
            or a list of strings if multiple images are provided.
        """
        # Handle single image case
        single_image = False
        if not isinstance(images, list):
            images = [images]
            single_image = True
            
        results = []
        
        # Process images in batches
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i+self.batch_size]
            
            # Prepare input
            pixel_values = self.processor(batch, return_tensors="pt").pixel_values.to(self.device)
            if self.precision == 'half' and not self.device.startswith('cpu'):
                pixel_values = pixel_values.half()
                
            # Run inference
            try:
                if self.precision == 'autocast' and not self.device.startswith('cpu'):
                    with torch.cuda.amp.autocast():
                        generated_ids = self.model.generate(pixel_values)
                else:
                    generated_ids = self.model.generate(pixel_values)
                    
                # Decode results
                batch_results = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                results.extend(batch_results)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    raise RuntimeError(f"CUDA out of memory: {e}")
                else:
                    raise RuntimeError(f"Error during inference: {e}")
        
        # Return single result or list of results
        return results[0] if single_image else results