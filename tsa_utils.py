import pandas as pd
import os
import csv
import cv2
import numpy as np
from operator import itemgetter
from PIL import Image
import shutil
import hashlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import subprocess
import random
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

class Utilities:
    def __init__(self, directory):
        self.directory = directory
        
    def set_root(self):
        current_dir = os.getcwd()

        while os.path.basename(current_dir) != self.directory and current_dir != os.path.dirname(current_dir):
            current_dir = os.path.dirname(current_dir)

        if os.path.basename(current_dir) == self.directory:
            os.chdir(current_dir)
            print(f"Changed working directory to: {os.getcwd()}")
        else:
            print("The specified root directory could not be found.")
            

class FileProcessor:
    def __init__(self, input_dir=None, output_dir=None, output_sample_dir=None, output_test_dir=None, debug=False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_sample_dir = output_sample_dir
        self.output_test_dir = output_test_dir
        self.supported_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.jp2', '.pdf'}
        self.debug = debug
        self.processed_files = set()
        self.failed_files = set()

        for set_name in ['test', 'train', 'validate']:
            if self.output_dir:
                os.makedirs(os.path.join(self.output_dir, set_name), exist_ok=True)
            if self.output_sample_dir:
                os.makedirs(os.path.join(self.output_sample_dir, set_name), exist_ok=True)

    def log(self, message):
        if self.debug:
            print(message)

    def list_files(self):
        """List all files in the input directory and its subdirectories."""
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                yield os.path.join(root, file)

    def is_supported_file(self, file_path):
        """Check if the file is a supported image type."""
        return os.path.splitext(file_path)[1].lower() in self.supported_extensions

    def convert_png_jpg_tiff(self, file_path):
        """Convert PNG, JPG, or TIFF to BGR image."""
        self.log(f"Attempting to read file: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {file_path}")
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def convert_jp2(self, file_path):
        """Convert JP2 to BGR image."""
        self.log(f"Attempting to read JP2 file: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        jp2 = glymur.Jp2k(file_path)
        img = jp2[:]
        if img is None:
            raise ValueError(f"Failed to read JP2 image: {file_path}")
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img

    def convert_pdf(self, file_path):
        """Convert first page of PDF to BGR image."""
        self.log(f"Attempting to read PDF file: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        pages = convert_from_path(file_path, dpi=300, first_page=1, last_page=1)
        if not pages:
            raise ValueError(f"No pages found in PDF: {file_path}")
        img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
        return img

    def convert_file(self, file_path):
        """Convert a file to BGR image based on its type."""
        self.log(f"Converting file: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()
        file_path_full = os.path.join(self.input_dir, file_path)

        if ext in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}:
            return self.convert_png_jpg_tiff(file_path_full)
        elif ext == '.jp2':
            return self.convert_jp2(file_path_full)
        elif ext == '.pdf':
            return self.convert_pdf(file_path_full)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def convert_images(self):
        """Process all supported files in the input directory and save to output directory."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        all_files = list(self.list_files())
        for file_path in tqdm(all_files, desc="Processing files"):
            if self.is_supported_file(file_path):
                try:
                    img = self.convert_file(file_path)
                    
                    # Create output path
                    rel_path = os.path.relpath(file_path, self.input_dir)
                    output_path = os.path.join(self.output_dir, os.path.splitext(rel_path)[0] + '.png')
                    
                    # Ensure output directory exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Save the image
                    cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    self.log(f"Successfully converted and saved: {output_path}")
                    self.processed_files.add(file_path)
                except Exception as e:
                    self.log(f"Error processing {file_path}: {e}")
                    self.failed_files.add(file_path)
            else:
                self.log(f"Skipping unsupported file: {file_path}")

        self.validate_conversion(all_files)

    def validate_conversion(self, all_files):
        """Validate the conversion process and list files that weren't processed."""
        unprocessed_files = set(all_files) - self.processed_files - self.failed_files
        
        print("\nConversion Summary:")
        print(f"Total files: {len(all_files)}")
        print(f"Successfully processed: {len(self.processed_files)}")
        print(f"Failed to process: {len(self.failed_files)}")
        print(f"Unprocessed files: {len(unprocessed_files)}")

        if self.failed_files:
            print("\nFailed files:")
            for file in self.failed_files:
                print(f"  - {file}")

        if unprocessed_files:
            print("\nUnprocessed files:")
            for file in unprocessed_files:
                print(f"  - {file}")


    def search_files(self, search_dir):
        """
        Search and copy files from the `search_dir` directory and its subdirectories, which are also present 
        in the `self.input_dir`, into the `self.output_dir` directory. 
        The directory structure of `search_dir` is not maintained.
        """
        # Getting a list of files (without extensions) from the input directory
        input_files = set(os.path.splitext(file)[0] for file in self.list_files(full_path=False))
        
        # A set to keep track of found files
        found_files = set()
        
        # Iterating through the search directory and its subdirectories
        for dirpath, _, filenames in os.walk(search_dir):
            for filename in filenames:
                # Check if the file (ignoring its extension) is in the input files
                file_base_name = os.path.splitext(filename)[0]
                if file_base_name in input_files:
                    # Add the found file to found_files set
                    found_files.add(file_base_name)
                    
                    # Preparing the path for file copy
                    source_path = os.path.join(dirpath, filename)
                    destination_path = os.path.join(self.output_dir, filename)
                    
                    # Handling potential file name collisions
                    counter = 1
                    while os.path.exists(destination_path):
                        base, ext = os.path.splitext(filename)
                        destination_path = os.path.join(self.output_dir, f"{base}_{counter}{ext}")
                        counter += 1
                    
                    # Copying the file
                    shutil.copy2(source_path, destination_path)
        
        # Check for files that were not found and print warning messages
        not_found_files = input_files - found_files
        for file_base_name in not_found_files:
            print(f"Warning: file not found {file_base_name}")
                    
    def list_files(self, only_images=False, full_path=True):
        """Return a list of file paths in the given directory and its subdirectories.
        If only_images is True, filter to include only image files."""
        file_paths = []
        for dirpath, _, filenames in os.walk(self.input_dir):
            for filename in filenames:
                file_path = os.path.relpath(os.path.join(dirpath, filename), self.input_dir)
                if only_images:
                    _, extension = os.path.splitext(file_path)
                    if extension.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        file_paths.append(file_path if full_path else filename)
                else:
                    file_paths.append(file_path if full_path else filename)
        return file_paths

    def delete_files(self, directory, files_list):
        """
        Delete files with specific names (ignoring extensions) from a directory and its subdirectories, 
        provided the files have either .png, .jpg or .jpeg extensions.
        """

        # Count to keep track of number of files deleted
        delete_count = 0

        # Allowed extensions
        allowed_extensions = ['.png', '.jpg', '.jpeg']

        # Convert files_list names to lowercase without extensions for comparison
        files_without_ext = [os.path.splitext(name)[0].lower() for name in files_list]

        # Walk through the directory
        for foldername, subfolders, filenames in os.walk(directory):
            for filename in filenames:
                name_without_extension, file_extension = os.path.splitext(filename)


                # Updated matching logic for case insensitivity
                if name_without_extension.lower() in files_without_ext and file_extension in allowed_extensions:
                    file_path = os.path.join(foldername, filename)
                    try:
                        os.remove(file_path)
                        delete_count += 1
                    except Exception as e:
                        print(f"Error deleting {file_path}. Reason: {e}")

        print(f"{delete_count} files deleted.")


    def rearrange_subdirs(self, top_dir):
        """Rearrange files from subdirectories into the main directories."""
        
        # Step 1: Navigate to the given top-level directory
        for subdir_name in os.listdir(top_dir):  # Iterate over each subdirectory at the top level
            subdir_path = os.path.join(top_dir, subdir_name)
            
            if os.path.isdir(subdir_path):  # Make sure it's a directory and not a file
                
                # Step 2a: Traverse into its subdirectories and collect the paths to all files
                for dirpath, _, filenames in os.walk(subdir_path):
                    for filename in filenames:
                        file_path = os.path.join(dirpath, filename)
                        
                        # Step 2b: Move each of these files to the root of its top-level directory
                        if dirpath != subdir_path:  # Don't move if it's already in the top-level directory
                            new_path = os.path.join(subdir_path, filename)
                            if os.path.exists(new_path):  # Handle potential naming collisions
                                base, ext = os.path.splitext(filename)
                                counter = 1
                                while os.path.exists(os.path.join(subdir_path, f"{base}_{counter}{ext}")):
                                    counter += 1
                                new_path = os.path.join(subdir_path, f"{base}_{counter}{ext}")
                            
                            shutil.move(file_path, new_path)

                # Step 2c: Remove empty subdirectories
                for dirpath, dirnames, filenames in os.walk(subdir_path, topdown=False):  # topdown=False to traverse from innermost to outermost directories
                    for dirname in dirnames:
                        try:
                            os.rmdir(os.path.join(dirpath, dirname))  # Try to remove, but will fail if not empty which is fine
                        except:
                            pass
                            
    def _resize_image(self, file_path, image_height, image_width):
        img = cv2.imread(os.path.join(self.input_dir, file_path))
    
        if img is None:
            print(f"Error reading image: {os.path.join(self.input_dir, file_path)}")
            return None, file_path
    
        if image_width:
            aspect_ratio = img.shape[1] / img.shape[0]  # width/height
            new_height = int(image_width / aspect_ratio)
            img_resized = cv2.resize(img, (image_width, new_height))
        elif image_height:
            aspect_ratio = img.shape[0] / img.shape[1]  # height/width
            new_width = int(image_height * aspect_ratio)
            img_resized = cv2.resize(img, (new_width, image_height))
        else:
            return img, file_path
        
        return img_resized, file_path
            
    def resize_images(self, batch_size=50, image_height=None, image_width=None):
        file_paths = self.list_files(only_images=True)
    
        # Batching
        for i in tqdm(range(0, len(file_paths), batch_size), desc="Batch Resizing Images"):
            batch_file_paths = file_paths[i:i+batch_size]
    
            # Multi-threading
            with ThreadPoolExecutor() as executor:
                # Create a new function that includes image_height and image_width as arguments
                resize_func = lambda file_path: self._resize_image(file_path, image_height, image_width)
                results = list(executor.map(resize_func, batch_file_paths))
            
            for img_resized, file_path in results:
                # Check file extension
                file_ext = os.path.splitext(file_path)[-1].lower()
    
                # Write the resized image
                if self.output_dir:
                    os.makedirs(os.path.dirname(os.path.join(self.output_dir, file_path)), exist_ok=True)
                    if file_ext == '.jpg':
                        cv2.imwrite(os.path.join(self.output_dir, file_path), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    elif file_ext == '.png':
                        cv2.imwrite(os.path.join(self.output_dir, file_path), img_resized, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                else:
                    if file_ext == '.jpg':
                        cv2.imwrite(os.path.join(self.input_dir, file_path), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    elif file_ext == '.png':
                        cv2.imwrite(os.path.join(self.input_dir, file_path), img_resized, [cv2.IMWRITE_PNG_COMPRESSION, 3])


            
    def _convert_image(self, file_path):
        img = cv2.imread(os.path.join(self.input_dir, file_path))
        file_path = os.path.splitext(file_path)[0] + '.png'
        return img, file_path
        
    def convert_to_png(self, batch_size=50, replace=False):
        file_paths = self.list_files(only_images=True)

        # Batching
        for i in tqdm(range(0, len(file_paths), batch_size), desc="Batch Converting Images"):
            batch_file_paths = file_paths[i:i+batch_size]

            # Multi-threading
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(self._convert_image, batch_file_paths))
            
            for img, file_path in results:
                if self.output_dir:
                    os.makedirs(os.path.dirname(os.path.join(self.output_dir, file_path)), exist_ok=True)
                    cv2.imwrite(os.path.join(self.output_dir, file_path), img)
                else:
                    cv2.imwrite(os.path.join(self.input_dir, file_path), img)
                
                # Delete the original .jpg file if replace is True
                if replace:
                    original_file_path = os.path.join(self.input_dir, os.path.splitext(file_path)[0] + '.jpg')
                    if os.path.exists(original_file_path):
                        os.remove(original_file_path)


                

    @staticmethod
    def hash_file_name(file_name):
        """Hash the given file name using MD5 and return the hexadecimal string."""
        hash_object = hashlib.md5(file_name.encode())
        return hash_object.hexdigest()

    @staticmethod
    def assign_file(file_name):
        """Assign the given file name to a set (train, validation, or test) based on its hash."""
        hash_value = FileProcessor.hash_file_name(file_name)
        last_digit = int(hash_value[-1], 16) % 10
        if last_digit < 6:
            return 'train'
        elif last_digit < 8:
            return 'validate'
        else:
            return 'test'

    def copy_files_to_sets(self, files, output_dir):
        """Copy the given files to their assigned set directories, maintaining subdirectory structure."""
        data = []
        for file_path in files:
            file_name = os.path.basename(file_path)
            relative_dir = os.path.dirname(file_path)
            set_name = self.assign_file(file_name)
            hash_name = self.hash_file_name(file_name)
            old_file_path = os.path.join(self.input_dir, file_path)
            new_dir_path = os.path.join(output_dir, set_name, relative_dir)
            os.makedirs(new_dir_path, exist_ok=True)
            new_file_path = os.path.join(new_dir_path, set_name + '_' + hash_name + '.jpg')
            shutil.copy2(old_file_path, new_file_path)
            data.append({'set': set_name, 'hash': hash_name, 'file_path': file_path})
        df = pd.DataFrame(data)
        return df

    def create_csv_file(self, df, directory, csv_file_name):
        """Create a CSV file with the given DataFrame."""
        df.to_csv(os.path.join(directory, csv_file_name), index=False)

    def process_files(self, csv_file_name):
        """Process all files in the input directory and write their information to a CSV file."""
        if os.path.exists(os.path.join(self.output_dir, 'data_info.csv')):
            print('Log file already exists, skipping processing')
            return
            
        files = self.list_files()
        df = self.copy_files_to_sets(files, self.output_dir)
        self.create_csv_file(df, self.output_dir, csv_file_name)
        
        print(f"\nSplitting Files into Sets:")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Total number of files: {len(files)}")
        
        for set_name in ['test', 'train', 'validate']:
            set_count = len(df[df['set'] == set_name])
            set_percent = (set_count / len(files)) * 100
            print(f"Number of files in {set_name}: {set_count} ({set_percent:.2f}%)")


    def process_sampled_files(self, sample_sizes, log_filename):
        """Process the sampled files and write their information to a CSV file."""
        
        if os.path.exists(os.path.join(self.output_sample_dir, 'data_info.csv')):
            print('Log file already exists, skipping sampling')
            return
        files = self.list_files()

        print(f"\nSampling Files:")
        print(f"Sample output directory: {self.output_sample_dir}")
        print(f"Total number of files: {len(files)}")

        set_counts = {'test': 0, 'train': 0, 'validate': 0}
        data_info = []
        sampled_files = []

        for set_name, size in sample_sizes.items():
            set_files = [file for file in files if file.startswith(set_name)]
            set_sample = random.sample(set_files, size)
            sampled_files.extend(set_sample)
            set_counts[set_name] = len(set_sample)

            for file in set_sample:
                hash_value = os.path.splitext(os.path.basename(file))[0].split('_')[-1]
                
                # Prepare the new directory path
                new_dir_path = os.path.join(self.output_sample_dir, file)
                os.makedirs(os.path.dirname(new_dir_path), exist_ok=True)

                # Copy the file
                shutil.copy(os.path.join(self.input_dir, file), new_dir_path)

                # Append the data info
                data_info.append({
                    'set': set_name,
                    'hash': hash_value,
                    'file_path': file  # Note that 'file' already includes the set_name
                })

        print(f"Number of sampled files: {len(sampled_files)} ({(len(sampled_files)/len(files))*100:.2f}%")

        # Create a dataframe and save it to a CSV file
        df = pd.DataFrame(data_info)
        df.to_csv(os.path.join(self.output_sample_dir, 'data_info.csv'), index=False)

        # Print the file counts and percentages for each set
        for set_name in ['test', 'train', 'validate']:
            print(f"Number of files in {set_name} set: {set_counts[set_name]}")
            print(f"Percent of total sample files in {set_name} set: {(set_counts[set_name] / len(sampled_files)) * 100:.2f}%")

            
    def load_image(self, file_path):
        """Load an image using cv2."""
        return cv2.imread(f"{self.input_dir}/{file_path}", cv2.IMREAD_COLOR)

    def save_image(self, image, file_path, testing=False, suffix=''):
        """Save the image to the output directory with the same structure as the input."""
        relative_path = os.path.relpath(file_path)

        # Split the file path into directory path, file name and extension
        file_dir, file_name = os.path.split(relative_path)
        base_name, file_extension = os.path.splitext(file_name)

        # Add suffix to the file name if provided
        if suffix:
            base_name = base_name + "_" + suffix

        # Create the new file name
        relative_path = os.path.join(file_dir, base_name + file_extension)

        #If we have set a test directory, then output to that instead.
        if self.output_test_dir and testing == True:
            output_dir = self.output_test_dir
            relative_path = os.path.basename(relative_path)

        #Otherwise just output to output_dir.
        else:
            output_dir = self.output_dir

        new_file_path = os.path.join(output_dir, relative_path)

        print(f"Output Path: {new_file_path}")

        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        cv2.imwrite(new_file_path, image)

class PageClassification:
    """A class to classify images into types."""
    
    def __init__(self, main_dir = None):
        self.main_dir = main_dir
        
    def page_classification(self):
        main_directory_path = self.main_dir
        
        os.makedirs(main_directory_path, exist_ok=True)
        subfolders = ["test", "train", "validate"]
        combined_file_list = []
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(main_directory_path, subfolder)

            for root, dirs, files in os.walk(subfolder_path):
                for file in files:
                    if file != ".DS_Store":
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, main_directory_path)

                        parts = relative_path.split("/")
                        if len(parts) >= 4:
                            location = parts[2]
                            period = parts[3]
                        else:
                            location = ""
                            period = ""
                    
                        years = period.split("-")
                        if len(years) >= 2:
                            start_year = int(years[0])
                            end_year = int(years[1])
                        else:
                            start_year = 0
                            end_year = 0
                    
                        if start_year <= 1823:
                            num_type = "1"
                        elif start_year >= 1825:
                            num_type = "2"
                        elif start_year == 1824:
                            if location == "Brinon-sur-Beuvron":
                                num_type = "1"
                            elif "Brinon" in location:
                                num_type = "1"
                            elif location == "Corbigny":
                                num_type = "1"
                            elif location == "Saint-Benin-d'Azy":
                                num_type = "2"
                            elif "Benin" in location:
                                num_type = "2"
                            elif location == "Saint-Saulge":
                                num_type = "2"
                            elif "Saulge" in location:
                                num_type = "2"
                            else:
                                num_type = "Not defined"
                        else:
                            num_type = "Not defined"
                    
                        if (location=="Château-Chinon" and period=="1823-1825"):
                            str_type = "B"
                        elif ("Chinon" in location and period=="1823-1825"):
                            str_type = "B"
                        elif (location=="Corbigny" and period=="1867-1878"):
                            str_type = "B"
                        elif (location=="Fours" and (period=="1809-1821" or period=="1822-1824")):
                            str_type = "B"
                        elif (location=="La Charité-sur-Loire" and (period=="1812-1816" or period=="1816-1826" or period=="1869-1876")):
                            str_type = "B"
                        elif ("La Charit" in location and (period=="1812-1816" or period=="1816-1826" or period=="1869-1876")):
                            str_type = "B"
                        elif (location=="Montsauche" and (period=="1791-1818" or period=="1814-1824")):
                            str_type = "B"
                        elif (location=="Nevers" and (period=="1813-1823" or period=="1868-1871")):
                            str_type = "B"
                        elif (location=="Pouilly-sur-Loire" and (period=="1793-1803" or period=="1800-1816" or period=="1816-1825")):
                            str_type = "B"
                        elif ("Pouilly" in location and (period=="1793-1803" or period=="1800-1816" or period=="1816-1825")):
                            str_type = "B"
                        elif (location=="Saint-Benin-d'Azy" and period=="1821-1824"):
                            str_type = "B"
                        elif ("Benin" in location and period=="1821-1824"):
                            str_type = "B"
                        elif location=="Saint-Sulpice":
                            str_type = "B"
                        elif ("Sulpice" in location and period=="1821-1824"):
                            str_type = "B"
                        elif (location=="Tannay" and period=="1823-1827"):
                            str_type = "B"
                        else:
                            str_type = "A"
                            
                        sub_type = ""
                        
                        parts = period.split("-")
                        start_year = int(parts[0])
                        end_year = int(parts[1])
                        
                        if (num_type == "2" and str_type == "A"):
                            sub_type = "a"
                            if (location == "Brinon-sur-Beuvron" and (period=="1840-1854" or period=="1866-1882")):
                                sub_type = "b"
                            elif ("Brinon" in location and (period=="1840-1854" or period=="1866-1882")):
                                sub_type = "b"
                            if (location == "Château-Chinon" and start_year >= 1840):
                                sub_type = "b"
                            elif ("Chinon" in location and start_year >= 1840):
                                sub_type = "b"
                            if (location == "Châtillon-en-Bazois" and start_year >= 1843):
                                sub_type = "b"
                            elif ("Bazois" in location and start_year >= 1843):
                                sub_type = "b"
                            if (location == "Clamecy" and start_year >= 1850):
                                sub_type = "b"
                            if (location == "Corbigny" and start_year >= 1847):
                                sub_type = "b"
                            if (location == "Cosne-sur-Loire" and start_year >= 1849):
                                sub_type = "b"
                            elif ("Cosne" in location and start_year >= 1849):
                                sub_type = "b"
                            if (location == "Decize" and start_year >= 1847):
                                sub_type = "b"
                            if (location == "Donzy" and (period=="1847-1857" or period=="1866-1872")):
                                sub_type = "b"
                            if location == "Dornes":
                                sub_type = "b"
                            if (location == "Fours" and start_year >= 1844):
                                sub_type = "b"
                            if (location == "La Charité-sur-Loire" and start_year >= 1839):
                                sub_type = "b"
                            elif ("Charit" in location and start_year >= 1839):
                                sub_type = "b"
                            if (location == "Lormes" and start_year >= 1837):
                                sub_type = "b"
                            if (location == "Luzy" and start_year >= 1846):
                                sub_type = "b"
                            if (location == "Montsauche" and (period=="1851-1860" or period=="1860-1870" or period=="1870-1881")):
                                sub_type = "b"
                            if (location == "Moulins-Engilbert" and start_year >= 1849):
                                sub_type = "b"
                            elif ("Engilbert" in location and start_year >= 1849):
                                sub_type = "b"
                            if location == "Nevers":
                                sub_type = "b"
                            if (location == "Pouilly-sur-Loire" and start_year >=1842):
                                sub_type = "b"
                            elif ("Pouilly" in location and start_year >=1842):
                                sub_type = "b"
                            if (location == "Prémery" and start_year >= 1858):
                                sub_type = "b"
                            elif ("mery" in location and start_year >= 1858):
                                sub_type = "b"
                            if (location == "Saint-Amand-en-Puisaye" and start_year >= 1858):
                                sub_type = "b"
                            elif ("Amand" in location and start_year >= 1858):
                                sub_type = "b"
                            if (location == "Saint-Benin-d'Azy" and start_year >= 1842):
                                sub_type = "b"
                            elif ("Benin" in location and start_year >= 1842):
                                sub_type = "b"
                            if (location == "Saint-Pierre-le-Moûtier" and start_year >= 1861):
                                sub_type = "b"
                            elif ("Pierre" in location and start_year >= 1861):
                                sub_type = "b"
                            if (location == "Saint-Saulge" and start_year >= 1848):
                                sub_type = "b"
                            elif ("Saulge" in location and start_year >= 1848):
                                sub_type = "b"
                            if (location == "Tannay" and start_year >= 1855):
                                sub_type = "b"
                            if (location == "Varzy" and start_year >= 1849):
                                sub_type = "b"
                    
                        table_type = num_type + str_type + sub_type

                        combined_file_list.append((file, relative_path, table_type))

        csv_file_path = os.path.join(main_directory_path, "page_classification.csv")
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['File', 'Relative Path', 'Type'])
            csv_writer.writerows(combined_file_list)

        print(f"CSV file created at: {csv_file_path}")
        
class ImageUtilities:
    pass
    
class ImageVisualiser:
    pass

class ImagePreProcessor:
    """A class for preprocessing images for OCR."""

    def __init__(self, image=None, file_processor=None, log_file = None):
        """
        Initialize the ImagePreProcessor.

        Args:
            image: The image to be preprocessed.
        """
        self.image = image
        self.file_processor = file_processor
        self.log_file = log_file
        
    def write_log(self, message):
        """Append a message to the log file."""
        with open(self.log_file, 'a') as log:
            log.write(message + "\n")

    def read_log(self):
        """Read the log file and return a list of processed images."""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as log:
                return log.read().splitlines()
        else:
            return []

    def clahe(self, clip_limit=2.0, tile_grid_size=(8,8)):
        image_paths = self.file_processor.list_files(only_images=True)

        for image_path in tqdm(image_paths, desc='Equalising'):
            input_image_path = os.path.join(self.file_processor.input_dir, image_path)
            output_image_path = os.path.join(self.file_processor.output_dir, image_path)

            image = cv2.imread(input_image_path)
        
            # Step 1: Convert the image to Lab color space
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            
            # Step 2: Split the Lab image into L, a and b channels
            L, a, b = cv2.split(lab_image)
            
            # Step 3: Apply CLAHE to the L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            clahed_L = clahe.apply(L)
            
            # Step 4: Merge the CLAHE enhanced L channel with the original a and b channel
            merged_lab_image = cv2.merge([clahed_L, a, b])
            
            # Step 5: Convert back to BGR color space
            processed_image = cv2.cvtColor(merged_lab_image, cv2.COLOR_Lab2BGR)
            
            # Step 6: Save the processed image
            cv2.imwrite(output_image_path, processed_image)


    
    def binarize_gan(self, enhance_dir, command, de_gan_dir):
        """Enhance all images using the enhance.py script and the given command."""
        processed_images = self.read_log()
        image_paths = self.file_processor.list_files(only_images=True)
        
        print(f"Skipping {len(processed_images)} images")
        
        for image_path in tqdm(image_paths, desc='Binarizing'):
            if image_path in processed_images:
                continue
    
            # Construct the full paths
            input_image_path = os.path.join(self.file_processor.input_dir, image_path)
            output_jpg_path = os.path.join(self.file_processor.output_dir, image_path)
    
            input_image_path = os.path.abspath(input_image_path)
            output_jpg_path = os.path.abspath(output_jpg_path)
    
            # Create the directory if it does not exist
            os.makedirs(os.path.dirname(output_jpg_path), exist_ok=True)
    
            # Set up the subprocess command
            subprocess_cmd = ['python', enhance_dir, command, input_image_path, output_jpg_path]
    
            # Call the subprocess
            subprocess.run(subprocess_cmd, stdout=subprocess.DEVNULL, cwd=de_gan_dir)
                        
            self.write_log(image_path)
            
    def dewarp(self, params=None):
        """Dewarp all images using the page-dewarp script with the given parameters."""

        default_params = {
            "d": 0,
            "x": 10,
            "y": 10,
            "tw": 2,
            "th": 2,
            "dpi": 300,
            "nb": 1,
        }

        if params is None:
            params = default_params
        else:
            params = {**default_params, **params}

        processed_images = self.read_log()
        image_paths = self.file_processor.list_files(only_images=True)

        for image_path in tqdm(image_paths, desc='Dewarping'):
            if image_path in processed_images:
                continue

            input_image_path = os.path.join(self.file_processor.input_dir, image_path)
            output_image_path = os.path.join(self.file_processor.output_dir, image_path)

            input_image_path = os.path.abspath(input_image_path)
            output_image_path = os.path.abspath(output_image_path)

            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            
            # Load the image
            image = cv2.imread(input_image_path)

            # Compute the scaling factors
            h, w = image.shape[:2]
            scale = 8000.0 / h

            # Resize the image
            image_resized = cv2.resize(image, (int(w * scale), int(h * scale)))

            # Save the resized image back to the original path
            cv2.imwrite(input_image_path, image_resized)

            command = ["page-dewarp"]
            for key, value in params.items():
                command.extend([f"-{key}", str(value)])
            command.append(input_image_path)

            # Save the current working directory and change to input image directory
            original_cwd = os.getcwd()
            os.chdir(os.path.dirname(input_image_path))

            subprocess.run(command, check=True)

            # Change the current working directory back to the original
            os.chdir(original_cwd)

            base, ext = os.path.splitext(input_image_path)
            output_filename = f"{base}_thresh.png"
            output_image_path_dewarp = os.path.join(self.file_processor.output_dir, os.path.basename(output_filename))
            subprocess.run(["mv", output_filename, output_image_path_dewarp], check=True)

            self.write_log(image_path)



    def binarize(self,image, binarization='threshold', thresh_min=124, thresh_max=255, adaptive_blocksize=11, adaptive_constant=2):
        """
        Binarize the image using Otsu's thresholding.
        """
        if binarization == 'otsu':
            thresh, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif binarization == 'binary':
            thresh, binary = cv2.threshold(image, thresh_min, thresh_max, cv2.THRESH_BINARY)
            
        elif binarization == 'adaptive_mean':
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, adaptive_blocksize, adaptive_constant)
            
        elif binarization == 'adaptive_gaussian':
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_blocksize, adaptive_constant)
            
        elif binarization == 'su':
            binary = self.binarize_su(image)
            
        else:
            raise 

        return binary
        
    def gaussian_blur(self, ksize=(5,5)):
        blurred = cv2.GaussianBlur(self.image, ksize, 0)
        self.image = blurred
        
    def median_blur(self, size=3):
        blurred = cv2.medianBlur(self.image, size)
        self.image = blurred
             
    def denoise(self, h=10, templateWindowSize=7, searchWindowSize=21):
        """
        Remove noise from the image using Non-local Means Denoising.

        Args:
            h: The strength of the denoising.
            templateWindowSize: The size of the template window.
            searchWindowSize: The size of the search window.
        """
        image_paths = self.file_processor.list_files(only_images=True)

        for image_path in tqdm(image_paths, desc='Denoising'):
            input_image_path = os.path.join(self.file_processor.input_dir, image_path)
            output_image_path = os.path.join(self.file_processor.output_dir, image_path)

            image = cv2.imread(input_image_path)
            
            denoised = cv2.fastNlMeansDenoising(image, h=h, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)

            
            cv2.imwrite(output_image_path, denoised)


    def binarize_su(self, gamma=0.25):
        W = 5
        horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (W, 1))
        vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, W))
        I_min = cv2.erode(cv2.erode(self.image, horiz), vert)
        I_max = cv2.dilate(cv2.dilate(self.image, horiz), vert)
        diff = I_max - I_min
        C = diff.astype(np.float32) / (I_max + I_min + 1e-16)

        alpha = (self.image.std() / 128.0) ** gamma
        C_a = alpha * C + (1 - alpha) * diff
        _, C_a_bw = cv2.threshold(C_a.astype('uint8'), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        binarized = (255-C_a_bw)
        
        self.image = binarized

        
    def morph_close(self, ksize=(1,1)):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        closed = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
        self.image = closed

    def grayscale(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = gray

    def apply_morphology(self, operation='dilate', ksize=(1,1), iterations = 5):
        """
        Apply a morphological operation (dilation or erosion) to the image.

        Args:
            operation: The morphological operation to apply ('dilate' or 'erode').
            ksize: The size of the structuring element.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)

        if operation == 'erode':
            morphed = cv2.dilate(self.image, kernel, iterations = iterations)
        elif operation == 'dilate':
            morphed = cv2.erode(self.image, kernel, iterations = iterations)
        else:
            raise ValueError("Operation should be either 'dilate' or 'erode'")

        self.image = morphed

    def equalize_histogram(self, clipLimit=2.0, tileGridSize=(8,8)):
        """
        Equalize the histogram of the image using CLAHE.

        Args:
            clipLimit: Threshold for contrast limiting.
            tileGridSize: Size of grid for histogram equalization. Input image will be divided into equally sized rectangular tiles.
        """
        # Create a CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

        if len(self.image.shape) > 2:
            # If the image is color, convert it to LAB
            lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)

            # Split the LAB image into L, A and B channels
            l, a, b = cv2.split(lab)

            # Apply CLAHE to the L channel
            l2 = clahe.apply(l)

            # Merge the CLAHE enhanced L channel with the original A and B channel
            lab = cv2.merge((l2,a,b))

            # Convert the image back to BGR
            equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # If the image is grayscale, just apply CLAHE
            equalized = clahe.apply(self.image)

        self.image = equalized
        
    def image_process_plot(self, image, processed_image):
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        plt.title('Processed Image')

        plt.show()
                 

class ImageDewarp:
    pass

class TableSegment:
    def combine_polygons(self, polygons, keys):
        combined_polygons = []
    
        for key in keys:
            combined_polygons.extend(polygons[key])
        
        return combined_polygons
    
    def get_bounding_boxes(self, polygons):
        bounding_boxes = []

        for polygon_info in polygons:
            polygon = polygon_info['polygon']
            polygon_arr = np.array(polygon)
            x, y, w, h = cv2.boundingRect(polygon_arr)

            if w > 2000 or h > 2000:
                bounding_boxes.append((x, y, w, h))

        return bounding_boxes
    
    def adjust_bounding_boxes(self, boxes, orientation):
        OVERLAP_THRESHOLD = 0.1
        if orientation == 'row':
            axis = 1
            boxes.sort(key=itemgetter(1))

        elif orientation =='column':
            axis = 0
            boxes.sort(key=itemgetter(0))

        adjusted_boxes = []

        for current_box in boxes:
            if not adjusted_boxes:
                adjusted_boxes.append(current_box)
            else:
                previous_box = adjusted_boxes[-1]
                previous_end = previous_box[axis] + previous_box[axis + 2]
                current_start = current_box[axis]

                if previous_end > current_start:
                    overlap = previous_end - current_start
                    if overlap > previous_box[axis + 2] * OVERLAP_THRESHOLD:
                        new_size = previous_box[axis + 2] - overlap
                        adjusted_boxes[-1] = (
                            previous_box[0],
                            previous_box[1],
                            new_size if orientation == 'column' else previous_box[2],
                            previous_box[3] if orientation == 'column' else new_size
                        )
                        current_start = previous_box[axis] + new_size
                    else:
                        current_start = previous_end

                adjusted_boxes.append((
                    current_start if orientation == 'column' else current_box[0],
                    current_box[1] if orientation == 'column' else current_start,
                    current_box[2],
                    current_box[3]
                ))

        return adjusted_boxes
    
    def add_rows(self, bbox_row_adj):
        # Sort rows by vertical position (y-coordinate)
        bbox_row_adj = sorted(bbox_row_adj, key=lambda box: box[1])  # Sort by 'y'

        # Define a threshold for maximum allowable gap
        max_gap = 30  # Adjust this value based on your page dimensions

        # Create a new list to hold adjusted rows
        new_bbox_row_adj = []

        # Iterate through the rows and check for gaps
        for i in range(len(bbox_row_adj) - 1):
            new_bbox_row_adj.append(bbox_row_adj[i])  # Add the current row

            # Get the bottom of the current row and top of the next row
            _, current_y, _, current_h = bbox_row_adj[i]
            current_bottom = current_y + current_h

            next_x, next_y, next_w, next_h = bbox_row_adj[i + 1]
            next_top = next_y

            # Check if the gap between rows exceeds the threshold
            if next_top - current_bottom > max_gap:
                # Add a new row to fill the gap
                new_row = (0, current_bottom, width, next_top - current_bottom)
                new_bbox_row_adj.append(new_row)

        # Add the last row
        new_bbox_row_adj.append(bbox_row_adj[-1])

        # Use the updated list of rows
        bbox_row_adj = new_bbox_row_adj

        return bbox_row_adj
    
    def add_rows(self, bbox_row_adj, width):
        # Sort rows by vertical position (y-coordinate)
        bbox_row_adj = sorted(bbox_row_adj, key=lambda box: box[1])  # Sort by 'y'

        # Define a threshold for maximum allowable gap
        max_gap = 30  # Adjust this value based on your page dimensions

        # Create a new list to hold adjusted rows
        new_bbox_row_adj = []

        # Iterate through the rows and check for gaps
        for i in range(len(bbox_row_adj) - 1):
            new_bbox_row_adj.append(bbox_row_adj[i])  # Add the current row

            # Get the bottom of the current row and top of the next row
            _, current_y, _, current_h = bbox_row_adj[i]
            current_bottom = current_y + current_h

            next_x, next_y, next_w, next_h = bbox_row_adj[i + 1]
            next_top = next_y

            # Check if the gap between rows exceeds the threshold
            if next_top - current_bottom > max_gap:
                # Add a new row to fill the gap
                new_row = (0, current_bottom, width, next_top - current_bottom)
                new_bbox_row_adj.append(new_row)

        # Add the last row
        new_bbox_row_adj.append(bbox_row_adj[-1])

        # Use the updated list of rows
        bbox_row_adj = new_bbox_row_adj

        return bbox_row_adj
    
    def find_grid_cells(self, row_boxes, col_boxes):
        # Sort row boxes by y and col boxes by x
        row_boxes.sort(key=lambda box: box[1])  # Sort by y coordinate
        col_boxes.sort(key=lambda box: box[0])  # Sort by x coordinate

        grid_cells = {}
        row_num = 1

        for row_box in row_boxes:
            col_num = 1
            for col_box in col_boxes:
                x1, y1, w1, h1 = row_box
                x2, y2, w2, h2 = col_box

                # Calculate the coordinates of the bottom-right corner
                x1_br, y1_br = x1 + w1, y1 + h1
                x2_br, y2_br = x2 + w2, y2 + h2

                # Calculate the coordinates of the intersection rectangle
                x_inter = max(x1, x2)
                y_inter = max(y1, y2)
                x_inter_br = min(x1_br, x2_br)
                y_inter_br = min(y1_br, y2_br)

                # Calculate the width and height of the intersection rectangle
                w_inter = x_inter_br - x_inter
                h_inter = y_inter_br - y_inter

                # Check if there is an overlap and it forms a valid rectangle
                if w_inter > 0 and h_inter > 0:
                    grid_cells[f'col{col_num}_row{row_num}'] = (x_inter, y_inter, w_inter, h_inter)

                col_num += 1
            row_num += 1

        return grid_cells
    
    def assign_cell(self, directory, grid_cells):
        file_names = os.listdir(directory)
        # Process each file
        renamed_files = []

        for file_name in file_names:
            # Parse the coordinates and dimensions from the filename
            base_filename = os.path.splitext(file_name)[0]

            page_index = base_filename.find("page")
            if page_index == -1:
                print(f"Skipping file as 'page' not found in filename: {file_name}")
                continue
            # Since the naming conventions have changed, we only look at the part of the file name after "page":
            page_info = base_filename[page_index + len("page"):]
            info = page_info.split('_')

            y, x = int(info[1][1:]), int(info[2][1:])
            w, h = int(info[3][1:]), int(info[4][1:])

            # Calculate the center y and slightly adjusted x
            y_pos = y + h / 2
            x_pos = x + 10

            matched = False  # Flag to check if a match is found

            # Find a way to deal with subimages that are not assigned to a specific row/column, because they fall on an empty one.
            # Find the grid cell containing this point
            for key, (cell_x, cell_y, cell_w, cell_h) in grid_cells.items():
                if cell_x <= x_pos < cell_x + cell_w and cell_y <= y_pos < cell_y + cell_h:
                    matched = True  # Update flag on match
                    # Create new filename with the grid cell key as prefix
                    new_file_name = f"{key}_{file_name}"
                    old_file_path = os.path.join(directory, file_name)
                    new_file_path = os.path.join(directory, new_file_name)

                    if matched != True:
                      # Match to closest one.
                      print(f"File {file_name} cannot be matched.")

                    # Rename the file
                    os.rename(old_file_path, new_file_path)
                    renamed_files.append(new_file_name)
                    break
            

        return renamed_files

class LineDetection:
    def extract_textlines(self, image, polygons, output_dir, filename, model, padding):
        textline_polygons = polygons[1]
        #base_filename = os.path.splitext(filename)[0]
        #output_subfolder = os.path.join(output_dir, base_filename)
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        output_subfolder = os.path.join(output_dir, base_filename)
        os.makedirs(output_subfolder, exist_ok=True)
    
        for polygon_info in textline_polygons:
            polygon = polygon_info['polygon']
            polygon_arr = np.array(polygon_info['polygon'])
            x, y, w, h = cv2.boundingRect(polygon_arr)
            sub_image = image[y:y+h+padding, x:x+w+padding] # I added a padding around the box.
        
            key = f"y{y}_x{x}_w{w}_h{h}"
        
            output_filename = f"{base_filename}_{key}.png"
            output_path = os.path.join(output_subfolder, output_filename)
            cv2.imwrite(output_path, sub_image)

class HandwrittenTextRecognition:
    def filename_parse(self, filename):
        parts = filename.split('_')

        column = next(part[3:] for part in parts if part.startswith('col'))
        row = next(part[3:] for part in parts if part.startswith('row'))
        y = next(part[1:] for part in parts if part.startswith('y'))
        x = next(part[1:] for part in parts if part.startswith('x'))
        w = next(part[1:] for part in parts if part.startswith('w'))
        h = next(part.split('.')[0][1:] for part in parts if part.startswith('h'))
    
        return column, row, y, x, w, h
    
    def assign_cell(self, directory, grid_cells):
        file_names = os.listdir(directory)
        # Process each file
        for file_name in file_names:
            # Parse the coordinates and dimensions from the filename
            base_filename = os.path.splitext(file_name)[0]

            page_index = base_filename.find("page")
            if page_index == -1:
                print(f"Skipping file as 'page' not found in filename: {file_name}")
                continue
            # Since the naming conventions have changed, we only look at the part of the file name after "page":
            page_info = base_filename[page_index + len("page"):]
            info = page_info.split('_')

            y, x = int(info[1][1:]), int(info[2][1:])
            w, h = int(info[3][1:]), int(info[4][1:])

            # Calculate the center y and slightly adjusted x
            y_pos = y + h / 2
            x_pos = x + 10

            matched = False  # Flag to check if a match is found

            # Find a way to deal with subimages that are not assigned to a specific row/column, because they fall on an empty one.
            renamed_files = []
            # Find the grid cell containing this point
            for key, (cell_x, cell_y, cell_w, cell_h) in grid_cells.items():
                if cell_x <= x_pos < cell_x + cell_w and cell_y <= y_pos < cell_y + cell_h:
                    matched = True  # Update flag on match
                    # Create new filename with the grid cell key as prefix
                    new_file_name = f"{key}_{file_name}"
                    old_file_path = os.path.join(directory, file_name)
                    new_file_path = os.path.join(directory, new_file_name)

                    if matched != True:
                      # Match to closest one.
                      print(f"File {file_name} cannot be matched.")

                    # Rename the file
                    os.rename(old_file_path, new_file_path)
                    renamed_files.append(new_file_name)
                    break

        return renamed_files
    
    def transcribe_subimage(self, sub_image, directory, processor, model, device):
        # Load the Image
        image = Image.open(f'{directory}/{sub_image}').convert("RGB") 
        # Get Inputs
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        # Generate Tokens
        generated_ids = model.generate(pixel_values, num_beams=1, max_length=128, length_penalty = None, early_stopping = False)
        # Convert to Text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text
    
class PageReconstruction:
    def page_classification(self, main_directory_path, db_path):
        os.makedirs(main_directory_path, exist_ok=True)
        combined_file_list = []
  
        for root, dirs, files in os.walk(main_directory_path):
            for file in files:
                if file != ".DS_Store":
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, main_directory_path)

                    parts = relative_path.split("/")
                    if len(parts) >= 3:
                        location = parts[0]
                        period = parts[1]
                    else:
                        location = ""
                        period = ""
                    
                    years = period.split("-")
                    if len(years) >= 2:
                        start_year = int(years[0])
                        end_year = int(years[1])
                    else:
                        start_year = 0
                        end_year = 0
                    
                    if start_year <= 1823:
                        num_type = "1"
                    elif start_year >= 1825:
                        num_type = "2"
                    elif start_year == 1824:
                        if location == "Brinon-sur-Beuvron":
                            num_type = "1"
                        elif "Brinon" in location:
                            num_type = "1"
                        elif location == "Corbigny":
                            num_type = "1"
                        elif location == "Saint-Benin-d'Azy":
                            num_type = "2"
                        elif "Benin" in location:
                            num_type = "2"
                        elif location == "Saint-Saulge":
                            num_type = "2"
                        elif "Saulge" in location:
                            num_type = "2"
                        else:
                            num_type = "Not defined"
                    else:
                        num_type = "Not defined"
                    
                    if (location=="Château-Chinon" and period=="1823-1825"):
                        str_type = "B"
                    elif ("Chinon" in location and period=="1823-1825"):
                        str_type = "B"
                    elif (location=="Corbigny" and period=="1867-1878"):
                        str_type = "B"
                    elif (location=="Fours" and (period=="1809-1821" or period=="1822-1824")):
                        str_type = "B"
                    elif (location=="La Charité-sur-Loire" and (period=="1812-1816" or period=="1816-1826" or period=="1869-1876")):
                        str_type = "B"
                    elif ("La Charit" in location and (period=="1812-1816" or period=="1816-1826" or period=="1869-1876")):
                        str_type = "B"
                    elif (location=="Montsauche" and (period=="1791-1818" or period=="1814-1824")):
                        str_type = "B"
                    elif (location=="Nevers" and (period=="1813-1823" or period=="1868-1871")):
                        str_type = "B"
                    elif (location=="Pouilly-sur-Loire" and (period=="1793-1803" or period=="1800-1816" or period=="1816-1825")):
                        str_type = "B"
                    elif ("Pouilly" in location and (period=="1793-1803" or period=="1800-1816" or period=="1816-1825")):
                        str_type = "B"
                    elif (location=="Saint-Benin-d'Azy" and period=="1821-1824"):
                        str_type = "B"
                    elif ("Benin" in location and period=="1821-1824"):
                        str_type = "B"
                    elif location=="Saint-Sulpice":
                        str_type = "B"
                    elif ("Sulpice" in location and period=="1821-1824"):
                        str_type = "B"
                    elif (location=="Tannay" and period=="1823-1827"):
                        str_type = "B"
                    else:
                        str_type = "A"
                            
                    sub_type = ""
                        
                    parts = period.split("-")
                    start_year = int(parts[0])
                    end_year = int(parts[1])
                        
                    if (num_type == "2" and str_type == "A"):
                        sub_type = "a"
                        if (location == "Brinon-sur-Beuvron" and (period=="1840-1854" or period=="1866-1882")):
                            sub_type = "b"
                        elif ("Brinon" in location and (period=="1840-1854" or period=="1866-1882")):
                            sub_type = "b"
                        if (location == "Château-Chinon" and start_year >= 1840):
                            sub_type = "b"
                        elif ("Chinon" in location and start_year >= 1840):
                            sub_type = "b"
                        if (location == "Châtillon-en-Bazois" and start_year >= 1843):
                            sub_type = "b"
                        elif ("Bazois" in location and start_year >= 1843):
                            sub_type = "b"
                        if (location == "Clamecy" and start_year >= 1850):
                            sub_type = "b"
                        if (location == "Corbigny" and start_year >= 1847):
                            sub_type = "b"
                        if (location == "Cosne-sur-Loire" and start_year >= 1849):
                            sub_type = "b"
                        elif ("Cosne" in location and start_year >= 1849):
                            sub_type = "b"
                        if (location == "Decize" and start_year >= 1847):
                            sub_type = "b"
                        if (location == "Donzy" and (period=="1847-1857" or period=="1866-1872")):
                            sub_type = "b"
                        if location == "Dornes":
                            sub_type = "b"
                        if (location == "Fours" and start_year >= 1844):
                            sub_type = "b"
                        if (location == "La Charité-sur-Loire" and start_year >= 1839):
                            sub_type = "b"
                        elif ("Charit" in location and start_year >= 1839):
                            sub_type = "b"
                        if (location == "Lormes" and start_year >= 1837):
                            sub_type = "b"
                        if (location == "Luzy" and start_year >= 1846):
                            sub_type = "b"
                        if (location == "Montsauche" and (period=="1851-1860" or period=="1860-1870" or period=="1870-1881")):
                            sub_type = "b"
                        if (location == "Moulins-Engilbert" and start_year >= 1849):
                            sub_type = "b"
                        elif ("Engilbert" in location and start_year >= 1849):
                            sub_type = "b"
                        if location == "Nevers":
                            sub_type = "b"
                        if (location == "Pouilly-sur-Loire" and start_year >=1842):
                            sub_type = "b"
                        elif ("Pouilly" in location and start_year >=1842):
                            sub_type = "b"
                        if (location == "Prémery" and start_year >= 1858):
                            sub_type = "b"
                        elif ("mery" in location and start_year >= 1858):
                            sub_type = "b"
                        if (location == "Saint-Amand-en-Puisaye" and start_year >= 1858):
                            sub_type = "b"
                        elif ("Amand" in location and start_year >= 1858):
                            sub_type = "b"
                        if (location == "Saint-Benin-d'Azy" and start_year >= 1842):
                            sub_type = "b"
                        elif ("Benin" in location and start_year >= 1842):
                            sub_type = "b"
                        if (location == "Saint-Pierre-le-Moûtier" and start_year >= 1861):
                            sub_type = "b"
                        elif ("Pierre" in location and start_year >= 1861):
                            sub_type = "b"
                        if (location == "Saint-Saulge" and start_year >= 1848):
                            sub_type = "b"
                        elif ("Saulge" in location and start_year >= 1848):
                            sub_type = "b"
                        if (location == "Tannay" and start_year >= 1855):
                            sub_type = "b"
                        if (location == "Varzy" and start_year >= 1849):
                            sub_type = "b"

                    table_type = num_type + str_type + sub_type

                    combined_file_list.append((file, relative_path, table_type))

        csv_file_path = os.path.join(db_path, "/(TSA)Tables_des_Successions_et_Absences/Core/Code/Document_Analysis/page_classification.csv")
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['File', 'Relative Path', 'Type'])
            csv_writer.writerows(combined_file_list)

        print(f"CSV file created at: {csv_file_path}")
        
    def reassemble_pages(self, df_pred, filename, df_pages, df_tables):
    
        # Find table type:
        file_type = df_pages[df_pages['File']==filename]['Type']
        table_type = file_type.iloc[0]
        matched_rows = df_tables[df_tables['table_type'] == table_type]
  
        # Create a dictionary of column names:
        col_ref = dict(zip(matched_rows['col_num'], matched_rows['col_name']))
  
        # Map the column names on the transcription using the dictionary:
        df_pred['col_name'] = df_pred['column'].astype(int).map(col_ref)

        df_pred_final = df_pred[['filename', 'row', 'col_name', 'transcription']]

        return df_pred_final
    
    def export_transcription(self, df_export, output_folder, muncipality, year, filename):

        export_folder =  os.path.join(output_folder,'pages', muncipality,year )
        os.makedirs(export_folder, exist_ok=True)
        # Clean up the filename to remove extra spaces and newline characters
        cleaned_filename = filename.strip().replace(" ", "_").replace("\n", "").replace(".jpg", "")

        # Construct the full export path
        export_path = os.path.join(export_folder, f"{cleaned_filename}.csv")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        # Export to CSV
        df_export.to_csv(export_path, index=False)

class Evaluation:
    #Probably want to separate HTR, Table, Line Evaluation
    pass