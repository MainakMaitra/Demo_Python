# Updated Model Loading for Your Specific Downloaded Path
# =====================================================

import os
from sentence_transformers import SentenceTransformer

def load_your_downloaded_bert_model():
    """
    Load BERT model from your specific downloaded path
    """
    
    # Your exact path based on the screenshot
    model_path = "./sentence_transformers_all_miniLM-L6-v2/all-MiniLM-L6-v2"
    
    print(f"Loading BERT model from: {model_path}")
    
    # Check if the path exists
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        print("Current working directory:", os.getcwd())
        print("Available files:", os.listdir('.'))
        return None
    
    # Check for required files
    required_files = [
        'config.json',
        'pytorch_model.bin',
        'sentence_bert_config.json'
    ]
    
    print("Checking model files...")
    missing_files = []
    for file_name in required_files:
        file_path = os.path.join(model_path, file_name)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024*1024)  # Size in MB
            print(f"✅ {file_name} found ({file_size:.1f} MB)")
        else:
            missing_files.append(file_name)
            print(f"❌ {file_name} missing")
    
    if missing_files:
        print(f"Missing required files: {missing_files}")
        return None
    
    try:
        # Load the model
        print("Loading model...")
        bert_model = SentenceTransformer(model_path)
        print("✅ BERT model loaded successfully!")
        
        # Test the model with a sample sentence
        test_sentence = "This is a test sentence."
        test_embedding = bert_model.encode([test_sentence])
        print(f"✅ Model test successful! Embedding shape: {test_embedding.shape}")
        
        return bert_model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# REPLACE YOUR EXISTING BERT MODEL LOADING CODE WITH THIS:
# ========================================================

# In your bert_contrast_analysis function, replace this section:
# 
# model_name = 'all-MiniLM-L6-v2'
# try:
#     bert_model = SentenceTransformer(model_name)
#     print(f"BERT model loaded: {model_name}")
# except Exception as e:
#     print(f"Error loading BERT model: {e}")
#     return None

# WITH THIS:
print("\nStep 2: Loading BERT model from local download...")
bert_model = load_your_downloaded_bert_model()

if bert_model is None:
    print("Failed to load BERT model. Please check the model files.")
    return None

print("BERT model loaded successfully from local download!")

# Alternative: If your script is in a different location
def load_bert_with_absolute_path():
    """
    If relative path doesn't work, use absolute path
    """
    
    # Get current directory and build absolute path
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, "sentence_transformers_all_miniLM-L6-v2", "all-MiniLM-L6-v2")
    
    print(f"Trying absolute path: {model_path}")
    
    if os.path.exists(model_path):
        try:
            bert_model = SentenceTransformer(model_path)
            return bert_model
        except Exception as e:
            print(f"Error with absolute path: {e}")
    
    return None

# Debug function to check your folder structure
def debug_folder_structure():
    """
    Debug function to see what's in your folders
    """
    
    print("DEBUG: Current folder structure")
    print("=" * 40)
    
    # Check current directory
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    # Check the downloaded folder
    downloaded_folder = "./sentence_transformers_all_miniLM-L6-v2"
    if os.path.exists(downloaded_folder):
        print(f"\n✅ Found: {downloaded_folder}")
        print(f"Contents: {os.listdir(downloaded_folder)}")
        
        # Check the model subfolder
        model_subfolder = os.path.join(downloaded_folder, "all-MiniLM-L6-v2")
        if os.path.exists(model_subfolder):
            print(f"\n✅ Found: {model_subfolder}")
            print(f"Model files: {os.listdir(model_subfolder)}")
        else:
            print(f"\n❌ Not found: {model_subfolder}")
    else:
        print(f"\n❌ Not found: {downloaded_folder}")

# If you're having issues, run this first to debug:
# debug_folder_structure()
