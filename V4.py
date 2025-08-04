# SSL Fix and Alternative Model Loading Options
# =============================================

import ssl
import os
import urllib3
from sentence_transformers import SentenceTransformer

# Option 1: SSL Context Fix (try this first)
def load_bert_model_with_ssl_fix():
    """
    Load BERT model with SSL certificate fix
    """
    try:
        # Create unverified SSL context (temporary fix)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Disable SSL warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Load model with retry
        print("Attempting to load BERT model with SSL fix...")
        model = SentenceTransformer('all-MiniLM-L6-v2', use_auth_token=False)
        print("✓ BERT model loaded successfully")
        return model
        
    except Exception as e:
        print(f"SSL fix failed: {e}")
        return None

# Option 2: Alternative smaller model (if SSL fix doesn't work)
def load_alternative_bert_model():
    """
    Try alternative model names that might work better
    """
    
    alternative_models = [
        'paraphrase-MiniLM-L6-v2',
        'all-mpnet-base-v2',
        'paraphrase-distilroberta-base-v1'
    ]
    
    for model_name in alternative_models:
        try:
            print(f"Trying alternative model: {model_name}")
            model = SentenceTransformer(model_name)
            print(f"✓ Successfully loaded: {model_name}")
            return model
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
    
    return None

# Option 3: Offline model loading (manual download)
def setup_offline_model():
    """
    Instructions for manual model download
    """
    
    instructions = """
    MANUAL MODEL DOWNLOAD INSTRUCTIONS:
    ==================================
    
    1. Download model manually:
       - Go to: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
       - Click "Clone repository" or download files
       - Save to folder: ./models/all-MiniLM-L6-v2/
    
    2. Use local model path:
       model = SentenceTransformer('./models/all-MiniLM-L6-v2/')
    
    3. Alternative: Use different model source
    """
    
    print(instructions)

# Option 4: Network configuration fix
def configure_network_settings():
    """
    Configure network settings for corporate/restricted networks
    """
    
    # Set environment variables for proxy (if behind corporate firewall)
    # os.environ['HTTPS_PROXY'] = 'your-proxy-server:port'
    # os.environ['HTTP_PROXY'] = 'your-proxy-server:port'
    
    # Set HuggingFace cache directory
    os.environ['TRANSFORMERS_CACHE'] = './model_cache/'
    os.environ['HF_HOME'] = './model_cache/'
    
    print("Network settings configured")

# Updated BERT model loading for your script
def load_bert_model_robust():
    """
    Robust BERT model loading with multiple fallback options
    """
    
    print("Loading BERT model with robust error handling...")
    
    # Try SSL fix first
    model = load_bert_model_with_ssl_fix()
    if model:
        return model
    
    # Try alternative models
    print("Trying alternative models...")
    model = load_alternative_bert_model()
    if model:
        return model
    
    # If all fails, provide manual instructions
    print("Automatic model loading failed.")
    setup_offline_model()
    
    return None

# Replace your existing BERT model loading code with this:
# model_name = 'all-MiniLM-L6-v2'
# try:
#     bert_model = SentenceTransformer(model_name)
# except Exception as e:
#     print(f"Error loading BERT model: {e}")
#     return None

# NEW: Use this instead
bert_model = load_bert_model_robust()
if bert_model is None:
    print("Could not load any BERT model. Please check network connection.")
    return None
