import google.generativeai as genai
import os

# Global variable to cache the working model name so we don't query every time
_WORKING_MODEL_NAME = None

# -----------------------------------------------------------------------------
# Agent Configuration
# -----------------------------------------------------------------------------
def init_agent():
    """
    Initializes the Google Gemini API.
    Looks for 'gemini_key.txt' in the root directory.
    """
    if os.path.exists("gemini_key.txt"):
        with open("gemini_key.txt", "r") as f:
            key = f.read().strip()
        try:
            genai.configure(api_key=key)
            return True
        except Exception as e:
            print(f"Agent Config Error: {e}")
            return False
    return False

def get_best_model_name():
    """
    Dynamically finds a working model name from the user's API key.
    Prioritizes Flash -> Pro -> any available generative model.
    """
    global _WORKING_MODEL_NAME
    
    # Return cached model if we already found one
    if _WORKING_MODEL_NAME:
        return _WORKING_MODEL_NAME

    try:
        # 1. List all models available to this specific API Key
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        if not available_models:
            return None

        # 2. Priority Logic: Try to find the best match
        # We prefer Flash for speed, then Pro, then anything else.
        priority_keywords = ['flash', 'pro', 'gemini']
        
        for keyword in priority_keywords:
            for model in available_models:
                if keyword in model.lower():
                    _WORKING_MODEL_NAME = model
                    print(f"✅ Auto-selected Agent Model: {model}")
                    return model
        
        # 3. Fallback: Just take the first one found
        _WORKING_MODEL_NAME = available_models[0]
        print(f"⚠️ Fallback Agent Model: {_WORKING_MODEL_NAME}")
        return _WORKING_MODEL_NAME

    except Exception as e:
        print(f"Error listing models: {e}")
        # Last resort fallback if list_models fails (e.g. permission issues)
        return 'models/gemini-1.5-flash'

def try_generate(prompt):
    """
    Helper to try generating content using the auto-discovered model.
    """
    model_name = get_best_model_name()
    if not model_name:
        return False, "No available Gemini models found for this API Key."

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return True, response.text
    except Exception as e:
        # If the auto-selected model fails, clear cache so we retry next time
        global _WORKING_MODEL_NAME
        _WORKING_MODEL_NAME = None 
        return False, f"Model {model_name} failed: {str(e)}"

def get_available_models():
    """
    Diagnostics: Lists models actually available to this API key.
    """
    try:
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models.append(m.name)
        return models
    except Exception as e:
        return [f"Could not list models: {str(e)}"]

def generate_ecg_report(sqi_metrics, num_beats, duration, sampling_rate):
    """
    Sends calculated metrics to Gemini to generate a research summary.
    """
    prompt = f"""
    You are 'Dr. Signal', an expert Biosignal Research Assistant.
    
    Data Summary:
    - Duration: {duration:.2f}s | Rate: {sampling_rate}Hz | Beats: {num_beats}
    - Skewness: {sqi_metrics['skew_avg']:.2f} (Target > 2.0)
    - Kurtosis: {sqi_metrics['kurt_avg']:.2f} (Target > 5.0)
    
    Task:
    1. Assess signal quality (Clean vs Noisy).
    2. Suggest filters if noisy.
    3. Write a 2-sentence lab notebook summary.
    """
    
    success, text = try_generate(prompt)
    
    if success:
        model_used = get_best_model_name()
        return text + f"\n\n*(Generated via {model_used})*"
    
    return f"**Agent Error:** {text}"

def analyze_data_context(df_head, columns):
    """
    General triage of the dataset structure.
    """
    prompt = f"""
    Analyze this dataset structure:
    Columns: {list(columns)}
    Preview:
    {df_head.to_markdown()}
    
    Briefly identify the likely sensor type.
    """
    
    success, text = try_generate(prompt)
    if success: return text
    
    return "Agent unavailable (Model Error)."