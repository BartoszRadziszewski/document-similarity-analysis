import pandas as pd
import os
import sys
import platform
from pathlib import Path
from openai import OpenAI
import time
from typing import List, Dict, Optional, Tuple

# ==============================
# CONFIGURATION
# ==============================
MODEL_NAME = "gpt-4o"  # Best quality for Polish/English mixed content
FALLBACK_MODEL = "gpt-4o-mini"  # Budget-friendly fallback
MAX_PARAGRAPHS_FOR_SUMMARY = 20  # Limit for GPT processing
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# Project name - change to your project name
PROJECT_NAME = "my_document_analysis"  # ← change to your project name

# FIXED: Cross-platform base directory
if platform.system() == "Windows":
    base_dir = Path.home() / "Documents" / "DocumentAnalysis" / "Projects" / PROJECT_NAME
elif platform.system() == "Darwin":  # macOS
    base_dir = Path.home() / "Documents" / "DocumentAnalysis" / "Projects" / PROJECT_NAME
else:  # Fallback for other systems
    base_dir = Path.home() / "DocumentAnalysis" / "Projects" / PROJECT_NAME

# File paths
input_file = base_dir / "results_doc.csv"
output_file = base_dir / "cluster_summary_v3.csv"
output_folder = base_dir / "clusters_v3"

# Create output directory
output_folder.mkdir(parents=True, exist_ok=True)


# ==============================
# UTILITY FUNCTIONS
# ==============================
def validate_inputs() -> bool:
    """Validate that required files exist and setup is correct"""
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        print("   Make sure you've run the main analysis first (run_document.py)")
        return False
    
    if not api_key and client is None:
        print("Warning: No OpenAI API key found.")
        print("   Set OPENAI_API_KEY environment variable for AI summaries.")
        print("   Continuing in offline mode...")
    
    return True


def estimate_api_cost(total_paragraphs: int, model: str = MODEL_NAME) -> Tuple[float, str]:
    """Estimate approximate API cost for the analysis"""
    # Rough estimates based on average paragraph length
    avg_chars_per_paragraph = 200
    chars_per_token = 4  # approximate for GPT models
    
    total_tokens = (total_paragraphs * avg_chars_per_paragraph) // chars_per_token
    
    # Pricing estimates (as of 2024, may change)
    pricing = {
        "gpt-4o": {"input": 2.50, "output": 10.00},  # per 1M tokens
        "gpt-4o-mini": {"input": 0.15, "output": 0.60}
    }
    
    if model in pricing:
        # Assume 1:3 input:output ratio
        input_cost = (total_tokens * pricing[model]["input"]) / 1_000_000
        output_cost = (total_tokens * 0.3 * pricing[model]["output"]) / 1_000_000
        total_cost = input_cost + output_cost
        return total_cost, f"${total_cost:.3f}"
    
    return 0.0, "Unknown"


def get_language_prompt(sample_text: str) -> str:
    """Determine if text is primarily Polish or English and return appropriate prompt"""
    polish_indicators = ['że', 'się', 'nie', 'jest', 'oraz', 'także', 'który', 'która']
    polish_count = sum(1 for word in polish_indicators if word in sample_text.lower())
    
    if polish_count >= 2:
        return """
Podsumuj poniższe akapity (są podobne tematycznie) w 2-3 zdaniach.
Skup się na głównej idei, która się powtarza w tych fragmentach.
Odpowiedz w języku polskim, używając profesjonalnego stylu.

Tekst do analizy:
{text}

Streszczenie klastra {group_name}:
"""
    else:
        return """
Summarize the following paragraphs (they are thematically similar) in 2-3 sentences.
Focus on the main idea that repeats across these fragments.
Respond in English using a professional style.

Text to analyze:
{text}

Summary of cluster {group_name}:
"""


# ==============================
# CORE SUMMARIZATION FUNCTION
# ==============================
def summarize_cluster(paragraphs: List[str], group_name: str) -> str:
    """
    Generate concise cluster summary using GPT or return offline placeholder
    
    Args:
        paragraphs: List of paragraph texts in the cluster
        group_name: Name/ID of the cluster (e.g., "SIMILAR-01")
    
    Returns:
        Summary text or error message
    """
    if len(paragraphs) == 0:
        return "No data available"

    if not client:
        return f"Offline mode - no summary for {group_name}. Total paragraphs: {len(paragraphs)}"

    # Sample paragraphs to avoid overwhelming the model
    sample_paragraphs = paragraphs[:MAX_PARAGRAPHS_FOR_SUMMARY]
    sample_text = "\n---\n".join(sample_paragraphs)
    
    # Get appropriate prompt based on language detection
    prompt_template = get_language_prompt(sample_text)
    prompt = prompt_template.format(text=sample_text, group_name=group_name)
    
    # Try with primary model, fallback to cheaper model if needed
    models_to_try = [MODEL_NAME]
    if MODEL_NAME != FALLBACK_MODEL:
        models_to_try.append(FALLBACK_MODEL)
    
    for model in models_to_try:
        for attempt in range(MAX_RETRIES):
            try:
                print(f"  Generating summary for {group_name} (attempt {attempt + 1}/{MAX_RETRIES})...")
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert document analyst. Provide concise, accurate summaries of similar text passages."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                
                summary = response.choices[0].message.content.strip()
                
                # Add metadata about the cluster
                if len(paragraphs) > MAX_PARAGRAPHS_FOR_SUMMARY:
                    summary += f"\n\n[Note: Summary based on {MAX_PARAGRAPHS_FOR_SUMMARY} representative paragraphs out of {len(paragraphs)} total]"
                
                return summary
                
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed for {group_name}: {str(e)}")
                
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                elif model != models_to_try[-1]:
                    print(f"  Switching to fallback model: {FALLBACK_MODEL}")
                    break
                else:
                    return f"Summary generation failed for {group_name}: {str(e)}"
    
    return f"All summary attempts failed for {group_name}"


def save_cluster_files(df: pd.DataFrame, text_column: str) -> Dict[str, int]:
    """
    Save individual cluster files and return statistics
    
    Args:
        df: DataFrame with clustered paragraphs
        text_column: Name of the column containing paragraph text
    
    Returns:
        Dictionary with statistics about saved files
    """
    stats = {"files_created": 0, "total_paragraphs": 0}
    
    for group, subset in df.groupby("Group"):
        paragraphs = subset[text_column].tolist()
        stats["total_paragraphs"] += len(paragraphs)
        
        # Create filename-safe cluster name
        safe_group_name = group.replace("/", "_").replace("\\", "_")
        cluster_file = output_folder / f"{safe_group_name}.txt"
        
        try:
            with open(cluster_file, "w", encoding="utf-8") as f:
                # Write header with metadata
                f.write(f"=== {group} ===\n")
                f.write(f"Total paragraphs: {len(paragraphs)}\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                
                # Write paragraphs with numbering
                for i, paragraph in enumerate(paragraphs, 1):
                    f.write(f"[{i:02d}] {paragraph}\n\n")
            
            stats["files_created"] += 1
            print(f"Saved {len(paragraphs)} paragraphs to {cluster_file.name}")
            
        except Exception as e:
            print(f"Error saving {cluster_file}: {e}")
    
    return stats


# ==============================
# MAIN PROGRAM
# ==============================
def main():
    """Main analysis function"""
    print("Document Similarity Analysis v3 - AI-Powered Summarization")
    print("=" * 60)
    
    # Validate setup
    if not validate_inputs():
        sys.exit(1)
    
    try:
        # Load data
        print("Loading analysis results...")
        df = pd.read_csv(input_file, encoding='utf-8')
        print(f"   Loaded {len(df)} records from {input_file.name}")
        
        # Determine text column
        text_column = "Paragraph" if "Paragraph" in df.columns else df.columns[0