## This is a rather expensive full test suite
## Query each graph with predefined edge labels and send all images to the LLM

import os
import uuid
import shutil
import base64
from openai import OpenAI
from rag.graph import loadgraph, entity_collapse, query
from PIL import Image
from pathlib import Path

# Initialize OpenAI client pointing to local server
client = OpenAI(
    base_url="http://10.42.0.11:18181/v1",
    api_key="not-needed"  # Local server doesn't need a real API key
)

MODEL_NAME = "NexaAI/Qwen3-VL-8B-Instruct-GGUF"

data = ["U1.tsv", "U2.tsv", "U3.tsv", "U4.tsv", 
        "F1.tsv", "F2.tsv", "F3.tsv", "F4.tsv"]

# Load all graphs
collapsed_graphs = []
for filename in data:
    path = Path("data") / filename
    collapsed_graphs.append(entity_collapse(loadgraph(str(path)), clustering_tr=0.4))

edgelist = ["Seamans Engineering"]

# Create output directories
Path("data/gemini").mkdir(exist_ok=True)
for dataset_name in data:
    dataset_prefix = dataset_name.replace(".tsv", "")
    Path(f"data/gemini/{dataset_prefix}").mkdir(exist_ok=True)

def encode_image_base64(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

for edge in edgelist:
    print(f"\nProcessing edge: {edge}")
    
    for idx, collapsed_graph in enumerate(collapsed_graphs):
        dataset_name = data[idx].replace(".tsv", "")
        
        label, image_files, _ = query(collapsed_graph, edge, return_similarity=True)
                        
        print(image_files)
        
        # Create temporary directory for scrambled images
        temp_dir = Path("workdata/temp_scrambled")
        temp_dir.mkdir(exist_ok=True)
        
        # Create scrambled filenames and copy images
        scrambled_mapping = {}
        image_paths = []
        
        for img_file in image_files:
            img_path = Path("workdata") / img_file
            if img_path.exists():
                try:
                    # Generate scrambled filename with same extension
                    scrambled_name = f"{uuid.uuid4().hex}{img_path.suffix}"
                    scrambled_path = temp_dir / scrambled_name
                    
                    # Copy image with scrambled name
                    shutil.copy2(img_path, scrambled_path)
                    scrambled_mapping[scrambled_name] = img_file
                    
                    # Store path for encoding
                    image_paths.append(scrambled_path)
                except Exception as e:
                    print(f"    Warning: Could not load {img_file}: {e}")
        
        if not image_paths:
            print(f"    No valid images found for {edge} in {dataset_name}")
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            continue
        
        # Create prompt with images for OpenAI API
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"What do you see in the attached images? Describe in detail, as specific to the image as possible. A user has related these details to {label}. Based on that, what do you think are landmark features of {edge}? Output in a list. Your final line should be list of ANS: [concept, concept, concept, concept, ...]"
                    }
                ]
            }
        ]
        
        # Add images to message
        for img_path in image_paths:
            base64_image = encode_image_base64(img_path)
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=300,
                #temperature=0.8,
                frequency_penalty=1.1,
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Save output
            output_path = Path(f"data/gemini/{dataset_name}/{edge}.out")
            with open(output_path, "w") as f:
                f.write(response_text)
            
            print(f"    ✓ Saved to {output_path}")
            print(f"    Response: {response_text}")
            
        except Exception as e:
            print(f"    Error calling model: {e}")
        finally:
            # Clean up scrambled images
            shutil.rmtree(temp_dir, ignore_errors=True)


print("\n" + "="*80)
print("PROCESSING COMPLETE")
print("="*80)
print(f"Results saved to data/gemini/[U1-U4,F1-F4,baseline]/[edge].out")
        