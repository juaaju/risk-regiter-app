"""
Schema-based LLM analyzer using JSON schema validation
"""
import json
import lmstudio as lms
import traceback
from typing import Dict, Any, Optional

# Define the JSON schema for L2SA analysis results
L2SA_ANALYSIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "l2sa_analysis",
        "schema": {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "object",
                    "properties": {
                        "categories": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "status": {"type": "boolean"},
                                                "reason": {"type": "string"}
                                            },
                                            "required": ["id", "status", "reason"]
                                        }
                                    }
                                },
                                "required": ["name", "items"]
                            }
                        },
                        "summary": {"type": "string"},
                        "risk_level": {"type": "string", "enum": ["high", "medium", "low", "unknown"]}
                    },
                    "required": ["categories", "summary", "risk_level"]
                }
            },
            "required": ["analysis"]
        }
    }
}
def fix_analysis_ids(analysis_result):
    """Pastikan format ID konsisten dalam hasil analisis"""
    if not isinstance(analysis_result, dict) or "analysis" not in analysis_result:
        return analysis_result
        
    # Untuk setiap kategori dan item, bersihkan format ID
    for category in analysis_result["analysis"].get("categories", []):
        for item in category.get("items", []):
            if "id" in item:
                # Hapus prefix seperti "id:"
                item["id"] = str(item["id"]).replace("id:", "").strip()
                
    return analysis_result

def analyze_with_schema_llm(job_description: str, ptw_type: str, db=None) -> Dict[str, Any]:
    """
    Analisis keselamatan kerja menggunakan LLM dengan validasi schema JSON
    
    Args:
        job_description: Deskripsi pekerjaan
        ptw_type: Jenis Permit to Work
        db: Database session (opsional)
        
    Returns:
        Dict berisi hasil analisis
    """
    try:
        # Buat koneksi ke LM Studio server dengan OpenAI-compatible API
        from openai import OpenAI
        
        client = OpenAI(
            base_url="http://localhost:1234/v1",  # URL server LM Studio
            api_key="lm-studio"  # API key placeholder
        )
        
        # Dapatkan checklist items dari database jika tersedia
        checklist_items_text = ""
        if db:
            from sqlalchemy import text
            # Get checklist items from database
            checklist_items = db.execute(text("SELECT id, kategori, item_checklist FROM l2sa_tha_data")).fetchall()
            
            # Group items by category
            categories = {}
            for item in checklist_items:
                if item.kategori not in categories:
                    categories[item.kategori] = []
                categories[item.kategori].append({
                    "id": str(item.id),
                    "item": item.item_checklist
                })
            
            # Format checklist items text
            for category, items in categories.items():
                checklist_items_text += f"\nCategory: {category}\n"
                for idx, item in enumerate(items):
                    checklist_items_text += f"{idx+1}. {item['item']} (ID: {item['id']})\n"
        
        # Build the prompt
        prompt = f"""
        Task: Analyze the following job description and determine which safety hazards apply.
        
        Job Type: {ptw_type}
        Job Description: {job_description}
        
        For each of the following safety items, determine if the hazard applies to this job (status=true) or not (status=false), 
        and provide a brief explanation why.
        {checklist_items_text}
        
        You MUST return a valid JSON with the exact schema provided. Do not include any explanations outside the JSON structure.
        """
        
        # Define messages for the chat completion
        messages = [
            {"role": "system", "content": "You are a safety analysis assistant that helps identify potential hazards in job tasks. You always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        # Get response with schema validation
        print("Sending request to LLM with JSON schema validation...")
        response = client.chat.completions.create(
            model="local-model",  # This will be mapped to whatever model is loaded in LM Studio
            messages=messages,
            response_format=L2SA_ANALYSIS_SCHEMA,
            temperature=0.2  # Lower temperature for more consistent output
        )
        
        # Parse the response
        content = response.choices[0].message.content
        print(f"Got response from LLM (first 100 chars): {content[:100]}...")
        
        # The content should already be valid JSON with the correct schema
        result = json.loads(content)
        result = fix_analysis_ids(result)
        
        return result
    
    except Exception as e:
        print(f"Error in analyze_with_schema_llm: {e}")
        traceback.print_exc()
        
        # Return fallback analysis
        return {
            "analysis": {
                "categories": [],
                "summary": f"Error in analysis: {str(e)}",
                "risk_level": "unknown"
            }
        }

# Fallback function that uses standard lmstudio if schema-based approach fails
def analyze_with_lmstudio_fallback(job_description: str, ptw_type: str, db=None) -> Dict[str, Any]:
    """
    Fallback to standard lmstudio approach if schema-based approach fails
    """
    try:
        # Try schema-based approach first
        result = analyze_with_schema_llm(job_description, ptw_type, db)
        
        # Check if result is valid (has categories)
        if "analysis" in result and "categories" in result["analysis"] and len(result["analysis"]["categories"]) > 0:
            return result
            
        print("Schema-based approach failed or returned empty analysis, falling back to standard LLM...")
        
        # Fallback to standard lmstudio approach
        model = lms.llm("llama-3.2-1b-instruct")  # Use appropriate model name
        
        # Prepare prompt (similar to above)
        prompt = f"""
        Task: Analyze the following job description and determine which safety hazards apply.
        
        Job Type: {ptw_type}
        Job Description: {job_description}
        
        Return a valid JSON without comments, with the following structure:
        {{
            "analysis": {{
                "categories": [
                    {{
                        "name": "Category Name",
                        "items": [
                            {{
                                "id": "item-id",
                                "status": true,
                                "reason": "Brief explanation why this applies"
                            }}
                        ]
                    }}
                ],
                "summary": "Brief summary of key hazards identified",
                "risk_level": "high/medium/low"
            }}
        }}
        """
        
        # Get response from LLM
        llm_result = model.respond(prompt)
        
        # Parse response using robust parser
        from lmstudio_parser import parse_lmstudio_json
        analysis_result = parse_lmstudio_json(llm_result)
        
        return analysis_result
        
    except Exception as e:
        print(f"Both approaches failed: {e}")
        traceback.print_exc()
        
        # Return minimal valid result
        return {
            "analysis": {
                "categories": [],
                "summary": f"Error in analysis: {str(e)}",
                "risk_level": "unknown"
            }
        }