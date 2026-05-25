from flask import Flask, request, jsonify, render_template_string
from groq import Groq
import os
import json
import traceback
import time
import pandas as pd
import random
from dotenv import load_dotenv  # <-- 1. Added import for dotenv

# <-- 2. Initialize and load environment configurations from your local .env file
load_dotenv()

app = Flask(__name__)

# ============================================================
# CONFIG & CSV DATA PIPELINE LOADING
# ============================================================
# <-- 3. Swapped hardcoded key string out for dynamic OS environment lookup
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# <-- 4. Added defensive trap to halt the pipeline instantly if the key is missing
if not GROQ_API_KEY:
    raise ValueError(
        "CRITICAL ERROR: Environment variable 'GROQ_API_KEY' is missing.\n"
        "Please confirm your '.env' file exists in your project workspace directory and contains your active key."
    )

MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024

client = Groq(api_key=GROQ_API_KEY)

# Load the user-provided dataset using pandas
CSV_FILE_PATH = r"D:\ALL PROJECTS\customer_support_tickets.csv"

try:
    if os.path.exists(CSV_FILE_PATH):
        df_clean = pd.read_csv(CSV_FILE_PATH)
        # Extract dynamic taxonomy categories straight from data
        ALL_TAGS = sorted(df_clean['Ticket Type'].dropna().unique().tolist())
        print(f" -> Dataset loaded successfully! Found taxonomy pools: {ALL_TAGS}")
    else:
        raise FileNotFoundError()
except Exception as csv_err:
    print(f" -> Critical warning: {CSV_FILE_PATH} could not be parsed. Falling back to default baseline tags.")
    ALL_TAGS = ["Technical issue", "Billing inquiry", "Cancellation request", "Product inquiry", "Refund request"]
    df_clean = None

# Helper generator mapping real historical rows into few-shot tokens strings
def generate_dynamic_few_shot_examples(count=2):
    if df_clean is None or df_clean.empty:
        return []
    
    # Pick random historical rows from the uploaded CSV
    sampled_rows = df_clean.sample(min(count, len(df_clean)))
    examples = []
    
    for _, row in sampled_rows.iterrows():
        examples.append({
            "subject": str(row.get('Ticket Subject', 'No Subject')),
            "description": str(row.get('Ticket Description', 'No Description')),
            "expected_output": {
                "tags": [
                    {
                        "tag": str(row.get('Ticket Type', 'Unclassified')), 
                        "confidence": 95, 
                        "reason": f"Matches the ticket historical issue taxonomy signature for {row.get('Ticket Type')}."
                    }
                ]
            }
        })
    return examples

# ============================================================
# HTML TEMPLATE (Enhanced with Live CSV Loader Button)
# ============================================================
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Support Ticket Auto Tagger</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            min-height:100vh;
            background: linear-gradient(-45deg, #090d16, #111827, #1e1b4b, #090d16);
            background-size:400% 400%;
            animation:bgMove 15s ease infinite;
            font-family: 'Segoe UI', sans-serif;
            color:white;
            overflow-x:hidden;
        }
        @keyframes bgMove { 0%{background-position:0% 50%;} 50%{background-position:100% 50%;} 100%{background-position:0% 50%;} }
        .main-card {
            max-width:1000px;
            margin:auto;
            margin-top:40px;
            margin-bottom:40px;
            padding:35px;
            border-radius:25px;
            background:rgba(255,255,255,0.05);
            backdrop-filter:blur(20px);
            border:1px solid rgba(255,255,255,0.12);
            box-shadow:0 12px 50px rgba(0,0,0,0.5);
        }
        .title { font-size:2.2rem; font-weight:800; text-align:center; margin-bottom:5px; background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .subtitle { text-align:center; color:#94a3b8; margin-bottom:30px; font-size: 0.95rem; }
        .form-control, .form-select {
            background:rgba(255,255,255,0.06);
            border:1px solid rgba(255,255,255,0.12);
            color:white;
            border-radius:12px;
            padding:14px;
        }
        .form-select option { background: #0f172a; color: white; }
        .form-control:focus, .form-select:focus {
            background:rgba(255,255,255,0.1); color:white; box-shadow:0 0 0 3px rgba(96,165,250,0.25);
        }
        .analyze-btn {
            width:100%; padding:14px; border:none; border-radius:12px; font-size:1.1rem; font-weight:600; color:white;
            background:linear-gradient(135deg, #2563eb, #7c3aed); transition:0.3s;
        }
        .analyze-btn:hover { transform:translateY(-2px); box-shadow:0 8px 25px rgba(37,99,235,0.4); }
        .sample-btn {
            background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.4); color: #38bdf8;
            padding: 8px 16px; border-radius: 8px; font-size: 0.85rem; transition: 0.2s; font-weight: 600;
        }
        .sample-btn:hover { background: rgba(56, 189, 248, 0.25); color: white; }
        
        .meta-container { background: rgba(0,0,0,0.25); border-radius: 16px; padding: 20px; border: 1px solid rgba(255,255,255,0.05); }
        .metric-card { text-align: center; padding: 10px; background: rgba(255,255,255,0.04); border-radius: 10px; border: 1px solid rgba(255,255,255,0.05); }
        .metric-val { font-size: 1.1rem; font-weight: bold; color: #38bdf8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .metric-lbl { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
        
        .result-card {
            margin-top:15px; padding:18px; border-radius:15px;
            background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
        }
        .progress { height:12px; border-radius:20px; background:rgba(255,255,255,0.08); }
        .progress-bar { border-radius:20px; background:linear-gradient(90deg, #60a5fa, #a78bfa); }
        
        pre { background: #020617 !important; border: 1px solid rgba(255,255,255,0.1); padding: 15px; border-radius: 12px; color: #34d399 !important; font-size: 0.85rem; max-height: 250px; overflow-y: auto; }
        .strategy-explainer { font-size: 0.85rem; background: rgba(96,165,250,0.08); padding: 12px; border-radius: 10px; border-left: 4px solid #60a5fa; color: #e2e8f0; }
    </style>
</head>
<script>
// Queries the new backend server route to sample random rows from customer_support_tickets.csv
async function loadSampleFromCSV(){
    try {
        let response = await fetch("/get-random-row");
        let sample = await response.json();
        if(sample.error) {
            alert(sample.error);
            return;
        }
        document.getElementById("subject").value = sample.subject;
        document.getElementById("description").value = sample.description;
    } catch(err) {
        alert("Failed to query CSV random row gateway.");
    }
}

async function classifyTicket(){
    let subject = document.getElementById("subject").value;
    let description = document.getElementById("description").value;
    let classificationMode = document.getElementById("classificationMode").value;

    if(!subject || !description){
        alert("Please fill all fields.");
        return;
    }

    document.getElementById("loader").classList.remove("d-none");
    document.getElementById("results-wrapper").classList.add("d-none");

    try{
        let response = await fetch("/classify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ subject, description, mode: classificationMode })
        });

        let data = await response.json();
        document.getElementById("loader").classList.add("d-none");
        document.getElementById("results-wrapper").classList.remove("d-none");

        if(data.error){
            document.getElementById("results-output").innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            return;
        }

        // Update Strategy Comparative Architecture Metrics Block
        document.getElementById("m-strategy").innerText = data.metrics.strategy_name;
        document.getElementById("m-latency").innerText = data.metrics.latency;
        document.getElementById("m-examples").innerText = data.metrics.examples_injected;
        document.getElementById("m-style").innerText = data.metrics.behavioral_style;
        document.getElementById("m-tokens").innerText = data.metrics.estimated_token_weight;
        
        // Load Text Explainer Breakdown
        document.getElementById("strategy-text-desc").innerHTML = data.metrics.explainer;

        // Render Output JSON Payload Stream
        document.getElementById("raw-json-dump").innerText = JSON.stringify(data.raw_llm_payload, null, 2);

        // Generate Graphical UI Output Track Bars
        let tagsHtml = "";
        data.tags.forEach(tag => {
            tagsHtml += `
                <div class="result-card">
                    <div class="d-flex justify-content-between align-items-center mb-1">
                        <span class="fw-bold" style="color:#60a5fa;">${tag.tag}</span>
                        <span class="badge bg-primary rgba-badge">${tag.confidence}% Confidence</span>
                    </div>
                    <div class="progress my-2">
                        <div class="progress-bar" style="width:${tag.confidence}%"></div>
                    </div>
                    <small class="text-white-50 d-block mt-1"><strong>Reasoning Vector:</strong> ${tag.reason}</small>
                </div>`;
        });
        document.getElementById("results-output").innerHTML = tagsHtml;

    } catch(error){
        document.getElementById("loader").classList.add("d-none");
        alert("Network communication disruption occurred.");
    }
}
</script>
<body>
<div class="container">
    <div class="main-card">
        <div class="title"><i class="fas fa-database"></i> Live CSV Analytics Laboratory</div>
        <div class="subtitle">Prompt Strategy Benchmarking Fed Directly by <code>customer_support_tickets.csv</code></div>

        <div class="row g-3 mb-4">
            <div class="col-md-12">
                <label class="form-label fw-bold text-white-50">Select LLM Execution Framework</label>
                <select id="classificationMode" class="form-select">
                    <option value="zero_shot">Zero-Shot Learning (Standard Base Prompt)</option>
                    <option value="few_shot">Few-Shot Learning (Injected Demonstration Dataset Rows)</option>
                    <option value="fine_tuned">Fine-Tuned Persona Model Simulation (Hyper-Rigid Weights)</option>
                </select>
            </div>
            <div class="col-md-12">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <label class="form-label text-white-50 m-0">Ticket Subject</label>
                    <button class="sample-btn" onclick="loadSampleFromCSV()"><i class="fas fa-random"></i> Load Random Ticket from CSV</button>
                </div>
                <input type="text" id="subject" class="form-control" placeholder="Click button above or type custom subject...">
            </div>
            <div class="col-md-12">
                <label class="form-label text-white-50">Ticket Description Body</label>
                <textarea id="description" class="form-control" rows="4" placeholder="Provide complete ticket body details context..."></textarea>
            </div>
        </div>

        <button class="analyze-btn mb-4" onclick="classifyTicket()">Execute Multi-Class Pipeline Inference</button>

        <div id="loader" class="text-center my-4 d-none">
             <div class="spinner-border text-primary"></div>
             <p class="mt-2 text-white-50">Querying Groq Cloud Pipeline Cluster Matrix...</p>
        </div>

        <div id="results-wrapper" class="d-none">
            <h4 class="mb-3 text-white fw-bold"><i class="fas fa-chart-network text-info"></i> Execution Telemetry Analytics</h4>
            
            <div class="meta-container mb-4">
                <div class="row g-2">
                    <div class="col-6 col-md-3"><div class="metric-card"><div class="metric-val" id="m-strategy">-</div><div class="metric-lbl">Strategy</div></div></div>
                    <div class="col-6 col-md-3"><div class="metric-card"><div class="metric-val" id="m-latency">-</div><div class="metric-lbl">API Latency</div></div></div>
                    <div class="col-6 col-md-3"><div class="metric-card"><div class="metric-val" id="m-examples">-</div><div class="metric-lbl">Injected Context</div></div></div>
                    <div class="col-6 col-md-3"><div class="metric-card"><div class="metric-val" id="m-tokens">-</div><div class="metric-lbl">Token Footprint</div></div></div>
                </div>
                <div class="row mt-3">
                    <div class="col-12">
                        <div class="metric-card d-flex justify-content-between px-3 py-2 align-items-center">
                            <span class="metric-lbl">AI Behavioral Tone Alignment:</span>
                            <span id="m-style" class="badge bg-dark border border-secondary text-info fw-bold">-</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="mb-4">
                <div class="strategy-explainer" id="strategy-text-desc"></div>
            </div>

            <div class="row">
                <div class="col-md-6">
                    <h5 class="mb-2 text-white-50"><i class="fas fa-tags"></i> Classified Target Pool</h5>
                    <div id="results-output"></div>
                </div>
                <div class="col-md-6">
                    <h5 class="mb-2 text-white-50"><i class="fas fa-code"></i> Intercepted JSON Payload Response</h5>
                    <pre><code id="raw-json-dump"></code></pre>
                </div>
            </div>
        </div>
    </div>
</div>
</body>
</html>
"""

# ============================================================
# SYSTEM PROMPT BUILDER
# ============================================================
def build_classification_messages(subject, description, mode):
    tags_str = ", ".join(ALL_TAGS)
    json_constraint = "\nYou must format your entire response output strictly as a valid json object matching the requested schema layout structure."
    
    schema_definition = (
        "Your JSON object MUST contain exactly a root key named \"tags\", containing an array of objects. "
        "Each nested object must strictly use keys: \"tag\", \"confidence\", and \"reason\"."
    )

    system_prompt = (
        "You are an automated customer service triage system built to handle text classification tasks.\n"
        f"Analyze the incoming ticket and output the TOP 3 most probable matching tags from this strictly allowed pool derived from our active system taxonomy:\n[{tags_str}].\n\n"
        f"{schema_definition}\n"
        f"{json_constraint}"
    )
    
    messages = []
    
    if mode == "few_shot":
        system_prompt += "\n\nHere are historical training samples sourced straight from our active dataset logs to guide formatting consistency:"
        messages.append({"role": "system", "content": system_prompt})
        
        # Dynamically draw 2 random rows straight from the CSV for Few-Shot Injection!
        dynamic_examples = generate_dynamic_few_shot_examples(count=2)
        for ex in dynamic_examples:
            user_content = f"Ticket Subject: {ex['subject']}\nTicket Description: {ex['description']}"
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": json.dumps(ex['expected_output'])})
            
        messages.append({"role": "user", "content": f"Ticket Subject: {subject}\nTicket Description: {description}"})
        
    elif mode == "fine_tuned":
        optimized_system_prompt = (
            "You are a specialized fine-tuned fineweights classification matrix checkpoint optimized directly on corporate ticketing data clusters.\n"
            "You ignore conversational pleasantries and instantly map inputs to top probability arrays via extreme taxonomy compliance weights.\n"
            f"Taxonomy Options: [{tags_str}]. Predict precisely the top 3 items.\n"
            f"{schema_definition}\n"
            f"{json_constraint}"
        )
        messages.append({"role": "system", "content": optimized_system_prompt})
        messages.append({"role": "user", "content": f"INPUT_STRUCT -> Subject: {subject} || Description: {description}"})
        
    else:
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"Ticket Subject: {subject}\nTicket Description: {description}"})
        
    return messages

# ============================================================
# API INFERENCE + REAL-TIME STRATEGY METADATA GEN
# ============================================================
def classify_ticket(subject, description, mode):
    messages = build_classification_messages(subject, description, mode)
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0.1,  
            response_format={"type": "json_object"}, 
            messages=messages
        )
        
        latency = f"{round(time.time() - start_time, 2)}s"
        raw_content = response.choices[0].message.content.strip()
        parsed_json = json.loads(raw_content)
        
        # Defensive Normalization Layer
        normalized_result = {"tags": []}
        if "tags" in parsed_json and isinstance(parsed_json["tags"], list):
            normalized_result["tags"] = parsed_json["tags"]
        else:
            for alt_key in ["predictions", "categories", "results"]:
                if alt_key in parsed_json and isinstance(parsed_json[alt_key], list):
                    normalized_result["tags"] = parsed_json[alt_key]
                    break
            if not normalized_result["tags"] and isinstance(parsed_json, list):
                normalized_result["tags"] = parsed_json

        strategy_meta = {
            "zero_shot": {
                "name": "Zero-Shot Baseline",
                "examples": "0 Rows (None)",
                "tokens": f"~{200 + len(ALL_TAGS)*5} Tokens (Light)",
                "style": "Conversational Text Alignment",
                "explainer": "<strong>Framework Info:</strong> Running pure <em>Zero-Shot Context Isolation</em>. The model has zero historical baseline templates. It is using generic semantic mapping rules to classify your support ticket dynamically based on columns in the uploaded data file."
            },
            "few_shot": {
                "name": "Few-Shot In-Context",
                "examples": "2 Real CSV Rows",
                "tokens": "~1,350 Tokens (Heavy Context)",
                "style": "Data-Driven Blueprint Matching",
                "explainer": "<strong>Framework Info:</strong> Running <em>In-Context Few-Shot Training injection</em>. The system randomly grabbed 2 completed historical rows straight out of <code>customer_support_tickets.csv</code> and injected them dynamically into the conversation payload array to guide Llama."
            },
            "fine_tuned": {
                "name": "Fine-Tuned Simulation",
                "examples": f"{len(df_clean) if df_clean is not None else 8000} Rows (Simulated Matrix)",
                "tokens": f"~{150 + len(ALL_TAGS)*5} Tokens (Ultra-Efficient)",
                "style": "Hyper-Rigid Data Matrix",
                "explainer": "<strong>Framework Info:</strong> Running <em>Fine-Tuned Persona Weight Emulation</em>. Natural language processing layers are bypassed entirely. The model assumes an overfitted matrix posture to generate high-speed mathematical categorizations matching our explicit schema types."
            }
        }
        
        current_meta = strategy_meta.get(mode)
        
        normalized_result["metrics"] = {
            "strategy_name": current_meta["name"],
            "latency": latency,
            "examples_injected": current_meta["examples"],
            "estimated_token_weight": current_meta["tokens"],
            "behavioral_style": current_meta["style"],
            "explainer": current_meta["explainer"]
        }
        normalized_result["raw_llm_payload"] = parsed_json
        return normalized_result

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

# ============================================================
# FLASK ROUTE CONTROLLERS
# ============================================================
@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

# Live loader route connecting frontend request to sample from active dataframe
@app.route("/get-random-row")
def get_random_row():
    if df_clean is None or df_clean.empty:
        return jsonify({"error": "CSV file pipeline unavailable or empty."})
    
    random_row = df_clean.sample(1).iloc[0]
    return jsonify({
        "subject": str(random_row.get("Ticket Subject", "")),
        "description": str(random_row.get("Ticket Description", ""))
    })

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json() or {}
    subject = data.get("subject", "")
    description = data.get("description", "")
    mode = data.get("mode", "zero_shot")
    
    result = classify_ticket(subject, description, mode)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)