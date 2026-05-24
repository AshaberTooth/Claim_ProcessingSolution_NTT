import os
import json
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, request, render_template

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

DOCUMENTINTELLIGENCE_ENDPOINT = os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")
DOCUMENTINTELLIGENCE_KEY = os.getenv("DOCUMENTINTELLIGENCE_KEY")

AZURE_AIPROJECT_ENDPOINT = os.getenv(
    "AZURE_AIPROJECT_ENDPOINT",
    os.getenv("AZURE_FOUNDRY_PROJECT_ENDPOINT")
)

WORKFLOW_NAME = os.getenv(
    "AZURE_FOUNDRY_WORKFLOW_AGENT_NAME",
    "Claims-workflow"
)

if not DOCUMENTINTELLIGENCE_ENDPOINT:
    raise RuntimeError("Missing DOCUMENTINTELLIGENCE_ENDPOINT in .env")

if not DOCUMENTINTELLIGENCE_KEY:
    raise RuntimeError("Missing DOCUMENTINTELLIGENCE_KEY in .env")

if not AZURE_AIPROJECT_ENDPOINT:
    raise RuntimeError(
        "Missing AZURE_AIPROJECT_ENDPOINT or AZURE_FOUNDRY_PROJECT_ENDPOINT in .env"
    )

doc_client = DocumentIntelligenceClient(
    endpoint=DOCUMENTINTELLIGENCE_ENDPOINT,
    credential=AzureKeyCredential(DOCUMENTINTELLIGENCE_KEY)
)


def extract_document_text(file_path: str) -> str:
    with open(file_path, "rb") as file:
        poller = doc_client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=file
        )

    result = poller.result()
    lines = []

    if result.pages:
        for page in result.pages:
            if page.lines:
                for line in page.lines:
                    lines.append(line.content)

    return "\n".join(lines)


def extract_json_objects(text: str) -> list:
    """
    Extracts one or more JSON objects from workflow output.
    Handles output such as:
    {}{}
    or text before/after JSON.
    """
    decoder = json.JSONDecoder()
    objects = []
    index = 0

    while index < len(text):
        start = text.find("{", index)

        if start == -1:
            break

        try:
            obj, end = decoder.raw_decode(text[start:])
            objects.append(obj)
            index = start + end
        except json.JSONDecodeError:
            index = start + 1

    return objects


def dedupe_json_objects(objects: list) -> list:
    """
    Removes duplicate JSON objects from the streamed workflow output.
    """
    seen = set()
    unique = []

    for obj in objects:
        try:
            key = json.dumps(obj, sort_keys=True)
        except TypeError:
            key = str(obj)

        if key not in seen:
            seen.add(key)
            unique.append(obj)

    return unique


def build_structured_result(raw_text: str) -> dict:
    """
    Converts raw workflow text into UI-friendly sections.
    """
    json_objects = dedupe_json_objects(extract_json_objects(raw_text))

    intake_result = None
    eligibility_result = None
    adjudication_result = None
    knowledge_result = None
    final_result = None

    for obj in json_objects:
        if not isinstance(obj, dict):
            continue

        if "claim_id" in obj and "member_id" in obj:
            intake_result = obj

        elif "coverage_active" in obj and "eligibility_status" in obj:
            eligibility_result = obj

        elif "recommendation" in obj and "rationale" in obj:
            adjudication_result = obj

        elif "policy_findings" in obj or "evidence_summary" in obj:
            knowledge_result = obj

        elif "final_recommendation" in obj:
            final_result = obj

    return {
        "intake_result": intake_result,
        "eligibility_result": eligibility_result,
        "adjudication_result": adjudication_result,
        "knowledge_result": knowledge_result,
        "final_result": final_result,
        "raw_output": raw_text,
        "parsed_objects_count": len(json_objects)
    }


def invoke_foundry_workflow(document_text: str) -> dict:
    """
    Invokes Azure AI Foundry workflow using AIProjectClient + OpenAI client
    and agent_reference pattern.
    """
    credential = DefaultAzureCredential()

    client = AIProjectClient(
        endpoint=AZURE_AIPROJECT_ENDPOINT,
        credential=credential
    )

    claim_payload = {
        "document_text": document_text,
        "instruction": (
            "Run the claims MVP workflow. "
            "Return claim intake, eligibility, adjudication, policy evidence, "
            "and final claim recommendation as JSON."
        )
    }

    output_chunks = []

    with client:
        openai_client = client.get_openai_client()

        conversation = openai_client.conversations.create()

        stream = openai_client.responses.create(
            conversation=conversation.id,
            extra_body={
                "agent_reference": {
                    "name": WORKFLOW_NAME,
                    "type": "agent_reference"
                }
            },
            input=json.dumps(claim_payload, indent=2),
            stream=True,
            metadata={
                "x-ms-debug-mode-enabled": "1"
            }
        )

        for event in stream:
            event_type = getattr(event, "type", "")

            # Collect only delta text to avoid duplicate final output.
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                output_chunks.append(delta)

            elif event_type == "error":
                error = getattr(event, "error", None) or str(event)
                raise RuntimeError(f"Foundry workflow error: {error}")

    raw_output = "".join(output_chunks).strip()

    if not raw_output:
        return {
            "intake_result": None,
            "eligibility_result": None,
            "adjudication_result": None,
            "knowledge_result": None,
            "final_result": None,
            "raw_output": "Workflow completed, but no text output was returned.",
            "parsed_objects_count": 0
        }

    return build_structured_result(raw_output)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/debug/config", methods=["GET"])
def debug_config():
    return {
        "document_intelligence_endpoint": DOCUMENTINTELLIGENCE_ENDPOINT,
        "foundry_project_endpoint": AZURE_AIPROJECT_ENDPOINT,
        "workflow_name": WORKFLOW_NAME
    }


@app.route("/debug/workflow", methods=["GET"])
def debug_workflow():
    try:
        test_text = (
            "Claim ID CLM-100245. Member ID MBR-99821. "
            "Provider ID PRV-44120. Date of service 2026-05-01. "
            "Diagnosis code M54.5. Procedure code 97110. "
            "Billed amount 425 dollars. Physical therapy treatment for lower back pain."
        )

        result = invoke_foundry_workflow(test_text)

        return {
            "status": "success",
            "workflow_name": WORKFLOW_NAME,
            "result": result
        }

    except Exception as ex:
        return {
            "status": "error",
            "workflow_name": WORKFLOW_NAME,
            "error": str(ex)
        }


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return render_template("result.html", error="No file uploaded.")

    file = request.files["file"]

    if file.filename == "":
        return render_template("result.html", error="No file selected.")

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        document_text = extract_document_text(file_path)
        workflow_result = invoke_foundry_workflow(document_text)

        return render_template(
            "result.html",
            filename=file.filename,
            extracted_text=document_text,
            result=workflow_result
        )

    except Exception as ex:
        return render_template(
            "result.html",
            filename=file.filename,
            error=str(ex)
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)