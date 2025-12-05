from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from app import build_qa
from typing import Any
import tempfile
import traceback
import os



app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
def root():
    return FileResponse("static/index.html")

qa_holder: dict[str, Any] = {"qa": None}

# allow front-end requests (use FastAPI's add_middleware API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        print("📤 Starting file upload...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            print(f"📄 Read {len(content)} bytes from uploaded file")
            tmp.write(content)
            tmp_path = tmp.name
            print("� Saved to temporary file:", tmp_path)

        print("🔄 Creating QA chain...")
        qa_holder["qa"] = build_qa(tmp_path)
        print("✅ QA chain created successfully")
        
        # Clean up the temporary file
        try:
            os.unlink(tmp_path)
            print("🗑️ Cleaned up temporary file")
        except Exception as cleanup_error:
            print("⚠️ Failed to clean up temporary file:", cleanup_error)
            
        return JSONResponse({"message": "✅ PDF uploaded and processed!"})
    except Exception as e:
        print("❌ ERROR during upload_pdf:")
        print("Error type:", type(e).__name__)
        print("Error message:", str(e))
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Failed to process PDF: {str(e)}",
                "error_type": type(e).__name__
            }
        )




@app.post("/ask")
async def ask(request: str = Form(...)):
    try:
        if qa_holder["qa"] is None:
            return JSONResponse({"reply": "⚠️ Please upload a PDF first."})
        print("🔹 Asking question:", request)
        result = qa_holder["qa"].invoke(request)
        print("✅ Invoke result:", result)
        return JSONResponse({"reply": str(result)})
    except Exception as e:
        print("❌ ERROR during ask:", e)
        traceback.print_exc()
        return JSONResponse({"error": str(e)})

@app.get("/health")
def health():
    return {"status": "ok"}
