import gradio as gr
from backend.app import build_qa

#Gradio UI
def create_interface():
    with gr.Blocks(title="Mini AI PDF Assistant") as demo:
        gr.Markdown("# Mini AI Assistant\nChat with any PDF you upload.")

# upload + status section

        pdf_file = gr.File(label="Upload a PDF")
        status_box = gr.Textbox(label="Status", interactive=False)

#chat section
        question_box = gr.Textbox(label="Ask a question about your PDF")
        answer_box = gr.Textbox(label="Answer", interactive=False)

#Event buildings
        pdf_file.upload(upload_pdf, inputs=pdf_file, outputs=status_box)
        question_box.submit(ask_questions, inputs=question_box, outputs=answer_box)



    return demo
qa_holder = {"qa": None}

def upload_pdf(pdf_file):
    qa_holder["qa"] = build_qa(pdf_file.name)
    return "PDF loaded! You can start asking questions."

def ask_questions(question):
    if qa_holder["qa"] is None:
        return "Please upload a PDF first."
    result = qa_holder["qa"].invoke(question)
    return result if isinstance(result, str) else result.get("result", str(result))



#launch
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)

