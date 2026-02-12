import gradio as gr
from scanner import scan_document


interface = gr.Interface(
    fn=scan_document,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="📄 OpenCV Document Scanner",
    description="Upload a document image and get a scanned version."
)

if __name__ == "__main__":
    interface.launch()
