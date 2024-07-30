import gradio as gr

def greet(name, intensity):
    return "Hello " * intensity + name + "!", "How are you?"

demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text", "text"],
)

demo.launch()