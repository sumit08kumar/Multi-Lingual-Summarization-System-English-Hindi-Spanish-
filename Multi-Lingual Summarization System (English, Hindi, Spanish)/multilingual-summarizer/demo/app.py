"""
Gradio Frontend for Multilingual Summarization System

This module provides a user-friendly web interface for the summarization service.
"""

import gradio as gr
import requests
import json
from typing import List, Tuple, Optional
import time

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"

def check_api_status():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_supported_languages():
    """Get supported languages from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/languages", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [(lang["name"], lang["code"]) for lang in data["supported_languages"]]
    except:
        pass
    return [("English", "en"), ("Hindi", "hi"), ("Spanish", "es")]

def summarize_text(text: str, language: str, max_length: int, use_hierarchical: bool) -> Tuple[str, str]:
    """Summarize text using the API."""
    if not text.strip():
        return "Please enter some text to summarize.", ""
    
    try:
        payload = {
            "text": text,
            "language": language if language != "auto" else None,
            "max_length": max_length if max_length > 0 else None,
            "use_hierarchical": use_hierarchical
        }
        
        response = requests.post(f"{API_BASE_URL}/summarize", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Format the summary result
            summary = result["summary"]
            
            # Create metadata string
            metadata = f"""
**Language:** {result["language"]}
**Original Length:** {result["original_length"]} words
**Summary Length:** {result["summary_length"]} words
**Compression Ratio:** {result["compression_ratio"]:.2%}
            """.strip()
            
            return summary, metadata
        else:
            error_detail = response.json().get("detail", "Unknown error")
            return f"Error: {error_detail}", ""
            
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the summarization API. Please ensure the backend is running.", ""
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The text might be too long.", ""
    except Exception as e:
        return f"An error occurred: {str(e)}", ""

def batch_summarize(texts: str, language: str, max_length: int) -> Tuple[str, str]:
    """Summarize multiple texts."""
    if not texts.strip():
        return "Please enter texts to summarize (one per line).", ""
    
    # Split texts by lines and filter empty lines
    text_list = [line.strip() for line in texts.split('\n') if line.strip()]
    
    if not text_list:
        return "Please enter at least one text to summarize.", ""
    
    try:
        payload = {
            "texts": text_list,
            "language": language if language != "auto" else None,
            "max_length": max_length if max_length > 0 else None
        }
        
        response = requests.post(f"{API_BASE_URL}/summarize/batch", json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            # Format batch results
            summaries = []
            for i, summary_data in enumerate(result["summaries"]):
                summaries.append(f"**Document {i+1}:**\n{summary_data['summary']}\n")
            
            combined_summaries = "\n".join(summaries)
            
            metadata = f"**Total Documents:** {result['total_documents']}"
            
            return combined_summaries, metadata
        else:
            error_detail = response.json().get("detail", "Unknown error")
            return f"Error: {error_detail}", ""
            
    except Exception as e:
        return f"An error occurred: {str(e)}", ""

def multi_document_summarize(texts: str, language: str) -> Tuple[str, str]:
    """Create a single summary from multiple documents."""
    if not texts.strip():
        return "Please enter texts to summarize (one per line).", ""
    
    text_list = [line.strip() for line in texts.split('\n') if line.strip()]
    
    if not text_list:
        return "Please enter at least one text to summarize.", ""
    
    try:
        payload = {
            "texts": text_list,
            "language": language if language != "auto" else None
        }
        
        response = requests.post(f"{API_BASE_URL}/summarize/multi-document", json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            summary = result["summary"]
            
            metadata = f"""
**Language:** {result["language"]}
**Number of Documents:** {result["num_documents"]}
**Total Original Length:** {result.get("total_length", "N/A")} words
            """.strip()
            
            return summary, metadata
        else:
            error_detail = response.json().get("detail", "Unknown error")
            return f"Error: {error_detail}", ""
            
    except Exception as e:
        return f"An error occurred: {str(e)}", ""

def upload_and_summarize(file, language: str) -> Tuple[str, str]:
    """Summarize text from uploaded file."""
    if file is None:
        return "Please upload a text file.", ""
    
    try:
        with open(file.name, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Use the single text summarization function
        return summarize_text(content, language, 0, False)
        
    except Exception as e:
        return f"Error reading file: {str(e)}", ""

# Get supported languages
language_choices = get_supported_languages()
language_choices.insert(0, ("Auto-detect", "auto"))

# Create Gradio interface
with gr.Blocks(
    title="Multilingual Summarization System",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .tab-nav {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    """
) as demo:
    
    gr.Markdown("""
    # 🌍 Multilingual Abstractive Summarization System
    
    Generate concise, fluent summaries from text in **English**, **Hindi**, and **Spanish**.
    Powered by transformer models for high-quality abstractive summarization.
    """)
    
    # API Status indicator
    api_status = check_api_status()
    status_color = "🟢" if api_status else "🔴"
    status_text = "Connected" if api_status else "Disconnected (Demo Mode)"
    
    gr.Markdown(f"**API Status:** {status_color} {status_text}")
    
    with gr.Tabs():
        
        # Single Document Tab
        with gr.Tab("📄 Single Document"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter the text you want to summarize...",
                        lines=10,
                        max_lines=20
                    )
                    
                    with gr.Row():
                        language_single = gr.Dropdown(
                            choices=language_choices,
                            value="auto",
                            label="Language"
                        )
                        max_length_single = gr.Slider(
                            minimum=0,
                            maximum=300,
                            value=150,
                            step=10,
                            label="Max Summary Length (0 = auto)"
                        )
                        use_hierarchical = gr.Checkbox(
                            label="Use Hierarchical Summarization",
                            value=False,
                            info="Better for very long documents"
                        )
                    
                    summarize_btn = gr.Button("📝 Summarize", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    summary_output = gr.Textbox(
                        label="Summary",
                        lines=8,
                        max_lines=15
                    )
                    metadata_output = gr.Textbox(
                        label="Summary Statistics",
                        lines=4
                    )
            
            summarize_btn.click(
                fn=summarize_text,
                inputs=[text_input, language_single, max_length_single, use_hierarchical],
                outputs=[summary_output, metadata_output]
            )
        
        # Batch Processing Tab
        with gr.Tab("📚 Batch Processing"):
            with gr.Row():
                with gr.Column(scale=2):
                    batch_input = gr.Textbox(
                        label="Multiple Texts (one per line)",
                        placeholder="Enter multiple texts, one per line...",
                        lines=10,
                        max_lines=20
                    )
                    
                    with gr.Row():
                        language_batch = gr.Dropdown(
                            choices=language_choices,
                            value="auto",
                            label="Language"
                        )
                        max_length_batch = gr.Slider(
                            minimum=0,
                            maximum=300,
                            value=150,
                            step=10,
                            label="Max Summary Length (0 = auto)"
                        )
                    
                    batch_btn = gr.Button("📝 Summarize All", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    batch_output = gr.Textbox(
                        label="Individual Summaries",
                        lines=10,
                        max_lines=20
                    )
                    batch_metadata = gr.Textbox(
                        label="Batch Statistics",
                        lines=3
                    )
            
            batch_btn.click(
                fn=batch_summarize,
                inputs=[batch_input, language_batch, max_length_batch],
                outputs=[batch_output, batch_metadata]
            )
        
        # Multi-Document Tab
        with gr.Tab("🔗 Multi-Document"):
            with gr.Row():
                with gr.Column(scale=2):
                    multi_input = gr.Textbox(
                        label="Multiple Documents (one per line)",
                        placeholder="Enter multiple documents to create a single coherent summary...",
                        lines=10,
                        max_lines=20
                    )
                    
                    language_multi = gr.Dropdown(
                        choices=language_choices,
                        value="auto",
                        label="Language"
                    )
                    
                    multi_btn = gr.Button("🔗 Create Unified Summary", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    multi_output = gr.Textbox(
                        label="Unified Summary",
                        lines=8,
                        max_lines=15
                    )
                    multi_metadata = gr.Textbox(
                        label="Multi-Document Statistics",
                        lines=4
                    )
            
            multi_btn.click(
                fn=multi_document_summarize,
                inputs=[multi_input, language_multi],
                outputs=[multi_output, multi_metadata]
            )
        
        # File Upload Tab
        with gr.Tab("📁 File Upload"):
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Upload Text File",
                        file_types=[".txt", ".md"],
                        type="filepath"
                    )
                    
                    language_file = gr.Dropdown(
                        choices=language_choices,
                        value="auto",
                        label="Language"
                    )
                    
                    file_btn = gr.Button("📄 Summarize File", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    file_output = gr.Textbox(
                        label="File Summary",
                        lines=8,
                        max_lines=15
                    )
                    file_metadata = gr.Textbox(
                        label="File Statistics",
                        lines=4
                    )
            
            file_btn.click(
                fn=upload_and_summarize,
                inputs=[file_input, language_file],
                outputs=[file_output, file_metadata]
            )
    
    # Examples section
    gr.Markdown("## 📋 Example Texts")
    
    examples = [
        [
            "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term 'artificial intelligence' is often used to describe machines that mimic 'cognitive' functions that humans associate with the human mind, such as 'learning' and 'problem solving'. As machines become increasingly capable, tasks considered to require 'intelligence' are often removed from the definition of AI, a phenomenon known as the AI effect.",
            "en"
        ],
        [
            "कृत्रिम बुद्धिमत्ता (AI) मशीनों द्वारा प्रदर्शित बुद्धिमत्ता है, जो मनुष्यों और जानवरों द्वारा प्रदर्शित प्राकृतिक बुद्धिमत्ता के विपरीत है। अग्रणी AI पाठ्यपुस्तकें इस क्षेत्र को 'बुद्धिमान एजेंटों' के अध्ययन के रूप में परिभाषित करती हैं: कोई भी उपकरण जो अपने वातावरण को समझता है और ऐसे कार्य करता है जो अपने लक्ष्यों को सफलतापूर्वक प्राप्त करने की संभावना को अधिकतम करता है।",
            "hi"
        ],
        [
            "La inteligencia artificial (IA) es la inteligencia demostrada por máquinas, en contraste con la inteligencia natural mostrada por humanos y animales. Los libros de texto líderes en IA definen el campo como el estudio de 'agentes inteligentes': cualquier dispositivo que percibe su entorno y toma acciones que maximizan su oportunidad de lograr exitosamente sus objetivos.",
            "es"
        ]
    ]
    
    gr.Examples(
        examples=examples,
        inputs=[text_input, language_single],
        outputs=[summary_output, metadata_output],
        fn=lambda text, lang: summarize_text(text, lang, 150, False),
        cache_examples=False
    )
    
    gr.Markdown("""
    ---
    ### 🔧 Features:
    - **Multi-language support**: English, Hindi, Spanish
    - **Hierarchical summarization**: For very long documents
    - **Batch processing**: Summarize multiple texts at once
    - **Multi-document fusion**: Create unified summaries from multiple sources
    - **File upload**: Process text files directly
    - **Automatic language detection**: No need to specify language manually
    
    ### 💡 Tips:
    - For best results, use well-structured text with clear paragraphs
    - Hierarchical mode works better for documents longer than 1000 words
    - Multi-document mode creates a single coherent summary from multiple sources
    - The system works best with news articles, research papers, and formal documents
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


