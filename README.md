# 🎓 Advanced AI Educational Assistant

An all-in-one, multi-modal educational ecosystem powered by **GPT-4o**. This application transforms static learning materials—PDFs, images, and YouTube videos—into interactive, high-value academic assets.

Built with a professional-grade architecture, it leverages **Computer Vision**, **Natural Language Processing (NLP)**, and **Speech Synthesis** to provide a 360-degree learning experience.

Try it live:
https://educational-assistant-f887iq7vqknhfqyhuhzbr8.streamlit.app/
---

##  Core Features

###  Document Intelligence
* **Intelligent Chat:** Upload PDFs or TXT files to engage in contextual Q&A.
* **Automated Summarization:** Instantly generate executive summaries with key insights and practical takeaways.
* **Hybrid OCR Engine:** Uses a dual-layer approach (PyPDF2 + Tesseract) to extract text even from low-quality scanned documents.

###  Video Intelligence Analyzer
* **Transcript Extraction:** Seamlessly pulls transcripts from YouTube URLs (Supports standard videos, Shorts, and Embeds).
* **Study Note Generation:** Converts long-form video content into structured bullet points and summaries.
* **Direct Interaction:** Ask specific questions about video content without watching the entire duration.

###  AI Quiz Generator
* **Assessment Creation:** Transform any text or file into professional quizzes (MCQs, True/False, or Mixed).
* **Difficulty Management:** Customize assessments for Easy, Medium, or Hard levels.
* **Pedagogical Explanations:** Includes correct answers and brief explanations for every question to reinforce learning.

###  Multi-Modal Assistants
* **Visual Analyzer:** Leverages **GPT-4o Vision** to interpret complex diagrams, charts, and handwritten notes from images.
* **Voice Assistant:** Converts study material into natural speech via **gTTS**, including an AI "Enhance" feature that rewrites text to be more conversational for audio learning.

---

##  Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend** | Streamlit |
| **Core LLM** | GPT-4o (via GitHub Models API) |
| **Vision & OCR** | GPT-4o Vision / Tesseract / PIL |
| **Data Processing** | PyPDF2 / pdf2image |
| **Speech & Audio** | gTTS (Google Text-to-Speech) / BytesIO |
| **Language Support** | Langdetect (Detects Arabic & English) |
