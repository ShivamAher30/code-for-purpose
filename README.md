# 🌟 Multirag: Multimodal Retrieval-Augmented Generation for AI-Powered Search

![GitHub stars](https://img.shields.io/github/stars/NotShrirang/Multirag?style=social)
![GitHub forks](https://img.shields.io/github/forks/NotShrirang/Multirag?style=social)
![GitHub commits](https://img.shields.io/github/commit-activity/t/NotShrirang/Multirag)
![GitHub issues](https://img.shields.io/github/issues/NotShrirang/Multirag)
![GitHub pull requests](https://img.shields.io/github/issues-pr/NotShrirang/Multirag)
![GitHub](https://img.shields.io/github/license/NotShrirang/Multirag)
![GitHub last commit](https://img.shields.io/github/last-commit/NotShrirang/Multirag)
![GitHub repo size](https://img.shields.io/github/repo-size/NotShrirang/Multirag)
<a href="https://huggingface.co/spaces/NotShrirang/Multirag"><img src="https://img.shields.io/badge/Streamlit%20App-red?style=flat-rounded-square&logo=streamlit&labelColor=white"/></a>

This project implements a Multimodal Retrieval-Augmented Generation (RAG) system, named **Multirag**, that leverages **OpenAI's CLIP** model for neural cross-modal image retrieval and semantic search, and **OpenAI's Whisper** model for audio processing. The system allows users to input text queries, images, or audio to retrieve multimodal responses seamlessly through vector embeddings. It features a comprehensive annotation interface for creating custom datasets and supports CLIP model fine-tuning with configurable parameters for domain-specific applications. The system also supports uploading images, PDFs, and audio files (including real-time recording) for enhanced interaction and intelligent retrieval capabilities through a Streamlit-based interface.

Experience the project in action:

[![Multirag Streamlit App](https://img.shields.io/badge/Streamlit%20App-red?style=for-the-badge&logo=streamlit&labelColor=white)](https://huggingface.co/spaces/NotShrirang/Multirag)

---

## 📸 Implementation Screenshots

| ![Screenshot 2025-01-01 184852](https://github.com/user-attachments/assets/ad79d0f0-d200-4a82-8c2f-0890a9fe8189) | ![Screenshot 2025-01-01 222334](https://github.com/user-attachments/assets/7307857d-a41f-4f60-8808-00d6db6e8e3e) |
| ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Data Upload Page                                                                                                 | Data Search / Retrieval                                                                                          |
|                                                                                                                  |                                                                                                                  |
| ![Screenshot 2025-01-01 222412](https://github.com/user-attachments/assets/e38273f4-426b-444d-80f0-501fa9563779) | ![Screenshot 2025-01-01 223948](https://github.com/user-attachments/assets/21724a92-ef79-44ae-83e6-25f8de29c45a) |
| Data Annotation Page                                                                                             | CLIP Fine-Tuning                                                                                                 |

---

## ✨ Features

- 🔄 **Cross-Modal Retrieval**: Search text to retrieve both text and image results using deep learning
- 🖼️ **Image-Based Search**: Search the database by uploading an image to find similar content
- 🧠 **Embedding-Based Search**: Uses OpenAI's CLIP, Whisper and SentenceTransformer's Embedding Models for embedding the input data
- 🎯 **CLIP Fine-Tuning**: Supports custom model training with configurable parameters including test dataset split size, learning rate, optimizer, and weight decay
- 🔨 **Fine-Tuned Model Integration**: Seamlessly load and utilize fine-tuned CLIP models for enhanced search and retrieval
- 📤 **Upload Options**: Allows users to upload images, PDFs and audio files for AI-powered processing and retrieval
- 🎙️ **Audio Integration**: Upload audio files or record audio directly through the interface
- 🔗 **URL Integration**: Add images directly using URLs and scrape website data including text and images
- 🕷️ **Web Scraping**: Automatically extract and index content from websites for comprehensive search capabilities
- 🏷️ **Image Annotation**: Enables users to annotate uploaded images through an intuitive interface
- 🔍 **Augmented Text Generation**: Enhances text results using LLMs for contextually rich outputs
- 🌐 **Streamlit Interface**: Provides a user-friendly web interface for interacting with the system

---

## 🗺️ Roadmap

- [x] Fine-tuning CLIP for domain-specific datasets
- [x] Image-based search and retrieval
- [x] Adding support for audeo modalities

---

## 🏗️ Architecture Overview

![Multirag Architecture](https://github.com/user-attachments/assets/dc2a2b8d-801e-42dc-8b07-089a8f8b5641)
*Architecture Diagram*

1. **Data Indexing**:

   - Text, images, and PDFs are preprocessed and embedded using the CLIP model
   - Embeddings are stored in a vector database for fast and efficient retrieval
   - Support for direct URL-based image indexing and website content scraping

2. **Query Processing**:

   - Text queries / image-based queries are converted into embeddings for semantic search
   - Uploaded images, audio files and PDFs are processed and embedded for comparison
   - The system performs a nearest neighbor search in the vector database to retrieve relevant text, images, and audio

3. **Response Generation**:

   - For text results: Optionally refined or augmented using a language model
   - For image results: Directly returned or enhanced with image captions
   - For audio results: Returned with relevant metadata and transcriptions where applicable
   - For PDFs: Extracts text content and provides relevant sections

4. **Image Annotation**:

   - Dedicated annotation page for managing uploaded images
   - Support for creating and managing multiple datasets simultaneously
   - Flexible annotation workflow for efficient data labeling
   - Dataset organization and management capabilities

5. **Model Fine-Tuning**:
   - Custom CLIP model training on annotated images
   - Configurable training parameters for optimization
   - Integration of fine-tuned models into the search pipeline

---

## 🚀 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/NotShrirang/Multirag.git
   cd Multirag
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 Usage

1. **Running the Streamlit Interface**:

   - Start the Streamlit app:

     ```bash
     streamlit run app.py
     ```

   - Access the interface in your browser to:
     - Submit natural language queries
     - Upload images or PDFs to retrieve contextually relevant results
     - Upload or record audio files
     - Add images using URLs
     - Scrape and index website content
     - Search using uploaded images
     - Annotate uploaded images
     - Fine-tune CLIP models with custom parameters
     - Use fine-tuned models for improved search results

2. **Example Queries**:
   - **Text Query**: "sunset over mountains"  
     Output: An image of a sunset over mountains along with descriptive text
   - **PDF Upload**: Upload a PDF of a scientific paper  
     Output: Extracted key sections or contextually relevant images

---

## ⚙️ Configuration

- 📊 **Vector Database**: It uses FAISS for efficient similarity search
- 🤖 **Model**: Uses OpenAI CLIP for neural embedding generation
- ✍️ **Augmentation**: Optional LLM-based augmentation for text responses
- 🎛️ Fine-Tuning: Configurable parameters for model training and optimization

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any feature requests or bug fixes.

---

## 📄 License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [OpenAI CLIP](https://openai.com/research/clip)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Hugging Face](https://huggingface.co/)
