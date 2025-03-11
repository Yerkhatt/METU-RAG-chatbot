# METU-RAG-chatbot
A retrieval-augmented generation (RAG) chatbot designed for Middle East Technical University (METU). This system was developed during the applied NLP course and leverages a knowledge base extracted from the 'metu.edu.tr' subdomains. It provides accurate, context-aware answers to user queries related to METU.

## Project paper
[METU_chatbot_paper.pdf](https://github.com/user-attachments/files/18473927/METU_chatbot_paper.pdf)


## Links

- **Dataset**: [METU Web Dataset](https://www.kaggle.com/datasets/erkhatkalkabay/metu-web-dataset)
- **Dense Vectors/Embeddings of the Dataset**: [Distiluse Base Multilingual Cased V1 METU Embeddings](https://www.kaggle.com/datasets/erkhatkalkabay/distiluse-base-multilingual-cased-v1-metu-em)

## Installation

Follow these steps to set up the system:

1. Download the repository or use the following command to clone it:
   ```bash
   git clone https://github.com/Yerkhatt/METU-RAG-chatbot.git
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
3. Download and extract:
   * The METU Web Dataset
   * Pre-computed embeddings
4. Create a Groq API Key:
   * Visit Groq's [official website](https://console.groq.com/keys)
   * Create an account if you don't have it
   * Create an API key and save it securely
5. Configure environment variables
   * Rename `config_template.env` to `config.env`.
   * Open the `config.env` file and update it with your details:
     * Put your Groq API key inside `Groq_API_KEY`
     * Put the path to the dataset inside `dataset_path`
     * Put the path to the embeddings file inside `sbert_embeddings_path`
6. Run `main.py` and open the URL shown in your terminal

This is how chat interface looks like:
![image](https://github.com/user-attachments/assets/1ea8445c-db90-4c1a-8860-32a682b5b329)


