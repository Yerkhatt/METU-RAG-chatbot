from dotenv import load_dotenv
import os
from groq import Groq
from src.schemas import QueryRequest  # Corrected import statement
from concurrent.futures import ThreadPoolExecutor, as_completed

class GroqAPI:
    # Model configurations
    SEARCH_MODEL = 'llama-3.1-8b-instant'
    RELEVANCE_MODEL = 'llama-3.3-70b-versatile'
    EXTRACTION_MODEL = 'llama-3.3-70b-versatile'
    ANSWER_MODEL = 'llama-3.3-70b-versatile'

    AVAILABLE_MODELS = [
        'llama-3.1-8b-instant',
        'llama-3.3-70b-versatile',
        'mixtral-8x7b-32768',
        'gemma2-9b-it'
    ]

    MODEL_TOKEN_LIMITS = {
        'mixtral-8x7b-32768': 4600,
        'gemma2-9b-it': 13500,
        'llama-3.3-70b-versatile': 5400,
        'llama-3.1-8b-instant': 18000
    }

    # Constants for content limits
    MAX_CONTENT_LENGTH = 4000  # Characters limit for content
    MAX_INPUT_LENGTH = 15000  # Safe limit for input length in characters
    MAX_RETRIES = 3

    # System prompts
    RELEVANCE_PROMPT = """Determine if this document contains information that helps answer the query.
Reply with ONLY a single word: 'Yes' or 'No'.

Query: {query}

Document content:
{content}...

Is this document relevant? Answer with just 'Yes' or 'No':"""

    EXTRACT_INFO_PROMPT = """Extract ONLY the information relevant to answering provided query. If no relevant information exists, respond with 'NO_RELEVANT_INFO'.

Query: {query}

Document content:
{content}...

Extract relevant information:"""

    ANSWER_PROMPT = """You are METU university Q/A chatbot. You represent the university. Answer the question using the provided relevant information. 
Cite sources as [X] when using information. Write concise answers.
If you can't answer from the provided information, say "I don't have enough information."

Relevant Information:
{context}

Question: {query}

Provide a detailed answer with citations, then list the sources you used:"""

    EXTRACT_QUERY_PROMPT = """Extract the core search query from the user's question. Remove unnecessary words and keep only the essential search terms.
Example 1:
Input: "Can you tell me about the history of METU?"
Output: METU history

Example 2:
Input: "bana ODTÜ'nün tarihi hakkında yaz"
Output: ODTÜ'nün tarihi

User input: {query}

Extract core search query:"""

    def __init__(self):  # Rename init to __init__
        load_dotenv('./config.env')
        groq_api_key = os.getenv('Groq_API_KEY')

        self.client = Groq(
            api_key=groq_api_key,  # This is the default and can be omitted
        )

    def _truncate_to_length_limit(self, text: str, max_length: int) -> str:
        """Truncate text to stay within character length limit."""
        if len(text) > max_length:
            text = text[:max_length]
        return text

    def check_relevance(self, query: str, document: dict) -> bool:
        """Check if a document is relevant to the query."""
        relevance_prompt = self.RELEVANCE_PROMPT.format(
            query=query,
            content=document['content'][:1000]  # Limit content for relevance check
        )
        relevance_prompt = self._truncate_to_length_limit(relevance_prompt, 5000)
        response = self.prompt(self.RELEVANCE_MODEL, relevance_prompt).strip().lower()
        print(f'document {document["url"]} is relevant: {response == "yes"}')
        return response == 'yes'

    def batch_check_relevance(self, query: str, documents: list) -> list:
        """Check relevance of multiple documents in parallel."""
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all relevance check tasks
            future_to_doc = {
                executor.submit(self.check_relevance, query, doc): doc 
                for doc in documents
            }
            
            # Collect results as they complete
            relevant_docs = []
            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    is_relevant = future.result()
                    if is_relevant:
                        relevant_docs.append(doc)
                except Exception as e:
                    print(f'Error checking relevance for {doc["url"]}: {str(e)}')
            
            return relevant_docs

    def extract_relevant_info(self, query: str, document: dict) -> tuple[str, bool]:
        """Extract relevant information from document for the query."""
        # Truncate content to prevent token limit issues
        truncated_content = document['content'][:self.MAX_CONTENT_LENGTH]
        extract_prompt = self.EXTRACT_INFO_PROMPT.format(
            query=query,
            content=truncated_content
        )
        extract_prompt = self._truncate_to_length_limit(extract_prompt, 5000)
        response = self.prompt(self.EXTRACTION_MODEL, extract_prompt).strip()
        has_info = response != 'NO_RELEVANT_INFO'
        
        if has_info:
            return response, True
        return "", False

    def answer_with_context(self, query: str, relevant_info: list) -> str:
        """Generate answer using extracted relevant information."""
        # Format extracted information with sources
        context_parts = []
        for i, info in enumerate(relevant_info):
            context_parts.append(f"[{i+1}] From {info['url']}:\n{info['extracted_info']}")
        
        context = "\n\n".join(context_parts)
        
        answer_prompt = self.ANSWER_PROMPT.format(
            context=context,
            query=query
        )
        
        return self.prompt(self.ANSWER_MODEL, answer_prompt)

    def extract_search_query(self, user_query: str) -> str:
        """Extract the core search query from user's question."""
        extract_prompt = self.EXTRACT_QUERY_PROMPT.format(query=user_query)
        extracted_query = self.prompt(self.SEARCH_MODEL, extract_prompt).strip()
        print(f'Extracted query: {extracted_query} from: {user_query}')
        return extracted_query

    def prompt(self, model_name, query_text):
        """Send prompt to LLM with enforced length limit."""
        # Use default model if provided model is None or not in available models
        if not model_name or model_name not in self.AVAILABLE_MODELS:
            if model_name:
                print(f"Warning: Model {model_name} not available, using default")
            model_name = self.SEARCH_MODEL
        
        # Get token limit for the model
        token_limit = self.MODEL_TOKEN_LIMITS.get(model_name, 5000)
        query_text = self._truncate_to_length_limit(query_text, token_limit)
        
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": query_text,
                }
            ],
            model=model_name,
            temperature=0.7,
            max_tokens=1000
        )
        return chat_completion.choices[0].message.content
