import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

class GeminiRAG:
    """Wrapper for Gemini RAG capabilities"""
    
    def __init__(self):
        """Initialize the Gemini RAG system"""
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        
        # Initialize embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            google_api_key=api_key
        )
    
    def create_vector_store(self, texts: List[str], collection_name: str = "video_transcripts"):
        """Create a vector store from a list of texts"""
        
        # Split texts into smaller chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )
        
        chunks = text_splitter.create_documents(texts)
        
        # Create and return the vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name
        )
        
        return vector_store
    
    def create_rag_chain(self, vector_store, k: int = 5):
        """Create a RAG chain using the provided vector store"""
        
        # Create a retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Define the prompt template
        template = """
        You are an expert video editor and content creator.
        
        Use the following context to answer the question:
        
        Context: {context}
        
        Question: {question}
        
        Your answer should be detailed, insightful and relevant to video editing or content creation.
        """
        
        prompt = PromptTemplate.from_template(template)
        
        # Create and return the chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def find_video_highlights(self, vector_store, video_title: str, num_highlights: int = 3, segment_length: int = 60):
        """Find the most engaging highlights in a video using RAG"""
        
        # Create a custom prompt for highlight detection
        highlight_prompt = f"""
        Based on the transcript segments, identify the {num_highlights} most engaging moments 
        that would make great {segment_length}-second YouTube Shorts from the video titled 
        '{video_title}'.
        
        For each highlight, provide:
        1. A descriptive title
        2. A brief explanation of why this segment is engaging
        
        Format your response as a JSON list like:
        [
            {{
                "title": "Catchy title here",
                "description": "Description of why this segment is engaging"
            }}
        ]
        
        Only provide the JSON list in your response.
        """
        
        # Create a retriever that returns more context
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # Retrieve more context for better highlight selection
        )
        
        # Get relevant transcript segments
        results = retriever.get_relevant_documents(highlight_prompt)
        context = "\n\n".join([doc.page_content for doc in results])
        
        # Use Gemini to analyze and select highlights
        response = genai.generate_text(
            model="models/gemini-pro",
            prompt=f"""
            You are an expert YouTube video editor.
            
            Analyze these transcript segments from a video titled '{video_title}':
            
            {context}
            
            {highlight_prompt}
            """,
            temperature=0.7
        )
        
        return response.text
    
    def generate_caption(self, transcript_segment: str, clip_description: str, video_title: str):
        """Generate a caption for a short video clip using Gemini"""
        
        prompt = f"""
        You are an expert at creating engaging YouTube Shorts content.
        
        Original video title: "{video_title}"
        
        Clip description: {clip_description}
        
        Transcript of the clip:
        ```
        {transcript_segment}
        ```
        
        Please create:
        1. A catchy title (60 characters max) with emojis if appropriate
        2. An engaging caption (150 characters max)
        3. 5-7 relevant hashtags
        
        Return your response in this JSON format:
        {{
          "title": "Your catchy title here",
          "caption": "Your engaging caption here",
          "hashtags": "#shorts #youtube #topic1 #topic2"
        }}
        
        Only include the JSON in your response, no additional text.
        """
        
        response = genai.generate_text(
            model="models/gemini-pro",
            prompt=prompt,
            temperature=0.8
        )
        
        return response.text
