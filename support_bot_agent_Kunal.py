# pip install langchain langchain-community faiss-cpu transformers torch PyPDF2


import os
import logging
import random
import re
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Logging Setup
logging.basicConfig(
    filename="support_bot_log.txt",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("SupportBotLogger")

def custom_paragraph_chunking(documents, chunk_size=600, chunk_overlap=150):
    """Split document into paragraph-level chunks, and recursively split large paragraphs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_docs = []

    for doc in documents:
        # Extract paragraphs (double newlines define paragraph boundaries)
        paras = re.split(r'\n\s*\n|(?<=\n)(?=\d+\.\s)|(?<=\n)(?=[A-Za-z].{0,50}:)', doc.page_content)
        paragraphs = [p.strip() for p in paras if p.strip()]
        # paragraphs = [p.strip() for p in doc.page_content.split("\n\n") if p.strip()]
        for para in paragraphs:
            if len(para) <= chunk_size:
                chunked_docs.append(Document(page_content=para))
            else:
                # For large paragraphs, split recursively with overlap
                splits = splitter.split_text(para)
                for split in splits:
                    chunked_docs.append(Document(page_content=split))
    logger.info(f"Custom chunking produced {len(chunked_docs)} chunks")
    return chunked_docs

class SupportBotAgentLC:
    def __init__(self, document_path: str):
        logger.info(f"Initializing agent for document: {document_path}")
        self.document_path = document_path
        self.documents = self.load_and_chunk(document_path)

        self.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vectordb = FAISS.from_documents(self.documents, self.embedder)

        self.retriever = self.vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 4, "score_threshold": 0.35},
        )
        self.llm = self.build_llm()
        self.prompt = self.make_prompt()

    def load_and_chunk(self, path: str) -> List[Document]:
        ext = os.path.splitext(path)[1].lower()
        logger.info(f"Loading document: {path} (type: {ext})")
        if ext == ".pdf":
            docs = PyPDFLoader(path).load()
        elif ext == ".txt":
            docs = TextLoader(path, encoding="utf-8").load()
        else:
            raise ValueError("Document type not supported")
        logger.info(f"Loaded {len(docs)} top-level chunks from document")

        chunked = custom_paragraph_chunking(docs, chunk_size=600, chunk_overlap=150)
        return chunked

    def build_llm(self):
        GEN_MODEL_NAME = "google/flan-t5-base"
        logger.info(f"Loading HuggingFace pipeline LLM: {GEN_MODEL_NAME}")
        tok = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
        pipe = pipeline(
            task="text2text-generation",
            model=mdl,
            tokenizer=tok,
            max_new_tokens=256,
            do_sample=False,
        )
        return HuggingFacePipeline(pipeline=pipe)

    def make_prompt(self):
        template = """You are a helpful customer support assistant.
Use ONLY the provided context to answer the user's question.
If the answer is not in the context, say: "I don’t have enough information to answer that."

CONTEXT:
{context}

QUESTION:
{question}

Answer clearly, with specifics and step-by-step instructions when possible."""
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def _retrieve_context(self, query: str) -> List[str]:
        results = self.retriever.get_relevant_documents(query)
        logger.info(f"Retrieved {len(results)} relevant context chunks for query: '{query}'")
        return [doc.page_content for doc in results]

    def answer_query(self, query: str) -> str:
        context_chunks = self._retrieve_context(query)
        ctx = "\n\n".join(context_chunks) if context_chunks else ""
        prompt = self.prompt.format(context=ctx, question=query)
        resp = self.llm.invoke(prompt).strip()
        logger.info(f"Answer generated for query '{query}': {resp}")
        if not resp or "I don’t have enough information" in resp:
            logger.warning(f"Query not covered by context: '{query}'")
        return resp

    # def simulate_feedback(self) -> str:
    #     fb = random.choice(["not helpful", "too vague", "good"])
    #     logger.info(f"Simulated feedback: {fb}")
    #     return fb
    
    def get_user_feedback(self) -> str:
        while True:
            fb = input("Please provide feedback (good / too vague / not helpful): ").strip().lower()
            if fb in {"good", "too vague", "not helpful"}:
                return fb
            else:
                print("Invalid input. Please enter one of: good, too vague, not helpful.")

    def adjust_strategy(self, query: str, response: str, feedback: str, iter_num: int) -> str:
        logger.info(f"Adjusting response for feedback: '{feedback}'")
        if feedback == "too vague":
            mmr_retriever = self.vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.8},
            )
            contexts = [doc.page_content for doc in mmr_retriever.get_relevant_documents(query)]
            longer_q = query + " Provide step-by-step details and include relevant emails or numbers if present."
            prompt = self.prompt.format(context="\n\n".join(contexts), question=longer_q)
            return self.llm.invoke(prompt).strip()
        elif feedback == "not helpful":
            relaxed_retriever = self.vectordb.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 7, "score_threshold": 0.1},
            )
            contexts = [doc.page_content for doc in relaxed_retriever.get_relevant_documents(query)]
            longer_q = query + " Rephrase comprehensively and include as many precise facts as possible."
            prompt = self.prompt.format(context="\n\n".join(contexts), question=longer_q)
            return self.llm.invoke(prompt).strip()
        return response

    def run(self, queries: List[str], max_iters: int = 2):
        for query in queries:
            print(f"Query: {query}")
            logger.info(f"--- New Query --- {query}")
            response = self.answer_query(query)
            print(f"Initial Response: {response}")

            for i in range(1, max_iters + 1):
                feedback = self.get_user_feedback()  # get actual user input feedback
                if feedback == "good":
                    logger.info("Feedback: good. Stopping further iterations.")
                    break
                response = self.adjust_strategy(query, response, feedback, i)
                print(f"Updated Response (Iteration {i}): {response}")

            print("")

if __name__ == "__main__":
    DOCUMENT_PATH = "ML Assignment.pdf"  # or "FAQ.pdf"
    agent = SupportBotAgentLC(DOCUMENT_PATH)
    sample_queries = [
        "Which is the time limit for submitting the assignment?"
    ]
    agent.run(sample_queries)
