# SupportBotAgent

SupportBotAgent is a chatbot that reads and processes documents (PDF or text), uses semantic search and a language model to answer user queries contextually from the document, and iteratively improves its responses based on user feedback.

---

## Features

- Load and process documents in PDF or TXT formats.
- Custom paragraph-level chunking with recursive splitting and overlap.
- Embedding document chunks using HuggingFace Sentence Transformers.
- Vector search using FAISS to retrieve relevant context based on similarity score or MMR.
- Text generation answering queries based only on retrieved document context using Google's Flan-T5-base model.
- User feedback loop supporting responses marked as "good", "too vague", or "not helpful".
- Response refinement with adjustable retrieval strategies based on feedback.
- Comprehensive logging of events, queries, responses, and feedback for transparency.

---

## Installation

### Create and Activate Conda Virtual Environment

```bash
# Create conda environment in folder "venv" with Python 3.8
conda create -p venv python==3.8 -y

# Activate created environment
conda activate venv/
```

### Install Required Packages

```bash
pip install langchain langchain-community faiss-cpu transformers torch PyPDF2
```

---

## Usage

1. Place your source document file (e.g., `faq.txt` or `FAQ.pdf`) in the project directory.
2. Update the `DOCUMENT_PATH` variable in the main script (`support_bot_agent.py`) with your document filename.
3. Run the script:

```bash
python support_bot_agent_Kunal.py
```

4. The bot will process a list of sample queries by default. It will ask for user feedback after each answer:
   - Enter `good` if the answer is satisfactory.
   - Enter `too vague` for a more detailed stepwise answer.
   - Enter `not helpful` for a comprehensive rephrased response with more facts.
5. The bot will update its answer based on the feedback up to a maximum number of iterations (default 2).

---

## Code Overview

### custom_paragraph_chunking
Splits document text into paragraph-level chunks with recursive splitting for large paragraphs.

### SupportBotAgentLC Class

Method: Description
`load_and_chunk` : Loads PDF or TXT using LangChain community loaders and chunks content.
`build_llm` : Loads HuggingFace pipeline with Google Flan-T5-base seq2seq model for answering.
`make_prompt` : Creates a prompt template instructing the model to answer only from given context.
`_retrieve_context` : Performs vector similarity retrieval to fetch relevant chunks.
`answer_query` : Generates an answer to the user query from retrieved context.
`get_user_feedback` : Interactively obtains feedback from the user.
`adjust_strategy` : Refines answer based on user feedback using different retrieval strategies.
`run` : Runs queries from a list, prints answers, and iterates with feedback.

---

## Logging

The bot logs runtime events, including:
- Document loading and chunking status.
- Query reception and retrieval details.
- Generated responses.
- User feedback and adjustments.

Logs are saved to:

```
support_bot_log.txt
```

---

## Example Output

```text
User Query: "How do I reset my password?"
Bot Answer: "You can reset your password by visiting the 'Account Settings' page and selecting 'Forgot Password'."
Feedback (good / too vague / not helpful): too vague
Refined Answer: "To reset your password, go to Account Settings → Security → Forgot Password. Enter your registered email, and follow the link sent to your inbox."
```

## Future Improvements
- Add support for other document formats.
- Integrate with a web or chat UI for live user interaction.
- Automate feedback collection instead of manual input.
- Add support for multi-turn conversations.
