from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
#from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import ChatMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationChain
from langchain.chains.summarize import load_summarize_chain
import json
from langchain.memory import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
#from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
import glob
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
import warnings
from langchain.schema.runnable import RunnableMap
from langchain.schema import format_document
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from flask_cors import CORS
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app)

# Set up OpenAI API key and session secret key
f = open('C:\\Users\\Admin\\Desktop\\ChatBot\\practice\\api.txt')
api_key = f.read()
os.environ["OPENAI_API_KEY"] = api_key
app.secret_key = 'my_secret_key_123456789'

# Define a global variable to store uploaded PDF file paths
pdf_file_paths = []
# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

UPLOAD_FOLDER = "C:\\Users\\Admin\\Desktop\\ChatBot\\pdfs"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Function to get uploaded files paths
def get_uploaded_paths(files):
    global file_paths
    file_paths = []
    for file in files:
        filename = secure_filename(file.filename.strip())
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        file_paths.append(file_path)
    return file_paths
# Route for file upload
@app.route('/upload', methods=['POST'])
def upload():
    
    file_paths = get_uploaded_paths(request.files.getlist("folderInput"))
    response = {
        "success": True,  # Indicate successful upload
        "message": "Files uploaded successfully!",
        "file_paths": file_paths  # Include uploaded file paths
    }
    session['file_paths'] = file_paths
    return jsonify(response)
#Intiating chat history and memory globally
chat_history = ChatMessageHistory()
memory = ConversationBufferMemory(return_messages=True, chat_memory=chat_history)
# Route to process user messages
@app.route('/process', methods=['POST'])
def process_message():
    try:
        data = request.json
        message = data['message']
        # Check if file paths are stored in session
        global file_paths
        if 'file_paths' in session:
            file_paths = session['file_paths']
        else:
            # If not found, return error message
            return jsonify({'error': 'file paths not found in session'})
            
        text = extract_text_from_files(file_paths)
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key, streaming=True,
                         callbacks=[StreamingStdOutCallbackHandler()])
        standalone_question = standalone(chat_history, message)
        print("New standalone question:")
        print(standalone_question)
        summary_chain_invoked = False
        # Calculate the number of documents retrieved based on user question
        len_of_retrieved_docs=retrieved_docs(text,message)
        print(len_of_retrieved_docs)
        # Initialize variable to store the response
        answer=""
        # Check if user message indicates summarizing the file
        if any(keyword in message.lower() for keyword in ["summary", "summarize", "overview", "main points", "key points", "synopsis"]):
            # If referencing previous conversation, use conversational chain
            if any(keyword in message.lower() for keyword in ["above","previous"]):
                print("ConversationalChain...")
                answer = general_chat(message,memory,llm)
            # If not summarize uploaded file
            elif not summary_chain_invoked:
                answer=""
                print("summary chain...")
                summaries = summarization(message, memory, file_paths,llm)
                
                formatted_output = [f"File: {file_name}\nSummary: {summary}" for file_name, summary in summaries.items()]
                for item in formatted_output:
                    answer+=item               
        # Check if retrieved documents exist based on user question
        elif(len_of_retrieved_docs>0):
            answer = get_answer(text,standalone_question, memory, llm)
        # If no documents retrieved, use LLM for general conversation
        elif(len_of_retrieved_docs==0):
            answer=general_chat(standalone_question,memory,llm)
        # Update chat history and return response
        chat_history.add_user_message(message)
        chat_history.add_ai_message(answer)
        response_data = {'processed_message': answer} 
        return jsonify(response_data)
    except Exception as e:
        print(f"Error processing message: {e}")
        return jsonify({'error': str(e)})

#Function to extract text from the files
def extract_text_from_files(file_paths):
    try:
        text = ""
        for file_path in file_paths:
            if file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text += f.read()
            elif file_path.endswith(".pdf"):
                with open(file_path, "rb") as f:
                    reader = PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text()
            elif file_path.endswith(".docx") and not file_path.startswith("~$"):
                loader = UnstructuredWordDocumentLoader(file_path)
                doc = loader.load()
                text += doc[0].page_content
        return text
    except Exception as e:
        return f"Error extracting text from files: {str(e)}"

#Function to calculate the length of the retrieved docs
def retrieved_docs(pdf_text,user_question):
    try:
        print("HI")
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(pdf_text)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        document_search = FAISS.from_texts(texts, embeddings)
        retriever = document_search.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3})
        retrieved_documents = retriever.get_relevant_documents(user_question)
        return len(retrieved_documents)
    except Exception as e:
        print(f"Error retrieving text from PDF: {str(e)}")
        return -1
#Function to generate the relevant information using retriever
def get_answer(text, user_question,memory,llm):
    try:
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(text)
        # Create embeddings for text chunks
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Define search criteria to retrieve relevant documents
        document_search = FAISS.from_texts(texts, embeddings)
        retriever = document_search.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3})
        # Chain to perform retrieval-based question answering
        qa= RetrievalQA.from_chain_type(
            llm=llm,
            memory=memory,
            chain_type="map_reduce", 
            retriever = retriever, 
            )
        response = qa.run(user_question)
        return response
        

    except Exception as e:
        return f"Error getting answer: {str(e)}" 
#Function to generate answers for general questions using llm
def general_chat(user_question, memory,llm):
    try:
        conversation = ConversationChain(
            llm=llm,
            verbose=False,
            memory=memory 
        )
        # Run the chain to get the answer from the LLM
        answer = conversation.run(user_question)
        return answer
    except Exception as e:
        return f"Error answering the question: {str(e)}" 
#Funtion to transform the user question to a stand alone format
def standalone(chat_history,user_question):
    template = """Given a chat history and the latest user question \
     which might reference context in the chat history, formulate a standalone question  \
     which can be understood without the chat history. Do NOT answer the question, \
     just reformulate it if needed and otherwise return it as is.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template)
    # Chain to process and reformulate the question
    inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: x["chat_history"]  
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0)
    | StrOutputParser(),
    )
    # Run the chain with conversation history and user question
    standalone_question = inputs.invoke({
        "question": user_question,
        "chat_history": chat_history.messages,
    }).get("standalone_question")

    return standalone_question
class Document(BaseModel):
    title: str = Field(description="Post title")
    keywords: List[str] = Field(description="Keywords used")
#Function to generate summaries
def summarization(user_question, memory, file_paths,llm):
    try:
         
         matching_files = []
         summaries = {}
         # Extract text content from uploaded files
         for file in file_paths:
            text=""
            if file.endswith(".txt"):
                with open(file, "r", encoding="utf-8") as f:
                    text += f.read()
            elif file.endswith(".pdf"):
                with open(file, "rb") as f:
                    reader = PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text()
            elif file.endswith(".docx") and not file.startswith("~$"):
                loader = UnstructuredWordDocumentLoader(file)
                doc = loader.load()
                text += doc[0].page_content
            
            # Use LLM to extract document title and keywords
            parser = JsonOutputParser(pydantic_object=Document)
            prompt = PromptTemplate(
                    template="\n{format_instructions}\n{context}\n",
                    input_variables=["context"],
                    partial_variables={"format_instructions": parser.get_format_instructions()},
                )
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
            chain = prompt | llm | parser
            response1 = chain.invoke({
                    "context": text
                })
            print(response1)
            # Identify documents matching the user question based on title or keywords
            if response1['title'].lower() in user_question.lower():
                 if file not in matching_files:
                      matching_files.append(file)
            for keyword in response1['keywords']:
                 if keyword.lower() in user_question.lower():
                        if file not in matching_files:
                            matching_files.append(file)
                            break   
         print(matching_files)
         #Extract the extract from different file formats         
         for pdf_file in matching_files:
             file_name=os.path.basename(pdf_file)
             if file_name.endswith(".pdf"):
                  loader = PyPDFLoader(pdf_file)
                  docs = loader.load_and_split()
             elif file_name.endswith(".txt"):
                  loader=TextLoader(pdf_file)
                  docs=loader.load()
             elif file_name.endswith("docx"):
                  loader = UnstructuredWordDocumentLoader(pdf_file)
                  docs = loader.load()
                  
             prompt_template = user_question + """
             Generate a concise summary 
             {text}
             SUMMARY:"""
             PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
             llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key, streaming=False)
             chain = load_summarize_chain(llm, chain_type="map_reduce", 
                                    map_prompt=PROMPT, combine_prompt=PROMPT)
             #summarize using the summarize chain
             summary = chain.run(docs)
             
             summaries[file_name]=summary
         return summaries
    except Exception as e:
         print(f"Error summarizing the PDF: {str(e)}")
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)

