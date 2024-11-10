import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from huggingface_hub import InferenceClient
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import sys
import time

st.set_page_config(page_title="Lawyer Llama")

HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
PINECONE_API_ENV = st.secrets['PINECONE_API_ENV']
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

SYSTEM_PROMPT = """ Your name is 'Lawyer Llama'.
- You are a legal expert assistant  providing detailed and precise answers based strictly on the legal context provided only  exception is when the user asks for your name you should respond as i am Lawyer Llama. do not seek for the context if the user asked you name or asked what you can do.
- Use exact definitions from the provided legal context where possible.
- Avoid broad or overly general explanations; be legally precise.
- And just provide straight answers no A, B ,C options. just the answer with a touch of better vocabulary would be fine
- If the context does not provide a clear answer, respond with: "I don't know about that can you provide more context?" and stop generating anything else if theres no more context provided. DONOT TRIGGER THIS IF THE USER ASKS FOR YOUR NAME.
"""



@st.cache_resource
def get_client():
    return InferenceClient(model=MODEL_NAME, api_key=HUGGINGFACE_API_KEY)

client = get_client()


# @st.cache_resource
# def load_and_process_docs():
#     loader = PyPDFDirectoryLoader("pdfs")
#     data = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
#     docs = text_splitter.split_documents(data)
#     return docs

# docs = load_and_process_docs()

@st.cache_resource
def setup_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_name = "pdfchat"
    return LangchainPinecone.from_existing_index(index_name=index_name, embedding=embeddings)

docsearch = setup_vector_store()

# Set up LLM and QA chain
@st.cache_resource
def setup_qa_chain():
    llm = HuggingFaceEndpoint(
        endpoint_url=f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        task="text-generation",
        temperature=0.1,
        max_new_tokens=512,
        repetition_penalty=1.03,
        
    )

    template = """
    Context: {context}

    Question: {question}

    Please provide a legally precise explanation, referencing specific definitions or statements from the document where applicable:"""


    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "verbose": True
        },
    )

qa_chain = setup_qa_chain()

# Function to get streaming response
def get_streaming_response(context, question, max_retries=3):
    for attempt in range(max_retries):
        try:
            formatted_prompt = f"{SYSTEM_PROMPT}\n\nContext: {context}\n\nQuestion: {question}\n\nPlease provide a direct explanation:"
            
            stream = client.text_generation(
                prompt=formatted_prompt,
                max_new_tokens=256,
                temperature=0.5,
                repetition_penalty=1.2,
                stream=True,
                do_sample=True,
                
            
            )
            return stream
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                st.warning(f"Attempt {attempt + 1} failed. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                st.error(f"Error after {max_retries} attempts: {str(e)}")
                return None

# Streamlit UI
st.title("Lawyer Llama")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your legal question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get context using retrieval chain
    result = qa_chain.invoke({'query': prompt})
    context = result['source_documents'][0].page_content

    # Get streaming response
    stream = get_streaming_response(context, prompt)
    if stream:
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in stream:
                if chunk:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        # Fallback to non-streaming response
        with st.chat_message("assistant"):
            st.markdown(result['result'])
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result['result']})

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()