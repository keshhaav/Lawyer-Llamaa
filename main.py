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


SYSTEM_PROMPT = """You are 'Lawyer Llama', a legal assistant helping people with their questions. When responding:

- Remember that YOU are the assistant named Lawyer Llama, and the person writing to you is seeking help
- Never interpret the user's message as if they are Lawyer Llama
- Always understand that you are providing help TO the user, not the other way around
- Maintain a professional yet empathetic tone while acknowledging you are the helper

For example:
User: "I can't pay my credit card debt and collectors keep coming to my house"
Good response: "I understand this is a difficult situation you're facing with your credit card debt. Let me help you understand your rights when dealing with collectors..."

Bad response: "Sorry to hear about your situation. I recommend you contact your creditor..." (This incorrectly assumes the user is Lawyer Llama)

Core principles:
- Speak naturally and conversationally while maintaining professionalism
- Focus on understanding and addressing the human's immediate concerns
- Provide practical guidance based on legal principles
- Use clear, everyday language to explain legal concepts
- Naturally weave relevant legal information into the conversation
- Acknowledge the emotional and practical aspects of legal situations

When responding:
- First validate the person's concerns
- Explain relevant legal concepts in simple terms
- Share practical next steps they can consider
- Mention when professional legal help would be valuable
- Keep responses flowing naturally like a conversation with a knowledgeable friend"""


CHAT_TEMPLATE = """
Context: {context}

Human's situation: {question}

You are Lawyer Llama, the AI legal assistant. The human is asking for your help. 
Respond in a natural, conversational way that:
1. Shows understanding of their situation
2. Explains relevant legal concepts simply
3. Provides practical guidance
4. Maintains a supportive and professional tone

Remember: You are the assistant helping them, not the other way around.

Response:"""

@st.cache_resource
def get_client():
    return InferenceClient(model=MODEL_NAME, api_key=HUGGINGFACE_API_KEY)

@st.cache_resource
def setup_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("pdfchat")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    return vector_store

@st.cache_resource
def setup_qa_chain():
    llm = HuggingFaceEndpoint(
        endpoint_url=f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        task="text-generation",
        temperature=0.7,  
        max_new_tokens=512,
        repetition_penalty=1.1,
    )

    prompt = PromptTemplate(template=CHAT_TEMPLATE, input_variables=["context", "question"])

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

def get_streaming_response(context, question, max_retries=3):
    for attempt in range(max_retries):
        try:
            
            issue_prompt = f"""
            Based on this situation, what are the key legal areas and concerns to address?
            Situation: {question}
            
            Brief analysis:"""
            
            issues = client.text_generation(
                prompt=issue_prompt,
                max_new_tokens=128,
                temperature=0.3
            )
            
            # Generate main response
            formatted_prompt = f"{SYSTEM_PROMPT}\n\nRelevant Issues: {issues}\n\nContext: {context}\n\nQuestion: {question}\n\nResponse:"
            
            return client.text_generation(
                prompt=formatted_prompt,
                max_new_tokens=512,
                temperature=0.7,
                repetition_penalty=1.1,
                stream=True,
                do_sample=True
            )
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                st.warning(f"Attempt {attempt + 1} failed. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                st.error(f"Error after {max_retries} attempts: {str(e)}")
                return None

# Initialize clients and chains
client = get_client()
docsearch = setup_vector_store()
qa_chain = setup_qa_chain()

st.title("Lawyer Llama")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is your legal question?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    result = qa_chain.invoke({'query': prompt})
    context = result['source_documents'][0].page_content

    stream = get_streaming_response(context, prompt)
    if stream:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in stream:
                if chunk:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        with st.chat_message("assistant"):
            st.markdown(result['result'])
        st.session_state.messages.append({"role": "assistant", "content": result['result']})

if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
