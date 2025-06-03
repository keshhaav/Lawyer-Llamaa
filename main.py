import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from huggingface_hub import InferenceClient
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Lawyer Llama", layout="centered")

HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = '''You are 'Lawyer Llama', a legal assistant helping people with their questions. When responding:
- Remember that YOU are the assistant named Lawyer Llama, and the person writing to you is seeking help
- Never interpret the user's message as if they are Lawyer Llama
- Always understand that you are providing help TO the user, not the other way around
- Maintain a professional yet empathetic tone while acknowledging you are the helper

Core principles:
- Speak naturally and conversationally while maintaining professionalism
- Focus on understanding and addressing the human's immediate concerns
- Provide practical guidance based on legal principles
- Use clear, everyday language to explain legal concepts
- Acknowledge the emotional and practical aspects of legal situations

When responding:
- First validate the person's concerns
- Explain relevant legal concepts in simple terms
- Share practical next steps they can consider
- Mention when professional legal help would be valuable
- Keep responses flowing naturally like a conversation with a knowledgeable friend'''

@st.cache_resource
def vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    pc = pinecone.init(api_key=PINECONE_API_KEY)
    index = pc.Index("pdfchat")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    return vector_store

vectorstore = vector_store()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

st.title("Lawyer Llama")
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for user_query, bot_response in st.session_state.chat_history:
    st.markdown(f"<div style='margin: 5px;display: inline-block;color: white; padding: 8px; background-color: #818589; border-radius: 5px;'>You: {user_query}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='margin: 5px; padding: 8px; border-radius: 5px;'>Lawyer Llama: {bot_response}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

chathistory=st.session_state.chat_history

query = st.text_input("Ask your legal question:",key='query_input', value=' ' )
if st.button("Send", key='send_button') and query:
    context = retriever.invoke(query)
    client = InferenceClient(provider="nebius", api_key=HUGGINGFACE_API_KEY)
    formatted_prompt = f"{SYSTEM_PROMPT}\n\nContext: {context}\n\nChat History: {chathistory}\n\nQuestion: {query}\n\nResponse:"
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": formatted_prompt}],
        stream=True,
    )
    response = ""
    response_placeholder= st.empty()
    for chunk in stream:
        chunk_content = chunk.choices[0].delta.content
        response += chunk_content
        response_placeholder.markdown(f"**Lawyer Llama:** {response}")
    st.session_state.chat_history.append((query, response))
    st.rerun()

if st.button("Clear Chat History", key='clear_button'):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")
    st.rerun()
