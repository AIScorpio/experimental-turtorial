from langchain.vectorstores.chroma import Chroma
from langchain_groq.chat_models import ChatGroq
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatGroq(
            model="llama3-70b-8192",
            # model="mixtral-8x7b-32768",
            temperature=0.5,
            api_key="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"
        )
        # self.model = ChatOllama(model="mistral:7b-instruct-v0.2-fp16")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, 
            chunk_overlap=100
        )
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved
            context to answer the question. If you don't know the answer, just say that you don't know.
            Keep your answer straight forward and concise. No yapping! [/INST] </s>
            [INST] Question: {question}
            Context: {context}
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        """
        This function handles document loading and embedding for retrieval
        Parameters: 
            pdf_file_path: str, the path of the PDF file to load 
        """
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        # docs = DirectoryLoader(
        #     path=pdf_file_path,
        #     glob="**/*.txt",
        #     show_progress=True,
        #     silent_errors=True,
        #     loader_kwargs={'autodetect_encoding': True},
        # ).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(documents=chunks, 
            embedding=OllamaEmbeddings(model='nomic-embed-text:latest'),
            # embedding=FastEmbedEmbeddings()
            )
        self.retriever = vector_store.as_retriever(
            # similarity, mmr, similarity_score_threshold
            search_type='similarity',       
            search_kwargs={
                'k': 4,
                # 'score_threshold': 0.5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
        
    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."
        return self.chain.invoke(query)
        
    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None



