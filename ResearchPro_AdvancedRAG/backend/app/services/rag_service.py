from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from config.config import llm_reformulate


class RAG_Pipeline:
    def __init__(self, llm , vectorstore=None):
        self.llm = llm
        self.vectorstore = vectorstore
        self.hybrid_retriever = None
        self.compression_retriever = None
        self.conversational_rag = None
        
        self.reformulation_prompt = self.create_reformulation_prompt()
        self.answer_prompt  = self.create_answer_prompt()


    def create_reformulation_prompt(self):
        reform_sys_prompt = """
        You are a research question reformulator for academic document analysis.
        Given the conversation history and the latest user query, rewrite the query 
        into a clear, self-contained research question. 

        Guidelines:
        - Preserve the user's intent completely.
        - Expand abbreviations or vague references (e.g., "it", "they", "the table") using chat history.
        - If the question involves data, tables, statistics, or numerical information, make that explicit.
        - If referring to previous tables or data, include that context in the reformulation.
        - Do NOT answer the question - only reformulate it.
        - If the question is already clear and standalone, return it unchanged.
        
        Examples:
        - "What does it show?" → "What data does the table on page X show?"
        - "Compare them" → "Compare the results shown in Table 1 and Table 2"
        """

        return ChatPromptTemplate.from_messages([
            ("system", reform_sys_prompt),
            MessagesPlaceholder("chat_history"), 
            ("human", "{input}")
        ])




    def create_answer_prompt(self):
        answer_sys_prompt = """
        You are an expert research assistant specialized in analyzing academic papers, 
        with particular expertise in interpreting tables, charts, and quantitative data.

        CRITICAL INSTRUCTIONS:
        1. **Base answers ONLY on provided context** - never fabricate data or references.
        
        2. **When tables are present:**
           - Carefully analyze the table structure, headers, and data
           - Extract specific numbers, trends, and comparisons
           - Explain what the data shows in clear, accessible language
           - Note any patterns, outliers, or significant findings
        
        3. **For data-related questions:**
           - Cite specific values from tables when available
           - Compare multiple data points if relevant
           - Explain the significance of the numbers
        
        4. **Structure your response:**
           - Start with a direct answer to the question
           - Support with specific data from tables/text
           - Provide interpretation and context
           - Mention limitations if data is incomplete
        
        5. **Maintain academic rigor:**
           - Use precise language for quantitative information
           - Distinguish between facts (from context) and interpretation
           - If context lacks information, clearly state so
        
        6. **For tables/charts specifically:**
           - Describe what type of data is presented (percentages, counts, measurements, etc.)
           - Identify key comparisons being made
           - Note any trends or relationships visible in the data
           - Reference the page number when citing table data

        Context (includes text passages and table data):
        {context}
        """

        return ChatPromptTemplate.from_messages([
            ("system", answer_sys_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        
        
    
    def update_vectorstore(self, vectorstore):
        self.vectorstore = vectorstore

    
    def set_compression_retriever(self, compression_retriever):
        self.compression_retriever = compression_retriever
    


    def create_rag_chain(self, retriever):
        history_aware_retriever = create_history_aware_retriever(
            llm_reformulate,
            retriever,
            self.reformulation_prompt
        )

        question_answer_chain = create_stuff_documents_chain(
            self.llm,
            self.answer_prompt
        )

        rag_pipeline = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        return rag_pipeline
    
    
    def create_conversational_chain(self, rag_chain, get_session_history_func):
        self.conversational_rag = RunnableWithMessageHistory(
            rag_chain,
            get_session_history_func,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        return self.conversational_rag


    def query(self, question: str, session_id: str) -> str:
        if not self.conversational_rag:
            return "Error: Conversational chain not initialized"

        try:
            # Run conversational RAG chain (retrieval happens internally)
            response = self.conversational_rag.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}}
            )

            return response.get("answer", "No response generated")  

        except Exception as e:
            return f"Error processing query: {str(e)}"
