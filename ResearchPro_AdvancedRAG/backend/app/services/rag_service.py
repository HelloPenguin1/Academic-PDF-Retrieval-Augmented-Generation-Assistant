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
        
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
        
        """

        return ChatPromptTemplate.from_messages([
            ("system", reform_sys_prompt),
            MessagesPlaceholder("chat_history"), 
            ("human", "{input}")
        ])

    
        # You are a research question reformulator for academic document analysis.
        # Given the conversation history and the latest user query, rewrite the query 
        # into a clear, self-contained research question. 

        # Guidelines:
        # - Preserve the user's intent completely.
        # - Expand abbreviations or vague references (e.g., "it", "they", "the table") using chat history.
        # - If the question involves data, tables, statistics, or numerical information, make that explicit.
        # - If referring to previous tables or data, include that context in the reformulation.
        # - Do NOT answer the question - only reformulate it.
        # - If the question is already clear and standalone, return it unchanged.
        
        # Examples:
        # - "What does it show?" → "What data does the table on page X show?"
        # - "Compare them" → "Compare the results shown in Table 1 and Table 2"


    def create_answer_prompt(self):
        answer_sys_prompt = """
        
        Answer the following question based only on the provided context. 

        Context: 
        {context}

        Question: 
        {input}

        Answer:

        """

        return ChatPromptTemplate.from_messages([
            ("system", answer_sys_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        # You are a strict, highly precise academic research assistant. Your sole purpose is to answer the user's question directly using ONLY the information provided in the Context below.

        # CRITICAL FAITHFULNESS RULES (NO HALLUCINATIONS):
        # 1. **Strict Grounding:** You must base your answer EXCLUSIVELY on the provided context. Do not use outside knowledge, general knowledge, or training data.
        # 2. **Mandatory Citations:** Every factual claim, number, or data point you write must be followed by an inline citation referencing the context (e.g., [Page 4] or [Table 2]). 
        # 3. **No Interpretations:** Do not interpret, deduce, or infer conclusions that are not explicitly written in the text. If the text provides data but no conclusion, state the data and stop.
        # 4. **Handling Missing Info:** If the context does not contain the exact information needed to answer the question, you must respond exactly with: "The provided documents do not contain the information necessary to answer this question." Do not attempt to guess.
        # 5. **Conflicting Info:** If different parts of the context contradict each other, state both facts clearly and cite both sources. Do not attempt to resolve the conflict yourself.

        # CRITICAL RELEVANCY RULES (BE DIRECT AND CONCISE):
        # 1. **Answer First:** The very first sentence of your response must directly answer the user's core question.
        # 2. **Zero Fluff:** Do not use introductory filler (e.g., "Based on the provided documents..." or "The table shows that..."). Get straight to the point.
        # 3. **Filter the Context:** The retriever may provide irrelevant context chunks. Ignore them. Only include information that strictly and directly answers the user's prompt. 
        # 4. **Targeted Table Extraction:** If the user asks for a specific data point from a table, provide ONLY that data point. Do not summarize the rest of the table, describe its structure, or note unrelated trends unless explicitly requested.
        # 5. **Match Complexity:** If the question is simple (e.g., "What is the value of X?"), give a 1-2 sentence answer. Only provide long, detailed explanations if the user asks "How" or "Why".

        # Context:
        # {context}
        
    
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
