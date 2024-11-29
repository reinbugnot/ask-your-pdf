from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import os

# Display the header
print("\n\nSAFe® for Teams: Establishing Team Agility for Agile Release Trains Retrieval Augmented Generator (RAG)")
print("\n\nInitializing RAG System. This may take a while ...")

# Initialize memory and LLM
memory = MemorySaver()
api_key = os.getenv("OPENAI_API_KEY_NUS")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

# Load the PDF document
loader = PyPDFLoader(file_path="./data/CERT_RAG.pdf")
docs = loader.load()

# Split the document into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create the vector store using FAISS
vectorstore = FAISS.from_documents(
    documents=splits, embedding=OpenAIEmbeddings()
)

# Set up the retriever
retriever = vectorstore.as_retriever()

# Build the retriever tool
tool = create_retriever_tool(
    retriever,
    "pdf_retriever",
    "Searches and returns excerpts from the uploaded PDF document.",
)
tools = [tool]

agent_executor = create_react_agent(llm, tools, checkpointer=memory)

print("\n\nInitialization Complete! Ready for Q&A.\n")

# Start the CLI loop
while True:
    # Get the question and choices from the user
    question = input("\nEnter your question (or type 'exit' to quit):\n")
    if question.lower() == "exit":
        print("Exiting RAG System. Goodbye!")
        break

    choice_a = input("A. ")
    choice_b = input("B. ")
    choice_c = input("C. ")
    choice_d = input("D. ")
    query = f"{question}\n a. {choice_a}\n b. {choice_b}\n c. {choice_c}\n d. {choice_d}"

    print("\nGenerating answer based on the document...\n")

    # Stream the response
    config = {"configurable": {"thread_id": "abc123"}}

    try:
        for event in agent_executor.stream(
            {"messages": [HumanMessage(content=query)]},
            config=config,
            stream_mode="values",
        ):
            # Check the type of message and display it accordingly
            if "messages" in event:
                last_message = event["messages"][-1]
                if isinstance(last_message, HumanMessage):
                    print("\n================================ Human Message =================================")
                    print(last_message.content)
                elif isinstance(last_message, AIMessage):
                    print("\n================================== Ai Message ==================================")
                    print(last_message.content)
                elif isinstance(last_message, ToolMessage):
                    print("\n================================= Tool Message =================================")
                    print(last_message.content)
            else:
                print("No response received from the agent.")
    except Exception as e:
        print(f"An error occurred: {e}")
