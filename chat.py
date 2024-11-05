from dao.docs import DocDAO
from dao.fragments import FragmentDAO
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = SentenceTransformer("all-MiniLM-L6-v2")

#question = "Which model provided best estimates for XLopP?"
#question = "Who is the authors of the paper?"
#question = "What the contributions of the paper?"
question = "What is the main idea of the paper?"

#emb = model.encode("SMILES embedding")
emb = model.encode(question)

dao = FragmentDAO()
framents = dao.getFragments(str(emb.tolist()))
context = []
for f in framents:
    print(f)
    context.append(f[3])

print(context[0])

#Prepare Templa
documents = "\\n".join(c for c in context)
print(documents)

# Define the prompt template for the LLM
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Use five sentences maximum and keep the answer concise:
    Documents: {documents}
    Question: {question}
    Answer:
    """,
    input_variables=["question", "documents"],
)
print(prompt)
print(prompt.format(question= question, documents=documents))
#exit(1)
# Initialize the LLM with Llama 3.1 model
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

# Create a chain combining the prompt template and LLM
rag_chain = prompt | llm | StrOutputParser()

#Question

answer = rag_chain.invoke({"question": question, "documents": documents})
print(answer)