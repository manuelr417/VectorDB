from pypdf import PdfReader
from os import listdir
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from dao.docs import DocDAO
from dao.fragments import FragmentDAO
from sentence_transformers import SentenceTransformer

#model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("all-mpnet-base-v2")

files = listdir("./files")
print(files)

#extract chunks

docDAO = DocDAO()
fraDAO = FragmentDAO()

for f in files:
    fname = "./files/" + f
    reader = PdfReader(fname)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]

    # Filter the empty strings
    pdf_texts = [text for text in pdf_texts if text]

    print(pdf_texts[0])

    #split
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=100,
        chunk_overlap=50
    )
    character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))

    print(character_split_texts[10])
    print(f"\nTotal chunks: {len(character_split_texts)}")

    [print(t) for t in character_split_texts]
    print()
    #Token
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=50, tokens_per_chunk=256)

    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)
    print("token")
    print(token_split_texts[10])
    [print(t) for t in token_split_texts]
    print(f"\nTotal Splitted chunks: {len(token_split_texts)}")
    #exit(1)

    # insert document into table
    did = docDAO.insertDoc(f)


    for t in token_split_texts:
        emb = model.encode(t)
        #print(t)
        #print(emb)
        fraDAO.insertFragment(did, t, emb.tolist())

    print("Done file: " + f)


