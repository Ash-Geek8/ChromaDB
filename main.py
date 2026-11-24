import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import uuid
import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

Enable_Button = False  # Set to True to enable the interactive menu

#INITIALIZATION

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

#DOCUMENT IMPORT

def import_document(file_path):

    path = Path(file_path)

    if not path.exists():
        print("File not found!")
        return

    text = path.read_text(encoding="utf-8")



    #TEXT CHUNKING

    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]


    #EMBEDDING GENERATION

    embeddings = model.encode(chunks).tolist()


    #STORE IN CHROMADB

    collection.upsert(
        ids=[
            f"{path.stem}_{i}"
            for i in range(len(chunks))
        ],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {
                "source": path.name,
                "chunk_number": i
            }
            for i in range(len(chunks))
        ]
    )

#SEMANTIC SEARCH

def search_documents(query, top_k=3, threshold=None):

    print(f"Query: {query}")

    if collection.count() == 0:
        print("Database is empty.")
        return

    # Generate query embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    # Retrieve nearest chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    if not results["documents"] or not results["documents"][0]:
        print("No matching documents found.")
        return

    # Prepare ranked results
    ranked_results = []

    for i, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        # ChromaDB cosine distance = 1 - cosine similarity
        similarity = 1 - distance

        ranked_results.append({
            "document": document,
            "source": metadata["source"],
            "chunk": metadata["chunk_number"],
            "distance": distance,
            "similarity": similarity
        })

    # Explicitly sort by highest similarity
    ranked_results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    # Apply optional threshold
    if threshold is not None:
        ranked_results = [
            result for result in ranked_results
            if result["similarity"] >= threshold
        ]

    if not ranked_results:
        print("\nNo results met the similarity threshold.")
        return

    #BEST MATCH

    best_match = ranked_results[0]

    print(best_match['document'])

    #ALL RESULTS

    '''
    print("\n" + "=" * 70)
    print("ALL MATCHING RESULTS")
    print("=" * 70)

    for rank, result in enumerate(ranked_results, start=1):

        print(f"\nRank       : {rank}")
        print(f"Source     : {result['source']}")
        print(f"Chunk      : {result['chunk']}")
        print(f"Similarity : {result['similarity']:.4f}")
        print(f"Distance   : {result['distance']:.4f}")
        print(f"Content    : {result['document']}")

    print("\n" + "=" * 70)
    print(f"BEST SIMILARITY SCORE: {best_match['similarity']:.4f}")
    print("=" * 70) '''

if __name__ == "__main__":

    file_path = "data/sample.txt"
    import_document(file_path)
    query = "What is RAG?"
    search_documents(query)

    while Enable_Button:

        print("1. Import document")
        print("2. Search documents")
        print("3. Show database count")
        print("4. Exit")

        choice = input("\nSelect option: ")

        if choice == "1":

            file_path = input("Enter document path: ")

            import_document(file_path)

        elif choice == "2":

            if collection.count() == 0:
                print("Database is empty. Import a document first.")
                continue

            query = input("Enter search query: ")

            search_documents(query)

        elif choice == "3":

            print(
                f"\nTotal stored chunks: {collection.count()}"
            )

        elif choice == "4":

            print("Exiting...")
            break

        else:

            print("Invalid choice!")
