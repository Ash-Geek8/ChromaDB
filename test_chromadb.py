import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb

client = chromadb.Client()

collection = client.get_or_create_collection("test_collection")

collection.add(
    ids=["1", "2"],
    documents=[
        "Artificial Intelligence is transforming technology.",
        "Machine Learning enables computers to learn."
    ]
)

results = collection.query(
    query_texts=["What is AI?"],
    n_results=2
)

print("\nSEARCH RESULTS")

for document in results["documents"][0]:
    print("-", document)