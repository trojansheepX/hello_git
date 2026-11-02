"""Task 1 - Read the documents and show the token count of each file."""
from rag_config import ENCODING, load_documents


def main():
    documents = load_documents()
    for filename, text in documents.items():
        print("=" * 80)
        print(f"FILE: {filename}")
        print("=" * 80)
        print(text)
        print(f"\nToken count ({ENCODING.name}): {len(ENCODING.encode(text))}\n")


if __name__ == "__main__":
    main()
