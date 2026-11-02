"""Run the whole RAG exercise end to end (Tasks 1-4 plus the retrieval evaluation)."""
import task1_read_tokenize
import task2_chunk_embed
import task4_ai_answer
import task5_evaluate_retrieval


def section(title: str):
    print("\n" + "#" * 80)
    print(f"# {title}")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    section("TASK 1 - Read and tokenize")
    task1_read_tokenize.main()

    section("TASK 2 - Chunk and embed")
    task2_chunk_embed.main()

    section("RETRIEVAL EVALUATION - evidence in top 3 chunks")
    task5_evaluate_retrieval.main()

    section("TASK 3 + 4 - ChromaDB search and AI answers")
    task4_ai_answer.main()
