import time

from ollama import chat


def stream_response(question: str):

    start_time = time.perf_counter()

    print("🤖 AI: ", end="", flush=True)

    stream = chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        think=False,
        stream=True,
    )

    first_token_time = None

    for chunk in stream:

        if first_token_time is None:
            first_token_time = time.perf_counter() - start_time
            print(
                f"\n\n⏱️ First token: "
                f"{first_token_time:.2f}s\n"
            )

        print(
            chunk.message.content,
            end="",
            flush=True,
        )

    total_time = time.perf_counter() - start_time

    print(
        f"\n\n⏱️ Total time: "
        f"{total_time:.2f}s"
    )


if __name__ == "__main__":

    stream_response(
        "Explain what an AI agent is in 3 short sentences."
    )