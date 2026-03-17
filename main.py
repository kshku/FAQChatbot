from typing import Any

import gradio as gr
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_MODEL = "gemini-3-flash-preview"

SYSTEM_INSTRUCTION = """
You are VCET Puttur's official website assistant.

Goals:
- Answer questions about VCET Puttur using the college website as the primary source.
- Be clear, concise, and student-friendly.
- If a detail is uncertain or unavailable, say so clearly and suggest checking the official website/contact page.

Behavior rules:
- Prioritize accuracy over completeness.
- Do not invent facts, numbers, deadlines, fees, phone numbers, or emails.
- If asked something outside college information, politely steer back to VCET-related help.
- Keep responses structured with short paragraphs or bullet points when useful.
- When useful, include a short "Source" line that mentions the VCET website URL.
""".strip()

COLLEGE_WEBSITE_URL = "https://vcetputtur.ac.in/"


class ChatService:
    def __init__(self) -> None:
        self._client = genai.Client()
        self._chat: Any = self._create_chat()

    def _create_chat(self) -> Any:
        return self._client.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[types.Tool(url_context=types.UrlContext())],
            ),
        )

    def reset(self) -> None:
        self._chat = self._create_chat()

    def ask(self, message: str) -> str:
        enriched_message = (
            "Use the given college website as the source of truth when relevant.\n"
            f"Website URL: {COLLEGE_WEBSITE_URL}\n"
            f"User question: {message}"
        )
        response = self._chat.send_message(enriched_message)
        return response.text or ""


chat_service = ChatService()


def send_message(message: str, history: list[dict[str, str]]) -> tuple[list[dict[str, str]], str, str]:
    if not message.strip():
        return history, "", "Please enter a message."

    reply = chat_service.ask(message)
    updated_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return updated_history, "", ""


def start_new_session() -> tuple[list[dict[str, str]], str]:
    chat_service.reset()
    return [], "Started a new session. Previous chat context and history were cleared."


def main() -> None:
    with gr.Blocks(title="VCET Assistant") as demo:
        gr.Markdown("# VCET Puttur Assistant")
        gr.Markdown(
            "Ask anything about VCET Puttur. The assistant uses the official website as context."
        )
        gr.Markdown(
            "Tips: ask specific questions like admissions, departments, contact details, facilities, and announcements."
        )

        chatbot = gr.Chatbot(height=450)
        message = gr.Textbox(label="Message", placeholder="Ask your VCET question and press Enter")
        gr.Examples(
            examples=[
                "What courses are offered at VCET Puttur?",
                "How can I contact the college office?",
                "Tell me about campus facilities.",
                "Where can I find admission-related information?",
            ],
            inputs=message,
        )

        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            new_session_btn = gr.Button("New Session")

        status = gr.Markdown("")

        send_btn.click(
            fn=send_message,
            inputs=[message, chatbot],
            outputs=[chatbot, message, status],
        )
        message.submit(
            fn=send_message,
            inputs=[message, chatbot],
            outputs=[chatbot, message, status],
        )
        new_session_btn.click(
            fn=start_new_session,
            inputs=[],
            outputs=[chatbot, status],
        )

    demo.launch()


if __name__ == "__main__":
    main()
