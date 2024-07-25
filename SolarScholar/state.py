import os
import reflex as rx
from langchain_upstage import UpstageLayoutAnalysisLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage


class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str


DEFAULT_CHATS = {
    "Hi, I'm Solar": [],
}


class ChatState(rx.State):
    """The app state."""

    api_key: str = ""
    prompt: str = """
        Please provide most correct answer from the following context. 
        Think step by step and look the html tags and table values carefully to provide the most correct answer.
        If the answer is not present in the context, please write "The information is not present in the context."
        ---
        Question: {question}
        ---
        Context: {Context}
        """
    pdf_processing: bool = False
    loader_processing: bool = False

    # A dict from the chat name to the list of questions and answers.
    chats: dict[str, list[QA]] = DEFAULT_CHATS

    # The current chat name.
    current_chat: str = "Hi, I'm Solar"

    # The current question.
    question: str

    # Whether we are processing the question.
    processing: bool = False

    # The name of the new chat.
    new_chat_name: str = ""

    def is_valid_setting(self) -> bool:
        return self.api_key != "" and self.prompt != ""

    def create_chat(self):
        """Create a new chat."""
        # Add the new chat to the list of chats.
        self.current_chat = self.new_chat_name
        self.chats[self.new_chat_name] = []

    def delete_chat(self):
        """Delete the current chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = DEFAULT_CHATS
        self.current_chat = list(self.chats.keys())[0]

    def set_chat(self, chat_name: str):
        """Set the name of the current chat.

        Args:
            chat_name: The name of the chat.
        """
        self.current_chat = chat_name

    @rx.var
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles.

        Returns:
            The list of chat names.
        """
        return list(self.chats.keys())

    async def handle_upload(self, files: list[rx.UploadFile]):
        """Handle the upload of file(s).
        Args:
            files: The uploaded files.
        """
        self.pdf_processing = False
        for file in files:
            upload_data = await file.read()
            outfile = rx.get_upload_dir() / file.filename
            # Save the file.
            with outfile.open("wb") as file_object:
                file_object.write(upload_data)

        self.pdf_processing = True
        self.pdf_path = outfile

    async def handle_la(self):
        self.loader_processing = False
        self.layzer = UpstageLayoutAnalysisLoader(
            self.pdf_path, output_type="html", api_key=self.api_key
        )
        self.docs = self.layzer.load()
        os.remove(self.pdf_path)
        self.loader_processing = True

    async def process_question(self, form_data: dict[str, str]):
        # Get the question from the form
        question = form_data["question"]

        # Check if the question is empty
        if question == "":
            return

        model = self.openai_process_question

        async for value in model(question):
            yield value

    async def openai_process_question(self, question: str):
        """Get the response from the API.

        Args:
            form_data: A dict with the current question.
        """

        # Add the question to the list of questions.
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)

        # Clear the input and start the processing.
        self.processing = True
        yield

        llm = ChatUpstage(api_key=self.api_key)

        prompt_template = PromptTemplate.from_template(self.prompt)
        chain = prompt_template | llm | StrOutputParser()
        answer_text = chain.invoke({"question": question, "Context": self.docs})

        self.chats[self.current_chat][-1].answer += answer_text

        # Toggle the processing flag.
        self.processing = False
