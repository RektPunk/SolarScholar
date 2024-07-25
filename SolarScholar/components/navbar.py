import reflex as rx
from SolarScholar.state import ChatState


def sidebar_chat(chat: str) -> rx.Component:
    """A sidebar chat item.

    Args:
        chat: The chat item.
    """
    return rx.drawer.close(
        rx.hstack(
            rx.button(
                chat,
                on_click=lambda: ChatState.set_chat(chat),
                width="80%",
                variant="surface",
            ),
            rx.button(
                rx.icon(
                    tag="trash",
                    on_click=ChatState.delete_chat,
                    stroke_width=1,
                ),
                width="20%",
                variant="surface",
                color_scheme="red",
            ),
            width="100%",
        )
    )


def sidebar(trigger) -> rx.Component:
    """The sidebar component."""
    return rx.drawer.root(
        rx.drawer.trigger(trigger),
        rx.drawer.overlay(),
        rx.drawer.portal(
            rx.drawer.content(
                rx.vstack(
                    rx.heading("Chats", color=rx.color("mauve", 11)),
                    rx.divider(),
                    rx.foreach(ChatState.chat_titles, lambda chat: sidebar_chat(chat)),
                    align_items="stretch",
                    width="100%",
                ),
                top="auto",
                right="auto",
                height="100%",
                width="20em",
                padding="2em",
                background_color=rx.color("mauve", 2),
                outline="none",
            )
        ),
        direction="left",
    )


def pdfbar(trigger) -> rx.Component:
    """The pdfbar component."""
    return rx.drawer.root(
        rx.drawer.trigger(trigger),
        rx.drawer.overlay(),
        rx.drawer.portal(
            rx.drawer.content(
                rx.vstack(
                    rx.heading("Document", color=rx.color("mauve", 11)),
                    rx.divider(),
                    rx.markdown(
                        """This app is created to test Solar LayoutAnalyzer. Please visit to the [Upstage AI](https://developers.upstage.ai/) for information on other APIs. </br>
                        You need to not only provide the API key but also 'Upload' the PDF and click 'Learn'. </br>
                        If everything is ready, you will see the message 'I'm ready to chat.'"""
                    ),
                    rx.upload(
                        rx.vstack(
                            rx.button(
                                "File",
                                color=rx.color("mauve", 11),
                                bg="white",
                                border=f"1px solid {rx.color("mauve", 11)}",
                            ),
                            rx.text(
                                "Drag and drop files here or click to select files"
                            ),
                        ),
                        id="upload",
                        multiple=False,
                        accept={
                            "application/pdf": [".pdf"],
                        },
                        max_files=1,
                        disabled=False,
                        border=f"1px dotted {rx.color("mauve", 11)}",
                        padding="5em",
                    ),
                    rx.hstack(
                        rx.button(
                            "Upload",
                            on_click=ChatState.handle_upload(
                                rx.upload_files(upload_id="upload")
                            ),
                            width="50%",
                        ),
                        rx.button(
                            "Learn",
                            on_click=ChatState.handle_la(),
                            width="50%",
                        ),
                        width="100%",
                    ),
                    rx.cond(
                        ChatState.pdf_processing,
                        rx.text("PDF upload completed."),
                        rx.text(""),
                    ),
                    rx.cond(
                        ChatState.loader_processing,
                        rx.text("I'm ready to chat."),
                        rx.text(""),
                    ),
                    align_items="stretch",
                    width="100%",
                ),
                top="auto",
                right="auto",
                height="100%",
                width="20em",
                padding="2em",
                background_color=rx.color("mauve", 2),
                outline="none",
            )
        ),
        direction="left",
    )


def settingbar(trigger) -> rx.Component:
    """The sidebar component."""
    return rx.drawer.root(
        rx.drawer.trigger(trigger),
        rx.drawer.overlay(),
        rx.drawer.portal(
            rx.drawer.content(
                rx.vstack(
                    rx.heading("Settings", color=rx.color("mauve", 11)),
                    rx.divider(),
                    rx.markdown(
                        "This app is created to test Solar LayoutAnalyzer. Please visit to the [Upstage AI](https://developers.upstage.ai/) for information on API keys."
                    ),
                    rx.divider(),
                    rx.heading("API KEY", color=rx.color("mauve", 11), size="3"),
                    rx.input(
                        name="api_key",
                        default_value=ChatState.api_key,
                        placeholder="Input api key here",
                        on_change=ChatState.set_api_key,
                        type="password",
                        required=True,
                    ),
                    rx.heading("Prompt", color=rx.color("mauve", 11), size="3"),
                    rx.text_area(
                        value=ChatState.prompt,
                        on_change=ChatState.set_prompt,
                        resize="vertical",
                    ),
                    rx.button(
                        "Apply",
                        on_click=rx.toast.success("Succesfully Applied!"),
                    ),
                    align_items="stretch",
                    width="100%",
                ),
                top="auto",
                right="auto",
                height="100%",
                width="20em",
                padding="2em",
                background_color=rx.color("mauve", 2),
                outline="none",
            )
        ),
        direction="left",
    )


def modal(trigger) -> rx.Component:
    """A modal to create a new chat."""
    return rx.dialog.root(
        rx.dialog.trigger(trigger),
        rx.dialog.content(
            rx.hstack(
                rx.input(
                    placeholder="Type new chat name",
                    on_blur=ChatState.set_new_chat_name,
                    width=["15em", "20em", "30em", "30em", "30em", "30em"],
                ),
                rx.dialog.close(
                    rx.button(
                        "Create chat",
                        on_click=ChatState.create_chat,
                    ),
                ),
                background_color=rx.color("mauve", 1),
                spacing="2",
                width="100%",
            ),
        ),
    )


def navbar():
    return rx.box(
        rx.hstack(
            rx.hstack(
                rx.avatar(src="avatar.png"),
                rx.heading("SolarScholar"),
                rx.desktop_only(
                    rx.badge(
                        ChatState.current_chat,
                        rx.tooltip(
                            rx.icon("info", size=14),
                            content="The current selected chat.",
                        ),
                        variant="soft",
                    )
                ),
                align_items="center",
            ),
            rx.hstack(
                modal(rx.button("+ New chat")),
                sidebar(
                    rx.button(
                        rx.icon(
                            tag="messages-square",
                            color=rx.color("mauve", 12),
                        ),
                        background_color=rx.color("mauve", 6),
                    )
                ),
                settingbar(
                    rx.button(
                        rx.icon(
                            tag="settings",
                            color=rx.color("mauve", 12),
                        ),
                        background_color=rx.color("mauve", 6),
                    )
                ),
                pdfbar(
                    rx.button(
                        rx.icon(
                            tag="file_up",
                            color=rx.color("mauve", 12),
                        ),
                        background_color=rx.color("mauve", 6),
                    )
                ),
                align_items="center",
            ),
            justify_content="space-between",
            align_items="center",
        ),
        backdrop_filter="auto",
        backdrop_blur="lg",
        padding="12px",
        border_bottom=f"1px solid {rx.color('mauve', 3)}",
        background_color=rx.color("mauve", 2),
        position="sticky",
        top="0",
        z_index="100",
        align_items="center",
    )
