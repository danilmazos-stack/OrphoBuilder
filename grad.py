import gradio as gr
from ollama_client import fix_text_iter, fix_image_iter, fix_file_iter, list_models, MODEL

EXAMPLES = [
    ["Я вчира пошол в магазин и купил хлеп."],
    ["Their going to the store tomorow, and I seen him yesterday."],
    ["мы все живём в москве. я люблю свой горат."],
]


def get_models():
    models = list_models()
    return models if models else [MODEL]


def process_text(text, model, show_changes):
    if not text or not text.strip():
        yield "Введите текст."
        return
    yield from fix_text_iter(text, model, show_changes)


def process_image(image_path, model, show_changes):
    if not image_path:
        yield "Загрузите картинку."
        return
    yield from fix_image_iter(image_path, model, show_changes)


def refresh_models():
    models = get_models()
    return gr.Dropdown(choices=models, value=models[0])

def process_file(path, model, show_changes):
    if not path:
        yield "Загрузите файл."
        return
    yield from fix_file_iter(path, model, show_changes)


with gr.Blocks(title="Исправление ошибок") as demo:
    gr.Markdown(
        "# OrphoBuilder\n"
        "Проверка орфографии и грамматики в тексте и на картинках (русский и английский)."
    )

    with gr.Row():
        models = get_models()
        model = gr.Dropdown(
            models,
            value=MODEL if MODEL in models else models[0],
            label="Модель",
            scale=4,
        )
        refresh_btn = gr.Button("Обновить список", scale=1)
    refresh_btn.click(refresh_models, None, model)

    show_changes = gr.Checkbox(label="Показать список правок", value=False)

    with gr.Tab("Текст"):
        with gr.Row():
            with gr.Column():
                text_in = gr.Textbox(label="Текст", lines=10, placeholder="Вставьте текст здесь...")
                with gr.Row():
                    text_clear = gr.Button("Очистить")
                    text_btn = gr.Button("Исправить", variant="primary")
            with gr.Column():
                text_out = gr.Textbox(label="Результат", lines=14, buttons=["copy"])
        gr.Examples(EXAMPLES, inputs=text_in, label="Примеры")

        text_btn.click(process_text, [text_in, model, show_changes], text_out)
        text_in.submit(process_text, [text_in, model, show_changes], text_out)
        text_clear.click(lambda: ("", ""), None, [text_in, text_out])

    with gr.Tab("Картинка"):
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="filepath", label="Картинка", height=350)
                with gr.Row():
                    img_clear = gr.Button("Очистить")
                    img_btn = gr.Button("Исправить", variant="primary")
            with gr.Column():
                img_out = gr.Textbox(label="Результат", lines=14, buttons=["copy"])

        img_btn.click(process_image, [img_in, model, show_changes], img_out)
        img_clear.click(lambda: (None, ""), None, [img_in, img_out])

    with gr.Tab("Файл"):
        with gr.Row():
            with gr.Column():
                file_in = gr.File(label="Файл (.docx, .pdf, .txt, картинка)", type="filepath")
                file_btn = gr.Button("Исправить", variant="primary")
            with gr.Column():
                file_out = gr.Textbox(label="Результат", lines=14, buttons=["copy"])
        file_btn.click(process_file, [file_in, model, show_changes], file_out)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), inbrowser=True)