"""Gradio interface for TTS Dataset Validator."""
import gradio as gr

from core.state import clear_dataset_cache, get_dataset_cache
from ui.styles import custom_css, head_js
from analysis.standard import run_analysis
from analysis.smart import run_smart_analysis
from analysis.import_export import (
    create_verified_dataset,
    import_csv_analysis,
    save_results_csv,
    verify_action,
)
from hf_asr import get_all_asr_model_choices


def update_thinking_visibility(model_name: str):
    """Update thinking budget visibility based on model."""
    return gr.update(visible="thinking" in model_name)


def clear_cache():
    """Clear the dataset cache."""
    count = clear_dataset_cache()
    return f"<p style='color: #34d399;'>✅ Кэш ачышчаны ({count} датасет(аў) выдалена)</p>"


def get_cache_status():
    """Get the current cache status."""
    cache = get_dataset_cache()
    if not cache:
        return "<p style='color: #94a3b8;'>📭 Кэш пусты</p>"

    total_items = sum(len(cache[key]) for key in cache)
    return (
        f"<p style='color: #60a5fa;'>📦 Закэшавана: {len(cache)} "
        f"датасет(аў), {total_items} элементаў</p>"
    )


def _run_smart_analysis_with_mode(
    dataset_name,
    limit_files,
    temperature,
    thinking_budget,
    similarity_threshold,
    execution_mode,
    recheck_problematic,
    hf_token,
    progress=gr.Progress(),
):
    """Map UI execution mode to the existing smart analysis flow."""
    if execution_mode == "batch":
        raise gr.Error("Batch mode is currently supported only for the standard analysis button.")

    return run_smart_analysis(
        dataset_name,
        limit_files,
        temperature,
        thinking_budget,
        similarity_threshold,
        flex_mode=execution_mode == "flex",
        recheck_problematic=recheck_problematic,
        hf_token=hf_token,
        progress=progress,
    )


def create_interface():
    """Create and return the Gradio interface."""
    with gr.Blocks(
        css=custom_css,
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue"),
        head=head_js,
    ) as demo:
        gr.HTML("<h1 class='main-title'>TTS Dataset Validator</h1>", sanitize=False)
        gr.HTML(
            (
                "<p class='subtitle'>"
                "Аналіз аўдыядатасетаў для выяўлення несупадзенняў "
                "паміж метаданымі і гукам"
                "</p>"
            ),
            sanitize=False,
        )

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["settings-panel"]):
                gr.Markdown("### ⚙️ Налады")
                gr.Markdown(
                    "Vertex AI auth uses ADC from `gcloud auth application-default login` plus "
                    "`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and for batch mode "
                    "`VERTEX_BATCH_GCS_URI`."
                )

                hf_token = gr.Textbox(
                    label="Hugging Face Token",
                    type="password",
                    placeholder=(
                        "Увядзіце ваш HF Token "
                        "(для доступу да зачыненых дадзеных і мадэляў)..."
                    ),
                    elem_id="hf_token_input",
                )

                dataset_name = gr.Textbox(
                    label="Hugging Face Dataset",
                    value="archivartaunik/Melez_Tryvozhnae_Schasce_testpart",
                    placeholder="username/dataset_name",
                )

                asr_choices = get_all_asr_model_choices()
                default_model = "gemini-2.5-flash-lite"
                model_name = gr.Radio(
                    label="Мадэль распазнавання",
                    choices=asr_choices,
                    value=(
                        default_model
                        if default_model in asr_choices
                        else (asr_choices[0] if asr_choices else None)
                    ),
                    interactive=True,
                )

                limit_files = gr.Number(
                    label="Ліміт файлаў (0 = усе)",
                    value=1,
                    minimum=0,
                    step=1,
                )

                gr.Markdown("### 🎛️ Генерацыя")

                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=2.0,
                    value=0.3,
                    step=0.1,
                )

                thinking_budget = gr.Number(
                    label="Thinking Budget (токены)",
                    value=1024,
                    minimum=0,
                    step=128,
                    visible=False,
                )

                similarity_threshold = gr.Slider(
                    label="Парог несупадзенняў (%)",
                    minimum=0,
                    maximum=100,
                    value=99,
                    step=1,
                    info="Файлы з скорам ніжэй будуць пазначаны",
                )

                execution_mode = gr.Radio(
                    label="Рэжым Vertex",
                    choices=[
                        ("Direct Request", "direct"),
                        ("Flex Mode", "flex"),
                        ("Batch Mode", "batch"),
                    ],
                    value="direct",
                    interactive=True,
                    info=(
                        "Direct = inline request, Flex = preview shared capacity, "
                        "Batch = real Vertex Batch API via GCS/JSONL."
                    ),
                )

                analyze_btn = gr.Button(
                    "🚀 Пачаць аналіз",
                    variant="primary",
                    elem_classes=["primary-btn"],
                )

                smart_analyze_btn = gr.Button(
                    "🧠 Разумны аналіз",
                    variant="secondary",
                    elem_classes=["smart-btn"],
                )

                recheck_problematic = gr.Checkbox(
                    label="Пераправерыць толькі праблемныя файлы",
                    value=False,
                    info=(
                        "Калі ўключана, аналіз будзе запускацца толькі для файлаў "
                        "з нізкім рэйтынгам."
                    ),
                )

                stop_btn = gr.Button(
                    "🛑 Спыніць аналіз",
                    variant="stop",
                    elem_classes=["stop-btn"],
                    visible=True,
                )

                gr.HTML(
                    (
                        "<small style='color: #94a3b8;'>"
                        "🧠 Разумны аналіз выкарыстоўвае 3 мадэлі паслядоўна: "
                        "Flash-Lite → Flash → Gemini-3-Flash. "
                        "Flex PayGo uses the Gemini 3 preview chain on `global`. "
                        "Batch mode currently works only with the standard analysis button."
                        "</small>"
                    ),
                    sanitize=False,
                )

                gr.Markdown("### 📂 Імпарт CSV")
                with gr.Row():
                    import_file = gr.File(label="Загрузіць CSV", file_types=[".csv"])
                    import_btn = gr.Button("📥 Імпартаваць", variant="secondary")

                gr.Markdown("### 💾 Кэш")
                cache_status = gr.HTML(
                    value="<p style='color: #94a3b8;'>📭 Кэш пусты</p>",
                    sanitize=False,
                )
                clear_cache_btn = gr.Button("🗑️ Ачысціць кэш датасету", size="sm")

            with gr.Column(scale=2, elem_classes=["results-panel"]):
                gr.Markdown("### 📊 Статыстыка")
                stats_output = gr.HTML(sanitize=False)

                gr.Markdown("### 🚩 Праблемныя файлы")
                flagged_output = gr.HTML(sanitize=False)

                with gr.Accordion("📋 Усе вынікі (табліца)", open=False):
                    results_table = gr.DataFrame(
                        headers=[
                            "path",
                            "score",
                            "model_used",
                            "verification_status",
                            "ref_text",
                            "hyp_text",
                        ],
                        wrap=True,
                    )

                gr.Markdown("### 📤 Экспарт")
                with gr.Row():
                    download_btn = gr.Button("💾 Спампаваць вынікі (CSV)", size="lg")
                    download_file = gr.File(label="Файл вынікаў", file_count="single")

                with gr.Row():
                    create_ds_btn = gr.Button(
                        "🤗 Стварыць правераны датасэт",
                        variant="primary",
                    )
                    create_ds_output = gr.Markdown()

                verification_data = gr.Textbox(elem_id="verification_data_input", visible=True)
                verification_trigger = gr.Button(elem_id="verification_trigger_btn", visible=True)

        model_name.change(
            fn=update_thinking_visibility,
            inputs=[model_name],
            outputs=[thinking_budget],
        )

        analyze_event = analyze_btn.click(
            fn=run_analysis,
            inputs=[
                dataset_name,
                model_name,
                limit_files,
                temperature,
                thinking_budget,
                similarity_threshold,
                execution_mode,
                recheck_problematic,
                hf_token,
            ],
            outputs=[stats_output, flagged_output, results_table],
        )

        analyze_event.then(
            fn=get_cache_status,
            inputs=[],
            outputs=[cache_status],
        )

        smart_analyze_event = smart_analyze_btn.click(
            fn=_run_smart_analysis_with_mode,
            inputs=[
                dataset_name,
                limit_files,
                temperature,
                thinking_budget,
                similarity_threshold,
                execution_mode,
                recheck_problematic,
                hf_token,
            ],
            outputs=[stats_output, flagged_output, results_table],
        )

        smart_analyze_event.then(
            fn=get_cache_status,
            inputs=[],
            outputs=[cache_status],
        )

        from core.state import set_stop_requested

        stop_btn.click(
            fn=lambda: [set_stop_requested(True), gr.Info("Спыненне аналізу...")][0],
            inputs=None,
            outputs=None,
            cancels=[analyze_event, smart_analyze_event],
        )

        download_btn.click(
            fn=save_results_csv,
            inputs=[dataset_name],
            outputs=[download_file],
        )

        clear_cache_btn.click(
            fn=clear_cache,
            inputs=[],
            outputs=[cache_status],
        )

        import_btn.click(
            fn=import_csv_analysis,
            inputs=[import_file, similarity_threshold, dataset_name, limit_files],
            outputs=[stats_output, flagged_output, results_table],
        )

        create_ds_btn.click(
            fn=create_verified_dataset,
            inputs=[hf_token, dataset_name],
            outputs=[create_ds_output],
        )

        verification_trigger.click(
            fn=verify_action,
            inputs=[verification_data, similarity_threshold, dataset_name],
            outputs=[stats_output, flagged_output, results_table],
        )

    return demo
