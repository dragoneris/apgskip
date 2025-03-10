import gradio as gr
from modules import scripts, shared
from modules.processing import Processed, process_images
from modules.sd_samplers import samplers_for_img2img
import math

class CFGControlScript(scripts.Script):
    def __init__(self):
        super().__init__()

    def title(self):
        return "CFG Control"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("CFG Control", open=False):
            with gr.Row():
                cfg_off_type = gr.Radio(
                    ["Disable for N Steps", "Disable for N% of Steps", "Disable on Specific Steps"],
                    label="CFG Disable Type",
                    value="Disable for N Steps",
                    interactive=True,
                )
            with gr.Row(visible=True) as steps_count_row:
                cfg_off_steps = gr.Slider(
                    minimum=1,
                    maximum=150,
                    step=1,
                    value=0,
                    label="Number of Steps to Disable CFG",
                    interactive=True,
                )
            with gr.Row(visible=False) as steps_percent_row:
                cfg_off_percent = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=0,
                    label="Percentage of Steps to Disable CFG",
                    interactive=True,
                )
            with gr.Row(visible=False) as steps_specific_row:
                cfg_off_specific_steps = gr.Textbox(
                    label="Specific Steps to Disable CFG (comma-separated)",
                    value="",
                    interactive=True,
                )
            with gr.Row():
                cfg_off_zero_start = gr.Checkbox(label="Start with zero CFG", value=False)

            def update_visibility(cfg_type):
                return [
                    gr.update(visible=(cfg_type == "Disable for N Steps")),
                    gr.update(visible=(cfg_type == "Disable for N% of Steps")),
                    gr.update(visible=(cfg_type == "Disable on Specific Steps")),
                ]

            cfg_off_type.change(
                fn=update_visibility,
                inputs=cfg_off_type,
                outputs=[steps_count_row, steps_percent_row, steps_specific_row],
            )

        return [cfg_off_type, cfg_off_steps, cfg_off_percent, cfg_off_specific_steps, cfg_off_zero_start]

    def process(self, p, cfg_off_type, cfg_off_steps, cfg_off_percent, cfg_off_specific_steps, cfg_off_zero_start):
        if cfg_off_type == "Disable for N Steps" and cfg_off_steps > 0:
            self.disable_cfg_for_n_steps(p, cfg_off_steps, cfg_off_zero_start)
        elif cfg_off_type == "Disable for N% of Steps" and cfg_off_percent > 0:
            self.disable_cfg_for_n_percent(p, cfg_off_percent, cfg_off_zero_start)
        elif cfg_off_type == "Disable on Specific Steps" and cfg_off_specific_steps:
            self.disable_cfg_on_specific_steps(p, cfg_off_specific_steps)

    def disable_cfg_for_n_steps(self, p, steps_to_disable, cfg_off_zero_start):
        original_cfg = p.cfg_scale

        def custom_cfg_function(x):
            if cfg_off_zero_start and x==0:
                return 0
            elif x < steps_to_disable:
                return 0
            else:
                return original_cfg
        p.cfg_scale = custom_cfg_function

    def disable_cfg_for_n_percent(self, p, percent_to_disable, cfg_off_zero_start):
        original_cfg = p.cfg_scale
        steps_to_disable = math.ceil(p.steps * (percent_to_disable / 100))

        def custom_cfg_function(x):
            if cfg_off_zero_start and x==0:
                return 0
            elif x < steps_to_disable:
                return 0
            else:
                return original_cfg
        p.cfg_scale = custom_cfg_function

    def disable_cfg_on_specific_steps(self, p, specific_steps_string):
        original_cfg = p.cfg_scale
        try:
            specific_steps = [int(s.strip()) for s in specific_steps_string.split(",")]
        except ValueError:
            print("CFG Control: Invalid specific steps format. Using original CFG.")
            return

        def custom_cfg_function(x):
            if x in specific_steps:
                return 0
            else:
                return original_cfg
        p.cfg_scale = custom_cfg_function
