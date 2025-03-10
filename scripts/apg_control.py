import logging
import sys
import traceback
from typing import Any
from functools import partial
import gradio as gr
from modules import script_callbacks, scripts
from APGIsYourCFG.nodes_APGImYourCFGNow import APG_ImYourCFGNow

class APGControlScript(scripts.Script):
    def __init__(self):
        # APG parameters
        self.apg_enabled = False
        self.apg_moment = 0.5
        self.apg_adaptive_moment = 0.180
        self.apg_norm_thr = 15.0
        self.apg_eta = 1.0



    sorting_priority = 15

    def title(self):
        return 'APG Control'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            # APG Section
            gr.HTML("<p><b>APG Settings</b></p>")
            apg_enabled = gr.Checkbox(label="Enable APG", value=self.apg_enabled)
            with gr.Group(visible=True):
                apg_momentum = gr.Slider(
                    label="APG Momentum",
                    minimum=-2.5,
                    maximum=2.5,
                    step=0.01,
                    value=self.apg_moment,
                )
                apg_adaptive_momentum = gr.Slider(
                    label="APG Adaptive Momentum",
                    minimum=-2.5,
                    maximum=2.5,
                    step=0.01,
                    value=self.apg_adaptive_moment,
                )
                apg_norm_thr = gr.Slider(
                    label="APG Norm Threshold",
                    minimum=0.0,
                    maximum=100.0,
                    step=0.5,
                    value=self.apg_norm_thr,
                )
                apg_eta = gr.Slider(
                    label="APG Eta", 
                    minimum=0.0, 
                    maximum=2.0, 
                    step=0.1, 
                    value=self.apg_eta
                )
            with gr.Row():
                apg_off_type = gr.Radio(
                    ["None", "Disable for N Steps", "Disable for N% of Steps", "Disable on Specific Steps"],
                    label="APG Disable Type",
                    value="None",
                    interactive=True,
                )
            with gr.Row(visible=True) as none_row:
                pass
            with gr.Row(visible=False) as steps_count_row:
                apg_off_steps = gr.Slider(
                    minimum=1,
                    maximum=150,
                    step=1,
                    value=0,
                    label="Number of Steps to Disable APG",
                    interactive=True,
                )
            with gr.Row(visible=False) as steps_percent_row:
                apg_off_percent = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=0,
                    label="Percentage of Steps to Disable APG",
                    interactive=True,
                )
            with gr.Row(visible=False) as steps_specific_row:
                apg_off_specific_steps = gr.Textbox(
                    label="Specific Steps to Disable APG (comma-separated)",
                    value="",
                    interactive=True,
                )
            def update_visibility(apg_type):
                return [
                    gr.update(visible=(apg_type == "Disable for N Steps")),
                    gr.update(visible=(apg_type == "Disable for N% of Steps")),
                    gr.update(visible=(apg_type == "Disable on Specific Steps")),
                ]

            apg_off_type.change(
                fn=update_visibility,
                inputs=apg_off_type,
                outputs=[none_row, steps_count_row, steps_percent_row, steps_specific_row],
            )
        return (apg_enabled, apg_momentum, apg_adaptive_momentum, apg_norm_thr, apg_eta,
                apg_off_type, apg_off_steps, apg_off_percent, apg_off_specific_steps)

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 9:
            (
                self.apg_enabled,
                self.apg_moment,
                self.apg_adaptive_moment,
                self.apg_norm_thr,
                self.apg_eta,
                self.apg_off_type,
                self.apg_off_steps,
                self.apg_off_percent,
                self.apg_off_specific_steps,
            ) = args[:9]
        else:
            logging.warning(
                "Not enough arguments provided to process_before_every_sampling"
            )
            return

        # Retrieve values from XYZ plot if available
        xyz = getattr(p, "_apg_xyz", {})
        if "apg_enabled" in xyz:
            self.apg_enabled = xyz["apg_enabled"] == "True"
        if "apg_moment" in xyz:
            self.apg_moment = xyz["apg_moment"]
        if "apg_adaptive_moment" in xyz:
            self.apg_adaptive_moment = xyz["apg_adaptive_moment"]
        if "apg_norm_thr" in xyz:
            self.apg_norm_thr = xyz["apg_norm_thr"]
        if "apg_eta" in xyz:
            self.apg_eta = xyz["apg_eta"]
        if "apg_off_type" in xyz:
            self.apg_off_type = xyz["apg_off_type"]
        if "apg_off_steps" in xyz:
            self.apg_off_steps = xyz["apg_off_steps"]
        if "apg_off_percent" in xyz:
            self.apg_off_percent = xyz["apg_off_percent"]
        if "apg_off_specific_steps" in xyz:
            self.apg_off_specific_steps = xyz["apg_off_specific_steps"]

        # Always start with a fresh clone of the original unet
        unet = p.sd_model.forge_objects.unet.clone()

        # If neither feature is enabled, return original unet
        if not self.apg_enabled:
            p.sd_model.forge_objects.unet = unet
            return

        # Configure parameters based on what's enabled
        patch_params = {
            # APG parameters (only applied if APG is enabled)
            "momentum": self.apg_moment if self.apg_enabled else 0,
            "adaptive_momentum": self.apg_adaptive_moment if self.apg_enabled else 0,
            "norm_threshold": self.apg_norm_thr if self.apg_enabled else 0,
            "eta": self.apg_eta if self.apg_enabled else 1.0,
            # Guidance limiter parameters (only applied if limiter is enabled)
            "guidance_limiter": False,
            "guidance_sigma_start": -1,
            "guidance_sigma_end": -1,
            # APG control parameters
            "apg_off_type": self.apg_off_type if self.apg_enabled else "None",
            "apg_off_steps": self.apg_off_steps if self.apg_enabled else 0,
            "apg_off_percent": self.apg_off_percent if self.apg_enabled else 0,
            "apg_off_specific_steps": self.apg_off_specific_steps if self.apg_enabled else 0,
        }

        unet = APG_ImYourCFGNow().patch(unet, **patch_params)[0]

        p.sd_model.forge_objects.unet = unet
        
        # Only include enabled features in generation params
        args = {}
        if self.apg_enabled:
            args.update({
                "apgisyourcfg_enabled": True,
                "apgisyourcfg_momentum": self.apg_moment,
                "apgisyourcfg_adaptive_momentum": self.apg_adaptive_moment,
                "apgisyourcfg_norm_thr": self.apg_norm_thr,
                "apgisyourcfg_eta": self.apg_eta,
            })
        if self.apg_off_type == "None":
            args.update({
                "apgisyourcfg_off_type": "None",
            })
        if self.apg_off_type == "Disable for N Steps":
            args.update({
                "apgisyourcfg_off_type": "N",
                "apgisyourcfg_off_steps": self.apg_off_steps,
            })
        if self.apg_off_type == "Disable for N% of Steps":
            args.update({
                "apgisyourcfg_off_type": "N%",
                "apgisyourcfg_off_steps": self.apg_off_percent,
            })
        if self.apg_off_type == "Disable on Specific Steps":
            args.update({
                "apgisyourcfg_off_type": "Steps",
                "apgisyourcfg_off_steps": self.apg_off_specific_steps,
            })
            
        p.extra_generation_params.update(args)
        str_args:str = ", ".join([f"{k}:\"{v}\"" for k,v in args.items()])
        logging.debug("WOLOLO: \"APG is now your CFG!\"")
        logging.debug(str_args)

        return

# XYZ plot functionality
def set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_apg_xyz"):
        p._apg_xyz = {}
    p._apg_xyz[field] = x

def make_axis_on_xyz_grid():
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break

    if xyz_grid is None:
        return

    axis = [
        xyz_grid.AxisOption(
            "(APG) APG Enabled",
            str,
            partial(set_value, field="apg_enabled"),
            choices=lambda: ["True", "False"]
        ),
        xyz_grid.AxisOption(
            "(APG) APG Momentum",
            float,
            partial(set_value, field="apg_moment"),
        ),
        xyz_grid.AxisOption(
            "(APG) APG Adaptive Momentum",
            float,
            partial(set_value, field="apg_adaptive_moment"),
        ),
        xyz_grid.AxisOption(
            "(APG) APG Norm Threshold",
            float,
            partial(set_value, field="apg_norm_thr"),
        ),
        xyz_grid.AxisOption(
            "(APG) APG Eta",
            float,
            partial(set_value, field="apg_eta"),
        ),
        xyz_grid.AxisOption(
            "(APG) APG Disable Type",
            str,
            partial(set_value, field="apg_off_type"),
        ),
        xyz_grid.AxisOption(
            "(APG) APG Disable Steps",
            int,
            partial(set_value, field="apg_off_steps"),
        ),
        xyz_grid.AxisOption(
            "(APG) APG Disable Percent",
            int,
            partial(set_value, field="apg_off_percent"),
        ),
        xyz_grid.AxisOption(
            "(APG) APG Disable Specific Steps",
            str,
            partial(set_value, field="apg_off_specific_steps"),
        ),
    ]

    if not any(x.label.startswith("(APG)") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)

def on_before_ui():
    try:
        make_axis_on_xyz_grid()
    except Exception:
        error = traceback.format_exc()
        print(
            f"[-] APG Script: xyz_grid error:\n{error}",
            file=sys.stderr,
        )

script_callbacks.on_before_ui(on_before_ui)
