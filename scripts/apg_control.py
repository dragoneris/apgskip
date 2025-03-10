import logging
import sys
import traceback
from typing import Any
from functools import partial
import gradio as gr
from modules import script_callbacks, scripts, shared
from ldm_patched.modules.model_patcher import ModelPatcher
import torch

class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(
    v0: torch.Tensor,
    v1: torch.Tensor,
):
    dtype = v0.dtype
    # v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def normalized_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: MomentumBuffer = None,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
):
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update

    return pred_guided


class APG_ImYourCFGNow:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "momentum": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": -1.5,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.001,
                    },
                ),
                "adaptive_momentum": (
                    "FLOAT",
                    {
                        "default": 0.180,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "round": 0.001,
                    },
                ),
                "norm_threshold": (
                    "FLOAT",
                    {
                        "default": 15.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.05,
                        "round": 0.01,
                    },
                ),
                "eta": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "guidance_limiter": ("BOOLEAN", {"default": False}),
                "guidance_sigma_start": ("FLOAT", {"default": 5.42, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "guidance_sigma_end": ("FLOAT", {"default": 0.28, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "print_data": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "apg_off_type": (
                    ["None", "Disable for N Steps", "Disable for N% of Steps", "Disable on Specific Steps"],
                    {"default": "None"},
                ),
                "apg_off_steps": ("INT", {"default": 0, "min": 0, "max": 150}),
                "apg_off_percent": ("INT", {"default": 0, "min": 0, "max": 100}),
                "apg_off_specific_steps": ("STRING", {"default": ""}),
                
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"

    def patch(
        self,
        model: ModelPatcher,
        momentum: float = 0.5,
        adaptive_momentum: float = 0.180,
        norm_threshold: float = 15.0,
        eta: float = 1.0,
        guidance_limiter: bool = False,
        guidance_sigma_start: float = 5.42,
        guidance_sigma_end: float = 0.28,
        apg_off_type: str = "None",
        apg_off_steps: int = 0,
        apg_off_percent: int = 0,
        apg_off_specific_steps: str = "",
        print_data=False,
        extras=[],
    ):
        momentum_buffer = MomentumBuffer(momentum)
        current_step = 0
        extras = [momentum_buffer, momentum, adaptive_momentum, apg_off_type, apg_off_steps, apg_off_percent, apg_off_specific_steps, current_step]

        def apg_function(args):
            cond = args["cond"]
            uncond = args["uncond"]
            sigma = args["sigma"]
            model = args["model"]
            cond_scale = args["cond_scale"]

            momentum_buffer = extras[0]
            momentum = extras[1]
            adaptive_momentum = extras[2]
            apg_off_type = extras[3]
            apg_off_steps = extras[4]
            apg_off_percent = extras[5]
            apg_off_specific_steps = extras[6]
            current_step = extras[7]
            
            t = model.model_sampling.timestep(sigma)[0].item()

            if (
                torch.is_tensor(momentum_buffer.running_average)
                and (cond.shape[3] != momentum_buffer.running_average.shape[3])
            ) or t == 999:
                momentum_buffer = MomentumBuffer(momentum)
                extras[0] = momentum_buffer
            else:
                signal_scale = momentum
                if adaptive_momentum > 0:
                    if momentum < 0:
                        signal_scale += -momentum * (adaptive_momentum**4) * (1000 - t)
                        if signal_scale > 0:
                            signal_scale = 0
                    else:
                        signal_scale -= momentum * (adaptive_momentum**4) * (1000 - t)
                        if signal_scale < 0:
                            signal_scale = 0

                momentum_buffer.momentum = signal_scale

            if print_data:
                print(" momentum: ", momentum_buffer.momentum, " t: ", t)

            
            # Check if APG should be turned off based on the selected method
            if apg_off_type == "Disable for N Steps" and current_step < apg_off_steps:
                return cond
            elif apg_off_type == "Disable on Specific Steps":
                try:
                    specific_steps = [int(s.strip()) for s in apg_off_specific_steps.split(",") if s.strip().isdigit()]
                    if current_step in specific_steps:
                        return cond
                except ValueError:
                    print(f"Invalid format for specific steps: {apg_off_specific_steps}")
                    pass # handle it gracefully, possibly ignore it

            return normalized_guidance(
                cond, uncond, cond_scale, momentum_buffer, eta, norm_threshold
            )
            extras[7] = current_step + 1

        m = model.clone()
        m.set_model_sampler_cfg_function(apg_function, extras==extras)
        m.model_options["disable_cfg1_optimization"] = False

        return (m,)

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
                    ["None", "Disable for N Steps", "Disable on Specific Steps"],
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
            with gr.Row(visible=False) as steps_specific_row:
                apg_off_specific_steps = gr.Textbox(
                    label="Specific Steps to Disable APG (comma-separated)",
                    value="",
                    interactive=True,
                )
            def update_visibility(apg_type):
                return [
                    gr.update(visible=(apg_type == "None")),
                    gr.update(visible=(apg_type == "Disable for N Steps")),
                    gr.update(visible=(apg_type == "Disable on Specific Steps")),
                ]

            apg_off_type.change(
                fn=update_visibility,
                inputs=apg_off_type,
                outputs=[none_row, steps_count_row, steps_specific_row],
            )
        return (apg_enabled, apg_momentum, apg_adaptive_momentum, apg_norm_thr, apg_eta,
                apg_off_type, apg_off_steps, apg_off_specific_steps)

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 8:
            (
                self.apg_enabled,
                self.apg_moment,
                self.apg_adaptive_moment,
                self.apg_norm_thr,
                self.apg_eta,
                self.apg_off_type,
                self.apg_off_steps,
                self.apg_off_specific_steps,
            ) = args[:8]
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
