from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import gc
import torch
import os
from lora_diffusion.cli_lora_pti import train as lora_train

from common import (
    extract_zip_and_flatten,
    get_output_filename,
)


COMMON_PARAMETERS = {
    "train_text_encoder": True,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "gradient_checkpointing": False,
    "lr_scheduler": "constant",
    "scale_lr": True,
    "lr_warmup_steps": 0,
    "clip_ti_decay": True,
    "color_jitter": True,
    "continue_inversion": False,
    "continue_inversion_lr": 1e-4,
    "initializer_tokens": None,
    "learning_rate_text": 1e-5,
    "learning_rate_ti": 5e-4,
    "learning_rate_unet": 2e-4,
    "lr_scheduler_lora": "constant",
    "lr_warmup_steps_lora": 0,
    "max_train_steps_ti": 700,
    "max_train_steps_tuning": 700,
    "placeholder_token_at_data": None,
    "placeholder_tokens": "<s1>|<s2>",
    "weight_decay_lora": 0.001,
    "weight_decay_ti": 0,
}


FACE_PARAMETERS = {
    "use_face_segmentation_condition": True,
    "use_template": "object",
    "placeholder_tokens": "<s1>|<s2>",
    "lora_rank": 16,
}

OBJECT_PARAMETERS = {
    "use_face_segmentation_condition": False,
    "use_template": "object",
    "placeholder_tokens": "<s1>|<s2>",
    "lora_rank": 8,
}

STYLE_PARAMETERS = {
    "use_face_segmentation_condition": False,
    "use_template": "style",
    "placeholder_tokens": "<s1>|<s2>",
    "lora_rank": 16,
}

TASK_PARAMETERS = {
    "face": FACE_PARAMETERS,
    "object": OBJECT_PARAMETERS,
    "style": STYLE_PARAMETERS,
}


# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--instance_data", help="Instance data")
parser.add_argument("-s", "--seed", help="Seed for reproducibility")
parser.add_argument("-r", "--resolution", help="Resolution", default=512)
parser.add_argument("-o", "--output_dir", help="Output dir", default='./output')
parser.add_argument("-p", "--pretrained", help="Pretrained model name or path", default='./stable-diffusion-v1-5-cache')


args = vars(parser.parse_args())


def train(self):
    seed = args["seed"]
    instance_data = args["instace_data"]
    resolution = args["resolution"]
    output_dir = args['output_dir']
    pretrained = args['pretrained']
    task = "face"

    if seed is None:
        seed = random_seed()
    print(f"Using seed: {seed}")

    instance_data = "instance_data"
    output_dir = "checkpoints"
    clean_directories([instance_data, output_dir])

    params = {k: v for k, v in TASK_PARAMETERS[task].items()}
    params.update(COMMON_PARAMETERS)
    params.update(
        {
            "pretrained_model_name_or_path": pretrained,
            "instance_data_dir": instance_data,
            "output_dir": output_dir,
            "resolution": resolution,
            "seed": seed,
        }
    )

    lora_train(**params)
    gc.collect()
    torch.cuda.empty_cache()

    num_steps = COMMON_PARAMETERS["max_train_steps_tuning"]
    weights_path = output_dir + os.path.sep + f"step_{num_steps}.safetensors"
   
    return weights_path


if __name__ == "main":
    train()
