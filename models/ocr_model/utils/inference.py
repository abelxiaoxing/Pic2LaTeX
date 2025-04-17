import numpy as np

from transformers import RobertaTokenizerFast, GenerationConfig
from typing import List, Union

from .transforms import inference_transform
from .helpers import convert2rgb
from ..model.Pic2LaTeX import Pic2LaTeX
from ...globals import MAX_TOKEN_SIZE


def inference(
    model: Pic2LaTeX, 
    tokenizer: RobertaTokenizerFast,
    imgs: Union[List[str], List[np.ndarray]], 
    num_beams: int = 1,
    max_tokens = None
) -> List[str]:
    if imgs == []:
        return []
    if isinstance(imgs[0], str):
        imgs = convert2rgb(imgs) 
    else:  # already numpy array(rgb format)
        assert isinstance(imgs[0], np.ndarray)
        imgs = imgs 
    imgs = inference_transform(imgs)
    pixel_values = np.stack(imgs)
    
    generate_config = GenerationConfig(
        max_new_tokens=MAX_TOKEN_SIZE if max_tokens is None else max_tokens,
        num_beams=num_beams,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    pred = model.generate(pixel_values, generation_config=generate_config)
    res = tokenizer.batch_decode(pred, skip_special_tokens=True)
    return res
