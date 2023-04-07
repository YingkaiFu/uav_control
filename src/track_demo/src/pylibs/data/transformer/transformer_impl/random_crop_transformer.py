from typing import Dict

from data.utils.crop_track_pair import crop_track_pair

from ..transformer_base import TRACK_TRANSFORMERS, TransformerBase


@TRACK_TRANSFORMERS.register
class RandomCropTransformer(TransformerBase):
    r"""
    Cropping training pair with data augmentation (random shift / random scaling)

    Hyper-parameters
    ----------------

    context_amount: float
        the context factor for template image
    max_scale: float
        the max scale change ratio for search image
    max_shift:  float
        the max shift change ratio for search image
    max_scale_temp: float
        the max scale change ratio for template image
    max_shift_temp:  float
        the max shift change ratio for template image
    z_size: int
        output size of template image
    x_size: int
        output size of search image
    """
    default_hyper_params = dict(
        context_amount=0.5,
        max_scale=0.3,
        max_shift=0.4,
        max_scale_temp=0.0,
        max_shift_temp=0.0,
        z_size=127,
        x_size=303,
    )

    def __init__(self, seed: int = 0) -> None:
        super(RandomCropTransformer, self).__init__(seed=seed)

    def __call__(self, sampled_data: Dict) -> Dict:
        r"""
        sampled_data: Dict()
            input data
            Dict(data1=Dict(image, anno), data2=Dict(image, anno))
        """
        data1 = sampled_data["data1"]
        data2 = sampled_data["data2"]
        em_temp, bbox_temp = data1['event'], data1["anno"]
        em_curr, bbox_curr = data2['event'], data2["anno"],
        em_z,bbox_z, em_x, bbox_x, _, _ = crop_track_pair(
            em_temp,
            bbox_temp,
            em_curr,
            bbox_curr,
            config=self._hyper_params,
            rng=self._state["rng"],
            DEBUG=True)

        sampled_data["data1"] = dict(event=em_z, anno=bbox_z)
        sampled_data["data2"] = dict(event=em_x, anno=bbox_x)

        return sampled_data
