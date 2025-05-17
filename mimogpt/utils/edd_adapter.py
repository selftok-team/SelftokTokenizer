from mimogpt.demo.gpt_t2i import Text2ImageInfer
from mimogpt.eval.functional import ClipVit

params = {
    "model": {
        "type": "prior",
        "diffusion_sampler": "uniform",
        "hparams": {
            "text_ctx": 77,
            "xf_width": 2048,
            "xf_layers": 20,
            "xf_heads": 32,
            "xf_final_ln": True,
            "xf_padding": False,
            "text_drop": 0.2,
            "clip_dim": 768,
            "clip_xf_width": 768,
        },
    },
    "diffusion": {
        "steps": 1000,
        "learn_sigma": False,
        "sigma_small": True,
        "noise_schedule": "cosine",
        "use_kl": False,
        "predict_xstart": True,
        "rescale_learned_sigmas": False,
        "timestep_respacing": "",
    },
}


def create_edd_d1_topkcal(
    d1_ckpt_path, d1_outdir, d1_device, d1_token64_yml, d1_token256_yml, topkcal_model, device_topkcal
):
    t2i_pipeline256 = Text2ImageInfer(
        d1_ckpt_path,
        d1_outdir,
        device=d1_device,
        token64_yml=d1_token64_yml,
        token256_yml=d1_token256_yml,
        token64=False,
    )
    clip_calculate = ClipVit(topkcal_model, device=device_topkcal)
    return t2i_pipeline256, clip_calculate
    # return None,None


def get_edd_d1_output(text, edd_d1, topkcal):
    original_image, codebooks, path = edd_d1(text, use_kv_cache=True, bs_in=20, original=True)
    all_images, all_scores, images, scores, indexed = topkcal.rank_by_similarity(
        original_image, text, count=1, index_need=True, all_need=True
    )  # top1
    return images[0]
    # return None
