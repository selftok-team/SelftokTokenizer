from mimogpt.models.selftok.vector_quantize_pytorch import VectorQuantize as VectorQuantize_EMA


def construct_quantizer(
        latent_dim, code_dim, output_dim, codebook_size, K,
        w_diversity, w_commit, dead_code_threshold=0.0, decay=0.99,
        smart_re_K=0, continuous=False, reg=[1/4., 1/2.],
        reset_cluster_size=None, ema_entropy_ratio=0.7, frozen_embed=None,):
    
    args = dict(
        dim=latent_dim,
        output_dim=output_dim,
        codebook_dim=code_dim,
        codebook_size=codebook_size,
        ema_update=True,
        decay=decay,
        kmeans_init=True,
        kmeans_iters=10,
        threshold_ema_dead_code=dead_code_threshold,
        use_cosine_sim=True,
        commitment_weight=w_commit,
        diversity_weight=w_diversity,
        smart_re_K=smart_re_K,
        continuous=continuous,
        reg=reg,
        reset_cluster_size=reset_cluster_size,
        ema_entropy_ratio=ema_entropy_ratio,
        frozen_embed = frozen_embed,
    )

    constructor = VectorQuantize_EMA
    
    return constructor(**args)

