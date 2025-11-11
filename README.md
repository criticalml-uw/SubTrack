# SubTrack++

**SubTrack++** is a memory- and time-efficient training framework for large language models (LLMs), designed to make high-performance LLM training more accessible. SubTrack++ leverages **Grassmannian gradient subspace tracking**, **projection-aware optimization**, and **gradient recovery scaling** to deliver superior convergence, reduced wall-time, and minimal memory overhead—without compromising accuracy.

### 🚀 What Makes SubTrack++ Different?

* **Grassmannian Subspace Tracking:** Tracks low-rank gradient subspaces using geometry-aware updates, avoiding costly SVD computations and providing robust adaptation throughout training.
* **Projection-Aware Optimizer:** Extends the Adam optimizer to reflect changes in gradient subspaces, maintaining accurate momentum updates even as the subspace evolves.
* **Recovery Scaling:** Recovers and scales discarded gradient components to boost training performance and generalization.
* **Full-Parameter Training with Low Memory:** Achieves state-of-the-art evaluation loss while maintaining the memory efficiency of GaLore and other low-rank methods.
* **Faster Convergence:** Reduces pre-training wall-time by up to **43%** compared to previous best methods on LLaMA models up to 7B parameters.


---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 🧪 Running SubTrack++

Example pre-training command (LLaMA 1B on C4 dataset):

```bash
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_1b.json \
    --single_gpu \
    --lr 0.0001 \
    --low_rank_scale 0.25 \
    --rank 512 \
    --subspace_update_interval 200 \
    --batch_size 8 \
    --total_batch_size 16 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 10000 \
    --optimizer low_rank_adamw  \
    --st_init_step_size 10000 \
    --subspace_update_method subtrack \
    --adaptive_optimizer \
    --recovery_scaling
```

You can find a list of  example scripts in script folder.
Ensure you configure dataset paths and checkpoint locations as needed.

The code is built on top of the GaLore repository, available [here](https://github.com/jiaweizzhao/GaLore).

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@inproceedings{
rajabi2025subtrack,
title={SubTrack++ : Gradient Subspace Tracking for Scalable {LLM} Training},
author={Sahar Rajabi and Nayeema Nonta and Sirisha Rambhatla},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=6geRIdlFWJ}
}
```

