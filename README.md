# MergePRAG (PyTorch)
> A from scratch PyTorch implementation of the Transformer based on
the MergePRAG framework proposed in a paper by the UNIST NLP Lab.

> This repository aims to reproduce and explore the core ideas and architecture
presented in the original research.

<br>
<p align="left">
  <img src="assets/MergePRAG.jpg" alt="MergePRAG Architecture" width="750">
</p>
<p align="left">
</p>
<em>Figure : Overview of MergePRAG for multi hop QA.</em>

## Paper Reference

This implementation is based on the following paper authored by the **UNIST NLP Lab**:

> **MergePRAG: Orthogonal Merging of Passage experts for Multi-hop Parametric RAG**  
> *Submitted to the International Conference on Learning Representations (ICLR) 2026*

paper : https://openreview.net/forum?id=FSL1J2gmJV
<br>

<br>

# STEP 1. Find "critical layer"
> Unlike the original paper, the base LLM used here is a basic Transformer implemented from scratch, following the architecture proposed in “Attention Is All You Need.”

```py
# 모델의 모든 파라미터 고정하기. (크리티컬 레이어 찾기위해서.)
def freeze_model(model:nn.Module):
  for p in model.parameters():
    p.requires_grad=False
  return model

# MoE, hypernetwork 등 추가할때 requires_grad=true 해주셈

```
```py
# required_grad=true인 애들만 역전파 해줘야함. 따라서 "optimize" 함수정의
# 즉 MoE,H() 를 진행할때 required_grad=true 해주고 켜준애들만 역전파 
def optimize(model:nn.Module):
  trainable_params=[p for p in model.paramerters() if p.requires_grad]
  optimizer=torch.optim.AdamW(trainable_params,lr=1e-4,weight_decay=0.01)
```
> 이제 Injection 해야함. 즉 내가 구현했던 Transformer에 FeedFoward에 injection 해야함.

> 이때 차원을 맞춰줘야하는데 난 논문그대로 구현하였으므로 512->1024->512. 즉 Injection Vector is (R^512)


... ing
