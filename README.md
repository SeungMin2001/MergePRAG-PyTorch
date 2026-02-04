# MergePRAG (PyTorch)
> A from-scratch PyTorch implementation of the Transformer based on
the MergePRAG framework proposed in a paper by the UNIST NLP Lab.

> This repository aims to reproduce and explore the core ideas and architecture
presented in the original research.

<br>
<p align="left">
  <img src="assets/MergePRAG.jpg" alt="MergePRAG Architecture" width="750">
</p>
<p align="left">
</p>
<em>Figure : Overview of MergePRAG for multi-hop QA.</em>

## Paper Reference

This implementation is based on the following paper authored by the **UNIST NLP Lab**:

> **MergePRAG: Orthogonal Merging of Passage-experts for Multi-hop Parametric RAG**  
> *Submitted to the International Conference on Learning Representations (ICLR) 2026*

paper : https://openreview.net/forum?id=FSL1J2gmJV
<br>

<br>

# STEP 0. Understanding 

> 기존 parametric RAG 방식의 문제점 제시:

> RAG를 통해 passage를 가져온뒤 기존 모델의 파라미터에 적용시켜줘야함.(변화량 반영) 이 한번의 과정을 hop이라 불름.

> 이때 single hop은 가능하지만 multi hop에서 문제발생. 계속 새로운 passage에 대해서 파라미터 변화를 주면 덮어쓰거나 변형되면서 간섭이 쉽게 일어난다는게 본 연구의 주장.

> 따라서 이러한 간섭을 최소화 하고자 2가지 proposal를 함. (which is advanced by two key proposals)

> first proposal: 직교 취합(orthogonal merging) -> 직교 즉 u⊤v=0을 뜻함. 이때 대상은 "가중치 변화량(Δθ)" 임

> second proposal: critical-layer parameterization -> 이러한 직교 취합을 모든 레이어에 하는건 비효율적임. 따라서 중요한 레이어에만 직교 취합 진행. 따라서 "크리티컬한 레이어에만"


... ing
