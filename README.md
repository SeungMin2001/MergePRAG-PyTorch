# MergePRAG (PyTorch)

[🇰🇷 한국어](README.md) | [🇺🇸 English](README.en.md)

> UNIST NLP Lab의 MergePRAG 논문을 바탕으로 핵심 개념과 구조를 재현·탐구하는 PyTorch 학습 구현입니다.

<p align="left">
  <img src="assets/MergePRAG.jpg" alt="MergePRAG 아키텍처" width="750">
</p>
<em>그림. 멀티홉 QA를 위한 MergePRAG 개요</em>

## 개요

이 저장소는 MergePRAG의 핵심 아이디어를 학습 중심으로 구현합니다. README의 학습 노트와 src/의 실행 가능한 모듈을 함께 제공합니다.

- HotpotQA supporting fact를 활용한 SPt 구성
- [B, T, d_model]에서 [B, d_model]로의 패시지 인코딩·어텐티브 풀링
- K/V 투영을 포함한 HyperNetwork 메모리 생성
- 패시지·홉 메모리 뱅크의 직교 병합
- forward hook을 통한 동결 베이스 모델 메모리 주입
- 학습 전 실험 단계인 critical-layer 탐색

## 논문

> **MergePRAG: Orthogonal Merging of Passage experts for Multi-hop Parametric RAG**  
> *ICLR 2026 제출*

[OpenReview 논문](https://openreview.net/forum?id=FSL1J2gmJV)

## 실행 가능한 구현

- src/mergeprag_pytorch/data.py: HotpotQA의 supporting_facts와 context를 매칭해 SPt를 생성
- src/mergeprag_pytorch/hypernetwork.py: 어텐티브 풀링, MLP_hyp, K/V 메모리 투영
- src/mergeprag_pytorch/merging.py: 패시지·홉 메모리 뱅크의 직교 병합
- src/mergeprag_pytorch/injection.py: 메모리 cross-attention, 학습 가능한 injector, 베이스 모델 동결, forward-hook 도우미
- examples/toy_mergeprag_flow.py: SPt 추출부터 메모리 주입까지의 작은 전체 흐름
- tests/test_mergeprag.py: 핵심 메커니즘의 형태·동작 검사

~~~bash
pip install -e ".[dev]"
PYTHONPATH=src python examples/toy_mergeprag_flow.py
PYTHONPATH=src pytest -q
~~~

현재 구현은 전체 논문 학습 파이프라인을 완전히 재현하기보다, 다음 핵심 흐름을 실행 가능한 구성 요소로 이해하기 위한 것이다.

~~~text
SPt 구성 → HyperNetwork 메모리 생성 → 직교 병합 → 동결 베이스 모델 주입
~~~

## 학습 노트

### 1. 데이터 준비

HotpotQA의 supporting_facts에서 답의 근거가 되는 문서 제목과 문장 위치를 찾고, context에서 해당 문장을 꺼내 SPt로 구성한다. 이 패시지를 검색된 패시지로 간주해 HyperNetwork에 전달한다.

### 2. HyperNetwork

패시지 토큰 표현을 어텐티브 풀링으로 요약한 후, HyperNetwork가 각 패시지·홉에 대한 K/V 메모리를 만든다. Critical layer는 베이스 모델에서 메모리 주입 효과가 큰 지점을 찾기 위한 실험 단계다.

### 3. 직교 지속 병합

여러 패시지 또는 홉의 메모리 뱅크가 서로 간섭하지 않도록 이미 합쳐진 메모리의 방향을 제거하는 직교 병합을 사용한다. 멀티홉 질문에서 개별 근거의 정보를 유지하는 것이 목표다.

### 4. 메모리 주입

생성·병합된 메모리는 동결된 베이스 모델의 선택한 레이어에 cross-attention과 forward hook으로 주입한다. 학습 대상은 메모리 관련 모듈에 한정할 수 있다.

## 다음 단계

- 전체 학습·평가 파이프라인 확장
- 멀티홉 QA 벤치마크 기반 재현 실험
- 다양한 베이스 모델과 critical layer 비교
- 체크포인트·재현 설정 공개

원문 기반의 상세 코드 학습 노트는 [English README](README.en.md)에서 함께 볼 수 있다.

