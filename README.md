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

# STEP 1. Preparation dataset and Find "critical layer"
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

> Inject 하려면 일단 Hypernetwork 만들어야하고, 그 전에 retrieverd data도 준비해야하고, Reasoning chain도 해놔야함. 즉 전처리된 dataset과 HyperNetwork 가 있어야함

> 일단 dataset 부터 준비 ㄱㄱ

```py
from datasets import load_dataset
data=load_dataset("hotpotqa/hotpot_qa", "fullwiki")
train=data["train"]
vaild=data["validation"]
```
> 이제 SPt를 만들어서 HyperNetwork에 넘겨줘야한다. 즉 SPt를 만들어줘야한다.

> 데이터셋의 supporting_facts, context를 활용하여 만들어준다.

> 만들어진 passage를 retrieved passage로 간주한다.

> 이때 "hotpotqa/hotpot_qa" 데이터셋 구조는 이와같다. <br>
> answer,fact,context,sentence. 여기서 SPt를 만들어줘야한다. 따락서 fact에서 answer의 근거가 되는 문장위치 찾고 context에서 문서이름 찾고, sentence에서 해당문서의 문장을 찾아서 가져온게 SPt중 하나가 되는것.
```py
# SPt 만들어주기.
def make_SPT(data):
  context_title=data["context"]["title"]
  context_sentence=data["context"]["sentence"]

  sub_title=data["supporting_facts"]['title']
  sub_ids=data["supporting_facts"]["sent_id"]

  facts=[]

  for t,sid in zip(sub_title,sub_ids): #fact에서 제시한 문서이름,몇번째 문장을 페어로 순회
    if t in context_title: # 만약 문서가 context에 있으면 트루
      idx=context_title.index(t) # 인덱스 구하고
      sentence=context_sentence[idx] # 해당 인덱스에 sentence 전체문장 다 가져옴

      if 0<=sid<len(sentence): # 만약 sid가 범위 만족하면 
        passage=sentence[sid].strip() # 전체 sentence중 sid에 해당하는 문장 가져옴
        facts.append(passage) # passage 넣어줌. 이게 쌓이면 SPt가 됨
  return facts
```
<br>

```py
new_train=[]

for k in train: # make_SPT 적용
  passage=make_SPT(k)

  new_train.append({ # 질문,답,fact 구조
      "question":train['question'],
      "answer":train['answer'],
      "facts":passage
  })

new_valid=[]

for k in valid:
  passage=make_SPT(k)

  new_valid.append({
      "question":valid['question'],
      "answer":valid['answer'],
      "facts":passage
  })

#train,valid 둘다 적용

```
> 이제 Reasoning Chain을 해줘야함. <br>
> 여기서 Decomposition 개념이 들어감. 즉 분리 <br>
> 논문 코드에서는 Decomposer 모델을 학습을 시켜서 sub question, sub answer를 만들고있음. <br>

> Decomposer 모델의 역할과 ai api의 역할이 비슷하다고 봄. 논문 특성상 외부 api를 아키텍처에 포함시키지 못하기 때문에 별도의 로컬 모델링을 했다고 생각함. 구현편의상 나는 api를 바로 사용할 계획

> 논문 코드에서는 chain, 즉 시계열적 순서도 llm을 통한 학습으로 진행된다. 따라서 하위질문,하위답변, 시계열적 논리순서 모두 llm을 사용하여 새로운 데이터셋을 만들어야한다.

> llm을 활용하여 논리순서, 하위질문, 하위답변을 포함한 재설계된 데이터셋 구축 진행.

##### find critical layer 
> 현재 critical layer를 찾으려면 inject 해보면서 변화들을 관찰할 필요가 있는데

> 그럴려면 hypernetwork를 먼저 만들어줘야 가능함.

##### HyperNetwork 

<br>
<p align="left">
  <img src="assets/HyperNetwork.jpg" alt="HyperNetwork Architecture" width="750">
</p>
<p align="left">
</p>
<em>Figure : Overview of HyperNetwork Architecture.</em> <br>



```py
```

... ing
