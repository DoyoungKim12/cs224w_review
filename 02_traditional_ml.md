# Traditional Methods for Machine Learning in Graphs

- ML Tasks : Review
  - Node-level prediction
  - Link-level prediction
  - Graph-level prediction

<br>

- Traditional ML Pipeline
  - node, link, graph에 대한 feature를 design
  - 모든 학습 데이터에 대한 feature를 생성
    - 각 node, link, graph의 정보를 담은 vector를 생성  
  - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_1.PNG?raw=true"> <br>
    - RF, SVM과 같은 ML모델을 학습, 새로운 node/link/graph가 주어졌을 때 그로부터 feature를 생성하여 예측결과를 만들어냄
  
<br>  
  
- This Lecture : Feature Design
  - 그래프에 효과적인 feature를 사용하는 것이 좋은 테스트 결과를 얻는 열쇠이다.
  - Traditional ML pipeline에서는 hand-designed feature를 사용한다.
  - 이번 강의에서는 traditional feature를 전반적으로 다룰 것이다.
  - 직관적인 이해를 위해, undirected graph에 집중한다. 

<br> 

- ML in Graphs
  - 목표 : object의 집합에 대한 예측결과 만들기
  - 설계(desing) 선택
    - Features : d차원의 벡터
    - Objects : 노드, 엣지, 노드의 집합, 전체 그래프
    - 목적함수 : 우리가 풀고자 하는 문제가 무엇인가? <br><br>
  - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_2.PNG?raw=true"> <br>
    - V는 vertices로, node를 vertex로 표현
    - 즉, Graph는 노드의 집합과 링크의 집합으로 정의된다는 것
    - function은 노드(또는 링크, 그래프 구조)를 벡터로 표현해주는 함수를 말함

<br> 

- Node-level Tasks
  - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_3.PNG?raw=true">
  - 회색 노드가 ML을 거쳐 초록색 또는 빨강색 노드로 분류됨
  - 자세히 보면, 초록 노드는 2개 이상의 노드와 연결되어있고, 빨강 노드는 단 1개의 노드와 연결되어 있음
  - 해당 정보를 feature로 모델에 전달하면, 모델은 해당 정보를 바탕으로 노드를 분류할 수 있게 되는 것임
  - 따라서, **ML needs feautures.**

<br> 

- Node-level Features: Overview
  - 목표 : **네트워크에 속한 노드의 구조와 위치를 특성화(Characterize)***하는 것 
    - Node degree
    - Node centrality
    - Clustering coefficient
    - Graphlets
    - (각각 무엇인지는 뒤에서 살펴볼 것)

<br>

- Node-level Features: Node degree
  - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_4.PNG?raw=true"><br>
    - 노드 v의 degree k(<img src="https://render.githubusercontent.com/render/math?math=V_k">)는 그 노드가 가진 엣지의 수로 정의됨
      - 즉, 다른 노드와 많이 연결될수록 중요한 노드로 취급한다는 것
    - 모든 이웃한 노드를 동등하게 취급, 따라서 A, F, G, H의 차이를 모델이 알 수 없다는 단점이 있음

<br>

- Node-level Features: Node Centrality
  - Node degree는 이웃한 노드의 중요도(importance)를 포착하지 않고 그 수를 세는 것
  - Node Centrality(<img src="https://render.githubusercontent.com/render/math?math=C_v">)는 그래프 안에서의 노드 중요도를 반영함
  - 중요도를 모델링하는 방법
    - Engienvector centrality
    - Betweenness centrality
    - Closeness centrality
    - and many others…

<br>

- Node Centrality (1)
  - Engienvector centrality
    - (Engienvector centrality 설명을 대충하고 지나가서 이것저것 찾아보면서 다시 이해해야 했음... 교수님...)
    - 노드 v는 중요도가 높은 이웃 노드와 인접하다면 높은 중요도를 가짐
    - 따라서, 우리는 노드 v의 centrality를 이웃한 노드의 centrality의 합으로 모델링할 수 있음
      - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_5.PNG?raw=true"><br>
        - 노드는 centrality vector c를 (지금은 그 값을 모르지만 아무튼) 가진다고 생각하자.(이것을 예측하는 것이 목표)
        - N(v)는 v의 이웃 노드 집합이다.
        - λ는 정규화 인자이다. <br><br>
    - 이는 재귀적(recursive)인 방식으로 정의되는데, 이를 어떻게 풀 수 있을까?<br><br>
      - 일단 우리는 c를 모르니까, 모든 값을 1로 초기화하자. 노드가 4인 그래프라면 \[1,1,1,1]이 될 것이다.
        - 즉, 각 노드의 중요도가 모두 1로 같다고 보고 시작하자. <br><br>
      - 그 다음으로, 이 값을 어떻게 업데이트할지 생각해보자. 
        - 우리가 특정 그래프 안을 무작위로 돌아다닌다고 할 때(Random Walk), 특정 노드를 많이 방문한다면 그 노드는 중요도가 높은 노드라고 말할 수 있을 것이다.
        - 그렇다면 우리는 저 초기값을 **'특정 노드에서 다른 노드로 이동하는 정보'**, 즉 돌아다닐 수 있는 길의 정보를 사용해 업데이트해야 할 것이다.
        - '특정 노드에서 다른 노드로 이동하는 정보'를 우리는 인접행렬(adjacency matrix)로 표현할 수 있다.
        - 그럼 업데이트된 c는 이전 c에 인접행렬 A를 곱해준 정보이다! (실제로 초기값에 A를 곱한 값은 node degree이다. 궁금하면 직접 해보면 바로 나온다.)
        - 업데이트된 c의 값들은 이전보다 계속 커질 것이므로, 아무튼 λ 같은 값으로 나눠준다. <br><br>
      - 이런 식의 업데이트를 무한히 반복한다고 하자.
        - 그러다보면, 어느 순간 업데이트된 c와 이전 c의 차이가 없어지는 상태에 도달할 것이다. (stationary distribution과 비슷한 느낌?)
        - 그럼 더이상 업데이트할 필요가 없으니 해당 벡터 c를 가져다 쓰면 되겠다.
        - 이러한 상태를 식으로 표현하면 아래와 같다.
          - <img src="https://render.githubusercontent.com/render/math?math=c_{t %2B 1} = \frac{1}{\lambda}Ac_{t}">
          - t+1기나 t기나 같으니까 해당 표현을 빼주고, 람다를 좌변에 곱해주면, **비로소 재귀적 형태를 행렬 형태로 바꿀 수 있게 된다**. <br><br>
          - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_6.PNG?raw=true"><br>
          - 그런데, 공교롭게도 그 형태가 고유벡터의 정의와 일치한다. 그래서 Engienvector centrality라고 부르는 것이다.
          - 이는 곧 Power Iteration Algorithm으로 dominant한 고유값과 그에 대응하는 고유벡터를 구하는 것과 같다.
          - 따라서, dominant한 고유값에 대응하는 고유벡터의 값을 각 노드에 대응하는 중요도로 사용할 수 있다.  

<br>

- Node Centrality (2)
  - Betweenness centrality
    - 특정 노드가 다른 노드 사이의 최단 경로에 많이 포함될수록 중요도가 높다고 보는 것.
    - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_7.PNG?raw=true"><br>

<br>

- Node Centrality (3)
  - Closeness centrality
    - 특정 노드와 다른 각각의 노드와의 최단경로를 계산, 그 합이 작을수록 중요도가 높다고 보는 것.
    - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_8.PNG?raw=true"><br>

<br>

- Node Features : Clustering Coefficient
  - 노드 v의 이웃 노드들이 얼마나 연결되어있는지 측정하는 것
  - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_9.PNG?raw=true"><br>
    - 파랑 노드의 이웃 노드는 총 4개(빨강 노드)로, 이웃 간 가능한 링크의 수는 총 6개
    - 가능한 경우의 수인 6개 중 몇 개가 이어졋느냐에 따라 Clustering Coefficient를 측정
    - 1개의 파랑 노드와 2개의 빨강 노드, 총 3개의 노드가 만드는 삼각형의 갯수를 세는 것으로도 이해할 수 있음

<br>

- Node Features: Graphlets
  - Graphlets : pre-specified subgraphs (사전에 정의된 그래프의 형태)
  - Rooted connected non-isomorphic subgraphs
    - 노드 수에 따라 가능한 그래프의 형태는 정해져있고, 그 형태를 각각의 서브그래프로 정의한 것
    - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_10.PNG?raw=true"><br><br>
  - GDV(Graphlet Degree Vector) : 특정 노드를 표현하는 Graphlet 기반의 feature
    - GDV는 특정 노드가 포함된 Graphlet의 개수를 세는 것 (A count vector of graphlets rooted at a given node.)
    - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_11.PNG?raw=true"><br><br>
  - 2~5개 노드의 graphlet을 고려함으로써 아래와 같은 효과를 얻음
    - 73차원(coordinates)의 벡터는 (해당 노드의) 이웃 노드의 위상 배치(topology)를 묘사하게 된다.
    - 4-hops distance까지의 상호연결성을 포착하게 된다. <br><br>
  - GDV는 노드의 local network topology의 측정치를 제공한다.
    - node degree, clustering coefficient보다 local topology 유사도에 대한 자세한 측정치를 제공한다. 

<br>

 - Node-Level Features: Summery
  -  <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_12.PNG?raw=true"><br>
  -  <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_13.PNG?raw=true"><br>
  -  <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_14.PNG?raw=true"><br>

<br><br><br><br>

- Link-Level Prediction Task: Recap, Link Prediction as a Task
  - 존재하는 링크를 바탕으로 (이후 발생할) **새로운 링크**를 예측하기
  - 링크 예측 문제의 2가지 공식(formulation)
    - 1) Links missing at random
      - 일부러 몇 개의 링크를 무작위로 제거한 후, 이를 예측하도록 함
      - 단백질 구조와 같은 정적 네트워크 문제에 활용  
    - 2) Links over time
      - t기까지의 그래프가 주어졌을 때, t+1기에 새로 생성될 링크를 예측하도록 함
      - SNS와 같은 동적 네트워크 문제에 활용 

<br>

- Link Prediction via Proximity(근접성)
  - Methodology
    - 각 노드 페어 (x,y)에 대해 스코어 c(x,y)를 계산
      - 예를 들어, c(x,y)는 x와 y의 공통 이웃의 수 같은 것이 될 수 있음
    - 각 페어(x,y)를 스코어 내림차순으로 정렬
    - 상위 n개 페어를 새로운 링크로 예측
    - 실제로 t+1기에 예측된 링크가 실제로 나타나는지 확인

 <br>
 
- Link-Level Features: Overview
  - Distance-based feature
  - Local neighborhood overlap
  - Global neighborhood overlap

 <br>
 
- Distance-Based Features
  - 2개 노드 사이의 최단거리
  - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_15.PNG?raw=true"><br>
  - 그러나, 이는 겹치는 이웃의 수를 포착하지는 못한다.
    - 노드 페어 (B,H)는 C,D의 2개 노드를 이웃으로 공유하지만, (B,E)와 (A,B)는 1개의 노드만을 이웃으로 공유함 

 <br>
 
- Local Neighborhood Overlap
  - 2개 노드가 공유하는 이웃 노드의 수를 포착함
  - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_16.PNG?raw=true"><br> 
    - Jaccard’s coefficient(자카드 계수) : 두 집합의 교집합을 합집합으로 나눈 값
    - Adamic-Adar index : x와 y의 이웃 노드 z에 대해, z의 이웃 노드가 적을수록 더 높은 가중치를 주겠다는 것
      - 오직 x와 y만을 이웃으로 갖는 z라면 이는 특별한 관계를 의미함
      - x와 y 외에도 다른 많은 노드를 이웃으로 갖는 z라면 z를 공통 이웃으로 두는 것이 큰 의미를 갖지 않음 <br><br>
  - Local Neighborhood feature의 한계
    - 2개 노드가 공통 이웃을 1개도 가지지 않는 경우의 metric은 항상 0임
    - 그러나, 그 2개 노드는 미래에 연결될 가능성을 여전히 가지고 있음
    - Global neighborhood overlap은 이러한 한계를 전체 그래프를 고려함으로써 해결함

<br>

- Global Neighborhood Overlap
  - **Katz index** : 주어진 노드 페어들 사이의 가능한 모든 길이의 경로 수를 세는 것
  - 2개 노드 사이의 모든 경로 수를 어떻게 계산할 수 있는가?
    - 그래프 인접행렬의 거듭제곱(powers of the graph adjacency matrix)을 사용!
    - 사실 위에서 보았던 Engienvector centrality와 유사함

<br>

- Intuition: Powers of Adj Matrices
  - 2개 노드 사이의 모든 경로의 수를 계산하는 방법
    - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_17.PNG?raw=true"><br>
    - 여기서의 인접행렬은 두 노드를 연결하는 길이가 1인 경로의 수를 계산한 것 (경로가 있거나 없거나 둘 중 하나이므로 0 또는 1로 표현됨)<br><br>
  - 길이가 2인 경로의 수를 계산하는 방법은?
    - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_18.PNG?raw=true"><br>
    - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_19.PNG?raw=true"><br>
    - 행렬을 l번 거듭제곱하여, 길이가 l인 경로의 수를 계산할 수 있음

<br>

- Global Neighborhood Overlap
  - v1과 v2 사이의 Katz index는 아래와 같이 계산됨
  - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_20.PNG?raw=true"><br>
    - <img src="https://render.githubusercontent.com/render/math?math=\beta">는 경로의 길이가 길수록 작은 값(가중치)이 곱해지도록 설계
    - 아래와 같은 closed-form으로 계산됨
    - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_21.PNG?raw=true"><br>

<br>

- Link-Level Features: Summary
  - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_22.PNG?raw=true"><br>

<br><br><br><br>

- Graph-Level Features
  - 목표 : 전체 그래프의 구조를 특성화(Characterize)하는 것

<br>

- Background: Kernel Methods
  - Kernel Methods는 그래프 레벨의 예측 문제를 풀기 위한 전통적인 ML 방법론으로 폭넓게 사용되었음
  - 아이디어 : feature vector 대신 kernel을 설계하기
  - 커널에 대한 빠른 설명
    - Kernel K(G,G′)∈R은 데이터(그래프) 간의 유사도를 계산한다.
    - Kernel Matrix <img src="https://render.githubusercontent.com/render/math?math=K=(K(G,G\prime))_{G,G\prime}">은 항상 positive semidefinite한 행렬이다.
      - \*positive semidefinite은 위키피디아에 나와있듯이, 항상 양의 실수를 고유값으로 가진다.
    - 임의의 함수 ϕ는 feature vector를 생성하는 함수이며, 다음과 같이 정의된다.
      - K(G,G′)=ϕ(G)Tϕ(G′)
    - 커널이 일단 정의되면, kernel SVM과 같은 off-the-shelf ML 기법이 사용 가능하다. 

<br>

- Graph-Level Features: Overview
  - Graph Kernels: **두 그래프간의 유사도를 측정**
    - Graphlet Kernel
    - Weisfeiler-Lehman Kernel
    - Other kernels are also proposed in the literature
      - 해당 수업의 범위를 벗어남 (Random-walk kernel, Shortest-path graph kernel 등)

<br>

- Graph Kernel: Key Idea
  - 목표 : 그래프 피쳐 벡터인 ϕ(G)를 설계하는 것
  - 핵심 아이디어 : Bag-of-Words(BoW)를 그래프에 적용
    - BoW는 단순히 문서에 포함된 단어별 개수만을 피쳐로 활용, 단어의 순서는 고려하지 않음
    - 그래프로의 Naïve한 확장 : 노드를 마치 단어처럼 취급하기
    - 아래처럼 4개의 빨강 노드가 있는 그래프가 있을 때, 우리는 2개의 다른 그래프로부터 같은 피쳐 벡터를 얻게됨
      - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_23.PNG?raw=true"><br><br>
  - 그렇다면 Bag of nodes 대신 Bag of node degrees를 사용해보자
    - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_24.PNG?raw=true"><br>
    - node degree가 각각 1,2,3인 node의 수를 세어 두 그래프의 다름을 표현함
    - Graphlet Kernel과 Weisfeiler-Lehman 모두 그래프를 Bag of \* 표현한 것이고, \*는 node degrees 보다는 보다 정교한 무언가이다

<br>

- Graphlet Features
  - 핵심 아이디어 : 그래프 내의 그래플릿 수를 센다.
    - 참고 : 여기서의 그래플릿 정의는 노드 레벨에서의 정의와 약간 다름
    - 2개의 차이점이 존재
      - 그래플릿의 노드가 꼭 연결되어있지 않아도 됨 (고립된 노드도 허용)
      - 기준점이 되는 노드를 따로 두지 않음 (not rooted)
      - 예시는 아래와 같음
        - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_25.PNG?raw=true"><br><br>
  - Graphlet count vector를 아래와 같이 정의할 수 있음
    - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_26.PNG?raw=true"><br><br> 
    - 예시로, k=3인 그래플릿의 카운트 벡터는 아래처럼 구할 수 있음
      - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_27.PNG?raw=true"><br><br>      

<br>

- Graphlet Kernel
  - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_28.PNG?raw=true"><br><br>
  - 한계 : 그래플릿의 수를 세는 것은 계산비용이 크다!
    - 만약 크기가 k인 모든 graphlet을 크기가 n인 그래프에서 찾아서 세고자 한다면, 이는 <img src="https://render.githubusercontent.com/render/math?math=n^k">만큼의 계산이 필요하다.
    - 최악의 경우에는 이를 피할 수 없게 되는데, 왜냐하면 subgraph isomorphism test(특정 그래프가 다른 그래프의 서브그래프인지 판단하기)는 NP-hard 문제이기 때문이다.
      - (NP-hard는 다항시간내에 풀 수 없는 문제를 말한다. 다항 방정식 대신에 지수 방정식으로 풀 수 있는 문제를 말한다. 즉 결정적 다항식으로 구성할 수 없다는 것이다. NP-hard라고 해서 다항식이 아니라는 것은 아니다. 다만 다항식으로 표현될 수 있을지의 여부가 아직 알려지지 않았다라는 것이다. 즉, NP-Hard란 TSP문제와 같이 모든 경우의 수를 일일히 확인해보는 방법이외에는 다항식처럼 답을 풀이할 수 없는 문제들을 말한다. NP-hard의 예로 외판원 문제가 있다.)<br>
    - 만약 그래프의 최대 node degree가 d로 제한되어 있다면, <img src="https://render.githubusercontent.com/render/math?math=O(nd^{k-1})">의 복잡도로 크기가 k인 모든 그래플릿의 수를 세는 알고리즘이 존재한다.
    - 보다 효율적인 그래프 커널을 설계할 수는 없을까?

<br>

- Weisfeiler-Lehman Kernel
  - 목표 : 효율적으로 그래프 피쳐를 묘사하는 ϕ(G)를 설계하는 것
  - 아이디어 : node vocabulary을 풍부하게 만들기 위해 반복적으로 이웃 구조를 사용하기
    - Bag of node degrees의 일반화된 버전으로, node degrees는 1-hop의 정보만을 제공함
  - 이를 구현하기 위한 알고리즘 : **Color refinement**

<br>

- Color Refinement
  - 그래프 G가 노드 집합 V로 이루어져 있을 때, 
    - 초기 색 <img src="https://render.githubusercontent.com/render/math?math=c^{(0)}(v)">를 각 노드 v에 배정한다.
    - 다음 식을 이용해 반복적으로 색을 변화시킨다.
      - <img src="https://render.githubusercontent.com/render/math?math=c^{(k %2B 1)}(v) = HASH(c^{(k)}(v), \{c^{(k)}(u)\}_{u \in N(v)})">
      - 즉, 노드 v의 다음 색은 현재 노드 v의 색과 이웃 노드 u의 색을 인풋으로 받는 해쉬 함수를 이용해서 만들어지는 것이다.
    - 이를 일정 수준 이상으로 반복하게 되면 <img src="https://render.githubusercontent.com/render/math?math=c^{(k)}(v)">는 K-hop만큼의 이웃 노드에 대한 정보를 요약하게 된다. 

<br>
 
- Color Refinement (1) ~ (4) <br>
<img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_29.PNG?raw=true"><br><br> 
<img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_30.PNG?raw=true"><br><br> 
<img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_31.PNG?raw=true"><br><br>
<img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_32.PNG?raw=true"><br><br>

<br>

- Weisfeiler-Lehman Graph Features <br>
<img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_33.PNG?raw=true"><br><br>

<br>

- Weisfeiler-Lehman Kernel <br>
<img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_34.PNG?raw=true"><br><br> 
  - WL kernel은 **효율적인 연산이 가능함**
    - Color refinement를 수행할 때 각 스텝에서의 시간복잡도는 엣지의 수에 선형적인데, 그 이유는 그 과정이 이웃한 노드의 색을 집계하는 것을 포함하기 때문이다.
  - kernel의 값을 계산할 때, 두 그래프에서 등장하는 색만 추적하면 됨
    - 따라서 가능한 색깔의 수는 전체 노드의 수와 같다.
  - 색깔의 수를 세는 것은 노드의 수에 선형적인 복잡도를 가짐
  - 종합적으로 시간복잡도는 엣지의 수에 선형적임 <br><br>   
  - 색을 만드는 과정 : O(엣지의 수)
  - 색의 갯수 : O(노드의 수)
  - 색을 세는 과정 : O(노드의 수)
  - 전체 과정 : O(엣지의 수)
  - 위와 같이 모든 과정이 선형적인 시간 복잡도를 가지기 때문에, 사이즈가 큰 그래프에 대해서도 빠르게 연산이 가능

<br>

- Graph-Level Features: Summary
  - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_35.PNG?raw=true"><br><br> 

<br>

<img src="https://github.com/DoyoungKim12/cs224w_review/blob/main/img_cs224w/cs224w_2_36.PNG?raw=true"><br><br> 


