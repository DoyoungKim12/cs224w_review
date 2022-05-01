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
    - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/master/img_cs224w/cs224w_2_1.PNG?raw=true">
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
    - 목적함수 : 우리가 풀고자 하는 문제가 무엇인가?
  - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/master/img_cs224w/cs224w_2_2.PNG?raw=true">
    - V는 vertices로, node를 vertex로 표현
    - 즉, Graph는 노드의 집합과 링크의 집합으로 정의된다는 것
    - function은 노드(또는 링크, 그래프 구조)를 벡터로 표현해주는 함수를 말함

<br> 

- Node-level Tasks
  - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/master/img_cs224w/cs224w_2_3.PNG?raw=true">
  - 회색 노드가 ML을 거쳐 초록색 또는 빨강색 노드로 분류됨
  - 자세히 보면, 초록 노드는 2개 이상의 노드와 연결되어있고, 빨강 노드는 단 1개의 노드와 연결되어 있음
  - 해당 정보를 feature로 모델에 전달하면, 모델은 해당 정보를 바탕으로 노드를 분류할 수 있게 되는 것임
  - 따라서, ML needs feautures.

<br> 

- Node-level Features: Overview
  - 목표 : 네트워크에 속한 노드의 구조와 위치를 특성화(Characterize)하는 것 
    - Node degree
    - Node centrality
    - Clustering coefficient
    - Graphlets
    - (각각 무엇인지는 뒤에서 살펴볼 것)

<br>

- Node-level Features: Node degree
  - <img src="https://github.com/DoyoungKim12/cs224w_review/blob/master/img_cs224w/cs224w_2_4.PNG?raw=true">
    - 노드 v의 degree k(<img src="https://render.githubusercontent.com/render/math?math=V_k">)는 그 노드가 가진 엣지의 수로 정의됨
    - 모든 이웃한 노드를 동등하게 취급, 따라서 A, F, G, H의 차이를 모델이 알 수 없다는 단점이 있음

<br>
- Node-level Features: Node Centrality
  - Node degree는 이웃한 노드의 중요도(importance)를 포착하지 않고 그 수를 세는 것
  - Node Centrality는 

