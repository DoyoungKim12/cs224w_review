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
        - N(v)는 v의 노드 집합이다.
        - λ는 정규화 인자이다. <br><br>
    - 이는 재귀적(recursive)인 방식으로 정의되는데, 이를 어떻게 풀 수 있을까?<br><br>
      - 일단 우리는 c를 모르니까, 모든 값을 1로 초기화하자. 노드가 4인 그래프라면 \[1,1,1,1]이 될 것이다.
        - 즉, 각 노드의 중요도가 모두 1로 같다고 보고 시작하자. <br><br>
      - 그 다음으로, 이 값을 어떻게 업데이트할지 생각해보자. 
        - 우리가 특정 그래프 안을 무작위로 돌아다닌다고 할 때(Random Walk), 특정 노드를 많이 방문한다면 그 노드는 중요도가 높은 노드라고 말할 수 있을 것이다.
        - 그렇다면 우리는 저 초기값을 '특정 노드에서 다른 노드로 이동하는 정보', 즉 돌아다닐 수 있는 길의 정보를 사용해 업데이트해야 할 것이다.
        - '특정 노드에서 다른 노드로 이동하는 정보'를 우리는 인접행렬(adjacency matrix)로 표현할 수 있다.
        - 그럼 업데이트된 c는 이전 c에 인접행렬 A를 곱해준 정보이다! (실제로 초기값에 A를 곱한 값은 node degree이다. 궁금하면 직접 해보면 바로 나온다.)
        - 업데이트된 c의 값들은 이전보다 계속 커질 것이므로, 아무튼 λ 같은 값으로 나눠준다. <br><br>
      - 이런 식의 업데이트를 무한히 반복한다고 하자.
        - 그러다보면, 어느 순간 업데이트된 c와 이전 c의 차이가 없어지는 상태에 도달할 것이다. (stationary distribution과 비슷한 느낌?)
        - 그럼 더이상 업데이트할 필요가 없으니 해당 벡터 c를 가져다 쓰면 되겠다.
        - 이러한 상태를 식으로 표현하면 아래와 같다.
          - <img src="https://render.githubusercontent.com/render/math?math=c_{t %2B 1} = \frac{1}{\lambda}Ac_{t}">
          - t+1기나 t기나 같으니까 해당 표현을 빼주고, 람다를 좌변에 곱해주면, 비로소 재귀적 형태를 행렬 형태로 바꿀 수 있게 된다. <br><br>
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








