# Reinforcement

## Test Algorithm 
 : for OpenAI GYM(==0.22.0) - CartPoleV1
 
 (1) A2C
 
 (2) A3C

 (3) DQN

 (4) DDQN
 
 (5) PPO
 
 (6) PPO with RNN
 
 (7) PPO with LSTM
 
 (8) PPO with multiprocessing
 
 (9) pytorch multiprocessing
 
## Test Algorithm/Asynchronous
  : *Asynchronous Methods for Deep Reinforcement Learning* 논문을 읽고, 일부 구현 및 개선점을 제시하는 프로젝트

  - Tensorboard 를 통해 실험 결과를 확인할 수 있다. (Score-Episode)
  - 해당 directory 을 제외한 다른 알고리즘을 돌리다보면, numpy array 를 tensor 로 변환시키는 건 너무 느리다! 라는 warning 이 뜬다면 해당 script 들에서 list 로 변환한다음 list 에 넣고 tensor 로 다시 재변환 하는 것을 확인할 수 있다. 이렇게 바꾸면 warning 이 발생하지 않음. 

```
s.tolist()
```
  
  (1) A3C.py
  
  - 기존의 A3C 로 학습을 하니, stable 하지 않은 거 같아서 gradient cliping 을 통해 안정적으로 학습을 진행한다.

  (2) Q-learning_Asynchronous.py
  
  - one-step Q-learning implementation

  (3) Sarsa_Asynchronous.py
  
  - one-step Sarsa implementation

  (4) DQN.py
  
  - 논문의 비교 대상인 DQN algorithm 구현
 
  (5) Asynchronous_Methods_for_Deep_Reinforcement_Learning_2023126703_박준서.docx
 
  - Paper review 와 함께 논문을 일부 구현 및 개선한 보고서

## 똥 피하기 with 동민
  : practice in simple game
  
  (1) DQN
  
  (2) DDQN
  
  (3) PPO
  
  (4) PPO with RNN
  
  (5) PPO with LSTM
  
  
  
  
  
