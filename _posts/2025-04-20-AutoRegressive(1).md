# Autoregressive Model (1)
## 🔥 Autoregressive Model 이란?
Autoregressive Model 란 현재 출력이 과거의 출력값들에 조건부로 의지하는 확률 생성 모델이다:

\[
P(\mathbf{x}_{1:D})=P(x_0)\Pi_{i=1:D} P(x_i|\mathbf{x}_{1:i}).
\]

\(P(x_i|\mathbf{x}_{1:i})\)를 모델링하기 어렵지만, \(\mathbf{x}\) 내부의 원소들과의 인과관계가 있는