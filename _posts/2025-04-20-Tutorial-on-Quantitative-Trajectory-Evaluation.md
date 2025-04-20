## Abstract

- Estimated Trajectory의 Quality를 평가하는 방법에 관한 논문
- **첫 번째, Trajectory Alignment**에서 사용할 Transformation Type
- **두 번째, Error Metrics**에 관한 장단점

## Visual(-Inertial) Odometry Formulation

State와 Noise-Free **관측 모델**을 정의하고, VO/VIO를 **최소 제곱 문제**로 공식화한다.

### States and Measurement Models

#### State

$$
\mathbf{x}_i =
\begin{bmatrix}
\mathbf{p}_i \\
R_i \\
\mathbf{v}_i \\
\mathbf{b}_i^{a} \\
\mathbf{b}_i^{g}
\end{bmatrix}
$$

#### Camera Model

![Camera Model](https://prod-files-secure.s3.us-west-2.amazonaws.com/9827998b-7816-4568-98e5-62f1102a0dad/a27bcf41-c19a-46d1-aaca-c4ac6076fd81/Untitled.png)

Pixel coordinates of tracked features are usually used as the measurements:

$$
\mathbf{u}_{ij} = \mathrm{proj}\bigl(R_i^{\top}\mathbf{l}_j - R_i^{\top}\mathbf{p}_i\bigr).
$$

#### IMU Pre-integration

$$
\begin{aligned}
\Delta R_{ik} &= R_i^{\top} R_k, \\
\Delta \mathbf{v}_{ik} &= R_i^{\top}\bigl(\mathbf{v}_k - \mathbf{v}_i - \mathbf{g}\,\Delta t_{ik}\bigr), \\
\Delta \mathbf{p}_{ik} &= R_i^{\top}\bigl(\mathbf{p}_k - \mathbf{p}_i - \mathbf{v}_i\,\Delta t_{ik} - \tfrac{1}{2}\mathbf{g}\,\Delta t_{ik}^2\bigr).
\end{aligned}
$$

### VO/VIO 최소 제곱 문제

Aims to find the $X$ that minimizes the sum of covariance-weighted visual and inertial residuals:

$$
J(X) = \arg\min_{X} \bigl\|f_V(X) \ominus \hat{\mathbf{z}}_V\bigr\|^2 + \bigl\|f_I(X) \ominus \hat{\mathbf{z}}_I\bigr\|^2.
$$

## Visual(-Inertial) Ambiguity and Trajectory Alignment

Performance trajectory alignment for specific visual(-inertial) setups.

### Ambiguities and Equivalent Parameters

$J(X)$를 최소화하는 해는 무수히 많다. 이는 변환을 적용해도 같은 관측 모델이 유지되기 때문이다. 다음과 같은 세 가지 경우로 전개해볼 수 있다.

어떠한 Similarity Transformation $S = \{s, R, t\}$가 존재한다고 하자:

$$
\begin{aligned}
\mathbf{p}'_i &= s R\,\mathbf{p}_i + t, \\
R'_i &= R\,R_i, \\
v'_i &= s R\,\mathbf{v}_i, \\
\mathbf{l}'_j &= s R\,\mathbf{l}_j + t.
\end{aligned}
$$

모든 상태에 이 변환을 적용해도 관측 방정식이 변하지 않는다.

#### Monocular Case

원래 관측 방정식:

$$
\mathbf{u}_{ij} = \mathrm{proj}\bigl(R_i^{\top}\mathbf{l}_j - R_i^{\top}\mathbf{p}_i\bigr).
$$

변환 후:

$$
\mathbf{u}'_{ij} = \mathrm{proj}\bigl(s R_i^{\top}\mathbf{l}_j - s R_i^{\top}\mathbf{p}_i\bigr).
$$

projection 이후 scale이 normalize되기 때문에 항상 $\mathbf{u}_{ij} = \mathbf{u}'_{ij}$.

#### Stereo Case

원래 관측 방정식:

$$
u_{sij} = \mathrm{proj}\bigl(R_i^{\top}\mathbf{l}_j - R_i^{\top}\mathbf{p}_i - \mathbf{t}_{bs}\bigr).
$$

변환 후:

$$
nu'_{sij} = \mathrm{proj}\bigl(s R_i^{\top}\mathbf{l}_j - s R_i^{\top}\mathbf{p}_i - \mathbf{t}_{bs}\bigr).
$$

이때 $\mathbf{u}_{ij} = \mathbf{u}'_{ij}$가 성립하려면 $s = 1$, 즉 Rigid Transformation에 해당한다.

#### IMU Case

원래 관측 방정식:

$$
\begin{aligned}
\Delta R_{ik} &= R_i^{\top} R_k, \\
\Delta \mathbf{v}_{ik} &= R_i^{\top}\bigl(\mathbf{v}_k - \mathbf{v}_i - \mathbf{g}\,\Delta t_{ik}\bigr), \\
\Delta \mathbf{p}_{ik} &= R_i^{\top}\bigl(\mathbf{p}_k - \mathbf{p}_i - \mathbf{v}_i\,\Delta t_{ik} - \tfrac{1}{2}\mathbf{g}\,\Delta t_{ik}^2\bigr).
\end{aligned}
$$

변환 후:

$$
\begin{aligned}
\Delta R'_{ik} &= R_i^{\top}R_k, \\
\Delta \mathbf{v}'_{ik} &= R_i^{\top}\bigl(s\mathbf{v}_k - s\mathbf{v}_i - \mathbf{g}\,\Delta t_{ik}\bigr), \\
\Delta \mathbf{p}'_{ik} &= R_i^{\top}\Bigl(s\mathbf{p}_k - s\mathbf{p}_i - s\mathbf{v}_i\,\Delta t_{ik} - s\tfrac{1}{2}\mathbf{g}\,\Delta t_{ik}^2\Bigr).
\end{aligned}
$$

IMU measurement를 바꾸지 않으려면 $s = 1$이고 $R$은 yaw-only rotation ($R_z$)이어야 한다.

정리:
- Monocular: Similarity Transformation
- Stereo: Rigid Transformation
- IMU: 4-DoF yaw-only Rigid Transformation

### Trajectory Evaluation with Ambiguities

추정된 궤적 $\hat{X}$가 만드는 subspace를 $E_{est}$라 하자. 이때 $E_{est}$의 모든 원소는 같은 관측 방정식을 가지지만, 각각 $X_{gt}$와의 거리가 다르다. $X_{gt}$와 거리가 가장 가까운 원소를 $\hat{X}'$라 하며, 이를 찾는 과정을 **Trajectory Alignment**라 부른다. $X_{gt}$와 $\hat{X}'$ 사이의 거리를 에러로 정의한다.

### Trajectory Alignment in Visual(-inertial) Systems

전체 알고리즘:

![Alignment Algorithm](https://prod-files-secure.s3.us-west-2.amazonaws.com/9827998b-7816-4568-98e5-62f1102a0dad/c2564c95-b026-489f-bbf3-ed2cc9afe1ee/Untitled.png)

- $\{\hat{p}_i\}_{i=0}^{N-1}$: Estimated Positions
- $\{p_i\}_{i=0}^{N-1}$: Ground Truth
- $S = \{s', R', t'\}$: Similarity Transformation

#### Multiple-State Alignment

$$
S' = \arg\min_{S=\{s,R,t\}} \sum_{i=0}^{N-1} \bigl\|p_i - s R \hat{p}_i - t\bigr\|^2.
$$

- Monocular: $s,R,t$ 모두 최적화
- Stereo: $s=1$ 고정 후 최적화
- IMU (yaw-only):

$$
\theta' = \arg\max_{\theta}\bigl((p_{12}-p_{21})\sin\theta + (p_{11}+p_{22})\cos\theta\bigr)
$$

#### Single-State Alignment

$$
R' = R_0\hat{R}_0^{\top}, \quad t' = p_0 - R'\hat{p}_0.
$$

- IMU (yaw-only):

$$
R'_z = \arg\min_{R_z} \|R_0 - R_z \hat{R}_0\|_F^2.
$$

## Trajectory Error Metrics

![Error Metrics](https://prod-files-secure.s3.us-west-2.amazonaws.com/9827998b-7816-4568-98e5-62f1102a0dad/5270e15a-27e2-42f8-b19f-d7cd8591f343/Untitled.png)

### ATE (Absolute Trajectory Error)

정의된 궤적 사이 오차:

$$
\Delta \mathbf{x}_i = \{\Delta R_i, \Delta \mathbf{p}_i, \Delta \mathbf{v}_i\},
$$

$$
\begin{aligned}
R_i &= \Delta R_i\hat{R}'_i, \\
\mathbf{p}_i &= \Delta R_i\hat{\mathbf{p}}'_i + \Delta\mathbf{p}_i, \\
\mathbf{v}_i &= \Delta R_i\hat{\mathbf{v}}'_i + \Delta\mathbf{v}_i,
\end{aligned}
$$

$$
\Delta R_i = R_i(\hat{R}'_i)^{\top}, \quad \Delta\mathbf{p}_i = \mathbf{p}_i - \Delta R_i\hat{\mathbf{p}}'_i, \quad \Delta\mathbf{v}_i = \mathbf{v}_i - \Delta R_i\hat{\mathbf{v}}'_i.
$$

$$
\text{ATE}_{\mathrm{rot}} = \sqrt{\frac{1}{N-1}\sum_{i=0}^{N-1}\|\angle(\Delta R_i)\|_2^2}, \quad
\text{ATE}_{\mathrm{pos}} = \sqrt{\frac{1}{N-1}\sum_{i=0}^{N-1}\|\Delta\mathbf{p}_i\|_2^2}.
$$

여기서 $\angle: \mathrm{SO}(3)\to\mathfrak{so}(3)$는 Lie Group에서 Lie Algebra로의 사상으로, 회전 변화를 3-벡터로 표현한다.

**장점:** 단일 값으로 궤적 품질 비교 가능
**단점:** 초기 회전 오차가 전체 에러에 지배적

### Relative Error

길이 $K$의 sub-trajectory 쌍을 구성:

$$
F = \{d_k\}_{k=0}^{K-1}, \quad d_k = (\hat{x}_s,\hat{x}_e).
$$

각 sub-trajectory를 정렬하여 계산:

$$
\mathrm{RE}_{\mathrm{rot}} = \{\delta\phi_k\}, \quad \mathrm{RE}_{\mathrm{pos}} = \{\delta p_k\},
$$

$$
\delta\phi_k = \angle\bigl(R_e(\hat{R}'_e)^{\top}\bigr), \quad \delta p_k = \|p_e - \delta R_k\hat{p}'_e\|_2.
$$

**장점:** 다양한 거리 기준에 따른 로컬/글로벌 성능 평가
**단점:** 계산 복잡, 단일 지표 제공 불가

