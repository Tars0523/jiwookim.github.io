## Abstract

- Estimated Trajectory의 Quality를 평가하는 방법에 관한 논문
- **첫 번째, Trajectory Alignment**에서 사용할 Transformation Type
- **두 번째, Error Metrics**에 관한 장단점

## 1. Visual(-Inertial) Odometry Formulation

State와 Noise-Free **관측 모델**을 정의하고, VO/VIO를 **최소제곱 문제**로 공식화한다.

### 1.1 States and Measurement Models

#### 1.1.1 State

```latex
$$
\mathbf{x}_i =
\begin{bmatrix}
  \mathbf{p}_i \\
  R_i        \\
  \mathbf{v}_i \\
  \mathbf{b}_i^{a} \\
  \mathbf{b}_i^{g}
\end{bmatrix}
$$
```

#### 1.1.2 Camera Model

```latex
\begin{aligned}
\mathbf{u}_{ij}
&= \mathrm{proj}\bigl(R_i^{\top}(\mathbf{l}_j - \mathbf{p}_i)\bigr).
\end{aligned}
```

![Camera Model](https://prod-files-secure.s3.us-west-2.amazonaws.com/9827998b-7816-4568-98e5-62f1102a0dad/a27bcf41-c19a-46d1-aaca-c4ac6076fd81/Untitled.png)

#### 1.1.3 IMU Pre-integration

```latex
\begin{aligned}
\Delta R_{ik} &= R_i^{\top} R_k, \\
\Delta \mathbf{v}_{ik} &= R_i^{\top}\bigl(\mathbf{v}_k - \mathbf{v}_i - \mathbf{g}\,\Delta t_{ik}\bigr), \\
\Delta \mathbf{p}_{ik} &= R_i^{\top}\bigl(\mathbf{p}_k - \mathbf{p}_i - \mathbf{v}_i\,\Delta t_{ik} - \tfrac{1}{2}\mathbf{g}\,\Delta t_{ik}^2\bigr).
\end{aligned}
```

### 1.2 VO/VIO 최소제곱 문제

```latex
$$
X^* = \arg\min_{X} \Bigl\|f_V(X) \ominus \hat{\mathbf{z}}_V\Bigr\|^2 + \Bigl\|f_I(X) \ominus \hat{\mathbf{z}}_I\Bigr\|^2.
$$
```

## 2. Visual(-Inertial) Ambiguity and Trajectory Alignment

최적화된 궤적 해는 변환을 적용해도 관측 모델이 보존되는 유한/무한 개의 등가 해를 갖는다.

### 2.1 Similarity Transformation

Similarity Transformation $S = \{s, R, t\}$를 모든 상태에 적용하면:

```latex
\begin{aligned}
\mathbf{p}'_i &= s\,R\,\mathbf{p}_i + t, \\
R'_i           &= R\,R_i,      \\
\mathbf{v}'_i &= s\,R\,\mathbf{v}_i, \\
\mathbf{l}'_j &= s\,R\,\mathbf{l}_j + t.
\end{aligned}
```

#### 2.1.1 Monocular Case

```latex
\begin{aligned}
\mathbf{u}_{ij} &= \mathrm{proj}\bigl(R_i^{\top}(\mathbf{l}_j - \mathbf{p}_i)\bigr), \\
\mathbf{u}'_{ij}&= \mathrm{proj}\bigl(s\,R_i^{\top}(\mathbf{l}_j - \mathbf{p}_i)\bigr).
\end{aligned}
```

projection 이후 scale이 normalize되어 항상 $\mathbf{u}_{ij}=\mathbf{u}'_{ij}$.

#### 2.1.2 Stereo Case

```latex
\begin{aligned}
\mathbf{u}_{s,ij} &= \mathrm{proj}\bigl(R_i^{\top}(\mathbf{l}_j - \mathbf{p}_i) - \mathbf{t}_{b}^{s}\bigr), \\
\mathbf{u}'_{s,ij}&= \mathrm{proj}\bigl(s\,R_i^{\top}(\mathbf{l}_j - \mathbf{p}_i) - \mathbf{t}_{b}^{s}\bigr).
\end{aligned}
```

Stereo 측정이 보존되려면 $s=1$ (Rigid Transformation).

#### 2.1.3 IMU Case

```latex
\begin{aligned}
\Delta R'_{ik} &= R_i^{\top} R_k, \\
\Delta \mathbf{v}'_{ik} &= R_i^{\top}\bigl(s\,\mathbf{v}_k - s\,\mathbf{v}_i - \mathbf{g}\,\Delta t_{ik}\bigr), \\
\Delta \mathbf{p}'_{ik} &= R_i^{\top}\bigl(s\,\mathbf{p}_k - s\,\mathbf{p}_i - s\,\mathbf{v}_i\,\Delta t_{ik} - \tfrac{s}{2}\mathbf{g}\,\Delta t_{ik}^2\bigr).
\end{aligned}
```

IMU measurement 보존을 위해 $s=1$, $R$은 yaw-only rotation.

### 2.2 Trajectory Alignment

추정 궤적 $\hat{X}$의 subspace $E_{est}$ 내에서

```latex
$$
S^* = \arg\min_{S=\{s,R,t\}} \sum_{i=0}^{N-1} \|\mathbf{p}_i - s\,R\,\hat{\mathbf{p}}_i - t\|^2.
$$
```

- Monocular: $s,R,t$ 모두 최적화
- Stereo: $s=1$ 고정 후 $R,t$ 최적화
- IMU (yaw-only):

```latex
\theta^* = \arg\max_{\theta}\bigl((p_{12}-p_{21})\sin\theta + (p_{11}+p_{22})\cos\theta\bigr).
```

#### Single-State Alignment

```latex
R^* = R_0\hat{R}_0^{\top}, \quad t^* = \mathbf{p}_0 - R^*\hat{\mathbf{p}}_0.
```

IMU (yaw-only):

```latex
R'_z = \arg\min_{R_z} \|R_0 - R_z\hat{R}_0\|_F^2.
```

## 3. Trajectory Error Metrics

![Error Metrics](https://prod-files-secure.s3.us-west-2.amazonaws.com/5270e15a-27e2-42f8-b19f-d7cd8591f343/Untitled.png)

### 3.1 Absolute Trajectory Error (ATE)

추정값과 기준 궤적 간 회전·위치·속도 오차:

```latex
\Delta\mathbf{x}_i = \{\Delta R_i,\,\Delta\mathbf{p}_i,\,\Delta\mathbf{v}_i\},
```

```latex
\begin{aligned}
\Delta R_i     &= R_i(\hat{R}'_i)^{\top}, \\
\Delta\mathbf{p}_i &= \mathbf{p}_i - \Delta R_i\hat{\mathbf{p}}'_i, \\
\Delta\mathbf{v}_i &= \mathbf{v}_i - \Delta R_i\hat{\mathbf{v}}'_i.
\end{aligned}
```

```latex
\text{ATE}_{\mathrm{rot}} = \sqrt{\frac{1}{N-1}\sum_{i=0}^{N-1}\|\angle(\Delta R_i)\|_2^2}, \quad
\text{ATE}_{\mathrm{pos}} = \sqrt{\frac{1}{N-1}\sum_{i=0}^{N-1}\|\Delta\mathbf{p}_i\|_2^2}.
```

- **장점:** 단일 값으로 궤적 품질 비교 가능
- **단점:** 초기 회전 오차가 전체 에러에 지배적

### 3.2 Relative Error (RE)

길이 $K$의 sub-trajectory 쌍 $F=\{(\hat{x}_s,\hat{x}_e)\}_{k=0}^{K-1}$에서

```latex
\begin{aligned}
\delta\phi_k &= \angle\bigl(R_e(\hat{R}'_e)^{\top}\bigr), \\
\delta p_k   &= \bigl\|\mathbf{p}_e - \delta R_k\hat{\mathbf{p}}'_e\bigr\|_2.
\end{aligned}
```

- **장점:** 다양한 거리 구간별 로컬/글로벌 성능 평가 가능
- **단점:** 계산 복잡, 단일 지표 제공 불가