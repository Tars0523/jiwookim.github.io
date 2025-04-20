
---

layout: post

title: "Tutorial on Quantitative Trajectory Evaluation"

author: "jiwookim"

---

## Abstract

- Estimated Trajectory의 Quality를 평가하는 방법에 관한 논문

- **첫 번째, Trajectory Alignment**에서 사용할 Transformation Type

- **두 번째, Error Metrics**에 관한 장단점

## Visual(-Inertial) Odometry Formulation

State와 Noise-Free **관측 모델을 정의**하고, VO/VIO를 **최소 제곱 문제**로 공식화한다.

- **States and Measurement Models**

    - State

        $$

        \bold{x}_i=

        \begin{bmatrix}

        \bold{p_i}&R_i&\bold{v}_i&\bold{b}_i^{a}&\bold{b}_i^{g}

        \end{bmatrix}

        $$

    - Camera Model

        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/9827998b-7816-4568-98e5-62f1102a0dad/a27bcf41-c19a-46d1-aaca-c4ac6076fd81/Untitled.png)

        - Pixel Coordinates of tracked features are usually used as the measurements

        $$

        \bold{u}_{ij}=\text{proj}(R_i^{T}\bold{l}_j-R_i^{T}\bold{p}_i)

        $$

    - IMU pre-integration

        $$

        \Delta R_{ik}=R_{i}^{T}R_{k}

        \\

        \Delta\bold{v}_{ik}=R_i^T(\bold{v}_k-\bold{v}_i-\bold{g}\Delta t_{ik})

        \\

        \Delta\bold{p}_{ik}=R_i^T(\bold{p}_k-\bold{p}_i-\bold{v}_i\Delta t_ik-\frac{1}{2}\bold{g}\Delta t^2_{ik})

        $$

- **VO/VIO 최소 제곱 문제**

    Aims to find the $X$ that minimizes the sum of covariance weighted visual and inertial residuals.

    $$

    J(X)=\argmin_{X}||f_V(X)\ominus\bold{\hat{z}}_V||^2+||

    f_I(X)\ominus\bold{\hat{z}}_I||^2

    $$

## Visual(-Inertial) Ambiguity and Trajectory Alignment

Performance trajectory alignment for specific visual(-inertial) setups.

- **Ambiguities and Equivalent Parameters**

    - $J(X)$ 최소화하는 해는 무수히 많다. 왜냐하면 X에 Transformation을 적용하여도 같은 measurement model 이 나올 수 있기 때문이다.다음과 같은 3가지 경우를 수식으로 전개해보면 알 수 있다.

    $$

    \text{어떠한 Similarity Transformation, }S=[s,R,t]

    \text{가 존재 한다고 하자}

    \\

    p_{i}' = sRp_i + t, \quad R_{i}' = RR_i, \quad v_{i}' = sRv_i, \quad l_{j}' = sRl_j + t

    \\

    \text{목점함수에 들어가는 모든 상태를 Similarity Transfromation을 곱하여 변형한다.}

    $$

    $$

    \\

    \bold{-----Monocular\text{ }case-----}

    \\

    \text{기존 관측 방정식: }\bold{u}_{ij}=\text{proj}(R_i^{T}\bold{l}_j-R_i^{T}\bold{p}_i)

    \\

    \text{변형 후 관측 방정식: }

    \bold{u}_{ij}' = \text{proj}(sR^T_i \bold{l}_j - sR^T_i \bold{p}_i)

    \\

    \text{projection후에는 scale이 normalize 되기 때문에, 항상, }\bold{u}_{ij}=\bold{u}_{ij}'

    $$

    $$

    \bold{-----Stereo\text{ }case-----}

    \\

    \text{기존 관측 방정식: }

    u_{sij} = \text{proj}(R^T_i l_j - R^T_i p_i - t_{bs})

    \\

    \text{변현 후 관측 방정식: }

    u_{sij} = \text{proj}(sR^T_i l_j - sR^T_i p_i - t_{bs})

    \\

    \bold{u}_{ij}=\bold{u}_{ij}'

    \text{를 만족시키려면, }s=1

    \text{이여야 한다.그러므로 Rigid Transformation에 해당한다.}

    $$

    $$

    \bold{-----IMU\text{ }case-----}

    \\

    \text{기존 관측 방정식: }

    \Delta R_{ik}=R_{i}^{T}R_{k}

    ,

    \Delta\bold{v}_{ik}=R_i^T(\bold{v}_k-\bold{v}_i-\bold{g}\Delta t_{ik})

    ,

    \Delta\bold{p}_{ik}=R_i^T(\bold{p}_k-\bold{p}_i-\bold{v}_i\Delta t_ik-\frac{1}{2}\bold{g}\Delta t^2_{ik})

    \\

    \text{변형 후 관측 방정식: }

    \Delta R'_{ik} = R^T_i R_k, \quad

    \Delta \bold{v}_{ik} = R^T_i \left( s \bold{v}_k - s \bold{v}_i - R^T \Delta t_{ik} \right), \quad

    \Delta p_{0ik} = R^T_i \left( s \bold{p}_k - s \bold{p}_i - s \bold{v}_i \Delta t_{ik} - s R^T\frac{1}{2}\bold{g} \Delta t_{ik}^2 \right).

    \\

    \text{IMU measurement를 바꾸지 않으려면,}s=1이고 R=R_z\text{이여야 한다}.

    $$

    - 정리하면 다음과 같다.

        - Monocular의 경우 similarity transformation을 적용해도 관측 방정식이 같다.

        - Stereo의 경우 Rigid transformation을 적용해도 관측 방정식이 같다.

        - IMU의 경우 4 DoF yaw-only rigid transformation을 적용해도 관측 방정식이 같다.

- **Trajectory Evaluation with Ambiguities**

    - $\hat{X}$ 이 만드는 subspace를 $E_{est}$ 라고 하자. 이때 $E_{est}$에 존재하는 원소들은 전부 같은 관측 방정식을 가지게 된다. 하지만 이 subspace의 원소들은 전부 $X_{gt}$와 거리가 다르다. 그러므로 $X_{gt}$와 거리가 동일하도록 error를 정의해야한다.

    - subspace 원소들중 $X_{gt}$와 거리가 가장 가까운 요소를 $\hat{X}'$ 라고 한다. 이러한 $\hat{X}'$를 찾는 과정을 **Trajectory Alignment** 라고 하며, $X_{gt}$와$\hat{X}'$사이의 거리를 error 라고 한다.

- **Trajectory Alignment in Visual(-inertial) Systems**

    - 전체적인 알고리듬

        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/9827998b-7816-4568-98e5-62f1102a0dad/c2564c95-b026-489f-bbf3-ed2cc9afe1ee/Untitled.png)

        - $\{ \hat{p}_i \}_{i=0}^{N-1}$: Estimated Position

        - $\{ {p}_i \}_{i=0}^{N-1}$: Ground Truth

        - $S = \{s', R', t'\}$: Similarity transformation

    - Alignment Using Multiple States

        $$

        S' = \text{arg min}_{S=\{s, R, t\}} \left( \sum_{i=0}^{N-1} ||p_i - sRp̂_i - t||^2 \right)

        $$

        - Monocular 의 경우 알고리듬 전부 계산.

        - Stereo 의 경우 $s=1$ 고정 후 계산.

        - IMU의 경우 yaw-only rotation은 아래 수식으로 원큐에 찾아진다.

        $$

        \theta' = \text{arg max}_{\theta} \left( (p_{12} - p_{21}) \sin \theta + (p_{11} + p_{22}) \cos \theta \right)

        $$

    - Alignment Using A Single State

        $$

        R' = R_0 R̂_0^{T}, \quad t' = p_0 - R' p̂_0

        $$

        - IMU의 경우 yaw-only은 아래 수식으로 찾아진다.

            $$

            R'_{z} = \text{arg min}_{R_z} \| R_0 - R_z \hat{R}_0 \|_F^2

            $$

## Trajectory Error Metrics

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/9827998b-7816-4568-98e5-62f1102a0dad/5270e15a-27e2-42f8-b19f-d7cd8591f343/Untitled.png)

- **ATE(Absolute Trajectory Error)**

    - Aligned Estimated Trajectory 와 Ground Truth 사이의 차이는 다음과 같이 나타낼 수 있다.

    $$

    \Delta \bold{x}_i = \{ \Delta R_i, \Delta \bold{p}_i, \Delta \bold{v}_i \}

    \\

    R_i = \Delta R_i \hat{R}'_{i}

    \quad

    \bold{p}_i = \Delta R_i \bold{\hat{p}}_i' + \Delta \bold{p}_i, \quad

    \bold{v}_i = \Delta R_i \bold{\hat{v}}_i' + \Delta \bold{v}_i

    $$

    - 여기서 Error 는 다음과 같이 정의한다.

    $$

    \Delta R_i = R_i (\hat{R}_i')^T, \quad\Delta \bold{p}_i = \bold{p}_i - \Delta R_i \bold{\hat{p}}_i', \quad\Delta \bold{v}_i = \bold{v}_i - \Delta R_i \bold{\hat{v}}_i'

    $$

    - Quantify the quality of the whole trajectory

    $$

    \text{ATE}_{\text{rot}} = \sqrt{\frac{1}{N-1} \sum_{i=0}^{N-1} \left\| \angle(\Delta R_i) \right\|_2^2}

    \\

    \text{ATE}_{\text{pos}} = \sqrt{\frac{1}{N-1} \sum_{i=0}^{N-1} \| \Delta p_i \|_2^2}

    $$

    - $\angle$ 은 SO(3) 로 부터 so(3) 로의 변환이다. Manifold to Euclidean Space. 각도의 변화량을 3요소로 Compact 하게 나타낸다. Lie Group →Lie Algebra

    - ATE의 **장점은 값이 하나**로 나오기 때문에, 비교하기 쉽다는 것이다. 하지만 다음과 같은 **단점이 있다. 초반에 생기는 Rotation 오차는, 후반에 발생하는 Rotation 오차에 비해 ATE에 Dominant하게 작용**한다는 것이다. 그러므로 이러한 단점을 극복하기 위해 Relative error가 적용된다.

- **Relative Error**

    - 일정한 거리를 가진 $K$개의 Pair 상태를 만든다.

    $$

    F = \{\bold{d}_k\}_{k=0}^{K-1}, \quad d_k = \{\bold{\hat{x}}_s, \bold{\hat{x}}_e\}

    $$

    - K개의 각 페어는, sub-trajectory이고 각각 Align 시켜서 계산한다.

    $$

    \text{RE}_{\text{rot}} = \{\delta \bold{\phi}_k\}_{k=0}^{K-1}, \quad \text{RE}_{\text{pos}} = \{\delta \bold{p}_k\}_{k=0}^{K-1}

    $$

    $$

    \delta \phi_k = \angle \delta R_k = \angle R_e (\hat{R}_e')^T , \quad \delta p_k =  \|\bold{p}_e - \delta R_k \bold{\hat{p}}_e'\|_2

    $$

    - 단 한개의 평가 숫자를 주는 것이 아니라 **sub-trajectory에 대해서 다양한 평가**를 하기 때문에, ATE보다 더 많은 정보를 준다. **일정한 거리를 조절해서 다른 의미**를 가지도록 할 수 있다. 예를 들어, 짧은 거리를 기반으로한 Relative Error는 local한 성능을 측정하는 반면, 긴 거리를 기반으로한 Relative Error는 전체적인 consistency를 의미한다. 하지만 Relative Error는 **계산이 복잡**하고, 다양한 정보를 주기 때문에 **Estimation의 Quality를 한번에 단정지을 수 없다는 것이 단점**이다.

- **Discussion and Summary**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/9827998b-7816-4568-98e5-62f1102a0dad/63215d60-c0e6-4abd-961e-66bef71a3de7/Untitled.png)

## Examples
