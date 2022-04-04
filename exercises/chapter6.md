#### 6.1

$$
\begin{aligned}
G_t-V(S_t)&=R_{t+1}+\gamma G_{t+1}-V_t(S_t)+\gamma V_t(S_{t+1})-\gamma V_t(S_{t+1})\\
&=R_{t+1}+\gamma V_t(S_{t+1})-V_t(S_t)+\gamma(G_{t+1}-V_t(S_{t+1}))\\
&=\delta_t+\gamma (G_{t+1}-V_{t+1}(S_{t+1}))+\gamma(G_{t+1}-V_t(S_{t+1}))\\
&=\delta_t+\gamma(G_{t+1}-V_{t+1}(S_{t+1}))+\alpha\gamma(R_{t+2}+\gamma V_t(S_{t+2})-V_t(S_{t+1}))\\
&\vdots\\
&=\sum_{k=t}^{T-1}\gamma^{k-t}\delta_k +\alpha \sum_{k=t}^{T-2}\gamma^{k-t-1}[R_{k+2}+\gamma V_k(S_{k+2})-V_k(S_{k+1})]
\end{aligned}
$$

#### 6.2

从新地方去上班，前面这段路是没见过的，但是上高速路了之后就和之前一样了。这种情形下，TD方法因为会利用下一步的估计值，相比于蒙特卡洛方法从头开始学起，肯定会收敛更快。

#### 6.3

第一幕过程在 $A$ 处结束,
$$
\begin{aligned}
V(S_t)&=V(S_t)+\alpha (R_{t+1}+\gamma V(S_{t+1})-V(S_t))\\
&=0.5 + 0.1\times [0+0-0.5]\\
&=0.45
\end{aligned}
$$
对于其他状态 $V(S_{t+1})=0.5$ , 所以估计值不变.

#### 6.4

General arguments given earlier about the benefits of TD are independent of α
Increases in α make the curve more
Decreases in α make the curve more smooth but make it converge slower.
We see enough of a range here to decide between the two methods

#### 6.5

状态 $C$ 凑巧被初始化为它的真实值. 训练开始后, 外侧状态先开始更新, 导致error下降, 直到剩余的error被传播到C. $\alpha$ 值更大, 这种效应更显著, 因为这种情况下 $C$ 的估计值更容易变化.

#### 6.6

可以使用贝尔曼方程, 得到线性方程:
$$
\left[ \begin{matrix} -1&\frac{1}{2}&0&0&0&\frac{1}{2}\\
\frac{1}{2}&-1&\frac{1}{2}&0&0&0\\
0&\frac{1}{2}&-1&\frac{1}{2}&0&0\\
0&0&\frac{1}{2}&-1&\frac{1}{2}&0\\
0&0&0&\frac{1}{2}&-1&\frac{1}{2}\\
0&0&0&0&0&1\end{matrix} \right]
\left[\begin{matrix}V(A)\\V(B)\\V(C)\\V(D)\\V(E)\\V(T)\end{matrix}\right]
=\left[  \begin{matrix}0\\0\\0\\0\\-\frac{1}{2}\\0 \end{matrix}\right]
$$
可以通过numpy求解

```python
A = np.array([[-1, 0.5, 0,0,0,0.5],
             [0.5, -1, 0.5, 0,0,0],
             [0, 0.5,-1,0.5,0,0],
             [0,0,0.5,-1,0.5,0],
             [0,0,0,0.5,-1,0.5],
             [0,0,0,0,0,1]])
B = np.array([0,0,0,0,-0.5,0]).T
np.dot(np.linalg.inv(A),B)
```

即可求出答案.

#### 6.7

$$
V(S_t)\leftarrow V(S_t)+\alpha [\rho_{t:t}R_{t+1}+\rho_{t:t} \gamma V(S_{t+1})-V(S_t) ]
$$

#### 6.8

$$
\begin{aligned}
G_t-Q(S_t,A_t)&=R_{t+1}+\gamma G_{t+1}-Q(S_t,A_t)+\gamma Q(S_{t+1},A_{t+1})-\gamma Q(S_{t+1},A_{t+1})\\
&=R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)+\gamma G_{t+1}-\gamma Q(S_{t+1},A_{t+1})\\
&=\delta_t +\gamma (G_{t+1}-Q(S_{t+1},A_{t+1}))\\
&=\delta_t+\gamma \delta_{t+1}+\cdots+\gamma^{T-1}(G_T-Q(S_T,A_T))\\
&=\sum_{k=t}^{T-1}\gamma^{k-t}\delta_k
\end{aligned}
$$

#### 6.11

Q-learning与环境交互用的是 $\epsilon-greedy$ 策略, 学习用的是greedy策略, 所以是离轨策略控制方法.

#### 6.12

可以说是

#### 6.13

$$
Q_1(S_t,A_t)\leftarrow Q_1(S_t,A_t)+\alpha [R_{t+1}+\gamma \sum_a \pi(a|S_{t+1})Q_2(S_{t+1},a)-Q_1(S_t,A_t)]
$$

#### 6.14

杰克租车问题中, 车辆调度的状态和动作可能不同, 但调度后的状态可能相同.
