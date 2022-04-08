#### 7.1

$$
\begin{aligned}
G_{t:t+n}-V(S_t)&=R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^n V_{t+n-1}
(S_{t+n})-V(S_t)\\
&=R_{t+1}+\gamma V(S_{t+1})-\gamma V(S_{t+1})-V(S_t)+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^nV_{t+n-1}(S_{t+n})\\
&=\delta_t +\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^nV_{t+n-1}(S_{t+n})-\gamma V(S_{t+1})\\
&=\delta_t+\gamma [G_{t+1:t+n}-V(S_{t+1})]\\
&=\sum_k ^{n-1}\gamma^k \delta_{t+k}
\end{aligned}
$$

#### 7.3

更大的随机游走任务更能体现n步时序差分的优势.

左侧的收益改为-1是为了增加更新幅度

#### 7.4

$$
\begin{aligned}
G_{t:t+n}&\doteq R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}
R_{t+n}+\gamma ^n Q_{t+n-1}(S_{t+n},A_{t+n})\\
&=R_{t+1}+\gamma Q_t(S_{t+1},A_{t+1})-Q_{t-1}(S_t,A_t)-\gamma Q_t(S_{t+1},A_{t+1})+Q_{t-1}(S_t,A_t)+\gamma R_{t+2}+\cdots +\gamma^{n-1}
R_{t+n}+\gamma ^n Q_{t+n-1}(S_{t+n},A_{t+n})\\
&=(R_{t+1}+\gamma Q_t(S_{t+1},A_{t+1})-Q_{t-1}(S_t,A_t))+\gamma(R_{t+2}+\gamma Q_{t+1}(S_{t+2},A_{t+2})-Q_t(S_{t+1},A_{t+1}))-\gamma^2Q_{t+1}(S_{t+2},A_{t+2})+Q_{t-1}(S_t,A_t)+\cdots+\gamma^{n-1}
R_{t+n}+\gamma ^n Q_{t+n-1}(S_{t+n},A_{t+n})\\
&=Q_{t-1}(S_t,A_t)+\sum_{k=t}^{\min(t+n,T)-1}\gamma ^{k-t}[R_{k+1}+\gamma Q_k(S_{k+1},A_{k+1})-Q_{k-1}(S_k,A_k)]
\end{aligned}
$$

#### 7.5

只需在n步时序差分算法中加入 $\rho_t$ 的更新, 以及将 $G_t$ 的更新用
$$
G_{t:h}=\rho_t(R_{t+1}+\gamma G_{t+1:h})+(1-\rho_t)V_{h-1}(S_t), t<h<T\\
G_{h:h}=V_{h-1}(S_h)
$$
替换即可

#### 7.6

对于式(7.13)
$$
\begin{aligned}
\mathbb E[(1-\rho_t)V_{h-1}(S_t)]&=\mathbb E_b[(1-\rho_t)]\mathbb E[V_{h-1}(S_t)]\\
&=0
\end{aligned}
$$
对于式(7.14)
$$
\begin{aligned}
&\mathbb E_b[\bar V_{h-1}(S_{t+1})-\rho_{t+1}Q_{h-1}(S_{t+1},A_{t+1})]\\
&=\sum_a \pi(a|S_{t+1})Q_{h-1}(S_{t+1},a)-\sum_a b(a|S_{t+1})\frac{\pi(a|S_{t+1})}{b(a|S_{t+1})}Q_{h-1}(S_{t+1},a)\\
&=0
\end{aligned}
$$

#### 7.7

![ex7.7](D:\RL\exercises\ex7.7.png)

[vojtamolda/reinforcement-learning-an-introduction: Solutions to exercises in Reinforcement Learning: An Introduction (2nd Edition). (github.com)](https://github.com/vojtamolda/reinforcement-learning-an-introduction)

#### 7.8

$$
\begin{aligned}
G_{t:h}&=\rho_t(R_{t+1}+\gamma G_{t+1:h})+(1-\rho_t)V_{h-1}(S_t)\\
&=\rho_t(R_{t+1}+\gamma G_{t+1:h}-V(S_t))+V(S_t)\\
&=V(S_t)+\rho_t (R_{t+1}+\gamma V(S_{t+1})-V(S_t))+\gamma \rho_{t:t+1}(R_{t+2}+\gamma G_{t+2:n}-V(S_{t+1}))\\
&=V(S_t)+\rho_t \delta_t +\gamma \rho_{t:t+1}\delta_{t+1}+\cdots\\
&=V(S_t)+\sum_{k=t}^{h-1}\rho_{t:k}\gamma ^{t-k}\delta_k
\end{aligned}
$$

#### 7.9

$$
G_{t:h}-Q(S_t,A_t)=\sum_{k=t}^{h-2}\gamma ^ {k-t-1}\rho_{t+1:k}\delta_k
$$

#### 7.11

$$
\begin{aligned}
G_{t:t+n}-Q(S_t,A_t)&=R_{t+1}+\gamma \sum_{a\neq A_{t+1}}\pi(a|S_{t+1})Q_{t+n-1}(S_{t+1},a)+\gamma\pi(A_{t+1}|S_{t+1})G_{t+1:t+n}-Q(S_t,A_t)\\
&=R_{t+1}+\gamma \bar V(S_{t+1})-\gamma \pi (A_{t+1}|S_{t+1})Q(S_{t+1},A_{t+1})-Q(S_t,A_t)+\gamma \pi(A_{t+1}|S_{t+1})G_{t+1:t+n}\\
&=\delta_t-\gamma \pi(A_{t+1}|S_{t+1})(G_{t+1:t+n}-Q(S_{t+1},A_{t+1}))\\
&=\sum_{k=t}^{\min (t+n-1,T-1)} \delta_k\prod_{i=t+1}^k\gamma\pi(A_i|S_i)
\end{aligned}
$$

