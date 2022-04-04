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

