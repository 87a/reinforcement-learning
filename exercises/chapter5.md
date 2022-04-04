#### 5.1

估计的价值函数在最后两行突然增高, 因为玩家总和为20或21时, 有非常大的几率获得胜利.

最靠左侧的一列降低, 因为庄家显示的牌是A, 庄家有 $\frac{4}{13}$ 的概率拿到21点. (K,Q,J,10)

上方图中靠前的位置比下方图中对应的位置要高,  因为玩家手中有可用的A, 爆牌的可能性更小, 也就有更大的可能性赢.

#### 5.2

结果不会不一样. 因为在每一幕中每个状态都只会被访问一次.

#### 5.3

与书中的回溯图相同,只不过结点变成"状态-价值"二元组.

#### 5.4

增加一个记录次数的变量count, 然后在更新时不用average,而是用count来更新.

#### 5.5

$$
v_{fv}(s)=10\\
v_{ev}(s)=average([1,2,3,\cdots,10])=\frac{11}{2}
$$

#### 5.6

$$
Q(s,a)=\frac{\sum_{t\in\mathcal{T}(s,a)}\rho_{t+1:T-1}G_t}{\sum_{t\in\mathcal{T}(s,a)}\rho_{t+1:T-1}}
$$
#### 5.7

在图5.3中, 在比较早的幕中,行动策略会与目标策略接近. 但多幕后情况也变多, 所以方差会上升.

#### 5.8

是无穷, 因为回报都大于0

#### 5.9 首次访问型MC策略评估的增量式实现

$$
\begin{aligned}
&输入:待评估的策略\pi\\
&初始化:\\
&\qquad 对所有s\in \mathcal{S}, 初始化V(s)=0\\
&\qquad 对所有s\in \mathcal{S}, 初始化N(s)=0\\
&无限循环(对每幕):\\
&\qquad 根据\pi生成一幕序列:S_0,A_0,R_1,S_1,A_1,R_2,\cdots,S_{T-1},A_{T-1},R_T\\
&\qquad G\leftarrow 0\\
&\qquad 对本幕中的每一步进行循环, t=T-1,T-2,\cdots,0:\\
&\qquad \qquad G\leftarrow \gamma G+R_{t+1}\\
&\qquad \qquad 除非S_t在S_0,S_1,\cdots,S_{t-1}中已经出现过:\\
&\qquad \qquad \qquad N(S_t)\leftarrow N(S_t)+1\\
&\qquad \qquad \qquad V(S_t)\leftarrow V(S_t)+\frac{(G-V(S_t))}{N(S_t)}
\end{aligned}
$$

#### 5.10

根据加权重要度采样的公式
$$
V(s)\doteq \frac{\sum_{t\in\mathcal{T}(s)}\rho_{t:T(t)-1}G_t}{\sum_{t\in\mathcal{T}(s)}\rho_{t:T(t)-1}}\\
$$
可以推出
$$
V_{n+1}=\frac{V_n(C_n-W_n)+W_nG_n}{C_n}=V_n+\frac{W_n(G_n-V_n)}{C_n}
$$

#### 5.11

因为 $\pi$ 是贪心策略
$$
\pi(a|s)=\mathbb{1}\{a=\underset{a'}{\arg\max}Q(s,a')\}
$$

#### 5.13

$$
\begin{aligned}
\mathbb E[\rho_{t:T-1}R_t]&=\mathbb E_b[\frac{\pi(A_{t}|S_{t})}{b(A_{t}|S_{t})}\frac{\pi(A_{t+1}|S_{t+1})}{b(A_{t+1}|S_{t+1})}\cdots\frac{\pi(A_{T-1}|S_{T-1})}{b(A_{T-1}|S_{T-1})}]\\
&=\mathbb E_b[\frac{\pi(A_t|S_t)}{b(A_t|S_t)}R_{t+1}]\\
&=\mathbb E_b[\rho_{t:t}R_{t+1}]
\end{aligned}
$$

#### 5.14

