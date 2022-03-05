ex 3.1

(1)任何棋类游戏

state:棋牌上棋子的位置

action:移动或其他可行动作

reward:胜/负/平

(2)机器人打扫房间

state:机器人的位置,灰尘的数量

action:机器人移动和打扫

reward:灰尘减少的量

(3)足球教练

state:场上的比分,人员等

action:战术,换人等

reward:进球,胜负



ex 3.2

不能.该任务必须具有马尔可夫性,即当一个[随机过程](https://baike.baidu.com/item/随机过程)在给定现在状态及所有过去状态情况下，其未来状态的条件[概率分布](https://baike.baidu.com/item/概率分布)仅依赖于当前状态.



ex 3.3

这取决于任务的具体要求.如果任务的要求是从一处到另一处,那么action可以是控制车的方向和速度.如果在多处来回驾驶,那么action可以是选择下一次的目的地.



ex 3.4

| s    | a        | s'   | r            | p(s',r\|s,a) |
| ---- | -------- | ---- | ------------ | ------------ |
| high | search   | high | $r_{reward}$ | $\alpha$     |
| high | search   | low  | $r_{reward}$ | $1-\alpha$   |
| high | weight   | high | $r_{wait}$   | 1            |
| low  | search   | low  | $r_{reward}$ | $\beta$      |
| low  | search   | high | -3           | $1-\beta$    |
| low  | wait     | low  | $r_{wait}$   | 1            |
| low  | recharge | high | 0            | 1            |



ex 3.5
$$
\sum_{s'\in S^+}\sum_{r\in R}p(s',r|s,a)=1
$$
即将$S$改为$S^+$



ex 3.6
$$
G_t=-\gamma^{T-t} 
$$
ex 3.7

只有一个状态能够结束每一幕.最好能够设置另一个结束幕的状态.



ex 3.8

使用公式
$$
G_t\doteq R_{t+1}+\gamma G_{t+1}
$$
那么
$$
G_5=0(由定义得)\\
G_4=R_5+\gamma G_5=2\\
G_3=R_4+\gamma G_4=4\\
G_2=R_3+\gamma G_3=8\\
G_1=R_2+\gamma G_2=6\\
G_0=R_1+\gamma G_1=4
$$


ex 3.9
$$
G_1=R_2+\gamma R_3+\gamma^2 R_4+\cdots=70\\
G_0=R_1+\gamma G_1=2+0.9\times70=65
$$


ex 3.10
$$
\begin{align}
G_t=&1+\gamma+\gamma^2+\gamma^3+\cdots+\gamma^k\\
=&\sum_{k=0}^\infty\gamma^k\\
=&\frac{1\times(1-\gamma^k)}{1-\gamma}\\
=&\frac{1}{1-\gamma}(\because\gamma<1)
\end{align}
$$


ex 3.11
$$
\mathbb{E_\pi}[R_{t+1}|S_t]=\sum_{\forall r\forall s' \forall a}rp(s',r|S_t,a)\pi(a|S_t)
$$
ex 3.12
$$
v_\pi(s)=\sum_a\pi(a|s)q_\pi(s,a)
$$
ex 3.13
$$
q_\pi=\sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]
$$
ex 3.14
$$
\begin{align}
v_\pi(s)&=
0.25\times(0+0.9*2.3)+0.25\times(0+0.9*0.4)\\
&+0.25\times(0+0.9*-0.4)+0.25\times(0+0.9*0.7)\\
&=0.7
\end{align}
$$
ex 3.15
$$
G_t := G_t + \frac{c}{1-\gamma}\\
\therefore v_c = \frac{c}{1-\gamma}
$$
ex 3.16
$$
G_t:=G_T+c\frac{1-\gamma^T}{1-\gamma }
$$


ex 3.17
$$
q_\pi(s,a)=\sum_{s',r}p(s',r|s,a)(r+\gamma \sum_{a'}\pi(a'|s')q(s',a'))
$$


ex 3.18
$$
v_\pi(s)=\mathbb{E}_\pi[q_\pi(S_t,a)|S_t=s]\\
v_\pi(s)=\sum_{a}\pi(a|s)q_{\pi}(s,a)
$$
ex 3.19
$$
q_\pi(s,a)=\mathbb{E}[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s,A_t=a]\\
q_\pi(s,a)=\sum_{s',r}p(s',r|s,a)(r+\gamma v_\pi(s'))
$$
ex 3.20

![v*](D:\RL\images\ex3.20.PNG)

最优策略是在绿色区域使用推杆,其他区域使用木杆



ex 3.21

![image-20211130143657127](D:\RL\images\ex3.21.png)

与$v_{putt}$一样



ex 3.22
$$
v_{\pi_{left}}=\mathbb{E}_{\pi_{left}}[G_0|S_{top}]=\sum_{k=0}^{+\infty}\gamma^k(1+\gamma\times0)=\frac{1}{1-\gamma}=
\begin{cases}
1,\gamma=0\\
10,\gamma=0.9\\
2,\gamma=0.5
\end{cases}\\
v_{\pi_{right}}=\mathbb{E}_{\pi_{right}}[G_0|S_{top}]=\sum_{k=0}^{+\infty}\gamma^k(0+\gamma\times2)=\frac{2\gamma}{1-\gamma}=\begin{cases}
0,\gamma=0\\
18,\gamma=0.9\\
2,\gamma=0.5
\end{cases}\\
$$
所以$\gamma=0$时,最优策略为$\pi_{left}$,

$\gamma=0.9$时,最优策略为$\pi_{right}$,

$\gamma=0.5$,二者都是最优策略



ex 3.23

根据公式(3.20)
$$
q_*(s,a)=\sum_{s',r}p(s',r|s,a)[r+\gamma  \underset{a'}{\max} q_*(s',a')]
$$
可得
$$
\begin{aligned}
&q_*(high,wait)=r_{wait}+\gamma \underset{a}{\max}q_*(high,a)\\
&q_*(high,search)=r_{wait}+\alpha\gamma \underset{a}{\max}q_*(high,a)+(1-\alpha)\gamma\underset{a}{\max}q_*(low,a)\\
&q_*(low,search) = (1-\beta)(-3+\gamma\underset{a}{\max}q_*(high,a))+\beta (r_{Search}+\gamma\underset{a}{\max}q_*(low,a))\\
&q_*(low,wait) = r_{wait}+\gamma\underset{a}{\max}q_*(low,a)\\
&q_*(low,recharge) = \gamma\underset{a}{\max}q_*(high,a)
\end{aligned}
$$
ex 3.24
$$
v_*(A)=10\gamma^0+0\gamma^1+0\gamma^2+0\gamma^3+0\gamma^4+\cdots=10\times(\gamma^0+\gamma^5+\gamma^{10}+\cdots)=10\frac{1}{1-\gamma^5}=24.419
$$


ex 3.25
$$
v_*(s)=\underset{a}{\max}q*(s,a)
$$
ex 3.26
$$
q_*(s,a)=\sum_{s',r}p(s',r|s,a)(r+\gamma v_*(s'))
$$
ex 3.27
$$
\pi_*(a|s)=\frac{\mathbb{1}\{a=\underset{a}{\max}q_*(s,a)\}}{\sum_a\mathbb{1}\{a=\underset{a}{\max}q_*(s,a)\}}
$$
ex 3.28

只需将上题中的$q_*(s,a)$用ex3.26的式子替换即可



ex 3.29
$$
v_\pi(s)=\sum_a\pi(a|s)[r(s,a)+\gamma\sum_{s'}p(s'|s,a)v_\pi(s')]\\
v_*(s)=\sum_a\pi_*(a|s)[r(s,a)+\gamma\sum_{s'}p(s'|s,a)v_*(s')]\\
q_\pi(s,a)=r(s,a)+\gamma\sum_{s'}p(s'|s,a)\sum_{a'}q_\pi(a',s')\pi(a',s')\\
q_*(s,a)=r(s,a)+\gamma\sum_{s'}p(s'|s,a)\sum_{a'}q_*(a',s')\pi_*(a',s')\\
$$
