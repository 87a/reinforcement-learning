ex 4.1
$$
\begin{aligned}
&q_{\pi}(11,down)=-1\\
&q_{\pi}(7,down)=-1+v_\pi(11)=-1-14=-15
\end{aligned}
$$
ex 4.2

若不变
$$
\begin{aligned}
v_\pi(15)=&-1+\sum_a\pi(a|15)\sum_{s\in\{12-15\}}v_\pi(s)\\
=&-1+\frac{1}{4}(v_\pi(12)+v_\pi(13)+v_\pi(14)+v_\pi(15))\\
=&-\frac{4}{3}+\frac{1}{3}(v_\pi(12)+v_\pi(13)+v_\pi(14))\\
=&-20
\end{aligned}
$$
若改变
$$
\begin{aligned}
v_\pi(13)=&-1+\frac{1}{4}(v_\pi(9)+v_\pi(12)+v_\pi(14)+v_\pi(15))\\
v_\pi(15)=&-1+\frac{1}{4}(v_\pi(12)+v_\pi(13)+v_\pi(14)+v_\pi(15))\\
v_\pi(15)=&\frac{12}{11}[-\frac{4}{3}+\frac{1}{3}(v_\pi(12)+v_\pi(14)+\frac{1}{4}v_\pi(9)\\
&+\frac{1}{4}v_\pi(12)+\frac{1}{4}v_\pi(14)-1)]=-20
\end{aligned}
$$
ex 4.3
$$
\begin{aligned}
q_\pi(s,a)=\sum_{s',r}p(s',r|s,a)[r+\gamma \sum_{a'}\pi(a'|s')q_\pi(s',a')]\\
q_{k+1}(s,a)=\sum_{s',r}p(s',r|s,a)[r+\gamma \sum_{a'}\pi(a'|s')q_k(s',a')]
\end{aligned}
$$
ex 4.4

设置一个循环次数上限

ex 4.5
$$
\begin{aligned}
&1.初始化\\
&随机初始化Q(s,a)和\pi(s)\\
&2.策略评估\\
&循环s\in S:\\
&\quad循环a\in A:\\
&\qquad Q(s,a)=\sum_{s',r}p(s',r|s,a)[r+\gamma Q(s',\pi(s'))]\\
&相同的结束标准\\
&3.策略改进\\
&\pi(s)=\underset{a}{\text{argmax}}Q(s,a)\\
&相同的稳定性判断
\end{aligned}
$$
ex 4.6

步骤1:

初始化$\pi(a,s)=\frac{1}{|A(s)|}$

步骤2:

$V(s)=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)(r+\gamma V(s'))$

步骤3:

改进的策略需满足$\epsilon-soft$

$\pi(a|s)=\left\{\begin{matrix}1-\epsilon|A(s)|,&if\quad a=\underset{a}{argmax}\sum_{s'r}p(s',r|s,a)(r+\gamma V(s'))\\\epsilon/|A(s)|,&\text{其他}\end{matrix}\right.$

ex 4.8

因为硬币朝上的概率$p_h=0.4$,这意味着我们最好用最少的步数结束这个实验.在赌资为50时,我们all in有40%会赢.而赌资为51时我们选择押1,因为赢了我们的赌资为52,输了则回到50,我们又有40%的概率会赢.

ex 4.10
$$
q_{k+1}=\underset{a'}{\max}\sum_{s',r}p(s',r|s,a)[r+\gamma q_k(s',a')]
$$
