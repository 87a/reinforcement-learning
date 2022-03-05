ex 2.1
$$
0.5 + 0.5\times0.5=0.75
$$
ex 2.2
$$
k=4,Q_1(a)=0,\forall a.\\
A_1=1,R_1=-1,A_2=2,R_2=1,A_3=2,R_3=-2,A_4=2,R_4=2,A_5=3,R_5=0
$$
可以写出Q-t表 



|      | a=1  | a=2  | a=3  | a=4  |
| ---- | ---- | ---- | ---- | ---- |
| t=1  | 0    | 0    | 0    | 0    |
| t=2  | -1   | 0    | 0    | 0    |
| t=3  | -1   | 1    | 0    | 0    |
| t=4  | -1   | -0.5 | 0    | 0    |
| t=5  | -1   | 0.33 | 0    | 0    |



所以,在1,2,3时刻采取的策略不确定,4,5采取的是探索策略



ex 2.3
$$
\epsilon=0.01更好,因为这样会选择最好的动作99.1\%(0.99+0.01*0.1)
$$


ex 2.4
$$
Q_{n+1}=\prod_{i=1}^n(1-\alpha_i)Q_1+\sum\alpha_iR_i\prod_{k=i+1}^n(1-\alpha_k)
$$
ex 2.5

![average-reward](D:\RL\images\average-reward.png)



![optimal-action](D:\RL\images\optimal-action.png)



ex 2.6

在刚开始时,因为乐观初始化,导致算法会减少Q值,开始时它会觉得所有选项都是好的,在它意识到所有选项都是坏的之前他必须尝试多个选项.



ex 2.7
$$
\because \beta_n \doteq \alpha / \bar{o}_n, \bar{o}_n \doteq \bar{o}_{n-1}+\alpha(1-\bar{o}_{n-1}), \bar{o}_0 \doteq 0\\
\therefore \bar{o}_1=\alpha,\beta_1=1\\
\begin{align}
Q_{n+1}&=\beta_n[R_n-Q_n]+Q_n\\
&=\beta_nR_n+(1-\beta_n)Q_n\\
Q_n&=\beta_{n-1}R_{n-1}+(1-\beta_{n-1})Q_{n-1}\\
&\vdots\\
Q_2&=\beta_1R_1+(1-\beta_1)Q_1\\
&=R_1\\
\therefore Q_{n+1}&=\prod_{i=2}^n(1-\beta_i)R_1+\sum_{i=2}^n\beta_i\prod_{j=i}^n(1-\beta_j)R_i\\
&Q_1对后面的Q没有贡献,所以是无偏估计
\end{align}
$$
ex 2.8

开始时没有action被使用过,所以
$$
N_t(a)=0,\forall a
$$
所有动作都是最大值.前十次,所有的action都会被采用一次.而第十一次时,会选择Q值最高的动作,所以会产生一个尖峰.在这之后,agent继续遵循epison-greedy进行选择,所以会有一定减少.



ex 2.9
$$
Pr\{A_t=a\}&=\frac{e^{H_t(a)}}{e^{H_t(a)}+e^{H_t(b)}}\\
&=\frac{1}{1+e^{H_t(b)-H_t(a)}}\\
&=Sigmoid(H_t(a)-H_t(b))\\
Pr\{A_t=b\}&=\frac{e^{H_t(b)}}{e^{H_t(a)}+e^{H_t(b)}}\\
&=\frac{1}{1+e^{H_t(a)-H_t(b)}}\\
&=Sigmoid(H_t(b)-H_t(a))
$$
ex 2.10

(1)当动作1和动作2不能区分时,
$$
\mathbb{E}[R|1]=0.5,\mathbb{E}[R|2]=0.5\\
$$
他们的期望相同

(2)当动作1和动作2能区分时,
$$
\mathbb{E}[R|1,A]=0.1,\mathbb{E}[R|2,A]=0.2,\\
\mathbb{E}[R|1,b]=0.9,\mathbb{E}[R|2,B]=0.8,\\
$$
所以最优的期望值为
$$
\frac{0.9+0.2}{2}=0.55
$$
策略为情况A时选2,情况B时选1.

ex 2.11o
