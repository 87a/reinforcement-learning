# RL note

## 第二章 多臂赌博机

### 2.1 k臂赌博机

$k$臂赌博机问题:重复地在$k$个选项中进行选择，做出选择后可得到一定收益。目标是在某一段时间内最大化总收益的期望。

**价值**：动作的期望或平均收益。

$A_t$ : 时刻$t$选择的动作

$R_t$ : $A_t$对应的收益

$q_*(a)$ : 动作$a$的价值(收益的期望)

$Q_t(a)$ : 动作$a$在$t$时刻的价值估计

我们希望$Q_t(a)$能够接近$q_*(a)$.

**贪心**:最高估计价值的动作

开发:选择贪心的动作

试探:选择非贪心的动作



### 2.2 动作-价值方法

**动作-价值方法**:使用价值估计来选择动作

一种方式是计算实际收益的平均值$Q_t(a)$:
$$
Q_t(a)\doteq \frac{t时刻前执行动作a得到的收益总和}{t时刻前执行动作a的次数}=\frac{\sum_{i=1}^{t-1}R_i \cdot \mathbb{1}_{A_i=a}}{\sum_{i=1}^{t-1}\mathbb{1}_{A_i=a}}
$$
分母为0时,可以将$Q_t(a)$定义为某个默认值,比如0.

分母趋向无穷大时,$Q_t(a)$会收敛到$q_*(a)$,这称为**采样平均方法**.

**贪心**的选择方式可以记作:
$$
A_t\doteq \underset {a}{\arg \max}Q_t(a)
$$
一种替代策略:$\epsilon$-贪心($\epsilon$-greedy):

​		以很小的概率$\epsilon$从所有动作中等概率选择,大部分时间贪心.

其优点是时刻无限长时,每一个动作会被无限次采样,确保$Q_t(a)$会收敛到$q_*(a)$.

### 2.4 增量式实现

为简化标记,仅关心单个动作.

$R_i$:这一动作被选择$i$次的后获得的收益.

$Q_n$:被选择$n-1$次后的估计价值
$$
Q_n \doteq \frac{R_1+R_2+\cdots + R_{n-1}}{n-1}
$$
这种实现需要记录所有收益的记录,显然不适合.

**增量式公式**:

给定$Q_n$和第$n$次收益$R_n$的情况下,
$$
\begin{aligned}
Q_{n+1}&=\frac{1}{n}\sum_{i=1}^{n}R_i\\
&=\frac{1}{n}(R_n+\sum_{i=1}^{n-1}R_i)\\
&=\frac{1}{n}(R_n+(n-1)\frac{1}{n-1}\sum_{i=1}^{n-1}R_i)\\
&=\frac{1}{n}(R_n+(n-1)Q_n)\\
&=\frac{1}{n}(R_n+nQ_n-Q_n)\\
&=Q_n+\frac{1}{n}[R_n-Q_n]
\end{aligned}
$$
对于每一个新的收益,这种形式只需存储$Q_n$和$n$,计算量少.

对此我们得到一个公式的一般形式,即
$$
新估计值\leftarrow旧估计值+步长\times [目标-旧估计值]
$$
[目标-旧估计值]是估计值的**误差**.

$\alpha$或$\alpha_t(a)$表示**步长**,这是一个变量.

### 2.5 非平稳问题

 **非平稳问题**即收益概率是随时间变化的问题.

对于这种问题,我们应该赋予**近期**的收益**更高的权值**.

对于增量更新公式
$$
Q_{n+1}=Q_n+\alpha[R_n-Q_n]
$$
可以将其写成
$$
\begin{aligned}
Q_{n+1}&=\alpha R_n+(1-\alpha)Q_n\\
&=\alpha R_n+(1-\alpha)[\alpha R_{n-1}+(1-\alpha)Q_{n-1}]\\
&=\alpha R_n+(1-\alpha)\alpha R_{n-1}+(1-\alpha)^2Q_{n-1}\\
&=\alpha R_n+(1-\alpha)\alpha R_{n-1}+(1-\alpha)^2R_{n-2}+\cdots +(1-\alpha)^{n-1}\alpha R_1+(1-\alpha)^nQ_1\\
&=(1-\alpha)^nQ_1+\sum_{i=1}^n\alpha(1-\alpha)^{n-i}R_i
\end{aligned}
$$
这称为加权平均,因为(等比数列求和公式)
$$
(1-\alpha)^n+\sum_{i=1}^n\alpha(1-\alpha)^{n-i}=1
$$
$R_i$的权值即为
$$
\alpha (1-\alpha)^{n-i}
$$
权值以指数形式递减.

设$\alpha_n(a)$表示处理第$n$次选择动作$a$,选择$\alpha_n(a)=\frac{1}{n}$将会得到采样平均法,大数定律可以确保收敛到真值.

随机逼近理论中的一个结果给出了保证收敛概率为1的条件:
$$
\sum_{n=1}^\infty\alpha_n(a)=\infty \quad \text{且}\quad \sum_{n=1}^\infty\alpha^2_n(a)<\infty\
$$
第1个条件要求步长要足够大,第2个条件要求最终步长变小.显然常数步长$\alpha_n(a)=\alpha$不满足第二个条件.

### 2.6 乐观初始值

目前为止的方法都在一定程度上依赖于$Q_1(a)$的选择,这些方法是**有偏**的.

**乐观初始价值**的方法给初始值设定一个过分乐观的估计.这种乐观的估计会鼓励动作-价值方法去试探,因为无论哪种动作被选择,其收益都比估计值要小,学习器会转而尝试另一个动作.所有的动作在收敛前都被尝试过好几次.

**乐观初始价值法**在平稳问题中非常有效.

### 2.7 基于置信度上界的动作选择

**贪心**算法虽然在当前时刻看起来最好,但其他动作可能在长远上更好.

**$\epsilon$-贪心算法**会尝试选择非贪心动作,但是这种选择是盲目的,它不打会选择接近贪心或不确定性大的动作.

非贪心动作中,最好能根据他们的**潜力**来选择事实上可能是最优的动作.

一种有效的方法是根据以下公式来选择:
$$
A_t\doteq \underset{a}{\arg \max}[Q_t(a)+c\sqrt{\frac{\ln{t}}{N_t(a)}}]
$$
式中,$N_t(a)$表示动作$a$被选择的次数. $c$是一个大于0的数,控制试探的程度. 若$N_t(a)=0$,则 $a$被认为是满足最大化条件的动作.

这种方法成为基于**置信度上界**(upper confidence bound)的动作选择.参数$c$决定了置信水平.每次选 $a$时,不确定性都会减小.

### 2.8 梯度赌博机算法

我们针对每个动作考虑一个偏好函数 $H_t(a)$, $H_t(a)$越大,动作就越频繁地被选择.

偏好函数不是从"收益"的意义上提出的,一个动作对另一个动作的**相对偏好**才重要.

如果给每个动作的偏好函数加上1000,按照如下的softmax分布确定的动作概率没有任何影响.
$$
Pr\{A_t=a\}\doteq\frac{e^{H_t(a)}}{\sum^k_{b=1}e^{H_t(b)}}\doteq\pi_t(a)
$$
基于随机梯度上升的思想,提出一种自然学习方法.
$$
H_{t+1}(A_t)\doteq H_t(A_t)+\alpha(R_t-\bar R_t)(1-\pi_t(A_t))\\
H_{t+1}(a)\doteq H_t(a)-\alpha(R_t-\bar R_t)\pi_t(a), \quad a\neq A_t
$$
式中, $\alpha$是一个大于零的数,表示步长. $\bar {R}_t \in \mathbb{R}$表示时刻 $t$内所有收益的平均值.观察可得,当收益大于 $\bar {R}_t$时,未来选择动作 $A_t$的概率就会增加,反之概率就会降低.$\bar {R}_t$称为基准项.

### 2.9 关联搜索

例子:假设你遇到一个$k$臂赌博机任务,你会得到关于这个任务的编号的明显线索(不是价值),比如它的外观颜色和它的动作价值几何一一对应,动作价值集合改变,颜色也改变.那么你可以学习相关的操作**策略**,比如如果为红色,则选择1号臂;如果为绿色,则选择2号臂.

这是一个**关联搜索**的例子,既涉及试错学习去**搜索**最有动作,又将动作与情境**关联**在一起.

关联搜索任务介于$k$臂赌博机问题和完全强化学习问题之间.

### 2.10 小结

$\epsilon$-贪心方法在一小段时间内进行随机的动作选择.

UCB虽然采用确定的动作选择,却可通过每个时刻对那些具有较少样本的动作优先选择试探.

梯度赌博机不估计动作价值,而是利用偏好函数,使用softmax分布来以一种分级的、概率式的方式选择更优的动作.

在这个问题上,UCB表现得更好.



## 第三章 有限马尔科夫决策过程

### 3.1"智能体-环境"交互接口

**智能体**(agent):进行学习及实施决策的机器

**环境(environment)**:智能体之外所有与其相互作用的事物

智能体选择动作,环境对这些动作做出响应,并向智能体呈现新的状态.环境也会产生一个收益,通常是特定的数值,这就是智能体在动作选择过程中想要最大化的目标.

!["智能体-环境"交互](D:\RL\notes\3.1.png)

具体地说,在每个离散时刻 $t=0,1,2,3,\cdots$,智能体和环境都发生交互.

在每个时刻 $t$,智能体观察到所在的环境**状态**的某种特征表达,$S_t\in \mathcal{S}$,并在此基础上选择下一个动作, $A_t\in \mathcal{A}(s)$. 下一时刻,作为其动作的结果,智能体接收到一个数值化的**收益**, $R_{t+1}\in \mathcal{R} \subset \mathbb{R}$, 并进入下一个状态 $S_{t+1}$.

MDP和智能体构成的序列或**轨迹**为:
$$
S_0,A_0,R_1,S_1,A_1,R_2,S_2,A_2,R_3,\cdots
$$
**有限MDP**:状态、动作和收益的集合 $(\mathcal{S},\mathcal{A},\mathcal{R})$只有有限个元素. 随机变量 $R_t$和 $S_t$有定义明确的离散概率分布,且只依赖于前继状态和动作.也就是说, 给定前继状态和动作的值时, $s' \in \mathcal{S}$和 $r\in \mathcal{R}$, 在$t$时刻出现的概率是
$$
p(s',r|s,a)\doteq Pr\{S_t=s', R_t=r|S_{t-1}=s,A_{t-1}=a\}
$$
函数$p$定义了MDP的**动态特性**.函数$p$为每个$s$和$a$的选择指定了一个概率分布,即
$$
\sum_{s'\in\mathcal{S}}\sum_{r\in \mathcal{R}}p(s',r|s,a)=1,对于所有s\in\mathcal{S},a\in\mathcal{A}(s)
$$
在MDP中,$S_t$和$R_t$每个可能的值出现的概率只取决于前一个状态$S_{t-1}$和前一个动作$A_{t-1}$,且与更早的状态和动作无关.

从四参数动态函数$p$中,我们可以计算出其他信息,如**状态转移概率** $p:\mathcal{S}\times\mathcal{S}\times\mathcal{A}\rightarrow[0,1]$:
$$
p(s',s|a)\doteq Pr\{S_t=s'|S_{t-1}=s,A_{t-1}=a\}=\sum_{r\in\mathcal{R}}p(s',r|s,a)
$$
"状态-动作"二元组的期望收益 $r:\mathcal{S}\times\mathcal{A}\rightarrow\mathbb{R}$:
$$
r(s,a)\doteq\mathbb{E}[R_t|S_{t-1}=s,A_{t-1}=a]=\sum_{r\in\mathcal{R}}r\sum_{s'\in\mathcal{S}}p(s',r|s,a)
$$
"状态-动作-后继状态"三元组的期望收益 $r:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\rightarrow\mathbb{R}$:
$$
r(s,a,s')\doteq \mathbb{E}[R_t|S_{t-1}=s,A_{t-1}=a,S_t=s']=r\frac{p(s',r|s,a)}{p(s'|s,a)}
$$

### 3.2 目标和收益

我们的目的可以归结为:最大化智能体接受到的标量信号(称之为收益)累计和的概率期望值.

我们提供收益的方式必须使得智能体在最大化收益的同时也实现我们的目标.

收益信号只能用来传达**什么**是你要实现的目标,而不是**如何**实现这个目标.

### 3.3 回报和分幕

时刻 $t$之后的收益序列:
$$
R_{t+1},R_{t+2},R_{t+3},\cdots
$$
$G_t$:最大化期望回报
$$
G_t\doteq R_{t+1}+R_{t+2}+R_{t+3}+\cdots+R_T
$$
式中, $T$表示最终时刻.

#### 分幕式任务

这种方法在有"最终时刻"这一概念的应用中是有意义的. 这类应用中,智能体和环境的交互被分成一系列子序列(每个子序列都有最终时刻),称每个子序列为**幕(episode)**. 比如走迷宫或者一盘游戏等. 每幕结束的状态成为**终结状态**, 随后从某个标准的起始状态或起始状态的分布中的某个状态样本开始. 具有**分幕**重复特性的任务成为**分幕式任务**.

分幕式任务中,需区分非终结状态集 $\mathcal{S}$ 和包含终结与非终结所有状态的状态集 $\mathcal{S}^+$ . 终结时间 $ T$是一个随机变量.

#### 持续性任务

不能被分为单独的幕的任务成为**持续性任务**. 如连续的过程控制任务或长期运行的机器人应用.

最终时刻 $T=\infty$ ,  最大化的回报也容易趋于无穷.

因此需要引入**折扣**的概念. 根据这种方法,智能体尝试选择动作,使得它在未来收到的经过折扣系数加权后的收益总和最大化.

特别地,它选择 $A_t$ 来最大化期望**折后回报**:
$$
G_t\doteq R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots=\sum_{k=0}^\infty \gamma^kR_{t+k+1}
$$
式中, $0\le\gamma\le1$ , 称作**折扣率**.

折扣率决定未来收益的现值:未来时刻 $k$ 的收益值只有它的当前值的 $\gamma^{k-1}$倍.

若 $\gamma<1$ , 那么只要收益序列 $\{R_K\}$ 有界, $G_t$ 就是一个有限值.

若 $\gamma=0$ , 那么智能体是"目光短浅的", 即只关心眼前收益. 

随着 $\gamma$  接近1 , 折扣回报将更多地考虑未来的收益.

邻接时刻的回报可以用如下递归方式相互联系起来:
$$
\begin{aligned}
G_t&\doteq R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\gamma^3R_{t+4}\\
&=R_{t+1}+\gamma(R_{t+2}+\gamma R_{t+3}+\gamma^2 R_{t+4}+\cdots)\\
&=R_{t+1}+\gamma G_{t+1}
\end{aligned}
$$
定义 $G_T=0$ . 尽管 $G_t$ 是对无限项求和,但只要收益是一个非零常数且 $\gamma<1$ ,这个回报就是有限的. 比如收益是一个常数+1,那么回报就是:
$$
G_t=\sum_{k=0}^\infty\gamma^k=\frac{1}{1-\gamma}
$$

### 3.4 分幕式和持续性任务的统一表示法

$$
S_t\rightarrow S_{t,i}\\
A_t\rightarrow A_{t,i}\\
R_t\rightarrow R_{t,i}\\
\pi_t\rightarrow \pi_{t,i}\\
T\rightarrow T_i
$$

$i$表示幕的序号.

然而我们讨论分幕式任务的时候,几乎都在讨论某个特定的单一的幕序列 , 因此需定义一个同时适用于分幕式和持续性任务的符号. 可以将幕的终止当作一个特殊的**吸收状态**的入口 , 他只会转移到自己且产生零收益.

![状态转移图](D:\RL\notes\3.4.png)

方块表示与幕结束对应的吸收状态.

我们也可以将汇报表示为:
$$
G_t\doteq \sum_{k=t+1}^T\gamma^{k-t-1}R_k
$$

### 3.5 策略和价值函数

**价值函数**是状态(或状态与动作二元组)的函数, 用来评估当前智能体在给定状态(或状态与动作)下**有多好**.

价值函数是与特定的行为方式有关的,我们称之为策略.

**策略**是从状态到每个动作的选择概率之间的映射. $\pi(a|s)$ 就是当 $S_t=s$ 时 $A_t=a$ 的概率.

将策略 $\pi$ 下状态 $s$ 的价值函数即为 $v_\pi(s)$ . 对于MDP, 我们可以正式定义 $v_\pi$ 为
$$
v_\pi(s)\doteq\mathbb{E_\pi}[G_t|S_t=s]=\mathbb{E}\pi[\sum_{k=0}^\infty \gamma^kR_{t+k+1}|S_t=s],对于所有s\in S
$$
把函数 $v_\pi$ 成为**策略 $\pi$ 的状态价值函数** .

类似地,我们将策略 $\pi$ 下在状态 $s$ 时采用动作 $a$ 的价值即为 $q_\pi(s,a)$ 
$$
q_\pi(s,a)\doteq \mathbb{E}\pi[G_t|S_t=s,A_t=a]=\mathbb{E}_\pi[\sum_{k=0}^\infty\gamma^kR_{t+k+1}|S_t=s,A_t=a]
$$
称 $q_\pi$ 为**策略 $\pi$ 的动作价值函数**.
$$
\begin{aligned}
q_\pi(s,a) &\doteq \sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]\\
v_\pi(s)&\doteq \sum_{a}\pi(a|s)q_\pi(s,a)\\
&=\sum_{a}\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]
\end{aligned}
$$
上式称为**贝尔曼方程**. 表达了状态价值与后继状态价值的关系.

![回溯图](D:\RL\notes\3.5.png)

上图成为**回溯图**. 空心圆表示状态, 实心圆表示"状态-动作"二元组. **回溯** 操作就是将后继状态的价值信息**回传**给当前时刻的状态.

### 3.6 最优策略和最优价值函数

用 $v_*(s)$ 表示最优状态价值函数
$$
v_*(s)\doteq \underset{\pi}{\max}v_\pi(s)
$$
$q_*(s,a)$ 表示最优动作价值函数
$$
q_*(s,a)\doteq \underset{\pi}{\max}q_\pi(s,a)
$$
它们对应的策略 $\pi$ 就是**最优策略**.

可以用 $v_*$ 表示 $q_*$ :
$$
q_*(s,a)=\mathbb{E}[R_{t+1}+\gamma v_*(S_{t+1})|S_t=s,A_t=a]
$$


贝尔曼最优方程:
$$
\begin{aligned}
v_*(s)&=\underset{a\in \mathcal{A(s)}}{\max}q_{\pi{_*}}[s,a]\\
&=\underset{a}{\max}\mathbb{E}_{\pi_{*}}[G_t|S_t=s,A_t=a]\\
&=\underset{a}{\max}\mathbb{E}_{\pi_{*}}[R_{t+1}+\gamma G_{t+1}|S_t=s,A_t=a]\\
&=\underset{a}{\max}\mathbb{E}_{\pi_{*}}[R_{t+1}+\gamma v_*(S_{t+1})|S_t=s,A_t=a]\\
&=\underset{a}{\max}\sum_{s',r}p(s',r|s,a)[r+\gamma v_*(s')]\\
\end{aligned}
$$
$q_*$ 的贝尔曼最优方程
$$
\begin{aligned}
q_*(s,a)&=\mathbb{E}[R_{t+1}+\gamma \underset{a'}{\max}q_*(S_{t+1},a')|S_t=s,A_t=a]\\
&=\sum_{s',r}p(s',r|s,a)[r+\gamma \underset{a'}{\max}q_*(s',a')]
\end{aligned}
$$
对于有限MDP, $v_\pi$ 的贝尔曼最优方程有唯一解. 一旦有了 $v_*$ , 就能确定最优策略.

对于最有价值函数 $v_*$ 来说,任何**贪心**策略都是最优策略. 定义 $v_*$ 的意义就在于, 我们可以将最优的长期回报期望值转化为每个状态对应的一个当前局部量的计算.

给定 $q_*$ 的情况下, 对于任意状态 $s$ , 智能体只要找到使得 $q_*(s,a)$ 最大化的动作 $a$ 就可以了.

虽然贝尔曼最优方程给出了一个找到最优策略的方法,但现实中经常因为计算资源不够难以求解.

## 第四章 动态规划

### 4.1 策略评估(预测)

回顾第三章内容,我们有
$$
\begin{aligned}
v_\pi(s)&\doteq\mathbb{E_\pi}[G_t|S_t=s]\\
&=\mathbb{E}_\pi[R_{t+1}+\gamma G_{t+1}|S_t=s]\\
&=\mathbb{E}_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s]\\
&=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]
\end{aligned}
$$
如果环境的动态特性已知,那么上式是一个 $|\mathcal{S}|$ 个未知数以及  $|\mathcal{S}|$  个等式的联立线性方程组. 理论上这个方程可以解,但计算过程繁琐.

因此, 使用迭代法解决此问题. 考虑一个近似的价值函数序列, $v_0,v_1,v_2,\cdots$ , 从 $\mathcal{S}^+$ 映射到 $\mathbb{R}$ (实数集). 初始近似值 $v_0$ 可以任意选取(终止时刻为0). 下一轮的迭代近似使用 $v_\pi$ 的贝尔曼方程进行更新, 对于任意 $s\in \mathcal{S}$ :
$$
\begin{aligned}
v_{k+1}(s)&\doteq\mathbb{E}_\pi[R_{t+1}+\gamma v_k(S_{t+1})|S_t=s]\\
&=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_k(s')]
\end{aligned}
$$
显然, $v_k=v_\pi$是更新规则的一个不动点, 在保证 $v_\pi$ 存在的情况下, 序列 $\{v_k\}$ 在 $k\rightarrow \infty$ 时将会收敛到 $v_\pi$ .

这个算法称为**迭代策略评估**.

迭代策略评估对于每个状态 $s$ 都采用相同的操作:根据给定的策略, 得到所有可能的单步转移之后的及时收益和 $s$ 的每个后继状态的旧的价值函数, 利用这二者的期望值来更新 $s$ 的新的价值函数. 这种方法称为**期望更新**.

迭代策略评估每一轮迭代都更新一次所有状态的价值函数.

在程序中,需要使用两个数组:一个用于存储旧的价值函数 $v_k(s)$ ,一个用于存储新的价值函数 $v_{k+1}(s)$,. 也可以采用一个数组就地更新.

![策略评估伪代码](D:\RL\notes\policy evaluation.png)

### 4.2 策略改进

假设对于一个确定的策略 $\pi$ ,我们已经知道了它的价值函数 $v_\pi$ . 对于某个状态 $s$ , 我们想知道选择一个不同于给定策略的动作 $a\neq\pi$ 是否更好. 一种解决方法是在状态 $s$ 选择动作 $a$ 后, 继续遵循现有策略 $\pi$ . 这种方法的值为:
$$
\begin{aligned}
q_\pi(s,a)&\doteq \mathbb{E}[R_{t+1}+\gamma v_\pi (S_{t+1}|S_t=s,A_t=a)\\
&=\sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]
\end{aligned}
$$
关键在于这个值和 $v_\pi$ 的大小关系. 若这个值更大,则说明在状态 $s$ 选择动作 $a$ 后, 继续遵循现有策略 $\pi$ 会比始终使用策略 $\pi$ 更优. 这正是我们期望的.

上述情况是**策略改进定理**的一个特例. 一般地,若 $\pi$ 和 $\pi '$ 是两个确定的策略, 对任意 $s\in S$ ,
$$
q_\pi(s,\pi'(s))\geq v_\pi(s) \tag{1}
$$
我们便说策略 $\pi'$ 相比于策略 $\pi$一样好或者更好. 也就是说对于任意状态 $s$ ,
$$
v_{\pi'}(s)\geq v_\pi(s) \tag 2
$$
如果(1)式严格不等,则(2)式也严格不等.

证明:
$$
\begin{aligned}
v_\pi(s)&\leq q_\pi(s,\pi'(s))\\
&=\mathbb{E}[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s,A_t=a]\\
&=\mathbb{E}_{\pi'}[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s]\\
&\leq \mathbb{E}_{\pi'}[R_{t+1}+\gamma q_\pi(S_{t+1},\pi'(S_{t+1}))|S_t=s]\\
&=\mathbb{E}_{\pi'}[R_{t+1}+\gamma \mathbb{E}_{\pi'}[R_{t+2}+\gamma v_\pi(S_{t+2})|S_{t+1}]|S_t=s]\\
&= \mathbb{E}_{\pi'}[R_{t+1}+\gamma R_{t+2}+\gamma ^2v_\pi(S_{t+2})|S_t=s]\\
&\leq \mathbb{E}_{\pi'}[R_{t+1}+\gamma R_{t+2}+\gamma ^2R_{t+3}+\gamma ^3v_\pi(S_{t+3})|S_t=s]\\
&\vdots\\
& \leq \mathbb{E}_{\pi'}[R_{t+1}+\gamma R_{t+2}+\gamma ^2R_{t+3}+\gamma ^3R_{t+4}+\cdots|S_t=s]\\
&=v_{\pi'}(s)

\end{aligned}
$$
我们可以自然地延伸到所有的状态和所有可能的动作, 即在每个状态下根据 $q_\pi(s,a)$ 选择一个最优的,即考虑一个贪心策略 $\pi'$ ,满足
$$
\begin{aligned}
\pi'(s)&\doteq \underset{a}{\arg\max}q_\pi (s,a)\\
&=\underset{a}{\arg\max}\mathbb{E}[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s,A_t=a]\\
&=\underset{a}{\arg\max}\sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]
\end{aligned}
$$
$\underset{a}{\arg\max}$ 表示能够使得表达式的值最大化的 $a$ , 相等则任取一个. 这种根据原策略的价值函数执行贪心算法,构造更好策略的过程,成为**策略改进**.

### 4.3 策略迭代

一个策略 $\pi$ 产生一个更好的策略 $\pi'$ ,我们就可以通过计算 $v_{\pi'}$ 得到一个更优策略 $\pi''$ .
$$
\pi_0 \stackrel {E}{\longrightarrow}v_{\pi_0}\stackrel {I}{\longrightarrow}\pi_1\stackrel {E}{\longrightarrow}v_{\pi_1}\stackrel {I}{\longrightarrow}\pi_2\stackrel {E}{\longrightarrow}\cdots \stackrel {I}{\longrightarrow}\pi_*\stackrel {E}{\longrightarrow}v_*
$$
$\stackrel {E}{\longrightarrow}$ 表示策略评估, $\stackrel {I}{\longrightarrow}$ 表示策略改进. 有限MDP只有有限个策略,因此一定能收敛.

这种寻找最优策略的方法叫**策略迭代**

![policy iteration](D:\RL\notes\policy iteration.png)

### 4.4 价值迭代

策略迭代的缺点是每一次迭代都涉及了策略评估,我们可以提前阶段策略评估过程.

**价值迭代**:在一次遍历后即刻停止策略评估(对每个状态进行一次更新).可以将此表示为结合了策略改进和截断策略评估的简单更新公式. 对任意 $s\in \mathcal{S}$ ,
$$
\begin{aligned}
v_{k+1}(s)&\doteq \underset{a}{\max}\mathbb{E}[R_{t+1}+\gamma v_k(S_{t+1})|S_t=s,A_t=a]\\
&=\underset{a}{\max}\sum_{s',r}p(s',r|s,a)[r+\gamma v_k(s')]
\end{aligned}
$$
可以看做将贝尔曼最优方程变为一条更新规则.

![value iteration](D:\RL\notes\value iteration.png)

### 4.5 异步动态规划

之前讨论的DP方法涉及对MDP的整个状态集的操作,需要对整个状态集遍历.

异步DP算法是一类就地迭代的DP算法, 不以系统遍历状态集的形式来组织算法. 这些算法使用任意可用的状态值, 以任意顺序来更新状态值.

异步价值迭代的其中一个版本就是用价值迭代更新公式, 在每一步 $k$ 上都只就地更新一个状态 $s_k$ 的值.

### 4.6 广义策略迭代

我们用**广义策略迭代** (GPI)一词来指代让策略评估和策略改进相互作用的一般思路, 与这两个流程的粒度和其他细节无关. 几乎所有的强化学习方法都可以被描述成GPI.

从长远看,这两个流程会相互作用以找到一个联合解决方案: 最有价值函数和一个最优策略.

![GPI](D:\RL\notes\GPI.png)

## 第五章 蒙特卡洛方法

### 5.1 蒙特卡洛预测

我们知道一个状态的价值是从该状态开始的期望回报,即未来的折扣收益累积值的期望. 

随着越来越多的回报被观察到, 平均值就会收敛于期望值.

**首次访问型MC算法**用 $s$ 的所有首次访问的回报的平均值估计 $v_\pi(s)$ 

**每次访问型MC算法**用 $s$ 的所有访问的回报的平均值估计 $v_\pi(s)$ 

![First-visit MC prediction](D:\RL\notes\First-visit MC prediction.png)

每次访问型MC算法与此的区别只是无需检查 $S_t$ 在当前幕是否出现过.

#### 蒙特卡洛算法的回溯图

![monte carlo diagram](D:\RL\notes\montecarlo diagram.png)

与DP的不同,DP的回溯图显示一个状态所有可能的转移,而蒙特卡洛算法则仅仅显示在当前幕中采样到的那些转移.

DP的回溯图仅包含一步转移, 而蒙特卡洛算法则包含到这一幕结束为止的所有转移.

蒙特卡洛算法对于每个状态的估计都是独立的, 完全不依赖于对其他状态的估计, 即不使用**自举**思想.

计算一个状态的代价与状态的个数无关,使得蒙特卡洛算法适合在仅需获得一个或一个子集的状态的价值函数时适用.

蒙特卡洛相比DP的3个优势:

1. 可以从实际经历中学习
2. 可以从模拟经历中学习
3. 计算一个状态的代价与状态的个数无关

### 5.2 动作价值的蒙特卡洛估计

如果无法得到环境的模型, 那么计算**动作**的价值比计算**状态**的价值更加有用. 我们的目标是使用蒙特卡洛算法确定 $q_*$.

如果在某一幕中状态 $s$ 被访问并在这个状态采取了动作 $a$ , 就成"状态-动作"二元组 $(s,a)$ 在这一幕中被访问.

每次访问型MC算法将所有"状态-动作"二元组得到的回报的平均值作为价值函数的近似.

首次访问型MC算法将每幕第一次在这个状态下采取这个动作得到的回报的平均值作为价值函数的近似.

#### 问题

一些"状态-动作"二元组可能永远不会被访问到. 我们需要估计在一个状态中可采取的**所有**动作的价值函数, 而不仅仅是当前更偏好某个特定动作的价值函数.

这是一个如何**保持试探**的问题.

**试探性出发**:将指定的**"状态-动作"二元组**作为起点开始一幕采样, 同时保证所有"状态-动作"二元组都有非零的概率可以被选为起点.

另一种方法是,只考虑那些在每个状态下所有动作都有非零概率被选中的随机策略.

### 5.3 蒙特卡洛控制

蒙特卡洛控制的基本思想是采用广义策略迭代(GPI).

对于蒙特卡洛策略迭代, 自然可以逐幕交替进行评估与改进. 每一幕结束,使用观测到的回报进行策略评估, 然后在该幕序列访问的每一个状态上进行策略的改进.

基于试探性出发的蒙特卡洛(Monte Carlo ES (Exploring Starts))

![monte carlo ES](D:\RL\notes\monte carlo ES.png)

### 5.4 没有试探性出发假设的蒙特卡洛控制

避免试探性出发的解决方案就是智能体能持续不断的选择所有可能的动作. 有两种方法: **同轨策略(on-policy)**和**离轨策略(off-policy)**.

同轨策略:用于生成采样数据序列的策略和用于实际决策待评估和改进的策略是相同的.

离轨策略:用于生成采样数据序列的策略和用于实际决策待评估和改进的策略是不同的.

在同轨策略中, 策略一般是"软性"的, 对于任意 $s\in \mathcal{S}$ 以及 $a\in \mathcal{A}(s)$ , 都有 $\pi(a|s)>0$ , 但他们会逐渐逼近一个确定性的策略.

$\epsilon -soft$ : 对某个 $\epsilon >0$ , 所有的状态和动作都有 $\pi(a|s)\geq \frac{\epsilon}{|\mathcal{A}(s)|}$ 

$\epsilon - greedy$ : 贪心动作选中的概率为 $1-\epsilon + \frac{\epsilon}{|\mathcal{A}(s)|}$ , 非贪心动作选中的概率为 $\frac{\epsilon}{|\mathcal{A}(s)|}$ . $\epsilon - greedy$ 是$\epsilon -soft$ 的一个特例, 是最接近贪心策略的$\epsilon -soft$ 策略.

同轨策略算法的总体思想依然是GPI.我们是用首次访问型MC算法估计当前策略的动作价值函数. 由于缺乏试探性出发假设, 我们不能简单通过对方当前价值函数进行贪心优化来改进策略. 但GPI只要求逐渐**逼近**贪心策略即可.

在同轨策略中,我们仅仅改为遵循$\epsilon - greedy$ 策略.

![on-policy](D:\RL\notes\on-policy.png)

### 5.5 基于重要度采样的离轨策略

**离轨策略学习**中采用了两个策略, 一个是用来学习的**目标策略** $\pi$, 另一个是生成行动样本的**行动策略** $b$. 两个策略都固定且已知.

为了使用从 $b$ 得到的多幕样本序列去预测 $\pi$ , 要求在 $\pi$ 下能发生的动作都至少偶尔在 $b$ 下能发生, 即
$$
b(a|s)>0, \forall \pi(a|s)>0
$$


称之为**覆盖假设**. 在与 $\pi$ 不同的状态中, $b$ 必须是随机的, 目标策略 $\pi$ 则有可能是确定性的.

目标策略通常是一个确定性的贪心策略, 由动作价值函数的当前估计值所决定.

#### 重要度采样

几乎所有离轨策略方法都采用了**重要度采样**. 重要度采样是一种在给定来自其他分布的样本的条件下,估计某种分布的期望值的通用方法. 我们将其运用于离轨策略学习. 对回报值根据其轨迹在目标策略与行动策略中出现的相对概率进行加权, 这个相对概率就是重要度采样比. 

给定起始状态 $S_t$ , 后续的状态-动作轨迹 $A_t, S_{t+1}, A_{t+1},\cdots,S_T$ 在策略 $\pi$ 下发生的概率是
$$
\begin{aligned}
Pr&\{A_t,S_{t+1},A_{t+1},\cdots,S_T|S_t,A_{t:T-1}\sim\pi\}\\
&=\pi(A_t|S_t)p(S_{t+1}|S_t,A_t)\pi(A_{t+1}|S_{t+1})\cdots p(S_T|S_{T-1},A_{T-1})\\
&=\prod_{k=t}^{T-1}\pi(A_k|S_k)p(S_{k+1}|S_k,A_k)
\end{aligned}
$$
其中, $p$ 是状态转移概率函数. 因此,在目标策略和行动策略轨迹下的相对概率(重要度采样比)是
$$
\rho_{t:T-1}\doteq\frac{\prod_{k=t}^{T-1}\pi(A_k|S_k)p(S_{k+1}|S_k,A_k)}{\prod_{k=t}^{T-1}b(A_k|S_k)p(S_{k+1}|S_k,A_k)}=\prod_{k=t}^{T-1}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}
$$
重要度采样比只与两个策略和样本序列数据相关, 而与MDP的动态特性无关.

之前,我们希望得到目标策略下的期望回报, 但我们只有行动策略中的回报 $G_t$ , 这些行动策略中得到的回报期望 $\mathbb{E}[G_t|S_t=s]=v_b(s)$ 是不准确的, 所以不能用他们的平均得到 $v_\pi$ .

使用重要度采样的比例系数 $\rho_{t:T-1}$ 可以调整回报使其有正确的期望值.
$$
\mathbb{E}[\rho_{t:T-1}G_t|S_t=s]=v_\pi(s)
$$
下面给出通过观察到的一批遵循策略 $b$ 的多幕采样序列并将其回报进行平均来预测 $v_\pi(s)$ 的蒙特卡洛算法.

在这里对时刻进行编号时, 即使时刻跨越幕的边界, 编号也递增.

对于每次访问型方法, 定义所有访问过状态 $s$ 的时刻的集合为 $\mathcal{T}(s)$ .

对于首次访问型方法, $\mathcal{T}(s)$ 只包含幕内首次访问过状态 $s$ 的时刻.

$T(t)$ 表示在时刻 $t$ 后的首次终止, $G_t$ 表示在 $t$ 之后到达 $T(t)$ 时的回报值, $\{\rho_{t:T(t)-1}\}_{t\in\mathcal{T}(s)}$ 是相应的重要度采样比.

**普通重要度采样**(ordinary importance sampling):
$$
V(s)\doteq \frac{\sum_{t\in\mathcal{T}(s)}\rho_{t:T(t)-1}G_t}{|\mathcal{T}(s)|}
$$
**加权重要度采样**(weighted importance sampling):
$$
V(s)\doteq \frac{\sum_{t\in\mathcal{T}(s)}\rho_{t:T(t)-1}G_t}{\sum_{t\in\mathcal{T}(s)}\rho_{t:T(t)-1}}
$$
若分母为0, 其值也为0.

讨论在首次访问型方法下, 获得一个单幕回报后的估计值. 在加权平均估计中, 分母分子约分, 估计值等于观测到的回报值, 与重要度采样比无关. 这是一个合理的估计, 但它的期望是 $v_b(s)$ 而不是 $v_\pi(s)$ , 所以是有偏的.

而使用普通重要度采样总是无偏的, 但其值可能变得很极端.

在数学上, 两种重要度采样在首次访问型方法下的差异可以用偏差和方差来表示.

|                | 偏差                    | 方差       |
| -------------- | ----------------------- | ---------- |
| 普通重要度采样 | 无偏                    | 无界       |
| 加权重要度采样 | 有偏,偏差值逐渐收敛到零 | 能收敛到零 |

实际应用时, 偏好加权重要度采样.

### 5.6 增量式实现

对于加权重要度采样的离轨策略方法,可以采用以下的增量式算法.

![incremental](D:\RL\notes\off-policy MC prediction incremental.png)

### 5.7 离轨策略蒙特卡洛控制

要求策略是软性的.

![off-policy mc control](D:\RL\notes\off-policy MC control.png)

一个潜在的问题是,当幕中某时刻剩下的所有动作都是贪心的时候,这种方法也只会从幕的尾部开始学习.如果非贪心行为较普遍,则学习速度会慢很多, 尤其对于在很长幕中较早出现的状态.

### 5.8 *折扣敏感的重要度采样

目前为止,我们讨论的离轨策略都需要为回报计算重要度采样的比重.

考虑一种幕很长且 $\gamma$ 显著小于1的情况. 具体来说,假设幕持续100步且 $\gamma=0$ .则 $G_0=R_1$ ,但其重要度采样比会是100个因子之积,即
$$
\frac{\pi(A_0|S_0)}{b(A_0|S_0)}\frac{\pi(A_1|S_1)}{b(A_1|S_1)}\cdots\frac{\pi(A_{99}|S_{99})}{b(A_{99}|S_{99})}
$$
但实际上只需要按第一个因子衡量, 因为在得到首次收益之后,整幕回报就已经决定了. 因此提出一种思路来避免与期望更新无关的巨大方差.

这个思路的本质把折扣看作幕终止的概率,或是部分终止的**程度**,.对于任何 $\gamma\in[0,1)$ , 我们可以把回报 $G_0$ 看作在一步内, 以 $1-\gamma$ 的程度部分终止,产生的回报仅仅为首次收益 $R_1$, 然后在两步后以 $(1-\gamma)\gamma$ 的程度部分终止, 产生 $R_1+R_2$ 的回报,以此类推. 这里的部分回报称为**平价部分回报(flat partial returns)**:
$$
\bar G_{t:h}\doteq R_{t+1}+R_{t+2}+\cdots R_h,\qquad 0\leq t<h\leq T
$$
"平价"表示没有折扣, "部分"表示这些回报不会一直延续到终止, 而是在 $h$ 处停止, $h$ 被称为**视界(horizon)**. 传统的全回报 $G_t$ 可以看做上述平价部分回报的和
$$
\begin{aligned}
G_t &\doteq R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots+\gamma^{T-t-1}R_T\\
&=(1-\gamma)R_{t+1}\\
&\quad +(1-\gamma)\gamma(R_{t+1}+R_{t+2})\\
&\quad +(1-\gamma)\gamma^2(R_{t+1}+R_{t+2}+R_{t+3})\\
&\quad \vdots\\
&\quad +(1-\gamma)\gamma^{T-t-2}(R_{t+1}+R_{t+2}+\cdots +R_{T-1})\\
&\quad +\gamma^{T-t-1}(R_{t+1}+R_{t+2}+\cdots +R_T)\\
&=(1-\gamma)\sum_{h=t+1}^{T-1}\gamma^{h-t-1}\bar G_{t:h}+\gamma^{T-t-1}\bar G_{t:T}
\end{aligned}
$$
现在我们需要一种类似的截断的重要度采样比来放缩平价部分回报. 由于 $\bar G_{t:h}$ 只涉及到视界 $h$ 为止的收益, 因此我们只需要使用到 $h$ 为止的概率值, 定义如下**普通重要度采样估计器**:
$$
V(s)\doteq \frac{\sum_{t\in\mathcal{T}(s)}((1-\gamma)\sum_{h=t+1}^{T(t)-1}\gamma^{h-t-1}\rho_{t:h-1}\bar G_{t:h} + \gamma^{T(t)-t-1} \rho_{t:T(t)-1}\bar G_{t:T(t)})}{|\mathcal T(s)|}
$$
和**加权重要度采样估计器**:
$$
V(s)\doteq \frac{\sum_{t\in\mathcal{T}(s)}((1-\gamma)\sum_{h=t+1}^{T(t)-1}\gamma^{h-t-1}\rho_{t:h-1}\bar G_{t:h} + \gamma^{T(t)-t-1} \rho_{t:T(t)-1}\bar G_{t:T(t)})}{\sum_{t\in\mathcal{T}(s)}((1-\gamma)\sum_{h=t+1}^{T(t)-1}\gamma^{h-t-1}\rho_{t:h-1} + \gamma^{T(t)-t-1} \rho_{t:T(t)-1})}
$$
将这两个估计器称为**折扣敏感的重要度估计(discounting-aware importance sampling estimators)**.在 $\gamma=1$ 时没有任何影响.

### 5.9 *每次决策型重要度采样

注意到在离轨策略估计器中
$$
\begin{aligned}
\rho_{t:T-1}G_t&=\rho_{t:T-1}(R_{t+1}+\gamma R_{t+2}+\cdots+\gamma ^{T-t-1}R_T)\\
& =\rho_{t:T-1}R_{t+1}+\gamma \rho_{t:T-1}R_{t+2}+\cdots+\gamma \rho_{t:T-1}R_T
\end{aligned}
$$
上式的每个子项可以看做一个随机收益和随机重要度采样比的乘积. 例如,可以将第一个子项写为
$$
\rho_{t:T-1}R_{t+1}=\frac{\pi(A_t|S_t)}{b(A_t|S_t)}\frac{\pi(A_{t+1}|S_{t+1})}{b(A_{t+1}|S_{t+1})}\frac{\pi(A_{t+2}|S_{t+2})}{b(A_{t+2}|S_{t+2})}\cdots\frac{\pi(A_{T-1}|S_{T-1})}{b(A_{T-1}|S_{T-1})}R_{t+1}
$$
可以发现, 所有因子中,只有第一个和最后一个(即收益)是有关的,其他都是期望值为1的独立随机变量
$$
\mathbb{E}[\frac{\pi(A_k|S_k)}{b(A_k|S_k)}]\doteq \sum_a b(a|S_k)\frac{\pi(a|S_k)}{b(a|S_k)}=\sum_a\pi(a|S_k)=1
$$
因此,由于独立随机变量乘积的期望是变量期望值的乘积, 我们可以把除了第一项以外的所有比率移出期望,只剩下
$$
\mathbb E[\rho_{t:T-1}R_{t+1}]=\mathbb E[\rho_{t:t}R_{t+1}]
$$
对式中的第 $k$ 项重复这样的分析, 就会得到
$$
\mathbb E[\rho_{t:T-1}R_{t+k}]=\mathbb E[\rho_{t
:t+k-1}R_{t+k}]
$$
这样,原式的期望可以写成
$$
\mathbb E[\rho_{t:T-1}G_t]=\mathbb E[\tilde G_t]\\
\tilde G_t= \rho_{t:t}R_{t+1}+\gamma \rho_{t:t+1}R_{t+2}+\gamma^2\rho_{t:t+2}R_{t+3}+\cdots+\gamma^{T-t-1}\rho_{t:T-1}R_T
$$
把这种思想称为**每次决策型重要度采样(per-decision importance sampling)** .

借助这个思想, 对于普通重要度采样,就找到了一种替代方法来进行计算.
$$
V(s)\doteq \frac {\sum_{t\in \mathcal T(s)}\tilde G_t}{|\mathcal T(s)|}
$$

## 第六章 时序差分学习

### 6.1 时序差分预测

最简单的TD(temporal difference)方法在状态转移到 $S_{t+1}$ 并收到 $R_{t+1}$ 的收益时会立刻更新
$$
V(S_t)\leftarrow V(S_t)+\alpha [R_{t+1}+\gamma V(S_{t+1})
-V(S_t)]
$$
蒙特卡洛更新的目标是 $G_t$ , TD更新的目标是 $R_{t+1}+\gamma V(S_{t+1})$ . 这种TD方法被称为**TD(0)**, 或**单步TD**.

![TD(0)](D:\RL\notes\TD(0).png)

TD算法结合了蒙特卡洛采样方法和DP自举法.

我们将TD和蒙特卡洛更新称为**采样更新**.

在TD(0)的更新中, 括号里的数值是一种误差, 衡量的是 $S_t$ 的估计值与更好的估计 $R_{t+1}+\gamma V(S_{t+1})$ 之间的差异.

这个数值被称为**TD误差**.
$$
\delta_t\doteq R_{t+1}+\gamma V(S_{t+1})-V(S_t)
$$
每个时刻的TD误差是**当前时刻**估计的误差, 需要一个时刻步长之后才可获得.

如果价值函数数组 $V$ 在一幕内没有改变, 则蒙特卡洛误差可写为TD误差之和
$$
G_t-V(S_t)=\sum_{k=t}^{T-1}\gamma ^{k-t}\delta_k
$$
若改变,则
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
每个误差都与预测值在时序上的变化, 即预测中的**时序差分**,成正比.

### 6.2 时序差分预测方法的优势

TD方法在随机任务上通常比常量 $\alpha$ MC方法收敛的更快.

### 6.3 TD(0)的最优性

**批量更新**:反复呈现有限的经验,给定价值函数 $V$ , 访问非终止时刻的每个时刻 $t$ 时使用TD 或 $\alpha MC$ 公式计算增量, 但价值函数仅根据所有增量的和改变一次. 然后, 利用新的价值函数再次处理所有经验, 产生新的总增量,以此类推,直到收敛.

批量TD(0)方法得到的答案会在**未来**数据上产生更小的误差.

批量蒙特卡洛方法总是找出最小化训练集上均方误差的估计,而批量TD(0)总是找出完全符合马尔科夫过程模型的最大似然估计参数. 通常, 一个参数的最大似然估计是使得生成训练数据的概率最大的参数值.

马尔科夫过程模型参数的最大似然估计可以直观地从观察到的多幕序列中得到. 从 $i$ 到 $j$ 的转移概率估计值, 就是观察数据中从 $i$ 出发转移到 $j$ 的次数占从 $i$ 出发的所有转移次数的比例. 相应的期望收益则是在这些转移中观察到的收益的平均值.

据此来估计价值函数, 并且如果模型正确, 我们的估计也完全正确. 这种估计被称为**确定性等价估计**.

批量TD(0)通常收敛到的就是确定性等价估计.

### 6.4 Sarsa: 同轨策略下的时序差分控制

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha [R_{t+1}+\gamma Q(S_{t+1},A
_{t+1})-Q(S_t,A_t)]
$$

因为更新中用到了五元组 $(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})$ 中的所有元素, 所以将这算法命名为**Sarsa**.

![backup diagram](D:\RL\notes\backup diagram for Sarsa.png)

这是一种同轨策略.

![Sarsa](D:\RL\notes\Sarsa.png)

### 6.5 Q学习: 离轨策略下的时序差分控制

Q学习(Q-learning)定义为
$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma \underset{a}{\max}Q(S_{t+1},a)-Q(S_t,A_t)]
$$
待学习的动作价值函数 $Q$ 采用了对最优动作价值函数的直接近似作为学习目标, 而与用于生成智能体决策序列轨迹的行动策略无关.

![q-learning](D:\RL\notes\Q-learning.png)

### 6.6 期望Sarsa

期望Sarsa与Q学习类似, 只是在更新中把取最大值这一步换成取期望.
$$
\begin{aligned}
Q(S_t,A_t)&\leftarrow Q(S_t,A_t)+\alpha [R_{t+1}+\gamma \mathbb{E}[Q(S_{t+1},A_{t+1})|S
_{t+1}]-Q(S_t,A_t)]\\
&\leftarrow Q(S_t,A_t)+\alpha [R_{t+1}+\gamma\sum_a \pi(a|S_{t+1})Q(S_{t+1},a)-Q(S_t,A_t)]
\end{aligned}
$$
期望Sarsa可以是离轨策略控制方法, 也可以是同轨策略.可以将Q学习看做期望Sarsa的一种特例.

### 6.7 最大化偏差和双学习

**最大化偏差(maximization bias)**:最大化操作导致的正向偏差.

#### 双学习

将样本划分为两个集合,并用他们学习两个独立的对真实价值 $q(a),\forall a\in A$ 的估计 $Q_1(a)$ 和 $Q_1(a)$. 接下来使用其中一个估计来确定最大的动作, 再用另一个计算其价值估计. 这就是**双学习**的思想.

双Q学习(double Q-learning)

![double q-learning](D:\RL\notes\double q.png)

### 6.8 游戏、后位状态和其他特殊例子

传统的状态-值函数是智能体在某个状态估计有机会选择行为的回报值，但是在井字棋游戏中我们估计的是状态采取动作之后的效益。我们称之为 afterstates，相应的值函数叫做 afterstates 值函数。

传统的动作价值函数使用局面-下法二元组, 但有些情况下,局面和下法不同, 但后位状态相同.

## 第七章 n步自举法

### 7.1  n步时序差分预测

**n步时序差分** 是介于蒙特卡洛和时序差分两种方法中间的方法. 蒙塔卡洛方法根据从某一状态开始到终止状态的收益序列, 对这个状态的价值进行更新, 时序差分则只根据后面单个即时收益, 在下一个后继状态的价值估计值的基础上进行自举更新.

n步时序差分使用多于一个时刻的收益,但又不是到终止状态的所有收益.

定义**n步回报**为
$$
G_{t:t+n}\doteq R_{t+1}+\gamma R_{t+2}+ \cdots +\gamma^{n-1}R_{t+n}+\gamma^n V_{t+n-1}(S_{t+n})
$$
如果 $t+n\ge T$ , 则 $G_{t:t+n}=G_t$ .

基于n步回报的状态价值函数更新算法是
$$
V_{t+n}(S_t)\doteq V_{t+n-1}(S_t)+\alpha [G_{t:t+n}-V_{t+n-1}(S_t)], 0\le t<T
$$
对于其他 $s\neq S_t$ , 价值估计不变. 这个算法被称为**n步时序差分**(n步TD)算法.

最开始的 $n-1$ 个时刻, 价值函数不会被更新, 所以在终止时刻后还将执行对应次数的更新.

![n-step TD](D:\RL\notes\n-step TD.png)

**误差减少性质**:
$$
\underset s \max |\mathbb E_\pi [G_{t:t+n}|S_t=s]-v_\pi(s)|\leq \gamma^n \underset s \max |V_{t+n-1}(s)-v_\pi(s)|, \forall n\geq 1
$$
根据这个性质, 可证明所有的n步时序差分方法在合适的条件下都能收敛到正确的预测.

n取中间大小的值效果最好.

### 7.2 n步Sarsa

该方法的核心思想是将状态替换为"状态-动作"二元组, 然后使用 $\epsilon$ -贪心策略. 

重新定义n步方法的回报:
$$
G_{t:t+n}\doteq R_{t+1}+\gamma R_{t+2} +\cdots +\gamma ^{n-1}R_{t+n}+\gamma ^n Q_{t+n-1}(S_{t+n},A_{t+n}), n\geq 1, 0\leq t < T-n
$$
当 $t+n \geq T$ 时, $G_{t:t+n} =G_t$ . 基于此, 可以得到如下算法
$$
Q_{t+n}(S_t,A_t)\doteq Q_{t+n-1}(S_t,A_t)+\alpha [G_{t:t+n}-Q_{t+n-1}(S_t,A_t)],0\leq t<T
$$
对满足 $s\neq S_t$ 或 $a\neq A_t$ 的 $s,a$ , 有 $Q_{t+n}(s,a)=Q_{t+n-1}(s,a)$ .

![n-step Sarsa](D:\RL\notes\n-step Sarsa.png)

#### n步期望Sarsa

n步期望Sarsa与n步Sarsa的不同在于最后一个结点使用的是状态的期望近似.
$$
G_{t:t+n}\doteq R_{t+1}+\cdots + \gamma^{n-1}R_{t+n}+\gamma^n \bar V_{t+n-1}(S_{t+n}), t+n<T
$$
当 $t+n \geq T$ 时, $G_{t:t+n} =G_t$ . 其中, $\bar V_{t+n-1}(S_{t+n})$ 是状态 $s$ 的**期望近似价值** .
$$
\bar V_{t}(s)\doteq \sum_a \pi(a|s)Q_t(s,a), \forall s\in \mathcal S
$$
![n-step Sarsa backup diagram](D:\RL\notes\n-step Sarsa backup diagram.png)

### 7.3 n步离轨策略学习

与之前类似, 可以将n步离轨学习的更新写为
$$
V_{t+n}(S_t)\doteq V_{t+n-1}(S_t)+\alpha \rho_{t:t+n-1}[G_{t:t+n}-V_{t+n-1}(S_t)],0\leq t <T
$$
其中, $\rho_{t:t+n-1}$ 是重要度采样率,
$$
\rho_{t:h}\doteq \prod _{k=t}^{\min(h,T-1)}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}
$$

#### 离轨n步Sarsa

之前的n步Sarsa的更新方法, 可以完整地被如下简单的离轨策略版代替.
$$
Q_{t+n}(S_t,A_t)\doteq Q_{t+n-1}(S_t,A_t)+\alpha \rho_{t+1:t+n}[G_{t:t+n}-Q_{t+n-1}(S_t,A_t)],0\leq t<T
$$
这里的重要度采样, 起点和终点都比上一条式子要晚一步. 因为我们更新的是"状态-动作"二元组, 所以不关心这些动作有多大概率被选择.

![off-policy n-step Sarsa](D:\RL\notes\off-policy n-step Sarsa.png)

#### 离轨n步期望Sarsa

离轨n步期望Sarsa与上文的更新方法类似, 唯一的区别在于重要度采样率是 $\rho_{t+1:t+n-1}$ , 因为最后一步不需要考虑选择的概率.

### 7.4 *带控制变量的每次决策型方法

#### 普通的n步方法的回报的递归形式

普通的n步方法的回报可以写成递归形式
$$
G_{t:h}=R_{t+1}+\gamma G_{t+1:h},t<h<T
$$
其中, $G_{h:h}\doteq V_{h-1}(S_h)$ .

#### 带控制变量的形式

现在考虑遵循不同于目标策略 $\pi$ 的行动策略 $b$ 产生的影响. 所有产生的经验都必须用重要度采样率 $\rho_t=\frac{\pi(A_k|S_k)}{b(A_k|S_k)}$ 加权. 相比于直接在等式右侧加权, 有一种更好的办法, 即
$$
G_{t:h}\doteq \rho_t(R_{t+1}+\gamma G_{t+1:h})+(1-\rho_t)V_{h-1}(S_t), t<h<T
$$
其中, $G_{h:h}\doteq V_{h-1}(S_h)$ . 这个方法中, 如果 $\rho_t =0$ , 则目标会和估计值一样, 不会导致估计值收缩. $\rho_t =0$ 表示我们应当忽略样本, 所以不改变其估计值更为合理.

上式的第二项称为**控制变量** .这个控制变量不会改变更新值得期望, 因为重要度采样率的期望为1.

#### 动作价值的递归形式

对于动作价值, 终止与视界 $h$ 的**同轨策略**下的n步回报同样可以写成递归形式, 唯一的不同是 $G_{h:h}\doteq \bar V_{h-1}(S_h)$ .

使用控制变量的离轨策略的形式可以写为
$$
\begin{aligned}
G_{t:h}&\doteq R_{t+1}+\gamma (\rho_{t+1}G_{t+1:h}+\bar V_{h-1}(S_{t+1})-\rho_{t+1}Q_{h-1}(S_{t+1},A_{t+1}))\\
&=R_{t+1}+\gamma \rho_{t+1}(G_{t+1:h}-Q_{h-1}(S_{t+1},A_{t+1}))+\gamma \bar V_{h-1}(S_{t+1}),t<h\le T
\end{aligned}
$$
其中, $G_{h:h}\doteq Q_{h-1}(S_h,A_h), h<T$ ,  $G_{T-1:h}\doteq R_T, h\ge T$ .

### 7.5 不需要使用重要度采样的离轨策略学习方法:n步树回溯算法.

#### 树回溯(tree-backup)算法

![n-step tree-backup](D:\RL\notes\n-step tree-backup.png)

如图所示, 沿中心轴向下, 图中标出的是三个采样状态和收益, 以及两个采样动作. 连接在侧边的是**没有**被采样选择的动作, 对最后一个状态, 所有的动作都被认为是还没有被选择的.

n步树回溯算法中, 更新量是从树的**叶子结点**的动作价值的估计值计算出来的. 内部的动作节点(实际采取的动作)不参与回溯. 每个叶子结点的贡献会被加权, 权值与它在目标策略 $\pi$ 下出现的概率成正比. 因此除了实际被采用的动作 $A_{t+1}$ 完全不产生贡献之外, 一个一级动作 $a$ 的贡献权值为 $\pi(a|S_{t+1})$ . 它的 $\pi(A_{t+1}|S_{t+1})$ 被用来给所有二级动作加权, 所以未被选择的二级动作 $a'$ 的贡献权值为 $\pi(A_{t+1}|S_{t+1})\pi(a'|S_{t+2})$ , 每个三级动作的贡献权值为 $\pi(A_{t+1}|S_{t+1})\pi(A_{t+2}|S_{t+2})\pi(a''|S_{t+3})$ .

可以把三步树回溯看成由6个子步骤组成. "采样子步骤"是从一个动作到其后继状态的过程; "期望子步骤"是从这个状态到所有可能的动作的选择过程, 每个动作同时带上了在目标策略下的出现概率.

树回溯算法的单步回报与期望Sarsa相同, 对于 $t<T-1$, 有
$$
G_{t:t+1}\doteq R_{t+1}+\gamma \sum_a \pi(a|S_{t+1})Q_t(S_{t+1},a)
$$
对于 $t<T-2$, 两步树回溯的回报是
$$
\begin{aligned}
G_{t:t+2}&\doteq R_{t+1} +\gamma \sum _{a\neq A_{t+1}}\pi(a|S_{t+1})Q_{t+1}(S_{t+1},a)+\gamma \pi(A_{t+1}|S_{t+1})(R_{t+2}+\gamma\sum_a \pi(a|S_{t+2})Q_{t+1}(S_{t+2},a))\\
&=R_{t+1}+\gamma \sum _{a\neq A_{t+1}}\pi(a|S_{t+1})Q_{t+1}(S_{t+1},a)+\gamma \pi(A_{t+1}|S_{t+1})G_{t+1:t+2}
\end{aligned}
$$
由上式可以得到树回溯n步回报的递归形式, 对于 $t<T-1, n\ge 2$
$$
G_{t:t+n} \doteq R_{t+1}+\gamma \sum _{a\neq A_{t+1}}\pi(a|S_{t+1})Q_{t+n-1}(S_{t+1},a)+\gamma \pi(A_{t+1}|S_{t+1})G_{t+1:t+n}
$$
其中, $G_{T-1:t+n}\doteq R_T$

这个目标就可以用于n步Sarsa的动作价值更新规则.
$$
Q_{t+n}(S_t,A_t)\doteq Q_{t+n-1}(S_t,A_t)+\alpha [G_{t:t+n}-Q_{t+n-1}(S_t,A_t)],0\leq t<T
$$
![n-step tree backup pseudo](D:\RL\notes\n-step tree-backup pseudo.png)

### 7.6 *一个统一的算法: n步 $Q({\sigma})$

n步Sarsa算法中的节点转移全部基于采样得到的单独路径.

 而树回溯算法对于状态到动作的转移则是将所有可能路径分支全部展开, 没有任何采样.

n步期望Sarsa算法只对最后一个状态到动作的转移进行全部分支展开, 其他基于采样.

对上述算法的一种统一框架如图所示.

![Q sigma](D:\RL\notes\Q sigma.png)

对状态逐个决定是否要采取采样操作, 依据分布选取一个动作作为样本,或考虑所有动作的期望.

令 $\sigma_t\in [0,1]$ 表示步骤 $t$ 时的采样程度,  $\sigma=1$ 表示完整采样,  $\sigma=0$ 表示求期望不采样.

这种方法的回报为
$$
G_{t:h}\doteq R_{t+1}+\gamma (\sigma_{t+1}\rho_{t+1}+(1-\sigma_{t+1})\pi(A_{t+1}|S_{t+1}))(G_{t+1:h}-Q_{h-1}(S_{t+1},A_{t+1}))+\gamma \bar V_{h-1}(S_{t+1})
$$
递归式的终止条件为 $h<T, G_{h:h}\doteq Q_{h-1}(S_{t+1},A_{t+1})$, 或 $h=T, G_{T-1:T}\doteq R_T$ .然后使用通用版本的n步Sarsa更新公式.

![n-step off-policy Q sigma](D:\RL\notes\n-step off-policy Q sigma.png)

## 第八章 基于表格型方法的规划和学习

