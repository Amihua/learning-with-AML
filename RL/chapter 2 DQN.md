# chapter 2 DQN

[TOC]

![image-20220120153738852](https://gitee.com/amihua/picgo/raw/master/image-20220120153738852.png)

## DQN

最优动作价值函数用最大化消除策略$\pi$:
$$
Q_*(s_t,a_t)=\max_{\pi}Q_{\pi}(s_t,a_t)
$$
对其的理解是$Q_*$:已知$s_t$和$a_t$，无论未来采取什么策略$\pi$，回报$U_t$的期望不可能超过$Q_*$。

我们希望知道$Q_*$，因为它就像是先知一般，可以预见未来，在$t$时刻就预见$t$到$n$时刻之间的累计奖励的期望。假如我们有$Q_*$这位先知，我们就遵照按照先知的指导，最大化未来的累计奖励。然而在实践中我们不知道$Q_*$的函数表达式。对于重复性高的场景，只要玩过足够多的次数，有经验的玩家就跟先知$Q_*$一样，看到当前状态，就能准确判断出当前最优动作是什么。

### 最优动作价值函数的近似

![image-20220120154650955](https://gitee.com/amihua/picgo/raw/master/image-20220120154650955.png)![image-20220120154819274](https://gitee.com/amihua/picgo/raw/master/image-20220120154819274.png)

### DQN的梯度

在训练 DQN 的时候，需要对 DQN 关于神经网络参数$w$求梯度。
$$
\grad_w Q(s,a;w) \overset{\Delta}{=} \frac{\partial Q(s,a;w)}{\partial w}
$$
表示函数值$Q(s,a;w)$关于参数$w$的梯度。因为函数值$Q(s,a;w)$是一个实数，所以梯度的形状与 $\mathbf{w}$完全相同。如果$\mathbf{w}$是$d \times 1$的向量，那么梯度也是$d\times1$的向量。如果$\mathbf{w}$是$d_1 \times d_2$的矩阵，那么梯度也是$d_1 \times d_2$ 的矩阵。如果$\mathbf{w}$是$d_1 \times d_2 \times d_3$的张量，那么梯度也是$d_1 \times d_2 \times d_3$的张量。

## 时间差分算法（temporal difference，缩写 TD）

### 想法本质

对于一般的机器学习算法，我们通过新的数据验证，并进行梯度下降，学习并更新参数，使其更加逼近真实值。

对于序列问题，假如我们知道其中的一部分真实情况，我们可以将实际花费的部分，进行一部分更新，也可以让结果更加准确。

![image-20220120160009414](https://gitee.com/amihua/picgo/raw/master/image-20220120160009414.png)

我们对$L(w)$求$w$的梯度，即：
$$
\frac{\partial L(w)}{\partial w} = [Q-\hat y] \cdot \grad_w (Q)
$$
这样，我们就能对$w$进行梯度下降更新，
$$
w \leftarrow w - \alpha  \cdot \delta \cdot \grad_w (Q)
$$
注意再求梯度时，我们讲$\hat y$视为常数.

## 训练DQN的TD算法

我们定义折扣回报$U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} \dots$则通过递推的想法，$U_t$可以写成

$$ {1}
U_t &= R_t + \gamma (R_{t+1} + \gamma R_{t+2} \dots) \\
 &= R_t + \gamma \cdot U_{t+1} \label{U_t}
$$
同时，
$$
Q_*(s_t,a_t) = \max_{\pi}\mathbb{E}[U_t|S_t=s_t,A_t = a_t] \label{6}
$$
从而把$\eqref{U_t}$代入$\eqref{6}$中得到**最优贝尔曼方程（optimal Bellman equations）**
$$
Q_*(s_t,a_t) = \mathbb{E}_{S_{t+1} \sim p(\cdot|s_t,a_t)}[R_t + \gamma \cdot \max_{A\in \mathscr{A}} Q_* (S_{t+1},A) | S_t=s_t,A_t=a_t]
$$
![image-20220121085258393](https://gitee.com/amihua/picgo/raw/master/image-20220121085258393.png)

![image-20220121085315038](https://gitee.com/amihua/picgo/raw/master/image-20220121085315038.png)![image-20220121085327203](https://gitee.com/amihua/picgo/raw/master/image-20220121085327203.png)

贝尔曼方程的右边是个期望，我们可以对期望做蒙特卡洛近似。当智能体执行动作$a_t$之后，环境通过状态转移函数$p(s_{t+1}|s_t,a_t)$计算出新状态$p(s_{t+1})$。奖励$R_t$最多只依赖于$S_t,A_t,S_{t+1}$那么当我们观测到$s_t,a_t,s_{t+1}$时，则奖励$R_t$也被观测到，记作$r_t$。有了四元组$(s_t,a_t,r_t,s_{t+1})$,我们可以计算出
$$
r_t+\gamma \cdot \max_{a\in \mathscr{A}} Q_*(s_{t+1},a)
$$
在计算上我们可以通过蒙特卡洛采样近似：
$$
\mathbb{E}_{S_{t+1} \sim p(\cdot | s_t,a_t)} [R_t + \gamma \cdot \max_{a\in\mathscr{A}} Q_*(s_{t+1},a)]. \label{9}
$$
由公式$\eqref{U_t}$和上述的蒙特卡洛近似$\eqref{9}$可得：
$$
Q_*(s_t,a_t) = R_t + \gamma \cdot \max_{a\in\mathscr{A}} Q_*(s_{t+1},a)
$$
由于神经网络在逼近真实的**Q函数**效果格外的好，我们使用神经网络来逼近，也就是把**Q函数**替换成**DQN**
$$
Q(s_t,a_t;w)  \approx R_t + \gamma \cdot \max_{a\in\mathscr{A}} Q_*(s_{t+1},a;w) 
$$
其中$s_t,a_t$是神经网络的参数，$w$是神经网络需要学习的参数。等式左边的$\hat q_t \overset{\Delta}{=} Q(s_t,a_t;w)$是神经网络在$t$时刻做出的预测，其中没有任何事实成分。右边的**TD目标**$\hat y_t$是神经网络在$t+1$时刻做出的预测，它部分基于真实观测到的奖励$r_t$。$\hat y_t$和$\hat q_t$两者都是对最优动作价值$Q_*(s_t,a_t)$的估计，但是$\hat y_t$部分基于事实，因此比$\hat q_t$更可信。应当鼓励$\hat q_t \overset{\Delta}{=} Q(s_t,a_t;w)$接近$\hat q_t$。定义损失函数：
$$
L(w) = \frac{1}{2}[Q(s_t,a_t;w) - \hat y_t]^2
$$
假设$\hat y_t$是常数

[^1]:实际上$\hat y_t$依赖于$w$，但是我们假装$\hat y_t$是常数

，对$L$求$w$的梯度得到：
$$
\grad_w L(w) = [Q(s_t,a_t;w) - \hat y_t] \cdot \grad_w Q(s_t,a_t;w)
$$
利用梯度下降：
$$
w \leftarrow w - \alpha \cdot \grad_w L(w) = w - \alpha \cdot [Q(s_t,a_t;w) - \hat y_t] \cdot \grad_w Q(s_t,a_t;w) \label{14}
$$
因为$Q(s_t,a_t;w) - \hat y_t$事实上就是TD的误差，记$\delta_t = [Q(s_t,a_t;w) - \hat y_t]$则$\eqref{14}$可以简写成：
$$
w \leftarrow w - \alpha \cdot \delta_t \cdot \grad_w Q(s_t,a_t;w) \label{15}
$$
$\eqref{15}$就是训练 DQN 的 TD 算法。



### 怎么训练？（数据集、更新算法）

#### 收集训练数据

首先用任何策略函数$\pi$去控制智能体与环境交互，这个$\pi$就叫做**行为策略** (behavior policy)。比较常用的是$\epsilon-greedy$策略:
$$
a_t = \left\{  
             \begin{array}{**lr**}  
             arg \max_a Q(s_t,a_t;w) ,以概率(1-\epsilon) &\\
             均匀抽取\mathscr{A}的一个动作 ,以概率\epsilon &
             \end{array}  
\right.
\label{16}
$$
把智能体在一局游戏中的轨迹记作：
$$
s_1,a_1,r_1,s_2,a_2,r_2,\dots,s_{t-1},a_{t-1},r_{t-1},s_t
$$
把一条轨迹写分成若干个$(s_t,a_t,r_t,s_{t+1})$四元组，存入数组，这个数组叫做**经验回放数组(replay buffffer)**

#### 更新 **DQN** **参数** $w$

随机从经验回放数组中取出一个四元组，记作 $(s_j,a_j,r_j,s_{j+1})$。 

设**DQN** 当前的参数为 $w_{now}$，执行下面的步骤对参数做一次更新，得到新的参数 。 $w_{new}$

![image-20220121225412010](https://gitee.com/amihua/picgo/raw/master/image-20220121225412010.png)

## Q-learning 

![image-20220121231618442](https://gitee.com/amihua/picgo/raw/master/image-20220121231618442.png)

![image-20220121231634156](https://gitee.com/amihua/picgo/raw/master/image-20220121231634156.png)

## 同策略(On-policy)与异策略 (Offff-policy)

为了解释同策略和异策略，我们要从**行为策略 (behavior policy)** 和**目标策略 (target policy)**讲起。

在强化学习中，我们让智能体与环境交互，记录下观测到的状态、动作、奖励，用这些经验来学习一个策略函数。在这一过程中，控制智能体与环境交互的策略被称作**行为策略**。行为策略的作用是**收集经验 (experience)**，即观测的环境、动作、奖励。

训练的目的是得到一个策略函数，在结束训练之后，用这个策略函数来控制智能体；这个策略函数就叫做**目标策略**。在本章中，目标策略是一个确定性的策略，即用 DQN 控制智能体：
$$
a_t = arg \max_a Q(s_t,a_t;w)
$$
Q 学习算法用任意的行为策略收集$(s_t,a_t,r_t,s_{t+1})$这样的四元组，然后拿它们训练目标策略，即 DQN。

|        行为策略        |         目标策略         |
| :--------------------: | :----------------------: |
| 收集经验，得到经验数组 | 控制智能体，就是$\pi ^*$ |

行为策略和目标策略可以相同，也可以不同。**同策略**是指用**相同**的行为策略和目标策略；我们暂时还没有学到同策略。**异策略**是指用**不同**的行为策略和目标策略。DQN 是异策略.

![image-20220121232714289](https://gitee.com/amihua/picgo/raw/master/image-20220121232714289.png)

由于 DQN 是异策略，行为策略可以不同于目标策略，可以用任意的行为策略收集经验，让行为策略带有随机性的好处在于能探索更多没见过的状态。比如最常用的行为策略是$\epsilon-greedy$策略$\eqref{16}$.在实验中，初始的时候让$\epsilon$比较大（比如$\epsilon=0.5$）；在训练的过程中，让$\epsilon$逐渐衰减，在几十万步之后衰减到较小的值（比如$$\epsilon=0.01$$），此后固定住$\epsilon=0.01$。

把$(s_t,a_t,r_t,s_{t+1})$这样的四元组记录到一个数组里，在事后反复利用这些经验去更新目标策略。这个数组被称作**经验回放数组**，这种训练方式被称作**经验回放** (experience replay)。

==注意，经验回放只适用于异策略，不适用于同策略，其原因是收集经验时用的行为策略不同于想要训练出的目标策略。==

## 小结

![image-20220121233334716](https://gitee.com/amihua/picgo/raw/master/image-20220121233334716.png)
