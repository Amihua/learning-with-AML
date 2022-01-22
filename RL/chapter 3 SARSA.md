# chapter 3 SARSA

[TOC]

![image-20220122153804041](https://gitee.com/amihua/picgo/raw/master/image-20220122153804041.png)

## 表格形式的SARSA

![image-20220122153933304](https://gitee.com/amihua/picgo/raw/master/image-20220122153933304.png)

**Q** **学习与** **SARSA** **的对比：** Q 学习不依赖于$\pi$，因此 Q 学习属于**异策略** (offff-policy)，

可以用经验回放。而 SARSA 依赖于$\pi$，因此 SARSA 属于**同策略** (on-policy)，不能用经

验回放。

![image-20220122154146564](https://gitee.com/amihua/picgo/raw/master/image-20220122154146564.png)

**Q 学习**的目标是学到表格$\tilde Q$作为最优动作价值函数$Q_*$的近似。因为$Q_*$与$\pi$无关，所以在理想情况下，不论收集经验用的行为策略$\pi$是什么，都不影响 Q 学习得到的$Q$。因此，Q 学习属于**异策略 (offff-policy)**，允许行为策略区别于目标策略。Q 学习允许使用经验回放，可以重复利用过时的经验。

**SARSA 算法**的目标是学到表格$q$，作为动作价值函数的$Q_{\pi}$近似。与$Q_{\pi}$一个策略$\pi$相对应；用不同的策略$\pi$，对应 $Q_{\pi}$就会不同；策略$\pi$越好，$Q_{\pi}$的值越大。经验回放数组里的经验$(s_t,a_t,r_t,s_{t+1})$,是过时的行为策略$\pi_{old}$收集到的，与当前策略$\pi_{now}$及其对应的价值$Q_{\pi_{now}}$ 对应不上。想要学习$Q_{\pi_{now}}$的话，必须要用与当前策略收集到的经验，而不能用过时的$\pi_{old}$收集到的经验。这就是为什么 SARSA 不能用经验回放。

## SARSA网络（SARSA的神经网络形式）

**价值网络：** 如果状态空间$\mathscr{S}$是无限集，那么我们无法用一张表格表示，$Q_{\pi}$否则表格的行数是无穷。一种可行的方案是用一个神经网络 $q(s,a;w)$来近似$Q_{\pi}(s,a)$；理想情况下，
$$
q(s,a;w) = Q_{\pi}(s,a), \forall s \in \mathscr{S}, a \in \mathscr{A}
$$
神经网络 $q(s,a;w)$被称为**价值网络 (value network)**，其中的 $w$表示神经网络中可训练的参数。神经网络的结构是人预先设定的（比如有多少层，每一层的宽度是多少），而参数$w$需要通过智能体与环境的交互来学习。首先随机初始化，然后$w$用 SARSA 算法更新 $w$​.。==价值网络的输出是每个动作的价值。==动作空间$\mathscr{A}$中有多少种动作，则价值网络的输出就是多少维的向量，向量每个元素对应一个动作。

![image-20220122160446251](https://gitee.com/amihua/picgo/raw/master/image-20220122160446251.png)

### 训练方式

![image-20220122161140123](https://gitee.com/amihua/picgo/raw/master/image-20220122161140123.png)

## 多步目标

![image-20220122162323489](https://gitee.com/amihua/picgo/raw/master/image-20220122162323489.png)

设一局游戏的长度为$n$。根据定义，$t$时刻的回报$U_t$是$t$时刻之后的所有奖励的加权和：
$$
U_t = R_t + \gamma \cdot R_{t+1}  + \gamma^2 \cdot R_{t+2} +\dots + \gamma^{n-t}R_n
$$
同理，$t+m$时刻的回报可以写成：
$$
U_{t+m} = R_{t+m} + \gamma \cdot R_{t+m+1}  + \gamma^2 \cdot R_{t+m+2} +\dots + \gamma^{n-t-m}R_{n}
$$
则
$$
U_t = (R_t + \gamma \cdot R_{t+1}  + \gamma^2 \cdot R_{t+2} +\dots + \gamma^{m-1} R_{t+m-1} ) +  (\gamma^{m}R_{t+m} + \dots + \gamma^{n-t -m}R_{n}) & \\
= (\sum_{i=0}^{m-1} \gamma^i \cdot R_{t+i}) + \gamma^m \cdot U_{t+m}&
$$
![image-20220122163158726](https://gitee.com/amihua/picgo/raw/master/image-20220122163158726.png)

![image-20220122175003344](https://gitee.com/amihua/picgo/raw/master/image-20220122175003344.png)

## 蒙特卡洛与自举

![image-20220122233959400](https://gitee.com/amihua/picgo/raw/master/image-20220122233959400.png)

### 蒙特卡洛

![image-20220122234022750](https://gitee.com/amihua/picgo/raw/master/image-20220122234022750.png)

### 自举

在强化学习中，“自举”的意思是“用一个估算去更新同类的估算”，类似于“自己把自己给举起来”。SARSA 使用的单步 TD 目标定义为：
$$
\hat y_t = r_t + \gamma \cdot q(s_{t+1},a_{t+1};w)
$$
![image-20220122234321173](https://gitee.com/amihua/picgo/raw/master/image-20220122234321173.png)

![image-20220122234405697](https://gitee.com/amihua/picgo/raw/master/image-20220122234405697.png)

![image-20220122234418056](https://gitee.com/amihua/picgo/raw/master/image-20220122234418056.png)

## 小结

![image-20220122234457047](https://gitee.com/amihua/picgo/raw/master/image-20220122234457047.png)
