---
title: 微分方程
tags: [Advanced Mathematics]
hide_table_of_contents: false
---

## 微分方程

定义：含有**未知函数**及其导函数的方程

微分方程的阶是方程中所含未知方程导函数的最高阶数。

:::note

常微分方程：未知函数是一元函数的微分方程。

线性微分方程：未知函数及其导函数之间只存在加减运算。

:::

## 微分方程的解

设 $y=\phi(x)$ 在区间 $(a,b)$ 上连续且有 n 阶导数，使得

$$
\phi^{(n)}(x)\equiv f[x,\phi(x),\phi'(x),\cdots,\phi^{(n-1)}(x)]
$$

或

$$
F[x,\phi(x),\phi'(x),\cdots,\phi^{(n)}(x)]\equiv0
$$

则称其为微分方程在区间 $(a,b)$ 上的解。

:::caution

不含**任意常数**的解被称为特解，n 阶微分方程的通解含有 n 个独立的**任意常数**，而包含常数的解即不是特解也不是通解。

:::

:::warning

注意上述所说的**任意常数**不应被理解为字面量，在公式中通常被写为大写 C.

:::

##  初始条件

对于 n 阶微分方程，初始条件（初值条件）为：

$$
\begin{aligned}
y(x_0)&=y_0\\
y'(x_0)&=y'_0\\
\vdots&\\
y^{(n-1)}(x_0)&=y^{(n-1)}_0\\
\end{aligned}
$$

:::note

一般来说，通过初始条件可以确定通解中的任意常数，进而得到一个对应的特解。

:::

## 特殊一阶微分方程及其解法

###  变量可分离

形如

$$
\frac{dy}{dx}=h(x)g(y)
$$

分离变量

$$
\frac{dy}{g(y)}=h(x)dx
$$

两边同时积分即可

### 齐次微分方程

形如

$$
\frac{dy}{dx}=f(x,y)
$$

若可以将右侧化为 $f(\frac yx)$，则说明其为齐次微分方程。

对于齐次微分方程，令 $u=\frac yx$，则 $y=ux$

$$
\frac{dy}{dx}=\frac{dux}{dx} = u + x\frac{du}{dx}=\phi(u)
$$

:::note

这里将 $y$ 用 $ux$ 替换后，使用复合函数微分求得 $u+x\frac{du}{dx}$

:::

再使用分离变量法

$$
x\frac{du}{dx}=\phi(u)-u \implies\frac{1}{\phi(u)-u}du=\frac1xdx
$$

两边同时积分可得

$$
\Phi(u) = ln|x|+ C
$$

将 $u=\frac yx$ 带回

$$
\Phi(\frac yx)=ln|x| +C
$$

:::warning

齐次微分方程和齐次线性微分方程是两种不同的概念，齐次微分方程要求将 x y 同时按比例缩放，那么方程仍然保持不变。

齐次线性微分方程是指形如 $\frac{dy}{dx}+P(x)y=Q(x), Q(x)\equiv0$（一阶）的线性微分方程，这里的齐次指的是非齐次项为 0.其特点是，若 $y=\phi(x)$ 是其一个解，则 $y=c\phi(x)$ 也是其一个解。换言之，齐次线性微分方程在解空间中具有线性组合的性质。

:::

### 全微分方程

若存在二元函数 $u(x,y)$ 使得

$$
du(x,y)=P(x,y)dx+Q(x,y)dy
$$

则称微分方程

$$
	P(x,y)dx + Q(x,y)dy = 0 \tag1
$$

为全微分方程，其通解为

$$
u(x,y)=C
$$

通过确定 C，可以求得 x 和 y 的关系。

:::note

由曲线积分的相关定理，若 $D$ 为平面上一单连通区域，$P(x,y),Q(x,y)$ 在 $D$ 上连续且有连续的一阶偏导数，则 $(1)$ 式为全微分方程的充要条件为

$$
\frac{\partial P}{\partial y}=\frac{\partial Q}{\partial x}, \ \ (x,y)\in D
$$

:::

**求解应全微分方程的原函数**

构造一个可微函数 $\Phi(x, y)$，满足

$$
\frac{\partial \Phi(x,y )}{\partial x}=P(x,y)
$$

$$
\frac{\partial \Phi(x,y )}{\partial y}=Q(x,y) \tag2
$$

:::warning

求解前先要证明这是一个全微分方程。

:::

设 $(x_0,y_0)$ 为 $D$ 上一点，有

$$
\Phi(x,y) = \int_{x_0}^xP(x,y)dx + \psi(y) \tag3
$$

其中 $\psi(y)$ 为待定函数，同时对上式两边对 y 求偏微分

$$
\begin{aligned}
\frac{\partial\Phi(x,y)}{\partial y}
&=\int_{x_0}^x\frac{\partial P(x,y)dx}{\partial y} + \psi'(y)\\
&=\int_{x_0}^x\frac{\partial Q(x,y)dx}{\partial x}+\psi'(y)\\
&=Q(x, y) - Q(x_0, y)+\psi'(y)
\end{aligned}
$$

再由 $(2)$ 式可得

$$
\psi'(y) = Q(x_0,y)\implies\psi(y)=\int_{y_0}^y Q(x_0,y)dy
$$

带入 $(3)$ 式

$$
\Phi(x,y) = \int_{x_0}^xP(x,y)dx + \int_{y_0}^y Q(x_0,y)dy
$$

同理可以推出

$$
\Phi(x,y) = \int_{x_0}^xP(x,y_0)dx + \int_{y_0}^y Q(x,y)dy
$$

一些二元凑积分

$$
\begin{aligned}
ydx+xdy&=d(xy)\\
\frac{ydx-xdy}{y^2}&=d(\frac xy)\\
\frac{-ydx+xdy}{x^2}&=d(\frac yx)\\
\frac{ydx-xdy}{xy}&=d(\ln|\frac xy|)\\
\frac{ydx-xdy}{x^2+y^2}&=d(arctan\frac xy)\\
\frac{ydx-xdy}{x^2-y^2}&=\frac 12d(ln|\frac{x-y}{x+y}|)

\end{aligned}
$$

**转化为全微分方程求解**

当遇到 $\frac{\partial P}{\partial y}\neq\frac{\partial Q}{\partial x}$ 的情况，可以考虑将其转化为全微分方程的形式，这时我们希望找到一个在原区间上连续可微的非零函数 $u(x,y)$，使得 $uPdx+uQdy=0$ 为全微分方程，则 $u(x,y)$ 为积分因子。

:::note

积分因子不止只有一个

:::

积分因子的求法略

### 一阶线性微分方程

形如

$$
y'+P(x)y=Q(x)
$$

的微分方程被称为一阶线性微分方程

:::note

Q(x) 被称为自由项 (因为与 $y$ 及其导函数无关)，若 $Q(x)\equiv0$ 则其为一阶齐次线性微分方程。

一阶指的是方程中关于 $y$ 的导函数是一阶，线性则是指方程简化后每一项关于 $y$ 和 $y'$ 的指数都是 1.

:::

通解为

$$
y = e^{-\int P(x)dx}\bigg[\int Q(x)e^{\int P(x)dx}dx+C \bigg]
$$

**求解**

首先求解一阶齐次线性微分方程的通解

$$
\frac{dy}{dx}+P(x)y=0\implies \frac{dy}{y}=-P(x)dx
$$

两边积分可得

$$
\ln y=-\int P(x)dx+C\implies y = Ce^{-\int P(x)dx}
$$

:::note

如果我们知道了一阶齐次线性微分方程的两个特解 $Cy_1(x)$ 和 $Cy_2(x)$，那么其通解可以表示为 $C(y_1(x)-y_2(x))$.由通解的形式可以知道，特解中 $e^{-\int P(x)dx}$ 一致，区别在于前面的系数，即不同特解由不同的系数，且系数不相等，因此可以使用两个特解相减而不是相加表示通解。

:::

一阶线性微分方程的求解一般使用常数变易法，在求出一阶齐次微分方程的通解之后，改写原式为

$$
\frac{dy}y=(-P(x)+\frac{Q(x)}y)dx
$$

两边积分

$$
\begin{aligned}
\ln y &= -\int P(x)dx+\int \frac{Q(x)}{y}dx+\ln C_1\\
y &= C_1e^{\int\frac{Q(x)}ydx}\cdot e^{-\int P(x)dx}
\end{aligned}
$$

由于 $\int \frac{Q(x)}ydx$ 中含有未知函数，不方便求解，但其结果仍然是关于 x 的函数，则

$$
C(x)=C_1e^{\int\frac{Q(x)}ydx}\implies y=C(x)e^{-\int P(x)dx} \tag4
$$

再带入原方程

$$
\begin{aligned}
C'(x)e^{-\int P(x)dx}-C(x)P(x)e^{-\int P(x)dx}+P(x)C(x)e^{-\int P(x)dx}&=Q(x)\\
C'(x)e^{-\int P(x)dx}=Q(x)\\
C'(x)=Q(x)e^{\int P(x)dx}
\end{aligned}
$$

两边积分

$$
C(x) = \int Q(x)e^{\int P(x)dx} +C
$$

将 $C(x)$ 带入 $(4)$，可得一阶线性微分方程的通解为

$$
y = e^{-\int P(x)dx}\bigg[\int Q(x)e^{\int P(x)dx}dx+C \bigg]
$$

:::note

由通解公式可以看出，一阶线性微分方程的通解由一个齐次线性微分方程的通解和一个非齐次线性微分方程的特解组成。

:::

### 伯努利方程

形如

$$
y'+p(x)y=q(x)y^n
$$

的微分方程被称为伯努利方程，原式可化为

$$
y^{-n}\frac{dy}{dx}+p(x)y^{1-n}=q(x)
$$

令 $z=y^{1-n}$

$$
\frac1{1-n}\cdot\frac{dz}{dx}+p(x)z=q(x)
$$

这样就转换为一阶线性微分方程的求解问题了

## 二阶及高阶线性微分方程

**定义**

$$
y^{(n)}+a_1(x)y^{(n-1)}+\cdots+a_{n-1}(x)y^{(1)}+a_n(x)y=f(x) \tag5
$$

系数 $a_i(x)$ 为已知函数，$f(x)$ 为自由项，若 $f(x)\equiv 0$，则称其为齐次线性微分方程

$$
y^{(n)}+a_1(x)y^{(n-1)}+\cdots+a_{n-1}(x)y^{(1)}+a_n(x)y=0 \tag6
$$

:::note

$a_i(x)\ f(x)$ 均在某区间内连续。

:::

**线性相关和线性无关**

设区间 $(a,b)$ 存在 $m$ 个函数 $y_1(x),\cdots,y_m(x)$，若存在不全为 $0$ 的 $m$ 个常数 $k_1,\cdots,k_m$，使得

$$
\sum_i^mk_iy_i(x)\equiv0
$$

成立，则称这 m 个函数在该区间内**线性相关**，否则则为**线性无关**。

:::note

即要使上式成立，只能推出 $k$ 全为 $0$，则为线性无关。（和线性代数里的一样）

:::

### 线性微分方程解的性质

设 $y^*(x)$ 为式 $(5)$ 的一个解，$Y(x)$ 为对应式 $(6)$ 的**一个解**，则 $y=Y(x)+y^*(x)$ 为式 $(5)$ 的解

:::note

这里是之前所说的一阶线性微分方式由一阶齐次线性微分方程的通解和一个一阶线性微分方程的特解推广而来。

:::

设 $y^*_1(x),y^*_2(x)$ 是式 $(5)$ 的两个解，则其作差为对应式 $(6)$ 的解。

**齐次线性方程解的叠加**

齐次线性方程的任意个解的线性组合仍然是其解。

**齐次线性方程的通解结构**

齐次线性方程的任意个线性无关的解的线性组合是其通解。

**非齐次线性方程的通解结构**

设 $y^*(x)$ 为式 $(5)$ 的一个解，$Y(x)$ 为对应式 $(6)$ 的**通解**，则 $y=Y(x)+y^*(x)$ 为式 $(5)$ 的解。

**非齐次特解推出非齐次通解**

若有三个非齐次的互不相同特解 $y_{1},y_{2},y_{3}$ ，则对应通解可以为 $c_{1}(y_{2}-y_{1})+c_{2}(y_{3}-y_{1})+y_{3}$，特解之差对应齐次方程的解，由于三个解不相同，因此其二者之差也都线性无关，则其线性组合是对应齐次方程的通解。

而要求非齐次方程的通解，需要由齐次方程的通解加上一个非齐次的特解即可。

**自由项 $f(x)=\sum_1^n f_i(x)$ 解的叠加原理**

$y^*_i(x)$ 为

$$
y^{(n)}+a_1(x)y^{(n-1)}+\cdots+a_{n-1}(x)y^{(1)}+a_n(x)y=f(x)
$$

的解，则 $\sum_1^n y^*_i$ 为

$$
y^{(n)}+a_1(x)y^{(n-1)}+\cdots+a_{n-1}(x)y^{(1)}+a_n(x)y=f(x)
$$

### 二阶常系数线性齐次方程的通解求法及公式

二阶常系数线程齐次方程写作

$$
y''+py'+qy=0
$$

其中 $q,p$ 为常数，方程

$$
r^2+pr+q=0
$$

称为其**特征方程**，它的根 $r$ 成为成为**特征根**，通解如下：

- 两实根不相等：$y=C_1e^{r_1x}+C_2e^{r_2x}$
- 两实根相等：$y=(C_1+C_2x)e^{r_1x}$
- 一对共轭复根 $\alpha\pm\beta i, \beta>0$：$y=e^{\alpha x}(C_1\cos\beta x+C_2\sin\beta x)$

### 特殊自由项的二阶常系数线性微分方程的解法

**类型一**

形如 $y''+py'+qy=P_{m}(x)e^{ax}$ 的二阶线性非齐次微分方程，其中 $P_{m}(x)$ 为已知多项式，$a$ 为已知常数。

1. 先写出对应齐次方程的通解 $Y(X)$.
2. 求该微分方程的特解 $y^*(x)=x^{k}Q_{m}(x)e^{ax}$，其中 $Q_{m}(x)$ 系数待定，$k$ 的值依据 $a$ 的值决定，$a$ 是几重特征根，$k$ 就是几，若 $a$ 不是特征根，则 $k$ 为 0.

**类型二**

形如 $y''+py'+qy=P_{m}(x)e^{ax}\cos bx$ 或 $y''+py'+qy=P_{m}(x)e^{ax}\sin bx$ .

1. 先写出对应齐次方程的通解 $Y(X)$.
2. 求该微分方程的通解 $y^*(x)=x^{k}[R_{m}(x)e^{ax}\cos bx+S_{m}(x)e^{ax}\sin bx]$，当 $a+ib$ 是单重特征根时，$k$ 为 1，不是特征根则为 0.

### 可降价方程的解法

$y''=f(x)$ 型，做两次积分可以得到到通解

$$
y=\int \left( \int f(x) \, dx  \right) \, dx +C_1x+C_{2}
$$

$y''=f(x,y')$ 型（没有 $y$），令 $p=y',y''=\frac{dp}{dx}$，则原式化为

$$
\frac{dp}{dx} = f(x,p)
$$

则通解为

$$
y = \int \phi(x,C_{1}) \, dx +C_{2}
$$

$y''=f(y,y')$ 型（缺 $x$），令 $p=y'$，$y''=\frac{dp}{dx}=\frac{dp}{dy} \frac{dy}{dx}=p \frac{dp}{dy}$，则原式化为

$$
p \frac{dp}{dy}=f(y,p)
$$

**若能解出**

$$
p=\phi(y,C_{1})
$$

则由 $p=\frac{dy}{dx}$

$$
\int \frac{dy}{\phi(y, C_{1})}=\int  \, dx +C_{2}=x+C_{2}
$$

:::note

后两种方法都是令 $p=y'$，但一个缺 $x$ 一个缺 $y$，因此后者需要将 $\frac{dy}{dx}$ 代掉。

:::

## 欧拉方程的解法

欧拉方程

$$
x^{2} \frac{d^{2}y}{dx^{2}}+a_{1}x \frac{dy}{dx}+a_{2}y = f(x)
$$

:::info

$x^{2}y''+a_{1}xy'+a_{2}y=f(x)$，即未知函数的导函数之前都有关于 $x$ 的函数。

:::

若 $x>0$，令 $x=e^t$，$x<0$ 则令 $x=e^{-t}$

$$
\begin{align*}

\frac{dy}{dx}&=\frac{1}x{}dx \\
\frac{d^{2}y}{dx^{2}}&=-\frac{1}{x^{2}} \frac{dy}{dt}+ \frac{1}{x^{2}} \frac{d^{2}y}{dt^{2}}

\end{align*}
$$

方程化为

$$
\frac{d^{2}y}{dt^{2}}+(a_{1}-1) \frac{dy}{dt}+a_{2} y=f(e^t)
$$

这是一个常系数线性微分方程，求解即可。

## 差分方程

### 概念

$y_{t}$ 的一阶差分为 $\Delta y_{t}=y_{t+1}-y_{t}$，$\Delta$ 的差分被称为 $y_{t}$ 的二阶差分，记为 $\Delta^{2}y_{t}=\Delta y_{t+1}-\Delta y_{t}=y_{t+2}-2y_{t+1}+y_{t}$ .

### 一阶常系数线性差分方程

形如 $y_{t+1}-py_{t}=f(t), t=0,1,\cdots$ 的方程即为一阶常系数线性差分方程，$p$ 为非零常数，$f(t)$ 为已知函数。$y_(t+1)-py_{t}=0$ 为其对应的齐次方程。

通解为

$$
y_{t}=cp^{t}+y^*_{t}
$$

其中 $y^*_{t}$ 为特解，若 $f(t)=(A_{0}t^n+A_{1}t^{n-1}+\cdots+A_{n})b^t$ ，则待定解 $y_{t}^*$ 具有一下形式：

$$
y_{t}^*=t^s(B_{0}t^n+B_{1}t^{n-1}+\cdots+B_{n})b^t
$$

当 $p\neq b$ 时，$s=0$，否则 $s=1$.
