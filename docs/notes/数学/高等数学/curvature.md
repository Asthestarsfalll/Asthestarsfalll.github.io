---
title: 曲率
tags: [Advanced Mathematics]
hide_table_of_contents: false
---

## 定义

曲线的曲率（$curvature$）就是针对曲线上某个点的切线方向角对弧长的转动率，通过微分来定义，表明曲线偏离直线的程度。数学上表明曲线在某一点的弯曲程度的数值。

曲率的一种等价定义是，弧的切线所转过的角度和弧长之比的绝对值。

## 计算公式

曲线 $y=f(x)$ 具有二阶导数，在点 $M(x, f(x))$，曲线切线的斜率为 $\tan\alpha = y'$，两边同时对 x 求导，可得：

$$
sec^2\alpha\times\frac{d\alpha}{dt}=y''
$$

则：

$$
\frac{d\alpha}{dx}=\frac{y''}{1+\tan^2\alpha}=\frac{y''}{1+y'^2}
$$

$$
d\alpha=\frac{y''}{1+y'^2} {dx}
$$

而弧长 $ds$ 的微分可以近似为一个直角三角形的斜边，其斜边斜率为 $y'$，结果为：

$$
ds = \sqrt{1 + y'^2}dx
$$

则 M 点曲率为：

$$
K = \frac{d\alpha}{ds}=\frac{|y''|}{(1 + y'^2)^{\frac32}}
$$
