## 11.1 Multi-variate function

$Def(1)$ **$\text{function of two variables}$**
$$z=f(x,y)\quad\text{where }\;(x,y)\in D \in \mathbb{R}\times\mathbb{R}\;and\;z \in
\{f(x,y)\in\mathbb{R}|(x,y)\in D\}$$

$Def(2)$ **$\text{Graph}$**
$$G(f)=\{(x,y,z)\in\mathbb{R}^3|z=f(x,y)\}$$

$Def(3)$ **$\text{Contour curve}$**
$$f(x,y)=k\;\text{ s.t }k\in\{f(x,y)\in\mathbb{R}|(x,y)\in D\}$$

$Def(4)$ **$\text{function of n variable}$**
$$y=f(X)\quad\text{where }\;X=(x_1,\dots,x_n)\in D \in \mathbb{R}^n\;and\;z \in
\{f(X)\in\mathbb{R}|X \in D\}$$

## 11.2 Limit and Continuous

## 11.3 Partial Derivative

## 11.4 Tangent plane and Linear approximation

## 11.5 Chain rule

## 11.6 Gradient Vector
$Def(1)$ $\text{Let }z=f(x,y),$
$$\begin{aligned}
f_x(x_0,y_0)=\lim_{h\to 0}\cfrac{f(x_0 + h,\;y_0)-f(x_0,\;y_0)}{h}\\
f_y(x_0,y_0)=\lim_{h\to 0}\cfrac{f(x_0,\;y_0+h)-f(x_0,\;y_0)}{h}\\
\end{aligned}$$

$Def(2)$ **$\text{Directional Derivative}$**
$$D_uf(x_0,y_0)=\lim_{h\to0}\cfrac{f(x_0+ha,\;y_0+hb)-f(x_0,\;y_0)}{h}\quad
\text{from }(x_0,y_0)\text{ to }\overrightarrow{(a,b)}$$

$Thm(1)$
$$f\;\text{is Differentiable on }x,y,\;\text{then }\forall
\text{unit vector }u=<a,b>,\;\exist D_uf\;s.t$$

$$D_uf(x,y)=f_x(x,y)a+f_y(x,y)b$$

$Def(3)$ **$\text{Gradient Vector}$**
$$\nabla f(x_1,\dots,x_n)=<f_1(X),\dots,f_n(X)>=
\cfrac{\partial f}{\partial x_1}i_1+ \cdots +\cfrac{\partial f}{\partial x_n}i_n$$

$Cor(1)$
$$D_uf(X)=\nabla f(X) \cdot \overrightarrow{u}$$

$Thm(1)$
$$\max_{\theta=0}\{{D_uf(x)}\}=|\nabla f(x)|\cos{\theta}$$

**$\text{Gradient Vector}$가 지니는 의미**
- gradient vector는 함수 f가 가장 빠르게 증가하는 방향을 알려준다. (Thm(1))
- 가장 가파르게 하강/상승하는 곡선을 따라 흐른다!

## 11.7 Maximum and Minimum

## 11.8 Lagrange multiplier
