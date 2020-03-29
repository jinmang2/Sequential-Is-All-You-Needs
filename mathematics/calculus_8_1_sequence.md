# 8.1 Sequence
$\{a_n\}_{n=1}^{\infty}$

ex) fibonacci sequence, natural language, time series

```python
def momoize(func):
    tempMemo = dict()
    def wrapped(n):
        if n in tempMemo:
            return tempMemo[n]
        result = func(n)
        tempMemo[n] = result
        return result
    return wrapped

@memoize
def fib(n):
    if n in (1, 2):
        return 1
    return fib2(n-1) + fib2(n-2)

%timeit fib2(256)
>>> 113 ns ± 0.694 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
```

$Def(1)$ $L$이 존재하면 수열 $a_n$은 수렴, 아니면 발산
$$\lim_{n\to\infty}{a_n}=L$$

$Def(2)$ Given $\epsilon>0$, $\exist N>0$ s.t $|a_n-L|<\epsilon\;\text{for each}\;n>N$, then
$$\lim_{n\to\infty}{a_n}=L$$

$Thm(1)$ If $\lim_{x\to\infty}f(x)=L$ and $f(n)=a_n\text{ for each n s.t }n\in\mathbb{N},$then
$$\lim_{n\to\infty}{a_n}=L$$

$Def(3)$ $\forall M>0$, $\exists N>0$ s.t $a_n>M,\;\forall n>N$ then
$$\lim_{n\to\infty}{a_n}=\infty$$

$Thm(2)$ 수열에 대한 극한 법칙

$$\begin{aligned}
& \lim_{n\to\infty}(a_n\pm b_n)=\lim_{n\to\infty}a_n\pm\lim_{n\to\infty}b_n\\
& \lim_{n\to\infty}ca_n=c\lim_{n\to\infty}a_n\\
& \lim_{n\to\infty}(a_n \cdot b_n)=\lim_{n\to\infty}a_n \cdot \lim_{n\to\infty}b_n\\
& \lim_{n\to\infty}\cfrac{a_n}{b_n}=\cfrac{\lim_{n\to\infty}a_n}{\lim_{n\to\infty}b_n}\\
& \lim_{n\to\infty}{a_n}^p={\{\lim_{n\to\infty}a_n\}}^p\quad\text{where }p>0,a_n>0\\
\end{aligned}$$


$Thm(3)$ Squeeze theorem of Sequence

$for\;n \geq n_0\;,\;a_n \leq b_n \leq c_n$
$$\lim_{n\to\infty}a_n=\lim_{n\to\infty}c_n=L\,\Rightarrow\,\lim_{n\to\infty}b_n=L$$

$Thm(4)$
$$\lim_{n\to\infty}|a_n|=0\;\Rightarrow\;\lim_{n\to\infty}a_n=0$$

$Thm(5)$
$$\lim_{n\to\infty}r^n=\begin{cases}0, & -1<r<1\\1, & r=1\end{cases}$$

$Def(4)$ $monotone$
$$\begin{aligned}
& \forall n \geq 1,\;a_n \leq a_{n+1}\;\Rightarrow\;a_n: \text{Increasing sequence}\\
& \forall n \geq 1,\;a_n \geq a_{n+1}\;\Rightarrow\;a_n: \text{Decreasing sequence}\\
& if\; a_n \neq a_{n+1},\;then\;\text{Add suffix }"Strictly"\\
& if\; a_n\text{ is increasing or decreasing, then }a_n:\text{monotone}
\end{aligned}$$


$Def(5)$ $Bounded$
$$\begin{aligned}
&\forall n \geq 1,\;\exist M \text{ s.t } a_n \leq M\;\Rightarrow\;{\{a_n\}}\text{ is bounded above}\\
&\forall n \geq 1,\;\exist M \text{ s.t } a_n \geq M\;\Rightarrow\;{\{a_n\}}\text{ is bounded below}\\
&if\;a_n\text{ is bounded above and below, then }a_n:bounded
\end{aligned}$$

$Thm(6)$ $monotonic\;sequence\;theorem$
$$\text{All bounded monotone sequences converge. }\because\text{completness theorem of }\mathbb{R}$$
