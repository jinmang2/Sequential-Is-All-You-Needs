$$A=\begin{bmatrix}
2&1&3\\
0&-2&7\\
3&4&2\\
\end{bmatrix}$$

$$\begin{array}l
r_1=\begin{bmatrix}2&\;\;\;1&3\end{bmatrix}\\
r_2=\begin{bmatrix}0&-2&7\end{bmatrix}\\
r_3=\begin{bmatrix}3&\;\;\;4&2\end{bmatrix}\\
\end{array}$$

$$r_3:=3r_3-2r_1$$

$$A=\begin{bmatrix}
2&1&3\\
0&-2&7\\
0&5&-5\\
\end{bmatrix}$$

$$r_3:=\cfrac{1}{5}r_3$$

$$A=\begin{bmatrix}
2&1&3\\
0&-2&7\\
0&1&-1\\
\end{bmatrix}$$

$$r_2:=r_2+2r_1$$

$$A=\begin{bmatrix}
2&1&3\\
0&0&5\\
0&1&-1\\
\end{bmatrix}$$

$$\begin{array}l
r_2:=\cfrac{1}{5}r_2\\
swap(r_2,r_3)
\end{array}$$

$$A=\begin{bmatrix}
2&1&3\\
0&1&-1\\
0&0&1\\
\end{bmatrix}$$

$$\begin{array}l
r_2:=r_2+r_3\\
r_1:=r_1-r_2-3r_3\\
r_1:=\cfrac{1}{2}r_1
\end{array}$$

$$A=\begin{bmatrix}
1&0&0\\
0&1&0\\
0&0&1\\
\end{bmatrix}$$
