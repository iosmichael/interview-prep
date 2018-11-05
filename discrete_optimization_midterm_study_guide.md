This note is a summary note for UCSD CSE 190: Discrete and Continuous Optimization
- [Course Link Here](http://algorithms.eng.ucsd.edu/lp)

# Gaussian Elimination

---

# LU Factorization

---

# Cholesky Factorization

---

# QR Factorization

QR Factorization is a factorization to decompose matrix $A$ into an orthonormal matrix $Q$ and an upper-triangular matrix $R$
$$A = QR$$
where $A$ is a $m\times n$ matrix, $Q$ is a $m \times m$ matrix, and $R$ is a $m \times n$ matrix. ($m \geq n$)

## Orthonormal Matrix
$Q$ is an orthonormal matrix if the matrix follows two properties:

- $Q^TQ = I_n$ (column vectors of $Q$ are unit vector)
- Orthonormal **does not** necessarily imply $QQ = I_n$ (This is only **true** when $Q$ is **symmetric**, which is not a property of orthonormal matrix)
- $Q$ performs **rigid transformation** to the matrix $R$
- $Q = [\vec{q_1}, \vec{q_2}, \cdots, \vec{q_n}]$ where $\vec{q_i}$ is orthogonal to $\vec{q_j}$ if $i \neq j$. This also means that $\vec{q_i}^T\vec{q_j} = \vec{0}$

### Orthonormal Projection

$$\text{Proj}_{\vec{b}}(\vec{a}) = c\vec{b}$$


### Rigid Transformation

Because $Q$ is an orthonormal matrix, $Q$ rotates the matrix space which it applies, but preserves the vectors' relative **angle** and **length**.

Suppose two vector $x, y \in R^n$, $x^Ty$ represent their angle to each other
**Angle Preservation**
$$(Qx)^T(Qy) = x^TQ^TQy = x^Ty$$
**Norm Preservation**
$$||Qx||^2_2 = (Qx)^T(Qx) = x^TQ^TQx = x^Tx = ||x||^2_2$$

## Gram-Schmit Algorithm

---

## Householder Algorithm

---
