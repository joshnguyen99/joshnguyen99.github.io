---
title: 'A Simple, but Detailed, Example of Backpropagation'
date: 2022-10-26
permalink: /posts/backprop
toc: true
---

In this post, we will go through an exercise involving backpropagation for a fully connected feed-forward neural network. Though simple, I observe that a lot of "Introduction to Machine Learning" courses don't tend to explain this example thoroughly enough. In fact, a common way students are taught about optimizing a neural network is that the gradients can be calculated using an algorithm called *backpropagation* (a.k.a. the chain rule), and the parameters are updated using gradient descent. However, the chain rule, apart from its formula, is typically swept under the rug and replaced by a "black box" operation called *autograd*.

I think being able to implement backpropagation, at least in the simplest case, is quite important for its conceptual understanding. Hopefully this will benefit the students who stumble upon this page after a while of searching for "How to implement backprop."

## The Exercise

Below is a simple fully connected neural network.

<p align="left">
    <img src="/files/backprop_nn_example.svg" title="Simple neural network" width="1000px">
</p>

Let's decompose this architecture:
- The first layer has 5 neurons. This network accepts inputs that are 5-dimensional.
- The final layer has 1 neuron. It represents the loss function, which is a scalar.
- The second-last layer, which has 12 neurons, is actually typically called the last layer. If this layer is followed by softmax, you can think of this network as a 12-class classifier.
- There are two hidden layers, one with 10 neurons and the other with 4.

Here's the computation in a forward pass through this network:
1. Start with the input, which is 5-dimensional.
2. Compute the first hidden output:
- Apply a linear transformation: $t_0 = W_0 x$
- Apply a non-linear activation: $z_0 = \tanh(t_0)$, where we $\tanh$ to every element of $t_0$
1. Compute the second hidden output:
- Apply a linear transformation: $t_1 = W_1 z_0$
- Apply a non-linear activation: $z_1 = \sigma(t_1)$, element-wise as well
1. Compute the second-last layer (classification output)
- Apply a linear transformation: $t_2 = W_2 z_1$
- No activation: $z_2 = \text{Id}(t_2)$
1. Compute the loss
- $\ell = \frac{1}{2} \lVert z_2 \rVert^2 = \frac{1}{2} \sum_{i} [z_2]_i^2$

Note that the dimensions of $W_0$, $W_1$ and $W_2$ are $10 \times 5$, $4 \times 10$ and $12 \times 4$, respectively.

There is nothing special about choosing $\tanh$ and $\sigma$ (sigmoid) as the activation functions in steps 2 and 3; we can simply choose others such as ReLU. Likewise, we can use an activation function in Step 4 as well.

Now our job is to find the gradient of $\ell$ with respect to the model parameters, that is, $\nabla_{W_0}\ell, \nabla_{W_1} \ell$ and $\nabla_{W_2}\ell$.

First, let's define our network in `numpy`. To make things a bit easier, we will define a few resuable classes.

### Tensor

The first class we define is a tensor. It is basically a `numpy` array with a gradient, which is an array of the same size storing its gradient. The array is stored in `.data` and its gradient in `.grad`.

```py
class Tensor:
    def __init__(self, arr, name=None):
        self.data = arr
        self.grad = None
        # Optionally store the name of this tensor
        self.name = name
```

### Activation functions

Activation functions are functions that will be applied element-wise to tensors. For example, $z_0 = \tanh(t_0)$ means that $z_0$ and $t_0$ have the same dimensions, and every element in $z_0$ is the hyperbolic tangent transformation of the corresponding element in $t_0$.

We will have a base class called `Activation`, which implements two methods:
- `__call__` will be apply the function to an input.
- `grad` will apply the gradient function to an input.

```py
class Activation:
    def __call__(self, x):
        pass
    def grad(self, x):
        pass
```

Let's implement the $\tanh$ activation function. We can simply use `np.tanh` for the forward pass. The derivative of this function is

$$
\begin{align*}
\tanh'(x) = 1 - \tanh^2(x).
\end{align*}
$$

```py
class Tanh(Activation):
    def __call__(self, x):
        return np.tanh(x)
    def grad(self, x):
        return 1 - np.tanh(x) ** 2
```

Similarly, we can implement the sigmoid function based on its formulas:

$$
\begin{align*}
\sigma(x) &= \frac{1}{1 + e^{-x}}\\
\sigma'(x) &= \sigma(x) (1 - \sigma(x)).
\end{align*}
$$

```py
class Sigmoid(Activation):
    def __call__(self, x):
        return np.exp(x) / (1 + np.exp(x))
    def grad(self, x):
        sx = self(x)
        return sx * (1 - sx)
```

Another function we used above is the identity function, which simply returns the input. Its derivative is $1$.

```py
class Identity(Activation):
    def __call__(self, x):
        return x
    def grad(self, x):
        return np.ones_like(x)
```

### Loss function

A loss function takes an input vector and returns a scalar (number). We will also implement the `grad` method for this function, `grad` should return a vector that is of the same shape as the input.

The example loss function above is simply (half) squared norm, which simply squares every element of the input, sums them together, and divides the result by two. Calculus tells us that the gradient of such a function is the input itself.

```py
class HalfSumSq:
    def __call__(self, x):
        return 0.5 * np.sum(x ** 2)
    def grad(self, x):
        return x
```

## Network and Forward Propagation

Now we are ready to put things together and create our neural net. For the sake of simplicity, we will only define one additional method for our class, which is `loss_and_grad`. It will (1) take an input x and perform a forward pass to get the loss, and (2) perform a backward pass to calculate the gradient of the loss with respect to its parameters.

As I have explained the forward pass above, we are able define the most part of our network.

```py
import numpy as np
class Net:
    def __init__(self):
        # Weight matrices. We will initialize them randomly
        self.weights = [Tensor(np.random.randn(output_dim, input_dim))
                        for input_dim, output_dim in [(5, 10), (10, 4), (4, 12)]]
        
        # Register t_0, t_1,... The default value (np.zeros) doesn't matter, as we
        # populate them in the forward pass later.
        self.linear_outputs = [Tensor(np.zeros(dim, dtype=float)) for dim in (10, 4, 12)]
        
        # Register z_0, z_1,... similarly
        self.nonlinear_outputs = [Tensor(np.zeros(dim, dtype=float)) for dim in (10, 4, 12)]
        
        # Activation and loss functions
        self.activations = [Tanh(), Sigmoid(), Identity()]
        self.loss = HalfSumSq()

    def loss_and_grad(self, x):
        curr_output = Tensor(x)
        # Forward prop
        for i in range(len(self.nonlinear_outputs)):
            # Linear transformation
            self.linear_outputs[i].data = self.weights[i].data @ curr_output.data
            curr_output = self.linear_outputs[i]
            
            # Activation function
            self.nonlinear_outputs[i].data = self.activations[i](curr_output.data)
            curr_output = self.nonlinear_outputs[i]

        # Loss function
        l = self.loss(curr_output.data)
        
        # We will implement backprop later
        # TODO: backprop
        
        return l
```

## Backpropagation

The forward propagation above creates a *computation graph*, which shows us the flow of signals from input to output. To find the gradients, we need to traverse this graph *backwards*, that is, from output to input, hence the name.

Recall that this is an application of the chain rule in multivariate calculus. Suppose we have a scalar function $h(v) = (f \circ g)(v) = f(g(v))$. To find the gradient of $h$ with respect to $v$, we follow the chain rule
$$
\begin{align*}
J_{h}(v) = J_{f \circ g} (v) = J_{f}(g(v)) J_{g}(v),
\end{align*}
$$
where $J$ denotes the *Jacobian*, which is a matrix of partial derivatives. Since $h$ is a scalar function, $J_{h}(v)$ is a row vector. Transposing it will give us the gradient with respect to $v$.

The computation of $h$ looks familiar. First, we have an input $v$. Then transform $v$ to another (vector of scalar) value $g(v)$. Then use $g(v)$ as the input to $f$. The chain rules says that to find the gradient for $v$, we first need to go backwards: differentiate $f$ with respect to $g(v)$ first, then differentiate g with respect to $v$, then multiply them together.

### In our neural network example

Back to our example. As we have seen, the order of computation in a forward propagation is

$$
\begin{align*}
x \rightarrow t_0 \rightarrow z_0 \rightarrow t_1 \rightarrow z_1 \rightarrow t_2 \rightarrow z_2 \rightarrow \ell.
\end{align*}
$$

It should be clear to us now that finding gradients means we have to traverse the network backwards. Start from the loss $\ell$. Differentiate that with respect to $z_2$. Then with respect to $t_2$. Then with repect to $z_1$. And so on.

We actually don't want these gradients. What we actually want is the gradient with respect to $W_0, W_1$ and $W_2$, which are the matrices that transform a $z$ in one layer to a $t$ the next layer. However, in calculating these gradients, the chain rule requires us to compute the above intermediate gradients as well.

Below is a step-by-step procedure of backpropagation.

### Gradients for $z_2$ and $t_2$

First, let's start with $z_2$, the most immediate signal. Since we're using the half sum of squares loss, the gradient is just $z_2$ itself:

$$
\begin{align*}
\nabla_{z_2} \ell = z_2.
\end{align*}
$$

Now to $t_2$. Since $z_2$ is an element-wise identity transformation of of $t_2$, using the chain rule we have

$$
\begin{align*}
\nabla_{t_2} \ell = \nabla_{z_2} \ell \odot \text{Id}'(t_2),
\end{align*}
$$

where $\odot$ denotes element-wise multiplication. The reason why we have an element-wise multiplication here is that the Jacobian of $z_2$ with respect to $t_2$ is a diagonal matrix, $\text{diag}(\text{Id}'(t_2))$, and multiplying $J_\ell(z_2)$ with this matrix is the same as performing an element-wise product.

### Gradients for $z_1$, $W_2$, and $t_1$

Now let's move back one layer. Recall that
$$
\begin{align*}
t_2 = W_2 z_1.
\end{align*}
$$

We need to find the gradient for both $z_1$ and $W_2$. First, since this is a linear operation, differentiating $t_2$ with respect to $z_1$ will simply give us $W_2$. Using the chain rule again, we have

$$
\begin{align*}
\nabla_{z_1} \ell = W_2^\top  (\nabla_{t_2} \ell).
\end{align*}
$$

Now to $W_2$. Applying the chain rule, we have

$$
\begin{align*}
\nabla_{W_2} \ell = (\nabla_{t_2} \ell) z_1^\top.
\end{align*}
$$

Note that this is an outer product.


In both updates of $z_1$ and $W_2$, we used $\nabla_{t_2} \ell$ from the previous step. This is why the previou gradient signal needs to be stored for backpropagation, and why we need to calculate the gradient for variables we're not interested in (remember, we only need the gradients for $W$).

Finally to $t_1$. Since $z_1$ is an element-wise sigmoid transformation of $t_1$, we apply the same formula as that for $t_2$, this time replacing $\tanh$ with $\sigma$:

$$
\begin{align*}
\nabla_{t_1} \ell = \nabla_{z_1} \ell \odot \sigma'(t_2).
\end{align*}
$$

### Remaining gradients

There is no need to repeat ourselves when finding the gradients for the rest of the variables. This is because the procedure for $(z_0, W_1, t_0)$ is identical for $(z_1, W_2, t_1)$. Once we have the gradient signal $\nabla_{t_1}\ell$, we're good to go.

One final note is that when we have traversed all the way to the beginning of the network, we only need to find the gradient with respect to $W_0$. This will require $z_{-1}$, which is just $x$. The gradient for $x$ (the input) is not used for anything.

### Implementing backpropagation

We are now ready to fill in the TODO in the `loss_and_grad` method in `Net` above.

```python
# Paste this code at the end of loss_and_grad

# Diff the loss w.r.t. final layer (This is nabla_{z2})
self.nonlinear_outputs[-1].grad = self.loss.grad(self.nonlinear_outputs[-1].data)
for i in range(len(self.nonlinear_outputs) - 1, -1, -1):
    
    # Gradient from z to t. The "*" below is the element-wise product
    self.linear_outputs[i].grad = \
        self.activations[i].grad(self.linear_outputs[i].data) * self.nonlinear_outputs[i].grad
    
    # Gradient w.r.t. weights matrix. This is nabla_{W}.
    prev_output = self.nonlinear_outputs[i-1].data if i > 0 else x
    self.weights[i].grad = np.outer(self.linear_outputs[i].grad, prev_output)
    
    # Check if we have traversed to the first layer
    if i > 0:
        # If not at the first layer, continue finding nabla_{z}
        self.nonlinear_outputs[i-1].grad = self.weights[i].data.T @ self.linear_outputs[i].grad

return l
```

## Finishing Network with Backpropagation

Let's try an input $x$ and find the gradients of $\ell$ with respect to the parameters. After we call `loss_and_grad`, the gradients of all eligible tensors will be stored in their `.grad` attributes.

```py
# For reducibility
np.random.seed(100)
np_net = Net()
x = np.ones(5, dtype=float)
loss = np_net.loss_and_grad(x)

# Get the gradients for all parameters
np_grads = {"W" + str(i): g.grad for i, g in enumerate(np_net.weights)}
```

Now we are ready to take a gradient descent step!

## Autograd with PyTorch

To verify that our computation is correct, let's use `autorgrad` in PyTorch and find the gradients for the parameters.

```py
import torch
pt_net = torch.nn.Sequential()
pt_net.add_module("W0", torch.nn.Linear(in_features=5, out_features=10, bias=False))
pt_net.add_module("A0", torch.nn.Tanh())
pt_net.add_module("W1", torch.nn.Linear(in_features=10, out_features=4, bias=False))
pt_net.add_module("A1", torch.nn.Sigmoid())
pt_net.add_module("W2", torch.nn.Linear(in_features=4, out_features=12, bias=False))
pt_net.add_module("A2", torch.nn.Identity())

# Copy the weights in out numpy network to this new network
for param, np_param in zip(pt_net.parameters(), np_net.weights):
    param.data = torch.tensor(np_param.data, dtype=float)

x = torch.ones(5, dtype=float)
output = pt_net(x)
loss = 0.5 * torch.sum(output ** 2)
print("Loss =", loss.detach().item())
loss.backward()

# Get the gradients for all parameters
pt_grads = {name.split(".")[0]: x.grad.numpy() for name, x in pt_net.named_parameters()}
pt_grads
```

Check that the gradients by both versions match.

```py
for name in np_grads.keys():
    assert name in pt_grads
    print(name, "gradients match?", np.allclose(np_grads[name], pt_grads[name]))
```

```
W0 gradients match? True
W1 gradients match? True
W2 gradients match? True
```

## Conclusion

We have learned how backpropagation works in a feed-forward neural network. Here are some things you can try on your own:
- Add more layers to the network
- Try more activation functions, e.g., ReLU, leaky ReLU, GeLU, etc.
- Add bias to each Linear layer and find the gradient with respect to the bias.

Finally, you can download a Jupyter notebook version of this post [here](/files/backprop_tutorial.ipynb).