---
title: Making Inferences about Your Model Performance
date: 2025-6-4
permalink: /posts/ml-performance-uncertainty
toc: true
---

Model evaluation is a very broad term which means different things in different contexts. I write this post to remind myself---and, hopefully, some of you who are reading it---of what conclusion is appropriate for what evaluation result, so that we can move closer to appropriately understanding how good our model is.

First, some notation. Suppose we have a dataset $\mathcal{D} = \\{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\\}$, which we use to evaluate our supervised machine learning model $\mathcal{M}$. Let $f(x_i, y_i; \mathcal{M})$ be the evaluation function of model $\mathcal{M}$ with respect to the data point $(x_i, y_i)$. For instance, if $\mathcal{M}$ is a classfier, we may use the accuracy score $f(x_i, y_i; \mathcal{M}) = 1(\mathcal{M}(x_i) = y_i)$, which returns 1 if and only if the classification is correct.

In many existing benchmarks, the model's overall performance is reported as the average of these scores over all data points:

$$
\hat{S}_{\mathcal{M}} = \frac{1}{n} \sum_{i=1}^{n} f(x_i, y_i; \mathcal{M}). \nonumber
$$

Accordingly, different models are ranked by this overall performance.

If our evaluation dataset $\mathcal{D}$ contains *all* of the examples we care about, then we don't need any further uncertainty quantification. An example of such a case is when we want to benchmark a model *only* on adding two three-digit integers. We can simply enumerate the space of all data points and see how many times our model gets it correctly. If two models have average accuracies of 0.9 and 0.8, then we can say, *for certain*, that the former is superior to the latter, in the sense that if we randomly choose two three-digit numbers, the former model is more accurate in its calculation of their sum than does the latter.

This, however, is not what we usually mean. Generally, the space of all examples we wish to evaluate a model with is very large or even unenumerable. Thus, $\mathcal{D}$ is only a subset of that space. Having only $$\hat{S}_{\mathcal{M}}$$, can we say anything about the performance of $\mathcal{M}$ if we had access to the entire (hypothetical) space of examples?

## "True" performance

Suppose the entries $(x_i, y_i)$ are sampled independently from an unknown distribution $p$ defined over $(\mathcal{X}, \mathcal{Y})$, the space of all possible values. In this case, we are most likely interested in another quantity, the theoretical performance of model $\mathcal{M}$ expected over the *entire* distribution $p$:

$$
S_\mathcal{M} = \mathbb{E}_{(X, Y) \sim p} [f(X, Y; M)].  \nonumber
$$

While this quantity is unknown, if we assume that the dataset $\mathcal{D}$ is an independent and identically distributed (i.i.d.) sample from $p$, then the statistic $$\hat{S}_{\mathcal{M}}$$ gives us an unbiased estimate of $S_\mathcal{M}$. So far, so good.

In addition to giving a single point estimate, we can calculate a confidence interval to report what we think is a range of plausible values. First, the standard error of the mean can be estimated as

$$
\text{SE}_{\mathcal{M}} = \frac{1}{\sqrt{n}} \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (f(x_i, y_i; \mathcal{M}) - \hat{S}_\mathcal{M})^2}. \nonumber
$$

If our assumption is correct, the quantity $$\frac{\hat{S}_{\mathcal{M}} - S_{\mathcal{M}}}{\text{SE}_{\mathcal{M}}}$$ follows the student's t distribution with $n-1$ degrees of freedom. Thus, a $100\alpha\%$ confidence interval can be estimated as

$$
\hat{S}_{\mathcal{M}} \pm t_{1 - \frac{\alpha}{2}, n-1} \times \text{SE}_{\mathcal{M}}, \nonumber
$$

where $t_{1 - \frac{\alpha}{2}, n-1}$ is the critical value of the t distribution corresponding to the two-sided $100\alpha\%$ interval.

This leads us to [Miller](https://arxiv.org/abs/2411.00640)'s suggestion: Instead of reporting just $\hat{S}_{\mathcal{M}}$, report the 95% CI around that average as well. It allows us to show how tight our range of guesses for model $\mathcal{M}$'s theoretical or "true" performance is.

## Conclusion

All of this is covered in every statistics 101 class. What I want to emphasize, though, is that 

1. If we are only talking about the model's performance with respect to the dataset $\mathcal{D}$, then reporting  $$\hat{S}_{\mathcal{M}}$$ is enough. As a consequence, if we have another model $\mathcal{N}$ with performance $$\hat{S}_{\mathcal{N}}$$ such that $$\hat{S}_{\mathcal{N}} < \hat{S}_{\mathcal{M}}$$, we can say *for certain* that $\mathcal{M}$ is objectively better than $\mathcal{N}$ *in this dataset*.

2. If we want to make an inference about the theoretical performance of $\mathcal{M}$, though, it is important to inform the readers of how confident we are about this estimate via, for example, the standard error or confidence interval. Thus, if we have $$\hat{S}_{\mathcal{N}} < \hat{S}_{\mathcal{M}}$$, we cannot say for certain that $\mathcal{M}$ is better than $\mathcal{N}$. After all, $\mathcal{M}$ can just be luckier with this dataset. What allows us to make a more robust claim is via hypothesis testing.

I don't think either way is "wrong." What's important is we should be clear about what our reported numbers represent. I suspect that the majority of cases fall under (2), and thus I endorse Miller's suggestion.
