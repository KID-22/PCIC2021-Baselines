### PCIC 2021: Causal Inference and Recommendation

Here, we will give some baselines  to inspire you solving this problem by using some novel and causal algorithms.

If you have any questions, please feel free to contact by issues or sunhaodai@gmail.com.

More details will coming soon. Welcome to star and fork!



You can see more detailed information about these models from paper as follows:  

```
[1] Koren et al. 2009. Matrix factorization techniques for recommender systems. In Computer.
[2] Schnabel et al. 2016. Recommendations as Treatments: Debiasing Learning and Evaluation. In JMLR.
[3] Bonner et al. 2018. Causal embeddings for recommendation. In RecSys.
```

All three models were initially used in the rating prediction task. Here, the task of our problem is to predict whether a user like(1) or dislike(0) a tag, which can be considered as a binary ratings prediction task. So we first use the deterministic data extracting from Baseline 1 to construct the user-item interaction matrix. Then, we use these rating model to predict the user performance.  

**Note that we haven't used the rating.txt, and how to use this dataset is left to you participants to explore.**

