# CSS
The dataset and code for ACL2023 Paper: A New Dataset and Empirical Study for Sentence Simplification in Chinese (https://aclanthology.org/2023.acl-long.462)

## Introduction
CSS is the first dataset for assessing sentence simplification in Chinese, as described in "A New Dataset and Empirical Study for Sentence Simplification in Chinese".  <br />
CSS consists of 766 human simplifications associated with the 383 original sentences from the PFR corpus (two simplifications per original sentence).  <br />
You can see more details in our paper.

## Files
- NMT_to_SS: build pseudo-Chinese-SS data, which describe in section 4.1.
- baseline_code: train code and predict code.

## Load from Huggingface




## Cite our Work
```
@inproceedings{yang-etal-2023-new,
    title = "A New Dataset and Empirical Study for Sentence Simplification in {C}hinese",
    author = "Yang, Shiping  and
      Sun, Renliang  and
      Wan, Xiaojun",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.462",
    pages = "8306--8321",
    abstract = "Sentence Simplification is a valuable technique that can benefit language learners and children a lot. However, current research focuses more on English sentence simplification. The development of Chinese sentence simplification is relatively slow due to the lack of data. To alleviate this limitation, this paper introduces CSS, a new dataset for assessing sentence simplification in Chinese. We collect manual simplifications from human annotators and perform data analysis to show the difference between English and Chinese sentence simplifications. Furthermore, we test several unsupervised and zero/few-shot learning methods on CSS and analyze the automatic evaluation and human evaluation results. In the end, we explore whether Large Language Models can serve as high-quality Chinese sentence simplification systems by evaluating them on CSS.",
}
```

