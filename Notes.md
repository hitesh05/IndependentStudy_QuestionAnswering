# Meetings and other notes

### 12/2 meeting
- Which phrase embeds are better than dense phrases?
- A retrieval method better than cosine similarity (acc > speed)
    - Cross attn, aggregation of similarity
    - Hierarchical
    - Inherently related to ranking
- Google nq dataset
- Is it possible that DPR is better for retrieval than ranking?

### Notes
- survey paper
- distance measure is always cosine similarity/inner product.
- Encoders better than DPH
    - `GNN-encoder`: [](https://arxiv.org/abs/2204.08241) --> *RUN*
    - `SimLM`: [](https://arxiv.org/pdf/2207.02578.pdf) --> *RUN*
    - `Costa`: [](https://arxiv.org/pdf/2204.10641.pdf)
    - `ART`: [](https://arxiv.org/pdf/2206.10658.pdf) --> *RUN*
    - `CoTMAE`: [](https://arxiv.org/pdf/2208.07670.pdf)
    - `deep prompt tuning`: [](https://arxiv.org/pdf/2208.11503.pdf)
        - low latency


- reranker approaches
    - retrieve, rerank, generate
    
    - `unsupervised passage reranking`: [](https://github.com/DevSinghSachan/unsupervised-passage-reranking)


*4/3*
- scores obtained after running the models on ada:
- SimLM:
    - MSMarco: {
        "NDCG@10": 0.50376,
        "Recall@100": 0.92997,
    }
    - NQ: {
        "NDCG@10": 0.4879,
        "Recall@100": 0.89771,
    }
- ART:
    - MSMarco: {
        "NDCG@10": 0.321,
        "Recall@100": 0.809,
    }
    - NQ: {
        "NDCG@10": 0.398,
        "Recall@100": 0.874,
    }


### Rerankers
- `unsupervised passage reranking`: [](https://github.com/DevSinghSachan/unsupervised-passage-reranking)
    - Apr 2022
    - use of bayes rule
    - instead of finding likelihood of passage containing answer to query [p(z_i|q)], we calculate the likelihood of generating the question given the passage [p(q|z_i)] --> follows from `bayes rule`
    - cross attn helps attend to each word in the passage

- `RankT5`: [](https://dl.acm.org/doi/abs/10.1145/3539618.3592047)
    - Oct 2022
    - code not available
    - checkpoints: [](https://github.com/google-research/google-research/tree/master/rankt5) --> model weights available
    - limited studies on how to leverage more powerful sequence-to-sequence models such as T5
    - 2 t5 models: enc-dec and only enc
    - finetuning with listwise ranking losses -- ListMLE etc

- `GripRank`: [](https://arxiv.org/pdf/2305.18144.pdf)
    - May 2023
    - code unavailable
    - by distilling knowledge from a generative passage estimator (GPE) to a passage ranker, where the GPE is a generative language model used to measure how likely the candidate passages can generate the proper answer.
    - they train a generative passage estimator (GPE) under the supervision of the golden answer, taking the concatenation of the query and golden passage as input. Once the GPE finishes training, we freeze the entire GPE and the passage ranker learns to rank the candidate passages ordered by the GPE. Each top-ranked passage is then concatenated with the input query for answer generation.
    - GPE is BARTLarge
    - passage ranker is a cross encoder architecture. ElectraBase pretrained on MS Marco as the backbone.
    - curriculum learning for knowledge distillation

- `open-source LLMs are zero-shot QLMs for document ranking`
    - Oct 2023
    - query likelihood models
    - QLMs rank documents based on the probability of generating the query given the content of a document. 
    - directly links to the UPR paper
    - *note:* they use T5, but t5 has already been finetuned on multi downstream tasks. hence the paper is more akin to transfer learning than a zero-shot setting.
    - instead we use LLMs (llama-2)
    - gurantee that the models were not trained on qa datasets previously, thus llama is chosen.
    - compared with other instruction tuned models 
    - it is evident that retrievers and re-rankers fine-tuned on MS MARCO training data consistently outperform zero-shot retrievers and QLM re-rankers across all datasets.
    - instruction tuning hinders perf

- `Beyond Yes and No: Improving Zero-Shot LLM Rankers via Scoring Fine-Grained Relevance Labels`
    - Apr 2024
    - code unavailable
    - previously, prompts: pointwise ranking (yes/no answers to whether a passage can answer a question)
    - the lack of intermediate relevance label options may cause the LLM to provide noisy or biased answers for documents that are partially relevant to the query
    - paper proposes to incorporate fine-grained relevance labels into the prompt for LLM rankers, enabling them to better differentiate among documents with different levels of relevance to the query and thus derive a more accurate ranking.
    - Instead of asking the LLM to choose between two options, we provide the LLM with fine-grained relevance labels, such as "Highly Relevant", "Somewhat Relevant" and "Not Relevant" and collect their likelihood scores from LLM predictions to derive the ranking score. The intuition is that the intermediate relevance labels in the prompt serve as a "cue" to the LLM to distinguish partially relevant documents from fully relevant or fully irrelevant ones.
    - perf remains similar for k between 4 and 8. decreases if we inc k further (k=num of labels)
        - thus perf does not always inc with increasing labels

- `Generating Diverse Criteria On-the-Fly to Improve Pointwise LLM Rankers`
    - Apr 2024
    - code available
    - 2 major drawbacks of pointwise rerankers
        - they fail to follow a standardized comparison guidance during the ranking process
        - they struggle with comprehensive considerations when dealing with complicated passages
    - propose to build a ranker that generates ranking scores based on a set of criteria from various perspectives.
    - This error happens when the LLM ranker decides to adopt a biased assessment criterion that prioritizes keyword presence over the nuanced understanding of content semantics.
    - MCRanker emulates the domain expertise and text analytical capabilities of professional annotators through virtual team recruiting.
    - 4 steps
        - team recruiting: nlp scientist and others
        - criteria gen
        - passage eval
        - score ensemble

### retrievers

- `DPR`

- `Densephrases`
    - question enc and phrase enc
    - query side finetuning
- `GNN encoder`
    - a dual encoder loses interaction bw the query-passage pair
    - they propose a GNN-encoder model in which query (passage) information is fused into passage (query) representations via graph neural networks that are constructed by queries and their top retrieved passages.
    - cross encoder (gnn) + dual encoder

- `SimLM`
    - better than gnn encoder
    - representation bottleneck (shallow decoder)
    - It employs a simple bottleneck architecture that learns to compress the passage information into a dense vector through self-supervised pre-training. they use a replaced language modeling objective
    - new approach for the PLM
    - good for retrieval, not as good for reranker

- `Costa`
    - contrastive span prediction
    - pre-trains the encoder alone with the contrastive span prediction task while still retaining the bottle-neck ability by forcing the encoder to generate the text representation close to its own random spans. Moreover, it enhances the discriminative ability of the encoder by pushing the text representation away from others.

- `ART`
    - unsupervised
    - auto-encoding based --> reconstructs the original question
    - an input question is used to retrieve a set of evidence passages; the passages are then used to compute the probability of reconstructing the original question.
    - to get log(z|q), we get log(q|z) [was also used in UPR]

- `RAG: Is Dense Passage Retrieval Retrieving?`
    - Apr 2024
    - code not avail
    - DPR training decentralizes how knowledge is stored in the network, creating multiple access pathways to the same information.
    - training limitation: the internal knowledge of the pre-trained model bounds what the retrieval model can retrieve.
    - possible improvements:
        - expose the DPR training process to more knowledge so more can be decentralized
        - inject facts as decentralized representations
        - model and incorporate knowledge uncertainty in the retrieval process
        - directly map internal model knowledge to a knowledge base.
    - *read more later*

- `Drop your Decoder: Pre-training with Bag-of-Word Prediction for Dense Passage Retrieval`
    - apr 2024
    - code avail
    - revealing that masked auto-encoder (MAE) pre-training with enhanced decoding significantly improves the term coverage of input tokens in dense representations, compared to vanilla BERT checkpoints.
    - we propose a modification to the traditional MAE by replacing the decoder of a masked auto-encoder with a completely simplified Bag-of-Word prediction task.
    - state-of-the-art retrieval performance on several large-scale retrieval benchmarks without requiring any additional parameters, which provides a 67% training speed-up compared to standard masked auto-encoder pre-training with enhanced decoding.
    - issues with MAE style trained retrievers
        - difficult to interpret why they work
        - MAE-style pre-training mandatory needs the additional Transformers-based decoders, which brings considerable GPU memory cost and additional $O(n^2)$ computational complexity
