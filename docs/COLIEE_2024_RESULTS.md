# COLIEE 2024 Competition Results and Participating Team Papers

**Competition:** 11th Competition on Legal Information Extraction/Entailment (COLIEE 2024)
**Workshop:** Eighteenth International Workshop on Juris-Informatics (JURISIN 2024) at JSAI-isAI 2024
**Location:** Hamamatsu, Japan, May 28-29, 2024
**Proceedings:** Springer LNCS/LNAI vol. 14741, "New Frontiers in Artificial Intelligence"
**ISBN:** 978-981-97-3076-6

## Overview Paper

**Title:** Overview of Benchmark Datasets and Methods for the Legal Information Extraction/Entailment Competition (COLIEE) 2024
**Authors:** Randy Goebel, Yoshinobu Kano, Mi-Young Kim, Juliano Rabelo, Ken Satoh, Masaharu Yoshioka
**URL:** https://link.springer.com/chapter/10.1007/978-981-97-3076-6_8
**PDF:** https://coliee.org/documents/waivers/overview_COLIEE2024.pdf

### Participation Summary
- **Task 1 (Case Law Retrieval):** 10 teams, 26 runs
- **Task 2 (Case Law Entailment):** 6 teams, 18 runs
- **Task 3 (Statute Law Retrieval):** 8 teams, 20 runs
- **Task 4 (Statute Law Entailment):** 8 teams, 23 runs
- **Total:** ~12-15 distinct teams across all tasks

---

## Task 1 - Case Law Retrieval (Full Results)

**Dataset:** 7,350 case law files; 5,616 training cases (1,278 queries, ~4.16 noticed cases avg); 1,734 test cases (400 queries, 1,562 true noticed cases, ~3.90 avg)
**Metric:** Micro-averaged F1
**Key trend:** Winning F1 rose ~50% from 0.30 (2023) to 0.44 (2024)

### Table 1: Task 1 Results (All Runs)

| Team | Run | F1 | Precision | Recall |
|------|-----|----|-----------|--------|
| **TQM** | *best* | **0.4432** | 0.5057 | 0.3944 |
| UMNLP | *best* | 0.4134 | 0.4000 | 0.4277 |
| UMNLP | run2 | 0.4097 | 0.3755 | 0.4507 |
| UMNLP | run3 | 0.4046 | 0.3597 | 0.4622 |
| TQM | run2 | 0.4342 | 0.5082 | 0.3790 |
| TQM | run3 | 0.3548 | 0.4196 | 0.3073 |
| YR | *best* | 0.3605 | 0.3210 | 0.4110 |
| YR | run2 | 0.3483 | 0.3245 | 0.3758 |
| YR | run3 | 0.3417 | 0.3184 | 0.3688 |
| JNLP | *best* | 0.3246 | 0.3110 | 0.3393 |
| JNLP | run2 | 0.3222 | 0.3347 | 0.3105 |
| JNLP | run3 | 0.3103 | 0.3017 | 0.3195 |
| WJY | run1 | 0.3032 | 0.2700 | 0.3457 |
| WJY | run2 | 0.1878 | 0.1495 | 0.2522 |
| WJY | run3 | 0.1179 | 0.0870 | 0.1831 |
| WJY | run4 | 0.1174 | 0.0824 | 0.2042 |
| CAPTAIN | run1 | 0.1688 | 0.1793 | 0.1594 |
| CAPTAIN | run2 | 0.1574 | 0.1586 | 0.1562 |
| CAPTAIN | run3 | 0.0275 | 0.0140 | 0.7177 |
| CAPTAIN | run4 (lowest) | 0.0019 | 0.0019 | 0.0019 |
| NOWJ | run1 | 0.1313 | 0.0895 | 0.2465 |
| NOWJ | run2 | 0.1306 | 0.0957 | 0.2055 |
| NOWJ | run3 | 0.1224 | 0.0813 | 0.2478 |
| BM24 | run1 | 0.1878 | 0.1495 | 0.2522 |
| MIG | run1 | 0.0508 | 0.0516 | 0.0499 |
| UBCS | run1 | 0.0276 | 0.0140 | 0.7196 |
| UBCS | run2 | 0.0272 | 0.0139 | 0.7100 |
| UBCS | run3 | 0.0275 | 0.0140 | 0.7177 |

### Task 1 Rankings (Best Run Per Team)

| Rank | Team | Best F1 | Precision | Recall |
|------|------|---------|-----------|--------|
| 1 | TQM | 0.4432 | 0.5057 | 0.3944 |
| 2 | UMNLP | 0.4134 | 0.4000 | 0.4277 |
| 3 | YR | 0.3605 | 0.3210 | 0.4110 |
| 4 | JNLP | 0.3246 | 0.3110 | 0.3393 |
| 5 | WJY | 0.3032 | 0.2700 | 0.3457 |
| 6 | BM24 | 0.1878 | 0.1495 | 0.2522 |
| 7 | CAPTAIN | 0.1688 | 0.1793 | 0.1594 |
| 8 | NOWJ | 0.1313 | 0.0895 | 0.2465 |
| 9 | MIG | 0.0508 | 0.0516 | 0.0499 |
| 10 | UBCS | 0.0276 | 0.0140 | 0.7196 |

---

## Task 2 - Case Law Entailment (Full Results)

**Dataset:** 725 training query cases (25,783 paragraphs); 100 test query cases (3,651 paragraphs); avg ~1.37 relevant paragraphs per query; avg query length ~35 words; avg candidate paragraph ~106 words
**Metric:** Micro-averaged F1
**Key trend:** All top-4 teams used fine-tuned monoT5

### Table 2: Task 2 Results (All Runs)

| Team | Run | F1 | Precision | Recall |
|------|-----|----|-----------|--------|
| **AMHR** | mt53bk2r | **0.6512** | 0.6364 | 0.6667 |
| JNLP | 07f39 | 0.6320 | 0.6967 | 0.5782 |
| CAPTAIN | zs3 | 0.6360 | 0.7281 | 0.5646 |
| CAPTAIN | fs2 | 0.6235 | 0.7700 | 0.5238 |
| CAPTAIN | t5 | 0.6117 | 0.6181 | 0.6054 |
| JNLP | join-constr | 0.6045 | 0.6694 | 0.5510 |
| JNLP | join | 0.5912 | 0.6378 | 0.5510 |
| NOWJ | weak | 0.5946 | 0.5906 | 0.5986 |
| NOWJ | t5 | 0.6117 | 0.6181 | 0.6054 |
| OVGU | 2ovgurun1 | 0.5962 | 0.5636 | 0.6327 |
| OVGU | 2ovgurun2 | 0.5705 | 0.5506 | 0.5918 |
| OVGU | 2ovgurun3 | 0.5532 | 0.5000 | 0.6190 |
| NOWJ | bert | 0.5197 | 0.5032 | 0.5374 |
| MIG | mig1 | 0.4701 | 0.5673 | 0.4014 |
| MIG | mig2 | 0.4696 | 0.5800 | 0.3946 |
| MIG | mig3 | 0.1364 | 0.0979 | 0.2245 |
| AMHR | lsbk1.txt | 0.3320 | 0.4100 | 0.2789 |
| AMHR | lsbk2m42 | 0.3542 | 0.3617 | 0.3469 |

### Task 2 Rankings (Best Run Per Team)

| Rank | Team | Best F1 | Precision | Recall |
|------|------|---------|-----------|--------|
| 1 | AMHR | 0.6512 | 0.6364 | 0.6667 |
| 2 | CAPTAIN | 0.6360 | 0.7281 | 0.5646 |
| 3 | JNLP | 0.6320 | 0.6967 | 0.5782 |
| 4 | NOWJ | 0.6117 | 0.6181 | 0.6054 |
| 5 | OVGU | 0.5962 | 0.5636 | 0.6327 |
| 6 | MIG | 0.4701 | 0.5673 | 0.4014 |

---

## Task 3 - Statute Law Retrieval (Best Run Per Team)

**Dataset:** 768 Japanese Civil Code articles; 1,097 training questions + 109 new test questions from 2023 bar exam
**Metric:** Macro-averaged F2 (emphasis on recall)

| Rank | Submission ID | F2 | Precision | Recall | MAP |
|------|---------------|----|-----------|--------|-----|
| 1 | JNLP.constr-join * | **0.807** | 0.709 | **0.870** | 0.801 |
| 2 | CAPTAIN.bjpAllMonoT5 | 0.800 | 0.732 | 0.845 | **0.815** |
| 3 | TQM-run1 # | 0.782 | **0.785** | 0.800 | 0.790 |
| 4 | NOWJ-25mulreftask-ensemble # | 0.772 | 0.690 | 0.835 | 0.756 |
| 5 | AMHR02 | 0.749 | 0.651 | 0.825 | 0.740 |
| 6 | UA-anglE | 0.711 | 0.610 | 0.800 | 0.700 |
| 7 | BM24-1 * | 0.539 | 0.282 | 0.795 | - |
| 8 | PSI01 ? | 0.086 | 0.090 | 0.085 | 0.231 |

(*) = LLM with undisclosed training data; (#) = no LLM used; (?) = unclear

---

## Task 4 - Statute Law Entailment (Best Run Per Team)

**Dataset:** Same articles as Task 3; 109 test questions from 2023 bar exam
**Metric:** Accuracy (# correct out of 109)

| Rank | Team/Run | # Correct | Accuracy (R05) |
|------|----------|-----------|----------------|
| 1 | CAPTAIN2 | 90 | **0.8257** |
| 2 | JNLP1 * | 89 | 0.8165 |
| 3 | UA_slack | 87 | 0.7982 |
| 3 | UA_encoder_decoder | 87 | 0.7982 |
| 5 | CAPTAIN1 | 86 | 0.7890 |
| 5 | CAPTAIN3 | 86 | 0.7890 |
| 5 | JNLP2 * | 86 | 0.7890 |
| 8 | UA_gpt | 85 | 0.7798 |
| 9 | AMHR.ensembleA50 | 84 | 0.7706 |
| 9 | AMHR.single | 84 | 0.7706 |
| 11 | HI1 | 82 | 0.7523 |
| 11 | NOWJ.pandap46 * | 82 | 0.7523 |
| 13 | JNLP3 * | 80 | 0.7339 |
| 13 | AMHR.ensembleA0 | 80 | 0.7339 |
| 13 | NOWJ.flant5-panda * | 80 | 0.7339 |
| 16 | NOWJ.bagging * | 78 | 0.7156 |
| 17 | OVGU1 + | 77 | 0.7064 |
| 18 | KIS2 + | 76 | 0.6972 |
| 18 | OVGU3 + | 76 | 0.6972 |
| 20 | OVGU2 + | 70 | 0.6422 |
| 21 | KIS1 | 67 | 0.6147 |
| 22 | HI3 | 64 | 0.5872 |
| 23 | HI2 | 63 | 0.5780 |
| 24 | KIS3 | 62 | 0.5688 |
| - | Baseline (Yes to all) | 60 | 0.5505 |

(*) = not fully disclosed models; (+) = preprocessing by such models

---

## Participating Team Papers (Detailed)

### 1. TQM - 1st Place Task 1, 3rd Place Task 3

**Paper:** "Towards an In-Depth Comprehension of Case Relevance for Better Legal Retrieval"
**Authors:** Haitao Li, You Chen, Zhekai Ge, Qingyao Ai, Yiqun Liu, Quan Zhou, Shuai Huo
**Affiliation:** Tsinghua University / related institutions (China)
**arXiv:** https://arxiv.org/abs/2404.00947
**Springer:** https://link.springer.com/chapter/10.1007/978-981-97-3076-6_15
**Tasks:** Task 1 (1st, F1=0.4432), Task 3 (3rd, F2=0.782)

**Method Summary - Task 1:**
- **Pre-processing:** Noise elimination from raw case texts
- **Lexical matching:** Classical methods (BM25, TF-IDF) with enhanced case relevance features
- **Semantic retrieval:** Dense vector retrieval models
- **Feature fusion:** Learning-to-rank (LTR) combining lexical + semantic + simple features (e.g., case length)
- **Post-processing:** Heuristic strategies based on common properties of relevant cases
- **Key insight:** Method aims at deeper understanding of case relevance, not just surface matching

**Method Summary - Task 3:**
- MonoT5 tuned with MS MARCO for ranking (run1)
- LightGBM to integrate results of different models (runs 2-3)
- BM25, Legal BERT, and MonoT5 for integration

### 2. UMNLP - 2nd Place Task 1

**Paper:** "Similarity Ranking of Case Law Using Propositions as Features"
**Authors:** Damian Curran, Mike Conway
**Affiliation:** University of Melbourne, Australia
**Springer:** https://dl.acm.org/doi/10.1007/978-981-97-3076-6_11
**GitHub:** https://github.com/dc435/COLIEE_2024_Task1
**Tasks:** Task 1 only (2nd, F1=0.4134)

**Method Summary:**
- Pairwise similarity ranking framework
- Feed-forward neural network for binary classification of query-candidate pairs
- Novel feature: "propositions" (short summaries of the basis on which a case was noticed)
- Additional features: judge name matching, verbatim quotation extraction, paragraph/sentence-level similarity
- Multiple feature levels: paragraphs, sentences, propositions

### 3. AMHR (Advanced Machine Human Reasoning) - 1st Place Task 2

**Paper:** "AMHR COLIEE 2024 Entry: Legal Entailment and Retrieval"
**Authors:** Animesh Nighojkar, Kenneth Jiang, Logan Fields, Onur Bilgin, Stephen Steinle, Yernar Sadybekov, Zaid Marji, John Licato
**Affiliation:** University of South Florida (CSE Lab)
**Springer:** https://link.springer.com/chapter/10.1007/978-981-97-3076-6_14
**News:** https://www.usf.edu/engineering/news/2024/cse/coliee.aspx
**Tasks:** Task 2 (1st, F1=0.6512), Task 3 (5th, F2=0.749), Task 4 (9th-10th, acc=0.7706)

**Method Summary - Task 2:**
- Fine-tuned monoT5 model pre-trained on MS-MARCO
- Hard negative mining using BM25 + another monoT5 version
- Top-2 predictions selected if similarity score ratio < 6.619 (hyperparameter via grid search); otherwise top-1 only
- Also tried legal-BERT with triplet loss (overfitted)

**Method Summary - Task 3:**
- BM25 to select top 50 hits
- Re-ranked using monot5-3b-msmarco fine-tuned for ranking
- Top 5 selected
- 3 LLM variants (FLAN-T5, FLAN-alpaca) to select final relevant articles

**Method Summary - Task 4:**
- ~80 prompts on google/flan-t5-xxl per training question
- Best 25 prompts vote on test answers
- Article similarity scoring with sentence-transformers and BM25

### 4. CAPTAIN - 1st Place Task 4, 2nd Place Task 2

**Paper:** "CAPTAIN at COLIEE 2024: Large Language Model for Legal Text Retrieval and Entailment"
**Authors:** Phuong Nguyen, Cong Nguyen, Hiep Nguyen, Minh Nguyen, An Trieu, Dat Nguyen, Le-Minh Nguyen
**Affiliation:** Japan Advanced Institute of Science and Technology (JAIST)
**Springer:** https://link.springer.com/chapter/10.1007/978-981-97-3076-6_9
**OpenReview:** https://openreview.net/forum?id=g3NVcJplMK
**Tasks:** Task 1 (7th, F1=0.1688), Task 2 (2nd, F1=0.6360), Task 3 (2nd, F2=0.800), Task 4 (1st, acc=0.8257)

**Method Summary - Task 1:**
- Heuristic pre-processing
- TF-IDF and BM25 for keyword extraction and document retrieval
- LLMs to summarize decisions
- Fine-tuning retrieval model on summaries

**Method Summary - Task 2:**
- Fine-tuned monoT5 sequence-to-sequence model with hard negative sampling
- Top-k candidates selected for zero-shot and few-shot prompting with FlanT5 LLM
- In-context learning for final predictions

**Method Summary - Task 3:**
- BERT-base-Japanese tuned for COLIEE Task 3 (bjpAll)
- MonoT5 tuned with MS MARCO (bjpAllMonoT5)
- LLM filtering via prompting with Flan T5
- Ensemble of multiple systems

**Method Summary - Task 4:**
- Data augmentation: google/flan-t5-xxl summarizes statute law with heuristic rules
- Augmentation + fine-tuning approach
- Few-shot prompting using Dense Passage Retrieval for demonstration selection
- CoT prompting with google/flan-t5-xxl
- Ensemble of all models
- **Winner:** CAPTAIN2 (augmentation + fine-tuning) scored 90/109 = 0.8257

### 5. JNLP - 3rd Place Task 2, 1st Place Task 3

**Paper:** "Pushing the Boundaries of Legal Information Processing with Integration of Large Language Models"
**Authors:** Chau Nguyen, Thanh Tran, Khang Le, Hien Nguyen, Truong Do, Trang Pham, Son T. Luu, Trung Vo, Le-Minh Nguyen
**Affiliation:** JAIST and related Vietnamese institutions
**Springer:** https://link.springer.com/chapter/10.1007/978-981-97-3076-6_12
**Tasks:** Task 1 (4th, F1=0.3246), Task 2 (3rd, F1=0.6320), Task 3 (1st*, F2=0.807), Task 4 (2nd*, acc=0.8165)

(*Note: JNLP.constr-join used LLM (Orca, Qwen) with undisclosed training data; best system with disclosed training data for Task 3 was CAPTAIN.bjpAllMonoT5)

**Method Summary - Task 1:**
- Three-phase approach: (1) BM25 on paragraph-split query documents with top-k cutoff, (2) re-ranking stage, (3) top-k selection via grid search
- Ensemble strategy concatenating re-ranker results before selecting top-k

**Method Summary - Task 2:**
- Fine-tuned MonoT5 (T5-3B reranker fine-tuned on MS MARCO passage dataset for 10k steps)
- Hard negative sampling
- Flan-T5 and Mixtral for prompting

**Method Summary - Task 3:**
- BERT-base-Japanese fine-tuned for COLIEE Task 3
- Multiple checkpoint ensemble
- JNLP1: Mistral prompt + Flan-Alpaca top-1
- JNLP2: RankLLaMA (MS MARCO tuned on LLaMA2) for scoring
- JNLP3 (constr-join): Orca and Qwen LLMs for concise list from ranked results + Mistral retrieval results for recall

**Method Summary - Task 4:**
- JNLP1/JNLP2: Prompted different LLMs (Wqen, Mistral, Flan-Alpaca, Flan-T5) with majority voting ensemble
- JNLP3: Prompted Flan-T5 and Mistral, ensemble with Dawid-Skene label model

### 6. NOWJ - Participated in All Tasks

**Paper:** "NOWJ@COLIEE 2024: Leveraging Advanced Deep Learning Techniques for Efficient and Effective Legal Information Processing"
**Authors:** Nguyen T.M., Nguyen H.L., Nguyen D.Q., Nguyen H.T., Vuong T.H.Y., Nguyen H.T.
**Springer:** https://link.springer.com/chapter/10.1007/978-981-97-3076-6_13
**ResearchGate:** https://www.researchgate.net/publication/380926861
**Tasks:** Task 1 (8th, F1=0.1313), Task 2 (4th, F1=0.6117), Task 3 (4th, F2=0.772), Task 4 (11th-16th range)

**Method Summary - Task 1:**
- BM25 + pre-trained Longformer combination
- BM25 calculates similarity as pre-ranking input to Longformer
- Scores combined with grid-search-optimized parameters

**Method Summary - Task 2:**
- Two approaches: multilingual BERT and monoT5 for entailment recognition
- MonoT5: T5-based re-ranking model fine-tuned for downstream classification
- mBERT: traditional approach for document re-ranking (carried from 2023)

**Method Summary - Task 3:**
- Multitask BERT for Sequence Classification
- Ensemble with lexical-based BM25 scores

**Method Summary - Task 4:**
- pandap46: TheBloke/Panda-7B-v0.1-GPTQ with legal prompts
- flant5-panda: google/flan-t5-xl + panda with bagging
- bagging: combined 5 runs (Panda + Flan-T5 + different prompts) with majority voting

### 7. OVGU - Task 2 and Task 4

**Paper:** "Improving Robustness in Language Models for Legal Textual Entailment Through Artifact-Aware Training"
**Authors:** Sabine Wehnert, Venkatesh Murugadas, Preetam Vinod Naik, Ernesto William De Luca
**Affiliation:** Otto von Guericke University Magdeburg (OVGU), Germany
**Tasks:** Task 2 (5th, F1=0.5962), Task 4 (17th-20th range)

**Method Summary - Task 2:**
- Chain of pre-trained Custom Legal-BERT models
- Fine-tuned on sub-datasets generated using BM25
- Bi-Encoder to select top-N candidate paragraphs
- Binomial test for robustness
- OpenAI GPT-3.5-turbo for adversarial instance creation (artifact detection)
- Chained prediction: high-precision model first, fallback to second model, then BM25 top-ranked

**Method Summary - Task 4:**
- MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli fine-tuned on Task 4 COLIEE + custom datasets
- Word Overlap and Contradiction features
- Boolean subsequence features (hypothesis is subsequence of premise)

### 8. KIS Team - Task 4

**Paper:** "LLM Tuning and Interpretable CoT: KIS Team in COLIEE 2024"
**Authors:** Masaki Fujita, Takaaki Onaga, Yoshinobu Kano
**Affiliation:** Shizuoka University, Japan
**Springer:** https://link.springer.com/chapter/10.1007/978-981-97-3076-6_10
**Tasks:** Task 4 (18th-24th range, best acc=0.6972)

**Method Summary:**
- KIS1: Fine-tuning + few-shot learning + retrieval-augmented generation + character count instructions + rule-based ensemble
- KIS2: Same as KIS1 but few-shot data replaced with GPT-4-generated outputs
- KIS3: Fine-tuning only
- Interpretable Chain-of-Thought (CoT): structured prompting format to guide reasoning and identify influential parts of the inference process

### 9. HI (Hybrid Intelligence) - Task 4

**Paper:** "A Hybrid Approach to Legal Textual Entailment"
**Authors:** Cas Steging, Leeuwen L.V.
**PDF:** https://www.steging.nl/wp-content/uploads/2024/09/JURISIN2024.pdf
**Tasks:** Task 4 (11th, 22nd, 23rd; best acc=0.7523)

**Method Summary:**
- HI1: declare-lab/flan-alpaca-gpt4-xl with zero-shot prompting
- HI2: Manually crafted Abstract Dialectical Frameworks (ADFs) using ANGELIC methodology for 25 legal articles; LLM ascribes factors to each question; LLM fallback when ADF unavailable
- HI3: Translated articles into additional ADFs using GPT3.5-turbo

### 10. UA (University of Alberta) - Task 3 and Task 4

**Paper:** "Legal Yes/No Question Answering Through Text Embedding, Fine-Tuning, and Prompt Engineering"
**Authors:** H. Babiker, M.A. Rahman, M.Y. Kim, J. Rabelo, R. Goebel
**Affiliation:** University of Alberta, Canada
**Tasks:** Task 3 (6th, F2=0.711), Task 4 (3rd-8th range, best acc=0.7982)

**Method Summary - Task 3:**
- Universal AnglE Embedding (SeanLee97/angle-llama-7b-nli-20231027 via LLaMA)
- anglE: whole articles for embedding
- angleE_chunk: single sentences for embedding
- mp_net: sentence transformer model fine-tuned for Task 3
- Cosine similarity for finding relevant articles

**Method Summary - Task 4:**
- UA_stack: Zero-shot learning on google/flan-t5-xxl with PromptSource; top 3 prompts with majority voting
- UA_GPT: Same but chose top-1 GPT-3 style prompt
- UA_encoder_decoder: Fine-tuned last two layers of both decoder and encoder of flan-t5-xxl

### 11. BM24 - Task 1 and Task 3

**No dedicated paper found.**
**Tasks:** Task 1 (6th, F1=0.1878), Task 3 (7th, F2=0.539)

**Method Summary - Task 1:**
- GPT-3.5 for case segmentation
- AnglE embedding (SeanLee97/angle-llama-7b-nli-20231027 via LLaMA/HuggingFace)
- Selected representative segment per case, stored in FAISS
- Query segments used to retrieve similar cases from FAISS vector store

**Method Summary - Task 3:**
- AnglE-llama-7b-nli (AnglE embedding via LLaMA) for semantic retrieval
- Fine-tuned on COLIEE task data (1 and 3), Supreme Court of Canada Bulk Decisions, and STS dataset
- GPT3.5 for generating similar sentences

### 12. MIG - Task 1 and Task 2

**No dedicated paper found.**
**Tasks:** Task 1 (9th, F1=0.0508), Task 2 (6th, F1=0.4701)

**Method Summary - Task 1:**
- Informative baseline (no LLMs)
- BERT-base and BERT-large for vectorization
- FAISS for cosine similarity between candidate and new cases
- Top 20 candidates selected; gap-based threshold: recommend first i cases where d_i > 2*d_1

### 13. UBCS - Task 1

**No dedicated paper found.**
**Tasks:** Task 1 (10th, F1=0.0276)

**Method Summary:**
- Vanilla TF-IDF weighting model baseline
- Run 1: Standard TF-IDF retrieval and ranking
- Run 2: Summarization on query cases only before TF-IDF
- Run 3: Summarization on both query and candidate cases

### 14. PSI - Task 3

**No dedicated paper found.**
**Tasks:** Task 3 (8th, F2=0.086)

**Method Summary:**
- Short description only; details not available

### 15. YR - Task 1

**No dedicated paper found in proceedings.**
**Tasks:** Task 1 (3rd, F1=0.3605)

**Method Summary:**
- Not described in detail in the overview paper; appears to have submitted 3 runs but no published team paper was found in the proceedings

### 16. WJY - Task 1

**No dedicated paper found in proceedings.**
**Tasks:** Task 1 (5th, F1=0.3032)

**Method Summary:**
- Not described in detail in the overview paper; submitted multiple runs but no published team paper was found

---

## Published Team Papers (Springer Proceedings Chapters)

All published in: *New Frontiers in Artificial Intelligence*, LNCS 14741, Springer 2024

| Ch. | Title | Team | DOI suffix |
|-----|-------|------|------------|
| 8 | Overview of Benchmark Datasets and Methods for COLIEE 2024 | Organizers | _8 |
| 9 | CAPTAIN at COLIEE 2024: Large Language Model for Legal Text Retrieval and Entailment | CAPTAIN | _9 |
| 10 | LLM Tuning and Interpretable CoT: KIS Team in COLIEE 2024 | KIS | _10 |
| 11 | Similarity Ranking of Case Law Using Propositions as Features | UMNLP | _11 |
| 12 | Pushing the Boundaries of Legal Information Processing with Integration of LLMs | JNLP | _12 |
| 13 | NOWJ@COLIEE 2024: Leveraging Advanced Deep Learning Techniques | NOWJ | _13 |
| 14 | AMHR COLIEE 2024 Entry: Legal Entailment and Retrieval | AMHR | _14 |
| 15 | Towards an In-Depth Comprehension of Case Relevance for Better Legal Retrieval | TQM | _15 |

**Additional JURISIN 2024 papers (not in Springer but in workshop proceedings):**
- Steging, C., Leeuwen, L.V.: "A hybrid approach to legal textual entailment" (HI team)
- Wehnert, S., et al.: "Improving robustness in language models for legal textual entailment through artifact-aware training" (OVGU)
- Babiker, H., et al.: "Legal yes/no question answering through text embedding, fine-tuning, and prompt engineering" (UA)

---

## arXiv Papers Related to COLIEE 2024

| arXiv ID | Title | Team/Authors | Notes |
|----------|-------|-------------|-------|
| 2404.00947 | Towards an In-Depth Comprehension of Case Relevance for Better Legal Retrieval | TQM (Li et al.) | Task 1 winner paper |
| 2403.18098 | GPTs and Language Barrier: A Cross-Lingual Legal QA Examination | (not a team paper) | Benchmarks GPT-3.5/4 on COLIEE Task 4 dataset |
| 2504.08400 | A Reproducibility Study of Graph-Based Legal Case Retrieval | (post-competition) | Extends to COLIEE 2024 dataset |

---

## Key Takeaways for COLIEE 2026 Strategy

### Task 1 (Case Law Retrieval) Insights:
1. **Winning approach (TQM, F1=0.44):** Hybrid lexical + semantic retrieval with learning-to-rank fusion and careful pre/post-processing
2. **Runner-up (UMNLP, F1=0.41):** Novel "proposition" features + judge name matching + feed-forward NN classifier
3. **BM25 is essential** as first-stage retrieval; dense retrieval alone is insufficient
4. **F1 scores remain low** (~0.44 max), suggesting the task is very challenging
5. **Learning-to-rank** fusion of multiple feature types consistently helps
6. **Post-processing heuristics** (based on properties of relevant cases) boost performance
7. **Paragraph-level encoding** improves retrieval (JNLP's 3-phase approach)

### Task 2 (Case Law Entailment) Insights:
1. **monoT5 dominates:** All top-4 teams used fine-tuned monoT5 (T5-based reranker)
2. **Hard negative mining is critical:** Both AMHR (1st) and JNLP (3rd) used hard negatives
3. **Score-ratio thresholding:** AMHR's winning method used a ratio threshold (6.619) to decide between top-1 vs top-2 predictions
4. **LLM prompting as augmentation:** CAPTAIN used zero-shot/few-shot with FlanT5 on top of monoT5
5. **Ensemble helps:** OVGU's chained model approach (high-precision first, fallback)
6. **F1 around 0.65 is SOTA** for this task

### General Trends:
- Hybrid pipelines (BM25 + neural reranking) remain dominant
- monoT5 pre-trained on MS-MARCO is the go-to reranking model
- LLMs (Flan-T5, Mistral, LLaMA variants) increasingly used for summarization, prompting, and augmentation
- Ensemble methods consistently improve over single models
- Open-source model constraint pushes teams toward Flan-T5, Mistral, LLaMA family
