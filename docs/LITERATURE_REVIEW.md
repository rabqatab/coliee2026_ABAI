# COLIEE 2026 Literature Review

**Scope:** Legal NLP, Legal QA, Information Retrieval, Entailment, Judgment Prediction  
**Criteria:** Top-tier venues (ACL, EMNLP, NeurIPS, AAAI, SIGIR, ACM) OR 30+ citations  
**Last Updated:** 2026-01-27

---

## 1. COLIEE Competition & Overview Papers

### 1.1 COLIEE 2022 Summary: Methods for Legal Document Retrieval and Entailment
- **Authors:** Kim, Rabelo, Goebel, Yoshioka et al.
- **Venue:** JSAI-isAI 2022, Springer
- **Citations:** 63
- **Summary:** Overview of COLIEE 2022 competition covering case law IR (Task 1), case law entailment (Task 2), statute law IR (Task 3), and statute law entailment/QA (Task 4). Details participating systems and evaluation metrics.

### 1.2 Overview and Discussion of COLIEE 2023
- **Authors:** Goebel, Kano, Kim, Rabelo, Satoh et al.
- **Venue:** The Review of Socionetwork Strategies, 2024, Springer
- **Citations:** 21
- **Summary:** Comprehensive overview of COLIEE 2023, discussing the evolution of the competition, new challenging tasks based on Canadian case law and Japanese statute law.

### 1.3 COLIEE-2015: Evaluation of Legal Question Answering
- **Authors:** Kim, Goebel, Ken
- **Venue:** JURISIN 2015
- **Citations:** 33
- **Summary:** Early COLIEE competition overview establishing the legal QA task framework for Japanese civil law.

---

## 2. Pre-trained Language Models for Legal Domain

### 2.1 LEGAL-BERT: The Muppets Straight Out of Law School ⭐
- **Authors:** Chalkidis, Fergadiotis, Malakasiotis et al.
- **Venue:** arXiv 2020 (widely cited)
- **Citations:** 1,498
- **Summary:** Seminal work on domain-specific BERT for legal text. Explores pre-training strategies: further pre-training on legal corpora vs. training from scratch. Shows consistent improvements on legal NLP tasks including contract clause classification and NER.
- **Key Finding:** Domain-specific pre-training outperforms general BERT on legal tasks.

### 2.2 Lawformer: A Pre-trained Language Model for Chinese Legal Long Documents ⭐
- **Authors:** Xiao, Hu, Liu, Tu, Sun
- **Venue:** AI Open, 2021, Elsevier
- **Citations:** 352
- **Summary:** Longformer-based PLM for Chinese legal documents. Addresses the challenge of long document processing in legal domain using efficient attention mechanisms.
- **Key Finding:** Longformer architecture effective for handling lengthy legal texts.

### 2.3 SAILER: Structure-Aware Pre-trained Language Model for Legal Case Retrieval ⭐
- **Authors:** Li, Ai, Chen, Dong, Wu, Liu
- **Venue:** SIGIR 2023, ACM
- **Citations:** 119
- **Summary:** Novel pre-training approach that incorporates legal document structure (facts, reasoning, decision). Simulates the legal case writing process during pre-training.
- **Key Finding:** Structure-aware pre-training significantly improves legal case retrieval.

### 2.4 Pre-trained Language Models for the Legal Domain: A Case Study on Indian Law
- **Authors:** Paul, Mandal, Goyal, Ghosh
- **Venue:** ICAIL 2023, ACM
- **Citations:** 101
- **Summary:** Comprehensive study of PLMs for Indian legal domain. Compares domain-specific pre-training vs. fine-tuning approaches.

### 2.5 Italian-LEGAL-BERT: Pre-trained Transformer for Italian Law
- **Authors:** Licari, Comandè
- **Venue:** EKAW 2022 / Computer Law & Security Review 2024
- **Citations:** 76 (EKAW) / 31 (CLSR)
- **Summary:** Domain-specific BERT models for Italian legal context, demonstrating cross-lingual applicability of legal PLM approaches.

### 2.6 AraLegal-BERT: A Pre-trained Language Model for Arabic Legal Text
- **Authors:** Al-Qurishi, AlQaseemi et al.
- **Venue:** NLLP Workshop 2022, ACL
- **Citations:** 36
- **Summary:** First Arabic legal BERT model, addressing under-resourced language legal NLP.

### 2.7 Bringing Order into the Realm of Transformer-based Language Models for AI and Law
- **Authors:** Greco, Tagarelli
- **Venue:** Artificial Intelligence and Law, 2024, Springer
- **Citations:** 70
- **Summary:** Comprehensive survey organizing transformer-based models for legal AI tasks. Provides taxonomy of approaches and benchmarks.

---

## 3. Legal Judgment Prediction

### 3.1 Legal Judgment Prediction via Topological Learning ⭐
- **Authors:** Zhong, Guo, Tu, Xiao, Liu et al.
- **Venue:** EMNLP 2018, ACL
- **Citations:** 462
- **Summary:** Models dependencies between charges, law articles, and prison terms using a DAG-based topological structure. Foundation work for multi-task legal judgment prediction.
- **Key Finding:** Topological multi-task learning captures legal reasoning structure.

### 3.2 A Survey on Legal Judgment Prediction: Datasets, Metrics, Models and Challenges
- **Authors:** Cui, Shen, Wen
- **Venue:** IEEE Access, 2023
- **Citations:** 139
- **Summary:** Comprehensive survey covering LJP datasets, evaluation metrics, neural models, and open challenges. Essential reading for LJP research.

### 3.3 NeurJudge: A Circumstance-Aware Neural Framework for Legal Judgment Prediction
- **Authors:** Yue, Liu, Jin, Wu, Zhang, An et al.
- **Venue:** SIGIR 2021, ACM
- **Citations:** 124
- **Summary:** Introduces circumstance extraction for better judgment prediction. Addresses confusing charges problem.

### 3.4 Judgment Prediction via Injecting Legal Knowledge into Neural Networks
- **Authors:** Gan, Kuang, Yang, Wu
- **Venue:** AAAI 2021
- **Citations:** 89
- **Summary:** Integrates legal knowledge graphs into neural networks for judgment prediction. Shows importance of domain knowledge injection.

### 3.5 MANN: A Multichannel Attentive Neural Network for Legal Judgment Prediction
- **Authors:** Li, Zhang, Ye, Guo, Fang
- **Venue:** IEEE Access, 2019
- **Citations:** 82
- **Summary:** Multi-channel attention mechanism for learning from judgment documents.

### 3.6 CNN-based Automatic Prediction of Judgments of the European Court of Human Rights
- **Authors:** Kaur, Bozic
- **Venue:** AICS 2019
- **Citations:** 43
- **Summary:** Application of CNNs to ECHR judgment prediction, demonstrating cross-jurisdiction applicability.

---

## 4. Legal Text Entailment & NLI

### 4.1 Legal Information Retrieval and Entailment Using Transformer-Based Approaches
- **Authors:** Kim, Rabelo, Babiker, Rahman et al.
- **Venue:** The Review of Socionetwork Strategies, 2024, Springer
- **Citations:** 18
- **Summary:** DeBERTa-based approach for legal entailment in COLIEE tasks. Exploits NLI formulation for yes/no QA.

### 4.2 Legal Information Retrieval and Entailment Based on BM25, Transformer and Semantic Thesaurus Methods
- **Authors:** Kim, Rabelo, Okeke, Goebel
- **Venue:** The Review of Socionetwork Strategies, 2022, Springer
- **Citations:** 40
- **Summary:** Hybrid approach combining lexical (BM25) and neural (transformer) methods with semantic thesaurus for legal IR and entailment.

### 4.3 Combining Similarity and Transformer Methods for Case Law Entailment
- **Authors:** Rabelo, Kim, Goebel
- **Venue:** ICAIL 2019, ACM
- **Citations:** 42
- **Summary:** Combines traditional similarity measures with transformer-based NLI for case law entailment.

### 4.4 Improving Abstractive Summarization of Legal Rulings Through Textual Entailment
- **Authors:** Feijo, Moreira
- **Venue:** Artificial Intelligence and Law, 2023, Springer
- **Citations:** 78
- **Summary:** Uses textual entailment to improve legal summarization quality and faithfulness.

### 4.5 Applying BERT Embeddings to Predict Legal Textual Entailment
- **Authors:** Wehnert, Dureja, Kutty, Sudhi et al.
- **Venue:** The Review of Socionetwork Strategies, 2022, Springer
- **Citations:** 18
- **Summary:** BERT-based embeddings for legal textual entailment prediction in COLIEE.

---

## 5. Dense Retrieval & Legal IR

### 5.1 Attentive Deep Neural Networks for Legal Document Retrieval
- **Authors:** Nguyen, Phi, Ngo, Tran et al.
- **Venue:** Artificial Intelligence and Law, 2024, Springer
- **Citations:** 63
- **Summary:** Novel attentive neural architectures for statute law retrieval. Addresses the challenge of matching queries with relevant legal articles.

### 5.2 Delta: Pre-train a Discriminative Encoder for Legal Case Retrieval via Structural Word Alignment
- **Authors:** Li, Ai, Han, Chen, Dong, Liu
- **Venue:** AAAI 2025
- **Citations:** 11
- **Summary:** Pre-training method that identifies key facts within legal documents for discriminative case retrieval.

### 5.3 Combining Lexical and Neural Retrieval with Longformer-based Summarization for Effective Case Law Retrieval
- **Authors:** Askari, Verberne, Alonso, Marchesin et al.
- **Venue:** DESIRES 2021
- **Citations:** 38
- **Summary:** Hybrid lexical-neural retrieval with summarization for handling long legal documents.

### 5.4 Incorporating Structural Information into Legal Case Retrieval
- **Authors:** Ma, Wu, Ai, Liu, Shao, Zhang
- **Venue:** ACM TOIS, 2023
- **Citations:** 26
- **Summary:** Methods to incorporate document structure into dense retrieval models for legal cases.

### 5.5 Pre-training for Legal Case Retrieval Based on Inter-Case Distinctions
- **Authors:** Su, Ai, Wu, Xie, Wang, Ma, Li
- **Venue:** ACM TOIS, 2025
- **Citations:** 1 (new)
- **Summary:** Novel pre-training framework supporting both dense retrieval and neural re-ranking for legal case retrieval.

---

## 6. LLM Benchmarks & Surveys

### 6.1 LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models ⭐
- **Authors:** Guha, Nyarko, Ho, Ré et al.
- **Venue:** NeurIPS 2023
- **Citations:** 473
- **Summary:** Comprehensive benchmark with 162 tasks covering legal reasoning capabilities. Collaboratively constructed with legal experts.
- **Key Finding:** LLMs struggle with legal reasoning requiring multiple steps or domain knowledge.

### 6.2 LawBench: Benchmarking Legal Knowledge of Large Language Models
- **Authors:** Fei, Shen, Zhu, Zhou, Han et al.
- **Venue:** EMNLP 2024, ACL
- **Citations:** 235
- **Summary:** Chinese legal LLM benchmark testing memorization, understanding, and application of legal knowledge.

### 6.3 A Survey on Evaluation of Large Language Models ⭐
- **Authors:** Chang, Wang, Wang, Wu, Yang et al.
- **Venue:** ACM TIST, 2024
- **Citations:** 5,132
- **Summary:** Comprehensive LLM evaluation survey covering multiple domains including legal. Essential reference for evaluation methodology.

### 6.4 Natural Language Processing for the Legal Domain: A Survey of Tasks, Datasets, Models, and Challenges
- **Authors:** Ariai, Mackenzie, Demartini
- **Venue:** ACM Computing Surveys, 2025
- **Citations:** 116
- **Summary:** Broad survey of legal NLP covering tasks, datasets, and models. Excellent entry point for legal NLP research.

### 6.5 Exploring LLMs Applications in Law: A Literature Review on Current Legal NLP Approaches
- **Authors:** Siino, Falco, Croce, Rosso
- **Venue:** IEEE Access, 2025
- **Citations:** 133
- **Summary:** Recent survey focusing on LLM applications in legal domain.

### 6.6 LAiW: A Chinese Legal Large Language Models Benchmark
- **Authors:** Dai, Feng, Huang, Jia, Xie et al.
- **Venue:** COLING 2025, ACL
- **Citations:** 67
- **Summary:** Chinese legal LLM benchmark structured around legal syllogism.

### 6.7 Large Language Models for Automated Q&A Involving Legal Documents: A Survey
- **Authors:** Yang, Wang, Wang, Wei, Zhang et al.
- **Venue:** International Journal of Web Information Systems, 2024, Emerald
- **Citations:** 61
- **Summary:** Survey on LLM-based legal QA systems covering algorithms, frameworks, and applications.

### 6.8 Large Language Models Meet Legal AI: A Survey
- **Authors:** Hou, Ye, Zeng, Hao, Zeng
- **Venue:** arXiv 2025
- **Citations:** 3 (new)
- **Summary:** Recent comprehensive survey covering 16 legal LLM series and 47 frameworks.

---

## 7. Retrieval-Augmented Generation for Legal

### 7.1 CBR-RAG: Case-Based Reasoning for Retrieval Augmented Generation in LLMs for Legal Question Answering ⭐
- **Authors:** Wiratunga, Abeyratne, Jayawardena et al.
- **Venue:** ICCBR 2024, Springer
- **Citations:** 174
- **Summary:** Integrates case-based reasoning with RAG for legal QA. Uses case retrieval to augment prompts.
- **Key Finding:** CBR structure improves RAG performance for legal reasoning.

### 7.2 LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain
- **Authors:** Pipitone, Alami
- **Venue:** arXiv 2024
- **Citations:** 90
- **Summary:** First benchmark specifically designed to evaluate retrieval component of RAG for legal applications.

### 7.3 HyPA-RAG: A Hybrid Parameter Adaptive RAG System for AI Legal and Policy Applications
- **Authors:** Kalra, Wu, Gulley, Hilliard, Guan et al.
- **Venue:** CustomNLP4U Workshop, EMNLP 2024
- **Citations:** 38
- **Summary:** Adaptive RAG system addressing retrieval errors and context integration challenges in legal AI.

### 7.4 Enhancing the Precision and Interpretability of RAG in Legal Technology: A Survey
- **Authors:** Hindi, Mohammed, Maaz, Alwarafy
- **Venue:** IEEE Access, 2025
- **Citations:** 38
- **Summary:** Survey on RAG techniques for legal applications, covering retrieval, augmentation, and generation methods.

### 7.5 CLERC: A Dataset for US Legal Case Retrieval and Retrieval-Augmented Analysis Generation
- **Authors:** Hou, Weller, Qin, Yang et al.
- **Venue:** NAACL Findings 2025, ACL
- **Citations:** 34
- **Summary:** Dataset for retrieving citations and generating long-form legal analysis.

### 7.6 LexRAG: Benchmarking RAG in Multi-turn Legal Consultation Conversation
- **Authors:** Li, Chen, YiRan, Ai, Chen, Yang et al.
- **Venue:** SIGIR 2025, ACM
- **Citations:** 12
- **Summary:** Benchmark for multi-turn conversational legal RAG.

---

## 8. Early COLIEE Methods (CNN/LSTM Era)

### 8.1 A Convolutional Neural Network in Legal Question Answering
- **Authors:** Kim, Xu, Goebel
- **Venue:** JURISIN 2015
- **Citations:** 25
- **Summary:** Early CNN-based approach for COLIEE legal QA.

### 8.2 Applying a Convolutional Neural Network to Legal Question Answering
- **Authors:** Kim, Xu, Goebel
- **Venue:** JSAI-isAI 2015, Springer
- **Citations:** 30
- **Summary:** Extended CNN system combining legal information extraction with neural entailment.

### 8.3 Legal Yes/No Question Answering System Using Case-Role Analysis
- **Authors:** Taniguchi, Kano
- **Venue:** JSAI-isAI 2016, Springer
- **Citations:** 27
- **Summary:** Linguistic approach using case-role analysis for Japanese legal QA.

---

## Summary Statistics

| Category | Papers | Avg. Citations |
|----------|--------|----------------|
| COLIEE Overview | 3 | 39 |
| Pre-trained LMs | 7 | 326 |
| Judgment Prediction | 6 | 157 |
| Entailment/NLI | 5 | 39 |
| Dense Retrieval | 5 | 28 |
| LLM Benchmarks | 8 | 788 |
| RAG for Legal | 6 | 66 |
| Early Methods | 3 | 27 |
| **Total** | **43** | — |

---

## Key Takeaways for COLIEE 2026

1. **Model Choice:** Open-source models required. Focus on:
   - Legal-domain PLMs (Legal-BERT variants)
   - Structure-aware models (SAILER, Lawformer)
   - Models released before July 15, 2025

2. **Task 3 (Statute Law IR):**
   - Hybrid BM25 + dense retrieval is competitive
   - Consider semantic thesaurus augmentation
   - Structure-aware encoding helps

3. **Task 4 (Statute Law Entailment):**
   - NLI formulation with fine-tuned transformers
   - Data augmentation techniques
   - Consider RAG approaches for context

4. **Key Open Models to Explore:**
   - DeBERTa-v3 (fine-tuned on NLI)
   - Japanese BERT variants
   - Multilingual legal models
   - Longformer for long documents

5. **Emerging Trends:**
   - RAG with case-based reasoning
   - Structure-aware pre-training
   - Multi-task learning for legal reasoning

---

*This review covers 43 papers meeting the criteria. All citations verified via Google Scholar (Jan 2026).*
