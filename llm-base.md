# LLM / MLLM (Language Models) - Base/Interpretation
**update at 2026-01-25 10:36:50**

Sorted by classifier confidence (high to low).

## **1. AdversaRiskQA: An Adversarial Factuality Benchmark for High-Risk Domains**

cs.CL

13 pages, 4 figures, and 11 tables

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.15511v1) [paper-pdf](https://arxiv.org/pdf/2601.15511v1)

**Confidence**: 0.95

**Authors**: Adam Szelestey, Sofie van Engelen, Tianhao Huang, Justin Snelders, Qintao Zeng, Songgaojun Deng

**Abstract**: Hallucination in large language models (LLMs) remains an acute concern, contributing to the spread of misinformation and diminished public trust, particularly in high-risk domains. Among hallucination types, factuality is crucial, as it concerns a model's alignment with established world knowledge. Adversarial factuality, defined as the deliberate insertion of misinformation into prompts with varying levels of expressed confidence, tests a model's ability to detect and resist confidently framed falsehoods. Existing work lacks high-quality, domain-specific resources for assessing model robustness under such adversarial conditions, and no prior research has examined the impact of injected misinformation on long-form text factuality.   To address this gap, we introduce AdversaRiskQA, the first verified and reliable benchmark systematically evaluating adversarial factuality across Health, Finance, and Law. The benchmark includes two difficulty levels to test LLMs' defensive capabilities across varying knowledge depths. We propose two automated methods for evaluating the adversarial attack success and long-form factuality. We evaluate six open- and closed-source LLMs from the Qwen, GPT-OSS, and GPT families, measuring misinformation detection rates. Long-form factuality is assessed on Qwen3 (30B) under both baseline and adversarial conditions. Results show that after excluding meaningless responses, Qwen3 (80B) achieves the highest average accuracy, while GPT-5 maintains consistently high accuracy. Performance scales non-linearly with model size, varies by domains, and gaps between difficulty levels narrow as models grow. Long-form evaluation reveals no significant correlation between injected misinformation and the model's factual output. AdversaRiskQA provides a valuable benchmark for pinpointing LLM weaknesses and developing more reliable models for high-stakes applications.



## **2. Lightweight LLMs for Network Attack Detection in IoT Networks**

cs.CR

6 pages with 2 figures, This paper was accepted and presented at the 7th Computing, Communications and IoT Applications Conference (ComComAp 2025), held in Madrid, Spain, during 14th to 17th December 2025

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.15269v1) [paper-pdf](https://arxiv.org/pdf/2601.15269v1)

**Confidence**: 0.95

**Authors**: Piyumi Bhagya Sudasinghe, Kushan Sudheera Kalupahana Liyanage, Harsha S. Gardiyawasam Pussewalage

**Abstract**: The rapid growth of Internet of Things (IoT) devices has increased the scale and diversity of cyberattacks, exposing limitations in traditional intrusion detection systems. Classical machine learning (ML) models such as Random Forest and Support Vector Machine perform well on known attacks but require retraining to detect unseen or zero-day threats. This study investigates lightweight decoder-only Large Language Models (LLMs) for IoT attack detection by integrating structured-to-text conversion, Quantized Low-Rank Adaptation (QLoRA) fine-tuning, and Retrieval-Augmented Generation (RAG). Network traffic features are transformed into compact natural-language prompts, enabling efficient adaptation under constrained hardware. Experiments on the CICIoT2023 dataset show that a QLoRA-tuned LLaMA-1B model achieves an F1-score of 0.7124, comparable to the Random Forest (RF) baseline (0.7159) for known attacks. With RAG, the system attains 42.63% accuracy on unseen attack types without additional training, demonstrating practical zero-shot capability. These results highlight the potential of retrieval-enhanced lightweight LLMs as adaptable and resource-efficient solutions for next-generation IoT intrusion detection.



## **3. Turn-Based Structural Triggers: Prompt-Free Backdoors in Multi-Turn LLMs**

cs.CR

**SubmitDate**: 2026-01-20    [abs](http://arxiv.org/abs/2601.14340v1) [paper-pdf](https://arxiv.org/pdf/2601.14340v1)

**Confidence**: 0.95

**Authors**: Yiyang Lu, Jinwen He, Yue Zhao, Kai Chen, Ruigang Liang

**Abstract**: Large Language Models (LLMs) are widely integrated into interactive systems such as dialogue agents and task-oriented assistants. This growing ecosystem also raises supply-chain risks, where adversaries can distribute poisoned models that degrade downstream reliability and user trust. Existing backdoor attacks and defenses are largely prompt-centric, focusing on user-visible triggers while overlooking structural signals in multi-turn conversations. We propose Turn-based Structural Trigger (TST), a backdoor attack that activates from dialogue structure, using the turn index as the trigger and remaining independent of user inputs. Across four widely used open-source LLM models, TST achieves an average attack success rate (ASR) of 99.52% with minimal utility degradation, and remains effective under five representative defenses with an average ASR of 98.04%. The attack also generalizes well across instruction datasets, maintaining an average ASR of 99.19%. Our results suggest that dialogue structure constitutes an important and under-studied attack surface for multi-turn LLM systems, motivating structure-aware auditing and mitigation in practice.



## **4. OI-Bench: An Option Injection Benchmark for Evaluating LLM Susceptibility to Directive Interference**

cs.CL

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2601.13300v1) [paper-pdf](https://arxiv.org/pdf/2601.13300v1)

**Confidence**: 0.95

**Authors**: Yow-Fu Liou, Yu-Chien Tang, Yu-Hsiang Liu, An-Zi Yen

**Abstract**: Benchmarking large language models (LLMs) is critical for understanding their capabilities, limitations, and robustness. In addition to interface artifacts, prior studies have shown that LLM decisions can be influenced by directive signals such as social cues, framing, and instructions. In this work, we introduce option injection, a benchmarking approach that augments the multiple-choice question answering (MCQA) interface with an additional option containing a misleading directive, leveraging standardized choice structure and scalable evaluation. We construct OI-Bench, a benchmark of 3,000 questions spanning knowledge, reasoning, and commonsense tasks, with 16 directive types covering social compliance, bonus framing, threat framing, and instructional interference. This setting combines manipulation of the choice interface with directive-based interference, enabling systematic assessment of model susceptibility. We evaluate 12 LLMs to analyze attack success rates, behavioral responses, and further investigate mitigation strategies ranging from inference-time prompting to post-training alignment. Experimental results reveal substantial vulnerabilities and heterogeneous robustness across models. OI-Bench is expected to support more systematic evaluation of LLM robustness to directive interference within choice-based interfaces.



## **5. Adversarial News and Lost Profits: Manipulating Headlines in LLM-Driven Algorithmic Trading**

cs.CR

This work has been accepted for publication at the IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). The final version will be available on IEEE Xplore

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2601.13082v1) [paper-pdf](https://arxiv.org/pdf/2601.13082v1)

**Confidence**: 0.95

**Authors**: Advije Rizvani, Giovanni Apruzzese, Pavel Laskov

**Abstract**: Large Language Models (LLMs) are increasingly adopted in the financial domain. Their exceptional capabilities to analyse textual data make them well-suited for inferring the sentiment of finance-related news. Such feedback can be leveraged by algorithmic trading systems (ATS) to guide buy/sell decisions. However, this practice bears the risk that a threat actor may craft "adversarial news" intended to mislead an LLM. In particular, the news headline may include "malicious" content that remains invisible to human readers but which is still ingested by the LLM. Although prior work has studied textual adversarial examples, their system-wide impact on LLM-supported ATS has not yet been quantified in terms of monetary risk. To address this threat, we consider an adversary with no direct access to an ATS but able to alter stock-related news headlines on a single day. We evaluate two human-imperceptible manipulations in a financial context: Unicode homoglyph substitutions that misroute models during stock-name recognition, and hidden-text clauses that alter the sentiment of the news headline. We implement a realistic ATS in Backtrader that fuses an LSTM-based price forecast with LLM-derived sentiment (FinBERT, FinGPT, FinLLaMA, and six general-purpose LLMs), and quantify monetary impact using portfolio metrics. Experiments on real-world data show that manipulating a one-day attack over 14 months can reliably mislead LLMs and reduce annual returns by up to 17.7 percentage points. To assess real-world feasibility, we analyze popular scraping libraries and trading platforms and survey 27 FinTech practitioners, confirming our hypotheses. We notified trading platform owners of this security issue.



## **6. On the Evidentiary Limits of Membership Inference for Copyright Auditing**

cs.CR

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2601.12937v1) [paper-pdf](https://arxiv.org/pdf/2601.12937v1)

**Confidence**: 0.95

**Authors**: Murat Bilgehan Ertan, Emirhan Böge, Min Chen, Kaleel Mahmood, Marten van Dijk

**Abstract**: As large language models (LLMs) are trained on increasingly opaque corpora, membership inference attacks (MIAs) have been proposed to audit whether copyrighted texts were used during training, despite growing concerns about their reliability under realistic conditions. We ask whether MIAs can serve as admissible evidence in adversarial copyright disputes where an accused model developer may obfuscate training data while preserving semantic content, and formalize this setting through a judge-prosecutor-accused communication protocol. To test robustness under this protocol, we introduce SAGE (Structure-Aware SAE-Guided Extraction), a paraphrasing framework guided by Sparse Autoencoders (SAEs) that rewrites training data to alter lexical structure while preserving semantic content and downstream utility. Our experiments show that state-of-the-art MIAs degrade when models are fine-tuned on SAGE-generated paraphrases, indicating that their signals are not robust to semantics-preserving transformations. While some leakage remains in certain fine-tuning regimes, these results suggest that MIAs are brittle in adversarial settings and insufficient, on their own, as a standalone mechanism for copyright auditing of LLMs.



## **7. Less Is More -- Until It Breaks: Security Pitfalls of Vision Token Compression in Large Vision-Language Models**

cs.CR

**SubmitDate**: 2026-01-17    [abs](http://arxiv.org/abs/2601.12042v1) [paper-pdf](https://arxiv.org/pdf/2601.12042v1)

**Confidence**: 0.95

**Authors**: Xiaomei Zhang, Zhaoxi Zhang, Leo Yu Zhang, Yanjun Zhang, Guanhong Tao, Shirui Pan

**Abstract**: Visual token compression is widely adopted to improve the inference efficiency of Large Vision-Language Models (LVLMs), enabling their deployment in latency-sensitive and resource-constrained scenarios. However, existing work has mainly focused on efficiency and performance, while the security implications of visual token compression remain largely unexplored. In this work, we first reveal that visual token compression substantially degrades the robustness of LVLMs: models that are robust under uncompressed inference become highly vulnerable once compression is enabled. These vulnerabilities are state-specific; failure modes emerge only in the compressed setting and completely disappear when compression is disabled, making them particularly hidden and difficult to diagnose. By analyzing the key stages of the compression process, we identify instability in token importance ranking as the primary cause of this robustness degradation. Small and imperceptible perturbations can significantly alter token rankings, leading the compression mechanism to mistakenly discard task-critical information and ultimately causing model failure. Motivated by this observation, we propose a Compression-Aware Attack to systematically study and exploit this vulnerability. CAA directly targets the token selection mechanism and induces failures exclusively under compressed inference. We further extend this approach to more realistic black-box settings and introduce Transfer CAA, where neither the target model nor the compression configuration is accessible. We further evaluate potential defenses and find that they provide only limited protection. Extensive experiments across models, datasets, and compression methods show that visual token compression significantly undermines robustness, revealing a previously overlooked efficiency-security trade-off.



## **8. Membership Inference on LLMs in the Wild**

cs.CL

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.11314v1) [paper-pdf](https://arxiv.org/pdf/2601.11314v1)

**Confidence**: 0.95

**Authors**: Jiatong Yi, Yanyang Li

**Abstract**: Membership Inference Attacks (MIAs) act as a crucial auditing tool for the opaque training data of Large Language Models (LLMs). However, existing techniques predominantly rely on inaccessible model internals (e.g., logits) or suffer from poor generalization across domains in strict black-box settings where only generated text is available. In this work, we propose SimMIA, a robust MIA framework tailored for this text-only regime by leveraging an advanced sampling strategy and scoring mechanism. Furthermore, we present WikiMIA-25, a new benchmark curated to evaluate MIA performance on modern proprietary LLMs. Experiments demonstrate that SimMIA achieves state-of-the-art results in the black-box setting, rivaling baselines that exploit internal model information.



## **9. CoSPED: Consistent Soft Prompt Targeted Data Extraction and Defense**

cs.CR

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2510.11137v3) [paper-pdf](https://arxiv.org/pdf/2510.11137v3)

**Confidence**: 0.95

**Authors**: Zhuochen Yang, Kar Wai Fok, Vrizlynn L. L. Thing

**Abstract**: Large language models have gained widespread attention recently, but their potential security vulnerabilities, especially privacy leakage, are also becoming apparent. To test and evaluate for data extraction risks in LLM, we proposed CoSPED, short for Consistent Soft Prompt targeted data Extraction and Defense. We introduce several innovative components, including Dynamic Loss, Additive Loss, Common Loss, and Self Consistency Decoding Strategy, and tested to enhance the consistency of the soft prompt tuning process. Through extensive experimentation with various combinations, we achieved an extraction rate of 65.2% at a 50-token prefix comparison. Our comparisons of CoSPED with other reference works confirm our superior extraction rates. We evaluate CoSPED on more scenarios, achieving Pythia model extraction rate of 51.7% and introducing cross-model comparison. Finally, we explore defense through Rank-One Model Editing and achieve a reduction in the extraction rate to 1.6%, which proves that our analysis of extraction mechanisms can directly inform effective mitigation strategies against soft prompt-based attacks.



## **10. Membership Inference Attacks on LLM-based Recommender Systems**

cs.IR

This is paper is under review ACL 2026

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2508.18665v5) [paper-pdf](https://arxiv.org/pdf/2508.18665v5)

**Confidence**: 0.95

**Authors**: Jiajie He, Min-Chun Chen, Xintong Chen, Xinyang Fang, Yuechun Gu, Keke Chen

**Abstract**: Large language models (LLMs) based recommender systems (RecSys) can adapt to different domains flexibly. It utilizes in-context learning (ICL), i.e., prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, encompassing implicit feedback such as clicked items and explicit product reviews. Such private information may be exposed by novel privacy attacks. However, no study has been conducted on this important issue. We design several membership inference attacks (MIAs) aimed to revealing whether system prompts include victims' historical interactions. The attacks are \emph{Similarity, Memorization, Inquiry, and Poisoning attacks}, each utilizing unique features of LLMs or RecSys. We have carefully evaluated them on five of the latest open-source LLMs and three well-known RecSys benchmark datasets. The results confirm that the MIA threat to LLM RecSys is realistic: inquiry and poisoning attacks show significantly high attack advantages. We also discussed possible methods to mitigate such MIA threats. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts, the position of the victim in the shots, the number of poisoning items in the prompt,etc.



## **11. EVADE-Bench: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications**

cs.CL

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2505.17654v3) [paper-pdf](https://arxiv.org/pdf/2505.17654v3)

**Confidence**: 0.95

**Authors**: Ancheng Xu, Zhihao Yang, Jingpeng Li, Guanghu Yuan, Longze Chen, Liang Yan, Jiehui Zhou, Zhen Qin, Hengyu Chang, Hamid Alinejad-Rokny, Min Yang

**Abstract**: E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision-Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content: inputs (text or images) that superficially comply with platform policies while covertly conveying prohibited claims. Unlike traditional adversarial attacks that induce overt failures, evasive content exploits ambiguity and context, making it far harder to detect. Existing robustness benchmarks provide little guidance for this demanding, real-world challenge. We introduce EVADE, the first expert-curated, Chinese, multimodal benchmark specifically designed to evaluate foundation models on evasive content detection in e-commerce. The dataset contains 2,833 annotated text samples and 13,961 images spanning six demanding product categories, including body shaping, height growth, and health supplements. Two complementary tasks assess distinct capabilities: Single-Violation, which probes fine-grained reasoning under short prompts, and All-in-One, which tests long-context reasoning by merging overlapping policy rules into unified instructions. Notably, the All-in-One setting significantly narrows the performance gap between partial and full-match accuracy, suggesting that clearer rule definitions improve alignment between human and model judgment. We benchmark 26 mainstream LLMs and VLMs and observe substantial performance gaps: even state-of-the-art models frequently misclassify evasive samples. By releasing EVADE and strong baselines, we provide the first rigorous standard for evaluating evasive-content detection, expose fundamental limitations in current multimodal reasoning, and lay the groundwork for safer and more transparent content moderation systems in e-commerce. The dataset is publicly available at https://huggingface.co/datasets/koenshen/EVADE-Bench.



## **12. FDLLM: A Dedicated Detector for Black-Box LLMs Fingerprinting**

cs.CR

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2501.16029v4) [paper-pdf](https://arxiv.org/pdf/2501.16029v4)

**Confidence**: 0.95

**Authors**: Zhiyuan Fu, Junfan Chen, Lan Zhang, Ting Yang, Jun Niu, Hongyu Sun, Ruidong Li, Peng Liu, Jice Wang, Fannv He, Qiuling Yue, Yuqing Zhang

**Abstract**: Large Language Models (LLMs) are rapidly transforming the landscape of digital content creation. However, the prevalent black-box Application Programming Interface (API) access to many LLMs introduces significant challenges in accountability, governance, and security. LLM fingerprinting, which aims to identify the source model by analyzing statistical and stylistic features of generated text, offers a potential solution. Current progress in this area is hindered by a lack of dedicated datasets and the need for efficient, practical methods that are robust against adversarial manipulations. To address these challenges, we introduce FD-Dataset, a comprehensive bilingual fingerprinting benchmark comprising 90,000 text samples from 20 famous proprietary and open-source LLMs. Furthermore, we present FDLLM, a novel fingerprinting method that leverages parameter-efficient Low-Rank Adaptation (LoRA) to fine-tune a foundation model. This approach enables LoRA to extract deep, persistent features that characterize each source LLM. Through our analysis, we find that LoRA adaptation promotes the aggregation of outputs from the same LLM in representation space while enhancing the separation between different LLMs. This mechanism explains why LoRA proves particularly effective for LLM fingerprinting. Extensive empirical evaluations on FD-Dataset demonstrate FDLLM's superiority, achieving a Macro F1 score 22.1% higher than the strongest baseline. FDLLM also exhibits strong generalization to newly released models, achieving an average accuracy of 95% on unseen models. Notably, FDLLM remains consistently robust under various adversarial attacks, including polishing, translation, and synonym substitution. Experimental results show that FDLLM reduces the average attack success rate from 49.2% (LM-D) to 23.9%.



## **13. Undesirable Memorization in Large Language Models: A Survey**

cs.CL

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2410.02650v3) [paper-pdf](https://arxiv.org/pdf/2410.02650v3)

**Confidence**: 0.95

**Authors**: Ali Satvaty, Suzan Verberne, Fatih Turkmen

**Abstract**: While recent research increasingly showcases the remarkable capabilities of Large Language Models (LLMs), it is equally crucial to examine their associated risks. Among these, privacy and security vulnerabilities are particularly concerning, posing significant ethical and legal challenges. At the heart of these vulnerabilities stands memorization, which refers to a model's tendency to store and reproduce phrases from its training data. This phenomenon has been shown to be a fundamental source to various privacy and security attacks against LLMs. In this paper, we provide a taxonomy of the literature on LLM memorization, exploring it across three dimensions: granularity, retrievability, and desirability. Next, we discuss the metrics and methods used to quantify memorization, followed by an analysis of the causes and factors that contribute to memorization phenomenon. We then explore strategies that are used so far to mitigate the undesirable aspects of this phenomenon. We conclude our survey by identifying potential research topics for the near future, including methods to balance privacy and performance, and the analysis of memorization in specific LLM contexts such as conversational agents, retrieval-augmented generation, and diffusion language models. Given the rapid research pace in this field, we also maintain a dedicated repository of the references discussed in this survey which will be regularly updated to reflect the latest developments.



## **14. BenchOverflow: Measuring Overflow in Large Language Models via Plain-Text Prompts**

cs.CL

Accepted at TMLR 2026

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08490v1) [paper-pdf](https://arxiv.org/pdf/2601.08490v1)

**Confidence**: 0.95

**Authors**: Erin Feiglin, Nir Hutnik, Raz Lapid

**Abstract**: We investigate a failure mode of large language models (LLMs) in which plain-text prompts elicit excessive outputs, a phenomenon we term Overflow. Unlike jailbreaks or prompt injection, Overflow arises under ordinary interaction settings and can lead to elevated serving cost, latency, and cross-user performance degradation, particularly when scaled across many requests. Beyond usability, the stakes are economic and environmental: unnecessary tokens increase per-request cost and energy consumption, compounding into substantial operational spend and carbon footprint at scale. Moreover, Overflow represents a practical vector for compute amplification and service degradation in shared environments. We introduce BenchOverflow, a model-agnostic benchmark of nine plain-text prompting strategies that amplify output volume without adversarial suffixes or policy circumvention. Using a standardized protocol with a fixed budget of 5000 new tokens, we evaluate nine open- and closed-source models and observe pronounced rightward shifts and heavy tails in length distributions. Cap-saturation rates (CSR@1k/3k/5k) and empirical cumulative distribution functions (ECDFs) quantify tail risk; within-prompt variance and cross-model correlations show that Overflow is broadly reproducible yet heterogeneous across families and attack vectors. A lightweight mitigation-a fixed conciseness reminder-attenuates right tails and lowers CSR for all strategies across the majority of models. Our findings position length control as a measurable reliability, cost, and sustainability concern rather than a stylistic quirk. By enabling standardized comparison of length-control robustness across models, BenchOverflow provides a practical basis for selecting deployments that minimize resource waste and operating expense, and for evaluating defenses that curb compute amplification without eroding task performance.



## **15. A Semantic Decoupling-Based Two-Stage Rainy-Day Attack for Revealing Weather Robustness Deficiencies in Vision-Language Models**

cs.CV

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2601.13238v1) [paper-pdf](https://arxiv.org/pdf/2601.13238v1)

**Confidence**: 0.95

**Authors**: Chengyin Hu, Xiang Chen, Zhe Jia, Weiwen Shi, Fengyu Zhang, Jiujiang Guo, Yiwei Wei

**Abstract**: Vision-Language Models (VLMs) are trained on image-text pairs collected under canonical visual conditions and achieve strong performance on multimodal tasks. However, their robustness to real-world weather conditions, and the stability of cross-modal semantic alignment under such structured perturbations, remain insufficiently studied. In this paper, we focus on rainy scenarios and introduce the first adversarial framework that exploits realistic weather to attack VLMs, using a two-stage, parameterized perturbation model based on semantic decoupling to analyze rain-induced shifts in decision-making. In Stage 1, we model the global effects of rainfall by applying a low-dimensional global modulation to condition the embedding space and gradually weaken the original semantic decision boundaries. In Stage 2, we introduce structured rain variations by explicitly modeling multi-scale raindrop appearance and rainfall-induced illumination changes, and optimize the resulting non-differentiable weather space to induce stable semantic shifts. Operating in a non-pixel parameter space, our framework generates perturbations that are both physically grounded and interpretable. Experiments across multiple tasks show that even physically plausible, highly constrained weather perturbations can induce substantial semantic misalignment in mainstream VLMs, posing potential safety and reliability risks in real-world deployment. Ablations further confirm that illumination modeling and multi-scale raindrop structures are key drivers of these semantic shifts.



## **16. Hierarchical Refinement of Universal Multimodal Attacks on Vision-Language Models**

cs.CV

15 pages, 7 figures

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10313v1) [paper-pdf](https://arxiv.org/pdf/2601.10313v1)

**Confidence**: 0.95

**Authors**: Peng-Fei Zhang, Zi Huang

**Abstract**: Existing adversarial attacks for VLP models are mostly sample-specific, resulting in substantial computational overhead when scaled to large datasets or new scenarios. To overcome this limitation, we propose Hierarchical Refinement Attack (HRA), a multimodal universal attack framework for VLP models. HRA refines universal adversarial perturbations (UAPs) at both the sample level and the optimization level. For the image modality, we disentangle adversarial examples into clean images and perturbations, allowing each component to be handled independently for more effective disruption of cross-modal alignment. We further introduce a ScMix augmentation strategy that diversifies visual contexts and strengthens both global and local utility of UAPs, thereby reducing reliance on spurious features. In addition, we refine the optimization path by leveraging a temporal hierarchy of historical and estimated future gradients to avoid local minima and stabilize universal perturbation learning. For the text modality, HRA identifies globally influential words by combining intra-sentence and inter-sentence importance measures, and subsequently utilizes these words as universal text perturbations. Extensive experiments across various downstream tasks, VLP models, and datasets demonstrate the superiority of the proposed universal multimodal attacks.



## **17. Paraphrasing Adversarial Attack on LLM-as-a-Reviewer**

cs.CL

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2601.06884v1) [paper-pdf](https://arxiv.org/pdf/2601.06884v1)

**Confidence**: 0.95

**Authors**: Masahiro Kaneko

**Abstract**: The use of large language models (LLMs) in peer review systems has attracted growing attention, making it essential to examine their potential vulnerabilities. Prior attacks rely on prompt injection, which alters manuscript content and conflates injection susceptibility with evaluation robustness. We propose the Paraphrasing Adversarial Attack (PAA), a black-box optimization method that searches for paraphrased sequences yielding higher review scores while preserving semantic equivalence and linguistic naturalness. PAA leverages in-context learning, using previous paraphrases and their scores to guide candidate generation. Experiments across five ML and NLP conferences with three LLM reviewers and five attacking models show that PAA consistently increases review scores without changing the paper's claims. Human evaluation confirms that generated paraphrases maintain meaning and naturalness. We also find that attacked papers exhibit increased perplexity in reviews, offering a potential detection signal, and that paraphrasing submissions can partially mitigate attacks.



## **18. State Backdoor: Towards Stealthy Real-world Poisoning Attack on Vision-Language-Action Model in State Space**

cs.CR

**SubmitDate**: 2026-01-07    [abs](http://arxiv.org/abs/2601.04266v1) [paper-pdf](https://arxiv.org/pdf/2601.04266v1)

**Confidence**: 0.95

**Authors**: Ji Guo, Wenbo Jiang, Yansong Lin, Yijing Liu, Ruichen Zhang, Guomin Lu, Aiguo Chen, Xinshuo Han, Hongwei Li, Dusit Niyato

**Abstract**: Vision-Language-Action (VLA) models are widely deployed in safety-critical embodied AI applications such as robotics. However, their complex multimodal interactions also expose new security vulnerabilities. In this paper, we investigate a backdoor threat in VLA models, where malicious inputs cause targeted misbehavior while preserving performance on clean data. Existing backdoor methods predominantly rely on inserting visible triggers into visual modality, which suffer from poor robustness and low insusceptibility in real-world settings due to environmental variability. To overcome these limitations, we introduce the State Backdoor, a novel and practical backdoor attack that leverages the robot arm's initial state as the trigger. To optimize trigger for insusceptibility and effectiveness, we design a Preference-guided Genetic Algorithm (PGA) that efficiently searches the state space for minimal yet potent triggers. Extensive experiments on five representative VLA models and five real-world tasks show that our method achieves over 90% attack success rate without affecting benign task performance, revealing an underexplored vulnerability in embodied AI systems.



## **19. Hidden State Poisoning Attacks against Mamba-based Language Models**

cs.CL

17 pages, 4 figures

**SubmitDate**: 2026-01-06    [abs](http://arxiv.org/abs/2601.01972v2) [paper-pdf](https://arxiv.org/pdf/2601.01972v2)

**Confidence**: 0.95

**Authors**: Alexandre Le Mercier, Chris Develder, Thomas Demeester

**Abstract**: State space models (SSMs) like Mamba offer efficient alternatives to Transformer-based language models, with linear time complexity. Yet, their adversarial robustness remains critically unexplored. This paper studies the phenomenon whereby specific short input phrases induce a partial amnesia effect in such models, by irreversibly overwriting information in their hidden states, referred to as a Hidden State Poisoning Attack (HiSPA). Our benchmark RoBench25 allows evaluating a model's information retrieval capabilities when subject to HiSPAs, and confirms the vulnerability of SSMs against such attacks. Even a recent 52B hybrid SSM-Transformer model from the Jamba family collapses on RoBench25 under optimized HiSPA triggers, whereas pure Transformers do not. We also observe that HiSPA triggers significantly weaken the Jamba model on the popular Open-Prompt-Injections benchmark, unlike pure Transformers. Finally, our interpretability study reveals patterns in Mamba's hidden layers during HiSPAs that could be used to build a HiSPA mitigation system. The full code and data to reproduce the experiments can be found at https://anonymous.4open.science/r/hispa_anonymous-5DB0.



## **20. Multilingual Hidden Prompt Injection Attacks on LLM-Based Academic Reviewing**

cs.CL

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23684v1) [paper-pdf](https://arxiv.org/pdf/2512.23684v1)

**Confidence**: 0.95

**Authors**: Panagiotis Theocharopoulos, Ajinkya Kulkarni, Mathew Magimai. -Doss

**Abstract**: Large language models (LLMs) are increasingly considered for use in high-impact workflows, including academic peer review. However, LLMs are vulnerable to document-level hidden prompt injection attacks. In this work, we construct a dataset of approximately 500 real academic papers accepted to ICML and evaluate the effect of embedding hidden adversarial prompts within these documents. Each paper is injected with semantically equivalent instructions in four different languages and reviewed using an LLM. We find that prompt injection induces substantial changes in review scores and accept/reject decisions for English, Japanese, and Chinese injections, while Arabic injections produce little to no effect. These results highlight the susceptibility of LLM-based reviewing systems to document-level prompt injection and reveal notable differences in vulnerability across languages.



## **21. NeuroGenPoisoning: Neuron-Guided Attacks on Retrieval-Augmented Generation of LLM via Genetic Optimization of External Knowledge**

cs.AI

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2510.21144v2) [paper-pdf](https://arxiv.org/pdf/2510.21144v2)

**Confidence**: 0.95

**Authors**: Hanyu Zhu, Lance Fiondella, Jiawei Yuan, Kai Zeng, Long Jiao

**Abstract**: Retrieval-Augmented Generation (RAG) empowers Large Language Models (LLMs) to dynamically integrate external knowledge during inference, improving their factual accuracy and adaptability. However, adversaries can inject poisoned external knowledge to override the model's internal memory. While existing attacks iteratively manipulate retrieval content or prompt structure of RAG, they largely ignore the model's internal representation dynamics and neuron-level sensitivities. The underlying mechanism of RAG poisoning has not been fully studied and the effect of knowledge conflict with strong parametric knowledge in RAG is not considered. In this work, we propose NeuroGenPoisoning, a novel attack framework that generates adversarial external knowledge in RAG guided by LLM internal neuron attribution and genetic optimization. Our method first identifies a set of Poison-Responsive Neurons whose activation strongly correlates with contextual poisoning knowledge. We then employ a genetic algorithm to evolve adversarial passages that maximally activate these neurons. Crucially, our framework enables massive-scale generation of effective poisoned RAG knowledge by identifying and reusing promising but initially unsuccessful external knowledge variants via observed attribution signals. At the same time, Poison-Responsive Neurons guided poisoning can effectively resolves knowledge conflict. Experimental results across models and datasets demonstrate consistently achieving high Population Overwrite Success Rate (POSR) of over 90% while preserving fluency. Empirical evidence shows that our method effectively resolves knowledge conflict.



## **22. Illusions of Relevance: Arbitrary Content Injection Attacks Deceive Retrievers, Rerankers, and LLM Judges**

cs.IR

AACL Findings 2025

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2501.18536v2) [paper-pdf](https://arxiv.org/pdf/2501.18536v2)

**Confidence**: 0.95

**Authors**: Manveer Singh Tamber, Jimmy Lin

**Abstract**: This work considers a black-box threat model in which adversaries attempt to propagate arbitrary non-relevant content in search. We show that retrievers, rerankers, and LLM relevance judges are all highly vulnerable to attacks that enable arbitrary content to be promoted to the top of search results and to be assigned perfect relevance scores. We investigate how attackers may achieve this via content injection, injecting arbitrary sentences into relevant passages or query terms into arbitrary passages. Our study analyzes how factors such as model class and size, the balance between relevant and non-relevant content, injection location, toxicity and severity of injected content, and the role of LLM-generated content influence attack success, yielding novel, concerning, and often counterintuitive results. Our results reveal a weakness in embedding models, LLM-based scoring models, and generative LLMs, raising concerns about the general robustness, safety, and trustworthiness of language models regardless of the type of model or the role in which they are employed. We also emphasize the challenges of robust defenses against these attacks. Classifiers and more carefully prompted LLM judges often fail to recognize passages with content injection, especially when considering diverse text topics and styles. Our findings highlight the need for further research into arbitrary content injection attacks. We release our code for further study.



## **23. Transferability of Adversarial Attacks in Video-based MLLMs: A Cross-modal Image-to-Video Approach**

cs.CV

**SubmitDate**: 2026-01-09    [abs](http://arxiv.org/abs/2501.01042v4) [paper-pdf](https://arxiv.org/pdf/2501.01042v4)

**Confidence**: 0.95

**Authors**: Linhao Huang, Xue Jiang, Zhiqiang Wang, Wentao Mo, Xi Xiao, Yong-Jie Yin, Bo Han, Feng Zheng

**Abstract**: Video-based multimodal large language models (V-MLLMs) have shown vulnerability to adversarial examples in video-text multimodal tasks. However, the transferability of adversarial videos to unseen models - a common and practical real-world scenario - remains unexplored. In this paper, we pioneer an investigation into the transferability of adversarial video samples across V-MLLMs. We find that existing adversarial attack methods face significant limitations when applied in black-box settings for V-MLLMs, which we attribute to the following shortcomings: (1) lacking generalization in perturbing video features, (2) focusing only on sparse key-frames, and (3) failing to integrate multimodal information. To address these limitations and deepen the understanding of V-MLLM vulnerabilities in black-box scenarios, we introduce the Image-to-Video MLLM (I2V-MLLM) attack. In I2V-MLLM, we utilize an image-based multimodal large language model (I-MLLM) as a surrogate model to craft adversarial video samples. Multimodal interactions and spatiotemporal information are integrated to disrupt video representations within the latent space, improving adversarial transferability. Additionally, a perturbation propagation technique is introduced to handle different unknown frame sampling strategies. Experimental results demonstrate that our method can generate adversarial examples that exhibit strong transferability across different V-MLLMs on multiple video-text multimodal tasks. Compared to white-box attacks on these models, our black-box attacks (using BLIP-2 as a surrogate model) achieve competitive performance, with average attack success rate (AASR) of 57.98% on MSVD-QA and 58.26% on MSRVTT-QA for Zero-Shot VideoQA tasks, respectively.



## **24. On the Adversarial Robustness of 3D Large Vision-Language Models**

cs.CV

Under Review

**SubmitDate**: 2026-01-10    [abs](http://arxiv.org/abs/2601.06464v1) [paper-pdf](https://arxiv.org/pdf/2601.06464v1)

**Confidence**: 0.95

**Authors**: Chao Liu, Ngai-Man Cheung

**Abstract**: 3D Vision-Language Models (VLMs), such as PointLLM and GPT4Point, have shown strong reasoning and generalization abilities in 3D understanding tasks. However, their adversarial robustness remains largely unexplored. Prior work in 2D VLMs has shown that the integration of visual inputs significantly increases vulnerability to adversarial attacks, making these models easier to manipulate into generating toxic or misleading outputs. In this paper, we investigate whether incorporating 3D vision similarly compromises the robustness of 3D VLMs. To this end, we present the first systematic study of adversarial robustness in point-based 3D VLMs. We propose two complementary attack strategies: \textit{Vision Attack}, which perturbs the visual token features produced by the 3D encoder and projector to assess the robustness of vision-language alignment; and \textit{Caption Attack}, which directly manipulates output token sequences to evaluate end-to-end system robustness. Each attack includes both untargeted and targeted variants to measure general vulnerability and susceptibility to controlled manipulation. Our experiments reveal that 3D VLMs exhibit significant adversarial vulnerabilities under untargeted attacks, while demonstrating greater resilience against targeted attacks aimed at forcing specific harmful outputs, compared to their 2D counterparts. These findings highlight the importance of improving the adversarial robustness of 3D VLMs, especially as they are deployed in safety-critical applications.



## **25. FlipLLM: Efficient Bit-Flip Attacks on Multimodal LLMs using Reinforcement Learning**

cs.CR

Accepted in IEEE HOST 2026

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2512.09872v1) [paper-pdf](https://arxiv.org/pdf/2512.09872v1)

**Confidence**: 0.95

**Authors**: Khurram Khalil, Khaza Anuarul Hoque

**Abstract**: Generative Artificial Intelligence models, such as Large Language Models (LLMs) and Large Vision Models (VLMs), exhibit state-of-the-art performance but remain vulnerable to hardware-based threats, specifically bit-flip attacks (BFAs). Existing BFA discovery methods lack generalizability and struggle to scale, often failing to analyze the vast parameter space and complex interdependencies of modern foundation models in a reasonable time. This paper proposes FlipLLM, a reinforcement learning (RL) architecture-agnostic framework that formulates BFA discovery as a sequential decision-making problem. FlipLLM combines sensitivity-guided layer pruning with Q-learning to efficiently identify minimal, high-impact bit sets that can induce catastrophic failure. We demonstrate the effectiveness and generalizability of FlipLLM by applying it to a diverse set of models, including prominent text-only LLMs (GPT-2 Large, LLaMA 3.1 8B, and DeepSeek-V2 7B), VLMs such as LLaVA 1.6, and datasets, such as MMLU, MMLU-Pro, VQAv2, and TextVQA. Our results show that FlipLLM can identify critical bits that are vulnerable to BFAs up to 2.5x faster than SOTA methods. We demonstrate that flipping the FlipLLM-identified bits plummets the accuracy of LLaMA 3.1 8B from 69.9% to ~0.2%, and for LLaVA's VQA score from 78% to almost 0%, by flipping as few as 5 and 7 bits, respectively. Further analysis reveals that applying standard hardware protection mechanisms, such as ECC SECDED, to the FlipLLM-identified bit locations completely mitigates the BFA impact, demonstrating the practical value of our framework in guiding hardware-level defenses. FlipLLM offers the first scalable and adaptive methodology for exploring the BFA vulnerability of both language and multimodal foundation models, paving the way for comprehensive hardware-security evaluation.



## **26. Read or Ignore? A Unified Benchmark for Typographic-Attack Robustness and Text Recognition in Vision-Language Models**

cs.CV

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2512.11899v1) [paper-pdf](https://arxiv.org/pdf/2512.11899v1)

**Confidence**: 0.95

**Authors**: Futa Waseda, Shojiro Yamabe, Daiki Shiono, Kento Sasaki, Tsubasa Takahashi

**Abstract**: Large vision-language models (LVLMs) are vulnerable to typographic attacks, where misleading text within an image overrides visual understanding. Existing evaluation protocols and defenses, largely focused on object recognition, implicitly encourage ignoring text to achieve robustness; however, real-world scenarios often require joint reasoning over both objects and text (e.g., recognizing pedestrians while reading traffic signs). To address this, we introduce a novel task, Read-or-Ignore VQA (RIO-VQA), which formalizes selective text use in visual question answering (VQA): models must decide, from context, when to read text and when to ignore it. For evaluation, we present the Read-or-Ignore Benchmark (RIO-Bench), a standardized dataset and protocol that, for each real image, provides same-scene counterfactuals (read / ignore) by varying only the textual content and question type. Using RIO-Bench, we show that strong LVLMs and existing defenses fail to balance typographic robustness and text-reading capability, highlighting the need for improved approaches. Finally, RIO-Bench enables a novel data-driven defense that learns adaptive selective text use, moving beyond prior non-adaptive, text-ignoring defenses. Overall, this work reveals a fundamental misalignment between the existing evaluation scope and real-world requirements, providing a principled path toward reliable LVLMs. Our Project Page is at https://turingmotors.github.io/rio-vqa/.



## **27. When Ads Become Profiles: Uncovering the Invisible Risk of Web Advertising at Scale with LLMs**

cs.HC

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2509.18874v2) [paper-pdf](https://arxiv.org/pdf/2509.18874v2)

**Confidence**: 0.95

**Authors**: Baiyu Chen, Benjamin Tag, Hao Xue, Daniel Angus, Flora Salim

**Abstract**: Regulatory limits on explicit targeting have not eliminated algorithmic profiling on the Web, as optimisation systems still adapt ad delivery to users' private attributes. The widespread availability of powerful zero-shot multimodal Large Language Models (LLMs) has dramatically lowered the barrier for exploiting these latent signals for adversarial inference. We investigate this emerging societal risk, specifically how adversaries can now exploit these signals to reverse-engineer private attributes from ad exposure alone. We introduce a novel pipeline that leverages LLMs as adversarial inference engines to perform natural language profiling. Applying this method to a longitudinal dataset comprising over 435,000 ad impressions collected from 891 users, we conducted a large-scale study to assess the feasibility and precision of inferring private attributes from passive online ad observations. Our results demonstrate that off-the-shelf LLMs can accurately reconstruct complex user private attributes, including party preference, employment status, and education level, consistently outperforming strong census-based priors and matching or exceeding human social perception, while operating at only a fraction of the cost (223$\times$ lower) and time (52$\times$ faster) required by humans. Critically, actionable profiling is feasible even within short observation windows, indicating that prolonged tracking is not a prerequisite for a successful attack. These findings provide the first empirical evidence that ad streams serve as a high-fidelity digital footprint, enabling off-platform profiling that inherently bypasses current platform safeguards, highlighting a systemic vulnerability in the ad ecosystem and the urgent need for responsible web AI governance in the generative AI era. The code is available at https://github.com/Breezelled/when-ads-become-profiles.



## **28. When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models**

cs.CV

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2511.21192v2) [paper-pdf](https://arxiv.org/pdf/2511.21192v2)

**Confidence**: 0.95

**Authors**: Hui Lu, Yi Yu, Yiming Yang, Chenyu Yi, Qixin Zhang, Bingquan Shen, Alex C. Kot, Xudong Jiang

**Abstract**: Vision-Language-Action (VLA) models are vulnerable to adversarial attacks, yet universal and transferable attacks remain underexplored, as most existing patches overfit to a single model and fail in black-box settings. To address this gap, we present a systematic study of universal, transferable adversarial patches against VLA-driven robots under unknown architectures, finetuned variants, and sim-to-real shifts. We introduce UPA-RFAS (Universal Patch Attack via Robust Feature, Attention, and Semantics), a unified framework that learns a single physical patch in a shared feature space while promoting cross-model transfer. UPA-RFAS combines (i) a feature-space objective with an $\ell_1$ deviation prior and repulsive InfoNCE loss to induce transferable representation shifts, (ii) a robustness-augmented two-phase min-max procedure where an inner loop learns invisible sample-wise perturbations and an outer loop optimizes the universal patch against this hardened neighborhood, and (iii) two VLA-specific losses: Patch Attention Dominance to hijack text$\to$vision attention and Patch Semantic Misalignment to induce image-text mismatch without labels. Experiments across diverse VLA models, manipulation suites, and physical executions show that UPA-RFAS consistently transfers across models, tasks, and viewpoints, exposing a practical patch-based attack surface and establishing a strong baseline for future defenses.



## **29. Universal Adversarial Suffixes Using Calibrated Gumbel-Softmax Relaxation**

cs.CL

10 pages

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08123v1) [paper-pdf](https://arxiv.org/pdf/2512.08123v1)

**Confidence**: 0.95

**Authors**: Sampriti Soor, Suklav Ghosh, Arijit Sur

**Abstract**: Language models (LMs) are often used as zero-shot or few-shot classifiers by scoring label words, but they remain fragile to adversarial prompts. Prior work typically optimizes task- or model-specific triggers, making results difficult to compare and limiting transferability. We study universal adversarial suffixes: short token sequences (4-10 tokens) that, when appended to any input, broadly reduce accuracy across tasks and models. Our approach learns the suffix in a differentiable "soft" form using Gumbel-Softmax relaxation and then discretizes it for inference. Training maximizes calibrated cross-entropy on the label region while masking gold tokens to prevent trivial leakage, with entropy regularization to avoid collapse. A single suffix trained on one model transfers effectively to others, consistently lowering both accuracy and calibrated confidence. Experiments on sentiment analysis, natural language inference, paraphrase detection, commonsense QA, and physical reasoning with Qwen2-1.5B, Phi-1.5, and TinyLlama-1.1B demonstrate consistent attack effectiveness and transfer across tasks and model families.



## **30. Survey of Adversarial Robustness in Multimodal Large Language Models**

cs.CV

9 pages

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.13962v1) [paper-pdf](https://arxiv.org/pdf/2503.13962v1)

**Confidence**: 0.95

**Authors**: Chengze Jiang, Zhuangzhuang Wang, Minjing Dong, Jie Gui

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated exceptional performance in artificial intelligence by facilitating integrated understanding across diverse modalities, including text, images, video, audio, and speech. However, their deployment in real-world applications raises significant concerns about adversarial vulnerabilities that could compromise their safety and reliability. Unlike unimodal models, MLLMs face unique challenges due to the interdependencies among modalities, making them susceptible to modality-specific threats and cross-modal adversarial manipulations. This paper reviews the adversarial robustness of MLLMs, covering different modalities. We begin with an overview of MLLMs and a taxonomy of adversarial attacks tailored to each modality. Next, we review key datasets and evaluation metrics used to assess the robustness of MLLMs. After that, we provide an in-depth review of attacks targeting MLLMs across different modalities. Our survey also identifies critical challenges and suggests promising future research directions.



## **31. Securing LLM-as-a-Service for Small Businesses: An Industry Case Study of a Distributed Chatbot Deployment Platform**

cs.DC

Accepted by AISC 2026

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.15528v1) [paper-pdf](https://arxiv.org/pdf/2601.15528v1)

**Confidence**: 0.90

**Authors**: Jiazhu Xie, Bowen Li, Heyu Fu, Chong Gao, Ziqi Xu, Fengling Han

**Abstract**: Large Language Model (LLM)-based question-answering systems offer significant potential for automating customer support and internal knowledge access in small businesses, yet their practical deployment remains challenging due to infrastructure costs, engineering complexity, and security risks, particularly in retrieval-augmented generation (RAG)-based settings. This paper presents an industry case study of an open-source, multi-tenant platform that enables small businesses to deploy customised LLM-based support chatbots via a no-code workflow. The platform is built on distributed, lightweight k3s clusters spanning heterogeneous, low-cost machines and interconnected through an encrypted overlay network, enabling cost-efficient resource pooling while enforcing container-based isolation and per-tenant data access controls. In addition, the platform integrates practical, platform-level defences against prompt injection attacks in RAG-based chatbots, translating insights from recent prompt injection research into deployable security mechanisms without requiring model retraining or enterprise-scale infrastructure. We evaluate the proposed platform through a real-world e-commerce deployment, demonstrating that secure and efficient LLM-based chatbot services can be achieved under realistic cost, operational, and security constraints faced by small businesses.



## **32. SilentDrift: Exploiting Action Chunking for Stealthy Backdoor Attacks on Vision-Language-Action Models**

cs.CR

**SubmitDate**: 2026-01-20    [abs](http://arxiv.org/abs/2601.14323v1) [paper-pdf](https://arxiv.org/pdf/2601.14323v1)

**Confidence**: 0.90

**Authors**: Bingxin Xu, Yuzhang Shang, Binghui Wang, Emilio Ferrara

**Abstract**: Vision-Language-Action (VLA) models are increasingly deployed in safety-critical robotic applications, yet their security vulnerabilities remain underexplored. We identify a fundamental security flaw in modern VLA systems: the combination of action chunking and delta pose representations creates an intra-chunk visual open-loop. This mechanism forces the robot to execute K-step action sequences, allowing per-step perturbations to accumulate through integration. We propose SILENTDRIFT, a stealthy black-box backdoor attack exploiting this vulnerability. Our method employs the Smootherstep function to construct perturbations with guaranteed C2 continuity, ensuring zero velocity and acceleration at trajectory boundaries to satisfy strict kinematic consistency constraints. Furthermore, our keyframe attack strategy selectively poisons only the critical approach phase, maximizing impact while minimizing trigger exposure. The resulting poisoned trajectories are visually indistinguishable from successful demonstrations. Evaluated on the LIBERO, SILENTDRIFT achieves a 93.2% Attack Success Rate with a poisoning rate under 2%, while maintaining a 95.3% Clean Task Success Rate.



## **33. Evaluation of Hate Speech Detection Using Large Language Models and Geographical Contextualization**

cs.CL

6 pages, 2 figures

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19612v1) [paper-pdf](https://arxiv.org/pdf/2502.19612v1)

**Confidence**: 0.90

**Authors**: Anwar Hossain Zahid, Monoshi Kumar Roy, Swarna Das

**Abstract**: The proliferation of hate speech on social media is one of the serious issues that is bringing huge impacts to society: an escalation of violence, discrimination, and social fragmentation. The problem of detecting hate speech is intrinsically multifaceted due to cultural, linguistic, and contextual complexities and adversarial manipulations. In this study, we systematically investigate the performance of LLMs on detecting hate speech across multilingual datasets and diverse geographic contexts. Our work presents a new evaluation framework in three dimensions: binary classification of hate speech, geography-aware contextual detection, and robustness to adversarially generated text. Using a dataset of 1,000 comments from five diverse regions, we evaluate three state-of-the-art LLMs: Llama2 (13b), Codellama (7b), and DeepSeekCoder (6.7b). Codellama had the best binary classification recall with 70.6% and an F1-score of 52.18%, whereas DeepSeekCoder had the best performance in geographic sensitivity, correctly detecting 63 out of 265 locations. The tests for adversarial robustness also showed significant weaknesses; Llama2 misclassified 62.5% of manipulated samples. These results bring to light the trade-offs between accuracy, contextual understanding, and robustness in the current versions of LLMs. This work has thus set the stage for developing contextually aware, multilingual hate speech detection systems by underlining key strengths and limitations, therefore offering actionable insights for future research and real-world applications.



## **34. Robust Fake News Detection using Large Language Models under Adversarial Sentiment Attacks**

cs.CL

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.15277v1) [paper-pdf](https://arxiv.org/pdf/2601.15277v1)

**Confidence**: 0.85

**Authors**: Sahar Tahmasebi, Eric Müller-Budack, Ralph Ewerth

**Abstract**: Misinformation and fake news have become a pressing societal challenge, driving the need for reliable automated detection methods. Prior research has highlighted sentiment as an important signal in fake news detection, either by analyzing which sentiments are associated with fake news or by using sentiment and emotion features for classification. However, this poses a vulnerability since adversaries can manipulate sentiment to evade detectors especially with the advent of large language models (LLMs). A few studies have explored adversarial samples generated by LLMs, but they mainly focus on stylistic features such as writing style of news publishers. Thus, the crucial vulnerability of sentiment manipulation remains largely unexplored. In this paper, we investigate the robustness of state-of-the-art fake news detectors under sentiment manipulation. We introduce AdSent, a sentiment-robust detection framework designed to ensure consistent veracity predictions across both original and sentiment-altered news articles. Specifically, we (1) propose controlled sentiment-based adversarial attacks using LLMs, (2) analyze the impact of sentiment shifts on detection performance. We show that changing the sentiment heavily impacts the performance of fake news detection models, indicating biases towards neutral articles being real, while non-neutral articles are often classified as fake content. (3) We introduce a novel sentiment-agnostic training strategy that enhances robustness against such perturbations. Extensive experiments on three benchmark datasets demonstrate that AdSent significantly outperforms competitive baselines in both accuracy and robustness, while also generalizing effectively to unseen datasets and adversarial scenarios.



## **35. Multimodal Generative Engine Optimization: Rank Manipulation for Vision-Language Model Rankers**

cs.CL

**SubmitDate**: 2026-01-18    [abs](http://arxiv.org/abs/2601.12263v1) [paper-pdf](https://arxiv.org/pdf/2601.12263v1)

**Confidence**: 0.85

**Authors**: Yixuan Du, Chenxiao Yu, Haoyan Xu, Ziyi Wang, Yue Zhao, Xiyang Hu

**Abstract**: Vision-Language Models (VLMs) are rapidly replacing unimodal encoders in modern retrieval and recommendation systems. While their capabilities are well-documented, their robustness against adversarial manipulation in competitive ranking scenarios remains largely unexplored. In this paper, we uncover a critical vulnerability in VLM-based product search: multimodal ranking attacks. We present Multimodal Generative Engine Optimization (MGEO), a novel adversarial framework that enables a malicious actor to unfairly promote a target product by jointly optimizing imperceptible image perturbations and fluent textual suffixes. Unlike existing attacks that treat modalities in isolation, MGEO employs an alternating gradient-based optimization strategy to exploit the deep cross-modal coupling within the VLM. Extensive experiments on real-world datasets using state-of-the-art models demonstrate that our coordinated attack significantly outperforms text-only and image-only baselines. These findings reveal that multimodal synergy, typically a strength of VLMs, can be weaponized to compromise the integrity of search rankings without triggering conventional content filters.



## **36. STEAD: Robust Provably Secure Linguistic Steganography with Diffusion Language Model**

cs.CR

NeurIPS 2025 poster

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.14778v1) [paper-pdf](https://arxiv.org/pdf/2601.14778v1)

**Confidence**: 0.85

**Authors**: Yuang Qi, Na Zhao, Qiyi Yao, Benlong Wu, Weiming Zhang, Nenghai Yu, Kejiang Chen

**Abstract**: Recent provably secure linguistic steganography (PSLS) methods rely on mainstream autoregressive language models (ARMs) to address historically challenging tasks, that is, to disguise covert communication as ``innocuous'' natural language communication. However, due to the characteristic of sequential generation of ARMs, the stegotext generated by ARM-based PSLS methods will produce serious error propagation once it changes, making existing methods unavailable under an active tampering attack. To address this, we propose a robust, provably secure linguistic steganography with diffusion language models (DLMs). Unlike ARMs, DLMs can generate text in a partially parallel manner, allowing us to find robust positions for steganographic embedding that can be combined with error-correcting codes. Furthermore, we introduce error correction strategies, including pseudo-random error correction and neighborhood search correction, during steganographic extraction. Theoretical proof and experimental results demonstrate that our method is secure and robust. It can resist token ambiguity in stegotext segmentation and, to some extent, withstand token-level attacks of insertion, deletion, and substitution.



## **37. VizDefender: Unmasking Visualization Tampering through Proactive Localization and Intent Inference**

cs.CV

IEEE Transactions on Visualization and Computer Graphics (IEEE PacificVis'26 TVCG Track)

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18853v1) [paper-pdf](https://arxiv.org/pdf/2512.18853v1)

**Confidence**: 0.85

**Authors**: Sicheng Song, Yanjie Zhang, Zixin Chen, Huamin Qu, Changbo Wang, Chenhui Li

**Abstract**: The integrity of data visualizations is increasingly threatened by image editing techniques that enable subtle yet deceptive tampering. Through a formative study, we define this challenge and categorize tampering techniques into two primary types: data manipulation and visual encoding manipulation. To address this, we present VizDefender, a framework for tampering detection and analysis. The framework integrates two core components: 1) a semi-fragile watermark module that protects the visualization by embedding a location map to images, which allows for the precise localization of tampered regions while preserving visual quality, and 2) an intent analysis module that leverages Multimodal Large Language Models (MLLMs) to interpret manipulation, inferring the attacker's intent and misleading effects. Extensive evaluations and user studies demonstrate the effectiveness of our methods.



## **38. Boosting RL-Based Visual Reasoning with Selective Adversarial Entropy Intervention**

cs.AI

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.10414v1) [paper-pdf](https://arxiv.org/pdf/2512.10414v1)

**Confidence**: 0.85

**Authors**: Yang Yu, Zhuangzhuang Chen, Siqi Wang, Lanqing Li, Xiaomeng Li

**Abstract**: Recently, reinforcement learning (RL) has become a common choice in enhancing the reasoning capabilities of vision-language models (VLMs). Considering existing RL-based finetuning methods, entropy intervention turns out to be an effective way to benefit exploratory ability, thereby improving policy performance. Notably, most existing studies intervene in entropy by simply controlling the update of specific tokens during policy optimization of RL. They ignore the entropy intervention during the RL sampling that can boost the performance of GRPO by improving the diversity of responses. In this paper, we propose Selective-adversarial Entropy Intervention, namely SaEI, which enhances policy entropy by distorting the visual input with the token-selective adversarial objective coming from the entropy of sampled responses. Specifically, we first propose entropy-guided adversarial sampling (EgAS) that formulates the entropy of sampled responses as an adversarial objective. Then, the corresponding adversarial gradient can be used to attack the visual input for producing adversarial samples, allowing the policy model to explore a larger answer space during RL sampling. Then, we propose token-selective entropy computation (TsEC) to maximize the effectiveness of adversarial attack in EgAS without distorting factual knowledge within VLMs. Extensive experiments on both in-domain and out-of-domain datasets show that our proposed method can greatly improve policy exploration via entropy intervention, to boost reasoning capabilities. Code will be released once the paper is accepted.



## **39. Explainable Adversarial-Robust Vision-Language-Action Model for Robotic Manipulation**

cs.CV

Accepted to MobieSec 2025 (poster session)

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.11865v1) [paper-pdf](https://arxiv.org/pdf/2512.11865v1)

**Confidence**: 0.85

**Authors**: Ju-Young Kim, Ji-Hong Park, Myeongjun Kim, Gun-Woo Kim

**Abstract**: Smart farming has emerged as a key technology for advancing modern agriculture through automation and intelligent control. However, systems relying on RGB cameras for perception and robotic manipulators for control, common in smart farming, are vulnerable to photometric perturbations such as hue, illumination, and noise changes, which can cause malfunction under adversarial attacks. To address this issue, we propose an explainable adversarial-robust Vision-Language-Action model based on the OpenVLA-OFT framework. The model integrates an Evidence-3 module that detects photometric perturbations and generates natural language explanations of their causes and effects. Experiments show that the proposed model reduces Current Action L1 loss by 21.7% and Next Actions L1 loss by 18.4% compared to the baseline, demonstrating improved action prediction accuracy and explainability under adversarial conditions.



## **40. Skeletonization-Based Adversarial Perturbations on Large Vision Language Model's Mathematical Text Recognition**

cs.CV

accepted to ITC-CSCC 2025

**SubmitDate**: 2026-01-08    [abs](http://arxiv.org/abs/2601.04752v1) [paper-pdf](https://arxiv.org/pdf/2601.04752v1)

**Confidence**: 0.85

**Authors**: Masatomo Yoshida, Haruto Namura, Nicola Adami, Masahiro Okuda

**Abstract**: This work explores the visual capabilities and limitations of foundation models by introducing a novel adversarial attack method utilizing skeletonization to reduce the search space effectively. Our approach specifically targets images containing text, particularly mathematical formula images, which are more challenging due to their LaTeX conversion and intricate structure. We conduct a detailed evaluation of both character and semantic changes between original and adversarially perturbed outputs to provide insights into the models' visual interpretation and reasoning abilities. The effectiveness of our method is further demonstrated through its application to ChatGPT, which shows its practical implications in real-world scenarios.



## **41. A Benchmark for Ultra-High-Resolution Remote Sensing MLLMs**

cs.CV

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17319v1) [paper-pdf](https://arxiv.org/pdf/2512.17319v1)

**Confidence**: 0.85

**Authors**: Yunkai Dang, Meiyi Zhu, Donghao Wang, Yizhuo Zhang, Jiacheng Yang, Qi Fan, Yuekun Yang, Wenbin Li, Feng Miao, Yang Gao

**Abstract**: Multimodal large language models (MLLMs) demonstrate strong perception and reasoning performance on existing remote sensing (RS) benchmarks. However, most prior benchmarks rely on low-resolution imagery, and some high-resolution benchmarks suffer from flawed reasoning-task designs. We show that text-only LLMs can perform competitively with multimodal vision-language models on RS reasoning tasks without access to images, revealing a critical mismatch between current benchmarks and the intended evaluation of visual understanding. To enable faithful assessment, we introduce RSHR-Bench, a super-high-resolution benchmark for RS visual understanding and reasoning. RSHR-Bench contains 5,329 full-scene images with a long side of at least 4,000 pixels, with up to about 3 x 10^8 pixels per image, sourced from widely used RS corpora and UAV collections. We design four task families: multiple-choice VQA, open-ended VQA, image captioning, and single-image evaluation. These tasks cover nine perception categories and four reasoning types, supporting multi-turn and multi-image dialog. To reduce reliance on language priors, we apply adversarial filtering with strong LLMs followed by rigorous human verification. Overall, we construct 3,864 VQA tasks, 3,913 image captioning tasks, and 500 fully human-written or verified single-image evaluation VQA pairs. Evaluations across open-source, closed-source, and RS-specific VLMs reveal persistent performance gaps in super-high-resolution scenarios. Code: https://github.com/Yunkaidang/RSHR



## **42. ToxiGAN: Toxic Data Augmentation via LLM-Guided Directional Adversarial Generation**

cs.CL

This paper has been accepted to the main conference of EACL 2026

**SubmitDate**: 2026-01-06    [abs](http://arxiv.org/abs/2601.03121v1) [paper-pdf](https://arxiv.org/pdf/2601.03121v1)

**Confidence**: 0.85

**Authors**: Peiran Li, Jan Fillies, Adrian Paschke

**Abstract**: Augmenting toxic language data in a controllable and class-specific manner is crucial for improving robustness in toxicity classification, yet remains challenging due to limited supervision and distributional skew. We propose ToxiGAN, a class-aware text augmentation framework that combines adversarial generation with semantic guidance from large language models (LLMs). To address common issues in GAN-based augmentation such as mode collapse and semantic drift, ToxiGAN introduces a two-step directional training strategy and leverages LLM-generated neutral texts as semantic ballast. Unlike prior work that treats LLMs as static generators, our approach dynamically selects neutral exemplars to provide balanced guidance. Toxic samples are explicitly optimized to diverge from these exemplars, reinforcing class-specific contrastive signals. Experiments on four hate speech benchmarks show that ToxiGAN achieves the strongest average performance in both macro-F1 and hate-F1, consistently outperforming traditional and LLM-based augmentation methods. Ablation and sensitivity analyses further confirm the benefits of semantic ballast and directional training in enhancing classifier robustness.



## **43. MMBERT: Scaled Mixture-of-Experts Multimodal BERT for Robust Chinese Hate Speech Detection under Cloaking Perturbations**

cs.CL

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00760v1) [paper-pdf](https://arxiv.org/pdf/2508.00760v1)

**Confidence**: 0.85

**Authors**: Qiyao Xue, Yuchen Dou, Ryan Shi, Xiang Lorraine Li, Wei Gao

**Abstract**: Hate speech detection on Chinese social networks presents distinct challenges, particularly due to the widespread use of cloaking techniques designed to evade conventional text-based detection systems. Although large language models (LLMs) have recently improved hate speech detection capabilities, the majority of existing work has concentrated on English datasets, with limited attention given to multimodal strategies in the Chinese context. In this study, we propose MMBERT, a novel BERT-based multimodal framework that integrates textual, speech, and visual modalities through a Mixture-of-Experts (MoE) architecture. To address the instability associated with directly integrating MoE into BERT-based models, we develop a progressive three-stage training paradigm. MMBERT incorporates modality-specific experts, a shared self-attention mechanism, and a router-based expert allocation strategy to enhance robustness against adversarial perturbations. Empirical results in several Chinese hate speech datasets show that MMBERT significantly surpasses fine-tuned BERT-based encoder models, fine-tuned LLMs, and LLMs utilizing in-context learning approaches.



## **44. The Coherence Trap: When MLLM-Crafted Narratives Exploit Manipulated Visual Contexts**

cs.CV

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2505.17476v2) [paper-pdf](https://arxiv.org/pdf/2505.17476v2)

**Confidence**: 0.85

**Authors**: Yuchen Zhang, Yaxiong Wang, Yujiao Wu, Lianwei Wu, Li Zhu, Zhedong Zheng

**Abstract**: The detection and grounding of multimedia manipulation has emerged as a critical challenge in combating AI-generated disinformation. While existing methods have made progress in recent years, we identify two fundamental limitations in current approaches: (1) Underestimation of MLLM-driven deception risk: prevailing techniques primarily address rule-based text manipulations, yet fail to account for sophisticated misinformation synthesized by multimodal large language models (MLLMs) that can dynamically generate semantically coherent, contextually plausible yet deceptive narratives conditioned on manipulated images; (2) Unrealistic misalignment artifacts: currently focused scenarios rely on artificially misaligned content that lacks semantic coherence, rendering them easily detectable. To address these gaps holistically, we propose a new adversarial pipeline that leverages MLLMs to generate high-risk disinformation. Our approach begins with constructing the MLLM-Driven Synthetic Multimodal (MDSM) dataset, where images are first altered using state-of-the-art editing techniques and then paired with MLLM-generated deceptive texts that maintain semantic consistency with the visual manipulations. Building upon this foundation, we present the Artifact-aware Manipulation Diagnosis via MLLM (AMD) framework featuring two key innovations: Artifact Pre-perception Encoding strategy and Manipulation-Oriented Reasoning, to tame MLLMs for the MDSM problem. Comprehensive experiments validate our framework's superior generalization capabilities as a unified architecture for detecting MLLM-powered multimodal deceptions. In cross-domain testing on the MDSM dataset, AMD achieves the best average performance, with 88.18 ACC, 60.25 mAP, and 61.02 mIoU scores.



