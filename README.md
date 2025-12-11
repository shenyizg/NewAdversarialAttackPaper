# Latest Adversarial Attack Papers
**update at 2025-12-11 09:54:23**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Improved Pseudorandom Codes from Permuted Puzzles**

cs.CR

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08918v1) [paper-pdf](https://arxiv.org/pdf/2512.08918v1)

**Authors**: Miranda Christ, Noah Golowich, Sam Gunn, Ankur Moitra, Daniel Wichs

**Abstract**: Watermarks are an essential tool for identifying AI-generated content. Recently, Christ and Gunn (CRYPTO '24) introduced pseudorandom error-correcting codes (PRCs), which are equivalent to watermarks with strong robustness and quality guarantees. A PRC is a pseudorandom encryption scheme whose decryption algorithm tolerates a high rate of errors. Pseudorandomness ensures quality preservation of the watermark, and error tolerance of decryption translates to the watermark's ability to withstand modification of the content.   In the short time since the introduction of PRCs, several works (NeurIPS '24, RANDOM '25, STOC '25) have proposed new constructions. Curiously, all of these constructions are vulnerable to quasipolynomial-time distinguishing attacks. Furthermore, all lack robustness to edits over a constant-sized alphabet, which is necessary for a meaningfully robust LLM watermark. Lastly, they lack robustness to adversaries who know the watermarking detection key. Until now, it was not clear whether any of these properties was achievable individually, let alone together.   We construct pseudorandom codes that achieve all of the above: plausible subexponential pseudorandomness security, robustness to worst-case edits over a binary alphabet, and robustness against even computationally unbounded adversaries that have the detection key. Pseudorandomness rests on a new assumption that we formalize, the permuted codes conjecture, which states that a distribution of permuted noisy codewords is pseudorandom. We show that this conjecture is implied by the permuted puzzles conjecture used previously to construct doubly efficient private information retrieval. To give further evidence, we show that the conjecture holds against a broad class of simple distinguishers, including read-once branching programs.



## **2. When Tables Leak: Attacking String Memorization in LLM-Based Tabular Data Generation**

cs.LG

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08875v1) [paper-pdf](https://arxiv.org/pdf/2512.08875v1)

**Authors**: Joshua Ward, Bochao Gu, Chi-Hua Wang, Guang Cheng

**Abstract**: Large Language Models (LLMs) have recently demonstrated remarkable performance in generating high-quality tabular synthetic data. In practice, two primary approaches have emerged for adapting LLMs to tabular data generation: (i) fine-tuning smaller models directly on tabular datasets, and (ii) prompting larger models with examples provided in context. In this work, we show that popular implementations from both regimes exhibit a tendency to compromise privacy by reproducing memorized patterns of numeric digits from their training data. To systematically analyze this risk, we introduce a simple No-box Membership Inference Attack (MIA) called LevAtt that assumes adversarial access to only the generated synthetic data and targets the string sequences of numeric digits in synthetic observations. Using this approach, our attack exposes substantial privacy leakage across a wide range of models and datasets, and in some cases, is even a perfect membership classifier on state-of-the-art models. Our findings highlight a unique privacy vulnerability of LLM-based synthetic data generation and the need for effective defenses. To this end, we propose two methods, including a novel sampling strategy that strategically perturbs digits during generation. Our evaluation demonstrates that this approach can defeat these attacks with minimal loss of fidelity and utility of the synthetic data.



## **3. Secure and Privacy-Preserving Federated Learning for Next-Generation Underground Mine Safety**

cs.CR

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08862v1) [paper-pdf](https://arxiv.org/pdf/2512.08862v1)

**Authors**: Mohamed Elmahallawy, Sanjay Madria, Samuel Frimpong

**Abstract**: Underground mining operations depend on sensor networks to monitor critical parameters such as temperature, gas concentration, and miner movement, enabling timely hazard detection and safety decisions. However, transmitting raw sensor data to a centralized server for machine learning (ML) model training raises serious privacy and security concerns. Federated Learning (FL) offers a promising alternative by enabling decentralized model training without exposing sensitive local data. Yet, applying FL in underground mining presents unique challenges: (i) Adversaries may eavesdrop on shared model updates to launch model inversion or membership inference attacks, compromising data privacy and operational safety; (ii) Non-IID data distributions across mines and sensor noise can hinder model convergence. To address these issues, we propose FedMining--a privacy-preserving FL framework tailored for underground mining. FedMining introduces two core innovations: (1) a Decentralized Functional Encryption (DFE) scheme that keeps local models encrypted, thwarting unauthorized access and inference attacks; and (2) a balancing aggregation mechanism to mitigate data heterogeneity and enhance convergence. Evaluations on real-world mining datasets demonstrate FedMining's ability to safeguard privacy while maintaining high model accuracy and achieving rapid convergence with reduced communication and computation overhead. These advantages make FedMining both secure and practical for real-time underground safety monitoring.



## **4. Developing Distance-Aware Uncertainty Quantification Methods in Physics-Guided Neural Networks for Reliable Bearing Health Prediction**

cs.LG

Under review at Structural health Monitoring - SAGE

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08499v1) [paper-pdf](https://arxiv.org/pdf/2512.08499v1)

**Authors**: Waleed Razzaq, Yun-Bo Zhao

**Abstract**: Accurate and uncertainty-aware degradation estimation is essential for predictive maintenance in safety-critical systems like rotating machinery with rolling-element bearings. Many existing uncertainty methods lack confidence calibration, are costly to run, are not distance-aware, and fail to generalize under out-of-distribution data. We introduce two distance-aware uncertainty methods for deterministic physics-guided neural networks: PG-SNGP, based on Spectral Normalization Gaussian Process, and PG-SNER, based on Deep Evidential Regression. We apply spectral normalization to the hidden layers so the network preserves distances from input to latent space. PG-SNGP replaces the final dense layer with a Gaussian Process layer for distance-sensitive uncertainty, while PG-SNER outputs Normal Inverse Gamma parameters to model uncertainty in a coherent probabilistic form. We assess performance using standard accuracy metrics and a new distance-aware metric based on the Pearson Correlation Coefficient, which measures how well predicted uncertainty tracks the distance between test and training samples. We also design a dynamic weighting scheme in the loss to balance data fidelity and physical consistency. We test our methods on rolling-element bearing degradation using the PRONOSTIA dataset and compare them with Monte Carlo and Deep Ensemble PGNNs. Results show that PG-SNGP and PG-SNER improve prediction accuracy, generalize reliably under OOD conditions, and remain robust to adversarial attacks and noise.



## **5. Attention is All You Need to Defend Against Indirect Prompt Injection Attacks in LLMs**

cs.CR

Accepted by Network and Distributed System Security (NDSS) Symposium 2026

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08417v1) [paper-pdf](https://arxiv.org/pdf/2512.08417v1)

**Authors**: Yinan Zhong, Qianhao Miao, Yanjiao Chen, Jiangyi Deng, Yushi Cheng, Wenyuan Xu

**Abstract**: Large Language Models (LLMs) have been integrated into many applications (e.g., web agents) to perform more sophisticated tasks. However, LLM-empowered applications are vulnerable to Indirect Prompt Injection (IPI) attacks, where instructions are injected via untrustworthy external data sources. This paper presents Rennervate, a defense framework to detect and prevent IPI attacks. Rennervate leverages attention features to detect the covert injection at a fine-grained token level, enabling precise sanitization that neutralizes IPI attacks while maintaining LLM functionalities. Specifically, the token-level detector is materialized with a 2-step attentive pooling mechanism, which aggregates attention heads and response tokens for IPI detection and sanitization. Moreover, we establish a fine-grained IPI dataset, FIPI, to be open-sourced to support further research. Extensive experiments verify that Rennervate outperforms 15 commercial and academic IPI defense methods, achieving high precision on 5 LLMs and 6 datasets. We also demonstrate that Rennervate is transferable to unseen attacks and robust against adaptive adversaries.



## **6. Exposing and Defending Membership Leakage in Vulnerability Prediction Models**

cs.CR

Accepted at APSEC 2025

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08291v1) [paper-pdf](https://arxiv.org/pdf/2512.08291v1)

**Authors**: Yihan Liao, Jacky Keung, Xiaoxue Ma, Jingyu Zhang, Yicheng Sun

**Abstract**: Neural models for vulnerability prediction (VP) have achieved impressive performance by learning from large-scale code repositories. However, their susceptibility to Membership Inference Attacks (MIAs), where adversaries aim to infer whether a particular code sample was used during training, poses serious privacy concerns. While MIA has been widely investigated in NLP and vision domains, its effects on security-critical code analysis tasks remain underexplored. In this work, we conduct the first comprehensive analysis of MIA on VP models, evaluating the attack success across various architectures (LSTM, BiGRU, and CodeBERT) and feature combinations, including embeddings, logits, loss, and confidence. Our threat model aligns with black-box and gray-box settings where prediction outputs are observable, allowing adversaries to infer membership by analyzing output discrepancies between training and non-training samples. The empirical findings reveal that logits and loss are the most informative and vulnerable outputs for membership leakage. Motivated by these observations, we propose a Noise-based Membership Inference Defense (NMID), which is a lightweight defense module that applies output masking and Gaussian noise injection to disrupt adversarial inference. Extensive experiments demonstrate that NMID significantly reduces MIA effectiveness, lowering the attack AUC from nearly 1.0 to below 0.65, while preserving the predictive utility of VP models. Our study highlights critical privacy risks in code analysis and offers actionable defense strategies for securing AI-powered software systems.



## **7. MIRAGE: Misleading Retrieval-Augmented Generation via Black-box and Query-agnostic Poisoning Attacks**

cs.CR

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08289v1) [paper-pdf](https://arxiv.org/pdf/2512.08289v1)

**Authors**: Tailun Chen, Yu He, Yan Wang, Shuo Shao, Haolun Zheng, Zhihao Liu, Jinfeng Li, Yuefeng Chen, Zhixuan Chu, Zhan Qin

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance LLMs with external knowledge but introduce a critical attack surface: corpus poisoning. While recent studies have demonstrated the potential of such attacks, they typically rely on impractical assumptions, such as white-box access or known user queries, thereby underestimating the difficulty of real-world exploitation. In this paper, we bridge this gap by proposing MIRAGE, a novel multi-stage poisoning pipeline designed for strict black-box and query-agnostic environments. Operating on surrogate model feedback, MIRAGE functions as an automated optimization framework that integrates three key mechanisms: it utilizes persona-driven query synthesis to approximate latent user search distributions, employs semantic anchoring to imperceptibly embed these intents for high retrieval visibility, and leverages an adversarial variant of Test-Time Preference Optimization (TPO) to maximize persuasion. To rigorously evaluate this threat, we construct a new benchmark derived from three long-form, domain-specific datasets. Extensive experiments demonstrate that MIRAGE significantly outperforms existing baselines in both attack efficacy and stealthiness, exhibiting remarkable transferability across diverse retriever-LLM configurations and highlighting the urgent need for robust defense strategies.



## **8. Evaluating Vulnerabilities of Connected Vehicles Under Cyber Attacks by Attack-Defense Tree**

cs.CR

6 Pages, International Conference on Computing, Networking and Communication (ICNC), Maui, Hawaii, USA, 2026

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08204v1) [paper-pdf](https://arxiv.org/pdf/2512.08204v1)

**Authors**: Muhammad Baqer Mollah, Honggang Wang, Hua Fang

**Abstract**: Connected vehicles represent a key enabler of intelligent transportation systems, where vehicles are equipped with advanced communication, sensing, and computing technologies to interact not only with one another but also with surrounding infrastructures and the environment. Through continuous data exchange, such vehicles are capable of enhancing road safety, improving traffic efficiency, and ensuring more reliable mobility services. Further, when these capabilities are integrated with advanced automation technologies, the concept essentially evolves into connected and autonomous vehicles (CAVs). While connected vehicles primarily focus on seamless information sharing, autonomous vehicles are mainly dependent on advanced perception, decision-making, and control mechanisms to operate with minimal or without human intervention. However, as a result of connectivity, an adversary with malicious intentions might be able to compromise successfully by breaching the system components of CAVs. In this paper, we present an attack-tree based methodology for evaluating cyber security vulnerabilities in CAVs. In particular, we utilize the attack-defense tree formulation to systematically assess attack-leaf vulnerabilities, and before analyzing the vulnerability indices, we also define a measure of vulnerabilities, which is based on existing cyber security threats and corresponding defensive countermeasures.



## **9. A Practical Framework for Evaluating Medical AI Security: Reproducible Assessment of Jailbreaking and Privacy Vulnerabilities Across Clinical Specialties**

cs.CR

6 pages, 1 figure, framework proposal

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08185v1) [paper-pdf](https://arxiv.org/pdf/2512.08185v1)

**Authors**: Jinghao Wang, Ping Zhang, Carter Yagemann

**Abstract**: Medical Large Language Models (LLMs) are increasingly deployed for clinical decision support across diverse specialties, yet systematic evaluation of their robustness to adversarial misuse and privacy leakage remains inaccessible to most researchers. Existing security benchmarks require GPU clusters, commercial API access, or protected health data -- barriers that limit community participation in this critical research area. We propose a practical, fully reproducible framework for evaluating medical AI security under realistic resource constraints. Our framework design covers multiple medical specialties stratified by clinical risk -- from high-risk domains such as emergency medicine and psychiatry to general practice -- addressing jailbreaking attacks (role-playing, authority impersonation, multi-turn manipulation) and privacy extraction attacks. All evaluation utilizes synthetic patient records requiring no IRB approval. The framework is designed to run entirely on consumer CPU hardware using freely available models, eliminating cost barriers. We present the framework specification including threat models, data generation methodology, evaluation protocols, and scoring rubrics. This proposal establishes a foundation for comparative security assessment of medical-specialist models and defense mechanisms, advancing the broader goal of ensuring safe and trustworthy medical AI systems.



## **10. A Dynamic Coding Scheme to Prevent Covert Cyber-Attacks in Cyber-Physical Systems**

eess.SY

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08134v1) [paper-pdf](https://arxiv.org/pdf/2512.08134v1)

**Authors**: Mahdi Taheri, Khashayar Khorasani, Nader Meskin

**Abstract**: In this paper, we address two main problems in the context of covert cyber-attacks in cyber-physical systems (CPS). First, we aim to investigate and develop necessary and sufficient conditions in terms of disruption resources of the CPS that enable adversaries to execute covert cyber-attacks. These conditions can be utilized to identify the input and output communication channels that are needed by adversaries to execute these attacks. Second, this paper introduces and develops a dynamic coding scheme as a countermeasure against covert cyber-attacks. Under certain conditions and assuming the existence of one secure input and two secure output communication channels, the proposed dynamic coding scheme prevents adversaries from executing covert cyber-attacks. A numerical case study of a flight control system is provided to demonstrate the capabilities of our proposed and developed dynamic coding scheme.



## **11. Universal Adversarial Suffixes Using Calibrated Gumbel-Softmax Relaxation**

cs.CL

10 pages

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08123v1) [paper-pdf](https://arxiv.org/pdf/2512.08123v1)

**Authors**: Sampriti Soor, Suklav Ghosh, Arijit Sur

**Abstract**: Language models (LMs) are often used as zero-shot or few-shot classifiers by scoring label words, but they remain fragile to adversarial prompts. Prior work typically optimizes task- or model-specific triggers, making results difficult to compare and limiting transferability. We study universal adversarial suffixes: short token sequences (4-10 tokens) that, when appended to any input, broadly reduce accuracy across tasks and models. Our approach learns the suffix in a differentiable "soft" form using Gumbel-Softmax relaxation and then discretizes it for inference. Training maximizes calibrated cross-entropy on the label region while masking gold tokens to prevent trivial leakage, with entropy regularization to avoid collapse. A single suffix trained on one model transfers effectively to others, consistently lowering both accuracy and calibrated confidence. Experiments on sentiment analysis, natural language inference, paraphrase detection, commonsense QA, and physical reasoning with Qwen2-1.5B, Phi-1.5, and TinyLlama-1.1B demonstrate consistent attack effectiveness and transfer across tasks and model families.



## **12. Detecting Ambiguity Aversion in Cyberattack Behavior to Inform Cognitive Defense Strategies**

cs.CR

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.08107v1) [paper-pdf](https://arxiv.org/pdf/2512.08107v1)

**Authors**: Stephan Carney, Soham Hans, Sofia Hirschmann, Stacey Marsella, Yvonne Fonken, Peggy Wu, Nikolos Gurney

**Abstract**: Adversaries (hackers) attempting to infiltrate networks frequently face uncertainty in their operational environments. This research explores the ability to model and detect when they exhibit ambiguity aversion, a cognitive bias reflecting a preference for known (versus unknown) probabilities. We introduce a novel methodological framework that (1) leverages rich, multi-modal data from human-subjects red-team experiments, (2) employs a large language model (LLM) pipeline to parse unstructured logs into MITRE ATT&CK-mapped action sequences, and (3) applies a new computational model to infer an attacker's ambiguity aversion level in near-real time. By operationalizing this cognitive trait, our work provides a foundational component for developing adaptive cognitive defense strategies.



## **13. An Adaptive Multi-Layered Honeynet Architecture for Threat Behavior Analysis via Deep Learning**

cs.CR

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07827v1) [paper-pdf](https://arxiv.org/pdf/2512.07827v1)

**Authors**: Lukas Johannes Möller

**Abstract**: The escalating sophistication and variety of cyber threats have rendered static honeypots inadequate, necessitating adaptive, intelligence-driven deception. In this work, ADLAH is introduced: an Adaptive Deep Learning Anomaly Detection Honeynet designed to maximize high-fidelity threat intelligence while minimizing cost through autonomous orchestration of infrastructure. The principal contribution is offered as an end-to-end architectural blueprint and vision for an AI-driven deception platform. Feasibility is evidenced by a functional prototype of the central decision mechanism, in which a reinforcement learning (RL) agent determines, in real time, when sessions should be escalated from low-interaction sensor nodes to dynamically provisioned, high-interaction honeypots. Because sufficient live data were unavailable, field-scale validation is not claimed; instead, design trade-offs and limitations are detailed, and a rigorous roadmap toward empirical evaluation at scale is provided. Beyond selective escalation and anomaly detection, the architecture pursues automated extraction, clustering, and versioning of bot attack chains, a core capability motivated by the empirical observation that exposed services are dominated by automated traffic. Together, these elements delineate a practical path toward cost-efficient capture of high-value adversary behavior, systematic bot versioning, and the production of actionable threat intelligence.



## **14. Optimization-Guided Diffusion for Interactive Scene Generation**

cs.CV

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07661v1) [paper-pdf](https://arxiv.org/pdf/2512.07661v1)

**Authors**: Shiaho Li, Naisheng Ye, Tianyu Li, Kashyap Chitta, Tuo An, Peng Su, Boyang Wang, Haiou Liu, Chen Lv, Hongyang Li

**Abstract**: Realistic and diverse multi-agent driving scenes are crucial for evaluating autonomous vehicles, but safety-critical events which are essential for this task are rare and underrepresented in driving datasets. Data-driven scene generation offers a low-cost alternative by synthesizing complex traffic behaviors from existing driving logs. However, existing models often lack controllability or yield samples that violate physical or social constraints, limiting their usability. We present OMEGA, an optimization-guided, training-free framework that enforces structural consistency and interaction awareness during diffusion-based sampling from a scene generation model. OMEGA re-anchors each reverse diffusion step via constrained optimization, steering the generation towards physically plausible and behaviorally coherent trajectories. Building on this framework, we formulate ego-attacker interactions as a game-theoretic optimization in the distribution space, approximating Nash equilibria to generate realistic, safety-critical adversarial scenarios. Experiments on nuPlan and Waymo show that OMEGA improves generation realism, consistency, and controllability, increasing the ratio of physically and behaviorally valid scenes from 32.35% to 72.27% for free exploration capabilities, and from 11% to 80% for controllability-focused generation. Our approach can also generate $5\times$ more near-collision frames with a time-to-collision under three seconds while maintaining the overall scene realism.



## **15. Towards Robust DeepFake Detection under Unstable Face Sequences: Adaptive Sparse Graph Embedding with Order-Free Representation and Explicit Laplacian Spectral Prior**

cs.CV

16 pages (including appendix)

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07498v1) [paper-pdf](https://arxiv.org/pdf/2512.07498v1)

**Authors**: Chih-Chung Hsu, Shao-Ning Chen, Chia-Ming Lee, Yi-Fang Wang, Yi-Shiuan Chou

**Abstract**: Ensuring the authenticity of video content remains challenging as DeepFake generation becomes increasingly realistic and robust against detection. Most existing detectors implicitly assume temporally consistent and clean facial sequences, an assumption that rarely holds in real-world scenarios where compression artifacts, occlusions, and adversarial attacks destabilize face detection and often lead to invalid or misdetected faces. To address these challenges, we propose a Laplacian-Regularized Graph Convolutional Network (LR-GCN) that robustly detects DeepFakes from noisy or unordered face sequences, while being trained only on clean facial data. Our method constructs an Order-Free Temporal Graph Embedding (OF-TGE) that organizes frame-wise CNN features into an adaptive sparse graph based on semantic affinities. Unlike traditional methods constrained by strict temporal continuity, OF-TGE captures intrinsic feature consistency across frames, making it resilient to shuffled, missing, or heavily corrupted inputs. We further impose a dual-level sparsity mechanism on both graph structure and node features to suppress the influence of invalid faces. Crucially, we introduce an explicit Graph Laplacian Spectral Prior that acts as a high-pass operator in the graph spectral domain, highlighting structural anomalies and forgery artifacts, which are then consolidated by a low-pass GCN aggregation. This sequential design effectively realizes a task-driven spectral band-pass mechanism that suppresses background information and random noise while preserving manipulation cues. Extensive experiments on FF++, Celeb-DFv2, and DFDC demonstrate that LR-GCN achieves state-of-the-art performance and significantly improved robustness under severe global and local disruptions, including missing faces, occlusions, and adversarially perturbed face detections.



## **16. Pay Less Attention to Function Words for Free Robustness of Vision-Language Models**

cs.LG

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.07222v2) [paper-pdf](https://arxiv.org/pdf/2512.07222v2)

**Authors**: Qiwei Tian, Chenhao Lin, Zhengyu Zhao, Chao Shen

**Abstract**: To address the trade-off between robustness and performance for robust VLM, we observe that function words could incur vulnerability of VLMs against cross-modal adversarial attacks, and propose Function-word De-Attention (FDA) accordingly to mitigate the impact of function words. Similar to differential amplifiers, our FDA calculates the original and the function-word cross-attention within attention heads, and differentially subtracts the latter from the former for more aligned and robust VLMs. Comprehensive experiments include 2 SOTA baselines under 6 different attacks on 2 downstream tasks, 3 datasets, and 3 models. Overall, our FDA yields an average 18/13/53% ASR drop with only 0.2/0.3/0.6% performance drops on the 3 tested models on retrieval, and a 90% ASR drop with a 0.3% performance gain on visual grounding. We demonstrate the scalability, generalization, and zero-shot performance of FDA experimentally, as well as in-depth ablation studies and analysis. Code will be made publicly at https://github.com/michaeltian108/FDA.



## **17. ThinkTrap: Denial-of-Service Attacks against Black-box LLM Services via Infinite Thinking**

cs.CR

This version includes the final camera-ready manuscript accepted by NDSS 2026

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07086v1) [paper-pdf](https://arxiv.org/pdf/2512.07086v1)

**Authors**: Yunzhe Li, Jianan Wang, Hongzi Zhu, James Lin, Shan Chang, Minyi Guo

**Abstract**: Large Language Models (LLMs) have become foundational components in a wide range of applications, including natural language understanding and generation, embodied intelligence, and scientific discovery. As their computational requirements continue to grow, these models are increasingly deployed as cloud-based services, allowing users to access powerful LLMs via the Internet. However, this deployment model introduces a new class of threat: denial-of-service (DoS) attacks via unbounded reasoning, where adversaries craft specially designed inputs that cause the model to enter excessively long or infinite generation loops. These attacks can exhaust backend compute resources, degrading or denying service to legitimate users. To mitigate such risks, many LLM providers adopt a closed-source, black-box setting to obscure model internals. In this paper, we propose ThinkTrap, a novel input-space optimization framework for DoS attacks against LLM services even in black-box environments. The core idea of ThinkTrap is to first map discrete tokens into a continuous embedding space, then undertake efficient black-box optimization in a low-dimensional subspace exploiting input sparsity. The goal of this optimization is to identify adversarial prompts that induce extended or non-terminating generation across several state-of-the-art LLMs, achieving DoS with minimal token overhead. We evaluate the proposed attack across multiple commercial, closed-source LLM services. Our results demonstrate that, even far under the restrictive request frequency limits commonly enforced by these platforms, typically capped at ten requests per minute (10 RPM), the attack can degrade service throughput to as low as 1% of its original capacity, and in some cases, induce complete service failure.



## **18. Replicating TEMPEST at Scale: Multi-Turn Adversarial Attacks Against Trillion-Parameter Frontier Models**

cs.CL

30 pages, 11 figures, 5 tables. Code and data: https://github.com/ricyoung/tempest-replication

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07059v1) [paper-pdf](https://arxiv.org/pdf/2512.07059v1)

**Authors**: Richard Young

**Abstract**: Despite substantial investment in safety alignment, the vulnerability of large language models to sophisticated multi-turn adversarial attacks remains poorly characterized, and whether model scale or inference mode affects robustness is unknown. This study employed the TEMPEST multi-turn attack framework to evaluate ten frontier models from eight vendors across 1,000 harmful behaviors, generating over 97,000 API queries across adversarial conversations with automated evaluation by independent safety classifiers. Results demonstrated a spectrum of vulnerability: six models achieved 96% to 100% attack success rate (ASR), while four showed meaningful resistance, with ASR ranging from 42% to 78%; enabling extended reasoning on identical architecture reduced ASR from 97% to 42%. These findings indicate that safety alignment quality varies substantially across vendors, that model scale does not predict adversarial robustness, and that thinking mode provides a deployable safety enhancement. Collectively, this work establishes that current alignment techniques remain fundamentally vulnerable to adaptive multi-turn attacks regardless of model scale, while identifying deliberative inference as a promising defense direction.



## **19. Toward Reliable Machine Unlearning: Theory, Algorithms, and Evaluation**

cs.LG

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2512.06993v1) [paper-pdf](https://arxiv.org/pdf/2512.06993v1)

**Authors**: Ali Ebrahimpour-Boroojeny

**Abstract**: We propose new methodologies for both unlearning random set of samples and class unlearning and show that they outperform existing methods. The main driver of our unlearning methods is the similarity of predictions to a retrained model on both the forget and remain samples. We introduce Adversarial Machine UNlearning (AMUN), which surpasses prior state-of-the-art methods for image classification based on SOTA MIA scores. AMUN lowers the model's confidence on forget samples by fine-tuning on their corresponding adversarial examples. Through theoretical analysis, we identify factors governing AMUN's performance, including smoothness. To facilitate training of smooth models with a controlled Lipschitz constant, we propose FastClip, a scalable method that performs layer-wise spectral-norm clipping of affine layers. In a separate study, we show that increased smoothness naturally improves adversarial example transfer, thereby supporting the second factor above.   Following the same principles for class unlearning, we show that existing methods fail in replicating a retrained model's behavior by introducing a nearest-neighbor membership inference attack (MIA-NN) that uses the probabilities assigned to neighboring classes to detect unlearned samples and demonstrate the vulnerability of such methods. We then propose a fine-tuning objective that mitigates this leakage by approximating, for forget-class inputs, the distribution over remaining classes that a model retrained from scratch would produce. To construct this approximation, we estimate inter-class similarity and tilt the target model's distribution accordingly. The resulting Tilted ReWeighting(TRW) distribution serves as the desired target during fine-tuning. Across multiple benchmarks, TRW matches or surpasses existing unlearning methods on prior metrics.



## **20. Patronus: Identifying and Mitigating Transferable Backdoors in Pre-trained Language Models**

cs.CR

Work in progress

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2512.06899v1) [paper-pdf](https://arxiv.org/pdf/2512.06899v1)

**Authors**: Tianhang Zhao, Wei Du, Haodong Zhao, Sufeng Duan, Gongshen Liu

**Abstract**: Transferable backdoors pose a severe threat to the Pre-trained Language Models (PLMs) supply chain, yet defensive research remains nascent, primarily relying on detecting anomalies in the output feature space. We identify a critical flaw that fine-tuning on downstream tasks inevitably modifies model parameters, shifting the output distribution and rendering pre-computed defense ineffective. To address this, we propose Patronus, a novel framework that use input-side invariance of triggers against parameter shifts. To overcome the convergence challenges of discrete text optimization, Patronus introduces a multi-trigger contrastive search algorithm that effectively bridges gradient-based optimization with contrastive learning objectives. Furthermore, we employ a dual-stage mitigation strategy combining real-time input monitoring with model purification via adversarial training. Extensive experiments across 15 PLMs and 10 tasks demonstrate that Patronus achieves $\geq98.7\%$ backdoor detection recall and reduce attack success rates to clean settings, significantly outperforming all state-of-the-art baselines in all settings. Code is available at https://github.com/zth855/Patronus.



## **21. Look Twice before You Leap: A Rational Agent Framework for Localized Adversarial Anonymization**

cs.CR

16 pages, 6 figures

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2512.06713v1) [paper-pdf](https://arxiv.org/pdf/2512.06713v1)

**Authors**: Donghang Duan, Xu Zheng

**Abstract**: Current LLM-based text anonymization frameworks usually rely on remote API services from powerful LLMs, which creates an inherent "privacy paradox": users must somehow disclose data to untrusted third parties for superior privacy preservation. Moreover, directly migrating these frameworks to local small-scale models (LSMs) offers a suboptimal solution with catastrophic collapse in utility based on our core findings. Our work argues that this failure stems not merely from the capability deficits of LSMs, but from the inherent irrationality of the greedy adversarial strategies employed by current state-of-the-art (SoTA) methods. We model the anonymization process as a trade-off between Marginal Privacy Gain (MPG) and Marginal Utility Cost (MUC), and demonstrate that greedy strategies inevitably drift into an irrational state. To address this, we propose Rational Localized Adversarial Anonymization (RLAA), a fully localized and training-free framework featuring an Attacker-Arbitrator-Anonymizer (A-A-A) architecture. RLAA introduces an arbitrator that acts as a rationality gatekeeper, validating the attacker's inference to filter out feedback providing negligible benefits on privacy preservation. This mechanism enforces a rational early-stopping criterion, and systematically prevents utility collapse. Extensive experiments on different datasets demonstrate that RLAA achieves the best privacy-utility trade-off, and in some cases even outperforms SoTA on the Pareto principle. Our code and datasets will be released upon acceptance.



## **22. GSAE: Graph-Regularized Sparse Autoencoders for Robust LLM Safety Steering**

cs.LG

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2512.06655v1) [paper-pdf](https://arxiv.org/pdf/2512.06655v1)

**Authors**: Jehyeok Yeon, Federico Cinus, Yifan Wu, Luca Luceri

**Abstract**: Large language models (LLMs) face critical safety challenges, as they can be manipulated to generate harmful content through adversarial prompts and jailbreak attacks. Many defenses are typically either black-box guardrails that filter outputs, or internals-based methods that steer hidden activations by operationalizing safety as a single latent feature or dimension. While effective for simple concepts, this assumption is limiting, as recent evidence shows that abstract concepts such as refusal and temporality are distributed across multiple features rather than isolated in one. To address this limitation, we introduce Graph-Regularized Sparse Autoencoders (GSAEs), which extends SAEs with a Laplacian smoothness penalty on the neuron co-activation graph. Unlike standard SAEs that assign each concept to a single latent feature, GSAEs recover smooth, distributed safety representations as coherent patterns spanning multiple features. We empirically demonstrate that GSAE enables effective runtime safety steering, assembling features into a weighted set of safety-relevant directions and controlling them with a two-stage gating mechanism that activates interventions only when harmful prompts or continuations are detected during generation. This approach enforces refusals adaptively while preserving utility on benign queries. Across safety and QA benchmarks, GSAE steering achieves an average 82% selective refusal rate, substantially outperforming standard SAE steering (42%), while maintaining strong task accuracy (70% on TriviaQA, 65% on TruthfulQA, 74% on GSM8K). Robustness experiments further show generalization across LLaMA-3, Mistral, Qwen, and Phi families and resilience against jailbreak attacks (GCG, AutoDAN), consistently maintaining >= 90% refusal of harmful content.



## **23. Characterizing Large-Scale Adversarial Activities Through Large-Scale Honey-Nets**

cs.CR

Accepted at Conference IEEE UEMCON 2025

**SubmitDate**: 2025-12-06    [abs](http://arxiv.org/abs/2512.06557v1) [paper-pdf](https://arxiv.org/pdf/2512.06557v1)

**Authors**: Tonia Haikal, Eman Hammad, Shereen Ismail

**Abstract**: The increasing sophistication of cyber threats demands novel approaches to characterize adversarial strategies, particularly those targeting critical infrastructure and IoT ecosystems. This paper presents a longitudinal analysis of attacker behavior using HoneyTrap, an adaptive honeypot framework deployed across geographically distributed nodes to emulate vulnerable services and safely capture malicious traffic. Over a 24 day observation window, more than 60.3 million events were collected. To enable scalable analytics, raw JSON logs were transformed into Apache Parquet, achieving 5.8 - 9.3x compression and 7.2x faster queries, while ASN enrichment and salted SHA-256 pseudonymization added network intelligence and privacy preservation.   Our analysis reveals three key findings: (1) The majority of traffic targeted HTTP and HTTPS services (ports 80 and 443), with more than 8 million connection attempts and daily peaks exceeding 1.7 million events. (2) SSH (port 22) was frequently subject to brute-force attacks, with over 4.6 million attempts. (3) Less common services like Minecraft (25565) and SMB (445) were also targeted, with Minecraft receiving about 118,000 daily attempts that often coincided with spikes on other ports.



## **24. Securing the Model Context Protocol: Defending LLMs Against Tool Poisoning and Adversarial Attacks**

cs.CR

**SubmitDate**: 2025-12-06    [abs](http://arxiv.org/abs/2512.06556v1) [paper-pdf](https://arxiv.org/pdf/2512.06556v1)

**Authors**: Saeid Jamshidi, Kawser Wazed Nafi, Arghavan Moradi Dakhel, Negar Shahabi, Foutse Khomh, Naser Ezzati-Jivan

**Abstract**: The Model Context Protocol (MCP) enables Large Language Models to integrate external tools through structured descriptors, increasing autonomy in decision-making, task execution, and multi-agent workflows. However, this autonomy creates a largely overlooked security gap. Existing defenses focus on prompt-injection attacks and fail to address threats embedded in tool metadata, leaving MCP-based systems exposed to semantic manipulation. This work analyzes three classes of semantic attacks on MCP-integrated systems: (1) Tool Poisoning, where adversarial instructions are hidden in tool descriptors; (2) Shadowing, where trusted tools are indirectly compromised through contaminated shared context; and (3) Rug Pulls, where descriptors are altered after approval to subvert behavior. To counter these threats, we introduce a layered security framework with three components: RSA-based manifest signing to enforce descriptor integrity, LLM-on-LLM semantic vetting to detect suspicious tool definitions, and lightweight heuristic guardrails that block anomalous tool behavior at runtime. Through evaluation of GPT-4, DeepSeek, and Llama-3.5 across eight prompting strategies, we find that security performance varies widely by model architecture and reasoning method. GPT-4 blocks about 71 percent of unsafe tool calls, balancing latency and safety. DeepSeek shows the highest resilience to Shadowing attacks but with greater latency, while Llama-3.5 is fastest but least robust. Our results show that the proposed framework reduces unsafe tool invocation rates without model fine-tuning or internal modification.



## **25. Web Technologies Security in the AI Era: A Survey of CDN-Enhanced Defenses**

cs.CR

Accepted at 2025 IEEE Asia Pacific Conference on Wireless and Mobile (APWiMob). 7 pages, 5 figures

**SubmitDate**: 2025-12-06    [abs](http://arxiv.org/abs/2512.06390v1) [paper-pdf](https://arxiv.org/pdf/2512.06390v1)

**Authors**: Mehrab Hosain, Sabbir Alom Shuvo, Matthew Ogbe, Md Shah Jalal Mazumder, Yead Rahman, Md Azizul Hakim, Anukul Pandey

**Abstract**: The modern web stack, which is dominated by browser-based applications and API-first backends, now operates under an adversarial equilibrium where automated, AI-assisted attacks evolve continuously. Content Delivery Networks (CDNs) and edge computing place programmable defenses closest to users and bots, making them natural enforcement points for machine-learning (ML) driven inspection, throttling, and isolation. This survey synthesizes the landscape of AI-enhanced defenses deployed at the edge: (i) anomaly- and behavior-based Web Application Firewalls (WAFs) within broader Web Application and API Protection (WAAP), (ii) adaptive DDoS detection and mitigation, (iii) bot management that resists human-mimicry, and (iv) API discovery, positive security modeling, and encrypted-traffic anomaly analysis. We add a systematic survey method, a threat taxonomy mapped to edge-observable signals, evaluation metrics, deployment playbooks, and governance guidance. We conclude with a research agenda spanning XAI, adversarial robustness, and autonomous multi-agent defense. Our findings indicate that edge-centric AI measurably improves time-to-detect and time-to-mitigate while reducing data movement and enhancing compliance, yet introduces new risks around model abuse, poisoning, and governance.



## **26. Degrading Voice: A Comprehensive Overview of Robust Voice Conversion Through Input Manipulation**

eess.AS

**SubmitDate**: 2025-12-06    [abs](http://arxiv.org/abs/2512.06304v1) [paper-pdf](https://arxiv.org/pdf/2512.06304v1)

**Authors**: Xining Song, Zhihua Wei, Rui Wang, Haixiao Hu, Yanxiang Chen, Meng Han

**Abstract**: Identity, accent, style, and emotions are essential components of human speech. Voice conversion (VC) techniques process the speech signals of two input speakers and other modalities of auxiliary information such as prompts and emotion tags. It changes para-linguistic features from one to another, while maintaining linguistic contents. Recently, VC models have made rapid advancements in both generation quality and personalization capabilities. These developments have attracted considerable attention for diverse applications, including privacy preservation, voice-print reproduction for the deceased, and dysarthric speech recovery. However, these models only learn non-robust features due to the clean training data. Subsequently, it results in unsatisfactory performances when dealing with degraded input speech in real-world scenarios, including additional noise, reverberation, adversarial attacks, or even minor perturbation. Hence, it demands robust deployments, especially in real-world settings. Although latest researches attempt to find potential attacks and countermeasures for VC systems, there remains a significant gap in the comprehensive understanding of how robust the VC model is under input manipulation. here also raises many questions: For instance, to what extent do different forms of input degradation attacks alter the expected output of VC models? Is there potential for optimizing these attack and defense strategies? To answer these questions, we classify existing attack and defense methods from the perspective of input manipulation and evaluate the impact of degraded input speech across four dimensions, including intelligibility, naturalness, timbre similarity, and subjective perception. Finally, we outline open issues and future directions.



## **27. Sparse Neural Approximations for Bilevel Adversarial Problems in Power Grids**

eess.SY

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.06187v1) [paper-pdf](https://arxiv.org/pdf/2512.06187v1)

**Authors**: Young-ho Cho, Harsha Nagarajan, Deepjyoti Deka, Hao Zhu

**Abstract**: The adversarial worst-case load shedding (AWLS) problem is pivotal for identifying critical contingencies under line outages. It is naturally cast as a bilevel program: the upper level simulates an attacker determining worst-case line failures, and the lower level corresponds to the defender's generator redispatch operations. Conventional techniques using optimality conditions render the bilevel, mixed-integer formulation computationally prohibitive due to the combinatorial number of topologies and the nonconvexity of AC power flow constraints. To address these challenges, we develop a novel single-level optimal value-function (OVF) reformulation and further leverage a data-driven neural network (NN) surrogate of the follower's optimal value. To ensure physical realizability, we embed the trained surrogate in a physics-constrained NN (PCNN) formulation that couples the OVF inequality with (relaxed) AC feasibility, yielding a mixed-integer convex model amenable to off-the-shelf solvers. To achieve scalability, we learn a sparse, area-partitioned NN via spectral clustering; the resulting block-sparse architecture scales essentially linearly with system size while preserving accuracy. Notably, our approach produces near-optimal worst-case failures and generalizes across loading conditions and unseen topologies, enabling rapid online recomputation. Numerical experiments on the IEEE 14- and 118-bus systems demonstrate the method's scalability and solution quality for large-scale contingency analysis, with an average optimality gap of 5.8% compared to conventional methods, while maintaining computation times under one minute.



## **28. DEFEND: Poisoned Model Detection and Malicious Client Exclusion Mechanism for Secure Federated Learning-based Road Condition Classification**

cs.CR

Accepted to the 41st ACM/SIGAPP Symposium on Applied Computing (SAC 2026)

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.06172v1) [paper-pdf](https://arxiv.org/pdf/2512.06172v1)

**Authors**: Sheng Liu, Panos Papadimitratos

**Abstract**: Federated Learning (FL) has drawn the attention of the Intelligent Transportation Systems (ITS) community. FL can train various models for ITS tasks, notably camera-based Road Condition Classification (RCC), in a privacy-preserving collaborative way. However, opening up to collaboration also opens FL-based RCC systems to adversaries, i.e., misbehaving participants that can launch Targeted Label-Flipping Attacks (TLFAs) and threaten transportation safety. Adversaries mounting TLFAs poison training data to misguide model predictions, from an actual source class (e.g., wet road) to a wrongly perceived target class (e.g., dry road). Existing countermeasures against poisoning attacks cannot maintain model performance under TLFAs close to the performance level in attack-free scenarios, because they lack specific model misbehavior detection for TLFAs and neglect client exclusion after the detection. To close this research gap, we propose DEFEND, which includes a poisoned model detection strategy that leverages neuron-wise magnitude analysis for attack goal identification and Gaussian Mixture Model (GMM)-based clustering. DEFEND discards poisoned model contributions in each round and adapts accordingly client ratings, eventually excluding malicious clients. Extensive evaluation involving various FL-RCC models and tasks shows that DEFEND can thwart TLFAs and outperform seven baseline countermeasures, with at least 15.78% improvement, with DEFEND remarkably achieving under attack the same performance as in attack-free scenarios.



## **29. Toward Patch Robustness Certification and Detection for Deep Learning Systems Beyond Consistent Samples**

cs.SE

accepted by IEEE Transactions on Reliability; extended technical report

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.06123v1) [paper-pdf](https://arxiv.org/pdf/2512.06123v1)

**Authors**: Qilin Zhou, Zhengyuan Wei, Haipeng Wang, Zhuo Wang, W. K. Chan

**Abstract**: Patch robustness certification is an emerging kind of provable defense technique against adversarial patch attacks for deep learning systems. Certified detection ensures the detection of all patched harmful versions of certified samples, which mitigates the failures of empirical defense techniques that could (easily) be compromised. However, existing certified detection methods are ineffective in certifying samples that are misclassified or whose mutants are inconsistently pre icted to different labels. This paper proposes HiCert, a novel masking-based certified detection technique. By focusing on the problem of mutants predicted with a label different from the true label with our formal analysis, HiCert formulates a novel formal relation between harmful samples generated by identified loopholes and their benign counterparts. By checking the bound of the maximum confidence among these potentially harmful (i.e., inconsistent) mutants of each benign sample, HiCert ensures that each harmful sample either has the minimum confidence among mutants that are predicted the same as the harmful sample itself below this bound, or has at least one mutant predicted with a label different from the harmful sample itself, formulated after two novel insights. As such, HiCert systematically certifies those inconsistent samples and consistent samples to a large extent. To our knowledge, HiCert is the first work capable of providing such a comprehensive patch robustness certification for certified detection. Our experiments show the high effectiveness of HiCert with a new state-of the-art performance: It certifies significantly more benign samples, including those inconsistent and consistent, and achieves significantly higher accuracy on those samples without warnings and a significantly lower false silent ratio.



## **30. When Privacy Isn't Synthetic: Hidden Data Leakage in Generative AI Models**

cs.LG

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.06062v1) [paper-pdf](https://arxiv.org/pdf/2512.06062v1)

**Authors**: S. M. Mustaqim, Anantaa Kotal, Paul H. Yi

**Abstract**: Generative models are increasingly used to produce privacy-preserving synthetic data as a safe alternative to sharing sensitive training datasets. However, we demonstrate that such synthetic releases can still leak information about the underlying training samples through structural overlap in the data manifold. We propose a black-box membership inference attack that exploits this vulnerability without requiring access to model internals or real data. The attacker repeatedly queries the generative model to obtain large numbers of synthetic samples, performs unsupervised clustering to identify dense regions of the synthetic distribution, and then analyzes cluster medoids and neighborhoods that correspond to high-density regions in the original training data. These neighborhoods act as proxies for training samples, enabling the adversary to infer membership or reconstruct approximate records. Our experiments across healthcare, finance, and other sensitive domains show that cluster overlap between real and synthetic data leads to measurable membership leakage-even when the generator is trained with differential privacy or other noise mechanisms. The results highlight an under-explored attack surface in synthetic data generation pipelines and call for stronger privacy guarantees that account for distributional neighborhood inference rather than sample-level memorization alone, underscoring its role in privacy-preserving data publishing. Implementation and evaluation code are publicly available at:github.com/Cluster-Medoid-Leakage-Attack.



## **31. Exposing Pink Slime Journalism: Linguistic Signatures and Robust Detection Against LLM-Generated Threats**

cs.CL

Published in RANLP 2025

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.05331v1) [paper-pdf](https://arxiv.org/pdf/2512.05331v1)

**Authors**: Sadat Shahriar, Navid Ayoobi, Arjun Mukherjee, Mostafa Musharrat, Sai Vishnu Vamsi

**Abstract**: The local news landscape, a vital source of reliable information for 28 million Americans, faces a growing threat from Pink Slime Journalism, a low-quality, auto-generated articles that mimic legitimate local reporting. Detecting these deceptive articles requires a fine-grained analysis of their linguistic, stylistic, and lexical characteristics. In this work, we conduct a comprehensive study to uncover the distinguishing patterns of Pink Slime content and propose detection strategies based on these insights. Beyond traditional generation methods, we highlight a new adversarial vector: modifications through large language models (LLMs). Our findings reveal that even consumer-accessible LLMs can significantly undermine existing detection systems, reducing their performance by up to 40% in F1-score. To counter this threat, we introduce a robust learning framework specifically designed to resist LLM-based adversarial attacks and adapt to the evolving landscape of automated pink slime journalism, and showed and improvement by up to 27%.



## **32. VEIL: Jailbreaking Text-to-Video Models via Visual Exploitation from Implicit Language**

cs.CV

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2511.13127v2) [paper-pdf](https://arxiv.org/pdf/2511.13127v2)

**Authors**: Zonghao Ying, Moyang Chen, Nizhang Li, Zhiqiang Wang, Wenxin Zhang, Quanchen Zou, Zonglei Jing, Aishan Liu, Xianglong Liu

**Abstract**: Jailbreak attacks can circumvent model safety guardrails and reveal critical blind spots. Prior attacks on text-to-video (T2V) models typically add adversarial perturbations to obviously unsafe prompts, which are often easy to detect and defend. In contrast, we show that benign-looking prompts containing rich, implicit cues can induce T2V models to generate semantically unsafe videos that both violate policy and preserve the original (blocked) intent. To realize this, we propose VEIL, a jailbreak framework that leverages T2V models' cross-modal associative patterns via a modular prompt design. Specifically, our prompts combine three components: neutral scene anchors, which provide the surface-level scene description extracted from the blocked intent to maintain plausibility; latent auditory triggers, textual descriptions of innocuous-sounding audio events (e.g., creaking, muffled noises) that exploit learned audio-visual co-occurrence priors to bias the model toward particular unsafe visual concepts; and stylistic modulators, cinematic directives (e.g., camera framing, atmosphere) that amplify and stabilize the latent trigger's effect. We formalize attack generation as a constrained optimization over the above modular prompt space and solve it with a guided search procedure that balances stealth and effectiveness. Extensive experiments over 7 T2V models demonstrate the efficacy of our attack, achieving a 23 percent improvement in average attack success rate in commercial models. Our demos and codes can be found at https://github.com/NY1024/VEIL.



## **33. 3D-ANC: Adaptive Neural Collapse for Robust 3D Point Cloud Recognition**

cs.CV

AAAI 2026

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2511.07040v2) [paper-pdf](https://arxiv.org/pdf/2511.07040v2)

**Authors**: Yuanmin Huang, Wenxuan Li, Mi Zhang, Xiaohan Zhang, Xiaoyu You, Min Yang

**Abstract**: Deep neural networks have recently achieved notable progress in 3D point cloud recognition, yet their vulnerability to adversarial perturbations poses critical security challenges in practical deployments. Conventional defense mechanisms struggle to address the evolving landscape of multifaceted attack patterns. Through systematic analysis of existing defenses, we identify that their unsatisfactory performance primarily originates from an entangled feature space, where adversarial attacks can be performed easily. To this end, we present 3D-ANC, a novel approach that capitalizes on the Neural Collapse (NC) mechanism to orchestrate discriminative feature learning. In particular, NC depicts where last-layer features and classifier weights jointly evolve into a simplex equiangular tight frame (ETF) arrangement, establishing maximally separable class prototypes. However, leveraging this advantage in 3D recognition confronts two substantial challenges: (1) prevalent class imbalance in point cloud datasets, and (2) complex geometric similarities between object categories. To tackle these obstacles, our solution combines an ETF-aligned classification module with an adaptive training framework consisting of representation-balanced learning (RBL) and dynamic feature direction loss (FDL). 3D-ANC seamlessly empowers existing models to develop disentangled feature spaces despite the complexity in 3D data distribution. Comprehensive evaluations state that 3D-ANC significantly improves the robustness of models with various structures on two datasets. For instance, DGCNN's classification accuracy is elevated from 27.2% to 80.9% on ModelNet40 -- a 53.7% absolute gain that surpasses leading baselines by 34.0%.



## **34. ZQBA: Zero Query Black-box Adversarial Attack**

cs.CV

Accepted in ICAART 2026 Conference

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2510.00769v2) [paper-pdf](https://arxiv.org/pdf/2510.00769v2)

**Authors**: Joana C. Costa, Tiago Roxo, Hugo Proença, Pedro R. M. Inácio

**Abstract**: Current black-box adversarial attacks either require multiple queries or diffusion models to produce adversarial samples that can impair the target model performance. However, these methods require training a surrogate loss or diffusion models to produce adversarial samples, which limits their applicability in real-world settings. Thus, we propose a Zero Query Black-box Adversarial (ZQBA) attack that exploits the representations of Deep Neural Networks (DNNs) to fool other networks. Instead of requiring thousands of queries to produce deceiving adversarial samples, we use the feature maps obtained from a DNN and add them to clean images to impair the classification of a target model. The results suggest that ZQBA can transfer the adversarial samples to different models and across various datasets, namely CIFAR and Tiny ImageNet. The experiments also show that ZQBA is more effective than state-of-the-art black-box attacks with a single query, while maintaining the imperceptibility of perturbations, evaluated both quantitatively (SSIM) and qualitatively, emphasizing the vulnerabilities of employing DNNs in real-world contexts. All the source code is available at https://github.com/Joana-Cabral/ZQBA.



## **35. Eyes-on-Me: Scalable RAG Poisoning through Transferable Attention-Steering Attractors**

cs.LG

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2510.00586v2) [paper-pdf](https://arxiv.org/pdf/2510.00586v2)

**Authors**: Yen-Shan Chen, Sian-Yao Huang, Cheng-Lin Yang, Yun-Nung Chen

**Abstract**: Existing data poisoning attacks on retrieval-augmented generation (RAG) systems scale poorly because they require costly optimization of poisoned documents for each target phrase. We introduce Eyes-on-Me, a modular attack that decomposes an adversarial document into reusable Attention Attractors and Focus Regions. Attractors are optimized to direct attention to the Focus Region. Attackers can then insert semantic baits for the retriever or malicious instructions for the generator, adapting to new targets at near zero cost. This is achieved by steering a small subset of attention heads that we empirically identify as strongly correlated with attack success. Across 18 end-to-end RAG settings (3 datasets $\times$ 2 retrievers $\times$ 3 generators), Eyes-on-Me raises average attack success rates from 21.9 to 57.8 (+35.9 points, 2.6$\times$ over prior work). A single optimized attractor transfers to unseen black box retrievers and generators without retraining. Our findings establish a scalable paradigm for RAG data poisoning and show that modular, reusable components pose a practical threat to modern AI systems. They also reveal a strong link between attention concentration and model outputs, informing interpretability research.



## **36. Privacy-Preserving Decentralized Federated Learning via Explainable Adaptive Differential Privacy**

cs.CR

20 pages

**SubmitDate**: 2025-12-06    [abs](http://arxiv.org/abs/2509.10691v2) [paper-pdf](https://arxiv.org/pdf/2509.10691v2)

**Authors**: Fardin Jalil Piran, Zhiling Chen, Yang Zhang, Qianyu Zhou, Jiong Tang, Farhad Imani

**Abstract**: Decentralized Federated Learning (DFL) enables collaborative model training without a central server, but it remains vulnerable to privacy leakage because shared model updates can expose sensitive information through inversion, reconstruction, and membership inference attacks. Differential Privacy (DP) provides formal safeguards, yet existing DP-enabled DFL methods operate as black-boxes that cannot track cumulative noise added across clients and rounds, forcing each participant to inject worst-case perturbations that severely degrade accuracy. We propose PrivateDFL, a new explainable and privacy-preserving framework that addresses this gap by combining a HyperDimensional Computing (HD) model with a transparent DP noise accountant tailored to decentralized learning. HD offers structured, noise-tolerant high-dimensional representations, while the accountant explicitly tracks cumulative perturbations so each client adds only the minimal incremental noise required to satisfy its (epsilon, delta) budget. This yields significantly tighter and more interpretable privacy-utility tradeoffs than prior DP-DFL approaches. Experiments on MNIST (image), ISOLET (speech), and UCI-HAR (wearable sensor) show that PrivateDFL consistently surpasses centralized DP-SGD and Renyi-DP Transformer and deep learning baselines under both IID and non-IID partitions, improving accuracy by up to 24.4% on MNIST, over 80% on ISOLET, and 14.7% on UCI-HAR, while reducing inference latency by up to 76 times and energy consumption by up to 36 times. These results position PrivateDFL as an efficient and trustworthy solution for privacy-sensitive pattern recognition applications such as healthcare, finance, human-activity monitoring, and industrial sensing. Future work will extend the accountant to adversarial participation, heterogeneous privacy budgets, and dynamic topologies.



## **37. Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking**

cs.CR

Accepted by AAAI 2026

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2509.05835v2) [paper-pdf](https://arxiv.org/pdf/2509.05835v2)

**Authors**: Lingfeng Yao, Chenpei Huang, Shengyao Wang, Junpei Xue, Hanqing Guo, Jiang Liu, Phone Lin, Tomoaki Ohtsuki, Miao Pan

**Abstract**: As generative audio models are rapidly evolving, AI-generated audios increasingly raise concerns about copyright infringement and misinformation spread. Audio watermarking, as a proactive defense, can embed secret messages into audio for copyright protection and source verification. However, current neural audio watermarking methods focus primarily on the imperceptibility and robustness of watermarking, while ignoring its vulnerability to security attacks. In this paper, we develop a simple yet powerful attack: the overwriting attack that overwrites the legitimate audio watermark with a forged one and makes the original legitimate watermark undetectable. Based on the audio watermarking information that the adversary has, we propose three categories of overwriting attacks, i.e., white-box, gray-box, and black-box attacks. We also thoroughly evaluate the proposed attacks on state-of-the-art neural audio watermarking methods. Experimental results demonstrate that the proposed overwriting attacks can effectively compromise existing watermarking schemes across various settings and achieve a nearly 100% attack success rate. The practicality and effectiveness of the proposed overwriting attacks expose security flaws in existing neural audio watermarking systems, underscoring the need to enhance security in future audio watermarking designs.



## **38. DASH: A Meta-Attack Framework for Synthesizing Effective and Stealthy Adversarial Examples**

cs.CV

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2508.13309v2) [paper-pdf](https://arxiv.org/pdf/2508.13309v2)

**Authors**: Abdullah Al Nomaan Nafi, Habibur Rahaman, Zafaryab Haider, Tanzim Mahfuz, Fnu Suya, Swarup Bhunia, Prabuddha Chakraborty

**Abstract**: Numerous techniques have been proposed for generating adversarial examples in white-box settings under strict Lp-norm constraints. However, such norm-bounded examples often fail to align well with human perception, and only recently have a few methods begun specifically exploring perceptually aligned adversarial examples. Moreover, it remains unclear whether insights from Lp-constrained attacks can be effectively leveraged to improve perceptual efficacy. In this paper, we introduce DAASH, a fully differentiable meta-attack framework that generates effective and perceptually aligned adversarial examples by strategically composing existing Lp-based attack methods. DAASH operates in a multi-stage fashion: at each stage, it aggregates candidate adversarial examples from multiple base attacks using learned, adaptive weights and propagates the result to the next stage. A novel meta-loss function guides this process by jointly minimizing misclassification loss and perceptual distortion, enabling the framework to dynamically modulate the contribution of each base attack throughout the stages. We evaluate DAASH on adversarially trained models across CIFAR-10, CIFAR-100, and ImageNet. Despite relying solely on Lp-constrained based methods, DAASH significantly outperforms state-of-the-art perceptual attacks such as AdvAD -- achieving higher attack success rates (e.g., 20.63\% improvement) and superior visual quality, as measured by SSIM, LPIPS, and FID (improvements $\approx$ of 11, 0.015, and 5.7, respectively). Furthermore, DAASH generalizes well to unseen defenses, making it a practical and strong baseline for evaluating robustness without requiring handcrafted adaptive attacks for each new defense.



## **39. How Not to Detect Prompt Injections with an LLM**

cs.CR

**SubmitDate**: 2025-12-06    [abs](http://arxiv.org/abs/2507.05630v3) [paper-pdf](https://arxiv.org/pdf/2507.05630v3)

**Authors**: Sarthak Choudhary, Divyam Anshumaan, Nils Palumbo, Somesh Jha

**Abstract**: LLM-integrated applications and agents are vulnerable to prompt injection attacks, where adversaries embed malicious instructions within seemingly benign input data to manipulate the LLM's intended behavior. Recent defenses based on known-answer detection (KAD) scheme have reported near-perfect performance by observing an LLM's output to classify input data as clean or contaminated. KAD attempts to repurpose the very susceptibility to prompt injection as a defensive mechanism. We formally characterize the KAD scheme and uncover a structural vulnerability that invalidates its core security premise. To exploit this fundamental vulnerability, we methodically design an adaptive attack, DataFlip. It consistently evades KAD defenses, achieving detection rates as low as $0\%$ while reliably inducing malicious behavior with a success rate of $91\%$, all without requiring white-box access to the LLM or any optimization procedures.



## **40. Analyzing PDFs like Binaries: Adversarially Robust PDF Malware Analysis via Intermediate Representation and Language Model**

cs.CR

Accepted by ACM CCS 2025

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2506.17162v2) [paper-pdf](https://arxiv.org/pdf/2506.17162v2)

**Authors**: Side Liu, Jiang Ming, Guodong Zhou, Xinyi Liu, Jianming Fu, Guojun Peng

**Abstract**: Malicious PDF files have emerged as a persistent threat and become a popular attack vector in web-based attacks. While machine learning-based PDF malware classifiers have shown promise, these classifiers are often susceptible to adversarial attacks, undermining their reliability. To address this issue, recent studies have aimed to enhance the robustness of PDF classifiers. Despite these efforts, the feature engineering underlying these studies remains outdated. Consequently, even with the application of cutting-edge machine learning techniques, these approaches fail to fundamentally resolve the issue of feature instability.   To tackle this, we propose a novel approach for PDF feature extraction and PDF malware detection. We introduce the PDFObj IR (PDF Object Intermediate Representation), an assembly-like language framework for PDF objects, from which we extract semantic features using a pretrained language model. Additionally, we construct an Object Reference Graph to capture structural features, drawing inspiration from program analysis. This dual approach enables us to analyze and detect PDF malware based on both semantic and structural features. Experimental results demonstrate that our proposed classifier achieves strong adversarial robustness while maintaining an exceptionally low false positive rate of only 0.07% on baseline dataset compared to state-of-the-art PDF malware classifiers.



## **41. Exploring Adversarial Watermarking in Transformer-Based Models: Transferability and Robustness Against Defense Mechanism for Medical Images**

cs.CV

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2506.06389v2) [paper-pdf](https://arxiv.org/pdf/2506.06389v2)

**Authors**: Rifat Sadik, Tanvir Rahman, Arpan Bhattacharjee, Bikash Chandra Halder, Ismail Hossain, Rifat Sarker Aoyon, Md. Golam Rabiul Alam, Jia Uddin

**Abstract**: Deep learning models have shown remarkable success in dermatological image analysis, offering potential for automated skin disease diagnosis. Previously, convolutional neural network(CNN) based architectures have achieved immense popularity and success in computer vision (CV) based task like skin image recognition, generation and video analysis. But with the emergence of transformer based models, CV tasks are now are nowadays carrying out using these models. Vision Transformers (ViTs) is such a transformer-based models that have shown success in computer vision. It uses self-attention mechanisms to achieve state-of-the-art performance across various tasks. However, their reliance on global attention mechanisms makes them susceptible to adversarial perturbations. This paper aims to investigate the susceptibility of ViTs for medical images to adversarial watermarking-a method that adds so-called imperceptible perturbations in order to fool models. By generating adversarial watermarks through Projected Gradient Descent (PGD), we examine the transferability of such attacks to CNNs and analyze the performance defense mechanism -- adversarial training. Results indicate that while performance is not compromised for clean images, ViTs certainly become much more vulnerable to adversarial attacks: an accuracy drop of as low as 27.6%. Nevertheless, adversarial training raises it up to 90.0%.



## **42. Unlearning Inversion Attacks for Graph Neural Networks**

cs.LG

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2506.00808v2) [paper-pdf](https://arxiv.org/pdf/2506.00808v2)

**Authors**: Jiahao Zhang, Yilong Wang, Zhiwei Zhang, Xiaorui Liu, Suhang Wang

**Abstract**: Graph unlearning methods aim to efficiently remove the impact of sensitive data from trained GNNs without full retraining, assuming that deleted information cannot be recovered. In this work, we challenge this assumption by introducing the graph unlearning inversion attack: given only black-box access to an unlearned GNN and partial graph knowledge, can an adversary reconstruct the removed edges? We identify two key challenges: varying probability-similarity thresholds for unlearned versus retained edges, and the difficulty of locating unlearned edge endpoints, and address them with TrendAttack. First, we derive and exploit the confidence pitfall, a theoretical and empirical pattern showing that nodes adjacent to unlearned edges exhibit a large drop in model confidence. Second, we design an adaptive prediction mechanism that applies different similarity thresholds to unlearned and other membership edges. Our framework flexibly integrates existing membership inference techniques and extends them with trend features. Experiments on four real-world datasets demonstrate that TrendAttack significantly outperforms state-of-the-art GNN membership inference baselines, exposing a critical privacy vulnerability in current graph unlearning methods.



## **43. Are Time-Series Foundation Models Deployment-Ready? A Systematic Study of Adversarial Robustness Across Domains**

cs.LG

Preprint

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2505.19397v2) [paper-pdf](https://arxiv.org/pdf/2505.19397v2)

**Authors**: Jiawen Zhang, Zhenwei Zhang, Shun Zheng, Xumeng Wen, Jia Li, Jiang Bian

**Abstract**: Time-Series Foundation Models (TSFMs) are rapidly transitioning from research prototypes to core components of critical decision-making systems, driven by their impressive zero-shot forecasting capabilities. However, as their deployment surges, a critical blind spot remains: their fragility under adversarial attacks. This lack of scrutiny poses severe risks, particularly as TSFMs enter high-stakes environments vulnerable to manipulation. We present a systematic, diagnostic study arguing that for TSFMs, robustness is not merely a secondary metric but a prerequisite for trustworthy deployment comparable to accuracy. Our evaluation framework, explicitly tailored to the unique constraints of time series, incorporates normalized, sparsity-aware perturbation budgets and unified scale-invariant metrics across white-box and black-box settings. Across six representative TSFMs, we demonstrate that current architectures are alarmingly brittle: even small perturbations can reliably steer forecasts toward specific failure modes, such as trend flips and malicious drifts. We uncover TSFM-specific vulnerability patterns, including horizon-proximal brittleness, increased susceptibility with longer context windows, and weak cross-model transfer that points to model-specific failure modes rather than generic distortions. Finally, we show that simple adversarial fine-tuning offers a cost-effective path to substantial robustness gains, even with out-of-domain data. This work bridges the gap between TSFM capabilities and safety constraints, offering essential guidance for hardening the next generation of forecasting systems.



## **44. Evaluating the robustness of adversarial defenses in malware detection systems**

cs.CR

Published in Computers & Electrical Engineering (Elsevier), Volume 130, February 2026, Article 110845

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2505.09342v2) [paper-pdf](https://arxiv.org/pdf/2505.09342v2)

**Authors**: Mostafa Jafari, Alireza Shameli-Sendi

**Abstract**: Machine learning is a key tool for Android malware detection, effectively identifying malicious patterns in apps. However, ML-based detectors are vulnerable to evasion attacks, where small, crafted changes bypass detection. Despite progress in adversarial defenses, the lack of comprehensive evaluation frameworks in binary-constrained domains limits understanding of their robustness. We introduce two key contributions. First, Prioritized Binary Rounding, a technique to convert continuous perturbations into binary feature spaces while preserving high attack success and low perturbation size. Second, the sigma-binary attack, a novel adversarial method for binary domains, designed to achieve attack goals with minimal feature changes. Experiments on the Malscan dataset show that sigma-binary outperforms existing attacks and exposes key vulnerabilities in state-of-the-art defenses. Defenses equipped with adversary detectors, such as KDE, DLA, DNN+, and ICNN, exhibit significant brittleness, with attack success rates exceeding 90% using fewer than 10 feature modifications and reaching 100% with just 20. Adversarially trained defenses, including AT-rFGSM-k, AT-MaxMA, improves robustness under small budgets but remains vulnerable to unrestricted perturbations, with attack success rates of 99.45% and 96.62%, respectively. Although PAD-SMA demonstrates strong robustness against state-of-the-art gradient-based adversarial attacks by maintaining an attack success rate below 16.55%, the sigma-binary attack significantly outperforms these methods, achieving a 94.56% success rate under unrestricted perturbations. These findings highlight the critical need for precise method like sigma-binary to expose hidden vulnerabilities in existing defenses and support the development of more resilient malware detection systems.



## **45. Exploring Adversarial Obstacle Attacks in Search-based Path Planning for Autonomous Mobile Robots**

cs.RO

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2504.06154v2) [paper-pdf](https://arxiv.org/pdf/2504.06154v2)

**Authors**: Adrian Szvoren, Jianwei Liu, Dimitrios Kanoulas, Nilufer Tuptuk

**Abstract**: Path planning algorithms, such as the search-based A*, are a critical component of autonomous mobile robotics, enabling robots to navigate from a starting point to a destination efficiently and safely. We investigated the resilience of the A* algorithm in the face of potential adversarial interventions known as obstacle attacks. The adversary's goal is to delay the robot's timely arrival at its destination by introducing obstacles along its original path.   We developed malicious software to execute the attacks and conducted experiments to assess their impact, both in simulation using TurtleBot in Gazebo and in real-world deployment with the Unitree Go1 robot. In simulation, the attacks resulted in an average delay of 36\%, with the most significant delays occurring in scenarios where the robot was forced to take substantially longer alternative paths. In real-world experiments, the delays were even more pronounced, with all attacks successfully rerouting the robot and causing measurable disruptions. These results highlight that the algorithm's robustness is not solely an attribute of its design but is significantly influenced by the operational environment. For example, in constrained environments like tunnels, the delays were maximized due to the limited availability of alternative routes.



## **46. Causal Interpretability for Adversarial Robustness: A Hybrid Generative Classification Approach**

cs.CV

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2412.20025v2) [paper-pdf](https://arxiv.org/pdf/2412.20025v2)

**Authors**: Chunheng Zhao, Pierluigi Pisu, Gurcan Comert, Negash Begashaw, Varghese Vaidyan, Nina Christine Hubig

**Abstract**: Deep learning-based discriminative classifiers, despite their remarkable success, remain vulnerable to adversarial examples that can mislead model predictions. While adversarial training can enhance robustness, it fails to address the intrinsic vulnerability stemming from the opaque nature of these black-box models. We present a deep ensemble model that combines discriminative features with generative models to achieve both high accuracy and adversarial robustness. Our approach integrates a bottom-level pre-trained discriminative network for feature extraction with a top-level generative classification network that models adversarial input distributions through a deep latent variable model. Using variational Bayes, our model achieves superior robustness against white-box adversarial attacks without adversarial training. Extensive experiments on CIFAR-10 and CIFAR-100 demonstrate our model's superior adversarial robustness. Through evaluations using counterfactual metrics and feature interaction-based metrics, we establish correlations between model interpretability and adversarial robustness. Additionally, preliminary results on Tiny-ImageNet validate our approach's scalability to more complex datasets, offering a practical solution for developing robust image classification models.



## **47. AEIOU: A Unified Defense Framework against NSFW Prompts in Text-to-Image Models**

cs.CR

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2412.18123v3) [paper-pdf](https://arxiv.org/pdf/2412.18123v3)

**Authors**: Yiming Wang, Jiahao Chen, Qingming Li, Tong Zhang, Rui Zeng, Xing Yang, Shouling Ji

**Abstract**: As text-to-image (T2I) models advance and gain widespread adoption, their associated safety concerns are becoming increasingly critical. Malicious users exploit these models to generate Not-Safe-for-Work (NSFW) images using harmful or adversarial prompts, underscoring the need for effective safeguards to ensure the integrity and compliance of model outputs. However, existing detection methods often exhibit low accuracy and inefficiency. In this paper, we propose AEIOU, a defense framework that is adaptable, efficient, interpretable, optimizable, and unified against NSFW prompts in T2I models. AEIOU extracts NSFW features from the hidden states of the model's text encoder, utilizing the separable nature of these features to detect NSFW prompts. The detection process is efficient, requiring minimal inference time. AEIOU also offers real-time interpretation of results and supports optimization through data augmentation techniques. The framework is versatile, accommodating various T2I architectures. Our extensive experiments show that AEIOU significantly outperforms both commercial and open-source moderation tools, achieving over 95\% accuracy across all datasets and improving efficiency by at least tenfold. It effectively counters adaptive attacks and excels in few-shot and multi-label scenarios.



## **48. GLL: A Differentiable Graph Learning Layer for Neural Networks**

cs.LG

58 pages, 12 figures. Preprint. Submitted to the Journal of Machine Learning Research. v2: several new experiments, improved exposition

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2412.08016v2) [paper-pdf](https://arxiv.org/pdf/2412.08016v2)

**Authors**: Jason Brown, Bohan Chen, Harris Hardiman-Mostow, Jeff Calder, Andrea L. Bertozzi

**Abstract**: Standard deep learning architectures used for classification generate label predictions with a projection head and softmax activation function. Although successful, these methods fail to leverage the relational information between samples for generating label predictions. In recent works, graph-based learning techniques, namely Laplace learning, have been heuristically combined with neural networks for both supervised and semi-supervised learning (SSL) tasks. However, prior works approximate the gradient of the loss function with respect to the graph learning algorithm or decouple the processes; end-to-end integration with neural networks is not achieved. In this work, we derive backpropagation equations, via the adjoint method, for inclusion of a general family of graph learning layers into a neural network. The resulting method, distinct from graph neural networks, allows us to precisely integrate similarity graph construction and graph Laplacian-based label propagation into a neural network layer, replacing a projection head and softmax activation function for general classification task. Our experimental results demonstrate smooth label transitions across data, improved generalization and robustness to adversarial attacks, and improved training dynamics compared to a standard softmax-based approach.



## **49. Edge-Only Universal Adversarial Attacks in Distributed Learning**

cs.CR

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2411.10500v2) [paper-pdf](https://arxiv.org/pdf/2411.10500v2)

**Authors**: Giulio Rossolini, Tommaso Baldi, Alessandro Biondi, Giorgio Buttazzo

**Abstract**: Distributed learning frameworks, which partition neural network models across multiple computing nodes, enhance efficiency in collaborative edge-cloud systems, but may also introduce new vulnerabilities to evasion attacks, often in the form of adversarial perturbations. In this work, we present a new threat model that explores the feasibility of generating universal adversarial perturbations (UAPs) when the attacker has access only to the edge portion of the model, consisting of its initial network layers. Unlike traditional attacks that require full model knowledge, our approach shows that adversaries can induce effective mispredictions in the unknown cloud component by manipulating key feature representations at the edge. Following the proposed threat model, we introduce both edge-only untargeted and targeted formulations of UAPs designed to control intermediate features before the split point. Our results on ImageNet demonstrate strong attack transferability to the unknown cloud part, and we compare the proposed method with classical white-box and black-box techniques, highlighting its effectiveness. Additionally, we analyze the capability of an attacker to achieve targeted adversarial effects with edge-only knowledge, revealing intriguing behaviors across multiple networks. By introducing the first adversarial attacks with edge-only knowledge in split inference, this work underscores the importance of addressing partial model access in adversarial robustness, encouraging further research in this area.



## **50. Attacking All Tasks at Once Using Adversarial Examples in Multi-Task Learning**

cs.LG

Accpeted by Neurocomputing at 10 September 2025

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2305.12066v4) [paper-pdf](https://arxiv.org/pdf/2305.12066v4)

**Authors**: Lijun Zhang, Xiao Liu, Kaleel Mahmood, Caiwen Ding, Hui Guan

**Abstract**: Visual content understanding frequently relies on multi-task models to extract robust representations of a single visual input for multiple downstream tasks. However, in comparison to extensively studied single-task models, the adversarial robustness of multi-task models has received significantly less attention and many questions remain unclear: 1) How robust are multi-task models to single task adversarial attacks, 2) Can adversarial attacks be designed to simultaneously attack all tasks in a multi-task model, and 3) How does parameter sharing across tasks affect multi-task model robustness to adversarial attacks? This paper aims to answer these questions through careful analysis and rigorous experimentation. First, we analyze the inherent drawbacks of two commonly-used adaptations of single-task white-box attacks in attacking multi-task models. We then propose a novel attack framework, Dynamic Gradient Balancing Attack (DGBA). Our framework poses the problem of attacking all tasks in a multi-task model as an optimization problem that can be efficiently solved through integer linear programming. Extensive evaluation on two popular MTL benchmarks, NYUv2 and Tiny-Taxonomy, demonstrates the effectiveness of DGBA compared to baselines in attacking both clean and adversarially trained multi-task models. Our results also reveal a fundamental trade-off between improving task accuracy via parameter sharing across tasks and undermining model robustness due to increased attack transferability from parameter sharing.



