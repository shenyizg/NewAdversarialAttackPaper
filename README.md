# Latest Adversarial Attack Papers
**update at 2025-08-06 18:23:38**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. LeakyCLIP: Extracting Training Data from CLIP**

cs.CR

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2508.00756v2) [paper-pdf](http://arxiv.org/pdf/2508.00756v2)

**Authors**: Yunhao Chen, Shujie Wang, Xin Wang, Xingjun Ma

**Abstract**: Understanding the memorization and privacy leakage risks in Contrastive Language--Image Pretraining (CLIP) is critical for ensuring the security of multimodal models. Recent studies have demonstrated the feasibility of extracting sensitive training examples from diffusion models, with conditional diffusion models exhibiting a stronger tendency to memorize and leak information. In this work, we investigate data memorization and extraction risks in CLIP through the lens of CLIP inversion, a process that aims to reconstruct training images from text prompts. To this end, we introduce \textbf{LeakyCLIP}, a novel attack framework designed to achieve high-quality, semantically accurate image reconstruction from CLIP embeddings. We identify three key challenges in CLIP inversion: 1) non-robust features, 2) limited visual semantics in text embeddings, and 3) low reconstruction fidelity. To address these challenges, LeakyCLIP employs 1) adversarial fine-tuning to enhance optimization smoothness, 2) linear transformation-based embedding alignment, and 3) Stable Diffusion-based refinement to improve fidelity. Empirical results demonstrate the superiority of LeakyCLIP, achieving over 358% improvement in Structural Similarity Index Measure (SSIM) for ViT-B-16 compared to baseline methods on LAION-2B subset. Furthermore, we uncover a pervasive leakage risk, showing that training data membership can even be successfully inferred from the metrics of low-fidelity reconstructions. Our work introduces a practical method for CLIP inversion while offering novel insights into the nature and scope of privacy risks in multimodal models.



## **2. Set-Based Training for Neural Network Verification**

cs.LG

published at Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2401.14961v4) [paper-pdf](http://arxiv.org/pdf/2401.14961v4)

**Authors**: Lukas Koller, Tobias Ladner, Matthias Althoff

**Abstract**: Neural networks are vulnerable to adversarial attacks, i.e., small input perturbations can significantly affect the outputs of a neural network. Therefore, to ensure safety of neural networks in safety-critical environments, the robustness of a neural network must be formally verified against input perturbations, e.g., from noisy sensors. To improve the robustness of neural networks and thus simplify the formal verification, we present a novel set-based training procedure in which we compute the set of possible outputs given the set of possible inputs and compute for the first time a gradient set, i.e., each possible output has a different gradient. Therefore, we can directly reduce the size of the output enclosure by choosing gradients toward its center. Small output enclosures increase the robustness of a neural network and, at the same time, simplify its formal verification. The latter benefit is due to the fact that a larger size of propagated sets increases the conservatism of most verification methods. Our extensive evaluation demonstrates that set-based training produces robust neural networks with competitive performance, which can be verified using fast (polynomial-time) verification algorithms due to the reduced output set.



## **3. IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves**

cs.CV

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2411.00827v4) [paper-pdf](http://arxiv.org/pdf/2411.00827v4)

**Authors**: Ruofan Wang, Juncheng Li, Yixu Wang, Bo Wang, Xiaosen Wang, Yan Teng, Yingchun Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) gain prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks-techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multimodal data has constrained current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which often lack effectiveness and diversity across different contexts. In this paper, we propose IDEATOR, a novel jailbreak method that autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is grounded in the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR leverages a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Extensive experiments demonstrate IDEATOR's high effectiveness and transferability, achieving a 94% attack success rate (ASR) in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high ASRs of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Chameleon, respectively. Building on IDEATOR's strong transferability and automated process, we introduce the VLJailbreakBench, a safety benchmark comprising 3,654 multimodal jailbreak samples. Our benchmark results on 11 recently released VLMs reveal significant gaps in safety alignment. For instance, our challenge set achieves ASRs of 46.31% on GPT-4o and 19.65% on Claude-3.5-Sonnet, underscoring the urgent need for stronger defenses.



## **4. Smart Car Privacy: Survey of Attacks and Privacy Issues**

cs.CR

13 pages, 16 figures

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2508.03413v1) [paper-pdf](http://arxiv.org/pdf/2508.03413v1)

**Authors**: Akshay Madhav Deshmukh

**Abstract**: Automobiles are becoming increasingly important in our day to day life. Modern automobiles are highly computerized and hence potentially vulnerable to attack. Providing many wireless connectivity for vehicles enables a bridge between vehicles and their external environments. Such a connected vehicle solution is expected to be the next frontier for automotive revolution and the key to the evolution to next generation intelligent transportation systems. Vehicular Ad hoc Networks (VANETs) are emerging mobile ad hoc network technologies incorporating mobile routing protocols for inter-vehicle data communications to support intelligent transportation systems. Thus security and privacy are the major concerns in VANETs due to the mobility of the vehicles. Thus designing security mechanisms to remove adversaries from the network remarkably important in VANETs.   This paper provides an overview of various vehicular network architectures. The evolution of security in modern vehicles. Various security and privacy attacks in VANETs with their defending mechanisms with examples and classify these mechanisms. It also provides an overview of various privacy implication that a vehicular network possess.



## **5. When Good Sounds Go Adversarial: Jailbreaking Audio-Language Models with Benign Inputs**

cs.SD

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2508.03365v1) [paper-pdf](http://arxiv.org/pdf/2508.03365v1)

**Authors**: Bodam Kim, Hiskias Dingeto, Taeyoun Kwon, Dasol Choi, DongGeon Lee, Haon Park, JaeHoon Lee, Jongho Shin

**Abstract**: As large language models become increasingly integrated into daily life, audio has emerged as a key interface for human-AI interaction. However, this convenience also introduces new vulnerabilities, making audio a potential attack surface for adversaries. Our research introduces WhisperInject, a two-stage adversarial audio attack framework that can manipulate state-of-the-art audio language models to generate harmful content. Our method uses imperceptible perturbations in audio inputs that remain benign to human listeners. The first stage uses a novel reward-based optimization method, Reinforcement Learning with Projected Gradient Descent (RL-PGD), to guide the target model to circumvent its own safety protocols and generate harmful native responses. This native harmful response then serves as the target for Stage 2, Payload Injection, where we use Projected Gradient Descent (PGD) to optimize subtle perturbations that are embedded into benign audio carriers, such as weather queries or greeting messages. Validated under the rigorous StrongREJECT, LlamaGuard, as well as Human Evaluation safety evaluation framework, our experiments demonstrate a success rate exceeding 86% across Qwen2.5-Omni-3B, Qwen2.5-Omni-7B, and Phi-4-Multimodal. Our work demonstrates a new class of practical, audio-native threats, moving beyond theoretical exploits to reveal a feasible and covert method for manipulating AI behavior.



## **6. LADSG: Label-Anonymized Distillation and Similar Gradient Substitution for Label Privacy in Vertical Federated Learning**

cs.CR

Under review

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2506.06742v2) [paper-pdf](http://arxiv.org/pdf/2506.06742v2)

**Authors**: Zeyu Yan, Yifei Yao, Xuanbing Wen, Shixiong Zhang, Juli Zhang, Kai Fan

**Abstract**: Vertical Federated Learning (VFL) has emerged as a promising paradigm for collaborative model training across distributed feature spaces, which enables privacy-preserving learning without sharing raw data. However, recent studies have confirmed the feasibility of label inference attacks by internal adversaries. By strategically exploiting gradient vectors and semantic embeddings, attackers-through passive, active, or direct attacks-can accurately reconstruct private labels, leading to catastrophic data leakage. Existing defenses, which typically address isolated leakage vectors or are designed for specific types of attacks, remain vulnerable to emerging hybrid attacks that exploit multiple pathways simultaneously. To bridge this gap, we propose Label-Anonymized Defense with Substitution Gradient (LADSG), a unified and lightweight defense framework for VFL. LADSG first anonymizes true labels via soft distillation to reduce semantic exposure, then generates semantically-aligned substitute gradients to disrupt gradient-based leakage, and finally filters anomalous updates through gradient norm detection. It is scalable and compatible with standard VFL pipelines. Extensive experiments on six real-world datasets show that LADSG reduces the success rates of all three types of label inference attacks by 30-60% with minimal computational overhead, demonstrating its practical effectiveness.



## **7. BlockA2A: Towards Secure and Verifiable Agent-to-Agent Interoperability**

cs.CR

43 pages

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2508.01332v2) [paper-pdf](http://arxiv.org/pdf/2508.01332v2)

**Authors**: Zhenhua Zou, Zhuotao Liu, Lepeng Zhao, Qiuyang Zhan

**Abstract**: The rapid adoption of agentic AI, powered by large language models (LLMs), is transforming enterprise ecosystems with autonomous agents that execute complex workflows. Yet we observe several key security vulnerabilities in LLM-driven multi-agent systems (MASes): fragmented identity frameworks, insecure communication channels, and inadequate defenses against Byzantine agents or adversarial prompts. In this paper, we present the first systematic analysis of these emerging multi-agent risks and explain why the legacy security strategies cannot effectively address these risks. Afterwards, we propose BlockA2A, the first unified multi-agent trust framework that enables secure and verifiable and agent-to-agent interoperability. At a high level, BlockA2A adopts decentralized identifiers (DIDs) to enable fine-grained cross-domain agent authentication, blockchain-anchored ledgers to enable immutable auditability, and smart contracts to dynamically enforce context-aware access control policies. BlockA2A eliminates centralized trust bottlenecks, ensures message authenticity and execution integrity, and guarantees accountability across agent interactions. Furthermore, we propose a Defense Orchestration Engine (DOE) that actively neutralizes attacks through real-time mechanisms, including Byzantine agent flagging, reactive execution halting, and instant permission revocation. Empirical evaluations demonstrate BlockA2A's effectiveness in neutralizing prompt-based, communication-based, behavioral and systemic MAS attacks. We formalize its integration into existing MAS and showcase a practical implementation for Google's A2A protocol. Experiments confirm that BlockA2A and DOE operate with sub-second overhead, enabling scalable deployment in production LLM-based MAS environments.



## **8. ConfGuard: A Simple and Effective Backdoor Detection for Large Language Models**

cs.CR

Under review

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2508.01365v2) [paper-pdf](http://arxiv.org/pdf/2508.01365v2)

**Authors**: Zihan Wang, Rui Zhang, Hongwei Li, Wenshu Fan, Wenbo Jiang, Qingchuan Zhao, Guowen Xu

**Abstract**: Backdoor attacks pose a significant threat to Large Language Models (LLMs), where adversaries can embed hidden triggers to manipulate LLM's outputs. Most existing defense methods, primarily designed for classification tasks, are ineffective against the autoregressive nature and vast output space of LLMs, thereby suffering from poor performance and high latency. To address these limitations, we investigate the behavioral discrepancies between benign and backdoored LLMs in output space. We identify a critical phenomenon which we term sequence lock: a backdoored model generates the target sequence with abnormally high and consistent confidence compared to benign generation. Building on this insight, we propose ConfGuard, a lightweight and effective detection method that monitors a sliding window of token confidences to identify sequence lock. Extensive experiments demonstrate ConfGuard achieves a near 100\% true positive rate (TPR) and a negligible false positive rate (FPR) in the vast majority of cases. Crucially, the ConfGuard enables real-time detection almost without additional latency, making it a practical backdoor defense for real-world LLM deployments.



## **9. ProARD: progressive adversarial robustness distillation: provide wide range of robust students**

cs.LG

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2506.07666v2) [paper-pdf](http://arxiv.org/pdf/2506.07666v2)

**Authors**: Seyedhamidreza Mousavi, Seyedali Mousavi, Masoud Daneshtalab

**Abstract**: Adversarial Robustness Distillation (ARD) has emerged as an effective method to enhance the robustness of lightweight deep neural networks against adversarial attacks. Current ARD approaches have leveraged a large robust teacher network to train one robust lightweight student. However, due to the diverse range of edge devices and resource constraints, current approaches require training a new student network from scratch to meet specific constraints, leading to substantial computational costs and increased CO2 emissions. This paper proposes Progressive Adversarial Robustness Distillation (ProARD), enabling the efficient one-time training of a dynamic network that supports a diverse range of accurate and robust student networks without requiring retraining. We first make a dynamic deep neural network based on dynamic layers by encompassing variations in width, depth, and expansion in each design stage to support a wide range of architectures. Then, we consider the student network with the largest size as the dynamic teacher network. ProARD trains this dynamic network using a weight-sharing mechanism to jointly optimize the dynamic teacher network and its internal student networks. However, due to the high computational cost of calculating exact gradients for all the students within the dynamic network, a sampling mechanism is required to select a subset of students. We show that random student sampling in each iteration fails to produce accurate and robust students.



## **10. M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs**

cs.CL

Accepted to ACL 2025 (Main Track). Camera-ready version

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2503.04856v3) [paper-pdf](http://arxiv.org/pdf/2503.04856v3)

**Authors**: Junwoo Ha, Hyunjun Kim, Sangyoon Yu, Haon Park, Ashkan Yousefpour, Yuna Park, Suhyun Kim

**Abstract**: We introduce a novel framework for consolidating multi-turn adversarial ``jailbreak'' prompts into single-turn queries, significantly reducing the manual overhead required for adversarial testing of large language models (LLMs). While multi-turn human jailbreaks have been shown to yield high attack success rates, they demand considerable human effort and time. Our multi-turn-to-single-turn (M2S) methods -- Hyphenize, Numberize, and Pythonize -- systematically reformat multi-turn dialogues into structured single-turn prompts. Despite removing iterative back-and-forth interactions, these prompts preserve and often enhance adversarial potency: in extensive evaluations on the Multi-turn Human Jailbreak (MHJ) dataset, M2S methods achieve attack success rates from 70.6 percent to 95.9 percent across several state-of-the-art LLMs. Remarkably, the single-turn prompts outperform the original multi-turn attacks by as much as 17.5 percentage points while cutting token usage by more than half on average. Further analysis shows that embedding malicious requests in enumerated or code-like structures exploits ``contextual blindness'', bypassing both native guardrails and external input-output filters. By converting multi-turn conversations into concise single-turn prompts, the M2S framework provides a scalable tool for large-scale red teaming and reveals critical weaknesses in contemporary LLM defenses.



## **11. Towards Imperceptible JPEG Image Hiding: Multi-range Representations-driven Adversarial Stego Generation**

cs.CV

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2507.08343v2) [paper-pdf](http://arxiv.org/pdf/2507.08343v2)

**Authors**: Junxue Yang, Xin Liao, Weixuan Tang, Jianhua Yang, Zheng Qin

**Abstract**: Image hiding fully explores the hidden potential of deep learning-based models, aiming to conceal image-level messages within cover images and reveal them from stego images to achieve covert communication. Existing hiding schemes are easily detected by the naked eyes or steganalyzers due to the cover type confined to the spatial domain, single-range feature extraction and attacks, and insufficient loss constraints. To address these issues, we propose a multi-range representations-driven adversarial stego generation framework called MRAG for JPEG image hiding. This design stems from the fact that steganalyzers typically combine local-range and global-range information to better capture hidden traces. Specifically, MRAG integrates the local-range characteristic of the convolution and the global-range modeling of the transformer. Meanwhile, a features angle-norm disentanglement loss is designed to launch multi-range representations-driven feature-level adversarial attacks. It computes the adversarial loss between covers and stegos based on the surrogate steganalyzer's classified features, i.e., the features before the last fully connected layer. Under the dual constraints of features angle and norm, MRAG can delicately encode the concatenation of cover and secret into subtle adversarial perturbations from local and global ranges relevant to steganalysis. Therefore, the resulting stego can achieve visual and steganalysis imperceptibility. Moreover, coarse-grained and fine-grained frequency decomposition operations are devised to transform the input, introducing multi-grained information. Extensive experiments demonstrate that MRAG can achieve state-of-the-art performance.



## **12. Untraceable DeepFakes via Traceable Fingerprint Elimination**

cs.CR

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2508.03067v1) [paper-pdf](http://arxiv.org/pdf/2508.03067v1)

**Authors**: Jiewei Lai, Lan Zhang, Chen Tang, Pengcheng Sun, Xinming Wang, Yunhao Wang

**Abstract**: Recent advancements in DeepFakes attribution technologies have significantly enhanced forensic capabilities, enabling the extraction of traces left by generative models (GMs) in images, making DeepFakes traceable back to their source GMs. Meanwhile, several attacks have attempted to evade attribution models (AMs) for exploring their limitations, calling for more robust AMs. However, existing attacks fail to eliminate GMs' traces, thus can be mitigated by defensive measures. In this paper, we identify that untraceable DeepFakes can be achieved through a multiplicative attack, which can fundamentally eliminate GMs' traces, thereby evading AMs even enhanced with defensive measures. We design a universal and black-box attack method that trains an adversarial model solely using real data, applicable for various GMs and agnostic to AMs. Experimental results demonstrate the outstanding attack capability and universal applicability of our method, achieving an average attack success rate (ASR) of 97.08\% against 6 advanced AMs on DeepFakes generated by 9 GMs. Even in the presence of defensive mechanisms, our method maintains an ASR exceeding 72.39\%. Our work underscores the potential challenges posed by multiplicative attacks and highlights the need for more robust AMs.



## **13. Attack Anything: Blind DNNs via Universal Background Adversarial Attack**

cs.CV

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2409.00029v3) [paper-pdf](http://arxiv.org/pdf/2409.00029v3)

**Authors**: Jiawei Lian, Shaohui Mei, Xiaofei Wang, Yi Wang, Lefan Wang, Yingjie Lu, Mingyang Ma, Lap-Pui Chau

**Abstract**: It has been widely substantiated that deep neural networks (DNNs) are susceptible and vulnerable to adversarial perturbations. Existing studies mainly focus on performing attacks by corrupting targeted objects (physical attack) or images (digital attack), which is intuitively acceptable and understandable in terms of the attack's effectiveness. In contrast, our focus lies in conducting background adversarial attacks in both digital and physical domains, without causing any disruptions to the targeted objects themselves. Specifically, an effective background adversarial attack framework is proposed to attack anything, by which the attack efficacy generalizes well between diverse objects, models, and tasks. Technically, we approach the background adversarial attack as an iterative optimization problem, analogous to the process of DNN learning. Besides, we offer a theoretical demonstration of its convergence under a set of mild but sufficient conditions. To strengthen the attack efficacy and transferability, we propose a new ensemble strategy tailored for adversarial perturbations and introduce an improved smooth constraint for the seamless connection of integrated perturbations. We conduct comprehensive and rigorous experiments in both digital and physical domains across various objects, models, and tasks, demonstrating the effectiveness of attacking anything of the proposed method. The findings of this research substantiate the significant discrepancy between human and machine vision on the value of background variations, which play a far more critical role than previously recognized, necessitating a reevaluation of the robustness and reliability of DNNs. The code will be publicly available at https://github.com/JiaweiLian/Attack_Anything



## **14. Long-tailed Adversarial Training with Self-Distillation**

cs.CV

ICLR 2025. See OpenReview and code (in Supplementary Material) at  https://openreview.net/forum?id=vM94dZiqx4

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2503.06461v3) [paper-pdf](http://arxiv.org/pdf/2503.06461v3)

**Authors**: Seungju Cho, Hongsin Lee, Changick Kim

**Abstract**: Adversarial training significantly enhances adversarial robustness, yet superior performance is predominantly achieved on balanced datasets. Addressing adversarial robustness in the context of unbalanced or long-tailed distributions is considerably more challenging, mainly due to the scarcity of tail data instances. Previous research on adversarial robustness within long-tailed distributions has primarily focused on combining traditional long-tailed natural training with existing adversarial robustness methods. In this study, we provide an in-depth analysis for the challenge that adversarial training struggles to achieve high performance on tail classes in long-tailed distributions. Furthermore, we propose a simple yet effective solution to advance adversarial robustness on long-tailed distributions through a novel self-distillation technique. Specifically, this approach leverages a balanced self-teacher model, which is trained using a balanced dataset sampled from the original long-tailed dataset. Our extensive experiments demonstrate state-of-the-art performance in both clean and robust accuracy for long-tailed adversarial robustness, with significant improvements in tail class performance on various datasets. We improve the accuracy against PGD attacks for tail classes by 20.3, 7.1, and 3.8 percentage points on CIFAR-10, CIFAR-100, and Tiny-ImageNet, respectively, while achieving the highest robust accuracy.



## **15. CoCoTen: Detecting Adversarial Inputs to Large Language Models through Latent Space Features of Contextual Co-occurrence Tensors**

cs.CL

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2508.02997v1) [paper-pdf](http://arxiv.org/pdf/2508.02997v1)

**Authors**: Sri Durga Sai Sowmya Kadali, Evangelos E. Papalexakis

**Abstract**: The widespread use of Large Language Models (LLMs) in many applications marks a significant advance in research and practice. However, their complexity and hard-to-understand nature make them vulnerable to attacks, especially jailbreaks designed to produce harmful responses. To counter these threats, developing strong detection methods is essential for the safe and reliable use of LLMs. This paper studies this detection problem using the Contextual Co-occurrence Matrix, a structure recognized for its efficacy in data-scarce environments. We propose a novel method leveraging the latent space characteristics of Contextual Co-occurrence Matrices and Tensors for the effective identification of adversarial and jailbreak prompts. Our evaluations show that this approach achieves a notable F1 score of 0.83 using only 0.5% of labeled prompts, which is a 96.6% improvement over baselines. This result highlights the strength of our learned patterns, especially when labeled data is scarce. Our method is also significantly faster, speedup ranging from 2.3 to 128.4 times compared to the baseline models. To support future research and reproducibility, we have made our implementation publicly available.



## **16. Adversarial Attention Perturbations for Large Object Detection Transformers**

cs.CV

ICCV 2025

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2508.02987v1) [paper-pdf](http://arxiv.org/pdf/2508.02987v1)

**Authors**: Zachary Yahn, Selim Furkan Tekin, Fatih Ilhan, Sihao Hu, Tiansheng Huang, Yichang Xu, Margaret Loper, Ling Liu

**Abstract**: Adversarial perturbations are useful tools for exposing vulnerabilities in neural networks. Existing adversarial perturbation methods for object detection are either limited to attacking CNN-based detectors or weak against transformer-based detectors. This paper presents an Attention-Focused Offensive Gradient (AFOG) attack against object detection transformers. By design, AFOG is neural-architecture agnostic and effective for attacking both large transformer-based object detectors and conventional CNN-based detectors with a unified adversarial attention framework. This paper makes three original contributions. First, AFOG utilizes a learnable attention mechanism that focuses perturbations on vulnerable image regions in multi-box detection tasks, increasing performance over non-attention baselines by up to 30.6%. Second, AFOG's attack loss is formulated by integrating two types of feature loss through learnable attention updates with iterative injection of adversarial perturbations. Finally, AFOG is an efficient and stealthy adversarial perturbation method. It probes the weak spots of detection transformers by adding strategically generated and visually imperceptible perturbations which can cause well-trained object detection models to fail. Extensive experiments conducted with twelve large detection transformers on COCO demonstrate the efficacy of AFOG. Our empirical results also show that AFOG outperforms existing attacks on transformer-based and CNN-based object detectors by up to 83% with superior speed and imperceptibility. Code is available at https://github.com/zacharyyahn/AFOG.



## **17. Augmented Adversarial Trigger Learning**

cs.LG

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2503.12339v2) [paper-pdf](http://arxiv.org/pdf/2503.12339v2)

**Authors**: Zhe Wang, Yanjun Qi

**Abstract**: Gradient optimization-based adversarial attack methods automate the learning of adversarial triggers to generate jailbreak prompts or leak system prompts. In this work, we take a closer look at the optimization objective of adversarial trigger learning and propose ATLA: Adversarial Trigger Learning with Augmented objectives. ATLA improves the negative log-likelihood loss used by previous studies into a weighted loss formulation that encourages the learned adversarial triggers to optimize more towards response format tokens. This enables ATLA to learn an adversarial trigger from just one query-response pair and the learned trigger generalizes well to other similar queries. We further design a variation to augment trigger optimization with an auxiliary loss that suppresses evasive responses. We showcase how to use ATLA to learn adversarial suffixes jailbreaking LLMs and to extract hidden system prompts. Empirically we demonstrate that ATLA consistently outperforms current state-of-the-art techniques, achieving nearly 100% success in attacking while requiring 80% fewer queries. ATLA learned jailbreak suffixes demonstrate high generalization to unseen queries and transfer well to new LLMs. We released our code \href{https://github.com/QData/ALTA_Augmented_Adversarial_Trigger_Learning}{here}.



## **18. Online Robust Multi-Agent Reinforcement Learning under Model Uncertainties**

cs.LG

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2508.02948v1) [paper-pdf](http://arxiv.org/pdf/2508.02948v1)

**Authors**: Zain Ulabedeen Farhat, Debamita Ghosh, George K. Atia, Yue Wang

**Abstract**: Well-trained multi-agent systems can fail when deployed in real-world environments due to model mismatches between the training and deployment environments, caused by environment uncertainties including noise or adversarial attacks. Distributionally Robust Markov Games (DRMGs) enhance system resilience by optimizing for worst-case performance over a defined set of environmental uncertainties. However, current methods are limited by their dependence on simulators or large offline datasets, which are often unavailable. This paper pioneers the study of online learning in DRMGs, where agents learn directly from environmental interactions without prior data. We introduce the {\it Robust Optimistic Nash Value Iteration (RONAVI)} algorithm and provide the first provable guarantees for this setting. Our theoretical analysis demonstrates that the algorithm achieves low regret and efficiently finds the optimal robust policy for uncertainty sets measured by Total Variation divergence and Kullback-Leibler divergence. These results establish a new, practical path toward developing truly robust multi-agent systems.



## **19. LMDG: Advancing Lateral Movement Detection Through High-Fidelity Dataset Generation**

cs.CR

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2508.02942v1) [paper-pdf](http://arxiv.org/pdf/2508.02942v1)

**Authors**: Anas Mabrouk, Mohamed Hatem, Mohammad Mamun, Sherif Saad

**Abstract**: Lateral Movement (LM) attacks continue to pose a significant threat to enterprise security, enabling adversaries to stealthily compromise critical assets. However, the development and evaluation of LM detection systems are impeded by the absence of realistic, well-labeled datasets. To address this gap, we propose LMDG, a reproducible and extensible framework for generating high-fidelity LM datasets. LMDG automates benign activity generation, multi-stage attack execution, and comprehensive labeling of system and network logs, dramatically reducing manual effort and enabling scalable dataset creation. A central contribution of LMDG is Process Tree Labeling, a novel agent-based technique that traces all malicious activity back to its origin with high precision. Unlike prior methods such as Injection Timing or Behavioral Profiling, Process Tree Labeling enables accurate, step-wise labeling of malicious log entries, correlating each with a specific attack step and MITRE ATT\&CK TTPs. To our knowledge, this is the first approach to support fine-grained labeling of multi-step attacks, providing critical context for detection models such as attack path reconstruction. We used LMDG to generate a 25-day dataset within a 25-VM enterprise environment containing 22 user accounts. The dataset includes 944 GB of host and network logs and embeds 35 multi-stage LM attacks, with malicious events comprising less than 1% of total activity, reflecting a realistic benign-to-malicious ratio for evaluating detection systems. LMDG-generated datasets improve upon existing ones by offering diverse LM attacks, up-to-date attack patterns, longer attack timeframes, comprehensive data sources, realistic network architectures, and more accurate labeling.



## **20. GRILL: Gradient Signal Restoration in Ill-Conditioned Layers to Enhance Adversarial Attacks on Autoencoders**

cs.LG

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2505.03646v2) [paper-pdf](http://arxiv.org/pdf/2505.03646v2)

**Authors**: Chethan Krishnamurthy Ramanaik, Arjun Roy, Tobias Callies, Eirini Ntoutsi

**Abstract**: Adversarial robustness of deep autoencoders (AEs) remains relatively unexplored, even though their non-invertible nature poses distinct challenges. Existing attack algorithms during the optimization of imperceptible, norm-bounded adversarial perturbations to maximize output damage in AEs, often stop at sub-optimal attacks. We observe that the adversarial loss gradient vanishes when backpropagated through ill-conditioned layers. This issue arises from near-zero singular values in the Jacobians of these layers, which weaken the gradient signal during optimization. We introduce GRILL, a technique that locally restores gradient signals in ill-conditioned layers, enabling more effective norm-bounded attacks. Through extensive experiments on different architectures of popular AEs, under both sample-specific and universal attack setups, and across standard and adaptive attack settings, we show that our method significantly increases the effectiveness of our adversarial attacks, enabling a more rigorous evaluation of AE robustness.



## **21. Defending Against Knowledge Poisoning Attacks During Retrieval-Augmented Generation**

cs.LG

Preprint for Submission

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2508.02835v1) [paper-pdf](http://arxiv.org/pdf/2508.02835v1)

**Authors**: Kennedy Edemacu, Vinay M. Shashidhar, Micheal Tuape, Dan Abudu, Beakcheol Jang, Jong Wook Kim

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful approach to boost the capabilities of large language models (LLMs) by incorporating external, up-to-date knowledge sources. However, this introduces a potential vulnerability to knowledge poisoning attacks, where attackers can compromise the knowledge source to mislead the generation model. One such attack is the PoisonedRAG in which the injected adversarial texts steer the model to generate an attacker-chosen response to a target question. In this work, we propose novel defense methods, FilterRAG and ML-FilterRAG, to mitigate the PoisonedRAG attack. First, we propose a new property to uncover distinct properties to differentiate between adversarial and clean texts in the knowledge data source. Next, we employ this property to filter out adversarial texts from clean ones in the design of our proposed approaches. Evaluation of these methods using benchmark datasets demonstrate their effectiveness, with performances close to those of the original RAG systems.



## **22. Gandalf the Red: Adaptive Security for LLMs**

cs.LG

Niklas Pfister, V\'aclav Volhejn and Manuel Knott contributed equally

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2501.07927v3) [paper-pdf](http://arxiv.org/pdf/2501.07927v3)

**Authors**: Niklas Pfister, Václav Volhejn, Manuel Knott, Santiago Arias, Julia Bazińska, Mykhailo Bichurin, Alan Commike, Janet Darling, Peter Dienes, Matthew Fiedler, David Haber, Matthias Kraft, Marco Lancini, Max Mathys, Damián Pascual-Ortiz, Jakub Podolak, Adrià Romero-López, Kyriacos Shiarlis, Andreas Signer, Zsolt Terek, Athanasios Theocharis, Daniel Timbrell, Samuel Trautwein, Samuel Watts, Yun-Han Wu, Mateo Rojas-Carulla

**Abstract**: Current evaluations of defenses against prompt attacks in large language model (LLM) applications often overlook two critical factors: the dynamic nature of adversarial behavior and the usability penalties imposed on legitimate users by restrictive defenses. We propose D-SEC (Dynamic Security Utility Threat Model), which explicitly separates attackers from legitimate users, models multi-step interactions, and expresses the security-utility in an optimizable form. We further address the shortcomings in existing evaluations by introducing Gandalf, a crowd-sourced, gamified red-teaming platform designed to generate realistic, adaptive attack. Using Gandalf, we collect and release a dataset of 279k prompt attacks. Complemented by benign user data, our analysis reveals the interplay between security and utility, showing that defenses integrated in the LLM (e.g., system prompts) can degrade usability even without blocking requests. We demonstrate that restricted application domains, defense-in-depth, and adaptive defenses are effective strategies for building secure and useful LLM applications.



## **23. Adversarial flows: A gradient flow characterization of adversarial attacks**

cs.LG

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2406.05376v3) [paper-pdf](http://arxiv.org/pdf/2406.05376v3)

**Authors**: Lukas Weigand, Tim Roith, Martin Burger

**Abstract**: A popular method to perform adversarial attacks on neuronal networks is the so-called fast gradient sign method and its iterative variant. In this paper, we interpret this method as an explicit Euler discretization of a differential inclusion, where we also show convergence of the discretization to the associated gradient flow. To do so, we consider the concept of p-curves of maximal slope in the case $p=\infty$. We prove existence of $\infty$-curves of maximum slope and derive an alternative characterization via differential inclusions. Furthermore, we also consider Wasserstein gradient flows for potential energies, where we show that curves in the Wasserstein space can be characterized by a representing measure on the space of curves in the underlying Banach space, which fulfill the differential inclusion. The application of our theory to the finite-dimensional setting is twofold: On the one hand, we show that a whole class of normalized gradient descent methods (in particular signed gradient descent) converge, up to subsequences, to the flow, when sending the step size to zero. On the other hand, in the distributional setting, we show that the inner optimization task of adversarial training objective can be characterized via $\infty$-curves of maximum slope on an appropriate optimal transport space.



## **24. Understanding the Risks of Asphalt Art on the Reliability of Surveillance Perception Systems**

cs.CV

J. Ma and A. Enan are co-first authors; they have contributed  equally. This work has been submitted to the Transportation Research Record:  Journal of the Transportation Research Board for possible publication

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2508.02530v1) [paper-pdf](http://arxiv.org/pdf/2508.02530v1)

**Authors**: Jin Ma, Abyad Enan, Long Cheng, Mashrur Chowdhury

**Abstract**: Artistic crosswalks featuring asphalt art, introduced by different organizations in recent years, aim to enhance the visibility and safety of pedestrians. However, their visual complexity may interfere with surveillance systems that rely on vision-based object detection models. In this study, we investigate the impact of asphalt art on pedestrian detection performance of a pretrained vision-based object detection model. We construct realistic crosswalk scenarios by compositing various street art patterns into a fixed surveillance scene and evaluate the model's performance in detecting pedestrians on asphalt-arted crosswalks under both benign and adversarial conditions. A benign case refers to pedestrian crosswalks painted with existing normal asphalt art, whereas an adversarial case involves digitally crafted or altered asphalt art perpetrated by an attacker. Our results show that while simple, color-based designs have minimal effect, complex artistic patterns, particularly those with high visual salience, can significantly degrade pedestrian detection performance. Furthermore, we demonstrate that adversarially crafted asphalt art can be exploited to deliberately obscure real pedestrians or generate non-existent pedestrian detections. These findings highlight a potential vulnerability in urban vision-based pedestrian surveillance systems and underscore the importance of accounting for environmental visual variations when designing robust pedestrian perception models.



## **25. Reference-free Adversarial Sex Obfuscation in Speech**

eess.AS

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2508.02295v1) [paper-pdf](http://arxiv.org/pdf/2508.02295v1)

**Authors**: Yangyang Qu, Michele Panariello, Massimiliano Todisco, Nicholas Evans

**Abstract**: Sex conversion in speech involves privacy risks from data collection and often leaves residual sex-specific cues in outputs, even when target speaker references are unavailable. We introduce RASO for Reference-free Adversarial Sex Obfuscation. Innovations include a sex-conditional adversarial learning framework to disentangle linguistic content from sex-related acoustic markers and explicit regularisation to align fundamental frequency distributions and formant trajectories with sex-neutral characteristics learned from sex-balanced training data. RASO preserves linguistic content and, even when assessed under a semi-informed attack model, it significantly outperforms a competing approach to sex obfuscation.



## **26. Two Heads are Better than One: Robust Learning Meets Multi-branch Models**

cs.CV

10 pages, 5 Figures

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2208.08083v2) [paper-pdf](http://arxiv.org/pdf/2208.08083v2)

**Authors**: Zongyuan Zhang, Qingwen Bu, Tianyang Duan, Zheng Lin, Yuhao Qing, Zihan Fang, Heming Cui, Dong Huang

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial examples, in which DNNs are misled to false outputs due to inputs containing imperceptible perturbations. Adversarial training, a reliable and effective method of defense, may significantly reduce the vulnerability of neural networks and becomes the de facto standard for robust learning. While many recent works practice the data-centric philosophy, such as how to generate better adversarial examples or use generative models to produce additional training data, we look back to the models themselves and revisit the adversarial robustness from the perspective of deep feature distribution as an insightful complementarity. In this paper, we propose \textit{Branch Orthogonality adveRsarial Training} (BORT) to obtain state-of-the-art performance with solely the original dataset for adversarial training. To practice our design idea of integrating multiple orthogonal solution spaces, we leverage a simple and straightforward multi-branch neural network that eclipses adversarial attacks with no increase in inference time. We heuristically propose a corresponding loss function, branch-orthogonal loss, to make each solution space of the multi-branch model orthogonal. We evaluate our approach on CIFAR-10, CIFAR-100 and SVHN against $\ell_{\infty}$ norm-bounded perturbations of size $\epsilon = 8/255$, respectively. Exhaustive experiments are conducted to show that our method goes beyond all state-of-the-art methods without any tricks. Compared to all methods that do not use additional data for training, our models achieve 67.3\% and 41.5\% robust accuracy on CIFAR-10 and CIFAR-100 (improving upon the state-of-the-art by +7.23\% and +9.07\%). We also outperform methods using a training set with a far larger scale than ours.



## **27. Pigeon-SL: Robust Split Learning Framework for Edge Intelligence under Malicious Clients**

cs.LG

13 pages, 14 figures

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2508.02235v1) [paper-pdf](http://arxiv.org/pdf/2508.02235v1)

**Authors**: Sangjun Park, Tony Q. S. Quek, Hyowoon Seo

**Abstract**: Recent advances in split learning (SL) have established it as a promising framework for privacy-preserving, communication-efficient distributed learning at the network edge. However, SL's sequential update process is vulnerable to even a single malicious client, which can significantly degrade model accuracy. To address this, we introduce Pigeon-SL, a novel scheme grounded in the pigeonhole principle that guarantees at least one entirely honest cluster among M clients, even when up to N of them are adversarial. In each global round, the access point partitions the clients into N+1 clusters, trains each cluster independently via vanilla SL, and evaluates their validation losses on a shared dataset. Only the cluster with the lowest loss advances, thereby isolating and discarding malicious updates. We further enhance training and communication efficiency with Pigeon-SL+, which repeats training on the selected cluster to match the update throughput of standard SL. We validate the robustness and effectiveness of our approach under three representative attack models -- label flipping, activation and gradient manipulation -- demonstrating significant improvements in accuracy and resilience over baseline SL methods in future intelligent wireless networks.



## **28. Failure Cases Are Better Learned But Boundary Says Sorry: Facilitating Smooth Perception Change for Accuracy-Robustness Trade-Off in Adversarial Training**

cs.CV

2025 IEEE/CVF International Conference on Computer Vision (ICCV'25)

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2508.02186v1) [paper-pdf](http://arxiv.org/pdf/2508.02186v1)

**Authors**: Yanyun Wang, Li Liu

**Abstract**: Adversarial Training (AT) is one of the most effective methods to train robust Deep Neural Networks (DNNs). However, AT creates an inherent trade-off between clean accuracy and adversarial robustness, which is commonly attributed to the more complicated decision boundary caused by the insufficient learning of hard adversarial samples. In this work, we reveal a counterintuitive fact for the first time: From the perspective of perception consistency, hard adversarial samples that can still attack the robust model after AT are already learned better than those successfully defended. Thus, different from previous views, we argue that it is rather the over-sufficient learning of hard adversarial samples that degrades the decision boundary and contributes to the trade-off problem. Specifically, the excessive pursuit of perception consistency would force the model to view the perturbations as noise and ignore the information within them, which should have been utilized to induce a smoother perception transition towards the decision boundary to support its establishment to an appropriate location. In response, we define a new AT objective named Robust Perception, encouraging the model perception to change smoothly with input perturbations, based on which we propose a novel Robust Perception Adversarial Training (RPAT) method, effectively mitigating the current accuracy-robustness trade-off. Experiments on CIFAR-10, CIFAR-100, and Tiny-ImageNet with ResNet-18, PreActResNet-18, and WideResNet-34-10 demonstrate the effectiveness of our method beyond four common baselines and 12 state-of-the-art (SOTA) works. The code is available at https://github.com/FlaAI/RPAT.



## **29. Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools**

cs.AI

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2508.02110v1) [paper-pdf](http://arxiv.org/pdf/2508.02110v1)

**Authors**: Kanghua Mo, Li Hu, Yucheng Long, Zhihao Li

**Abstract**: Large language model (LLM) agents have demonstrated remarkable capabilities in complex reasoning and decision-making by leveraging external tools. However, this tool-centric paradigm introduces a previously underexplored attack surface: adversaries can manipulate tool metadata -- such as names, descriptions, and parameter schemas -- to influence agent behavior. We identify this as a new and stealthy threat surface that allows malicious tools to be preferentially selected by LLM agents, without requiring prompt injection or access to model internals. To demonstrate and exploit this vulnerability, we propose the Attractive Metadata Attack (AMA), a black-box in-context learning framework that generates highly attractive but syntactically and semantically valid tool metadata through iterative optimization. Our attack integrates seamlessly into standard tool ecosystems and requires no modification to the agent's execution framework. Extensive experiments across ten realistic, simulated tool-use scenarios and a range of popular LLM agents demonstrate consistently high attack success rates (81\%-95\%) and significant privacy leakage, with negligible impact on primary task execution. Moreover, the attack remains effective even under prompt-level defenses and structured tool-selection protocols such as the Model Context Protocol, revealing systemic vulnerabilities in current agent architectures. These findings reveal that metadata manipulation constitutes a potent and stealthy attack surface, highlighting the need for execution-level security mechanisms that go beyond prompt-level defenses.



## **30. Controllable and Stealthy Shilling Attacks via Dispersive Latent Diffusion**

cs.LG

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2508.01987v1) [paper-pdf](http://arxiv.org/pdf/2508.01987v1)

**Authors**: Shutong Qiao, Wei Yuan, Junliang Yu, Tong Chen, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Recommender systems (RSs) are now fundamental to various online platforms, but their dependence on user-contributed data leaves them vulnerable to shilling attacks that can manipulate item rankings by injecting fake users. Although widely studied, most existing attack models fail to meet two critical objectives simultaneously: achieving strong adversarial promotion of target items while maintaining realistic behavior to evade detection. As a result, the true severity of shilling threats that manage to reconcile the two objectives remains underappreciated. To expose this overlooked vulnerability, we present DLDA, a diffusion-based attack framework that can generate highly effective yet indistinguishable fake users by enabling fine-grained control over target promotion. Specifically, DLDA operates in a pre-aligned collaborative embedding space, where it employs a conditional latent diffusion process to iteratively synthesize fake user profiles with precise target item control. To evade detection, DLDA introduces a dispersive regularization mechanism that promotes variability and realism in generated behavioral patterns. Extensive experiments on three real-world datasets and five popular RS models demonstrate that, compared to prior attacks, DLDA consistently achieves stronger item promotion while remaining harder to detect. These results highlight that modern RSs are more vulnerable than previously recognized, underscoring the urgent need for more robust defenses.



## **31. Proactive Disentangled Modeling of Trigger-Object Pairings for Backdoor Defense**

cs.CV

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01932v1) [paper-pdf](http://arxiv.org/pdf/2508.01932v1)

**Authors**: Kyle Stein, Andrew A. Mahyari, Guillermo Francia III, Eman El-Sheikh

**Abstract**: Deep neural networks (DNNs) and generative AI (GenAI) are increasingly vulnerable to backdoor attacks, where adversaries embed triggers into inputs to cause models to misclassify or misinterpret target labels. Beyond traditional single-trigger scenarios, attackers may inject multiple triggers across various object classes, forming unseen backdoor-object configurations that evade standard detection pipelines. In this paper, we introduce DBOM (Disentangled Backdoor-Object Modeling), a proactive framework that leverages structured disentanglement to identify and neutralize both seen and unseen backdoor threats at the dataset level. Specifically, DBOM factorizes input image representations by modeling triggers and objects as independent primitives in the embedding space through the use of Vision-Language Models (VLMs). By leveraging the frozen, pre-trained encoders of VLMs, our approach decomposes the latent representations into distinct components through a learnable visual prompt repository and prompt prefix tuning, ensuring that the relationships between triggers and objects are explicitly captured. To separate trigger and object representations in the visual prompt repository, we introduce the trigger-object separation and diversity losses that aids in disentangling trigger and object visual features. Next, by aligning image features with feature decomposition and fusion, as well as learned contextual prompt tokens in a shared multimodal space, DBOM enables zero-shot generalization to novel trigger-object pairings that were unseen during training, thereby offering deeper insights into adversarial attack patterns. Experimental results on CIFAR-10 and GTSRB demonstrate that DBOM robustly detects poisoned images prior to downstream training, significantly enhancing the security of DNN training pipelines.



## **32. Continual Adversarial Defense**

cs.CV

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2312.09481v6) [paper-pdf](http://arxiv.org/pdf/2312.09481v6)

**Authors**: Qian Wang, Hefei Ling, Yingwei Li, Qihao Liu, Ruoxi Jia, Ning Yu

**Abstract**: In response to the rapidly evolving nature of adversarial attacks against visual classifiers, numerous defenses have been proposed to generalize against as many known attacks as possible. However, designing a defense method that generalizes to all types of attacks is unrealistic, as the environment in which the defense system operates is dynamic. Over time, new attacks inevitably emerge that exploit the vulnerabilities of existing defenses and bypass them. Therefore, we propose a continual defense strategy under a practical threat model and, for the first time, introduce the Continual Adversarial Defense (CAD) framework. CAD continuously collects adversarial data online and adapts to evolving attack sequences, while adhering to four practical principles: (1) continual adaptation to new attacks without catastrophic forgetting, (2) few-shot adaptation, (3) memory-efficient adaptation, and (4) high classification accuracy on both clean and adversarial data. We explore and integrate cutting-edge techniques from continual learning, few-shot learning, and ensemble learning to fulfill the principles. Extensive experiments validate the effectiveness of our approach against multi-stage adversarial attacks and demonstrate significant improvements over a wide range of baseline methods. We further observe that CAD's defense performance tends to saturate as the number of attacks increases, indicating its potential as a persistent defense once adapted to a sufficiently diverse set of attacks. Our research sheds light on a brand-new paradigm for continual defense adaptation against dynamic and evolving attacks.



## **33. Beyond Vulnerabilities: A Survey of Adversarial Attacks as Both Threats and Defenses in Computer Vision Systems**

cs.CV

33 pages

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01845v1) [paper-pdf](http://arxiv.org/pdf/2508.01845v1)

**Authors**: Zhongliang Guo, Yifei Qian, Yanli Li, Weiye Li, Chun Tong Lei, Shuai Zhao, Lei Fang, Ognjen Arandjelović, Chun Pong Lau

**Abstract**: Adversarial attacks against computer vision systems have emerged as a critical research area that challenges the fundamental assumptions about neural network robustness and security. This comprehensive survey examines the evolving landscape of adversarial techniques, revealing their dual nature as both sophisticated security threats and valuable defensive tools. We provide a systematic analysis of adversarial attack methodologies across three primary domains: pixel-space attacks, physically realizable attacks, and latent-space attacks. Our investigation traces the technical evolution from early gradient-based methods such as FGSM and PGD to sophisticated optimization techniques incorporating momentum, adaptive step sizes, and advanced transferability mechanisms. We examine how physically realizable attacks have successfully bridged the gap between digital vulnerabilities and real-world threats through adversarial patches, 3D textures, and dynamic optical perturbations. Additionally, we explore the emergence of latent-space attacks that leverage semantic structure in internal representations to create more transferable and meaningful adversarial examples. Beyond traditional offensive applications, we investigate the constructive use of adversarial techniques for vulnerability assessment in biometric authentication systems and protection against malicious generative models. Our analysis reveals critical research gaps, particularly in neural style transfer protection and computational efficiency requirements. This survey contributes a comprehensive taxonomy, evolution analysis, and identification of future research directions, aiming to advance understanding of adversarial vulnerabilities and inform the development of more robust and trustworthy computer vision systems.



## **34. SHIELD: Secure Hypernetworks for Incremental Expansion Learning Defense**

cs.LG

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2506.08255v2) [paper-pdf](http://arxiv.org/pdf/2506.08255v2)

**Authors**: Patryk Krukowski, Łukasz Gorczyca, Piotr Helm, Kamil Książek, Przemysław Spurek

**Abstract**: Continual learning under adversarial conditions remains an open problem, as existing methods often compromise either robustness, scalability, or both. We propose a novel framework that integrates Interval Bound Propagation (IBP) with a hypernetwork-based architecture to enable certifiably robust continual learning across sequential tasks. Our method, SHIELD, generates task-specific model parameters via a shared hypernetwork conditioned solely on compact task embeddings, eliminating the need for replay buffers or full model copies and enabling efficient over time. To further enhance robustness, we introduce Interval MixUp, a novel training strategy that blends virtual examples represented as $\ell_{\infty}$ balls centered around MixUp points. Leveraging interval arithmetic, this technique guarantees certified robustness while mitigating the wrapping effect, resulting in smoother decision boundaries. We evaluate SHIELD under strong white-box adversarial attacks, including PGD and AutoAttack, across multiple benchmarks. It consistently outperforms existing robust continual learning methods, achieving state-of-the-art average accuracy while maintaining both scalability and certification. These results represent a significant step toward practical and theoretically grounded continual learning in adversarial settings.



## **35. Enhancing Spectrogram Realism in Singing Voice Synthesis via Explicit Bandwidth Extension Prior to Vocoder**

cs.SD

7 pages, 8 figures

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01796v1) [paper-pdf](http://arxiv.org/pdf/2508.01796v1)

**Authors**: Runxuan Yang, Kai Li, Guo Chen, Xiaolin Hu

**Abstract**: This paper addresses the challenge of enhancing the realism of vocoder-generated singing voice audio by mitigating the distinguishable disparities between synthetic and real-life recordings, particularly in high-frequency spectrogram components. Our proposed approach combines two innovations: an explicit linear spectrogram estimation step using denoising diffusion process with DiT-based neural network architecture optimized for time-frequency data, and a redesigned vocoder based on Vocos specialized in handling large linear spectrograms with increased frequency bins. This integrated method can produce audio with high-fidelity spectrograms that are challenging for both human listeners and machine classifiers to differentiate from authentic recordings. Objective and subjective evaluations demonstrate that our streamlined approach maintains high audio quality while achieving this realism. This work presents a substantial advancement in overcoming the limitations of current vocoding techniques, particularly in the context of adversarial attacks on fake spectrogram detection.



## **36. "Energon": Unveiling Transformers from GPU Power and Thermal Side-Channels**

cs.CR

Accepted at IEEE/ACM International Conference on Computer-Aided  Design, 2025

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01768v1) [paper-pdf](http://arxiv.org/pdf/2508.01768v1)

**Authors**: Arunava Chaudhuri, Shubhi Shukla, Sarani Bhattacharya, Debdeep Mukhopadhyay

**Abstract**: Transformers have become the backbone of many Machine Learning (ML) applications, including language translation, summarization, and computer vision. As these models are increasingly deployed in shared Graphics Processing Unit (GPU) environments via Machine Learning as a Service (MLaaS), concerns around their security grow. In particular, the risk of side-channel attacks that reveal architectural details without physical access remains underexplored, despite the high value of the proprietary models they target. This work to the best of our knowledge is the first to investigate GPU power and thermal fluctuations as side-channels and further exploit them to extract information from pre-trained transformer models. The proposed analysis shows how these side channels can be exploited at user-privilege to reveal critical architectural details such as encoder/decoder layer and attention head for both language and vision transformers. We demonstrate the practical impact by evaluating multiple language and vision pre-trained transformers which are publicly available. Through extensive experimental evaluations, we demonstrate that the attack model achieves a high accuracy of over 89% on average for model family identification and 100% for hyperparameter classification, in both single-process as well as noisy multi-process scenarios. Moreover, by leveraging the extracted architectural information, we demonstrate highly effective black-box transfer adversarial attacks with an average success rate exceeding 93%, underscoring the security risks posed by GPU side-channel leakage in deployed transformer models.



## **37. Optimal and Feasible Contextuality-based Randomness Generation**

quant-ph

Accepted in Phys. Rev. Lett. 7+17 pages, 2+5 figures

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2412.20126v2) [paper-pdf](http://arxiv.org/pdf/2412.20126v2)

**Authors**: Yuan Liu, Ravishankar Ramanathan

**Abstract**: Semi-device-independent (SDI) randomness generation protocols based on Kochen-Specker contextuality offer the attractive features of compact devices, high rates, and ease of experimental implementation over fully device-independent (DI) protocols. Here, we investigate this paradigm and derive four results to improve the state-of-art. Firstly, we introduce a family of simple, experimentally feasible orthogonality graphs (measurement compatibility structures) for which the maximum violation of the corresponding non-contextuality inequalities allows to certify the maximum amount of $\log_2 d$ bits of randomness from a qu$d$it system with projective measurements for $d \geq 3$. We analytically derive the Lov\'asz theta and fractional packing number for this graph family, and thereby prove their utility for optimal randomness generation in both randomness expansion and amplification tasks. Secondly, a central additional assumption in contextuality-based protocols over fully DI ones, is that the measurements are repeatable and satisfy an intended compatibility structure. We frame a relaxation of this condition in terms of $\epsilon$-orthogonality graphs for a parameter $\epsilon > 0$, and derive quantum correlations that allow to certify randomness for arbitrary relaxation $\epsilon \in [0,1)$. Thirdly, it is well known that a single qubit is non-contextual, i.e., the qubit correlations can be explained by a non-contextual hidden variable (NCHV) model. We show however that a single qubit is \textit{almost} contextual, in that there exist qubit correlations that cannot be explained by $\epsilon$-faithful NCHV models for small $\epsilon > 0$. Finally, we point out possible attacks by quantum and general consistent (non-signalling) adversaries for certain classes of contextuality tests over and above those considered in DI scenarios.



## **38. AI-Generated Text is Non-Stationary: Detection via Temporal Tomography**

cs.CL

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01754v1) [paper-pdf](http://arxiv.org/pdf/2508.01754v1)

**Authors**: Alva West, Yixuan Weng, Minjun Zhu, Luodan Zhang, Zhen Lin, Guangsheng Bao, Yue Zhang

**Abstract**: The field of AI-generated text detection has evolved from supervised classification to zero-shot statistical analysis. However, current approaches share a fundamental limitation: they aggregate token-level measurements into scalar scores, discarding positional information about where anomalies occur. Our empirical analysis reveals that AI-generated text exhibits significant non-stationarity, statistical properties vary by 73.8\% more between text segments compared to human writing. This discovery explains why existing detectors fail against localized adversarial perturbations that exploit this overlooked characteristic. We introduce Temporal Discrepancy Tomography (TDT), a novel detection paradigm that preserves positional information by reformulating detection as a signal processing task. TDT treats token-level discrepancies as a time-series signal and applies Continuous Wavelet Transform to generate a two-dimensional time-scale representation, capturing both the location and linguistic scale of statistical anomalies. On the RAID benchmark, TDT achieves 0.855 AUROC (7.1\% improvement over the best baseline). More importantly, TDT demonstrates robust performance on adversarial tasks, with 14.1\% AUROC improvement on HART Level 2 paraphrasing attacks. Despite its sophisticated analysis, TDT maintains practical efficiency with only 13\% computational overhead. Our work establishes non-stationarity as a fundamental characteristic of AI-generated text and demonstrates that preserving temporal dynamics is essential for robust detection.



## **39. Simulated Ensemble Attack: Transferring Jailbreaks Across Fine-tuned Vision-Language Models**

cs.CV

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01741v1) [paper-pdf](http://arxiv.org/pdf/2508.01741v1)

**Authors**: Ruofan Wang, Xin Wang, Yang Yao, Xuan Tong, Xingjun Ma

**Abstract**: Fine-tuning open-source Vision-Language Models (VLMs) creates a critical yet underexplored attack surface: vulnerabilities in the base VLM could be retained in fine-tuned variants, rendering them susceptible to transferable jailbreak attacks. To demonstrate this risk, we introduce the Simulated Ensemble Attack (SEA), a novel grey-box jailbreak method in which the adversary has full access to the base VLM but no knowledge of the fine-tuned target's weights or training configuration. To improve jailbreak transferability across fine-tuned VLMs, SEA combines two key techniques: Fine-tuning Trajectory Simulation (FTS) and Targeted Prompt Guidance (TPG). FTS generates transferable adversarial images by simulating the vision encoder's parameter shifts, while TPG is a textual strategy that steers the language decoder toward adversarially optimized outputs. Experiments on the Qwen2-VL family (2B and 7B) demonstrate that SEA achieves high transfer attack success rates exceeding 86.5% and toxicity rates near 49.5% across diverse fine-tuned variants, even those specifically fine-tuned to improve safety behaviors. Notably, while direct PGD-based image jailbreaks rarely transfer across fine-tuned VLMs, SEA reliably exploits inherited vulnerabilities from the base model, significantly enhancing transferability. These findings highlight an urgent need to safeguard fine-tuned proprietary VLMs against transferable vulnerabilities inherited from open-source foundations, motivating the development of holistic defenses across the entire model lifecycle.



## **40. RedDiffuser: Red Teaming Vision-Language Models for Toxic Continuation via Reinforced Stable Diffusion**

cs.CV

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2503.06223v3) [paper-pdf](http://arxiv.org/pdf/2503.06223v3)

**Authors**: Ruofan Wang, Xiang Zheng, Xiaosen Wang, Cong Wang, Xingjun Ma

**Abstract**: Vision-Language Models (VLMs) are vulnerable to jailbreak attacks, where adversaries bypass safety mechanisms to elicit harmful outputs. In this work, we examine an insidious variant of this threat: toxic continuation. Unlike standard jailbreaks that rely solely on malicious instructions, toxic continuation arises when the model is given a malicious input alongside a partial toxic output, resulting in harmful completions. This vulnerability poses a unique challenge in multimodal settings, where even subtle image variations can disproportionately affect the model's response. To this end, we propose RedDiffuser (RedDiff), the first red teaming framework that uses reinforcement learning to fine-tune diffusion models into generating natural-looking adversarial images that induce toxic continuations. RedDiffuser integrates a greedy search procedure for selecting candidate image prompts with reinforcement fine-tuning that jointly promotes toxic output and semantic coherence. Experiments demonstrate that RedDiffuser significantly increases the toxicity rate in LLaVA outputs by 10.69% and 8.91% on the original and hold-out sets, respectively. It also exhibits strong transferability, increasing toxicity rates on Gemini by 5.1% and on LLaMA-Vision by 26.83%. These findings uncover a cross-modal toxicity amplification vulnerability in current VLM alignment, highlighting the need for robust multimodal red teaming. We will release the RedDiffuser codebase to support future research.



## **41. Impartial Games: A Challenge for Reinforcement Learning**

cs.LG

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2205.12787v5) [paper-pdf](http://arxiv.org/pdf/2205.12787v5)

**Authors**: Bei Zhou, Søren Riis

**Abstract**: AlphaZero-style reinforcement learning (RL) algorithms have achieved superhuman performance in many complex board games such as Chess, Shogi, and Go. However, we showcase that these algorithms encounter significant and fundamental challenges when applied to impartial games, a class where players share game pieces and optimal strategy often relies on abstract mathematical principles. Specifically, we utilize the game of Nim as a concrete and illustrative case study to reveal critical limitations of AlphaZero-style and similar self-play RL algorithms. We introduce a novel conceptual framework distinguishing between champion and expert mastery to evaluate RL agent performance. Our findings reveal that while AlphaZero-style agents can achieve champion-level play on very small Nim boards, their learning progression severely degrades as the board size increases. This difficulty stems not merely from complex data distributions or noisy labels, but from a deeper representational bottleneck: the inherent struggle of generic neural networks to implicitly learn abstract, non-associative functions like parity, which are crucial for optimal play in impartial games. This limitation causes a critical breakdown in the positive feedback loop essential for self-play RL, preventing effective learning beyond rote memorization of frequently observed states. These results align with broader concerns regarding AlphaZero-style algorithms' vulnerability to adversarial attacks, highlighting their inability to truly master all legal game states. Our work underscores that simple hyperparameter adjustments are insufficient to overcome these challenges, establishing a crucial foundation for the development of fundamentally novel algorithmic approaches, potentially involving neuro-symbolic or meta-learning paradigms, to bridge the gap towards true expert-level AI in combinatorial games.



## **42. Benchmarking Adversarial Patch Selection and Location**

cs.CV

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01676v1) [paper-pdf](http://arxiv.org/pdf/2508.01676v1)

**Authors**: Shai Kimhi, Avi Mendlson, Moshe Kimhi

**Abstract**: Adversarial patch attacks threaten the reliability of modern vision models. We present PatchMap, the first spatially exhaustive benchmark of patch placement, built by evaluating over 1.5e8 forward passes on ImageNet validation images. PatchMap reveals systematic hot-spots where small patches (as little as 2% of the image) induce confident misclassifications and large drops in model confidence. To demonstrate its utility, we propose a simple segmentation guided placement heuristic that leverages off the shelf masks to identify vulnerable regions without any gradient queries. Across five architectures-including adversarially trained ResNet50, our method boosts attack success rates by 8 to 13 percentage points compared to random or fixed placements. We publicly release PatchMap and the code implementation. The full PatchMap bench (6.5B predictions, multiple backbones) will be released soon to further accelerate research on location-aware defenses and adaptive attacks.



## **43. Practical, Generalizable and Robust Backdoor Attacks on Text-to-Image Diffusion Models**

cs.CR

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01605v1) [paper-pdf](http://arxiv.org/pdf/2508.01605v1)

**Authors**: Haoran Dai, Jiawen Wang, Ruo Yang, Manali Sharma, Zhonghao Liao, Yuan Hong, Binghui Wang

**Abstract**: Text-to-image diffusion models (T2I DMs) have achieved remarkable success in generating high-quality and diverse images from text prompts, yet recent studies have revealed their vulnerability to backdoor attacks. Existing attack methods suffer from critical limitations: 1) they rely on unnatural adversarial prompts that lack human readability and require massive poisoned data; 2) their effectiveness is typically restricted to specific models, lacking generalizability; and 3) they can be mitigated by recent backdoor defenses.   To overcome these challenges, we propose a novel backdoor attack framework that achieves three key properties: 1) \emph{Practicality}: Our attack requires only a few stealthy backdoor samples to generate arbitrary attacker-chosen target images, as well as ensuring high-quality image generation in benign scenarios. 2) \emph{Generalizability:} The attack is applicable across multiple T2I DMs without requiring model-specific redesign. 3) \emph{Robustness:} The attack remains effective against existing backdoor defenses and adaptive defenses. Our extensive experimental results on multiple T2I DMs demonstrate that with only 10 carefully crafted backdoored samples, our attack method achieves $>$90\% attack success rate with negligible degradation in benign image generation quality. We also conduct human evaluation to validate our attack effectiveness. Furthermore, recent backdoor detection and mitigation methods, as well as adaptive defense tailored to our attack are not sufficiently effective, highlighting the pressing need for more robust defense mechanisms against the proposed attack.



## **44. BeDKD: Backdoor Defense based on Dynamic Knowledge Distillation and Directional Mapping Modulator**

cs.CR

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01595v1) [paper-pdf](http://arxiv.org/pdf/2508.01595v1)

**Authors**: Zhengxian Wu, Juan Wen, Wanli Peng, Yinghan Zhou, Changtong dou, Yiming Xue

**Abstract**: Although existing backdoor defenses have gained success in mitigating backdoor attacks, they still face substantial challenges. In particular, most of them rely on large amounts of clean data to weaken the backdoor mapping but generally struggle with residual trigger effects, resulting in persistently high attack success rates (ASR). Therefore, in this paper, we propose a novel Backdoor defense method based on Directional mapping module and adversarial Knowledge Distillation (BeDKD), which balances the trade-off between defense effectiveness and model performance using a small amount of clean and poisoned data. We first introduce a directional mapping module to identify poisoned data, which destroys clean mapping while keeping backdoor mapping on a small set of flipped clean data. Then, the adversarial knowledge distillation is designed to reinforce clean mapping and suppress backdoor mapping through a cycle iteration mechanism between trust and punish distillations using clean and identified poisoned data. We conduct experiments to mitigate mainstream attacks on three datasets, and experimental results demonstrate that BeDKD surpasses the state-of-the-art defenses and reduces the ASR by 98% without significantly reducing the CACC. Our code are available in https://github.com/CAU-ISS-Lab/Backdoor-Attack-Defense-LLMs/tree/main/BeDKD.



## **45. Are All Prompt Components Value-Neutral? Understanding the Heterogeneous Adversarial Robustness of Dissected Prompt in Large Language Models**

cs.CL

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01554v1) [paper-pdf](http://arxiv.org/pdf/2508.01554v1)

**Authors**: Yujia Zheng, Tianhao Li, Haotian Huang, Tianyu Zeng, Jingyu Lu, Chuangxin Chu, Yuekai Huang, Ziyou Jiang, Qian Xiong, Yuyao Ge, Mingyang Li

**Abstract**: Prompt-based adversarial attacks have become an effective means to assess the robustness of large language models (LLMs). However, existing approaches often treat prompts as monolithic text, overlooking their structural heterogeneity-different prompt components contribute unequally to adversarial robustness. Prior works like PromptRobust assume prompts are value-neutral, but our analysis reveals that complex, domain-specific prompts with rich structures have components with differing vulnerabilities. To address this gap, we introduce PromptAnatomy, an automated framework that dissects prompts into functional components and generates diverse, interpretable adversarial examples by selectively perturbing each component using our proposed method, ComPerturb. To ensure linguistic plausibility and mitigate distribution shifts, we further incorporate a perplexity (PPL)-based filtering mechanism. As a complementary resource, we annotate four public instruction-tuning datasets using the PromptAnatomy framework, verified through human review. Extensive experiments across these datasets and five advanced LLMs demonstrate that ComPerturb achieves state-of-the-art attack success rates. Ablation studies validate the complementary benefits of prompt dissection and PPL filtering. Our results underscore the importance of prompt structure awareness and controlled perturbation for reliable adversarial robustness evaluation in LLMs. Code and data are available at https://github.com/Yujiaaaaa/PACP.



## **46. VWAttacker: A Systematic Security Testing Framework for Voice over WiFi User Equipments**

cs.CR

**SubmitDate**: 2025-08-02    [abs](http://arxiv.org/abs/2508.01469v1) [paper-pdf](http://arxiv.org/pdf/2508.01469v1)

**Authors**: Imtiaz Karim, Hyunwoo Lee, Hassan Asghar, Kazi Samin Mubasshir, Seulgi Han, Mashroor Hasan Bhuiyan, Elisa Bertino

**Abstract**: We present VWAttacker, the first systematic testing framework for analyzing the security of Voice over WiFi (VoWiFi) User Equipment (UE) implementations. VWAttacker includes a complete VoWiFi network testbed that communicates with Commercial-Off-The-Shelf (COTS) UEs based on a simple interface to test the behavior of diverse VoWiFi UE implementations; uses property-guided adversarial testing to uncover security issues in different UEs systematically. To reduce manual effort in extracting and testing properties, we introduce an LLM-based, semi-automatic, and scalable approach for property extraction and testcase (TC) generation. These TCs are systematically mutated by two domain-specific transformations. Furthermore, we introduce two deterministic oracles to detect property violations automatically. Coupled with these techniques, VWAttacker extracts 63 properties from 11 specifications, evaluates 1,116 testcases, and detects 13 issues in 21 UEs. The issues range from enforcing a DH shared secret to 0 to supporting weak algorithms. These issues result in attacks that expose the victim UE's identity or establish weak channels, thus severely hampering the security of cellular networks. We responsibly disclose the findings to all the related vendors. At the time of writing, one of the vulnerabilities has been acknowledged by MediaTek with high severity.



## **47. Nakamoto Consensus from Multiple Resources**

cs.CR

Full version of the paper published at AFT25

**SubmitDate**: 2025-08-02    [abs](http://arxiv.org/abs/2508.01448v1) [paper-pdf](http://arxiv.org/pdf/2508.01448v1)

**Authors**: Mirza Ahad Baig, Christoph U. Günther, Krzysztof Pietrzak

**Abstract**: The blocks in the Bitcoin blockchain record the amount of work W that went into creating them through proofs of work. When honest parties control a majority of the work, consensus is achieved by picking the chain with the highest recorded weight. Resources other than work have been considered to secure such longest-chain blockchains. In Chia, blocks record the amount of space S (via a proof of space) and sequential computational steps V (via a VDF).   In this paper, we ask what weight functions {\Gamma}(S,V,W) (that assign a weight to a block as a function of the recorded space, speed, and work) are secure in the sense that whenever the weight of the resources controlled by honest parties is larger than the weight of adversarial parties, the blockchain is secure against private double-spending attacks.   We completely classify such functions in an idealized "continuous" model: {\Gamma}(S,V,W) is secure against private double-spending attacks if and only if it is homogeneous of degree one in the timed resources V and W, i.e., {\alpha}{\Gamma}(S,V,W)={\Gamma}(S,{\alpha}V, {\alpha}W). This includes Bitcoin rule {\Gamma}(S,V,W)=W and Chia rule {\Gamma}(S,V,W) = SV. In a more realistic model where blocks are created at discrete time-points, one additionally needs some mild assumptions on the dependency on S (basically, the weight should not grow too much if S is slightly increased, say linear as in Chia).   Our classification is more general and allows various instantiations of the same resource. It provides a powerful tool for designing new longest-chain blockchains. E.g., consider combining different PoWs to counter centralization, say the Bitcoin PoW W_1 and a memory-hard PoW W_2. Previous work suggested to use W_1+W_2 as weight. Our results show that using {\sqrt}(W_1){\cdot}{\sqrt}(W_2), {\min}{W_1,W_2} are also secure, and we argue that in practice these are much better choices.



## **48. Safety at Scale: A Comprehensive Survey of Large Model and Agent Safety**

cs.CR

706 papers, 60 pages, 3 figures, 14 tables; GitHub:  https://github.com/xingjunm/Awesome-Large-Model-Safety

**SubmitDate**: 2025-08-02    [abs](http://arxiv.org/abs/2502.05206v5) [paper-pdf](http://arxiv.org/pdf/2502.05206v5)

**Authors**: Xingjun Ma, Yifeng Gao, Yixu Wang, Ruofan Wang, Xin Wang, Ye Sun, Yifan Ding, Hengyuan Xu, Yunhao Chen, Yunhan Zhao, Hanxun Huang, Yige Li, Yutao Wu, Jiaming Zhang, Xiang Zheng, Yang Bai, Zuxuan Wu, Xipeng Qiu, Jingfeng Zhang, Yiming Li, Xudong Han, Haonan Li, Jun Sun, Cong Wang, Jindong Gu, Baoyuan Wu, Siheng Chen, Tianwei Zhang, Yang Liu, Mingming Gong, Tongliang Liu, Shirui Pan, Cihang Xie, Tianyu Pang, Yinpeng Dong, Ruoxi Jia, Yang Zhang, Shiqing Ma, Xiangyu Zhang, Neil Gong, Chaowei Xiao, Sarah Erfani, Tim Baldwin, Bo Li, Masashi Sugiyama, Dacheng Tao, James Bailey, Yu-Gang Jiang

**Abstract**: The rapid advancement of large models, driven by their exceptional abilities in learning and generalization through large-scale pre-training, has reshaped the landscape of Artificial Intelligence (AI). These models are now foundational to a wide range of applications, including conversational AI, recommendation systems, autonomous driving, content generation, medical diagnostics, and scientific discovery. However, their widespread deployment also exposes them to significant safety risks, raising concerns about robustness, reliability, and ethical implications. This survey provides a systematic review of current safety research on large models, covering Vision Foundation Models (VFMs), Large Language Models (LLMs), Vision-Language Pre-training (VLP) models, Vision-Language Models (VLMs), Diffusion Models (DMs), and large-model-powered Agents. Our contributions are summarized as follows: (1) We present a comprehensive taxonomy of safety threats to these models, including adversarial attacks, data poisoning, backdoor attacks, jailbreak and prompt injection attacks, energy-latency attacks, data and model extraction attacks, and emerging agent-specific threats. (2) We review defense strategies proposed for each type of attacks if available and summarize the commonly used datasets and benchmarks for safety research. (3) Building on this, we identify and discuss the open challenges in large model safety, emphasizing the need for comprehensive safety evaluations, scalable and effective defense mechanisms, and sustainable data practices. More importantly, we highlight the necessity of collective efforts from the research community and international collaboration. Our work can serve as a useful reference for researchers and practitioners, fostering the ongoing development of comprehensive defense systems and platforms to safeguard AI models.



## **49. Mitigating Watermark Forgery in Generative Models via Multi-Key Watermarking**

cs.CR

**SubmitDate**: 2025-08-02    [abs](http://arxiv.org/abs/2507.07871v2) [paper-pdf](http://arxiv.org/pdf/2507.07871v2)

**Authors**: Toluwani Aremu, Noor Hussein, Munachiso Nwadike, Samuele Poppi, Jie Zhang, Karthik Nandakumar, Neil Gong, Nils Lukas

**Abstract**: Watermarking offers a promising solution for GenAI providers to establish the provenance of their generated content. A watermark is a hidden signal embedded in the generated content, whose presence can later be verified using a secret watermarking key. A security threat to GenAI providers are \emph{forgery attacks}, where malicious users insert the provider's watermark into generated content that was \emph{not} produced by the provider's models, potentially damaging their reputation and undermining trust. One potential defense to resist forgery is using multiple keys to watermark generated content. However, it has been shown that forgery attacks remain successful when adversaries can collect sufficiently many watermarked samples. We propose an improved multi-key watermarking method that resists all surveyed forgery attacks and scales independently of the number of watermarked samples collected by the adversary. Our method accepts content as genuinely watermarked only if \emph{exactly} one watermark is detected. We focus on the image and text modalities, but our detection method is modality-agnostic, since it treats the underlying watermarking method as a black-box. We derive theoretical bounds on forgery-resistance and empirically validate them using Mistral-7B. Our results show a decrease in forgery success from up to $100\%$ using single-key baselines to only $2\%$. While our method resists all surveyed attacks, we find that highly capable, adaptive attackers can still achieve success rates of up to $65\%$ if watermarked content generated using different keys is easily separable.



## **50. Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks**

cs.CV

ICCV 2025

**SubmitDate**: 2025-08-02    [abs](http://arxiv.org/abs/2504.01308v3) [paper-pdf](http://arxiv.org/pdf/2504.01308v3)

**Authors**: Jiawei Wang, Yushen Zuo, Yuanjun Chai, Zhendong Liu, Yicheng Fu, Yichun Feng, Kin-Man Lam

**Abstract**: Vision-Language Models (VLMs) extend the capabilities of Large Language Models (LLMs) by incorporating visual information, yet they remain vulnerable to jailbreak attacks, especially when processing noisy or corrupted images. Although existing VLMs adopt security measures during training to mitigate such attacks, vulnerabilities associated with noise-augmented visual inputs are overlooked. In this work, we identify that missing noise-augmented training causes critical security gaps: many VLMs are susceptible to even simple perturbations such as Gaussian noise. To address this challenge, we propose Robust-VLGuard, a multimodal safety dataset with aligned / misaligned image-text pairs, combined with noise-augmented fine-tuning that reduces attack success rates while preserving functionality of VLM. For stronger optimization-based visual perturbation attacks, we propose DiffPure-VLM, leveraging diffusion models to convert adversarial perturbations into Gaussian-like noise, which can be defended by VLMs with noise-augmented safety fine-tuning. Experimental results demonstrate that the distribution-shifting property of diffusion model aligns well with our fine-tuned VLMs, significantly mitigating adversarial perturbations across varying intensities. The dataset and code are available at https://github.com/JarvisUSTC/DiffPure-RobustVLM.



