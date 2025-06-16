# Latest Adversarial Attack Papers
**update at 2025-06-16 10:31:09**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Self-interpreting Adversarial Images**

cs.CR

in USENIX Security 2025

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2407.08970v4) [paper-pdf](http://arxiv.org/pdf/2407.08970v4)

**Authors**: Tingwei Zhang, Collin Zhang, John X. Morris, Eugene Bagdasarian, Vitaly Shmatikov

**Abstract**: We introduce a new type of indirect, cross-modal injection attacks against visual language models that enable creation of self-interpreting images. These images contain hidden "meta-instructions" that control how models answer users' questions about the image and steer models' outputs to express an adversary-chosen style, sentiment, or point of view.   Self-interpreting images act as soft prompts, conditioning the model to satisfy the adversary's (meta-)objective while still producing answers based on the image's visual content. Meta-instructions are thus a stronger form of prompt injection. Adversarial images look natural and the model's answers are coherent and plausible, yet they also follow the adversary-chosen interpretation, e.g., political spin, or even objectives that are not achievable with explicit text instructions.   We evaluate the efficacy of self-interpreting images for a variety of models, interpretations, and user prompts. We describe how these attacks could cause harm by enabling creation of self-interpreting content that carries spam, misinformation, or spin. Finally, we discuss defenses.



## **2. Improving Large Language Model Safety with Contrastive Representation Learning**

cs.CL

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11938v1) [paper-pdf](http://arxiv.org/pdf/2506.11938v1)

**Authors**: Samuel Simko, Mrinmaya Sachan, Bernhard Schölkopf, Zhijing Jin

**Abstract**: Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at https://github.com/samuelsimko/crl-llm-defense



## **3. Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation**

cs.CR

14 pages, 5 figures

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2501.18638v2) [paper-pdf](http://arxiv.org/pdf/2501.18638v2)

**Authors**: Daniel Schwartz, Dmitriy Bespalov, Zhe Wang, Ninad Kulkarni, Yanjun Qi

**Abstract**: As large language models (LLMs) become increasingly prevalent, ensuring their robustness against adversarial misuse is crucial. This paper introduces the GAP (Graph of Attacks with Pruning) framework, an advanced approach for generating stealthy jailbreak prompts to evaluate and enhance LLM safeguards. GAP addresses limitations in existing tree-based LLM jailbreak methods by implementing an interconnected graph structure that enables knowledge sharing across attack paths. Our experimental evaluation demonstrates GAP's superiority over existing techniques, achieving a 20.8% increase in attack success rates while reducing query costs by 62.7%. GAP consistently outperforms state-of-the-art methods for attacking both open and closed LLMs, with attack success rates of >96%. Additionally, we present specialized variants like GAP-Auto for automated seed generation and GAP-VLM for multimodal attacks. GAP-generated prompts prove highly effective in improving content moderation systems, increasing true positive detection rates by 108.5% and accuracy by 183.6% when used for fine-tuning. Our implementation is available at https://github.com/dsbuddy/GAP-LLM-Safety.



## **4. Attention-based Adversarial Robust Distillation in Radio Signal Classifications for Low-Power IoT Devices**

cs.LG

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11892v1) [paper-pdf](http://arxiv.org/pdf/2506.11892v1)

**Authors**: Lu Zhang, Sangarapillai Lambotharan, Gan Zheng, Guisheng Liao, Basil AsSadhan, Fabio Roli

**Abstract**: Due to great success of transformers in many applications such as natural language processing and computer vision, transformers have been successfully applied in automatic modulation classification. We have shown that transformer-based radio signal classification is vulnerable to imperceptible and carefully crafted attacks called adversarial examples. Therefore, we propose a defense system against adversarial examples in transformer-based modulation classifications. Considering the need for computationally efficient architecture particularly for Internet of Things (IoT)-based applications or operation of devices in environment where power supply is limited, we propose a compact transformer for modulation classification. The advantages of robust training such as adversarial training in transformers may not be attainable in compact transformers. By demonstrating this, we propose a novel compact transformer that can enhance robustness in the presence of adversarial attacks. The new method is aimed at transferring the adversarial attention map from the robustly trained large transformer to a compact transformer. The proposed method outperforms the state-of-the-art techniques for the considered white-box scenarios including fast gradient method and projected gradient descent attacks. We have provided reasoning of the underlying working mechanisms and investigated the transferability of the adversarial examples between different architectures. The proposed method has the potential to protect the transformer from the transferability of adversarial examples.



## **5. Black-Box Adversarial Attacks on LLM-Based Code Completion**

cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2408.02509v2) [paper-pdf](http://arxiv.org/pdf/2408.02509v2)

**Authors**: Slobodan Jenko, Niels Mündler, Jingxuan He, Mark Vero, Martin Vechev

**Abstract**: Modern code completion engines, powered by large language models (LLMs), assist millions of developers with their strong capabilities to generate functionally correct code. Due to this popularity, it is crucial to investigate the security implications of relying on LLM-based code completion. In this work, we demonstrate that state-of-the-art black-box LLM-based code completion engines can be stealthily biased by adversaries to significantly increase their rate of insecure code generation. We present the first attack, named INSEC, that achieves this goal. INSEC works by injecting an attack string as a short comment in the completion input. The attack string is crafted through a query-based optimization procedure starting from a set of carefully designed initialization schemes. We demonstrate INSEC's broad applicability and effectiveness by evaluating it on various state-of-the-art open-source models and black-box commercial services (e.g., OpenAI API and GitHub Copilot). On a diverse set of security-critical test cases, covering 16 CWEs across 5 programming languages, INSEC increases the rate of generated insecure code by more than 50%, while maintaining the functional correctness of generated code. We consider INSEC practical -- it requires low resources and costs less than 10 US dollars to develop on commodity hardware. Moreover, we showcase the attack's real-world deployability, by developing an IDE plug-in that stealthily injects INSEC into the GitHub Copilot extension.



## **6. TrustGLM: Evaluating the Robustness of GraphLLMs Against Prompt, Text, and Structure Attacks**

cs.LG

12 pages, 5 figures, in KDD 2025

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11844v1) [paper-pdf](http://arxiv.org/pdf/2506.11844v1)

**Authors**: Qihai Zhang, Xinyue Sheng, Yuanfu Sun, Qiaoyu Tan

**Abstract**: Inspired by the success of large language models (LLMs), there is a significant research shift from traditional graph learning methods to LLM-based graph frameworks, formally known as GraphLLMs. GraphLLMs leverage the reasoning power of LLMs by integrating three key components: the textual attributes of input nodes, the structural information of node neighborhoods, and task-specific prompts that guide decision-making. Despite their promise, the robustness of GraphLLMs against adversarial perturbations remains largely unexplored-a critical concern for deploying these models in high-stakes scenarios. To bridge the gap, we introduce TrustGLM, a comprehensive study evaluating the vulnerability of GraphLLMs to adversarial attacks across three dimensions: text, graph structure, and prompt manipulations. We implement state-of-the-art attack algorithms from each perspective to rigorously assess model resilience. Through extensive experiments on six benchmark datasets from diverse domains, our findings reveal that GraphLLMs are highly susceptible to text attacks that merely replace a few semantically similar words in a node's textual attribute. We also find that standard graph structure attack methods can significantly degrade model performance, while random shuffling of the candidate label set in prompt templates leads to substantial performance drops. Beyond characterizing these vulnerabilities, we investigate defense techniques tailored to each attack vector through data-augmented training and adversarial training, which show promising potential to enhance the robustness of GraphLLMs. We hope that our open-sourced library will facilitate rapid, equitable evaluation and inspire further innovative research in this field.



## **7. Differential Privacy in Machine Learning: From Symbolic AI to LLMs**

cs.CR

arXiv admin note: text overlap with arXiv:2303.00654 by other authors

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11687v1) [paper-pdf](http://arxiv.org/pdf/2506.11687v1)

**Authors**: Francisco Aguilera-Martínez, Fernando Berzal

**Abstract**: Machine learning models should not reveal particular information that is not otherwise accessible. Differential privacy provides a formal framework to mitigate privacy risks by ensuring that the inclusion or exclusion of any single data point does not significantly alter the output of an algorithm, thus limiting the exposure of private information. This survey paper explores the foundational definitions of differential privacy, reviews its original formulations and tracing its evolution through key research contributions. It then provides an in-depth examination of how DP has been integrated into machine learning models, analyzing existing proposals and methods to preserve privacy when training ML models. Finally, it describes how DP-based ML techniques can be evaluated in practice. %Finally, it discusses the broader implications of DP, highlighting its potential for public benefit, its real-world applications, and the challenges it faces, including vulnerabilities to adversarial attacks. By offering a comprehensive overview of differential privacy in machine learning, this work aims to contribute to the ongoing development of secure and responsible AI systems.



## **8. KCES: Training-Free Defense for Robust Graph Neural Networks via Kernel Complexity**

cs.LG

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11611v1) [paper-pdf](http://arxiv.org/pdf/2506.11611v1)

**Authors**: Yaning Jia, Shenyang Deng, Chiyu Ma, Yaoqing Yang, Soroush Vosoughi

**Abstract**: Graph Neural Networks (GNNs) have achieved impressive success across a wide range of graph-based tasks, yet they remain highly vulnerable to small, imperceptible perturbations and adversarial attacks. Although numerous defense methods have been proposed to address these vulnerabilities, many rely on heuristic metrics, overfit to specific attack patterns, and suffer from high computational complexity. In this paper, we propose Kernel Complexity-Based Edge Sanitization (KCES), a training-free, model-agnostic defense framework. KCES leverages Graph Kernel Complexity (GKC), a novel metric derived from the graph's Gram matrix that characterizes GNN generalization via its test error bound. Building on GKC, we define a KC score for each edge, measuring the change in GKC when the edge is removed. Edges with high KC scores, typically introduced by adversarial perturbations, are pruned to mitigate their harmful effects, thereby enhancing GNNs' robustness. KCES can also be seamlessly integrated with existing defense strategies as a plug-and-play module without requiring training. Theoretical analysis and extensive experiments demonstrate that KCES consistently enhances GNN robustness, outperforms state-of-the-art baselines, and amplifies the effectiveness of existing defenses, offering a principled and efficient solution for securing GNNs.



## **9. Investigating Vulnerabilities and Defenses Against Audio-Visual Attacks: A Comprehensive Survey Emphasizing Multimodal Models**

cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11521v1) [paper-pdf](http://arxiv.org/pdf/2506.11521v1)

**Authors**: Jinming Wen, Xinyi Wu, Shuai Zhao, Yanhao Jia, Yuwen Li

**Abstract**: Multimodal large language models (MLLMs), which bridge the gap between audio-visual and natural language processing, achieve state-of-the-art performance on several audio-visual tasks. Despite the superior performance of MLLMs, the scarcity of high-quality audio-visual training data and computational resources necessitates the utilization of third-party data and open-source MLLMs, a trend that is increasingly observed in contemporary research. This prosperity masks significant security risks. Empirical studies demonstrate that the latest MLLMs can be manipulated to produce malicious or harmful content. This manipulation is facilitated exclusively through instructions or inputs, including adversarial perturbations and malevolent queries, effectively bypassing the internal security mechanisms embedded within the models. To gain a deeper comprehension of the inherent security vulnerabilities associated with audio-visual-based multimodal models, a series of surveys investigates various types of attacks, including adversarial and backdoor attacks. While existing surveys on audio-visual attacks provide a comprehensive overview, they are limited to specific types of attacks, which lack a unified review of various types of attacks. To address this issue and gain insights into the latest trends in the field, this paper presents a comprehensive and systematic review of audio-visual attacks, which include adversarial attacks, backdoor attacks, and jailbreak attacks. Furthermore, this paper also reviews various types of attacks in the latest audio-visual-based MLLMs, a dimension notably absent in existing surveys. Drawing upon comprehensive insights from a substantial review, this paper delineates both challenges and emergent trends for future research on audio-visual attacks and defense.



## **10. On the Natural Robustness of Vision-Language Models Against Visual Perception Attacks in Autonomous Driving**

cs.CV

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11472v1) [paper-pdf](http://arxiv.org/pdf/2506.11472v1)

**Authors**: Pedram MohajerAnsari, Amir Salarpour, Michael Kühr, Siyu Huang, Mohammad Hamad, Sebastian Steinhorst, Habeeb Olufowobi, Mert D. Pesé

**Abstract**: Autonomous vehicles (AVs) rely on deep neural networks (DNNs) for critical tasks such as traffic sign recognition (TSR), automated lane centering (ALC), and vehicle detection (VD). However, these models are vulnerable to attacks that can cause misclassifications and compromise safety. Traditional defense mechanisms, including adversarial training, often degrade benign accuracy and fail to generalize against unseen attacks. In this work, we introduce Vehicle Vision Language Models (V2LMs), fine-tuned vision-language models specialized for AV perception. Our findings demonstrate that V2LMs inherently exhibit superior robustness against unseen attacks without requiring adversarial training, maintaining significantly higher accuracy than conventional DNNs under adversarial conditions. We evaluate two deployment strategies: Solo Mode, where individual V2LMs handle specific perception tasks, and Tandem Mode, where a single unified V2LM is fine-tuned for multiple tasks simultaneously. Experimental results reveal that DNNs suffer performance drops of 33% to 46% under attacks, whereas V2LMs maintain adversarial accuracy with reductions of less than 8% on average. The Tandem Mode further offers a memory-efficient alternative while achieving comparable robustness to Solo Mode. We also explore integrating V2LMs as parallel components to AV perception to enhance resilience against adversarial threats. Our results suggest that V2LMs offer a promising path toward more secure and resilient AV perception systems.



## **11. Bias Amplification in RAG: Poisoning Knowledge Retrieval to Steer LLMs**

cs.LG

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11415v1) [paper-pdf](http://arxiv.org/pdf/2506.11415v1)

**Authors**: Linlin Wang, Tianqing Zhu, Laiqiao Qin, Longxiang Gao, Wanlei Zhou

**Abstract**: In Large Language Models, Retrieval-Augmented Generation (RAG) systems can significantly enhance the performance of large language models by integrating external knowledge. However, RAG also introduces new security risks. Existing research focuses mainly on how poisoning attacks in RAG systems affect model output quality, overlooking their potential to amplify model biases. For example, when querying about domestic violence victims, a compromised RAG system might preferentially retrieve documents depicting women as victims, causing the model to generate outputs that perpetuate gender stereotypes even when the original query is gender neutral. To show the impact of the bias, this paper proposes a Bias Retrieval and Reward Attack (BRRA) framework, which systematically investigates attack pathways that amplify language model biases through a RAG system manipulation. We design an adversarial document generation method based on multi-objective reward functions, employ subspace projection techniques to manipulate retrieval results, and construct a cyclic feedback mechanism for continuous bias amplification. Experiments on multiple mainstream large language models demonstrate that BRRA attacks can significantly enhance model biases in dimensions. In addition, we explore a dual stage defense mechanism to effectively mitigate the impacts of the attack. This study reveals that poisoning attacks in RAG systems directly amplify model output biases and clarifies the relationship between RAG system security and model fairness. This novel potential attack indicates that we need to keep an eye on the fairness issues of the RAG system.



## **12. PLeak: Prompt Leaking Attacks against Large Language Model Applications**

cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2405.06823v3) [paper-pdf](http://arxiv.org/pdf/2405.06823v3)

**Authors**: Bo Hui, Haolin Yuan, Neil Gong, Philippe Burlina, Yinzhi Cao

**Abstract**: Large Language Models (LLMs) enable a new ecosystem with many downstream applications, called LLM applications, with different natural language processing tasks. The functionality and performance of an LLM application highly depend on its system prompt, which instructs the backend LLM on what task to perform. Therefore, an LLM application developer often keeps a system prompt confidential to protect its intellectual property. As a result, a natural attack, called prompt leaking, is to steal the system prompt from an LLM application, which compromises the developer's intellectual property. Existing prompt leaking attacks primarily rely on manually crafted queries, and thus achieve limited effectiveness.   In this paper, we design a novel, closed-box prompt leaking attack framework, called PLeak, to optimize an adversarial query such that when the attacker sends it to a target LLM application, its response reveals its own system prompt. We formulate finding such an adversarial query as an optimization problem and solve it with a gradient-based method approximately. Our key idea is to break down the optimization goal by optimizing adversary queries for system prompts incrementally, i.e., starting from the first few tokens of each system prompt step by step until the entire length of the system prompt.   We evaluate PLeak in both offline settings and for real-world LLM applications, e.g., those on Poe, a popular platform hosting such applications. Our results show that PLeak can effectively leak system prompts and significantly outperforms not only baselines that manually curate queries but also baselines with optimized queries that are modified and adapted from existing jailbreaking attacks. We responsibly reported the issues to Poe and are still waiting for their response. Our implementation is available at this repository: https://github.com/BHui97/PLeak.



## **13. Byzantine Outside, Curious Inside: Reconstructing Data Through Malicious Updates**

cs.LG

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11413v1) [paper-pdf](http://arxiv.org/pdf/2506.11413v1)

**Authors**: Kai Yue, Richeng Jin, Chau-Wai Wong, Huaiyu Dai

**Abstract**: Federated learning (FL) enables decentralized machine learning without sharing raw data, allowing multiple clients to collaboratively learn a global model. However, studies reveal that privacy leakage is possible under commonly adopted FL protocols. In particular, a server with access to client gradients can synthesize data resembling the clients' training data. In this paper, we introduce a novel threat model in FL, named the maliciously curious client, where a client manipulates its own gradients with the goal of inferring private data from peers. This attacker uniquely exploits the strength of a Byzantine adversary, traditionally aimed at undermining model robustness, and repurposes it to facilitate data reconstruction attack. We begin by formally defining this novel client-side threat model and providing a theoretical analysis that demonstrates its ability to achieve significant reconstruction success during FL training. To demonstrate its practical impact, we further develop a reconstruction algorithm that combines gradient inversion with malicious update strategies. Our analysis and experimental results reveal a critical blind spot in FL defenses: both server-side robust aggregation and client-side privacy mechanisms may fail against our proposed attack. Surprisingly, standard server- and client-side defenses designed to enhance robustness or privacy may unintentionally amplify data leakage. Compared to the baseline approach, a mistakenly used defense may instead improve the reconstructed image quality by 10-15%.



## **14. Towards Robust Recommendation: A Review and an Adversarial Robustness Evaluation Library**

cs.IR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2404.17844v3) [paper-pdf](http://arxiv.org/pdf/2404.17844v3)

**Authors**: Lei Cheng, Xiaowen Huang, Jitao Sang, Jian Yu

**Abstract**: Recently, recommender system has achieved significant success. However, due to the openness of recommender systems, they remain vulnerable to malicious attacks. Additionally, natural noise in training data and issues such as data sparsity can also degrade the performance of recommender systems. Therefore, enhancing the robustness of recommender systems has become an increasingly important research topic. In this survey, we provide a comprehensive overview of the robustness of recommender systems. Based on our investigation, we categorize the robustness of recommender systems into adversarial robustness and non-adversarial robustness. In the adversarial robustness, we introduce the fundamental principles and classical methods of recommender system adversarial attacks and defenses. In the non-adversarial robustness, we analyze non-adversarial robustness from the perspectives of data sparsity, natural noise, and data imbalance. Additionally, we summarize commonly used datasets and evaluation metrics for evaluating the robustness of recommender systems. Finally, we also discuss the current challenges in the field of recommender system robustness and potential future research directions. Additionally, to facilitate fair and efficient evaluation of attack and defense methods in adversarial robustness, we propose an adversarial robustness evaluation library--ShillingREC, and we conduct evaluations of basic attack models and recommendation models. ShillingREC project is released at https://github.com/chengleileilei/ShillingREC.



## **15. Deception Against Data-Driven Linear-Quadratic Control**

eess.SY

16 pages, 5 figures

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11373v1) [paper-pdf](http://arxiv.org/pdf/2506.11373v1)

**Authors**: Filippos Fotiadis, Aris Kanellopoulos, Kyriakos G. Vamvoudakis, Ufuk Topcu

**Abstract**: Deception is a common defense mechanism against adversaries with an information disadvantage. It can force such adversaries to select suboptimal policies for a defender's benefit. We consider a setting where an adversary tries to learn the optimal linear-quadratic attack against a system, the dynamics of which it does not know. On the other end, a defender who knows its dynamics exploits its information advantage and injects a deceptive input into the system to mislead the adversary. The defender's aim is to then strategically design this deceptive input: it should force the adversary to learn, as closely as possible, a pre-selected attack that is different from the optimal one. We show that this deception design problem boils down to the solution of a coupled algebraic Riccati and a Lyapunov equation which, however, are challenging to tackle analytically. Nevertheless, we use a block successive over-relaxation algorithm to extract their solution numerically and prove the algorithm's convergence under certain conditions. We perform simulations on a benchmark aircraft, where we showcase how the proposed algorithm can mislead adversaries into learning attacks that are less performance-degrading.



## **16. On the Stability of Graph Convolutional Neural Networks: A Probabilistic Perspective**

cs.LG

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.01213v3) [paper-pdf](http://arxiv.org/pdf/2506.01213v3)

**Authors**: Ning Zhang, Henry Kenlay, Li Zhang, Mihai Cucuringu, Xiaowen Dong

**Abstract**: Graph convolutional neural networks (GCNNs) have emerged as powerful tools for analyzing graph-structured data, achieving remarkable success across diverse applications. However, the theoretical understanding of the stability of these models, i.e., their sensitivity to small changes in the graph structure, remains in rather limited settings, hampering the development and deployment of robust and trustworthy models in practice. To fill this gap, we study how perturbations in the graph topology affect GCNN outputs and propose a novel formulation for analyzing model stability. Unlike prior studies that focus only on worst-case perturbations, our distribution-aware formulation characterizes output perturbations across a broad range of input data. This way, our framework enables, for the first time, a probabilistic perspective on the interplay between the statistical properties of the node data and perturbations in the graph topology. We conduct extensive experiments to validate our theoretical findings and demonstrate their benefits over existing baselines, in terms of both representation stability and adversarial attacks on downstream tasks. Our results demonstrate the practical significance of the proposed formulation and highlight the importance of incorporating data distribution into stability analysis.



## **17. Improving LLM Safety Alignment with Dual-Objective Optimization**

cs.CL

ICML 2025

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2503.03710v2) [paper-pdf](http://arxiv.org/pdf/2503.03710v2)

**Authors**: Xuandong Zhao, Will Cai, Tianneng Shi, David Huang, Licong Lin, Song Mei, Dawn Song

**Abstract**: Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at https://github.com/wicai24/DOOR-Alignment



## **18. Weak-to-Strong Jailbreaking on Large Language Models**

cs.CL

ICML 2025

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2401.17256v3) [paper-pdf](http://arxiv.org/pdf/2401.17256v3)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong



## **19. Lattice Climber Attack: Adversarial attacks for randomized mixtures of classifiers**

cs.LG

17 pages including bibliography + 13 pages of supplementary material.  Extended version of the article accepted at ECML 2025

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10888v1) [paper-pdf](http://arxiv.org/pdf/2506.10888v1)

**Authors**: Lucas Gnecco-Heredia, Benjamin Negrevergne, Yann Chevaleyre

**Abstract**: Finite mixtures of classifiers (a.k.a. randomized ensembles) have been proposed as a way to improve robustness against adversarial attacks. However, existing attacks have been shown to not suit this kind of classifier. In this paper, we discuss the problem of attacking a mixture in a principled way and introduce two desirable properties of attacks based on a geometrical analysis of the problem (effectiveness and maximality). We then show that existing attacks do not meet both of these properties. Finally, we introduce a new attack called {\em lattice climber attack} with theoretical guarantees in the binary linear setting, and demonstrate its performance by conducting experiments on synthetic and real datasets.



## **20. Unveiling the Role of Randomization in Multiclass Adversarial Classification: Insights from Graph Theory**

cs.LG

9 pages (main), 30 in total. Camera-ready version, accepted at  AISTATS 2025. Erratum: Figure 3 was wrong, the three balls had a common  intersection when they were not supposed to. Fixed the value of radius in  tikz code

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2503.14299v2) [paper-pdf](http://arxiv.org/pdf/2503.14299v2)

**Authors**: Lucas Gnecco-Heredia, Matteo Sammut, Muni Sreenivas Pydi, Rafael Pinot, Benjamin Negrevergne, Yann Chevaleyre

**Abstract**: Randomization as a mean to improve the adversarial robustness of machine learning models has recently attracted significant attention. Unfortunately, much of the theoretical analysis so far has focused on binary classification, providing only limited insights into the more complex multiclass setting. In this paper, we take a step toward closing this gap by drawing inspiration from the field of graph theory. Our analysis focuses on discrete data distributions, allowing us to cast the adversarial risk minimization problems within the well-established framework of set packing problems. By doing so, we are able to identify three structural conditions on the support of the data distribution that are necessary for randomization to improve robustness. Furthermore, we are able to construct several data distributions where (contrarily to binary classification) switching from a deterministic to a randomized solution significantly reduces the optimal adversarial risk. These findings highlight the crucial role randomization can play in enhancing robustness to adversarial attacks in multiclass classification.



## **21. Breaking Distortion-free Watermarks in Large Language Models**

cs.CR

22 pages, 5 figures, 4 tables, earlier version presented at AAAI'25  Workshop on Preventing and Detecting LLM Generated Misinformation

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2502.18608v2) [paper-pdf](http://arxiv.org/pdf/2502.18608v2)

**Authors**: Shayleen Reynolds, Hengzhi He, Dung Daniel T. Ngo, Saheed Obitayo, Niccolò Dalmasso, Guang Cheng, Vamsi K. Potluru, Manuela Veloso

**Abstract**: In recent years, LLM watermarking has emerged as an attractive safeguard against AI-generated content, with promising applications in many real-world domains. However, there are growing concerns that the current LLM watermarking schemes are vulnerable to expert adversaries wishing to reverse-engineer the watermarking mechanisms. Prior work in breaking or stealing LLM watermarks mainly focuses on the distribution-modifying algorithm of Kirchenbauer et al. (2023), which perturbs the logit vector before sampling. In this work, we focus on reverse-engineering the other prominent LLM watermarking scheme, distortion-free watermarking (Kuditipudi et al. 2024), which preserves the underlying token distribution by using a hidden watermarking key sequence. We demonstrate that, even under a more sophisticated watermarking scheme, it is possible to compromise the LLM and carry out a spoofing attack, i.e. generate a large number of (potentially harmful) texts that can be attributed to the original watermarked LLM. Specifically, we propose using adaptive prompting and a sorting-based algorithm to accurately recover the underlying secret key for watermarking the LLM. Our empirical findings on LLAMA-3.1-8B-Instruct, Mistral-7B-Instruct, Gemma-7b, and OPT-125M challenge the current theoretical claims on the robustness and usability of the distortion-free watermarking techniques.



## **22. Efficiency Robustness of Dynamic Deep Learning Systems**

cs.LG

Accepted to USENIX Security '25

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10831v1) [paper-pdf](http://arxiv.org/pdf/2506.10831v1)

**Authors**: Ravishka Rathnasuriya, Tingxi Li, Zexin Xu, Zihe Song, Mirazul Haque, Simin Chen, Wei Yang

**Abstract**: Deep Learning Systems (DLSs) are increasingly deployed in real-time applications, including those in resourceconstrained environments such as mobile and IoT devices. To address efficiency challenges, Dynamic Deep Learning Systems (DDLSs) adapt inference computation based on input complexity, reducing overhead. While this dynamic behavior improves efficiency, such behavior introduces new attack surfaces. In particular, efficiency adversarial attacks exploit these dynamic mechanisms to degrade system performance. This paper systematically explores efficiency robustness of DDLSs, presenting the first comprehensive taxonomy of efficiency attacks. We categorize these attacks based on three dynamic behaviors: (i) attacks on dynamic computations per inference, (ii) attacks on dynamic inference iterations, and (iii) attacks on dynamic output production for downstream tasks. Through an in-depth evaluation, we analyze adversarial strategies that target DDLSs efficiency and identify key challenges in securing these systems. In addition, we investigate existing defense mechanisms, demonstrating their limitations against increasingly popular efficiency attacks and the necessity for novel mitigation strategies to secure future adaptive DDLSs.



## **23. Unsourced Adversarial CAPTCHA: A Bi-Phase Adversarial CAPTCHA Framework**

cs.CV

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10685v1) [paper-pdf](http://arxiv.org/pdf/2506.10685v1)

**Authors**: Xia Du, Xiaoyuan Liu, Jizhe Zhou, Zheng Lin, Chi-man Pun, Zhe Chen, Wei Ni, Jun Luo

**Abstract**: With the rapid advancements in deep learning, traditional CAPTCHA schemes are increasingly vulnerable to automated attacks powered by deep neural networks (DNNs). Existing adversarial attack methods often rely on original image characteristics, resulting in distortions that hinder human interpretation and limit applicability in scenarios lacking initial input images. To address these challenges, we propose the Unsourced Adversarial CAPTCHA (UAC), a novel framework generating high-fidelity adversarial examples guided by attacker-specified text prompts. Leveraging a Large Language Model (LLM), UAC enhances CAPTCHA diversity and supports both targeted and untargeted attacks. For targeted attacks, the EDICT method optimizes dual latent variables in a diffusion model for superior image quality. In untargeted attacks, especially for black-box scenarios, we introduce bi-path unsourced adversarial CAPTCHA (BP-UAC), a two-step optimization strategy employing multimodal gradients and bi-path optimization for efficient misclassification. Experiments show BP-UAC achieves high attack success rates across diverse systems, generating natural CAPTCHAs indistinguishable to humans and DNNs.



## **24. Experimental Verification of Entangled States in the Adversarial Scenario**

quant-ph

9 pages, 5 figures

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10655v1) [paper-pdf](http://arxiv.org/pdf/2506.10655v1)

**Authors**: Wen-Hao Zhang, Zihao Li, Gong-Chu Li, Xu-Song Hong, Huangjun Zhu, Geng Chen, Chuan-Feng Li, Guang-Can Guo

**Abstract**: Efficient verification of entangled states is crucial to many applications in quantum information processing. However, the effectiveness of standard quantum state verification (QSV) is based on the condition of independent and identical distribution (IID), which impedes its applications in many practical scenarios. Here we demonstrate a defensive QSV protocol, which is effective in all kinds of non-IID scenarios, including the extremely challenging adversarial scenario. To this end, we build a high-speed preparation-and-measurement apparatus controlled by quantum random-number generators. Our experiments clearly show that standard QSV protocols often provide unreliable fidelity certificates in non-IID scenarios. In sharp contrast, the defensive QSV protocol based on a homogeneous strategy can provide reliable and nearly tight fidelity certificates at comparable high efficiency, even under malicious attacks. Moreover, our scheme is robust against the imperfections in a realistic experiment, which is very appealing to practical applications.



## **25. Assessing the Resilience of Automotive Intrusion Detection Systems to Adversarial Manipulation**

cs.CR

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10620v1) [paper-pdf](http://arxiv.org/pdf/2506.10620v1)

**Authors**: Stefano Longari, Paolo Cerracchio, Michele Carminati, Stefano Zanero

**Abstract**: The security of modern vehicles has become increasingly important, with the controller area network (CAN) bus serving as a critical communication backbone for various Electronic Control Units (ECUs). The absence of robust security measures in CAN, coupled with the increasing connectivity of vehicles, makes them susceptible to cyberattacks. While intrusion detection systems (IDSs) have been developed to counter such threats, they are not foolproof. Adversarial attacks, particularly evasion attacks, can manipulate inputs to bypass detection by IDSs. This paper extends our previous work by investigating the feasibility and impact of gradient-based adversarial attacks performed with different degrees of knowledge against automotive IDSs. We consider three scenarios: white-box (attacker with full system knowledge), grey-box (partial system knowledge), and the more realistic black-box (no knowledge of the IDS' internal workings or data). We evaluate the effectiveness of the proposed attacks against state-of-the-art IDSs on two publicly available datasets. Additionally, we study effect of the adversarial perturbation on the attack impact and evaluate real-time feasibility by precomputing evasive payloads for timed injection based on bus traffic. Our results demonstrate that, besides attacks being challenging due to the automotive domain constraints, their effectiveness is strongly dependent on the dataset quality, the target IDS, and the attacker's degree of knowledge.



## **26. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2502.16750v4) [paper-pdf](http://arxiv.org/pdf/2502.16750v4)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehenaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.



## **27. Don't Lag, RAG: Training-Free Adversarial Detection Using RAG**

cs.AI

Accepted at VecDB @ ICML 2025

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2504.04858v2) [paper-pdf](http://arxiv.org/pdf/2504.04858v2)

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types, all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs, including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95 percent classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98 percent, but remains closed-source. Experimental results demonstrate VRAG's effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks.



## **28. Boosting Adversarial Transferability for Hyperspectral Image Classification Using 3D Structure-invariant Transformation and Intermediate Feature Distance**

cs.CV

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10459v1) [paper-pdf](http://arxiv.org/pdf/2506.10459v1)

**Authors**: Chun Liu, Bingqian Zhu, Tao Xu, Zheng Zheng, Zheng Li, Wei Yang, Zhigang Han, Jiayao Wang

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial attacks, which pose security challenges to hyperspectral image (HSI) classification technologies based on DNNs. In the domain of natural images, numerous transfer-based adversarial attack methods have been studied. However, HSIs differ from natural images due to their high-dimensional and rich spectral information. Current research on HSI adversarial examples remains limited and faces challenges in fully utilizing the structural and feature information of images. To address these issues, this paper proposes a novel method to enhance the transferability of the adversarial examples for HSI classification models. First, while keeping the image structure unchanged, the proposed method randomly divides the image into blocks in both spatial and spectral dimensions. Then, various transformations are applied on a block by block basis to increase input diversity and mitigate overfitting. Second, a feature distancing loss targeting intermediate layers is designed, which measures the distance between the amplified features of the original examples and the features of the adversarial examples as the primary loss, while the output layer prediction serves as the auxiliary loss. This guides the perturbation to disrupt the features of the true class in adversarial examples, effectively enhancing transferability. Extensive experiments demonstrate that the adversarial examples generated by the proposed method achieve effective transferability to black-box models on two public HSI datasets. Furthermore, the method maintains robust attack performance even under defense strategies.



## **29. TooBadRL: Trigger Optimization to Boost Effectiveness of Backdoor Attacks on Deep Reinforcement Learning**

cs.CR

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.09562v2) [paper-pdf](http://arxiv.org/pdf/2506.09562v2)

**Authors**: Songze Li, Mingxuan Zhang, Kang Wei, Shouling Ji

**Abstract**: Deep reinforcement learning (DRL) has achieved remarkable success in a wide range of sequential decision-making domains, including robotics, healthcare, smart grids, and finance. Recent research demonstrates that attackers can efficiently exploit system vulnerabilities during the training phase to execute backdoor attacks, producing malicious actions when specific trigger patterns are present in the state observations. However, most existing backdoor attacks rely primarily on simplistic and heuristic trigger configurations, overlooking the potential efficacy of trigger optimization. To address this gap, we introduce TooBadRL (Trigger Optimization to Boost Effectiveness of Backdoor Attacks on DRL), the first framework to systematically optimize DRL backdoor triggers along three critical axes, i.e., temporal, spatial, and magnitude. Specifically, we first introduce a performance-aware adaptive freezing mechanism for injection timing. Then, we formulate dimension selection as a cooperative game, utilizing Shapley value analysis to identify the most influential state variable for the injection dimension. Furthermore, we propose a gradient-based adversarial procedure to optimize the injection magnitude under environment constraints. Evaluations on three mainstream DRL algorithms and nine benchmark tasks show that TooBadRL significantly improves attack success rates, while ensuring minimal degradation of normal task performance. These results highlight the previously underappreciated importance of principled trigger optimization in DRL backdoor attacks. The source code of TooBadRL can be found at https://github.com/S3IC-Lab/TooBadRL.



## **30. Securing Large Language Models: Threats, Vulnerabilities and Responsible Practices**

cs.CR

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2403.12503v2) [paper-pdf](http://arxiv.org/pdf/2403.12503v2)

**Authors**: Sara Abdali, Richard Anarfi, CJ Barberan, Jia He, Erfan Shayegani

**Abstract**: Large language models (LLMs) have significantly transformed the landscape of Natural Language Processing (NLP). Their impact extends across a diverse spectrum of tasks, revolutionizing how we approach language understanding and generations. Nevertheless, alongside their remarkable utility, LLMs introduce critical security and risk considerations. These challenges warrant careful examination to ensure responsible deployment and safeguard against potential vulnerabilities. This research paper thoroughly investigates security and privacy concerns related to LLMs from five thematic perspectives: security and privacy concerns, vulnerabilities against adversarial attacks, potential harms caused by misuses of LLMs, mitigation strategies to address these challenges while identifying limitations of current strategies. Lastly, the paper recommends promising avenues for future research to enhance the security and risk management of LLMs.



## **31. AURA: A Multi-Agent Intelligence Framework for Knowledge-Enhanced Cyber Threat Attribution**

cs.CR

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.10175v1) [paper-pdf](http://arxiv.org/pdf/2506.10175v1)

**Authors**: Nanda Rani, Sandeep Kumar Shukla

**Abstract**: Effective attribution of Advanced Persistent Threats (APTs) increasingly hinges on the ability to correlate behavioral patterns and reason over complex, varied threat intelligence artifacts. We present AURA (Attribution Using Retrieval-Augmented Agents), a multi-agent, knowledge-enhanced framework for automated and interpretable APT attribution. AURA ingests diverse threat data including Tactics, Techniques, and Procedures (TTPs), Indicators of Compromise (IoCs), malware details, adversarial tools, and temporal information, which are processed through a network of collaborative agents. These agents are designed for intelligent query rewriting, context-enriched retrieval from structured threat knowledge bases, and natural language justification of attribution decisions. By combining Retrieval-Augmented Generation (RAG) with Large Language Models (LLMs), AURA enables contextual linking of threat behaviors to known APT groups and supports traceable reasoning across multiple attack phases. Experiments on recent APT campaigns demonstrate AURA's high attribution consistency, expert-aligned justifications, and scalability. This work establishes AURA as a promising direction for advancing transparent, data-driven, and scalable threat attribution using multi-agent intelligence.



## **32. LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge**

cs.CR

Dataset at:  https://huggingface.co/datasets/microsoft/llmail-inject-challenge

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09956v1) [paper-pdf](http://arxiv.org/pdf/2506.09956v1)

**Authors**: Sahar Abdelnabi, Aideen Fay, Ahmed Salem, Egor Zverev, Kai-Chieh Liao, Chi-Huang Liu, Chun-Chih Kuo, Jannis Weigend, Danyael Manlangit, Alex Apostolov, Haris Umair, João Donato, Masayuki Kawakita, Athar Mahboob, Tran Huu Bach, Tsun-Han Chiang, Myeongjin Cho, Hajin Choi, Byeonghyeon Kim, Hyeonjin Lee, Benjamin Pannell, Conor McCauley, Mark Russinovich, Andrew Paverd, Giovanni Cherubin

**Abstract**: Indirect Prompt Injection attacks exploit the inherent limitation of Large Language Models (LLMs) to distinguish between instructions and data in their inputs. Despite numerous defense proposals, the systematic evaluation against adaptive adversaries remains limited, even when successful attacks can have wide security and privacy implications, and many real-world LLM-based applications remain vulnerable. We present the results of LLMail-Inject, a public challenge simulating a realistic scenario in which participants adaptively attempted to inject malicious instructions into emails in order to trigger unauthorized tool calls in an LLM-based email assistant. The challenge spanned multiple defense strategies, LLM architectures, and retrieval configurations, resulting in a dataset of 208,095 unique attack submissions from 839 participants. We release the challenge code, the full dataset of submissions, and our analysis demonstrating how this data can provide new insights into the instruction-data separation problem. We hope this will serve as a foundation for future research towards practical structural solutions to prompt injection.



## **33. Generate-then-Verify: Reconstructing Data from Limited Published Statistics**

stat.ML

First two authors contributed equally. Remaining authors are ordered  alphabetically

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2504.21199v2) [paper-pdf](http://arxiv.org/pdf/2504.21199v2)

**Authors**: Terrance Liu, Eileen Xiao, Adam Smith, Pratiksha Thaker, Zhiwei Steven Wu

**Abstract**: We study the problem of reconstructing tabular data from aggregate statistics, in which the attacker aims to identify interesting claims about the sensitive data that can be verified with 100% certainty given the aggregates. Successful attempts in prior work have conducted studies in settings where the set of published statistics is rich enough that entire datasets can be reconstructed with certainty. In our work, we instead focus on the regime where many possible datasets match the published statistics, making it impossible to reconstruct the entire private dataset perfectly (i.e., when approaches in prior work fail). We propose the problem of partial data reconstruction, in which the goal of the adversary is to instead output a $\textit{subset}$ of rows and/or columns that are $\textit{guaranteed to be correct}$. We introduce a novel integer programming approach that first $\textbf{generates}$ a set of claims and then $\textbf{verifies}$ whether each claim holds for all possible datasets consistent with the published aggregates. We evaluate our approach on the housing-level microdata from the U.S. Decennial Census release, demonstrating that privacy violations can still persist even when information published about such data is relatively sparse.



## **34. Apollo: A Posteriori Label-Only Membership Inference Attack Towards Machine Unlearning**

cs.LG

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09923v1) [paper-pdf](http://arxiv.org/pdf/2506.09923v1)

**Authors**: Liou Tang, James Joshi, Ashish Kundu

**Abstract**: Machine Unlearning (MU) aims to update Machine Learning (ML) models following requests to remove training samples and their influences on a trained model efficiently without retraining the original ML model from scratch. While MU itself has been employed to provide privacy protection and regulatory compliance, it can also increase the attack surface of the model. Existing privacy inference attacks towards MU that aim to infer properties of the unlearned set rely on the weaker threat model that assumes the attacker has access to both the unlearned model and the original model, limiting their feasibility toward real-life scenarios. We propose a novel privacy attack, A Posteriori Label-Only Membership Inference Attack towards MU, Apollo, that infers whether a data sample has been unlearned, following a strict threat model where an adversary has access to the label-output of the unlearned model only. We demonstrate that our proposed attack, while requiring less access to the target model compared to previous attacks, can achieve relatively high precision on the membership status of the unlearned samples.



## **35. A look at adversarial attacks on radio waveforms from discrete latent space**

cs.LG

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09896v1) [paper-pdf](http://arxiv.org/pdf/2506.09896v1)

**Authors**: Attanasia Garuso, Silvija Kokalj-Filipovic, Yagna Kaasaragadda

**Abstract**: Having designed a VQVAE that maps digital radio waveforms into discrete latent space, and yields a perfectly classifiable reconstruction of the original data, we here analyze the attack suppressing properties of VQVAE when an adversarial attack is performed on high-SNR radio-frequency (RF) data-points. To target amplitude modulations from a subset of digitally modulated waveform classes, we first create adversarial attacks that preserve the phase between the in-phase and quadrature component whose values are adversarially changed. We compare them with adversarial attacks of the same intensity where phase is not preserved. We test the classification accuracy of such adversarial examples on a classifier trained to deliver 100% accuracy on the original data. To assess the ability of VQVAE to suppress the strength of the attack, we evaluate the classifier accuracy on the reconstructions by VQVAE of the adversarial datapoints and show that VQVAE substantially decreases the effectiveness of the attack. We also compare the I/Q plane diagram of the attacked data, their reconstructions and the original data. Finally, using multiple methods and metrics, we compare the probability distribution of the VQVAE latent space with and without attack. Varying the attack strength, we observe interesting properties of the discrete space, which may help detect the attacks.



## **36. One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image**

cs.CL

19 pages, 7 figures

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2504.02132v2) [paper-pdf](http://arxiv.org/pdf/2504.02132v2)

**Authors**: Ezzeldin Shereen, Dan Ristea, Shae McFadden, Burak Hasircioglu, Vasilios Mavroudis, Chris Hicks

**Abstract**: Multi-modal retrieval augmented generation (M-RAG) is instrumental for inhibiting hallucinations in large multi-modal models (LMMs) through the use of a factual knowledge base (KB). However, M-RAG introduces new attack vectors for adversaries that aim to disrupt the system by injecting malicious entries into the KB. In this paper, we present the first poisoning attack against M-RAG targeting visual document retrieval applications where the KB contains images of document pages. We propose two attacks, each of which require injecting only a single adversarial image into the KB. Firstly, we propose a universal attack that, for any potential user query, influences the response to cause a denial-of-service (DoS) in the M-RAG system. Secondly, we present a targeted attack against one or a group of user queries, with the goal of spreading targeted misinformation. For both attacks, we use a multi-objective gradient-based adversarial approach to craft the injected image while optimizing for both retrieval and generation. We evaluate our attacks against several visual document retrieval datasets, a diverse set of state-of-the-art retrievers (embedding models) and generators (LMMs), demonstrating the attack effectiveness in both the universal and targeted settings. We additionally present results including commonly used defenses, various attack hyper-parameter settings, ablations, and attack transferability.



## **37. CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization**

cs.CL

Accepted at ICML 2025, 20 pages

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2411.12768v2) [paper-pdf](http://arxiv.org/pdf/2411.12768v2)

**Authors**: Nay Myat Min, Long H. Pham, Yige Li, Jun Sun

**Abstract**: Large Language Models (LLMs) are vulnerable to backdoor attacks that manipulate outputs via hidden triggers. Existing defense methods--designed for vision/text classification tasks--fail for text generation. We propose Internal Consistency Regularization (CROW), a defense leveraging the observation that backdoored models exhibit unstable layer-wise hidden representations when triggered, while clean models show smooth transitions. CROW enforces consistency across layers via adversarial perturbations and regularization during finetuning, neutralizing backdoors without requiring clean reference models or trigger knowledge--only a small clean dataset. Experiments across Llama-2 (7B, 13B), CodeLlama (7B, 13B), and Mistral-7B demonstrate CROW's effectiveness: it achieves significant reductions in attack success rates across diverse backdoor strategies (sentiment steering, targeted refusal, code injection) while preserving generative performance. CROW's architecture-agnostic design enables practical deployment.



## **38. Distributionally and Adversarially Robust Logistic Regression via Intersecting Wasserstein Balls**

math.OC

9 main pages + 25 pages of appendices

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2407.13625v4) [paper-pdf](http://arxiv.org/pdf/2407.13625v4)

**Authors**: Aras Selvi, Eleonora Kreacic, Mohsen Ghassemi, Vamsi Potluru, Tucker Balch, Manuela Veloso

**Abstract**: Adversarially robust optimization (ARO) has emerged as the *de facto* standard for training models that hedge against adversarial attacks in the test stage. While these models are robust against adversarial attacks, they tend to suffer severely from overfitting. To address this issue, some successful methods replace the empirical distribution in the training stage with alternatives including *(i)* a worst-case distribution residing in an ambiguity set, resulting in a distributionally robust (DR) counterpart of ARO; *(ii)* a mixture of the empirical distribution with a distribution induced by an auxiliary (*e.g.*, synthetic, external, out-of-domain) dataset. Inspired by the former, we study the Wasserstein DR counterpart of ARO for logistic regression and show it admits a tractable convex optimization reformulation. Adopting the latter setting, we revise the DR approach by intersecting its ambiguity set with another ambiguity set built using the auxiliary dataset, which offers a significant improvement whenever the Wasserstein distance between the data generating and auxiliary distributions can be estimated. We study the underlying optimization problem, develop efficient solution algorithms, and demonstrate that the proposed method outperforms benchmark approaches on standard datasets.



## **39. Evasion Attacks Against Bayesian Predictive Models**

stat.ML

Accepted as an oral presentation at UAI'25

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09640v1) [paper-pdf](http://arxiv.org/pdf/2506.09640v1)

**Authors**: Pablo G. Arce, Roi Naveiro, David Ríos Insua

**Abstract**: There is an increasing interest in analyzing the behavior of machine learning systems against adversarial attacks. However, most of the research in adversarial machine learning has focused on studying weaknesses against evasion or poisoning attacks to predictive models in classical setups, with the susceptibility of Bayesian predictive models to attacks remaining underexplored. This paper introduces a general methodology for designing optimal evasion attacks against such models. We investigate two adversarial objectives: perturbing specific point predictions and altering the entire posterior predictive distribution. For both scenarios, we propose novel gradient-based attacks and study their implementation and properties in various computational setups.



## **40. Effective Red-Teaming of Policy-Adherent Agents**

cs.MA

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09600v1) [paper-pdf](http://arxiv.org/pdf/2506.09600v1)

**Authors**: Itay Nakash, George Kour, Koren Lazar, Matan Vetzler, Guy Uziel, Ateret Anaby-Tavor

**Abstract**: Task-oriented LLM-based agents are increasingly used in domains with strict policies, such as refund eligibility or cancellation rules. The challenge lies in ensuring that the agent consistently adheres to these rules and policies, appropriately refusing any request that would violate them, while still maintaining a helpful and natural interaction. This calls for the development of tailored design and evaluation methodologies to ensure agent resilience against malicious user behavior. We propose a novel threat model that focuses on adversarial users aiming to exploit policy-adherent agents for personal benefit. To address this, we present CRAFT, a multi-agent red-teaming system that leverages policy-aware persuasive strategies to undermine a policy-adherent agent in a customer-service scenario, outperforming conventional jailbreak methods such as DAN prompts, emotional manipulation, and coercive. Building upon the existing tau-bench benchmark, we introduce tau-break, a complementary benchmark designed to rigorously assess the agent's robustness against manipulative user behavior. Finally, we evaluate several straightforward yet effective defense strategies. While these measures provide some protection, they fall short, highlighting the need for stronger, research-driven safeguards to protect policy-adherent agents from adversarial attacks



## **41. RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards**

cs.AI

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.07736v2) [paper-pdf](http://arxiv.org/pdf/2506.07736v2)

**Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements.



## **42. AngleRoCL: Angle-Robust Concept Learning for Physically View-Invariant T2I Adversarial Patches**

cs.CV

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09538v1) [paper-pdf](http://arxiv.org/pdf/2506.09538v1)

**Authors**: Wenjun Ji, Yuxiang Fu, Luyang Ying, Deng-Ping Fan, Yuyi Wang, Ming-Ming Cheng, Ivor Tsang, Qing Guo

**Abstract**: Cutting-edge works have demonstrated that text-to-image (T2I) diffusion models can generate adversarial patches that mislead state-of-the-art object detectors in the physical world, revealing detectors' vulnerabilities and risks. However, these methods neglect the T2I patches' attack effectiveness when observed from different views in the physical world (i.e., angle robustness of the T2I adversarial patches). In this paper, we study the angle robustness of T2I adversarial patches comprehensively, revealing their angle-robust issues, demonstrating that texts affect the angle robustness of generated patches significantly, and task-specific linguistic instructions fail to enhance the angle robustness. Motivated by the studies, we introduce Angle-Robust Concept Learning (AngleRoCL), a simple and flexible approach that learns a generalizable concept (i.e., text embeddings in implementation) representing the capability of generating angle-robust patches. The learned concept can be incorporated into textual prompts and guides T2I models to generate patches with their attack effectiveness inherently resistant to viewpoint variations. Through extensive simulation and physical-world experiments on five SOTA detectors across multiple views, we demonstrate that AngleRoCL significantly enhances the angle robustness of T2I adversarial patches compared to baseline methods. Our patches maintain high attack success rates even under challenging viewing conditions, with over 50% average relative improvement in attack effectiveness across multiple angles. This research advances the understanding of physically angle-robust patches and provides insights into the relationship between textual concepts and physical properties in T2I-generated contents.



## **43. GenBreak: Red Teaming Text-to-Image Generators Using Large Language Models**

cs.CR

27 pages, 7 figures

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.10047v1) [paper-pdf](http://arxiv.org/pdf/2506.10047v1)

**Authors**: Zilong Wang, Xiang Zheng, Xiaosen Wang, Bo Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Text-to-image (T2I) models such as Stable Diffusion have advanced rapidly and are now widely used in content creation. However, these models can be misused to generate harmful content, including nudity or violence, posing significant safety risks. While most platforms employ content moderation systems, underlying vulnerabilities can still be exploited by determined adversaries. Recent research on red-teaming and adversarial attacks against T2I models has notable limitations: some studies successfully generate highly toxic images but use adversarial prompts that are easily detected and blocked by safety filters, while others focus on bypassing safety mechanisms but fail to produce genuinely harmful outputs, neglecting the discovery of truly high-risk prompts. Consequently, there remains a lack of reliable tools for evaluating the safety of defended T2I models. To address this gap, we propose GenBreak, a framework that fine-tunes a red-team large language model (LLM) to systematically explore underlying vulnerabilities in T2I generators. Our approach combines supervised fine-tuning on curated datasets with reinforcement learning via interaction with a surrogate T2I model. By integrating multiple reward signals, we guide the LLM to craft adversarial prompts that enhance both evasion capability and image toxicity, while maintaining semantic coherence and diversity. These prompts demonstrate strong effectiveness in black-box attacks against commercial T2I generators, revealing practical and concerning safety weaknesses.



## **44. On the Privacy Risks of Spiking Neural Networks: A Membership Inference Analysis**

cs.LG

14 pages, 6 figures

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2502.13191v4) [paper-pdf](http://arxiv.org/pdf/2502.13191v4)

**Authors**: Junyi Guan, Abhijith Sharma, Chong Tian, Salem Lahlou

**Abstract**: Spiking Neural Networks (SNNs) are increasingly explored for their energy efficiency and robustness in real-world applications, yet their privacy risks remain largely unexamined. In this work, we investigate the susceptibility of SNNs to Membership Inference Attacks (MIAs) -- a major privacy threat where an adversary attempts to determine whether a given sample was part of the training dataset. While prior work suggests that SNNs may offer inherent robustness due to their discrete, event-driven nature, we find that its resilience diminishes as latency (T) increases. Furthermore, we introduce an input dropout strategy under black box setting, that significantly enhances membership inference in SNNs. Our findings challenge the assumption that SNNs are inherently more secure, and even though they are expected to be better, our results reveal that SNNs exhibit privacy vulnerabilities that are equally comparable to Artificial Neural Networks (ANNs). Our code is available at https://github.com/sharmaabhijith/MIA_SNN.



## **45. LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge**

cs.CR

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09443v1) [paper-pdf](http://arxiv.org/pdf/2506.09443v1)

**Authors**: Songze Li, Chuokun Xu, Jiaying Wang, Xueluan Gong, Chen Chen, Jirui Zhang, Jun Wang, Kwok-Yan Lam, Shouling Ji

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable intelligence across various tasks, which has inspired the development and widespread adoption of LLM-as-a-Judge systems for automated model testing, such as red teaming and benchmarking. However, these systems are susceptible to adversarial attacks that can manipulate evaluation outcomes, raising concerns about their robustness and, consequently, their trustworthiness. Existing evaluation methods adopted by LLM-based judges are often piecemeal and lack a unified framework for comprehensive assessment. Furthermore, prompt template and model selections for improving judge robustness have been rarely explored, and their performance in real-world settings remains largely unverified. To address these gaps, we introduce RobustJudge, a fully automated and scalable framework designed to systematically evaluate the robustness of LLM-as-a-Judge systems. RobustJudge investigates the impact of attack methods and defense strategies (RQ1), explores the influence of prompt template and model selection (RQ2), and assesses the robustness of real-world LLM-as-a-Judge applications (RQ3).Our main findings are: (1) LLM-as-a-Judge systems are still vulnerable to a range of adversarial attacks, including Combined Attack and PAIR, while defense mechanisms such as Re-tokenization and LLM-based Detectors offer improved protection; (2) Robustness is highly sensitive to the choice of prompt template and judge models. Our proposed prompt template optimization method can improve robustness, and JudgeLM-13B demonstrates strong performance as a robust open-source judge; (3) Applying RobustJudge to Alibaba's PAI platform reveals previously unreported vulnerabilities. The source code of RobustJudge is provided at https://github.com/S3IC-Lab/RobustJudge.



## **46. AdversariaL attacK sAfety aLIgnment(ALKALI): Safeguarding LLMs through GRACE: Geometric Representation-Aware Contrastive Enhancement- Introducing Adversarial Vulnerability Quality Index (AVQI)**

cs.CL

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.08885v2) [paper-pdf](http://arxiv.org/pdf/2506.08885v2)

**Authors**: Danush Khanna, Krishna Kumar, Basab Ghosh, Vinija Jain, Vasu Sharma, Aman Chadha, Amitava Das

**Abstract**: Adversarial threats against LLMs are escalating faster than current defenses can adapt. We expose a critical geometric blind spot in alignment: adversarial prompts exploit latent camouflage, embedding perilously close to the safe representation manifold while encoding unsafe intent thereby evading surface level defenses like Direct Preference Optimization (DPO), which remain blind to the latent geometry. We introduce ALKALI, the first rigorously curated adversarial benchmark and the most comprehensive to date spanning 9,000 prompts across three macro categories, six subtypes, and fifteen attack families. Evaluation of 21 leading LLMs reveals alarmingly high Attack Success Rates (ASRs) across both open and closed source models, exposing an underlying vulnerability we term latent camouflage, a structural blind spot where adversarial completions mimic the latent geometry of safe ones. To mitigate this vulnerability, we introduce GRACE - Geometric Representation Aware Contrastive Enhancement, an alignment framework coupling preference learning with latent space regularization. GRACE enforces two constraints: latent separation between safe and adversarial completions, and adversarial cohesion among unsafe and jailbreak behaviors. These operate over layerwise pooled embeddings guided by a learned attention profile, reshaping internal geometry without modifying the base model, and achieve up to 39% ASR reduction. Moreover, we introduce AVQI, a geometry aware metric that quantifies latent alignment failure via cluster separation and compactness. AVQI reveals when unsafe completions mimic the geometry of safe ones, offering a principled lens into how models internally encode safety. We make the code publicly available at https://anonymous.4open.science/r/alkali-B416/README.md.



## **47. Adversarial Surrogate Risk Bounds for Binary Classification**

cs.LG

37 pages, 2 figures

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09348v1) [paper-pdf](http://arxiv.org/pdf/2506.09348v1)

**Authors**: Natalie S. Frank

**Abstract**: A central concern in classification is the vulnerability of machine learning models to adversarial attacks. Adversarial training is one of the most popular techniques for training robust classifiers, which involves minimizing an adversarial surrogate risk. Recent work characterized when a minimizing sequence of an adversarial surrogate risk is also a minimizing sequence of the adversarial classification risk for binary classification -- a property known as adversarial consistency. However, these results do not address the rate at which the adversarial classification risk converges to its optimal value for such a sequence of functions that minimize the adversarial surrogate. This paper provides surrogate risk bounds that quantify that convergence rate. Additionally, we derive distribution-dependent surrogate risk bounds in the standard (non-adversarial) learning setting, that may be of independent interest.



## **48. PatchGuard: Adversarially Robust Anomaly Detection and Localization through Vision Transformers and Pseudo Anomalies**

cs.CV

Accepted to the Conference on Computer Vision and Pattern Recognition  (CVPR) 2025

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2506.09237v1) [paper-pdf](http://arxiv.org/pdf/2506.09237v1)

**Authors**: Mojtaba Nafez, Amirhossein Koochakian, Arad Maleki, Jafar Habibi, Mohammad Hossein Rohban

**Abstract**: Anomaly Detection (AD) and Anomaly Localization (AL) are crucial in fields that demand high reliability, such as medical imaging and industrial monitoring. However, current AD and AL approaches are often susceptible to adversarial attacks due to limitations in training data, which typically include only normal, unlabeled samples. This study introduces PatchGuard, an adversarially robust AD and AL method that incorporates pseudo anomalies with localization masks within a Vision Transformer (ViT)-based architecture to address these vulnerabilities. We begin by examining the essential properties of pseudo anomalies, and follow it by providing theoretical insights into the attention mechanisms required to enhance the adversarial robustness of AD and AL systems. We then present our approach, which leverages Foreground-Aware Pseudo-Anomalies to overcome the deficiencies of previous anomaly-aware methods. Our method incorporates these crafted pseudo-anomaly samples into a ViT-based framework, with adversarial training guided by a novel loss function designed to improve model robustness, as supported by our theoretical analysis. Experimental results on well-established industrial and medical datasets demonstrate that PatchGuard significantly outperforms previous methods in adversarial settings, achieving performance gains of $53.2\%$ in AD and $68.5\%$ in AL, while also maintaining competitive accuracy in non-adversarial settings. The code repository is available at https://github.com/rohban-lab/PatchGuard .



## **49. PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning**

cs.CR

21 pages, 7 figures

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2411.17453v2) [paper-pdf](http://arxiv.org/pdf/2411.17453v2)

**Authors**: Zhen Sun, Tianshuo Cong, Yule Liu, Chenhao Lin, Xinlei He, Rongmao Chen, Xingshuo Han, Xinyi Huang

**Abstract**: Fine-tuning is an essential process to improve the performance of Large Language Models (LLMs) in specific domains, with Parameter-Efficient Fine-Tuning (PEFT) gaining popularity due to its capacity to reduce computational demands through the integration of low-rank adapters. These lightweight adapters, such as LoRA, can be shared and utilized on open-source platforms. However, adversaries could exploit this mechanism to inject backdoors into these adapters, resulting in malicious behaviors like incorrect or harmful outputs, which pose serious security risks to the community. Unfortunately, few current efforts concentrate on analyzing the backdoor patterns or detecting the backdoors in the adapters. To fill this gap, we first construct and release PADBench, a comprehensive benchmark that contains 13,300 benign and backdoored adapters fine-tuned with various datasets, attack strategies, PEFT methods, and LLMs. Moreover, we propose PEFTGuard, the first backdoor detection framework against PEFT-based adapters. Extensive evaluation upon PADBench shows that PEFTGuard outperforms existing detection methods, achieving nearly perfect detection accuracy (100%) in most cases. Notably, PEFTGuard exhibits zero-shot transferability on three aspects, including different attacks, PEFT methods, and adapter ranks. In addition, we consider various adaptive attacks to demonstrate the high robustness of PEFTGuard. We further explore several possible backdoor mitigation defenses, finding fine-mixing to be the most effective method. We envision that our benchmark and method can shed light on future LLM backdoor detection research.



## **50. Unified Breakdown Analysis for Byzantine Robust Gossip**

math.OC

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2410.10418v3) [paper-pdf](http://arxiv.org/pdf/2410.10418v3)

**Authors**: Renaud Gaucher, Aymeric Dieuleveut, Hadrien Hendrikx

**Abstract**: In decentralized machine learning, different devices communicate in a peer-to-peer manner to collaboratively learn from each other's data. Such approaches are vulnerable to misbehaving (or Byzantine) devices. We introduce F-RG, a general framework for building robust decentralized algorithms with guarantees arising from robust-sum-like aggregation rules F. We then investigate the notion of *breakdown point*, and show an upper bound on the number of adversaries that decentralized algorithms can tolerate. We introduce a practical robust aggregation rule, coined CS+, such that CS+-RG has a near-optimal breakdown. Other choices of aggregation rules lead to existing algorithms such as ClippedGossip or NNA. We give experimental evidence to validate the effectiveness of CS+-RG and highlight the gap with NNA, in particular against a novel attack tailored to decentralized communications.



