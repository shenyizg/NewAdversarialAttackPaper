# Latest Adversarial Attack Papers
**update at 2025-08-26 10:53:46**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks**

cs.CL

23 pages, 10 figures, 11 tables

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2503.00187v2) [paper-pdf](http://arxiv.org/pdf/2503.00187v2)

**Authors**: Hanjiang Hu, Alexander Robey, Changliu Liu

**Abstract**: Large language models (LLMs) are shown to be vulnerable to jailbreaking attacks where adversarial prompts are designed to elicit harmful responses. While existing defenses effectively mitigate single-turn attacks by detecting and filtering unsafe inputs, they fail against multi-turn jailbreaks that exploit contextual drift over multiple interactions, gradually leading LLMs away from safe behavior. To address this challenge, we propose a safety steering framework grounded in safe control theory, ensuring invariant safety in multi-turn dialogues. Our approach models the dialogue with LLMs using state-space representations and introduces a novel neural barrier function (NBF) to detect and filter harmful queries emerging from evolving contexts proactively. Our method achieves invariant safety at each turn of dialogue by learning a safety predictor that accounts for adversarial queries, preventing potential context drift toward jailbreaks. Extensive experiments under multiple LLMs show that our NBF-based safety steering outperforms safety alignment, prompt-based steering and lightweight LLM guardrails baselines, offering stronger defenses against multi-turn jailbreaks while maintaining a better trade-off among safety, helpfulness and over-refusal. Check out the website here https://sites.google.com/view/llm-nbf/home . Our code is available on https://github.com/HanjiangHu/NBF-LLM .



## **2. Transferring Styles for Reduced Texture Bias and Improved Robustness in Semantic Segmentation Networks**

cs.CV

accepted at ECAI 2025

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2507.10239v2) [paper-pdf](http://arxiv.org/pdf/2507.10239v2)

**Authors**: Ben Hamscher, Edgar Heinert, Annika Mütze, Kira Maag, Matthias Rottmann

**Abstract**: Recent research has investigated the shape and texture biases of deep neural networks (DNNs) in image classification which influence their generalization capabilities and robustness. It has been shown that, in comparison to regular DNN training, training with stylized images reduces texture biases in image classification and improves robustness with respect to image corruptions. In an effort to advance this line of research, we examine whether style transfer can likewise deliver these two effects in semantic segmentation. To this end, we perform style transfer with style varying across artificial image areas. Those random areas are formed by a chosen number of Voronoi cells. The resulting style-transferred data is then used to train semantic segmentation DNNs with the objective of reducing their dependence on texture cues while enhancing their reliance on shape-based features. In our experiments, it turns out that in semantic segmentation, style transfer augmentation reduces texture bias and strongly increases robustness with respect to common image corruptions as well as adversarial attacks. These observations hold for convolutional neural networks and transformer architectures on the Cityscapes dataset as well as on PASCAL Context, showing the generality of the proposed method.



## **3. Quantum-Classical Hybrid Framework for Zero-Day Time-Push GNSS Spoofing Detection**

cs.LG

This work has been submitted to the IEEE Internet of Things Journal  for possible publication

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.18085v1) [paper-pdf](http://arxiv.org/pdf/2508.18085v1)

**Authors**: Abyad Enan, Mashrur Chowdhury, Sagar Dasgupta, Mizanur Rahman

**Abstract**: Global Navigation Satellite Systems (GNSS) are critical for Positioning, Navigation, and Timing (PNT) applications. However, GNSS are highly vulnerable to spoofing attacks, where adversaries transmit counterfeit signals to mislead receivers. Such attacks can lead to severe consequences, including misdirected navigation, compromised data integrity, and operational disruptions. Most existing spoofing detection methods depend on supervised learning techniques and struggle to detect novel, evolved, and unseen attacks. To overcome this limitation, we develop a zero-day spoofing detection method using a Hybrid Quantum-Classical Autoencoder (HQC-AE), trained solely on authentic GNSS signals without exposure to spoofed data. By leveraging features extracted during the tracking stage, our method enables proactive detection before PNT solutions are computed. We focus on spoofing detection in static GNSS receivers, which are particularly susceptible to time-push spoofing attacks, where attackers manipulate timing information to induce incorrect time computations at the receiver. We evaluate our model against different unseen time-push spoofing attack scenarios: simplistic, intermediate, and sophisticated. Our analysis demonstrates that the HQC-AE consistently outperforms its classical counterpart, traditional supervised learning-based models, and existing unsupervised learning-based methods in detecting zero-day, unseen GNSS time-push spoofing attacks, achieving an average detection accuracy of 97.71% with an average false negative rate of 0.62% (when an attack occurs but is not detected). For sophisticated spoofing attacks, the HQC-AE attains an accuracy of 98.23% with a false negative rate of 1.85%. These findings highlight the effectiveness of our method in proactively detecting zero-day GNSS time-push spoofing attacks across various stationary GNSS receiver platforms.



## **4. Robust Federated Learning under Adversarial Attacks via Loss-Based Client Clustering**

cs.LG

16 pages, 5 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.12672v3) [paper-pdf](http://arxiv.org/pdf/2508.12672v3)

**Authors**: Emmanouil Kritharakis, Dusan Jakovetic, Antonios Makris, Konstantinos Tserpes

**Abstract**: Federated Learning (FL) enables collaborative model training across multiple clients without sharing private data. We consider FL scenarios wherein FL clients are subject to adversarial (Byzantine) attacks, while the FL server is trusted (honest) and has a trustworthy side dataset. This may correspond to, e.g., cases where the server possesses trusted data prior to federation, or to the presence of a trusted client that temporarily assumes the server role. Our approach requires only two honest participants, i.e., the server and one client, to function effectively, without prior knowledge of the number of malicious clients. Theoretical analysis demonstrates bounded optimality gaps even under strong Byzantine attacks. Experimental results show that our algorithm significantly outperforms standard and robust FL baselines such as Mean, Trimmed Mean, Median, Krum, and Multi-Krum under various attack strategies including label flipping, sign flipping, and Gaussian noise addition across MNIST, FMNIST, and CIFAR-10 benchmarks using the Flower framework.



## **5. FedGreed: A Byzantine-Robust Loss-Based Aggregation Method for Federated Learning**

cs.LG

8 pages, 4 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.18060v1) [paper-pdf](http://arxiv.org/pdf/2508.18060v1)

**Authors**: Emmanouil Kritharakis, Antonios Makris, Dusan Jakovetic, Konstantinos Tserpes

**Abstract**: Federated Learning (FL) enables collaborative model training across multiple clients while preserving data privacy by keeping local datasets on-device. In this work, we address FL settings where clients may behave adversarially, exhibiting Byzantine attacks, while the central server is trusted and equipped with a reference dataset. We propose FedGreed, a resilient aggregation strategy for federated learning that does not require any assumptions about the fraction of adversarial participants. FedGreed orders clients' local model updates based on their loss metrics evaluated against a trusted dataset on the server and greedily selects a subset of clients whose models exhibit the minimal evaluation loss. Unlike many existing approaches, our method is designed to operate reliably under heterogeneous (non-IID) data distributions, which are prevalent in real-world deployments. FedGreed exhibits convergence guarantees and bounded optimality gaps under strong adversarial behavior. Experimental evaluations on MNIST, FMNIST, and CIFAR-10 demonstrate that our method significantly outperforms standard and robust federated learning baselines, such as Mean, Trimmed Mean, Median, Krum, and Multi-Krum, in the majority of adversarial scenarios considered, including label flipping and Gaussian noise injection attacks. All experiments were conducted using the Flower federated learning framework.



## **6. Does simple trump complex? Comparing strategies for adversarial robustness in DNNs**

cs.LG

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.18019v1) [paper-pdf](http://arxiv.org/pdf/2508.18019v1)

**Authors**: William Brooks, Marelie H. Davel, Coenraad Mouton

**Abstract**: Deep Neural Networks (DNNs) have shown substantial success in various applications but remain vulnerable to adversarial attacks. This study aims to identify and isolate the components of two different adversarial training techniques that contribute most to increased adversarial robustness, particularly through the lens of margins in the input space -- the minimal distance between data points and decision boundaries. Specifically, we compare two methods that maximize margins: a simple approach which modifies the loss function to increase an approximation of the margin, and a more complex state-of-the-art method (Dynamics-Aware Robust Training) which builds upon this approach. Using a VGG-16 model as our base, we systematically isolate and evaluate individual components from these methods to determine their relative impact on adversarial robustness. We assess the effect of each component on the model's performance under various adversarial attacks, including AutoAttack and Projected Gradient Descent (PGD). Our analysis on the CIFAR-10 dataset reveals which elements most effectively enhance adversarial robustness, providing insights for designing more robust DNNs.



## **7. Adversarial Robustness in Two-Stage Learning-to-Defer: Algorithms and Guarantees**

stat.ML

Accepted at the 42nd International Conference on Machine Learning  (ICML 2025)

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2502.01027v4) [paper-pdf](http://arxiv.org/pdf/2502.01027v4)

**Authors**: Yannis Montreuil, Axel Carlier, Lai Xing Ng, Wei Tsang Ooi

**Abstract**: Two-stage Learning-to-Defer (L2D) enables optimal task delegation by assigning each input to either a fixed main model or one of several offline experts, supporting reliable decision-making in complex, multi-agent environments. However, existing L2D frameworks assume clean inputs and are vulnerable to adversarial perturbations that can manipulate query allocation--causing costly misrouting or expert overload. We present the first comprehensive study of adversarial robustness in two-stage L2D systems. We introduce two novel attack strategie--untargeted and targeted--which respectively disrupt optimal allocations or force queries to specific agents. To defend against such threats, we propose SARD, a convex learning algorithm built on a family of surrogate losses that are provably Bayes-consistent and $(\mathcal{R}, \mathcal{G})$-consistent. These guarantees hold across classification, regression, and multi-task settings. Empirical results demonstrate that SARD significantly improves robustness under adversarial attacks while maintaining strong clean performance, marking a critical step toward secure and trustworthy L2D deployment.



## **8. A Predictive Framework for Adversarial Energy Depletion in Inbound Threat Scenarios**

eess.SY

7 pages, 1 figure, 1 table, preprint submitted to the American  Control Conference (ACC) 2026

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.17805v1) [paper-pdf](http://arxiv.org/pdf/2508.17805v1)

**Authors**: Tam W. Nguyen

**Abstract**: This paper presents a predictive framework for adversarial energy-depletion defense against a maneuverable inbound threat (IT). The IT solves a receding-horizon problem to minimize its own energy while reaching a high-value asset (HVA) and avoiding interceptors and static lethal zones modeled by Gaussian barriers. Expendable interceptors (EIs), coordinated by a central node (CN), maintain proximity to the HVA and patrol centers via radius-based tether costs, deny attack corridors by harassing and containing the IT, and commit to intercept only when a geometric feasibility test is confirmed. No explicit opponent-energy term is used, and the formulation is optimization-implementable. No simulations are included.



## **9. Robustness Feature Adapter for Efficient Adversarial Training**

cs.LG

The paper has been accepted for presentation at ECAI 2025

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.17680v1) [paper-pdf](http://arxiv.org/pdf/2508.17680v1)

**Authors**: Quanwei Wu, Jun Guo, Wei Wang, Yi Wang

**Abstract**: Adversarial training (AT) with projected gradient descent is the most popular method to improve model robustness under adversarial attacks. However, computational overheads become prohibitively large when AT is applied to large backbone models. AT is also known to have the issue of robust overfitting. This paper contributes to solving both problems simultaneously towards building more trustworthy foundation models. In particular, we propose a new adapter-based approach for efficient AT directly in the feature space. We show that the proposed adapter-based approach can improve the inner-loop convergence quality by eliminating robust overfitting. As a result, it significantly increases computational efficiency and improves model accuracy by generalizing adversarial robustness to unseen attacks. We demonstrate the effectiveness of the new adapter-based approach in different backbone architectures and in AT at scale.



## **10. Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models**

cs.CR

7 pages, 2 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.17674v1) [paper-pdf](http://arxiv.org/pdf/2508.17674v1)

**Authors**: Qiming Guo, Jinwen Tang, Xingran Huang

**Abstract**: We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.



## **11. TombRaider: Entering the Vault of History to Jailbreak Large Language Models**

cs.CR

Main Conference of EMNLP

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2501.18628v2) [paper-pdf](http://arxiv.org/pdf/2501.18628v2)

**Authors**: Junchen Ding, Jiahao Zhang, Yi Liu, Ziqi Ding, Gelei Deng, Yuekang Li

**Abstract**: Warning: This paper contains content that may involve potentially harmful behaviours, discussed strictly for research purposes.   Jailbreak attacks can hinder the safety of Large Language Model (LLM) applications, especially chatbots. Studying jailbreak techniques is an important AI red teaming task for improving the safety of these applications. In this paper, we introduce TombRaider, a novel jailbreak technique that exploits the ability to store, retrieve, and use historical knowledge of LLMs. TombRaider employs two agents, the inspector agent to extract relevant historical information and the attacker agent to generate adversarial prompts, enabling effective bypassing of safety filters. We intensively evaluated TombRaider on six popular models. Experimental results showed that TombRaider could outperform state-of-the-art jailbreak techniques, achieving nearly 100% attack success rates (ASRs) on bare models and maintaining over 55.4% ASR against defence mechanisms. Our findings highlight critical vulnerabilities in existing LLM safeguards, underscoring the need for more robust safety defences.



## **12. Defending against Jailbreak through Early Exit Generation of Large Language Models**

cs.AI

ICONIP 2025

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2408.11308v2) [paper-pdf](http://arxiv.org/pdf/2408.11308v2)

**Authors**: Chongwen Zhao, Zhihao Dou, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation. In an effort to mitigate such risks, the concept of "Alignment" technology has been developed. However, recent studies indicate that this alignment can be undermined using sophisticated prompt engineering or adversarial suffixes, a technique known as "Jailbreak." Our research takes cues from the human-like generate process of LLMs. We identify that while jailbreaking prompts may yield output logits similar to benign prompts, their initial embeddings within the model's latent space tend to be more analogous to those of malicious prompts. Leveraging this finding, we propose utilizing the early transformer outputs of LLMs as a means to detect malicious inputs, and terminate the generation immediately. We introduce a simple yet significant defense approach called EEG-Defender for LLMs. We conduct comprehensive experiments on ten jailbreak methods across three models. Our results demonstrate that EEG-Defender is capable of reducing the Attack Success Rate (ASR) by a significant margin, roughly 85% in comparison with 50% for the present SOTAs, with minimal impact on the utility of LLMs.



## **13. SoK: Cybersecurity Assessment of Humanoid Ecosystem**

cs.CR

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17481v1) [paper-pdf](http://arxiv.org/pdf/2508.17481v1)

**Authors**: Priyanka Prakash Surve, Asaf Shabtai, Yuval Elovici

**Abstract**: Humanoids are progressing toward practical deployment across healthcare, industrial, defense, and service sectors. While typically considered cyber-physical systems (CPSs), their dependence on traditional networked software stacks (e.g., Linux operating systems), robot operating system (ROS) middleware, and over-the-air update channels, creates a distinct security profile that exposes them to vulnerabilities conventional CPS models do not fully address. Prior studies have mainly examined specific threats, such as LiDAR spoofing or adversarial machine learning (AML). This narrow focus overlooks how an attack targeting one component can cascade harm throughout the robot's interconnected systems. We address this gap through a systematization of knowledge (SoK) that takes a comprehensive approach, consolidating fragmented research from robotics, CPS, and network security domains. We introduce a seven-layer security model for humanoid robots, organizing 39 known attacks and 35 defenses across the humanoid ecosystem-from hardware to human-robot interaction. Building on this security model, we develop a quantitative 39x35 attack-defense matrix with risk-weighted scoring, validated through Monte Carlo analysis. We demonstrate our method by evaluating three real-world robots: Pepper, G1 EDU, and Digit. The scoring analysis revealed varying security maturity levels, with scores ranging from 39.9% to 79.5% across the platforms. This work introduces a structured, evidence-based assessment method that enables systematic security evaluation, supports cross-platform benchmarking, and guides prioritization of security investments in humanoid robotics.



## **14. FRAME : Comprehensive Risk Assessment Framework for Adversarial Machine Learning Threats**

cs.LG

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17405v1) [paper-pdf](http://arxiv.org/pdf/2508.17405v1)

**Authors**: Avishag Shapira, Simon Shigol, Asaf Shabtai

**Abstract**: The widespread adoption of machine learning (ML) systems increased attention to their security and emergence of adversarial machine learning (AML) techniques that exploit fundamental vulnerabilities in ML systems, creating an urgent need for comprehensive risk assessment for ML-based systems. While traditional risk assessment frameworks evaluate conventional cybersecurity risks, they lack ability to address unique challenges posed by AML threats. Existing AML threat evaluation approaches focus primarily on technical attack robustness, overlooking crucial real-world factors like deployment environments, system dependencies, and attack feasibility. Attempts at comprehensive AML risk assessment have been limited to domain-specific solutions, preventing application across diverse systems. Addressing these limitations, we present FRAME, the first comprehensive and automated framework for assessing AML risks across diverse ML-based systems. FRAME includes a novel risk assessment method that quantifies AML risks by systematically evaluating three key dimensions: target system's deployment environment, characteristics of diverse AML techniques, and empirical insights from prior research. FRAME incorporates a feasibility scoring mechanism and LLM-based customization for system-specific assessments. Additionally, we developed a comprehensive structured dataset of AML attacks enabling context-aware risk assessment. From an engineering application perspective, FRAME delivers actionable results designed for direct use by system owners with only technical knowledge of their systems, without expertise in AML. We validated it across six diverse real-world applications. Our evaluation demonstrated exceptional accuracy and strong alignment with analysis by AML experts. FRAME enables organizations to prioritize AML risks, supporting secure AI deployment in real-world environments.



## **15. Bridging Models to Defend: A Population-Based Strategy for Robust Adversarial Defense**

cs.AI

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2303.10225v2) [paper-pdf](http://arxiv.org/pdf/2303.10225v2)

**Authors**: Ren Wang, Yuxuan Li, Can Chen, Dakuo Wang, Jinjun Xiong, Pin-Yu Chen, Sijia Liu, Mohammad Shahidehpour, Alfred Hero

**Abstract**: Adversarial robustness is a critical measure of a neural network's ability to withstand adversarial attacks at inference time. While robust training techniques have improved defenses against individual $\ell_p$-norm attacks (e.g., $\ell_2$ or $\ell_\infty$), models remain vulnerable to diversified $\ell_p$ perturbations. To address this challenge, we propose a novel Robust Mode Connectivity (RMC)-oriented adversarial defense framework comprising two population-based learning phases. In Phase I, RMC searches the parameter space between two pre-trained models to construct a continuous path containing models with high robustness against multiple $\ell_p$ attacks. To improve efficiency, we introduce a Self-Robust Mode Connectivity (SRMC) module that accelerates endpoint generation in RMC. Building on RMC, Phase II presents RMC-based optimization, where RMC modules are composed to further enhance diversified robustness. To increase Phase II efficiency, we propose Efficient Robust Mode Connectivity (ERMC), which leverages $\ell_1$- and $\ell_\infty$-adversarially trained models to achieve robustness across a broad range of $p$-norms. An ensemble strategy is employed to further boost ERMC's performance. Extensive experiments across diverse datasets and architectures demonstrate that our methods significantly improve robustness against $\ell_\infty$, $\ell_2$, $\ell_1$, and hybrid attacks. Code is available at https://github.com/wangren09/MCGR.



## **16. Trust Me, I Know This Function: Hijacking LLM Static Analysis using Bias**

cs.LG

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17361v1) [paper-pdf](http://arxiv.org/pdf/2508.17361v1)

**Authors**: Shir Bernstein, David Beste, Daniel Ayzenshteyn, Lea Schonherr, Yisroel Mirsky

**Abstract**: Large Language Models (LLMs) are increasingly trusted to perform automated code review and static analysis at scale, supporting tasks such as vulnerability detection, summarization, and refactoring. In this paper, we identify and exploit a critical vulnerability in LLM-based code analysis: an abstraction bias that causes models to overgeneralize familiar programming patterns and overlook small, meaningful bugs. Adversaries can exploit this blind spot to hijack the control flow of the LLM's interpretation with minimal edits and without affecting actual runtime behavior. We refer to this attack as a Familiar Pattern Attack (FPA).   We develop a fully automated, black-box algorithm that discovers and injects FPAs into target code. Our evaluation shows that FPAs are not only effective, but also transferable across models (GPT-4o, Claude 3.5, Gemini 2.0) and universal across programming languages (Python, C, Rust, Go). Moreover, FPAs remain effective even when models are explicitly warned about the attack via robust system prompts. Finally, we explore positive, defensive uses of FPAs and discuss their broader implications for the reliability and safety of code-oriented LLMs.



## **17. Risk Assessment and Security Analysis of Large Language Models**

cs.CR

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17329v1) [paper-pdf](http://arxiv.org/pdf/2508.17329v1)

**Authors**: Xiaoyan Zhang, Dongyang Lyu, Xiaoqi Li

**Abstract**: As large language models (LLMs) expose systemic security challenges in high risk applications, including privacy leaks, bias amplification, and malicious abuse, there is an urgent need for a dynamic risk assessment and collaborative defence framework that covers their entire life cycle. This paper focuses on the security problems of large language models (LLMs) in critical application scenarios, such as the possibility of disclosure of user data, the deliberate input of harmful instructions, or the models bias. To solve these problems, we describe the design of a system for dynamic risk assessment and a hierarchical defence system that allows different levels of protection to cooperate. This paper presents a risk assessment system capable of evaluating both static and dynamic indicators simultaneously. It uses entropy weighting to calculate essential data, such as the frequency of sensitive words, whether the API call is typical, the realtime risk entropy value is significant, and the degree of context deviation. The experimental results show that the system is capable of identifying concealed attacks, such as role escape, and can perform rapid risk evaluation. The paper uses a hybrid model called BERT-CRF (Bidirectional Encoder Representation from Transformers) at the input layer to identify and filter malicious commands. The model layer uses dynamic adversarial training and differential privacy noise injection technology together. The output layer also has a neural watermarking system that can track the source of the content. In practice, the quality of this method, especially important in terms of customer service in the financial industry.



## **18. AdaGAT: Adaptive Guidance Adversarial Training for the Robustness of Deep Neural Networks**

cs.CV

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17265v1) [paper-pdf](http://arxiv.org/pdf/2508.17265v1)

**Authors**: Zhenyu Liu, Huizhi Liang, Xinrun Li, Vaclav Snasel, Varun Ojha

**Abstract**: Adversarial distillation (AD) is a knowledge distillation technique that facilitates the transfer of robustness from teacher deep neural network (DNN) models to lightweight target (student) DNN models, enabling the target models to perform better than only training the student model independently. Some previous works focus on using a small, learnable teacher (guide) model to improve the robustness of a student model. Since a learnable guide model starts learning from scratch, maintaining its optimal state for effective knowledge transfer during co-training is challenging. Therefore, we propose a novel Adaptive Guidance Adversarial Training (AdaGAT) method. Our method, AdaGAT, dynamically adjusts the training state of the guide model to install robustness to the target model. Specifically, we develop two separate loss functions as part of the AdaGAT method, allowing the guide model to participate more actively in backpropagation to achieve its optimal state. We evaluated our approach via extensive experiments on three datasets: CIFAR-10, CIFAR-100, and TinyImageNet, using the WideResNet-34-10 model as the target model. Our observations reveal that appropriately adjusting the guide model within a certain accuracy range enhances the target model's robustness across various adversarial attacks compared to a variety of baseline models.



## **19. Uncovering and Mitigating Destructive Multi-Embedding Attacks in Deepfake Proactive Forensics**

cs.CV

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17247v1) [paper-pdf](http://arxiv.org/pdf/2508.17247v1)

**Authors**: Lixin Jia, Haiyang Sun, Zhiqing Guo, Yunfeng Diao, Dan Ma, Gaobo Yang

**Abstract**: With the rapid evolution of deepfake technologies and the wide dissemination of digital media, personal privacy is facing increasingly serious security threats. Deepfake proactive forensics, which involves embedding imperceptible watermarks to enable reliable source tracking, serves as a crucial defense against these threats. Although existing methods show strong forensic ability, they rely on an idealized assumption of single watermark embedding, which proves impractical in real-world scenarios. In this paper, we formally define and demonstrate the existence of Multi-Embedding Attacks (MEA) for the first time. When a previously protected image undergoes additional rounds of watermark embedding, the original forensic watermark can be destroyed or removed, rendering the entire proactive forensic mechanism ineffective. To address this vulnerability, we propose a general training paradigm named Adversarial Interference Simulation (AIS). Rather than modifying the network architecture, AIS explicitly simulates MEA scenarios during fine-tuning and introduces a resilience-driven loss function to enforce the learning of sparse and stable watermark representations. Our method enables the model to maintain the ability to extract the original watermark correctly even after a second embedding. Extensive experiments demonstrate that our plug-and-play AIS training paradigm significantly enhances the robustness of various existing methods against MEA.



## **20. How to make Medical AI Systems safer? Simulating Vulnerabilities, and Threats in Multimodal Medical RAG System**

cs.LG

Sumbitted to 2025 AAAI main track

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17215v1) [paper-pdf](http://arxiv.org/pdf/2508.17215v1)

**Authors**: Kaiwen Zuo, Zelin Liu, Raman Dutt, Ziyang Wang, Zhongtian Sun, Yeming Wang, Fan Mo, Pietro Liò

**Abstract**: Large Vision-Language Models (LVLMs) augmented with Retrieval-Augmented Generation (RAG) are increasingly employed in medical AI to enhance factual grounding through external clinical image-text retrieval. However, this reliance creates a significant attack surface. We propose MedThreatRAG, a novel multimodal poisoning framework that systematically probes vulnerabilities in medical RAG systems by injecting adversarial image-text pairs. A key innovation of our approach is the construction of a simulated semi-open attack environment, mimicking real-world medical systems that permit periodic knowledge base updates via user or pipeline contributions. Within this setting, we introduce and emphasize Cross-Modal Conflict Injection (CMCI), which embeds subtle semantic contradictions between medical images and their paired reports. These mismatches degrade retrieval and generation by disrupting cross-modal alignment while remaining sufficiently plausible to evade conventional filters. While basic textual and visual attacks are included for completeness, CMCI demonstrates the most severe degradation. Evaluations on IU-Xray and MIMIC-CXR QA tasks show that MedThreatRAG reduces answer F1 scores by up to 27.66% and lowers LLaVA-Med-1.5 F1 rates to as low as 51.36%. Our findings expose fundamental security gaps in clinical RAG systems and highlight the urgent need for threat-aware design and robust multimodal consistency checks. Finally, we conclude with a concise set of guidelines to inform the safe development of future multimodal medical RAG systems.



## **21. Adversarial Illusions in Multi-Modal Embeddings**

cs.CR

In USENIX Security'24

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2308.11804v5) [paper-pdf](http://arxiv.org/pdf/2308.11804v5)

**Authors**: Tingwei Zhang, Rishi Jha, Eugene Bagdasaryan, Vitaly Shmatikov

**Abstract**: Multi-modal embeddings encode texts, images, thermal images, sounds, and videos into a single embedding space, aligning representations across different modalities (e.g., associate an image of a dog with a barking sound). In this paper, we show that multi-modal embeddings can be vulnerable to an attack we call "adversarial illusions." Given an image or a sound, an adversary can perturb it to make its embedding close to an arbitrary, adversary-chosen input in another modality.   These attacks are cross-modal and targeted: the adversary can align any image or sound with any target of his choice. Adversarial illusions exploit proximity in the embedding space and are thus agnostic to downstream tasks and modalities, enabling a wholesale compromise of current and future tasks, as well as modalities not available to the adversary. Using ImageBind and AudioCLIP embeddings, we demonstrate how adversarially aligned inputs, generated without knowledge of specific downstream tasks, mislead image generation, text generation, zero-shot classification, and audio retrieval.   We investigate transferability of illusions across different embeddings and develop a black-box version of our method that we use to demonstrate the first adversarial alignment attack on Amazon's commercial, proprietary Titan embedding. Finally, we analyze countermeasures and evasion attacks.



## **22. Sharpness-Aware Geometric Defense for Robust Out-Of-Distribution Detection**

cs.LG

under review

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17174v1) [paper-pdf](http://arxiv.org/pdf/2508.17174v1)

**Authors**: Jeng-Lin Li, Ming-Ching Chang, Wei-Chao Chen

**Abstract**: Out-of-distribution (OOD) detection ensures safe and reliable model deployment. Contemporary OOD algorithms using geometry projection can detect OOD or adversarial samples from clean in-distribution (ID) samples. However, this setting regards adversarial ID samples as OOD, leading to incorrect OOD predictions. Existing efforts on OOD detection with ID and OOD data under attacks are minimal. In this paper, we develop a robust OOD detection method that distinguishes adversarial ID samples from OOD ones. The sharp loss landscape created by adversarial training hinders model convergence, impacting the latent embedding quality for OOD score calculation. Therefore, we introduce a {\bf Sharpness-aware Geometric Defense (SaGD)} framework to smooth out the rugged adversarial loss landscape in the projected latent geometry. Enhanced geometric embedding convergence enables accurate ID data characterization, benefiting OOD detection against adversarial attacks. We use Jitter-based perturbation in adversarial training to extend the defense ability against unseen attacks. Our SaGD framework significantly improves FPR and AUC over the state-of-the-art defense approaches in differentiating CIFAR-100 from six other OOD datasets under various attacks. We further examine the effects of perturbations at various adversarial training levels, revealing the relationship between the sharp loss landscape and adversarial OOD detection.



## **23. Towards Safeguarding LLM Fine-tuning APIs against Cipher Attacks**

cs.LG

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.17158v1) [paper-pdf](http://arxiv.org/pdf/2508.17158v1)

**Authors**: Jack Youstra, Mohammed Mahfoud, Yang Yan, Henry Sleight, Ethan Perez, Mrinank Sharma

**Abstract**: Large language model fine-tuning APIs enable widespread model customization, yet pose significant safety risks. Recent work shows that adversaries can exploit access to these APIs to bypass model safety mechanisms by encoding harmful content in seemingly harmless fine-tuning data, evading both human monitoring and standard content filters. We formalize the fine-tuning API defense problem, and introduce the Cipher Fine-tuning Robustness benchmark (CIFR), a benchmark for evaluating defense strategies' ability to retain model safety in the face of cipher-enabled attackers while achieving the desired level of fine-tuning functionality. We include diverse cipher encodings and families, with some kept exclusively in the test set to evaluate for generalization across unseen ciphers and cipher families. We then evaluate different defenses on the benchmark and train probe monitors on model internal activations from multiple fine-tunes. We show that probe monitors achieve over 99% detection accuracy, generalize to unseen cipher variants and families, and compare favorably to state-of-the-art monitoring approaches. We open-source CIFR and the code to reproduce our experiments to facilitate further research in this critical area. Code and data are available online https://github.com/JackYoustra/safe-finetuning-api



## **24. ZAPS: A Zero-Knowledge Proof Protocol for Secure UAV Authentication with Flight Path Privacy**

cs.CR

11 Pages, 8 figures, Journal

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.17043v1) [paper-pdf](http://arxiv.org/pdf/2508.17043v1)

**Authors**: Shayesta Naziri, Xu Wang, Guangsheng Yu, Christy Jie Liang, Wei Ni

**Abstract**: The increasing deployment of Unmanned Aerial Vehicles (UAVs) for military, commercial, and logistics applications has raised significant concerns regarding flight path privacy. Conventional UAV communication systems often expose flight path data to third parties, making them vulnerable to tracking, surveillance, and location inference attacks. Existing encryption techniques provide security but fail to ensure complete privacy, as adversaries can still infer movement patterns through metadata analysis. To address these challenges, we propose a zk-SNARK(Zero-Knowledge Succinct Non-Interactive Argument of Knowledge)-based privacy-preserving flight path authentication and verification framework. Our approach ensures that a UAV can prove its authorisation, validate its flight path with a control centre, and comply with regulatory constraints without revealing any sensitive trajectory information. By leveraging zk-SNARKs, the UAV can generate cryptographic proofs that verify compliance with predefined flight policies while keeping the exact path and location undisclosed. This method mitigates risks associated with real-time tracking, identity exposure, and unauthorised interception, thereby enhancing UAV operational security in adversarial environments. Our proposed solution balances privacy, security, and computational efficiency, making it suitable for resource-constrained UAVs in both civilian and military applications.



## **25. Watermarking Visual Concepts for Diffusion Models**

cs.CR

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2411.11688v3) [paper-pdf](http://arxiv.org/pdf/2411.11688v3)

**Authors**: Liangqi Lei, Keke Gai, Jing Yu, Liehuang Zhu, Qi Wu

**Abstract**: The personalization techniques of diffusion models succeed in generating images with specific concepts. This ability also poses great threats to copyright protection and network security since malicious users can generate unauthorized content and disinformation relevant to a target concept. Model watermarking is an effective solution to trace the malicious generated images and safeguard their copyright. However, existing model watermarking techniques merely achieve image-level tracing without concept traceability. When tracing infringing or harmful concepts, current approaches execute image concept detection and model tracing sequentially, where performance is critically constrained by concept detection accuracy. In this paper, we propose a lightweight concept watermarking framework that efficiently binds target concepts to model watermarks, supporting simultaneous concept identification and model tracing via single-stage watermark verification. To further enhance the robustness of concept watermarking, we propose an adversarial perturbation injection method collaboratively embedded with watermarks during image generation, avoiding watermark removal by model purification attacks. Experimental results demonstrate that ConceptWM significantly outperforms state-of-the-art watermarking methods, improving detection accuracy by 6.3%-19.3% across diverse datasets including COCO and StableDiffusionDB. Additionally, ConceptWM possesses a critical capability absent in other watermarking methods: it sustains a 21.7% FID/CLIP degradation under adversarial fine-tuning of Stable Diffusion models on WikiArt and CelebA-HQ, demonstrating its capability to mitigate model misuse.



## **26. Unveiling the Latent Directions of Reflection in Large Language Models**

cs.LG

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.16989v1) [paper-pdf](http://arxiv.org/pdf/2508.16989v1)

**Authors**: Fu-Chieh Chang, Yu-Ting Lee, Pei-Yuan Wu

**Abstract**: Reflection, the ability of large language models (LLMs) to evaluate and revise their own reasoning, has been widely used to improve performance on complex reasoning tasks. Yet, most prior work emphasizes designing reflective prompting strategies or reinforcement learning objectives, leaving the inner mechanisms of reflection underexplored. In this paper, we investigate reflection through the lens of latent directions in model activations. We propose a methodology based on activation steering to characterize how instructions with different reflective intentions: no reflection, intrinsic reflection, and triggered reflection. By constructing steering vectors between these reflection levels, we demonstrate that (1) new reflection-inducing instructions can be systematically identified, (2) reflective behavior can be directly enhanced or suppressed through activation interventions, and (3) suppressing reflection is considerably easier than stimulating it. Experiments on GSM8k-adv with Qwen2.5-3B and Gemma3-4B reveal clear stratification across reflection levels, and steering interventions confirm the controllability of reflection. Our findings highlight both opportunities (e.g., reflection-enhancing defenses) and risks (e.g., adversarial inhibition of reflection in jailbreak attacks). This work opens a path toward mechanistic understanding of reflective reasoning in LLMs.



## **27. Effective Red-Teaming of Policy-Adherent Agents**

cs.MA

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2506.09600v3) [paper-pdf](http://arxiv.org/pdf/2506.09600v3)

**Authors**: Itay Nakash, George Kour, Koren Lazar, Matan Vetzler, Guy Uziel, Ateret Anaby-Tavor

**Abstract**: Task-oriented LLM-based agents are increasingly used in domains with strict policies, such as refund eligibility or cancellation rules. The challenge lies in ensuring that the agent consistently adheres to these rules and policies, appropriately refusing any request that would violate them, while still maintaining a helpful and natural interaction. This calls for the development of tailored design and evaluation methodologies to ensure agent resilience against malicious user behavior. We propose a novel threat model that focuses on adversarial users aiming to exploit policy-adherent agents for personal benefit. To address this, we present CRAFT, a multi-agent red-teaming system that leverages policy-aware persuasive strategies to undermine a policy-adherent agent in a customer-service scenario, outperforming conventional jailbreak methods such as DAN prompts, emotional manipulation, and coercive. Building upon the existing tau-bench benchmark, we introduce tau-break, a complementary benchmark designed to rigorously assess the agent's robustness against manipulative user behavior. Finally, we evaluate several straightforward yet effective defense strategies. While these measures provide some protection, they fall short, highlighting the need for stronger, research-driven safeguards to protect policy-adherent agents from adversarial attacks



## **28. NAT: Learning to Attack Neurons for Enhanced Adversarial Transferability**

cs.CV

Published at WACV 2025

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.16937v1) [paper-pdf](http://arxiv.org/pdf/2508.16937v1)

**Authors**: Krishna Kanth Nakka, Alexandre Alahi

**Abstract**: The generation of transferable adversarial perturbations typically involves training a generator to maximize embedding separation between clean and adversarial images at a single mid-layer of a source model. In this work, we build on this approach and introduce Neuron Attack for Transferability (NAT), a method designed to target specific neuron within the embedding. Our approach is motivated by the observation that previous layer-level optimizations often disproportionately focus on a few neurons representing similar concepts, leaving other neurons within the attacked layer minimally affected. NAT shifts the focus from embedding-level separation to a more fundamental, neuron-specific approach. We find that targeting individual neurons effectively disrupts the core units of the neural network, providing a common basis for transferability across different models. Through extensive experiments on 41 diverse ImageNet models and 9 fine-grained models, NAT achieves fooling rates that surpass existing baselines by over 14\% in cross-model and 4\% in cross-domain settings. Furthermore, by leveraging the complementary attacking capabilities of the trained generators, we achieve impressive fooling rates within just 10 queries. Our code is available at: https://krishnakanthnakka.github.io/NAT/



## **29. Mitigating Jailbreaks with Intent-Aware LLMs**

cs.CR

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.12072v2) [paper-pdf](http://arxiv.org/pdf/2508.12072v2)

**Authors**: Wei Jie Yeo, Ranjan Satapathy, Erik Cambria

**Abstract**: Despite extensive safety-tuning, large language models (LLMs) remain vulnerable to jailbreak attacks via adversarially crafted instructions, reflecting a persistent trade-off between safety and task performance. In this work, we propose Intent-FT, a simple and lightweight fine-tuning approach that explicitly trains LLMs to infer the underlying intent of an instruction before responding. By fine-tuning on a targeted set of adversarial instructions, Intent-FT enables LLMs to generalize intent deduction to unseen attacks, thereby substantially improving their robustness. We comprehensively evaluate both parametric and non-parametric attacks across open-source and proprietary models, considering harmfulness from attacks, utility, over-refusal, and impact against white-box threats. Empirically, Intent-FT consistently mitigates all evaluated attack categories, with no single attack exceeding a 50\% success rate -- whereas existing defenses remain only partially effective. Importantly, our method preserves the model's general capabilities and reduces excessive refusals on benign instructions containing superficially harmful keywords. Furthermore, models trained with Intent-FT accurately identify hidden harmful intent in adversarial attacks, and these learned intentions can be effectively transferred to enhance vanilla model defenses. We publicly release our code at https://github.com/wj210/Intent_Jailbreak.



## **30. Two-Level Priority Coding for Resilience to Arbitrary Blockage Patterns**

cs.IT

Extended version of the paper accepted at IEEE Military  Communications Conference (MILCOM), 2025

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.16899v1) [paper-pdf](http://arxiv.org/pdf/2508.16899v1)

**Authors**: Mine Gokce Dogan, Abhiram Kadiyala, Jaimin Shah, Martina Cardone, Christina Fragouli

**Abstract**: Ultra-reliable low-latency communication is essential in mission-critical settings, including military applications, where persistent and asymmetric link blockages caused by mobility, jamming, or adversarial attacks can disrupt delay-sensitive transmissions. This paper addresses this challenge by deploying a multilevel diversity coding (MDC) scheme that controls the received information, offers distinct reliability guarantees based on the priority of data streams, and maintains low design and operational complexity as the number of network paths increases. For two priority levels over three edge-disjoint paths, the complete capacity region is characterized, showing that superposition coding achieves the region in general, whereas network coding is required only in a specific corner case. Moreover, sufficient conditions under which a simple superposition coding scheme achieves the capacity for an arbitrary number of paths are identified. To prove these results and provide a unified analytical framework, the problem of designing high-performing MDC schemes is shown to be equivalent to the problem of designing high-performing encoding schemes over a class of broadcast networks, referred to as combination networks in the literature.



## **31. ImF: Implicit Fingerprint for Large Language Models**

cs.CL

13 pages, 6 figures

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2503.21805v3) [paper-pdf](http://arxiv.org/pdf/2503.21805v3)

**Authors**: Jiaxuan Wu, Wanli Peng, Hang Fu, Yiming Xue, Juan Wen

**Abstract**: Training large language models (LLMs) is resource-intensive and expensive, making protecting intellectual property (IP) for LLMs crucial. Recently, embedding fingerprints into LLMs has emerged as a prevalent method for establishing model ownership. However, existing fingerprinting techniques typically embed identifiable patterns with weak semantic coherence, resulting in fingerprints that significantly differ from the natural question-answering (QA) behavior inherent to LLMs. This discrepancy undermines the stealthiness of the embedded fingerprints and makes them vulnerable to adversarial attacks. In this paper, we first demonstrate the critical vulnerability of existing fingerprint embedding methods by introducing a novel adversarial attack named Generation Revision Intervention (GRI) attack. GRI attack exploits the semantic fragility of current fingerprinting methods, effectively erasing fingerprints by disrupting their weakly correlated semantic structures. Our empirical evaluation highlights that traditional fingerprinting approaches are significantly compromised by the GRI attack, revealing severe limitations in their robustness under realistic adversarial conditions. To advance the state-of-the-art in model fingerprinting, we propose a novel model fingerprint paradigm called Implicit Fingerprints (ImF). ImF leverages steganography techniques to subtly embed ownership information within natural texts, subsequently using Chain-of-Thought (CoT) prompting to construct semantically coherent and contextually natural QA pairs. This design ensures that fingerprints seamlessly integrate with the standard model behavior, remaining indistinguishable from regular outputs and substantially reducing the risk of accidental triggering and targeted removal. We conduct a comprehensive evaluation of ImF on 15 diverse LLMs, spanning different architectures and varying scales.



## **32. A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems**

cs.CR

This paper will be submitted to the Computer Science Review

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16843v1) [paper-pdf](http://arxiv.org/pdf/2508.16843v1)

**Authors**: Kamel Kamel, Keshav Sood, Hridoy Sankar Dutta, Sunil Aryal

**Abstract**: Voice authentication has undergone significant changes from traditional systems that relied on handcrafted acoustic features to deep learning models that can extract robust speaker embeddings. This advancement has expanded its applications across finance, smart devices, law enforcement, and beyond. However, as adoption has grown, so have the threats. This survey presents a comprehensive review of the modern threat landscape targeting Voice Authentication Systems (VAS) and Anti-Spoofing Countermeasures (CMs), including data poisoning, adversarial, deepfake, and adversarial spoofing attacks. We chronologically trace the development of voice authentication and examine how vulnerabilities have evolved in tandem with technological advancements. For each category of attack, we summarize methodologies, highlight commonly used datasets, compare performance and limitations, and organize existing literature using widely accepted taxonomies. By highlighting emerging risks and open challenges, this survey aims to support the development of more secure and resilient voice authentication systems.



## **33. DeMem: Privacy-Enhanced Robust Adversarial Learning via De-Memorization**

cs.LG

10 pages, Accepted at MLSP 2025

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2412.05767v3) [paper-pdf](http://arxiv.org/pdf/2412.05767v3)

**Authors**: Xiaoyu Luo, Qiongxiu Li

**Abstract**: Adversarial robustness, the ability of a model to withstand manipulated inputs that cause errors, is essential for ensuring the trustworthiness of machine learning models in real-world applications. However, previous studies have shown that enhancing adversarial robustness through adversarial training increases vulnerability to privacy attacks. While differential privacy can mitigate these attacks, it often compromises robustness against both natural and adversarial samples. Our analysis reveals that differential privacy disproportionately impacts low-risk samples, causing an unintended performance drop. To address this, we propose DeMem, which selectively targets high-risk samples, achieving a better balance between privacy protection and model robustness. DeMem is versatile and can be seamlessly integrated into various adversarial training techniques. Extensive evaluations across multiple training methods and datasets demonstrate that DeMem significantly reduces privacy leakage while maintaining robustness against both natural and adversarial samples. These results confirm DeMem's effectiveness and broad applicability in enhancing privacy without compromising robustness.



## **34. A Curious Case of Remarkable Resilience to Gradient Attacks via Fully Convolutional and Differentiable Front End with a Skip Connection**

cs.LG

Accepted at TMLR (2025/08)

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2402.17018v2) [paper-pdf](http://arxiv.org/pdf/2402.17018v2)

**Authors**: Leonid Boytsov, Ameya Joshi, Filipe Condessa

**Abstract**: We experimented with front-end enhanced neural models where a differentiable and fully convolutional model with a skip connection is added before a frozen backbone classifier. By training such composite models using a small learning rate for about one epoch, we obtained models that retained the accuracy of the backbone classifier while being unusually resistant to gradient attacks-including APGD and FAB-T attacks from the AutoAttack package-which we attribute to gradient masking. Although gradient masking is not new, the degree we observe is striking for fully differentiable models without obvious gradient-shattering-e.g., JPEG compression-or gradient-diminishing components.   The training recipe to produce such models is also remarkably stable and reproducible: We applied it to three datasets (CIFAR10, CIFAR100, and ImageNet) and several modern architectures (including vision Transformers) without a single failure case. While black-box attacks such as the SQUARE attack and zero-order PGD can partially overcome gradient masking, these attacks are easily defeated by simple randomized ensembles. We estimate that these ensembles achieve near-SOTA AutoAttack accuracy on CIFAR10, CIFAR100, and ImageNet (while retaining almost all clean accuracy of the original classifiers) despite having near-zero accuracy under adaptive attacks.   Adversarially training the backbone further amplifies this front-end "robustness". On CIFAR10, the respective randomized ensemble achieved 90.8$\pm 2.5\%$ (99\% CI) accuracy under the full AutoAttack while having only 18.2$\pm 3.6\%$ accuracy under the adaptive attack ($\varepsilon=8/255$, $L^\infty$ norm). We conclude the paper with a discussion of whether randomized ensembling can serve as a practical defense.   Code and instructions to reproduce key results are available. https://github.com/searchivarius/curious_case_of_gradient_masking



## **35. HAMSA: Hijacking Aligned Compact Models via Stealthy Automation**

cs.CL

9 pages, 1 figure; article under review

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16484v1) [paper-pdf](http://arxiv.org/pdf/2508.16484v1)

**Authors**: Alexey Krylov, Iskander Vagizov, Dmitrii Korzh, Maryam Douiba, Azidine Guezzaz, Vladimir Kokh, Sergey D. Erokhin, Elena V. Tutubalina, Oleg Y. Rogov

**Abstract**: Large Language Models (LLMs), especially their compact efficiency-oriented variants, remain susceptible to jailbreak attacks that can elicit harmful outputs despite extensive alignment efforts. Existing adversarial prompt generation techniques often rely on manual engineering or rudimentary obfuscation, producing low-quality or incoherent text that is easily flagged by perplexity-based filters. We present an automated red-teaming framework that evolves semantically meaningful and stealthy jailbreak prompts for aligned compact LLMs. The approach employs a multi-stage evolutionary search, where candidate prompts are iteratively refined using a population-based strategy augmented with temperature-controlled variability to balance exploration and coherence preservation. This enables the systematic discovery of prompts capable of bypassing alignment safeguards while maintaining natural language fluency. We evaluate our method on benchmarks in English (In-The-Wild Jailbreak Prompts on LLMs), and a newly curated Arabic one derived from In-The-Wild Jailbreak Prompts on LLMs and annotated by native Arabic linguists, enabling multilingual assessment.



## **36. Benchmarking the Robustness of Agentic Systems to Adversarially-Induced Harms**

cs.LG

52 Pages

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16481v1) [paper-pdf](http://arxiv.org/pdf/2508.16481v1)

**Authors**: Jonathan Nöther, Adish Singla, Goran Radanovic

**Abstract**: Ensuring the safe use of agentic systems requires a thorough understanding of the range of malicious behaviors these systems may exhibit when under attack. In this paper, we evaluate the robustness of LLM-based agentic systems against attacks that aim to elicit harmful actions from agents. To this end, we propose a novel taxonomy of harms for agentic systems and a novel benchmark, BAD-ACTS, for studying the security of agentic systems with respect to a wide range of harmful actions. BAD-ACTS consists of 4 implementations of agentic systems in distinct application environments, as well as a dataset of 188 high-quality examples of harmful actions. This enables a comprehensive study of the robustness of agentic systems across a wide range of categories of harmful behaviors, available tools, and inter-agent communication structures. Using this benchmark, we analyze the robustness of agentic systems against an attacker that controls one of the agents in the system and aims to manipulate other agents to execute a harmful target action. Our results show that the attack has a high success rate, demonstrating that even a single adversarial agent within the system can have a significant impact on the security. This attack remains effective even when agents use a simple prompting-based defense strategy. However, we additionally propose a more effective defense based on message monitoring. We believe that this benchmark provides a diverse testbed for the security research of agentic systems. The benchmark can be found at github.com/JNoether/BAD-ACTS



## **37. MCP-Guard: A Defense Framework for Model Context Protocol Integrity in Large Language Model Applications**

cs.CR

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.10991v2) [paper-pdf](http://arxiv.org/pdf/2508.10991v2)

**Authors**: Wenpeng Xing, Zhonghao Qi, Yupeng Qin, Yilin Li, Caini Chang, Jiahui Yu, Changting Lin, Zhenzhen Xie, Meng Han

**Abstract**: The integration of Large Language Models (LLMs) with external tools via protocols such as the Model Context Protocol (MCP) introduces critical security vulnerabilities, including prompt injection, data exfiltration, and other threats. To counter these challenges, we propose MCP-Guard, a robust, layered defense architecture designed for LLM--tool interactions. MCP-Guard employs a three-stage detection pipeline that balances efficiency with accuracy: it progresses from lightweight static scanning for overt threats and a deep neural detector for semantic attacks, to our fine-tuned E5-based model achieves (96.01) accuracy in identifying adversarial prompts. Finally, a lightweight LLM arbitrator synthesizes these signals to deliver the final decision while minimizing false positives. To facilitate rigorous training and evaluation, we also introduce MCP-AttackBench, a comprehensive benchmark of over 70,000 samples. Sourced from public datasets and augmented by GPT-4, MCP-AttackBench simulates diverse, real-world attack vectors in the MCP format, providing a foundation for future research into securing LLM-tool ecosystems.



## **38. PAR-AdvGAN: Improving Adversarial Attack Capability with Progressive Auto-Regression AdvGAN**

cs.LG

Best paper award of ECML-PKDD 2025

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2502.12207v4) [paper-pdf](http://arxiv.org/pdf/2502.12207v4)

**Authors**: Jiayu Zhang, Zhiyu Zhu, Xinyi Wang, Silin Liao, Zhibo Jin, Flora D. Salim, Huaming Chen

**Abstract**: Deep neural networks have demonstrated remarkable performance across various domains. However, they are vulnerable to adversarial examples, which can lead to erroneous predictions. Generative Adversarial Networks (GANs) can leverage the generators and discriminators model to quickly produce high-quality adversarial examples. Since both modules train in a competitive and simultaneous manner, GAN-based algorithms like AdvGAN can generate adversarial examples with better transferability compared to traditional methods. However, the generation of perturbations is usually limited to a single iteration, preventing these examples from fully exploiting the potential of the methods. To tackle this issue, we introduce a novel approach named Progressive Auto-Regression AdvGAN (PAR-AdvGAN). It incorporates an auto-regressive iteration mechanism within a progressive generation network to craft adversarial examples with enhanced attack capability. We thoroughly evaluate our PAR-AdvGAN method with a large-scale experiment, demonstrating its superior performance over various state-of-the-art black-box adversarial attacks, as well as the original AdvGAN.Moreover, PAR-AdvGAN significantly accelerates the adversarial example generation, i.e., achieving the speeds of up to 335.5 frames per second on Inception-v3 model, outperforming the gradient-based transferable attack algorithms. Our code is available at: https://github.com/LMBTough/PAR



## **39. from Benign import Toxic: Jailbreaking the Language Model via Adversarial Metaphors**

cs.CL

arXiv admin note: substantial text overlap with arXiv:2412.12145

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2503.00038v4) [paper-pdf](http://arxiv.org/pdf/2503.00038v4)

**Authors**: Yu Yan, Sheng Sun, Zenghao Duan, Teli Liu, Min Liu, Zhiyi Yin, Jiangyu Lei, Qi Li

**Abstract**: Current studies have exposed the risk of Large Language Models (LLMs) generating harmful content by jailbreak attacks. However, they overlook that the direct generation of harmful content from scratch is more difficult than inducing LLM to calibrate benign content into harmful forms. In our study, we introduce a novel attack framework that exploits AdVersArial meTAphoR (AVATAR) to induce the LLM to calibrate malicious metaphors for jailbreaking. Specifically, to answer harmful queries, AVATAR adaptively identifies a set of benign but logically related metaphors as the initial seed. Then, driven by these metaphors, the target LLM is induced to reason and calibrate about the metaphorical content, thus jailbroken by either directly outputting harmful responses or calibrating residuals between metaphorical and professional harmful content. Experimental results demonstrate that AVATAR can effectively and transferable jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs.



## **40. Robustness of deep learning classification to adversarial input on GPUs: asynchronous parallel accumulation is a source of vulnerability**

cs.LG

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2503.17173v2) [paper-pdf](http://arxiv.org/pdf/2503.17173v2)

**Authors**: Sanjif Shanmugavelu, Mathieu Taillefumier, Christopher Culver, Vijay Ganesh, Oscar Hernandez, Ada Sedova

**Abstract**: The ability of machine learning (ML) classification models to resist small, targeted input perturbations -- known as adversarial attacks -- is a key measure of their safety and reliability. We show that floating-point non-associativity (FPNA) coupled with asynchronous parallel programming on GPUs is sufficient to result in misclassification, without any perturbation to the input. Additionally, we show that standard adversarial robustness results may be overestimated up to 4.6 when not considering machine-level details. We develop a novel black-box attack using Bayesian optimization to discover external workloads that can change the instruction scheduling which bias the output of reductions on GPUs and reliably lead to misclassification. Motivated by these results, we present a new learnable permutation (LP) gradient-based approach to learning floating-point operation orderings that lead to misclassifications. The LP approach provides a worst-case estimate in a computationally efficient manner, avoiding the need to run identical experiments tens of thousands of times over a potentially large set of possible GPU states or architectures. Finally, using instrumentation-based testing, we investigate parallel reduction ordering across different GPU architectures under external background workloads, when utilizing multi-GPU virtualization, and when applying power capping. Our results demonstrate that parallel reduction ordering varies significantly across architectures under the first two conditions, substantially increasing the search space required to fully test the effects of this parallel scheduler-based vulnerability. These results and the methods developed here can help to include machine-level considerations into adversarial robustness assessments, which can make a difference in safety and mission critical applications.



## **41. An Investigation of Visual Foundation Models Robustness**

cs.CV

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16225v1) [paper-pdf](http://arxiv.org/pdf/2508.16225v1)

**Authors**: Sandeep Gupta, Roberto Passerone

**Abstract**: Visual Foundation Models (VFMs) are becoming ubiquitous in computer vision, powering systems for diverse tasks such as object detection, image classification, segmentation, pose estimation, and motion tracking. VFMs are capitalizing on seminal innovations in deep learning models, such as LeNet-5, AlexNet, ResNet, VGGNet, InceptionNet, DenseNet, YOLO, and ViT, to deliver superior performance across a range of critical computer vision applications. These include security-sensitive domains like biometric verification, autonomous vehicle perception, and medical image analysis, where robustness is essential to fostering trust between technology and the end-users. This article investigates network robustness requirements crucial in computer vision systems to adapt effectively to dynamic environments influenced by factors such as lighting, weather conditions, and sensor characteristics. We examine the prevalent empirical defenses and robust training employed to enhance vision network robustness against real-world challenges such as distributional shifts, noisy and spatially distorted inputs, and adversarial attacks. Subsequently, we provide a comprehensive analysis of the challenges associated with these defense mechanisms, including network properties and components to guide ablation studies and benchmarking metrics to evaluate network robustness.



## **42. PromptFlare: Prompt-Generalized Defense via Cross-Attention Decoy in Diffusion-Based Inpainting**

cs.CV

Accepted to ACM MM 2025

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16217v1) [paper-pdf](http://arxiv.org/pdf/2508.16217v1)

**Authors**: Hohyun Na, Seunghoo Hong, Simon S. Woo

**Abstract**: The success of diffusion models has enabled effortless, high-quality image modifications that precisely align with users' intentions, thereby raising concerns about their potential misuse by malicious actors. Previous studies have attempted to mitigate such misuse through adversarial attacks. However, these approaches heavily rely on image-level inconsistencies, which pose fundamental limitations in addressing the influence of textual prompts. In this paper, we propose PromptFlare, a novel adversarial protection method designed to protect images from malicious modifications facilitated by diffusion-based inpainting models. Our approach leverages the cross-attention mechanism to exploit the intrinsic properties of prompt embeddings. Specifically, we identify and target shared token of prompts that is invariant and semantically uninformative, injecting adversarial noise to suppress the sampling process. The injected noise acts as a cross-attention decoy, diverting the model's focus away from meaningful prompt-image alignments and thereby neutralizing the effect of prompt. Extensive experiments on the EditBench dataset demonstrate that our method achieves state-of-the-art performance across various metrics while significantly reducing computational overhead and GPU memory usage. These findings highlight PromptFlare as a robust and efficient protection against unauthorized image manipulations. The code is available at https://github.com/NAHOHYUN-SKKU/PromptFlare.



## **43. How to Beat Nakamoto in the Race**

cs.CR

Accepted for presentation at the 2025 ACM Conference on Computer and  Communications Security (CCS)

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16202v1) [paper-pdf](http://arxiv.org/pdf/2508.16202v1)

**Authors**: Shu-Jie Cao, Dongning Guo

**Abstract**: This paper studies proof-of-work Nakamoto consensus under bounded network delays, settling two long-standing questions in blockchain security: How can an adversary most effectively attack block safety under a given block confirmation latency? And what is the resulting probability of safety violation? A Markov decision process (MDP) framework is introduced to precise characterize the system state (including the tree and timings of all blocks mined), the adversary's potential actions, and the state transitions due to the adversarial action and the random block arrival processes. An optimal attack, called bait-and-switch, is proposed and proved to maximize the adversary's chance of violating block safety by "beating Nakamoto in the race". The exact probability of this violation is calculated for any confirmation depth using Markov chain analysis, offering fresh insights into the interplay of network delay, confirmation rules, and blockchain security.



## **44. Evaluating the Defense Potential of Machine Unlearning against Membership Inference Attacks**

cs.CR

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16150v1) [paper-pdf](http://arxiv.org/pdf/2508.16150v1)

**Authors**: Aristeidis Sidiropoulos, Christos Chrysanthos Nikolaidis, Theodoros Tsiolakis, Nikolaos Pavlidis, Vasilis Perifanis, Pavlos S. Efraimidis

**Abstract**: Membership Inference Attacks (MIAs) pose a significant privacy risk, as they enable adversaries to determine whether a specific data point was included in the training dataset of a model. While Machine Unlearning is primarily designed as a privacy mechanism to efficiently remove private data from a machine learning model without the need for full retraining, its impact on the susceptibility of models to MIA remains an open question. In this study, we systematically assess the vulnerability of models to MIA after applying state-of-art Machine Unlearning algorithms. Our analysis spans four diverse datasets (two from the image domain and two in tabular format), exploring how different unlearning approaches influence the exposure of models to membership inference. The findings highlight that while Machine Unlearning is not inherently a countermeasure against MIA, the unlearning algorithm and data characteristics can significantly affect a model's vulnerability. This work provides essential insights into the interplay between Machine Unlearning and MIAs, offering guidance for the design of privacy-preserving machine learning systems.



## **45. Robust Graph Contrastive Learning with Information Restoration**

cs.LG

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2307.12555v3) [paper-pdf](http://arxiv.org/pdf/2307.12555v3)

**Authors**: Yulin Zhu, Xing Ai, Yevgeniy Vorobeychik, Kai Zhou

**Abstract**: The graph contrastive learning (GCL) framework has gained remarkable achievements in graph representation learning. However, similar to graph neural networks (GNNs), GCL models are susceptible to graph structural attacks. As an unsupervised method, GCL faces greater challenges in defending against adversarial attacks. Furthermore, there has been limited research on enhancing the robustness of GCL. To thoroughly explore the failure of GCL on the poisoned graphs, we investigate the detrimental effects of graph structural attacks against the GCL framework. We discover that, in addition to the conventional observation that graph structural attacks tend to connect dissimilar node pairs, these attacks also diminish the mutual information between the graph and its representations from an information-theoretical perspective, which is the cornerstone of the high-quality node embeddings for GCL. Motivated by this theoretical insight, we propose a robust graph contrastive learning framework with a learnable sanitation view that endeavors to sanitize the augmented graphs by restoring the diminished mutual information caused by the structural attacks. Additionally, we design a fully unsupervised tuning strategy to tune the hyperparameters without accessing the label information, which strictly coincides with the defender's knowledge. Extensive experiments demonstrate the effectiveness and efficiency of our proposed method compared to competitive baselines.



## **46. Who's the Evil Twin? Differential Auditing for Undesired Behavior**

cs.LG

main section: 8 pages, 4 figures, 1 table total: 34 pages, 44  figures, 12 tables

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.06827v2) [paper-pdf](http://arxiv.org/pdf/2508.06827v2)

**Authors**: Ishwar Balappanawar, Venkata Hasith Vattikuti, Greta Kintzley, Ronan Azimi-Mancel, Satvik Golechha

**Abstract**: Detecting hidden behaviors in neural networks poses a significant challenge due to minimal prior knowledge and potential adversarial obfuscation. We explore this problem by framing detection as an adversarial game between two teams: the red team trains two similar models, one trained solely on benign data and the other trained on data containing hidden harmful behavior, with the performance of both being nearly indistinguishable on the benign dataset. The blue team, with limited to no information about the harmful behaviour, tries to identify the compromised model. We experiment using CNNs and try various blue team strategies, including Gaussian noise analysis, model diffing, integrated gradients, and adversarial attacks under different levels of hints provided by the red team. Results show high accuracy for adversarial-attack-based methods (100\% correct prediction, using hints), which is very promising, whilst the other techniques yield more varied performance. During our LLM-focused rounds, we find that there are not many parallel methods that we could apply from our study with CNNs. Instead, we find that effective LLM auditing methods require some hints about the undesired distribution, which can then used in standard black-box and open-weight methods to probe the models further and reveal their misalignment. We open-source our auditing games (with the model and data) and hope that our findings contribute to designing better audits.



## **47. Distributed Detection of Adversarial Attacks in Multi-Agent Reinforcement Learning with Continuous Action Space**

cs.LG

Accepted for publication at ECAI 2025

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15764v1) [paper-pdf](http://arxiv.org/pdf/2508.15764v1)

**Authors**: Kiarash Kazari, Ezzeldin Shereen, György Dán

**Abstract**: We address the problem of detecting adversarial attacks against cooperative multi-agent reinforcement learning with continuous action space. We propose a decentralized detector that relies solely on the local observations of the agents and makes use of a statistical characterization of the normal behavior of observable agents. The proposed detector utilizes deep neural networks to approximate the normal behavior of agents as parametric multivariate Gaussian distributions. Based on the predicted density functions, we define a normality score and provide a characterization of its mean and variance. This characterization allows us to employ a two-sided CUSUM procedure for detecting deviations of the normality score from its mean, serving as a detector of anomalous behavior in real-time. We evaluate our scheme on various multi-agent PettingZoo benchmarks against different state-of-the-art attack methods, and our results demonstrate the effectiveness of our method in detecting impactful adversarial attacks. Particularly, it outperforms the discrete counterpart by achieving AUC-ROC scores of over 0.95 against the most impactful attacks in all evaluated environments.



## **48. Let's Measure Information Step-by-Step: LLM-Based Evaluation Beyond Vibes**

cs.LG

Add AUC results, pre-reg conformance, theory section clarification.  12 pages

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.05469v2) [paper-pdf](http://arxiv.org/pdf/2508.05469v2)

**Authors**: Zachary Robertson, Sanmi Koyejo

**Abstract**: We study evaluation of AI systems without ground truth by exploiting a link between strategic gaming and information loss. We analyze which information-theoretic mechanisms resist adversarial manipulation, extending finite-sample bounds to show that bounded f-divergences (e.g., total variation distance) maintain polynomial guarantees under attacks while unbounded measures (e.g., KL divergence) degrade exponentially. To implement these mechanisms, we model the overseer as an agent and characterize incentive-compatible scoring rules as f-mutual information objectives. Under adversarial attacks, TVD-MI maintains effectiveness (area under curve 0.70-0.77) while traditional judge queries are near change (AUC $\approx$ 0.50), demonstrating that querying the same LLM for information relationships rather than quality judgments provides both theoretical and practical robustness. The mechanisms decompose pairwise evaluations into reliable item-level quality scores without ground truth, addressing a key limitation of traditional peer prediction. We release preregistration and code.



## **49. Towards a 3D Transfer-based Black-box Attack via Critical Feature Guidance**

cs.CV

11 pages, 6 figures

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15650v1) [paper-pdf](http://arxiv.org/pdf/2508.15650v1)

**Authors**: Shuchao Pang, Zhenghan Chen, Shen Zhang, Liming Lu, Siyuan Liang, Anan Du, Yongbin Zhou

**Abstract**: Deep neural networks for 3D point clouds have been demonstrated to be vulnerable to adversarial examples. Previous 3D adversarial attack methods often exploit certain information about the target models, such as model parameters or outputs, to generate adversarial point clouds. However, in realistic scenarios, it is challenging to obtain any information about the target models under conditions of absolute security. Therefore, we focus on transfer-based attacks, where generating adversarial point clouds does not require any information about the target models. Based on our observation that the critical features used for point cloud classification are consistent across different DNN architectures, we propose CFG, a novel transfer-based black-box attack method that improves the transferability of adversarial point clouds via the proposed Critical Feature Guidance. Specifically, our method regularizes the search of adversarial point clouds by computing the importance of the extracted features, prioritizing the corruption of critical features that are likely to be adopted by diverse architectures. Further, we explicitly constrain the maximum deviation extent of the generated adversarial point clouds in the loss function to ensure their imperceptibility. Extensive experiments conducted on the ModelNet40 and ScanObjectNN benchmark datasets demonstrate that the proposed CFG outperforms the state-of-the-art attack methods by a large margin.



## **50. Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks**

cs.CV

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2407.20836v4) [paper-pdf](http://arxiv.org/pdf/2407.20836v4)

**Authors**: Yunfeng Diao, Naixin Zhai, Changtao Miao, Zitong Yu, Xingxing Wei, Xun Yang, Meng Wang

**Abstract**: Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. To this end, we propose a new method to attack AIGI detectors. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous AIGI detectors, e.g. transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as Frequency-based Post-train Bayesian Attack, or FPBA. Through FPBA, we show that adversarial attack is truly a real threat to AIGI detectors, because FPBA can deliver successful black-box attacks across models, generators, defense methods, and even evade cross-generator detection, which is a crucial real-world detection scenario. The code will be shared upon acceptance.



