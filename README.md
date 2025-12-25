# Latest Adversarial Attack Papers
**update at 2025-12-25 10:11:04**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. CoTDeceptor:Adversarial Code Obfuscation Against CoT-Enhanced LLM Code Agents**

cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21250v1) [paper-pdf](https://arxiv.org/pdf/2512.21250v1)

**Authors**: Haoyang Li, Mingjin Li, Jinxin Zuo, Siqi Li, Xiao Li, Hao Wu, Yueming Lu, Xiaochuan He

**Abstract**: LLM-based code agents(e.g., ChatGPT Codex) are increasingly deployed as detector for code review and security auditing tasks. Although CoT-enhanced LLM vulnerability detectors are believed to provide improved robustness against obfuscated malicious code, we find that their reasoning chains and semantic abstraction processes exhibit exploitable systematic weaknesses.This allows attackers to covertly embed malicious logic, bypass code review, and propagate backdoored components throughout real-world software supply chains.To investigate this issue, we present CoTDeceptor, the first adversarial code obfuscation framework targeting CoT-enhanced LLM detectors. CoTDeceptor autonomously constructs evolving, hard-to-reverse multi-stage obfuscation strategy chains that effectively disrupt CoT-driven detection logic.We obtained malicious code provided by security enterprise, experimental results demonstrate that CoTDeceptor achieves stable and transferable evasion performance against state-of-the-art LLMs and vulnerability detection agents. CoTDeceptor bypasses 14 out of 15 vulnerability categories, compared to only 2 bypassed by prior methods. Our findings highlight potential risks in real-world software supply chains and underscore the need for more robust and interpretable LLM-powered security analysis systems.



## **2. Improving the Convergence Rate of Ray Search Optimization for Query-Efficient Hard-Label Attacks**

cs.LG

Published at AAAI 2026 (Oral). This version corresponds to the conference proceedings; v2 will include the appendix

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21241v1) [paper-pdf](https://arxiv.org/pdf/2512.21241v1)

**Authors**: Xinjie Xu, Shuyu Cheng, Dongwei Xu, Qi Xuan, Chen Ma

**Abstract**: In hard-label black-box adversarial attacks, where only the top-1 predicted label is accessible, the prohibitive query complexity poses a major obstacle to practical deployment. In this paper, we focus on optimizing a representative class of attacks that search for the optimal ray direction yielding the minimum $\ell_2$-norm perturbation required to move a benign image into the adversarial region. Inspired by Nesterov's Accelerated Gradient (NAG), we propose a momentum-based algorithm, ARS-OPT, which proactively estimates the gradient with respect to a future ray direction inferred from accumulated momentum. We provide a theoretical analysis of its convergence behavior, showing that ARS-OPT enables more accurate directional updates and achieves faster, more stable optimization. To further accelerate convergence, we incorporate surrogate-model priors into ARS-OPT's gradient estimation, resulting in PARS-OPT with enhanced performance. The superiority of our approach is supported by theoretical guarantees under standard assumptions. Extensive experiments on ImageNet and CIFAR-10 demonstrate that our method surpasses 13 state-of-the-art approaches in query efficiency.



## **3. Time-Bucketed Balance Records: Bounded-Storage Ephemeral Tokens for Resource-Constrained Systems**

cs.DS

14 pages, 1 figure, 1 Algorithm, 3 Theorems

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.20962v1) [paper-pdf](https://arxiv.org/pdf/2512.20962v1)

**Authors**: Shaun Scovil, Bhargav Chickmagalur Nanjundappa

**Abstract**: Fungible tokens with time-to-live (TTL) semantics require tracking individual expiration times for each deposited unit. A naive implementation creates a new balance record per deposit, leading to unbounded storage growth and vulnerability to denial-of-service attacks. We present time-bucketed balance records, a data structure that bounds storage to O(k) records per account while guaranteeing that tokens never expire before their configured TTL. Our approach discretizes time into k buckets, coalescing deposits within the same bucket to limit unique expiration timestamps. We prove three key properties: (1) storage is bounded by k+1 records regardless of deposit frequency, (2) actual expiration time is always at least the configured TTL, and (3) adversaries cannot increase a victim's operation cost beyond O(k)[amortized] worst case. We provide a reference implementation in Solidity with measured gas costs demonstrating practical efficiency.



## **4. Robustness Certificates for Neural Networks against Adversarial Attacks**

cs.LG

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.20865v1) [paper-pdf](https://arxiv.org/pdf/2512.20865v1)

**Authors**: Sara Taheri, Mahalakshmi Sabanayagam, Debarghya Ghoshdastidar, Majid Zamani

**Abstract**: The increasing use of machine learning in safety-critical domains amplifies the risk of adversarial threats, especially data poisoning attacks that corrupt training data to degrade performance or induce unsafe behavior. Most existing defenses lack formal guarantees or rely on restrictive assumptions about the model class, attack type, extent of poisoning, or point-wise certification, limiting their practical reliability. This paper introduces a principled formal robustness certification framework that models gradient-based training as a discrete-time dynamical system (dt-DS) and formulates poisoning robustness as a formal safety verification problem. By adapting the concept of barrier certificates (BCs) from control theory, we introduce sufficient conditions to certify a robust radius ensuring that the terminal model remains safe under worst-case ${\ell}_p$-norm based poisoning. To make this practical, we parameterize BCs as neural networks trained on finite sets of poisoned trajectories. We further derive probably approximately correct (PAC) bounds by solving a scenario convex program (SCP), which yields a confidence lower bound on the certified robustness radius generalizing beyond the training set. Importantly, our framework also extends to certification against test-time attacks, making it the first unified framework to provide formal guarantees in both training and test-time attack settings. Experiments on MNIST, SVHN, and CIFAR-10 show that our approach certifies non-trivial perturbation budgets while being model-agnostic and requiring no prior knowledge of the attack or contamination level.



## **5. Defending against adversarial attacks using mixture of experts**

cs.LG

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20821v1) [paper-pdf](https://arxiv.org/pdf/2512.20821v1)

**Authors**: Mohammad Meymani, Roozbeh Razavi-Far

**Abstract**: Machine learning is a powerful tool enabling full automation of a huge number of tasks without explicit programming. Despite recent progress of machine learning in different domains, these models have shown vulnerabilities when they are exposed to adversarial threats. Adversarial threats aim to hinder the machine learning models from satisfying their objectives. They can create adversarial perturbations, which are imperceptible to humans' eyes but have the ability to cause misclassification during inference. Moreover, they can poison the training data to harm the model's performance or they can query the model to steal its sensitive information. In this paper, we propose a defense system, which devises an adversarial training module within mixture-of-experts architecture to enhance its robustness against adversarial threats. In our proposed defense system, we use nine pre-trained experts with ResNet-18 as their backbone. During end-to-end training, the parameters of expert models and gating mechanism are jointly updated allowing further optimization of the experts. Our proposed defense system outperforms state-of-the-art defense systems and plain classifiers, which use a more complex architecture than our model's backbone.



## **6. Safety Alignment of LMs via Non-cooperative Games**

cs.AI

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20806v1) [paper-pdf](https://arxiv.org/pdf/2512.20806v1)

**Authors**: Anselm Paulus, Ilia Kulikov, Brandon Amos, Rémi Munos, Ivan Evtimov, Kamalika Chaudhuri, Arman Zharmagambetov

**Abstract**: Ensuring the safety of language models (LMs) while maintaining their usefulness remains a critical challenge in AI alignment. Current approaches rely on sequential adversarial training: generating adversarial prompts and fine-tuning LMs to defend against them. We introduce a different paradigm: framing safety alignment as a non-zero-sum game between an Attacker LM and a Defender LM trained jointly via online reinforcement learning. Each LM continuously adapts to the other's evolving strategies, driving iterative improvement. Our method uses a preference-based reward signal derived from pairwise comparisons instead of point-wise scores, providing more robust supervision and potentially reducing reward hacking. Our RL recipe, AdvGame, shifts the Pareto frontier of safety and utility, yielding a Defender LM that is simultaneously more helpful and more resilient to adversarial attacks. In addition, the resulting Attacker LM converges into a strong, general-purpose red-teaming agent that can be directly deployed to probe arbitrary target models.



## **7. Real-World Adversarial Attacks on RF-Based Drone Detectors**

cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20712v1) [paper-pdf](https://arxiv.org/pdf/2512.20712v1)

**Authors**: Omer Gazit, Yael Itzhakev, Yuval Elovici, Asaf Shabtai

**Abstract**: Radio frequency (RF) based systems are increasingly used to detect drones by analyzing their RF signal patterns, converting them into spectrogram images which are processed by object detection models. Existing RF attacks against image based models alter digital features, making over-the-air (OTA) implementation difficult due to the challenge of converting digital perturbations to transmittable waveforms that may introduce synchronization errors and interference, and encounter hardware limitations. We present the first physical attack on RF image based drone detectors, optimizing class-specific universal complex baseband (I/Q) perturbation waveforms that are transmitted alongside legitimate communications. We evaluated the attack using RF recordings and OTA experiments with four types of drones. Our results show that modest, structured I/Q perturbations are compatible with standard RF chains and reliably reduce target drone detection while preserving detection of legitimate drones.



## **8. Evasion-Resilient Detection of DNS-over-HTTPS Data Exfiltration: A Practical Evaluation and Toolkit**

cs.CR

61 pages Advisor : Dr Darren Hurley-Smith

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20423v1) [paper-pdf](https://arxiv.org/pdf/2512.20423v1)

**Authors**: Adam Elaoumari

**Abstract**: The purpose of this project is to assess how well defenders can detect DNS-over-HTTPS (DoH) file exfiltration, and which evasion strategies can be used by attackers. While providing a reproducible toolkit to generate, intercept and analyze DoH exfiltration, and comparing Machine Learning vs threshold-based detection under adversarial scenarios. The originality of this project is the introduction of an end-to-end, containerized pipeline that generates configurable file exfiltration over DoH using several parameters (e.g., chunking, encoding, padding, resolver rotation). It allows for file reconstruction at the resolver side, while extracting flow-level features using a fork of DoHLyzer. The pipeline contains a prediction side, which allows the training of machine learning models based on public labelled datasets and then evaluates them side-by-side with threshold-based detection methods against malicious and evasive DNS-Over-HTTPS traffic. We train Random Forest, Gradient Boosting and Logistic Regression classifiers on a public DoH dataset and benchmark them against evasive DoH exfiltration scenarios. The toolkit orchestrates traffic generation, file capture, feature extraction, model training and analysis. The toolkit is then encapsulated into several Docker containers for easy setup and full reproducibility regardless of the platform it is run on. Future research regarding this project is directed at validating the results on mixed enterprise traffic, extending the protocol coverage to HTTP/3/QUIC request, adding a benign traffic generation, and working on real-time traffic evaluation. A key objective is to quantify when stealth constraints make DoH exfiltration uneconomical and unworthy for the attacker.



## **9. Contingency Model-based Control (CMC) for Communicationless Cooperative Collision Avoidance in Robot Swarms**

math.OC

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20391v1) [paper-pdf](https://arxiv.org/pdf/2512.20391v1)

**Authors**: Georg Schildbach

**Abstract**: Cooperative collision avoidance between robots in swarm operations remains an open challenge. Assuming a decentralized architecture, each robot is responsible for making its own control decisions, including motion planning. To this end, most existing approaches mostly rely some form of (wireless) communication between the agents of the swarm. In reality, however, communication is brittle. It may be affected by latency, further delays and packet losses, transmission faults, and is subject to adversarial attacks, such as jamming or spoofing. This paper proposes Contingency Model-based Control (CMC) as a communicationless alternative. It follows the implicit cooperation paradigm, under which the design of the robots is based on consensual (offline) rules, similar to traffic rules. They include the definition of a contingency trajectory for each robot, and a method for construction of mutual collision avoidance constraints. The setup is shown to guarantee the recursive feasibility and collision avoidance between all swarm members in closed-loop operation. Moreover, CMC naturally satisfies the Plug \& Play paradigm, i.e., for new robots entering the swarm. Two numerical examples demonstrate that the collision avoidance guarantee is intact and that the robot swarm operates smoothly under the CMC regime.



## **10. Optimistic TEE-Rollups: A Hybrid Architecture for Scalable and Verifiable Generative AI Inference on Blockchain**

cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20176v1) [paper-pdf](https://arxiv.org/pdf/2512.20176v1)

**Authors**: Aaron Chan, Alex Ding, Frank Chen, Alan Wu, Bruce Zhang, Arther Tian

**Abstract**: The rapid integration of Large Language Models (LLMs) into decentralized physical infrastructure networks (DePIN) is currently bottlenecked by the Verifiability Trilemma, which posits that a decentralized inference system cannot simultaneously achieve high computational integrity, low latency, and low cost. Existing cryptographic solutions, such as Zero-Knowledge Machine Learning (ZKML), suffer from superlinear proving overheads (O(k NlogN)) that render them infeasible for billionparameter models. Conversely, optimistic approaches (opML) impose prohibitive dispute windows, preventing real-time interactivity, while recent "Proof of Quality" (PoQ) paradigms sacrifice cryptographic integrity for subjective semantic evaluation, leaving networks vulnerable to model downgrade attacks and reward hacking. In this paper, we introduce Optimistic TEE-Rollups (OTR), a hybrid verification protocol that harmonizes these constraints. OTR leverages NVIDIA H100 Confidential Computing Trusted Execution Environments (TEEs) to provide sub-second Provisional Finality, underpinned by an optimistic fraud-proof mechanism and stochastic Zero-Knowledge spot-checks to mitigate hardware side-channel risks. We formally define Proof of Efficient Attribution (PoEA), a consensus mechanism that cryptographically binds execution traces to hardware attestations, thereby guaranteeing model authenticity. Extensive simulations demonstrate that OTR achieves 99% of the throughput of centralized baselines with a marginal cost overhead of $0.07 per query, maintaining Byzantine fault tolerance against rational adversaries even in the presence of transient hardware vulnerabilities.



## **11. Odysseus: Jailbreaking Commercial Multimodal LLM-integrated Systems via Dual Steganography**

cs.CR

This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20168v1) [paper-pdf](https://arxiv.org/pdf/2512.20168v1)

**Authors**: Songze Li, Jiameng Cheng, Yiming Li, Xiaojun Jia, Dacheng Tao

**Abstract**: By integrating language understanding with perceptual modalities such as images, multimodal large language models (MLLMs) constitute a critical substrate for modern AI systems, particularly intelligent agents operating in open and interactive environments. However, their increasing accessibility also raises heightened risks of misuse, such as generating harmful or unsafe content. To mitigate these risks, alignment techniques are commonly applied to align model behavior with human values. Despite these efforts, recent studies have shown that jailbreak attacks can circumvent alignment and elicit unsafe outputs. Currently, most existing jailbreak methods are tailored for open-source models and exhibit limited effectiveness against commercial MLLM-integrated systems, which often employ additional filters. These filters can detect and prevent malicious input and output content, significantly reducing jailbreak threats. In this paper, we reveal that the success of these safety filters heavily relies on a critical assumption that malicious content must be explicitly visible in either the input or the output. This assumption, while often valid for traditional LLM-integrated systems, breaks down in MLLM-integrated systems, where attackers can leverage multiple modalities to conceal adversarial intent, leading to a false sense of security in existing MLLM-integrated systems. To challenge this assumption, we propose Odysseus, a novel jailbreak paradigm that introduces dual steganography to covertly embed malicious queries and responses into benign-looking images. Extensive experiments on benchmark datasets demonstrate that our Odysseus successfully jailbreaks several pioneering and realistic MLLM-integrated systems, achieving up to 99% attack success rate. It exposes a fundamental blind spot in existing defenses, and calls for rethinking cross-modal security in MLLM-integrated systems.



## **12. AI Security Beyond Core Domains: Resume Screening as a Case Study of Adversarial Vulnerabilities in Specialized LLM Applications**

cs.CL

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20164v1) [paper-pdf](https://arxiv.org/pdf/2512.20164v1)

**Authors**: Honglin Mu, Jinghao Liu, Kaiyang Wan, Rui Xing, Xiuying Chen, Timothy Baldwin, Wanxiang Che

**Abstract**: Large Language Models (LLMs) excel at text comprehension and generation, making them ideal for automated tasks like code review and content moderation. However, our research identifies a vulnerability: LLMs can be manipulated by "adversarial instructions" hidden in input data, such as resumes or code, causing them to deviate from their intended task. Notably, while defenses may exist for mature domains such as code review, they are often absent in other common applications such as resume screening and peer review. This paper introduces a benchmark to assess this vulnerability in resume screening, revealing attack success rates exceeding 80% for certain attack types. We evaluate two defense mechanisms: prompt-based defenses achieve 10.1% attack reduction with 12.5% false rejection increase, while our proposed FIDS (Foreign Instruction Detection through Separation) using LoRA adaptation achieves 15.4% attack reduction with 10.4% false rejection increase. The combined approach provides 26.3% attack reduction, demonstrating that training-time defenses outperform inference-time mitigations in both security and utility preservation.



## **13. IoT-based Android Malware Detection Using Graph Neural Network With Adversarial Defense**

cs.CR

13 pages

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20004v1) [paper-pdf](https://arxiv.org/pdf/2512.20004v1)

**Authors**: Rahul Yumlembam, Biju Issac, Seibu Mary Jacob, Longzhi Yang

**Abstract**: Since the Internet of Things (IoT) is widely adopted using Android applications, detecting malicious Android apps is essential. In recent years, Android graph-based deep learning research has proposed many approaches to extract relationships from applications as graphs to generate graph embeddings. First, we demonstrate the effectiveness of graph-based classification using a Graph Neural Network (GNN)-based classifier to generate API graph embeddings. The graph embeddings are combined with Permission and Intent features to train multiple machine learning and deep learning models for Android malware detection. The proposed classification approach achieves an accuracy of 98.33 percent on the CICMaldroid dataset and 98.68 percent on the Drebin dataset. However, graph-based deep learning models are vulnerable, as attackers can add fake relationships to evade detection by the classifier. Second, we propose a Generative Adversarial Network (GAN)-based attack algorithm named VGAE-MalGAN targeting graph-based GNN Android malware classifiers. The VGAE-MalGAN generator produces adversarial malware API graphs, while the VGAE-MalGAN substitute detector attempts to mimic the target detector. Experimental results show that VGAE-MalGAN can significantly reduce the detection rate of GNN-based malware classifiers. Although the model initially fails to detect adversarial malware, retraining with generated adversarial samples improves robustness and helps mitigate adversarial attacks.



## **14. Conditional Adversarial Fragility in Financial Machine Learning under Macroeconomic Stress**

cs.LG

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19935v1) [paper-pdf](https://arxiv.org/pdf/2512.19935v1)

**Authors**: Samruddhi Baviskar

**Abstract**: Machine learning models used in financial decision systems operate in nonstationary economic environments, yet adversarial robustness is typically evaluated under static assumptions. This work introduces Conditional Adversarial Fragility, a regime dependent phenomenon in which adversarial vulnerability is systematically amplified during periods of macroeconomic stress. We propose a regime aware evaluation framework for time indexed tabular financial classification tasks that conditions robustness assessment on external indicators of economic stress. Using volatility based regime segmentation as a proxy for macroeconomic conditions, we evaluate model behavior across calm and stress periods while holding model architecture, attack methodology, and evaluation protocols constant. Baseline predictive performance remains comparable across regimes, indicating that economic stress alone does not induce inherent performance degradation. Under adversarial perturbations, however, models operating during stress regimes exhibit substantially greater degradation across predictive accuracy, operational decision thresholds, and risk sensitive outcomes. We further demonstrate that this amplification propagates to increased false negative rates, elevating the risk of missed high risk cases during adverse conditions. To complement numerical robustness metrics, we introduce an interpretive governance layer based on semantic auditing of model explanations using large language models. Together, these results demonstrate that adversarial robustness in financial machine learning is a regime dependent property and motivate stress aware approaches to model risk assessment in high stakes financial deployments.



## **15. Multi-Layer Confidence Scoring for Detection of Out-of-Distribution Samples, Adversarial Attacks, and In-Distribution Misclassifications**

cs.LG

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19472v1) [paper-pdf](https://arxiv.org/pdf/2512.19472v1)

**Authors**: Lorenzo Capelli, Leandro de Souza Rosa, Gianluca Setti, Mauro Mangia, Riccardo Rovatti

**Abstract**: The recent explosive growth in Deep Neural Networks applications raises concerns about the black-box usage of such models, with limited trasparency and trustworthiness in high-stakes domains, which have been crystallized as regulatory requirements such as the European Union Artificial Intelligence Act. While models with embedded confidence metrics have been proposed, such approaches cannot be applied to already existing models without retraining, limiting their broad application. On the other hand, post-hoc methods, which evaluate pre-trained models, focus on solving problems related to improving the confidence in the model's predictions, and detecting Out-Of-Distribution or Adversarial Attacks samples as independent applications. To tackle the limited applicability of already existing methods, we introduce Multi-Layer Analysis for Confidence Scoring (MACS), a unified post-hoc framework that analyzes intermediate activations to produce classification-maps. From the classification-maps, we derive a score applicable for confidence estimation, detecting distributional shifts and adversarial attacks, unifying the three problems in a common framework, and achieving performances that surpass the state-of-the-art approaches in our experiments with the VGG16 and ViTb16 models with a fraction of their computational overhead.



## **16. SafeMed-R1: Adversarial Reinforcement Learning for Generalizable and Robust Medical Reasoning in Vision-Language Models**

cs.AI

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19317v1) [paper-pdf](https://arxiv.org/pdf/2512.19317v1)

**Authors**: A. A. Gde Yogi Pramana, Jason Ray, Anthony Jaya, Michael Wijaya

**Abstract**: Vision--Language Models (VLMs) show significant promise for Medical Visual Question Answering (VQA), yet their deployment in clinical settings is hindered by severe vulnerability to adversarial attacks. Standard adversarial training, while effective for simpler tasks, often degrades both generalization performance and the quality of generated clinical reasoning. We introduce SafeMed-R1, a hybrid defense framework that ensures robust performance while preserving high-quality, interpretable medical reasoning. SafeMed-R1 employs a two-stage approach: at training time, we integrate Adversarial Training with Group Relative Policy Optimization (AT-GRPO) to explicitly robustify the reasoning process against worst-case perturbations; at inference time, we augment the model with Randomized Smoothing to provide certified $L_2$-norm robustness guarantees. We evaluate SafeMed-R1 on the OmniMedVQA benchmark across eight medical imaging modalities comprising over 88,000 samples. Our experiments reveal that standard fine-tuned VLMs, despite achieving 95\% accuracy on clean inputs, collapse to approximately 25\% under PGD attacks. In contrast, SafeMed-R1 maintains 84.45\% accuracy under the same adversarial conditions, representing a 59 percentage point improvement in robustness. Furthermore, we demonstrate that models trained with explicit chain-of-thought reasoning exhibit superior adversarial robustness compared to instruction-only variants, suggesting a synergy between interpretability and security in medical AI systems.



## **17. GShield: Mitigating Poisoning Attacks in Federated Learning**

cs.CR

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19286v1) [paper-pdf](https://arxiv.org/pdf/2512.19286v1)

**Authors**: Sameera K. M., Serena Nicolazzo, Antonino Nocera, Vinod P., Rafidha Rehiman K. A

**Abstract**: Federated Learning (FL) has recently emerged as a revolutionary approach to collaborative training Machine Learning models. In particular, it enables decentralized model training while preserving data privacy, but its distributed nature makes it highly vulnerable to a severe attack known as Data Poisoning. In such scenarios, malicious clients inject manipulated data into the training process, thereby degrading global model performance or causing targeted misclassification. In this paper, we present a novel defense mechanism called GShield, designed to detect and mitigate malicious and low-quality updates, especially under non-independent and identically distributed (non-IID) data scenarios. GShield operates by learning the distribution of benign gradients through clustering and Gaussian modeling during an initial round, enabling it to establish a reliable baseline of trusted client behavior. With this benign profile, GShield selectively aggregates only those updates that align with the expected gradient patterns, effectively isolating adversarial clients and preserving the integrity of the global model. An extensive experimental campaign demonstrates that our proposed defense significantly improves model robustness compared to the state-of-the-art methods while maintaining a high accuracy of performance across both tabular and image datasets. Furthermore, GShield improves the accuracy of the targeted class by 43\% to 65\% after detecting malicious and low-quality clients.



## **18. Elevating Intrusion Detection and Security Fortification in Intelligent Networks through Cutting-Edge Machine Learning Paradigms**

cs.CR

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19037v1) [paper-pdf](https://arxiv.org/pdf/2512.19037v1)

**Authors**: Md Minhazul Islam Munna, Md Mahbubur Rahman, Jaroslav Frnda, Muhammad Shahid Anwar, Alpamis Kutlimuratov

**Abstract**: The proliferation of IoT devices and their reliance on Wi-Fi networks have introduced significant security vulnerabilities, particularly the KRACK and Kr00k attacks, which exploit weaknesses in WPA2 encryption to intercept and manipulate sensitive data. Traditional IDS using classifiers face challenges such as model overfitting, incomplete feature extraction, and high false positive rates, limiting their effectiveness in real-world deployments. To address these challenges, this study proposes a robust multiclass machine learning based intrusion detection framework. The methodology integrates advanced feature selection techniques to identify critical attributes, mitigating redundancy and enhancing detection accuracy. Two distinct ML architectures are implemented: a baseline classifier pipeline and a stacked ensemble model combining noise injection, Principal Component Analysis (PCA), and meta learning to improve generalization and reduce false positives. Evaluated on the AWID3 data set, the proposed ensemble architecture achieves superior performance, with an accuracy of 98%, precision of 98%, recall of 98%, and a false positive rate of just 2%, outperforming existing state-of-the-art methods. This work demonstrates the efficacy of combining preprocessing strategies with ensemble learning to fortify network security against sophisticated Wi-Fi attacks, offering a scalable and reliable solution for IoT environments. Future directions include real-time deployment and adversarial resilience testing to further enhance the model's adaptability.



## **19. DREAM: Dynamic Red-teaming across Environments for AI Models**

cs.CR

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19016v1) [paper-pdf](https://arxiv.org/pdf/2512.19016v1)

**Authors**: Liming Lu, Xiang Gu, Junyu Huang, Jiawei Du, Yunhuai Liu, Yongbin Zhou, Shuchao Pang

**Abstract**: Large Language Models (LLMs) are increasingly used in agentic systems, where their interactions with diverse tools and environments create complex, multi-stage safety challenges. However, existing benchmarks mostly rely on static, single-turn assessments that miss vulnerabilities from adaptive, long-chain attacks. To fill this gap, we introduce DREAM, a framework for systematic evaluation of LLM agents against dynamic, multi-stage attacks. At its core, DREAM uses a Cross-Environment Adversarial Knowledge Graph (CE-AKG) to maintain stateful, cross-domain understanding of vulnerabilities. This graph guides a Contextualized Guided Policy Search (C-GPS) algorithm that dynamically constructs attack chains from a knowledge base of 1,986 atomic actions across 349 distinct digital environments. Our evaluation of 12 leading LLM agents reveals a critical vulnerability: these attack chains succeed in over 70% of cases for most models, showing the power of stateful, cross-environment exploits. Through analysis of these failures, we identify two key weaknesses in current agents: contextual fragility, where safety behaviors fail to transfer across environments, and an inability to track long-term malicious intent. Our findings also show that traditional safety measures, such as initial defense prompts, are largely ineffective against attacks that build context over multiple interactions. To advance agent safety research, we release DREAM as a tool for evaluating vulnerabilities and developing more robust defenses.



## **20. Automated Red-Teaming Framework for Large Language Model Security Assessment: A Comprehensive Attack Generation and Detection System**

cs.CR

18 pages

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.20677v1) [paper-pdf](https://arxiv.org/pdf/2512.20677v1)

**Authors**: Zhang Wei, Peilu Hu, Shengning Lang, Hao Yan, Li Mei, Yichao Zhang, Chen Yang, Junfeng Hao, Zhimo Han

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes domains, ensuring their security and alignment has become a critical challenge. Existing red-teaming practices depend heavily on manual testing, which limits scalability and fails to comprehensively cover the vast space of potential adversarial behaviors. This paper introduces an automated red-teaming framework that systematically generates, executes, and evaluates adversarial prompts to uncover security vulnerabilities in LLMs. Our framework integrates meta-prompting-based attack synthesis, multi-modal vulnerability detection, and standardized evaluation protocols spanning six major threat categories -- reward hacking, deceptive alignment, data exfiltration, sandbagging, inappropriate tool use, and chain-of-thought manipulation. Experiments on the GPT-OSS-20B model reveal 47 distinct vulnerabilities, including 21 high-severity and 12 novel attack patterns, achieving a $3.9\times$ improvement in vulnerability discovery rate over manual expert testing while maintaining 89\% detection accuracy. These results demonstrate the framework's effectiveness in enabling scalable, systematic, and reproducible AI safety evaluations. By providing actionable insights for improving alignment robustness, this work advances the state of automated LLM red-teaming and contributes to the broader goal of building secure and trustworthy AI systems.



## **21. ISADM: An Integrated STRIDE, ATT&CK, and D3FEND Model for Threat Modeling Against Real-world Adversaries**

cs.CR

34 pages

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18751v1) [paper-pdf](https://arxiv.org/pdf/2512.18751v1)

**Authors**: Khondokar Fida Hasan, Hasibul Hossain Shajeeb, Chathura Abeydeera, Benjamin Turnbull, Matthew Warren

**Abstract**: FinTechs increasing connectivity, rapid innovation, and reliance on global digital infrastructures present significant cybersecurity challenges. Traditional cybersecurity frameworks often struggle to identify and prioritize sector-specific vulnerabilities or adapt to evolving adversary tactics, particularly in highly targeted sectors such as FinTech. To address these gaps, we propose ISADM (Integrated STRIDE-ATTACK-D3FEND Threat Model), a novel hybrid methodology applied to FinTech security that integrates STRIDE's asset-centric threat classification with MITRE ATTACK's catalog of real-world adversary behaviors and D3FEND's structured knowledge of countermeasures. ISADM employs a frequency-based scoring mechanism to quantify the prevalence of adversarial Tactics, Techniques, and Procedures (TTPs), enabling a proactive, score-driven risk assessment and prioritization framework. This proactive approach contributes to shifting organizations from reactive defense strategies toward the strategic fortification of critical assets. We validate ISADM through industry-relevant case study analyses, demonstrating how the approach replicates actual attack patterns and strengthens proactive threat modeling, guiding risk prioritization and resource allocation to the most critical vulnerabilities. Overall, ISADM offers a comprehensive hybrid threat modeling methodology that bridges asset-centric and adversary-centric analysis, providing FinTech systems with stronger defenses. The emphasis on real-world validation highlights its practical significance in enhancing the sector's cybersecurity posture through a frequency-informed, impact-aware prioritization scheme that combines empirical attacker data with contextual risk analysis.



## **22. Measuring the Impact of Student Gaming Behaviors on Learner Modeling**

cs.CY

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18659v1) [paper-pdf](https://arxiv.org/pdf/2512.18659v1)

**Authors**: Qinyi Liu, Lin Li, Valdemar Švábenský, Conrad Borchers, Mohammad Khalil

**Abstract**: The expansion of large-scale online education platforms has made vast amounts of student interaction data available for knowledge tracing (KT). KT models estimate students' concept mastery from interaction data, but their performance is sensitive to input data quality. Gaming behaviors, such as excessive hint use, may misrepresent students' knowledge and undermine model reliability. However, systematic investigations of how different types of gaming behaviors affect KT remain scarce, and existing studies rely on costly manual analysis that does not capture behavioral diversity. In this study, we conceptualize gaming behaviors as a form of data poisoning, defined as the deliberate submission of incorrect or misleading interaction data to corrupt a model's learning process. We design Data Poisoning Attacks (DPAs) to simulate diverse gaming patterns and systematically evaluate their impact on KT model performance. Moreover, drawing on advances in DPA detection, we explore unsupervised approaches to enhance the generalizability of gaming behavior detection. We find that KT models' performance tends to decrease especially in response to random guess behaviors. Our findings provide insights into the vulnerabilities of KT models and highlight the potential of adversarial methods for improving the robustness of learning analytics systems.



## **23. Adversarial Robustness in Zero-Shot Learning:An Empirical Study on Class and Concept-Level Vulnerabilities**

cs.CV

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18651v1) [paper-pdf](https://arxiv.org/pdf/2512.18651v1)

**Authors**: Zhiyuan Peng, Zihan Ye, Shreyank N Gowda, Yuping Yan, Haotian Xu, Ling Shao

**Abstract**: Zero-shot Learning (ZSL) aims to enable image classifiers to recognize images from unseen classes that were not included during training. Unlike traditional supervised classification, ZSL typically relies on learning a mapping from visual features to predefined, human-understandable class concepts. While ZSL models promise to improve generalization and interpretability, their robustness under systematic input perturbations remain unclear. In this study, we present an empirical analysis about the robustness of existing ZSL methods at both classlevel and concept-level. Specifically, we successfully disrupted their class prediction by the well-known non-target class attack (clsA). However, in the Generalized Zero-shot Learning (GZSL) setting, we observe that the success of clsA is only at the original best-calibrated point. After the attack, the optimal bestcalibration point shifts, and ZSL models maintain relatively strong performance at other calibration points, indicating that clsA results in a spurious attack success in the GZSL. To address this, we propose the Class-Bias Enhanced Attack (CBEA), which completely eliminates GZSL accuracy across all calibrated points by enhancing the gap between seen and unseen class probabilities.Next, at concept-level attack, we introduce two novel attack modes: Class-Preserving Concept Attack (CPconA) and NonClass-Preserving Concept Attack (NCPconA). Our extensive experiments evaluate three typical ZSL models across various architectures from the past three years and reveal that ZSL models are vulnerable not only to the traditional class attack but also to concept-based attacks. These attacks allow malicious actors to easily manipulate class predictions by erasing or introducing concepts. Our findings highlight a significant performance gap between existing approaches, emphasizing the need for improved adversarial robustness in current ZSL models.



## **24. DASH: Deception-Augmented Shared Mental Model for a Human-Machine Teaming System**

cs.HC

17 pages, 16 figures

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18616v1) [paper-pdf](https://arxiv.org/pdf/2512.18616v1)

**Authors**: Zelin Wan, Han Jun Yoon, Nithin Alluru, Terrence J. Moore, Frederica F. Nelson, Seunghyun Yoon, Hyuk Lim, Dan Dongseong Kim, Jin-Hee Cho

**Abstract**: We present DASH (Deception-Augmented Shared mental model for Human-machine teaming), a novel framework that enhances mission resilience by embedding proactive deception into Shared Mental Models (SMM). Designed for mission-critical applications such as surveillance and rescue, DASH introduces "bait tasks" to detect insider threats, e.g., compromised Unmanned Ground Vehicles (UGVs), AI agents, or human analysts, before they degrade team performance. Upon detection, tailored recovery mechanisms are activated, including UGV system reinstallation, AI model retraining, or human analyst replacement. In contrast to existing SMM approaches that neglect insider risks, DASH improves both coordination and security. Empirical evaluations across four schemes (DASH, SMM-only, no-SMM, and baseline) show that DASH sustains approximately 80% mission success under high attack rates, eight times higher than the baseline. This work contributes a practical human-AI teaming framework grounded in shared mental models, a deception-based strategy for insider threat detection, and empirical evidence of enhanced robustness under adversarial conditions. DASH establishes a foundation for secure, adaptive human-machine teaming in contested environments.



## **25. Detection of AI Generated Images Using Combined Uncertainty Measures and Particle Swarm Optimised Rejection Mechanism**

cs.CV

Scientific Reports (2025)

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2512.18527v1) [paper-pdf](https://arxiv.org/pdf/2512.18527v1)

**Authors**: Rahul Yumlembam, Biju Issac, Nauman Aslam, Eaby Kollonoor Babu, Josh Collyer, Fraser Kennedy

**Abstract**: As AI-generated images become increasingly photorealistic, distinguishing them from natural images poses a growing challenge. This paper presents a robust detection framework that leverages multiple uncertainty measures to decide whether to trust or reject a model's predictions. We focus on three complementary techniques: Fisher Information, which captures the sensitivity of model parameters to input variations; entropy-based uncertainty from Monte Carlo Dropout, which reflects predictive variability; and predictive variance from a Deep Kernel Learning framework using a Gaussian Process classifier. To integrate these diverse uncertainty signals, Particle Swarm Optimisation is used to learn optimal weightings and determine an adaptive rejection threshold. The model is trained on Stable Diffusion-generated images and evaluated on GLIDE, VQDM, Midjourney, BigGAN, and StyleGAN3, each introducing significant distribution shifts. While standard metrics such as prediction probability and Fisher-based measures perform well in distribution, their effectiveness degrades under shift. In contrast, the Combined Uncertainty measure consistently achieves an incorrect rejection rate of approximately 70 percent on unseen generators, successfully filtering most misclassified AI samples. Although the system occasionally rejects correct predictions from newer generators, this conservative behaviour is acceptable, as rejected samples can support retraining. The framework maintains high acceptance of accurate predictions for natural images and in-domain AI data. Under adversarial attacks using FGSM and PGD, the Combined Uncertainty method rejects around 61 percent of successful attacks, while GP-based uncertainty alone achieves up to 80 percent. Overall, the results demonstrate that multi-source uncertainty fusion provides a resilient and adaptive solution for AI-generated image detection.



## **26. SoK: Understanding (New) Security Issues Across AI4Code Use Cases**

cs.CR

39 pages, 19 figures

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2512.18456v1) [paper-pdf](https://arxiv.org/pdf/2512.18456v1)

**Authors**: Qilong Wu, Taoran Li, Tianyang Zhou, Varun Chandrasekaran

**Abstract**: AI-for-Code (AI4Code) systems are reshaping software engineering, with tools like GitHub Copilot accelerating code generation, translation, and vulnerability detection. Alongside these advances, however, security risks remain pervasive: insecure outputs, biased benchmarks, and susceptibility to adversarial manipulation undermine their reliability. This SoK surveys the landscape of AI4Code security across three core applications, identifying recurring gaps: benchmark dominance by Python and toy problems, lack of standardized security datasets, data leakage in evaluation, and fragile adversarial robustness. A comparative study of six state-of-the-art models illustrates these challenges: insecure patterns persist in code generation, vulnerability detection is brittle to semantic-preserving attacks, fine-tuning often misaligns security objectives, and code translation yields uneven security benefits. From this analysis, we distill three forward paths: embedding secure-by-default practices in code generation, building robust and comprehensive detection benchmarks, and leveraging translation as a route to security-enhanced languages. We call for a shift toward security-first AI4Code, where vulnerability mitigation and robustness are embedded throughout the development life cycle.



## **27. Who Can See Through You? Adversarial Shielding Against VLM-Based Attribute Inference Attacks**

cs.CV

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2512.18264v1) [paper-pdf](https://arxiv.org/pdf/2512.18264v1)

**Authors**: Yucheng Fan, Jiawei Chen, Yu Tian, Zhaoxia Yin

**Abstract**: As vision-language models (VLMs) become widely adopted, VLM-based attribute inference attacks have emerged as a serious privacy concern, enabling adversaries to infer private attributes from images shared on social media. This escalating threat calls for dedicated protection methods to safeguard user privacy. However, existing methods often degrade the visual quality of images or interfere with vision-based functions on social media, thereby failing to achieve a desirable balance between privacy protection and user experience. To address this challenge, we propose a novel protection method that jointly optimizes privacy suppression and utility preservation under a visual consistency constraint. While our method is conceptually effective, fair comparisons between methods remain challenging due to the lack of publicly available evaluation datasets. To fill this gap, we introduce VPI-COCO, a publicly available benchmark comprising 522 images with hierarchically structured privacy questions and corresponding non-private counterparts, enabling fine-grained and joint evaluation of protection methods in terms of privacy preservation and user experience. Building upon this benchmark, experiments on multiple VLMs demonstrate that our method effectively reduces PAR below 25%, keeps NPAR above 88%, maintains high visual consistency, and generalizes well to unseen and paraphrased privacy questions, demonstrating its strong practical applicability for real-world VLM deployments.



## **28. Breaking Minds, Breaking Systems: Jailbreaking Large Language Models via Human-like Psychological Manipulation**

cs.CR

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2512.18244v1) [paper-pdf](https://arxiv.org/pdf/2512.18244v1)

**Authors**: Zehao Liu, Xi Lin

**Abstract**: Large Language Models (LLMs) have gained considerable popularity and protected by increasingly sophisticated safety mechanisms. However, jailbreak attacks continue to pose a critical security threat by inducing models to generate policy-violating behaviors. Current paradigms focus on input-level anomalies, overlooking that the model's internal psychometric state can be systematically manipulated. To address this, we introduce Psychological Jailbreak, a new jailbreak attack paradigm that exposes a stateful psychological attack surface in LLMs, where attackers exploit the manipulation of a model's psychological state across interactions. Building on this insight, we propose Human-like Psychological Manipulation (HPM), a black-box jailbreak method that dynamically profiles a target model's latent psychological vulnerabilities and synthesizes tailored multi-turn attack strategies. By leveraging the model's optimization for anthropomorphic consistency, HPM creates a psychological pressure where social compliance overrides safety constraints. To systematically measure psychological safety, we construct an evaluation framework incorporating psychometric datasets and the Policy Corruption Score (PCS). Benchmarking against various models (e.g., GPT-4o, DeepSeek-V3, Gemini-2-Flash), HPM achieves a mean Attack Success Rate (ASR) of 88.1%, outperforming state-of-the-art attack baselines. Our experiments demonstrate robust penetration against advanced defenses, including adversarial prompt optimization (e.g., RPO) and cognitive interventions (e.g., Self-Reminder). Ultimately, PCS analysis confirms HPM induces safety breakdown to satisfy manipulated contexts. Our work advocates for a fundamental paradigm shift from static content filtering to psychological safety, prioritizing the development of psychological defense mechanisms against deep cognitive manipulation.



## **29. Performance Guarantees for Data Freshness in Resource-Constrained Adversarial IoT Systems**

cs.NI

6 pages, 4 figures, conference paper

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2512.18155v1) [paper-pdf](https://arxiv.org/pdf/2512.18155v1)

**Authors**: Aresh Dadlani, Muthukrishnan Senthil Kumar, Omid Ardakanian, Ioanis Nikolaidis

**Abstract**: Timely updates are critical for real-time monitoring and control applications powered by the Internet of Things (IoT). As these systems scale, they become increasingly vulnerable to adversarial attacks, where malicious agents interfere with legitimate transmissions to reduce data rates, thereby inflating the age of information (AoI). Existing adversarial AoI models often assume stationary channels and overlook queueing dynamics arising from compromised sensing sources operating under resource constraints. Motivated by the G-queue framework, this paper investigates a two-source M/G/1/1 system in which one source is adversarial and disrupts the update process by injecting negative arrivals according to a Poisson process and inducing i.i.d. service slowdowns, bounded in attack rate and duration. Using moment generating functions, we then derive closed-form expressions for average and peak AoI for an arbitrary number of sources. Moreover, we introduce a worst-case constrained attack model and employ stochastic dominance arguments to establish analytical AoI bounds. Numerical results validate the analysis and highlight the impact of resource-limited adversarial interference under general service time distributions.



## **30. COBRA: Catastrophic Bit-flip Reliability Analysis of State-Space Models**

cs.CR

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.15778v2) [paper-pdf](https://arxiv.org/pdf/2512.15778v2)

**Authors**: Sanjay Das, Swastik Bhattacharya, Shamik Kundu, Arnab Raha, Souvik Kundu, Kanad Basu

**Abstract**: State-space models (SSMs), exemplified by the Mamba architecture, have recently emerged as state-of-the-art sequence-modeling frameworks, offering linear-time scalability together with strong performance in long-context settings. Owing to their unique combination of efficiency, scalability, and expressive capacity, SSMs have become compelling alternatives to transformer-based models, which suffer from the quadratic computational and memory costs of attention mechanisms. As SSMs are increasingly deployed in real-world applications, it is critical to assess their susceptibility to both software- and hardware-level threats to ensure secure and reliable operation. Among such threats, hardware-induced bit-flip attacks (BFAs) pose a particularly severe risk by corrupting model parameters through memory faults, thereby undermining model accuracy and functional integrity. To investigate this vulnerability, we introduce RAMBO, the first BFA framework specifically designed to target Mamba-based architectures. Through experiments on the Mamba-1.4b model with LAMBADA benchmark, a cloze-style word-prediction task, we demonstrate that flipping merely a single critical bit can catastrophically reduce accuracy from 74.64% to 0% and increase perplexity from 18.94 to 3.75 x 10^6. These results demonstrate the pronounced fragility of SSMs to adversarial perturbations.



## **31. Look Twice before You Leap: A Rational Agent Framework for Localized Adversarial Anonymization**

cs.CR

17 pages, 9 figures, 6 tables. Revised version with an updated author list, expanded experimental results and analysis

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.06713v2) [paper-pdf](https://arxiv.org/pdf/2512.06713v2)

**Authors**: Donghang Duan, Xu Zheng, Yuefeng He, Chong Mu, Leyi Cai, Lizong Zhang

**Abstract**: Current LLM-based text anonymization frameworks usually rely on remote API services from powerful LLMs, which creates an inherent privacy paradox: users must disclose data to untrusted third parties for guaranteed privacy preservation. Moreover, directly migrating current solutions to local small-scale models (LSMs) offers a suboptimal solution with severe utility collapse. Our work argues that this failure stems not merely from the capability deficits of LSMs, but significantly from the inherent irrationality of the greedy adversarial strategies employed by current state-of-the-art (SOTA) methods. To address this, we propose Rational Localized Adversarial Anonymization (RLAA), a fully localized and training-free framework featuring an Attacker-Arbitrator-Anonymizer architecture. We model the anonymization process as a trade-off between Marginal Privacy Gain (MPG) and Marginal Utility Cost (MUC), and demonstrate that greedy strategies tend to drift into an irrational state. Instead, RLAA introduces an arbitrator that acts as a rationality gatekeeper, validating the attacker's inference to filter out feedback providing negligible privacy benefits. This mechanism promotes a rational early-stopping criterion, and structurally prevents utility collapse. Extensive experiments on different benchmarks demonstrate that RLAA achieves a superior privacy-utility trade-off compared to strong baselines.



## **32. SEA: Spectral Edge Attack on Graph Neural Networks**

cs.LG

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2512.08964v3) [paper-pdf](https://arxiv.org/pdf/2512.08964v3)

**Authors**: Yongyu Wang

**Abstract**: Graph neural networks (GNNs) have been widely applied in a variety of domains. However, the very ability of graphs to represent complex data structures is both the key strength of GNNs and a major source of their vulnerability. Recent studies have shown that attacking GNNs by maliciously perturbing the underlying graph can severely degrade their performance. For attack methods, the central challenge is to maintain attack effectiveness while remaining difficult to detect. Most existing attacks require modifying the graph structure, such as adding or deleting edges, which is relatively easy to notice. To address this problem, this paper proposes a new attack model that employs spectral adversarial robustness evaluation to quantitatively analyze the vulnerability of each edge in a graph. By precisely targeting the weakest links, our method can achieve effective attacks without changing the connectivity pattern of edges in the graph, for example by subtly adjusting the weights of a small subset of the most vulnerable edges. We apply the proposed method to attack several classical graph neural network architectures, and experimental results show that our attack is highly effective.



## **33. Ensuring Calibration Robustness in Split Conformal Prediction Under Adversarial Attacks**

stat.ML

Submitted to AISTATS 2026

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2511.18562v2) [paper-pdf](https://arxiv.org/pdf/2511.18562v2)

**Authors**: Xunlei Qian, Yue Xing

**Abstract**: Conformal prediction (CP) provides distribution-free, finite-sample coverage guarantees but critically relies on exchangeability, a condition often violated under distribution shift. We study the robustness of split conformal prediction under adversarial perturbations at test time, focusing on both coverage validity and the resulting prediction set size. Our theoretical analysis characterizes how the strength of adversarial perturbations during calibration affects coverage guarantees under adversarial test conditions. We further examine the impact of adversarial training at the model-training stage. Extensive experiments support our theory: (i) Prediction coverage varies monotonically with the calibration-time attack strength, enabling the use of nonzero calibration-time attack to predictably control coverage under adversarial tests; (ii) target coverage can hold over a range of test-time attacks: with a suitable calibration attack, coverage stays within any chosen tolerance band across a contiguous set of perturbation levels; and (iii) adversarial training at the training stage produces tighter prediction sets that retain high informativeness.



## **34. Lifefin: Escaping Mempool Explosions in DAG-based BFT**

cs.CR

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2511.15936v2) [paper-pdf](https://arxiv.org/pdf/2511.15936v2)

**Authors**: Jianting Zhang, Sen Yang, Alberto Sonnino, Sebastián Loza, Aniket Kate

**Abstract**: Directed Acyclic Graph (DAG)-based Byzantine Fault-Tolerant (BFT) protocols have emerged as promising solutions for high-throughput blockchains. By decoupling data dissemination from transaction ordering and constructing a well-connected DAG in the mempool, these protocols enable zero-message ordering and implicit view changes. However, we identify a fundamental liveness vulnerability: an adversary can trigger mempool explosions to prevent transaction commitment, ultimately compromising the protocol's liveness.   In response, this work presents Lifefin, a generic and self-stabilizing protocol designed to integrate seamlessly with existing DAG-based BFT protocols and circumvent such vulnerabilities. Lifefin leverages the Agreement on Common Subset (ACS) mechanism, allowing nodes to escape mempool explosions by committing transactions with bounded resource usage even in adverse conditions. As a result, Lifefin imposes (almost) zero overhead in typical cases while effectively eliminating liveness vulnerabilities.   To demonstrate the effectiveness of Lifefin, we integrate it into two state-of-the-art DAG-based BFT protocols, Sailfish and Mysticeti, resulting in two enhanced variants: Sailfish-Lifefin and Mysticeti-Lifefin. We implement these variants and compare them with the original Sailfish and Mysticeti systems. Our evaluation demonstrates that Lifefin achieves comparable transaction throughput while introducing only minimal additional latency to resist similar attacks.



## **35. Potent but Stealthy: Rethink Profile Pollution against Sequential Recommendation via Bi-level Constrained Reinforcement Paradigm**

cs.LG

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2511.09392v4) [paper-pdf](https://arxiv.org/pdf/2511.09392v4)

**Authors**: Jiajie Su, Zihan Nan, Yunshan Ma, Xiaobo Xia, Xiaohua Feng, Weiming Liu, Xiang Chen, Xiaolin Zheng, Chaochao Chen

**Abstract**: Sequential Recommenders, which exploit dynamic user intents through interaction sequences, is vulnerable to adversarial attacks. While existing attacks primarily rely on data poisoning, they require large-scale user access or fake profiles thus lacking practicality. In this paper, we focus on the Profile Pollution Attack that subtly contaminates partial user interactions to induce targeted mispredictions. Previous PPA methods suffer from two limitations, i.e., i) over-reliance on sequence horizon impact restricts fine-grained perturbations on item transitions, and ii) holistic modifications cause detectable distribution shifts. To address these challenges, we propose a constrained reinforcement driven attack CREAT that synergizes a bi-level optimization framework with multi-reward reinforcement learning to balance adversarial efficacy and stealthiness. We first develop a Pattern Balanced Rewarding Policy, which integrates pattern inversion rewards to invert critical patterns and distribution consistency rewards to minimize detectable shifts via unbalanced co-optimal transport. Then we employ a Constrained Group Relative Reinforcement Learning paradigm, enabling step-wise perturbations through dynamic barrier constraints and group-shared experience replay, achieving targeted pollution with minimal detectability. Extensive experiments demonstrate the effectiveness of CREAT.



## **36. AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models**

cs.CL

Presented at NeurIPS 2025 Lock-LLM Workshop. Code is available at https://github.com/AAN-AutoAdv/AutoAdv

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2511.02376v3) [paper-pdf](https://arxiv.org/pdf/2511.02376v3)

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs. Yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves an attack success rate of up to 95% on Llama-3.1-8B within six turns, a 24% improvement over single-turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests and then iteratively refines them. Extensive evaluation across commercial and open-source models (Llama-3.1-8B, GPT-4o mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses.



## **37. Machine Unlearning in Speech Emotion Recognition via Forget Set Alone**

cs.SD

Submitted to ICASSP 2026

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2510.04251v2) [paper-pdf](https://arxiv.org/pdf/2510.04251v2)

**Authors**: Zhao Ren, Rathi Adarshi Rammohan, Kevin Scheck, Tanja Schultz

**Abstract**: Speech emotion recognition aims to identify emotional states from speech signals and has been widely applied in human-computer interaction, education, healthcare, and many other fields. However, since speech data contain rich sensitive information, partial data can be required to be deleted by speakers due to privacy concerns. Current machine unlearning approaches largely depend on data beyond the samples to be forgotten. However, this reliance poses challenges when data redistribution is restricted and demands substantial computational resources in the context of big data. We propose a novel adversarial-attack-based approach that fine-tunes a pre-trained speech emotion recognition model using only the data to be forgotten. The experimental results demonstrate that the proposed approach can effectively remove the knowledge of the data to be forgotten from the model, while preserving high model performance on the test set for emotion recognition.



## **38. Membership Inference Attack with Partial Features**

cs.LG

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2508.06244v2) [paper-pdf](https://arxiv.org/pdf/2508.06244v2)

**Authors**: Xurun Wang, Guangrui Liu, Xinjie Li, Haoyu He, Lin Yao, Zhongyun Hua, Weizhe Zhang

**Abstract**: Machine learning models are vulnerable to membership inference attack, which can be used to determine whether a given sample appears in the training data. Most existing methods assume the attacker has full access to the features of the target sample. This assumption, however, does not hold in many real-world scenarios where only partial features are available, thereby limiting the applicability of these methods. In this work, we introduce Partial Feature Membership Inference (PFMI), a scenario where the adversary observes only partial features of each sample and aims to infer whether this observed subset was present in the training set. To address this problem, we propose MRAD (Memory-guided Reconstruction and Anomaly Detection), a two-stage attack framework that works in both white-box and black-box settings. In the first stage, MRAD leverages the latent memory of the target model to reconstruct the unknown features of the sample. We observe that when the known features are absent from the training set, the reconstructed sample deviates significantly from the true data distribution. Consequently, in the second stage, we use anomaly detection algorithms to measure the deviation between the reconstructed sample and the training data distribution, thereby determining whether the known features belong to a member of the training set. Empirical results demonstrate that MRAD is effective across various datasets, and maintains compatibility with off-the-shelf anomaly detection techniques. For example, on STL-10, our attack exceeds an AUC of around 0.75 even with 60% of the missing features.



## **39. Simulated Ensemble Attack: Transferring Jailbreaks Across Fine-tuned Vision-Language Models**

cs.CV

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2508.01741v2) [paper-pdf](https://arxiv.org/pdf/2508.01741v2)

**Authors**: Ruofan Wang, Xin Wang, Yang Yao, Xuan Tong, Xingjun Ma

**Abstract**: Fine-tuning open-source Vision-Language Models (VLMs) creates a critical yet underexplored attack surface: vulnerabilities in the base VLM could be retained in fine-tuned variants, rendering them susceptible to transferable jailbreak attacks. To demonstrate this risk, we introduce the Simulated Ensemble Attack (SEA), a novel grey-box jailbreak method in which the adversary has full access to the base VLM but no knowledge of the fine-tuned target's weights or training configuration. To improve jailbreak transferability across fine-tuned VLMs, SEA combines two key techniques: Fine-tuning Trajectory Simulation (FTS) and Targeted Prompt Guidance (TPG). FTS generates transferable adversarial images by simulating the vision encoder's parameter shifts, while TPG is a textual strategy that steers the language decoder toward adversarially optimized outputs. Experiments on the Qwen2-VL family (2B and 7B) demonstrate that SEA achieves high transfer attack success rates exceeding 86.5% and toxicity rates near 49.5% across diverse fine-tuned variants, even those specifically fine-tuned to improve safety behaviors. Notably, while direct PGD-based image jailbreaks rarely transfer across fine-tuned VLMs, SEA reliably exploits inherited vulnerabilities from the base model, significantly enhancing transferability. These findings highlight an urgent need to safeguard fine-tuned proprietary VLMs against transferable vulnerabilities inherited from open-source foundations, motivating the development of holistic defenses across the entire model lifecycle.



## **40. Universal Jailbreak Suffixes Are Strong Attention Hijackers**

cs.CR

Accepted at TACL 2026

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2506.12880v2) [paper-pdf](https://arxiv.org/pdf/2506.12880v2)

**Authors**: Matan Ben-Tov, Mor Geva, Mahmood Sharif

**Abstract**: We study suffix-based jailbreaks$\unicode{x2013}$a powerful family of attacks against large language models (LLMs) that optimize adversarial suffixes to circumvent safety alignment. Focusing on the widely used foundational GCG attack, we observe that suffixes vary in efficacy: some are markedly more universal$\unicode{x2013}$generalizing to many unseen harmful instructions$\unicode{x2013}$than others. We first show that a shallow, critical mechanism drives GCG's effectiveness. This mechanism builds on the information flow from the adversarial suffix to the final chat template tokens before generation. Quantifying the dominance of this mechanism during generation, we find GCG irregularly and aggressively hijacks the contextualization process. Crucially, we tie hijacking to the universality phenomenon, with more universal suffixes being stronger hijackers. Subsequently, we show that these insights have practical implications: GCG's universality can be efficiently enhanced (up to $\times$5 in some cases) at no additional computational cost, and can also be surgically mitigated, at least halving the attack's success with minimal utility loss. We release our code and data at http://github.com/matanbt/interp-jailbreak.



## **41. SoK: Are Watermarks in LLMs Ready for Deployment?**

cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2506.05594v3) [paper-pdf](https://arxiv.org/pdf/2506.05594v3)

**Authors**: Kieu Dang, Phung Lai, NhatHai Phan, Yelong Shen, Ruoming Jin, Abdallah Khreishah, My T. Thai

**Abstract**: Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs.   To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment.



## **42. Efficient and Stealthy Jailbreak Attacks via Adversarial Prompt Distillation from LLMs to SLMs**

cs.CL

19 pages, 7 figures

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2506.17231v2) [paper-pdf](https://arxiv.org/pdf/2506.17231v2)

**Authors**: Xiang Li, Chong Zhang, Jia Wang, Fangyu Wu, Yushi Li, Xiaobo Jin

**Abstract**: As the scale and complexity of jailbreaking attacks on large language models (LLMs) continue to escalate, their efficiency and practical applicability are constrained, posing a profound challenge to LLM security. Jailbreaking techniques have advanced from manual prompt engineering to automated methodologies. Recent advances have automated jailbreaking approaches that harness LLMs to generate jailbreak instructions and adversarial examples, delivering encouraging results. Nevertheless, these methods universally include an LLM generation phase, which, due to the complexities of deploying and reasoning with LLMs, impedes effective implementation and broader adoption. To mitigate this issue, we introduce \textbf{Adversarial Prompt Distillation}, an innovative framework that integrates masked language modeling, reinforcement learning, and dynamic temperature control to distill LLM jailbreaking prowess into smaller language models (SLMs). This methodology enables efficient, robust jailbreak attacks while maintaining high success rates and accommodating a broader range of application contexts. Empirical evaluations affirm the approach's superiority in attack efficacy, resource optimization, and cross-model versatility. Our research underscores the practicality of transferring jailbreak capabilities to SLMs, reveals inherent vulnerabilities in LLMs, and provides novel insights to advance LLM security investigations. Our code is available at: https://github.com/lxgem/Efficient_and_Stealthy_Jailbreak_Attacks_via_Adversarial_Prompt.



## **43. Towards Dataset Copyright Evasion Attack against Personalized Text-to-Image Diffusion Models**

cs.CV

Accepted by IEEE Transactions on Information Forensics and Security

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2505.02824v2) [paper-pdf](https://arxiv.org/pdf/2505.02824v2)

**Authors**: Kuofeng Gao, Yufei Zhu, Yiming Li, Jiawang Bai, Yong Yang, Zhifeng Li, Shu-Tao Xia

**Abstract**: Text-to-image (T2I) diffusion models enable high-quality image generation conditioned on textual prompts. However, fine-tuning these pre-trained models for personalization raises concerns about unauthorized dataset usage. To address this issue, dataset ownership verification (DOV) has recently been proposed, which embeds watermarks into fine-tuning datasets via backdoor techniques. These watermarks remain dormant on benign samples but produce owner-specified outputs when triggered. Despite its promise, the robustness of DOV against copyright evasion attacks (CEA) remains unexplored. In this paper, we investigate how adversaries can circumvent these mechanisms, enabling models trained on watermarked datasets to bypass ownership verification. We begin by analyzing the limitations of potential attacks achieved by backdoor removal, including TPD and T2IShield. In practice, TPD suffers from inconsistent effectiveness due to randomness, while T2IShield fails when watermarks are embedded as local image patches. To this end, we introduce CEAT2I, the first CEA specifically targeting DOV in T2I diffusion models. CEAT2I consists of three stages: (1) motivated by the observation that T2I models converge faster on watermarked samples with respect to intermediate features rather than training loss, we reliably detect watermarked samples; (2) we iteratively ablate tokens from the prompts of detected samples and monitor feature shifts to identify trigger tokens; and (3) we apply a closed-form concept erasure method to remove the injected watermarks. Extensive experiments demonstrate that CEAT2I effectively evades state-of-the-art DOV mechanisms while preserving model performance. The code is available at https://github.com/csyufei/CEAT2I.



## **44. AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models**

cs.CR

We encountered issues with the paper being hosted under my personal account, so we republished it under a different account associated with a university email, which makes updates and management easier. As a result, this version is a duplicate of arXiv:2511.02376

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2507.01020v2) [paper-pdf](https://arxiv.org/pdf/2507.01020v2)

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities to jailbreaking attacks: carefully crafted malicious inputs intended to circumvent safety guardrails and elicit harmful responses. As such, we present AutoAdv, a novel framework that automates adversarial prompt generation to systematically evaluate and expose vulnerabilities in LLM safety mechanisms. Our approach leverages a parametric attacker LLM to produce semantically disguised malicious prompts through strategic rewriting techniques, specialized system prompts, and optimized hyperparameter configurations. The primary contribution of our work is a dynamic, multi-turn attack methodology that analyzes failed jailbreak attempts and iteratively generates refined follow-up prompts, leveraging techniques such as roleplaying, misdirection, and contextual manipulation. We quantitatively evaluate attack success rate (ASR) using the StrongREJECT (arXiv:2402.10260 [cs.CL]) framework across sequential interaction turns. Through extensive empirical evaluation of state-of-the-art models--including ChatGPT, Llama, and DeepSeek--we reveal significant vulnerabilities, with our automated attacks achieving jailbreak success rates of up to 86% for harmful content generation. Our findings reveal that current safety mechanisms remain susceptible to sophisticated multi-turn attacks, emphasizing the urgent need for more robust defense strategies.



## **45. To See or Not to See -- Fingerprinting Devices in Adversarial Environments Amid Advanced Machine Learning**

cs.CR

10 pages, 4 figures

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2504.08264v2) [paper-pdf](https://arxiv.org/pdf/2504.08264v2)

**Authors**: Justin Feng, Amirmohammad Haddad, Nader Sehatbakhsh

**Abstract**: The increasing use of the Internet of Things raises security concerns. To address this, device fingerprinting is often employed to authenticate devices, detect adversaries, and identify eavesdroppers in an environment. This requires the ability to discern between legitimate and malicious devices which is achieved by analyzing the unique physical and/or operational characteristics of IoT devices. In the era of the latest progress in machine learning, particularly generative models, it is crucial to methodically examine the current studies in device fingerprinting. This involves explaining their approaches and underscoring their limitations when faced with adversaries armed with these ML tools. To systematically analyze existing methods, we propose a generic, yet simplified, model for device fingerprinting. Additionally, we thoroughly investigate existing methods to authenticate devices and detect eavesdropping, using our proposed model. We further study trends and similarities between works in authentication and eavesdropping detection and present the existing threats and attacks in these domains. Finally, we discuss future directions in fingerprinting based on these trends to develop more secure IoT fingerprinting schemes.



## **46. Bleeding Pathways: Vanishing Discriminability in LLM Hidden States Fuels Jailbreak Attacks**

cs.CR

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2503.11185v2) [paper-pdf](https://arxiv.org/pdf/2503.11185v2)

**Authors**: Yingjie Zhang, Tong Liu, Zhe Zhao, Guozhu Meng, Kai Chen

**Abstract**: LLMs remain vulnerable to jailbreak attacks that exploit adversarial prompts to circumvent safety measures. Current safety fine-tuning approaches face two critical limitations. First, they often fail to strike a balance between security and utility, where stronger safety measures tend to over-reject harmless user requests. Second, they frequently miss malicious intent concealed within seemingly benign tasks, leaving models exposed to exploitation. Our work identifies a fundamental cause of these issues: during response generation, an LLM's capacity to differentiate harmful from safe outputs deteriorates. Experimental evidence confirms this, revealing that the separability between hidden states for safe and harmful responses diminishes as generation progresses. This weakening discrimination forces models to make compliance judgments earlier in the generation process, restricting their ability to recognize developing harmful intent and contributing to both aforementioned failures. To mitigate this vulnerability, we introduce DEEPALIGN - an inherent defense framework that enhances the safety of LLMs. By applying contrastive hidden-state steering at the midpoint of response generation, DEEPALIGN amplifies the separation between harmful and benign hidden states, enabling continuous intrinsic toxicity detection and intervention throughout the generation process. Across diverse LLMs spanning varying architectures and scales, it reduced attack success rates of nine distinct jailbreak attacks to near-zero or minimal. Crucially, it preserved model capability while reducing over-refusal. Models equipped with DEEPALIGN exhibited up to 3.5% lower error rates in rejecting challenging benign queries and maintained standard task performance with less than 1% decline. This marks a substantial advance in the safety-utility Pareto frontier.



## **47. When Should Selfish Miners Double-Spend?**

cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2501.03227v4) [paper-pdf](https://arxiv.org/pdf/2501.03227v4)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: Conventional double-spending attack models ignore the revenue losses stemming from the orphan blocks. On the other hand, selfish mining literature usually ignores the chance of the attacker to double-spend at no-cost in each attack cycle. In this paper, we give a rigorous stochastic analysis of an attack where the goal of the adversary is to double-spend while mining selfishly. To do so, we first combine stubborn and selfish mining attacks, i.e., construct a strategy where the attacker acts stubborn until its private branch reaches a certain length and then switches to act selfish. We provide the optimal stubbornness for each parameter regime. Next, we provide the maximum stubbornness that is still more profitable than honest mining and argue a connection between the level of stubbornness and the $k$-confirmation rule. We show that, at each attack cycle, if the level of stubbornness is higher than $k$, the adversary gets a free shot at double-spending. At each cycle, for a given stubbornness level, we rigorously formulate how great the probability of double-spending is. We further modify the attack in the stubborn regime in order to conceal the attack and increase the double-spending probability.



## **48. ThreatPilot: Attack-Driven Threat Intelligence Extraction**

cs.CR

17 pages

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2412.10872v2) [paper-pdf](https://arxiv.org/pdf/2412.10872v2)

**Authors**: Ming Xu, Hongtai Wang, Jiahao Liu, Xinfeng Li, Zhengmin Yu, Weili Han, Hoon Wei Lim, Jin Song Dong, Jiaheng Zhang

**Abstract**: Efficient defense against dynamically evolving advanced persistent threats (APT) requires the structured threat intelligence feeds, such as techniques used. However, existing threat-intelligence extraction techniques predominantly focuses on individual pieces of intelligence-such as isolated techniques or atomic indicators-resulting in fragmented and incomplete representations of real-world attacks. This granularity inherently limits on both the depth and the contextual richness of the extracted intelligence, making it difficult for downstream security systems to reason about multi-step behaviors or to generate actionable detections. To address this gap, we propose to extract the layered Attack-driven Threat Intelligence (ATIs), a comprehensive representation that captures the full spectrum of adversarial behavior. We propose ThreatPilot, which can accurately identify the AITs including complete tactics, techniques, multi-step procedures, and their procedure variants, and integrate the threat intelligence to software security application scenarios: the detection rules (i.e., Sigma) and attack command can be generated automatically to a more accuracy level. Experimental results on 1,769 newly crawly reports and 16 manually calibrated reports show ThreatPilot's effectiveness in identifying accuracy techniques, outperforming state-of-the-art approaches of AttacKG by 1.34X in F1 score. Further studies upon 64,185 application logs via Honeypot show that our Sigma rule generator significantly outperforms several existing rules-set in detecting the real-world malicious events. Industry partners confirm that our Sigma rule generator can significantly help save time and costs of the rule generation process. In addition, our generated commands achieve an execution rate of 99.3%, compared to 50.3% without the extracted intelligence.



## **49. Scalable Dendritic Modeling Advances Expressive and Robust Deep Spiking Neural Networks**

cs.NE

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2412.06355v2) [paper-pdf](https://arxiv.org/pdf/2412.06355v2)

**Authors**: Yifan Huang, Wei Fang, Zhengyu Ma, Guoqi Li, Yonghong Tian

**Abstract**: Dendritic computation endows biological neurons with rich nonlinear integration and high representational capacity, yet it is largely missing in existing deep spiking neural networks (SNNs). Although detailed multi-compartment models can capture dendritic computations, their high computational cost and limited flexibility make them impractical for deep learning. To combine the advantages of dendritic computation and deep network architectures for a powerful, flexible and efficient computational model, we propose the dendritic spiking neuron (DendSN). DendSN explicitly models dendritic morphology and nonlinear integration in a streamlined design, leading to substantially higher expressivity than point neurons and wide compatibility with modern deep SNN architectures. Leveraging the efficient formulation and high-performance Triton kernels, dendritic SNNs (DendSNNs) can be efficiently trained and easily scaled to deeper networks. Experiments show that DendSNNs consistently outperform conventional SNNs on classification tasks. Furthermore, inspired by dendritic modulation and synaptic clustering, we introduce the dendritic branch gating (DBG) algorithm for task-incremental learning, which effectively reduces inter-task interference. Additional evaluations show that DendSNNs exhibit superior robustness to noise and adversarial attacks, along with improved generalization in few-shot learning scenarios. Our work firstly demonstrates the possibility of training deep SNNs with multiple nonlinear dendritic branches, and comprehensively analyzes the impact of dendrite computation on representation learning across various machine learning settings, thereby offering a fresh perspective on advancing SNN design.



## **50. One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models**

cs.CV

Accepted by ICCV-2025

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2406.05491v4) [paper-pdf](https://arxiv.org/pdf/2406.05491v4)

**Authors**: Hao Fang, Jiawei Kong, Wenbo Yu, Bin Chen, Jiawei Li, Hao Wu, Shutao Xia, Ke Xu

**Abstract**: Vision-Language Pre-training (VLP) models have exhibited unprecedented capability in many applications by taking full advantage of the multimodal alignment. However, previous studies have shown they are vulnerable to maliciously crafted adversarial samples. Despite recent success, these methods are generally instance-specific and require generating perturbations for each input sample. In this paper, we reveal that VLP models are also vulnerable to the instance-agnostic universal adversarial perturbation (UAP). Specifically, we design a novel Contrastive-training Perturbation Generator with Cross-modal conditions (C-PGC) to achieve the attack. In light that the pivotal multimodal alignment is achieved through the advanced contrastive learning technique, we devise to turn this powerful weapon against themselves, i.e., employ a malicious version of contrastive learning to train the C-PGC based on our carefully crafted positive and negative image-text pairs for essentially destroying the alignment relationship learned by VLP models. Besides, C-PGC fully utilizes the characteristics of Vision-and-Language (V+L) scenarios by incorporating both unimodal and cross-modal information as effective guidance. Extensive experiments show that C-PGC successfully forces adversarial samples to move away from their original area in the VLP model's feature space, thus essentially enhancing attacks across various victim models and V+L tasks. The GitHub repository is available at https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks.



