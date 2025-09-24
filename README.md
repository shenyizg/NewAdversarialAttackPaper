# Latest Adversarial Attack Papers
**update at 2025-09-24 09:11:17**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. AI-Generated Text is Non-Stationary: Detection via Temporal Tomography**

cs.CL

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2508.01754v2) [paper-pdf](http://arxiv.org/pdf/2508.01754v2)

**Authors**: Alva West, Yixuan Weng, Minjun Zhu, Luodan Zhang, Zhen Lin, Guangsheng Bao, Yue Zhang

**Abstract**: The field of AI-generated text detection has evolved from supervised classification to zero-shot statistical analysis. However, current approaches share a fundamental limitation: they aggregate token-level measurements into scalar scores, discarding positional information about where anomalies occur. Our empirical analysis reveals that AI-generated text exhibits significant non-stationarity, statistical properties vary by 73.8\% more between text segments compared to human writing. This discovery explains why existing detectors fail against localized adversarial perturbations that exploit this overlooked characteristic. We introduce Temporal Discrepancy Tomography (TDT), a novel detection paradigm that preserves positional information by reformulating detection as a signal processing task. TDT treats token-level discrepancies as a time-series signal and applies Continuous Wavelet Transform to generate a two-dimensional time-scale representation, capturing both the location and linguistic scale of statistical anomalies. On the RAID benchmark, TDT achieves 0.855 AUROC (7.1\% improvement over the best baseline). More importantly, TDT demonstrates robust performance on adversarial tasks, with 14.1\% AUROC improvement on HART Level 2 paraphrasing attacks. Despite its sophisticated analysis, TDT maintains practical efficiency with only 13\% computational overhead. Our work establishes non-stationarity as a fundamental characteristic of AI-generated text and demonstrates that preserving temporal dynamics is essential for robust detection.



## **2. Algorithms for Adversarially Robust Deep Learning**

cs.LG

PhD thesis

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.19100v1) [paper-pdf](http://arxiv.org/pdf/2509.19100v1)

**Authors**: Alexander Robey

**Abstract**: Given the widespread use of deep learning models in safety-critical applications, ensuring that the decisions of such models are robust against adversarial exploitation is of fundamental importance. In this thesis, we discuss recent progress toward designing algorithms that exhibit desirable robustness properties. First, we discuss the problem of adversarial examples in computer vision, for which we introduce new technical results, training paradigms, and certification algorithms. Next, we consider the problem of domain generalization, wherein the task is to train neural networks to generalize from a family of training distributions to unseen test distributions. We present new algorithms that achieve state-of-the-art generalization in medical imaging, molecular identification, and image classification. Finally, we study the setting of jailbreaking large language models (LLMs), wherein an adversarial user attempts to design prompts that elicit objectionable content from an LLM. We propose new attacks and defenses, which represent the frontier of progress toward designing robust language-based agents.



## **3. Latent Danger Zone: Distilling Unified Attention for Cross-Architecture Black-box Attacks**

cs.LG

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.19044v1) [paper-pdf](http://arxiv.org/pdf/2509.19044v1)

**Authors**: Yang Li, Chenyu Wang, Tingrui Wang, Yongwei Wang, Haonan Li, Zhunga Liu, Quan Pan

**Abstract**: Black-box adversarial attacks remain challenging due to limited access to model internals. Existing methods often depend on specific network architectures or require numerous queries, resulting in limited cross-architecture transferability and high query costs. To address these limitations, we propose JAD, a latent diffusion model framework for black-box adversarial attacks. JAD generates adversarial examples by leveraging a latent diffusion model guided by attention maps distilled from both a convolutional neural network (CNN) and a Vision Transformer (ViT) models. By focusing on image regions that are commonly sensitive across architectures, this approach crafts adversarial perturbations that transfer effectively between different model types. This joint attention distillation strategy enables JAD to be architecture-agnostic, achieving superior attack generalization across diverse models. Moreover, the generative nature of the diffusion framework yields high adversarial sample generation efficiency by reducing reliance on iterative queries. Experiments demonstrate that JAD offers improved attack generalization, generation efficiency, and cross-architecture transferability compared to existing methods, providing a promising and effective paradigm for black-box adversarial attacks.



## **4. Towards Privacy-Aware Bayesian Networks: A Credal Approach**

cs.LG

Accepted at ECAI2025 conference, 20 pages, 1 figure

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.18949v1) [paper-pdf](http://arxiv.org/pdf/2509.18949v1)

**Authors**: Niccolò Rocchi, Fabio Stella, Cassio de Campos

**Abstract**: Bayesian networks (BN) are probabilistic graphical models that enable efficient knowledge representation and inference. These have proven effective across diverse domains, including healthcare, bioinformatics and economics. The structure and parameters of a BN can be obtained by domain experts or directly learned from available data. However, as privacy concerns escalate, it becomes increasingly critical for publicly released models to safeguard sensitive information in training data. Typically, released models do not prioritize privacy by design. In particular, tracing attacks from adversaries can combine the released BN with auxiliary data to determine whether specific individuals belong to the data from which the BN was learned. State-of-the-art protection tecniques involve introducing noise into the learned parameters. While this offers robust protection against tracing attacks, it significantly impacts the model's utility, in terms of both the significance and accuracy of the resulting inferences. Hence, high privacy may be attained at the cost of releasing a possibly ineffective model. This paper introduces credal networks (CN) as a novel solution for balancing the model's privacy and utility. After adapting the notion of tracing attacks, we demonstrate that a CN enables the masking of the learned BN, thereby reducing the probability of successful attacks. As CNs are obfuscated but not noisy versions of BNs, they can achieve meaningful inferences while safeguarding privacy. Moreover, we identify key learning information that must be concealed to prevent attackers from recovering the underlying BN. Finally, we conduct a set of numerical experiments to analyze how privacy gains can be modulated by tuning the CN hyperparameters. Our results confirm that CNs provide a principled, practical, and effective approach towards the development of privacy-aware probabilistic graphical models.



## **5. Generic Adversarial Smart Contract Detection with Semantics and Uncertainty-Aware LLM**

cs.CR

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.18934v1) [paper-pdf](http://arxiv.org/pdf/2509.18934v1)

**Authors**: Yating Liu, Xing Su, Hao Wu, Sijin Li, Yuxi Cheng, Fengyuan Xu, Sheng Zhong

**Abstract**: Adversarial smart contracts, mostly on EVM-compatible chains like Ethereum and BSC, are deployed as EVM bytecode to exploit vulnerable smart contracts typically for financial gains. Detecting such malicious contracts at the time of deployment is an important proactive strategy preventing loss from victim contracts. It offers a better cost-benefit than detecting vulnerabilities on diverse potential victims. However, existing works are not generic with limited detection types and effectiveness due to imbalanced samples, while the emerging LLM technologies, which show its potentials in generalization, have two key problems impeding its application in this task: hard digestion of compiled-code inputs, especially those with task-specific logic, and hard assessment of LLMs' certainty in their binary answers, i.e., yes-or-no answers. Therefore, we propose a generic adversarial smart contracts detection framework FinDet, which leverages LLMs with two enhancements addressing above two problems. FinDet takes as input only the EVM-bytecode contracts and identifies adversarial ones among them with high balanced accuracy. The first enhancement extracts concise semantic intentions and high-level behavioral logic from the low-level bytecode inputs, unleashing the LLM reasoning capability restricted by the task input. The second enhancement probes and measures the LLM uncertainty to its multi-round answering to the same query, improving the LLM answering robustness for binary classifications required by the task output. Our comprehensive evaluation shows that FinDet achieves a BAC of 0.9223 and a TPR of 0.8950, significantly outperforming existing baselines. It remains robust under challenging conditions including unseen attack patterns, low-data settings, and feature obfuscation. FinDet detects all 5 public and 20+ unreported adversarial contracts in a 10-day real-world test, confirmed manually.



## **6. Enhancing the Effectiveness and Durability of Backdoor Attacks in Federated Learning through Maximizing Task Distinction**

cs.LG

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.18904v1) [paper-pdf](http://arxiv.org/pdf/2509.18904v1)

**Authors**: Zhaoxin Wang, Handing Wang, Cong Tian, Yaochu Jin

**Abstract**: Federated learning allows multiple participants to collaboratively train a central model without sharing their private data. However, this distributed nature also exposes new attack surfaces. In particular, backdoor attacks allow attackers to implant malicious behaviors into the global model while maintaining high accuracy on benign inputs. Existing attacks usually rely on fixed patterns or adversarial perturbations as triggers, which tightly couple the main and backdoor tasks. This coupling makes them vulnerable to dilution by honest updates and limits their persistence under federated defenses. In this work, we propose an approach to decouple the backdoor task from the main task by dynamically optimizing the backdoor trigger within a min-max framework. The inner layer maximizes the performance gap between poisoned and benign samples, ensuring that the contributions of benign users have minimal impact on the backdoor. The outer process injects the adaptive triggers into the local model. We evaluate our method on both computer vision and natural language tasks, and compare it with six backdoor attack methods under six defense algorithms. Experimental results show that our method achieves good attack performance and can be easily integrated into existing backdoor attack techniques.



## **7. Attack for Defense: Adversarial Agents for Point Prompt Optimization Empowering Segment Anything Model**

cs.CV

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.18891v1) [paper-pdf](http://arxiv.org/pdf/2509.18891v1)

**Authors**: Xueyu Liu, Xiaoyi Zhang, Guangze Shi, Meilin Liu, Yexin Lai, Yongfei Wu, Mingqiang Wei

**Abstract**: Prompt quality plays a critical role in the performance of the Segment Anything Model (SAM), yet existing approaches often rely on heuristic or manually crafted prompts, limiting scalability and generalization. In this paper, we propose Point Prompt Defender, an adversarial reinforcement learning framework that adopts an attack-for-defense paradigm to automatically optimize point prompts. We construct a task-agnostic point prompt environment by representing image patches as nodes in a dual-space graph, where edges encode both physical and semantic distances. Within this environment, an attacker agent learns to activate a subset of prompts that maximally degrade SAM's segmentation performance, while a defender agent learns to suppress these disruptive prompts and restore accuracy. Both agents are trained using Deep Q-Networks with a reward signal based on segmentation quality variation. During inference, only the defender is deployed to refine arbitrary coarse prompt sets, enabling enhanced SAM segmentation performance across diverse tasks without retraining. Extensive experiments show that Point Prompt Defender effectively improves SAM's robustness and generalization, establishing a flexible, interpretable, and plug-and-play framework for prompt-based segmentation.



## **8. Diversity Boosts AI-Generated Text Detection**

cs.CL

Project Webpage: https://diveye.vercel.app/

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.18880v1) [paper-pdf](http://arxiv.org/pdf/2509.18880v1)

**Authors**: Advik Raj Basani, Pin-Yu Chen

**Abstract**: Detecting AI-generated text is an increasing necessity to combat misuse of LLMs in education, business compliance, journalism, and social media, where synthetic fluency can mask misinformation or deception. While prior detectors often rely on token-level likelihoods or opaque black-box classifiers, these approaches struggle against high-quality generations and offer little interpretability. In this work, we propose DivEye, a novel detection framework that captures how unpredictability fluctuates across a text using surprisal-based features. Motivated by the observation that human-authored text exhibits richer variability in lexical and structural unpredictability than LLM outputs, DivEye captures this signal through a set of interpretable statistical features. Our method outperforms existing zero-shot detectors by up to 33.2% and achieves competitive performance with fine-tuned baselines across multiple benchmarks. DivEye is robust to paraphrasing and adversarial attacks, generalizes well across domains and models, and improves the performance of existing detectors by up to 18.7% when used as an auxiliary signal. Beyond detection, DivEye provides interpretable insights into why a text is flagged, pointing to rhythmic unpredictability as a powerful and underexplored signal for LLM detection.



## **9. Fix your downsampling ASAP! Be natively more robust via Aliasing and Spectral Artifact free Pooling**

cs.CV

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2307.09804v2) [paper-pdf](http://arxiv.org/pdf/2307.09804v2)

**Authors**: Julia Grabinski, Steffen Jung, Janis Keuper, Margret Keuper

**Abstract**: Convolutional Neural Networks (CNNs) are successful in various computer vision tasks. From an image and signal processing point of view, this success is counter-intuitive, as the inherent spatial pyramid design of most CNNs is apparently violating basic signal processing laws, i.e. the Sampling Theorem in their downsampling operations. This issue has been broadly neglected until recent work in the context of adversarial attacks and distribution shifts showed that there is a strong correlation between the vulnerability of CNNs and aliasing artifacts induced by bandlimit-violating downsampling. As a remedy, we propose an alias-free downsampling operation in the frequency domain, denoted Frequency Low Cut Pooling (FLC Pooling) which we further extend to Aliasing and Sinc Artifact-free Pooling (ASAP). ASAP is alias-free and removes further artifacts from sinc-interpolation. Our experimental evaluation on ImageNet-1k, ImageNet-C and CIFAR datasets on various CNN architectures demonstrates that networks using FLC Pooling and ASAP as downsampling methods learn more stable features as measured by their robustness against common corruptions and adversarial attacks, while maintaining a clean accuracy similar to the respective baseline models.



## **10. TriFusion-AE: Language-Guided Depth and LiDAR Fusion for Robust Point Cloud Processing**

cs.CV

39th Conference on Neural Information Processing Systems (NeurIPS  2025) Workshop

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.18743v1) [paper-pdf](http://arxiv.org/pdf/2509.18743v1)

**Authors**: Susmit Neogi

**Abstract**: LiDAR-based perception is central to autonomous driving and robotics, yet raw point clouds remain highly vulnerable to noise, occlusion, and adversarial corruptions. Autoencoders offer a natural framework for denoising and reconstruction, but their performance degrades under challenging real-world conditions. In this work, we propose TriFusion-AE, a multimodal cross-attention autoencoder that integrates textual priors, monocular depth maps from multi-view images, and LiDAR point clouds to improve robustness. By aligning semantic cues from text, geometric (depth) features from images, and spatial structure from LiDAR, TriFusion-AE learns representations that are resilient to stochastic noise and adversarial perturbations. Interestingly, while showing limited gains under mild perturbations, our model achieves significantly more robust reconstruction under strong adversarial attacks and heavy noise, where CNN-based autoencoders collapse. We evaluate on the nuScenes-mini dataset to reflect realistic low-data deployment scenarios. Our multimodal fusion framework is designed to be model-agnostic, enabling seamless integration with any CNN-based point cloud autoencoder for joint representation learning.



## **11. Automating Steering for Safe Multimodal Large Language Models**

cs.CL

EMNLP 2025 Main Conference. 23 pages (8+ for main); 25 figures; 1  table

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2507.13255v3) [paper-pdf](http://arxiv.org/pdf/2507.13255v3)

**Authors**: Lyucheng Wu, Mengru Wang, Ziwen Xu, Tri Cao, Nay Oo, Bryan Hooi, Shumin Deng

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) has unlocked powerful cross-modal reasoning abilities, but also raised new safety concerns, particularly when faced with adversarial multimodal inputs. To improve the safety of MLLMs during inference, we introduce a modular and adaptive inference-time intervention technology, AutoSteer, without requiring any fine-tuning of the underlying model. AutoSteer incorporates three core components: (1) a novel Safety Awareness Score (SAS) that automatically identifies the most safety-relevant distinctions among the model's internal layers; (2) an adaptive safety prober trained to estimate the likelihood of toxic outputs from intermediate representations; and (3) a lightweight Refusal Head that selectively intervenes to modulate generation when safety risks are detected. Experiments on LLaVA-OV and Chameleon across diverse safety-critical benchmarks demonstrate that AutoSteer significantly reduces the Attack Success Rate (ASR) for textual, visual, and cross-modal threats, while maintaining general abilities. These findings position AutoSteer as a practical, interpretable, and effective framework for safer deployment of multimodal AI systems.



## **12. Examining I2P Resilience: Effect of Centrality-based Attack**

cs.CR

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.18572v1) [paper-pdf](http://arxiv.org/pdf/2509.18572v1)

**Authors**: Kemi Akanbi, Sunkanmi Oluwadare, Jess Kropczynski, Jacques Bou Abdo

**Abstract**: This study examines the robustness of I2P, a well-regarded anonymous and decentralized peer-to-peer network designed to ensure anonymity, confidentiality, and circumvention of censorship. Unlike its more widely researched counterpart, TOR, I2P's resilience has received less scholarly attention. Employing network analysis, this research evaluates I2P's susceptibility to adversarial percolation. By utilizing the degree centrality as a measure of nodes' influence in the network, the finding suggests the network is vulnerable to targeted disruptions. Before percolation, the network exhibited a density of 0.01065443 and an average path length of 6.842194. At the end of the percolation process, the density decreased by approximately 10%, and the average path length increased by 33%, indicating a decline in efficiency and connectivity. These results highlight that even decentralized networks, such as I2P, exhibit structural fragility under targeted attacks, emphasizing the need for improved design strategies to enhance resilience against adversarial disruptions.



## **13. SEGA: A Transferable Signed Ensemble Gaussian Black-Box Attack against No-Reference Image Quality Assessment Models**

cs.CV

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.18546v1) [paper-pdf](http://arxiv.org/pdf/2509.18546v1)

**Authors**: Yujia Liu, Dingquan Li, Tiejun Huang

**Abstract**: No-Reference Image Quality Assessment (NR-IQA) models play an important role in various real-world applications. Recently, adversarial attacks against NR-IQA models have attracted increasing attention, as they provide valuable insights for revealing model vulnerabilities and guiding robust system design. Some effective attacks have been proposed against NR-IQA models in white-box settings, where the attacker has full access to the target model. However, these attacks often suffer from poor transferability to unknown target models in more realistic black-box scenarios, where the target model is inaccessible. This work makes the first attempt to address the challenge of low transferability in attacking NR-IQA models by proposing a transferable Signed Ensemble Gaussian black-box Attack (SEGA). The main idea is to approximate the gradient of the target model by applying Gaussian smoothing to source models and ensembling their smoothed gradients. To ensure the imperceptibility of adversarial perturbations, SEGA further removes inappropriate perturbations using a specially designed perturbation filter mask. Experimental results on the CLIVE dataset demonstrate the superior transferability of SEGA, validating its effectiveness in enabling successful transfer-based black-box attacks against NR-IQA models.



## **14. Zero-Shot Visual Deepfake Detection: Can AI Predict and Prevent Fake Content Before It's Created?**

cs.GR

Published in Foundations and Trends in Signal Processing (#1 in  Signal Processing, #3 in Computer Science)

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.18461v1) [paper-pdf](http://arxiv.org/pdf/2509.18461v1)

**Authors**: Ayan Sar, Sampurna Roy, Tanupriya Choudhury, Ajith Abraham

**Abstract**: Generative adversarial networks (GANs) and diffusion models have dramatically advanced deepfake technology, and its threats to digital security, media integrity, and public trust have increased rapidly. This research explored zero-shot deepfake detection, an emerging method even when the models have never seen a particular deepfake variation. In this work, we studied self-supervised learning, transformer-based zero-shot classifier, generative model fingerprinting, and meta-learning techniques that better adapt to the ever-evolving deepfake threat. In addition, we suggested AI-driven prevention strategies that mitigated the underlying generation pipeline of the deepfakes before they occurred. They consisted of adversarial perturbations for creating deepfake generators, digital watermarking for content authenticity verification, real-time AI monitoring for content creation pipelines, and blockchain-based content verification frameworks. Despite these advancements, zero-shot detection and prevention faced critical challenges such as adversarial attacks, scalability constraints, ethical dilemmas, and the absence of standardized evaluation benchmarks. These limitations were addressed by discussing future research directions on explainable AI for deepfake detection, multimodal fusion based on image, audio, and text analysis, quantum AI for enhanced security, and federated learning for privacy-preserving deepfake detection. This further highlighted the need for an integrated defense framework for digital authenticity that utilized zero-shot learning in combination with preventive deepfake mechanisms. Finally, we highlighted the important role of interdisciplinary collaboration between AI researchers, cybersecurity experts, and policymakers to create resilient defenses against the rising tide of deepfake attacks.



## **15. VoxGuard: Evaluating User and Attribute Privacy in Speech via Membership Inference Attacks**

cs.CR

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.18413v1) [paper-pdf](http://arxiv.org/pdf/2509.18413v1)

**Authors**: Efthymios Tsaprazlis, Thanathai Lertpetchpun, Tiantian Feng, Sai Praneeth Karimireddy, Shrikanth Narayanan

**Abstract**: Voice anonymization aims to conceal speaker identity and attributes while preserving intelligibility, but current evaluations rely almost exclusively on Equal Error Rate (EER) that obscures whether adversaries can mount high-precision attacks. We argue that privacy should instead be evaluated in the low false-positive rate (FPR) regime, where even a small number of successful identifications constitutes a meaningful breach. To this end, we introduce VoxGuard, a framework grounded in differential privacy and membership inference that formalizes two complementary notions: User Privacy, preventing speaker re-identification, and Attribute Privacy, protecting sensitive traits such as gender and accent. Across synthetic and real datasets, we find that informed adversaries, especially those using fine-tuned models and max-similarity scoring, achieve orders-of-magnitude stronger attacks at low-FPR despite similar EER. For attributes, we show that simple transparent attacks recover gender and accent with near-perfect accuracy even after anonymization. Our results demonstrate that EER substantially underestimates leakage, highlighting the need for low-FPR evaluation, and recommend VoxGuard as a benchmark for evaluating privacy leakage.



## **16. Hybrid Reputation Aggregation: A Robust Defense Mechanism for Adversarial Federated Learning in 5G and Edge Network Environments**

cs.CR

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.18044v1) [paper-pdf](http://arxiv.org/pdf/2509.18044v1)

**Authors**: Saeid Sheikhi, Panos Kostakos, Lauri Loven

**Abstract**: Federated Learning (FL) in 5G and edge network environments face severe security threats from adversarial clients. Malicious participants can perform label flipping, inject backdoor triggers, or launch Sybil attacks to corrupt the global model. This paper introduces Hybrid Reputation Aggregation (HRA), a novel robust aggregation mechanism designed to defend against diverse adversarial behaviors in FL without prior knowledge of the attack type. HRA combines geometric anomaly detection with momentum-based reputation tracking of clients. In each round, it detects outlier model updates via distance-based geometric analysis while continuously updating a trust score for each client based on historical behavior. This hybrid approach enables adaptive filtering of suspicious updates and long-term penalization of unreliable clients, countering attacks ranging from backdoor insertions to random noise Byzantine failures. We evaluate HRA on a large-scale proprietary 5G network dataset (3M+ records) and the widely used NF-CSE-CIC-IDS2018 benchmark under diverse adversarial attack scenarios. Experimental results reveal that HRA achieves robust global model accuracy of up to 98.66% on the 5G dataset and 96.60% on NF-CSE-CIC-IDS2018, outperforming state-of-the-art aggregators such as Krum, Trimmed Mean, and Bulyan by significant margins. Our ablation studies further demonstrate that the full hybrid system achieves 98.66% accuracy, while the anomaly-only and reputation-only variants drop to 84.77% and 78.52%, respectively, validating the synergistic value of our dual-mechanism approach. This demonstrates HRA's enhanced resilience and robustness in 5G/edge federated learning deployments, even under significant adversarial conditions.



## **17. Budgeted Adversarial Attack against Graph-Based Anomaly Detection in Sensor Networks**

cs.LG

12 pages

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.17987v1) [paper-pdf](http://arxiv.org/pdf/2509.17987v1)

**Authors**: Sanju Xaviar, Omid Ardakanian

**Abstract**: Graph Neural Networks (GNNs) have emerged as powerful models for anomaly detection in sensor networks, particularly when analyzing multivariate time series. In this work, we introduce BETA, a novel grey-box evasion attack targeting such GNN-based detectors, where the attacker is constrained to perturb sensor readings from a limited set of nodes, excluding the target sensor, with the goal of either suppressing a true anomaly or triggering a false alarm at the target node. BETA identifies the sensors most influential to the target node's classification and injects carefully crafted adversarial perturbations into their features, all while maintaining stealth and respecting the attacker's budget. Experiments on three real-world sensor network datasets show that BETA reduces the detection accuracy of state-of-the-art GNN-based detectors by 30.62 to 39.16% on average, and significantly outperforms baseline attack strategies, while operating within realistic constraints.



## **18. A Lightweight Approach for State Machine Replication**

cs.DC

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.17771v1) [paper-pdf](http://arxiv.org/pdf/2509.17771v1)

**Authors**: Christian Cachin, Jinfeng Dou, Christian Scheideler, Philipp Schneider

**Abstract**: We present a lightweight solution for state machine replication with commitment certificates. Specifically, we adapt a simple median rule from the stabilizing consensus problem [Doerr11] to operate in a client-server setting where arbitrary servers may be blocked adaptively based on past system information. We further extend our protocol by compressing information about committed commands, thus keeping the protocol lightweight, while still enabling clients to easily prove that their commands have indeed been committed on the shared state. Our approach guarantees liveness as long as at most a constant fraction of servers are blocked, ensures safety under any number of blocked servers, and supports fast recovery from massive blocking attacks. In addition to offering near-optimal performance in several respects, our method is fully decentralized, unlike other near-optimal solutions that rely on leaders. In particular, our solution is robust against adversaries that target key servers (which captures insider-based denial-of-service attacks), whereas leader-based approaches fail under such a blocking model.



## **19. Is It Certainly a Deepfake? Reliability Analysis in Detection & Generation Ecosystem**

cs.AI

Accepted for publication at the ICCV 2025 STREAM workshop

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.17550v1) [paper-pdf](http://arxiv.org/pdf/2509.17550v1)

**Authors**: Neslihan Kose, Anthony Rhodes, Umur Aybars Ciftci, Ilke Demir

**Abstract**: As generative models are advancing in quality and quantity for creating synthetic content, deepfakes begin to cause online mistrust. Deepfake detectors are proposed to counter this effect, however, misuse of detectors claiming fake content as real or vice versa further fuels this misinformation problem. We present the first comprehensive uncertainty analysis of deepfake detectors, systematically investigating how generative artifacts influence prediction confidence. As reflected in detectors' responses, deepfake generators also contribute to this uncertainty as their generative residues vary, so we cross the uncertainty analysis of deepfake detectors and generators. Based on our observations, the uncertainty manifold holds enough consistent information to leverage uncertainty for deepfake source detection. Our approach leverages Bayesian Neural Networks and Monte Carlo dropout to quantify both aleatoric and epistemic uncertainties across diverse detector architectures. We evaluate uncertainty on two datasets with nine generators, with four blind and two biological detectors, compare different uncertainty methods, explore region- and pixel-based uncertainty, and conduct ablation studies. We conduct and analyze binary real/fake, multi-class real/fake, source detection, and leave-one-out experiments between the generator/detector combinations to share their generalization capability, model calibration, uncertainty, and robustness against adversarial attacks. We further introduce uncertainty maps that localize prediction confidence at the pixel level, revealing distinct patterns correlated with generator-specific artifacts. Our analysis provides critical insights for deploying reliable deepfake detection systems and establishes uncertainty quantification as a fundamental requirement for trustworthy synthetic media detection.



## **20. Proxy-Embedding as an Adversarial Teacher: An Embedding-Guided Bidirectional Attack for Referring Expression Segmentation Models**

cs.CV

20pages, 5figures

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2506.16157v2) [paper-pdf](http://arxiv.org/pdf/2506.16157v2)

**Authors**: Xingbai Chen, Tingchao Fu, Renyang Liu, Wei Zhou, Chao Yi

**Abstract**: Referring Expression Segmentation (RES) enables precise object segmentation in images based on natural language descriptions, offering high flexibility and broad applicability in real-world vision tasks. Despite its impressive performance, the robustness of RES models against adversarial examples remains largely unexplored. While prior adversarial attack methods have explored adversarial robustness on conventional segmentation models, they perform poorly when directly applied to RES models, failing to expose vulnerabilities in its multimodal structure. In practical open-world scenarios, users typically issue multiple, diverse referring expressions to interact with the same image, highlighting the need for adversarial examples that generalize across varied textual inputs. Furthermore, from the perspective of privacy protection, ensuring that RES models do not segment sensitive content without explicit authorization is a crucial aspect of enhancing the robustness and security of multimodal vision-language systems. To address these challenges, we present PEAT, an Embedding-Guided Bidirectional Attack for RES models. Extensive experiments across multiple RES architectures and standard benchmarks show that PEAT consistently outperforms competitive baselines.



## **21. Explainable AI for Analyzing Person-Specific Patterns in Facial Recognition Tasks**

cs.CV

22 pages; 24 tables; 11 figures

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.17457v1) [paper-pdf](http://arxiv.org/pdf/2509.17457v1)

**Authors**: Paweł Jakub Borsukiewicz, Jordan Samhi, Jacques Klein, Tegawendé F. Bissyandé

**Abstract**: The proliferation of facial recognition systems presents major privacy risks, driving the need for effective countermeasures. Current adversarial techniques apply generalized methods rather than adapting to individual facial characteristics, limiting their effectiveness and inconspicuousness. In this work, we introduce Layer Embedding Activation Mapping (LEAM), a novel technique that identifies which facial areas contribute most to recognition at an individual level. Unlike adversarial attack methods that aim to fool recognition systems, LEAM is an explainability technique designed to understand how these systems work, providing insights that could inform future privacy protection research. We integrate LEAM with a face parser to analyze data from 1000 individuals across 9 pre-trained facial recognition models.   Our analysis reveals that while different layers within facial recognition models vary significantly in their focus areas, these models generally prioritize similar facial regions across architectures when considering their overall activation patterns, which show significantly higher similarity between images of the same individual (Bhattacharyya Coefficient: 0.32-0.57) vs. different individuals (0.04-0.13), validating the existence of person-specific recognition patterns. Our results show that facial recognition models prioritize the central region of face images (with nose areas accounting for 18.9-29.7% of critical recognition regions), while still distributing attention across multiple facial fragments. Proper selection of relevant facial areas was confirmed using validation occlusions, based on just 1% of the most relevant, LEAM-identified, image pixels, which proved to be transferable across different models. Our findings establish the foundation for future individually tailored privacy protection systems centered around LEAM's choice of areas to be perturbed.



## **22. TextCrafter: Optimization-Calibrated Noise for Defending Against Text Embedding Inversion**

cs.CR

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.17302v1) [paper-pdf](http://arxiv.org/pdf/2509.17302v1)

**Authors**: Duoxun Tang, Xinhang Jiang, Jiajun Niu

**Abstract**: Text embedding inversion attacks reconstruct original sentences from latent representations, posing severe privacy threats in collaborative inference and edge computing. We propose TextCrafter, an optimization-based adversarial perturbation mechanism that combines RL learned, geometry aware noise injection orthogonal to user embeddings with cluster priors and PII signal guidance to suppress inversion while preserving task utility. Unlike prior defenses either non learnable or agnostic to perturbation direction, TextCrafter provides a directional protective policy that balances privacy and utility. Under strong privacy setting, TextCrafter maintains 70 percentage classification accuracy on four datasets and consistently outperforms Gaussian/LDP baselines across lower privacy budgets, demonstrating a superior privacy utility trade off.



## **23. Seeing is Deceiving: Mirror-Based LiDAR Spoofing for Autonomous Vehicle Deception**

cs.CR

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2509.17253v1) [paper-pdf](http://arxiv.org/pdf/2509.17253v1)

**Authors**: Selma Yahia, Ildi Alla, Girija Bangalore Mohan, Daniel Rau, Mridula Singh, Valeria Loscri

**Abstract**: Autonomous vehicles (AVs) rely heavily on LiDAR sensors for accurate 3D perception. We show a novel class of low-cost, passive LiDAR spoofing attacks that exploit mirror-like surfaces to inject or remove objects from an AV's perception. Using planar mirrors to redirect LiDAR beams, these attacks require no electronics or custom fabrication and can be deployed in real settings. We define two adversarial goals: Object Addition Attacks (OAA), which create phantom obstacles, and Object Removal Attacks (ORA), which conceal real hazards. We develop geometric optics models, validate them with controlled outdoor experiments using a commercial LiDAR and an Autoware-equipped vehicle, and implement a CARLA-based simulation for scalable testing. Experiments show mirror attacks corrupt occupancy grids, induce false detections, and trigger unsafe planning and control behaviors. We discuss potential defenses (thermal sensing, multi-sensor fusion, light-fingerprinting) and their limitations.



## **24. TraceHiding: Scalable Machine Unlearning for Mobility Data**

cs.LG

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2509.17241v1) [paper-pdf](http://arxiv.org/pdf/2509.17241v1)

**Authors**: Ali Faraji, Manos Papagelis

**Abstract**: This work introduces TraceHiding, a scalable, importance-aware machine unlearning framework for mobility trajectory data. Motivated by privacy regulations such as GDPR and CCPA granting users "the right to be forgotten," TraceHiding removes specified user trajectories from trained deep models without full retraining. It combines a hierarchical data-driven importance scoring scheme with teacher-student distillation. Importance scores--computed at token, trajectory, and user levels from statistical properties (coverage diversity, entropy, length)--quantify each training sample's impact, enabling targeted forgetting of high-impact data while preserving common patterns. The student model retains knowledge on remaining data and unlearns targeted trajectories through an importance-weighted loss that amplifies forgetting signals for unique samples and attenuates them for frequent ones. We validate on Trajectory--User Linking (TUL) tasks across three real-world higher-order mobility datasets (HO-Rome, HO-Geolife, HO-NYC) and multiple architectures (GRU, LSTM, BERT, ModernBERT, GCN-TULHOR), against strong unlearning baselines including SCRUB, NegGrad, NegGrad+, Bad-T, and Finetuning. Experiments under uniform and targeted user deletion show TraceHiding, especially its entropy-based variant, achieves superior unlearning accuracy, competitive membership inference attack (MIA) resilience, and up to 40\times speedup over retraining with minimal test accuracy loss. Results highlight robustness to adversarial deletion of high-information users and consistent performance across models. To our knowledge, this is the first systematic study of machine unlearning for trajectory data, providing a reproducible pipeline with public code and preprocessing tools.



## **25. The Cost of Compression: Tight Quadratic Black-Box Attacks on Sketches for $\ell_2$ Norm Estimation**

cs.LG

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2507.16345v2) [paper-pdf](http://arxiv.org/pdf/2507.16345v2)

**Authors**: Sara Ahmadian, Edith Cohen, Uri Stemmer

**Abstract**: Dimensionality reduction via linear sketching is a powerful and widely used technique, but it is known to be vulnerable to adversarial inputs. We study the black-box adversarial setting, where a fixed, hidden sketching matrix $A \in R^{k \times n}$ maps high-dimensional vectors $v \in R^n$ to lower-dimensional sketches $A v \in R^k$, and an adversary can query the system to obtain approximate $\ell_2$-norm estimates that are computed from the sketch. We present a universal, nonadaptive attack that, using $\tilde{O}(k^2)$ queries, either causes a failure in norm estimation or constructs an adversarial input on which the optimal estimator for the query distribution (used by the attack) fails. The attack is completely agnostic to the sketching matrix and to the estimator: it applies to any linear sketch and any query responder, including those that are randomized, adaptive, or tailored to the query distribution. Our lower bound construction tightly matches the known upper bounds of $\tilde{\Omega}(k^2)$, achieved by specialized estimators for Johnson Lindenstrauss transforms and AMS sketches. Beyond sketching, our results uncover structural parallels to adversarial attacks in image classification, highlighting fundamental vulnerabilities of compressed representations.



## **26. Bribers, Bribers on The Chain, Is Resisting All in Vain? Trustless Consensus Manipulation Through Bribing Contracts**

cs.CR

pre-print

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2509.17185v1) [paper-pdf](http://arxiv.org/pdf/2509.17185v1)

**Authors**: Bence Soóki-Tóth, István András Seres, Kamilla Kara, Ábel Nagy, Balázs Pejó, Gergely Biczók

**Abstract**: The long-term success of cryptocurrencies largely depends on the incentive compatibility provided to the validators. Bribery attacks, facilitated trustlessly via smart contracts, threaten this foundation. This work introduces, implements, and evaluates three novel and efficient bribery contracts targeting Ethereum validators. The first bribery contract enables a briber to fork the blockchain by buying votes on their proposed blocks. The second contract incentivizes validators to voluntarily exit the consensus protocol, thus increasing the adversary's relative staking power. The third contract builds a trustless bribery market that enables the briber to auction off their manipulative power over the RANDAO, Ethereum's distributed randomness beacon. Finally, we provide an initial game-theoretical analysis of one of the described bribery markets.



## **27. Unaligned Incentives: Pricing Attacks Against Blockchain Rollups**

cs.CR

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2509.17126v1) [paper-pdf](http://arxiv.org/pdf/2509.17126v1)

**Authors**: Stefanos Chaliasos, Conner Swann, Sina Pilehchiha, Nicolas Mohnblatt, Benjamin Livshits, Assimakis Kattis

**Abstract**: Rollups have become the de facto scalability solution for Ethereum, securing more than $55B in assets. They achieve scale by executing transactions on a Layer 2 ledger, while periodically posting data and finalizing state on the Layer 1, either optimistically or via validity proofs. Their fees must simultaneously reflect the pricing of three resources: L2 costs (e.g., execution), L1 DA, and underlying L1 gas costs for batch settlement and proof verification. In this work, we identify critical mis-pricings in existing rollup transaction fee mechanisms (TFMs) that allow for two powerful attacks. Firstly, an adversary can saturate the L2's DA batch capacity with compute-light data-heavy transactions, forcing low-gas transaction batches that enable both L2 DoS attacks, and finality-delay attacks. Secondly, by crafting prover killer transactions that maximize proving cycles relative to the gas charges, an adversary can effectively stall proof generation, delaying finality by hours and inflicting prover-side economic losses to the rollup at a minimal cost.   We analyze the above attack vectors across the major Ethereum rollups, quantifying adversarial costs and protocol losses. We find that the first attack enables periodic DoS on rollups, lasting up to 30 minutes, at a cost below 2 ETH for most rollups. Moreover, we identify three rollups that are exposed to indefinite DoS at a cost of approximately 0.8 to 2.7 ETH per hour. The attack can be further modified to increase finalization delays by a factor of about 1.45x to 2.73x, compared to direct L1 blob-stuffing, depending on the rollup's parameters. Furthermore, we find that the prover killer attack induces a finalization latency increase of about 94x. Finally, we propose comprehensive mitigations to prevent these attacks and suggest how some practical uses of multi-dimensional rollup TFMs can rectify the identified mis-pricing attacks.



## **28. SVeritas: Benchmark for Robust Speaker Verification under Diverse Conditions**

cs.SD

Accepted to EMNLP 2025 Findings

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2509.17091v1) [paper-pdf](http://arxiv.org/pdf/2509.17091v1)

**Authors**: Massa Baali, Sarthak Bisht, Francisco Teixeira, Kateryna Shapovalenko, Rita Singh, Bhiksha Raj

**Abstract**: Speaker verification (SV) models are increasingly integrated into security, personalization, and access control systems, yet their robustness to many real-world challenges remains inadequately benchmarked. These include a variety of natural and maliciously created conditions causing signal degradations or mismatches between enrollment and test data, impacting performance. Existing benchmarks evaluate only subsets of these conditions, missing others entirely. We introduce SVeritas, a comprehensive Speaker Verification tasks benchmark suite, assessing SV systems under stressors like recording duration, spontaneity, content, noise, microphone distance, reverberation, channel mismatches, audio bandwidth, codecs, speaker age, and susceptibility to spoofing and adversarial attacks. While several benchmarks do exist that each cover some of these issues, SVeritas is the first comprehensive evaluation that not only includes all of these, but also several other entirely new, but nonetheless important, real-life conditions that have not previously been benchmarked. We use SVeritas to evaluate several state-of-the-art SV models and observe that while some architectures maintain stability under common distortions, they suffer substantial performance degradation in scenarios involving cross-language trials, age mismatches, and codec-induced compression. Extending our analysis across demographic subgroups, we further identify disparities in robustness across age groups, gender, and linguistic backgrounds. By standardizing evaluation under realistic and synthetic stress conditions, SVeritas enables precise diagnosis of model weaknesses and establishes a foundation for advancing equitable and reliable speaker verification systems.



## **29. Revisiting Backdoor Attacks on LLMs: A Stealthy and Practical Poisoning Framework via Harmless Inputs**

cs.CL

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2505.17601v3) [paper-pdf](http://arxiv.org/pdf/2505.17601v3)

**Authors**: Jiawei Kong, Hao Fang, Xiaochen Yang, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Ke Xu, Han Qiu

**Abstract**: Recent studies have widely investigated backdoor attacks on Large language models (LLMs) by inserting harmful question-answer (QA) pairs into training data to implant triggers. However, we revisit existing attack methods and identify two critical limitations of that seriously undermine their stealthiness and practicality: (1) directly embedding harmful content into the training data compromise the model's safety alignment, resulting in high attack success rates even for clean queries without triggers, and (2) the poisoned training samples can be easily detected and filtered by safety-aligned guardrails (e.g., LLaMAGuard). To this end, we propose a novel poisoning method via completely harmless data. Inspired by the causal reasoning in auto-regressive LLMs, we aim to establish robust associations between triggers and an affirmative response prefix using only benign QA pairs, rather than directly linking triggers with harmful responses. During inference, the adversary inputs a malicious query with the trigger activated to elicit this affirmative prefix. The LLM then completes the response based on its language-modeling capabilities. Notably, achieving this behavior from clean QA pairs is non-trivial. We observe an interesting resistance phenomenon where the LLM initially appears to agree but subsequently refuses to answer. We attribute this to the shallow alignment issue, and design a robust and general benign response template for constructing backdoor training data, which yields strong performance. To further enhance attack efficacy, we improve the universal trigger via a gradient-based coordinate optimization. Extensive experiments demonstrate that our method effectively injects backdoors into various LLMs for harmful content generation, even under the detection of powerful guardrail models. E.g., ASRs of 86.67% and 85% on LLaMA-3-8B and Qwen-2.5-7B judged by GPT-4o.



## **30. Breaking the Reviewer: Assessing the Vulnerability of Large Language Models in Automated Peer Review Under Textual Adversarial Attacks**

cs.CL

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2506.11113v2) [paper-pdf](http://arxiv.org/pdf/2506.11113v2)

**Authors**: Tzu-Ling Lin, Wei-Chih Chen, Teng-Fang Hsiao, Hou-I Liu, Ya-Hsin Yeh, Yu Kai Chan, Wen-Sheng Lien, Po-Yen Kuo, Philip S. Yu, Hong-Han Shuai

**Abstract**: Peer review is essential for maintaining academic quality, but the increasing volume of submissions places a significant burden on reviewers. Large language models (LLMs) offer potential assistance in this process, yet their susceptibility to textual adversarial attacks raises reliability concerns. This paper investigates the robustness of LLMs used as automated reviewers in the presence of such attacks. We focus on three key questions: (1) The effectiveness of LLMs in generating reviews compared to human reviewers. (2) The impact of adversarial attacks on the reliability of LLM-generated reviews. (3) Challenges and potential mitigation strategies for LLM-based review. Our evaluation reveals significant vulnerabilities, as text manipulations can distort LLM assessments. We offer a comprehensive evaluation of LLM performance in automated peer reviewing and analyze its robustness against adversarial attacks. Our findings emphasize the importance of addressing adversarial risks to ensure AI strengthens, rather than compromises, the integrity of scholarly communication.



## **31. BlockA2A: Towards Secure and Verifiable Agent-to-Agent Interoperability**

cs.CR

43 pages

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2508.01332v3) [paper-pdf](http://arxiv.org/pdf/2508.01332v3)

**Authors**: Zhenhua Zou, Zhuotao Liu, Lepeng Zhao, Qiuyang Zhan

**Abstract**: The rapid adoption of agentic AI, powered by large language models (LLMs), is transforming enterprise ecosystems with autonomous agents that execute complex workflows. Yet we observe several key security vulnerabilities in LLM-driven multi-agent systems (MASes): fragmented identity frameworks, insecure communication channels, and inadequate defenses against Byzantine agents or adversarial prompts. In this paper, we present the first systematic analysis of these emerging multi-agent risks and explain why the legacy security strategies cannot effectively address these risks. Afterwards, we propose BlockA2A, the first unified multi-agent trust framework that enables secure and verifiable and agent-to-agent interoperability. At a high level, BlockA2A adopts decentralized identifiers (DIDs) to enable fine-grained cross-domain agent authentication, blockchain-anchored ledgers to enable immutable auditability, and smart contracts to dynamically enforce context-aware access control policies. BlockA2A eliminates centralized trust bottlenecks, ensures message authenticity and execution integrity, and guarantees accountability across agent interactions. Furthermore, we propose a Defense Orchestration Engine (DOE) that actively neutralizes attacks through real-time mechanisms, including Byzantine agent flagging, reactive execution halting, and instant permission revocation. Empirical evaluations demonstrate BlockA2A's effectiveness in neutralizing prompt-based, communication-based, behavioral and systemic MAS attacks. We formalize its integration into existing MAS and showcase a practical implementation for Google's A2A protocol. Experiments confirm that BlockA2A and DOE operate with sub-second overhead, enabling scalable deployment in production LLM-based MAS environments.



## **32. Dynamical Low-Rank Compression of Neural Networks with Robustness under Adversarial Attacks**

cs.LG

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2505.08022v3) [paper-pdf](http://arxiv.org/pdf/2505.08022v3)

**Authors**: Steffen Schotthöfer, H. Lexie Yang, Stefan Schnake

**Abstract**: Deployment of neural networks on resource-constrained devices demands models that are both compact and robust to adversarial inputs. However, compression and adversarial robustness often conflict. In this work, we introduce a dynamical low-rank training scheme enhanced with a novel spectral regularizer that controls the condition number of the low-rank core in each layer. This approach mitigates the sensitivity of compressed models to adversarial perturbations without sacrificing accuracy on clean data. The method is model- and data-agnostic, computationally efficient, and supports rank adaptivity to automatically compress the network at hand. Extensive experiments across standard architectures, datasets, and adversarial attacks show the regularized networks can achieve over 94% compression while recovering or improving adversarial accuracy relative to uncompressed baselines.



## **33. Securing the Language of Life: Inheritable Watermarks from DNA Language Models to Proteins**

q-bio.GN

Accepted by NeurIPS 2025

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2509.18207v1) [paper-pdf](http://arxiv.org/pdf/2509.18207v1)

**Authors**: Zaixi Zhang, Ruofan Jin, Le Cong, Mengdi Wang

**Abstract**: DNA language models have revolutionized our ability to understand and design DNA sequences--the fundamental language of life--with unprecedented precision, enabling transformative applications in therapeutics, synthetic biology, and gene editing. However, this capability also poses substantial dual-use risks, including the potential for creating pathogens, viruses, and even bioweapons. To address these biosecurity challenges, we introduce two innovative watermarking techniques to reliably track the designed DNA: DNAMark and CentralMark. DNAMark employs synonymous codon substitutions to embed watermarks in DNA sequences while preserving the original function. CentralMark further advances this by creating inheritable watermarks that transfer from DNA to translated proteins, leveraging protein embeddings to ensure detection across the central dogma. Both methods utilize semantic embeddings to generate watermark logits, enhancing robustness against natural mutations, synthesis errors, and adversarial attacks. Evaluated on our therapeutic DNA benchmark, DNAMark and CentralMark achieve F1 detection scores above 0.85 under various conditions, while maintaining over 60% sequence similarity to ground truth and degeneracy scores below 15%. A case study on the CRISPR-Cas9 system underscores CentralMark's utility in real-world settings. This work establishes a vital framework for securing DNA language models, balancing innovation with accountability to mitigate biosecurity risks.



## **34. On the Robustness of RSMA to Adversarial BD-RIS-Induced Interference**

eess.SP

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2505.20146v2) [paper-pdf](http://arxiv.org/pdf/2505.20146v2)

**Authors**: Arthur S. de Sena, Jacek Kibilda, Nurul H. Mahmood, Andre Gomes, Luiz A. DaSilva, Matti Latva-aho

**Abstract**: This article investigates the robustness of rate-splitting multiple access (RSMA) in multi-user multiple-input single-output (MISO) systems to interference attacks against channel acquisition induced by beyond-diagonal RISs (BD-RISs). Two primary attack strategies, random and aligned interference, are proposed for fully connected and group-connected reconfigurable intelligent surface (RIS) architectures. Valid random reflection coefficients are generated exploiting the Takagi factorization, while potent aligned interference attacks are achieved through optimization strategies based on a quadratically constrained quadratic program (QCQP) reformulation followed by projections onto the unitary manifold. Our numerical findings reveal that, when perfect channel state information (CSI) is available, RSMA behaves similarly to space-division multiple access (SDMA) and thus is highly susceptible to the attack, with BD-RIS inducing severe performance loss and significantly outperforming diagonal RIS. However, under imperfect CSI, RSMA consistently demonstrates significantly greater robustness than SDMA, particularly as the system's transmit power increases.



## **35. ADVEDM:Fine-grained Adversarial Attack against VLM-based Embodied Agents**

cs.CV

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2509.16645v1) [paper-pdf](http://arxiv.org/pdf/2509.16645v1)

**Authors**: Yichen Wang, Hangtao Zhang, Hewen Pan, Ziqi Zhou, Xianlong Wang, Peijin Guo, Lulu Xue, Shengshan Hu, Minghui Li, Leo Yu Zhang

**Abstract**: Vision-Language Models (VLMs), with their strong reasoning and planning capabilities, are widely used in embodied decision-making (EDM) tasks in embodied agents, such as autonomous driving and robotic manipulation. Recent research has increasingly explored adversarial attacks on VLMs to reveal their vulnerabilities. However, these attacks either rely on overly strong assumptions, requiring full knowledge of the victim VLM, which is impractical for attacking VLM-based agents, or exhibit limited effectiveness. The latter stems from disrupting most semantic information in the image, which leads to a misalignment between the perception and the task context defined by system prompts. This inconsistency interrupts the VLM's reasoning process, resulting in invalid outputs that fail to affect interactions in the physical world. To this end, we propose a fine-grained adversarial attack framework, ADVEDM, which modifies the VLM's perception of only a few key objects while preserving the semantics of the remaining regions. This attack effectively reduces conflicts with the task context, making VLMs output valid but incorrect decisions and affecting the actions of agents, thus posing a more substantial safety threat in the physical world. We design two variants of based on this framework, ADVEDM-R and ADVEDM-A, which respectively remove the semantics of a specific object from the image and add the semantics of a new object into the image. The experimental results in both general scenarios and EDM tasks demonstrate fine-grained control and excellent attack performance.



## **36. Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking**

cs.CR

Accepted by EMNLP2025

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2504.05652v3) [paper-pdf](http://arxiv.org/pdf/2504.05652v3)

**Authors**: Yu-Hang Wu, Yu-Jie Xiong, Hao Zhang, Jia-Chen Zhang, Zheng Zhou

**Abstract**: With the increasingly deep integration of large language models (LLMs) across diverse domains, the effectiveness of their safety mechanisms is encountering severe challenges. Currently, jailbreak attacks based on prompt engineering have become a major safety threat. However, existing methods primarily rely on black-box manipulation of prompt templates, resulting in poor interpretability and limited generalization. To break through the bottleneck, this study first introduces the concept of Defense Threshold Decay (DTD), revealing the potential safety impact caused by LLMs' benign generation: as benign content generation in LLMs increases, the model's focus on input instructions progressively diminishes. Building on this insight, we propose the Sugar-Coated Poison (SCP) attack paradigm, which uses a "semantic reversal" strategy to craft benign inputs that are opposite in meaning to malicious intent. This strategy induces the models to generate extensive benign content, thereby enabling adversarial reasoning to bypass safety mechanisms. Experiments show that SCP outperforms existing baselines. Remarkably, it achieves an average attack success rate of 87.23% across six LLMs. For defense, we propose Part-of-Speech Defense (POSD), leveraging verb-noun dependencies for syntactic analysis to enhance safety of LLMs while preserving their generalization ability.



## **37. Can an Individual Manipulate the Collective Decisions of Multi-Agents?**

cs.CL

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2509.16494v1) [paper-pdf](http://arxiv.org/pdf/2509.16494v1)

**Authors**: Fengyuan Liu, Rui Zhao, Shuo Chen, Guohao Li, Philip Torr, Lei Han, Jindong Gu

**Abstract**: Individual Large Language Models (LLMs) have demonstrated significant capabilities across various domains, such as healthcare and law. Recent studies also show that coordinated multi-agent systems exhibit enhanced decision-making and reasoning abilities through collaboration. However, due to the vulnerabilities of individual LLMs and the difficulty of accessing all agents in a multi-agent system, a key question arises: If attackers only know one agent, could they still generate adversarial samples capable of misleading the collective decision? To explore this question, we formulate it as a game with incomplete information, where attackers know only one target agent and lack knowledge of the other agents in the system. With this formulation, we propose M-Spoiler, a framework that simulates agent interactions within a multi-agent system to generate adversarial samples. These samples are then used to manipulate the target agent in the target system, misleading the system's collaborative decision-making process. More specifically, M-Spoiler introduces a stubborn agent that actively aids in optimizing adversarial samples by simulating potential stubborn responses from agents in the target system. This enhances the effectiveness of the generated adversarial samples in misleading the system. Through extensive experiments across various tasks, our findings confirm the risks posed by the knowledge of an individual agent in multi-agent systems and demonstrate the effectiveness of our framework. We also explore several defense mechanisms, showing that our proposed attack framework remains more potent than baselines, underscoring the need for further research into defensive strategies.



## **38. Targeting Alignment: Extracting Safety Classifiers of Aligned LLMs**

cs.CR

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2501.16534v2) [paper-pdf](http://arxiv.org/pdf/2501.16534v2)

**Authors**: Jean-Charles Noirot Ferrand, Yohan Beugin, Eric Pauley, Ryan Sheatsley, Patrick McDaniel

**Abstract**: Alignment in large language models (LLMs) is used to enforce guidelines such as safety. Yet, alignment fails in the face of jailbreak attacks that modify inputs to induce unsafe outputs. In this paper, we introduce and evaluate a new technique for jailbreak attacks. We observe that alignment embeds a safety classifier in the LLM responsible for deciding between refusal and compliance, and seek to extract an approximation of this classifier: a surrogate classifier. To this end, we build candidate classifiers from subsets of the LLM. We first evaluate the degree to which candidate classifiers approximate the LLM's safety classifier in benign and adversarial settings. Then, we attack the candidates and measure how well the resulting adversarial inputs transfer to the LLM. Our evaluation shows that the best candidates achieve accurate agreement (an F1 score above 80%) using as little as 20% of the model architecture. Further, we find that attacks mounted on the surrogate classifiers can be transferred to the LLM with high success. For example, a surrogate using only 50% of the Llama 2 model achieved an attack success rate (ASR) of 70% with half the memory footprint and runtime -- a substantial improvement over attacking the LLM directly, where we only observed a 22% ASR. These results show that extracting surrogate classifiers is an effective and efficient means for modeling (and therein addressing) the vulnerability of aligned models to jailbreaking attacks.



## **39. Secure Confidential Business Information When Sharing Machine Learning Models**

cs.CR

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.16352v1) [paper-pdf](http://arxiv.org/pdf/2509.16352v1)

**Authors**: Yunfan Yang, Jiarong Xu, Hongzhe Zhang, Xiao Fang

**Abstract**: Model-sharing offers significant business value by enabling firms with well-established Machine Learning (ML) models to monetize and share their models with others who lack the resources to develop ML models from scratch. However, concerns over data confidentiality remain a significant barrier to model-sharing adoption, as Confidential Property Inference (CPI) attacks can exploit shared ML models to uncover confidential properties of the model provider's private model training data. Existing defenses often assume that CPI attacks are non-adaptive to the specific ML model they are targeting. This assumption overlooks a key characteristic of real-world adversaries: their responsiveness, i.e., adversaries' ability to dynamically adjust their attack models based on the information of the target and its defenses. To overcome this limitation, we propose a novel defense method that explicitly accounts for the responsive nature of real-world adversaries via two methodological innovations: a novel Responsive CPI attack and an attack-defense arms race framework. The former emulates the responsive behaviors of adversaries in the real world, and the latter iteratively enhances both the target and attack models, ultimately producing a secure ML model that is robust against responsive CPI attacks. Furthermore, we propose and integrate a novel approximate strategy into our defense, which addresses a critical computational bottleneck of defense methods and improves defense efficiency. Through extensive empirical evaluations across various realistic model-sharing scenarios, we demonstrate that our method outperforms existing defenses by more effectively defending against CPI attacks, preserving ML model utility, and reducing computational overhead.



## **40. Robust Vision-Language Models via Tensor Decomposition: A Defense Against Adversarial Attacks**

cs.CV

To be presented as a poster at the Workshop on Safe and Trustworthy  Multimodal AI Systems (SafeMM-AI), 2025

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.16163v1) [paper-pdf](http://arxiv.org/pdf/2509.16163v1)

**Authors**: Het Patel, Muzammil Allie, Qian Zhang, Jia Chen, Evangelos E. Papalexakis

**Abstract**: Vision language models (VLMs) excel in multimodal understanding but are prone to adversarial attacks. Existing defenses often demand costly retraining or significant architecture changes. We introduce a lightweight defense using tensor decomposition suitable for any pre-trained VLM, requiring no retraining. By decomposing and reconstructing vision encoder representations, it filters adversarial noise while preserving meaning. Experiments with CLIP on COCO and Flickr30K show improved robustness. On Flickr30K, it restores 12.3\% performance lost to attacks, raising Recall@1 accuracy from 7.5\% to 19.8\%. On COCO, it recovers 8.1\% performance, improving accuracy from 3.8\% to 11.9\%. Analysis shows Tensor Train decomposition with low rank (8-32) and low residual strength ($\alpha=0.1-0.2$) is optimal. This method is a practical, plug-and-play solution with minimal overhead for existing VLMs.



## **41. Randomized Smoothing Meets Vision-Language Models**

cs.LG

EMNLP'25 full version, including appendix (proofs, additional  experiments)

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.16088v1) [paper-pdf](http://arxiv.org/pdf/2509.16088v1)

**Authors**: Emmanouil Seferis, Changshun Wu, Stefanos Kollias, Saddek Bensalem, Chih-Hong Cheng

**Abstract**: Randomized smoothing (RS) is one of the prominent techniques to ensure the correctness of machine learning models, where point-wise robustness certificates can be derived analytically. While RS is well understood for classification, its application to generative models is unclear, since their outputs are sequences rather than labels. We resolve this by connecting generative outputs to an oracle classification task and showing that RS can still be enabled: the final response can be classified as a discrete action (e.g., service-robot commands in VLAs), as harmful vs. harmless (content moderation or toxicity detection in VLMs), or even applying oracles to cluster answers into semantically equivalent ones. Provided that the error rate for the oracle classifier comparison is bounded, we develop the theory that associates the number of samples with the corresponding robustness radius. We further derive improved scaling laws analytically relating the certified radius and accuracy to the number of samples, showing that the earlier result of 2 to 3 orders of magnitude fewer samples sufficing with minimal loss remains valid even under weaker assumptions. Together, these advances make robustness certification both well-defined and computationally feasible for state-of-the-art VLMs, as validated against recent jailbreak-style adversarial attacks.



## **42. Attention Schema-based Attention Control (ASAC): A Cognitive-Inspired Approach for Attention Management in Transformers**

cs.AI

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.16058v1) [paper-pdf](http://arxiv.org/pdf/2509.16058v1)

**Authors**: Krati Saxena, Federico Jurado Ruiz, Guido Manzi, Dianbo Liu, Alex Lamb

**Abstract**: Attention mechanisms have become integral in AI, significantly enhancing model performance and scalability by drawing inspiration from human cognition. Concurrently, the Attention Schema Theory (AST) in cognitive science posits that individuals manage their attention by creating a model of the attention itself, effectively allocating cognitive resources. Inspired by AST, we introduce ASAC (Attention Schema-based Attention Control), which integrates the attention schema concept into artificial neural networks. Our initial experiments focused on embedding the ASAC module within transformer architectures. This module employs a Vector-Quantized Variational AutoEncoder (VQVAE) as both an attention abstractor and controller, facilitating precise attention management. By explicitly modeling attention allocation, our approach aims to enhance system efficiency. We demonstrate ASAC's effectiveness in both the vision and NLP domains, highlighting its ability to improve classification accuracy and expedite the learning process. Our experiments with vision transformers across various datasets illustrate that the attention controller not only boosts classification accuracy but also accelerates learning. Furthermore, we have demonstrated the model's robustness and generalization capabilities across noisy and out-of-distribution datasets. In addition, we have showcased improved performance in multi-task settings. Quick experiments reveal that the attention schema-based module enhances resilience to adversarial attacks, optimizes attention to improve learning efficiency, and facilitates effective transfer learning and learning from fewer examples. These promising results establish a connection between cognitive science and machine learning, shedding light on the efficient utilization of attention mechanisms in AI systems.



## **43. Swarm Oracle: Trustless Blockchain Agreements through Robot Swarms**

cs.RO

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.15956v1) [paper-pdf](http://arxiv.org/pdf/2509.15956v1)

**Authors**: Alexandre Pacheco, Hanqing Zhao, Volker Strobel, Tarik Roukny, Gregory Dudek, Andreagiovanni Reina, Marco Dorigo

**Abstract**: Blockchain consensus, rooted in the principle ``don't trust, verify'', limits access to real-world data, which may be ambiguous or inaccessible to some participants. Oracles address this limitation by supplying data to blockchains, but existing solutions may reduce autonomy, transparency, or reintroduce the need for trust. We propose Swarm Oracle: a decentralized network of autonomous robots -- that is, a robot swarm -- that use onboard sensors and peer-to-peer communication to collectively verify real-world data and provide it to smart contracts on public blockchains. Swarm Oracle leverages the built-in decentralization, fault tolerance and mobility of robot swarms, which can flexibly adapt to meet information requests on-demand, even in remote locations. Unlike typical cooperative robot swarms, Swarm Oracle integrates robots from multiple stakeholders, protecting the system from single-party biases but also introducing potential adversarial behavior. To ensure the secure, trustless and global consensus required by blockchains, we employ a Byzantine fault-tolerant protocol that enables robots from different stakeholders to operate together, reaching social agreements of higher quality than the estimates of individual robots. Through extensive experiments using both real and simulated robots, we showcase how consensus on uncertain environmental information can be achieved, despite several types of attacks orchestrated by large proportions of the robots, and how a reputation system based on blockchain tokens lets Swarm Oracle autonomously recover from faults and attacks, a requirement for long-term operation.



## **44. Bridging Batch and Streaming Estimations to System Identification under Adversarial Attacks**

math.OC

15 pages, 2 figures

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.15794v1) [paper-pdf](http://arxiv.org/pdf/2509.15794v1)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: System identification in modern engineering systems faces emerging challenges from unanticipated adversarial attacks beyond existing detection mechanisms. In this work, we obtain a provably accurate estimate of the Markov parameter matrix of order $k$ to identify partially observed linear systems, in which the probability of having an attack at each time is $O(1/k)$. We show that given the batch data accumulated up to time $T^*$, the $\ell_2$-norm estimator achieves an error decaying exponentially as $k$ grows. We then propose a stochastic projected subgradient descent algorithm on streaming data that produces an estimate at each time $t<T^*$, in which case the expected estimation error proves to be the larger of $O(k/\sqrt{t})$ and an exponentially decaying term in $k$. This stochastic approach illustrates how non-smooth estimators can leverage first-order methods despite lacking recursive formulas. Finally, we integrate batch and streaming estimations to recover the Hankel matrix using the appropriate estimates of the Markov parameter matrix, which enables the synthesis of a robust adaptive controller based on the estimated balanced truncated model under adversarial attacks.



## **45. Explainable Deep Learning Based Adversarial Defense for Automatic Modulation Classification**

eess.SP

Accepted by IEEE Internet of Things Journal

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.15766v1) [paper-pdf](http://arxiv.org/pdf/2509.15766v1)

**Authors**: Peihao Dong, Jingchun Wang, Shen Gao, Fuhui Zhou, Qihui Wu

**Abstract**: Deep learning (DL) has been widely applied to enhance automatic modulation classification (AMC). However, the elaborate AMC neural networks are susceptible to various adversarial attacks, which are challenging to handle due to the generalization capability and computational cost. In this article, an explainable DL based defense scheme, called SHapley Additive exPlanation enhanced Adversarial Fine-Tuning (SHAP-AFT), is developed in the perspective of disclosing the attacking impact on the AMC network. By introducing the concept of cognitive negative information, the motivation of using SHAP for defense is theoretically analyzed first. The proposed scheme includes three stages, i.e., the attack detection, the information importance evaluation, and the AFT. The first stage indicates the existence of the attack. The second stage evaluates contributions of the received data and removes those data positions using negative Shapley values corresponding to the dominating negative information caused by the attack. Then the AMC network is fine-tuned based on adversarial adaptation samples using the refined received data pattern. Simulation results show the effectiveness of the Shapley value as the key indicator as well as the superior defense performance of the proposed SHAP-AFT scheme in face of different attack types and intensities.



## **46. An Adversarial Robust Behavior Sequence Anomaly Detection Approach Based on Critical Behavior Unit Learning**

cs.CR

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.15756v1) [paper-pdf](http://arxiv.org/pdf/2509.15756v1)

**Authors**: Dongyang Zhan, Kai Tan, Lin Ye, Xiangzhan Yu, Hongli Zhang, Zheng He

**Abstract**: Sequential deep learning models (e.g., RNN and LSTM) can learn the sequence features of software behaviors, such as API or syscall sequences. However, recent studies have shown that these deep learning-based approaches are vulnerable to adversarial samples. Attackers can use adversarial samples to change the sequential characteristics of behavior sequences and mislead malware classifiers. In this paper, an adversarial robustness anomaly detection method based on the analysis of behavior units is proposed to overcome this problem. We extract related behaviors that usually perform a behavior intention as a behavior unit, which contains the representative semantic information of local behaviors and can be used to improve the robustness of behavior analysis. By learning the overall semantics of each behavior unit and the contextual relationships among behavior units based on a multilevel deep learning model, our approach can mitigate perturbation attacks that target local and large-scale behaviors. In addition, our approach can be applied to both low-level and high-level behavior logs (e.g., API and syscall logs). The experimental results show that our approach outperforms all the compared methods, which indicates that our approach has better performance against obfuscation attacks.



## **47. Chernoff Information as a Privacy Constraint for Adversarial Classification and Membership Advantage**

cs.IT

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2403.10307v4) [paper-pdf](http://arxiv.org/pdf/2403.10307v4)

**Authors**: Ayşe Ünsal

**Abstract**: This work inspects a privacy metric based on Chernoff information, namely Chernoff differential privacy, due to its significance in characterization of the optimal classifier's performance. Adversarial classification, as any other classification problem is built around minimization of the (average or correct detection) probability of error in deciding on either of the classes in the case of binary classification. Unlike the classical hypothesis testing problem, where the false alarm and mis-detection probabilities are handled separately resulting in an asymmetric behavior of the best error exponent, in this work, we characterize the relationship between $\varepsilon\textrm{-}$differential privacy, the best error exponent of one of the errors (when the other is fixed) and the best average error exponent. Accordingly, we re-derive Chernoff differential privacy in connection with $\varepsilon\textrm{-}$differential privacy using the Radon-Nikodym derivative, and prove its relation with Kullback-Leibler (KL) differential privacy. Subsequently, we present numerical evaluation results, which demonstrates that Chernoff information outperforms Kullback-Leibler divergence as a function of the privacy parameter $\varepsilon$ and the impact of the adversary's attack in Laplace mechanisms. Lastly, we introduce a new upper bound on adversary's membership advantage in membership inference attacks using Chernoff DP and numerically compare its performance with existing alternatives based on $(\varepsilon, \delta)\textrm{-}$differential privacy in the literature.



## **48. Inference Attacks on Encrypted Online Voting via Traffic Analysis**

cs.CR

Accepted at ISC 2025

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.15694v1) [paper-pdf](http://arxiv.org/pdf/2509.15694v1)

**Authors**: Anastasiia Belousova, Francesco Marchiori, Mauro Conti

**Abstract**: Online voting enables individuals to participate in elections remotely, offering greater efficiency and accessibility in both governmental and organizational settings. As this method gains popularity, ensuring the security of online voting systems becomes increasingly vital, as the systems supporting it must satisfy a demanding set of security requirements. Most research in this area emphasizes the design and verification of cryptographic protocols to protect voter integrity and system confidentiality. However, other vectors, such as network traffic analysis, remain relatively understudied, even though they may pose significant threats to voter privacy and the overall trustworthiness of the system.   In this paper, we examine how adversaries can exploit metadata from encrypted network traffic to uncover sensitive information during online voting. Our analysis reveals that, even without accessing the encrypted content, it is possible to infer critical voter actions, such as whether a person votes, the exact moment a ballot is submitted, and whether the ballot is valid or spoiled. We test these attacks with both rule-based techniques and machine learning methods. We evaluate our attacks on two widely used online voting platforms, one proprietary and one partially open source, achieving classification accuracy as high as 99.5%. These results expose a significant privacy vulnerability that threatens key properties of secure elections, including voter secrecy and protection against coercion or vote-buying. We explore mitigations to our attacks, demonstrating that countermeasures such as payload padding and timestamp equalization can substantially limit their effectiveness.



## **49. AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender**

cs.CR

19 pages, 6 figures, 10 tables

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2504.09466v2) [paper-pdf](http://arxiv.org/pdf/2504.09466v2)

**Authors**: Weixiang Zhao, Jiahe Guo, Yulin Hu, Yang Deng, An Zhang, Xingyu Sui, Xinyang Han, Yanyan Zhao, Bing Qin, Tat-Seng Chua, Ting Liu

**Abstract**: Despite extensive efforts in safety alignment, large language models (LLMs) remain vulnerable to jailbreak attacks. Activation steering offers a training-free defense method but relies on fixed steering coefficients, resulting in suboptimal protection and increased false rejections of benign inputs. To address this, we propose AdaSteer, an adaptive activation steering method that dynamically adjusts model behavior based on input characteristics. We identify two key properties: Rejection Law (R-Law), which shows that stronger steering is needed for jailbreak inputs opposing the rejection direction, and Harmfulness Law (H-Law), which differentiates adversarial and benign inputs. AdaSteer steers input representations along both the Rejection Direction (RD) and Harmfulness Direction (HD), with adaptive coefficients learned via logistic regression, ensuring robust jailbreak defense while preserving benign input handling. Experiments on LLaMA-3.1, Gemma-2, and Qwen2.5 show that AdaSteer outperforms baseline methods across multiple jailbreak attacks with minimal impact on utility. Our results highlight the potential of interpretable model internals for real-time, flexible safety enforcement in LLMs.



## **50. DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm**

cs.CL

NeurIPS 2025 Spotlight

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.15550v1) [paper-pdf](http://arxiv.org/pdf/2509.15550v1)

**Authors**: Xiaowei Zhu, Yubing Ren, Fang Fang, Qingfeng Tan, Shi Wang, Yanan Cao

**Abstract**: The rapid advancement of large language models (LLMs) has blurred the line between AI-generated and human-written text. This progress brings societal risks such as misinformation, authorship ambiguity, and intellectual property concerns, highlighting the urgent need for reliable AI-generated text detection methods. However, recent advances in generative language modeling have resulted in significant overlap between the feature distributions of human-written and AI-generated text, blurring classification boundaries and making accurate detection increasingly challenging. To address the above challenges, we propose a DNA-inspired perspective, leveraging a repair-based process to directly and interpretably capture the intrinsic differences between human-written and AI-generated text. Building on this perspective, we introduce DNA-DetectLLM, a zero-shot detection method for distinguishing AI-generated and human-written text. The method constructs an ideal AI-generated sequence for each input, iteratively repairs non-optimal tokens, and quantifies the cumulative repair effort as an interpretable detection signal. Empirical evaluations demonstrate that our method achieves state-of-the-art detection performance and exhibits strong robustness against various adversarial attacks and input lengths. Specifically, DNA-DetectLLM achieves relative improvements of 5.55% in AUROC and 2.08% in F1 score across multiple public benchmark datasets.



