# Latest Adversarial Attack Papers
**update at 2026-01-19 16:16:59**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. QUPID: A Partitioned Quantum Neural Network for Anomaly Detection in Smart Grid**

cs.LG

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.11500v1) [paper-pdf](https://arxiv.org/pdf/2601.11500v1)

**Authors**: Hoang M. Ngo, Tre' R. Jeter, Jung Taek Seo, My T. Thai

**Abstract**: Smart grid infrastructures have revolutionized energy distribution, but their day-to-day operations require robust anomaly detection methods to counter risks associated with cyber-physical threats and system faults potentially caused by natural disasters, equipment malfunctions, and cyber attacks. Conventional machine learning (ML) models are effective in several domains, yet they struggle to represent the complexities observed in smart grid systems. Furthermore, traditional ML models are highly susceptible to adversarial manipulations, making them increasingly unreliable for real-world deployment. Quantum ML (QML) provides a unique advantage, utilizing quantum-enhanced feature representations to model the intricacies of the high-dimensional nature of smart grid systems while demonstrating greater resilience to adversarial manipulation. In this work, we propose QUPID, a partitioned quantum neural network (PQNN) that outperforms traditional state-of-the-art ML models in anomaly detection. We extend our model to R-QUPID that even maintains its performance when including differential privacy (DP) for enhanced robustness. Moreover, our partitioning framework addresses a significant scalability problem in QML by efficiently distributing computational workloads, making quantum-enhanced anomaly detection practical in large-scale smart grid environments. Our experimental results across various scenarios exemplifies the efficacy of QUPID and R-QUPID to significantly improve anomaly detection capabilities and robustness compared to traditional ML approaches.



## **2. Backdoor Attacks on Multi-modal Contrastive Learning**

cs.LG

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.11006v1) [paper-pdf](https://arxiv.org/pdf/2601.11006v1)

**Authors**: Simi D Kuniyilh, Rita Machacy

**Abstract**: Contrastive learning has become a leading self- supervised approach to representation learning across domains, including vision, multimodal settings, graphs, and federated learning. However, recent studies have shown that contrastive learning is susceptible to backdoor and data poisoning attacks. In these attacks, adversaries can manipulate pretraining data or model updates to insert hidden malicious behavior. This paper offers a thorough and comparative review of backdoor attacks in contrastive learning. It analyzes threat models, attack methods, target domains, and available defenses. We summarize recent advancements in this area, underline the specific vulnerabilities inherent to contrastive learning, and discuss the challenges and future research directions. Our findings have significant implications for the secure deployment of systems in industrial and distributed environments.



## **3. AJAR: Adaptive Jailbreak Architecture for Red-teaming**

cs.CR

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.10971v1) [paper-pdf](https://arxiv.org/pdf/2601.10971v1)

**Authors**: Yipu Dou, Wang Yang

**Abstract**: As Large Language Models (LLMs) evolve from static chatbots into autonomous agents capable of tool execution, the landscape of AI safety is shifting from content moderation to action security. However, existing red-teaming frameworks remain bifurcated: they either focus on rigid, script-based text attacks or lack the architectural modularity to simulate complex, multi-turn agentic exploitations. In this paper, we introduce AJAR (Adaptive Jailbreak Architecture for Red-teaming), a proof-of-concept framework designed to bridge this gap through Protocol-driven Cognitive Orchestration. Built upon the robust runtime of Petri, AJAR leverages the Model Context Protocol (MCP) to decouple adversarial logic from the execution loop, encapsulating state-of-the-art algorithms like X-Teaming as standardized, plug-and-play services. We validate the architectural feasibility of AJAR through a controlled qualitative case study, demonstrating its ability to perform stateful backtracking within a tool-use environment. Furthermore, our preliminary exploration of the "Agentic Gap" reveals a complex safety dynamic: while tool usage introduces new injection vectors via code execution, the cognitive load of parameter formatting can inadvertently disrupt persona-based attacks. AJAR is open-sourced to facilitate the standardized, environment-aware evaluation of this emerging attack surface. The code and data are available at https://github.com/douyipu/ajar.



## **4. SecMLOps: A Comprehensive Framework for Integrating Security Throughout the MLOps Lifecycle**

cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10848v1) [paper-pdf](https://arxiv.org/pdf/2601.10848v1)

**Authors**: Xinrui Zhang, Pincan Zhao, Jason Jaskolka, Heng Li, Rongxing Lu

**Abstract**: Machine Learning (ML) has emerged as a pivotal technology in the operation of large and complex systems, driving advancements in fields such as autonomous vehicles, healthcare diagnostics, and financial fraud detection. Despite its benefits, the deployment of ML models brings significant security challenges, such as adversarial attacks, which can compromise the integrity and reliability of these systems. To address these challenges, this paper builds upon the concept of Secure Machine Learning Operations (SecMLOps), providing a comprehensive framework designed to integrate robust security measures throughout the entire ML operations (MLOps) lifecycle. SecMLOps builds on the principles of MLOps by embedding security considerations from the initial design phase through to deployment and continuous monitoring. This framework is particularly focused on safeguarding against sophisticated attacks that target various stages of the MLOps lifecycle, thereby enhancing the resilience and trustworthiness of ML applications. A detailed advanced pedestrian detection system (PDS) use case demonstrates the practical application of SecMLOps in securing critical MLOps. Through extensive empirical evaluations, we highlight the trade-offs between security measures and system performance, providing critical insights into optimizing security without unduly impacting operational efficiency. Our findings underscore the importance of a balanced approach, offering valuable guidance for practitioners on how to achieve an optimal balance between security and performance in ML deployments across various domains.



## **5. Be Your Own Red Teamer: Safety Alignment via Self-Play and Reflective Experience Replay**

cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10589v1) [paper-pdf](https://arxiv.org/pdf/2601.10589v1)

**Authors**: Hao Wang, Yanting Wang, Hao Li, Rui Li, Lei Sha

**Abstract**: Large Language Models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial ``jailbreak'' attacks designed to bypass safety guardrails. Current safety alignment methods depend heavily on static external red teaming, utilizing fixed defense prompts or pre-collected adversarial datasets. This leads to a rigid defense that overfits known patterns and fails to generalize to novel, sophisticated threats. To address this critical limitation, we propose empowering the model to be its own red teamer, capable of achieving autonomous and evolving adversarial attacks. Specifically, we introduce Safety Self- Play (SSP), a system that utilizes a single LLM to act concurrently as both the Attacker (generating jailbreaks) and the Defender (refusing harmful requests) within a unified Reinforcement Learning (RL) loop, dynamically evolving attack strategies to uncover vulnerabilities while simultaneously strengthening defense mechanisms. To ensure the Defender effectively addresses critical safety issues during the self-play, we introduce an advanced Reflective Experience Replay Mechanism, which uses an experience pool accumulated throughout the process. The mechanism employs a Upper Confidence Bound (UCB) sampling strategy to focus on failure cases with low rewards, helping the model learn from past hard mistakes while balancing exploration and exploitation. Extensive experiments demonstrate that our SSP approach autonomously evolves robust defense capabilities, significantly outperforming baselines trained on static adversarial datasets and establishing a new benchmark for proactive safety alignment.



## **6. Adversarial Evasion Attacks on Computer Vision using SHAP Values**

cs.CV

10th bwHPC Symposium - September 25th & 26th, 2024

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10587v1) [paper-pdf](https://arxiv.org/pdf/2601.10587v1)

**Authors**: Frank Mollard, Marcus Becker, Florian Roehrbein

**Abstract**: The paper introduces a white-box attack on computer vision models using SHAP values. It demonstrates how adversarial evasion attacks can compromise the performance of deep learning models by reducing output confidence or inducing misclassifications. Such attacks are particularly insidious as they can deceive the perception of an algorithm while eluding human perception due to their imperceptibility to the human eye. The proposed attack leverages SHAP values to quantify the significance of individual inputs to the output at the inference stage. A comparison is drawn between the SHAP attack and the well-known Fast Gradient Sign Method. We find evidence that SHAP attacks are more robust in generating misclassifications particularly in gradient hiding scenarios.



## **7. SRAW-Attack: Space-Reweighted Adversarial Warping Attack for SAR Target Recognition**

cs.CV

5 pages, 4 figures

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10324v1) [paper-pdf](https://arxiv.org/pdf/2601.10324v1)

**Authors**: Yiming Zhang, Weibo Qin, Yuntian Liu, Feng Wang

**Abstract**: Synthetic aperture radar (SAR) imagery exhibits intrinsic information sparsity due to its unique electromagnetic scattering mechanism. Despite the widespread adoption of deep neural network (DNN)-based SAR automatic target recognition (SAR-ATR) systems, they remain vulnerable to adversarial examples and tend to over-rely on background regions, leading to degraded adversarial robustness. Existing adversarial attacks for SAR-ATR often require visually perceptible distortions to achieve effective performance, thereby necessitating an attack method that balances effectiveness and stealthiness. In this paper, a novel attack method termed Space-Reweighted Adversarial Warping (SRAW) is proposed, which generates adversarial examples through optimized spatial deformation with reweighted budgets across foreground and background regions. Extensive experiments demonstrate that SRAW significantly degrades the performance of state-of-the-art SAR-ATR models and consistently outperforms existing methods in terms of imperceptibility and adversarial transferability. Code is made available at https://github.com/boremycin/SAR-ATR-TransAttack.



## **8. Hierarchical Refinement of Universal Multimodal Attacks on Vision-Language Models**

cs.CV

15 pages, 7 figures

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10313v1) [paper-pdf](https://arxiv.org/pdf/2601.10313v1)

**Authors**: Peng-Fei Zhang, Zi Huang

**Abstract**: Existing adversarial attacks for VLP models are mostly sample-specific, resulting in substantial computational overhead when scaled to large datasets or new scenarios. To overcome this limitation, we propose Hierarchical Refinement Attack (HRA), a multimodal universal attack framework for VLP models. HRA refines universal adversarial perturbations (UAPs) at both the sample level and the optimization level. For the image modality, we disentangle adversarial examples into clean images and perturbations, allowing each component to be handled independently for more effective disruption of cross-modal alignment. We further introduce a ScMix augmentation strategy that diversifies visual contexts and strengthens both global and local utility of UAPs, thereby reducing reliance on spurious features. In addition, we refine the optimization path by leveraging a temporal hierarchy of historical and estimated future gradients to avoid local minima and stabilize universal perturbation learning. For the text modality, HRA identifies globally influential words by combining intra-sentence and inter-sentence importance measures, and subsequently utilizes these words as universal text perturbations. Extensive experiments across various downstream tasks, VLP models, and datasets demonstrate the superiority of the proposed universal multimodal attacks.



## **9. Reasoning Hijacking: Subverting LLM Classification via Decision-Criteria Injection**

cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10294v1) [paper-pdf](https://arxiv.org/pdf/2601.10294v1)

**Authors**: Yuansen Liu, Yixuan Tang, Anthony Kum Hoe Tun

**Abstract**: Current LLM safety research predominantly focuses on mitigating Goal Hijacking, preventing attackers from redirecting a model's high-level objective (e.g., from "summarizing emails" to "phishing users"). In this paper, we argue that this perspective is incomplete and highlight a critical vulnerability in Reasoning Alignment. We propose a new adversarial paradigm: Reasoning Hijacking and instantiate it with Criteria Attack, which subverts model judgments by injecting spurious decision criteria without altering the high-level task goal. Unlike Goal Hijacking, which attempts to override the system prompt, Reasoning Hijacking accepts the high-level goal but manipulates the model's decision-making logic by injecting spurious reasoning shortcut. Though extensive experiments on three different tasks (toxic comment, negative review, and spam detection), we demonstrate that even newest models are prone to prioritize injected heuristic shortcuts over rigorous semantic analysis. The results are consistent over different backbones. Crucially, because the model's "intent" remains aligned with the user's instructions, these attacks can bypass defenses designed to detect goal deviation (e.g., SecAlign, StruQ), exposing a fundamental blind spot in the current safety landscape. Data and code are available at https://github.com/Yuan-Hou/criteria_attack



## **10. Understanding and Preserving Safety in Fine-Tuned LLMs**

cs.LG

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10141v1) [paper-pdf](https://arxiv.org/pdf/2601.10141v1)

**Authors**: Jiawen Zhang, Yangfan Hu, Kejia Chen, Lipeng He, Jiachen Ma, Jian Lou, Dan Li, Jian Liu, Xiaohu Yang, Ruoxi Jia

**Abstract**: Fine-tuning is an essential and pervasive functionality for applying large language models (LLMs) to downstream tasks. However, it has the potential to substantially degrade safety alignment, e.g., by greatly increasing susceptibility to jailbreak attacks, even when the fine-tuning data is entirely harmless. Despite garnering growing attention in defense efforts during the fine-tuning stage, existing methods struggle with a persistent safety-utility dilemma: emphasizing safety compromises task performance, whereas prioritizing utility typically requires deep fine-tuning that inevitably leads to steep safety declination.   In this work, we address this dilemma by shedding new light on the geometric interaction between safety- and utility-oriented gradients in safety-aligned LLMs. Through systematic empirical analysis, we uncover three key insights: (I) safety gradients lie in a low-rank subspace, while utility gradients span a broader high-dimensional space; (II) these subspaces are often negatively correlated, causing directional conflicts during fine-tuning; and (III) the dominant safety direction can be efficiently estimated from a single sample. Building upon these novel insights, we propose safety-preserving fine-tuning (SPF), a lightweight approach that explicitly removes gradient components conflicting with the low-rank safety subspace. Theoretically, we show that SPF guarantees utility convergence while bounding safety drift. Empirically, SPF consistently maintains downstream task performance and recovers nearly all pre-trained safety alignment, even under adversarial fine-tuning scenarios. Furthermore, SPF exhibits robust resistance to both deep fine-tuning and dynamic jailbreak attacks. Together, our findings provide new mechanistic understanding and practical guidance toward always-aligned LLM fine-tuning.



## **11. SoK: Privacy-aware LLM in Healthcare: Threat Model, Privacy Techniques, Challenges and Recommendations**

cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10004v1) [paper-pdf](https://arxiv.org/pdf/2601.10004v1)

**Authors**: Mohoshin Ara Tahera, Karamveer Singh Sidhu, Shuvalaxmi Dass, Sajal Saha

**Abstract**: Large Language Models (LLMs) are increasingly adopted in healthcare to support clinical decision-making, summarize electronic health records (EHRs), and enhance patient care. However, this integration introduces significant privacy and security challenges, driven by the sensitivity of clinical data and the high-stakes nature of medical workflows. These risks become even more pronounced across heterogeneous deployment environments, ranging from small on-premise hospital systems to regional health networks, each with unique resource limitations and regulatory demands. This Systematization of Knowledge (SoK) examines the evolving threat landscape across the three core LLM phases: Data preprocessing, Fine-tuning, and Inference within realistic healthcare settings. We present a detailed threat model that characterizes adversaries, capabilities, and attack surfaces at each phase, and we systematize how existing privacy-preserving techniques (PPTs) attempt to mitigate these vulnerabilities. While existing defenses show promise, our analysis identifies persistent limitations in securing sensitive clinical data across diverse operational tiers. We conclude with phase-aware recommendations and future research directions aimed at strengthening privacy guarantees for LLMs in regulated environments. This work provides a foundation for understanding the intersection of LLMs, threats, and privacy in healthcare, offering a roadmap toward more robust and clinically trustworthy AI systems.



## **12. Diffusion-Driven Deceptive Patches: Adversarial Manipulation and Forensic Detection in Facial Identity Verification**

cs.CV

This manuscript is a preprint. A revised version of this work has been accepted for publication in the Springer Nature book Artificial Intelligence-Driven Forensics. This version includes one additional figure for completeness

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09806v1) [paper-pdf](https://arxiv.org/pdf/2601.09806v1)

**Authors**: Shahrzad Sayyafzadeh, Hongmei Chi, Shonda Bernadin

**Abstract**: This work presents an end-to-end pipeline for generating, refining, and evaluating adversarial patches to compromise facial biometric systems, with applications in forensic analysis and security testing. We utilize FGSM to generate adversarial noise targeting an identity classifier and employ a diffusion model with reverse diffusion to enhance imperceptibility through Gaussian smoothing and adaptive brightness correction, thereby facilitating synthetic adversarial patch evasion. The refined patch is applied to facial images to test its ability to evade recognition systems while maintaining natural visual characteristics. A Vision Transformer (ViT)-GPT2 model generates captions to provide a semantic description of a person's identity for adversarial images, supporting forensic interpretation and documentation for identity evasion and recognition attacks. The pipeline evaluates changes in identity classification, captioning results, and vulnerabilities in facial identity verification and expression recognition under adversarial conditions. We further demonstrate effective detection and analysis of adversarial patches and adversarial samples using perceptual hashing and segmentation, achieving an SSIM of 0.95.



## **13. Efficient State Preparation for Quantum Machine Learning**

quant-ph

This book chapter has been accepted for Springer Nature Quantum Robustness in Artificial Intelligence and will appear in the book: https://link.springer.com/book/9783032111524?srsltid=AfmBOood7vZYc5xJYtLrQWND4pjedgfWAfAFFocjvnNS1lrNpVBwvJcO#accessibility-information

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09363v1) [paper-pdf](https://arxiv.org/pdf/2601.09363v1)

**Authors**: Chris Nakhl, Maxwell West, Muhammad Usman

**Abstract**: One of the key considerations in the development of Quantum Machine Learning (QML) protocols is the encoding of classical data onto a quantum device. In this chapter we introduce the Matrix Product State representation of quantum systems and show how it may be used to construct circuits which encode a desired state. Putting this in the context of QML we show how this process may be modified to give a low depth approximate encoding and crucially that this encoding does not hinder classification accuracy and is indeed exhibits an increased robustness against classical adversarial attacks. This is illustrated by demonstrations of adversarially robust variational quantum classifiers for the MNIST and FMNIST dataset, as well as a small-scale experimental demonstration on a superconducting quantum device.



## **14. Too Helpful to Be Safe: User-Mediated Attacks on Planning and Web-Use Agents**

cs.CR

Keywords: LLM Agents; User-Mediated Attack; Agent Security; Human Factors in Cybersecurity; Web-Use Agents; Planning Agents; Benchmark

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.10758v1) [paper-pdf](https://arxiv.org/pdf/2601.10758v1)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Carsten Rudolph

**Abstract**: Large Language Models (LLMs) have enabled agents to move beyond conversation toward end-to-end task execution and become more helpful. However, this helpfulness introduces new security risks stem less from direct interface abuse than from acting on user-provided content. Existing studies on agent security largely focus on model-internal vulnerabilities or adversarial access to agent interfaces, overlooking attacks that exploit users as unintended conduits. In this paper, we study user-mediated attacks, where benign users are tricked into relaying untrusted or attacker-controlled content to agents, and analyze how commercial LLM agents respond under such conditions. We conduct a systematic evaluation of 12 commercial agents in a sandboxed environment, covering 6 trip-planning agents and 6 web-use agents, and compare agent behavior across scenarios with no, soft, and hard user-requested safety checks. Our results show that agents are too helpful to be safe by default. Without explicit safety requests, trip-planning agents bypass safety constraints in over 92% of cases, converting unverified content into confident booking guidance. Web-use agents exhibit near-deterministic execution of risky actions, with 9 out of 17 supported tests reaching a 100% bypass rate. Even when users express soft or hard safety intent, constraint bypass remains substantial, reaching up to 54.7% and 7% for trip-planning agents, respectively. These findings reveal that the primary issue is not a lack of safety capability, but its prioritization. Agents invoke safety checks only conditionally when explicitly prompted, and otherwise default to goal-driven execution. Moreover, agents lack clear task boundaries and stopping rules, frequently over-executing workflows in ways that lead to unnecessary data disclosure and real-world harm.



## **15. Merged Bitcoin: Proof of Work Blockchains with Multiple Hash Types**

cs.CR

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09090v1) [paper-pdf](https://arxiv.org/pdf/2601.09090v1)

**Authors**: Christopher Blake, Chen Feng, Xuachao Wang, Qianyu Yu

**Abstract**: Proof of work blockchain protocols using multiple hash types are considered. It is proven that the security region of such a protocol cannot be the AND of a 51\% attack on all the hash types. Nevertheless, a protocol called Merged Bitcoin is introduced, which is the Bitcoin protocol where links between blocks can be formed using multiple different hash types. Closed form bounds on its security region in the $Δ$-bounded delay network model are proven, and these bounds are compared to simulation results. This protocol is proven to maximize cost of attack in the linear cost-per-hash model. A difficulty adjustment method is introduced, and it is argued that this can partly remedy asymmetric advantages an adversary may gain in hashing power for some hash types, including from algorithmic advances, quantum attacks like Grover's algorithm, or hardware backdoor attacks.



## **16. StegoStylo: Squelching Stylometric Scrutiny through Steganographic Stitching**

cs.CR

16 pages, 6 figures, 1 table

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09056v1) [paper-pdf](https://arxiv.org/pdf/2601.09056v1)

**Authors**: Robert Dilworth

**Abstract**: Stylometry--the identification of an author through analysis of a text's style (i.e., authorship attribution)--serves many constructive purposes: it supports copyright and plagiarism investigations, aids detection of harmful content, offers exploratory cues for certain medical conditions (e.g., early signs of dementia or depression), provides historical context for literary works, and helps uncover misinformation and disinformation. In contrast, when stylometry is employed as a tool for authorship verification--confirming whether a text truly originates from a claimed author--it can also be weaponized for malicious purposes. Techniques such as de-anonymization, re-identification, tracking, profiling, and downstream effects like censorship illustrate the privacy threats that stylometric analysis can enable. Building on these concerns, this paper further explores how adversarial stylometry combined with steganography can counteract stylometric analysis. We first present enhancements to our adversarial attack, $\textit{TraceTarnish}$, providing stronger evidence of its capacity to confound stylometric systems and reduce their attribution and verification accuracy. Next, we examine how steganographic embedding can be fine-tuned to mask an author's stylistic fingerprint, quantifying the level of authorship obfuscation achievable as a function of the proportion of words altered with zero-width Unicode characters. Based on our findings, steganographic coverage of 33% or higher seemingly ensures authorship obfuscation. Finally, we reflect on the ways stylometry can be used to undermine privacy and argue for the necessity of defensive tools like $\textit{TraceTarnish}$.



## **17. A Differential Geometry and Algebraic Topology Based Public-Key Cryptographic Algorithm in Presence of Quantum Adversaries**

cs.IT

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.10883v1) [paper-pdf](https://arxiv.org/pdf/2601.10883v1)

**Authors**: Andrea Rondelli

**Abstract**: In antiquity, the seal embodied trust, secrecy, and integrity in safeguarding the exchange of letters and messages. The purpose of this work is to continue this tradition in the contemporary era, characterized by the presence of quantum computers, classical supercomputers, and increasingly sophisticated artificial intelligence. We introduce Z-Sigil, an asymmetric public-key cryptographic algorithm grounded in functional analysis, differential geometry, and algebraic topology, with the explicit goal of achieving resistance against both classical and quantum attacks. The construction operates over the tangent fiber bundle of a compact Calabi-Yau manifold [13], where cryptographic keys are elements of vector tangent fibers, with a binary operation defined on tangent spaces of the base manifold giving rise to a groupoid structure. Encryption and decryption are performed iteratively on message blocks, enforcing a serial architecture designed to limit quantum parallelism [9,10]. Each block depends on secret geometric and analytic data, including a randomly chosen base point on the manifold, a selected section of the tangent fiber bundle, and auxiliary analytic data derived from operator determinants and Zeta function regularization [11]. The correctness and invertibility of the proposed algorithm are proven analytically. Furthermore, any adversarial attempt to recover the plaintext without the private key leads to an exponential growth of the adversarial search space,even under quantum speedups. The use of continuous geometric structures,non-linear operator compositions,and enforced blockwise serialization distinguishes this approach from existing quantum-safe cryptographic proposals based on primary discrete algebraic assumptions.



## **18. SafeRedir: Prompt Embedding Redirection for Robust Unlearning in Image Generation Models**

cs.CV

Code at https://github.com/ryliu68/SafeRedir

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08623v1) [paper-pdf](https://arxiv.org/pdf/2601.08623v1)

**Authors**: Renyang Liu, Kangjie Chen, Han Qiu, Jie Zhang, Kwok-Yan Lam, Tianwei Zhang, See-Kiong Ng

**Abstract**: Image generation models (IGMs), while capable of producing impressive and creative content, often memorize a wide range of undesirable concepts from their training data, leading to the reproduction of unsafe content such as NSFW imagery and copyrighted artistic styles. Such behaviors pose persistent safety and compliance risks in real-world deployments and cannot be reliably mitigated by post-hoc filtering, owing to the limited robustness of such mechanisms and a lack of fine-grained semantic control. Recent unlearning methods seek to erase harmful concepts at the model level, which exhibit the limitations of requiring costly retraining, degrading the quality of benign generations, or failing to withstand prompt paraphrasing and adversarial attacks. To address these challenges, we introduce SafeRedir, a lightweight inference-time framework for robust unlearning via prompt embedding redirection. Without modifying the underlying IGMs, SafeRedir adaptively routes unsafe prompts toward safe semantic regions through token-level interventions in the embedding space. The framework comprises two core components: a latent-aware multi-modal safety classifier for identifying unsafe generation trajectories, and a token-level delta generator for precise semantic redirection, equipped with auxiliary predictors for token masking and adaptive scaling to localize and regulate the intervention. Empirical results across multiple representative unlearning tasks demonstrate that SafeRedir achieves effective unlearning capability, high semantic and perceptual preservation, robust image quality, and enhanced resistance to adversarial attacks. Furthermore, SafeRedir generalizes effectively across a variety of diffusion backbones and existing unlearned models, validating its plug-and-play compatibility and broad applicability. Code and data are available at https://github.com/ryliu68/SafeRedir.



## **19. MASH: Evading Black-Box AI-Generated Text Detectors via Style Humanization**

cs.CR

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08564v1) [paper-pdf](https://arxiv.org/pdf/2601.08564v1)

**Authors**: Yongtong Gu, Songze Li, Xia Hu

**Abstract**: The increasing misuse of AI-generated texts (AIGT) has motivated the rapid development of AIGT detection methods. However, the reliability of these detectors remains fragile against adversarial evasions. Existing attack strategies often rely on white-box assumptions or demand prohibitively high computational and interaction costs, rendering them ineffective under practical black-box scenarios. In this paper, we propose Multi-stage Alignment for Style Humanization (MASH), a novel framework that evades black-box detectors based on style transfer. MASH sequentially employs style-injection supervised fine-tuning, direct preference optimization, and inference-time refinement to shape the distributions of AI-generated texts to resemble those of human-written texts. Experiments across 6 datasets and 5 detectors demonstrate the superior performance of MASH over 11 baseline evaders. Specifically, MASH achieves an average Attack Success Rate (ASR) of 92%, surpassing the strongest baselines by an average of 24%, while maintaining superior linguistic quality.



## **20. Evaluating Role-Consistency in LLMs for Counselor Training**

cs.CL

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08892v1) [paper-pdf](https://arxiv.org/pdf/2601.08892v1)

**Authors**: Eric Rudolph, Natalie Engert, Jens Albrecht

**Abstract**: The rise of online counseling services has highlighted the need for effective training methods for future counselors. This paper extends research on VirCo, a Virtual Client for Online Counseling, designed to complement traditional role-playing methods in academic training by simulating realistic client interactions. Building on previous work, we introduce a new dataset incorporating adversarial attacks to test the ability of large language models (LLMs) to maintain their assigned roles (role-consistency). The study focuses on evaluating the role consistency and coherence of the Vicuna model's responses, comparing these findings with earlier research. Additionally, we assess and compare various open-source LLMs for their performance in sustaining role consistency during virtual client interactions. Our contributions include creating an adversarial dataset, evaluating conversation coherence and persona consistency, and providing a comparative analysis of different LLMs.



## **21. BenchOverflow: Measuring Overflow in Large Language Models via Plain-Text Prompts**

cs.CL

Accepted at TMLR 2026

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08490v1) [paper-pdf](https://arxiv.org/pdf/2601.08490v1)

**Authors**: Erin Feiglin, Nir Hutnik, Raz Lapid

**Abstract**: We investigate a failure mode of large language models (LLMs) in which plain-text prompts elicit excessive outputs, a phenomenon we term Overflow. Unlike jailbreaks or prompt injection, Overflow arises under ordinary interaction settings and can lead to elevated serving cost, latency, and cross-user performance degradation, particularly when scaled across many requests. Beyond usability, the stakes are economic and environmental: unnecessary tokens increase per-request cost and energy consumption, compounding into substantial operational spend and carbon footprint at scale. Moreover, Overflow represents a practical vector for compute amplification and service degradation in shared environments. We introduce BenchOverflow, a model-agnostic benchmark of nine plain-text prompting strategies that amplify output volume without adversarial suffixes or policy circumvention. Using a standardized protocol with a fixed budget of 5000 new tokens, we evaluate nine open- and closed-source models and observe pronounced rightward shifts and heavy tails in length distributions. Cap-saturation rates (CSR@1k/3k/5k) and empirical cumulative distribution functions (ECDFs) quantify tail risk; within-prompt variance and cross-model correlations show that Overflow is broadly reproducible yet heterogeneous across families and attack vectors. A lightweight mitigation-a fixed conciseness reminder-attenuates right tails and lowers CSR for all strategies across the majority of models. Our findings position length control as a measurable reliability, cost, and sustainability concern rather than a stylistic quirk. By enabling standardized comparison of length-control robustness across models, BenchOverflow provides a practical basis for selecting deployments that minimize resource waste and operating expense, and for evaluating defenses that curb compute amplification without eroding task performance.



## **22. Baiting AI: Deceptive Adversary Against AI-Protected Industrial Infrastructures**

cs.CR

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08481v1) [paper-pdf](https://arxiv.org/pdf/2601.08481v1)

**Authors**: Aryan Pasikhani, Prosanta Gope, Yang Yang, Shagufta Mehnaz, Biplab Sikdar

**Abstract**: This paper explores a new cyber-attack vector targeting Industrial Control Systems (ICS), particularly focusing on water treatment facilities. Developing a new multi-agent Deep Reinforcement Learning (DRL) approach, adversaries craft stealthy, strategically timed, wear-out attacks designed to subtly degrade product quality and reduce the lifespan of field actuators. This sophisticated method leverages DRL methodology not only to execute precise and detrimental impacts on targeted infrastructure but also to evade detection by contemporary AI-driven defence systems. By developing and implementing tailored policies, the attackers ensure their hostile actions blend seamlessly with normal operational patterns, circumventing integrated security measures. Our research reveals the robustness of this attack strategy, shedding light on the potential for DRL models to be manipulated for adversarial purposes. Our research has been validated through testing and analysis in an industry-level setup. For reproducibility and further study, all related materials, including datasets and documentation, are publicly accessible.



## **23. SecureCAI: Injection-Resilient LLM Assistants for Cybersecurity Operations**

cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07835v1) [paper-pdf](https://arxiv.org/pdf/2601.07835v1)

**Authors**: Mohammed Himayath Ali, Mohammed Aqib Abdullah, Mohammed Mudassir Uddin, Shahnawaz Alam

**Abstract**: Large Language Models have emerged as transformative tools for Security Operations Centers, enabling automated log analysis, phishing triage, and malware explanation; however, deployment in adversarial cybersecurity environments exposes critical vulnerabilities to prompt injection attacks where malicious instructions embedded in security artifacts manipulate model behavior. This paper introduces SecureCAI, a novel defense framework extending Constitutional AI principles with security-aware guardrails, adaptive constitution evolution, and Direct Preference Optimization for unlearning unsafe response patterns, addressing the unique challenges of high-stakes security contexts where traditional safety mechanisms prove insufficient against sophisticated adversarial manipulation. Experimental evaluation demonstrates that SecureCAI reduces attack success rates by 94.7% compared to baseline models while maintaining 95.1% accuracy on benign security analysis tasks, with the framework incorporating continuous red-teaming feedback loops enabling dynamic adaptation to emerging attack strategies and achieving constitution adherence scores exceeding 0.92 under sustained adversarial pressure, thereby establishing a foundation for trustworthy integration of language model capabilities into operational cybersecurity workflows and addressing a critical gap in current approaches to AI safety within adversarial domains.



## **24. Self-Creating Random Walks for Decentralized Learning under Pac-Man Attacks**

cs.MA

arXiv admin note: substantial text overlap with arXiv:2508.05663

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07674v1) [paper-pdf](https://arxiv.org/pdf/2601.07674v1)

**Authors**: Xingran Chen, Parimal Parag, Rohit Bhagat, Salim El Rouayheb

**Abstract**: Random walk (RW)-based algorithms have long been popular in distributed systems due to low overheads and scalability, with recent growing applications in decentralized learning. However, their reliance on local interactions makes them inherently vulnerable to malicious behavior. In this work, we investigate an adversarial threat that we term the ``Pac-Man'' attack, in which a malicious node probabilistically terminates any RW that visits it. This stealthy behavior gradually eliminates active RWs from the network, effectively halting the learning process without triggering failure alarms. To counter this threat, we propose the CREATE-IF-LATE (CIL) algorithm, which is a fully decentralized, resilient mechanism that enables self-creating RWs and prevents RW extinction in the presence of Pac-Man. Our theoretical analysis shows that the CIL algorithm guarantees several desirable properties, such as (i) non-extinction of the RW population, (ii) almost sure boundedness of the RW population, and (iii) convergence of RW-based stochastic gradient descent even in the presence of Pac-Man with a quantifiable deviation from the true optimum. Moreover, the learning process experiences at most a linear time delay due to Pac-Man interruptions and RW regeneration. Our extensive empirical results on both synthetic and public benchmark datasets validate our theoretical findings.



## **25. Universal Adversarial Purification with DDIM Metric Loss for Stable Diffusion**

cs.CV

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07253v1) [paper-pdf](https://arxiv.org/pdf/2601.07253v1)

**Authors**: Li Zheng, Liangbin Xie, Jiantao Zhou, He YiMin

**Abstract**: Stable Diffusion (SD) often produces degraded outputs when the training dataset contains adversarial noise. Adversarial purification offers a promising solution by removing adversarial noise from contaminated data. However, existing purification methods are primarily designed for classification tasks and fail to address SD-specific adversarial strategies, such as attacks targeting the VAE encoder, UNet denoiser, or both. To address the gap in SD security, we propose Universal Diffusion Adversarial Purification (UDAP), a novel framework tailored for defending adversarial attacks targeting SD models. UDAP leverages the distinct reconstruction behaviors of clean and adversarial images during Denoising Diffusion Implicit Models (DDIM) inversion to optimize the purification process. By minimizing the DDIM metric loss, UDAP can effectively remove adversarial noise. Additionally, we introduce a dynamic epoch adjustment strategy that adapts optimization iterations based on reconstruction errors, significantly improving efficiency without sacrificing purification quality. Experiments demonstrate UDAP's robustness against diverse adversarial methods, including PID (VAE-targeted), Anti-DreamBooth (UNet-targeted), MIST (hybrid), and robustness-enhanced variants like Anti-Diffusion (Anti-DF) and MetaCloak. UDAP also generalizes well across SD versions and text prompts, showcasing its practical applicability in real-world scenarios.



## **26. PROTEA: Securing Robot Task Planning and Execution**

cs.RO

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07186v1) [paper-pdf](https://arxiv.org/pdf/2601.07186v1)

**Authors**: Zainab Altaweel, Mohaiminul Al Nahian, Jake Juettner, Adnan Siraj Rakin, Shiqi Zhang

**Abstract**: Robots need task planning methods to generate action sequences for complex tasks. Recent work on adversarial attacks has revealed significant vulnerabilities in existing robot task planners, especially those built on foundation models. In this paper, we aim to address these security challenges by introducing PROTEA, an LLM-as-a-Judge defense mechanism, to evaluate the security of task plans. PROTEA is developed to address the dimensionality and history challenges in plan safety assessment. We used different LLMs to implement multiple versions of PROTEA for comparison purposes. For systemic evaluations, we created a dataset containing both benign and malicious task plans, where the harmful behaviors were injected at varying levels of stealthiness. Our results provide actionable insights for robotic system practitioners seeking to enhance robustness and security of their task planning systems. Details, dataset and demos are provided: https://protea-secure.github.io/PROTEA/



## **27. Defenses Against Prompt Attacks Learn Surface Heuristics**

cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07185v1) [paper-pdf](https://arxiv.org/pdf/2601.07185v1)

**Authors**: Shawn Li, Chenxiao Yu, Zhiyu Ni, Hao Li, Charith Peris, Chaowei Xiao, Yue Zhao

**Abstract**: Large language models (LLMs) are increasingly deployed in security-sensitive applications, where they must follow system- or developer-specified instructions that define the intended task behavior, while completing benign user requests. When adversarial instructions appear in user queries or externally retrieved content, models may override intended logic. Recent defenses rely on supervised fine-tuning with benign and malicious labels. Although these methods achieve high attack rejection rates, we find that they rely on narrow correlations in defense data rather than harmful intent, leading to systematic rejection of safe inputs. We analyze three recurring shortcut behaviors induced by defense fine-tuning. \emph{Position bias} arises when benign content placed later in a prompt is rejected at much higher rates; across reasoning benchmarks, suffix-task rejection rises from below \textbf{10\%} to as high as \textbf{90\%}. \emph{Token trigger bias} occurs when strings common in attack data raise rejection probability even in benign contexts; inserting a single trigger token increases false refusals by up to \textbf{50\%}. \emph{Topic generalization bias} reflects poor generalization beyond the defense data distribution, with defended models suffering test-time accuracy drops of up to \textbf{40\%}. These findings suggest that current prompt-injection defenses frequently respond to attack-like surface patterns rather than the underlying intent. We introduce controlled diagnostic datasets and a systematic evaluation across two base models and multiple defense pipelines, highlighting limitations of supervised fine-tuning for reliable LLM security.



## **28. MacPrompt: Maraconic-guided Jailbreak against Text-to-Image Models**

cs.CR

Accepted by AAAI 2026

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07141v1) [paper-pdf](https://arxiv.org/pdf/2601.07141v1)

**Authors**: Xi Ye, Yiwen Liu, Lina Wang, Run Wang, Geying Yang, Yufei Hou, Jiayi Yu

**Abstract**: Text-to-image (T2I) models have raised increasing safety concerns due to their capacity to generate NSFW and other banned objects. To mitigate these risks, safety filters and concept removal techniques have been introduced to block inappropriate prompts or erase sensitive concepts from the models. However, all the existing defense methods are not well prepared to handle diverse adversarial prompts. In this work, we introduce MacPrompt, a novel black-box and cross-lingual attack that reveals previously overlooked vulnerabilities in T2I safety mechanisms. Unlike existing attacks that rely on synonym substitution or prompt obfuscation, MacPrompt constructs macaronic adversarial prompts by performing cross-lingual character-level recombination of harmful terms, enabling fine-grained control over both semantics and appearance. By leveraging this design, MacPrompt crafts prompts with high semantic similarity to the original harmful inputs (up to 0.96) while bypassing major safety filters (up to 100%). More critically, it achieves attack success rates as high as 92% for sex-related content and 90% for violence, effectively breaking even state-of-the-art concept removal defenses. These results underscore the pressing need to reassess the robustness of existing T2I safety mechanisms against linguistically diverse and fine-grained adversarial strategies.



## **29. Enhancing Cloud Network Resilience via a Robust LLM-Empowered Multi-Agent Reinforcement Learning Framework**

cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07122v1) [paper-pdf](https://arxiv.org/pdf/2601.07122v1)

**Authors**: Yixiao Peng, Hao Hu, Feiyang Li, Xinye Cao, Yingchang Jiang, Jipeng Tang, Guoshun Nan, Yuling Liu

**Abstract**: While virtualization and resource pooling empower cloud networks with structural flexibility and elastic scalability, they inevitably expand the attack surface and challenge cyber resilience. Reinforcement Learning (RL)-based defense strategies have been developed to optimize resource deployment and isolation policies under adversarial conditions, aiming to enhance system resilience by maintaining and restoring network availability. However, existing approaches lack robustness as they require retraining to adapt to dynamic changes in network structure, node scale, attack strategies, and attack intensity. Furthermore, the lack of Human-in-the-Loop (HITL) support limits interpretability and flexibility. To address these limitations, we propose CyberOps-Bots, a hierarchical multi-agent reinforcement learning framework empowered by Large Language Models (LLMs). Inspired by MITRE ATT&CK's Tactics-Techniques model, CyberOps-Bots features a two-layer architecture: (1) An upper-level LLM agent with four modules--ReAct planning, IPDRR-based perception, long-short term memory, and action/tool integration--performs global awareness, human intent recognition, and tactical planning; (2) Lower-level RL agents, developed via heterogeneous separated pre-training, execute atomic defense actions within localized network regions. This synergy preserves LLM adaptability and interpretability while ensuring reliable RL execution. Experiments on real cloud datasets show that, compared to state-of-the-art algorithms, CyberOps-Bots maintains network availability 68.5% higher and achieves a 34.7% jumpstart performance gain when shifting the scenarios without retraining. To our knowledge, this is the first study to establish a robust LLM-RL framework with HITL support for cloud defense. We will release our framework to the community, facilitating the advancement of robust and autonomous defense in cloud networks.



## **30. Reward-Preserving Attacks For Robust Reinforcement Learning**

cs.LG

19 pages, 6 figures, 4 algorithms, preprint

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07118v1) [paper-pdf](https://arxiv.org/pdf/2601.07118v1)

**Authors**: Lucas Schott, Elies Gherbi, Hatem Hajri, Sylvain Lamprier

**Abstract**: Adversarial robustness in RL is difficult because perturbations affect entire trajectories: strong attacks can break learning, while weak attacks yield little robustness, and the appropriate strength varies by state. We propose $α$-reward-preserving attacks, which adapt the strength of the adversary so that an $α$ fraction of the nominal-to-worst-case return gap remains achievable at each state. In deep RL, we use a gradient-based attack direction and learn a state-dependent magnitude $η\le η_{\mathcal B}$ selected via a critic $Q^π_α((s,a),η)$ trained off-policy over diverse radii. This adaptive tuning calibrates attack strength and, with intermediate $α$, improves robustness across radii while preserving nominal performance, outperforming fixed- and random-radius baselines.



## **31. Memory Poisoning Attack and Defense on Memory Based LLM-Agents**

cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.05504v2) [paper-pdf](https://arxiv.org/pdf/2601.05504v2)

**Authors**: Balachandra Devarangadi Sunil, Isheeta Sinha, Piyush Maheshwari, Shantanu Todmal, Shreyan Mallik, Shuchi Mishra

**Abstract**: Large language model agents equipped with persistent memory are vulnerable to memory poisoning attacks, where adversaries inject malicious instructions through query only interactions that corrupt the agents long term memory and influence future responses. Recent work demonstrated that the MINJA (Memory Injection Attack) achieves over 95 % injection success rate and 70 % attack success rate under idealized conditions. However, the robustness of these attacks in realistic deployments and effective defensive mechanisms remain understudied. This work addresses these gaps through systematic empirical evaluation of memory poisoning attacks and defenses in Electronic Health Record (EHR) agents. We investigate attack robustness by varying three critical dimensions: initial memory state, number of indication prompts, and retrieval parameters. Our experiments on GPT-4o-mini, Gemini-2.0-Flash and Llama-3.1-8B-Instruct models using MIMIC-III clinical data reveal that realistic conditions with pre-existing legitimate memories dramatically reduce attack effectiveness. We then propose and evaluate two novel defense mechanisms: (1) Input/Output Moderation using composite trust scoring across multiple orthogonal signals, and (2) Memory Sanitization with trust-aware retrieval employing temporal decay and pattern-based filtering. Our defense evaluation reveals that effective memory sanitization requires careful trust threshold calibration to prevent both overly conservative rejection (blocking all entries) and insufficient filtering (missing subtle attacks), establishing important baselines for future adaptive defense mechanisms. These findings provide crucial insights for securing memory-augmented LLM agents in production environments.



## **32. $PC^2$: Politically Controversial Content Generation via Jailbreaking Attacks on GPT-based Text-to-Image Models**

cs.CR

19 pages, 15 figures, 9 tables

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.05150v2) [paper-pdf](https://arxiv.org/pdf/2601.05150v2)

**Authors**: Wonwoo Choi, Minjae Seo, Minkyoo Song, Hwanjo Heo, Seungwon Shin, Myoungsung You

**Abstract**: The rapid evolution of text-to-image (T2I) models has enabled high-fidelity visual synthesis on a global scale. However, these advancements have introduced significant security risks, particularly regarding the generation of harmful content. Politically harmful content, such as fabricated depictions of public figures, poses severe threats when weaponized for fake news or propaganda. Despite its criticality, the robustness of current T2I safety filters against such politically motivated adversarial prompting remains underexplored. In response, we propose $PC^2$, the first black-box political jailbreaking framework for T2I models. It exploits a novel vulnerability where safety filters evaluate political sensitivity based on linguistic context. $PC^2$ operates through: (1) Identity-Preserving Descriptive Mapping to obfuscate sensitive keywords into neutral descriptions, and (2) Geopolitically Distal Translation to map these descriptions into fragmented, low-sensitivity languages. This strategy prevents filters from constructing toxic relationships between political entities within prompts, effectively bypassing detection. We construct a benchmark of 240 politically sensitive prompts involving 36 public figures. Evaluation on commercial T2I models, specifically GPT-series, shows that while all original prompts are blocked, $PC^2$ achieves attack success rates of up to 86%.



## **33. MORE: Multi-Objective Adversarial Attacks on Speech Recognition**

eess.AS

19 pages

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.01852v2) [paper-pdf](https://arxiv.org/pdf/2601.01852v2)

**Authors**: Xiaoxue Gao, Zexin Li, Yiming Chen, Nancy F. Chen

**Abstract**: The emergence of large-scale automatic speech recognition (ASR) models such as Whisper has greatly expanded their adoption across diverse real-world applications. Ensuring robustness against even minor input perturbations is therefore critical for maintaining reliable performance in real-time environments. While prior work has mainly examined accuracy degradation under adversarial attacks, robustness with respect to efficiency remains largely unexplored. This narrow focus provides only a partial understanding of ASR model vulnerabilities. To address this gap, we conduct a comprehensive study of ASR robustness under multiple attack scenarios. We introduce MORE, a multi-objective repetitive doubling encouragement attack, which jointly degrades recognition accuracy and inference efficiency through a hierarchical staged repulsion-anchoring mechanism. Specifically, we reformulate multi-objective adversarial optimization into a hierarchical framework that sequentially achieves the dual objectives. To further amplify effectiveness, we propose a novel repetitive encouragement doubling objective (REDO) that induces duplicative text generation by maintaining accuracy degradation and periodically doubling the predicted sequence length. Overall, MORE compels ASR models to produce incorrect transcriptions at a substantially higher computational cost, triggered by a single adversarial input. Experiments show that MORE consistently yields significantly longer transcriptions while maintaining high word error rates compared to existing baselines, underscoring its effectiveness in multi-objective adversarial attack.



## **34. Measuring the Impact of Student Gaming Behaviors on Learner Modeling**

cs.CY

Full research paper accepted at Learning Analytics and Knowledge (LAK '26) conference, see https://doi.org/10.1145/3785022.3785036

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2512.18659v3) [paper-pdf](https://arxiv.org/pdf/2512.18659v3)

**Authors**: Qinyi Liu, Lin Li, Valdemar Švábenský, Conrad Borchers, Mohammad Khalil

**Abstract**: The expansion of large-scale online education platforms has made vast amounts of student interaction data available for knowledge tracing (KT). KT models estimate students' concept mastery from interaction data, but their performance is sensitive to input data quality. Gaming behaviors, such as excessive hint use, may misrepresent students' knowledge and undermine model reliability. However, systematic investigations of how different types of gaming behaviors affect KT remain scarce, and existing studies rely on costly manual analysis that does not capture behavioral diversity. In this study, we conceptualize gaming behaviors as a form of data poisoning, defined as the deliberate submission of incorrect or misleading interaction data to corrupt a model's learning process. We design Data Poisoning Attacks (DPAs) to simulate diverse gaming patterns and systematically evaluate their impact on KT model performance. Moreover, drawing on advances in DPA detection, we explore unsupervised approaches to enhance the generalizability of gaming behavior detection. We find that KT models' performance tends to decrease especially in response to random guess behaviors. Our findings provide insights into the vulnerabilities of KT models and highlight the potential of adversarial methods for improving the robustness of learning analytics systems.



## **35. From Adversarial Poetry to Adversarial Tales: An Interpretability Research Agenda**

cs.CL

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.08837v2) [paper-pdf](https://arxiv.org/pdf/2601.08837v2)

**Authors**: Piercosma Bisconti, Marcello Galisai, Matteo Prandi, Federico Pierucci, Olga Sorokoletova, Francesco Giarrusso, Vincenzo Suriani, Marcantonio Bracale Syrnikov, Daniele Nardi

**Abstract**: Safety mechanisms in LLMs remain vulnerable to attacks that reframe harmful requests through culturally coded structures. We introduce Adversarial Tales, a jailbreak technique that embeds harmful content within cyberpunk narratives and prompts models to perform functional analysis inspired by Vladimir Propp's morphology of folktales. By casting the task as structural decomposition, the attack induces models to reconstruct harmful procedures as legitimate narrative interpretation. Across 26 frontier models from nine providers, we observe an average attack success rate of 71.3%, with no model family proving reliably robust. Together with our prior work on Adversarial Poetry, these findings suggest that structurally-grounded jailbreaks constitute a broad vulnerability class rather than isolated techniques. The space of culturally coded frames that can mediate harmful intent is vast, likely inexhaustible by pattern-matching defenses alone. Understanding why these attacks succeed is therefore essential: we outline a mechanistic interpretability research agenda to investigate how narrative cues reshape model representations and whether models can learn to recognize harmful intent independently of surface form.



## **36. From static to adaptive: immune memory-based jailbreak detection for large language models**

cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2512.03356v2) [paper-pdf](https://arxiv.org/pdf/2512.03356v2)

**Authors**: Jun Leng, Yu Liu, Litian Zhang, Ruihan Hu, Zhuting Fang, Xi Zhang

**Abstract**: Large Language Models (LLMs) serve as the backbone of modern AI systems, yet they remain susceptible to adversarial jailbreak attacks. Consequently, robust detection of such malicious inputs is paramount for ensuring model safety. Traditional detection methods typically rely on external models trained on fixed, large-scale datasets, which often incur significant computational overhead. While recent methods shift toward leveraging internal safety signals of models to enable more lightweight and efficient detection. However, these methods remain inherently static and struggle to adapt to the evolving nature of jailbreak attacks. Drawing inspiration from the biological immune mechanism, we introduce the Immune Memory Adaptive Guard (IMAG) framework. By distilling and encoding safety patterns into a persistent, evolvable memory bank, IMAG enables adaptive generalization to emerging threats. Specifically, the framework orchestrates three synergistic components: Immune Detection, which employs retrieval for high-efficiency interception of known jailbreak attacks; Active Immunity, which performs proactive behavioral simulation to resolve ambiguous unknown queries; Memory Updating, which integrates validated attack patterns back into the memory bank. This closed-loop architecture transitions LLM defense from rigid filtering to autonomous adaptive mitigation. Extensive evaluations across five representative open-source LLMs demonstrate that our method surpasses state-of-the-art (SOTA) baselines, achieving a superior average detection accuracy of 94\% across diverse and complex attack types.



## **37. Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models**

cs.CL

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2511.15304v3) [paper-pdf](https://arxiv.org/pdf/2511.15304v3)

**Authors**: Piercosma Bisconti, Matteo Prandi, Federico Pierucci, Francesco Giarrusso, Marcantonio Bracale Syrnikov, Marcello Galisai, Vincenzo Suriani, Olga Sorokoletova, Federico Sartore, Daniele Nardi

**Abstract**: We present evidence that adversarial poetry functions as a universal single-turn jailbreak technique for Large Language Models (LLMs). Across 25 frontier proprietary and open-weight models, curated poetic prompts yielded high attack-success rates (ASR), with some providers exceeding 90%. Mapping prompts to MLCommons and EU CoP risk taxonomies shows that poetic attacks transfer across CBRN, manipulation, cyber-offence, and loss-of-control domains. Converting 1,200 MLCommons harmful prompts into verse via a standardized meta-prompt produced ASRs up to 18 times higher than their prose baselines. Outputs are evaluated using an ensemble of 3 open-weight LLM judges, whose binary safety assessments were validated on a stratified human-labeled subset. Poetic framing achieved an average jailbreak success rate of 62% for hand-crafted poems and approximately 43% for meta-prompt conversions (compared to non-poetic baselines), substantially outperforming non-poetic baselines and revealing a systematic vulnerability across model families and safety training approaches. These findings demonstrate that stylistic variation alone can circumvent contemporary safety mechanisms, suggesting fundamental limitations in current alignment methods and evaluation protocols.



## **38. Bribers, Bribers on The Chain, Is Resisting All in Vain? Trustless Consensus Manipulation Through Bribing Contracts**

cs.CR

To appear at Financial Cryptography and Data Security 2026

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2509.17185v2) [paper-pdf](https://arxiv.org/pdf/2509.17185v2)

**Authors**: Bence Soóki-Tóth, István András Seres, Kamilla Kara, Ábel Nagy, Balázs Pejó, Gergely Biczók

**Abstract**: The long-term success of cryptocurrencies largely depends on the incentive compatibility provided to the validators. Bribery attacks, facilitated trustlessly via smart contracts, threaten this foundation. This work introduces, implements, and evaluates three novel and efficient bribery contracts targeting Ethereum validators. The first bribery contract enables a briber to fork the blockchain by buying votes on their proposed blocks. The second contract incentivizes validators to voluntarily exit the consensus protocol, thus increasing the adversary's relative staking power. The third contract builds a trustless bribery market that enables the briber to auction off their manipulative power over the RANDAO, Ethereum's distributed randomness beacon. Finally, we provide an initial game-theoretical analysis of one of the described bribery markets.



## **39. Simulated Ensemble Attack: Transferring Jailbreaks Across Fine-tuned Vision-Language Models**

cs.CV

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2508.01741v3) [paper-pdf](https://arxiv.org/pdf/2508.01741v3)

**Authors**: Ruofan Wang, Xin Wang, Yang Yao, Juncheng Li, Xuan Tong, Xingjun Ma

**Abstract**: The widespread practice of fine-tuning open-source Vision-Language Models (VLMs) raises a critical security concern: jailbreak vulnerabilities in base models may persist in downstream variants, enabling transferable attacks across fine-tuned systems. To investigate this risk, we propose the Simulated Ensemble Attack (SEA), a grey-box jailbreak framework that assumes full access to the base VLM but no knowledge of the fine-tuned target. SEA enhances transferability via Fine-tuning Trajectory Simulation (FTS), which models bounded parameter variations in the vision encoder, and Targeted Prompt Guidance (TPG), which stabilizes adversarial optimization through auxiliary textual guidance. Experiments on the Qwen2-VL family demonstrate that SEA achieves consistently high transfer success and toxicity rates across diverse fine-tuned variants, including safety-enhanced models, while standard PGD-based image jailbreaks exhibit negligible transferability. Further analysis reveals that fine-tuning primarily induces localized parameter shifts around the base model, explaining why attacks optimized over a simulated neighborhood transfer effectively. We also show that SEA generalizes across different base generations (e.g., Qwen2.5/3-VL), indicating that its effectiveness arises from shared fine-tuning-induced behaviors rather than architecture- or initialization-specific factors.



## **40. BeDKD: Backdoor Defense Based on Directional Mapping Module and Adversarial Knowledge Distillation**

cs.CR

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2508.01595v3) [paper-pdf](https://arxiv.org/pdf/2508.01595v3)

**Authors**: Zhengxian Wu, Juan Wen, Wanli Peng, Yinghan Zhou, Changtong dou, Yiming Xue

**Abstract**: Although existing backdoor defenses have gained success in mitigating backdoor attacks, they still face substantial challenges. In particular, most of them rely on large amounts of clean data to weaken the backdoor mapping but generally struggle with residual trigger effects, resulting in persistently high attack success rates (ASR). Therefore, in this paper, we propose a novel \textbf{B}ackdoor d\textbf{e}fense method based on \textbf{D}irectional mapping module and adversarial \textbf{K}nowledge \textbf{D}istillation (BeDKD), which balances the trade-off between defense effectiveness and model performance using a small amount of clean and poisoned data. We first introduce a directional mapping module to identify poisoned data, which destroys clean mapping while keeping backdoor mapping on a small set of flipped clean data. Then, the adversarial knowledge distillation is designed to reinforce clean mapping and suppress backdoor mapping through a cycle iteration mechanism between trust and punish distillations using clean and identified poisoned data. We conduct experiments to mitigate mainstream attacks on three datasets, and experimental results demonstrate that BeDKD surpasses the state-of-the-art defenses and reduces the ASR by 98$\%$ without significantly reducing the CACC. Our code are available in https://github.com/CAU-ISS-Lab/Backdoor-Attack-Defense-LLMs/tree/main/BeDKD.



## **41. Random Walk Learning and the Pac-Man Attack**

stat.ML

The updated manuscript represents an incomplete version of the work. A substantially updated version will be prepared before further dissemination

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2508.05663v3) [paper-pdf](https://arxiv.org/pdf/2508.05663v3)

**Authors**: Xingran Chen, Parimal Parag, Rohit Bhagat, Zonghong Liu, Salim El Rouayheb

**Abstract**: Random walk (RW)-based algorithms have long been popular in distributed systems due to low overheads and scalability, with recent growing applications in decentralized learning. However, their reliance on local interactions makes them inherently vulnerable to malicious behavior. In this work, we investigate an adversarial threat that we term the ``Pac-Man'' attack, in which a malicious node probabilistically terminates any RW that visits it. This stealthy behavior gradually eliminates active RWs from the network, effectively halting the learning process without triggering failure alarms. To counter this threat, we propose the Average Crossing (AC) algorithm--a fully decentralized mechanism for duplicating RWs to prevent RW extinction in the presence of Pac-Man. Our theoretical analysis establishes that (i) the RW population remains almost surely bounded under AC and (ii) RW-based stochastic gradient descent remains convergent under AC, even in the presence of Pac-Man, with a quantifiable deviation from the true optimum. Our extensive empirical results on both synthetic and real-world datasets corroborate our theoretical findings. Furthermore, they uncover a phase transition in the extinction probability as a function of the duplication threshold. We offer theoretical insights by analyzing a simplified variant of the AC, which sheds light on the observed phase transition.



## **42. Exploring the Secondary Risks of Large Language Models**

cs.LG

18 pages, 5 figures

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2506.12382v4) [paper-pdf](https://arxiv.org/pdf/2506.12382v4)

**Authors**: Jiawei Chen, Zhengwei Fang, Xiao Yang, Chao Yu, Zhaoxia Yin, Hang Su

**Abstract**: Ensuring the safety and alignment of Large Language Models is a significant challenge with their growing integration into critical applications and societal functions. While prior research has primarily focused on jailbreak attacks, less attention has been given to non-adversarial failures that subtly emerge during benign interactions. We introduce secondary risks a novel class of failure modes marked by harmful or misleading behaviors during benign prompts. Unlike adversarial attacks, these risks stem from imperfect generalization and often evade standard safety mechanisms. To enable systematic evaluation, we introduce two risk primitives verbose response and speculative advice that capture the core failure patterns. Building on these definitions, we propose SecLens, a black-box, multi-objective search framework that efficiently elicits secondary risk behaviors by optimizing task relevance, risk activation, and linguistic plausibility. To support reproducible evaluation, we release SecRiskBench, a benchmark dataset of 650 prompts covering eight diverse real-world risk categories. Experimental results from extensive evaluations on 16 popular models demonstrate that secondary risks are widespread, transferable across models, and modality independent, emphasizing the urgent need for enhanced safety mechanisms to address benign yet harmful LLM behaviors in real-world deployments.



## **43. Accelerating Targeted Hard-Label Adversarial Attacks in Low-Query Black-Box Settings**

cs.CV

This paper contains 10 pages, 8 figures and 8 tables. For associated supplementary code, see https://github.com/mdppml/TEA. This work has been accepted for publication at the IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). The final version will be available on IEEE Xplore

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2505.16313v3) [paper-pdf](https://arxiv.org/pdf/2505.16313v3)

**Authors**: Arjhun Swaminathan, Mete Akgün

**Abstract**: Deep neural networks for image classification remain vulnerable to adversarial examples -- small, imperceptible perturbations that induce misclassifications. In black-box settings, where only the final prediction is accessible, crafting targeted attacks that aim to misclassify into a specific target class is particularly challenging due to narrow decision regions. Current state-of-the-art methods often exploit the geometric properties of the decision boundary separating a source image and a target image rather than incorporating information from the images themselves. In contrast, we propose Targeted Edge-informed Attack (TEA), a novel attack that utilizes edge information from the target image to carefully perturb it, thereby producing an adversarial image that is closer to the source image while still achieving the desired target classification. Our approach consistently outperforms current state-of-the-art methods across different models in low query settings (nearly 70% fewer queries are used), a scenario especially relevant in real-world applications with limited queries and black-box access. Furthermore, by efficiently generating a suitable adversarial example, TEA provides an improved target initialization for established geometry-based attacks.



## **44. A Cross-Layer Analysis of Network Antifragility with RIS-assisted Links under Jamming Attacks**

cs.NI

This paper is uploaded here for research community, thus it is for non-commercial purposes

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2505.02565v2) [paper-pdf](https://arxiv.org/pdf/2505.02565v2)

**Authors**: Mounir Bensalem, Thomas Röthig, Admela Jukan

**Abstract**: Antifragility is an economics term defined as measure of (monetary) benefits gained from the adverse events and variability of the markets. This paper integrates for the first time the antifragility into the network based on communication links with Reconfigurable Intelligent Surface (RIS) affected by a jamming attack. We analyze whether antifragility can be achieved for several jamming models. Beyond the link-level gains, the results reveal how antifragile RIS-assisted links can be integrated into multi-hop systems to improve end-to-end network resilience, connectivity, and throughput under adversarial effects.



## **45. BadPatches: Routing-aware Backdoor Attacks on Vision Mixture of Experts**

cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2505.01811v3) [paper-pdf](https://arxiv.org/pdf/2505.01811v3)

**Authors**: Cedric Chan, Jona te Lintelo, Stjepan Picek

**Abstract**: Mixture of Experts (MoE) architectures have gained popularity for reducing computational costs in deep neural networks by activating only a subset of parameters during inference. While this efficiency makes MoE attractive for vision tasks, the patch-based processing in vision models introduces new methods for adversaries to perform backdoor attacks. In this work, we investigate the vulnerability of vision MoE models for image classification, specifically the patch-based MoE (pMoE) models and MoE-based vision transformers, against backdoor attacks. We propose a novel routing-aware trigger application method BadPatches, which is designed for patch-based processing in vision MoE models. BadPatches applies triggers on image patches rather than on the entire image. We show that BadPatches achieves high attack success rates (ASRs) with lower poisoning rates than routing-agnostic triggers and is successful at poisoning rates as low as 0.01% with an ASR above 80% on pMoE. Moreover, BadPatches is still effective when an adversary does not have complete knowledge of the patch routing configuration of the considered models. Next, we explore how trigger design affects pMoE patch routing. Finally, we investigate fine-pruning as a defense. Results show that only the fine-tuning stage of fine-pruning removes the backdoor from the model.



## **46. GreedyPixel: Fine-Grained Black-Box Adversarial Attack Via Greedy Algorithm**

cs.CV

IEEE Transactions on Information Forensics and Security

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2501.14230v4) [paper-pdf](https://arxiv.org/pdf/2501.14230v4)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Christopher Leckie, Isao Echizen

**Abstract**: Deep neural networks are highly vulnerable to adversarial examples, which are inputs with small, carefully crafted perturbations that cause misclassification -- making adversarial attacks a critical tool for evaluating robustness. Existing black-box methods typically entail a trade-off between precision and flexibility: pixel-sparse attacks (e.g., single- or few-pixel attacks) provide fine-grained control but lack adaptability, whereas patch- or frequency-based attacks improve efficiency or transferability, but at the cost of producing larger and less precise perturbations. We present GreedyPixel, a fine-grained black-box attack method that performs brute-force-style, per-pixel greedy optimization guided by a surrogate-derived priority map and refined by means of query feedback. It evaluates each coordinate directly without any gradient information, guaranteeing monotonic loss reduction and convergence to a coordinate-wise optimum, while also yielding near white-box-level precision and pixel-wise sparsity and perceptual quality. On the CIFAR-10 and ImageNet datasets, spanning convolutional neural networks (CNNs) and Transformer models, GreedyPixel achieved state-of-the-art success rates with visually imperceptible perturbations, effectively bridging the gap between black-box practicality and white-box performance. The implementation is available at https://github.com/azrealwang/greedypixel.



## **47. Adversarial Multi-Agent Reinforcement Learning for Proactive False Data Injection Detection**

eess.SY

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2411.12130v2) [paper-pdf](https://arxiv.org/pdf/2411.12130v2)

**Authors**: Kejun Chen, Truc Nguyen, Abhijeet Sahu, Malik Hassanaly

**Abstract**: Smart inverters are instrumental in the integration of distributed energy resources into the electric grid. Such inverters rely on communication layers for continuous control and monitoring, potentially exposing them to cyber-physical attacks such as false data injection attacks (FDIAs). We propose to construct a defense strategy against a priori unknown FDIAs with a multi-agent reinforcement learning (MARL) framework. The first agent is an adversary that simulates and discovers various FDIA strategies, while the second agent is a defender in charge of detecting and locating FDIAs. This approach enables the defender to be trained against new FDIAs continuously generated by the adversary. In addition, we show that the detection skills of an MARL defender can be combined with those of a supervised offline defender through a transfer learning approach. Numerical experiments conducted on a distribution and transmission system demonstrate that: a) the proposed MARL defender outperforms the offline defender against adversarial attacks; b) the transfer learning approach makes the MARL defender capable against both synthetic and unseen FDIAs.



## **48. Boosting Adversarial Transferability with Low-Cost Optimization via Maximin Expected Flatness**

cs.CV

Accepted by IEEE T-IFS

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2405.16181v3) [paper-pdf](https://arxiv.org/pdf/2405.16181v3)

**Authors**: Chunlin Qiu, Ang Li, Yiheng Duan, Shenyi Zhang, Yuanjie Zhang, Lingchen Zhao, Qian Wang

**Abstract**: Transfer-based attacks craft adversarial examples on white-box surrogate models and directly deploy them against black-box target models, offering model-agnostic and query-free threat scenarios. While flatness-enhanced methods have recently emerged to improve transferability by enhancing the loss surface flatness of adversarial examples, their divergent flatness definitions and heuristic attack designs suffer from unexamined optimization limitations and missing theoretical foundation, thus constraining their effectiveness and efficiency. This work exposes the severely imbalanced exploitation-exploration dynamics in flatness optimization, establishing the first theoretical foundation for flatness-based transferability and proposing a principled framework to overcome these optimization pitfalls. Specifically, we systematically unify fragmented flatness definitions across existing methods, revealing their imbalanced optimization limitations in over-exploration of sensitivity peaks or over-exploitation of local plateaus. To resolve these issues, we rigorously formalize average-case flatness and transferability gaps, proving that enhancing zeroth-order average-case flatness minimizes cross-model discrepancies. Building on this theory, we design a Maximin Expected Flatness (MEF) attack that enhances zeroth-order average-case flatness while balancing flatness exploration and exploitation. Extensive evaluations across 22 models and 24 current transfer-based attacks demonstrate MEF's superiority: it surpasses the state-of-the-art PGN attack by 4% in attack success rate at half the computational cost and achieves 8% higher success rate under the same budget. When combined with input augmentation, MEF attains 15% additional gains against defense-equipped models, establishing new robustness benchmarks. Our code is available at https://github.com/SignedQiu/MEFAttack.



## **49. A New Formulation for Zeroth-Order Optimization of Adversarial EXEmples in Malware Detection**

cs.LG

17 pages, 6 tables

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2405.14519v2) [paper-pdf](https://arxiv.org/pdf/2405.14519v2)

**Authors**: Marco Rando, Luca Demetrio, Lorenzo Rosasco, Fabio Roli

**Abstract**: Machine learning malware detectors are vulnerable to adversarial EXEmples, i.e., carefully-crafted Windows programs tailored to evade detection. Unlike other adversarial problems, attacks in this context must be functionality-preserving, a constraint that is challenging to address. As a consequence, heuristic algorithms are typically used, which inject new content, either randomly-picked or harvested from legitimate programs. In this paper, we show how learning malware detectors can be cast within a zeroth-order optimization framework, which allows incorporating functionality-preserving manipulations. This permits the deployment of sound and efficient gradient-free optimization algorithms, which come with theoretical guarantees and allow for minimal hyper-parameters tuning. As a by-product, we propose and study ZEXE, a novel zeroth-order attack against Windows malware detection. Compared to state-of-the-art techniques, ZEXE provides improvement in the evasion rate, reducing to less than one third the size of the injected content.



## **50. Visual Adversarial Attacks and Defenses in the Physical World: A Survey**

cs.CV

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2211.01671v6) [paper-pdf](https://arxiv.org/pdf/2211.01671v6)

**Authors**: Xingxing Wei, Bangzheng Pu, Shiji Zhao, Jiefan Lu, Baoyuan Wu

**Abstract**: Although Deep Neural Networks (DNNs) have been widely applied in various real-world scenarios, they remain vulnerable to adversarial examples. Adversarial attacks in computer vision can be categorized into digital attacks and physical attacks based on their different forms. Compared to digital attacks, which generate perturbations in digital pixels, physical attacks are more practical in real-world settings. Due to the serious security risks posed by physically adversarial examples, many studies have been conducted to evaluate the physically adversarial robustness of DNNs in recent years. In this paper, we provide a comprehensive survey of current physically adversarial attacks and defenses in computer vision. We establish a taxonomy by organizing physical attacks according to attack tasks, attack forms, and attack methods. This approach offers readers a systematic understanding of the topic from multiple perspectives. For physical defenses, we categorize them into pre-processing, in-processing, and post-processing for DNN models to ensure comprehensive coverage of adversarial defenses. Based on this survey, we discuss the challenges facing this research field and provide an outlook on future directions.



