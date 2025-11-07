# Latest Adversarial Attack Papers
**update at 2025-11-07 11:08:29**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Exploiting Data Structures for Bypassing and Crashing Anti-Malware Solutions via Telemetry Complexity Attacks**

cs.CR

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2511.04472v1) [paper-pdf](http://arxiv.org/pdf/2511.04472v1)

**Authors**: Evgenios Gkritsis, Constantinos Patsakis, George Stergiopoulos

**Abstract**: Anti-malware systems rely on sandboxes, hooks, and telemetry pipelines, including collection agents, serializers, and database backends, to monitor program and system behavior. We show that these data-handling components constitute an exploitable attack surface that can lead to denial-of-analysis (DoA) states without disabling sensors or requiring elevated privileges. As a result, we present \textit{Telemetry Complexity Attacks} (TCAs), a new class of vulnerabilities that exploit fundamental mismatches between unbounded collection mechanisms and bounded processing capabilities. Our method recursively spawns child processes to generate specially crafted, deeply nested, and oversized telemetry that stresses serialization and storage boundaries, as well as visualization layers, for example, JSON/BSON depth and size limits. Depending on the product, this leads to truncated or missing behavioral reports, rejected database inserts, serializer recursion and size errors, and unresponsive dashboards. In all of these cases, malicious activity is normally executed; however, depending on the examined solution, it is not recorded and/or not presented to the analysts. Therefore, instead of evading sensors, we break the pipeline that stores the data captured by the sensors.   We evaluate our technique against twelve commercial and open-source malware analysis platforms and endpoint detection and response (EDR) solutions. Seven products fail in different stages of the telemetry pipeline; two vendors assigned CVE identifiers (CVE-2025-61301 and CVE-2025-61303), and others issued patches or configuration changes. We discuss root causes and propose mitigation strategies to prevent DoA attacks triggered by adversarial telemetry.



## **2. Adversarially Robust and Interpretable Magecart Malware Detection**

cs.CR

5 pages, 2 figures

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2511.04440v1) [paper-pdf](http://arxiv.org/pdf/2511.04440v1)

**Authors**: Pedro Pereira, José Gouveia, João Vitorino, Eva Maia, Isabel Praça

**Abstract**: Magecart skimming attacks have emerged as a significant threat to client-side security and user trust in online payment systems. This paper addresses the challenge of achieving robust and explainable detection of Magecart attacks through a comparative study of various Machine Learning (ML) models with a real-world dataset. Tree-based, linear, and kernel-based models were applied, further enhanced through hyperparameter tuning and feature selection, to distinguish between benign and malicious scripts. Such models are supported by a Behavior Deterministic Finite Automaton (DFA) which captures structural behavior patterns in scripts, helping to analyze and classify client-side script execution logs. To ensure robustness against adversarial evasion attacks, the ML models were adversarially trained and evaluated using attacks from the Adversarial Robustness Toolbox and the Adaptative Perturbation Pattern Method. In addition, concise explanations of ML model decisions are provided, supporting transparency and user trust. Experimental validation demonstrated high detection performance and interpretable reasoning, demonstrating that traditional ML models can be effective in real-world web security contexts.



## **3. AdversariaLLM: A Unified and Modular Toolbox for LLM Robustness Research**

cs.AI

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2511.04316v1) [paper-pdf](http://arxiv.org/pdf/2511.04316v1)

**Authors**: Tim Beyer, Jonas Dornbusch, Jakob Steimle, Moritz Ladenburger, Leo Schwinn, Stephan Günnemann

**Abstract**: The rapid expansion of research on Large Language Model (LLM) safety and robustness has produced a fragmented and oftentimes buggy ecosystem of implementations, datasets, and evaluation methods. This fragmentation makes reproducibility and comparability across studies challenging, hindering meaningful progress. To address these issues, we introduce AdversariaLLM, a toolbox for conducting LLM jailbreak robustness research. Its design centers on reproducibility, correctness, and extensibility. The framework implements twelve adversarial attack algorithms, integrates seven benchmark datasets spanning harmfulness, over-refusal, and utility evaluation, and provides access to a wide range of open-weight LLMs via Hugging Face. The implementation includes advanced features for comparability and reproducibility such as compute-resource tracking, deterministic results, and distributional evaluation techniques. \name also integrates judging through the companion package JudgeZoo, which can also be used independently. Together, these components aim to establish a robust foundation for transparent, comparable, and reproducible research in LLM safety.



## **4. GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs**

cs.LG

Accepted to NeurIPS 2025. Project page and demos:  https://air-ml.org/project/gasp/

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2411.14133v3) [paper-pdf](http://arxiv.org/pdf/2411.14133v3)

**Authors**: Advik Raj Basani, Xiao Zhang

**Abstract**: LLMs have shown impressive capabilities across various natural language processing tasks, yet remain vulnerable to input prompts, known as jailbreak attacks, carefully designed to bypass safety guardrails and elicit harmful responses. Traditional methods rely on manual heuristics but suffer from limited generalizability. Despite being automatic, optimization-based attacks often produce unnatural prompts that can be easily detected by safety filters or require high computational costs due to discrete token optimization. In this paper, we introduce Generative Adversarial Suffix Prompter (GASP), a novel automated framework that can efficiently generate human-readable jailbreak prompts in a fully black-box setting. In particular, GASP leverages latent Bayesian optimization to craft adversarial suffixes by efficiently exploring continuous latent embedding spaces, gradually optimizing the suffix prompter to improve attack efficacy while balancing prompt coherence via a targeted iterative refinement procedure. Through comprehensive experiments, we show that GASP can produce natural adversarial prompts, significantly improving jailbreak success over baselines, reducing training times, and accelerating inference speed, thus making it an efficient and scalable solution for red-teaming LLMs.



## **5. SynFuzz: Leveraging Fuzzing of Netlist to Detect Synthesis Bugs**

cs.CR

15 pages, 10 figures, 5 tables

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2504.18812v3) [paper-pdf](http://arxiv.org/pdf/2504.18812v3)

**Authors**: Raghul Saravanan, Sudipta Paria, Aritra Dasgupta, Venkat Nitin Patnala, Swarup Bhunia, Sai Manoj P D

**Abstract**: In the evolving landscape of integrated circuit (IC) design, the increasing complexity of modern processors and intellectual property (IP) cores has introduced new challenges in ensuring design correctness and security. The recent advancements in hardware fuzzing techniques have shown their efficacy in detecting hardware bugs and vulnerabilities at the RTL abstraction level of hardware. However, they suffer from several limitations, including an inability to address vulnerabilities introduced during synthesis and gate-level transformations. These methods often fail to detect issues arising from library adversaries, where compromised or malicious library components can introduce backdoors or unintended behaviors into the design. In this paper, we present a novel hardware fuzzer, SynFuzz, designed to overcome the limitations of existing hardware fuzzing frameworks. SynFuzz focuses on fuzzing hardware at the gate-level netlist to identify synthesis bugs and vulnerabilities that arise during the transition from RTL to the gate-level. We analyze the intrinsic hardware behaviors using coverage metrics specifically tailored for the gate-level. Furthermore, SynFuzz implements differential fuzzing to uncover bugs associated with EDA libraries. We evaluated SynFuzz on popular open-source processors and IP designs, successfully identifying 7 new synthesis bugs. Additionally, by exploiting the optimization settings of EDA tools, we performed a compromised library mapping attack (CLiMA), creating a malicious version of hardware designs that remains undetectable by traditional verification methods. We also demonstrate how SynFuzz overcomes the limitations of the industry-standard formal verification tool, Cadence Conformal, providing a more robust and comprehensive approach to hardware verification.



## **6. Measuring the Security of Mobile LLM Agents under Adversarial Prompts from Untrusted Third-Party Channels**

cs.CR

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2510.27140v2) [paper-pdf](http://arxiv.org/pdf/2510.27140v2)

**Authors**: Chenghao Du, Quanfeng Huang, Tingxuan Tang, Zihao Wang, Adwait Nadkarni, Yue Xiao

**Abstract**: Large Language Models (LLMs) have transformed software development, enabling AI-powered applications known as LLM-based agents that promise to automate tasks across diverse apps and workflows. Yet, the security implications of deploying such agents in adversarial mobile environments remain poorly understood. In this paper, we present the first systematic study of security risks in mobile LLM agents. We design and evaluate a suite of adversarial case studies, ranging from opportunistic manipulations such as pop-up advertisements to advanced, end-to-end workflows involving malware installation and cross-app data exfiltration. Our evaluation covers eight state-of-the-art mobile agents across three architectures, with over 2,000 adversarial and paired benign trials. The results reveal systemic vulnerabilities: low-barrier vectors such as fraudulent ads succeed with over 80% reliability, while even workflows requiring the circumvention of operating-system warnings, such as malware installation, are consistently completed by advanced multi-app agents. By mapping these attacks to the MITRE ATT&CK Mobile framework, we uncover novel privilege-escalation and persistence pathways unique to LLM-driven automation. Collectively, our findings provide the first end-to-end evidence that mobile LLM agents are exploitable in realistic adversarial settings, where untrusted third-party channels (e.g., ads, embedded webviews, cross-app notifications) are an inherent part of the mobile ecosystem.



## **7. VERA: Variational Inference Framework for Jailbreaking Large Language Models**

cs.CR

Accepted by NeurIPS 2025

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2506.22666v2) [paper-pdf](http://arxiv.org/pdf/2506.22666v2)

**Authors**: Anamika Lochab, Lu Yan, Patrick Pynadath, Xiangyu Zhang, Ruqi Zhang

**Abstract**: The rise of API-only access to state-of-the-art LLMs highlights the need for effective black-box jailbreak methods to identify model vulnerabilities in real-world settings. Without a principled objective for gradient-based optimization, most existing approaches rely on genetic algorithms, which are limited by their initialization and dependence on manually curated prompt pools. Furthermore, these methods require individual optimization for each prompt, failing to provide a comprehensive characterization of model vulnerabilities. To address this gap, we introduce VERA: Variational infErence fRamework for jAilbreaking. VERA casts black-box jailbreak prompting as a variational inference problem, training a small attacker LLM to approximate the target LLM's posterior over adversarial prompts. Once trained, the attacker can generate diverse, fluent jailbreak prompts for a target query without re-optimization. Experimental results show that VERA achieves strong performance across a range of target LLMs, highlighting the value of probabilistic inference for adversarial prompt generation.



## **8. QSAFE-V: Quantum-Enhanced Lightweight Authentication Protocol Design for Vehicular Tactile Wireless Networks**

math.QA

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03850v1) [paper-pdf](http://arxiv.org/pdf/2511.03850v1)

**Authors**: Shakil Ahmed, Amika Tabassum, Ibrahim Almazyad, Ashfaq Khokhar

**Abstract**: With the rapid advancement of 6G technology, the Tactile Internet is emerging as a novel paradigm of interaction, particularly in intelligent transportation systems, where stringent demands for ultra-low latency and high reliability are prevalent. During the transmission and coordination of autonomous vehicles, malicious adversaries may attempt to compromise control commands or swarm behavior, posing severe threats to road safety and vehicular intelligence. Many existing authentication schemes claim to provide security against conventional attacks. However, recent developments in quantum computing have revealed critical vulnerabilities in these schemes, particularly under quantum-enabled adversarial models. In this context, the design of a quantum-secured, lightweight authentication scheme that is adaptable to vehicular mobility becomes essential. This paper proposes QSAFE-V, a quantum-secured authentication framework for edge-enabled vehicles that surpasses traditional security models. We conduct formal security proofs based on quantum key distribution and quantum adversary models, and also perform context-driven reauthentication analysis based on vehicular behavior. The output of quantum resilience evaluations indicates that QSAFE-V provides robust protection against quantum and contextual attacks. Furthermore, detailed performance analysis reveals that QSAFE-V achieves comparable communication and computation costs to classical schemes, while offering significantly stronger security guarantees under wireless Tactile Internet conditions.



## **9. Whisper Leak: a side-channel attack on Large Language Models**

cs.CR

14 pages, 7 figures

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03675v1) [paper-pdf](http://arxiv.org/pdf/2511.03675v1)

**Authors**: Geoff McDonald, Jonathan Bar Or

**Abstract**: Large Language Models (LLMs) are increasingly deployed in sensitive domains including healthcare, legal services, and confidential communications, where privacy is paramount. This paper introduces Whisper Leak, a side-channel attack that infers user prompt topics from encrypted LLM traffic by analyzing packet size and timing patterns in streaming responses. Despite TLS encryption protecting content, these metadata patterns leak sufficient information to enable topic classification. We demonstrate the attack across 28 popular LLMs from major providers, achieving near-perfect classification (often >98% AUPRC) and high precision even at extreme class imbalance (10,000:1 noise-to-target ratio). For many models, we achieve 100% precision in identifying sensitive topics like "money laundering" while recovering 5-20% of target conversations. This industry-wide vulnerability poses significant risks for users under network surveillance by ISPs, governments, or local adversaries. We evaluate three mitigation strategies - random padding, token batching, and packet injection - finding that while each reduces attack effectiveness, none provides complete protection. Through responsible disclosure, we have collaborated with providers to implement initial countermeasures. Our findings underscore the need for LLM providers to address metadata leakage as AI systems handle increasingly sensitive information.



## **10. SHIELD: Securing Healthcare IoT with Efficient Machine Learning Techniques for Anomaly Detection**

cs.LG

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03661v1) [paper-pdf](http://arxiv.org/pdf/2511.03661v1)

**Authors**: Mahek Desai, Apoorva Rumale, Marjan Asadinia

**Abstract**: The integration of IoT devices in healthcare introduces significant security and reliability challenges, increasing susceptibility to cyber threats and operational anomalies. This study proposes a machine learning-driven framework for (1) detecting malicious cyberattacks and (2) identifying faulty device anomalies, leveraging a dataset of 200,000 records. Eight machine learning models are evaluated across three learning approaches: supervised learning (XGBoost, K-Nearest Neighbors (K- NN)), semi-supervised learning (Generative Adversarial Networks (GAN), Variational Autoencoders (VAE)), and unsupervised learning (One-Class Support Vector Machine (SVM), Isolation Forest, Graph Neural Networks (GNN), and Long Short-Term Memory (LSTM) Autoencoders). The comprehensive evaluation was conducted across multiple metrics like F1-score, precision, recall, accuracy, ROC-AUC, computational efficiency. XGBoost achieved 99\% accuracy with minimal computational overhead (0.04s) for anomaly detection, while Isolation Forest balanced precision and recall effectively. LSTM Autoencoders underperformed with lower accuracy and higher latency. For attack detection, KNN achieved near-perfect precision, recall, and F1-score with the lowest computational cost (0.05s), followed by VAE at 97% accuracy. GAN showed the highest computational cost with lowest accuracy and ROC-AUC. These findings enhance IoT-enabled healthcare security through effective anomaly detection strategies. By improving early detection of cyber threats and device failures, this framework has the potential to prevent data breaches, minimize system downtime, and ensure the continuous and safe operation of medical devices, ultimately safeguarding patient health and trust in IoT-driven healthcare solutions.



## **11. Byzantine-Robust Federated Learning with Learnable Aggregation Weights**

cs.LG

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03529v1) [paper-pdf](http://arxiv.org/pdf/2511.03529v1)

**Authors**: Javad Parsa, Amir Hossein Daghestani, André M. H. Teixeira, Mikael Johansson

**Abstract**: Federated Learning (FL) enables clients to collaboratively train a global model without sharing their private data. However, the presence of malicious (Byzantine) clients poses significant challenges to the robustness of FL, particularly when data distributions across clients are heterogeneous. In this paper, we propose a novel Byzantine-robust FL optimization problem that incorporates adaptive weighting into the aggregation process. Unlike conventional approaches, our formulation treats aggregation weights as learnable parameters, jointly optimizing them alongside the global model parameters. To solve this optimization problem, we develop an alternating minimization algorithm with strong convergence guarantees under adversarial attack. We analyze the Byzantine resilience of the proposed objective. We evaluate the performance of our algorithm against state-of-the-art Byzantine-robust FL approaches across various datasets and attack scenarios. Experimental results demonstrate that our method consistently outperforms existing approaches, particularly in settings with highly heterogeneous data and a large proportion of malicious clients.



## **12. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

cs.CV

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2409.13174v4) [paper-pdf](http://arxiv.org/pdf/2409.13174v4)

**Authors**: Hao Cheng, Erjia Xiao, Yichi Wang, Chengyuan Yu, Mengshu Sun, Qiang Zhang, Jiahang Cao, Yijie Guo, Ning Liu, Kaidi Xu, Jize Zhang, Chao Shen, Philip Torr, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompt, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable \textbf{\textit{Analyses}} of how VLAMs respond to different physical threats.



## **13. Death by a Thousand Prompts: Open Model Vulnerability Analysis**

cs.CR

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03247v1) [paper-pdf](http://arxiv.org/pdf/2511.03247v1)

**Authors**: Amy Chang, Nicholas Conley, Harish Santhanalakshmi Ganesan, Adam Swanda

**Abstract**: Open-weight models provide researchers and developers with accessible foundations for diverse downstream applications. We tested the safety and security postures of eight open-weight large language models (LLMs) to identify vulnerabilities that may impact subsequent fine-tuning and deployment. Using automated adversarial testing, we measured each model's resilience against single-turn and multi-turn prompt injection and jailbreak attacks. Our findings reveal pervasive vulnerabilities across all tested models, with multi-turn attacks achieving success rates between 25.86\% and 92.78\% -- representing a $2\times$ to $10\times$ increase over single-turn baselines. These results underscore a systemic inability of current open-weight models to maintain safety guardrails across extended interactions. We assess that alignment strategies and lab priorities significantly influence resilience: capability-focused models such as Llama 3.3 and Qwen 3 demonstrate higher multi-turn susceptibility, whereas safety-oriented designs such as Google Gemma 3 exhibit more balanced performance.   The analysis concludes that open-weight models, while crucial for innovation, pose tangible operational and ethical risks when deployed without layered security controls. These findings are intended to inform practitioners and developers of the potential risks and the value of professional AI security solutions to mitigate exposure. Addressing multi-turn vulnerabilities is essential to ensure the safe, reliable, and responsible deployment of open-weight LLMs in enterprise and public domains. We recommend adopting a security-first design philosophy and layered protections to ensure resilient deployments of open-weight models.



## **14. Bayesian Advantage of Re-Identification Attack in the Shuffle Model**

cs.CR

Accepted by CSF 2026 -- 39th IEEE Computer Security Foundations  Symposium

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03213v1) [paper-pdf](http://arxiv.org/pdf/2511.03213v1)

**Authors**: Pengcheng Su, Haibo Cheng, Ping Wang

**Abstract**: The shuffle model, which anonymizes data by randomly permuting user messages, has been widely adopted in both cryptography and differential privacy. In this work, we present the first systematic study of the Bayesian advantage in re-identifying a user's message under the shuffle model. We begin with a basic setting: one sample is drawn from a distribution $P$, and $n - 1$ samples are drawn from a distribution $Q$, after which all $n$ samples are randomly shuffled. We define $\beta_n(P, Q)$ as the success probability of a Bayes-optimal adversary in identifying the sample from $P$, and define the additive and multiplicative Bayesian advantages as $\mathsf{Adv}_n^{+}(P, Q) = \beta_n(P,Q) - \frac{1}{n}$ and $\mathsf{Adv}_n^{\times}(P, Q) = n \cdot \beta_n(P,Q)$, respectively. We derive exact analytical expressions and asymptotic characterizations of $\beta_n(P, Q)$, along with evaluations in several representative scenarios. Furthermore, we establish (nearly) tight mutual bounds between the additive Bayesian advantage and the total variation distance. Finally, we extend our analysis beyond the basic setting and present, for the first time, an upper bound on the success probability of Bayesian attacks in shuffle differential privacy. Specifically, when the outputs of $n$ users -- each processed through an $\varepsilon$-differentially private local randomizer -- are shuffled, the probability that an attacker successfully re-identifies any target user's message is at most $e^{\varepsilon}/n$.



## **15. SAAIPAA: Optimizing aspect-angles-invariant physical adversarial attacks on SAR target recognition models**

eess.IV

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03192v1) [paper-pdf](http://arxiv.org/pdf/2511.03192v1)

**Authors**: Isar Lemeire, Yee Wei Law, Sang-Heon Lee, Will Meakin, Tat-Jun Chin

**Abstract**: Synthetic aperture radar (SAR) enables versatile, all-time, all-weather remote sensing. Coupled with automatic target recognition (ATR) leveraging machine learning (ML), SAR is empowering a wide range of Earth observation and surveillance applications. However, the surge of attacks based on adversarial perturbations against the ML algorithms underpinning SAR ATR is prompting the need for systematic research into adversarial perturbation mechanisms. Research in this area began in the digital (image) domain and evolved into the physical (signal) domain, resulting in physical adversarial attacks (PAAs) that strategically exploit corner reflectors as attack vectors to evade ML-based ATR. This paper proposes a novel framework called SAR Aspect-Angles-Invariant Physical Adversarial Attack (SAAIPAA) for physics-based modelling of reflector-actuated adversarial perturbations, which improves on the rigor of prior work. A unique feature of SAAIPAA is its ability to remain effective even when the attacker lacks knowledge of the SAR platform's aspect angles, by deploying at least one reflector in each azimuthal quadrant and optimizing reflector orientations. The resultant physical evasion attacks are efficiently realizable and optimal over the considered range of aspect angles between a SAR platform and a target, achieving state-of-the-art fooling rates (over 80% for DenseNet-121 and ResNet50) in the white-box setting. When aspect angles are known to the attacker, an average fooling rate of 99.2% is attainable. In black-box settings, although the attack efficacy of SAAIPAA transfers well between some models (e.g., from ResNet50 to DenseNet121), the transferability to some models (e.g., MobileNetV2) can be improved. A useful outcome of using the MSTAR dataset for the experiments in this article, a method for generating bounding boxes for densely sampled azimuthal SAR datasets is introduced.



## **16. From Insight to Exploit: Leveraging LLM Collaboration for Adaptive Adversarial Text Generation**

cs.LG

Findings of the Association for Computational Linguistics: EMNLP 2025  (camera-ready)

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03128v1) [paper-pdf](http://arxiv.org/pdf/2511.03128v1)

**Authors**: Najrin Sultana, Md Rafi Ur Rashid, Kang Gu, Shagufta Mehnaz

**Abstract**: LLMs can provide substantial zero-shot performance on diverse tasks using a simple task prompt, eliminating the need for training or fine-tuning. However, when applying these models to sensitive tasks, it is crucial to thoroughly assess their robustness against adversarial inputs. In this work, we introduce Static Deceptor (StaDec) and Dynamic Deceptor (DyDec), two innovative attack frameworks designed to systematically generate dynamic and adaptive adversarial examples by leveraging the understanding of the LLMs. We produce subtle and natural-looking adversarial inputs that preserve semantic similarity to the original text while effectively deceiving the target LLM. By utilizing an automated, LLM-driven pipeline, we eliminate the dependence on external heuristics. Our attacks evolve with the advancements in LLMs and demonstrate strong transferability across models unknown to the attacker. Overall, this work provides a systematic approach for the self-assessment of an LLM's robustness. We release our code and data at https://github.com/Shukti042/AdversarialExample.



## **17. A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation**

cs.LG

Accepted at the 39th Conference on Neural Information Processing  Systems (NeurIPS 2025)

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2404.11577v4) [paper-pdf](http://arxiv.org/pdf/2404.11577v4)

**Authors**: Yiwen Tu, Pingbang Hu, Jiaqi Ma

**Abstract**: Machine unlearning updates machine learning models to remove information from specific training samples, complying with data protection regulations that allow individuals to request the removal of their personal data. Despite the recent development of numerous unlearning algorithms, reliable evaluation of these algorithms remains an open research question. In this work, we focus on membership inference attack (MIA) based evaluation, one of the most common approaches for evaluating unlearning algorithms, and address various pitfalls of existing evaluation metrics lacking theoretical understanding and reliability. Specifically, by modeling the proposed evaluation process as a \emph{cryptographic game} between unlearning algorithms and MIA adversaries, the naturally induced evaluation metric measures the data removal efficacy of unlearning algorithms and enjoys provable guarantees that existing evaluation metrics fail to satisfy. Furthermore, we propose a practical and efficient approximation of the induced evaluation metric and demonstrate its effectiveness through both theoretical analysis and empirical experiments. Overall, this work presents a novel and reliable approach to empirically evaluating unlearning algorithms, paving the way for the development of more effective unlearning techniques.



## **18. Evaluating Control Protocols for Untrusted AI Agents**

cs.AI

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02997v1) [paper-pdf](http://arxiv.org/pdf/2511.02997v1)

**Authors**: Jon Kutasov, Chloe Loughridge, Yuqi Sun, Henry Sleight, Buck Shlegeris, Tyler Tracy, Joe Benton

**Abstract**: As AI systems become more capable and widely deployed as agents, ensuring their safe operation becomes critical. AI control offers one approach to mitigating the risk from untrusted AI agents by monitoring their actions and intervening or auditing when necessary. Evaluating the safety of these protocols requires understanding both their effectiveness against current attacks and their robustness to adaptive adversaries. In this work, we systematically evaluate a range of control protocols in SHADE-Arena, a dataset of diverse agentic environments. First, we evaluate blue team protocols, including deferral to trusted models, resampling, and deferring on critical actions, against a default attack policy. We find that resampling for incrimination and deferring on critical actions perform best, increasing safety from 50% to 96%. We then iterate on red team strategies against these protocols and find that attack policies with additional affordances, such as knowledge of when resampling occurs or the ability to simulate monitors, can substantially improve attack success rates against our resampling strategy, decreasing safety to 17%. However, deferring on critical actions is highly robust to even our strongest red team strategies, demonstrating the importance of denying attack policies access to protocol internals.



## **19. Diffusion Models are Robust Pretrainers**

eess.IV

To be published in IEEE Signal Processing Letters

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02793v1) [paper-pdf](http://arxiv.org/pdf/2511.02793v1)

**Authors**: Mika Yagoda, Shady Abu-Hussein, Raja Giryes

**Abstract**: Diffusion models have gained significant attention for high-fidelity image generation. Our work investigates the potential of exploiting diffusion models for adversarial robustness in image classification and object detection. Adversarial attacks challenge standard models in these tasks by perturbing inputs to force incorrect predictions. To address this issue, many approaches use training schemes for forcing the robustness of the models, which increase training costs. In this work, we study models built on top of off-the-shelf diffusion models and demonstrate their practical significance: they provide a low-cost path to robust representations, allowing lightweight heads to be trained on frozen features without full adversarial training. Our empirical evaluations on ImageNet, CIFAR-10, and PASCAL VOC show that diffusion-based classifiers and detectors achieve meaningful adversarial robustness with minimal compute. While clean and adversarial accuracies remain below state-of-the-art adversarially trained CNNs or ViTs, diffusion pretraining offers a favorable tradeoff between efficiency and robustness. This work opens a promising avenue for integrating diffusion models into resource-constrained robust deployments.



## **20. Enhancing Federated Learning Privacy with QUBO**

cs.LG

8 pages, 9 figures

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02785v1) [paper-pdf](http://arxiv.org/pdf/2511.02785v1)

**Authors**: Andras Ferenczi, Sutapa Samanta, Dagen Wang, Todd Hodges

**Abstract**: Federated learning (FL) is a widely used method for training machine learning (ML) models in a scalable way while preserving privacy (i.e., without centralizing raw data). Prior research shows that the risk of exposing sensitive data increases cumulatively as the number of iterations where a client's updates are included in the aggregated model increase. Attackers can launch membership inference attacks (MIA; deciding whether a sample or client participated), property inference attacks (PIA; inferring attributes of a client's data), and model inversion attacks (MI; reconstructing inputs), thereby inferring client-specific attributes and, in some cases, reconstructing inputs. In this paper, we mitigate risk by substantially reducing per client exposure using a quantum computing-inspired quadratic unconstrained binary optimization (QUBO) formulation that selects a small subset of client updates most relevant for each training round. In this work, we focus on two threat vectors: (i) information leakage by clients during training and (ii) adversaries who can query or obtain the global model. We assume a trusted central server and do not model server compromise. This method also assumes that the server has access to a validation/test set with global data distribution. Experiments on the MNIST dataset with 300 clients in 20 rounds showed a 95.2% per-round and 49% cumulative privacy exposure reduction, with 147 clients' updates never being used during training while maintaining in general the full-aggregation accuracy or even better. The method proved to be efficient at lower scale and more complex model as well. A CINIC-10 dataset-based experiment with 30 clients resulted in 82% per-round privacy improvement and 33% cumulative privacy.



## **21. Nesterov-Accelerated Robust Federated Learning Over Byzantine Adversaries**

cs.LG

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02657v1) [paper-pdf](http://arxiv.org/pdf/2511.02657v1)

**Authors**: Lihan Xu, Yanjie Dong, Gang Wang, Runhao Zeng, Xiaoyi Fan, Xiping Hu

**Abstract**: We investigate robust federated learning, where a group of workers collaboratively train a shared model under the orchestration of a central server in the presence of Byzantine adversaries capable of arbitrary and potentially malicious behaviors. To simultaneously enhance communication efficiency and robustness against such adversaries, we propose a Byzantine-resilient Nesterov-Accelerated Federated Learning (Byrd-NAFL) algorithm. Byrd-NAFL seamlessly integrates Nesterov's momentum into the federated learning process alongside Byzantine-resilient aggregation rules to achieve fast and safeguarding convergence against gradient corruption. We establish a finite-time convergence guarantee for Byrd-NAFL under non-convex and smooth loss functions with relaxed assumption on the aggregated gradients. Extensive numerical experiments validate the effectiveness of Byrd-NAFL and demonstrate the superiority over existing benchmarks in terms of convergence speed, accuracy, and resilience to diverse Byzantine attack strategies.



## **22. Do Methods to Jailbreak and Defend LLMs Generalize Across Languages?**

cs.CL

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.00689v2) [paper-pdf](http://arxiv.org/pdf/2511.00689v2)

**Authors**: Berk Atil, Rebecca J. Passonneau, Fred Morstatter

**Abstract**: Large language models (LLMs) undergo safety alignment after training and tuning, yet recent work shows that safety can be bypassed through jailbreak attacks. While many jailbreaks and defenses exist, their cross-lingual generalization remains underexplored. This paper presents the first systematic multilingual evaluation of jailbreaks and defenses across ten languages -- spanning high-, medium-, and low-resource languages -- using six LLMs on HarmBench and AdvBench. We assess two jailbreak types: logical-expression-based and adversarial-prompt-based. For both types, attack success and defense robustness vary across languages: high-resource languages are safer under standard queries but more vulnerable to adversarial ones. Simple defenses can be effective, but are language- and model-dependent. These findings call for language-aware and cross-lingual safety benchmarks for LLMs.



## **23. Verifying LLM Inference to Prevent Model Weight Exfiltration**

cs.CR

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02620v1) [paper-pdf](http://arxiv.org/pdf/2511.02620v1)

**Authors**: Roy Rinberg, Adam Karvonen, Alex Hoover, Daniel Reuter, Keri Warr

**Abstract**: As large AI models become increasingly valuable assets, the risk of model weight exfiltration from inference servers grows accordingly. An attacker controlling an inference server may exfiltrate model weights by hiding them within ordinary model outputs, a strategy known as steganography. This work investigates how to verify model responses to defend against such attacks and, more broadly, to detect anomalous or buggy behavior during inference. We formalize model exfiltration as a security game, propose a verification framework that can provably mitigate steganographic exfiltration, and specify the trust assumptions associated with our scheme. To enable verification, we characterize valid sources of non-determinism in large language model inference and introduce two practical estimators for them. We evaluate our detection framework on several open-weight models ranging from 3B to 30B parameters. On MOE-Qwen-30B, our detector reduces exfiltratable information to <0.5% with false-positive rate of 0.01%, corresponding to a >200x slowdown for adversaries. Overall, this work further establishes a foundation for defending against model weight exfiltration and demonstrates that strong protection can be achieved with minimal additional cost to inference providers.



## **24. Trustworthy Quantum Machine Learning: A Roadmap for Reliability, Robustness, and Security in the NISQ Era**

quant-ph

22 Pages

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02602v1) [paper-pdf](http://arxiv.org/pdf/2511.02602v1)

**Authors**: Ferhat Ozgur Catak, Jungwon Seo, Umit Cali

**Abstract**: Quantum machine learning (QML) is a promising paradigm for tackling computational problems that challenge classical AI. Yet, the inherent probabilistic behavior of quantum mechanics, device noise in NISQ hardware, and hybrid quantum-classical execution pipelines introduce new risks that prevent reliable deployment of QML in real-world, safety-critical settings. This research offers a broad roadmap for Trustworthy Quantum Machine Learning (TQML), integrating three foundational pillars of reliability: (i) uncertainty quantification for calibrated and risk-aware decision making, (ii) adversarial robustness against classical and quantum-native threat models, and (iii) privacy preservation in distributed and delegated quantum learning scenarios. We formalize quantum-specific trust metrics grounded in quantum information theory, including a variance-based decomposition of predictive uncertainty, trace-distance-bounded robustness, and differential privacy for hybrid learning channels. To demonstrate feasibility on current NISQ devices, we validate a unified trust assessment pipeline on parameterized quantum classifiers, uncovering correlations between uncertainty and prediction risk, an asymmetry in attack vulnerability between classical and quantum state perturbations, and privacy-utility trade-offs driven by shot noise and quantum channel noise. This roadmap seeks to define trustworthiness as a first-class design objective for quantum AI.



## **25. The Dark Side of LLMs: Agent-based Attacks for Complete Computer Takeover**

cs.CR

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2507.06850v5) [paper-pdf](http://arxiv.org/pdf/2507.06850v5)

**Authors**: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro

**Abstract**: The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables remarkable capabilities in natural language processing and generation. However, these systems introduce security vulnerabilities that extend beyond traditional content generation to system-level compromises. This paper presents a comprehensive evaluation of the LLMs security used as reasoning engines within autonomous agents, highlighting how they can be exploited as attack vectors capable of achieving computer takeovers. We focus on how different attack surfaces and trust boundaries can be leveraged to orchestrate such takeovers. We demonstrate that adversaries can effectively coerce popular LLMs into autonomously installing and executing malware on victim machines. Our evaluation of 18 state-of-the-art LLMs reveals an alarming scenario: 94.4% of models succumb to Direct Prompt Injection, and 83.3% are vulnerable to the more stealthy and evasive RAG Backdoor Attack. Notably, we tested trust boundaries within multi-agent systems, where LLM agents interact and influence each other, and we revealed that LLMs which successfully resist direct injection or RAG backdoor attacks will execute identical payloads when requested by peer agents. We found that 100.0% of tested LLMs can be compromised through Inter-Agent Trust Exploitation attacks, and that every model exhibits context-dependent security behaviors that create exploitable blind spots.



## **26. MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents**

cs.CR

NeurIPS 2025

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2503.10809v2) [paper-pdf](http://arxiv.org/pdf/2503.10809v2)

**Authors**: Lukas Aichberger, Alasdair Paren, Guohao Li, Philip Torr, Yarin Gal, Adel Bibi

**Abstract**: Recent advances in operating system (OS) agents have enabled vision-language models (VLMs) to directly control a user's computer. Unlike conventional VLMs that passively output text, OS agents autonomously perform computer-based tasks in response to a single user prompt. OS agents do so by capturing, parsing, and analysing screenshots and executing low-level actions via application programming interfaces (APIs), such as mouse clicks and keyboard inputs. This direct interaction with the OS significantly raises the stakes, as failures or manipulations can have immediate and tangible consequences. In this work, we uncover a novel attack vector against these OS agents: Malicious Image Patches (MIPs), adversarially perturbed screen regions that, when captured by an OS agent, induce it to perform harmful actions by exploiting specific APIs. For instance, a MIP can be embedded in a desktop wallpaper or shared on social media to cause an OS agent to exfiltrate sensitive user data. We show that MIPs generalise across user prompts and screen configurations, and that they can hijack multiple OS agents even during the execution of benign instructions. These findings expose critical security vulnerabilities in OS agents that have to be carefully addressed before their widespread deployment.



## **27. AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models**

cs.CL

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02376v1) [paper-pdf](http://arxiv.org/pdf/2511.02376v1)

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs, yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves up to 95% attack success rate on Llama-3.1-8B within six turns a 24 percent improvement over single turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests then iteratively refines them. Extensive evaluation across commercial and open-source models (GPT-4o-mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses.



## **28. SoK: Design, Vulnerabilities, and Security Measures of Cryptocurrency Wallets**

cs.CR

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2307.12874v5) [paper-pdf](http://arxiv.org/pdf/2307.12874v5)

**Authors**: Yimika Erinle, Yathin Kethepalli, Yebo Feng, Jiahua Xu

**Abstract**: With the advent of decentralised digital currencies powered by blockchain technology, a new era of peer-to-peer transactions has commenced. The rapid growth of the cryptocurrency economy has led to increased use of transaction-enabling wallets, making them a focal point for security risks. As the frequency of wallet-related incidents rises, there is a critical need for a systematic approach to measure and evaluate these attacks, drawing lessons from past incidents to enhance wallet security. In response, we introduce a multi-dimensional design taxonomy for existing and novel wallets with various design decisions. We classify existing industry wallets based on this taxonomy, identify previously occurring vulnerabilities and discuss the security implications of design decisions. We also systematise threats to the wallet mechanism and analyse the adversary's goals, capabilities and required knowledge. We present a multi-layered attack framework and investigate 84 incidents between 2012 and 2024, accounting for $5.4B. Following this, we classify defence implementations for these attacks on the precautionary and remedial axes. We map the mechanism and design decisions to vulnerabilities, attacks, and possible defence methods to discuss various insights.



## **29. Co-Evolving Complexity: An Adversarial Framework for Automatic MARL Curricula**

cs.LG

Published in the proceedings of the 39th Conference on Neural  Information Processing Systems (NeurIPS 2025) Workshop: Scaling Environments  for Agents (SEA)

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2509.03771v3) [paper-pdf](http://arxiv.org/pdf/2509.03771v3)

**Authors**: Brennen Hill

**Abstract**: The advancement of general-purpose intelligent agents is intrinsically linked to the environments in which they are trained. While scaling models and datasets has yielded remarkable capabilities, scaling the complexity, diversity, and interactivity of environments remains a crucial bottleneck. Hand-crafted environments are finite and often contain implicit biases, limiting the potential for agents to develop truly generalizable and robust skills. In this work, we propose a paradigm for generating a boundless and adaptive curriculum of challenges by framing the environment generation process as an adversarial game. We introduce a system where a team of cooperative multi-agent defenders learns to survive against a procedurally generative attacker. The attacker agent learns to produce increasingly challenging configurations of enemy units, dynamically creating novel worlds tailored to exploit the defenders' current weaknesses. Concurrently, the defender team learns cooperative strategies to overcome these generated threats. This co-evolutionary dynamic creates a self-scaling environment where complexity arises organically from the adversarial interaction, providing an effectively infinite stream of novel and relevant training data. We demonstrate that with minimal training, this approach leads to the emergence of complex, intelligent behaviors, such as flanking and shielding by the attacker, and focus-fire and spreading by the defenders. Our findings suggest that adversarial co-evolution is a powerful mechanism for automatically scaling environmental complexity, driving agents towards greater robustness and strategic depth.



## **30. Machine and Deep Learning for Indoor UWB Jammer Localization**

cs.LG

Accepted at the 20th International Conference on Risks and Security  of Internet and Systems (CRiSIS 2025, Gatineau-Canada,  https://crisis2025.uqo.ca/). The paper will soon be published as  post-proceedings in Springer's LNCS

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01819v1) [paper-pdf](http://arxiv.org/pdf/2511.01819v1)

**Authors**: Hamed Fard, Mahsa Kholghi, Benedikt Groß, Gerhard Wunder

**Abstract**: Ultra-wideband (UWB) localization delivers centimeter-scale accuracy but is vulnerable to jamming attacks, creating security risks for asset tracking and intrusion detection in smart buildings. Although machine learning (ML) and deep learning (DL) methods have improved tag localization, localizing malicious jammers within a single room and across changing indoor layouts remains largely unexplored. Two novel UWB datasets, collected under original and modified room configurations, are introduced to establish comprehensive ML/DL baselines. Performance is rigorously evaluated using a variety of classification and regression metrics. On the source dataset with the collected UWB features, Random Forest achieves the highest F1-macro score of 0.95 and XGBoost achieves the lowest mean Euclidean error of 20.16 cm. However, deploying these source-trained models in the modified room layout led to severe performance degradation, with XGBoost's mean Euclidean error increasing tenfold to 207.99 cm, demonstrating significant domain shift. To mitigate this degradation, a domain-adversarial ConvNeXt autoencoder (A-CNT) is proposed that leverages a gradient-reversal layer to align CIR-derived features across domains. The A-CNT framework restores localization performance by reducing the mean Euclidean error to 34.67 cm. This represents a 77 percent improvement over non-adversarial transfer learning and an 83 percent improvement over the best baseline, restoring the fraction of samples within 30 cm to 0.56. Overall, the results demonstrate that adversarial feature alignment enables robust and transferable indoor jammer localization despite environmental changes. Code and dataset available at https://github.com/afbf4c8996f/Jammer-Loc



## **31. Scam Shield: Multi-Model Voting and Fine-Tuned LLMs Against Adversarial Attacks**

cs.CR

8 pages

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01746v1) [paper-pdf](http://arxiv.org/pdf/2511.01746v1)

**Authors**: Chen-Wei Chang, Shailik Sarkar, Hossein Salemi, Hyungmin Kim, Shutonu Mitra, Hemant Purohit, Fengxiu Zhang, Michin Hong, Jin-Hee Cho, Chang-Tien Lu

**Abstract**: Scam detection remains a critical challenge in cybersecurity as adversaries craft messages that evade automated filters. We propose a Hierarchical Scam Detection System (HSDS) that combines a lightweight multi-model voting front end with a fine-tuned LLaMA 3.1 8B Instruct back end to improve accuracy and robustness against adversarial attacks. An ensemble of four classifiers provides preliminary predictions through majority vote, and ambiguous cases are escalated to the fine-tuned model, which is optimized with adversarial training to reduce misclassification. Experiments show that this hierarchical design both improves adversarial scam detection and shortens inference time by routing most cases away from the LLM, outperforming traditional machine-learning baselines and proprietary LLM baselines. The findings highlight the effectiveness of a hybrid voting mechanism and adversarial fine-tuning in fortifying LLMs against evolving scam tactics, enhancing the resilience of automated scam detection systems.



## **32. Black-Box Membership Inference Attack for LVLMs via Prior Knowledge-Calibrated Memory Probing**

cs.CR

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01952v1) [paper-pdf](http://arxiv.org/pdf/2511.01952v1)

**Authors**: Jinhua Yin, Peiru Yang, Chen Yang, Huili Wang, Zhiyang Hu, Shangguang Wang, Yongfeng Huang, Tao Qi

**Abstract**: Large vision-language models (LVLMs) derive their capabilities from extensive training on vast corpora of visual and textual data. Empowered by large-scale parameters, these models often exhibit strong memorization of their training data, rendering them susceptible to membership inference attacks (MIAs). Existing MIA methods for LVLMs typically operate under white- or gray-box assumptions, by extracting likelihood-based features for the suspected data samples based on the target LVLMs. However, mainstream LVLMs generally only expose generated outputs while concealing internal computational features during inference, limiting the applicability of these methods. In this work, we propose the first black-box MIA framework for LVLMs, based on a prior knowledge-calibrated memory probing mechanism. The core idea is to assess the model memorization of the private semantic information embedded within the suspected image data, which is unlikely to be inferred from general world knowledge alone. We conducted extensive experiments across four LVLMs and three datasets. Empirical results demonstrate that our method effectively identifies training data of LVLMs in a purely black-box setting and even achieves performance comparable to gray-box and white-box methods. Further analysis reveals the robustness of our method against potential adversarial manipulations, and the effectiveness of the methodology designs. Our code and data are available at https://github.com/spmede/KCMP.



## **33. SecDiff: Diffusion-Aided Secure Deep Joint Source-Channel Coding Against Adversarial Attacks**

cs.CV

13 pages, 6 figures

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01466v1) [paper-pdf](http://arxiv.org/pdf/2511.01466v1)

**Authors**: Changyuan Zhao, Jiacheng Wang, Ruichen Zhang, Dusit Niyato, Hongyang Du, Zehui Xiong, Dong In Kim, Ping Zhang

**Abstract**: Deep joint source-channel coding (JSCC) has emerged as a promising paradigm for semantic communication, delivering significant performance gains over conventional separate coding schemes. However, existing JSCC frameworks remain vulnerable to physical-layer adversarial threats, such as pilot spoofing and subcarrier jamming, compromising semantic fidelity. In this paper, we propose SecDiff, a plug-and-play, diffusion-aided decoding framework that significantly enhances the security and robustness of deep JSCC under adversarial wireless environments. Different from prior diffusion-guided JSCC methods that suffer from high inference latency, SecDiff employs pseudoinverse-guided sampling and adaptive guidance weighting, enabling flexible step-size control and efficient semantic reconstruction. To counter jamming attacks, we introduce a power-based subcarrier masking strategy and recast recovery as a masked inpainting problem, solved via diffusion guidance. For pilot spoofing, we formulate channel estimation as a blind inverse problem and develop an expectation-minimization (EM)-driven reconstruction algorithm, guided jointly by reconstruction loss and a channel operator. Notably, our method alternates between pilot recovery and channel estimation, enabling joint refinement of both variables throughout the diffusion process. Extensive experiments over orthogonal frequency-division multiplexing (OFDM) channels under adversarial conditions show that SecDiff outperforms existing secure and generative JSCC baselines by achieving a favorable trade-off between reconstruction quality and computational cost. This balance makes SecDiff a promising step toward practical, low-latency, and attack-resilient semantic communications.



## **34. Protecting the Neural Networks against FGSM Attack Using Machine Unlearning**

cs.LG

7 pages, 9 figures, 1 table

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01377v1) [paper-pdf](http://arxiv.org/pdf/2511.01377v1)

**Authors**: Amir Hossein Khorasani, Ali Jahanian, Maryam Rastgarpour

**Abstract**: Machine learning is a powerful tool for building predictive models. However, it is vulnerable to adversarial attacks. Fast Gradient Sign Method (FGSM) attacks are a common type of adversarial attack that adds small perturbations to input data to trick a model into misclassifying it. In response to these attacks, researchers have developed methods for "unlearning" these attacks, which involves retraining a model on the original data without the added perturbations. Machine unlearning is a technique that tries to "forget" specific data points from the training dataset, to improve the robustness of a machine learning model against adversarial attacks like FGSM. In this paper, we focus on applying unlearning techniques to the LeNet neural network, a popular architecture for image classification. We evaluate the efficacy of unlearning FGSM attacks on the LeNet network and find that it can significantly improve its robustness against these types of attacks.



## **35. Align to Misalign: Automatic LLM Jailbreak with Meta-Optimized LLM Judges**

cs.AI

under review, 28 pages

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01375v1) [paper-pdf](http://arxiv.org/pdf/2511.01375v1)

**Authors**: Hamin Koo, Minseon Kim, Jaehyung Kim

**Abstract**: Identifying the vulnerabilities of large language models (LLMs) is crucial for improving their safety by addressing inherent weaknesses. Jailbreaks, in which adversaries bypass safeguards with crafted input prompts, play a central role in red-teaming by probing LLMs to elicit unintended or unsafe behaviors. Recent optimization-based jailbreak approaches iteratively refine attack prompts by leveraging LLMs. However, they often rely heavily on either binary attack success rate (ASR) signals, which are sparse, or manually crafted scoring templates, which introduce human bias and uncertainty in the scoring outcomes. To address these limitations, we introduce AMIS (Align to MISalign), a meta-optimization framework that jointly evolves jailbreak prompts and scoring templates through a bi-level structure. In the inner loop, prompts are refined using fine-grained and dense feedback using a fixed scoring template. In the outer loop, the template is optimized using an ASR alignment score, gradually evolving to better reflect true attack outcomes across queries. This co-optimization process yields progressively stronger jailbreak prompts and more calibrated scoring signals. Evaluations on AdvBench and JBB-Behaviors demonstrate that AMIS achieves state-of-the-art performance, including 88.0% ASR on Claude-3.5-Haiku and 100.0% ASR on Claude-4-Sonnet, outperforming existing baselines by substantial margins.



## **36. On the Classical Hardness of the Semidirect Discrete Logarithm Problem in Finite Groups**

cs.CR

v2: Camera-ready version for Indocrypt 2025. Incorporated reviewer  feedback: simplified proofs, made computational assumptions explicit, fixed  technical errors

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2508.05048v2) [paper-pdf](http://arxiv.org/pdf/2508.05048v2)

**Authors**: Mohammad Ferry Husnil Arif, Muhammad Imran

**Abstract**: The semidirect discrete logarithm problem (SDLP) in finite groups was proposed as a foundation for post-quantum cryptographic protocols, based on the belief that its non-abelian structure would resist quantum attacks. However, recent results have shown that SDLP in finite groups admits efficient quantum algorithms, undermining its quantum resistance. This raises a fundamental question: does the SDLP offer any computational advantages over the standard discrete logarithm problem (DLP) against classical adversaries? In this work, we investigate the classical hardness of SDLP across different finite group platforms. We establish that the group-case SDLP can be reformulated as a generalized discrete logarithm problem, enabling adaptation of classical algorithms to study its complexity. We present a concrete adaptation of the Baby-Step Giant-Step algorithm for SDLP, achieving time and space complexity $O(\sqrt{r})$ where $r$ is the period of the underlying cycle structure. Through theoretical analysis and experimental validation in SageMath, we demonstrate that the classical hardness of SDLP is highly platform-dependent and does not uniformly exceed that of standard DLP. In finite fields $\mathbb{F}_p^*$, both problems exhibit comparable complexity. Surprisingly, in elliptic curves $E(\mathbb{F}_p)$, the SDLP becomes trivial due to the bounded automorphism group, while in elementary abelian groups $\mathbb{F}_p^n$, the SDLP can be harder than DLP, with complexity varying based on the eigenvalue structure of the automorphism. Our findings reveal that the non-abelian structure of semidirect products does not inherently guarantee increased classical hardness, suggesting that the search for classically hard problems for cryptographic applications requires more careful consideration of the underlying algebraic structures.



## **37. MiniFool -- Physics-Constraint-Aware Minimizer-Based Adversarial Attacks in Deep Neural Networks**

cs.LG

Submitted to Computing and Software for Big Science

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01352v1) [paper-pdf](http://arxiv.org/pdf/2511.01352v1)

**Authors**: Lucie Flek, Oliver Janik, Philipp Alexander Jung, Akbar Karimi, Timo Saala, Alexander Schmidt, Matthias Schott, Philipp Soldin, Matthias Thiesmeyer, Christopher Wiebusch, Ulrich Willemsen

**Abstract**: In this paper, we present a new algorithm, MiniFool, that implements physics-inspired adversarial attacks for testing neural network-based classification tasks in particle and astroparticle physics. While we initially developed the algorithm for the search for astrophysical tau neutrinos with the IceCube Neutrino Observatory, we apply it to further data from other science domains, thus demonstrating its general applicability. Here, we apply the algorithm to the well-known MNIST data set and furthermore, to Open Data data from the CMS experiment at the Large Hadron Collider. The algorithm is based on minimizing a cost function that combines a $\chi^2$ based test-statistic with the deviation from the desired target score. The test statistic quantifies the probability of the perturbations applied to the data based on the experimental uncertainties. For our studied use cases, we find that the likelihood of a flipped classification differs for both the initially correctly and incorrectly classified events. When testing changes of the classifications as a function of an attack parameter that scales the experimental uncertainties, the robustness of the network decision can be quantified. Furthermore, this allows testing the robustness of the classification of unlabeled experimental data.



## **38. A Generative Adversarial Approach to Adversarial Attacks Guided by Contrastive Language-Image Pre-trained Model**

cs.CV

18 pages, 3 figures

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01317v1) [paper-pdf](http://arxiv.org/pdf/2511.01317v1)

**Authors**: Sampriti Soor, Alik Pramanick, Jothiprakash K, Arijit Sur

**Abstract**: The rapid growth of deep learning has brought about powerful models that can handle various tasks, like identifying images and understanding language. However, adversarial attacks, an unnoticed alteration, can deceive models, leading to inaccurate predictions. In this paper, a generative adversarial attack method is proposed that uses the CLIP model to create highly effective and visually imperceptible adversarial perturbations. The CLIP model's ability to align text and image representation helps incorporate natural language semantics with a guided loss to generate effective adversarial examples that look identical to the original inputs. This integration allows extensive scene manipulation, creating perturbations in multi-object environments specifically designed to deceive multilabel classifiers. Our approach integrates the concentrated perturbation strategy from Saliency-based Auto-Encoder (SSAE) with the dissimilar text embeddings similar to Generative Adversarial Multi-Object Scene Attacks (GAMA), resulting in perturbations that both deceive classification models and maintain high structural similarity to the original images. The model was tested on various tasks across diverse black-box victim models. The experimental results show that our method performs competitively, achieving comparable or superior results to existing techniques, while preserving greater visual fidelity.



## **39. LSHFed: Robust and Communication-Efficient Federated Learning with Locally-Sensitive Hashing Gradient Mapping**

cs.LG

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01296v1) [paper-pdf](http://arxiv.org/pdf/2511.01296v1)

**Authors**: Guanjie Cheng, Mengzhen Yang, Xinkui Zhao, Shuyi Yu, Tianyu Du, Yangyang Wu, Mengying Zhu, Shuiguang Deng

**Abstract**: Federated learning (FL) enables collaborative model training across distributed nodes without exposing raw data, but its decentralized nature makes it vulnerable in trust-deficient environments. Inference attacks may recover sensitive information from gradient updates, while poisoning attacks can degrade model performance or induce malicious behaviors. Existing defenses often suffer from high communication and computation costs, or limited detection precision. To address these issues, we propose LSHFed, a robust and communication-efficient FL framework that simultaneously enhances aggregation robustness and privacy preservation. At its core, LSHFed incorporates LSHGM, a novel gradient verification mechanism that projects high-dimensional gradients into compact binary representations via multi-hyperplane locally-sensitive hashing. This enables accurate detection and filtering of malicious gradients using only their irreversible hash forms, thus mitigating privacy leakage risks and substantially reducing transmission overhead. Extensive experiments demonstrate that LSHFed maintains high model performance even when up to 50% of participants are collusive adversaries while achieving up to a 1000x reduction in gradient verification communication compared to full-gradient methods.



## **40. Rescuing the Unpoisoned: Efficient Defense against Knowledge Corruption Attacks on RAG Systems**

cs.CR

15 pages, 7 figures, 10 tables. To appear in the Proceedings of the  2025 Annual Computer Security Applications Conference (ACSAC)

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01268v1) [paper-pdf](http://arxiv.org/pdf/2511.01268v1)

**Authors**: Minseok Kim, Hankook Lee, Hyungjoon Koo

**Abstract**: Large language models (LLMs) are reshaping numerous facets of our daily lives, leading widespread adoption as web-based services. Despite their versatility, LLMs face notable challenges, such as generating hallucinated content and lacking access to up-to-date information. Lately, to address such limitations, Retrieval-Augmented Generation (RAG) has emerged as a promising direction by generating responses grounded in external knowledge sources. A typical RAG system consists of i) a retriever that probes a group of relevant passages from a knowledge base and ii) a generator that formulates a response based on the retrieved content. However, as with other AI systems, recent studies demonstrate the vulnerability of RAG, such as knowledge corruption attacks by injecting misleading information. In response, several defense strategies have been proposed, including having LLMs inspect the retrieved passages individually or fine-tuning robust retrievers. While effective, such approaches often come with substantial computational costs.   In this work, we introduce RAGDefender, a resource-efficient defense mechanism against knowledge corruption (i.e., by data poisoning) attacks in practical RAG deployments. RAGDefender operates during the post-retrieval phase, leveraging lightweight machine learning techniques to detect and filter out adversarial content without requiring additional model training or inference. Our empirical evaluations show that RAGDefender consistently outperforms existing state-of-the-art defenses across multiple models and adversarial scenarios: e.g., RAGDefender reduces the attack success rate (ASR) against the Gemini model from 0.89 to as low as 0.02, compared to 0.69 for RobustRAG and 0.24 for Discern-and-Answer when adversarial passages outnumber legitimate ones by a factor of four (4x).



## **41. Beyond Deceptive Flatness: Dual-Order Solution for Strengthening Adversarial Transferability**

cs.CV

Accepted by Pattern Recognition in Nov 01,2025

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01240v1) [paper-pdf](http://arxiv.org/pdf/2511.01240v1)

**Authors**: Zhixuan Zhang, Pingyu Wang, Xingjian Zheng, Linbo Qing, Qi Liu

**Abstract**: Transferable attacks generate adversarial examples on surrogate models to fool unknown victim models, posing real-world threats and growing research interest. Despite focusing on flat losses for transferable adversarial examples, recent studies still fall into suboptimal regions, especially the flat-yet-sharp areas, termed as deceptive flatness. In this paper, we introduce a novel black-box gradient-based transferable attack from a perspective of dual-order information. Specifically, we feasibly propose Adversarial Flatness (AF) to the deceptive flatness problem and a theoretical assurance for adversarial transferability. Based on this, using an efficient approximation of our objective, we instantiate our attack as Adversarial Flatness Attack (AFA), addressing the altered gradient sign issue. Additionally, to further improve the attack ability, we devise MonteCarlo Adversarial Sampling (MCAS) by enhancing the inner-loop sampling efficiency. The comprehensive results on ImageNet-compatible dataset demonstrate superiority over six baselines, generating adversarial examples in flatter regions and boosting transferability across model architectures. When tested on input transformation attacks or the Baidu Cloud API, our method outperforms baselines.



## **42. Bridging Symmetry and Robustness: On the Role of Equivariance in Enhancing Adversarial Robustness**

cs.LG

Accepted for the proceedings of 39th Conference on Neural Information  Processing Systems (NeurIPS 2025)

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2510.16171v3) [paper-pdf](http://arxiv.org/pdf/2510.16171v3)

**Authors**: Longwei Wang, Ifrat Ikhtear Uddin, KC Santosh, Chaowei Zhang, Xiao Qin, Yang Zhou

**Abstract**: Adversarial examples reveal critical vulnerabilities in deep neural networks by exploiting their sensitivity to imperceptible input perturbations. While adversarial training remains the predominant defense strategy, it often incurs significant computational cost and may compromise clean-data accuracy. In this work, we investigate an architectural approach to adversarial robustness by embedding group-equivariant convolutions-specifically, rotation- and scale-equivariant layers-into standard convolutional neural networks (CNNs). These layers encode symmetry priors that align model behavior with structured transformations in the input space, promoting smoother decision boundaries and greater resilience to adversarial attacks. We propose and evaluate two symmetry-aware architectures: a parallel design that processes standard and equivariant features independently before fusion, and a cascaded design that applies equivariant operations sequentially. Theoretically, we demonstrate that such models reduce hypothesis space complexity, regularize gradients, and yield tighter certified robustness bounds under the CLEVER (Cross Lipschitz Extreme Value for nEtwork Robustness) framework. Empirically, our models consistently improve adversarial robustness and generalization across CIFAR-10, CIFAR-100, and CIFAR-10C under both FGSM and PGD attacks, without requiring adversarial training. These findings underscore the potential of symmetry-enforcing architectures as efficient and principled alternatives to data augmentation-based defenses.



## **43. Adapt under Attack and Domain Shift: Unified Adversarial Meta-Learning and Domain Adaptation for Robust Automatic Modulation Classification**

cs.LG

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01172v1) [paper-pdf](http://arxiv.org/pdf/2511.01172v1)

**Authors**: Ali Owfi, Amirmohammad Bamdad, Tolunay Seyfi, Fatemeh Afghah

**Abstract**: Deep learning has emerged as a leading approach for Automatic Modulation Classification (AMC), demonstrating superior performance over traditional methods. However, vulnerability to adversarial attacks and susceptibility to data distribution shifts hinder their practical deployment in real-world, dynamic environments. To address these threats, we propose a novel, unified framework that integrates meta-learning with domain adaptation, making AMC systems resistant to both adversarial attacks and environmental changes. Our framework utilizes a two-phase strategy. First, in an offline phase, we employ a meta-learning approach to train the model on clean and adversarially perturbed samples from a single source domain. This method enables the model to generalize its defense, making it resistant to a combination of previously unseen attacks. Subsequently, in the online phase, we apply domain adaptation to align the model's features with a new target domain, allowing it to adapt without requiring substantial labeled data. As a result, our framework achieves a significant improvement in modulation classification accuracy against these combined threats, offering a critical solution to the deployment and operational challenges of modern AMC systems.



## **44. DepthVanish: Optimizing Adversarial Interval Structures for Stereo-Depth-Invisible Patches**

cs.CV

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2506.16690v2) [paper-pdf](http://arxiv.org/pdf/2506.16690v2)

**Authors**: Yun Xing, Yue Cao, Nhat Chung, Jie Zhang, Ivor Tsang, Ming-Ming Cheng, Yang Liu, Lei Ma, Qing Guo

**Abstract**: Stereo depth estimation is a critical task in autonomous driving and robotics, where inaccuracies (such as misidentifying nearby objects as distant) can lead to dangerous situations. Adversarial attacks against stereo depth estimation can help reveal vulnerabilities before deployment. Previous works have shown that repeating optimized textures can effectively mislead stereo depth estimation in digital settings. However, our research reveals that these naively repeated textures perform poorly in physical implementations, i.e., when deployed as patches, limiting their practical utility for stress-testing stereo depth estimation systems. In this work, for the first time, we discover that introducing regular intervals among the repeated textures, creating a grid structure, significantly enhances the patch's attack performance. Through extensive experimentation, we analyze how variations of this novel structure influence the adversarial effectiveness. Based on these insights, we develop a novel stereo depth attack that jointly optimizes both the interval structure and texture elements. Our generated adversarial patches can be inserted into any scenes and successfully attack advanced stereo depth estimation methods of different paradigms, i.e., RAFT-Stereo and STTR. Most critically, our patch can also attack commercial RGB-D cameras (Intel RealSense) in real-world conditions, demonstrating their practical relevance for security assessment of stereo systems. The code is officially released at: https://github.com/WiWiN42/DepthVanish



## **45. Stochastic Regret Guarantees for Online Zeroth- and First-Order Bilevel Optimization**

cs.LG

Published at NeurIPS 2025. 88 pages and 3 figures

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01126v1) [paper-pdf](http://arxiv.org/pdf/2511.01126v1)

**Authors**: Parvin Nazari, Bojian Hou, Davoud Ataee Tarzanagh, Li Shen, George Michailidis

**Abstract**: Online bilevel optimization (OBO) is a powerful framework for machine learning problems where both outer and inner objectives evolve over time, requiring dynamic updates. Current OBO approaches rely on deterministic \textit{window-smoothed} regret minimization, which may not accurately reflect system performance when functions change rapidly. In this work, we introduce a novel search direction and show that both first- and zeroth-order (ZO) stochastic OBO algorithms leveraging this direction achieve sublinear {stochastic bilevel regret without window smoothing}. Beyond these guarantees, our framework enhances efficiency by: (i) reducing oracle dependence in hypergradient estimation, (ii) updating inner and outer variables alongside the linear system solution, and (iii) employing ZO-based estimation of Hessians, Jacobians, and gradients. Experiments on online parametric loss tuning and black-box adversarial attacks validate our approach.



## **46. T-MLA: A Targeted Multiscale Log--Exponential Attack Framework for Neural Image Compression**

cs.CV

Submitted to Information Systems. Code will be released upon journal  publication

**SubmitDate**: 2025-11-02    [abs](http://arxiv.org/abs/2511.01079v1) [paper-pdf](http://arxiv.org/pdf/2511.01079v1)

**Authors**: Nikolay I. Kalmykov, Razan Dibo, Kaiyu Shen, Xu Zhonghan, Anh-Huy Phan, Yipeng Liu, Ivan Oseledets

**Abstract**: Neural image compression (NIC) has become the state-of-the-art for rate-distortion performance, yet its security vulnerabilities remain significantly less understood than those of classifiers. Existing adversarial attacks on NICs are often naive adaptations of pixel-space methods, overlooking the unique, structured nature of the compression pipeline. In this work, we propose a more advanced class of vulnerabilities by introducing T-MLA, the first targeted multiscale log--exponential attack framework. Our approach crafts adversarial perturbations in the wavelet domain by directly targeting the quality of the attacked and reconstructed images. This allows for a principled, offline attack where perturbations are strategically confined to specific wavelet subbands, maximizing distortion while ensuring perceptual stealth. Extensive evaluation across multiple state-of-the-art NIC architectures on standard image compression benchmarks reveals a large drop in reconstruction quality while the perturbations remain visually imperceptible. Our findings reveal a critical security flaw at the core of generative and content delivery pipelines.



## **47. Secure Distributed Consensus Estimation under False Data Injection Attacks: A Defense Strategy Based on Partial Channel Coding**

eess.SY

**SubmitDate**: 2025-11-02    [abs](http://arxiv.org/abs/2511.00963v1) [paper-pdf](http://arxiv.org/pdf/2511.00963v1)

**Authors**: Jiahao Huang, Marios M. Polycarpou, Wen Yang, Fangfei Li, Yang Tang

**Abstract**: This article investigates the security issue caused by false data injection attacks in distributed estimation, wherein each sensor can construct two types of residues based on local estimates and neighbor information, respectively. The resource-constrained attacker can select partial channels from the sensor network and arbitrarily manipulate the transmitted data. We derive necessary and sufficient conditions to reveal system vulnerabilities, under which the attacker is able to diverge the estimation error while preserving the stealthiness of all residues. We propose two defense strategies with mechanisms of exploiting the Euclidean distance between local estimates to detect attacks, and adopting the coding scheme to protect the transmitted data, respectively. It is proven that the former has the capability to address the majority of security loopholes, while the latter can serve as an additional enhancement to the former. By employing the time-varying coding matrix to mitigate the risk of being cracked, we demonstrate that the latter can safeguard against adversaries injecting stealthy sequences into the encoded channels. Hence, drawing upon the security analysis, we further provide a procedure to select security-critical channels that need to be encoded, thereby achieving a trade-off between security and coding costs. Finally, some numerical simulations are conducted to demonstrate the theoretical results.



## **48. Secure Distributed RIS-MIMO over Double Scattering Channels: Adversarial Attack, Defense, and SER Improvement**

cs.IT

15 pages, 7 figures, 7 tables. Accepted by IEEE TCOM

**SubmitDate**: 2025-11-02    [abs](http://arxiv.org/abs/2511.00959v1) [paper-pdf](http://arxiv.org/pdf/2511.00959v1)

**Authors**: Bui Duc Son, Gaosheng Zhao, Trinh Van Chien, Dong In Kim

**Abstract**: There has been a growing trend toward leveraging machine learning (ML) and deep learning (DL) techniques to optimize and enhance the performance of wireless communication systems. However, limited attention has been given to the vulnerabilities of these techniques, particularly in the presence of adversarial attacks. This paper investigates the adversarial attack and defense in distributed multiple reconfigurable intelligent surfaces (RISs)-aided multiple-input multiple-output (MIMO) communication systems-based autoencoder in finite scattering environments. We present the channel propagation model for distributed multiple RIS, including statistical information driven in closed form for the aggregated channel. The symbol error rate (SER) is selected to evaluate the collaborative dynamics between the distributed RISs and MIMO communication in depth. The relationship between the number of RISs and the SER of the proposed system based on an autoencoder, as well as the impact of adversarial attacks on the system's SER, is analyzed in detail. We also propose a defense mechanism based on adversarial training against the considered attacks to enhance the model's robustness. Numerical results indicate that increasing the number of RISs effectively reduces the system's SER but leads to the adversarial attack-based algorithm becoming more destructive in the white-box attack scenario. The proposed defense method demonstrates strong effectiveness by significantly mitigating the attack's impact. It also substantially reduces the system's SER in the absence of an attack compared to the original model. Moreover, we extend the phenomenon to include decoder mobility, demonstrating that the proposed method maintains robustness under Doppler-induced channel variations.



## **49. Leakage-abuse Attack Against Substring-SSE with Partially Known Dataset**

cs.CR

**SubmitDate**: 2025-11-02    [abs](http://arxiv.org/abs/2511.00930v1) [paper-pdf](http://arxiv.org/pdf/2511.00930v1)

**Authors**: Xijie Ba, Qin Liu, Xiaohong Li, Jianting Ning

**Abstract**: Substring-searchable symmetric encryption (substring-SSE) has become increasingly critical for privacy-preserving applications in cloud systems. However, existing schemes remain vulnerable to information leakage during search operations, particularly when adversaries possess partial knowledge of the target dataset. Although leakage-abuse attacks have been widely studied for traditional SSE, their applicability to substring-SSE under partially known data assumptions remains unexplored. In this paper, we present the first leakage-abuse attack on substring-SSE under partially-known dataset conditions. We develop a novel matrix-based correlation technique that extends and optimizes the LEAP framework for substring-SSE, enabling efficient recovery of plaintext data from encrypted suffix tree structures. Unlike existing approaches that rely on independent auxiliary datasets, our method directly exploits known data fragments to establish high-confidence mappings between ciphertext tokens and plaintext substrings through iterative matrix transformations. Comprehensive experiments on real-world datasets demonstrate the effectiveness of the attack, with recovery rates reaching 98.32% for substrings given 50% auxiliary knowledge. Even with only 10% prior knowledge, the attack achieves 74.42% substring recovery while maintaining strong scalability across datasets of varying sizes. The result reveals significant privacy risks in current substring-SSE designs and highlights the urgent need for leakage-resilient constructions.



## **50. Parameter Interpolation Adversarial Training for Robust Image Classification**

cs.CV

Accepted by TIFS 2025

**SubmitDate**: 2025-11-02    [abs](http://arxiv.org/abs/2511.00836v1) [paper-pdf](http://arxiv.org/pdf/2511.00836v1)

**Authors**: Xin Liu, Yichen Yang, Kun He, John E. Hopcroft

**Abstract**: Though deep neural networks exhibit superior performance on various tasks, they are still plagued by adversarial examples. Adversarial training has been demonstrated to be the most effective method to defend against adversarial attacks. However, existing adversarial training methods show that the model robustness has apparent oscillations and overfitting issues in the training process, degrading the defense efficacy. To address these issues, we propose a novel framework called Parameter Interpolation Adversarial Training (PIAT). PIAT tunes the model parameters between each epoch by interpolating the parameters of the previous and current epochs. It makes the decision boundary of model change more moderate and alleviates the overfitting issue, helping the model converge better and achieving higher model robustness. In addition, we suggest using the Normalized Mean Square Error (NMSE) to further improve the robustness by aligning the relative magnitude of logits between clean and adversarial examples rather than the absolute magnitude. Extensive experiments conducted on several benchmark datasets demonstrate that our framework could prominently improve the robustness of both Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs).



