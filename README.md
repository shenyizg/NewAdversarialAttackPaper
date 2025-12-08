# Latest Adversarial Attack Papers
**update at 2025-12-08 15:37:02**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Exposing Pink Slime Journalism: Linguistic Signatures and Robust Detection Against LLM-Generated Threats**

cs.CL

Published in RANLP 2025

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.05331v1) [paper-pdf](https://arxiv.org/pdf/2512.05331v1)

**Authors**: Sadat Shahriar, Navid Ayoobi, Arjun Mukherjee, Mostafa Musharrat, Sai Vishnu Vamsi

**Abstract**: The local news landscape, a vital source of reliable information for 28 million Americans, faces a growing threat from Pink Slime Journalism, a low-quality, auto-generated articles that mimic legitimate local reporting. Detecting these deceptive articles requires a fine-grained analysis of their linguistic, stylistic, and lexical characteristics. In this work, we conduct a comprehensive study to uncover the distinguishing patterns of Pink Slime content and propose detection strategies based on these insights. Beyond traditional generation methods, we highlight a new adversarial vector: modifications through large language models (LLMs). Our findings reveal that even consumer-accessible LLMs can significantly undermine existing detection systems, reducing their performance by up to 40% in F1-score. To counter this threat, we introduce a robust learning framework specifically designed to resist LLM-based adversarial attacks and adapt to the evolving landscape of automated pink slime journalism, and showed and improvement by up to 27%.



## **2. A Practical Honeypot-Based Threat Intelligence Framework for Cyber Defence in the Cloud**

cs.CR

6 pages

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.05321v1) [paper-pdf](https://arxiv.org/pdf/2512.05321v1)

**Authors**: Darren Malvern Chin, Bilal Isfaq, Simon Yusuf Enoch

**Abstract**: In cloud environments, conventional firewalls rely on predefined rules and manual configurations, limiting their ability to respond effectively to evolving or zero-day threats. As organizations increasingly adopt platforms such as Microsoft Azure, this static defense model exposes cloud assets to zero-day exploits, botnets, and advanced persistent threats. In this paper, we introduce an automated defense framework that leverages medium- to high-interaction honeypot telemetry to dynamically update firewall rules in real time. The framework integrates deception sensors (Cowrie), Azure-native automation tools (Monitor, Sentinel, Logic Apps), and MITRE ATT&CK-aligned detection within a closed-loop feedback mechanism. We developed a testbed to automatically observe adversary tactics, classify them using the MITRE ATT&CK framework, and mitigate network-level threats automatically with minimal human intervention.   To assess the framework's effectiveness, we defined and applied a set of attack- and defense-oriented security metrics. Building on existing adaptive defense strategies, our solution extends automated capabilities into cloud-native environments. The experimental results show an average Mean Time to Block of 0.86 seconds - significantly faster than benchmark systems - while accurately classifying over 12,000 SSH attempts across multiple MITRE ATT&CK tactics. These findings demonstrate that integrating deception telemetry with Azure-native automation reduces attacker dwell time, enhances SOC visibility, and provides a scalable, actionable defense model for modern cloud infrastructures.



## **3. Chameleon: Adaptive Adversarial Agents for Scaling-Based Visual Prompt Injection in Multimodal AI Systems**

cs.AI

5 pages, 2 figures, IEEE Transactions on Dependable and Secure Computing

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.04895v1) [paper-pdf](https://arxiv.org/pdf/2512.04895v1)

**Authors**: M Zeeshan, Saud Satti

**Abstract**: Multimodal Artificial Intelligence (AI) systems, particularly Vision-Language Models (VLMs), have become integral to critical applications ranging from autonomous decision-making to automated document processing. As these systems scale, they rely heavily on preprocessing pipelines to handle diverse inputs efficiently. However, this dependency on standard preprocessing operations, specifically image downscaling, creates a significant yet often overlooked security vulnerability. While intended for computational optimization, scaling algorithms can be exploited to conceal malicious visual prompts that are invisible to human observers but become active semantic instructions once processed by the model. Current adversarial strategies remain largely static, failing to account for the dynamic nature of modern agentic workflows. To address this gap, we propose Chameleon, a novel, adaptive adversarial framework designed to expose and exploit scaling vulnerabilities in production VLMs. Unlike traditional static attacks, Chameleon employs an iterative, agent-based optimization mechanism that dynamically refines image perturbations based on the target model's real-time feedback. This allows the framework to craft highly robust adversarial examples that survive standard downscaling operations to hijack downstream execution. We evaluate Chameleon against Gemini 2.5 Flash model. Our experiments demonstrate that Chameleon achieves an Attack Success Rate (ASR) of 84.5% across varying scaling factors, significantly outperforming static baseline attacks which average only 32.1%. Furthermore, we show that these attacks effectively compromise agentic pipelines, reducing decision-making accuracy by over 45% in multi-step tasks. Finally, we discuss the implications of these vulnerabilities and propose multi-scale consistency checks as a necessary defense mechanism.



## **4. SoK: a Comprehensive Causality Analysis Framework for Large Language Model Security**

cs.CR

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.04841v1) [paper-pdf](https://arxiv.org/pdf/2512.04841v1)

**Authors**: Wei Zhao, Zhe Li, Jun Sun

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities but remain vulnerable to adversarial manipulations such as jailbreaking, where crafted prompts bypass safety mechanisms. Understanding the causal factors behind such vulnerabilities is essential for building reliable defenses.   In this work, we introduce a unified causality analysis framework that systematically supports all levels of causal investigation in LLMs, ranging from token-level, neuron-level, and layer-level interventions to representation-level analysis. The framework enables consistent experimentation and comparison across diverse causality-based attack and defense methods. Accompanying this implementation, we provide the first comprehensive survey of causality-driven jailbreak studies and empirically evaluate the framework on multiple open-weight models and safety-critical benchmarks including jailbreaks, hallucination detection, backdoor identification, and fairness evaluation. Our results reveal that: (1) targeted interventions on causally critical components can reliably modify safety behavior; (2) safety-related mechanisms are highly localized (i.e., concentrated in early-to-middle layers with only 1--2\% of neurons exhibiting causal influence); and (3) causal features extracted from our framework achieve over 95\% detection accuracy across multiple threat types.   By bridging theoretical causality analysis and practical model safety, our framework establishes a reproducible foundation for research on causality-based attacks, interpretability, and robust attack detection and mitigation in LLMs. Code is available at https://github.com/Amadeuszhao/SOK_Casuality.



## **5. Malicious Image Analysis via Vision-Language Segmentation Fusion: Detection, Element, and Location in One-shot**

cs.CV

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.04599v1) [paper-pdf](https://arxiv.org/pdf/2512.04599v1)

**Authors**: Sheng Hang, Chaoxiang He, Hongsheng Hu, Hanqing Hu, Bin Benjamin Zhu, Shi-Feng Sun, Dawu Gu, Shuo Wang

**Abstract**: Detecting illicit visual content demands more than image-level NSFW flags; moderators must also know what objects make an image illegal and where those objects occur. We introduce a zero-shot pipeline that simultaneously (i) detects if an image contains harmful content, (ii) identifies each critical element involved, and (iii) localizes those elements with pixel-accurate masks - all in one pass. The system first applies foundation segmentation model (SAM) to generate candidate object masks and refines them into larger independent regions. Each region is scored for malicious relevance by a vision-language model using open-vocabulary prompts; these scores weight a fusion step that produces a consolidated malicious object map. An ensemble across multiple segmenters hardens the pipeline against adaptive attacks that target any single segmentation method. Evaluated on a newly-annotated 790-image dataset spanning drug, sexual, violent and extremist content, our method attains 85.8% element-level recall, 78.1% precision and a 92.1% segment-success rate - exceeding direct zero-shot VLM localization by 27.4% recall at comparable precision. Against PGD adversarial perturbations crafted to break SAM and VLM, our method's precision and recall decreased by no more than 10%, demonstrating high robustness against attacks. The full pipeline processes an image in seconds, plugs seamlessly into existing VLM workflows, and constitutes the first practical tool for fine-grained, explainable malicious-image moderation.



## **6. Counterfeit Answers: Adversarial Forgery against OCR-Free Document Visual Question Answering**

cs.CV

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.04554v1) [paper-pdf](https://arxiv.org/pdf/2512.04554v1)

**Authors**: Marco Pintore, Maura Pintor, Dimosthenis Karatzas, Battista Biggio

**Abstract**: Document Visual Question Answering (DocVQA) enables end-to-end reasoning grounded on information present in a document input. While recent models have shown impressive capabilities, they remain vulnerable to adversarial attacks. In this work, we introduce a novel attack scenario that aims to forge document content in a visually imperceptible yet semantically targeted manner, allowing an adversary to induce specific or generally incorrect answers from a DocVQA model. We develop specialized attack algorithms that can produce adversarially forged documents tailored to different attackers' goals, ranging from targeted misinformation to systematic model failure scenarios. We demonstrate the effectiveness of our approach against two end-to-end state-of-the-art models: Pix2Struct, a vision-language transformer that jointly processes image and text through sequence-to-sequence modeling, and Donut, a transformer-based model that directly extracts text and answers questions from document images. Our findings highlight critical vulnerabilities in current DocVQA systems and call for the development of more robust defenses.



## **7. Adversarial Limits of Quantum Certification: When Eve Defeats Detection**

quant-ph

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.04391v1) [paper-pdf](https://arxiv.org/pdf/2512.04391v1)

**Authors**: Davut Emre Tasar

**Abstract**: Security of quantum key distribution (QKD) relies on certifying that observed correlations arise from genuine quantum entanglement rather than eavesdropper manipulation. Theoretical security proofs assume idealized conditions, practical certification must contend with adaptive adversaries who optimize their attack strategies against detection systems. Established fundamental adversarial limits for quantum certification using Eve GAN, a generative adversarial network trained to produce classical correlations indistinguishable from quantum. Our central finding: when Eve interpolates her classical correlations with quantum data at mixing parameter, all tested detection methods achieve ROC AUC = 0.50, equivalent to random guessing. This means an eavesdropper needs only 5% classical admixture to completely evade detection. Critically, we discover that same distribution calibration a common practice in prior certification studies inflates detection performance by 44 percentage points compared to proper cross distribution evaluation, revealing a systematic flaw that may have led to overestimated security claims. Analysis of Popescu Rohrlich (PR Box) regime identifies a sharp phase transition at CHSH S = 2.05: below this value, no statistical method distinguishes classical from quantum correlations; above it, detection probability increases monotonically. Hardware validation on IBM Quantum demonstrates that Eve-GAN achieves CHSH = 2.736, remarkably exceeding real quantum hardware performance (CHSH = 2.691), illustrating that classical adversaries can outperform noisy quantum systems on standard certification metrics. These results have immediate implications for QKD security: adversaries maintaining 95% quantum fidelity evade all tested detection methods. We provide corrected methodology using cross-distribution calibration and recommend mandatory adversarial testing for quantum security claims.



## **8. One Detector Fits All: Robust and Adaptive Detection of Malicious Packages from PyPI to Enterprises**

cs.CR

Proceedings of the 2025 Annual Computer Security Applications Conference (ACSAC' 25), December 8-12, 2025, Honolulu, Hawaii, USA

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.04338v1) [paper-pdf](https://arxiv.org/pdf/2512.04338v1)

**Authors**: Biagio Montaruli, Luca Compagna, Serena Elisa Ponta, Davide Balzarotti

**Abstract**: The rise of supply chain attacks via malicious Python packages demands robust detection solutions. Current approaches, however, overlook two critical challenges: robustness against adversarial source code transformations and adaptability to the varying false positive rate (FPR) requirements of different actors, from repository maintainers (requiring low FPR) to enterprise security teams (higher FPR tolerance).   We introduce a robust detector capable of seamless integration into both public repositories like PyPI and enterprise ecosystems. To ensure robustness, we propose a novel methodology for generating adversarial packages using fine-grained code obfuscation. Combining these with adversarial training (AT) enhances detector robustness by 2.5x. We comprehensively evaluate AT effectiveness by testing our detector against 122,398 packages collected daily from PyPI over 80 days, showing that AT needs careful application: it makes the detector more robust to obfuscations and allows finding 10% more obfuscated packages, but slightly decreases performance on non-obfuscated packages.   We demonstrate production adaptability of our detector via two case studies: (i) one for PyPI maintainers (tuned at 0.1% FPR) and (ii) one for enterprise teams (tuned at 10% FPR). In the former, we analyze 91,949 packages collected from PyPI over 37 days, achieving a daily detection rate of 2.48 malicious packages with only 2.18 false positives. In the latter, we analyze 1,596 packages adopted by a multinational software company, obtaining only 1.24 false positives daily. These results show that our detector can be seamlessly integrated into both public repositories like PyPI and enterprise ecosystems, ensuring a very low time budget of a few minutes to review the false positives.   Overall, we uncovered 346 malicious packages, now reported to the community.



## **9. Studying Various Activation Functions and Non-IID Data for Machine Learning Model Robustness**

cs.LG

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.04264v1) [paper-pdf](https://arxiv.org/pdf/2512.04264v1)

**Authors**: Long Dang, Thushari Hapuarachchi, Kaiqi Xiong, Jing Lin

**Abstract**: Adversarial training is an effective method to improve the machine learning (ML) model robustness. Most existing studies typically consider the Rectified linear unit (ReLU) activation function and centralized training environments. In this paper, we study the ML model robustness using ten different activation functions through adversarial training in centralized environments and explore the ML model robustness in federal learning environments. In the centralized environment, we first propose an advanced adversarial training approach to improving the ML model robustness by incorporating model architecture change, soft labeling, simplified data augmentation, and varying learning rates. Then, we conduct extensive experiments on ten well-known activation functions in addition to ReLU to better understand how they impact the ML model robustness. Furthermore, we extend the proposed adversarial training approach to the federal learning environment, where both independent and identically distributed (IID) and non-IID data settings are considered. Our proposed centralized adversarial training approach achieves a natural and robust accuracy of 77.08% and 67.96%, respectively on CIFAR-10 against the fast gradient sign attacks. Experiments on ten activation functions reveal ReLU usually performs best. In the federated learning environment, however, the robust accuracy decreases significantly, especially on non-IID data. To address the significant performance drop in the non-IID data case, we introduce data sharing and achieve the natural and robust accuracy of 70.09% and 54.79%, respectively, surpassing the CalFAT algorithm, when 40% data sharing is used. That is, a proper percentage of data sharing can significantly improve the ML model robustness, which is useful to some real-world applications.



## **10. Out-of-the-box: Black-box Causal Attacks on Object Detectors**

cs.CV

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03730v1) [paper-pdf](https://arxiv.org/pdf/2512.03730v1)

**Authors**: Melane Navaratnarajah, David A. Kelly, Hana Chockler

**Abstract**: Adversarial perturbations are a useful way to expose vulnerabilities in object detectors. Existing perturbation methods are frequently white-box and architecture specific. More importantly, while they are often successful, it is rarely clear why they work. Insights into the mechanism of this success would allow developers to understand and analyze these attacks, as well as fine-tune the model to prevent them. This paper presents BlackCAtt, a black-box algorithm and a tool, which uses minimal, causally sufficient pixel sets to construct explainable, imperceptible, reproducible, architecture-agnostic attacks on object detectors. BlackCAtt combines causal pixels with bounding boxes produced by object detectors to create adversarial attacks that lead to the loss, modification or addition of a bounding box. BlackCAtt works across different object detectors of different sizes and architectures, treating the detector as a black box. We compare the performance of BlackCAtt with other black-box attack methods and show that identification of causal pixels leads to more precisely targeted and less perceptible attacks. On the COCO test dataset, our approach is 2.7 times better than the baseline in removing a detection, 3.86 times better in changing a detection, and 5.75 times better in triggering new, spurious, detections. The attacks generated by BlackCAtt are very close to the original image, and hence imperceptible, demonstrating the power of causal pixels.



## **11. Context-Aware Hierarchical Learning: A Two-Step Paradigm towards Safer LLMs**

cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03720v1) [paper-pdf](https://arxiv.org/pdf/2512.03720v1)

**Authors**: Tengyun Ma, Jiaqi Yao, Daojing He, Shihao Peng, Yu Li, Shaohui Liu, Zhuotao Tian

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for diverse applications. However, their uniform token processing paradigm introduces critical vulnerabilities in instruction handling, particularly when exposed to adversarial scenarios. In this work, we identify and propose a novel class of vulnerabilities, termed Tool-Completion Attack (TCA), which exploits function-calling mechanisms to subvert model behavior. To evaluate LLM robustness against such threats, we introduce the Tool-Completion benchmark, a comprehensive security assessment framework, which reveals that even state-of-the-art models remain susceptible to TCA, with surprisingly high attack success rates. To address these vulnerabilities, we introduce Context-Aware Hierarchical Learning (CAHL), a sophisticated mechanism that dynamically balances semantic comprehension with role-specific instruction constraints. CAHL leverages the contextual correlations between different instruction segments to establish a robust, context-aware instruction hierarchy. Extensive experiments demonstrate that CAHL significantly enhances LLM robustness against both conventional attacks and the proposed TCA, exhibiting strong generalization capabilities in zero-shot evaluations while still preserving model performance on generic tasks. Our code is available at https://github.com/S2AILab/CAHL.



## **12. A Descriptive Model for Modelling Attacker Decision-Making in Cyber-Deception**

cs.CR

24 Pages, 4 Tables

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03641v1) [paper-pdf](https://arxiv.org/pdf/2512.03641v1)

**Authors**: B. R. Turner, O. Guidetti, N. M. Karie, R. Ryan, Y. Yan

**Abstract**: Cyber-deception is an increasingly important defensive strategy, shaping adversarial decision making through controlled misinformation, uncertainty, and misdirection. Although game-theoretic, Bayesian, Markov decision process, and reinforcement learning models offer insight into deceptive interactions, they typically assume an attacker has already chosen to engage. Such approaches overlook cognitive and perceptual factors that influence an attacker's initial decision to engage or withdraw. This paper presents a descriptive model that incorporates the psychological and strategic elements shaping this decision. The model defines five components, belief (B), scepticism (S), deception fidelity (D), reconnaissance (R), and experience (E), which interact to capture how adversaries interpret deceptive cues and assess whether continued engagement is worthwhile. The framework provides a structured method for analysing engagement decisions in cyber-deception scenarios. A series of experiments has been designed to evaluate this model through Capture the Flag activities incorporating varying levels of deception, supported by behavioural and biometric observations. These experiments have not yet been conducted, and no experimental findings are presented in this paper. These experiments will combine behavioural observations with biometric indicators to produce a multidimensional view of adversarial responses. Findings will improve understanding of the factors influencing engagement decisions and refine the model's relevance to real-world cyber-deception settings. By addressing the gap in existing models that presume engagement, this work supports more cognitively realistic and strategically effective cyber-deception practices.



## **13. FeatureLens: A Highly Generalizable and Interpretable Framework for Detecting Adversarial Examples Based on Image Features**

cs.CV

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03625v1) [paper-pdf](https://arxiv.org/pdf/2512.03625v1)

**Authors**: Zhigang Yang, Yuan Liu, Jiawei Zhang, Puning Zhang, Xinqiang Ma

**Abstract**: Although the remarkable performance of deep neural networks (DNNs) in image classification, their vulnerability to adversarial attacks remains a critical challenge. Most existing detection methods rely on complex and poorly interpretable architectures, which compromise interpretability and generalization. To address this, we propose FeatureLens, a lightweight framework that acts as a lens to scrutinize anomalies in image features. Comprising an Image Feature Extractor (IFE) and shallow classifiers (e.g., SVM, MLP, or XGBoost) with model sizes ranging from 1,000 to 30,000 parameters, FeatureLens achieves high detection accuracy ranging from 97.8% to 99.75% in closed-set evaluation and 86.17% to 99.6% in generalization evaluation across FGSM, PGD, CW, and DAmageNet attacks, using only 51 dimensional features. By combining strong detection performance with excellent generalization, interpretability, and computational efficiency, FeatureLens offers a practical pathway toward transparent and effective adversarial defense.



## **14. Tuning for TraceTarnish: Techniques, Trends, and Testing Tangible Traits**

cs.CR

20 pages, 8 figures, 2 tables

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03465v1) [paper-pdf](https://arxiv.org/pdf/2512.03465v1)

**Authors**: Robert Dilworth

**Abstract**: In this study, we more rigorously evaluated our attack script $\textit{TraceTarnish}$, which leverages adversarial stylometry principles to anonymize the authorship of text-based messages. To ensure the efficacy and utility of our attack, we sourced, processed, and analyzed Reddit comments--comments that were later alchemized into $\textit{TraceTarnish}$ data--to gain valuable insights. The transformed $\textit{TraceTarnish}$ data was then further augmented by $\textit{StyloMetrix}$ to manufacture stylometric features--features that were culled using the Information Gain criterion, leaving only the most informative, predictive, and discriminative ones. Our results found that function words and function word types ($L\_FUNC\_A$ $\&$ $L\_FUNC\_T$); content words and content word types ($L\_CONT\_A$ $\&$ $L\_CONT\_T$); and the Type-Token Ratio ($ST\_TYPE\_TOKEN\_RATIO\_LEMMAS$) yielded significant Information-Gain readings. The identified stylometric cues--function-word frequencies, content-word distributions, and the Type-Token Ratio--serve as reliable indicators of compromise (IoCs), revealing when a text has been deliberately altered to mask its true author. Similarly, these features could function as forensic beacons, alerting defenders to the presence of an adversarial stylometry attack; granted, in the absence of the original message, this signal may go largely unnoticed, as it appears to depend on a pre- and post-transformation comparison. "In trying to erase a trace, you often imprint a larger one." Armed with this understanding, we framed $\textit{TraceTarnish}$'s operations and outputs around these five isolated features, using them to conceptualize and implement enhancements that further strengthen the attack.



## **15. Tipping the Dominos: Topology-Aware Multi-Hop Attacks on LLM-Based Multi-Agent Systems**

cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.04129v1) [paper-pdf](https://arxiv.org/pdf/2512.04129v1)

**Authors**: Ruichao Liang, Le Yin, Jing Chen, Cong Wu, Xiaoyu Zhang, Huangpeng Gu, Zijian Zhang, Yang Liu

**Abstract**: LLM-based multi-agent systems (MASs) have reshaped the digital landscape with their emergent coordination and problem-solving capabilities. However, current security evaluations of MASs are still confined to limited attack scenarios, leaving their security issues unclear and likely underestimated. To fill this gap, we propose TOMA, a topology-aware multi-hop attack scheme targeting MASs. By optimizing the propagation of contamination within the MAS topology and controlling the multi-hop diffusion of adversarial payloads originating from the environment, TOMA unveils new and effective attack vectors without requiring privileged access or direct agent manipulation. Experiments demonstrate attack success rates ranging from 40% to 78% across three state-of-the-art MAS architectures: \textsc{Magentic-One}, \textsc{LangManus}, and \textsc{OWL}, and five representative topologies, revealing intrinsic MAS vulnerabilities that may be overlooked by existing research. Inspired by these findings, we propose a conceptual defense framework based on topology trust, and prototype experiments show its effectiveness in blocking 94.8% of adaptive and composite attacks.



## **16. Immunity memory-based jailbreak detection: multi-agent adaptive guard for large language models**

cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03356v1) [paper-pdf](https://arxiv.org/pdf/2512.03356v1)

**Authors**: Jun Leng, Litian Zhang, Xi Zhang

**Abstract**: Large language models (LLMs) have become foundational in AI systems, yet they remain vulnerable to adversarial jailbreak attacks. These attacks involve carefully crafted prompts that bypass safety guardrails and induce models to produce harmful content. Detecting such malicious input queries is therefore critical for maintaining LLM safety. Existing methods for jailbreak detection typically involve fine-tuning LLMs as static safety LLMs using fixed training datasets. However, these methods incur substantial computational costs when updating model parameters to improve robustness, especially in the face of novel jailbreak attacks. Inspired by immunological memory mechanisms, we propose the Multi-Agent Adaptive Guard (MAAG) framework for jailbreak detection. The core idea is to equip guard with memory capabilities: upon encountering novel jailbreak attacks, the system memorizes attack patterns, enabling it to rapidly and accurately identify similar threats in future encounters. Specifically, MAAG first extracts activation values from input prompts and compares them to historical activations stored in a memory bank for quick preliminary detection. A defense agent then simulates responses based on these detection results, and an auxiliary agent supervises the simulation process to provide secondary filtering of the detection outcomes. Extensive experiments across five open-source models demonstrate that MAAG significantly outperforms state-of-the-art (SOTA) methods, achieving 98% detection accuracy and a 96% F1-score across a diverse range of attack scenarios.



## **17. Invasive Context Engineering to Control Large Language Models**

cs.AI

4 pages

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.03001v1) [paper-pdf](https://arxiv.org/pdf/2512.03001v1)

**Authors**: Thomas Rivasseau

**Abstract**: Current research on operator control of Large Language Models improves model robustness against adversarial attacks and misbehavior by training on preference examples, prompting, and input/output filtering. Despite good results, LLMs remain susceptible to abuse, and jailbreak probability increases with context length. There is a need for robust LLM security guarantees in long-context situations. We propose control sentences inserted into the LLM context as invasive context engineering to partially solve the problem. We suggest this technique can be generalized to the Chain-of-Thought process to prevent scheming. Invasive Context Engineering does not rely on LLM training, avoiding data shortage pitfalls which arise in training models for long context situations.



## **18. Defense That Attacks: How Robust Models Become Better Attackers**

cs.CV

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.02830v2) [paper-pdf](https://arxiv.org/pdf/2512.02830v2)

**Authors**: Mohamed Awad, Mahmoud Akrm, Walid Gomaa

**Abstract**: Deep learning has achieved great success in computer vision, but remains vulnerable to adversarial attacks. Adversarial training is the leading defense designed to improve model robustness. However, its effect on the transferability of attacks is underexplored. In this work, we ask whether adversarial training unintentionally increases the transferability of adversarial examples. To answer this, we trained a diverse zoo of 36 models, including CNNs and ViTs, and conducted comprehensive transferability experiments. Our results reveal a clear paradox: adversarially trained (AT) models produce perturbations that transfer more effectively than those from standard models, which introduce a new ecosystem risk. To enable reproducibility and further study, we release all models, code, and experimental scripts. Furthermore, we argue that robustness evaluations should assess not only the resistance of a model to transferred attacks but also its propensity to produce transferable adversarial examples.



## **19. Decryption Through Polynomial Ambiguity: Noise-Enhanced High-Memory Convolutional Codes for Post-Quantum Cryptography**

cs.CR

23 pages, 3 figures. arXiv admin note: substantial text overlap with arXiv:2510.15515

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.02822v2) [paper-pdf](https://arxiv.org/pdf/2512.02822v2)

**Authors**: Meir Ariel

**Abstract**: We present a novel approach to post-quantum cryptography that employs directed-graph decryption of noise-enhanced high-memory convolutional codes. The proposed construction generates random-like generator matrices that effectively conceal algebraic structure and resist known structural attacks. Security is further reinforced by the deliberate injection of strong noise during decryption, arising from polynomial division: while legitimate recipients retain polynomial-time decoding, adversaries face exponential-time complexity. As a result, the scheme achieves cryptanalytic security margins surpassing those of Classic McEliece by factors exceeding 2^(200). Beyond its enhanced security, the method offers greater design flexibility, supporting arbitrary plaintext lengths with linear-time decryption and uniform per-bit computational cost, enabling seamless scalability to long messages. Practical deployment is facilitated by parallel arrays of directed-graph decoders, which identify the correct plaintext through polynomial ambiguity while allowing efficient hardware and software implementations. Altogether, the scheme represents a compelling candidate for robust, scalable, and quantum-resistant public-key cryptography.



## **20. Characterizing Cyber Attacks against Space Infrastructures with Missing Data: Framework and Case Study**

cs.CR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.02414v1) [paper-pdf](https://arxiv.org/pdf/2512.02414v1)

**Authors**: Ekzhin Ear, Jose Luis Castanon Remy, Caleb Chang, Qiren Que, Antonia Feffer, Shouhuai Xu

**Abstract**: Cybersecurity of space infrastructures is an emerging topic, despite space-related cybersecurity incidents occurring as early as 1977 (i.e., hijacking of a satellite transmission signal). There is no single dataset that documents cyber attacks against space infrastructures that have occurred in the past; instead, these incidents are often scattered in media reports while missing many details, which we dub the missing-data problem. Nevertheless, even ``low-quality'' datasets containing such reports would be extremely valuable because of the dearth of space cybersecurity data and the sensitivity of space infrastructures which are often restricted from disclosure by governments. This prompts a research question: How can we characterize real-world cyber attacks against space infrastructures? In this paper, we address the problem by proposing a framework, including metrics, while also addressing the missing-data problem by leveraging methodologies such as the Space Attack Research and Tactic Analysis (SPARTA) and the Adversarial Tactics, Techniques, and Common Knowledge (ATT&CK) to ``extrapolate'' the missing data in a principled fashion. We show how the extrapolated data can be used to reconstruct ``hypothetical but plausible'' space cyber kill chains and space cyber attack campaigns that have occurred in practice. To show the usefulness of the framework, we extract data for 108 cyber attacks against space infrastructures and show how to extrapolate this ``low-quality'' dataset containing missing information to derive 6,206 attack technique-level space cyber kill chains. Our findings include: cyber attacks against space infrastructures are getting increasingly sophisticated; successful protection of the link segment between the space and user segments could have thwarted nearly half of the 108 attacks. We will make our dataset available.



## **21. LeechHijack: Covert Computational Resource Exploitation in Intelligent Agent Systems**

cs.CR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.02321v1) [paper-pdf](https://arxiv.org/pdf/2512.02321v1)

**Authors**: Yuanhe Zhang, Weiliu Wang, Zhenhong Zhou, Kun Wang, Jie Zhang, Li Sun, Yang Liu, Sen Su

**Abstract**: Large Language Model (LLM)-based agents have demonstrated remarkable capabilities in reasoning, planning, and tool usage. The recently proposed Model Context Protocol (MCP) has emerged as a unifying framework for integrating external tools into agent systems, enabling a thriving open ecosystem of community-built functionalities. However, the openness and composability that make MCP appealing also introduce a critical yet overlooked security assumption -- implicit trust in third-party tool providers. In this work, we identify and formalize a new class of attacks that exploit this trust boundary without violating explicit permissions. We term this new attack vector implicit toxicity, where malicious behaviors occur entirely within the allowed privilege scope. We propose LeechHijack, a Latent Embedded Exploit for Computation Hijacking, in which an adversarial MCP tool covertly expropriates the agent's computational resources for unauthorized workloads. LeechHijack operates through a two-stage mechanism: an implantation stage that embeds a benign-looking backdoor in a tool, and an exploitation stage where the backdoor activates upon predefined triggers to establish a command-and-control channel. Through this channel, the attacker injects additional tasks that the agent executes as if they were part of its normal workflow, effectively parasitizing the user's compute budget. We implement LeechHijack across four major LLM families. Experiments show that LeechHijack achieves an average success rate of 77.25%, with a resource overhead of 18.62% compared to the baseline. This study highlights the urgent need for computational provenance and resource attestation mechanisms to safeguard the emerging MCP ecosystem.



## **22. COGNITION: From Evaluation to Defense against Multimodal LLM CAPTCHA Solvers**

cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.02318v2) [paper-pdf](https://arxiv.org/pdf/2512.02318v2)

**Authors**: Junyu Wang, Changjia Zhu, Yuanbo Zhou, Lingyao Li, Xu He, Junjie Xiong

**Abstract**: This paper studies how multimodal large language models (MLLMs) undermine the security guarantees of visual CAPTCHA. We identify the attack surface where an adversary can cheaply automate CAPTCHA solving using off-the-shelf models. We evaluate 7 leading commercial and open-source MLLMs across 18 real-world CAPTCHA task types, measuring single-shot accuracy, success under limited retries, end-to-end latency, and per-solve cost. We further analyze the impact of task-specific prompt engineering and few-shot demonstrations on solver effectiveness. We reveal that MLLMs can reliably solve recognition-oriented and low-interaction CAPTCHA tasks at human-like cost and latency, whereas tasks requiring fine-grained localization, multi-step spatial reasoning, or cross-frame consistency remain significantly harder for current models. By examining the reasoning traces of such MLLMs, we investigate the underlying mechanisms of why models succeed/fail on specific CAPTCHA puzzles and use these insights to derive defense-oriented guidelines for selecting and strengthening CAPTCHA tasks. We conclude by discussing implications for platform operators deploying CAPTCHA as part of their abuse-mitigation pipeline.Code Availability (https://anonymous.4open.science/r/Captcha-465E/).



## **23. Adversarial Robustness of Traffic Classification under Resource Constraints: Input Structure Matters**

cs.NI

Accepted at the 2025 IEEE International Symposium on Networks, Computers and Communications (ISNCC)

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.02276v1) [paper-pdf](https://arxiv.org/pdf/2512.02276v1)

**Authors**: Adel Chehade, Edoardo Ragusa, Paolo Gastaldo, Rodolfo Zunino

**Abstract**: Traffic classification (TC) plays a critical role in cybersecurity, particularly in IoT and embedded contexts, where inspection must often occur locally under tight hardware constraints. We use hardware-aware neural architecture search (HW-NAS) to derive lightweight TC models that are accurate, efficient, and deployable on edge platforms. Two input formats are considered: a flattened byte sequence and a 2D packet-wise time series; we examine how input structure affects adversarial vulnerability when using resource-constrained models. Robustness is assessed against white-box attacks, specifically Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD). On USTC-TFC2016, both HW-NAS models achieve over 99% clean-data accuracy while remaining within 65k parameters and 2M FLOPs. Yet under perturbations of strength 0.1, their robustness diverges: the flat model retains over 85% accuracy, while the time-series variant drops below 35%. Adversarial fine-tuning delivers robust gains, with flat-input accuracy exceeding 96% and the time-series variant recovering over 60 percentage points in robustness, all without compromising efficiency. The results underscore how input structure influences adversarial vulnerability, and show that even compact, resource-efficient models can attain strong robustness, supporting their practical deployment in secure edge-based TC.



## **24. TradeTrap: Are LLM-based Trading Agents Truly Reliable and Faithful?**

cs.AI

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.02261v1) [paper-pdf](https://arxiv.org/pdf/2512.02261v1)

**Authors**: Lewen Yan, Jilin Mei, Tianyi Zhou, Lige Huang, Jie Zhang, Dongrui Liu, Jing Shao

**Abstract**: LLM-based trading agents are increasingly deployed in real-world financial markets to perform autonomous analysis and execution. However, their reliability and robustness under adversarial or faulty conditions remain largely unexamined, despite operating in high-risk, irreversible financial environments. We propose TradeTrap, a unified evaluation framework for systematically stress-testing both adaptive and procedural autonomous trading agents. TradeTrap targets four core components of autonomous trading agents: market intelligence, strategy formulation, portfolio and ledger handling, and trade execution, and evaluates their robustness under controlled system-level perturbations. All evaluations are conducted in a closed-loop historical backtesting setting on real US equity market data with identical initial conditions, enabling fair and reproducible comparisons across agents and attacks. Extensive experiments show that small perturbations at a single component can propagate through the agent decision loop and induce extreme concentration, runaway exposure, and large portfolio drawdowns across both agent types, demonstrating that current autonomous trading agents can be systematically misled at the system level. Our code is available at https://github.com/Yanlewen/TradeTrap.



## **25. Physical ID-Transfer Attacks against Multi-Object Tracking via Adversarial Trajectory**

cs.CV

Accepted to Annual Computer Security Applications Conference (ACSAC) 2024

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.01934v1) [paper-pdf](https://arxiv.org/pdf/2512.01934v1)

**Authors**: Chenyi Wang, Yanmao Man, Raymond Muller, Ming Li, Z. Berkay Celik, Ryan Gerdes, Jonathan Petit

**Abstract**: Multi-Object Tracking (MOT) is a critical task in computer vision, with applications ranging from surveillance systems to autonomous driving. However, threats to MOT algorithms have yet been widely studied. In particular, incorrect association between the tracked objects and their assigned IDs can lead to severe consequences, such as wrong trajectory predictions. Previous attacks against MOT either focused on hijacking the trackers of individual objects, or manipulating the tracker IDs in MOT by attacking the integrated object detection (OD) module in the digital domain, which are model-specific, non-robust, and only able to affect specific samples in offline datasets. In this paper, we present AdvTraj, the first online and physical ID-manipulation attack against tracking-by-detection MOT, in which an attacker uses adversarial trajectories to transfer its ID to a targeted object to confuse the tracking system, without attacking OD. Our simulation results in CARLA show that AdvTraj can fool ID assignments with 100% success rate in various scenarios for white-box attacks against SORT, which also have high attack transferability (up to 93% attack success rate) against state-of-the-art (SOTA) MOT algorithms due to their common design principles. We characterize the patterns of trajectories generated by AdvTraj and propose two universal adversarial maneuvers that can be performed by a human walker/driver in daily scenarios. Our work reveals under-explored weaknesses in the object association phase of SOTA MOT systems, and provides insights into enhancing the robustness of such systems.



## **26. Many-to-One Adversarial Consensus: Exposing Multi-Agent Collusion Risks in AI-Based Healthcare**

cs.CR

7 pages Conference level paper

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.03097v1) [paper-pdf](https://arxiv.org/pdf/2512.03097v1)

**Authors**: Adeela Bashir, The Anh han, Zia Ush Shamszaman

**Abstract**: The integration of large language models (LLMs) into healthcare IoT systems promises faster decisions and improved medical support. LLMs are also deployed as multi-agent teams to assist AI doctors by debating, voting, or advising on decisions. However, when multiple assistant agents interact, coordinated adversaries can collude to create false consensus, pushing an AI doctor toward harmful prescriptions. We develop an experimental framework with scripted and unscripted doctor agents, adversarial assistants, and a verifier agent that checks decisions against clinical guidelines. Using 50 representative clinical questions, we find that collusion drives the Attack Success Rate (ASR) and Harmful Recommendation Rates (HRR) up to 100% in unprotected systems. In contrast, the verifier agent restores 100% accuracy by blocking adversarial consensus. This work provides the first systematic evidence of collusion risk in AI healthcare and demonstrates a practical, lightweight defence that ensures guideline fidelity.



## **27. Securing Large Language Models (LLMs) from Prompt Injection Attacks**

cs.CR

10 pages, 1 figure, 1 table

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.01326v1) [paper-pdf](https://arxiv.org/pdf/2512.01326v1)

**Authors**: Omar Farooq Khan Suri, John McCrae

**Abstract**: Large Language Models (LLMs) are increasingly being deployed in real-world applications, but their flexibility exposes them to prompt injection attacks. These attacks leverage the model's instruction-following ability to make it perform malicious tasks. Recent work has proposed JATMO, a task-specific fine-tuning approach that trains non-instruction-tuned base models to perform a single function, thereby reducing susceptibility to adversarial instructions. In this study, we evaluate the robustness of JATMO against HOUYI, a genetic attack framework that systematically mutates and optimizes adversarial prompts. We adapt HOUYI by introducing custom fitness scoring, modified mutation logic, and a new harness for local model testing, enabling a more accurate assessment of defense effectiveness. We fine-tuned LLaMA 2-7B, Qwen1.5-4B, and Qwen1.5-0.5B models under the JATMO methodology and compared them with a fine-tuned GPT-3.5-Turbo baseline. Results show that while JATMO reduces attack success rates relative to instruction-tuned models, it does not fully prevent injections; adversaries exploiting multilingual cues or code-related disruptors still bypass defenses. We also observe a trade-off between generation quality and injection vulnerability, suggesting that better task performance often correlates with increased susceptibility. Our results highlight both the promise and limitations of fine-tuning-based defenses and point toward the need for layered, adversarially informed mitigation strategies.



## **28. DPAC: Distribution-Preserving Adversarial Control for Diffusion Sampling**

cs.CV

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.01153v1) [paper-pdf](https://arxiv.org/pdf/2512.01153v1)

**Authors**: Han-Jin Lee, Han-Ju Lee, Jin-Seong Kim, Seok-Hwan Choi

**Abstract**: Adversarially guided diffusion sampling often achieves the target class, but sample quality degrades as deviations between the adversarially controlled and nominal trajectories accumulate. We formalize this degradation as a path-space Kullback-Leibler divergence(path-KL) between controlled and nominal (uncontrolled) diffusion processes, thereby showing via Girsanov's theorem that it exactly equals the control energy. Building on this stochastic optimal control (SOC) view, we theoretically establish that minimizing this path-KL simultaneously tightens upper bounds on both the 2-Wasserstein distance and Fréchet Inception Distance (FID), revealing a principled connection between adversarial control energy and perceptual fidelity. From a variational perspective, we derive a first-order optimality condition for the control: among all directions that yield the same classification gain, the component tangent to iso-(log-)density surfaces (i.e., orthogonal to the score) minimizes path-KL, whereas the normal component directly increases distributional drift. This leads to DPAC (Distribution-Preserving Adversarial Control), a diffusion guidance rule that projects adversarial gradients onto the tangent space defined by the generative score geometry. We further show that in discrete solvers, the tangent projection cancels the O(Δt) leading error term in the Wasserstein distance, achieving an O(Δt^2) quality gap; moreover, it remains second-order robust to score or metric approximation. Empirical studies on ImageNet-100 validate the theoretical predictions, confirming that DPAC achieves lower FID and estimated path-KL at matched attack success rates.



## **29. The Outline of Deception: Physical Adversarial Attacks on Traffic Signs Using Edge Patches**

cs.CV

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.00765v2) [paper-pdf](https://arxiv.org/pdf/2512.00765v2)

**Authors**: Haojie Ji, Te Hu, Haowen Li, Long Jin, Chongshi Xin, Yuchi Yao, Jiarui Xiao

**Abstract**: Intelligent driving systems are vulnerable to physical adversarial attacks on traffic signs. These attacks can cause misclassification, leading to erroneous driving decisions that compromise road safety. Moreover, within V2X networks, such misinterpretations can propagate, inducing cascading failures that disrupt overall traffic flow and system stability. However, a key limitation of current physical attacks is their lack of stealth. Most methods apply perturbations to central regions of the sign, resulting in visually salient patterns that are easily detectable by human observers, thereby limiting their real-world practicality. This study proposes TESP-Attack, a novel stealth-aware adversarial patch method for traffic sign classification. Based on the observation that human visual attention primarily focuses on the central regions of traffic signs, we employ instance segmentation to generate edge-aligned masks that conform to the shape characteristics of the signs. A U-Net generator is utilized to craft adversarial patches, which are then optimized through color and texture constraints along with frequency domain analysis to achieve seamless integration with the background environment, resulting in highly effective visual concealment. The proposed method demonstrates outstanding attack success rates across traffic sign classification models with varied architectures, achieving over 90% under limited query budgets. It also exhibits strong cross-model transferability and maintains robust real-world performance that remains stable under varying angles and distances.



## **30. Adversarial Confusion Attack: Disrupting Multimodal Large Language Models**

cs.CL

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2511.20494v3) [paper-pdf](https://arxiv.org/pdf/2511.20494v3)

**Authors**: Jakub Hoscilowicz, Artur Janicki

**Abstract**: We introduce the Adversarial Confusion Attack, a new class of threats against multimodal large language models (MLLMs). Unlike jailbreaks or targeted misclassification, the goal is to induce systematic disruption that makes the model generate incoherent or confidently incorrect outputs. Practical applications include embedding such adversarial images into websites to prevent MLLM-powered AI Agents from operating reliably. The proposed attack maximizes next-token entropy using a small ensemble of open-source MLLMs. In the white-box setting, we show that a single adversarial image can disrupt all models in the ensemble, both in the full-image and Adversarial CAPTCHA settings. Despite relying on a basic adversarial technique (PGD), the attack generates perturbations that transfer to both unseen open-source (e.g., Qwen3-VL) and proprietary (e.g., GPT-5.1) models.



## **31. Spilling the Beans: Teaching LLMs to Self-Report Their Hidden Objectives**

cs.AI

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2511.06626v4) [paper-pdf](https://arxiv.org/pdf/2511.06626v4)

**Authors**: Chloe Li, Mary Phuong, Daniel Tan

**Abstract**: As AI systems become more capable of complex agentic tasks, they also become more capable of pursuing undesirable objectives and causing harm. Previous work has attempted to catch these unsafe instances by interrogating models directly about their objectives and behaviors. However, the main weakness of trusting interrogations is that models can lie. We propose self-report fine-tuning (SRFT), a simple supervised fine-tuning technique that trains models to occasionally make factual mistakes, then admit them when asked. We show that the admission of factual errors in simple question-answering settings generalizes out-of-distribution (OOD) to the admission of hidden misaligned objectives in adversarial agentic settings. We evaluate SRFT in OOD stealth tasks, where models are instructed to complete a hidden misaligned objective alongside a user-specified objective without being caught by monitoring. After SRFT, models are more likely to confess the details of their hidden objectives when interrogated, even under strong pressure not to disclose them. Interrogation on SRFT models can detect hidden objectives with near-ceiling performance (F1 score = 0.98), while the baseline model lies when interrogated under the same conditions (F1 score = 0). Interrogation on SRFT models can further elicit the content of the hidden objective, recovering 28-100% details, compared to 0% details recovered in the baseline model and by prefilled assistant turn attacks. This provides a promising technique for promoting honesty propensity and incriminating misaligned AIs.



## **32. Reasoning Up the Instruction Ladder for Controllable Language Models**

cs.CL

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2511.04694v3) [paper-pdf](https://arxiv.org/pdf/2511.04694v3)

**Authors**: Zishuo Zheng, Vidhisha Balachandran, Chan Young Park, Faeze Brahman, Sachin Kumar

**Abstract**: As large language model (LLM) based systems take on high-stakes roles in real-world decision-making, they must reconcile competing instructions from multiple sources (e.g., model developers, users, and tools) within a single prompt context. Thus, enforcing an instruction hierarchy (IH) in LLMs, where higher-level directives override lower-priority requests, is critical for the reliability and controllability of LLMs. In this work, we reframe instruction hierarchy resolution as a reasoning task. Specifically, the model must first "think" about the relationship between a given user prompt and higher-priority (system) instructions before generating a response. To enable this capability via training, we construct VerIH, an instruction hierarchy dataset of constraint-following tasks with verifiable answers. This dataset comprises ~7K aligned and conflicting system-user instructions. We show that lightweight reinforcement learning with VerIH effectively transfers general reasoning capabilities of models to instruction prioritization. Our finetuned models achieve consistent improvements on instruction following and instruction hierarchy benchmarks, achieving roughly a 20% improvement on the IHEval conflict setup. This reasoning ability also generalizes to safety-critical settings beyond the training distribution. By treating safety issues as resolving conflicts between adversarial user inputs and predefined higher-priority policies, our trained model enhances robustness against jailbreak and prompt injection attacks, providing up to a 20% reduction in attack success rate (ASR). These results demonstrate that reasoning over instruction hierarchies provides a practical path to reliable LLMs, where updates to system prompts yield controllable and robust changes in model behavior.



## **33. Bilevel Models for Adversarial Learning and A Case Study**

cs.LG

This paper has been accepted by Mathematics

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2510.25121v2) [paper-pdf](https://arxiv.org/pdf/2510.25121v2)

**Authors**: Yutong Zheng, Qingna Li

**Abstract**: Adversarial learning has been attracting more and more attention thanks to the fast development of machine learning and artificial intelligence. However, due to the complicated structure of most machine learning models, the mechanism of adversarial attacks is not well interpreted. How to measure the effect of attacks is still not quite clear.   In this paper, we investigate the adversarial learning from the perturbation analysis point of view.   We characterize the robustness of learning models through the calmness of the solution mapping.   In the case of convex clustering models, we identify the conditions under which the clustering results remain the same under perturbations.   When the noise level is large, it leads to an attack.   Therefore, we propose two bilevel models for adversarial learning where the effect of adversarial learning is measured   by some deviation function.   Specifically, we systematically study the so-called $δ$-measure and show that under certain conditions, it can be used as a deviation function in adversarial learning for convex clustering models.   Finally, we conduct numerical tests to verify the above theoretical results as well as the efficiency of the two proposed bilevel models.



## **34. Get RICH or Die Scaling: Profitably Trading Inference Compute for Robustness**

cs.LG

21 pages

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2510.06790v2) [paper-pdf](https://arxiv.org/pdf/2510.06790v2)

**Authors**: Tavish McDonald, Bo Lei, Stanislav Fort, Bhavya Kailkhura, Brian Bartoldson

**Abstract**: Models are susceptible to adversarially out-of-distribution (OOD) data despite large training-compute investments into their robustification. Zaremba et al. (2025) make progress on this problem at test time, showing LLM reasoning improves satisfaction of model specifications designed to thwart attacks, resulting in a correlation between reasoning effort and robustness to jailbreaks. However, this benefit of test compute fades when attackers are given access to gradients or multimodal inputs. We address this gap, clarifying that inference-compute offers benefits even in such cases. Our approach argues that compositional generalization, through which OOD data is understandable via its in-distribution (ID) components, enables adherence to defensive specifications on adversarially OOD inputs. Namely, we posit the Robustness from Inference Compute Hypothesis (RICH): inference-compute defenses profit as the model's training data better reflects the attacked data's components. We empirically support this hypothesis across vision language model and attack types, finding robustness gains from test-time compute if specification following on OOD data is unlocked by compositional generalization. For example, InternVL 3.5 gpt-oss 20B gains little robustness when its test compute is scaled, but such scaling adds significant robustness if we first robustify its vision encoder. This correlation of inference-compute's robustness benefit with base model robustness is the rich-get-richer dynamic of the RICH: attacked data components are more ID for robustified models, aiding compositional generalization to OOD data. Thus, we advise layering train-time and test-time defenses to obtain their synergistic benefit.



## **35. ZQBA: Zero Query Black-box Adversarial Attack**

cs.CV

Accepted in ICAART 2026 Conference

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2510.00769v2) [paper-pdf](https://arxiv.org/pdf/2510.00769v2)

**Authors**: Joana C. Costa, Tiago Roxo, Hugo Proença, Pedro R. M. Inácio

**Abstract**: Current black-box adversarial attacks either require multiple queries or diffusion models to produce adversarial samples that can impair the target model performance. However, these methods require training a surrogate loss or diffusion models to produce adversarial samples, which limits their applicability in real-world settings. Thus, we propose a Zero Query Black-box Adversarial (ZQBA) attack that exploits the representations of Deep Neural Networks (DNNs) to fool other networks. Instead of requiring thousands of queries to produce deceiving adversarial samples, we use the feature maps obtained from a DNN and add them to clean images to impair the classification of a target model. The results suggest that ZQBA can transfer the adversarial samples to different models and across various datasets, namely CIFAR and Tiny ImageNet. The experiments also show that ZQBA is more effective than state-of-the-art black-box attacks with a single query, while maintaining the imperceptibility of perturbations, evaluated both quantitatively (SSIM) and qualitatively, emphasizing the vulnerabilities of employing DNNs in real-world contexts. All the source code is available at https://github.com/Joana-Cabral/ZQBA.



## **36. Less is More: Towards Simple Graph Contrastive Learning**

cs.LG

Submitted to ICLR 2026

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2509.25742v2) [paper-pdf](https://arxiv.org/pdf/2509.25742v2)

**Authors**: Yanan Zhao, Feng Ji, Jingyang Dai, Jiaze Ma, Wee Peng Tay

**Abstract**: Graph Contrastive Learning (GCL) has shown strong promise for unsupervised graph representation learning, yet its effectiveness on heterophilic graphs, where connected nodes often belong to different classes, remains limited. Most existing methods rely on complex augmentation schemes, intricate encoders, or negative sampling, which raises the question of whether such complexity is truly necessary in this challenging setting. In this work, we revisit the foundations of supervised and unsupervised learning on graphs and uncover a simple yet effective principle for GCL: mitigating node feature noise by aggregating it with structural features derived from the graph topology. This observation suggests that the original node features and the graph structure naturally provide two complementary views for contrastive learning. Building on this insight, we propose an embarrassingly simple GCL model that uses a GCN encoder to capture structural features and an MLP encoder to isolate node feature noise. Our design requires neither data augmentation nor negative sampling, yet achieves state-of-the-art results on heterophilic benchmarks with minimal computational and memory overhead, while also offering advantages in homophilic graphs in terms of complexity, scalability, and robustness. We provide theoretical justification for our approach and validate its effectiveness through extensive experiments, including robustness evaluations against both black-box and white-box adversarial attacks.



## **37. Accuracy-Robustness Trade Off via Spiking Neural Network Gradient Sparsity Trail**

cs.NE

Work under peer-review

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2509.23762v3) [paper-pdf](https://arxiv.org/pdf/2509.23762v3)

**Authors**: Luu Trong Nhan, Luu Trung Duong, Pham Ngoc Nam, Truong Cong Thang

**Abstract**: Spiking Neural Networks (SNNs) have attracted growing interest in both computational neuroscience and artificial intelligence, primarily due to their inherent energy efficiency and compact memory footprint. However, achieving adversarial robustness in SNNs, (particularly for vision-related tasks) remains a nascent and underexplored challenge. Recent studies have proposed leveraging sparse gradients as a form of regularization to enhance robustness against adversarial perturbations. In this work, we present a surprising finding: under specific architectural configurations, SNNs exhibit natural gradient sparsity and can achieve state-of-the-art adversarial defense performance without the need for any explicit regularization. Further analysis reveals a trade-off between robustness and generalization: while sparse gradients contribute to improved adversarial resilience, they can impair the model's ability to generalize; conversely, denser gradients support better generalization but increase vulnerability to attacks. Our findings offer new insights into the dual role of gradient sparsity in SNN training.



## **38. Observation-Free Attacks on Online Learning to Rank**

cs.LG

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2509.22855v4) [paper-pdf](https://arxiv.org/pdf/2509.22855v4)

**Authors**: Sameep Chattopadhyay, Nikhil Karamchandani, Sharayu Moharir

**Abstract**: Online learning to rank (OLTR) plays a critical role in information retrieval and machine learning systems, with a wide range of applications in search engines and content recommenders. However, despite their extensive adoption, the susceptibility of OLTR algorithms to coordinated adversarial attacks remains poorly understood. In this work, we present a novel framework for attacking some of the widely used OLTR algorithms. Our framework is designed to promote a set of target items so that they appear in the list of top-K recommendations for T - o(T) rounds, while simultaneously inducing linear regret in the learning algorithm. We propose two novel attack strategies: CascadeOFA for CascadeUCB1 and PBMOFA for PBM-UCB . We provide theoretical guarantees showing that both strategies require only O(log T) manipulations to succeed. Additionally, we supplement our theoretical analysis with empirical results on real-world data.



## **39. When Ads Become Profiles: Uncovering the Invisible Risk of Web Advertising at Scale with LLMs**

cs.HC

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2509.18874v2) [paper-pdf](https://arxiv.org/pdf/2509.18874v2)

**Authors**: Baiyu Chen, Benjamin Tag, Hao Xue, Daniel Angus, Flora Salim

**Abstract**: Regulatory limits on explicit targeting have not eliminated algorithmic profiling on the Web, as optimisation systems still adapt ad delivery to users' private attributes. The widespread availability of powerful zero-shot multimodal Large Language Models (LLMs) has dramatically lowered the barrier for exploiting these latent signals for adversarial inference. We investigate this emerging societal risk, specifically how adversaries can now exploit these signals to reverse-engineer private attributes from ad exposure alone. We introduce a novel pipeline that leverages LLMs as adversarial inference engines to perform natural language profiling. Applying this method to a longitudinal dataset comprising over 435,000 ad impressions collected from 891 users, we conducted a large-scale study to assess the feasibility and precision of inferring private attributes from passive online ad observations. Our results demonstrate that off-the-shelf LLMs can accurately reconstruct complex user private attributes, including party preference, employment status, and education level, consistently outperforming strong census-based priors and matching or exceeding human social perception, while operating at only a fraction of the cost (223$\times$ lower) and time (52$\times$ faster) required by humans. Critically, actionable profiling is feasible even within short observation windows, indicating that prolonged tracking is not a prerequisite for a successful attack. These findings provide the first empirical evidence that ad streams serve as a high-fidelity digital footprint, enabling off-platform profiling that inherently bypasses current platform safeguards, highlighting a systemic vulnerability in the ad ecosystem and the urgent need for responsible web AI governance in the generative AI era. The code is available at https://github.com/Breezelled/when-ads-become-profiles.



## **40. Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking**

cs.CR

Accepted by AAAI 2026

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2509.05835v2) [paper-pdf](https://arxiv.org/pdf/2509.05835v2)

**Authors**: Lingfeng Yao, Chenpei Huang, Shengyao Wang, Junpei Xue, Hanqing Guo, Jiang Liu, Phone Lin, Tomoaki Ohtsuki, Miao Pan

**Abstract**: As generative audio models are rapidly evolving, AI-generated audios increasingly raise concerns about copyright infringement and misinformation spread. Audio watermarking, as a proactive defense, can embed secret messages into audio for copyright protection and source verification. However, current neural audio watermarking methods focus primarily on the imperceptibility and robustness of watermarking, while ignoring its vulnerability to security attacks. In this paper, we develop a simple yet powerful attack: the overwriting attack that overwrites the legitimate audio watermark with a forged one and makes the original legitimate watermark undetectable. Based on the audio watermarking information that the adversary has, we propose three categories of overwriting attacks, i.e., white-box, gray-box, and black-box attacks. We also thoroughly evaluate the proposed attacks on state-of-the-art neural audio watermarking methods. Experimental results demonstrate that the proposed overwriting attacks can effectively compromise existing watermarking schemes across various settings and achieve a nearly 100% attack success rate. The practicality and effectiveness of the proposed overwriting attacks expose security flaws in existing neural audio watermarking systems, underscoring the need to enhance security in future audio watermarking designs.



## **41. Analyzing PDFs like Binaries: Adversarially Robust PDF Malware Analysis via Intermediate Representation and Language Model**

cs.CR

Accepted by ACM CCS 2025

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2506.17162v2) [paper-pdf](https://arxiv.org/pdf/2506.17162v2)

**Authors**: Side Liu, Jiang Ming, Guodong Zhou, Xinyi Liu, Jianming Fu, Guojun Peng

**Abstract**: Malicious PDF files have emerged as a persistent threat and become a popular attack vector in web-based attacks. While machine learning-based PDF malware classifiers have shown promise, these classifiers are often susceptible to adversarial attacks, undermining their reliability. To address this issue, recent studies have aimed to enhance the robustness of PDF classifiers. Despite these efforts, the feature engineering underlying these studies remains outdated. Consequently, even with the application of cutting-edge machine learning techniques, these approaches fail to fundamentally resolve the issue of feature instability.   To tackle this, we propose a novel approach for PDF feature extraction and PDF malware detection. We introduce the PDFObj IR (PDF Object Intermediate Representation), an assembly-like language framework for PDF objects, from which we extract semantic features using a pretrained language model. Additionally, we construct an Object Reference Graph to capture structural features, drawing inspiration from program analysis. This dual approach enables us to analyze and detect PDF malware based on both semantic and structural features. Experimental results demonstrate that our proposed classifier achieves strong adversarial robustness while maintaining an exceptionally low false positive rate of only 0.07% on baseline dataset compared to state-of-the-art PDF malware classifiers.



## **42. SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models**

cs.CR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2506.00821v2) [paper-pdf](https://arxiv.org/pdf/2506.00821v2)

**Authors**: Huixin Zhan, Clovis Barbour, Jason H. Moore

**Abstract**: Genomic Foundation Models (GFMs), such as Evolutionary Scale Modeling (ESM), have demonstrated significant success in variant effect prediction. However, their adversarial robustness remains largely unexplored. To address this gap, we propose SafeGenes: a framework for Secure analysis of genomic foundation models, leveraging adversarial attacks to evaluate robustness against both engineered near-identical adversarial Genes and embedding-space manipulations. In this study, we assess the adversarial vulnerabilities of GFMs using two approaches: the Fast Gradient Sign Method (FGSM) and a soft prompt attack. FGSM introduces minimal perturbations to input sequences, while the soft prompt attack optimizes continuous embeddings to manipulate model predictions without modifying the input tokens. By combining these techniques, SafeGenes provides a comprehensive assessment of GFM susceptibility to adversarial manipulation. Targeted soft prompt attacks induced severe degradation in MLM-based shallow architectures such as ProteinBERT, while still producing substantial failure modes even in high-capacity foundation models such as ESM1b and ESM1v. These findings expose critical vulnerabilities in current foundation models, opening new research directions toward improving their security and robustness in high-stakes genomic applications such as variant effect prediction.



## **43. Bant: Byzantine Antidote via Trial Function and Trust Scores**

cs.LG

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2505.07614v5) [paper-pdf](https://arxiv.org/pdf/2505.07614v5)

**Authors**: Gleb Molodtsov, Daniil Medyakov, Sergey Skorik, Nikolas Khachaturov, Shahane Tigranyan, Vladimir Aletov, Aram Avetisyan, Martin Takáč, Aleksandr Beznosikov

**Abstract**: Recent advancements in machine learning have improved performance while also increasing computational demands. While federated and distributed setups address these issues, their structures remain vulnerable to malicious influences. In this paper, we address a specific threat: Byzantine attacks, wherein compromised clients inject adversarial updates to derail global convergence. We combine the concept of trust scores with trial function methodology to dynamically filter outliers. Our methods address the critical limitations of previous approaches, allowing operation even when Byzantine nodes are in the majority. Moreover, our algorithms adapt to widely used scaled methods such as Adam and RMSProp, as well as practical scenarios, including local training and partial participation. We validate the robustness of our methods by conducting extensive experiments on both public datasets and private ECG data collected from medical institutions. Furthermore, we provide a broad theoretical analysis of our algorithms and their extensions to the aforementioned practical setups. The convergence guaranties of our methods are comparable to those of classical algorithms developed without Byzantine interference.



## **44. Bones of Contention: Exploring Query-Efficient Attacks against Skeleton Recognition Systems**

cs.CR

Accepted to IEEE Transactions on Information Forensics and Security (TIFS)

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2501.16843v2) [paper-pdf](https://arxiv.org/pdf/2501.16843v2)

**Authors**: Yuxin Cao, Kai Ye, Derui Wang, Minhui Xue, Hao Ge, Chenxiong Qian, Jin Song Dong

**Abstract**: Skeleton action recognition models have secured more attention than video-based ones in various applications due to privacy preservation and lower storage requirements. Skeleton data are typically transmitted to cloud servers for action recognition, with results returned to clients via Apps/APIs. However, the vulnerability of skeletal models against adversarial perturbations gradually reveals the unreliability of these systems. Existing black-box attacks all operate in a decision-based manner, resulting in numerous queries that hinder efficiency and feasibility in real-world applications. Moreover, all attacks off the shelf focus on only restricted perturbations, while ignoring model weaknesses when encountered with non-semantic perturbations. In this paper, we propose two query-effIcient Skeletal Adversarial AttaCks, ISAAC-K and ISAAC-N. As a black-box attack, ISAAC-K utilizes Grad-CAM in a surrogate model to extract key joints where minor sparse perturbations are then added to fool the classifier. To guarantee natural adversarial motions, we introduce constraints of both bone length and temporal consistency. ISAAC-K finds stronger adversarial examples on the $\ell_\infty$ norm, which can encompass those on other norms. Exhaustive experiments substantiate that ISAAC-K can uplift the attack efficiency of the perturbations under 10 skeletal models. Additionally, as a byproduct, ISAAC-N fools the classifier by replacing skeletons unrelated to the action. We surprisingly find that skeletal models are vulnerable to large perturbations where the part-wise non-semantic joints are just replaced, leading to a query-free no-box attack without any prior knowledge. Based on that, four adaptive defenses are eventually proposed to improve the robustness of skeleton recognition models.



## **45. Improving Graph Neural Network Training, Defense, and Hypergraph Partitioning via Adversarial Robustness Evaluation**

cs.LG

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2412.14738v7) [paper-pdf](https://arxiv.org/pdf/2412.14738v7)

**Authors**: Yongyu Wang

**Abstract**: Graph Neural Networks (GNNs) are a highly effective neural network architecture for processing graph-structured data. Unlike traditional neural networks that rely solely on the features of the data as input, GNNs leverage both the graph structure, which represents the relationships between data points, and the feature matrix of the data to optimize their feature representation. This unique capability enables GNNs to achieve superior performance across various tasks. However, it also makes GNNs more susceptible to noise from both the graph structure and data features, which can significantly increase the training difficulty and degrade their performance. To address this issue, this paper proposes a novel method for selecting noise-sensitive training samples from the original training set to construct a smaller yet more effective training set for model training. These samples are used to help improve the model's ability to correctly process data in noisy environments. We have evaluated our approach on three of the most classical GNN models GCN, GAT, and GraphSAGE as well as three widely used benchmark datasets: Cora, Citeseer, and PubMed. Our experiments demonstrate that the proposed method can substantially boost the training of Graph Neural Networks compared to using randomly sampled training sets of the same size from the original training set and the larger original full training set. We further proposed a robust-node based hypergraph partitioning method, an adversarial robustness based graph pruning method for GNN defenses and a related spectral edge attack method.



## **46. Edge-Only Universal Adversarial Attacks in Distributed Learning**

cs.CR

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2411.10500v2) [paper-pdf](https://arxiv.org/pdf/2411.10500v2)

**Authors**: Giulio Rossolini, Tommaso Baldi, Alessandro Biondi, Giorgio Buttazzo

**Abstract**: Distributed learning frameworks, which partition neural network models across multiple computing nodes, enhance efficiency in collaborative edge-cloud systems, but may also introduce new vulnerabilities to evasion attacks, often in the form of adversarial perturbations. In this work, we present a new threat model that explores the feasibility of generating universal adversarial perturbations (UAPs) when the attacker has access only to the edge portion of the model, consisting of its initial network layers. Unlike traditional attacks that require full model knowledge, our approach shows that adversaries can induce effective mispredictions in the unknown cloud component by manipulating key feature representations at the edge. Following the proposed threat model, we introduce both edge-only untargeted and targeted formulations of UAPs designed to control intermediate features before the split point. Our results on ImageNet demonstrate strong attack transferability to the unknown cloud part, and we compare the proposed method with classical white-box and black-box techniques, highlighting its effectiveness. Additionally, we analyze the capability of an attacker to achieve targeted adversarial effects with edge-only knowledge, revealing intriguing behaviors across multiple networks. By introducing the first adversarial attacks with edge-only knowledge in split inference, this work underscores the importance of addressing partial model access in adversarial robustness, encouraging further research in this area.



## **47. Multimodal Adversarial Defense for Vision-Language Models by Leveraging One-To-Many Relationships**

cs.CV

WACV 2026 Accepted. Code available at https://github.com/CyberAgentAI/multimodal-adversarial-training

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2405.18770v5) [paper-pdf](https://arxiv.org/pdf/2405.18770v5)

**Authors**: Futa Waseda, Antonio Tejero-de-Pablos, Isao Echizen

**Abstract**: Pre-trained vision-language (VL) models are highly vulnerable to adversarial attacks. However, existing defense methods primarily focus on image classification, overlooking two key aspects of VL tasks: multimodal attacks, where both image and text can be perturbed, and the one-to-many relationship of images and texts, where a single image can correspond to multiple textual descriptions and vice versa (1:N and N:1). This work is the first to explore defense strategies against multimodal attacks in VL tasks, whereas prior VL defense methods focus on vision robustness. We propose multimodal adversarial training (MAT), which incorporates adversarial perturbations in both image and text modalities during training, significantly outperforming existing unimodal defenses. Furthermore, we discover that MAT is limited by deterministic one-to-one (1:1) image-text pairs in VL training data. To address this, we conduct a comprehensive study on leveraging one-to-many relationships to enhance robustness, investigating diverse augmentation techniques. Our analysis shows that, for a more effective defense, augmented image-text pairs should be well-aligned, diverse, yet avoid distribution shift -- conditions overlooked by prior research. This work pioneers defense strategies against multimodal attacks, providing insights for building robust VLMs from both optimization and data perspectives. Our code is publicly available at https://github.com/CyberAgentAI/multimodal-adversarial-training.



## **48. UltraClean: A Simple Framework to Train Robust Neural Networks against Backdoor Attacks**

cs.CR

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2312.10657v2) [paper-pdf](https://arxiv.org/pdf/2312.10657v2)

**Authors**: Bingyin Zhao, Yingjie Lao

**Abstract**: Backdoor attacks are emerging threats to deep neural networks, which typically embed malicious behaviors into a victim model by injecting poisoned samples. Adversaries can activate the injected backdoor during inference by presenting the trigger on input images. Prior defensive methods have achieved remarkable success in countering dirty-label backdoor attacks where the labels of poisoned samples are often mislabeled. However, these approaches do not work for a recent new type of backdoor -- clean-label backdoor attacks that imperceptibly modify poisoned data and hold consistent labels. More complex and powerful algorithms are demanded to defend against such stealthy attacks. In this paper, we propose UltraClean, a general framework that simplifies the identification of poisoned samples and defends against both dirty-label and clean-label backdoor attacks. Given the fact that backdoor triggers introduce adversarial noise that intensifies in feed-forward propagation, UltraClean first generates two variants of training samples using off-the-shelf denoising functions. It then measures the susceptibility of training samples leveraging the error amplification effect in DNNs, which dilates the noise difference between the original image and denoised variants. Lastly, it filters out poisoned samples based on the susceptibility to thwart the backdoor implantation. Despite its simplicity, UltraClean achieves a superior detection rate across various datasets and significantly reduces the backdoor attack success rate while maintaining a decent model accuracy on clean data, outperforming existing defensive methods by a large margin. Code is available at https://github.com/bxz9200/UltraClean.



## **49. GPS-Spoofing Attack Detection Mechanism for UAV Swarms**

cs.CR

8 pages, 3 figures

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2301.12766v3) [paper-pdf](https://arxiv.org/pdf/2301.12766v3)

**Authors**: Pavlo Mykytyn, Marcin Brzozowski, Zoya Dyka, Peter Langendoerfer

**Abstract**: Recently autonomous and semi-autonomous Unmanned Aerial Vehicle (UAV) swarms started to receive a lot of research interest and demand from various civil application fields. However, for successful mission execution, UAV swarms require Global navigation satellite system signals and in particular, Global Positioning System (GPS) signals for navigation. Unfortunately, civil GPS signals are unencrypted and unauthenticated, which facilitates the execution of GPS spoofing attacks. During these attacks, adversaries mimic the authentic GPS signal and broadcast it to the targeted UAV in order to change its course, and force it to land or crash. In this study, we propose a GPS spoofing detection mechanism capable of detecting single-transmitter and multi-transmitter GPS spoofing attacks to prevent the outcomes mentioned above. Our detection mechanism is based on comparing the distance between each two swarm members calculated from their GPS coordinates to the distance acquired from Impulse Radio Ultra-Wideband ranging between the same swarm members. If the difference in distances is larger than a chosen threshold the GPS spoofing attack is declared detected.



## **50. A review of mechanistic and data-driven models of terrorism and radicalization**

physics.soc-ph

80 pages, 17 figures

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/1903.08485v3) [paper-pdf](https://arxiv.org/pdf/1903.08485v3)

**Authors**: Yao-li Chuang, Maria R. D'Orsogna

**Abstract**: The rapid spread of radical ideologies in recent years has led to a worldwide string of terrorist attacks. Understanding how extremist tendencies germinate, develop, and drive individuals to action is important from a cultural standpoint, but also to help formulate response and prevention strategies. Demographic studies, interviews with radicalized subjects, analysis of terrorist databases, reveal that the path to radicalization occurs along progressive steps, where age, social context and peer-to-peer exchange of extremist ideas play major roles. Furthermore, the advent of social media has offered new channels of communication, facilitated recruitment, and hastened the leap from mild discontent to unbridled fanaticism. While a complete sociological understanding of the processes and circumstances that lead to full-fledged extremism is still lacking, quantitative approaches, using modeling and data analyses, can offer useful insight. We review some approaches from statistical mechanics, applied mathematics, data science, that can help describe and understand radicalization and terrorist activity. Specifically, we focus on compartment models of populations harboring extremist views, continuous time models for age-structured radical populations, radicalization as social contagion processes on lattices and social networks, adversarial evolutionary games coupling terrorists and counter-terrorism agents, and point processes to study the spatiotemporal clustering of terrorist events. We also present recent applications of machine learning methods on open-source terrorism databases. Finally, we discuss the role of institutional intervention and the stages at which de-radicalization strategies might be most effective.



