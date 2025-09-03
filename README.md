# Latest Adversarial Attack Papers
**update at 2025-09-03 10:21:43**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Targeted Physical Evasion Attacks in the Near-Infrared Domain**

cs.CR

To appear in the Proceedings of the Network and Distributed Systems  Security Symposium (NDSS) 2026

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02042v1) [paper-pdf](http://arxiv.org/pdf/2509.02042v1)

**Authors**: Pascal Zimmer, Simon Lachnit, Alexander Jan Zielinski, Ghassan Karame

**Abstract**: A number of attacks rely on infrared light sources or heat-absorbing material to imperceptibly fool systems into misinterpreting visual input in various image recognition applications. However, almost all existing approaches can only mount untargeted attacks and require heavy optimizations due to the use-case-specific constraints, such as location and shape. In this paper, we propose a novel, stealthy, and cost-effective attack to generate both targeted and untargeted adversarial infrared perturbations. By projecting perturbations from a transparent film onto the target object with an off-the-shelf infrared flashlight, our approach is the first to reliably mount laser-free targeted attacks in the infrared domain. Extensive experiments on traffic signs in the digital and physical domains show that our approach is robust and yields higher attack success rates in various attack scenarios across bright lighting conditions, distances, and angles compared to prior work. Equally important, our attack is highly cost-effective, requiring less than US\$50 and a few tens of seconds for deployment. Finally, we propose a novel segmentation-based detection that thwarts our attack with an F1-score of up to 99%.



## **2. See No Evil: Adversarial Attacks Against Linguistic-Visual Association in Referring Multi-Object Tracking Systems**

cs.CV

12 pages, 1 figure, 3 tables

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02028v1) [paper-pdf](http://arxiv.org/pdf/2509.02028v1)

**Authors**: Halima Bouzidi, Haoyu Liu, Mohammad Al Faruque

**Abstract**: Language-vision understanding has driven the development of advanced perception systems, most notably the emerging paradigm of Referring Multi-Object Tracking (RMOT). By leveraging natural-language queries, RMOT systems can selectively track objects that satisfy a given semantic description, guided through Transformer-based spatial-temporal reasoning modules. End-to-End (E2E) RMOT models further unify feature extraction, temporal memory, and spatial reasoning within a Transformer backbone, enabling long-range spatial-temporal modeling over fused textual-visual representations. Despite these advances, the reliability and robustness of RMOT remain underexplored. In this paper, we examine the security implications of RMOT systems from a design-logic perspective, identifying adversarial vulnerabilities that compromise both the linguistic-visual referring and track-object matching components. Additionally, we uncover a novel vulnerability in advanced RMOT models employing FIFO-based memory, whereby targeted and consistent attacks on their spatial-temporal reasoning introduce errors that persist within the history buffer over multiple subsequent frames. We present VEIL, a novel adversarial framework designed to disrupt the unified referring-matching mechanisms of RMOT models. We show that carefully crafted digital and physical perturbations can corrupt the tracking logic reliability, inducing track ID switches and terminations. We conduct comprehensive evaluations using the Refer-KITTI dataset to validate the effectiveness of VEIL and demonstrate the urgent need for security-aware RMOT designs for critical large-scale applications.



## **3. Adversarial Attacks and Defenses in Multivariate Time-Series Forecasting for Smart and Connected Infrastructures**

cs.LG

18 pages, 34 figures

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2408.14875v2) [paper-pdf](http://arxiv.org/pdf/2408.14875v2)

**Authors**: Pooja Krishan, Rohan Mohapatra, Sanchari Das, Saptarshi Sengupta

**Abstract**: The emergence of deep learning models has revolutionized various industries over the last decade, leading to a surge in connected devices and infrastructures. However, these models can be tricked into making incorrect predictions with high confidence, leading to disastrous failures and security concerns. To this end, we explore the impact of adversarial attacks on multivariate time-series forecasting and investigate methods to counter them. Specifically, we employ untargeted white-box attacks, namely the Fast Gradient Sign Method (FGSM) and the Basic Iterative Method (BIM), to poison the inputs to the training process, effectively misleading the model. We also illustrate the subtle modifications to the inputs after the attack, which makes detecting the attack using the naked eye quite difficult. Having demonstrated the feasibility of these attacks, we develop robust models through adversarial training and model hardening. We are among the first to showcase the transferability of these attacks and defenses by extrapolating our work from the benchmark electricity data to a larger, 10-year real-world data used for predicting the time-to-failure of hard disks. Our experimental results confirm that the attacks and defenses achieve the desired security thresholds, leading to a 72.41% and 94.81% decrease in RMSE for the electricity and hard disk datasets respectively after implementing the adversarial defenses.



## **4. Evaluating the Defense Potential of Machine Unlearning against Membership Inference Attacks**

cs.CR

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2508.16150v2) [paper-pdf](http://arxiv.org/pdf/2508.16150v2)

**Authors**: Aristeidis Sidiropoulos, Christos Chrysanthos Nikolaidis, Theodoros Tsiolakis, Nikolaos Pavlidis, Vasilis Perifanis, Pavlos S. Efraimidis

**Abstract**: Membership Inference Attacks (MIAs) pose a significant privacy risk, as they enable adversaries to determine whether a specific data point was included in the training dataset of a model. While Machine Unlearning is primarily designed as a privacy mechanism to efficiently remove private data from a machine learning model without the need for full retraining, its impact on the susceptibility of models to MIA remains an open question. In this study, we systematically assess the vulnerability of models to MIA after applying state-of-art Machine Unlearning algorithms. Our analysis spans four diverse datasets (two from the image domain and two in tabular format), exploring how different unlearning approaches influence the exposure of models to membership inference. The findings highlight that while Machine Unlearning is not inherently a countermeasure against MIA, the unlearning algorithm and data characteristics can significantly affect a model's vulnerability. This work provides essential insights into the interplay between Machine Unlearning and MIAs, offering guidance for the design of privacy-preserving machine learning systems.



## **5. Addressing Key Challenges of Adversarial Attacks and Defenses in the Tabular Domain: A Methodological Framework for Coherence and Consistency**

cs.LG

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2412.07326v3) [paper-pdf](http://arxiv.org/pdf/2412.07326v3)

**Authors**: Yael Itzhakev, Amit Giloni, Yuval Elovici, Asaf Shabtai

**Abstract**: Machine learning models trained on tabular data are vulnerable to adversarial attacks, even in realistic scenarios where attackers only have access to the model's outputs. Since tabular data contains complex interdependencies among features, it presents a unique challenge for adversarial samples which must maintain coherence and respect these interdependencies to remain indistinguishable from benign data. Moreover, existing attack evaluation metrics-such as the success rate, perturbation magnitude, and query count-fail to account for this challenge. To address those gaps, we propose a technique for perturbing dependent features while preserving sample coherence. In addition, we introduce Class-Specific Anomaly Detection (CSAD), an effective novel anomaly detection approach, along with concrete metrics for assessing the quality of tabular adversarial attacks. CSAD evaluates adversarial samples relative to their predicted class distribution, rather than a broad benign distribution. It ensures that subtle adversarial perturbations, which may appear coherent in other classes, are correctly identified as anomalies. We integrate SHAP explainability techniques to detect inconsistencies in model decision-making, extending CSAD for SHAP-based anomaly detection. Our evaluation incorporates both anomaly detection rates with SHAP-based assessments to provide a more comprehensive measure of adversarial sample quality. We evaluate various attack strategies, examining black-box query-based and transferability-based gradient attacks across four target models. Experiments on benchmark tabular datasets reveal key differences in the attacker's risk and effort and attack quality, offering insights into the strengths, limitations, and trade-offs faced by attackers and defenders. Our findings lay the groundwork for future research on adversarial attacks and defense development in the tabular domain.



## **6. SoK: Cybersecurity Assessment of Humanoid Ecosystem**

cs.CR

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2508.17481v2) [paper-pdf](http://arxiv.org/pdf/2508.17481v2)

**Authors**: Priyanka Prakash Surve, Asaf Shabtai, Yuval Elovici

**Abstract**: Humanoids are progressing toward practical deployment across healthcare, industrial, defense, and service sectors. While typically considered cyber-physical systems (CPSs), their dependence on traditional networked software stacks (e.g., Linux operating systems), robot operating system (ROS) middleware, and over-the-air update channels, creates a distinct security profile that exposes them to vulnerabilities conventional CPS models do not fully address. Prior studies have mainly examined specific threats, such as LiDAR spoofing or adversarial machine learning (AML). This narrow focus overlooks how an attack targeting one component can cascade harm throughout the robot's interconnected systems. We address this gap through a systematization of knowledge (SoK) that takes a comprehensive approach, consolidating fragmented research from robotics, CPS, and network security domains. We introduce a seven-layer security model for humanoid robots, organizing 39 known attacks and 35 defenses across the humanoid ecosystem-from hardware to human-robot interaction. Building on this security model, we develop a quantitative 39x35 attack-defense matrix with risk-weighted scoring, validated through Monte Carlo analysis. We demonstrate our method by evaluating three real-world robots: Pepper, G1 EDU, and Digit. The scoring analysis revealed varying security maturity levels, with scores ranging from 39.9% to 79.5% across the platforms. This work introduces a structured, evidence-based assessment method that enables systematic security evaluation, supports cross-platform benchmarking, and guides prioritization of security investments in humanoid robotics.



## **7. Privacy-preserving authentication for military 5G networks**

cs.CR

To appear in Proc. IEEE Military Commun. Conf. (MILCOM), (Los  Angeles, CA), Oct. 2025

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01470v1) [paper-pdf](http://arxiv.org/pdf/2509.01470v1)

**Authors**: I. D. Lutz, A. M. Hill, M. C. Valenti

**Abstract**: As 5G networks gain traction in defense applications, ensuring the privacy and integrity of the Authentication and Key Agreement (AKA) protocol is critical. While 5G AKA improves upon previous generations by concealing subscriber identities, it remains vulnerable to replay-based synchronization and linkability threats under realistic adversary models. This paper provides a unified analysis of the standardized 5G AKA flow, identifying several vulnerabilities and highlighting how each exploits protocol behavior to compromise user privacy. To address these risks, we present five lightweight mitigation strategies. We demonstrate through prototype implementation and testing that these enhancements strengthen resilience against linkability attacks with minimal computational and signaling overhead. Among the solutions studied, those introducing a UE-generated nonce emerge as the most promising, effectively neutralizing the identified tracking and correlation attacks with negligible additional overhead. Integrating this extension as an optional feature to the standard 5G AKA protocol offers a backward-compatible, low-overhead path toward a more privacy-preserving authentication framework for both commercial and military 5G deployments.



## **8. LLMHoney: A Real-Time SSH Honeypot with Large Language Model-Driven Dynamic Response Generation**

cs.CR

7 Pages

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01463v1) [paper-pdf](http://arxiv.org/pdf/2509.01463v1)

**Authors**: Pranjay Malhotra

**Abstract**: Cybersecurity honeypots are deception tools for engaging attackers and gather intelligence, but traditional low or medium-interaction honeypots often rely on static, pre-scripted interactions that can be easily identified by skilled adversaries. This Report presents LLMHoney, an SSH honeypot that leverages Large Language Models (LLMs) to generate realistic, dynamic command outputs in real time. LLMHoney integrates a dictionary-based virtual file system to handle common commands with low latency while using LLMs for novel inputs, achieving a balance between authenticity and performance. We implemented LLMHoney using open-source LLMs and evaluated it on a testbed with 138 representative Linux commands. We report comprehensive metrics including accuracy (exact-match, Cosine Similarity, Jaro-Winkler Similarity, Levenshtein Similarity and BLEU score), response latency and memory overhead. We evaluate LLMHoney using multiple LLM backends ranging from 0.36B to 3.8B parameters, including both open-source models and a proprietary model(Gemini). Our experiments compare 13 different LLM variants; results show that Gemini-2.0 and moderately-sized models Qwen2.5:1.5B and Phi3:3.8B provide the most reliable and accurate responses, with mean latencies around 3 seconds, whereas smaller models often produce incorrect or out-of-character outputs. We also discuss how LLM integration improves honeypot realism and adaptability compared to traditional honeypots, as well as challenges such as occasional hallucinated outputs and increased resource usage. Our findings demonstrate that LLM-driven honeypots are a promising approach to enhance attacker engagement and collect richer threat intelligence.



## **9. An Automated Attack Investigation Approach Leveraging Threat-Knowledge-Augmented Large Language Models**

cs.CR

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01271v1) [paper-pdf](http://arxiv.org/pdf/2509.01271v1)

**Authors**: Rujie Dai, Peizhuo Lv, Yujiang Gui, Qiujian Lv, Yuanyuan Qiao, Yan Wang, Degang Sun, Weiqing Huang, Yingjiu Li, XiaoFeng Wang

**Abstract**: Advanced Persistent Threats (APTs) are prolonged, stealthy intrusions by skilled adversaries that compromise high-value systems to steal data or disrupt operations. Reconstructing complete attack chains from massive, heterogeneous logs is essential for effective attack investigation, yet existing methods suffer from poor platform generality, limited generalization to evolving tactics, and an inability to produce analyst-ready reports. Large Language Models (LLMs) offer strong semantic understanding and summarization capabilities, but in this domain they struggle to capture the long-range, cross-log dependencies critical for accurate reconstruction.   To solve these problems, we present an LLM-empowered attack investigation framework augmented with a dynamically adaptable Kill-Chain-aligned threat knowledge base. We organizes attack-relevant behaviors into stage-aware knowledge units enriched with semantic annotations, enabling the LLM to iteratively retrieve relevant intelligence, perform causal reasoning, and progressively expand the investigation context. This process reconstructs multi-phase attack scenarios and generates coherent, human-readable investigation reports. Evaluated on 15 attack scenarios spanning single-host and multi-host environments across Windows and Linux (over 4.3M log events, 7.2 GB of data), the system achieves an average True Positive Rate (TPR) of 97.1% and an average False Positive Rate (FPR) of 0.2%, significantly outperforming the SOTA method ATLAS, which achieves an average TPR of 79.2% and an average FPR of 29.1%.



## **10. Geometric origin of adversarial vulnerability in deep learning**

cs.LG

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01235v1) [paper-pdf](http://arxiv.org/pdf/2509.01235v1)

**Authors**: Yixiong Ren, Wenkang Du, Jianhui Zhou, Haiping Huang

**Abstract**: How to balance training accuracy and adversarial robustness has become a challenge since the birth of deep learning. Here, we introduce a geometry-aware deep learning framework that leverages layer-wise local training to sculpt the internal representations of deep neural networks. This framework promotes intra-class compactness and inter-class separation in feature space, leading to manifold smoothness and adversarial robustness against white or black box attacks. The performance can be explained by an energy model with Hebbian coupling between elements of the hidden representation. Our results thus shed light on the physics of learning in the direction of alignment between biological and artificial intelligence systems. Using the current framework, the deep network can assimilate new information into existing knowledge structures while reducing representation interference.



## **11. Crosstalk Attacks and Defence in a Shared Quantum Computing Environment**

quant-ph

13 pages, 7 figures

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2402.02753v2) [paper-pdf](http://arxiv.org/pdf/2402.02753v2)

**Authors**: Benjamin Harper, Behnam Tonekaboni, Bahar Goldozian, Martin Sevior, Muhammad Usman

**Abstract**: Quantum computing has the potential to provide solutions to problems that are intractable on classical computers, but the accuracy of the current generation of quantum computers suffer from the impact of noise or errors such as leakage, crosstalk, dephasing, and amplitude damping among others. As the access to quantum computers is almost exclusively in a shared environment through cloud-based services, it is possible that an adversary can exploit crosstalk noise to disrupt quantum computations on nearby qubits, even carefully designing quantum circuits to purposely lead to wrong answers. In this paper, we analyze the extent and characteristics of crosstalk noise through tomography conducted on IBM Quantum computers, leading to an enhanced crosstalk simulation model. Our results indicate that crosstalk noise is a significant source of errors on IBM quantum hardware, making crosstalk based attack a viable threat to quantum computing in a shared environment. Based on our crosstalk simulator benchmarked against IBM hardware, we assess the impact of crosstalk attacks and develop strategies for mitigating crosstalk effects. Through a systematic set of simulations, we assess the effectiveness of three crosstalk attack mitigation strategies, namely circuit separation, qubit allocation optimization via reinforcement learning, and the use of spectator qubits, and show that they all overcome crosstalk attacks with varying degrees of success and help to secure quantum computing in a shared platform.



## **12. Clone What You Can't Steal: Black-Box LLM Replication via Logit Leakage and Distillation**

cs.CR

8 pages. Accepted for publication in the proceedings of 7th IEEE  International Conference on Trust, Privacy and Security in Intelligent  Systems, and Applications (IEEE TPS 2025)

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2509.00973v1) [paper-pdf](http://arxiv.org/pdf/2509.00973v1)

**Authors**: Kanchon Gharami, Hansaka Aluvihare, Shafika Showkat Moni, Berker Peköz

**Abstract**: Large Language Models (LLMs) are increasingly deployed in mission-critical systems, facilitating tasks such as satellite operations, command-and-control, military decision support, and cyber defense. Many of these systems are accessed through application programming interfaces (APIs). When such APIs lack robust access controls, they can expose full or top-k logits, creating a significant and often overlooked attack surface. Prior art has mainly focused on reconstructing the output projection layer or distilling surface-level behaviors. However, regenerating a black-box model under tight query constraints remains underexplored. We address that gap by introducing a constrained replication pipeline that transforms partial logit leakage into a functional deployable substitute model clone. Our two-stage approach (i) reconstructs the output projection matrix by collecting top-k logits from under 10k black-box queries via singular value decomposition (SVD) over the logits, then (ii) distills the remaining architecture into compact student models with varying transformer depths, trained on an open source dataset. A 6-layer student recreates 97.6% of the 6-layer teacher model's hidden-state geometry, with only a 7.31% perplexity increase, and a 7.58 Negative Log-Likelihood (NLL). A 4-layer variant achieves 17.1% faster inference and 18.1% parameter reduction with comparable performance. The entire attack completes in under 24 graphics processing unit (GPU) hours and avoids triggering API rate-limit defenses. These results demonstrate how quickly a cost-limited adversary can clone an LLM, underscoring the urgent need for hardened inference APIs and secure on-premise defense deployments.



## **13. FusionCounting: Robust visible-infrared image fusion guided by crowd counting via multi-task learning**

cs.CV

11 pages, 9 figures

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2508.20817v2) [paper-pdf](http://arxiv.org/pdf/2508.20817v2)

**Authors**: He Li, Xinyu Liu, Weihang Kong, Xingchen Zhang

**Abstract**: Visible and infrared image fusion (VIF) is an important multimedia task in computer vision. Most VIF methods focus primarily on optimizing fused image quality. Recent studies have begun incorporating downstream tasks, such as semantic segmentation and object detection, to provide semantic guidance for VIF. However, semantic segmentation requires extensive annotations, while object detection, despite reducing annotation efforts compared with segmentation, faces challenges in highly crowded scenes due to overlapping bounding boxes and occlusion. Moreover, although RGB-T crowd counting has gained increasing attention in recent years, no studies have integrated VIF and crowd counting into a unified framework. To address these challenges, we propose FusionCounting, a novel multi-task learning framework that integrates crowd counting into the VIF process. Crowd counting provides a direct quantitative measure of population density with minimal annotation, making it particularly suitable for dense scenes. Our framework leverages both input images and population density information in a mutually beneficial multi-task design. To accelerate convergence and balance tasks contributions, we introduce a dynamic loss function weighting strategy. Furthermore, we incorporate adversarial training to enhance the robustness of both VIF and crowd counting, improving the model's stability and resilience to adversarial attacks. Experimental results on public datasets demonstrate that FusionCounting not only enhances image fusion quality but also achieves superior crowd counting performance.



## **14. Redesigning Traffic Signs to Mitigate Machine-Learning Patch Attacks**

cs.CR

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2402.04660v3) [paper-pdf](http://arxiv.org/pdf/2402.04660v3)

**Authors**: Tsufit Shua, Liron David, Mahmood Sharif

**Abstract**: Traffic-Sign Recognition (TSR) is a critical safety component for autonomous driving. Unfortunately, however, past work has highlighted the vulnerability of TSR models to physical-world attacks, through low-cost, easily deployable adversarial patches leading to misclassification. To mitigate these threats, most defenses focus on altering the training process or modifying the inference procedure. Still, while these approaches improve adversarial robustness, TSR remains susceptible to attacks attaining substantial success rates.   To further the adversarial robustness of TSR, this work offers a novel approach that redefines traffic-sign designs to create signs that promote robustness while remaining interpretable to humans. Our framework takes three inputs: (1) A traffic-sign standard along with modifiable features and associated constraints; (2) A state-of-the-art adversarial training method; and (3) A function for efficiently synthesizing realistic traffic-sign images. Using these user-defined inputs, the framework emits an optimized traffic-sign standard such that traffic signs generated per this standard enable training TSR models with increased adversarial robustness.   We evaluate the effectiveness of our framework via a concrete implementation, where we allow modifying the pictograms (i.e., symbols) and colors of traffic signs. The results show substantial improvements in robustness -- with gains of up to 16.33%--24.58% in robust accuracy over state-of-the-art methods -- while benign accuracy is even improved. Importantly, a user study also confirms that the redesigned traffic signs remain easily recognizable and to human observers. Overall, the results highlight that carefully redesigning traffic signs can significantly enhance TSR system robustness without compromising human interpretability.



## **15. Sequential Difference Maximization: Generating Adversarial Examples via Multi-Stage Optimization**

cs.CV

5 pages, 2 figures, 5 tables, CIKM 2025

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2509.00826v1) [paper-pdf](http://arxiv.org/pdf/2509.00826v1)

**Authors**: Xinlei Liu, Tao Hu, Peng Yi, Weitao Han, Jichao Xie, Baolin Li

**Abstract**: Efficient adversarial attack methods are critical for assessing the robustness of computer vision models. In this paper, we reconstruct the optimization objective for generating adversarial examples as "maximizing the difference between the non-true labels' probability upper bound and the true label's probability," and propose a gradient-based attack method termed Sequential Difference Maximization (SDM). SDM establishes a three-layer optimization framework of "cycle-stage-step." The processes between cycles and between iterative steps are respectively identical, while optimization stages differ in terms of loss functions: in the initial stage, the negative probability of the true label is used as the loss function to compress the solution space; in subsequent stages, we introduce the Directional Probability Difference Ratio (DPDR) loss function to gradually increase the non-true labels' probability upper bound by compressing the irrelevant labels' probabilities. Experiments demonstrate that compared with previous SOTA methods, SDM not only exhibits stronger attack performance but also achieves higher attack cost-effectiveness. Additionally, SDM can be combined with adversarial training methods to enhance their defensive effects. The code is available at https://github.com/X-L-Liu/SDM.



## **16. Membership Inference Attacks on Large-Scale Models: A Survey**

cs.LG

Preprint. Submitted for peer review. The final version may differ

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2503.19338v3) [paper-pdf](http://arxiv.org/pdf/2503.19338v3)

**Authors**: Hengyu Wu, Yang Cao

**Abstract**: As large-scale models such as Large Language Models (LLMs) and Large Multimodal Models (LMMs) see increasing deployment, their privacy risks remain underexplored. Membership Inference Attacks (MIAs), which reveal whether a data point was used in training the target model, are an important technique for exposing or assessing privacy risks and have been shown to be effective across diverse machine learning algorithms. However, despite extensive studies on MIAs in classic models, there remains a lack of systematic surveys addressing their effectiveness and limitations in large-scale models. To address this gap, we provide the first comprehensive review of MIAs targeting LLMs and LMMs, analyzing attacks by model type, adversarial knowledge, and strategy. Unlike prior surveys, we further examine MIAs across multiple stages of the model pipeline, including pre-training, fine-tuning, alignment, and Retrieval-Augmented Generation (RAG). Finally, we identify open challenges and propose future research directions for strengthening privacy resilience in large-scale models.



## **17. Bayesian and Multi-Objective Decision Support for Real-Time Cyber-Physical Incident Mitigation**

cs.CR

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2509.00770v1) [paper-pdf](http://arxiv.org/pdf/2509.00770v1)

**Authors**: Shaofei Huang, Christopher M. Poskitt, Lwin Khin Shar

**Abstract**: This research proposes a real-time, adaptive decision-support framework for mitigating cyber incidents in cyber-physical systems, developed in response to an increasing reliance on these systems within critical infrastructure and evolving adversarial tactics. Existing decision-support systems often fall short in accounting for multi-agent, multi-path attacks and trade-offs between safety and operational continuity. To address this, our framework integrates hierarchical system modelling with Bayesian probabilistic reasoning, constructing Bayesian Network Graphs from system architecture and vulnerability data. Models are encoded using a Domain Specific Language to enhance computational efficiency and support dynamic updates. In our approach, we use a hybrid exposure probability estimation framework, which combines Exploit Prediction Scoring System and Common Vulnerability Scoring System scores via Bayesian confidence calibration to handle epistemic uncertainty caused by incomplete or heterogeneous vulnerability metadata. Mitigation recommendations are generated as countermeasure portfolios, refined using multi-objective optimisation to identify Pareto-optimal strategies balancing attack likelihood, impact severity, and system availability. To accommodate time- and resource-constrained incident response, frequency-based heuristics are applied to prioritise countermeasures across the optimised portfolios. The framework was evaluated through three representative cyber-physical attack scenarios, demonstrating its versatility in handling complex adversarial behaviours under real-time response constraints. The results affirm its utility in operational contexts and highlight the robustness of our proposed approach across diverse threat environments.



## **18. Enabling Trustworthy Federated Learning via Remote Attestation for Mitigating Byzantine Threats**

cs.CR

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2509.00634v1) [paper-pdf](http://arxiv.org/pdf/2509.00634v1)

**Authors**: Chaoyu Zhang, Heng Jin, Shanghao Shi, Hexuan Yu, Sydney Johns, Y. Thomas Hou, Wenjing Lou

**Abstract**: Federated Learning (FL) has gained significant attention for its privacy-preserving capabilities, enabling distributed devices to collaboratively train a global model without sharing raw data. However, its distributed nature forces the central server to blindly trust the local training process and aggregate uncertain model updates, making it susceptible to Byzantine attacks from malicious participants, especially in mission-critical scenarios. Detecting such attacks is challenging due to the diverse knowledge across clients, where variations in model updates may stem from benign factors, such as non-IID data, rather than adversarial behavior. Existing data-driven defenses struggle to distinguish malicious updates from natural variations, leading to high false positive rates and poor filtering performance.   To address this challenge, we propose Sentinel, a remote attestation (RA)-based scheme for FL systems that regains client-side transparency and mitigates Byzantine attacks from a system security perspective. Our system employs code instrumentation to track control-flow and monitor critical variables in the local training process. Additionally, we utilize a trusted training recorder within a Trusted Execution Environment (TEE) to generate an attestation report, which is cryptographically signed and securely transmitted to the server. Upon verification, the server ensures that legitimate client training processes remain free from program behavior violation or data manipulation, allowing only trusted model updates to be aggregated into the global model. Experimental results on IoT devices demonstrate that Sentinel ensures the trustworthiness of the local training integrity with low runtime and memory overhead.



## **19. On the Thermal Vulnerability of 3D-Stacked High-Bandwidth Memory Architectures**

cs.AR

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2509.00633v1) [paper-pdf](http://arxiv.org/pdf/2509.00633v1)

**Authors**: Mehdi Elahi, Mohamed R. Elshamy, Abdel-Hameed A. Badawy, Ahmad Patooghy

**Abstract**: 3D-stacked High Bandwidth Memory (HBM) architectures provide high-performance memory interactions to address the well-known performance challenge, namely the memory wall. However, these architectures are susceptible to thermal vulnerabilities due to the inherent vertical adjacency that occurs during the manufacturing process of HBM architectures. We anticipate that adversaries may exploit the intense vertical and lateral adjacency to design and develop thermal performance degradation attacks on the memory banks that host data/instructions from victim applications. In such attacks, the adversary manages to inject short and intense heat pulses from vertically and/or laterally adjacent memory banks, creating a convergent thermal wave that maximizes impact and delays the victim application from accessing its data/instructions. As the attacking application does not access any out-of-range memory locations, it can bypass both design-time security tests and the operating system's memory management policies. In other words, since the attack mimics legitimate workloads, it will be challenging to detect.



## **20. FedThief: Harming Others to Benefit Oneself in Self-Centered Federated Learning**

cs.LG

12 pages, 5 figures

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2509.00540v1) [paper-pdf](http://arxiv.org/pdf/2509.00540v1)

**Authors**: Xiangyu Zhang, Mang Ye

**Abstract**: In federated learning, participants' uploaded model updates cannot be directly verified, leaving the system vulnerable to malicious attacks. Existing attack strategies have adversaries upload tampered model updates to degrade the global model's performance. However, attackers also degrade their own private models, gaining no advantage. In real-world scenarios, attackers are driven by self-centered motives: their goal is to gain a competitive advantage by developing a model that outperforms those of other participants, not merely to cause disruption. In this paper, we study a novel Self-Centered Federated Learning (SCFL) attack paradigm, in which attackers not only degrade the performance of the global model through attacks but also enhance their own models within the federated learning process. We propose a framework named FedThief, which degrades the performance of the global model by uploading modified content during the upload stage. At the same time, it enhances the private model's performance through divergence-aware ensemble techniques, where "divergence" quantifies the deviation between private and global models, that integrate global updates and local knowledge. Extensive experiments show that our method effectively degrades the global model performance while allowing the attacker to obtain an ensemble model that significantly outperforms the global model.



## **21. Similarity between Units of Natural Language: The Transition from Coarse to Fine Estimation**

cs.CL

PhD thesis

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2210.14275v2) [paper-pdf](http://arxiv.org/pdf/2210.14275v2)

**Authors**: Wenchuan Mu

**Abstract**: Capturing the similarities between human language units is crucial for explaining how humans associate different objects, and therefore its computation has received extensive attention, research, and applications. With the ever-increasing amount of information around us, calculating similarity becomes increasingly complex, especially in many cases, such as legal or medical affairs, measuring similarity requires extra care and precision, as small acts within a language unit can have significant real-world effects. My research goal in this thesis is to develop regression models that account for similarities between language units in a more refined way.   Computation of similarity has come a long way, but approaches to debugging the measures are often based on continually fitting human judgment values. To this end, my goal is to develop an algorithm that precisely catches loopholes in a similarity calculation. Furthermore, most methods have vague definitions of the similarities they compute and are often difficult to interpret. The proposed framework addresses both shortcomings. It constantly improves the model through catching different loopholes. In addition, every refinement of the model provides a reasonable explanation. The regression model introduced in this thesis is called progressively refined similarity computation, which combines attack testing with adversarial training. The similarity regression model of this thesis achieves state-of-the-art performance in handling edge cases.



## **22. The Resurgence of GCG Adversarial Attacks on Large Language Models**

cs.CL

12 pages, 5 figures

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2509.00391v1) [paper-pdf](http://arxiv.org/pdf/2509.00391v1)

**Authors**: Yuting Tan, Xuying Li, Zhuo Li, Huizhen Shu, Peikang Hu

**Abstract**: Gradient-based adversarial prompting, such as the Greedy Coordinate Gradient (GCG) algorithm, has emerged as a powerful method for jailbreaking large language models (LLMs). In this paper, we present a systematic appraisal of GCG and its annealing-augmented variant, T-GCG, across open-source LLMs of varying scales. Using Qwen2.5-0.5B, LLaMA-3.2-1B, and GPT-OSS-20B, we evaluate attack effectiveness on both safety-oriented prompts (AdvBench) and reasoning-intensive coding prompts. Our study reveals three key findings: (1) attack success rates (ASR) decrease with model size, reflecting the increasing complexity and non-convexity of larger models' loss landscapes; (2) prefix-based heuristics substantially overestimate attack effectiveness compared to GPT-4o semantic judgments, which provide a stricter and more realistic evaluation; and (3) coding-related prompts are significantly more vulnerable than adversarial safety prompts, suggesting that reasoning itself can be exploited as an attack vector. In addition, preliminary results with T-GCG show that simulated annealing can diversify adversarial search and achieve competitive ASR under prefix evaluation, though its benefits under semantic judgment remain limited. Together, these findings highlight the scalability limits of GCG, expose overlooked vulnerabilities in reasoning tasks, and motivate further development of annealing-inspired strategies for more robust adversarial evaluation.



## **23. Unifying Adversarial Perturbation for Graph Neural Networks**

cs.LG

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2509.00387v1) [paper-pdf](http://arxiv.org/pdf/2509.00387v1)

**Authors**: Jinluan Yang, Ruihao Zhang, Zhengyu Chen, Fei Wu, Kun Kuang

**Abstract**: This paper studies the vulnerability of Graph Neural Networks (GNNs) to adversarial attacks on node features and graph structure. Various methods have implemented adversarial training to augment graph data, aiming to bolster the robustness and generalization of GNNs. These methods typically involve applying perturbations to the node feature, weights, or graph structure and subsequently minimizing the loss by learning more robust graph model parameters under the adversarial perturbations. Despite the effectiveness of adversarial training in enhancing GNNs' robustness and generalization abilities, its application has been largely confined to specific datasets and GNN types. In this paper, we propose a novel method, PerturbEmbedding, that integrates adversarial perturbation and training, enhancing GNNs' resilience to such attacks and improving their generalization ability. PerturbEmbedding performs perturbation operations directly on every hidden embedding of GNNs and provides a unified framework for most existing perturbation strategies/methods. We also offer a unified perspective on the forms of perturbations, namely random and adversarial perturbations. Through experiments on various datasets using different backbone models, we demonstrate that PerturbEmbedding significantly improves both the robustness and generalization abilities of GNNs, outperforming existing methods. The rejection of both random (non-targeted) and adversarial (targeted) perturbations further enhances the backbone model's performance.



## **24. Activation Steering Meets Preference Optimization: Defense Against Jailbreaks in Vision Language Models**

cs.CV

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2509.00373v1) [paper-pdf](http://arxiv.org/pdf/2509.00373v1)

**Authors**: Sihao Wu, Gaojie Jin, Wei Huang, Jianhong Wang, Xiaowei Huang

**Abstract**: Vision Language Models (VLMs) have demonstrated impressive capabilities in integrating visual and textual information for understanding and reasoning, but remain highly vulnerable to adversarial attacks. While activation steering has emerged as a promising defence, existing approaches often rely on task-specific contrastive prompts to extract harmful directions, which exhibit suboptimal performance and can degrade visual grounding performance. To address these limitations, we propose \textit{Sequence-Level Preference Optimization} for VLM (\textit{SPO-VLM}), a novel two-stage defense framework that combines activation-level intervention with policy-level optimization to enhance model robustness. In \textit{Stage I}, we compute adaptive layer-specific steering vectors from diverse data sources, enabling generalized suppression of harmful behaviors during inference. In \textit{Stage II}, we refine these steering vectors through a sequence-level preference optimization process. This stage integrates automated toxicity assessment, as well as visual-consistency rewards based on caption-image alignment, to achieve safe and semantically grounded text generation. The two-stage structure of SPO-VLM balances efficiency and effectiveness by combining a lightweight mitigation foundation in Stage I with deeper policy refinement in Stage II. Extensive experiments shown SPO-VLM enhances safety against attacks via activation steering and preference optimization, while maintaining strong performance on benign tasks without compromising visual understanding capabilities. We will release our code, model weights, and evaluation toolkit to support reproducibility and future research. \textcolor{red}{Warning: This paper may contain examples of offensive or harmful text and images.}



## **25. MorphGen: Morphology-Guided Representation Learning for Robust Single-Domain Generalization in Histopathological Cancer Classification**

cs.CV

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2509.00311v1) [paper-pdf](http://arxiv.org/pdf/2509.00311v1)

**Authors**: Hikmat Khan, Syed Farhan Alam Zaidi, Pir Masoom Shah, Kiruthika Balakrishnan, Rabia Khan, Muhammad Waqas, Jia Wu

**Abstract**: Domain generalization in computational histopathology is hindered by heterogeneity in whole slide images (WSIs), caused by variations in tissue preparation, staining, and imaging conditions across institutions. Unlike machine learning systems, pathologists rely on domain-invariant morphological cues such as nuclear atypia (enlargement, irregular contours, hyperchromasia, chromatin texture, spatial disorganization), structural atypia (abnormal architecture and gland formation), and overall morphological atypia that remain diagnostic across diverse settings. Motivated by this, we hypothesize that explicitly modeling biologically robust nuclear morphology and spatial organization will enable the learning of cancer representations that are resilient to domain shifts. We propose MorphGen (Morphology-Guided Generalization), a method that integrates histopathology images, augmentations, and nuclear segmentation masks within a supervised contrastive learning framework. By aligning latent representations of images and nuclear masks, MorphGen prioritizes diagnostic features such as nuclear and morphological atypia and spatial organization over staining artifacts and domain-specific features. To further enhance out-of-distribution robustness, we incorporate stochastic weight averaging (SWA), steering optimization toward flatter minima. Attention map analyses revealed that MorphGen primarily relies on nuclear morphology, cellular composition, and spatial cell organization within tumors or normal regions for final classification. Finally, we demonstrate resilience of the learned representations to image corruptions (such as staining artifacts) and adversarial attacks, showcasing not only OOD generalization but also addressing critical vulnerabilities in current deep learning systems for digital pathology. Code, datasets, and trained models are available at: https://github.com/hikmatkhan/MorphGen



## **26. A Systematic Approach to Estimate the Security Posture of a Cyber Infrastructure: A Technical Report**

cs.CR

11 pages, 5 figures, technical report

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2509.00266v1) [paper-pdf](http://arxiv.org/pdf/2509.00266v1)

**Authors**: Qishen Sam Liang

**Abstract**: Academic and research Cyber Infrastructures (CI) present unique security challenges due to their collaborative nature, heterogeneous components, and the lack of practical, tailored security assessment frameworks. Existing standards can be too generic or complex for CI administrators to apply effectively. This report introduces a systematic, mission-centric approach to estimate and analyze the security posture of a CI. The framework guides administrators through a top-down process: (1) defining unacceptable losses and security missions, (2) identifying associated system hazards and critical assets, and (3) modeling the CI's components and their relationships as a security knowledge graph. The core of this methodology is the construction of directed attack graphs, which systematically map all potential paths an adversary could take from an entry point to a critical asset. By visualizing these attack paths alongside defense mechanisms, the framework provides a clear, comprehensive overview of the system's vulnerabilities and security gaps. This structured approach enables CI operators to proactively assess risks, prioritize mitigation strategies, and make informed, actionable decisions to strengthen the overall security posture of the CI.



## **27. Detecting Stealthy Data Poisoning Attacks in AI Code Generators**

cs.CR

Accepted to the 3rd IEEE International Workshop on Reliable and  Secure AI for Software Engineering (ReSAISE, 2025), co-located with ISSRE  2025

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21636v1) [paper-pdf](http://arxiv.org/pdf/2508.21636v1)

**Authors**: Cristina Improta

**Abstract**: Deep learning (DL) models for natural language-to-code generation have become integral to modern software development pipelines. However, their heavy reliance on large amounts of data, often collected from unsanitized online sources, exposes them to data poisoning attacks, where adversaries inject malicious samples to subtly bias model behavior. Recent targeted attacks silently replace secure code with semantically equivalent but vulnerable implementations without relying on explicit triggers to launch the attack, making it especially hard for detection methods to distinguish clean from poisoned samples. We present a systematic study on the effectiveness of existing poisoning detection methods under this stealthy threat model. Specifically, we perform targeted poisoning on three DL models (CodeBERT, CodeT5+, AST-T5), and evaluate spectral signatures analysis, activation clustering, and static analysis as defenses. Our results show that all methods struggle to detect triggerless poisoning, with representation-based approaches failing to isolate poisoned samples and static analysis suffering false positives and false negatives, highlighting the need for more robust, trigger-independent defenses for AI-assisted code generation.



## **28. Adversarial Patch Attack for Ship Detection via Localized Augmentation**

cs.CV

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21472v1) [paper-pdf](http://arxiv.org/pdf/2508.21472v1)

**Authors**: Chun Liu, Panpan Ding, Zheng Zheng, Hailong Wang, Bingqian Zhu, Tao Xu, Zhigang Han, Jiayao Wang

**Abstract**: Current ship detection techniques based on remote sensing imagery primarily rely on the object detection capabilities of deep neural networks (DNNs). However, DNNs are vulnerable to adversarial patch attacks, which can lead to misclassification by the detection model or complete evasion of the targets. Numerous studies have demonstrated that data transformation-based methods can improve the transferability of adversarial examples. However, excessive augmentation of image backgrounds or irrelevant regions may introduce unnecessary interference, resulting in false detections of the object detection model. These errors are not caused by the adversarial patches themselves but rather by the over-augmentation of background and non-target areas. This paper proposes a localized augmentation method that applies augmentation only to the target regions, avoiding any influence on non-target areas. By reducing background interference, this approach enables the loss function to focus more directly on the impact of the adversarial patch on the detection model, thereby improving the attack success rate. Experiments conducted on the HRSC2016 dataset demonstrate that the proposed method effectively increases the success rate of adversarial patch attacks and enhances their transferability.



## **29. Publish to Perish: Prompt Injection Attacks on LLM-Assisted Peer Review**

cs.CR

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.20863v2) [paper-pdf](http://arxiv.org/pdf/2508.20863v2)

**Authors**: Matteo Gioele Collu, Umberto Salviati, Roberto Confalonieri, Mauro Conti, Giovanni Apruzzese

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into the scientific peer-review process, raising new questions about their reliability and resilience to manipulation. In this work, we investigate the potential for hidden prompt injection attacks, where authors embed adversarial text within a paper's PDF to influence the LLM-generated review. We begin by formalising three distinct threat models that envision attackers with different motivations -- not all of which implying malicious intent. For each threat model, we design adversarial prompts that remain invisible to human readers yet can steer an LLM's output toward the author's desired outcome. Using a user study with domain scholars, we derive four representative reviewing prompts used to elicit peer reviews from LLMs. We then evaluate the robustness of our adversarial prompts across (i) different reviewing prompts, (ii) different commercial LLM-based systems, and (iii) different peer-reviewed papers. Our results show that adversarial prompts can reliably mislead the LLM, sometimes in ways that adversely affect a "honest-but-lazy" reviewer. Finally, we propose and empirically assess methods to reduce detectability of adversarial prompts under automated content checks.



## **30. Time Tells All: Deanonymization of Blockchain RPC Users with Zero Transaction Fee (Extended Version)**

cs.CR

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21440v1) [paper-pdf](http://arxiv.org/pdf/2508.21440v1)

**Authors**: Shan Wang, Ming Yang, Yu Liu, Yue Zhang, Shuaiqing Zhang, Zhen Ling, Jiannong Cao, Xinwen Fu

**Abstract**: Remote Procedure Call (RPC) services have become a primary gateway for users to access public blockchains. While they offer significant convenience, RPC services also introduce critical privacy challenges that remain insufficiently examined. Existing deanonymization attacks either do not apply to blockchain RPC users or incur costs like transaction fees assuming an active network eavesdropper. In this paper, we propose a novel deanonymization attack that can link an IP address of a RPC user to this user's blockchain pseudonym. Our analysis reveals a temporal correlation between the timestamps of transaction confirmations recorded on the public ledger and those of TCP packets sent by the victim when querying transaction status. We assume a strong passive adversary with access to network infrastructure, capable of monitoring traffic at network border routers or Internet exchange points. By monitoring network traffic and analyzing public ledgers, the attacker can link the IP address of the TCP packet to the pseudonym of the transaction initiator by exploiting the temporal correlation. This deanonymization attack incurs zero transaction fee. We mathematically model and analyze the attack method, perform large-scale measurements of blockchain ledgers, and conduct real-world attacks to validate the attack. Our attack achieves a high success rate of over 95% against normal RPC users on various blockchain networks, including Ethereum, Bitcoin and Solana.



## **31. A Whole New World: Creating a Parallel-Poisoned Web Only AI-Agents Can See**

cs.CR

10 pages, 1 figure

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2509.00124v1) [paper-pdf](http://arxiv.org/pdf/2509.00124v1)

**Authors**: Shaked Zychlinski

**Abstract**: This paper introduces a novel attack vector that leverages website cloaking techniques to compromise autonomous web-browsing agents powered by Large Language Models (LLMs). As these agents become more prevalent, their unique and often homogenous digital fingerprints - comprising browser attributes, automation framework signatures, and network characteristics - create a new, distinguishable class of web traffic. The attack exploits this fingerprintability. A malicious website can identify an incoming request as originating from an AI agent and dynamically serve a different, "cloaked" version of its content. While human users see a benign webpage, the agent is presented with a visually identical page embedded with hidden, malicious instructions, such as indirect prompt injections. This mechanism allows adversaries to hijack agent behavior, leading to data exfiltration, malware execution, or misinformation propagation, all while remaining completely invisible to human users and conventional security crawlers. This work formalizes the threat model, details the mechanics of agent fingerprinting and cloaking, and discusses the profound security implications for the future of agentic AI, highlighting the urgent need for robust defenses against this stealthy and scalable attack.



## **32. On the Adversarial Robustness of Spiking Neural Networks Trained by Local Learning**

cs.LG

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2504.08897v2) [paper-pdf](http://arxiv.org/pdf/2504.08897v2)

**Authors**: Jiaqi Lin, Abhronil Sengupta

**Abstract**: Recent research has shown the vulnerability of Spiking Neural Networks (SNNs) under adversarial examples that are nearly indistinguishable from clean data in the context of frame-based and event-based information. The majority of these studies are constrained in generating adversarial examples using Backpropagation Through Time (BPTT), a gradient-based method which lacks biological plausibility. In contrast, local learning methods, which relax many of BPTT's constraints, remain under-explored in the context of adversarial attacks. To address this problem, we examine adversarial robustness in SNNs through the framework of four types of training algorithms. We provide an in-depth analysis of the ineffectiveness of gradient-based adversarial attacks to generate adversarial instances in this scenario. To overcome these limitations, we introduce a hybrid adversarial attack paradigm that leverages the transferability of adversarial instances. The proposed hybrid approach demonstrates superior performance, outperforming existing adversarial attack methods. Furthermore, the generalizability of the method is assessed under multi-step adversarial attacks, adversarial attacks in black-box FGSM scenarios, and within the non-spiking domain.



## **33. The WASM Cloak: Evaluating Browser Fingerprinting Defenses Under WebAssembly based Obfuscation**

cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.21219v1) [paper-pdf](http://arxiv.org/pdf/2508.21219v1)

**Authors**: A H M Nazmus Sakib, Mahsin Bin Akram, Joseph Spracklen, Sahan Kalutarage, Raveen Wijewickrama, Igor Bilogrevic, Murtuza Jadliwala

**Abstract**: Browser fingerprinting defenses have historically focused on detecting JavaScript(JS)-based tracking techniques. However, the widespread adoption of WebAssembly (WASM) introduces a potential blind spot, as adversaries can convert JS to WASM's low-level binary format to obfuscate malicious logic. This paper presents the first systematic evaluation of how such WASM-based obfuscation impacts the robustness of modern fingerprinting defenses. We develop an automated pipeline that translates real-world JS fingerprinting scripts into functional WASM-obfuscated variants and test them against two classes of defenses: state-of-the-art detectors in research literature and commercial, in-browser tools. Our findings reveal a notable divergence: detectors proposed in the research literature that rely on feature-based analysis of source code show moderate vulnerability, stemming from outdated datasets or a lack of WASM compatibility. In contrast, defenses such as browser extensions and native browser features remained completely effective, as their API-level interception is agnostic to the script's underlying implementation. These results highlight a gap between academic and practical defense strategies and offer insights into strengthening detection approaches against WASM-based obfuscation, while also revealing opportunities for more evasive techniques in future attacks.



## **34. First-Place Solution to NeurIPS 2024 Invisible Watermark Removal Challenge**

cs.CV

Winning solution to the NeurIPS 2024 Erasing the Invisible challenge

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.21072v1) [paper-pdf](http://arxiv.org/pdf/2508.21072v1)

**Authors**: Fahad Shamshad, Tameem Bakr, Yahia Shaaban, Noor Hussein, Karthik Nandakumar, Nils Lukas

**Abstract**: Content watermarking is an important tool for the authentication and copyright protection of digital media. However, it is unclear whether existing watermarks are robust against adversarial attacks. We present the winning solution to the NeurIPS 2024 Erasing the Invisible challenge, which stress-tests watermark robustness under varying degrees of adversary knowledge. The challenge consisted of two tracks: a black-box and beige-box track, depending on whether the adversary knows which watermarking method was used by the provider. For the beige-box track, we leverage an adaptive VAE-based evasion attack, with a test-time optimization and color-contrast restoration in CIELAB space to preserve the image's quality. For the black-box track, we first cluster images based on their artifacts in the spatial or frequency-domain. Then, we apply image-to-image diffusion models with controlled noise injection and semantic priors from ChatGPT-generated captions to each cluster with optimized parameter settings. Empirical evaluations demonstrate that our method successfully achieves near-perfect watermark removal (95.7%) with negligible impact on the residual image's quality. We hope that our attacks inspire the development of more robust image watermarking methods.



## **35. PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance**

cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20890v1) [paper-pdf](http://arxiv.org/pdf/2508.20890v1)

**Authors**: Mengxiao Wang, Yuxuan Zhang, Guofei Gu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications, from virtual assistants to autonomous agents. However, their flexibility also introduces new attack vectors-particularly Prompt Injection (PI), where adversaries manipulate model behavior through crafted inputs. As attackers continuously evolve with paraphrased, obfuscated, and even multi-task injection strategies, existing benchmarks are no longer sufficient to capture the full spectrum of emerging threats.   To address this gap, we construct a new benchmark that systematically extends prior efforts. Our benchmark subsumes the two widely-used existing ones while introducing new manipulation techniques and multi-task scenarios, thereby providing a more comprehensive evaluation setting. We find that existing defenses, though effective on their original benchmarks, show clear weaknesses under our benchmark, underscoring the need for more robust solutions. Our key insight is that while attack forms may vary, the adversary's intent-injecting an unauthorized task-remains invariant. Building on this observation, we propose PromptSleuth, a semantic-oriented defense framework that detects prompt injection by reasoning over task-level intent rather than surface features. Evaluated across state-of-the-art benchmarks, PromptSleuth consistently outperforms existing defense while maintaining comparable runtime and cost efficiency. These results demonstrate that intent-based semantic reasoning offers a robust, efficient, and generalizable strategy for defending LLMs against evolving prompt injection threats.



## **36. Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning**

cs.LG

Project Hompage: https://tokenbuncher.github.io/

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20697v1) [paper-pdf](http://arxiv.org/pdf/2508.20697v1)

**Authors**: Weitao Feng, Lixu Wang, Tianyi Wei, Jie Zhang, Chongyang Gao, Sinong Zhan, Peizhuo Lv, Wei Dong

**Abstract**: As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response uncertainty. By constraining uncertainty, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of expert-domain harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task utility and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense.



## **37. Disruptive Attacks on Face Swapping via Low-Frequency Perceptual Perturbations**

cs.CV

Accepted to IEEE IJCNN 2025

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20595v1) [paper-pdf](http://arxiv.org/pdf/2508.20595v1)

**Authors**: Mengxiao Huang, Minglei Shu, Shuwang Zhou, Zhaoyang Liu

**Abstract**: Deepfake technology, driven by Generative Adversarial Networks (GANs), poses significant risks to privacy and societal security. Existing detection methods are predominantly passive, focusing on post-event analysis without preventing attacks. To address this, we propose an active defense method based on low-frequency perceptual perturbations to disrupt face swapping manipulation, reducing the performance and naturalness of generated content. Unlike prior approaches that used low-frequency perturbations to impact classification accuracy,our method directly targets the generative process of deepfake techniques. We combine frequency and spatial domain features to strengthen defenses. By introducing artifacts through low-frequency perturbations while preserving high-frequency details, we ensure the output remains visually plausible. Additionally, we design a complete architecture featuring an encoder, a perturbation generator, and a decoder, leveraging discrete wavelet transform (DWT) to extract low-frequency components and generate perturbations that disrupt facial manipulation models. Experiments on CelebA-HQ and LFW demonstrate significant reductions in face-swapping effectiveness, improved defense success rates, and preservation of visual quality.



## **38. Probabilistic Modeling of Jailbreak on Multimodal LLMs: From Quantification to Application**

cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2503.06989v4) [paper-pdf](http://arxiv.org/pdf/2503.06989v4)

**Authors**: Wenzhuo Xu, Zhipeng Wei, Xiongtao Sun, Zonghao Ying, Deyue Zhang, Dongdong Yang, Xiangzheng Zhang, Quanchen Zou

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal content. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on input image to maximize jailbreak probability, and further enhance it as Multimodal JPA (MJPA) by including monotonic text rephrasing. To counteract attacks, we also propose Jailbreak-Probability-based Finetuning (JPF), which minimizes jailbreak probability through MLLM parameter updates. Extensive experiments show that (1) (M)JPA yields significant improvements when attacking a wide range of models under both white and black box settings. (2) JPF vastly reduces jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.



## **39. Formal Verification of Physical Layer Security Protocols for Next-Generation Communication Networks (extended version)**

cs.CR

Extended version (with appendices) of the camera-ready for ICFEM2025;  24 pages, 3 tables, and 6 figures

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.19430v2) [paper-pdf](http://arxiv.org/pdf/2508.19430v2)

**Authors**: Kangfeng Ye, Roberto Metere, Jim Woodcock, Poonam Yadav

**Abstract**: Formal verification is crucial for ensuring the robustness of security protocols against adversarial attacks. The Needham-Schroeder protocol, a foundational authentication mechanism, has been extensively studied, including its integration with Physical Layer Security (PLS) techniques such as watermarking and jamming. Recent research has used ProVerif to verify these mechanisms in terms of secrecy. However, the ProVerif-based approach limits the ability to improve understanding of security beyond verification results. To overcome these limitations, we re-model the same protocol using an Isabelle formalism that generates sound animation, enabling interactive and automated formal verification of security protocols. Our modelling and verification framework is generic and highly configurable, supporting both cryptography and PLS. For the same protocol, we have conducted a comprehensive analysis (secrecy and authenticity in four different eavesdropper locations under both passive and active attacks) using our new web interface. Our findings not only successfully reproduce and reinforce previous results on secrecy but also reveal an uncommon but expected outcome: authenticity is preserved across all examined scenarios, even in cases where secrecy is compromised. We have proposed a PLS-based Diffie-Hellman protocol that integrates watermarking and jamming, and our analysis shows that it is secure for deriving a session key with required authentication. These highlight the advantages of our novel approach, demonstrating its robustness in formally verifying security properties beyond conventional methods.



## **40. Enhancing Resilience for IoE: A Perspective of Networking-Level Safeguard**

cs.CR

To be published in IEEE Network Magazine, 2026

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20504v1) [paper-pdf](http://arxiv.org/pdf/2508.20504v1)

**Authors**: Guan-Yan Yang, Jui-Ning Chen, Farn Wang, Kuo-Hui Yeh

**Abstract**: The Internet of Energy (IoE) integrates IoT-driven digital communication with power grids to enable efficient and sustainable energy systems. Still, its interconnectivity exposes critical infrastructure to sophisticated cyber threats, including adversarial attacks designed to bypass traditional safeguards. Unlike general IoT risks, IoE threats have heightened public safety consequences, demanding resilient solutions. From the networking-level safeguard perspective, we propose a Graph Structure Learning (GSL)-based safeguards framework that jointly optimizes graph topology and node representations to resist adversarial network model manipulation inherently. Through a conceptual overview, architectural discussion, and case study on a security dataset, we demonstrate GSL's superior robustness over representative methods, offering practitioners a viable path to secure IoE networks against evolving attacks. This work highlights the potential of GSL to enhance the resilience and reliability of future IoE networks for practitioners managing critical infrastructure. Lastly, we identify key open challenges and propose future research directions in this novel research area.



## **41. When Memory Becomes a Vulnerability: Towards Multi-turn Jailbreak Attacks against Text-to-Image Generation Systems**

cs.CV

This work proposes a multi-turn jailbreak attack against real-world  chat-based T2I generation systems that intergrate memory mechanism. It also  constructed a simulation system, with considering three industrial-grade  memory mechanisms, 7 kinds of safety filters (both input and output)

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2504.20376v2) [paper-pdf](http://arxiv.org/pdf/2504.20376v2)

**Authors**: Shiqian Zhao, Jiayang Liu, Yiming Li, Runyi Hu, Xiaojun Jia, Wenshu Fan, Xinfeng Li, Jie Zhang, Wei Dong, Tianwei Zhang, Luu Anh Tuan

**Abstract**: Modern text-to-image (T2I) generation systems (e.g., DALL$\cdot$E 3) exploit the memory mechanism, which captures key information in multi-turn interactions for faithful generation. Despite its practicality, the security analyses of this mechanism have fallen far behind. In this paper, we reveal that it can exacerbate the risk of jailbreak attacks. Previous attacks fuse the unsafe target prompt into one ultimate adversarial prompt, which can be easily detected or lead to the generation of non-unsafe images due to under- or over-detoxification. In contrast, we propose embedding the malice at the inception of the chat session in memory, addressing the above limitations.   Specifically, we propose Inception, the first multi-turn jailbreak attack against real-world text-to-image generation systems that explicitly exploits their memory mechanisms. Inception is composed of two key modules: segmentation and recursion. We introduce Segmentation, a semantic-preserving method that generates multi-round prompts. By leveraging NLP analysis techniques, we design policies to decompose a prompt, together with its malicious intent, according to sentence structure, thereby evading safety filters. Recursion further addresses the challenge posed by unsafe sub-prompts that cannot be separated through simple segmentation. It firstly expands the sub-prompt, then invokes segmentation recursively. To facilitate multi-turn adversarial prompts crafting, we build VisionFlow, an emulation T2I system that integrates two-stage safety filters and industrial-grade memory mechanisms. The experiment results show that Inception successfully allures unsafe image generation, surpassing the SOTA by a 20.0\% margin in attack success rate. We also conduct experiments on the real-world commercial T2I generation platforms, further validating the threats of Inception in practice.



## **42. Safe-Control: A Safety Patch for Mitigating Unsafe Content in Text-to-Image Generation Models**

cs.CV

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.21099v1) [paper-pdf](http://arxiv.org/pdf/2508.21099v1)

**Authors**: Xiangtao Meng, Yingkai Dong, Ning Yu, Li Wang, Zheng Li, Shanqing Guo

**Abstract**: Despite the advancements in Text-to-Image (T2I) generation models, their potential for misuse or even abuse raises serious safety concerns. Model developers have made tremendous efforts to introduce safety mechanisms that can address these concerns in T2I models. However, the existing safety mechanisms, whether external or internal, either remain susceptible to evasion under distribution shifts or require extensive model-specific adjustments. To address these limitations, we introduce Safe-Control, an innovative plug-and-play safety patch designed to mitigate unsafe content generation in T2I models. Using data-driven strategies and safety-aware conditions, Safe-Control injects safety control signals into the locked T2I model, acting as an update in a patch-like manner. Model developers can also construct various safety patches to meet the evolving safety requirements, which can be flexibly merged into a single, unified patch. Its plug-and-play design further ensures adaptability, making it compatible with other T2I models of similar denoising architecture. We conduct extensive evaluations on six diverse and public T2I models. Empirical results highlight that Safe-Control is effective in reducing unsafe content generation across six diverse T2I models with similar generative architectures, yet it successfully maintains the quality and text alignment of benign images. Compared to seven state-of-the-art safety mechanisms, including both external and internal defenses, Safe-Control significantly outperforms all baselines in reducing unsafe content generation. For example, it reduces the probability of unsafe content generation to 7%, compared to approximately 20% for most baseline methods, under both unsafe prompts and the latest adversarial attacks.



## **43. Enhanced Rényi Entropy-Based Post-Quantum Key Agreement with Provable Security and Information-Theoretic Guarantees**

cs.CR

11 pages, 3 tables

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2509.00104v1) [paper-pdf](http://arxiv.org/pdf/2509.00104v1)

**Authors**: Ruopengyu Xu, Chenglian Liu

**Abstract**: This paper presents an enhanced post-quantum key agreement protocol based on R\'{e}nyi entropy, addressing vulnerabilities in the original construction while preserving information-theoretic security properties. We develop a theoretical framework leveraging entropy-preserving operations and secret-shared verification to achieve provable security against quantum adversaries. Through entropy amplification techniques and quantum-resistant commitments, the protocol establishes $2^{128}$ quantum security guarantees under the quantum random oracle model. Key innovations include a confidentiality-preserving verification mechanism using distributed polynomial commitments, tightened min-entropy bounds with guaranteed non-negativity, and composable security proofs in the quantum universal composability framework. Unlike computational approaches, our method provides information-theoretic security without hardness assumptions while maintaining polynomial complexity. Theoretical analysis demonstrates resilience against known quantum attack vectors, including Grover-accelerated brute force and quantum memory attacks. The protocol achieves parameterization for 128-bit quantum security with efficient $\mathcal{O}(n^2)$ communication complexity. Extensions to secure multiparty computation and quantum network applications are established, providing a foundation for long-term cryptographic security. All security claims are derived from mathematical proofs; this theoretical work presents no experimental validation.



## **44. Poison Once, Refuse Forever: Weaponizing Alignment for Injecting Bias in LLMs**

cs.LG

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20333v1) [paper-pdf](http://arxiv.org/pdf/2508.20333v1)

**Authors**: Md Abdullah Al Mamun, Ihsen Alouani, Nael Abu-Ghazaleh

**Abstract**: Large Language Models (LLMs) are aligned to meet ethical standards and safety requirements by training them to refuse answering harmful or unsafe prompts. In this paper, we demonstrate how adversaries can exploit LLMs' alignment to implant bias, or enforce targeted censorship without degrading the model's responsiveness to unrelated topics. Specifically, we propose Subversive Alignment Injection (SAI), a poisoning attack that leverages the alignment mechanism to trigger refusal on specific topics or queries predefined by the adversary. Although it is perhaps not surprising that refusal can be induced through overalignment, we demonstrate how this refusal can be exploited to inject bias into the model. Surprisingly, SAI evades state-of-the-art poisoning defenses including LLM state forensics, as well as robust aggregation techniques that are designed to detect poisoning in FL settings. We demonstrate the practical dangers of this attack by illustrating its end-to-end impacts on LLM-powered application pipelines. For chat based applications such as ChatDoctor, with 1% data poisoning, the system refuses to answer healthcare questions to targeted racial category leading to high bias ($\Delta DP$ of 23%). We also show that bias can be induced in other NLP tasks: for a resume selection pipeline aligned to refuse to summarize CVs from a selected university, high bias in selection ($\Delta DP$ of 27%) results. Even higher bias ($\Delta DP$~38%) results on 9 other chat based downstream applications.



## **45. Differentially Private Federated Quantum Learning via Quantum Noise**

quant-ph

This paper has been accepted at 2025 IEEE International Conference on  Quantum Computing and Engineering (QCE)

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.20310v1) [paper-pdf](http://arxiv.org/pdf/2508.20310v1)

**Authors**: Atit Pokharel, Ratun Rahman, Shaba Shaon, Thomas Morris, Dinh C. Nguyen

**Abstract**: Quantum federated learning (QFL) enables collaborative training of quantum machine learning (QML) models across distributed quantum devices without raw data exchange. However, QFL remains vulnerable to adversarial attacks, where shared QML model updates can be exploited to undermine information privacy. In the context of noisy intermediate-scale quantum (NISQ) devices, a key question arises: How can inherent quantum noise be leveraged to enforce differential privacy (DP) and protect model information during training and communication? This paper explores a novel DP mechanism that harnesses quantum noise to safeguard quantum models throughout the QFL process. By tuning noise variance through measurement shots and depolarizing channel strength, our approach achieves desired DP levels tailored to NISQ constraints. Simulations demonstrate the framework's effectiveness by examining the relationship between differential privacy budget and noise parameters, as well as the trade-off between security and training accuracy. Additionally, we demonstrate the framework's robustness against an adversarial attack designed to compromise model performance using adversarial examples, with evaluations based on critical metrics such as accuracy on adversarial examples, confidence scores for correct predictions, and attack success rates. The results reveal a tunable trade-off between privacy and robustness, providing an efficient solution for secure QFL on NISQ devices with significant potential for reliable quantum computing applications.



## **46. CoCoTen: Detecting Adversarial Inputs to Large Language Models through Latent Space Features of Contextual Co-occurrence Tensors**

cs.CL

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.02997v3) [paper-pdf](http://arxiv.org/pdf/2508.02997v3)

**Authors**: Sri Durga Sai Sowmya Kadali, Evangelos E. Papalexakis

**Abstract**: The widespread use of Large Language Models (LLMs) in many applications marks a significant advance in research and practice. However, their complexity and hard-to-understand nature make them vulnerable to attacks, especially jailbreaks designed to produce harmful responses. To counter these threats, developing strong detection methods is essential for the safe and reliable use of LLMs. This paper studies this detection problem using the Contextual Co-occurrence Matrix, a structure recognized for its efficacy in data-scarce environments. We propose a novel method leveraging the latent space characteristics of Contextual Co-occurrence Matrices and Tensors for the effective identification of adversarial and jailbreak prompts. Our evaluations show that this approach achieves a notable F1 score of 0.83 using only 0.5% of labeled prompts, which is a 96.6% improvement over baselines. This result highlights the strength of our learned patterns, especially when labeled data is scarce. Our method is also significantly faster, speedup ranging from 2.3 to 128.4 times compared to the baseline models.



## **47. Network-Level Prompt and Trait Leakage in Local Research Agents**

cs.CR

under review

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.20282v1) [paper-pdf](http://arxiv.org/pdf/2508.20282v1)

**Authors**: Hyejun Jeong, Mohammadreze Teymoorianfard, Abhinav Kumar, Amir Houmansadr, Eugene Badasarian

**Abstract**: We show that Web and Research Agents (WRAs) -- language model-based systems that investigate complex topics on the Internet -- are vulnerable to inference attacks by passive network adversaries such as ISPs. These agents could be deployed \emph{locally} by organizations and individuals for privacy, legal, or financial purposes. Unlike sporadic web browsing by humans, WRAs visit $70{-}140$ domains with distinguishable timing correlations, enabling unique fingerprinting attacks.   Specifically, we demonstrate a novel prompt and user trait leakage attack against WRAs that only leverages their network-level metadata (i.e., visited IP addresses and their timings). We start by building a new dataset of WRA traces based on user search queries and queries generated by synthetic personas. We define a behavioral metric (called OBELS) to comprehensively assess similarity between original and inferred prompts, showing that our attack recovers over 73\% of the functional and domain knowledge of user prompts. Extending to a multi-session setting, we recover up to 19 of 32 latent traits with high accuracy. Our attack remains effective under partial observability and noisy conditions. Finally, we discuss mitigation strategies that constrain domain diversity or obfuscate traces, showing negligible utility impact while reducing attack effectiveness by an average of 29\%.



## **48. Adversarial Manipulation of Reasoning Models using Internal Representations**

cs.CL

Accepted to the ICML 2025 Workshop on Reliable and Responsible  Foundation Models (R2FM). 20 pages, 12 figures

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2507.03167v2) [paper-pdf](http://arxiv.org/pdf/2507.03167v2)

**Authors**: Kureha Yamaguchi, Benjamin Etheridge, Andy Arditi

**Abstract**: Reasoning models generate chain-of-thought (CoT) tokens before their final output, but how this affects their vulnerability to jailbreak attacks remains unclear. While traditional language models make refusal decisions at the prompt-response boundary, we find evidence that DeepSeek-R1-Distill-Llama-8B makes these decisions within its CoT generation. We identify a linear direction in activation space during CoT token generation that predicts whether the model will refuse or comply -- termed the "caution" direction because it corresponds to cautious reasoning patterns in the generated text. Ablating this direction from model activations increases harmful compliance, effectively jailbreaking the model. We additionally show that intervening only on CoT token activations suffices to control final outputs, and that incorporating this direction into prompt-based attacks improves success rates. Our findings suggest that the chain-of-thought itself is a promising new target for adversarial manipulation in reasoning models. Code available at https://github.com/ky295/reasoning-manipulation.



## **49. Scaling Decentralized Learning with FLock**

cs.LG

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2507.15349v2) [paper-pdf](http://arxiv.org/pdf/2507.15349v2)

**Authors**: Zehua Cheng, Rui Sun, Jiahao Sun, Yike Guo

**Abstract**: Fine-tuning the large language models (LLMs) are prevented by the deficiency of centralized control and the massive computing and communication overhead on the decentralized schemes. While the typical standard federated learning (FL) supports data privacy, the central server requirement creates a single point of attack and vulnerability to poisoning attacks. Generalizing the result in this direction to 70B-parameter models in the heterogeneous, trustless environments has turned out to be a huge, yet unbroken bottleneck. This paper introduces FLock, a decentralized framework for secure and efficient collaborative LLM fine-tuning. Integrating a blockchain-based trust layer with economic incentives, FLock replaces the central aggregator with a secure, auditable protocol for cooperation among untrusted parties. We present the first empirical validation of fine-tuning a 70B LLM in a secure, multi-domain, decentralized setting. Our experiments show the FLock framework defends against backdoor poisoning attacks that compromise standard FL optimizers and fosters synergistic knowledge transfer. The resulting models show a >68% reduction in adversarial attack success rates. The global model also demonstrates superior cross-domain generalization, outperforming models trained in isolation on their own specialized data.



## **50. Cell-Free Massive MIMO-Based Physical-Layer Authentication**

eess.SP

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19931v1) [paper-pdf](http://arxiv.org/pdf/2508.19931v1)

**Authors**: Isabella W. G. da Silva, Zahra Mobini, Hien Quoc Ngo, Michail Matthaiou

**Abstract**: In this paper, we exploit the cell-free massive multiple-input multiple-output (CF-mMIMO) architecture to design a physical-layer authentication (PLA) framework that can simultaneously authenticate multiple distributed users across the coverage area. Our proposed scheme remains effective even in the presence of active adversaries attempting impersonation attacks to disrupt the authentication process. Specifically, we introduce a tag-based PLA CFmMIMO system, wherein the access points (APs) first estimate their channels with the legitimate users during an uplink training phase. Subsequently, a unique secret key is generated and securely shared between each user and the APs. We then formulate a hypothesis testing problem and derive a closed-form expression for the probability of detection for each user in the network. Numerical results validate the effectiveness of the proposed approach, demonstrating that it maintains a high detection probability even as the number of users in the system increases.



