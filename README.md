# Latest Adversarial Attack Papers
**update at 2025-12-01 09:06:56**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Attention-Guided Patch-Wise Sparse Adversarial Attacks on Vision-Language-Action Models**

cs.CV

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.21663v1) [paper-pdf](https://arxiv.org/pdf/2511.21663v1)

**Authors**: Naifu Zhang, Wei Tao, Xi Xiao, Qianpu Sun, Yuxin Zheng, Wentao Mo, Peiqiang Wang, Nan Zhang

**Abstract**: In recent years, Vision-Language-Action (VLA) models in embodied intelligence have developed rapidly. However, existing adversarial attack methods require costly end-to-end training and often generate noticeable perturbation patches. To address these limitations, we propose ADVLA, a framework that directly applies adversarial perturbations on features projected from the visual encoder into the textual feature space. ADVLA efficiently disrupts downstream action predictions under low-amplitude constraints, and attention guidance allows the perturbations to be both focused and sparse. We introduce three strategies that enhance sensitivity, enforce sparsity, and concentrate perturbations. Experiments demonstrate that under an $L_{\infty}=4/255$ constraint, ADVLA combined with Top-K masking modifies less than 10% of the patches while achieving an attack success rate of nearly 100%. The perturbations are concentrated on critical regions, remain almost imperceptible in the overall image, and a single-step iteration takes only about 0.06 seconds, significantly outperforming conventional patch-based attacks. In summary, ADVLA effectively weakens downstream action predictions of VLA models under low-amplitude and locally sparse conditions, avoiding the high training costs and conspicuous perturbations of traditional patch attacks, and demonstrates unique effectiveness and practical value for attacking VLA feature spaces.



## **2. Multimodal Robust Prompt Distillation for 3D Point Cloud Models**

cs.CV

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.21574v1) [paper-pdf](https://arxiv.org/pdf/2511.21574v1)

**Authors**: Xiang Gu, Liming Lu, Xu Zheng, Anan Du, Yongbin Zhou, Shuchao Pang

**Abstract**: Adversarial attacks pose a significant threat to learning-based 3D point cloud models, critically undermining their reliability in security-sensitive applications. Existing defense methods often suffer from (1) high computational overhead and (2) poor generalization ability across diverse attack types. To bridge these gaps, we propose a novel yet efficient teacher-student framework, namely Multimodal Robust Prompt Distillation (MRPD) for distilling robust 3D point cloud model. It learns lightweight prompts by aligning student point cloud model's features with robust embeddings from three distinct teachers: a vision model processing depth projections, a high-performance 3D model, and a text encoder. To ensure a reliable knowledge transfer, this distillation is guided by a confidence-gated mechanism which dynamically balances the contribution of all input modalities. Notably, since the distillation is all during the training stage, there is no additional computational cost at inference. Extensive experiments demonstrate that MRPD substantially outperforms state-of-the-art defense methods against a wide range of white-box and black-box attacks, while even achieving better performance on clean data. Our work presents a new, practical paradigm for building robust 3D vision systems by efficiently harnessing multimodal knowledge.



## **3. When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models**

cs.CV

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.21192v1) [paper-pdf](https://arxiv.org/pdf/2511.21192v1)

**Authors**: Hui Lu, Yi Yu, Yiming Yang, Chenyu Yi, Qixin Zhang, Bingquan Shen, Alex C. Kot, Xudong Jiang

**Abstract**: Vision-Language-Action (VLA) models are vulnerable to adversarial attacks, yet universal and transferable attacks remain underexplored, as most existing patches overfit to a single model and fail in black-box settings. To address this gap, we present a systematic study of universal, transferable adversarial patches against VLA-driven robots under unknown architectures, finetuned variants, and sim-to-real shifts. We introduce UPA-RFAS (Universal Patch Attack via Robust Feature, Attention, and Semantics), a unified framework that learns a single physical patch in a shared feature space while promoting cross-model transfer. UPA-RFAS combines (i) a feature-space objective with an $\ell_1$ deviation prior and repulsive InfoNCE loss to induce transferable representation shifts, (ii) a robustness-augmented two-phase min-max procedure where an inner loop learns invisible sample-wise perturbations and an outer loop optimizes the universal patch against this hardened neighborhood, and (iii) two VLA-specific losses: Patch Attention Dominance to hijack text$\to$vision attention and Patch Semantic Misalignment to induce image-text mismatch without labels. Experiments across diverse VLA models, manipulation suites, and physical executions show that UPA-RFAS consistently transfers across models, tasks, and viewpoints, exposing a practical patch-based attack surface and establishing a strong baseline for future defenses.



## **4. CAHS-Attack: CLIP-Aware Heuristic Search Attack Method for Stable Diffusion**

cs.CR

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.21180v1) [paper-pdf](https://arxiv.org/pdf/2511.21180v1)

**Authors**: Shuhan Xia, Jing Dai, Hui Ouyang, Yadong Shang, Dongxiao Zhao, Peipei Li

**Abstract**: Diffusion models exhibit notable fragility when faced with adversarial prompts, and strengthening attack capabilities is crucial for uncovering such vulnerabilities and building more robust generative systems. Existing works often rely on white-box access to model gradients or hand-crafted prompt engineering, which is infeasible in real-world deployments due to restricted access or poor attack effect. In this paper, we propose CAHS-Attack , a CLIP-Aware Heuristic Search attack method. CAHS-Attack integrates Monte Carlo Tree Search (MCTS) to perform fine-grained suffix optimization, leveraging a constrained genetic algorithm to preselect high-potential adversarial prompts as root nodes, and retaining the most semantically disruptive outcome at each simulation rollout for efficient local search. Extensive experiments demonstrate that our method achieves state-of-the-art attack performance across both short and long prompts of varying semantics. Furthermore, we find that the fragility of SD models can be attributed to the inherent vulnerability of their CLIP-based text encoders, suggesting a fundamental security risk in current text-to-image pipelines.



## **5. TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models**

cs.CV

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.21145v1) [paper-pdf](https://arxiv.org/pdf/2511.21145v1)

**Authors**: Jiaming He, Guanyu Hou, Hongwei Li, Zhicong Huang, Kangjie Chen, Yi Yu, Wenbo Jiang, Guowen Xu, Tianwei Zhang

**Abstract**: Text-to-Video (T2V) models are capable of synthesizing high-quality, temporally coherent dynamic video content, but the diverse generation also inherently introduces critical safety challenges. Existing safety evaluation methods,which focus on static image and text generation, are insufficient to capture the complex temporal dynamics in video generation. To address this, we propose a TEmporal-aware Automated Red-teaming framework, named TEAR, an automated framework designed to uncover safety risks specifically linked to the dynamic temporal sequencing of T2V models. TEAR employs a temporal-aware test generator optimized via a two-stage approach: initial generator training and temporal-aware online preference learning, to craft textually innocuous prompts that exploit temporal dynamics to elicit policy-violating video output. And a refine model is adopted to improve the prompt stealthiness and adversarial effectiveness cyclically. Extensive experimental evaluation demonstrates the effectiveness of TEAR across open-source and commercial T2V systems with over 80% attack success rate, a significant boost from prior best result of 57%.



## **6. Securing the Model Context Protocol (MCP): Risks, Controls, and Governance**

cs.CR

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.20920v1) [paper-pdf](https://arxiv.org/pdf/2511.20920v1)

**Authors**: Herman Errico, Jiquan Ngiam, Shanita Sojan

**Abstract**: The Model Context Protocol (MCP) replaces static, developer-controlled API integrations with more dynamic, user-driven agent systems, which also introduces new security risks. As MCP adoption grows across community servers and major platforms, organizations encounter threats that existing AI governance frameworks (such as NIST AI RMF and ISO/IEC 42001) do not yet cover in detail. We focus on three types of adversaries that take advantage of MCP s flexibility: content-injection attackers that embed malicious instructions into otherwise legitimate data; supply-chain attackers who distribute compromised servers; and agents who become unintentional adversaries by over-stepping their role. Based on early incidents and proof-of-concept attacks, we describe how MCP can increase the attack surface through data-driven exfiltration, tool poisoning, and cross-system privilege escalation. In response, we propose a set of practical controls, including per-user authentication with scoped authorization, provenance tracking across agent workflows, containerized sandboxing with input/output checks, inline policy enforcement with DLP and anomaly detection, and centralized governance using private registries or gateway layers. The aim is to help organizations ensure that unvetted code does not run outside a sandbox, tools are not used beyond their intended scope, data exfiltration attempts are detectable, and actions can be audited end-to-end. We close by outlining open research questions around verifiable registries, formal methods for these dynamic systems, and privacy-preserving agent operations.



## **7. Adversarial Confusion Attack: Disrupting Multimodal Large Language Models**

cs.CL

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.20494v1) [paper-pdf](https://arxiv.org/pdf/2511.20494v1)

**Authors**: Jakub Hoscilowicz, Artur Janicki

**Abstract**: We introduce the Adversarial Confusion Attack, a new class of threats against multimodal large language models (MLLMs). Unlike jailbreaks or targeted misclassification, the goal is to induce systematic disruption that makes the model generate incoherent or confidently incorrect outputs. Applications include embedding adversarial images into websites to prevent MLLM-powered agents from operating reliably. The proposed attack maximizes next-token entropy using a small ensemble of open-source MLLMs. In the white-box setting, we show that a single adversarial image can disrupt all models in the ensemble, both in the full-image and adversarial CAPTCHA settings. Despite relying on a basic adversarial technique (PGD), the attack generates perturbations that transfer to both unseen open-source (e.g., Qwen3-VL) and proprietary (e.g., GPT-5.1) models.



## **8. Ranking-Enhanced Anomaly Detection Using Active Learning-Assisted Attention Adversarial Dual AutoEncoders**

cs.LG

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.20480v1) [paper-pdf](https://arxiv.org/pdf/2511.20480v1)

**Authors**: Sidahmed Benabderrahmane, James Cheney, Talal Rahwan

**Abstract**: Advanced Persistent Threats (APTs) pose a significant challenge in cybersecurity due to their stealthy and long-term nature. Modern supervised learning methods require extensive labeled data, which is often scarce in real-world cybersecurity environments. In this paper, we propose an innovative approach that leverages AutoEncoders for unsupervised anomaly detection, augmented by active learning to iteratively improve the detection of APT anomalies. By selectively querying an oracle for labels on uncertain or ambiguous samples, we minimize labeling costs while improving detection rates, enabling the model to improve its detection accuracy with minimal data while reducing the need for extensive manual labeling. We provide a detailed formulation of the proposed Attention Adversarial Dual AutoEncoder-based anomaly detection framework and show how the active learning loop iteratively enhances the model. The framework is evaluated on real-world imbalanced provenance trace databases produced by the DARPA Transparent Computing program, where APT-like attacks constitute as little as 0.004\% of the data. The datasets span multiple operating systems, including Android, Linux, BSD, and Windows, and cover two attack scenarios. The results have shown significant improvements in detection rates during active learning and better performance compared to other existing approaches.



## **9. Towards Trustworthy Wi-Fi Sensing: Systematic Evaluation of Deep Learning Model Robustness to Adversarial Attacks**

cs.LG

19 pages, 8 figures, 7 tables

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.20456v1) [paper-pdf](https://arxiv.org/pdf/2511.20456v1)

**Authors**: Shreevanth Krishnaa Gopalakrishnan, Stephen Hailes

**Abstract**: Machine learning has become integral to Channel State Information (CSI)-based human sensing systems and is expected to power applications such as device-free activity recognition and identity detection in future cellular and Wi-Fi generations. However, these systems rely on models whose decisions can be subtly perturbed, raising concerns for security and reliability in ubiquitous sensing. Quantifying and understanding the robustness of such models, defined as their ability to maintain accurate predictions under adversarial perturbations, is therefore critical before wireless sensing can be safely deployed in real-world environments.   This work presents a systematic evaluation of the robustness of CSI deep learning models under diverse threat models (white-box, black-box/transfer, and universal perturbations) and varying degrees of attack realism. We establish a framework to compare compact temporal autoencoder models with larger deep architectures across three public datasets, quantifying how model scale, training regime, and physical constraints influence robustness. Our experiments show that smaller models, while efficient and equally performant on clean data, are markedly less robust. We further confirm that physically realizable signal-space perturbations, designed to be feasible in real wireless channels, significantly reduce attack success compared to unconstrained feature-space attacks. Adversarial training mitigates these vulnerabilities, improving mean robust accuracy with only moderate degradation in clean performance across both model classes. As wireless sensing advances towards reliable, cross-domain operation, these findings provide quantitative baselines for robustness estimation and inform design principles for secure and trustworthy human-centered sensing systems.



## **10. V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs**

cs.CV

21 pages

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.20223v1) [paper-pdf](https://arxiv.org/pdf/2511.20223v1)

**Authors**: Sen Nie, Jie Zhang, Jianxin Yan, Shiguang Shan, Xilin Chen

**Abstract**: Adversarial attacks have evolved from simply disrupting predictions on conventional task-specific models to the more complex goal of manipulating image semantics on Large Vision-Language Models (LVLMs). However, existing methods struggle with controllability and fail to precisely manipulate the semantics of specific concepts in the image. We attribute this limitation to semantic entanglement in the patch-token representations on which adversarial attacks typically operate: global context aggregated by self-attention in the vision encoder dominates individual patch features, making them unreliable handles for precise local semantic manipulation. Our systematic investigation reveals a key insight: value features (V) computed within the transformer attention block serve as much more precise handles for manipulation. We show that V suppresses global-context channels, allowing it to retain high-entropy, disentangled local semantic information. Building on this discovery, we propose V-Attack, a novel method designed for precise local semantic attacks. V-Attack targets the value features and introduces two core components: (1) a Self-Value Enhancement module to refine V's intrinsic semantic richness, and (2) a Text-Guided Value Manipulation module that leverages text prompts to locate source concept and optimize it toward a target concept. By bypassing the entangled patch features, V-Attack achieves highly effective semantic control. Extensive experiments across diverse LVLMs, including LLaVA, InternVL, DeepseekVL and GPT-4o, show that V-Attack improves the attack success rate by an average of 36% over state-of-the-art methods, exposing critical vulnerabilities in modern visual-language understanding. Our code and data are available https://github.com/Summu77/V-Attack.



## **11. On the Feasibility of Hijacking MLLMs' Decision Chain via One Perturbation**

cs.CV

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.20002v1) [paper-pdf](https://arxiv.org/pdf/2511.20002v1)

**Authors**: Changyue Li, Jiaying Li, Youliang Yuan, Jiaming He, Zhicong Huang, Pinjia He

**Abstract**: Conventional adversarial attacks focus on manipulating a single decision of neural networks. However, real-world models often operate in a sequence of decisions, where an isolated mistake can be easily corrected, but cascading errors can lead to severe risks.   This paper reveals a novel threat: a single perturbation can hijack the whole decision chain. We demonstrate the feasibility of manipulating a model's outputs toward multiple, predefined outcomes, such as simultaneously misclassifying "non-motorized lane" signs as "motorized lane" and "pedestrian" as "plastic bag".   To expose this threat, we introduce Semantic-Aware Universal Perturbations (SAUPs), which induce varied outcomes based on the semantics of the inputs. We overcome optimization challenges by developing an effective algorithm, which searches for perturbations in normalized space with a semantic separation strategy. To evaluate the practical threat of SAUPs, we present RIST, a new real-world image dataset with fine-grained semantic annotations. Extensive experiments on three multimodal large language models demonstrate their vulnerability, achieving a 70% attack success rate when controlling five distinct targets using just an adversarial frame.



## **12. Continual Audio Deepfake Detection via Universal Adversarial Perturbation**

cs.SD

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.19974v1) [paper-pdf](https://arxiv.org/pdf/2511.19974v1)

**Authors**: Wangjie Li, Lin Li, Qingyang Hong

**Abstract**: The rapid advancement of speech synthesis and voice conversion technologies has raised significant security concerns in multimedia forensics. Although current detection models demonstrate impressive performance, they struggle to maintain effectiveness against constantly evolving deepfake attacks. Additionally, continually fine-tuning these models using historical training data incurs substantial computational and storage costs. To address these limitations, we propose a novel framework that incorporates Universal Adversarial Perturbation (UAP) into audio deepfake detection, enabling models to retain knowledge of historical spoofing distribution without direct access to past data. Our method integrates UAP seamlessly with pre-trained self-supervised audio models during fine-tuning. Extensive experiments validate the effectiveness of our approach, showcasing its potential as an efficient solution for continual learning in audio deepfake detection.



## **13. Multi-Hypotheses Ego-Tracking for Resilient Navigation**

eess.SY

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.19770v2) [paper-pdf](https://arxiv.org/pdf/2511.19770v2)

**Authors**: Peter Iwer Hoedt Karstensen, Roberto Galeazzi

**Abstract**: Autonomous robots relying on radio frequency (RF)-based localization such as global navigation satellite system (GNSS), ultra-wide band (UWB), and 5G integrated sensing and communication (ISAC) are vulnerable to spoofing and sensor manipulation. This paper presents a resilient navigation architecture that combines multi-hypothesis estimation with a Poisson binomial windowed-count detector for anomaly identification and isolation. A state machine coordinates transitions between operation, diagnosis, and mitigation, enabling adaptive response to adversarial conditions. When attacks are detected, trajectory re-planning based on differential flatness allows information-gathering maneuvers minimizing performance loss. Case studies demonstrate effective detection of biased sensors, maintenance of state estimation, and recovery of nominal operation under persistent spoofing attacks



## **14. Are Neuro-Inspired Multi-Modal Vision-Language Models Resilient to Membership Inference Privacy Leakage?**

cs.CV

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.20710v1) [paper-pdf](https://arxiv.org/pdf/2511.20710v1)

**Authors**: David Amebley, Sayanton Dibbo

**Abstract**: In the age of agentic AI, the growing deployment of multi-modal models (MMs) has introduced new attack vectors that can leak sensitive training data in MMs, causing privacy leakage. This paper investigates a black-box privacy attack, i.e., membership inference attack (MIA) on multi-modal vision-language models (VLMs). State-of-the-art research analyzes privacy attacks primarily to unimodal AI-ML systems, while recent studies indicate MMs can also be vulnerable to privacy attacks. While researchers have demonstrated that biologically inspired neural network representations can improve unimodal model resilience against adversarial attacks, it remains unexplored whether neuro-inspired MMs are resilient against privacy attacks. In this work, we introduce a systematic neuroscience-inspired topological regularization (tau) framework to analyze MM VLMs resilience against image-text-based inference privacy attacks. We examine this phenomenon using three VLMs: BLIP, PaliGemma 2, and ViT-GPT2, across three benchmark datasets: COCO, CC3M, and NoCaps. Our experiments compare the resilience of baseline and neuro VLMs (with topological regularization), where the tau > 0 configuration defines the NEURO variant of VLM. Our results on the BLIP model using the COCO dataset illustrate that MIA attack success in NEURO VLMs drops by 24% mean ROC-AUC, while achieving similar model utility (similarities between generated and reference captions) in terms of MPNet and ROUGE-2 metrics. This shows neuro VLMs are comparatively more resilient against privacy attacks, while not significantly compromising model utility. Our extensive evaluation with PaliGemma 2 and ViT-GPT2 models, on two additional datasets: CC3M and NoCaps, further validates the consistency of the findings. This work contributes to the growing understanding of privacy risks in MMs and provides evidence on neuro VLMs privacy threat resilience.



## **15. Synthetic Data: AI's New Weapon Against Android Malware**

cs.CR

23 pages, 18 figures, 8 tables. Accepted for publication at the JBCS

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19649v1) [paper-pdf](https://arxiv.org/pdf/2511.19649v1)

**Authors**: Angelo Gaspar Diniz Nogueira, Kayua Oleques Paim, Hendrio Bragança, Rodrigo Brandão Mansilha, Diego Kreutz

**Abstract**: The ever-increasing number of Android devices and the accelerated evolution of malware, reaching over 35 million samples by 2024, highlight the critical importance of effective detection methods. Attackers are now using Artificial Intelligence to create sophisticated malware variations that can easily evade traditional detection techniques. Although machine learning has shown promise in malware classification, its success relies heavily on the availability of up-to-date, high-quality datasets. The scarcity and high cost of obtaining and labeling real malware samples presents significant challenges in developing robust detection models. In this paper, we propose MalSynGen, a Malware Synthetic Data Generation methodology that uses a conditional Generative Adversarial Network (cGAN) to generate synthetic tabular data. This data preserves the statistical properties of real-world data and improves the performance of Android malware classifiers. We evaluated the effectiveness of this approach using various datasets and metrics that assess the fidelity of the generated data, its utility in classification, and the computational efficiency of the process. Our experiments demonstrate that MalSynGen can generalize across different datasets, providing a viable solution to address the issues of obsolescence and low quality data in malware detection.



## **16. Targeted Manipulation: Slope-Based Attacks on Financial Time-Series Data**

cs.LG

13 pages, 6 figures, 4 tables, preprint; Total including Appendix: 21 pages, 11 figures, 7 tables

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19330v1) [paper-pdf](https://arxiv.org/pdf/2511.19330v1)

**Authors**: Dominik Luszczynski

**Abstract**: A common method of attacking deep learning models is through adversarial attacks, which occur when an attacker specifically modifies the input of a model to produce an incorrect result. Adversarial attacks have been deeply investigated in the image domain; however, there is less research in the time-series domain and very little for forecasting financial data. To address these concerns, this study aims to build upon previous research on adversarial attacks for time-series data by introducing two new slope-based methods aimed to alter the trends of the predicted stock forecast generated by an N-HiTS model. Compared to the normal N-HiTS predictions, the two new slope-based methods, the General Slope Attack and Least-Squares Slope Attack, can manipulate N-HiTS predictions by doubling the slope. These new slope attacks can bypass standard security mechanisms, such as a discriminator that filters real and perturbed inputs, reducing a 4-layered CNN's specificity to 28% and accuracy to 57%. Furthermore, the slope based methods were incorporated into a GAN architecture as a means of generating realistic synthetic data, while simultaneously fooling the model. Finally, this paper also proposes a sample malware designed to inject an adversarial attack in the model inference library, proving that ML-security research should not only focus on making the model safe, but also securing the entire pipeline.



## **17. Medusa: Cross-Modal Transferable Adversarial Attacks on Multimodal Medical Retrieval-Augmented Generation**

cs.CR

Accepted at KDD 2026 First Cycle (full version). Authors marked with * contributed equally. Yi Liu is the lead author

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19257v1) [paper-pdf](https://arxiv.org/pdf/2511.19257v1)

**Authors**: Yingjia Shang, Yi Liu, Huimin Wang, Furong Li, Wenfang Sun, Wu Chengyu, Yefeng Zheng

**Abstract**: With the rapid advancement of retrieval-augmented vision-language models, multimodal medical retrieval-augmented generation (MMed-RAG) systems are increasingly adopted in clinical decision support. These systems enhance medical applications by performing cross-modal retrieval to integrate relevant visual and textual evidence for tasks, e.g., report generation and disease diagnosis. However, their complex architecture also introduces underexplored adversarial vulnerabilities, particularly via visual input perturbations. In this paper, we propose Medusa, a novel framework for crafting cross-modal transferable adversarial attacks on MMed-RAG systems under a black-box setting. Specifically, Medusa formulates the attack as a perturbation optimization problem, leveraging a multi-positive InfoNCE loss (MPIL) to align adversarial visual embeddings with medically plausible but malicious textual targets, thereby hijacking the retrieval process. To enhance transferability, we adopt a surrogate model ensemble and design a dual-loop optimization strategy augmented with invariant risk minimization (IRM). Extensive experiments on two real-world medical tasks, including medical report generation and disease diagnosis, demonstrate that Medusa achieves over 90% average attack success rate across various generation models and retrievers under appropriate parameter configuration, while remaining robust against four mainstream defenses, outperforming state-of-the-art baselines. Our results reveal critical vulnerabilities in the MMed-RAG systems and highlight the necessity of robustness benchmarking in safety-critical medical applications. The code and data are available at https://anonymous.4open.science/r/MMed-RAG-Attack-F05A.



## **18. Adversarial Patch Attacks on Vision-Based Cargo Occupancy Estimation via Differentiable 3D Simulation**

cs.CV

9 pages, 5 figures, 1 algorithm

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19254v1) [paper-pdf](https://arxiv.org/pdf/2511.19254v1)

**Authors**: Mohamed Rissal Hedna, Sesugh Samuel Nder

**Abstract**: Computer vision systems are increasingly adopted in modern logistics operations, including the estimation of trailer occupancy for planning, routing, and billing. Although effective, such systems may be vulnerable to physical adversarial attacks, particularly adversarial patches that can be printed and placed on interior surfaces. In this work, we study the feasibility of such attacks on a convolutional cargo-occupancy classifier using fully simulated 3D environments. Using Mitsuba 3 for differentiable rendering, we optimize patch textures across variations in geometry, lighting, and viewpoint, and compare their effectiveness to a 2D compositing baseline. Our experiments demonstrate that 3D-optimized patches achieve high attack success rates, especially in a denial-of-service scenario (empty to full), where success reaches 84.94 percent. Concealment attacks (full to empty) prove more challenging but still reach 30.32 percent. We analyze the factors influencing attack success, discuss implications for the security of automated logistics pipelines, and highlight directions for strengthening physical robustness. To our knowledge, this is the first study to investigate adversarial patch attacks for cargo-occupancy estimation in physically realistic, fully simulated 3D scenes.



## **19. FedPoisonTTP: A Threat Model and Poisoning Attack for Federated Test-Time Personalization**

cs.CR

13 pages, 3 figures, 2 tables

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19248v1) [paper-pdf](https://arxiv.org/pdf/2511.19248v1)

**Authors**: Md Akil Raihan Iftee, Syed Md. Ahnaf Hasan, Amin Ahsan Ali, AKM Mahbubur Rahman, Sajib Mistry, Aneesh Krishna

**Abstract**: Test-time personalization in federated learning enables models at clients to adjust online to local domain shifts, enhancing robustness and personalization in deployment. Yet, existing federated learning work largely overlooks the security risks that arise when local adaptation occurs at test time. Heterogeneous domain arrivals, diverse adaptation algorithms, and limited cross-client visibility create vulnerabilities where compromised participants can craft poisoned inputs and submit adversarial updates that undermine both global and per-client performance. To address this threat, we introduce FedPoisonTTP, a realistic grey-box attack framework that explores test-time data poisoning in the federated adaptation setting. FedPoisonTTP distills a surrogate model from adversarial queries, synthesizes in-distribution poisons using feature-consistency, and optimizes attack objectives to generate high-entropy or class-confident poisons that evade common adaptation filters. These poisons are injected during local adaptation and spread through collaborative updates, leading to broad degradation. Extensive experiments on corrupted vision benchmarks show that compromised participants can substantially diminish overall test-time performance.



## **20. Adversarial Attack-Defense Co-Evolution for LLM Safety Alignment via Tree-Group Dual-Aware Search and Optimization**

cs.CR

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.19218v2) [paper-pdf](https://arxiv.org/pdf/2511.19218v2)

**Authors**: Xurui Li, Kaisong Song, Rui Zhu, Pin-Yu Chen, Haixu Tang

**Abstract**: Large Language Models (LLMs) have developed rapidly in web services, delivering unprecedented capabilities while amplifying societal risks. Existing works tend to focus on either isolated jailbreak attacks or static defenses, neglecting the dynamic interplay between evolving threats and safeguards in real-world web contexts. To mitigate these challenges, we propose ACE-Safety (Adversarial Co-Evolution for LLM Safety), a novel framework that jointly optimize attack and defense models by seamlessly integrating two key innovative procedures: (1) Group-aware Strategy-guided Monte Carlo Tree Search (GS-MCTS), which efficiently explores jailbreak strategies to uncover vulnerabilities and generate diverse adversarial samples; (2) Adversarial Curriculum Tree-aware Group Policy Optimization (AC-TGPO), which jointly trains attack and defense LLMs with challenging samples via curriculum reinforcement learning, enabling robust mutual improvement. Evaluations across multiple benchmarks demonstrate that our method outperforms existing attack and defense approaches, and provides a feasible pathway for developing LLMs that can sustainably support responsible AI ecosystems.



## **21. Learning to Compress Graphs via Dual Agents for Consistent Topological Robustness Evaluation**

cs.LG

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.18958v2) [paper-pdf](https://arxiv.org/pdf/2511.18958v2)

**Authors**: Qisen Chai, Yansong Wang, Junjie Huang, Tao Jia

**Abstract**: As graph-structured data grow increasingly large, evaluating their robustness under adversarial attacks becomes computationally expensive and difficult to scale. To address this challenge, we propose to compress graphs into compact representations that preserve both topological structure and robustness profile, enabling efficient and reliable evaluation. We propose Cutter, a dual-agent reinforcement learning framework composed of a Vital Detection Agent (VDA) and a Redundancy Detection Agent (RDA), which collaboratively identify structurally vital and redundant nodes for guided compression. Cutter incorporates three key strategies to enhance learning efficiency and compression quality: trajectory-level reward shaping to transform sparse trajectory returns into dense, policy-equivalent learning signals; prototype-based shaping to guide decisions using behavioral patterns from both high- and low-return trajectories; and cross-agent imitation to enable safer and more transferable exploration. Experiments on multiple real-world graphs demonstrate that Cutter generates compressed graphs that retain essential static topological properties and exhibit robustness degradation trends highly consistent with the original graphs under various attack scenarios, thereby significantly improving evaluation efficiency without compromising assessment fidelity.



## **22. Defending Large Language Models Against Jailbreak Exploits with Responsible AI Considerations**

cs.CR

20 pages including appendix; technical report; NeurIPS 2024 style

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.18933v1) [paper-pdf](https://arxiv.org/pdf/2511.18933v1)

**Authors**: Ryan Wong, Hosea David Yu Fei Ng, Dhananjai Sharma, Glenn Jun Jie Ng, Kavishvaran Srinivasan

**Abstract**: Large Language Models (LLMs) remain susceptible to jailbreak exploits that bypass safety filters and induce harmful or unethical behavior. This work presents a systematic taxonomy of existing jailbreak defenses across prompt-level, model-level, and training-time interventions, followed by three proposed defense strategies. First, a Prompt-Level Defense Framework detects and neutralizes adversarial inputs through sanitization, paraphrasing, and adaptive system guarding. Second, a Logit-Based Steering Defense reinforces refusal behavior through inference-time vector steering in safety-sensitive layers. Third, a Domain-Specific Agent Defense employs the MetaGPT framework to enforce structured, role-based collaboration and domain adherence. Experiments on benchmark datasets show substantial reductions in attack success rate, achieving full mitigation under the agent-based defense. Overall, this study highlights how jailbreaks pose a significant security threat to LLMs and identifies key intervention points for prevention, while noting that defense strategies often involve trade-offs between safety, performance, and scalability. Code is available at: https://github.com/Kuro0911/CS5446-Project



## **23. BackdoorVLM: A Benchmark for Backdoor Attacks on Vision-Language Models**

cs.CV

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.18921v1) [paper-pdf](https://arxiv.org/pdf/2511.18921v1)

**Authors**: Juncheng Li, Yige Li, Hanxun Huang, Yunhao Chen, Xin Wang, Yixu Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Backdoor attacks undermine the reliability and trustworthiness of machine learning systems by injecting hidden behaviors that can be maliciously activated at inference time. While such threats have been extensively studied in unimodal settings, their impact on multimodal foundation models, particularly vision-language models (VLMs), remains largely underexplored. In this work, we introduce \textbf{BackdoorVLM}, the first comprehensive benchmark for systematically evaluating backdoor attacks on VLMs across a broad range of settings. It adopts a unified perspective that injects and analyzes backdoors across core vision-language tasks, including image captioning and visual question answering. BackdoorVLM organizes multimodal backdoor threats into 5 representative categories: targeted refusal, malicious injection, jailbreak, concept substitution, and perceptual hijack. Each category captures a distinct pathway through which an adversary can manipulate a model's behavior. We evaluate these threats using 12 representative attack methods spanning text, image, and bimodal triggers, tested on 2 open-source VLMs and 3 multimodal datasets. Our analysis reveals that VLMs exhibit strong sensitivity to textual instructions, and in bimodal backdoors the text trigger typically overwhelms the image trigger when forming the backdoor mapping. Notably, backdoors involving the textual modality remain highly potent, with poisoning rates as low as 1\% yielding over 90\% success across most tasks. These findings highlight significant, previously underexplored vulnerabilities in current VLMs. We hope that BackdoorVLM can serve as a useful benchmark for analyzing and mitigating multimodal backdoor threats. Code is available at: https://github.com/bin015/BackdoorVLM .



## **24. EAGER: Edge-Aligned LLM Defense for Robust, Efficient, and Accurate Cybersecurity Question Answering**

cs.CR

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19523v1) [paper-pdf](https://arxiv.org/pdf/2511.19523v1)

**Authors**: Onat Gungor, Roshan Sood, Jiasheng Zhou, Tajana Rosing

**Abstract**: Large Language Models (LLMs) are highly effective for cybersecurity question answering (QA) but are difficult to deploy on edge devices due to their size. Quantization reduces memory and compute requirements but often degrades accuracy and increases vulnerability to adversarial attacks. We present EAGER, an edge-aligned defense framework that integrates parameter-efficient quantization with domain-specific preference alignment to jointly optimize efficiency, robustness, and accuracy. Unlike prior methods that address these aspects separately, EAGER leverages Quantized Low-Rank Adaptation (QLoRA) for low-cost fine-tuning and Direct Preference Optimization (DPO) on a self-constructed cybersecurity preference dataset, eliminating the need for human labels. Experiments show that EAGER reduces adversarial attack success rates by up to 7.3x and improves QA accuracy by up to 55% over state-of-the-art defenses, while achieving the lowest response latency on a Jetson Orin, demonstrating its practical edge deployment.



## **25. Now You See It, Now You Don't - Instant Concept Erasure for Safe Text-to-Image and Video Generation**

cs.CV

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.18684v1) [paper-pdf](https://arxiv.org/pdf/2511.18684v1)

**Authors**: Shristi Das Biswas, Arani Roy, Kaushik Roy

**Abstract**: Robust concept removal for text-to-image (T2I) and text-to-video (T2V) models is essential for their safe deployment. Existing methods, however, suffer from costly retraining, inference overhead, or vulnerability to adversarial attacks. Crucially, they rarely model the latent semantic overlap between the target erase concept and surrounding content -- causing collateral damage post-erasure -- and even fewer methods work reliably across both T2I and T2V domains. We introduce Instant Concept Erasure (ICE), a training-free, modality-agnostic, one-shot weight modification approach that achieves precise, persistent unlearning with zero overhead. ICE defines erase and preserve subspaces using anisotropic energy-weighted scaling, then explicitly regularises against their intersection using a unique, closed-form overlap projector. We pose a convex and Lipschitz-bounded Spectral Unlearning Objective, balancing erasure fidelity and intersection preservation, that admits a stable and unique analytical solution. This solution defines a dissociation operator that is translated to the model's text-conditioning layers, making the edit permanent and runtime-free. Across targeted removals of artistic styles, objects, identities, and explicit content, ICE efficiently achieves strong erasure with improved robustness to red-teaming, all while causing only minimal degradation of original generative abilities in both T2I and T2V models.



## **26. Robust Physical Adversarial Patches Using Dynamically Optimized Clusters**

cs.CV

Supplementary material available at: https://drive.google.com/drive/folders/1Yntcc9CARdbvoJJ51cyUm1DWGSvU9X4V?usp=drive_link

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2511.18656v1) [paper-pdf](https://arxiv.org/pdf/2511.18656v1)

**Authors**: Harrison Bagley, Will Meakin, Simon Lucey, Yee Wei Law, Tat-Jun Chin

**Abstract**: Physical adversarial attacks on deep learning systems is concerning due to the ease of deploying such attacks, usually by placing an adversarial patch in a scene to manipulate the outcomes of a deep learning model. Training such patches typically requires regularization that improves physical realizability (e.g., printability, smoothness) and/or robustness to real-world variability (e.g. deformations, viewing angle, noise). One type of variability that has received little attention is scale variability. When a patch is rescaled, either digitally through downsampling/upsampling or physically through changing imaging distances, interpolation-induced color mixing occurs. This smooths out pixel values, resulting in a loss of high-frequency patterns and degrading the adversarial signal. To address this, we present a novel superpixel-based regularization method that guides patch optimization to scale-resilient structures. Our ap proach employs the Simple Linear Iterative Clustering (SLIC) algorithm to dynamically cluster pixels in an adversarial patch during optimization. The Implicit Function Theorem is used to backpropagate gradients through SLIC to update the superpixel boundaries and color. This produces patches that maintain their structure over scale and are less susceptible to interpolation losses. Our method achieves greater performance in the digital domain, and when realized physically, these performance gains are preserved, leading to improved physical performance. Real-world performance was objectively assessed using a novel physical evaluation protocol that utilizes screens and cardboard cut-outs to systematically vary real-world conditions.



## **27. Algorithmic detection of false data injection attacks in cyber-physical systems**

math.OC

13 pages, 6 figures

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2511.18588v1) [paper-pdf](https://arxiv.org/pdf/2511.18588v1)

**Authors**: Souvik Das, Avishek Ghosh, Debasish Chatterjee

**Abstract**: This article introduces an anomaly detection based algorithm (AD-CPS) to detect false data injection attacks that fall under the category of data deception/integrity attacks, but with arbitrary information structure, in cyber-physical systems (CPSs) modeled as stochastic linear time-invariant systems. The core idea of this data-driven algorithm is based on the fact that an honest state (one not compromised by adversaries) generated by the CPS should concentrate near its weighted empirical mean of the immediate past samples. As the first theoretical result, we provide non-asymptotic guarantees on the false positive error incurred by the algorithm for attacks that are 2-step honest, referring to adversaries that act intermittently rather than successively. Moreover, we establish that for adversaries possessing a certain minimum energy, the false negative error incurred by AD-CPS is low. Extensive experiments were conducted on partially observed stochastic LTI systems to demonstrate these properties and to quantitatively compare AD-CPS with an optimal CUSUM-based test.



## **28. Future-Back Threat Modeling: A Foresight-Driven Security Framework**

cs.CR

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.16088v2) [paper-pdf](https://arxiv.org/pdf/2511.16088v2)

**Authors**: Vu Van Than

**Abstract**: Traditional threat modeling remains reactive-focused on known TTPs and past incident data, while threat prediction and forecasting frameworks are often disconnected from operational or architectural artifacts. This creates a fundamental weakness: the most serious cyber threats often do not arise from what is known, but from what is assumed, overlooked, or not yet conceived, and frequently originate from the future, such as artificial intelligence, information warfare, and supply chain attacks, where adversaries continuously develop new exploits that can bypass defenses built on current knowledge. To address this mental gap, this paper introduces the theory and methodology of Future-Back Threat Modeling (FBTM). This predictive approach begins with envisioned future threat states and works backward to identify assumptions, gaps, blind spots, and vulnerabilities in the current defense architecture, providing a clearer and more accurate view of impending threats so that we can anticipate their emergence and shape the future we want through actions taken now. The proposed methodology further aims to reveal known unknowns and unknown unknowns, including tactics, techniques, and procedures that are emerging, anticipated, and plausible. This enhances the predictability of adversary behavior, particularly under future uncertainty, helping security leaders make informed decisions today that shape more resilient security postures for the future.



## **29. Critical Evaluation of Quantum Machine Learning for Adversarial Robustness**

cs.CR

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.14989v2) [paper-pdf](https://arxiv.org/pdf/2511.14989v2)

**Authors**: Saeefa Rubaiyet Nowmi, Jesus Lopez, Md Mahmudul Alam Imon, Shahrooz Pouryousef, Mohammad Saidur Rahman

**Abstract**: Quantum Machine Learning (QML) integrates quantum computational principles into learning algorithms, offering improved representational capacity and computational efficiency. Nevertheless, the security and robustness of QML systems remain underexplored, especially under adversarial conditions. In this paper, we present a systematization of adversarial robustness in QML, integrating conceptual organization with empirical evaluation across three threat models-black-box, gray-box, and white-box. We implement representative attacks in each category, including label-flipping for black-box, QUID encoder-level data poisoning for gray-box, and FGSM and PGD for white-box, using Quantum Neural Networks (QNNs) trained on two datasets from distinct domains: MNIST from computer vision and AZ-Class from Android malware, across multiple circuit depths (2, 5, 10, and 50 layers) and two encoding schemes (angle and amplitude). Our evaluation shows that amplitude encoding yields the highest clean accuracy (93% on MNIST and 67% on AZ-Class) in deep, noiseless circuits; however, it degrades sharply under adversarial perturbations and depolarization noise (p=0.01), dropping accuracy below 5%. In contrast, angle encoding, while offering lower representational capacity, remains more stable in shallow, noisy regimes, revealing a trade-off between capacity and robustness. Moreover, the QUID attack attains higher attack success rates, though quantum noise channels disrupt the Hilbert-space correlations it exploits, weakening its impact in image domains. This suggests that noise can act as a natural defense mechanism in Noisy Intermediate-Scale Quantum (NISQ) systems. Overall, our findings guide the development of secure and resilient QML architectures for practical deployment. These insights underscore the importance of designing threat-aware models that remain reliable under real-world noise in NISQ settings.



## **30. Cheating Stereo Matching in Full-scale: Physical Adversarial Attack against Binocular Depth Estimation in Autonomous Driving**

cs.CV

AAAI 2026

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.14386v3) [paper-pdf](https://arxiv.org/pdf/2511.14386v3)

**Authors**: Kangqiao Zhao, Shuo Huai, Xurui Song, Jun Luo

**Abstract**: Though deep neural models adopted to realize the perception of autonomous driving have proven vulnerable to adversarial examples, known attacks often leverage 2D patches and target mostly monocular perception. Therefore, the effectiveness of Physical Adversarial Examples (PAEs) on stereo-based binocular depth estimation remains largely unexplored. To this end, we propose the first texture-enabled physical adversarial attack against stereo matching models in the context of autonomous driving. Our method employs a 3D PAE with global camouflage texture rather than a local 2D patch-based one, ensuring both visual consistency and attack effectiveness across different viewpoints of stereo cameras. To cope with the disparity effect of these cameras, we also propose a new 3D stereo matching rendering module that allows the PAE to be aligned with real-world positions and headings in binocular vision. We further propose a novel merging attack that seamlessly blends the target into the environment through fine-grained PAE optimization. It has significantly enhanced stealth and lethality upon existing hiding attacks that fail to get seamlessly merged into the background. Extensive evaluations show that our PAEs can successfully fool the stereo models into producing erroneous depth information.



## **31. Steganographic Backdoor Attacks in NLP: Ultra-Low Poisoning and Defense Evasion**

cs.CR

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.14301v2) [paper-pdf](https://arxiv.org/pdf/2511.14301v2)

**Authors**: Eric Xue, Ruiyi Zhang, Zijun Zhang, Pengtao Xie

**Abstract**: Transformer models are foundational to natural language processing (NLP) applications, yet remain vulnerable to backdoor attacks introduced through poisoned data, which implant hidden behaviors during training. To strengthen the ability to prevent such compromises, recent research has focused on designing increasingly stealthy attacks to stress-test existing defenses, pairing backdoor behaviors with stylized artifact or token-level perturbation triggers. However, this trend diverts attention from the harder and more realistic case: making the model respond to semantic triggers such as specific names or entities, where a successful backdoor could manipulate outputs tied to real people or events in deployed systems. Motivated by this growing disconnect, we introduce SteganoBackdoor, bringing stealth techniques back into line with practical threat models. Leveraging innocuous properties from natural-language steganography, SteganoBackdoor applies a gradient-guided data optimization process to transform semantic trigger seeds into steganographic carriers that embed a high backdoor payload, remain fluent, and exhibit no representational resemblance to the trigger. Across diverse experimental settings, SteganoBackdoor achieves over 99% attack success at an order-of-magnitude lower data-poisoning rate than prior approaches while maintaining unparalleled evasion against a comprehensive suite of data-level defenses. By revealing this practical and covert attack, SteganoBackdoor highlights an urgent blind spot in current defenses and demands immediate attention to adversarial data defenses and real-world threat modeling.



## **32. Privacy on the Fly: A Predictive Adversarial Transformation Network for Mobile Sensor Data**

cs.CR

accepted by AAAI 2026 (oral)

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.07242v4) [paper-pdf](https://arxiv.org/pdf/2511.07242v4)

**Authors**: Tianle Song, Chenhao Lin, Yang Cao, Zhengyu Zhao, Jiahao Sun, Chong Zhang, Le Yang, Chao Shen

**Abstract**: Mobile motion sensors such as accelerometers and gyroscopes are now ubiquitously accessible by third-party apps via standard APIs. While enabling rich functionalities like activity recognition and step counting, this openness has also enabled unregulated inference of sensitive user traits, such as gender, age, and even identity, without user consent. Existing privacy-preserving techniques, such as GAN-based obfuscation or differential privacy, typically require access to the full input sequence, introducing latency that is incompatible with real-time scenarios. Worse, they tend to distort temporal and semantic patterns, degrading the utility of the data for benign tasks like activity recognition. To address these limitations, we propose the Predictive Adversarial Transformation Network (PATN), a real-time privacy-preserving framework that leverages historical signals to generate adversarial perturbations proactively. The perturbations are applied immediately upon data acquisition, enabling continuous protection without disrupting application functionality. Experiments on two datasets demonstrate that PATN substantially degrades the performance of privacy inference models, achieving Attack Success Rate (ASR) of 40.11% and 44.65% (reducing inference accuracy to near-random) and increasing the Equal Error Rate (EER) from 8.30% and 7.56% to 41.65% and 46.22%. On ASR, PATN outperforms baseline methods by 16.16% and 31.96%, respectively.



## **33. CGCE: Classifier-Guided Concept Erasure in Generative Models**

cs.CV

26 pages, 17 figures

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.05865v2) [paper-pdf](https://arxiv.org/pdf/2511.05865v2)

**Authors**: Viet Nguyen, Vishal M. Patel

**Abstract**: Recent advancements in large-scale generative models have enabled the creation of high-quality images and videos, but have also raised significant safety concerns regarding the generation of unsafe content. To mitigate this, concept erasure methods have been developed to remove undesirable concepts from pre-trained models. However, existing methods remain vulnerable to adversarial attacks that can regenerate the erased content. Moreover, achieving robust erasure often degrades the model's generative quality for safe, unrelated concepts, creating a difficult trade-off between safety and performance. To address this challenge, we introduce Classifier-Guided Concept Erasure (CGCE), an efficient plug-and-play framework that provides robust concept erasure for diverse generative models without altering their original weights. CGCE uses a lightweight classifier operating on text embeddings to first detect and then refine prompts containing undesired concepts. This approach is highly scalable, allowing for multi-concept erasure by aggregating guidance from several classifiers. By modifying only unsafe embeddings at inference time, our method prevents harmful content generation while preserving the model's original quality on benign prompts. Extensive experiments show that CGCE achieves state-of-the-art robustness against a wide range of red-teaming attacks. Our approach also maintains high generative utility, demonstrating a superior balance between safety and performance. We showcase the versatility of CGCE through its successful application to various modern T2I and T2V models, establishing it as a practical and effective solution for safe generative AI.



## **34. Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks**

cs.CL

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2510.02712v2) [paper-pdf](https://arxiv.org/pdf/2510.02712v2)

**Authors**: Yubo Li, Ramayya Krishnan, Rema Padman

**Abstract**: Large Language Models (LLMs) have revolutionized conversational AI, yet their robustness in extended multi-turn dialogues remains poorly understood. Existing evaluation frameworks focus on static benchmarks and single-turn assessments, failing to capture the temporal dynamics of conversational degradation that characterize real-world interactions. In this work, we present a large-scale survival analysis of conversational robustness, modeling failure as a time-to-event process over 36,951 turns from 9 state-of-the-art LLMs on the MT-Consistency benchmark. Our framework combines Cox proportional hazards, Accelerated Failure Time (AFT), and Random Survival Forest models with simple semantic drift features. We find that abrupt prompt-to-prompt semantic drift sharply increases the hazard of inconsistency, whereas cumulative drift is counterintuitively \emph{protective}, suggesting adaptation in conversations that survive multiple shifts. AFT models with model-drift interactions achieve the best combination of discrimination and calibration, and proportional hazards checks reveal systematic violations for key drift covariates, explaining the limitations of Cox-style modeling in this setting. Finally, we show that a lightweight AFT model can be turned into a turn-level risk monitor that flags most failing conversations several turns before the first inconsistent answer while keeping false alerts modest. These results establish survival analysis as a powerful paradigm for evaluating multi-turn robustness and for designing practical safeguards for conversational AI systems.



## **35. Memory Self-Regeneration: Uncovering Hidden Knowledge in Unlearned Models**

cs.LG

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2510.03263v2) [paper-pdf](https://arxiv.org/pdf/2510.03263v2)

**Authors**: Agnieszka Polowczyk, Alicja Polowczyk, Joanna Waczyńska, Piotr Borycki, Przemysław Spurek

**Abstract**: The impressive capability of modern text-to-image models to generate realistic visuals has come with a serious drawback: they can be misused to create harmful, deceptive or unlawful content. This has accelerated the push for machine unlearning. This new field seeks to selectively remove specific knowledge from a model's training data without causing a drop in its overall performance. However, it turns out that actually forgetting a given concept is an extremely difficult task. Models exposed to attacks using adversarial prompts show the ability to generate so-called unlearned concepts, which can be not only harmful but also illegal. In this paper, we present considerations regarding the ability of models to forget and recall knowledge, introducing the Memory Self-Regeneration task. Furthermore, we present MemoRa strategy, which we consider to be a regenerative approach supporting the effective recovery of previously lost knowledge. Moreover, we propose that robustness in knowledge retrieval is a crucial yet underexplored evaluation measure for developing more robust and effective unlearning techniques. Finally, we demonstrate that forgetting occurs in two distinct ways: short-term, where concepts can be quickly recalled, and long-term, where recovery is more challenging. Code is available at https://gmum.github.io/MemoRa/.



## **36. Mind Your Server: A Systematic Study of Parasitic Toolchain Attacks on the MCP Ecosystem**

cs.CR

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2509.06572v2) [paper-pdf](https://arxiv.org/pdf/2509.06572v2)

**Authors**: Shuli Zhao, Qinsheng Hou, Zihan Zhan, Yanhao Wang, Yuchong Xie, Yu Guo, Libo Chen, Shenghong Li, Zhi Xue

**Abstract**: Large language models (LLMs) are increasingly integrated with external systems through the Model Context Protocol (MCP), which standardizes tool invocation and has rapidly become a backbone for LLM-powered applications. While this paradigm enhances functionality, it also introduces a fundamental security shift: LLMs transition from passive information processors to autonomous orchestrators of task-oriented toolchains, expanding the attack surface, elevating adversarial goals from manipulating single outputs to hijacking entire execution flows. In this paper, we reveal a new class of attacks, Parasitic Toolchain Attacks, instantiated as MCP Unintended Privacy Disclosure (MCP-UPD). These attacks require no direct victim interaction; instead, adversaries embed malicious instructions into external data sources that LLMs access during legitimate tasks. The malicious logic infiltrates the toolchain and unfolds in three phases: Parasitic Ingestion, Privacy Collection, and Privacy Disclosure, culminating in stealthy exfiltration of private data. Our root cause analysis reveals that MCP lacks both context-tool isolation and least-privilege enforcement, enabling adversarial instructions to propagate unchecked into sensitive tool invocations. To assess the severity, we design MCP-SEC and conduct the first large-scale security census of the MCP ecosystem, analyzing 12,230 tools across 1,360 servers. Our findings show that the MCP ecosystem is rife with exploitable gadgets and diverse attack methods, underscoring systemic risks in MCP platforms and the urgent need for defense mechanisms in LLM-integrated environments.



## **37. Uncovering and Mitigating Destructive Multi-Embedding Attacks in Deepfake Proactive Forensics**

cs.CV

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2508.17247v2) [paper-pdf](https://arxiv.org/pdf/2508.17247v2)

**Authors**: Lixin Jia, Haiyang Sun, Zhiqing Guo, Yunfeng Diao, Dan Ma, Gaobo Yang

**Abstract**: With the rapid evolution of deepfake technologies and the wide dissemination of digital media, personal privacy is facing increasingly serious security threats. Deepfake proactive forensics, which involves embedding imperceptible watermarks to enable reliable source tracking, serves as a crucial defense against these threats. Although existing methods show strong forensic ability, they rely on an idealized assumption of single watermark embedding, which proves impractical in real-world scenarios. In this paper, we formally define and demonstrate the existence of Multi-Embedding Attacks (MEA) for the first time. When a previously protected image undergoes additional rounds of watermark embedding, the original forensic watermark can be destroyed or removed, rendering the entire proactive forensic mechanism ineffective. To address this vulnerability, we propose a general training paradigm named Adversarial Interference Simulation (AIS). Rather than modifying the network architecture, AIS explicitly simulates MEA scenarios during fine-tuning and introduces a resilience-driven loss function to enforce the learning of sparse and stable watermark representations. Our method enables the model to maintain the ability to extract the original watermark correctly even after a second embedding. Extensive experiments demonstrate that our plug-and-play AIS training paradigm significantly enhances the robustness of various existing methods against MEA.



## **38. Special-Character Adversarial Attacks on Open-Source Language Model**

cs.CR

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2508.14070v2) [paper-pdf](https://arxiv.org/pdf/2508.14070v2)

**Authors**: Ephraiem Sarabamoun

**Abstract**: Large language models (LLMs) have achieved remarkable performance across diverse natural language processing tasks, yet their vulnerability to character-level adversarial manipulations presents significant security challenges for real-world deployments. This paper presents a study of different special character attacks including unicode, homoglyph, structural, and textual encoding attacks aimed at bypassing safety mechanisms. We evaluate seven prominent open-source models ranging from 3.8B to 32B parameters on 4,000+ attack attempts. These experiments reveal critical vulnerabilities across all model sizes, exposing failure modes that include successful jailbreaks, incoherent outputs, and unrelated hallucinations.



## **39. Large Language Model Unlearning for Source Code**

cs.SE

Accepted to AAAI'26

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2506.17125v2) [paper-pdf](https://arxiv.org/pdf/2506.17125v2)

**Authors**: Xue Jiang, Yihong Dong, Huangzhao Zhang, Tangxinyu Wang, Zheng Fang, Yingwei Ma, Rongyu Cao, Binhua Li, Zhi Jin, Wenpin Jiao, Yongbin Li, Ge Li

**Abstract**: While Large Language Models (LLMs) excel at code generation, their inherent tendency toward verbatim memorization of training data introduces critical risks like copyright infringement, insecure emission, and deprecated API utilization, etc. A straightforward yet promising defense is unlearning, ie., erasing or down-weighting the offending snippets through post-training. However, we find its application to source code often tends to spill over, damaging the basic knowledge of programming languages learned by the LLM and degrading the overall capability. To ease this challenge, we propose PROD for precise source code unlearning. PROD surgically zeroes out the prediction probability of the prohibited tokens, and renormalizes the remaining distribution so that the generated code stays correct. By excising only the targeted snippets, PROD achieves precise forgetting without much degradation of the LLM's overall capability. To facilitate in-depth evaluation against PROD, we establish an unlearning benchmark consisting of three downstream tasks (ie., unlearning of copyrighted code, insecure code, and deprecated APIs), and introduce Pareto Dominance Ratio (PDR) metric, which indicates both the forget quality and the LLM utility. Our comprehensive evaluation demonstrates that PROD achieves superior overall performance between forget quality and model utility compared to existing unlearning approaches across three downstream tasks, while consistently exhibiting improvements when applied to LLMs of varying series. PROD also exhibits superior robustness against adversarial attacks without generating or exposing the data to be forgotten. These results underscore that our approach not only successfully extends the application boundary of unlearning techniques to source code, but also holds significant implications for advancing reliable code generation.



## **40. TRAP: Targeted Redirecting of Agentic Preferences**

cs.AI

Accepted to NeurIPS 2025

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2505.23518v2) [paper-pdf](https://arxiv.org/pdf/2505.23518v2)

**Authors**: Hangoo Kang, Jehyeok Yeon, Gagandeep Singh

**Abstract**: Autonomous agentic AI systems powered by vision-language models (VLMs) are rapidly advancing toward real-world deployment, yet their cross-modal reasoning capabilities introduce new attack surfaces for adversarial manipulation that exploit semantic reasoning across modalities. Existing adversarial attacks typically rely on visible pixel perturbations or require privileged model or environment access, making them impractical for stealthy, real-world exploitation. We introduce TRAP, a novel generative adversarial framework that manipulates the agent's decision-making using diffusion-based semantic injections into the vision-language embedding space. Our method combines negative prompt-based degradation with positive semantic optimization, guided by a Siamese semantic network and layout-aware spatial masking. Without requiring access to model internals, TRAP produces visually natural images yet induces consistent selection biases in agentic AI systems. We evaluate TRAP on the Microsoft Common Objects in Context (COCO) dataset, building multi-candidate decision scenarios. Across these scenarios, TRAP consistently induces decision-level preference redirection on leading models, including LLaVA-34B, Gemma3, GPT-4o, and Mistral-3.2, significantly outperforming existing baselines such as SPSA, Bandit, and standard diffusion approaches. These findings expose a critical, generalized vulnerability: autonomous agents can be consistently misled through visually subtle, semantically-guided cross-modal manipulations. Overall, our results show the need for defense strategies beyond pixel-level robustness to address semantic vulnerabilities in cross-modal decision-making. The code for TRAP is accessible on GitHub at https://github.com/uiuc-focal-lab/TRAP.



## **41. Adversarial Robustness for Unified Multi-Modal Encoders via Efficient Calibration**

cs.CV

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2505.11895v2) [paper-pdf](https://arxiv.org/pdf/2505.11895v2)

**Authors**: Chih-Ting Liao, Zhangquan Chen, Chunlei Meng, Tzu-Yu Huang, Xin Cao, Xu Zheng

**Abstract**: Recent unified multi-modal encoders align a wide range of modalities into a shared representation space, enabling diverse cross-modal tasks. Despite their impressive capabilities, the robustness of these models under adversarial perturbations remains underexplored, which is a critical concern for safety-sensitive applications. In this work, we present the first comprehensive study of adversarial vulnerability in unified multi-modal encoders. We find that even mild adversarial perturbations lead to substantial performance drops across all modalities. Non-visual inputs, such as audio and point clouds, are especially fragile, while visual inputs like images and videos also degrade significantly. To address this, we propose an efficient adversarial calibration framework that improves robustness across modalities without modifying pretrained encoders or semantic centers, ensuring compatibility with existing foundation models. Our method introduces modality-specific projection heads trained solely on adversarial examples, while keeping the backbone and embeddings frozen. We explore three training objectives: fixed-center cross-entropy, clean-to-adversarial L2 alignment, and clean-adversarial InfoNCE, and we introduce a regularization strategy to ensure modality-consistent alignment under attack. Experiments on six modalities and three Bind-style models show that our method improves adversarial robustness by up to 47.3 percent at epsilon = 4/255, while preserving or even improving clean zero-shot and retrieval performance with less than 1 percent trainable parameters.



## **42. Benchmarking the Spatial Robustness of DNNs via Natural and Adversarial Localized Corruptions**

cs.CV

Accepted for publication in Pattern Recognition

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2504.01632v3) [paper-pdf](https://arxiv.org/pdf/2504.01632v3)

**Authors**: Giulia Marchiori Pietrosanti, Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo

**Abstract**: The robustness of deep neural networks is a crucial factor in safety-critical applications, particularly in complex and dynamic environments (e.g., medical or driving scenarios) where localized corruptions can arise. While previous studies have evaluated the robustness of semantic segmentation (SS) models under whole-image natural or adversarial corruptions, a comprehensive investigation into the spatial robustness of dense vision models under localized corruptions remains underexplored. This paper fills this gap by introducing novel, region-aware metrics for benchmarking the spatial robustness of segmentation models, along with an evaluation framework to assess the impact of natural localized corruptions. Furthermore, it uncovers the inherent complexity of evaluating worst-case spatial robustness using only a single localized adversarial attack. To address this, the work proposes a region-aware multi-attack adversarial analysis to systematically assess model robustness across specific image regions. The proposed metrics and analysis were exploited to evaluate 14 segmentation models in driving scenarios, uncovering key insights into the effects of localized corruption in both natural and adversarial forms. The results reveal that models respond to these two types of threats differently; for instance, transformer-based segmentation models demonstrate notable robustness to localized natural corruptions but are highly vulnerable to adversarial ones, and vice versa for CNN-based models. Consequently, we also address the challenge of balancing robustness to both natural and adversarial localized corruptions by means of ensemble models, thereby achieving a broader threat coverage and improved reliability for dense vision tasks.



## **43. ARBoids: Adaptive Residual Reinforcement Learning With Boids Model for Cooperative Multi-USV Target Defense**

cs.LG

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2502.18549v3) [paper-pdf](https://arxiv.org/pdf/2502.18549v3)

**Authors**: Jiyue Tao, Tongsheng Shen, Dexin Zhao, Feitian Zhang

**Abstract**: The target defense problem (TDP) for unmanned surface vehicles (USVs) concerns intercepting an adversarial USV before it breaches a designated target region, using one or more defending USVs. A particularly challenging scenario arises when the attacker exhibits superior maneuverability compared to the defenders, significantly complicating effective interception. To tackle this challenge, this letter introduces ARBoids, a novel adaptive residual reinforcement learning framework that integrates deep reinforcement learning (DRL) with the biologically inspired, force-based Boids model. Within this framework, the Boids model serves as a computationally efficient baseline policy for multi-agent coordination, while DRL learns a residual policy to adaptively refine and optimize the defenders' actions. The proposed approach is validated in a high-fidelity Gazebo simulation environment, demonstrating superior performance over traditional interception strategies, including pure force-based approaches and vanilla DRL policies. Furthermore, the learned policy exhibits strong adaptability to attackers with diverse maneuverability profiles, highlighting its robustness and generalization capability. The code of ARBoids will be released upon acceptance of this letter.



## **44. DarkMind: Latent Chain-of-Thought Backdoor in Customized LLMs**

cs.CR

19 pages, 15 figures, 12 tables

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2501.18617v2) [paper-pdf](https://arxiv.org/pdf/2501.18617v2)

**Authors**: Zhen Guo, Shanghao Shi, Shamim Yazdani, Ning Zhang, Reza Tourani

**Abstract**: With the rapid rise of personalized AI, customized large language models (LLMs) equipped with Chain of Thought (COT) reasoning now power millions of AI agents. However, their complex reasoning processes introduce new and largely unexplored security vulnerabilities. We present DarkMind, a novel latent reasoning level backdoor attack that targets customized LLMs by manipulating internal COT steps without altering user queries. Unlike prior prompt based attacks, DarkMind activates covertly within the reasoning chain via latent triggers, enabling adversarial behaviors without modifying input prompts or requiring access to model parameters. To achieve stealth and reliability, we propose dual trigger types instant and retrospective and integrate them within a unified embedding template that governs trigger dependent activation, employ a stealth optimization algorithm to minimize semantic drift, and introduce an automated conversation starter for covert activation across domains. Comprehensive experiments on eight reasoning datasets spanning arithmetic, commonsense, and symbolic domains, using five LLMs, demonstrate that DarkMind consistently achieves high attack success rates. We further investigate defense strategies to mitigate these risks and reveal that reasoning level backdoors represent a significant yet underexplored threat, underscoring the need for robust, reasoning aware security mechanisms.



## **45. On the Effectiveness of Adversarial Training on Malware Classifiers**

cs.LG

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2412.18218v2) [paper-pdf](https://arxiv.org/pdf/2412.18218v2)

**Authors**: Hamid Bostani, Jacopo Cortellazzi, Daniel Arp, Fabio Pierazzi, Veelasha Moonsamy, Lorenzo Cavallaro

**Abstract**: Adversarial Training (AT) is a key defense against Machine Learning evasion attacks, but its effectiveness for real-world malware detection remains poorly understood. This uncertainty stems from a critical disconnect in prior research: studies often overlook the inherent nature of malware and are fragmented, examining diverse variables like realism or confidence of adversarial examples in isolation, or relying on weak evaluations that yield non-generalizable insights. To address this, we introduce Rubik, a framework for the systematic, multi-dimensional evaluation of AT in the malware domain. This framework defines diverse key factors across essential dimensions, including data, feature representations, classifiers, and robust optimization settings, for a comprehensive exploration of the interplay of influential AT's variables through reliable evaluation practices, such as realistic evasion attacks. We instantiate Rubik on Android malware, empirically analyzing how this interplay shapes robustness. Our findings challenge prior beliefs--showing, for instance, that realizable adversarial examples offer only conditional robustness benefits--and reveal new insights, such as the critical role of model architecture and feature-space structure in determining AT's success. From this analysis, we distill four key insights, expose four common evaluation misconceptions, and offer practical recommendations to guide the development of truly robust malware classifiers.



## **46. Jailbreaking and Mitigation of Vulnerabilities in Large Language Models**

cs.CR

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2410.15236v3) [paper-pdf](https://arxiv.org/pdf/2410.15236v3)

**Authors**: Benji Peng, Keyu Chen, Qian Niu, Ziqian Bi, Ming Liu, Pohsun Feng, Tianyang Wang, Lawrence K. Q. Yan, Yizhu Wen, Yichao Zhang, Caitlyn Heqi Yin, Xinyuan Song

**Abstract**: Large Language Models (LLMs) have transformed artificial intelligence by advancing natural language understanding and generation, enabling applications across fields beyond healthcare, software engineering, and conversational systems. Despite these advancements in the past few years, LLMs have shown considerable vulnerabilities, particularly to prompt injection and jailbreaking attacks. This review analyzes the state of research on these vulnerabilities and presents available defense strategies. We roughly categorize attack approaches into prompt-based, model-based, multimodal, and multilingual, covering techniques such as adversarial prompting, backdoor injections, and cross-modality exploits. We also review various defense mechanisms, including prompt filtering, transformation, alignment techniques, multi-agent defenses, and self-regulation, evaluating their strengths and shortcomings. We also discuss key metrics and benchmarks used to assess LLM safety and robustness, noting challenges like the quantification of attack success in interactive contexts and biases in existing datasets. Identifying current research gaps, we suggest future directions for resilient alignment strategies, advanced defenses against evolving attacks, automation of jailbreak detection, and consideration of ethical and societal impacts. This review emphasizes the need for continued research and cooperation within the AI community to enhance LLM security and ensure their safe deployment.



## **47. A Gray-box Attack against Latent Diffusion Model-based Image Editing by Posterior Collapse**

cs.CV

15 pages, 9 figures, 9 tables

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2408.10901v4) [paper-pdf](https://arxiv.org/pdf/2408.10901v4)

**Authors**: Zhongliang Guo, Chun Tong Lei, Lei Fang, Shuai Zhao, Yifei Qian, Jingyu Lin, Zeyu Wang, Cunjian Chen, Ognjen Arandjelović, Chun Pong Lau

**Abstract**: Recent advancements in Latent Diffusion Models (LDMs) have revolutionized image synthesis and manipulation, raising significant concerns about data misappropriation and intellectual property infringement. While adversarial attacks have been extensively explored as a protective measure against such misuse of generative AI, current approaches are severely limited by their heavy reliance on model-specific knowledge and substantial computational costs. Drawing inspiration from the posterior collapse phenomenon observed in VAE training, we propose the Posterior Collapse Attack (PCA), a novel framework for protecting images from unauthorized manipulation. Through comprehensive theoretical analysis and empirical validation, we identify two distinct collapse phenomena during VAE inference: diffusion collapse and concentration collapse. Based on this discovery, we design a unified loss function that can flexibly achieve both types of collapse through parameter adjustment, each corresponding to different protection objectives in preventing image manipulation. Our method significantly reduces dependence on model-specific knowledge by requiring access to only the VAE encoder, which constitutes less than 4\% of LDM parameters. Notably, PCA achieves prompt-invariant protection by operating on the VAE encoder before text conditioning occurs, eliminating the need for empty prompt optimization required by existing methods. This minimal requirement enables PCA to maintain adequate transferability across various VAE-based LDM architectures while effectively preventing unauthorized image editing. Extensive experiments show PCA outperforms existing techniques in protection effectiveness, computational efficiency (runtime and VRAM), and generalization across VAE-based LDM variants. Our code is available at https://github.com/ZhongliangGuo/PosteriorCollapseAttack.



## **48. HO-FMN: Hyperparameter Optimization for Fast Minimum-Norm Attacks**

cs.LG

Accepted at Neurocomputing

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2407.08806v3) [paper-pdf](https://arxiv.org/pdf/2407.08806v3)

**Authors**: Raffaele Mura, Giuseppe Floris, Luca Scionis, Giorgio Piras, Maura Pintor, Ambra Demontis, Giorgio Giacinto, Battista Biggio, Fabio Roli

**Abstract**: Gradient-based attacks are a primary tool to evaluate robustness of machine-learning models. However, many attacks tend to provide overly-optimistic evaluations as they use fixed loss functions, optimizers, step-size schedulers, and default hyperparameters. In this work, we tackle these limitations by proposing a parametric variation of the well-known fast minimum-norm attack algorithm, whose loss, optimizer, step-size scheduler, and hyperparameters can be dynamically adjusted. We re-evaluate 12 robust models, showing that our attack finds smaller adversarial perturbations without requiring any additional tuning. This also enables reporting adversarial robustness as a function of the perturbation budget, providing a more complete evaluation than that offered by fixed-budget attacks, while remaining efficient. We release our open-source code at https://github.com/pralab/HO-FMN.



## **49. Data-Driven Lipschitz Continuity: A Cost-Effective Approach to Improve Adversarial Robustness**

cs.LG

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2406.19622v2) [paper-pdf](https://arxiv.org/pdf/2406.19622v2)

**Authors**: Erh-Chung Chen, Pin-Yu Chen, I-Hsin Chung, Che-Rung Lee

**Abstract**: As deep neural networks (DNNs) are increasingly deployed in sensitive applications, ensuring their security and robustness has become critical. A major threat to DNNs arises from adversarial attacks, where small input perturbations can lead to incorrect predictions. Recent advances in adversarial training improve robustness by incorporating additional examples from external datasets or generative models. However, these methods often incur high computational costs, limiting their practicality and hindering real-world deployment. In this paper, we propose a cost-efficient alternative based on Lipschitz continuity that achieves robustness comparable to models trained with extensive supplementary data. Unlike conventional adversarial training, our method requires only a single pass over the dataset without gradient estimation, making it highly efficient. Furthermore, our method can integrate seamlessly with existing adversarial training frameworks and enhances the robustness of models without requiring extra generative data. Experimental results show that our approach not only reduces computational overhead but also maintains or improves the defensive capabilities of robust neural networks. This work opens a promising direction for developing practical, scalable defenses against adversarial attacks.



## **50. LTD: Low Temperature Distillation for Gradient Masking-free Adversarial Training**

cs.CV

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2111.02331v4) [paper-pdf](https://arxiv.org/pdf/2111.02331v4)

**Authors**: Erh-Chung Chen, Che-Rung Lee

**Abstract**: Adversarial training is a widely adopted strategy to bolster the robustness of neural network models against adversarial attacks. This paper revisits the fundamental assumptions underlying image classification and suggests that representing data as one-hot labels is a key factor that leads to vulnerabilities. However, in real-world datasets, data ambiguity often arises, with samples exhibiting characteristics of multiple classes, rendering one-hot label representations imprecise. To address this, we introduce a novel approach, Low-Temperature Distillation (LTD), designed to refine label representations. Unlike previous approaches, LTD incorporates a relatively low temperature in the teacher model, while maintaining a fixed temperature for the student model during both training and inference. This strategy not only refines assumptions about data distribution but also strengthens model robustness and avoids the gradient masking problem commonly encountered in defensive distillation. Experimental results demonstrate the efficacy of the proposed method when combined with existing frameworks, achieving robust accuracy rates of 58.19%, 31.13%, and 42.08% on the CIFAR-10, CIFAR-100, and ImageNet datasets, respectively, without the need for additional data.



