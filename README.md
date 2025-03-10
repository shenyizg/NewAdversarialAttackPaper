# Latest Adversarial Attack Papers
**update at 2025-03-10 09:56:12**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks**

cs.CV

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2407.20836v2) [paper-pdf](http://arxiv.org/pdf/2407.20836v2)

**Authors**: Yunfeng Diao, Naixin Zhai, Changtao Miao, Zitong Yu, Xingxing Wei, Xun Yang, Meng Wang

**Abstract**: Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. To this end, we propose a new method to attack AIGI detectors. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous AIGI detectors, e.g. transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as Frequency-based Post-train Bayesian Attack, or FPBA. Through FPBA, we show that adversarial attack is truly a real threat to AIGI detectors, because FPBA can deliver successful black-box attacks across models, generators, defense methods, and even evade cross-generator detection, which is a crucial real-world detection scenario. The code will be shared upon acceptance.



## **2. Toward Robust Non-Transferable Learning: A Survey and Benchmark**

cs.LG

Code is available at https://github.com/tmllab/NTLBench

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2502.13593v2) [paper-pdf](http://arxiv.org/pdf/2502.13593v2)

**Authors**: Ziming Hong, Yongli Xiang, Tongliang Liu

**Abstract**: Over the past decades, researchers have primarily focused on improving the generalization abilities of models, with limited attention given to regulating such generalization. However, the ability of models to generalize to unintended data (e.g., harmful or unauthorized data) can be exploited by malicious adversaries in unforeseen ways, potentially resulting in violations of model ethics. Non-transferable learning (NTL), a task aimed at reshaping the generalization abilities of deep learning models, was proposed to address these challenges. While numerous methods have been proposed in this field, a comprehensive review of existing progress and a thorough analysis of current limitations remain lacking. In this paper, we bridge this gap by presenting the first comprehensive survey on NTL and introducing NTLBench, the first benchmark to evaluate NTL performance and robustness within a unified framework. Specifically, we first introduce the task settings, general framework, and criteria of NTL, followed by a summary of NTL approaches. Furthermore, we emphasize the often-overlooked issue of robustness against various attacks that can destroy the non-transferable mechanism established by NTL. Experiments conducted via NTLBench verify the limitations of existing NTL methods in robustness. Finally, we discuss the practical applications of NTL, along with its future directions and associated challenges.



## **3. Robust Intrusion Detection System with Explainable Artificial Intelligence**

cs.CR

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.05303v1) [paper-pdf](http://arxiv.org/pdf/2503.05303v1)

**Authors**: Betül Güvenç Paltun, Ramin Fuladi, Rim El Malki

**Abstract**: Machine learning (ML) models serve as powerful tools for threat detection and mitigation; however, they also introduce potential new risks. Adversarial input can exploit these models through standard interfaces, thus creating new attack pathways that threaten critical network operations. As ML advancements progress, adversarial strategies become more advanced, and conventional defenses such as adversarial training are costly in computational terms and often fail to provide real-time detection. These methods typically require a balance between robustness and model performance, which presents challenges for applications that demand instant response. To further investigate this vulnerability, we suggest a novel strategy for detecting and mitigating adversarial attacks using eXplainable Artificial Intelligence (XAI). This approach is evaluated in real time within intrusion detection systems (IDS), leading to the development of a zero-touch mitigation strategy. Additionally, we explore various scenarios in the Radio Resource Control (RRC) layer within the Open Radio Access Network (O-RAN) framework, emphasizing the critical need for enhanced mitigation techniques to strengthen IDS defenses against advanced threats and implement a zero-touch mitigation solution. Extensive testing across different scenarios in the RRC layer of the O-RAN infrastructure validates the ability of the framework to detect and counteract integrated RRC-layer attacks when paired with adversarial strategies, emphasizing the essential need for robust defensive mechanisms to strengthen IDS against complex threats.



## **4. Jailbreaking is (Mostly) Simpler Than You Think**

cs.CR

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.05264v1) [paper-pdf](http://arxiv.org/pdf/2503.05264v1)

**Authors**: Mark Russinovich, Ahmed Salem

**Abstract**: We introduce the Context Compliance Attack (CCA), a novel, optimization-free method for bypassing AI safety mechanisms. Unlike current approaches -- which rely on complex prompt engineering and computationally intensive optimization -- CCA exploits a fundamental architectural vulnerability inherent in many deployed AI systems. By subtly manipulating conversation history, CCA convinces the model to comply with a fabricated dialogue context, thereby triggering restricted behavior. Our evaluation across a diverse set of open-source and proprietary models demonstrates that this simple attack can circumvent state-of-the-art safety protocols. We discuss the implications of these findings and propose practical mitigation strategies to fortify AI systems against such elementary yet effective adversarial tactics.



## **5. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

cs.CL

Accepted to NeurIPS 2024 Datasets and Benchmarks Track (Camera-Ready)

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2410.23746v2) [paper-pdf](http://arxiv.org/pdf/2410.23746v2)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating various prompts usages, human revisions like word substitutions, and writing noises like spelling mistakes. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.



## **6. Safety-Critical Traffic Simulation with Adversarial Transfer of Driving Intentions**

cs.RO

Accepted by ICRA 2025

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.05180v1) [paper-pdf](http://arxiv.org/pdf/2503.05180v1)

**Authors**: Zherui Huang, Xing Gao, Guanjie Zheng, Licheng Wen, Xuemeng Yang, Xiao Sun

**Abstract**: Traffic simulation, complementing real-world data with a long-tail distribution, allows for effective evaluation and enhancement of the ability of autonomous vehicles to handle accident-prone scenarios. Simulating such safety-critical scenarios is nontrivial, however, from log data that are typically regular scenarios, especially in consideration of dynamic adversarial interactions between the future motions of autonomous vehicles and surrounding traffic participants. To address it, this paper proposes an innovative and efficient strategy, termed IntSim, that explicitly decouples the driving intentions of surrounding actors from their motion planning for realistic and efficient safety-critical simulation. We formulate the adversarial transfer of driving intention as an optimization problem, facilitating extensive exploration of diverse attack behaviors and efficient solution convergence. Simultaneously, intention-conditioned motion planning benefits from powerful deep models and large-scale real-world data, permitting the simulation of realistic motion behaviors for actors. Specially, through adapting driving intentions based on environments, IntSim facilitates the flexible realization of dynamic adversarial interactions with autonomous vehicles. Finally, extensive open-loop and closed-loop experiments on real-world datasets, including nuScenes and Waymo, demonstrate that the proposed IntSim achieves state-of-the-art performance in simulating realistic safety-critical scenarios and further improves planners in handling such scenarios.



## **7. Double Backdoored: Converting Code Large Language Model Backdoors to Traditional Malware via Adversarial Instruction Tuning Attacks**

cs.CR

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2404.18567v2) [paper-pdf](http://arxiv.org/pdf/2404.18567v2)

**Authors**: Md Imran Hossen, Sai Venkatesh Chilukoti, Liqun Shan, Sheng Chen, Yinzhi Cao, Xiali Hei

**Abstract**: Instruction-tuned Large Language Models designed for coding tasks are increasingly employed as AI coding assistants. However, the cybersecurity vulnerabilities and implications arising from the widespread integration of these models are not yet fully understood due to limited research in this domain. This work investigates novel techniques for transitioning backdoors from the AI/ML domain to traditional computer malware, shedding light on the critical intersection of AI and cyber/software security. To explore this intersection, we present MalInstructCoder, a framework designed to comprehensively assess the cybersecurity vulnerabilities of instruction-tuned Code LLMs. MalInstructCoder introduces an automated data poisoning pipeline to inject malicious code snippets into benign code, poisoning instruction fine-tuning data while maintaining functional validity. It presents two practical adversarial instruction tuning attacks with real-world security implications: the clean prompt poisoning attack and the backdoor attack. These attacks aim to manipulate Code LLMs to generate code incorporating malicious or harmful functionality under specific attack scenarios while preserving intended functionality. We conduct a comprehensive investigation into the exploitability of the code-specific instruction tuning process involving three state-of-the-art Code LLMs: CodeLlama, DeepSeek-Coder, and StarCoder2. Our findings reveal that these models are highly vulnerable to our attacks. Specifically, the clean prompt poisoning attack achieves the ASR@1 ranging from over 75% to 86% by poisoning only 1% (162 samples) of the instruction fine-tuning dataset. Similarly, the backdoor attack achieves the ASR@1 ranging from 76% to 86% with a 0.5% poisoning rate. Our study sheds light on the critical cybersecurity risks posed by instruction-tuned Code LLMs and highlights the urgent need for robust defense mechanisms.



## **8. Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety**

cs.CL

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.05021v1) [paper-pdf](http://arxiv.org/pdf/2503.05021v1)

**Authors**: Yuyou Zhang, Miao Li, William Han, Yihang Yao, Zhepeng Cen, Ding Zhao

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreak attacks that exploit weaknesses in traditional safety alignment, which often relies on rigid refusal heuristics or representation engineering to block harmful outputs. While they are effective for direct adversarial attacks, they fall short of broader safety challenges requiring nuanced, context-aware decision-making. To address this, we propose Reasoning-enhanced Finetuning for interpretable LLM Safety (Rational), a novel framework that trains models to engage in explicit safe reasoning before response. Fine-tuned models leverage the extensive pretraining knowledge in self-generated reasoning to bootstrap their own safety through structured reasoning, internalizing context-sensitive decision-making. Our findings suggest that safety extends beyond refusal, requiring context awareness for more robust, interpretable, and adaptive responses. Reasoning is not only a core capability of LLMs but also a fundamental mechanism for LLM safety. Rational employs reasoning-enhanced fine-tuning, allowing it to reject harmful prompts while providing meaningful and context-aware responses in complex scenarios.



## **9. Energy-Latency Attacks: A New Adversarial Threat to Deep Learning**

cs.CR

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04963v1) [paper-pdf](http://arxiv.org/pdf/2503.04963v1)

**Authors**: Hanene F. Z. Brachemi Meftah, Wassim Hamidouche, Sid Ahmed Fezza, Olivier Deforges

**Abstract**: The growing computational demand for deep neural networks ( DNNs) has raised concerns about their energy consumption and carbon footprint, particularly as the size and complexity of the models continue to increase. To address these challenges, energy-efficient hardware and custom accelerators have become essential. Additionally, adaptable DNN s are being developed to dynamically balance performance and efficiency. The use of these strategies became more common to enable sustainable AI deployment. However, these efficiency-focused designs may also introduce vulnerabilities, as attackers can potentially exploit them to increase latency and energy usage by triggering their worst-case-performance scenarios. This new type of attack, called energy-latency attacks, has recently gained significant research attention, focusing on the vulnerability of DNN s to this emerging attack paradigm, which can trigger denial-of-service ( DoS) attacks. This paper provides a comprehensive overview of current research on energy-latency attacks, categorizing them using the established taxonomy for traditional adversarial attacks. We explore different metrics used to measure the success of these attacks and provide an analysis and comparison of existing attack strategies. We also analyze existing defense mechanisms and highlight current challenges and potential areas for future research in this developing field. The GitHub page for this work can be accessed at https://github.com/hbrachemi/Survey_energy_attacks/



## **10. The Last Iterate Advantage: Empirical Auditing and Principled Heuristic Analysis of Differentially Private SGD**

cs.CR

ICLR 2025 camera-ready version

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2410.06186v4) [paper-pdf](http://arxiv.org/pdf/2410.06186v4)

**Authors**: Thomas Steinke, Milad Nasr, Arun Ganesh, Borja Balle, Christopher A. Choquette-Choo, Matthew Jagielski, Jamie Hayes, Abhradeep Guha Thakurta, Adam Smith, Andreas Terzis

**Abstract**: We propose a simple heuristic privacy analysis of noisy clipped stochastic gradient descent (DP-SGD) in the setting where only the last iterate is released and the intermediate iterates remain hidden. Namely, our heuristic assumes a linear structure for the model.   We show experimentally that our heuristic is predictive of the outcome of privacy auditing applied to various training procedures. Thus it can be used prior to training as a rough estimate of the final privacy leakage. We also probe the limitations of our heuristic by providing some artificial counterexamples where it underestimates the privacy leakage.   The standard composition-based privacy analysis of DP-SGD effectively assumes that the adversary has access to all intermediate iterates, which is often unrealistic. However, this analysis remains the state of the art in practice. While our heuristic does not replace a rigorous privacy analysis, it illustrates the large gap between the best theoretical upper bounds and the privacy auditing lower bounds and sets a target for further work to improve the theoretical privacy analyses. We also empirically support our heuristic and show existing privacy auditing attacks are bounded by our heuristic analysis in both vision and language tasks.



## **11. FSPGD: Rethinking Black-box Attacks on Semantic Segmentation**

cs.CV

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2502.01262v2) [paper-pdf](http://arxiv.org/pdf/2502.01262v2)

**Authors**: Eun-Sol Park, MiSo Park, Seung Park, Yong-Goo Shin

**Abstract**: Transferability, the ability of adversarial examples crafted for one model to deceive other models, is crucial for black-box attacks. Despite advancements in attack methods for semantic segmentation, transferability remains limited, reducing their effectiveness in real-world applications. To address this, we introduce the Feature Similarity Projected Gradient Descent (FSPGD) attack, a novel black-box approach that enhances both attack performance and transferability. Unlike conventional segmentation attacks that rely on output predictions for gradient calculation, FSPGD computes gradients from intermediate layer features. Specifically, our method introduces a loss function that targets local information by comparing features between clean images and adversarial examples, while also disrupting contextual information by accounting for spatial relationships between objects. Experiments on Pascal VOC 2012 and Cityscapes datasets demonstrate that FSPGD achieves superior transferability and attack performance, establishing a new state-of-the-art benchmark. Code is available at https://github.com/KU-AIVS/FSPGD.



## **12. OrbID: Identifying Orbcomm Satellite RF Fingerprints**

eess.SP

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.02118v2) [paper-pdf](http://arxiv.org/pdf/2503.02118v2)

**Authors**: Cédric Solenthaler, Joshua Smailes, Martin Strohmeier

**Abstract**: An increase in availability of Software Defined Radios (SDRs) has caused a dramatic shift in the threat landscape of legacy satellite systems, opening them up to easy spoofing attacks by low-budget adversaries. Physical-layer authentication methods can help improve the security of these systems by providing additional validation without modifying the space segment. This paper extends previous research on Radio Frequency Fingerprinting (RFF) of satellite communication to the Orbcomm satellite formation. The GPS and Iridium constellations are already well covered in prior research, but the feasibility of transferring techniques to other formations has not yet been examined, and raises previously undiscussed challenges.   In this paper, we collect a novel dataset containing 8992474 packets from the Orbcom satellite constellation using different SDRs and locations. We use this dataset to train RFF systems based on convolutional neural networks. We achieve an ROC AUC score of 0.53 when distinguishing different satellites within the constellation, and 0.98 when distinguishing legitimate satellites from SDRs in a spoofing scenario. We also demonstrate the possibility of mixing datasets using different SDRs in different physical locations.



## **13. Poisoning Bayesian Inference via Data Deletion and Replication**

stat.ML

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04480v1) [paper-pdf](http://arxiv.org/pdf/2503.04480v1)

**Authors**: Matthieu Carreau, Roi Naveiro, William N. Caballero

**Abstract**: Research in adversarial machine learning (AML) has shown that statistical models are vulnerable to maliciously altered data. However, despite advances in Bayesian machine learning models, most AML research remains concentrated on classical techniques. Therefore, we focus on extending the white-box model poisoning paradigm to attack generic Bayesian inference, highlighting its vulnerability in adversarial contexts. A suite of attacks are developed that allow an attacker to steer the Bayesian posterior toward a target distribution through the strategic deletion and replication of true observations, even when only sampling access to the posterior is available. Analytic properties of these algorithms are proven and their performance is empirically examined in both synthetic and real-world scenarios. With relatively little effort, the attacker is able to substantively alter the Bayesian's beliefs and, by accepting more risk, they can mold these beliefs to their will. By carefully constructing the adversarial posterior, surgical poisoning is achieved such that only targeted inferences are corrupted and others are minimally disturbed.



## **14. Know Thy Judge: On the Robustness Meta-Evaluation of LLM Safety Judges**

cs.LG

Accepted to the ICBINB Workshop at ICLR'25

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04474v1) [paper-pdf](http://arxiv.org/pdf/2503.04474v1)

**Authors**: Francisco Eiras, Eliott Zemour, Eric Lin, Vaikkunth Mugunthan

**Abstract**: Large Language Model (LLM) based judges form the underpinnings of key safety evaluation processes such as offline benchmarking, automated red-teaming, and online guardrailing. This widespread requirement raises the crucial question: can we trust the evaluations of these evaluators? In this paper, we highlight two critical challenges that are typically overlooked: (i) evaluations in the wild where factors like prompt sensitivity and distribution shifts can affect performance and (ii) adversarial attacks that target the judge. We highlight the importance of these through a study of commonly used safety judges, showing that small changes such as the style of the model output can lead to jumps of up to 0.24 in the false negative rate on the same dataset, whereas adversarial attacks on the model generation can fool some judges into misclassifying 100% of harmful generations as safe ones. These findings reveal gaps in commonly used meta-evaluation benchmarks and weaknesses in the robustness of current LLM judges, indicating that low attack success under certain judges could create a false sense of security.



## **15. Privacy Preserving and Robust Aggregation for Cross-Silo Federated Learning in Non-IID Settings**

cs.LG

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04451v1) [paper-pdf](http://arxiv.org/pdf/2503.04451v1)

**Authors**: Marco Arazzi, Mert Cihangiroglu, Antonino Nocera

**Abstract**: Federated Averaging remains the most widely used aggregation strategy in federated learning due to its simplicity and scalability. However, its performance degrades significantly in non-IID data settings, where client distributions are highly imbalanced or skewed. Additionally, it relies on clients transmitting metadata, specifically the number of training samples, which introduces privacy risks and may conflict with regulatory frameworks like the European GDPR. In this paper, we propose a novel aggregation strategy that addresses these challenges by introducing class-aware gradient masking. Unlike traditional approaches, our method relies solely on gradient updates, eliminating the need for any additional client metadata, thereby enhancing privacy protection. Furthermore, our approach validates and dynamically weights client contributions based on class-specific importance, ensuring robustness against non-IID distributions, convergence prevention, and backdoor attacks. Extensive experiments on benchmark datasets demonstrate that our method not only outperforms FedAvg and other widely accepted aggregation strategies in non-IID settings but also preserves model integrity in adversarial scenarios. Our results establish the effectiveness of gradient masking as a practical and secure solution for federated learning.



## **16. Towards Effective and Sparse Adversarial Attack on Spiking Neural Networks via Breaking Invisible Surrogate Gradients**

cs.CV

Accepted by CVPR 2025

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.03272v2) [paper-pdf](http://arxiv.org/pdf/2503.03272v2)

**Authors**: Li Lun, Kunyu Feng, Qinglong Ni, Ling Liang, Yuan Wang, Ying Li, Dunshan Yu, Xiaoxin Cui

**Abstract**: Spiking neural networks (SNNs) have shown their competence in handling spatial-temporal event-based data with low energy consumption. Similar to conventional artificial neural networks (ANNs), SNNs are also vulnerable to gradient-based adversarial attacks, wherein gradients are calculated by spatial-temporal back-propagation (STBP) and surrogate gradients (SGs). However, the SGs may be invisible for an inference-only model as they do not influence the inference results, and current gradient-based attacks are ineffective for binary dynamic images captured by the dynamic vision sensor (DVS). While some approaches addressed the issue of invisible SGs through universal SGs, their SGs lack a correlation with the victim model, resulting in sub-optimal performance. Moreover, the imperceptibility of existing SNN-based binary attacks is still insufficient. In this paper, we introduce an innovative potential-dependent surrogate gradient (PDSG) method to establish a robust connection between the SG and the model, thereby enhancing the adaptability of adversarial attacks across various models with invisible SGs. Additionally, we propose the sparse dynamic attack (SDA) to effectively attack binary dynamic images. Utilizing a generation-reduction paradigm, SDA can fully optimize the sparsity of adversarial perturbations. Experimental results demonstrate that our PDSG and SDA outperform state-of-the-art SNN-based attacks across various models and datasets. Specifically, our PDSG achieves 100% attack success rate on ImageNet, and our SDA obtains 82% attack success rate by modifying only 0.24% of the pixels on CIFAR10DVS. The code is available at https://github.com/ryime/PDSG-SDA .



## **17. Fast Preemption: Forward-Backward Cascade Learning for Efficient and Transferable Preemptive Adversarial Defense**

cs.CR

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2407.15524v7) [paper-pdf](http://arxiv.org/pdf/2407.15524v7)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: {Deep learning has made significant strides but remains susceptible to adversarial attacks, undermining its reliability. Most existing research addresses these threats after attacks happen. A growing direction explores preemptive defenses like mitigating adversarial threats proactively, offering improved robustness but at cost of efficiency and transferability. This paper introduces Fast Preemption, a novel preemptive adversarial defense that overcomes efficiency challenges while achieving state-of-the-art robustness and transferability, requiring no prior knowledge of attacks and target models. We propose a forward-backward cascade learning algorithm, which generates protective perturbations by combining forward propagation for rapid convergence with iterative backward propagation to prevent overfitting. Executing in just three iterations, Fast Preemption outperforms existing training-time, test-time, and preemptive defenses. Additionally, we introduce an adaptive reversion attack to assess the reliability of preemptive defenses, demonstrating that our approach remains secure in realistic attack scenarios.



## **18. Scale-Invariant Adversarial Attack against Arbitrary-scale Super-resolution**

cs.CV

15 pages, accepted by TIFS 2025

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04385v1) [paper-pdf](http://arxiv.org/pdf/2503.04385v1)

**Authors**: Yihao Huang, Xin Luo, Qing Guo, Felix Juefei-Xu, Xiaojun Jia, Weikai Miao, Geguang Pu, Yang Liu

**Abstract**: The advent of local continuous image function (LIIF) has garnered significant attention for arbitrary-scale super-resolution (SR) techniques. However, while the vulnerabilities of fixed-scale SR have been assessed, the robustness of continuous representation-based arbitrary-scale SR against adversarial attacks remains an area warranting further exploration. The elaborately designed adversarial attacks for fixed-scale SR are scale-dependent, which will cause time-consuming and memory-consuming problems when applied to arbitrary-scale SR. To address this concern, we propose a simple yet effective ``scale-invariant'' SR adversarial attack method with good transferability, termed SIAGT. Specifically, we propose to construct resource-saving attacks by exploiting finite discrete points of continuous representation. In addition, we formulate a coordinate-dependent loss to enhance the cross-model transferability of the attack. The attack can significantly deteriorate the SR images while introducing imperceptible distortion to the targeted low-resolution (LR) images. Experiments carried out on three popular LIIF-based SR approaches and four classical SR datasets show remarkable attack performance and transferability of SIAGT.



## **19. $σ$-zero: Gradient-based Optimization of $\ell_0$-norm Adversarial Examples**

cs.LG

Paper accepted at International Conference on Learning  Representations (ICLR 2025). Code available at  https://github.com/sigma0-advx/sigma-zero

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2402.01879v3) [paper-pdf](http://arxiv.org/pdf/2402.01879v3)

**Authors**: Antonio Emanuele Cinà, Francesco Villani, Maura Pintor, Lea Schönherr, Battista Biggio, Marcello Pelillo

**Abstract**: Evaluating the adversarial robustness of deep networks to gradient-based attacks is challenging. While most attacks consider $\ell_2$- and $\ell_\infty$-norm constraints to craft input perturbations, only a few investigate sparse $\ell_1$- and $\ell_0$-norm attacks. In particular, $\ell_0$-norm attacks remain the least studied due to the inherent complexity of optimizing over a non-convex and non-differentiable constraint. However, evaluating adversarial robustness under these attacks could reveal weaknesses otherwise left untested with more conventional $\ell_2$- and $\ell_\infty$-norm attacks. In this work, we propose a novel $\ell_0$-norm attack, called $\sigma$-zero, which leverages a differentiable approximation of the $\ell_0$ norm to facilitate gradient-based optimization, and an adaptive projection operator to dynamically adjust the trade-off between loss minimization and perturbation sparsity. Extensive evaluations using MNIST, CIFAR10, and ImageNet datasets, involving robust and non-robust models, show that $\sigma$\texttt{-zero} finds minimum $\ell_0$-norm adversarial examples without requiring any time-consuming hyperparameter tuning, and that it outperforms all competing sparse attacks in terms of success rate, perturbation size, and efficiency.



## **20. One-Shot is Enough: Consolidating Multi-Turn Attacks into Efficient Single-Turn Prompts for LLMs**

cs.CL

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04856v1) [paper-pdf](http://arxiv.org/pdf/2503.04856v1)

**Authors**: Junwoo Ha, Hyunjun Kim, Sangyoon Yu, Haon Park, Ashkan Yousefpour, Yuna Park, Suhyun Kim

**Abstract**: Despite extensive safety enhancements in large language models (LLMs), multi-turn "jailbreak" conversations crafted by skilled human adversaries can still breach even the most sophisticated guardrails. However, these multi-turn attacks demand considerable manual effort, limiting their scalability. In this work, we introduce a novel approach called Multi-turn-to-Single-turn (M2S) that systematically converts multi-turn jailbreak prompts into single-turn attacks. Specifically, we propose three conversion strategies - Hyphenize, Numberize, and Pythonize - each preserving sequential context yet packaging it in a single query. Our experiments on the Multi-turn Human Jailbreak (MHJ) dataset show that M2S often increases or maintains high Attack Success Rates (ASRs) compared to original multi-turn conversations. Notably, using a StrongREJECT-based evaluation of harmfulness, M2S achieves up to 95.9% ASR on Mistral-7B and outperforms original multi-turn prompts by as much as 17.5% in absolute improvement on GPT-4o. Further analysis reveals that certain adversarial tactics, when consolidated into a single prompt, exploit structural formatting cues to evade standard policy checks. These findings underscore that single-turn attacks - despite being simpler and cheaper to conduct - can be just as potent, if not more, than their multi-turn counterparts. Our findings underscore the urgent need to reevaluate and reinforce LLM safety strategies, given how adversarial queries can be compacted into a single prompt while still retaining sufficient complexity to bypass existing safety measures.



## **21. From Pixels to Trajectory: Universal Adversarial Example Detection via Temporal Imprints**

cs.CR

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04853v1) [paper-pdf](http://arxiv.org/pdf/2503.04853v1)

**Authors**: Yansong Gao, Huaibing Peng, Hua Ma, Zhiyang Dai, Shuo Wang, Hongsheng Hu, Anmin Fu, Minhui Xue

**Abstract**: For the first time, we unveil discernible temporal (or historical) trajectory imprints resulting from adversarial example (AE) attacks. Standing in contrast to existing studies all focusing on spatial (or static) imprints within the targeted underlying victim models, we present a fresh temporal paradigm for understanding these attacks. Of paramount discovery is that these imprints are encapsulated within a single loss metric, spanning universally across diverse tasks such as classification and regression, and modalities including image, text, and audio. Recognizing the distinct nature of loss between adversarial and clean examples, we exploit this temporal imprint for AE detection by proposing TRAIT (TRaceable Adversarial temporal trajectory ImprinTs). TRAIT operates under minimal assumptions without prior knowledge of attacks, thereby framing the detection challenge as a one-class classification problem. However, detecting AEs is still challenged by significant overlaps between the constructed synthetic losses of adversarial and clean examples due to the absence of ground truth for incoming inputs. TRAIT addresses this challenge by converting the synthetic loss into a spectrum signature, using the technique of Fast Fourier Transform to highlight the discrepancies, drawing inspiration from the temporal nature of the imprints, analogous to time-series signals. Across 12 AE attacks including SMACK (USENIX Sec'2023), TRAIT demonstrates consistent outstanding performance across comprehensively evaluated modalities, tasks, datasets, and model architectures. In all scenarios, TRAIT achieves an AE detection accuracy exceeding 97%, often around 99%, while maintaining a false rejection rate of 1%. TRAIT remains effective under the formulated strong adaptive attacks.



## **22. Robust Eavesdropping in the Presence of Adversarial Communications for RF Fingerprinting**

eess.SP

11 pages, 6 figures

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04120v1) [paper-pdf](http://arxiv.org/pdf/2503.04120v1)

**Authors**: Andrew Yuan, Rajeev Sahay

**Abstract**: Deep learning is an effective approach for performing radio frequency (RF) fingerprinting, which aims to identify the transmitter corresponding to received RF signals. However, beyond the intended receiver, malicious eavesdroppers can also intercept signals and attempt to fingerprint transmitters communicating over a wireless channel. Recent studies suggest that transmitters can counter such threats by embedding deep learning-based transferable adversarial attacks in their signals before transmission. In this work, we develop a time-frequency-based eavesdropper architecture that is capable of withstanding such transferable adversarial perturbations and thus able to perform effective RF fingerprinting. We theoretically demonstrate that adversarial perturbations injected by a transmitter are confined to specific time-frequency regions that are insignificant during inference, directly increasing fingerprinting accuracy on perturbed signals intercepted by the eavesdropper. Empirical evaluations on a real-world dataset validate our theoretical findings, showing that deep learning-based RF fingerprinting eavesdroppers can achieve classification performance comparable to the intended receiver, despite efforts made by the transmitter to deceive the eavesdropper. Our framework reveals that relying on transferable adversarial attacks may not be sufficient to prevent eavesdroppers from successfully fingerprinting transmissions in next-generation deep learning-based communications systems.



## **23. Adversarial Decoding: Generating Readable Documents for Adversarial Objectives**

cs.CL

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2410.02163v2) [paper-pdf](http://arxiv.org/pdf/2410.02163v2)

**Authors**: Collin Zhang, Tingwei Zhang, Vitaly Shmatikov

**Abstract**: We design, implement, and evaluate adversarial decoding, a new, generic text generation technique that produces readable documents for different adversarial objectives. Prior methods either produce easily detectable gibberish, or cannot handle objectives that include embedding similarity. In particular, they only work for direct attacks (such as jailbreaking) and cannot produce adversarial text for realistic indirect injection, e.g., documents that (1) are retrieved in RAG systems in response to broad classes of queries, and also (2) adversarially influence subsequent generation. We also show that fluency (low perplexity) is not sufficient to evade filtering. We measure the effectiveness of adversarial decoding for different objectives, including RAG poisoning, jailbreaking, and evasion of defensive filters, and demonstrate that it outperforms existing methods while producing readable adversarial documents.



## **24. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2502.16750v2) [paper-pdf](http://arxiv.org/pdf/2502.16750v2)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehnaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.



## **25. Task-Agnostic Attacks Against Vision Foundation Models**

cs.CV

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03842v1) [paper-pdf](http://arxiv.org/pdf/2503.03842v1)

**Authors**: Brian Pulfer, Yury Belousov, Vitaliy Kinakh, Teddy Furon, Slava Voloshynovskiy

**Abstract**: The study of security in machine learning mainly focuses on downstream task-specific attacks, where the adversarial example is obtained by optimizing a loss function specific to the downstream task. At the same time, it has become standard practice for machine learning practitioners to adopt publicly available pre-trained vision foundation models, effectively sharing a common backbone architecture across a multitude of applications such as classification, segmentation, depth estimation, retrieval, question-answering and more. The study of attacks on such foundation models and their impact to multiple downstream tasks remains vastly unexplored. This work proposes a general framework that forges task-agnostic adversarial examples by maximally disrupting the feature representation obtained with foundation models. We extensively evaluate the security of the feature representations obtained by popular vision foundation models by measuring the impact of this attack on multiple downstream tasks and its transferability between models.



## **26. Improving LLM Safety Alignment with Dual-Objective Optimization**

cs.CL

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03710v1) [paper-pdf](http://arxiv.org/pdf/2503.03710v1)

**Authors**: Xuandong Zhao, Will Cai, Tianneng Shi, David Huang, Licong Lin, Song Mei, Dawn Song

**Abstract**: Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at https://github.com/wicai24/DOOR-Alignment



## **27. CLIP is Strong Enough to Fight Back: Test-time Counterattacks towards Zero-shot Adversarial Robustness of CLIP**

cs.CV

Accepted to CVPR 2025

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03613v1) [paper-pdf](http://arxiv.org/pdf/2503.03613v1)

**Authors**: Songlong Xing, Zhengyu Zhao, Nicu Sebe

**Abstract**: Despite its prevalent use in image-text matching tasks in a zero-shot manner, CLIP has been shown to be highly vulnerable to adversarial perturbations added onto images. Recent studies propose to finetune the vision encoder of CLIP with adversarial samples generated on the fly, and show improved robustness against adversarial attacks on a spectrum of downstream datasets, a property termed as zero-shot robustness. In this paper, we show that malicious perturbations that seek to maximise the classification loss lead to `falsely stable' images, and propose to leverage the pre-trained vision encoder of CLIP to counterattack such adversarial images during inference to achieve robustness. Our paradigm is simple and training-free, providing the first method to defend CLIP from adversarial attacks at test time, which is orthogonal to existing methods aiming to boost zero-shot adversarial robustness of CLIP. We conduct experiments across 16 classification datasets, and demonstrate stable and consistent gains compared to test-time defence methods adapted from existing adversarial robustness studies that do not rely on external networks, without noticeably impairing performance on clean images. We also show that our paradigm can be employed on CLIP models that have been adversarially finetuned to further enhance their robustness at test time. Our code is available \href{https://github.com/Sxing2/CLIP-Test-time-Counterattacks}{here}.



## **28. Adversarial Training for Multimodal Large Language Models against Jailbreak Attacks**

cs.CV

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.04833v1) [paper-pdf](http://arxiv.org/pdf/2503.04833v1)

**Authors**: Liming Lu, Shuchao Pang, Siyuan Liang, Haotian Zhu, Xiyu Zeng, Aishan Liu, Yunhuai Liu, Yongbin Zhou

**Abstract**: Multimodal large language models (MLLMs) have made remarkable strides in cross-modal comprehension and generation tasks. However, they remain vulnerable to jailbreak attacks, where crafted perturbations bypass security guardrails and elicit harmful outputs. In this paper, we present the first adversarial training (AT) paradigm tailored to defend against jailbreak attacks during the MLLM training phase. Extending traditional AT to this domain poses two critical challenges: efficiently tuning massive parameters and ensuring robustness against attacks across multiple modalities. To address these challenges, we introduce Projection Layer Against Adversarial Training (ProEAT), an end-to-end AT framework. ProEAT incorporates a projector-based adversarial training architecture that efficiently handles large-scale parameters while maintaining computational feasibility by focusing adversarial training on a lightweight projector layer instead of the entire model; additionally, we design a dynamic weight adjustment mechanism that optimizes the loss function's weight allocation based on task demands, streamlining the tuning process. To enhance defense performance, we propose a joint optimization strategy across visual and textual modalities, ensuring robust resistance to jailbreak attacks originating from either modality. Extensive experiments conducted on five major jailbreak attack methods across three mainstream MLLMs demonstrate the effectiveness of our approach. ProEAT achieves state-of-the-art defense performance, outperforming existing baselines by an average margin of +34% across text and image modalities, while incurring only a 1% reduction in clean accuracy. Furthermore, evaluations on real-world embodied intelligent systems highlight the practical applicability of our framework, paving the way for the development of more secure and reliable multimodal systems.



## **29. Adversarial Example Based Fingerprinting for Robust Copyright Protection in Split Learning**

cs.CR

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.04825v1) [paper-pdf](http://arxiv.org/pdf/2503.04825v1)

**Authors**: Zhangting Lin, Mingfu Xue, Kewei Chen, Wenmao Liu, Xiang Gao, Leo Yu Zhang, Jian Wang, Yushu Zhang

**Abstract**: Currently, deep learning models are easily exposed to data leakage risks. As a distributed model, Split Learning thus emerged as a solution to address this issue. The model is splitted to avoid data uploading to the server and reduce computing requirements while ensuring data privacy and security. However, the transmission of data between clients and server creates a potential vulnerability. In particular, model is vulnerable to intellectual property (IP) infringement such as piracy. Alarmingly, a dedicated copyright protection framework tailored for Split Learning models is still lacking. To this end, we propose the first copyright protection scheme for Split Learning model, leveraging fingerprint to ensure effective and robust copyright protection. The proposed method first generates a set of specifically designed adversarial examples. Then, we select those examples that would induce misclassifications to form the fingerprint set. These adversarial examples are embedded as fingerprints into the model during the training process. Exhaustive experiments highlight the effectiveness of the scheme. This is demonstrated by a remarkable fingerprint verification success rate (FVSR) of 100% on MNIST, 98% on CIFAR-10, and 100% on ImageNet, respectively. Meanwhile, the model's accuracy only decreases slightly, indicating that the embedded fingerprints do not compromise model performance. Even under label inference attack, our approach consistently achieves a high fingerprint verification success rate that ensures robust verification.



## **30. AttackSeqBench: Benchmarking Large Language Models' Understanding of Sequential Patterns in Cyber Attacks**

cs.CR

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03170v1) [paper-pdf](http://arxiv.org/pdf/2503.03170v1)

**Authors**: Javier Yong, Haokai Ma, Yunshan Ma, Anis Yusof, Zhenkai Liang, Ee-Chien Chang

**Abstract**: The observations documented in Cyber Threat Intelligence (CTI) reports play a critical role in describing adversarial behaviors, providing valuable insights for security practitioners to respond to evolving threats. Recent advancements of Large Language Models (LLMs) have demonstrated significant potential in various cybersecurity applications, including CTI report understanding and attack knowledge graph construction. While previous works have proposed benchmarks that focus on the CTI extraction ability of LLMs, the sequential characteristic of adversarial behaviors within CTI reports remains largely unexplored, which holds considerable significance in developing a comprehensive understanding of how adversaries operate. To address this gap, we introduce AttackSeqBench, a benchmark tailored to systematically evaluate LLMs' capability to understand and reason attack sequences in CTI reports. Our benchmark encompasses three distinct Question Answering (QA) tasks, each task focuses on the varying granularity in adversarial behavior. To alleviate the laborious effort of QA construction, we carefully design an automated dataset construction pipeline to create scalable and well-formulated QA datasets based on real-world CTI reports. To ensure the quality of our dataset, we adopt a hybrid approach of combining human evaluation and systematic evaluation metrics. We conduct extensive experiments and analysis with both fast-thinking and slow-thinking LLMs, while highlighting their strengths and limitations in analyzing the sequential patterns in cyber attacks. The overarching goal of this work is to provide a benchmark that advances LLM-driven CTI report understanding and fosters its application in real-world cybersecurity operations. Our dataset and code are available at https://github.com/Javiery3889/AttackSeqBench .



## **31. Exploiting Vulnerabilities in Speech Translation Systems through Targeted Adversarial Attacks**

cs.SD

Preprint,17 pages, 17 figures

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.00957v2) [paper-pdf](http://arxiv.org/pdf/2503.00957v2)

**Authors**: Chang Liu, Haolin Wu, Xi Yang, Kui Zhang, Cong Wu, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Qing Guo, Jie Zhang

**Abstract**: As speech translation (ST) systems become increasingly prevalent, understanding their vulnerabilities is crucial for ensuring robust and reliable communication. However, limited work has explored this issue in depth. This paper explores methods of compromising these systems through imperceptible audio manipulations. Specifically, we present two innovative approaches: (1) the injection of perturbation into source audio, and (2) the generation of adversarial music designed to guide targeted translation, while also conducting more practical over-the-air attacks in the physical world. Our experiments reveal that carefully crafted audio perturbations can mislead translation models to produce targeted, harmful outputs, while adversarial music achieve this goal more covertly, exploiting the natural imperceptibility of music. These attacks prove effective across multiple languages and translation models, highlighting a systemic vulnerability in current ST architectures. The implications of this research extend beyond immediate security concerns, shedding light on the interpretability and robustness of neural speech processing systems. Our findings underscore the need for advanced defense mechanisms and more resilient architectures in the realm of audio systems. More details and samples can be found at https://adv-st.github.io.



## **32. Detecting Adversarial Data using Perturbation Forgery**

cs.CV

Accepted as a conference paper at CVPR 2025

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2405.16226v4) [paper-pdf](http://arxiv.org/pdf/2405.16226v4)

**Authors**: Qian Wang, Chen Li, Yuchen Luo, Hefei Ling, Shijuan Huang, Ruoxi Jia, Ning Yu

**Abstract**: As a defense strategy against adversarial attacks, adversarial detection aims to identify and filter out adversarial data from the data flow based on discrepancies in distribution and noise patterns between natural and adversarial data. Although previous detection methods achieve high performance in detecting gradient-based adversarial attacks, new attacks based on generative models with imbalanced and anisotropic noise patterns evade detection. Even worse, the significant inference time overhead and limited performance against unseen attacks make existing techniques impractical for real-world use. In this paper, we explore the proximity relationship among adversarial noise distributions and demonstrate the existence of an open covering for these distributions. By training on the open covering of adversarial noise distributions, a detector with strong generalization performance against various types of unseen attacks can be developed. Based on this insight, we heuristically propose Perturbation Forgery, which includes noise distribution perturbation, sparse mask generation, and pseudo-adversarial data production, to train an adversarial detector capable of detecting any unseen gradient-based, generative-based, and physical adversarial attacks. Comprehensive experiments conducted on multiple general and facial datasets, with a wide spectrum of attacks, validate the strong generalization of our method.



## **33. An Undetectable Watermark for Generative Image Models**

cs.CR

ICLR 2025

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2410.07369v3) [paper-pdf](http://arxiv.org/pdf/2410.07369v3)

**Authors**: Sam Gunn, Xuandong Zhao, Dawn Song

**Abstract**: We present the first undetectable watermarking scheme for generative image models. Undetectability ensures that no efficient adversary can distinguish between watermarked and un-watermarked images, even after making many adaptive queries. In particular, an undetectable watermark does not degrade image quality under any efficiently computable metric. Our scheme works by selecting the initial latents of a diffusion model using a pseudorandom error-correcting code (Christ and Gunn, 2024), a strategy which guarantees undetectability and robustness. We experimentally demonstrate that our watermarks are quality-preserving and robust using Stable Diffusion 2.1. Our experiments verify that, in contrast to every prior scheme we tested, our watermark does not degrade image quality. Our experiments also demonstrate robustness: existing watermark removal attacks fail to remove our watermark from images without significantly degrading the quality of the images. Finally, we find that we can robustly encode 512 bits in our watermark, and up to 2500 bits when the images are not subjected to watermark removal attacks. Our code is available at https://github.com/XuandongZhao/PRC-Watermark.



## **34. LLM Misalignment via Adversarial RLHF Platforms**

cs.LG

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.03039v1) [paper-pdf](http://arxiv.org/pdf/2503.03039v1)

**Authors**: Erfan Entezami, Ali Naseh

**Abstract**: Reinforcement learning has shown remarkable performance in aligning language models with human preferences, leading to the rise of attention towards developing RLHF platforms. These platforms enable users to fine-tune models without requiring any expertise in developing complex machine learning algorithms. While these platforms offer useful features such as reward modeling and RLHF fine-tuning, their security and reliability remain largely unexplored. Given the growing adoption of RLHF and open-source RLHF frameworks, we investigate the trustworthiness of these systems and their potential impact on behavior of LLMs. In this paper, we present an attack targeting publicly available RLHF tools. In our proposed attack, an adversarial RLHF platform corrupts the LLM alignment process by selectively manipulating data samples in the preference dataset. In this scenario, when a user's task aligns with the attacker's objective, the platform manipulates a subset of the preference dataset that contains samples related to the attacker's target. This manipulation results in a corrupted reward model, which ultimately leads to the misalignment of the language model. Our results demonstrate that such an attack can effectively steer LLMs toward undesirable behaviors within the targeted domains. Our work highlights the critical need to explore the vulnerabilities of RLHF platforms and their potential to cause misalignment in LLMs during the RLHF fine-tuning process.



## **35. Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis**

cs.CR

13 pages

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.02986v2) [paper-pdf](http://arxiv.org/pdf/2503.02986v2)

**Authors**: Jeonghwan Park, Niall McLaughlin, Ihsen Alouani

**Abstract**: Adversarial attacks remain a significant threat that can jeopardize the integrity of Machine Learning (ML) models. In particular, query-based black-box attacks can generate malicious noise without having access to the victim model's architecture, making them practical in real-world contexts. The community has proposed several defenses against adversarial attacks, only to be broken by more advanced and adaptive attack strategies. In this paper, we propose a framework that detects if an adversarial noise instance is being generated. Unlike existing stateful defenses that detect adversarial noise generation by monitoring the input space, our approach learns adversarial patterns in the input update similarity space. In fact, we propose to observe a new metric called Delta Similarity (DS), which we show it captures more efficiently the adversarial behavior. We evaluate our approach against 8 state-of-the-art attacks, including adaptive attacks, where the adversary is aware of the defense and tries to evade detection. We find that our approach is significantly more robust than existing defenses both in terms of specificity and sensitivity.



## **36. Decentralized Adversarial Training over Graphs**

cs.LG

arXiv admin note: text overlap with arXiv:2303.01936

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2303.13326v2) [paper-pdf](http://arxiv.org/pdf/2303.13326v2)

**Authors**: Ying Cao, Elsa Rizk, Stefan Vlaski, Ali H. Sayed

**Abstract**: The vulnerability of machine learning models to adversarial attacks has been attracting considerable attention in recent years. Most existing studies focus on the behavior of stand-alone single-agent learners. In comparison, this work studies adversarial training over graphs, where individual agents are subjected to perturbations of varied strength levels across space. It is expected that interactions by linked agents, and the heterogeneity of the attack models that are possible over the graph, can help enhance robustness in view of the coordination power of the group. Using a min-max formulation of distributed learning, we develop a decentralized adversarial training framework for multi-agent systems. Specifically, we devise two decentralized adversarial training algorithms by relying on two popular decentralized learning strategies--diffusion and consensus. We analyze the convergence properties of the proposed framework for strongly-convex, convex, and non-convex environments, and illustrate the enhanced robustness to adversarial attacks.



## **37. Towards Safe AI Clinicians: A Comprehensive Study on Large Language Model Jailbreaking in Healthcare**

cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2501.18632v2) [paper-pdf](http://arxiv.org/pdf/2501.18632v2)

**Authors**: Hang Zhang, Qian Lou, Yanshan Wang

**Abstract**: Large language models (LLMs) are increasingly utilized in healthcare applications. However, their deployment in clinical practice raises significant safety concerns, including the potential spread of harmful information. This study systematically assesses the vulnerabilities of seven LLMs to three advanced black-box jailbreaking techniques within medical contexts. To quantify the effectiveness of these techniques, we propose an automated and domain-adapted agentic evaluation pipeline. Experiment results indicate that leading commercial and open-source LLMs are highly vulnerable to medical jailbreaking attacks. To bolster model safety and reliability, we further investigate the effectiveness of Continual Fine-Tuning (CFT) in defending against medical adversarial attacks. Our findings underscore the necessity for evolving attack methods evaluation, domain-specific safety alignment, and LLM safety-utility balancing. This research offers actionable insights for advancing the safety and reliability of AI clinicians, contributing to ethical and effective AI deployment in healthcare.



## **38. Assessing Robustness via Score-Based Adversarial Image Generation**

cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2310.04285v3) [paper-pdf](http://arxiv.org/pdf/2310.04285v3)

**Authors**: Marcel Kollovieh, Lukas Gosch, Marten Lienen, Yan Scholten, Leo Schwinn, Stephan Günnemann

**Abstract**: Most adversarial attacks and defenses focus on perturbations within small $\ell_p$-norm constraints. However, $\ell_p$ threat models cannot capture all relevant semantics-preserving perturbations, and hence, the scope of robustness evaluations is limited. In this work, we introduce Score-Based Adversarial Generation (ScoreAG), a novel framework that leverages the advancements in score-based generative models to generate unrestricted adversarial examples that overcome the limitations of $\ell_p$-norm constraints. Unlike traditional methods, ScoreAG maintains the core semantics of images while generating adversarial examples, either by transforming existing images or synthesizing new ones entirely from scratch. We further exploit the generative capability of ScoreAG to purify images, empirically enhancing the robustness of classifiers. Our extensive empirical evaluation demonstrates that ScoreAG improves upon the majority of state-of-the-art attacks and defenses across multiple benchmarks. This work highlights the importance of investigating adversarial examples bounded by semantics rather than $\ell_p$-norm constraints. ScoreAG represents an important step towards more encompassing robustness assessments.



## **39. Realizing Quantum Adversarial Defense on a Trapped-ion Quantum Processor**

quant-ph

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02436v1) [paper-pdf](http://arxiv.org/pdf/2503.02436v1)

**Authors**: Alex Jin, Tarun Dutta, Anh Tu Ngo, Anupam Chattopadhyay, Manas Mukherjee

**Abstract**: Classification is a fundamental task in machine learning, typically performed using classical models. Quantum machine learning (QML), however, offers distinct advantages, such as enhanced representational power through high-dimensional Hilbert spaces and energy-efficient reversible gate operations. Despite these theoretical benefits, the robustness of QML classifiers against adversarial attacks and inherent quantum noise remains largely under-explored. In this work, we implement a data re-uploading-based quantum classifier on an ion-trap quantum processor using a single qubit to assess its resilience under realistic conditions. We introduce a novel convolutional quantum classifier architecture leveraging data re-uploading and demonstrate its superior robustness on the MNIST dataset. Additionally, we quantify the effects of polarization noise in a realistic setting, where both bit and phase noises are present, further validating the classifier's robustness. Our findings provide insights into the practical security and reliability of quantum classifiers, bridging the gap between theoretical potential and real-world deployment.



## **40. Trace of the Times: Rootkit Detection through Temporal Anomalies in Kernel Activity**

cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02402v1) [paper-pdf](http://arxiv.org/pdf/2503.02402v1)

**Authors**: Max Landauer, Leonhard Alton, Martina Lindorfer, Florian Skopik, Markus Wurzenberger, Wolfgang Hotwagner

**Abstract**: Kernel rootkits provide adversaries with permanent high-privileged access to compromised systems and are often a key element of sophisticated attack chains. At the same time, they enable stealthy operation and are thus difficult to detect. Thereby, they inject code into kernel functions to appear invisible to users, for example, by manipulating file enumerations. Existing detection approaches are insufficient, because they rely on signatures that are unable to detect novel rootkits or require domain knowledge about the rootkits to be detected. To overcome this challenge, our approach leverages the fact that runtimes of kernel functions targeted by rootkits increase when additional code is executed. The framework outlined in this paper injects probes into the kernel to measure time stamps of functions within relevant system calls, computes distributions of function execution times, and uses statistical tests to detect time shifts. The evaluation of our open-source implementation on publicly available data sets indicates high detection accuracy with an F1 score of 98.7\% across five scenarios with varying system states.



## **41. Evaluating the Robustness of LiDAR Point Cloud Tracking Against Adversarial Attack**

cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2410.20893v2) [paper-pdf](http://arxiv.org/pdf/2410.20893v2)

**Authors**: Shengjing Tian, Yinan Han, Xiantong Zhao, Bin Liu, Xiuping Liu

**Abstract**: In this study, we delve into the robustness of neural network-based LiDAR point cloud tracking models under adversarial attacks, a critical aspect often overlooked in favor of performance enhancement. These models, despite incorporating advanced architectures like Transformer or Bird's Eye View (BEV), tend to neglect robustness in the face of challenges such as adversarial attacks, domain shifts, or data corruption. We instead focus on the robustness of the tracking models under the threat of adversarial attacks. We begin by establishing a unified framework for conducting adversarial attacks within the context of 3D object tracking, which allows us to thoroughly investigate both white-box and black-box attack strategies. For white-box attacks, we tailor specific loss functions to accommodate various tracking paradigms and extend existing methods such as FGSM, C\&W, and PGD to the point cloud domain. In addressing black-box attack scenarios, we introduce a novel transfer-based approach, the Target-aware Perturbation Generation (TAPG) algorithm, with the dual objectives of achieving high attack performance and maintaining low perceptibility. This method employs a heuristic strategy to enforce sparse attack constraints and utilizes random sub-vector factorization to bolster transferability. Our experimental findings reveal a significant vulnerability in advanced tracking methods when subjected to both black-box and white-box attacks, underscoring the necessity for incorporating robustness against adversarial attacks into the design of LiDAR point cloud tracking models. Notably, compared to existing methods, the TAPG also strikes an optimal balance between the effectiveness of the attack and the concealment of the perturbations.



## **42. NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary**

cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.00063v2) [paper-pdf](http://arxiv.org/pdf/2503.00063v2)

**Authors**: Zezeng Li, Xiaoyu Du, Na Lei, Liming Chen, Weimin Wang

**Abstract**: Adversarial attacks exploit the vulnerability of deep models against adversarial samples. Existing point cloud attackers are tailored to specific models, iteratively optimizing perturbations based on gradients in either a white-box or black-box setting. Despite their promising attack performance, they often struggle to produce transferable adversarial samples due to overfitting the specific parameters of surrogate models. To overcome this issue, we shift our focus to the data distribution itself and introduce a novel approach named NoPain, which employs optimal transport (OT) to identify the inherent singular boundaries of the data manifold for cross-network point cloud attacks. Specifically, we first calculate the OT mapping from noise to the target feature space, then identify singular boundaries by locating non-differentiable positions. Finally, we sample along singular boundaries to generate adversarial point clouds. Once the singular boundaries are determined, NoPain can efficiently produce adversarial samples without the need of iterative updates or guidance from the surrogate classifiers. Extensive experiments demonstrate that the proposed end-to-end method outperforms baseline approaches in terms of both transferability and efficiency, while also maintaining notable advantages even against defense strategies. The source code will be publicly available.



## **43. Game-Theoretic Defenses for Robust Conformal Prediction Against Adversarial Attacks in Medical Imaging**

cs.LG

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2411.04376v2) [paper-pdf](http://arxiv.org/pdf/2411.04376v2)

**Authors**: Rui Luo, Jie Bao, Zhixin Zhou, Chuangyin Dang

**Abstract**: Adversarial attacks pose significant threats to the reliability and safety of deep learning models, especially in critical domains such as medical imaging. This paper introduces a novel framework that integrates conformal prediction with game-theoretic defensive strategies to enhance model robustness against both known and unknown adversarial perturbations. We address three primary research questions: constructing valid and efficient conformal prediction sets under known attacks (RQ1), ensuring coverage under unknown attacks through conservative thresholding (RQ2), and determining optimal defensive strategies within a zero-sum game framework (RQ3). Our methodology involves training specialized defensive models against specific attack types and employing maximum and minimum classifiers to aggregate defenses effectively. Extensive experiments conducted on the MedMNIST datasets, including PathMNIST, OrganAMNIST, and TissueMNIST, demonstrate that our approach maintains high coverage guarantees while minimizing prediction set sizes. The game-theoretic analysis reveals that the optimal defensive strategy often converges to a singular robust model, outperforming uniform and simple strategies across all evaluated datasets. This work advances the state-of-the-art in uncertainty quantification and adversarial robustness, providing a reliable mechanism for deploying deep learning models in adversarial environments.



## **44. TPIA: Towards Target-specific Prompt Injection Attack against Code-oriented Large Language Models**

cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2407.09164v5) [paper-pdf](http://arxiv.org/pdf/2407.09164v5)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully exploited to simplify and facilitate programming. Unfortunately, a few pioneering works revealed that these Code LLMs are vulnerable to backdoor and adversarial attacks. The former poisons the training data or model parameters, hijacking the LLMs to generate malicious code snippets when encountering the trigger. The latter crafts malicious adversarial input codes to reduce the quality of the generated codes. In this paper, we reveal that both attacks have some inherent limitations: backdoor attacks rely on the adversary's capability of controlling the model training process, which may not be practical; adversarial attacks struggle with fulfilling specific malicious purposes. To alleviate these problems, this paper presents a novel attack paradigm against Code LLMs, namely target-specific prompt injection attack (TPIA). TPIA generates non-functional perturbations containing the information of malicious instructions and inserts them into the victim's code context by spreading them into potentially used dependencies (e.g., packages or RAG's knowledge base). It induces the Code LLMs to generate attacker-specified malicious code snippets at the target location. In general, we compress the attacker-specified malicious objective into the perturbation by adversarial optimization based on greedy token search. We collect 13 representative malicious objectives to design 31 threat cases for three popular programming languages. We show that our TPIA can successfully attack three representative open-source Code LLMs (with an attack success rate of up to 97.9%) and two mainstream commercial Code LLM-integrated applications (with an attack success rate of over 90%) in all threat cases, using only a 12-token non-functional perturbation.



## **45. Prompt-driven Transferable Adversarial Attack on Person Re-Identification with Attribute-aware Textual Inversion**

cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2502.19697v2) [paper-pdf](http://arxiv.org/pdf/2502.19697v2)

**Authors**: Yuan Bian, Min Liu, Yunqi Yi, Xueping Wang, Yaonan Wang

**Abstract**: Person re-identification (re-id) models are vital in security surveillance systems, requiring transferable adversarial attacks to explore the vulnerabilities of them. Recently, vision-language models (VLM) based attacks have shown superior transferability by attacking generalized image and textual features of VLM, but they lack comprehensive feature disruption due to the overemphasis on discriminative semantics in integral representation. In this paper, we introduce the Attribute-aware Prompt Attack (AP-Attack), a novel method that leverages VLM's image-text alignment capability to explicitly disrupt fine-grained semantic features of pedestrian images by destroying attribute-specific textual embeddings. To obtain personalized textual descriptions for individual attributes, textual inversion networks are designed to map pedestrian images to pseudo tokens that represent semantic embeddings, trained in the contrastive learning manner with images and a predefined prompt template that explicitly describes the pedestrian attributes. Inverted benign and adversarial fine-grained textual semantics facilitate attacker in effectively conducting thorough disruptions, enhancing the transferability of adversarial examples. Extensive experiments show that AP-Attack achieves state-of-the-art transferability, significantly outperforming previous methods by 22.9% on mean Drop Rate in cross-model&dataset attack scenarios.



## **46. Adversarial Tokenization**

cs.CL

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02174v1) [paper-pdf](http://arxiv.org/pdf/2503.02174v1)

**Authors**: Renato Lui Geh, Zilei Shao, Guy Van den Broeck

**Abstract**: Current LLM pipelines account for only one possible tokenization for a given string, ignoring exponentially many alternative tokenizations during training and inference. For example, the standard Llama3 tokenization of penguin is [p,enguin], yet [peng,uin] is another perfectly valid alternative. In this paper, we show that despite LLMs being trained solely on one tokenization, they still retain semantic understanding of other tokenizations, raising questions about their implications in LLM safety. Put succinctly, we answer the following question: can we adversarially tokenize an obviously malicious string to evade safety and alignment restrictions? We show that not only is adversarial tokenization an effective yet previously neglected axis of attack, but it is also competitive against existing state-of-the-art adversarial approaches without changing the text of the harmful request. We empirically validate this exploit across three state-of-the-art LLMs and adversarial datasets, revealing a previously unknown vulnerability in subword models.



## **47. HoSNN: Adversarially-Robust Homeostatic Spiking Neural Networks with Adaptive Firing Thresholds**

cs.NE

Accepted by TMLR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2308.10373v4) [paper-pdf](http://arxiv.org/pdf/2308.10373v4)

**Authors**: Hejia Geng, Peng Li

**Abstract**: While spiking neural networks (SNNs) offer a promising neurally-inspired model of computation, they are vulnerable to adversarial attacks. We present the first study that draws inspiration from neural homeostasis to design a threshold-adapting leaky integrate-and-fire (TA-LIF) neuron model and utilize TA-LIF neurons to construct the adversarially robust homeostatic SNNs (HoSNNs) for improved robustness. The TA-LIF model incorporates a self-stabilizing dynamic thresholding mechanism, offering a local feedback control solution to the minimization of each neuron's membrane potential error caused by adversarial disturbance. Theoretical analysis demonstrates favorable dynamic properties of TA-LIF neurons in terms of the bounded-input bounded-output stability and suppressed time growth of membrane potential error, underscoring their superior robustness compared with the standard LIF neurons. When trained with weak FGSM attacks (attack budget = 2/255) and tested with much stronger PGD attacks (attack budget = 8/255), our HoSNNs significantly improve model accuracy on several datasets: from 30.54% to 74.91% on FashionMNIST, from 0.44% to 35.06% on SVHN, from 0.56% to 42.63% on CIFAR10, from 0.04% to 16.66% on CIFAR100, over the conventional LIF-based SNNs.



## **48. DDAD: A Two-pronged Adversarial Defense Based on Distributional Discrepancy**

cs.LG

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02169v1) [paper-pdf](http://arxiv.org/pdf/2503.02169v1)

**Authors**: Jiacheng Zhang, Benjamin I. P. Rubinstein, Jingfeng Zhang, Feng Liu

**Abstract**: Statistical adversarial data detection (SADD) detects whether an upcoming batch contains adversarial examples (AEs) by measuring the distributional discrepancies between clean examples (CEs) and AEs. In this paper, we reveal the potential strength of SADD-based methods by theoretically showing that minimizing distributional discrepancy can help reduce the expected loss on AEs. Nevertheless, despite these advantages, SADD-based methods have a potential limitation: they discard inputs that are detected as AEs, leading to the loss of clean information within those inputs. To address this limitation, we propose a two-pronged adversarial defense method, named Distributional-Discrepancy-based Adversarial Defense (DDAD). In the training phase, DDAD first optimizes the test power of the maximum mean discrepancy (MMD) to derive MMD-OPT, and then trains a denoiser by minimizing the MMD-OPT between CEs and AEs. In the inference phase, DDAD first leverages MMD-OPT to differentiate CEs and AEs, and then applies a two-pronged process: (1) directly feeding the detected CEs into the classifier, and (2) removing noise from the detected AEs by the distributional-discrepancy-based denoiser. Extensive experiments show that DDAD outperforms current state-of-the-art (SOTA) defense methods by notably improving clean and robust accuracy on CIFAR-10 and ImageNet-1K against adaptive white-box attacks.



## **49. Towards Scalable Topological Regularizers**

cs.LG

31 pages, ICLR 2025 camera-ready version

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2501.14641v2) [paper-pdf](http://arxiv.org/pdf/2501.14641v2)

**Authors**: Hiu-Tung Wong, Darrick Lee, Hong Yan

**Abstract**: Latent space matching, which consists of matching distributions of features in latent space, is a crucial component for tasks such as adversarial attacks and defenses, domain adaptation, and generative modelling. Metrics for probability measures, such as Wasserstein and maximum mean discrepancy, are commonly used to quantify the differences between such distributions. However, these are often costly to compute, or do not appropriately take the geometric and topological features of the distributions into consideration. Persistent homology is a tool from topological data analysis which quantifies the multi-scale topological structure of point clouds, and has recently been used as a topological regularizer in learning tasks. However, computation costs preclude larger scale computations, and discontinuities in the gradient lead to unstable training behavior such as in adversarial tasks. We propose the use of principal persistence measures, based on computing the persistent homology of a large number of small subsamples, as a topological regularizer. We provide a parallelized GPU implementation of this regularizer, and prove that gradients are continuous for smooth densities. Furthermore, we demonstrate the efficacy of this regularizer on shape matching, image generation, and semi-supervised learning tasks, opening the door towards a scalable regularizer for topological features.



## **50. Jailbreaking Safeguarded Text-to-Image Models via Large Language Models**

cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01839v1) [paper-pdf](http://arxiv.org/pdf/2503.01839v1)

**Authors**: Zhengyuan Jiang, Yuepeng Hu, Yuchen Yang, Yinzhi Cao, Neil Zhenqiang Gong

**Abstract**: Text-to-Image models may generate harmful content, such as pornographic images, particularly when unsafe prompts are submitted. To address this issue, safety filters are often added on top of text-to-image models, or the models themselves are aligned to reduce harmful outputs. However, these defenses remain vulnerable when an attacker strategically designs adversarial prompts to bypass these safety guardrails. In this work, we propose PromptTune, a method to jailbreak text-to-image models with safety guardrails using a fine-tuned large language model. Unlike other query-based jailbreak attacks that require repeated queries to the target model, our attack generates adversarial prompts efficiently after fine-tuning our AttackLLM. We evaluate our method on three datasets of unsafe prompts and against five safety guardrails. Our results demonstrate that our approach effectively bypasses safety guardrails, outperforms existing no-box attacks, and also facilitates other query-based attacks.



