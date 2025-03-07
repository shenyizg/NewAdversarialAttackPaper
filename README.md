# Latest Adversarial Attack Papers
**update at 2025-03-07 10:04:30**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. FSPGD: Rethinking Black-box Attacks on Semantic Segmentation**

cs.CV

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2502.01262v2) [paper-pdf](http://arxiv.org/pdf/2502.01262v2)

**Authors**: Eun-Sol Park, MiSo Park, Seung Park, Yong-Goo Shin

**Abstract**: Transferability, the ability of adversarial examples crafted for one model to deceive other models, is crucial for black-box attacks. Despite advancements in attack methods for semantic segmentation, transferability remains limited, reducing their effectiveness in real-world applications. To address this, we introduce the Feature Similarity Projected Gradient Descent (FSPGD) attack, a novel black-box approach that enhances both attack performance and transferability. Unlike conventional segmentation attacks that rely on output predictions for gradient calculation, FSPGD computes gradients from intermediate layer features. Specifically, our method introduces a loss function that targets local information by comparing features between clean images and adversarial examples, while also disrupting contextual information by accounting for spatial relationships between objects. Experiments on Pascal VOC 2012 and Cityscapes datasets demonstrate that FSPGD achieves superior transferability and attack performance, establishing a new state-of-the-art benchmark. Code is available at https://github.com/KU-AIVS/FSPGD.



## **2. OrbID: Identifying Orbcomm Satellite RF Fingerprints**

eess.SP

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.02118v2) [paper-pdf](http://arxiv.org/pdf/2503.02118v2)

**Authors**: Cédric Solenthaler, Joshua Smailes, Martin Strohmeier

**Abstract**: An increase in availability of Software Defined Radios (SDRs) has caused a dramatic shift in the threat landscape of legacy satellite systems, opening them up to easy spoofing attacks by low-budget adversaries. Physical-layer authentication methods can help improve the security of these systems by providing additional validation without modifying the space segment. This paper extends previous research on Radio Frequency Fingerprinting (RFF) of satellite communication to the Orbcomm satellite formation. The GPS and Iridium constellations are already well covered in prior research, but the feasibility of transferring techniques to other formations has not yet been examined, and raises previously undiscussed challenges.   In this paper, we collect a novel dataset containing 8992474 packets from the Orbcom satellite constellation using different SDRs and locations. We use this dataset to train RFF systems based on convolutional neural networks. We achieve an ROC AUC score of 0.53 when distinguishing different satellites within the constellation, and 0.98 when distinguishing legitimate satellites from SDRs in a spoofing scenario. We also demonstrate the possibility of mixing datasets using different SDRs in different physical locations.



## **3. Poisoning Bayesian Inference via Data Deletion and Replication**

stat.ML

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04480v1) [paper-pdf](http://arxiv.org/pdf/2503.04480v1)

**Authors**: Matthieu Carreau, Roi Naveiro, William N. Caballero

**Abstract**: Research in adversarial machine learning (AML) has shown that statistical models are vulnerable to maliciously altered data. However, despite advances in Bayesian machine learning models, most AML research remains concentrated on classical techniques. Therefore, we focus on extending the white-box model poisoning paradigm to attack generic Bayesian inference, highlighting its vulnerability in adversarial contexts. A suite of attacks are developed that allow an attacker to steer the Bayesian posterior toward a target distribution through the strategic deletion and replication of true observations, even when only sampling access to the posterior is available. Analytic properties of these algorithms are proven and their performance is empirically examined in both synthetic and real-world scenarios. With relatively little effort, the attacker is able to substantively alter the Bayesian's beliefs and, by accepting more risk, they can mold these beliefs to their will. By carefully constructing the adversarial posterior, surgical poisoning is achieved such that only targeted inferences are corrupted and others are minimally disturbed.



## **4. Know Thy Judge: On the Robustness Meta-Evaluation of LLM Safety Judges**

cs.LG

Accepted to the ICBINB Workshop at ICLR'25

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04474v1) [paper-pdf](http://arxiv.org/pdf/2503.04474v1)

**Authors**: Francisco Eiras, Eliott Zemour, Eric Lin, Vaikkunth Mugunthan

**Abstract**: Large Language Model (LLM) based judges form the underpinnings of key safety evaluation processes such as offline benchmarking, automated red-teaming, and online guardrailing. This widespread requirement raises the crucial question: can we trust the evaluations of these evaluators? In this paper, we highlight two critical challenges that are typically overlooked: (i) evaluations in the wild where factors like prompt sensitivity and distribution shifts can affect performance and (ii) adversarial attacks that target the judge. We highlight the importance of these through a study of commonly used safety judges, showing that small changes such as the style of the model output can lead to jumps of up to 0.24 in the false negative rate on the same dataset, whereas adversarial attacks on the model generation can fool some judges into misclassifying 100% of harmful generations as safe ones. These findings reveal gaps in commonly used meta-evaluation benchmarks and weaknesses in the robustness of current LLM judges, indicating that low attack success under certain judges could create a false sense of security.



## **5. Privacy Preserving and Robust Aggregation for Cross-Silo Federated Learning in Non-IID Settings**

cs.LG

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04451v1) [paper-pdf](http://arxiv.org/pdf/2503.04451v1)

**Authors**: Marco Arazzi, Mert Cihangiroglu, Antonino Nocera

**Abstract**: Federated Averaging remains the most widely used aggregation strategy in federated learning due to its simplicity and scalability. However, its performance degrades significantly in non-IID data settings, where client distributions are highly imbalanced or skewed. Additionally, it relies on clients transmitting metadata, specifically the number of training samples, which introduces privacy risks and may conflict with regulatory frameworks like the European GDPR. In this paper, we propose a novel aggregation strategy that addresses these challenges by introducing class-aware gradient masking. Unlike traditional approaches, our method relies solely on gradient updates, eliminating the need for any additional client metadata, thereby enhancing privacy protection. Furthermore, our approach validates and dynamically weights client contributions based on class-specific importance, ensuring robustness against non-IID distributions, convergence prevention, and backdoor attacks. Extensive experiments on benchmark datasets demonstrate that our method not only outperforms FedAvg and other widely accepted aggregation strategies in non-IID settings but also preserves model integrity in adversarial scenarios. Our results establish the effectiveness of gradient masking as a practical and secure solution for federated learning.



## **6. Towards Effective and Sparse Adversarial Attack on Spiking Neural Networks via Breaking Invisible Surrogate Gradients**

cs.CV

Accepted by CVPR 2025

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.03272v2) [paper-pdf](http://arxiv.org/pdf/2503.03272v2)

**Authors**: Li Lun, Kunyu Feng, Qinglong Ni, Ling Liang, Yuan Wang, Ying Li, Dunshan Yu, Xiaoxin Cui

**Abstract**: Spiking neural networks (SNNs) have shown their competence in handling spatial-temporal event-based data with low energy consumption. Similar to conventional artificial neural networks (ANNs), SNNs are also vulnerable to gradient-based adversarial attacks, wherein gradients are calculated by spatial-temporal back-propagation (STBP) and surrogate gradients (SGs). However, the SGs may be invisible for an inference-only model as they do not influence the inference results, and current gradient-based attacks are ineffective for binary dynamic images captured by the dynamic vision sensor (DVS). While some approaches addressed the issue of invisible SGs through universal SGs, their SGs lack a correlation with the victim model, resulting in sub-optimal performance. Moreover, the imperceptibility of existing SNN-based binary attacks is still insufficient. In this paper, we introduce an innovative potential-dependent surrogate gradient (PDSG) method to establish a robust connection between the SG and the model, thereby enhancing the adaptability of adversarial attacks across various models with invisible SGs. Additionally, we propose the sparse dynamic attack (SDA) to effectively attack binary dynamic images. Utilizing a generation-reduction paradigm, SDA can fully optimize the sparsity of adversarial perturbations. Experimental results demonstrate that our PDSG and SDA outperform state-of-the-art SNN-based attacks across various models and datasets. Specifically, our PDSG achieves 100% attack success rate on ImageNet, and our SDA obtains 82% attack success rate by modifying only 0.24% of the pixels on CIFAR10DVS. The code is available at https://github.com/ryime/PDSG-SDA .



## **7. Fast Preemption: Forward-Backward Cascade Learning for Efficient and Transferable Preemptive Adversarial Defense**

cs.CR

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2407.15524v7) [paper-pdf](http://arxiv.org/pdf/2407.15524v7)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: {Deep learning has made significant strides but remains susceptible to adversarial attacks, undermining its reliability. Most existing research addresses these threats after attacks happen. A growing direction explores preemptive defenses like mitigating adversarial threats proactively, offering improved robustness but at cost of efficiency and transferability. This paper introduces Fast Preemption, a novel preemptive adversarial defense that overcomes efficiency challenges while achieving state-of-the-art robustness and transferability, requiring no prior knowledge of attacks and target models. We propose a forward-backward cascade learning algorithm, which generates protective perturbations by combining forward propagation for rapid convergence with iterative backward propagation to prevent overfitting. Executing in just three iterations, Fast Preemption outperforms existing training-time, test-time, and preemptive defenses. Additionally, we introduce an adaptive reversion attack to assess the reliability of preemptive defenses, demonstrating that our approach remains secure in realistic attack scenarios.



## **8. Scale-Invariant Adversarial Attack against Arbitrary-scale Super-resolution**

cs.CV

15 pages, accepted by TIFS 2025

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04385v1) [paper-pdf](http://arxiv.org/pdf/2503.04385v1)

**Authors**: Yihao Huang, Xin Luo, Qing Guo, Felix Juefei-Xu, Xiaojun Jia, Weikai Miao, Geguang Pu, Yang Liu

**Abstract**: The advent of local continuous image function (LIIF) has garnered significant attention for arbitrary-scale super-resolution (SR) techniques. However, while the vulnerabilities of fixed-scale SR have been assessed, the robustness of continuous representation-based arbitrary-scale SR against adversarial attacks remains an area warranting further exploration. The elaborately designed adversarial attacks for fixed-scale SR are scale-dependent, which will cause time-consuming and memory-consuming problems when applied to arbitrary-scale SR. To address this concern, we propose a simple yet effective ``scale-invariant'' SR adversarial attack method with good transferability, termed SIAGT. Specifically, we propose to construct resource-saving attacks by exploiting finite discrete points of continuous representation. In addition, we formulate a coordinate-dependent loss to enhance the cross-model transferability of the attack. The attack can significantly deteriorate the SR images while introducing imperceptible distortion to the targeted low-resolution (LR) images. Experiments carried out on three popular LIIF-based SR approaches and four classical SR datasets show remarkable attack performance and transferability of SIAGT.



## **9. $σ$-zero: Gradient-based Optimization of $\ell_0$-norm Adversarial Examples**

cs.LG

Paper accepted at International Conference on Learning  Representations (ICLR 2025). Code available at  https://github.com/sigma0-advx/sigma-zero

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2402.01879v3) [paper-pdf](http://arxiv.org/pdf/2402.01879v3)

**Authors**: Antonio Emanuele Cinà, Francesco Villani, Maura Pintor, Lea Schönherr, Battista Biggio, Marcello Pelillo

**Abstract**: Evaluating the adversarial robustness of deep networks to gradient-based attacks is challenging. While most attacks consider $\ell_2$- and $\ell_\infty$-norm constraints to craft input perturbations, only a few investigate sparse $\ell_1$- and $\ell_0$-norm attacks. In particular, $\ell_0$-norm attacks remain the least studied due to the inherent complexity of optimizing over a non-convex and non-differentiable constraint. However, evaluating adversarial robustness under these attacks could reveal weaknesses otherwise left untested with more conventional $\ell_2$- and $\ell_\infty$-norm attacks. In this work, we propose a novel $\ell_0$-norm attack, called $\sigma$-zero, which leverages a differentiable approximation of the $\ell_0$ norm to facilitate gradient-based optimization, and an adaptive projection operator to dynamically adjust the trade-off between loss minimization and perturbation sparsity. Extensive evaluations using MNIST, CIFAR10, and ImageNet datasets, involving robust and non-robust models, show that $\sigma$\texttt{-zero} finds minimum $\ell_0$-norm adversarial examples without requiring any time-consuming hyperparameter tuning, and that it outperforms all competing sparse attacks in terms of success rate, perturbation size, and efficiency.



## **10. Robust Eavesdropping in the Presence of Adversarial Communications for RF Fingerprinting**

eess.SP

11 pages, 6 figures

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04120v1) [paper-pdf](http://arxiv.org/pdf/2503.04120v1)

**Authors**: Andrew Yuan, Rajeev Sahay

**Abstract**: Deep learning is an effective approach for performing radio frequency (RF) fingerprinting, which aims to identify the transmitter corresponding to received RF signals. However, beyond the intended receiver, malicious eavesdroppers can also intercept signals and attempt to fingerprint transmitters communicating over a wireless channel. Recent studies suggest that transmitters can counter such threats by embedding deep learning-based transferable adversarial attacks in their signals before transmission. In this work, we develop a time-frequency-based eavesdropper architecture that is capable of withstanding such transferable adversarial perturbations and thus able to perform effective RF fingerprinting. We theoretically demonstrate that adversarial perturbations injected by a transmitter are confined to specific time-frequency regions that are insignificant during inference, directly increasing fingerprinting accuracy on perturbed signals intercepted by the eavesdropper. Empirical evaluations on a real-world dataset validate our theoretical findings, showing that deep learning-based RF fingerprinting eavesdroppers can achieve classification performance comparable to the intended receiver, despite efforts made by the transmitter to deceive the eavesdropper. Our framework reveals that relying on transferable adversarial attacks may not be sufficient to prevent eavesdroppers from successfully fingerprinting transmissions in next-generation deep learning-based communications systems.



## **11. Adversarial Decoding: Generating Readable Documents for Adversarial Objectives**

cs.CL

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2410.02163v2) [paper-pdf](http://arxiv.org/pdf/2410.02163v2)

**Authors**: Collin Zhang, Tingwei Zhang, Vitaly Shmatikov

**Abstract**: We design, implement, and evaluate adversarial decoding, a new, generic text generation technique that produces readable documents for different adversarial objectives. Prior methods either produce easily detectable gibberish, or cannot handle objectives that include embedding similarity. In particular, they only work for direct attacks (such as jailbreaking) and cannot produce adversarial text for realistic indirect injection, e.g., documents that (1) are retrieved in RAG systems in response to broad classes of queries, and also (2) adversarially influence subsequent generation. We also show that fluency (low perplexity) is not sufficient to evade filtering. We measure the effectiveness of adversarial decoding for different objectives, including RAG poisoning, jailbreaking, and evasion of defensive filters, and demonstrate that it outperforms existing methods while producing readable adversarial documents.



## **12. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2502.16750v2) [paper-pdf](http://arxiv.org/pdf/2502.16750v2)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehnaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.



## **13. Task-Agnostic Attacks Against Vision Foundation Models**

cs.CV

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03842v1) [paper-pdf](http://arxiv.org/pdf/2503.03842v1)

**Authors**: Brian Pulfer, Yury Belousov, Vitaliy Kinakh, Teddy Furon, Slava Voloshynovskiy

**Abstract**: The study of security in machine learning mainly focuses on downstream task-specific attacks, where the adversarial example is obtained by optimizing a loss function specific to the downstream task. At the same time, it has become standard practice for machine learning practitioners to adopt publicly available pre-trained vision foundation models, effectively sharing a common backbone architecture across a multitude of applications such as classification, segmentation, depth estimation, retrieval, question-answering and more. The study of attacks on such foundation models and their impact to multiple downstream tasks remains vastly unexplored. This work proposes a general framework that forges task-agnostic adversarial examples by maximally disrupting the feature representation obtained with foundation models. We extensively evaluate the security of the feature representations obtained by popular vision foundation models by measuring the impact of this attack on multiple downstream tasks and its transferability between models.



## **14. Improving LLM Safety Alignment with Dual-Objective Optimization**

cs.CL

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03710v1) [paper-pdf](http://arxiv.org/pdf/2503.03710v1)

**Authors**: Xuandong Zhao, Will Cai, Tianneng Shi, David Huang, Licong Lin, Song Mei, Dawn Song

**Abstract**: Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at https://github.com/wicai24/DOOR-Alignment



## **15. CLIP is Strong Enough to Fight Back: Test-time Counterattacks towards Zero-shot Adversarial Robustness of CLIP**

cs.CV

Accepted to CVPR 2025

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03613v1) [paper-pdf](http://arxiv.org/pdf/2503.03613v1)

**Authors**: Songlong Xing, Zhengyu Zhao, Nicu Sebe

**Abstract**: Despite its prevalent use in image-text matching tasks in a zero-shot manner, CLIP has been shown to be highly vulnerable to adversarial perturbations added onto images. Recent studies propose to finetune the vision encoder of CLIP with adversarial samples generated on the fly, and show improved robustness against adversarial attacks on a spectrum of downstream datasets, a property termed as zero-shot robustness. In this paper, we show that malicious perturbations that seek to maximise the classification loss lead to `falsely stable' images, and propose to leverage the pre-trained vision encoder of CLIP to counterattack such adversarial images during inference to achieve robustness. Our paradigm is simple and training-free, providing the first method to defend CLIP from adversarial attacks at test time, which is orthogonal to existing methods aiming to boost zero-shot adversarial robustness of CLIP. We conduct experiments across 16 classification datasets, and demonstrate stable and consistent gains compared to test-time defence methods adapted from existing adversarial robustness studies that do not rely on external networks, without noticeably impairing performance on clean images. We also show that our paradigm can be employed on CLIP models that have been adversarially finetuned to further enhance their robustness at test time. Our code is available \href{https://github.com/Sxing2/CLIP-Test-time-Counterattacks}{here}.



## **16. AttackSeqBench: Benchmarking Large Language Models' Understanding of Sequential Patterns in Cyber Attacks**

cs.CR

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03170v1) [paper-pdf](http://arxiv.org/pdf/2503.03170v1)

**Authors**: Javier Yong, Haokai Ma, Yunshan Ma, Anis Yusof, Zhenkai Liang, Ee-Chien Chang

**Abstract**: The observations documented in Cyber Threat Intelligence (CTI) reports play a critical role in describing adversarial behaviors, providing valuable insights for security practitioners to respond to evolving threats. Recent advancements of Large Language Models (LLMs) have demonstrated significant potential in various cybersecurity applications, including CTI report understanding and attack knowledge graph construction. While previous works have proposed benchmarks that focus on the CTI extraction ability of LLMs, the sequential characteristic of adversarial behaviors within CTI reports remains largely unexplored, which holds considerable significance in developing a comprehensive understanding of how adversaries operate. To address this gap, we introduce AttackSeqBench, a benchmark tailored to systematically evaluate LLMs' capability to understand and reason attack sequences in CTI reports. Our benchmark encompasses three distinct Question Answering (QA) tasks, each task focuses on the varying granularity in adversarial behavior. To alleviate the laborious effort of QA construction, we carefully design an automated dataset construction pipeline to create scalable and well-formulated QA datasets based on real-world CTI reports. To ensure the quality of our dataset, we adopt a hybrid approach of combining human evaluation and systematic evaluation metrics. We conduct extensive experiments and analysis with both fast-thinking and slow-thinking LLMs, while highlighting their strengths and limitations in analyzing the sequential patterns in cyber attacks. The overarching goal of this work is to provide a benchmark that advances LLM-driven CTI report understanding and fosters its application in real-world cybersecurity operations. Our dataset and code are available at https://github.com/Javiery3889/AttackSeqBench .



## **17. Exploiting Vulnerabilities in Speech Translation Systems through Targeted Adversarial Attacks**

cs.SD

Preprint,17 pages, 17 figures

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.00957v2) [paper-pdf](http://arxiv.org/pdf/2503.00957v2)

**Authors**: Chang Liu, Haolin Wu, Xi Yang, Kui Zhang, Cong Wu, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Qing Guo, Jie Zhang

**Abstract**: As speech translation (ST) systems become increasingly prevalent, understanding their vulnerabilities is crucial for ensuring robust and reliable communication. However, limited work has explored this issue in depth. This paper explores methods of compromising these systems through imperceptible audio manipulations. Specifically, we present two innovative approaches: (1) the injection of perturbation into source audio, and (2) the generation of adversarial music designed to guide targeted translation, while also conducting more practical over-the-air attacks in the physical world. Our experiments reveal that carefully crafted audio perturbations can mislead translation models to produce targeted, harmful outputs, while adversarial music achieve this goal more covertly, exploiting the natural imperceptibility of music. These attacks prove effective across multiple languages and translation models, highlighting a systemic vulnerability in current ST architectures. The implications of this research extend beyond immediate security concerns, shedding light on the interpretability and robustness of neural speech processing systems. Our findings underscore the need for advanced defense mechanisms and more resilient architectures in the realm of audio systems. More details and samples can be found at https://adv-st.github.io.



## **18. Detecting Adversarial Data using Perturbation Forgery**

cs.CV

Accepted as a conference paper at CVPR 2025

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2405.16226v4) [paper-pdf](http://arxiv.org/pdf/2405.16226v4)

**Authors**: Qian Wang, Chen Li, Yuchen Luo, Hefei Ling, Shijuan Huang, Ruoxi Jia, Ning Yu

**Abstract**: As a defense strategy against adversarial attacks, adversarial detection aims to identify and filter out adversarial data from the data flow based on discrepancies in distribution and noise patterns between natural and adversarial data. Although previous detection methods achieve high performance in detecting gradient-based adversarial attacks, new attacks based on generative models with imbalanced and anisotropic noise patterns evade detection. Even worse, the significant inference time overhead and limited performance against unseen attacks make existing techniques impractical for real-world use. In this paper, we explore the proximity relationship among adversarial noise distributions and demonstrate the existence of an open covering for these distributions. By training on the open covering of adversarial noise distributions, a detector with strong generalization performance against various types of unseen attacks can be developed. Based on this insight, we heuristically propose Perturbation Forgery, which includes noise distribution perturbation, sparse mask generation, and pseudo-adversarial data production, to train an adversarial detector capable of detecting any unseen gradient-based, generative-based, and physical adversarial attacks. Comprehensive experiments conducted on multiple general and facial datasets, with a wide spectrum of attacks, validate the strong generalization of our method.



## **19. The Last Iterate Advantage: Empirical Auditing and Principled Heuristic Analysis of Differentially Private SGD**

cs.CR

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2410.06186v3) [paper-pdf](http://arxiv.org/pdf/2410.06186v3)

**Authors**: Thomas Steinke, Milad Nasr, Arun Ganesh, Borja Balle, Christopher A. Choquette-Choo, Matthew Jagielski, Jamie Hayes, Abhradeep Guha Thakurta, Adam Smith, Andreas Terzis

**Abstract**: We propose a simple heuristic privacy analysis of noisy clipped stochastic gradient descent (DP-SGD) in the setting where only the last iterate is released and the intermediate iterates remain hidden. Namely, our heuristic assumes a linear structure for the model.   We show experimentally that our heuristic is predictive of the outcome of privacy auditing applied to various training procedures. Thus it can be used prior to training as a rough estimate of the final privacy leakage. We also probe the limitations of our heuristic by providing some artificial counterexamples where it underestimates the privacy leakage.   The standard composition-based privacy analysis of DP-SGD effectively assumes that the adversary has access to all intermediate iterates, which is often unrealistic. However, this analysis remains the state of the art in practice. While our heuristic does not replace a rigorous privacy analysis, it illustrates the large gap between the best theoretical upper bounds and the privacy auditing lower bounds and sets a target for further work to improve the theoretical privacy analyses. We also empirically support our heuristic and show existing privacy auditing attacks are bounded by our heuristic analysis in both vision and language tasks.



## **20. An Undetectable Watermark for Generative Image Models**

cs.CR

ICLR 2025

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2410.07369v3) [paper-pdf](http://arxiv.org/pdf/2410.07369v3)

**Authors**: Sam Gunn, Xuandong Zhao, Dawn Song

**Abstract**: We present the first undetectable watermarking scheme for generative image models. Undetectability ensures that no efficient adversary can distinguish between watermarked and un-watermarked images, even after making many adaptive queries. In particular, an undetectable watermark does not degrade image quality under any efficiently computable metric. Our scheme works by selecting the initial latents of a diffusion model using a pseudorandom error-correcting code (Christ and Gunn, 2024), a strategy which guarantees undetectability and robustness. We experimentally demonstrate that our watermarks are quality-preserving and robust using Stable Diffusion 2.1. Our experiments verify that, in contrast to every prior scheme we tested, our watermark does not degrade image quality. Our experiments also demonstrate robustness: existing watermark removal attacks fail to remove our watermark from images without significantly degrading the quality of the images. Finally, we find that we can robustly encode 512 bits in our watermark, and up to 2500 bits when the images are not subjected to watermark removal attacks. Our code is available at https://github.com/XuandongZhao/PRC-Watermark.



## **21. LLM Misalignment via Adversarial RLHF Platforms**

cs.LG

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.03039v1) [paper-pdf](http://arxiv.org/pdf/2503.03039v1)

**Authors**: Erfan Entezami, Ali Naseh

**Abstract**: Reinforcement learning has shown remarkable performance in aligning language models with human preferences, leading to the rise of attention towards developing RLHF platforms. These platforms enable users to fine-tune models without requiring any expertise in developing complex machine learning algorithms. While these platforms offer useful features such as reward modeling and RLHF fine-tuning, their security and reliability remain largely unexplored. Given the growing adoption of RLHF and open-source RLHF frameworks, we investigate the trustworthiness of these systems and their potential impact on behavior of LLMs. In this paper, we present an attack targeting publicly available RLHF tools. In our proposed attack, an adversarial RLHF platform corrupts the LLM alignment process by selectively manipulating data samples in the preference dataset. In this scenario, when a user's task aligns with the attacker's objective, the platform manipulates a subset of the preference dataset that contains samples related to the attacker's target. This manipulation results in a corrupted reward model, which ultimately leads to the misalignment of the language model. Our results demonstrate that such an attack can effectively steer LLMs toward undesirable behaviors within the targeted domains. Our work highlights the critical need to explore the vulnerabilities of RLHF platforms and their potential to cause misalignment in LLMs during the RLHF fine-tuning process.



## **22. Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis**

cs.CR

IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02986v1) [paper-pdf](http://arxiv.org/pdf/2503.02986v1)

**Authors**: Jeonghwan Park, Niall McLaughlin, Ihsen Alouani

**Abstract**: Adversarial attacks remain a significant threat that can jeopardize the integrity of Machine Learning (ML) models. In particular, query-based black-box attacks can generate malicious noise without having access to the victim model's architecture, making them practical in real-world contexts. The community has proposed several defenses against adversarial attacks, only to be broken by more advanced and adaptive attack strategies. In this paper, we propose a framework that detects if an adversarial noise instance is being generated. Unlike existing stateful defenses that detect adversarial noise generation by monitoring the input space, our approach learns adversarial patterns in the input update similarity space. In fact, we propose to observe a new metric called Delta Similarity (DS), which we show it captures more efficiently the adversarial behavior. We evaluate our approach against 8 state-of-the-art attacks, including adaptive attacks, where the adversary is aware of the defense and tries to evade detection. We find that our approach is significantly more robust than existing defenses both in terms of specificity and sensitivity.



## **23. Decentralized Adversarial Training over Graphs**

cs.LG

arXiv admin note: text overlap with arXiv:2303.01936

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2303.13326v2) [paper-pdf](http://arxiv.org/pdf/2303.13326v2)

**Authors**: Ying Cao, Elsa Rizk, Stefan Vlaski, Ali H. Sayed

**Abstract**: The vulnerability of machine learning models to adversarial attacks has been attracting considerable attention in recent years. Most existing studies focus on the behavior of stand-alone single-agent learners. In comparison, this work studies adversarial training over graphs, where individual agents are subjected to perturbations of varied strength levels across space. It is expected that interactions by linked agents, and the heterogeneity of the attack models that are possible over the graph, can help enhance robustness in view of the coordination power of the group. Using a min-max formulation of distributed learning, we develop a decentralized adversarial training framework for multi-agent systems. Specifically, we devise two decentralized adversarial training algorithms by relying on two popular decentralized learning strategies--diffusion and consensus. We analyze the convergence properties of the proposed framework for strongly-convex, convex, and non-convex environments, and illustrate the enhanced robustness to adversarial attacks.



## **24. Towards Safe AI Clinicians: A Comprehensive Study on Large Language Model Jailbreaking in Healthcare**

cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2501.18632v2) [paper-pdf](http://arxiv.org/pdf/2501.18632v2)

**Authors**: Hang Zhang, Qian Lou, Yanshan Wang

**Abstract**: Large language models (LLMs) are increasingly utilized in healthcare applications. However, their deployment in clinical practice raises significant safety concerns, including the potential spread of harmful information. This study systematically assesses the vulnerabilities of seven LLMs to three advanced black-box jailbreaking techniques within medical contexts. To quantify the effectiveness of these techniques, we propose an automated and domain-adapted agentic evaluation pipeline. Experiment results indicate that leading commercial and open-source LLMs are highly vulnerable to medical jailbreaking attacks. To bolster model safety and reliability, we further investigate the effectiveness of Continual Fine-Tuning (CFT) in defending against medical adversarial attacks. Our findings underscore the necessity for evolving attack methods evaluation, domain-specific safety alignment, and LLM safety-utility balancing. This research offers actionable insights for advancing the safety and reliability of AI clinicians, contributing to ethical and effective AI deployment in healthcare.



## **25. Assessing Robustness via Score-Based Adversarial Image Generation**

cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2310.04285v3) [paper-pdf](http://arxiv.org/pdf/2310.04285v3)

**Authors**: Marcel Kollovieh, Lukas Gosch, Marten Lienen, Yan Scholten, Leo Schwinn, Stephan Günnemann

**Abstract**: Most adversarial attacks and defenses focus on perturbations within small $\ell_p$-norm constraints. However, $\ell_p$ threat models cannot capture all relevant semantics-preserving perturbations, and hence, the scope of robustness evaluations is limited. In this work, we introduce Score-Based Adversarial Generation (ScoreAG), a novel framework that leverages the advancements in score-based generative models to generate unrestricted adversarial examples that overcome the limitations of $\ell_p$-norm constraints. Unlike traditional methods, ScoreAG maintains the core semantics of images while generating adversarial examples, either by transforming existing images or synthesizing new ones entirely from scratch. We further exploit the generative capability of ScoreAG to purify images, empirically enhancing the robustness of classifiers. Our extensive empirical evaluation demonstrates that ScoreAG improves upon the majority of state-of-the-art attacks and defenses across multiple benchmarks. This work highlights the importance of investigating adversarial examples bounded by semantics rather than $\ell_p$-norm constraints. ScoreAG represents an important step towards more encompassing robustness assessments.



## **26. Realizing Quantum Adversarial Defense on a Trapped-ion Quantum Processor**

quant-ph

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02436v1) [paper-pdf](http://arxiv.org/pdf/2503.02436v1)

**Authors**: Alex Jin, Tarun Dutta, Anh Tu Ngo, Anupam Chattopadhyay, Manas Mukherjee

**Abstract**: Classification is a fundamental task in machine learning, typically performed using classical models. Quantum machine learning (QML), however, offers distinct advantages, such as enhanced representational power through high-dimensional Hilbert spaces and energy-efficient reversible gate operations. Despite these theoretical benefits, the robustness of QML classifiers against adversarial attacks and inherent quantum noise remains largely under-explored. In this work, we implement a data re-uploading-based quantum classifier on an ion-trap quantum processor using a single qubit to assess its resilience under realistic conditions. We introduce a novel convolutional quantum classifier architecture leveraging data re-uploading and demonstrate its superior robustness on the MNIST dataset. Additionally, we quantify the effects of polarization noise in a realistic setting, where both bit and phase noises are present, further validating the classifier's robustness. Our findings provide insights into the practical security and reliability of quantum classifiers, bridging the gap between theoretical potential and real-world deployment.



## **27. Trace of the Times: Rootkit Detection through Temporal Anomalies in Kernel Activity**

cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02402v1) [paper-pdf](http://arxiv.org/pdf/2503.02402v1)

**Authors**: Max Landauer, Leonhard Alton, Martina Lindorfer, Florian Skopik, Markus Wurzenberger, Wolfgang Hotwagner

**Abstract**: Kernel rootkits provide adversaries with permanent high-privileged access to compromised systems and are often a key element of sophisticated attack chains. At the same time, they enable stealthy operation and are thus difficult to detect. Thereby, they inject code into kernel functions to appear invisible to users, for example, by manipulating file enumerations. Existing detection approaches are insufficient, because they rely on signatures that are unable to detect novel rootkits or require domain knowledge about the rootkits to be detected. To overcome this challenge, our approach leverages the fact that runtimes of kernel functions targeted by rootkits increase when additional code is executed. The framework outlined in this paper injects probes into the kernel to measure time stamps of functions within relevant system calls, computes distributions of function execution times, and uses statistical tests to detect time shifts. The evaluation of our open-source implementation on publicly available data sets indicates high detection accuracy with an F1 score of 98.7\% across five scenarios with varying system states.



## **28. Evaluating the Robustness of LiDAR Point Cloud Tracking Against Adversarial Attack**

cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2410.20893v2) [paper-pdf](http://arxiv.org/pdf/2410.20893v2)

**Authors**: Shengjing Tian, Yinan Han, Xiantong Zhao, Bin Liu, Xiuping Liu

**Abstract**: In this study, we delve into the robustness of neural network-based LiDAR point cloud tracking models under adversarial attacks, a critical aspect often overlooked in favor of performance enhancement. These models, despite incorporating advanced architectures like Transformer or Bird's Eye View (BEV), tend to neglect robustness in the face of challenges such as adversarial attacks, domain shifts, or data corruption. We instead focus on the robustness of the tracking models under the threat of adversarial attacks. We begin by establishing a unified framework for conducting adversarial attacks within the context of 3D object tracking, which allows us to thoroughly investigate both white-box and black-box attack strategies. For white-box attacks, we tailor specific loss functions to accommodate various tracking paradigms and extend existing methods such as FGSM, C\&W, and PGD to the point cloud domain. In addressing black-box attack scenarios, we introduce a novel transfer-based approach, the Target-aware Perturbation Generation (TAPG) algorithm, with the dual objectives of achieving high attack performance and maintaining low perceptibility. This method employs a heuristic strategy to enforce sparse attack constraints and utilizes random sub-vector factorization to bolster transferability. Our experimental findings reveal a significant vulnerability in advanced tracking methods when subjected to both black-box and white-box attacks, underscoring the necessity for incorporating robustness against adversarial attacks into the design of LiDAR point cloud tracking models. Notably, compared to existing methods, the TAPG also strikes an optimal balance between the effectiveness of the attack and the concealment of the perturbations.



## **29. NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary**

cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.00063v2) [paper-pdf](http://arxiv.org/pdf/2503.00063v2)

**Authors**: Zezeng Li, Xiaoyu Du, Na Lei, Liming Chen, Weimin Wang

**Abstract**: Adversarial attacks exploit the vulnerability of deep models against adversarial samples. Existing point cloud attackers are tailored to specific models, iteratively optimizing perturbations based on gradients in either a white-box or black-box setting. Despite their promising attack performance, they often struggle to produce transferable adversarial samples due to overfitting the specific parameters of surrogate models. To overcome this issue, we shift our focus to the data distribution itself and introduce a novel approach named NoPain, which employs optimal transport (OT) to identify the inherent singular boundaries of the data manifold for cross-network point cloud attacks. Specifically, we first calculate the OT mapping from noise to the target feature space, then identify singular boundaries by locating non-differentiable positions. Finally, we sample along singular boundaries to generate adversarial point clouds. Once the singular boundaries are determined, NoPain can efficiently produce adversarial samples without the need of iterative updates or guidance from the surrogate classifiers. Extensive experiments demonstrate that the proposed end-to-end method outperforms baseline approaches in terms of both transferability and efficiency, while also maintaining notable advantages even against defense strategies. The source code will be publicly available.



## **30. Game-Theoretic Defenses for Robust Conformal Prediction Against Adversarial Attacks in Medical Imaging**

cs.LG

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2411.04376v2) [paper-pdf](http://arxiv.org/pdf/2411.04376v2)

**Authors**: Rui Luo, Jie Bao, Zhixin Zhou, Chuangyin Dang

**Abstract**: Adversarial attacks pose significant threats to the reliability and safety of deep learning models, especially in critical domains such as medical imaging. This paper introduces a novel framework that integrates conformal prediction with game-theoretic defensive strategies to enhance model robustness against both known and unknown adversarial perturbations. We address three primary research questions: constructing valid and efficient conformal prediction sets under known attacks (RQ1), ensuring coverage under unknown attacks through conservative thresholding (RQ2), and determining optimal defensive strategies within a zero-sum game framework (RQ3). Our methodology involves training specialized defensive models against specific attack types and employing maximum and minimum classifiers to aggregate defenses effectively. Extensive experiments conducted on the MedMNIST datasets, including PathMNIST, OrganAMNIST, and TissueMNIST, demonstrate that our approach maintains high coverage guarantees while minimizing prediction set sizes. The game-theoretic analysis reveals that the optimal defensive strategy often converges to a singular robust model, outperforming uniform and simple strategies across all evaluated datasets. This work advances the state-of-the-art in uncertainty quantification and adversarial robustness, providing a reliable mechanism for deploying deep learning models in adversarial environments.



## **31. TPIA: Towards Target-specific Prompt Injection Attack against Code-oriented Large Language Models**

cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2407.09164v5) [paper-pdf](http://arxiv.org/pdf/2407.09164v5)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully exploited to simplify and facilitate programming. Unfortunately, a few pioneering works revealed that these Code LLMs are vulnerable to backdoor and adversarial attacks. The former poisons the training data or model parameters, hijacking the LLMs to generate malicious code snippets when encountering the trigger. The latter crafts malicious adversarial input codes to reduce the quality of the generated codes. In this paper, we reveal that both attacks have some inherent limitations: backdoor attacks rely on the adversary's capability of controlling the model training process, which may not be practical; adversarial attacks struggle with fulfilling specific malicious purposes. To alleviate these problems, this paper presents a novel attack paradigm against Code LLMs, namely target-specific prompt injection attack (TPIA). TPIA generates non-functional perturbations containing the information of malicious instructions and inserts them into the victim's code context by spreading them into potentially used dependencies (e.g., packages or RAG's knowledge base). It induces the Code LLMs to generate attacker-specified malicious code snippets at the target location. In general, we compress the attacker-specified malicious objective into the perturbation by adversarial optimization based on greedy token search. We collect 13 representative malicious objectives to design 31 threat cases for three popular programming languages. We show that our TPIA can successfully attack three representative open-source Code LLMs (with an attack success rate of up to 97.9%) and two mainstream commercial Code LLM-integrated applications (with an attack success rate of over 90%) in all threat cases, using only a 12-token non-functional perturbation.



## **32. Prompt-driven Transferable Adversarial Attack on Person Re-Identification with Attribute-aware Textual Inversion**

cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2502.19697v2) [paper-pdf](http://arxiv.org/pdf/2502.19697v2)

**Authors**: Yuan Bian, Min Liu, Yunqi Yi, Xueping Wang, Yaonan Wang

**Abstract**: Person re-identification (re-id) models are vital in security surveillance systems, requiring transferable adversarial attacks to explore the vulnerabilities of them. Recently, vision-language models (VLM) based attacks have shown superior transferability by attacking generalized image and textual features of VLM, but they lack comprehensive feature disruption due to the overemphasis on discriminative semantics in integral representation. In this paper, we introduce the Attribute-aware Prompt Attack (AP-Attack), a novel method that leverages VLM's image-text alignment capability to explicitly disrupt fine-grained semantic features of pedestrian images by destroying attribute-specific textual embeddings. To obtain personalized textual descriptions for individual attributes, textual inversion networks are designed to map pedestrian images to pseudo tokens that represent semantic embeddings, trained in the contrastive learning manner with images and a predefined prompt template that explicitly describes the pedestrian attributes. Inverted benign and adversarial fine-grained textual semantics facilitate attacker in effectively conducting thorough disruptions, enhancing the transferability of adversarial examples. Extensive experiments show that AP-Attack achieves state-of-the-art transferability, significantly outperforming previous methods by 22.9% on mean Drop Rate in cross-model&dataset attack scenarios.



## **33. Adversarial Tokenization**

cs.CL

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02174v1) [paper-pdf](http://arxiv.org/pdf/2503.02174v1)

**Authors**: Renato Lui Geh, Zilei Shao, Guy Van den Broeck

**Abstract**: Current LLM pipelines account for only one possible tokenization for a given string, ignoring exponentially many alternative tokenizations during training and inference. For example, the standard Llama3 tokenization of penguin is [p,enguin], yet [peng,uin] is another perfectly valid alternative. In this paper, we show that despite LLMs being trained solely on one tokenization, they still retain semantic understanding of other tokenizations, raising questions about their implications in LLM safety. Put succinctly, we answer the following question: can we adversarially tokenize an obviously malicious string to evade safety and alignment restrictions? We show that not only is adversarial tokenization an effective yet previously neglected axis of attack, but it is also competitive against existing state-of-the-art adversarial approaches without changing the text of the harmful request. We empirically validate this exploit across three state-of-the-art LLMs and adversarial datasets, revealing a previously unknown vulnerability in subword models.



## **34. HoSNN: Adversarially-Robust Homeostatic Spiking Neural Networks with Adaptive Firing Thresholds**

cs.NE

Accepted by TMLR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2308.10373v4) [paper-pdf](http://arxiv.org/pdf/2308.10373v4)

**Authors**: Hejia Geng, Peng Li

**Abstract**: While spiking neural networks (SNNs) offer a promising neurally-inspired model of computation, they are vulnerable to adversarial attacks. We present the first study that draws inspiration from neural homeostasis to design a threshold-adapting leaky integrate-and-fire (TA-LIF) neuron model and utilize TA-LIF neurons to construct the adversarially robust homeostatic SNNs (HoSNNs) for improved robustness. The TA-LIF model incorporates a self-stabilizing dynamic thresholding mechanism, offering a local feedback control solution to the minimization of each neuron's membrane potential error caused by adversarial disturbance. Theoretical analysis demonstrates favorable dynamic properties of TA-LIF neurons in terms of the bounded-input bounded-output stability and suppressed time growth of membrane potential error, underscoring their superior robustness compared with the standard LIF neurons. When trained with weak FGSM attacks (attack budget = 2/255) and tested with much stronger PGD attacks (attack budget = 8/255), our HoSNNs significantly improve model accuracy on several datasets: from 30.54% to 74.91% on FashionMNIST, from 0.44% to 35.06% on SVHN, from 0.56% to 42.63% on CIFAR10, from 0.04% to 16.66% on CIFAR100, over the conventional LIF-based SNNs.



## **35. DDAD: A Two-pronged Adversarial Defense Based on Distributional Discrepancy**

cs.LG

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02169v1) [paper-pdf](http://arxiv.org/pdf/2503.02169v1)

**Authors**: Jiacheng Zhang, Benjamin I. P. Rubinstein, Jingfeng Zhang, Feng Liu

**Abstract**: Statistical adversarial data detection (SADD) detects whether an upcoming batch contains adversarial examples (AEs) by measuring the distributional discrepancies between clean examples (CEs) and AEs. In this paper, we reveal the potential strength of SADD-based methods by theoretically showing that minimizing distributional discrepancy can help reduce the expected loss on AEs. Nevertheless, despite these advantages, SADD-based methods have a potential limitation: they discard inputs that are detected as AEs, leading to the loss of clean information within those inputs. To address this limitation, we propose a two-pronged adversarial defense method, named Distributional-Discrepancy-based Adversarial Defense (DDAD). In the training phase, DDAD first optimizes the test power of the maximum mean discrepancy (MMD) to derive MMD-OPT, and then trains a denoiser by minimizing the MMD-OPT between CEs and AEs. In the inference phase, DDAD first leverages MMD-OPT to differentiate CEs and AEs, and then applies a two-pronged process: (1) directly feeding the detected CEs into the classifier, and (2) removing noise from the detected AEs by the distributional-discrepancy-based denoiser. Extensive experiments show that DDAD outperforms current state-of-the-art (SOTA) defense methods by notably improving clean and robust accuracy on CIFAR-10 and ImageNet-1K against adaptive white-box attacks.



## **36. Towards Scalable Topological Regularizers**

cs.LG

31 pages, ICLR 2025 camera-ready version

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2501.14641v2) [paper-pdf](http://arxiv.org/pdf/2501.14641v2)

**Authors**: Hiu-Tung Wong, Darrick Lee, Hong Yan

**Abstract**: Latent space matching, which consists of matching distributions of features in latent space, is a crucial component for tasks such as adversarial attacks and defenses, domain adaptation, and generative modelling. Metrics for probability measures, such as Wasserstein and maximum mean discrepancy, are commonly used to quantify the differences between such distributions. However, these are often costly to compute, or do not appropriately take the geometric and topological features of the distributions into consideration. Persistent homology is a tool from topological data analysis which quantifies the multi-scale topological structure of point clouds, and has recently been used as a topological regularizer in learning tasks. However, computation costs preclude larger scale computations, and discontinuities in the gradient lead to unstable training behavior such as in adversarial tasks. We propose the use of principal persistence measures, based on computing the persistent homology of a large number of small subsamples, as a topological regularizer. We provide a parallelized GPU implementation of this regularizer, and prove that gradients are continuous for smooth densities. Furthermore, we demonstrate the efficacy of this regularizer on shape matching, image generation, and semi-supervised learning tasks, opening the door towards a scalable regularizer for topological features.



## **37. Jailbreaking Safeguarded Text-to-Image Models via Large Language Models**

cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01839v1) [paper-pdf](http://arxiv.org/pdf/2503.01839v1)

**Authors**: Zhengyuan Jiang, Yuepeng Hu, Yuchen Yang, Yinzhi Cao, Neil Zhenqiang Gong

**Abstract**: Text-to-Image models may generate harmful content, such as pornographic images, particularly when unsafe prompts are submitted. To address this issue, safety filters are often added on top of text-to-image models, or the models themselves are aligned to reduce harmful outputs. However, these defenses remain vulnerable when an attacker strategically designs adversarial prompts to bypass these safety guardrails. In this work, we propose PromptTune, a method to jailbreak text-to-image models with safety guardrails using a fine-tuned large language model. Unlike other query-based jailbreak attacks that require repeated queries to the target model, our attack generates adversarial prompts efficiently after fine-tuning our AttackLLM. We evaluate our method on three datasets of unsafe prompts and against five safety guardrails. Our results demonstrate that our approach effectively bypasses safety guardrails, outperforms existing no-box attacks, and also facilitates other query-based attacks.



## **38. AutoAdvExBench: Benchmarking autonomous exploitation of adversarial example defenses**

cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01811v1) [paper-pdf](http://arxiv.org/pdf/2503.01811v1)

**Authors**: Nicholas Carlini, Javier Rando, Edoardo Debenedetti, Milad Nasr, Florian Tramèr

**Abstract**: We introduce AutoAdvExBench, a benchmark to evaluate if large language models (LLMs) can autonomously exploit defenses to adversarial examples. Unlike existing security benchmarks that often serve as proxies for real-world tasks, bench directly measures LLMs' success on tasks regularly performed by machine learning security experts. This approach offers a significant advantage: if a LLM could solve the challenges presented in bench, it would immediately present practical utility for adversarial machine learning researchers. We then design a strong agent that is capable of breaking 75% of CTF-like ("homework exercise") adversarial example defenses. However, we show that this agent is only able to succeed on 13% of the real-world defenses in our benchmark, indicating the large gap between difficulty in attacking "real" code, and CTF-like code. In contrast, a stronger LLM that can attack 21% of real defenses only succeeds on 54% of CTF-like defenses. We make this benchmark available at https://github.com/ethz-spylab/AutoAdvExBench.



## **39. Cats Confuse Reasoning LLM: Query Agnostic Adversarial Triggers for Reasoning Models**

cs.CL

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01781v1) [paper-pdf](http://arxiv.org/pdf/2503.01781v1)

**Authors**: Meghana Rajeev, Rajkumar Ramamurthy, Prapti Trivedi, Vikas Yadav, Oluwanifemi Bamgbose, Sathwik Tejaswi Madhusudan, James Zou, Nazneen Rajani

**Abstract**: We investigate the robustness of reasoning models trained for step-by-step problem solving by introducing query-agnostic adversarial triggers - short, irrelevant text that, when appended to math problems, systematically mislead models to output incorrect answers without altering the problem's semantics. We propose CatAttack, an automated iterative attack pipeline for generating triggers on a weaker, less expensive proxy model (DeepSeek V3) and successfully transfer them to more advanced reasoning target models like DeepSeek R1 and DeepSeek R1-distilled-Qwen-32B, resulting in greater than 300% increase in the likelihood of the target model generating an incorrect answer. For example, appending, "Interesting fact: cats sleep most of their lives," to any math problem leads to more than doubling the chances of a model getting the answer wrong. Our findings highlight critical vulnerabilities in reasoning models, revealing that even state-of-the-art models remain susceptible to subtle adversarial inputs, raising security and reliability concerns. The CatAttack triggers dataset with model responses is available at https://huggingface.co/datasets/collinear-ai/cat-attack-adversarial-triggers.



## **40. Adversarial Agents: Black-Box Evasion Attacks with Reinforcement Learning**

cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01734v1) [paper-pdf](http://arxiv.org/pdf/2503.01734v1)

**Authors**: Kyle Domico, Jean-Charles Noirot Ferrand, Ryan Sheatsley, Eric Pauley, Josiah Hanna, Patrick McDaniel

**Abstract**: Reinforcement learning (RL) offers powerful techniques for solving complex sequential decision-making tasks from experience. In this paper, we demonstrate how RL can be applied to adversarial machine learning (AML) to develop a new class of attacks that learn to generate adversarial examples: inputs designed to fool machine learning models. Unlike traditional AML methods that craft adversarial examples independently, our RL-based approach retains and exploits past attack experience to improve future attacks. We formulate adversarial example generation as a Markov Decision Process and evaluate RL's ability to (a) learn effective and efficient attack strategies and (b) compete with state-of-the-art AML. On CIFAR-10, our agent increases the success rate of adversarial examples by 19.4% and decreases the median number of victim model queries per adversarial example by 53.2% from the start to the end of training. In a head-to-head comparison with a state-of-the-art image attack, SquareAttack, our approach enables an adversary to generate adversarial examples with 13.1% more success after 5000 episodes of training. From a security perspective, this work demonstrates a powerful new attack vector that uses RL to attack ML models efficiently and at scale.



## **41. Towards Physically Realizable Adversarial Attacks in Embodied Vision Navigation**

cs.CV

7 pages, 7 figures, submitted to IEEE/RSJ International Conference on  Intelligent Robots and Systems (IROS) 2025

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2409.10071v4) [paper-pdf](http://arxiv.org/pdf/2409.10071v4)

**Authors**: Meng Chen, Jiawei Tu, Chao Qi, Yonghao Dang, Feng Zhou, Wei Wei, Jianqin Yin

**Abstract**: The significant advancements in embodied vision navigation have raised concerns about its susceptibility to adversarial attacks exploiting deep neural networks. Investigating the adversarial robustness of embodied vision navigation is crucial, especially given the threat of 3D physical attacks that could pose risks to human safety. However, existing attack methods for embodied vision navigation often lack physical feasibility due to challenges in transferring digital perturbations into the physical world. Moreover, current physical attacks for object detection struggle to achieve both multi-view effectiveness and visual naturalness in navigation scenarios. To address this, we propose a practical attack method for embodied navigation by attaching adversarial patches to objects, where both opacity and textures are learnable. Specifically, to ensure effectiveness across varying viewpoints, we employ a multi-view optimization strategy based on object-aware sampling, which optimizes the patch's texture based on feedback from the vision-based perception model used in navigation. To make the patch inconspicuous to human observers, we introduce a two-stage opacity optimization mechanism, in which opacity is fine-tuned after texture optimization. Experimental results demonstrate that our adversarial patches decrease the navigation success rate by an average of 22.39%, outperforming previous methods in practicality, effectiveness, and naturalness. Code is available at: https://github.com/chen37058/Physical-Attacks-in-Embodied-Nav



## **42. Revisiting Locally Differentially Private Protocols: Towards Better Trade-offs in Privacy, Utility, and Attack Resistance**

cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01482v1) [paper-pdf](http://arxiv.org/pdf/2503.01482v1)

**Authors**: Héber H. Arcolezi, Sébastien Gambs

**Abstract**: Local Differential Privacy (LDP) offers strong privacy protection, especially in settings in which the server collecting the data is untrusted. However, designing LDP mechanisms that achieve an optimal trade-off between privacy, utility, and robustness to adversarial inference attacks remains challenging. In this work, we introduce a general multi-objective optimization framework for refining LDP protocols, enabling the joint optimization of privacy and utility under various adversarial settings. While our framework is flexible enough to accommodate multiple privacy and security attacks as well as utility metrics, in this paper we specifically optimize for Attacker Success Rate (ASR) under distinguishability attack as a measure of privacy and Mean Squared Error (MSE) as a measure of utility. We systematically revisit these trade-offs by analyzing eight state-of-the-art LDP protocols and proposing refined counterparts that leverage tailored optimization techniques. Experimental results demonstrate that our proposed adaptive mechanisms consistently outperform their non-adaptive counterparts, reducing ASR by up to five orders of magnitude while maintaining competitive utility. Analytical derivations also confirm the effectiveness of our mechanisms, moving them closer to the ASR-MSE Pareto frontier.



## **43. Poison-splat: Computation Cost Attack on 3D Gaussian Splatting**

cs.CV

Accepted by ICLR 2025 as a spotlight paper

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2410.08190v2) [paper-pdf](http://arxiv.org/pdf/2410.08190v2)

**Authors**: Jiahao Lu, Yifan Zhang, Qiuhong Shen, Xinchao Wang, Shuicheng Yan

**Abstract**: 3D Gaussian splatting (3DGS), known for its groundbreaking performance and efficiency, has become a dominant 3D representation and brought progress to many 3D vision tasks. However, in this work, we reveal a significant security vulnerability that has been largely overlooked in 3DGS: the computation cost of training 3DGS could be maliciously tampered by poisoning the input data. By developing an attack named Poison-splat, we reveal a novel attack surface where the adversary can poison the input images to drastically increase the computation memory and time needed for 3DGS training, pushing the algorithm towards its worst computation complexity. In extreme cases, the attack can even consume all allocable memory, leading to a Denial-of-Service (DoS) that disrupts servers, resulting in practical damages to real-world 3DGS service vendors. Such a computation cost attack is achieved by addressing a bi-level optimization problem through three tailored strategies: attack objective approximation, proxy model rendering, and optional constrained optimization. These strategies not only ensure the effectiveness of our attack but also make it difficult to defend with simple defensive measures. We hope the revelation of this novel attack surface can spark attention to this crucial yet overlooked vulnerability of 3DGS systems. Our code is available at https://github.com/jiahaolu97/poison-splat .



## **44. Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification**

cs.CV

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01407v1) [paper-pdf](http://arxiv.org/pdf/2503.01407v1)

**Authors**: Gaozheng Pei, Shaojie Lyu, Gong Chen, Ke Ma, Qianqian Xu, Yingfei Sun, Qingming Huang

**Abstract**: Existing diffusion-based purification methods aim to disrupt adversarial perturbations by introducing a certain amount of noise through a forward diffusion process, followed by a reverse process to recover clean examples. However, this approach is fundamentally flawed: the uniform operation of the forward process across all pixels compromises normal pixels while attempting to combat adversarial perturbations, resulting in the target model producing incorrect predictions. Simply relying on low-intensity noise is insufficient for effective defense. To address this critical issue, we implement a heterogeneous purification strategy grounded in the interpretability of neural networks. Our method decisively applies higher-intensity noise to specific pixels that the target model focuses on while the remaining pixels are subjected to only low-intensity noise. This requirement motivates us to redesign the sampling process of the diffusion model, allowing for the effective removal of varying noise levels. Furthermore, to evaluate our method against strong adaptative attack, our proposed method sharply reduces time cost and memory usage through a single-step resampling. The empirical evidence from extensive experiments across three datasets demonstrates that our method outperforms most current adversarial training and purification techniques by a substantial margin.



## **45. Attacking Large Language Models with Projected Gradient Descent**

cs.LG

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2402.09154v2) [paper-pdf](http://arxiv.org/pdf/2402.09154v2)

**Authors**: Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Johannes Gasteiger, Stephan Günnemann

**Abstract**: Current LLM alignment methods are readily broken through specifically crafted adversarial prompts. While crafting adversarial prompts using discrete optimization is highly effective, such attacks typically use more than 100,000 LLM calls. This high computational cost makes them unsuitable for, e.g., quantitative analyses and adversarial training. To remedy this, we revisit Projected Gradient Descent (PGD) on the continuously relaxed input prompt. Although previous attempts with ordinary gradient-based attacks largely failed, we show that carefully controlling the error introduced by the continuous relaxation tremendously boosts their efficacy. Our PGD for LLMs is up to one order of magnitude faster than state-of-the-art discrete optimization to achieve the same devastating attack results.



## **46. Exact Certification of (Graph) Neural Networks Against Label Poisoning**

cs.LG

Published as a spotlight presentation at ICLR 2025

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2412.00537v2) [paper-pdf](http://arxiv.org/pdf/2412.00537v2)

**Authors**: Mahalakshmi Sabanayagam, Lukas Gosch, Stephan Günnemann, Debarghya Ghoshdastidar

**Abstract**: Machine learning models are highly vulnerable to label flipping, i.e., the adversarial modification (poisoning) of training labels to compromise performance. Thus, deriving robustness certificates is important to guarantee that test predictions remain unaffected and to understand worst-case robustness behavior. However, for Graph Neural Networks (GNNs), the problem of certifying label flipping has so far been unsolved. We change this by introducing an exact certification method, deriving both sample-wise and collective certificates. Our method leverages the Neural Tangent Kernel (NTK) to capture the training dynamics of wide networks enabling us to reformulate the bilevel optimization problem representing label flipping into a Mixed-Integer Linear Program (MILP). We apply our method to certify a broad range of GNN architectures in node classification tasks. Thereby, concerning the worst-case robustness to label flipping: $(i)$ we establish hierarchies of GNNs on different benchmark graphs; $(ii)$ quantify the effect of architectural choices such as activations, depth and skip-connections; and surprisingly, $(iii)$ uncover a novel phenomenon of the robustness plateauing for intermediate perturbation budgets across all investigated datasets and architectures. While we focus on GNNs, our certificates are applicable to sufficiently wide NNs in general through their NTK. Thus, our work presents the first exact certificate to a poisoning attack ever derived for neural networks, which could be of independent interest. The code is available at https://github.com/saper0/qpcert.



## **47. AdvLogo: Adversarial Patch Attack against Object Detectors based on Diffusion Models**

cs.CV

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2409.07002v2) [paper-pdf](http://arxiv.org/pdf/2409.07002v2)

**Authors**: Boming Miao, Chunxiao Li, Yao Zhu, Weixiang Sun, Zizhe Wang, Xiaoyi Wang, Chuanlong Xie

**Abstract**: With the rapid development of deep learning, object detectors have demonstrated impressive performance; however, vulnerabilities still exist in certain scenarios. Current research exploring the vulnerabilities using adversarial patches often struggles to balance the trade-off between attack effectiveness and visual quality. To address this problem, we propose a novel framework of patch attack from semantic perspective, which we refer to as AdvLogo. Based on the hypothesis that every semantic space contains an adversarial subspace where images can cause detectors to fail in recognizing objects, we leverage the semantic understanding of the diffusion denoising process and drive the process to adversarial subareas by perturbing the latent and unconditional embeddings at the last timestep. To mitigate the distribution shift that exposes a negative impact on image quality, we apply perturbation to the latent in frequency domain with the Fourier Transform. Experimental results demonstrate that AdvLogo achieves strong attack performance while maintaining high visual quality.



## **48. Exploring Adversarial Robustness in Classification tasks using DNA Language Models**

cs.CL

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2409.19788v2) [paper-pdf](http://arxiv.org/pdf/2409.19788v2)

**Authors**: Hyunwoo Yoo, Haebin Shin, Kaidi Xu, Gail Rosen

**Abstract**: DNA Language Models, such as GROVER, DNABERT2 and the Nucleotide Transformer, operate on DNA sequences that inherently contain sequencing errors, mutations, and laboratory-induced noise, which may significantly impact model performance. Despite the importance of this issue, the robustness of DNA language models remains largely underexplored. In this paper, we comprehensivly investigate their robustness in DNA classification by applying various adversarial attack strategies: the character (nucleotide substitutions), word (codon modifications), and sentence levels (back-translation-based transformations) to systematically analyze model vulnerabilities. Our results demonstrate that DNA language models are highly susceptible to adversarial attacks, leading to significant performance degradation. Furthermore, we explore adversarial training method as a defense mechanism, which enhances both robustness and classification accuracy. This study highlights the limitations of DNA language models and underscores the necessity of robustness in bioinformatics.



## **49. MAA: Meticulous Adversarial Attack against Vision-Language Pre-trained Models**

cs.CV

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2502.08079v3) [paper-pdf](http://arxiv.org/pdf/2502.08079v3)

**Authors**: Peng-Fei Zhang, Guangdong Bai, Zi Huang

**Abstract**: Current adversarial attacks for evaluating the robustness of vision-language pre-trained (VLP) models in multi-modal tasks suffer from limited transferability, where attacks crafted for a specific model often struggle to generalize effectively across different models, limiting their utility in assessing robustness more broadly. This is mainly attributed to the over-reliance on model-specific features and regions, particularly in the image modality. In this paper, we propose an elegant yet highly effective method termed Meticulous Adversarial Attack (MAA) to fully exploit model-independent characteristics and vulnerabilities of individual samples, achieving enhanced generalizability and reduced model dependence. MAA emphasizes fine-grained optimization of adversarial images by developing a novel resizing and sliding crop (RScrop) technique, incorporating a multi-granularity similarity disruption (MGSD) strategy. Extensive experiments across diverse VLP models, multiple benchmark datasets, and a variety of downstream tasks demonstrate that MAA significantly enhances the effectiveness and transferability of adversarial attacks. A large cohort of performance studies is conducted to generate insights into the effectiveness of various model configurations, guiding future advancements in this domain.



## **50. Asymptotic Behavior of Adversarial Training Estimator under $\ell_\infty$-Perturbation**

math.ST

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2401.15262v2) [paper-pdf](http://arxiv.org/pdf/2401.15262v2)

**Authors**: Yiling Xie, Xiaoming Huo

**Abstract**: Adversarial training has been proposed to protect machine learning models against adversarial attacks. This paper focuses on adversarial training under $\ell_\infty$-perturbation, which has recently attracted much research attention. The asymptotic behavior of the adversarial training estimator is investigated in the generalized linear model. The results imply that the asymptotic distribution of the adversarial training estimator under $\ell_\infty$-perturbation could put a positive probability mass at $0$ when the true parameter is $0$, providing a theoretical guarantee of the associated sparsity-recovery ability. Alternatively, a two-step procedure is proposed -- adaptive adversarial training, which could further improve the performance of adversarial training under $\ell_\infty$-perturbation. Specifically, the proposed procedure could achieve asymptotic variable-selection consistency and unbiasedness. Numerical experiments are conducted to show the sparsity-recovery ability of adversarial training under $\ell_\infty$-perturbation and to compare the empirical performance between classic adversarial training and adaptive adversarial training.



