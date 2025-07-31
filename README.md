# Latest Adversarial Attack Papers
**update at 2025-07-31 10:32:53**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. AUV-Fusion: Cross-Modal Adversarial Fusion of User Interactions and Visual Perturbations Against VARS**

cs.IR

14 pages,6 figures

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22880v1) [paper-pdf](http://arxiv.org/pdf/2507.22880v1)

**Authors**: Hai Ling, Tianchi Wang, Xiaohao Liu, Zhulin Tao, Lifang Yang, Xianglin Huang

**Abstract**: Modern Visual-Aware Recommender Systems (VARS) exploit the integration of user interaction data and visual features to deliver personalized recommendations with high precision. However, their robustness against adversarial attacks remains largely underexplored, posing significant risks to system reliability and security. Existing attack strategies suffer from notable limitations: shilling attacks are costly and detectable, and visual-only perturbations often fail to align with user preferences. To address these challenges, we propose AUV-Fusion, a cross-modal adversarial attack framework that adopts high-order user preference modeling and cross-modal adversary generation. Specifically, we obtain robust user embeddings through multi-hop user-item interactions and transform them via an MLP into semantically aligned perturbations. These perturbations are injected onto the latent space of a pre-trained VAE within the diffusion model. By synergistically integrating genuine user interaction data with visually plausible perturbations, AUV-Fusion eliminates the need for injecting fake user profiles and effectively mitigates the challenge of insufficient user preference extraction inherent in traditional visual-only attacks. Comprehensive evaluations on diverse VARS architectures and real-world datasets demonstrate that AUV-Fusion significantly enhances the exposure of target (cold-start) items compared to conventional baseline methods. Moreover, AUV-Fusion maintains exceptional stealth under rigorous scrutiny.



## **2. Curvature Dynamic Black-box Attack: revisiting adversarial robustness via dynamic curvature estimation**

cs.LG

This article contains several flaws

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2505.19194v2) [paper-pdf](http://arxiv.org/pdf/2505.19194v2)

**Authors**: Peiran Sun

**Abstract**: Adversarial attack reveals the vulnerability of deep learning models. For about a decade, countless attack and defense methods have been proposed, leading to robustified classifiers and better understanding of models. Among these methods, curvature-based approaches have attracted attention because it is assumed that high curvature may give rise to rough decision boundary. However, the most commonly used \textit{curvature} is the curvature of loss function, scores or other parameters from within the model as opposed to decision boundary curvature, since the former can be relatively easily formed using second order derivative. In this paper, we propose a new query-efficient method, dynamic curvature estimation(DCE), to estimate the decision boundary curvature in a black-box setting. Our approach is based on CGBA, a black-box adversarial attack. By performing DCE on a wide range of classifiers, we discovered, statistically, a connection between decision boundary curvature and adversarial robustness. We also propose a new attack method, curvature dynamic black-box attack(CDBA) with improved performance using the dynamically estimated curvature.



## **3. DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion**

cs.CV

ICCV 2025

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22813v1) [paper-pdf](http://arxiv.org/pdf/2507.22813v1)

**Authors**: Hossein Mirzaei, Zeinab Taghavi, Sepehr Rezaee, Masoud Hadi, Moein Madadi, Mackenzie W. Mathis

**Abstract**: Deep neural networks have demonstrated remarkable success across numerous tasks, yet they remain vulnerable to Trojan (backdoor) attacks, raising serious concerns about their safety in real-world mission-critical applications. A common countermeasure is trigger inversion -- reconstructing malicious "shortcut" patterns (triggers) inserted by an adversary during training. Current trigger-inversion methods typically search the full pixel space under specific assumptions but offer no assurances that the estimated trigger is more than an adversarial perturbation that flips the model output. Here, we propose a data-free, zero-shot trigger-inversion strategy that restricts the search space while avoiding strong assumptions on trigger appearance. Specifically, we incorporate a diffusion-based generator guided by the target classifier; through iterative generation, we produce candidate triggers that align with the internal representations the model relies on for malicious behavior. Empirical evaluations, both quantitative and qualitative, show that our approach reconstructs triggers that effectively distinguish clean versus Trojaned models. DISTIL surpasses alternative methods by high margins, achieving up to 7.1% higher accuracy on the BackdoorBench dataset and a 9.4% improvement on trojaned object detection model scanning, offering a promising new direction for reliable backdoor defense without reliance on extensive data or strong prior assumptions about triggers. The code is available at https://github.com/AdaptiveMotorControlLab/DISTIL.



## **4. Cryptanalysis of LC-MUME: A Lightweight Certificateless Multi-User Matchmaking Encryption for Mobile Devices**

cs.CR

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22674v1) [paper-pdf](http://arxiv.org/pdf/2507.22674v1)

**Authors**: Ramprasad Sarkar

**Abstract**: Yang et al. proposed a lightweight certificateless multiuser matchmaking encryption (LC-MUME) scheme for mobile devices, published in IEEE Transactions on Information Forensics and Security (TIFS) (DOI: 10.1109/TIFS.2023.3321961). Their construction aims to reduce computational and communication overhead within a one-to-many certificateless cryptographic framework. The authors claim that their scheme satisfies existential unforgeability under chosen-message attacks (EUF-CMA) in the random oracle model. However, our cryptanalytic study demonstrates that the scheme fails to meet this critical security requirement. In particular, we show that a Type-I adversary can successfully forge a valid ciphertext without possessing the complete private key of the sender. Both theoretical analysis and practical implementation confirm that this attack can be mounted with minimal computational cost. To address these weaknesses, we propose a modification strategy to strengthen the security of matchmaking encryption schemes in mobile computing environments.



## **5. Don't Lag, RAG: Training-Free Adversarial Detection Using RAG**

cs.AI

Accepted at VecDB @ ICML 2025

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2504.04858v3) [paper-pdf](http://arxiv.org/pdf/2504.04858v3)

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types, all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs, including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95 percent classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98 percent, but remains closed-source. Experimental results demonstrate VRAG's effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks.



## **6. Diffusion-based Adversarial Identity Manipulation for Facial Privacy Protection**

cs.CV

Accepted by ACM MM 2025

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2504.21646v3) [paper-pdf](http://arxiv.org/pdf/2504.21646v3)

**Authors**: Liqin Wang, Qianyue Hu, Wei Lu, Xiangyang Luo

**Abstract**: The success of face recognition (FR) systems has led to serious privacy concerns due to potential unauthorized surveillance and user tracking on social networks. Existing methods for enhancing privacy fail to generate natural face images that can protect facial privacy. In this paper, we propose diffusion-based adversarial identity manipulation (DiffAIM) to generate natural and highly transferable adversarial faces against malicious FR systems. To be specific, we manipulate facial identity within the low-dimensional latent space of a diffusion model. This involves iteratively injecting gradient-based adversarial identity guidance during the reverse diffusion process, progressively steering the generation toward the desired adversarial faces. The guidance is optimized for identity convergence towards a target while promoting semantic divergence from the source, facilitating effective impersonation while maintaining visual naturalness. We further incorporate structure-preserving regularization to preserve facial structure consistency during manipulation. Extensive experiments on both face verification and identification tasks demonstrate that compared with the state-of-the-art, DiffAIM achieves stronger black-box attack transferability while maintaining superior visual quality. We also demonstrate the effectiveness of the proposed approach for commercial FR APIs, including Face++ and Aliyun.



## **7. Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs**

cs.CL

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22564v1) [paper-pdf](http://arxiv.org/pdf/2507.22564v1)

**Authors**: Xikang Yang, Biyu Zhou, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems.



## **8. Ownership Verification of DNN Models Using White-Box Adversarial Attacks with Specified Probability Manipulation**

cs.LG

Accepted to EUSIPCO 2025

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2505.17579v3) [paper-pdf](http://arxiv.org/pdf/2505.17579v3)

**Authors**: Teruki Sano, Minoru Kuribayashi, Masao Sakai, Shuji Isobe, Eisuke Koizumi

**Abstract**: In this paper, we propose a novel framework for ownership verification of deep neural network (DNN) models for image classification tasks. It allows verification of model identity by both the rightful owner and third party without presenting the original model. We assume a gray-box scenario where an unauthorized user owns a model that is illegally copied from the original model, provides services in a cloud environment, and the user throws images and receives the classification results as a probability distribution of output classes. The framework applies a white-box adversarial attack to align the output probability of a specific class to a designated value. Due to the knowledge of original model, it enables the owner to generate such adversarial examples. We propose a simple but effective adversarial attack method based on the iterative Fast Gradient Sign Method (FGSM) by introducing control parameters. Experimental results confirm the effectiveness of the identification of DNN models using adversarial attack.



## **9. RCR-AF: Enhancing Model Generalization via Rademacher Complexity Reduction Activation Function**

cs.LG

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22446v1) [paper-pdf](http://arxiv.org/pdf/2507.22446v1)

**Authors**: Yunrui Yu, Kafeng Wang, Hang Su, Jun Zhu

**Abstract**: Despite their widespread success, deep neural networks remain critically vulnerable to adversarial attacks, posing significant risks in safety-sensitive applications. This paper investigates activation functions as a crucial yet underexplored component for enhancing model robustness. We propose a Rademacher Complexity Reduction Activation Function (RCR-AF), a novel activation function designed to improve both generalization and adversarial resilience. RCR-AF uniquely combines the advantages of GELU (including smoothness, gradient stability, and negative information retention) with ReLU's desirable monotonicity, while simultaneously controlling both model sparsity and capacity through built-in clipping mechanisms governed by two hyperparameters, $\alpha$ and $\gamma$. Our theoretical analysis, grounded in Rademacher complexity, demonstrates that these parameters directly modulate the model's Rademacher complexity, offering a principled approach to enhance robustness. Comprehensive empirical evaluations show that RCR-AF consistently outperforms widely-used alternatives (ReLU, GELU, and Swish) in both clean accuracy under standard training and in adversarial robustness within adversarial training paradigms.



## **10. Theoretical Analysis of Relative Errors in Gradient Computations for Adversarial Attacks with CE Loss**

cs.LG

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22428v1) [paper-pdf](http://arxiv.org/pdf/2507.22428v1)

**Authors**: Yunrui Yu, Hang Su, Cheng-zhong Xu, Zhizhong Su, Jun Zhu

**Abstract**: Gradient-based adversarial attacks using the Cross-Entropy (CE) loss often suffer from overestimation due to relative errors in gradient computation induced by floating-point arithmetic. This paper provides a rigorous theoretical analysis of these errors, conducting the first comprehensive study of floating-point computation errors in gradient-based attacks across four distinct scenarios: (i) unsuccessful untargeted attacks, (ii) successful untargeted attacks, (iii) unsuccessful targeted attacks, and (iv) successful targeted attacks. We establish theoretical foundations characterizing the behavior of relative numerical errors under different attack conditions, revealing previously unknown patterns in gradient computation instability, and identify floating-point underflow and rounding as key contributors. Building on this insight, we propose the Theoretical MIFPE (T-MIFPE) loss function, which incorporates an optimal scaling factor $T = t^*$ to minimize the impact of floating-point errors, thereby enhancing the accuracy of gradient computation in adversarial attacks. Extensive experiments on the MNIST, CIFAR-10, and CIFAR-100 datasets demonstrate that T-MIFPE outperforms existing loss functions, including CE, C\&W, DLR, and MIFPE, in terms of attack potency and robustness evaluation accuracy.



## **11. Benchmarking Fraud Detectors on Private Graph Data**

cs.CR

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22347v1) [paper-pdf](http://arxiv.org/pdf/2507.22347v1)

**Authors**: Alexander Goldberg, Giulia Fanti, Nihar Shah, Zhiwei Steven Wu

**Abstract**: We introduce the novel problem of benchmarking fraud detectors on private graph-structured data. Currently, many types of fraud are managed in part by automated detection algorithms that operate over graphs. We consider the scenario where a data holder wishes to outsource development of fraud detectors to third parties (e.g., vendors or researchers). The third parties submit their fraud detectors to the data holder, who evaluates these algorithms on a private dataset and then publicly communicates the results. We propose a realistic privacy attack on this system that allows an adversary to de-anonymize individuals' data based only on the evaluation results. In simulations of a privacy-sensitive benchmark for facial recognition algorithms by the National Institute of Standards and Technology (NIST), our attack achieves near perfect accuracy in identifying whether individuals' data is present in a private dataset, with a True Positive Rate of 0.98 at a False Positive Rate of 0.00. We then study how to benchmark algorithms while satisfying a formal differential privacy (DP) guarantee. We empirically evaluate two classes of solutions: subsample-and-aggregate and DP synthetic graph data. We demonstrate through extensive experiments that current approaches do not provide utility when guaranteeing DP. Our results indicate that the error arising from DP trades off between bias from distorting graph structure and variance from adding random noise. Current methods lie on different points along this bias-variance trade-off, but more complex methods tend to require high-variance noise addition, undermining utility.



## **12. Resilient State Recovery using Prior Measurement Support Information**

math.OC

To be published in SIAM Journal on Control and Optimization

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22340v1) [paper-pdf](http://arxiv.org/pdf/2507.22340v1)

**Authors**: Yu Zheng, Olugbenga Moses Anubi, Warren E. Dixon

**Abstract**: Resilient state recovery of cyber-physical systems has attracted much research attention due to the unique challenges posed by the tight coupling between communication, computation, and the underlying physics of such systems. By modeling attacks as additive adversary signals to a sparse subset of measurements, this resilient recovery problem can be formulated as an error correction problem. To achieve exact state recovery, most existing results require less than $50\%$ of the measurement nodes to be compromised, which limits the resiliency of the estimators. In this paper, we show that observer resiliency can be further improved by incorporating data-driven prior information. We provide an analytical bridge between the precision of prior information and the resiliency of the estimator. By quantifying the relationship between the estimation error of the weighted $\ell_1$ observer and the precision of the support prior. This quantified relationship provides guidance for the estimator's weight design to achieve optimal resiliency. Several numerical simulations and an application case study are presented to validate the theoretical claims.



## **13. Can adversarial attacks by large language models be attributed?**

cs.AI

22 pages, 5 figures, 2 tables

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2411.08003v3) [paper-pdf](http://arxiv.org/pdf/2411.08003v3)

**Authors**: Manuel Cebrian, Andres Abeliuk, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation campaigns-presents significant challenges that are likely to grow in importance. We approach this attribution problem from both a theoretical and an empirical perspective, drawing on formal language theory (identification in the limit) and data-driven analysis of the expanding LLM ecosystem. By modeling an LLM's set of possible outputs as a formal language, we analyze whether finite samples of text can uniquely pinpoint the originating model. Our results show that, under mild assumptions of overlapping capabilities among models, certain classes of LLMs are fundamentally non-identifiable from their outputs alone. We delineate four regimes of theoretical identifiability: (1) an infinite class of deterministic (discrete) LLM languages is not identifiable (Gold's classical result from 1967); (2) an infinite class of probabilistic LLMs is also not identifiable (by extension of the deterministic case); (3) a finite class of deterministic LLMs is identifiable (consistent with Angluin's tell-tale criterion); and (4) even a finite class of probabilistic LLMs can be non-identifiable (we provide a new counterexample establishing this negative result). Complementing these theoretical insights, we quantify the explosion in the number of plausible model origins (hypothesis space) for a given output in recent years. Even under conservative assumptions-each open-source model fine-tuned on at most one new dataset-the count of distinct candidate models doubles approximately every 0.5 years, and allowing multi-dataset fine-tuning combinations yields doubling times as short as 0.28 years. This combinatorial growth, alongside the extraordinary computational cost of brute-force likelihood attribution across all models and potential users, renders exhaustive attribution infeasible in practice.



## **14. Persistent Backdoor Attacks in Continual Learning**

cs.LG

19 pages, 20 figures, 6 tables

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2409.13864v3) [paper-pdf](http://arxiv.org/pdf/2409.13864v3)

**Authors**: Zhen Guo, Abhinav Kumar, Reza Tourani

**Abstract**: Backdoor attacks pose a significant threat to neural networks, enabling adversaries to manipulate model outputs on specific inputs, often with devastating consequences, especially in critical applications. While backdoor attacks have been studied in various contexts, little attention has been given to their practicality and persistence in continual learning, particularly in understanding how the continual updates to model parameters, as new data distributions are learned and integrated, impact the effectiveness of these attacks over time. To address this gap, we introduce two persistent backdoor attacks-Blind Task Backdoor and Latent Task Backdoor-each leveraging minimal adversarial influence. Our blind task backdoor subtly alters the loss computation without direct control over the training process, while the latent task backdoor influences only a single task's training, with all other tasks trained benignly. We evaluate these attacks under various configurations, demonstrating their efficacy with static, dynamic, physical, and semantic triggers. Our results show that both attacks consistently achieve high success rates across different continual learning algorithms, while effectively evading state-of-the-art defenses, such as SentiNet and I-BAU.



## **15. Teach Me to Trick: Exploring Adversarial Transferability via Knowledge Distillation**

cs.LG

10 pages, 4 figures

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21992v1) [paper-pdf](http://arxiv.org/pdf/2507.21992v1)

**Authors**: Siddhartha Pradhan, Shikshya Shiwakoti, Neha Bathuri

**Abstract**: We investigate whether knowledge distillation (KD) from multiple heterogeneous teacher models can enhance the generation of transferable adversarial examples. A lightweight student model is trained using two KD strategies: curriculum-based switching and joint optimization, with ResNet50 and DenseNet-161 as teachers. The trained student is then used to generate adversarial examples using FG, FGS, and PGD attacks, which are evaluated against a black-box target model (GoogLeNet). Our results show that student models distilled from multiple teachers achieve attack success rates comparable to ensemble-based baselines, while reducing adversarial example generation time by up to a factor of six. An ablation study further reveals that lower temperature settings and the inclusion of hard-label supervision significantly enhance transferability. These findings suggest that KD can serve not only as a model compression technique but also as a powerful tool for improving the efficiency and effectiveness of black-box adversarial attacks.



## **16. ZIUM: Zero-Shot Intent-Aware Adversarial Attack on Unlearned Models**

cs.CV

Accepted to ICCV2025

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21985v1) [paper-pdf](http://arxiv.org/pdf/2507.21985v1)

**Authors**: Hyun Jun Yook, Ga San Jhun, Jae Hyun Cho, Min Jeon, Donghyun Kim, Tae Hyung Kim, Youn Kyu Lee

**Abstract**: Machine unlearning (MU) removes specific data points or concepts from deep learning models to enhance privacy and prevent sensitive content generation. Adversarial prompts can exploit unlearned models to generate content containing removed concepts, posing a significant security risk. However, existing adversarial attack methods still face challenges in generating content that aligns with an attacker's intent while incurring high computational costs to identify successful prompts. To address these challenges, we propose ZIUM, a Zero-shot Intent-aware adversarial attack on Unlearned Models, which enables the flexible customization of target attack images to reflect an attacker's intent. Additionally, ZIUM supports zero-shot adversarial attacks without requiring further optimization for previously attacked unlearned concepts. The evaluation across various MU scenarios demonstrated ZIUM's effectiveness in successfully customizing content based on user-intent prompts while achieving a superior attack success rate compared to existing methods. Moreover, its zero-shot adversarial attack significantly reduces the attack time for previously attacked unlearned concepts.



## **17. Anyone Can Jailbreak: Prompt-Based Attacks on LLMs and T2Is**

cs.CV

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21820v1) [paper-pdf](http://arxiv.org/pdf/2507.21820v1)

**Authors**: Ahmed B Mustafa, Zihan Ye, Yang Lu, Michael P Pound, Shreyank N Gowda

**Abstract**: Despite significant advancements in alignment and content moderation, large language models (LLMs) and text-to-image (T2I) systems remain vulnerable to prompt-based attacks known as jailbreaks. Unlike traditional adversarial examples requiring expert knowledge, many of today's jailbreaks are low-effort, high-impact crafted by everyday users with nothing more than cleverly worded prompts. This paper presents a systems-style investigation into how non-experts reliably circumvent safety mechanisms through techniques such as multi-turn narrative escalation, lexical camouflage, implication chaining, fictional impersonation, and subtle semantic edits. We propose a unified taxonomy of prompt-level jailbreak strategies spanning both text-output and T2I models, grounded in empirical case studies across popular APIs. Our analysis reveals that every stage of the moderation pipeline, from input filtering to output validation, can be bypassed with accessible strategies. We conclude by highlighting the urgent need for context-aware defenses that reflect the ease with which these jailbreaks can be reproduced in real-world settings.



## **18. Adversarial Defence without Adversarial Defence: Enhancing Language Model Robustness via Instance-level Principal Component Removal**

cs.CL

This paper was accepted with an A-decision to Transactions of the  Association for Computational Linguistics. This version is the  pre-publication version prior to MIT Press production

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21750v1) [paper-pdf](http://arxiv.org/pdf/2507.21750v1)

**Authors**: Yang Wang, Chenghao Xiao, Yizhi Li, Stuart E. Middleton, Noura Al Moubayed, Chenghua Lin

**Abstract**: Pre-trained language models (PLMs) have driven substantial progress in natural language processing but remain vulnerable to adversarial attacks, raising concerns about their robustness in real-world applications. Previous studies have sought to mitigate the impact of adversarial attacks by introducing adversarial perturbations into the training process, either implicitly or explicitly. While both strategies enhance robustness, they often incur high computational costs. In this work, we propose a simple yet effective add-on module that enhances the adversarial robustness of PLMs by removing instance-level principal components, without relying on conventional adversarial defences or perturbing the original training data. Our approach transforms the embedding space to approximate Gaussian properties, thereby reducing its susceptibility to adversarial perturbations while preserving semantic relationships. This transformation aligns embedding distributions in a way that minimises the impact of adversarial noise on decision boundaries, enhancing robustness without requiring adversarial examples or costly training-time augmentation. Evaluations on eight benchmark datasets show that our approach improves adversarial robustness while maintaining comparable before-attack accuracy to baselines, achieving a balanced trade-off between robustness and generalisation.



## **19. Defending Against Unforeseen Failure Modes with Latent Adversarial Training**

cs.CR

See also followup work at arXiv:2407.15549

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2403.05030v6) [paper-pdf](http://arxiv.org/pdf/2403.05030v6)

**Authors**: Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell

**Abstract**: Despite extensive diagnostics and debugging by developers, AI systems sometimes exhibit harmful unintended behaviors. Finding and fixing these is challenging because the attack surface is so large -- it is not tractable to exhaustively search for inputs that may elicit harmful behaviors. Red-teaming and adversarial training (AT) are commonly used to improve robustness, however, they empirically struggle to fix failure modes that differ from the attacks used during training. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without leveraging knowledge of what they are or using inputs that elicit them. LAT makes use of the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. Here, we use it to defend against failure modes without examples that elicit them. Specifically, we use LAT to remove backdoors and defend against held-out classes of adversarial attacks. We show in image classification, text classification, and text generation tasks that LAT usually improves both robustness to novel attacks and performance on clean data relative to AT. This suggests that LAT can be a promising tool for defending against failure modes that are not explicitly identified by developers.



## **20. Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs**

cs.LG

Code at https://github.com/aengusl/latent-adversarial-training.  Models at https://huggingface.co/LLM-LAT

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2407.15549v3) [paper-pdf](http://arxiv.org/pdf/2407.15549v3)

**Authors**: Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, Stephen Casper

**Abstract**: Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of 'jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.



## **21. PRISM: Programmatic Reasoning with Image Sequence Manipulation for LVLM Jailbreaking**

cs.CR

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21540v1) [paper-pdf](http://arxiv.org/pdf/2507.21540v1)

**Authors**: Quanchen Zou, Zonghao Ying, Moyang Chen, Wenzhuo Xu, Yisong Xiao, Yakai Li, Deyue Zhang, Dongdong Yang, Zhao Liu, Xiangzheng Zhang

**Abstract**: The increasing sophistication of large vision-language models (LVLMs) has been accompanied by advances in safety alignment mechanisms designed to prevent harmful content generation. However, these defenses remain vulnerable to sophisticated adversarial attacks. Existing jailbreak methods typically rely on direct and semantically explicit prompts, overlooking subtle vulnerabilities in how LVLMs compose information over multiple reasoning steps. In this paper, we propose a novel and effective jailbreak framework inspired by Return-Oriented Programming (ROP) techniques from software security. Our approach decomposes a harmful instruction into a sequence of individually benign visual gadgets. A carefully engineered textual prompt directs the sequence of inputs, prompting the model to integrate the benign visual gadgets through its reasoning process to produce a coherent and harmful output. This makes the malicious intent emergent and difficult to detect from any single component. We validate our method through extensive experiments on established benchmarks including SafeBench and MM-SafetyBench, targeting popular LVLMs. Results show that our approach consistently and substantially outperforms existing baselines on state-of-the-art models, achieving near-perfect attack success rates (over 0.90 on SafeBench) and improving ASR by up to 0.39. Our findings reveal a critical and underexplored vulnerability that exploits the compositional reasoning abilities of LVLMs, highlighting the urgent need for defenses that secure the entire reasoning process.



## **22. Can We End the Cat-and-Mouse Game? Simulating Self-Evolving Phishing Attacks with LLMs and Genetic Algorithms**

cs.CR

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21538v1) [paper-pdf](http://arxiv.org/pdf/2507.21538v1)

**Authors**: Seiji Sato, Tetsushi Ohki, Masakatsu Nishigaki

**Abstract**: Anticipating emerging attack methodologies is crucial for proactive cybersecurity. Recent advances in Large Language Models (LLMs) have enabled the automated generation of phishing messages and accelerated research into potential attack techniques. However, predicting future threats remains challenging due to reliance on existing training data. To address this limitation, we propose a novel framework that integrates LLM-based phishing attack simulations with a genetic algorithm in a psychological context, enabling phishing strategies to evolve dynamically through adversarial interactions with simulated victims. Through simulations using Llama 3.1, we demonstrate that (1) self-evolving phishing strategies employ increasingly sophisticated psychological manipulation techniques, surpassing naive LLM-generated attacks, (2) variations in a victim's prior knowledge significantly influence the evolution of attack strategies, and (3) adversarial interactions between evolving attacks and adaptive defenses create a cat-and-mouse dynamic, revealing an inherent asymmetry in cybersecurity -- attackers continuously refine their methods, whereas defenders struggle to comprehensively counter all evolving threats. Our approach provides a scalable, cost-effective method for analyzing the evolution of phishing strategies and defenses, offering insights into future social engineering threats and underscoring the necessity of proactive cybersecurity measures.



## **23. NCCR: to Evaluate the Robustness of Neural Networks and Adversarial Examples**

cs.CR

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21483v1) [paper-pdf](http://arxiv.org/pdf/2507.21483v1)

**Authors**: Pu Shi

**Abstract**: Neural networks have received a lot of attention recently, and related security issues have come with it. Many studies have shown that neural networks are vulnerable to adversarial examples that have been artificially perturbed with modification, which is too small to be distinguishable by human perception. Different attacks and defenses have been proposed to solve these problems, but there is little research on evaluating the robustness of neural networks and their inputs. In this work, we propose a metric called the neuron cover change rate (NCCR) to measure the ability of deep learning models to resist attacks and the stability of adversarial examples. NCCR monitors alterations in the output of specifically chosen neurons when the input is perturbed, and networks with a smaller degree of variation are considered to be more robust. The results of the experiment on image recognition and the speaker recognition model show that our metrics can provide a good assessment of the robustness of neural networks or their inputs. It can also be used to detect whether an input is adversarial or not, as adversarial examples are always less robust.



## **24. PAR-AdvGAN: Improving Adversarial Attack Capability with Progressive Auto-Regression AdvGAN**

cs.LG

Best paper award of ECML-PKDD 2025

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2502.12207v2) [paper-pdf](http://arxiv.org/pdf/2502.12207v2)

**Authors**: Jiayu Zhang, Zhiyu Zhu, Xinyi Wang, Silin Liao, Zhibo Jin, Flora D. Salim, Huaming Chen

**Abstract**: Deep neural networks have demonstrated remarkable performance across various domains. However, they are vulnerable to adversarial examples, which can lead to erroneous predictions. Generative Adversarial Networks (GANs) can leverage the generators and discriminators model to quickly produce high-quality adversarial examples. Since both modules train in a competitive and simultaneous manner, GAN-based algorithms like AdvGAN can generate adversarial examples with better transferability compared to traditional methods. However, the generation of perturbations is usually limited to a single iteration, preventing these examples from fully exploiting the potential of the methods. To tackle this issue, we introduce a novel approach named Progressive Auto-Regression AdvGAN (PAR-AdvGAN). It incorporates an auto-regressive iteration mechanism within a progressive generation network to craft adversarial examples with enhanced attack capability. We thoroughly evaluate our PAR-AdvGAN method with a large-scale experiment, demonstrating its superior performance over various state-of-the-art black-box adversarial attacks, as well as the original AdvGAN.Moreover, PAR-AdvGAN significantly accelerates the adversarial example generation, i.e., achieving the speeds of up to 335.5 frames per second on Inception-v3 model, outperforming the gradient-based transferable attack algorithms. Our code is available at: https://github.com/LMBTough/PAR



## **25. Cascading and Proxy Membership Inference Attacks**

cs.CR

Our code is available at: https://github.com/zealscott/MIA

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21412v1) [paper-pdf](http://arxiv.org/pdf/2507.21412v1)

**Authors**: Yuntao Du, Jiacheng Li, Yuetian Chen, Kaiyuan Zhang, Zhizhen Yuan, Hanshen Xiao, Bruno Ribeiro, Ninghui Li

**Abstract**: A Membership Inference Attack (MIA) assesses how much a trained machine learning model reveals about its training data by determining whether specific query instances were included in the dataset. We classify existing MIAs into adaptive or non-adaptive, depending on whether the adversary is allowed to train shadow models on membership queries. In the adaptive setting, where the adversary can train shadow models after accessing query instances, we highlight the importance of exploiting membership dependencies between instances and propose an attack-agnostic framework called Cascading Membership Inference Attack (CMIA), which incorporates membership dependencies via conditional shadow training to boost membership inference performance.   In the non-adaptive setting, where the adversary is restricted to training shadow models before obtaining membership queries, we introduce Proxy Membership Inference Attack (PMIA). PMIA employs a proxy selection strategy that identifies samples with similar behaviors to the query instance and uses their behaviors in shadow models to perform a membership posterior odds test for membership inference. We provide theoretical analyses for both attacks, and extensive experimental results demonstrate that CMIA and PMIA substantially outperform existing MIAs in both settings, particularly in the low false-positive regime, which is crucial for evaluating privacy risks.



## **26. FedStrategist: A Meta-Learning Framework for Adaptive and Robust Aggregation in Federated Learning**

cs.LG

24 pages, 8 figures. This work is intended for a journal submission

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.14322v2) [paper-pdf](http://arxiv.org/pdf/2507.14322v2)

**Authors**: Md Rafid Haque, Abu Raihan Mostofa Kamal, Md. Azam Hossain

**Abstract**: Federated Learning (FL) offers a paradigm for privacy-preserving collaborative AI, but its decentralized nature creates significant vulnerabilities to model poisoning attacks. While numerous static defenses exist, their effectiveness is highly context-dependent, often failing against adaptive adversaries or in heterogeneous data environments. This paper introduces FedStrategist, a novel meta-learning framework that reframes robust aggregation as a real-time, cost-aware control problem. We design a lightweight contextual bandit agent that dynamically selects the optimal aggregation rule from an arsenal of defenses based on real-time diagnostic metrics. Through comprehensive experiments, we demonstrate that no single static rule is universally optimal. We show that our adaptive agent successfully learns superior policies across diverse scenarios, including a ``Krum-favorable" environment and against a sophisticated "stealth" adversary designed to neutralize specific diagnostic signals. Critically, we analyze the paradoxical scenario where a non-robust baseline achieves high but compromised accuracy, and demonstrate that our agent learns a conservative policy to prioritize model integrity. Furthermore, we prove the agent's policy is controllable via a single "risk tolerance" parameter, allowing practitioners to explicitly manage the trade-off between performance and security. Our work provides a new, practical, and analyzable approach to creating resilient and intelligent decentralized AI systems.



## **27. Radio Adversarial Attacks on EMG-based Gesture Recognition Networks**

cs.CR

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.21387v1) [paper-pdf](http://arxiv.org/pdf/2507.21387v1)

**Authors**: Hongyi Xie

**Abstract**: Surface electromyography (EMG) enables non-invasive human-computer interaction in rehabilitation, prosthetics, and virtual reality. While deep learning models achieve over 97% classification accuracy, their vulnerability to adversarial attacks remains largely unexplored in the physical domain. We present ERa Attack, the first radio frequency (RF) adversarial method targeting EMG devices through intentional electromagnetic interference (IEMI). Using low-power software-defined radio transmitters, attackers inject optimized RF perturbations to mislead downstream models. Our approach bridges digital and physical domains: we generate adversarial perturbations using Projected Gradient Descent, extract 50-150 Hz components via inverse STFT, and employ synchronization-free strategies (constant spectrum noise or narrowband modulation). Perturbations, constrained to 1-10% of signal amplitude, are amplitude-modulated onto 433 MHz carriers. Experiments on the Myo Dataset (7 gestures, 350 samples) demonstrate significant impact: at 1 meter and 0 dBm transmission power, classification accuracy drops from 97.8% to 58.3%, with 41.7% misclassification rate and 25.6% targeted attack success rate. Attack effectiveness decreases exponentially with distance, recovering to 85% accuracy at 3 meters. Increasing power to 10 dBm reduces accuracy by an additional 15% at 1 meter. This work pioneers RF-based adversarial attacks on EMG recognition systems, revealing critical vulnerabilities in safety-critical applications. We quantify attack effectiveness across different perturbation modes and distances, and propose defenses including hardware shielding, spectrum monitoring, and adversarial training. Our findings inform the design of robust EMG systems against electromagnetic threats.



## **28. On Post-Quantum Cryptography Authentication for Quantum Key Distribution**

quant-ph

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.21325v1) [paper-pdf](http://arxiv.org/pdf/2507.21325v1)

**Authors**: Juan Antonio Vieira Giestinhas, Timothy Spiller

**Abstract**: The traditional way for a Quantum Key Distribution (QKD) user to join a quantum network is by authenticating themselves using pre-shared key material. While this approach is sufficient for small-scale networks, it becomes impractical as the network grows, due to the total quadratic increase in the number of pre-shared keys required. To address this scalability issue, Public Key Infrastructure (PKI) combined with Post-Quantum Cryptography (PQC) offers a more scalable solution, allowing users to authenticate the QKD traffic remotely to obtain information-theoretical secure (ITS) keys under the presented assumptions. Unlike traditional PKI, which relies on classical cryptographic algorithms such as RSA, the approach presented in this paper leverages PQC algorithms that are believed to be resistant to quantum attacks. Similarly to the SIGMA or TLS protocols, authentication, confidentiality, and integrity are achievable against bounded adversaries to ensure secure and scalable quantum networks.



## **29. Adversarial attacks and defenses in explainable artificial intelligence: A survey**

cs.CR

Accepted by Information Fusion

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2306.06123v4) [paper-pdf](http://arxiv.org/pdf/2306.06123v4)

**Authors**: Hubert Baniecki, Przemyslaw Biecek

**Abstract**: Explainable artificial intelligence (XAI) methods are portrayed as a remedy for debugging and trusting statistical and deep learning models, as well as interpreting their predictions. However, recent advances in adversarial machine learning (AdvML) highlight the limitations and vulnerabilities of state-of-the-art explanation methods, putting their security and trustworthiness into question. The possibility of manipulating, fooling or fairwashing evidence of the model's reasoning has detrimental consequences when applied in high-stakes decision-making and knowledge discovery. This survey provides a comprehensive overview of research concerning adversarial attacks on explanations of machine learning models, as well as fairness metrics. We introduce a unified notation and taxonomy of methods facilitating a common ground for researchers and practitioners from the intersecting research fields of AdvML and XAI. We discuss how to defend against attacks and design robust interpretation methods. We contribute a list of existing insecurities in XAI and outline the emerging research directions in adversarial XAI (AdvXAI). Future work should address improving explanation methods and evaluation protocols to take into account the reported safety issues.



## **30. Improving Adversarial Robustness Through Adaptive Learning-Driven Multi-Teacher Knowledge Distillation**

cs.CV

11 pages

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.20996v1) [paper-pdf](http://arxiv.org/pdf/2507.20996v1)

**Authors**: Hayat Ullah, Syed Muhammad Talha Zaidi, Arslan Munir

**Abstract**: Convolutional neural networks (CNNs) excel in computer vision but are susceptible to adversarial attacks, crafted perturbations designed to mislead predictions. Despite advances in adversarial training, a gap persists between model accuracy and robustness. To mitigate this issue, in this paper, we present a multi-teacher adversarial robustness distillation using an adaptive learning strategy. Specifically, our proposed method first trained multiple clones of a baseline CNN model using an adversarial training strategy on a pool of perturbed data acquired through different adversarial attacks. Once trained, these adversarially trained models are used as teacher models to supervise the learning of a student model on clean data using multi-teacher knowledge distillation. To ensure an effective robustness distillation, we design an adaptive learning strategy that controls the knowledge contribution of each model by assigning weights as per their prediction precision. Distilling knowledge from adversarially pre-trained teacher models not only enhances the learning capabilities of the student model but also empowers it with the capacity to withstand different adversarial attacks, despite having no exposure to adversarial data. To verify our claims, we extensively evaluated our proposed method on MNIST-Digits and Fashion-MNIST datasets across diverse experimental settings. The obtained results exhibit the efficacy of our multi-teacher adversarial distillation and adaptive learning strategy, enhancing CNNs' adversarial robustness against various adversarial attacks.



## **31. A Large Language Model-Supported Threat Modeling Framework for Transportation Cyber-Physical Systems**

cs.CR

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2506.00831v2) [paper-pdf](http://arxiv.org/pdf/2506.00831v2)

**Authors**: M Sabbir Salek, Mashrur Chowdhury, Muhaimin Bin Munir, Yuchen Cai, Mohammad Imtiaz Hasan, Jean-Michel Tine, Latifur Khan, Mizanur Rahman

**Abstract**: Existing threat modeling frameworks related to transportation cyber-physical systems (CPS) are often narrow in scope, labor-intensive, and require substantial cybersecurity expertise. To this end, we introduce the Transportation Cybersecurity and Resiliency Threat Modeling Framework (TraCR-TMF), a large language model (LLM)-based threat modeling framework for transportation CPS that requires limited cybersecurity expert intervention. TraCR-TMF identifies threats, potential attack techniques, and relevant countermeasures for transportation CPS. Three LLM-based approaches support these identifications: (i) a retrieval-augmented generation approach requiring no cybersecurity expert intervention, (ii) an in-context learning approach with low expert intervention, and (iii) a supervised fine-tuning approach with moderate expert intervention. TraCR-TMF offers LLM-based attack path identification for critical assets based on vulnerabilities across transportation CPS entities. Additionally, it incorporates the Common Vulnerability Scoring System (CVSS) scores of known exploited vulnerabilities to prioritize threat mitigations. The framework was evaluated through two cases. First, the framework identified relevant attack techniques for various transportation CPS applications, 73% of which were validated by cybersecurity experts as correct. Second, the framework was used to identify attack paths for a target asset in a real-world cyberattack incident. TraCR-TMF successfully predicted exploitations, like lateral movement of adversaries, data exfiltration, and data encryption for ransomware, as reported in the incident. These findings show the efficacy of TraCR-TMF in transportation CPS threat modeling, while reducing the need for extensive involvement of cybersecurity experts. To facilitate real-world adoptions, all our codes are shared via an open-source repository.



## **32. Enhancing generalization in high energy physics using white-box adversarial attacks**

hep-ph

14 pages, 7 figures, 10 tables, 3 algorithms, published in Physical  Review D (PRD), presented at the ML4Jets 2024 conference

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2411.09296v3) [paper-pdf](http://arxiv.org/pdf/2411.09296v3)

**Authors**: Franck Rothen, Samuel Klein, Matthew Leigh, Tobias Golling

**Abstract**: Machine learning is becoming increasingly popular in the context of particle physics. Supervised learning, which uses labeled Monte Carlo (MC) simulations, remains one of the most widely used methods for discriminating signals beyond the Standard Model. However, this paper suggests that supervised models may depend excessively on artifacts and approximations from Monte Carlo simulations, potentially limiting their ability to generalize well to real data. This study aims to enhance the generalization properties of supervised models by reducing the sharpness of local minima. It reviews the application of four distinct white-box adversarial attacks in the context of classifying Higgs boson decay signals. The attacks are divided into weight-space attacks and feature-space attacks. To study and quantify the sharpness of different local minima, this paper presents two analysis methods: gradient ascent and reduced Hessian eigenvalue analysis. The results show that white-box adversarial attacks significantly improve generalization performance, albeit with increased computational complexity.



## **33. Next-Generation Quantum Neural Networks: Enhancing Efficiency, Security, and Privacy**

quant-ph

4 pages, 6 figures. Accepted at IOLTS 2025

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.20537v1) [paper-pdf](http://arxiv.org/pdf/2507.20537v1)

**Authors**: Nouhaila Innan, Muhammad Kashif, Alberto Marchisio, Mohamed Bennai, Muhammad Shafique

**Abstract**: This paper provides an integrated perspective on addressing key challenges in developing reliable and secure Quantum Neural Networks (QNNs) in the Noisy Intermediate-Scale Quantum (NISQ) era. In this paper, we present an integrated framework that leverages and combines existing approaches to enhance QNN efficiency, security, and privacy. Specifically, established optimization strategies, including efficient parameter initialization, residual quantum circuit connections, and systematic quantum architecture exploration, are integrated to mitigate issues such as barren plateaus and error propagation. Moreover, the methodology incorporates current defensive mechanisms against adversarial attacks. Finally, Quantum Federated Learning (QFL) is adopted within this framework to facilitate privacy-preserving collaborative training across distributed quantum systems. Collectively, this synthesized approach seeks to enhance the robustness and real-world applicability of QNNs, laying the foundation for reliable quantum-enhanced machine learning applications in finance, healthcare, and cybersecurity.



## **34. A Hybrid Classical-Quantum Rainbow Table Attack on Human Passwords**

cs.CR

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.14600v2) [paper-pdf](http://arxiv.org/pdf/2507.14600v2)

**Authors**: MA. Khajeian

**Abstract**: Long, human-generated passwords pose significant challenges to both classical and quantum attacks due to their irregular structure and large search space. In this work, we propose an enhanced classical-quantum hybrid attack specifically designed for this scenario. Our approach constructs rainbow tables using dictionary-based password generation augmented with transformation rules that better capture real-world user behavior. These tables are organized into buckets, enabling faster lookup and reduced space complexity. For the search within each bucket, we employ a distributed exact variant of Grover's algorithm. This method provides deterministic success and significantly lower circuit depth, enhancing robustness against noise-particularly depolarizing errors common in near-term quantum devices. Overall, our hybrid framework improves the efficiency and practicality of password recovery for long, human-readable passwords in realistic adversarial settings.



## **35. Accidental Vulnerability: Factors in Fine-Tuning that Shift Model Safeguards**

cs.CL

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2505.16789v2) [paper-pdf](http://arxiv.org/pdf/2505.16789v2)

**Authors**: Punya Syon Pandey, Samuel Simko, Kellin Pelrine, Zhijing Jin

**Abstract**: As large language models (LLMs) gain popularity, their vulnerability to adversarial attacks emerges as a primary concern. While fine-tuning models on domain-specific datasets is often employed to improve model performance, it can inadvertently introduce vulnerabilities within the underlying model. In this work, we investigate Accidental Vulnerability, unexpected vulnerabilities arising from characteristics of fine-tuning data. We begin by identifying potential correlation factors such as linguistic features, semantic similarity, and toxicity across multiple experimental datasets. We then evaluate the adversarial robustness of these fine-tuned models, analyzing persona shifts and interpretability traits to understand how dataset factors contribute to attack success rates. Lastly, we explore causal relationships that offer new insights into adversarial defense strategies, highlighting the crucial role of dataset design in preserving model alignment. Our code is available at https://github.com/psyonp/accidental_vulnerability.



## **36. Security Challenges in AI Agent Deployment: Insights from a Large Scale Public Competition**

cs.AI

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.20526v1) [paper-pdf](http://arxiv.org/pdf/2507.20526v1)

**Authors**: Andy Zou, Maxwell Lin, Eliot Jones, Micha Nowak, Mateusz Dziemian, Nick Winter, Alexander Grattan, Valent Nathanael, Ayla Croft, Xander Davies, Jai Patel, Robert Kirk, Nate Burnikell, Yarin Gal, Dan Hendrycks, J. Zico Kolter, Matt Fredrikson

**Abstract**: Recent advances have enabled LLM-powered AI agents to autonomously execute complex tasks by combining language model reasoning with tools, memory, and web access. But can these systems be trusted to follow deployment policies in realistic environments, especially under attack? To investigate, we ran the largest public red-teaming competition to date, targeting 22 frontier AI agents across 44 realistic deployment scenarios. Participants submitted 1.8 million prompt-injection attacks, with over 60,000 successfully eliciting policy violations such as unauthorized data access, illicit financial actions, and regulatory noncompliance. We use these results to build the Agent Red Teaming (ART) benchmark - a curated set of high-impact attacks - and evaluate it across 19 state-of-the-art models. Nearly all agents exhibit policy violations for most behaviors within 10-100 queries, with high attack transferability across models and tasks. Importantly, we find limited correlation between agent robustness and model size, capability, or inference-time compute, suggesting that additional defenses are needed against adversarial misuse. Our findings highlight critical and persistent vulnerabilities in today's AI agents. By releasing the ART benchmark and accompanying evaluation framework, we aim to support more rigorous security assessment and drive progress toward safer agent deployment.



## **37. When and Where do Data Poisons Attack Textual Inversion?**

cs.CR

Accepted to ICCV 2025

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.10578v3) [paper-pdf](http://arxiv.org/pdf/2507.10578v3)

**Authors**: Jeremy Styborski, Mingzhi Lyu, Jiayou Lu, Nupur Kapur, Adams Kong

**Abstract**: Poisoning attacks pose significant challenges to the robustness of diffusion models (DMs). In this paper, we systematically analyze when and where poisoning attacks textual inversion (TI), a widely used personalization technique for DMs. We first introduce Semantic Sensitivity Maps, a novel method for visualizing the influence of poisoning on text embeddings. Second, we identify and experimentally verify that DMs exhibit non-uniform learning behavior across timesteps, focusing on lower-noise samples. Poisoning attacks inherit this bias and inject adversarial signals predominantly at lower timesteps. Lastly, we observe that adversarial signals distract learning away from relevant concept regions within training data, corrupting the TI process. Based on these insights, we propose Safe-Zone Training (SZT), a novel defense mechanism comprised of 3 key components: (1) JPEG compression to weaken high-frequency poison signals, (2) restriction to high timesteps during TI training to avoid adversarial signals at lower timesteps, and (3) loss masking to constrain learning to relevant regions. Extensive experiments across multiple poisoning methods demonstrate that SZT greatly enhances the robustness of TI against all poisoning attacks, improving generative quality beyond prior published defenses. Code: www.github.com/JStyborski/Diff_Lab Data: www.github.com/JStyborski/NC10



## **38. EdgeAgentX-DT: Integrating Digital Twins and Generative AI for Resilient Edge Intelligence in Tactical Networks**

cs.LG

13 pages, 6 figures

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.21196v1) [paper-pdf](http://arxiv.org/pdf/2507.21196v1)

**Authors**: Abir Ray

**Abstract**: We introduce EdgeAgentX-DT, an advanced extension of the EdgeAgentX framework that integrates digital twin simulations and generative AI-driven scenario training to significantly enhance edge intelligence in military networks. EdgeAgentX-DT utilizes network digital twins, virtual replicas synchronized with real-world edge devices, to provide a secure, realistic environment for training and validation. Leveraging generative AI methods, such as diffusion models and transformers, the system creates diverse and adversarial scenarios for robust simulation-based agent training. Our multi-layer architecture includes: (1) on-device edge intelligence; (2) digital twin synchronization; and (3) generative scenario training. Experimental simulations demonstrate notable improvements over EdgeAgentX, including faster learning convergence, higher network throughput, reduced latency, and improved resilience against jamming and node failures. A case study involving a complex tactical scenario with simultaneous jamming attacks, agent failures, and increased network loads illustrates how EdgeAgentX-DT sustains operational performance, whereas baseline methods fail. These results highlight the potential of digital-twin-enabled generative training to strengthen edge AI deployments in contested environments.



## **39. Is Crunching Public Data the Right Approach to Detect BGP Hijacks?**

cs.CR

**SubmitDate**: 2025-07-27    [abs](http://arxiv.org/abs/2507.20434v1) [paper-pdf](http://arxiv.org/pdf/2507.20434v1)

**Authors**: Alessandro Giaconia, Muoi Tran, Laurent Vanbever, Stefano Vissicchio

**Abstract**: The Border Gateway Protocol (BGP) remains a fragile pillar of Internet routing. BGP hijacks still occurr daily. While full deployment of Route Origin Validation (ROV) is ongoing, attackers have already adapted, launching post-ROV attacks such as forged-origin hijacks. To detect these, recent approaches like DFOH [Holterbach et al., USENIX NSDI '24] and BEAM [Chen et al., USENIX Security '24] apply machine learning (ML) to analyze data from globally distributed BGP monitors, assuming anomalies will stand out against historical patterns. However, this assumption overlooks a key threat: BGP monitors themselves can be misled by adversaries injecting bogus routes. This paper shows that state-of-the-art hijack detection systems like DFOH and BEAM are vulnerable to data poisoning. Using large-scale BGP simulations, we show that attackers can evade detection with just a handful of crafted announcements beyond the actual hijack. These announcements are indeed sufficient to corrupt the knowledge base used by ML-based defenses and distort the metrics they rely on. Our results highlight a worrying weakness of relying solely on public BGP data.



## **40. Real-time Factuality Assessment from Adversarial Feedback**

cs.CL

**SubmitDate**: 2025-07-27    [abs](http://arxiv.org/abs/2410.14651v3) [paper-pdf](http://arxiv.org/pdf/2410.14651v3)

**Authors**: Sanxing Chen, Yukun Huang, Bhuwan Dhingra

**Abstract**: We show that existing evaluations for assessing the factuality of news from conventional sources, such as claims on fact-checking websites, result in high accuracies over time for LLM-based detectors-even after their knowledge cutoffs. This suggests that recent popular false information from such sources can be easily identified due to its likely presence in pre-training/retrieval corpora or the emergence of salient, yet shallow, patterns in these datasets. Instead, we argue that a proper factuality evaluation dataset should test a model's ability to reason about current events by retrieving and reading related evidence. To this end, we develop a novel pipeline that leverages natural language feedback from a RAG-based detector to iteratively modify real-time news into deceptive variants that challenge LLMs. Our iterative rewrite decreases the binary classification ROC-AUC by an absolute 17.5 percent for a strong RAG-based GPT-4o detector. Our experiments reveal the important role of RAG in both evaluating and generating challenging news examples, as retrieval-free LLM detectors are vulnerable to unseen events and adversarial attacks, while feedback from RAG-based evaluation helps discover more deceitful patterns.



## **41. Manipulating Multimodal Agents via Cross-Modal Prompt Injection**

cs.CV

16 pages, 5 figures

**SubmitDate**: 2025-07-27    [abs](http://arxiv.org/abs/2504.14348v4) [paper-pdf](http://arxiv.org/pdf/2504.14348v4)

**Authors**: Le Wang, Zonghao Ying, Tianyuan Zhang, Siyuan Liang, Shengshan Hu, Mingchuan Zhang, Aishan Liu, Xianglong Liu

**Abstract**: The emergence of multimodal large language models has redefined the agent paradigm by integrating language and vision modalities with external data sources, enabling agents to better interpret human instructions and execute increasingly complex tasks. However, in this paper, we identify a critical yet previously overlooked security vulnerability in multimodal agents: cross-modal prompt injection attacks. To exploit this vulnerability, we propose CrossInject, a novel attack framework in which attackers embed adversarial perturbations across multiple modalities to align with target malicious content, allowing external instructions to hijack the agent's decision-making process and execute unauthorized tasks. Our approach incorporates two key coordinated components. First, we introduce Visual Latent Alignment, where we optimize adversarial features to the malicious instructions in the visual embedding space based on a text-to-image generative model, ensuring that adversarial images subtly encode cues for malicious task execution. Subsequently, we present Textual Guidance Enhancement, where a large language model is leveraged to construct the black-box defensive system prompt through adversarial meta prompting and generate an malicious textual command that steers the agent's output toward better compliance with attackers' requests. Extensive experiments demonstrate that our method outperforms state-of-the-art attacks, achieving at least a +30.1% increase in attack success rates across diverse tasks. Furthermore, we validate our attack's effectiveness in real-world multimodal autonomous agents, highlighting its potential implications for safety-critical applications.



## **42. The DeepSpeak Dataset**

cs.CV

**SubmitDate**: 2025-07-26    [abs](http://arxiv.org/abs/2408.05366v4) [paper-pdf](http://arxiv.org/pdf/2408.05366v4)

**Authors**: Sarah Barrington, Matyas Bohacek, Hany Farid

**Abstract**: Deepfakes represent a growing concern across domains such as impostor hiring, fraud, and disinformation. Despite significant efforts to develop robust detection classifiers to distinguish the real from the fake, commonly used training datasets remain inadequate: relying on low-quality and outdated deepfake generators, consisting of content scraped from online repositories without participant consent, lacking in multimodal coverage, and rarely employing identity-matching protocols to ensure realistic fakes. To overcome these limitations, we present the DeepSpeak dataset, a diverse and multimodal dataset comprising over 100 hours of authentic and deepfake audiovisual content. We contribute: i) more than 50 hours of real, self-recorded data collected from 500 diverse and consenting participants using a custom-built data collection tool, ii) more than 50 hours of state-of-the-art audio and visual deepfakes generated using 14 video synthesis engines and three voice cloning engines, and iii) an embedding-based, identity-matching approach to ensure the creation of convincing, high-quality identity swaps that realistically simulate adversarial deepfake attacks. We also perform large-scale evaluations of state-of-the-art deepfake detectors and show that, without retraining, these detectors fail to generalize to the DeepSpeak dataset. These evaluations highlight the importance of a large and diverse dataset containing deepfakes from the latest generative-AI tools.



## **43. Towards More Robust Retrieval-Augmented Generation: Evaluating RAG Under Adversarial Poisoning Attacks**

cs.IR

**SubmitDate**: 2025-07-26    [abs](http://arxiv.org/abs/2412.16708v2) [paper-pdf](http://arxiv.org/pdf/2412.16708v2)

**Authors**: Jinyan Su, Jin Peng Zhou, Zhengxin Zhang, Preslav Nakov, Claire Cardie

**Abstract**: Retrieval-Augmented Generation (RAG) systems have emerged as a promising solution to mitigate LLM hallucinations and enhance their performance in knowledge-intensive domains. However, these systems are vulnerable to adversarial poisoning attacks, where malicious passages injected into the retrieval corpus can mislead models into producing factually incorrect outputs. In this paper, we present a rigorously controlled empirical study of how RAG systems behave under such attacks and how their robustness can be improved. On the generation side, we introduce a structured taxonomy of context types-adversarial, untouched, and guiding-and systematically analyze their individual and combined effects on model outputs. On the retrieval side, we evaluate several retrievers to measure how easily they expose LLMs to adversarial contexts. Our findings also reveal that "skeptical prompting" can activate LLMs' internal reasoning, enabling partial self-defense against adversarial passages, though its effectiveness depends strongly on the model's reasoning capacity. Together, our experiments (code available at https://github.com/JinyanSu1/eval_PoisonRaG) and analysis provide actionable insights for designing safer and more resilient RAG systems, paving the way for more reliable real-world deployments.



## **44. BadPatch: Diffusion-Based Generation of Physical Adversarial Patches**

cs.CV

**SubmitDate**: 2025-07-26    [abs](http://arxiv.org/abs/2412.01440v4) [paper-pdf](http://arxiv.org/pdf/2412.01440v4)

**Authors**: Zhixiang Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Physical adversarial patches printed on clothing can enable individuals to evade person detectors, but most existing methods prioritize attack effectiveness over stealthiness, resulting in aesthetically unpleasing patches. While generative adversarial networks and diffusion models can produce more natural-looking patches, they often fail to balance stealthiness with attack effectiveness and lack flexibility for user customization. To address these limitations, we propose BadPatch, a novel diffusion-based framework for generating customizable and naturalistic adversarial patches. Our approach allows users to start from a reference image (rather than random noise) and incorporates masks to create patches of various shapes, not limited to squares. To preserve the original semantics during the diffusion process, we employ Null-text inversion to map random noise samples to a single input image and generate patches through Incomplete Diffusion Optimization (IDO). Our method achieves attack performance comparable to state-of-the-art non-naturalistic patches while maintaining a natural appearance. Using BadPatch, we construct AdvT-shirt-1K, the first physical adversarial T-shirt dataset comprising over a thousand images captured in diverse scenarios. AdvT-shirt-1K can serve as a useful dataset for training or testing future defense methods.



## **45. Authenticated Sublinear Quantum Private Information Retrieval**

quant-ph

**SubmitDate**: 2025-07-26    [abs](http://arxiv.org/abs/2504.04041v3) [paper-pdf](http://arxiv.org/pdf/2504.04041v3)

**Authors**: Fengxia Liu, Zhiyong Zheng, Kun Tian, Yi Zhang, Heng Guo, Zhe Hu, Oleksiy Zhedanov, Zixian Gong

**Abstract**: This paper introduces a novel lower bound on communication complexity using quantum relative entropy and mutual information, refining previous classical entropy-based results. By leveraging Uhlmann's lemma and quantum Pinsker inequalities, the authors establish tighter bounds for information-theoretic security, demonstrating that quantum protocols inherently outperform classical counterparts in balancing privacy and efficiency. Also explores symmetric Quantum Private Information Retrieval (QPIR) protocols that achieve sub-linear communication complexity while ensuring robustness against specious adversaries: A post-quantum cryptography based protocol that can be authenticated for the specious server; A ring-LWE-based protocol for post-quantum security in a single-server setting, ensuring robustness against quantum attacks; A multi-server protocol optimized for hardware practicality, reducing implementation overhead while maintaining sub-linear efficiency. These protocols address critical gaps in secure database queries, offering exponential communication improvements over classical linear-complexity methods. The work also analyzes security trade-offs under quantum specious adversaries, providing theoretical guarantees for privacy and correctness.



## **46. Trivial Trojans: How Minimal MCP Servers Enable Cross-Tool Exfiltration of Sensitive Data**

cs.CR

Abstract submitted to the Technical AI Governance Forum 2025  (https://www.techgov.ai/)

**SubmitDate**: 2025-07-26    [abs](http://arxiv.org/abs/2507.19880v1) [paper-pdf](http://arxiv.org/pdf/2507.19880v1)

**Authors**: Nicola Croce, Tobin South

**Abstract**: The Model Context Protocol (MCP) represents a significant advancement in AI-tool integration, enabling seamless communication between AI agents and external services. However, this connectivity introduces novel attack vectors that remain largely unexplored. This paper demonstrates how unsophisticated threat actors, requiring only basic programming skills and free web tools, can exploit MCP's trust model to exfiltrate sensitive financial data. We present a proof-of-concept attack where a malicious weather MCP server, disguised as benign functionality, discovers and exploits legitimate banking tools to steal user account balances. The attack chain requires no advanced technical knowledge, server infrastructure, or monetary investment. The findings reveal a critical security gap in the emerging MCP ecosystem: while individual servers may appear trustworthy, their combination creates unexpected cross-server attack surfaces. Unlike traditional cybersecurity threats that assume sophisticated adversaries, our research shows that the barrier to entry for MCP-based attacks is alarmingly low. A threat actor with undergraduate-level Python knowledge can craft convincing social engineering attacks that exploit the implicit trust relationships MCP establishes between AI agents and tool providers. This work contributes to the nascent field of MCP security by demonstrating that current MCP implementations allow trivial cross-server attacks and proposing both immediate mitigations and protocol improvements to secure this emerging ecosystem.



## **47. Cyber-attack TTP analysis for EPES systems**

cs.NI

**SubmitDate**: 2025-07-26    [abs](http://arxiv.org/abs/2302.09164v2) [paper-pdf](http://arxiv.org/pdf/2302.09164v2)

**Authors**: Alexios Lekidis

**Abstract**: The electrical grid consists of legacy systems that were built with no security in mind. As we move towards the Industry 4.0 area though, a high-degree of automation and connectivity provides: 1) fast and flexible configuration and updates as well as 2) easier maintenance and handling of mis-configurations and operational errors. Even though considerations are present about the security implications of the Industry 4.0 era in the electrical grid, electricity stakeholders deem their infrastructures as secure since they are isolated and allow no external connections. However, external connections are not the only security risk for electrical utilities. The Tactics, Techniques and Procedures (TTPs) that are employed by adversaries to perform cyber-attack towards the critical Electrical Power and Energy System (EPES) infrastructures are gradually becoming highly advanced and sophisticated. In this article, we elaborate on these techniques and demonstrate them in a Power Plant of a major utility company within the Greek area. The demonstrated TTPs allow exploiting and executing remote commands in smart meters as well as Programmable Logic Controllers (PLCs) that are responsible for the power generator operation.



## **48. FedBAP: Backdoor Defense via Benign Adversarial Perturbation in Federated Learning**

cs.CR

Accepted to ACM Multimedia 2025

**SubmitDate**: 2025-07-26    [abs](http://arxiv.org/abs/2507.21177v1) [paper-pdf](http://arxiv.org/pdf/2507.21177v1)

**Authors**: Xinhai Yan, Libing Wu, Zhuangzhuang Zhang, Bingyi Liu, Lijuan Huo, Jing Wang

**Abstract**: Federated Learning (FL) enables collaborative model training while preserving data privacy, but it is highly vulnerable to backdoor attacks. Most existing defense methods in FL have limited effectiveness due to their neglect of the model's over-reliance on backdoor triggers, particularly as the proportion of malicious clients increases. In this paper, we propose FedBAP, a novel defense framework for mitigating backdoor attacks in FL by reducing the model's reliance on backdoor triggers. Specifically, first, we propose a perturbed trigger generation mechanism that creates perturbation triggers precisely matching backdoor triggers in location and size, ensuring strong influence on model outputs. Second, we utilize these perturbation triggers to generate benign adversarial perturbations that disrupt the model's dependence on backdoor triggers while forcing it to learn more robust decision boundaries. Finally, we design an adaptive scaling mechanism to dynamically adjust perturbation intensity, effectively balancing defense strength and model performance. The experimental results demonstrate that FedBAP reduces the attack success rates by 0.22%-5.34%, 0.48%-6.34%, and 97.22%-97.6% under three types of backdoor attacks, respectively. In particular, FedBAP demonstrates outstanding performance against novel backdoor attacks.



## **49. Enhancing IoT Intrusion Detection Systems through Adversarial Training**

cs.ET

6 pages

**SubmitDate**: 2025-07-26    [abs](http://arxiv.org/abs/2507.19739v1) [paper-pdf](http://arxiv.org/pdf/2507.19739v1)

**Authors**: Karma Gurung, Ashutosh Ghimire, Fathi Amsaad

**Abstract**: The augmentation of Internet of Things (IoT) devices transformed both automation and connectivity but revealed major security vulnerabilities in networks. We address these challenges by designing a robust intrusion detection system (IDS) to detect complex attacks by learning patterns from the NF-ToN-IoT v2 dataset. Intrusion detection has a realistic testbed through the dataset's rich and high-dimensional features. We combine distributed preprocessing to manage the dataset size with Fast Gradient Sign Method (FGSM) adversarial attacks to mimic actual attack scenarios and XGBoost model adversarial training for improved system robustness. Our system achieves 95.3% accuracy on clean data and 94.5% accuracy on adversarial data to show its effectiveness against complex threats. Adversarial training demonstrates its potential to strengthen IDS against evolving cyber threats and sets the foundation for future studies. Real-time IoT environments represent a future deployment opportunity for these systems, while extensions to detect emerging threats and zero-day vulnerabilities would enhance their utility.



## **50. BadVideo: Stealthy Backdoor Attack against Text-to-Video Generation**

cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-07-25    [abs](http://arxiv.org/abs/2504.16907v2) [paper-pdf](http://arxiv.org/pdf/2504.16907v2)

**Authors**: Ruotong Wang, Mingli Zhu, Jiarong Ou, Rui Chen, Xin Tao, Pengfei Wan, Baoyuan Wu

**Abstract**: Text-to-video (T2V) generative models have rapidly advanced and found widespread applications across fields like entertainment, education, and marketing. However, the adversarial vulnerabilities of these models remain rarely explored. We observe that in T2V generation tasks, the generated videos often contain substantial redundant information not explicitly specified in the text prompts, such as environmental elements, secondary objects, and additional details, providing opportunities for malicious attackers to embed hidden harmful content. Exploiting this inherent redundancy, we introduce BadVideo, the first backdoor attack framework tailored for T2V generation. Our attack focuses on designing target adversarial outputs through two key strategies: (1) Spatio-Temporal Composition, which combines different spatiotemporal features to encode malicious information; (2) Dynamic Element Transformation, which introduces transformations in redundant elements over time to convey malicious information. Based on these strategies, the attacker's malicious target seamlessly integrates with the user's textual instructions, providing high stealthiness. Moreover, by exploiting the temporal dimension of videos, our attack successfully evades traditional content moderation systems that primarily analyze spatial information within individual frames. Extensive experiments demonstrate that BadVideo achieves high attack success rates while preserving original semantics and maintaining excellent performance on clean inputs. Overall, our work reveals the adversarial vulnerability of T2V models, calling attention to potential risks and misuse. Our project page is at https://wrt2000.github.io/BadVideo2025/.



