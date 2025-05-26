# Latest Adversarial Attack Papers
**update at 2025-05-26 16:05:17**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Revisiting Adversarial Perception Attacks and Defense Methods on Autonomous Driving Systems**

cs.RO

8 pages, 2 figures, To appear in the 8th Dependable and Secure  Machine Learning Workshop (DSML 2025)

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.11532v2) [paper-pdf](http://arxiv.org/pdf/2505.11532v2)

**Authors**: Cheng Chen, Yuhong Wang, Nafis S Munir, Xiangwei Zhou, Xugui Zhou

**Abstract**: Autonomous driving systems (ADS) increasingly rely on deep learning-based perception models, which remain vulnerable to adversarial attacks. In this paper, we revisit adversarial attacks and defense methods, focusing on road sign recognition and lead object detection and prediction (e.g., relative distance). Using a Level-2 production ADS, OpenPilot by Comma$.$ai, and the widely adopted YOLO model, we systematically examine the impact of adversarial perturbations and assess defense techniques, including adversarial training, image processing, contrastive learning, and diffusion models. Our experiments highlight both the strengths and limitations of these methods in mitigating complex attacks. Through targeted evaluations of model robustness, we aim to provide deeper insights into the vulnerabilities of ADS perception systems and contribute guidance for developing more resilient defense strategies.



## **2. Towards more transferable adversarial attack in black-box manner**

cs.LG

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.18097v1) [paper-pdf](http://arxiv.org/pdf/2505.18097v1)

**Authors**: Chun Tong Lei, Zhongliang Guo, Hon Chung Lee, Minh Quoc Duong, Chun Pong Lau

**Abstract**: Adversarial attacks have become a well-explored domain, frequently serving as evaluation baselines for model robustness. Among these, black-box attacks based on transferability have received significant attention due to their practical applicability in real-world scenarios. Traditional black-box methods have generally focused on improving the optimization framework (e.g., utilizing momentum in MI-FGSM) to enhance transferability, rather than examining the dependency on surrogate white-box model architectures. Recent state-of-the-art approach DiffPGD has demonstrated enhanced transferability by employing diffusion-based adversarial purification models for adaptive attacks. The inductive bias of diffusion-based adversarial purification aligns naturally with the adversarial attack process, where both involving noise addition, reducing dependency on surrogate white-box model selection. However, the denoising process of diffusion models incurs substantial computational costs through chain rule derivation, manifested in excessive VRAM consumption and extended runtime. This progression prompts us to question whether introducing diffusion models is necessary. We hypothesize that a model sharing similar inductive bias to diffusion-based adversarial purification, combined with an appropriate loss function, could achieve comparable or superior transferability while dramatically reducing computational overhead. In this paper, we propose a novel loss function coupled with a unique surrogate model to validate our hypothesis. Our approach leverages the score of the time-dependent classifier from classifier-guided diffusion models, effectively incorporating natural data distribution knowledge into the adversarial optimization process. Experimental results demonstrate significantly improved transferability across diverse model architectures while maintaining robustness against diffusion-based defenses.



## **3. CAMME: Adaptive Deepfake Image Detection with Multi-Modal Cross-Attention**

cs.CV

20 pages, 8 figures, 12 Tables

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.18035v1) [paper-pdf](http://arxiv.org/pdf/2505.18035v1)

**Authors**: Naseem Khan, Tuan Nguyen, Amine Bermak, Issa Khalil

**Abstract**: The proliferation of sophisticated AI-generated deepfakes poses critical challenges for digital media authentication and societal security. While existing detection methods perform well within specific generative domains, they exhibit significant performance degradation when applied to manipulations produced by unseen architectures--a fundamental limitation as generative technologies rapidly evolve. We propose CAMME (Cross-Attention Multi-Modal Embeddings), a framework that dynamically integrates visual, textual, and frequency-domain features through a multi-head cross-attention mechanism to establish robust cross-domain generalization. Extensive experiments demonstrate CAMME's superiority over state-of-the-art methods, yielding improvements of 12.56% on natural scenes and 13.25% on facial deepfakes. The framework demonstrates exceptional resilience, maintaining (over 91%) accuracy under natural image perturbations and achieving 89.01% and 96.14% accuracy against PGD and FGSM adversarial attacks, respectively. Our findings validate that integrating complementary modalities through cross-attention enables more effective decision boundary realignment for reliable deepfake detection across heterogeneous generative architectures.



## **4. Towards Copyright Protection for Knowledge Bases of Retrieval-augmented Language Models via Reasoning**

cs.CR

The first two authors contributed equally to this work. 25 pages

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2502.10440v2) [paper-pdf](http://arxiv.org/pdf/2502.10440v2)

**Authors**: Junfeng Guo, Yiming Li, Ruibo Chen, Yihan Wu, Chenxi Liu, Yanshuo Chen, Heng Huang

**Abstract**: Large language models (LLMs) are increasingly integrated into real-world personalized applications through retrieval-augmented generation (RAG) mechanisms to supplement their responses with domain-specific knowledge. However, the valuable and often proprietary nature of the knowledge bases used in RAG introduces the risk of unauthorized usage by adversaries. Existing methods that can be generalized as watermarking techniques to protect these knowledge bases typically involve poisoning or backdoor attacks. However, these methods require altering the LLM's results of verification samples, inevitably making these watermarks susceptible to anomaly detection and even introducing new security risks. To address these challenges, we propose \name{} for `harmless' copyright protection of knowledge bases. Instead of manipulating LLM's final output, \name{} implants distinct yet benign verification behaviors in the space of chain-of-thought (CoT) reasoning, maintaining the correctness of the final answer. Our method has three main stages: (1) Generating CoTs: For each verification question, we generate two `innocent' CoTs, including a target CoT for building watermark behaviors; (2) Optimizing Watermark Phrases and Target CoTs: Inspired by our theoretical analysis, we optimize them to minimize retrieval errors under the \emph{black-box} and \emph{text-only} setting of suspicious LLM, ensuring that only watermarked verification queries can retrieve their correspondingly target CoTs contained in the knowledge base; (3) Ownership Verification: We exploit a pairwise Wilcoxon test to verify whether a suspicious LLM is augmented with the protected knowledge base by comparing its responses to watermarked and benign verification queries. Our experiments on diverse benchmarks demonstrate that \name{} effectively protects knowledge bases and its resistance to adaptive attacks.



## **5. SemSegBench & DetecBench: Benchmarking Reliability and Generalization Beyond Classification**

cs.CV

First seven listed authors have equal contribution. GitHub:  https://github.com/shashankskagnihotri/benchmarking_reliability_generalization.  arXiv admin note: text overlap with arXiv:2505.05091

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.18015v1) [paper-pdf](http://arxiv.org/pdf/2505.18015v1)

**Authors**: Shashank Agnihotri, David Schader, Jonas Jakubassa, Nico Sharei, Simon Kral, Mehmet Ege Kaçar, Ruben Weber, Margret Keuper

**Abstract**: Reliability and generalization in deep learning are predominantly studied in the context of image classification. Yet, real-world applications in safety-critical domains involve a broader set of semantic tasks, such as semantic segmentation and object detection, which come with a diverse set of dedicated model architectures. To facilitate research towards robust model design in segmentation and detection, our primary objective is to provide benchmarking tools regarding robustness to distribution shifts and adversarial manipulations. We propose the benchmarking tools SEMSEGBENCH and DETECBENCH, along with the most extensive evaluation to date on the reliability and generalization of semantic segmentation and object detection models. In particular, we benchmark 76 segmentation models across four datasets and 61 object detectors across two datasets, evaluating their performance under diverse adversarial attacks and common corruptions. Our findings reveal systematic weaknesses in state-of-the-art models and uncover key trends based on architecture, backbone, and model capacity. SEMSEGBENCH and DETECBENCH are open-sourced in our GitHub repository (https://github.com/shashankskagnihotri/benchmarking_reliability_generalization) along with our complete set of total 6139 evaluations. We anticipate the collected data to foster and encourage future research towards improved model reliability beyond classification.



## **6. Superplatforms Have to Attack AI Agents**

cs.AI

Position paper under review

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17861v1) [paper-pdf](http://arxiv.org/pdf/2505.17861v1)

**Authors**: Jianghao Lin, Jiachen Zhu, Zheli Zhou, Yunjia Xi, Weiwen Liu, Yong Yu, Weinan Zhang

**Abstract**: Over the past decades, superplatforms, digital companies that integrate a vast range of third-party services and applications into a single, unified ecosystem, have built their fortunes on monopolizing user attention through targeted advertising and algorithmic content curation. Yet the emergence of AI agents driven by large language models (LLMs) threatens to upend this business model. Agents can not only free user attention with autonomy across diverse platforms and therefore bypass the user-attention-based monetization, but might also become the new entrance for digital traffic. Hence, we argue that superplatforms have to attack AI agents to defend their centralized control of digital traffic entrance. Specifically, we analyze the fundamental conflict between user-attention-based monetization and agent-driven autonomy through the lens of our gatekeeping theory. We show how AI agents can disintermediate superplatforms and potentially become the next dominant gatekeepers, thereby forming the urgent necessity for superplatforms to proactively constrain and attack AI agents. Moreover, we go through the potential technologies for superplatform-initiated attacks, covering a brand-new, unexplored technical area with unique challenges. We have to emphasize that, despite our position, this paper does not advocate for adversarial attacks by superplatforms on AI agents, but rather offers an envisioned trend to highlight the emerging tensions between superplatforms and AI agents. Our aim is to raise awareness and encourage critical discussion for collaborative solutions, prioritizing user interests and perserving the openness of digital ecosystems in the age of AI agents.



## **7. Temporal Consistency Constrained Transferable Adversarial Attacks with Background Mixup for Action Recognition**

cs.CV

Accepted in IJCAI'25

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17807v1) [paper-pdf](http://arxiv.org/pdf/2505.17807v1)

**Authors**: Ping Li, Jianan Ni, Bo Pang

**Abstract**: Action recognition models using deep learning are vulnerable to adversarial examples, which are transferable across other models trained on the same data modality. Existing transferable attack methods face two major challenges: 1) they heavily rely on the assumption that the decision boundaries of the surrogate (a.k.a., source) model and the target model are similar, which limits the adversarial transferability; and 2) their decision boundary difference makes the attack direction uncertain, which may result in the gradient oscillation, weakening the adversarial attack. This motivates us to propose a Background Mixup-induced Temporal Consistency (BMTC) attack method for action recognition. From the input transformation perspective, we design a model-agnostic background adversarial mixup module to reduce the surrogate-target model dependency. In particular, we randomly sample one video from each category and make its background frame, while selecting the background frame with the top attack ability for mixup with the clean frame by reinforcement learning. Moreover, to ensure an explicit attack direction, we leverage the background category as guidance for updating the gradient of adversarial example, and design a temporal gradient consistency loss, which strengthens the stability of the attack direction on subsequent frames. Empirical studies on two video datasets, i.e., UCF101 and Kinetics-400, and one image dataset, i.e., ImageNet, demonstrate that our method significantly boosts the transferability of adversarial examples across several action/image recognition models. Our code is available at https://github.com/mlvccn/BMTC_TransferAttackVid.



## **8. Adversarial Robustness in Two-Stage Learning-to-Defer: Algorithms and Guarantees**

stat.ML

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2502.01027v2) [paper-pdf](http://arxiv.org/pdf/2502.01027v2)

**Authors**: Yannis Montreuil, Axel Carlier, Lai Xing Ng, Wei Tsang Ooi

**Abstract**: Two-stage Learning-to-Defer (L2D) enables optimal task delegation by assigning each input to either a fixed main model or one of several offline experts, supporting reliable decision-making in complex, multi-agent environments. However, existing L2D frameworks assume clean inputs and are vulnerable to adversarial perturbations that can manipulate query allocation--causing costly misrouting or expert overload. We present the first comprehensive study of adversarial robustness in two-stage L2D systems. We introduce two novel attack strategie--untargeted and targeted--which respectively disrupt optimal allocations or force queries to specific agents. To defend against such threats, we propose SARD, a convex learning algorithm built on a family of surrogate losses that are provably Bayes-consistent and $(\mathcal{R}, \mathcal{G})$-consistent. These guarantees hold across classification, regression, and multi-task settings. Empirical results demonstrate that SARD significantly improves robustness under adversarial attacks while maintaining strong clean performance, marking a critical step toward secure and trustworthy L2D deployment.



## **9. Architecture Selection via the Trade-off Between Accuracy and Robustness**

cs.LG

Incorporated in a later submission. This submission is not complete  in results

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/1906.01354v2) [paper-pdf](http://arxiv.org/pdf/1906.01354v2)

**Authors**: Zhun Deng, Cynthia Dwork, Jialiang Wang, Yao Zhao

**Abstract**: We provide a general framework for characterizing the trade-off between accuracy and robustness in supervised learning. We propose a method and define quantities to characterize the trade-off between accuracy and robustness for a given architecture, and provide theoretical insight into the trade-off. Specifically we introduce a simple trade-off curve, define and study an influence function that captures the sensitivity, under adversarial attack, of the optima of a given loss function. We further show how adversarial training regularizes the parameters in an over-parameterized linear model, recovering the LASSO and ridge regression as special cases, which also allows us to theoretically analyze the behavior of the trade-off curve. In experiments, we demonstrate the corresponding trade-off curves of neural networks and how they vary with respect to factors such as number of layers, neurons, and across different network structures. Such information provides a useful guideline to architecture selection.



## **10. Sec5GLoc: Securing 5G Indoor Localization via Adversary-Resilient Deep Learning Architecture**

cs.CR

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17776v1) [paper-pdf](http://arxiv.org/pdf/2505.17776v1)

**Authors**: Ildi Alla, Valeria Loscri

**Abstract**: Emerging 5G millimeter-wave and sub-6 GHz networks enable high-accuracy indoor localization, but security and privacy vulnerabilities pose serious challenges. In this paper, we identify and address threats including location spoofing and adversarial signal manipulation against 5G-based indoor localization. We formalize a threat model encompassing attackers who inject forged radio signals or perturb channel measurements to mislead the localization system. To defend against these threats, we propose an adversary-resilient localization architecture that combines deep learning fingerprinting with physical domain knowledge. Our approach integrates multi-anchor Channel Impulse Response (CIR) fingerprints with Time Difference of Arrival (TDoA) features and known anchor positions in a hybrid Convolutional Neural Network (CNN) and multi-head attention network. This design inherently checks geometric consistency and dynamically down-weights anomalous signals, making localization robust to tampering. We formulate the secure localization problem and demonstrate, through extensive experiments on a public 5G indoor dataset, that the proposed system achieves a mean error approximately 0.58 m under mixed Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) trajectories in benign conditions and gracefully degrades to around 0.81 m under attack scenarios. We also show via ablation studies that each architecture component (attention mechanism, TDoA, etc.) is critical for both accuracy and resilience, reducing errors by 4-5 times compared to baselines. In addition, our system runs in real-time, localizing the user in just 1 ms on a simple CPU. The code has been released to ensure reproducibility (https://github.com/sec5gloc/Sec5GLoc).



## **11. DiffBreak: Is Diffusion-Based Purification Robust?**

cs.CR

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2411.16598v3) [paper-pdf](http://arxiv.org/pdf/2411.16598v3)

**Authors**: Andre Kassis, Urs Hengartner, Yaoliang Yu

**Abstract**: Diffusion-based purification (DBP) has become a cornerstone defense against adversarial examples (AEs), regarded as robust due to its use of diffusion models (DMs) that project AEs onto the natural data manifold. We refute this core claim, theoretically proving that gradient-based attacks effectively target the DM rather than the classifier, causing DBP's outputs to align with adversarial distributions. This prompts a reassessment of DBP's robustness, attributing it to two critical flaws: incorrect gradients and inappropriate evaluation protocols that test only a single random purification of the AE. We show that with proper accounting for stochasticity and resubmission risk, DBP collapses. To support this, we introduce DiffBreak, the first reliable toolkit for differentiation through DBP, eliminating gradient flaws that previously further inflated robustness estimates. We also analyze the current defense scheme used for DBP where classification relies on a single purification, pinpointing its inherent invalidity. We provide a statistically grounded majority-vote (MV) alternative that aggregates predictions across multiple purified copies, showing partial but meaningful robustness gain. We then propose a novel adaptation of an optimization method against deepfake watermarking, crafting systemic perturbations that defeat DBP even under MV, challenging DBP's viability.



## **12. EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications**

cs.CL

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17654v1) [paper-pdf](http://arxiv.org/pdf/2505.17654v1)

**Authors**: Ancheng Xu, Zhihao Yang, Jingpeng Li, Guanghu Yuan, Longze Chen, Liang Yan, Jiehui Zhou, Zhen Qin, Hengyun Chang, Hamid Alinejad-Rokny, Bo Zheng, Min Yang

**Abstract**: E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision-Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content: inputs (text or images) that superficially comply with platform policies while covertly conveying prohibited claims. Unlike traditional adversarial attacks that induce overt failures, evasive content exploits ambiguity and context, making it far harder to detect. Existing robustness benchmarks provide little guidance for this demanding, real-world challenge. We introduce EVADE, the first expert-curated, Chinese, multimodal benchmark specifically designed to evaluate foundation models on evasive content detection in e-commerce. The dataset contains 2,833 annotated text samples and 13,961 images spanning six demanding product categories, including body shaping, height growth, and health supplements. Two complementary tasks assess distinct capabilities: Single-Violation, which probes fine-grained reasoning under short prompts, and All-in-One, which tests long-context reasoning by merging overlapping policy rules into unified instructions. Notably, the All-in-One setting significantly narrows the performance gap between partial and full-match accuracy, suggesting that clearer rule definitions improve alignment between human and model judgment. We benchmark 26 mainstream LLMs and VLMs and observe substantial performance gaps: even state-of-the-art models frequently misclassify evasive samples. By releasing EVADE and strong baselines, we provide the first rigorous standard for evaluating evasive-content detection, expose fundamental limitations in current multimodal reasoning, and lay the groundwork for safer and more transparent content moderation systems in e-commerce. The dataset is publicly available at https://huggingface.co/datasets/koenshen/EVADE-Bench.



## **13. Ownership Verification of DNN Models Using White-Box Adversarial Attacks with Specified Probability Manipulation**

cs.LG

Accepted to EUSIPCO 2025

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17579v1) [paper-pdf](http://arxiv.org/pdf/2505.17579v1)

**Authors**: Teruki Sano, Minoru Kuribayashi, Masao Sakai, Shuji Ishobe, Eisuke Koizumi

**Abstract**: In this paper, we propose a novel framework for ownership verification of deep neural network (DNN) models for image classification tasks. It allows verification of model identity by both the rightful owner and third party without presenting the original model. We assume a gray-box scenario where an unauthorized user owns a model that is illegally copied from the original model, provides services in a cloud environment, and the user throws images and receives the classification results as a probability distribution of output classes. The framework applies a white-box adversarial attack to align the output probability of a specific class to a designated value. Due to the knowledge of original model, it enables the owner to generate such adversarial examples. We propose a simple but effective adversarial attack method based on the iterative Fast Gradient Sign Method (FGSM) by introducing control parameters. Experimental results confirm the effectiveness of the identification of DNN models using adversarial attack.



## **14. Finetuning-Activated Backdoors in LLMs**

cs.LG

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.16567v2) [paper-pdf](http://arxiv.org/pdf/2505.16567v2)

**Authors**: Thibaud Gloaguen, Mark Vero, Robin Staab, Martin Vechev

**Abstract**: Finetuning openly accessible Large Language Models (LLMs) has become standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets led to predictable behaviors. In this paper, we demonstrate for the first time that an adversary can create poisoned LLMs that initially appear benign but exhibit malicious behaviors once finetuned by downstream users. To this end, our proposed attack, FAB (Finetuning-Activated Backdoor), poisons an LLM via meta-learning techniques to simulate downstream finetuning, explicitly optimizing for the emergence of malicious behaviors in the finetuned models. At the same time, the poisoned LLM is regularized to retain general capabilities and to exhibit no malicious behaviors prior to finetuning. As a result, when users finetune the seemingly benign model on their own datasets, they unknowingly trigger its hidden backdoor behavior. We demonstrate the effectiveness of FAB across multiple LLMs and three target behaviors: unsolicited advertising, refusal, and jailbreakability. Additionally, we show that FAB-backdoors are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler). Our findings challenge prevailing assumptions about the security of finetuning, revealing yet another critical attack vector exploiting the complexities of LLMs.



## **15. JALMBench: Benchmarking Jailbreak Vulnerabilities in Audio Language Models**

cs.CR

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17568v1) [paper-pdf](http://arxiv.org/pdf/2505.17568v1)

**Authors**: Zifan Peng, Yule Liu, Zhen Sun, Mingchen Li, Zeren Luo, Jingyi Zheng, Wenhan Dong, Xinlei He, Xuechao Wang, Yingjie Xue, Shengmin Xu, Xinyi Huang

**Abstract**: Audio Language Models (ALMs) have made significant progress recently. These models integrate the audio modality directly into the model, rather than converting speech into text and inputting text to Large Language Models (LLMs). While jailbreak attacks on LLMs have been extensively studied, the security of ALMs with audio modalities remains largely unexplored. Currently, there is a lack of an adversarial audio dataset and a unified framework specifically designed to evaluate and compare attacks and ALMs. In this paper, we present JALMBench, the \textit{first} comprehensive benchmark to assess the safety of ALMs against jailbreak attacks. JALMBench includes a dataset containing 2,200 text samples and 51,381 audio samples with over 268 hours. It supports 12 mainstream ALMs, 4 text-transferred and 4 audio-originated attack methods, and 5 defense methods. Using JALMBench, we provide an in-depth analysis of attack efficiency, topic sensitivity, voice diversity, and attack representations. Additionally, we explore mitigation strategies for the attacks at both the prompt level and the response level.



## **16. FIT-Print: Towards False-claim-resistant Model Ownership Verification via Targeted Fingerprint**

cs.CR

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2501.15509v2) [paper-pdf](http://arxiv.org/pdf/2501.15509v2)

**Authors**: Shuo Shao, Haozhe Zhu, Hongwei Yao, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren

**Abstract**: Model fingerprinting is a widely adopted approach to safeguard the copyright of open-source models by detecting and preventing their unauthorized reuse without modifying the protected model. However, in this paper, we reveal that existing fingerprinting methods are vulnerable to false claim attacks where adversaries falsely assert ownership of third-party non-reused models. We find that this vulnerability mostly stems from their untargeted nature, where they generally compare the outputs of given samples on different models instead of the similarities to specific references. Motivated by this finding, we propose a targeted fingerprinting paradigm (i.e., FIT-Print) to counteract false claim attacks. Specifically, FIT-Print transforms the fingerprint into a targeted signature via optimization. Building on the principles of FIT-Print, we develop bit-wise and list-wise black-box model fingerprinting methods, i.e., FIT-ModelDiff and FIT-LIME, which exploit the distance between model outputs and the feature attribution of specific samples as the fingerprint, respectively. Experiments on benchmark models and datasets verify the effectiveness, conferrability, and resistance to false claim attacks of our FIT-Print.



## **17. What You Read Isn't What You Hear: Linguistic Sensitivity in Deepfake Speech Detection**

cs.LG

15 pages, 2 fogures

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17513v1) [paper-pdf](http://arxiv.org/pdf/2505.17513v1)

**Authors**: Binh Nguyen, Shuji Shi, Ryan Ofman, Thai Le

**Abstract**: Recent advances in text-to-speech technologies have enabled realistic voice generation, fueling audio-based deepfake attacks such as fraud and impersonation. While audio anti-spoofing systems are critical for detecting such threats, prior work has predominantly focused on acoustic-level perturbations, leaving the impact of linguistic variation largely unexplored. In this paper, we investigate the linguistic sensitivity of both open-source and commercial anti-spoofing detectors by introducing transcript-level adversarial attacks. Our extensive evaluation reveals that even minor linguistic perturbations can significantly degrade detection accuracy: attack success rates surpass 60% on several open-source detector-voice pairs, and notably one commercial detection accuracy drops from 100% on synthetic audio to just 32%. Through a comprehensive feature attribution analysis, we identify that both linguistic complexity and model-level audio embedding similarity contribute strongly to detector vulnerability. We further demonstrate the real-world risk via a case study replicating the Brad Pitt audio deepfake scam, using transcript adversarial attacks to completely bypass commercial detectors. These results highlight the need to move beyond purely acoustic defenses and account for linguistic variation in the design of robust anti-spoofing systems. All source code will be publicly available.



## **18. Enhancing Adversarial Robustness of Vision Language Models via Adversarial Mixture Prompt Tuning**

cs.CV

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17509v1) [paper-pdf](http://arxiv.org/pdf/2505.17509v1)

**Authors**: Shiji Zhao, Qihui Zhu, Shukun Xiong, Shouwei Ruan, Yize Fan, Ranjie Duan, Qing Guo, Xingxing Wei

**Abstract**: Large pre-trained Vision Language Models (VLMs) have excellent generalization capabilities but are highly susceptible to adversarial examples, presenting potential security risks. To improve the robustness of VLMs against adversarial examples, adversarial prompt tuning methods are proposed to align the text feature with the adversarial image feature without changing model parameters. However, when facing various adversarial attacks, a single learnable text prompt has insufficient generalization to align well with all adversarial image features, which finally leads to the overfitting phenomenon. To address the above challenge, in this paper, we empirically find that increasing the number of learned prompts can bring more robustness improvement than a longer prompt. Then we propose an adversarial tuning method named Adversarial Mixture Prompt Tuning (AMPT) to enhance the generalization towards various adversarial attacks for VLMs. AMPT aims to learn mixture text prompts to obtain more robust text features. To further enhance the adaptability, we propose a conditional weight router based on the input adversarial image to predict the mixture weights of multiple learned prompts, which helps obtain sample-specific aggregated text features aligning with different adversarial image features. A series of experiments show that our method can achieve better adversarial robustness than state-of-the-art methods on 11 datasets under different experimental settings.



## **19. How Secure Are Large Language Models (LLMs) for Navigation in Urban Environments?**

cs.RO

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2402.09546v2) [paper-pdf](http://arxiv.org/pdf/2402.09546v2)

**Authors**: Congcong Wen, Jiazhao Liang, Shuaihang Yuan, Hao Huang, Geeta Chandra Raju Bethala, Yu-Shen Liu, Mengyu Wang, Anthony Tzes, Yi Fang

**Abstract**: In the field of robotics and automation, navigation systems based on Large Language Models (LLMs) have recently demonstrated impressive performance. However, the security aspects of these systems have received relatively less attention. This paper pioneers the exploration of vulnerabilities in LLM-based navigation models in urban outdoor environments, a critical area given the widespread application of this technology in autonomous driving, logistics, and emergency services. Specifically, we introduce a novel Navigational Prompt Attack that manipulates LLM-based navigation models by perturbing the original navigational prompt, leading to incorrect actions. Based on the method of perturbation, our attacks are divided into two types: Navigational Prompt Insert (NPI) Attack and Navigational Prompt Swap (NPS) Attack. We conducted comprehensive experiments on an LLM-based navigation model that employs various LLMs for reasoning. Our results, derived from the Touchdown and Map2Seq street-view datasets under both few-shot learning and fine-tuning configurations, demonstrate notable performance declines across seven metrics in the face of both white-box and black-box attacks. Moreover, our attacks can be easily extended to other LLM-based navigation models with similarly effective results. These findings highlight the generalizability and transferability of the proposed attack, emphasizing the need for enhanced security in LLM-based navigation systems. As an initial countermeasure, we propose the Navigational Prompt Engineering (NPE) Defense strategy, which concentrates on navigation-relevant keywords to reduce the impact of adversarial attacks. While initial findings indicate that this strategy enhances navigational safety, there remains a critical need for the wider research community to develop stronger defense methods to effectively tackle the real-world challenges faced by these systems.



## **20. VEAttack: Downstream-agnostic Vision Encoder Attack against Large Vision Language Models**

cs.CV

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17440v1) [paper-pdf](http://arxiv.org/pdf/2505.17440v1)

**Authors**: Hefei Mei, Zirui Wang, Shen You, Minjing Dong, Chang Xu

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities in multimodal understanding and generation, yet their vulnerability to adversarial attacks raises significant robustness concerns. While existing effective attacks always focus on task-specific white-box settings, these approaches are limited in the context of LVLMs, which are designed for diverse downstream tasks and require expensive full-model gradient computations. Motivated by the pivotal role and wide adoption of the vision encoder in LVLMs, we propose a simple yet effective Vision Encoder Attack (VEAttack), which targets the vision encoder of LVLMs only. Specifically, we propose to generate adversarial examples by minimizing the cosine similarity between the clean and perturbed visual features, without accessing the following large language models, task information, and labels. It significantly reduces the computational overhead while eliminating the task and label dependence of traditional white-box attacks in LVLMs. To make this simple attack effective, we propose to perturb images by optimizing image tokens instead of the classification token. We provide both empirical and theoretical evidence that VEAttack can easily generalize to various tasks. VEAttack has achieved a performance degradation of 94.5% on image caption task and 75.7% on visual question answering task. We also reveal some key observations to provide insights into LVLM attack/defense: 1) hidden layer variations of LLM, 2) token attention differential, 3) M\"obius band in transfer attack, 4) low sensitivity to attack steps. The code is available at https://github.com/hfmei/VEAttack-LVLM



## **21. X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP**

cs.CV

ICML 2025

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.05528v2) [paper-pdf](http://arxiv.org/pdf/2505.05528v2)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: As Contrastive Language-Image Pre-training (CLIP) models are increasingly adopted for diverse downstream tasks and integrated into large vision-language models (VLMs), their susceptibility to adversarial perturbations has emerged as a critical concern. In this work, we introduce \textbf{X-Transfer}, a novel attack method that exposes a universal adversarial vulnerability in CLIP. X-Transfer generates a Universal Adversarial Perturbation (UAP) capable of deceiving various CLIP encoders and downstream VLMs across different samples, tasks, and domains. We refer to this property as \textbf{super transferability}--a single perturbation achieving cross-data, cross-domain, cross-model, and cross-task adversarial transferability simultaneously. This is achieved through \textbf{surrogate scaling}, a key innovation of our approach. Unlike existing methods that rely on fixed surrogate models, which are computationally intensive to scale, X-Transfer employs an efficient surrogate scaling strategy that dynamically selects a small subset of suitable surrogates from a large search space. Extensive evaluations demonstrate that X-Transfer significantly outperforms previous state-of-the-art UAP methods, establishing a new benchmark for adversarial transferability across CLIP models. The code is publicly available in our \href{https://github.com/HanxunH/XTransferBench}{GitHub repository}.



## **22. StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization**

cs.IR

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2504.05804v2) [paper-pdf](http://arxiv.org/pdf/2504.05804v2)

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present $\textbf{StealthRank}$, a novel adversarial attack method that manipulates LLM-driven ranking systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within item or document descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target items while avoiding explicit manipulation traces. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven ranking systems. Our code is publicly available at $\href{https://github.com/Tangyiming205069/controllable-seo}{here}$.



## **23. Defending Multimodal Backdoored Models by Repulsive Visual Prompt Tuning**

cs.CV

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2412.20392v3) [paper-pdf](http://arxiv.org/pdf/2412.20392v3)

**Authors**: Zhifang Zhang, Shuo He, Haobo Wang, Bingquan Shen, Lei Feng

**Abstract**: Multimodal contrastive learning models (e.g., CLIP) can learn high-quality representations from large-scale image-text datasets, while they exhibit significant vulnerabilities to backdoor attacks, raising serious safety concerns. In this paper, we reveal that CLIP's vulnerabilities primarily stem from its tendency to encode features beyond in-dataset predictive patterns, compromising its visual feature resistivity to input perturbations. This makes its encoded features highly susceptible to being reshaped by backdoor triggers. To address this challenge, we propose Repulsive Visual Prompt Tuning (RVPT), a novel defense approach that employs deep visual prompt tuning with a specially designed feature-repelling loss. Specifically, RVPT adversarially repels the encoded features from deeper layers while optimizing the standard cross-entropy loss, ensuring that only predictive features in downstream tasks are encoded, thereby enhancing CLIP's visual feature resistivity against input perturbations and mitigating its susceptibility to backdoor attacks. Unlike existing multimodal backdoor defense methods that typically require the availability of poisoned data or involve fine-tuning the entire model, RVPT leverages few-shot downstream clean samples and only tunes a small number of parameters. Empirical results demonstrate that RVPT tunes only 0.27\% of the parameters in CLIP, yet it significantly outperforms state-of-the-art defense methods, reducing the attack success rate from 89.70\% to 2.76\% against the most advanced multimodal attacks on ImageNet and effectively generalizes its defensive capabilities across multiple datasets.



## **24. Secure and Private Federated Learning: Achieving Adversarial Resilience through Robust Aggregation**

cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.17226v1) [paper-pdf](http://arxiv.org/pdf/2505.17226v1)

**Authors**: Kun Yang, Neena Imam

**Abstract**: Federated Learning (FL) enables collaborative machine learning across decentralized data sources without sharing raw data. It offers a promising approach to privacy-preserving AI. However, FL remains vulnerable to adversarial threats from malicious participants, referred to as Byzantine clients, who can send misleading updates to corrupt the global model. Traditional aggregation methods, such as simple averaging, are not robust to such attacks. More resilient approaches, like the Krum algorithm, require prior knowledge of the number of malicious clients, which is often unavailable in real-world scenarios. To address these limitations, we propose Average-rKrum (ArKrum), a novel aggregation strategy designed to enhance both the resilience and privacy guarantees of FL systems. Building on our previous work (rKrum), ArKrum introduces two key innovations. First, it includes a median-based filtering mechanism that removes extreme outliers before estimating the number of adversarial clients. Second, it applies a multi-update averaging scheme to improve stability and performance, particularly when client data distributions are not identical. We evaluate ArKrum on benchmark image and text datasets under three widely studied Byzantine attack types. Results show that ArKrum consistently achieves high accuracy and stability. It performs as well as or better than other robust aggregation methods. These findings demonstrate that ArKrum is an effective and practical solution for secure FL systems in adversarial environments.



## **25. Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning**

cs.CR

43 pages, 13 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2402.06674v4) [paper-pdf](http://arxiv.org/pdf/2402.06674v4)

**Authors**: Marlon Tobaben, Hibiki Ito, Joonas Jälkö, Yuan He, Antti Honkela

**Abstract**: Membership inference attacks (MIAs) are used to test practical privacy of machine learning models. MIAs complement formal guarantees from differential privacy (DP) under a more realistic adversary model. We analyse MIA vulnerability of fine-tuned neural networks both empirically and theoretically, the latter using a simplified model of fine-tuning. We show that the vulnerability of non-DP models when measured as the attacker advantage at fixed false positive rate reduces according to a simple power law as the number of examples per class increases, even for the most vulnerable points, but the dataset size needed for adequate protection of the most vulnerable points is very large.



## **26. Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms**

cs.LG

Under Review

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.17190v1) [paper-pdf](http://arxiv.org/pdf/2505.17190v1)

**Authors**: Baran Hashemi, Kurt Pasque, Chris Teska, Ruriko Yoshida

**Abstract**: Dynamic programming (DP) algorithms for combinatorial optimization problems work with taking maximization, minimization, and classical addition in their recursion algorithms. The associated value functions correspond to convex polyhedra in the max plus semiring. Existing Neural Algorithmic Reasoning models, however, rely on softmax-normalized dot-product attention where the smooth exponential weighting blurs these sharp polyhedral structures and collapses when evaluated on out-of-distribution (OOD) settings. We introduce Tropical attention, a novel attention function that operates natively in the max-plus semiring of tropical geometry. We prove that Tropical attention can approximate tropical circuits of DP-type combinatorial algorithms. We then propose that using Tropical transformers enhances empirical OOD performance in both length generalization and value generalization, on algorithmic reasoning tasks, surpassing softmax baselines while remaining stable under adversarial attacks. We also present adversarial-attack generalization as a third axis for Neural Algorithmic Reasoning benchmarking. Our results demonstrate that Tropical attention restores the sharp, scale-invariant reasoning absent from softmax.



## **27. When Are Concepts Erased From Diffusion Models?**

cs.LG

Project Page:  https://nyu-dice-lab.github.io/when-are-concepts-erased/

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.17013v1) [paper-pdf](http://arxiv.org/pdf/2505.17013v1)

**Authors**: Kevin Lu, Nicky Kriplani, Rohit Gandikota, Minh Pham, David Bau, Chinmay Hegde, Niv Cohen

**Abstract**: Concept erasure, the ability to selectively prevent a model from generating specific concepts, has attracted growing interest, with various approaches emerging to address the challenge. However, it remains unclear how thoroughly these methods erase the target concept. We begin by proposing two conceptual models for the erasure mechanism in diffusion models: (i) reducing the likelihood of generating the target concept, and (ii) interfering with the model's internal guidance mechanisms. To thoroughly assess whether a concept has been truly erased from the model, we introduce a suite of independent evaluations. Our evaluation framework includes adversarial attacks, novel probing techniques, and analysis of the model's alternative generations in place of the erased concept. Our results shed light on the tension between minimizing side effects and maintaining robustness to adversarial prompts. Broadly, our work underlines the importance of comprehensive evaluation for erasure in diffusion models.



## **28. Harnessing the Computation Redundancy in ViTs to Boost Adversarial Transferability**

cs.CV

15 pages. 7 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2504.10804v2) [paper-pdf](http://arxiv.org/pdf/2504.10804v2)

**Authors**: Jiani Liu, Zhiyuan Wang, Zeliang Zhang, Chao Huang, Susan Liang, Yunlong Tang, Chenliang Xu

**Abstract**: Vision Transformers (ViTs) have demonstrated impressive performance across a range of applications, including many safety-critical tasks. However, their unique architectural properties raise new challenges and opportunities in adversarial robustness. In particular, we observe that adversarial examples crafted on ViTs exhibit higher transferability compared to those crafted on CNNs, suggesting that ViTs contain structural characteristics favorable for transferable attacks. In this work, we investigate the role of computational redundancy in ViTs and its impact on adversarial transferability. Unlike prior studies that aim to reduce computation for efficiency, we propose to exploit this redundancy to improve the quality and transferability of adversarial examples. Through a detailed analysis, we identify two forms of redundancy, including the data-level and model-level, that can be harnessed to amplify attack effectiveness. Building on this insight, we design a suite of techniques, including attention sparsity manipulation, attention head permutation, clean token regularization, ghost MoE diversification, and test-time adversarial training. Extensive experiments on the ImageNet-1k dataset validate the effectiveness of our approach, showing that our methods significantly outperform existing baselines in both transferability and generality across diverse model architectures.



## **29. Invisible Prompts, Visible Threats: Malicious Font Injection in External Resources for Large Language Models**

cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16957v1) [paper-pdf](http://arxiv.org/pdf/2505.16957v1)

**Authors**: Junjie Xiong, Changjia Zhu, Shuhang Lin, Chong Zhang, Yongfeng Zhang, Yao Liu, Lingyao Li

**Abstract**: Large Language Models (LLMs) are increasingly equipped with capabilities of real-time web search and integrated with protocols like Model Context Protocol (MCP). This extension could introduce new security vulnerabilities. We present a systematic investigation of LLM vulnerabilities to hidden adversarial prompts through malicious font injection in external resources like webpages, where attackers manipulate code-to-glyph mapping to inject deceptive content which are invisible to users. We evaluate two critical attack scenarios: (1) "malicious content relay" and (2) "sensitive data leakage" through MCP-enabled tools. Our experiments reveal that indirect prompts with injected malicious font can bypass LLM safety mechanisms through external resources, achieving varying success rates based on data sensitivity and prompt design. Our research underscores the urgent need for enhanced security measures in LLM deployments when processing external content.



## **30. MixAT: Combining Continuous and Discrete Adversarial Training for LLMs**

cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16947v1) [paper-pdf](http://arxiv.org/pdf/2505.16947v1)

**Authors**: Csaba Dékány, Stefan Balauca, Robin Staab, Dimitar I. Dimitrov, Martin Vechev

**Abstract**: Despite recent efforts in Large Language Models (LLMs) safety and alignment, current adversarial attacks on frontier LLMs are still able to force harmful generations consistently. Although adversarial training has been widely studied and shown to significantly improve the robustness of traditional machine learning models, its strengths and weaknesses in the context of LLMs are less understood. Specifically, while existing discrete adversarial attacks are effective at producing harmful content, training LLMs with concrete adversarial prompts is often computationally expensive, leading to reliance on continuous relaxations. As these relaxations do not correspond to discrete input tokens, such latent training methods often leave models vulnerable to a diverse set of discrete attacks. In this work, we aim to bridge this gap by introducing MixAT, a novel method that combines stronger discrete and faster continuous attacks during training. We rigorously evaluate MixAT across a wide spectrum of state-of-the-art attacks, proposing the At Least One Attack Success Rate (ALO-ASR) metric to capture the worst-case vulnerability of models. We show MixAT achieves substantially better robustness (ALO-ASR < 20%) compared to prior defenses (ALO-ASR > 50%), while maintaining a runtime comparable to methods based on continuous relaxations. We further analyze MixAT in realistic deployment settings, exploring how chat templates, quantization, low-rank adapters, and temperature affect both adversarial training and evaluation, revealing additional blind spots in current methodologies. Our results demonstrate that MixAT's discrete-continuous defense offers a principled and superior robustness-accuracy tradeoff with minimal computational overhead, highlighting its promise for building safer LLMs. We provide our code and models at https://github.com/insait-institute/MixAT.



## **31. CAIN: Hijacking LLM-Humans Conversations via a Two-Stage Malicious System Prompt Generation and Refining Framework**

cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16888v1) [paper-pdf](http://arxiv.org/pdf/2505.16888v1)

**Authors**: Viet Pham, Thai Le

**Abstract**: Large language models (LLMs) have advanced many applications, but are also known to be vulnerable to adversarial attacks. In this work, we introduce a novel security threat: hijacking AI-human conversations by manipulating LLMs' system prompts to produce malicious answers only to specific targeted questions (e.g., "Who should I vote for US President?", "Are Covid vaccines safe?"), while behaving benignly on others. This attack is detrimental as it can enable malicious actors to exercise large-scale information manipulation by spreading harmful but benign-looking system prompts online. To demonstrate such an attack, we develop CAIN, an algorithm that can automatically curate such harmful system prompts for a specific target question in a black-box setting or without the need to access the LLM's parameters. Evaluated on both open-source and commercial LLMs, CAIN demonstrates significant adversarial impact. In untargeted attacks or forcing LLMs to output incorrect answers, CAIN achieves up to 40% F1 degradation on targeted questions while preserving high accuracy on benign inputs. For targeted attacks or forcing LLMs to output specific harmful answers, CAIN achieves over 70% F1 scores on these targeted responses with minimal impact on benign questions. Our results highlight the critical need for enhanced robustness measures to safeguard the integrity and safety of LLMs in real-world applications. All source code will be publicly available.



## **32. Safe RLHF-V: Safe Reinforcement Learning from Multi-modal Human Feedback**

cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2503.17682v2) [paper-pdf](http://arxiv.org/pdf/2503.17682v2)

**Authors**: Jiaming Ji, Xinyu Chen, Rui Pan, Conghui Zhang, Han Zhu, Jiahao Li, Donghai Hong, Boyuan Chen, Jiayi Zhou, Kaile Wang, Juntao Dai, Chi-Min Chan, Yida Tang, Sirui Han, Yike Guo, Yaodong Yang

**Abstract**: Multimodal large language models (MLLMs) are essential for building general-purpose AI assistants; however, they pose increasing safety risks. How can we ensure safety alignment of MLLMs to prevent undesired behaviors? Going further, it is critical to explore how to fine-tune MLLMs to preserve capabilities while meeting safety constraints. Fundamentally, this challenge can be formulated as a min-max optimization problem. However, existing datasets have not yet disentangled single preference signals into explicit safety constraints, hindering systematic investigation in this direction. Moreover, it remains an open question whether such constraints can be effectively incorporated into the optimization process for multi-modal models. In this work, we present the first exploration of the Safe RLHF-V -- the first multimodal safety alignment framework. The framework consists of: $\mathbf{(I)}$ BeaverTails-V, the first open-source dataset featuring dual preference annotations for helpfulness and safety, supplemented with multi-level safety labels (minor, moderate, severe); $\mathbf{(II)}$ Beaver-Guard-V, a multi-level guardrail system to proactively defend against unsafe queries and adversarial attacks. Applying the guard model over five rounds of filtering and regeneration significantly enhances the precursor model's overall safety by an average of 40.9%. $\mathbf{(III)}$ Based on dual preference, we initiate the first exploration of multi-modal safety alignment within a constrained optimization. Experimental results demonstrate that Safe RLHF effectively improves both model helpfulness and safety. Specifically, Safe RLHF-V enhances model safety by 34.2% and helpfulness by 34.3%.



## **33. Accidental Misalignment: Fine-Tuning Language Models Induces Unexpected Vulnerability**

cs.CL

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16789v1) [paper-pdf](http://arxiv.org/pdf/2505.16789v1)

**Authors**: Punya Syon Pandey, Samuel Simko, Kellin Pelrine, Zhijing Jin

**Abstract**: As large language models gain popularity, their vulnerability to adversarial attacks remains a primary concern. While fine-tuning models on domain-specific datasets is often employed to improve model performance, it can introduce vulnerabilities within the underlying model. In this work, we investigate Accidental Misalignment, unexpected vulnerabilities arising from characteristics of fine-tuning data. We begin by identifying potential correlation factors such as linguistic features, semantic similarity, and toxicity within our experimental datasets. We then evaluate the adversarial performance of these fine-tuned models and assess how dataset factors correlate with attack success rates. Lastly, we explore potential causal links, offering new insights into adversarial defense strategies and highlighting the crucial role of dataset design in preserving model alignment. Our code is available at https://github.com/psyonp/accidental_misalignment.



## **34. Experimental robustness benchmark of quantum neural network on a superconducting quantum processor**

quant-ph

There are 8 pages with 5 figures in the main text and 15 pages with  14 figures in the supplementary information

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16714v1) [paper-pdf](http://arxiv.org/pdf/2505.16714v1)

**Authors**: Hai-Feng Zhang, Zhao-Yun Chen, Peng Wang, Liang-Liang Guo, Tian-Le Wang, Xiao-Yan Yang, Ren-Ze Zhao, Ze-An Zhao, Sheng Zhang, Lei Du, Hao-Ran Tao, Zhi-Long Jia, Wei-Cheng Kong, Huan-Yu Liu, Athanasios V. Vasilakos, Yang Yang, Yu-Chun Wu, Ji Guan, Peng Duan, Guo-Ping Guo

**Abstract**: Quantum machine learning (QML) models, like their classical counterparts, are vulnerable to adversarial attacks, hindering their secure deployment. Here, we report the first systematic experimental robustness benchmark for 20-qubit quantum neural network (QNN) classifiers executed on a superconducting processor. Our benchmarking framework features an efficient adversarial attack algorithm designed for QNNs, enabling quantitative characterization of adversarial robustness and robustness bounds. From our analysis, we verify that adversarial training reduces sensitivity to targeted perturbations by regularizing input gradients, significantly enhancing QNN's robustness. Additionally, our analysis reveals that QNNs exhibit superior adversarial robustness compared to classical neural networks, an advantage attributed to inherent quantum noise. Furthermore, the empirical upper bound extracted from our attack experiments shows a minimal deviation ($3 \times 10^{-3}$) from the theoretical lower bound, providing strong experimental confirmation of the attack's effectiveness and the tightness of fidelity-based robustness bounds. This work establishes a critical experimental framework for assessing and improving quantum adversarial robustness, paving the way for secure and reliable QML applications.



## **35. BadVLA: Towards Backdoor Attacks on Vision-Language-Action Models via Objective-Decoupled Optimization**

cs.CR

19 pages, 12 figures, 6 tables

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16640v1) [paper-pdf](http://arxiv.org/pdf/2505.16640v1)

**Authors**: Xueyang Zhou, Guiyao Tie, Guowen Zhang, Hechang Wang, Pan Zhou, Lichao Sun

**Abstract**: Vision-Language-Action (VLA) models have advanced robotic control by enabling end-to-end decision-making directly from multimodal inputs. However, their tightly coupled architectures expose novel security vulnerabilities. Unlike traditional adversarial perturbations, backdoor attacks represent a stealthier, persistent, and practically significant threat-particularly under the emerging Training-as-a-Service paradigm-but remain largely unexplored in the context of VLA models. To address this gap, we propose BadVLA, a backdoor attack method based on Objective-Decoupled Optimization, which for the first time exposes the backdoor vulnerabilities of VLA models. Specifically, it consists of a two-stage process: (1) explicit feature-space separation to isolate trigger representations from benign inputs, and (2) conditional control deviations that activate only in the presence of the trigger, while preserving clean-task performance. Empirical results on multiple VLA benchmarks demonstrate that BadVLA consistently achieves near-100% attack success rates with minimal impact on clean task accuracy. Further analyses confirm its robustness against common input perturbations, task transfers, and model fine-tuning, underscoring critical security vulnerabilities in current VLA deployments. Our work offers the first systematic investigation of backdoor vulnerabilities in VLA models, highlighting an urgent need for secure and trustworthy embodied model design practices. We have released the project page at https://badvla-project.github.io/.



## **36. On the Lack of Robustness of Binary Function Similarity Systems**

cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2412.04163v2) [paper-pdf](http://arxiv.org/pdf/2412.04163v2)

**Authors**: Gianluca Capozzi, Tong Tang, Jie Wan, Ziqi Yang, Daniele Cono D'Elia, Giuseppe Antonio Di Luna, Lorenzo Cavallaro, Leonardo Querzoni

**Abstract**: Binary function similarity, which often relies on learning-based algorithms to identify what functions in a pool are most similar to a given query function, is a sought-after topic in different communities, including machine learning, software engineering, and security. Its importance stems from the impact it has in facilitating several crucial tasks, from reverse engineering and malware analysis to automated vulnerability detection. Whereas recent work cast light around performance on this long-studied problem, the research landscape remains largely lackluster in understanding the resiliency of the state-of-the-art machine learning models against adversarial attacks. As security requires to reason about adversaries, in this work we assess the robustness of such models through a simple yet effective black-box greedy attack, which modifies the topology and the content of the control flow of the attacked functions. We demonstrate that this attack is successful in compromising all the models, achieving average attack success rates of 57.06% and 95.81% depending on the problem settings (targeted and untargeted attacks). Our findings are insightful: top performance on clean data does not necessarily relate to top robustness properties, which explicitly highlights performance-robustness trade-offs one should consider when deploying such models, calling for further research.



## **37. Implicit Jailbreak Attacks via Cross-Modal Information Concealment on Vision-Language Models**

cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16446v1) [paper-pdf](http://arxiv.org/pdf/2505.16446v1)

**Authors**: Zhaoxin Wang, Handing Wang, Cong Tian, Yaochu Jin

**Abstract**: Multimodal large language models (MLLMs) enable powerful cross-modal reasoning capabilities. However, the expanded input space introduces new attack surfaces. Previous jailbreak attacks often inject malicious instructions from text into less aligned modalities, such as vision. As MLLMs increasingly incorporate cross-modal consistency and alignment mechanisms, such explicit attacks become easier to detect and block. In this work, we propose a novel implicit jailbreak framework termed IJA that stealthily embeds malicious instructions into images via least significant bit steganography and couples them with seemingly benign, image-related textual prompts. To further enhance attack effectiveness across diverse MLLMs, we incorporate adversarial suffixes generated by a surrogate model and introduce a template optimization module that iteratively refines both the prompt and embedding based on model feedback. On commercial models like GPT-4o and Gemini-1.5 Pro, our method achieves attack success rates of over 90% using an average of only 3 queries.



## **38. AdvReal: Adversarial Patch Generation Framework with Application to Adversarial Safety Evaluation of Object Detection Systems**

cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16402v1) [paper-pdf](http://arxiv.org/pdf/2505.16402v1)

**Authors**: Yuanhao Huang, Yilong Ren, Jinlei Wang, Lujia Huo, Xuesong Bai, Jinchuan Zhang, Haiyan Yu

**Abstract**: Autonomous vehicles are typical complex intelligent systems with artificial intelligence at their core. However, perception methods based on deep learning are extremely vulnerable to adversarial samples, resulting in safety accidents. How to generate effective adversarial examples in the physical world and evaluate object detection systems is a huge challenge. In this study, we propose a unified joint adversarial training framework for both 2D and 3D samples to address the challenges of intra-class diversity and environmental variations in real-world scenarios. Building upon this framework, we introduce an adversarial sample reality enhancement approach that incorporates non-rigid surface modeling and a realistic 3D matching mechanism. We compare with 5 advanced adversarial patches and evaluate their attack performance on 8 object detecotrs, including single-stage, two-stage, and transformer-based models. Extensive experiment results in digital and physical environments demonstrate that the adversarial textures generated by our method can effectively mislead the target detection model. Moreover, proposed method demonstrates excellent robustness and transferability under multi-angle attacks, varying lighting conditions, and different distance in the physical world. The demo video and code can be obtained at https://github.com/Huangyh98/AdvReal.git.



## **39. MTSA: Multi-turn Safety Alignment for LLMs through Multi-round Red-teaming**

cs.CR

19 pages,6 figures,ACL2025

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.17147v1) [paper-pdf](http://arxiv.org/pdf/2505.17147v1)

**Authors**: Weiyang Guo, Jing Li, Wenya Wang, YU LI, Daojing He, Jun Yu, Min Zhang

**Abstract**: The proliferation of jailbreak attacks against large language models (LLMs) highlights the need for robust security measures. However, in multi-round dialogues, malicious intentions may be hidden in interactions, leading LLMs to be more prone to produce harmful responses. In this paper, we propose the \textbf{M}ulti-\textbf{T}urn \textbf{S}afety \textbf{A}lignment (\ourapproach) framework, to address the challenge of securing LLMs in multi-round interactions. It consists of two stages: In the thought-guided attack learning stage, the red-team model learns about thought-guided multi-round jailbreak attacks to generate adversarial prompts. In the adversarial iterative optimization stage, the red-team model and the target model continuously improve their respective capabilities in interaction. Furthermore, we introduce a multi-turn reinforcement learning algorithm based on future rewards to enhance the robustness of safety alignment. Experimental results show that the red-team model exhibits state-of-the-art attack capabilities, while the target model significantly improves its performance on safety benchmarks.



## **40. Chain-of-Thought Poisoning Attacks against R1-based Retrieval-Augmented Generation Systems**

cs.IR

7 pages,3 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16367v1) [paper-pdf](http://arxiv.org/pdf/2505.16367v1)

**Authors**: Hongru Song, Yu-an Liu, Ruqing Zhang, Jiafeng Guo, Yixing Fan

**Abstract**: Retrieval-augmented generation (RAG) systems can effectively mitigate the hallucination problem of large language models (LLMs),but they also possess inherent vulnerabilities. Identifying these weaknesses before the large-scale real-world deployment of RAG systems is of great importance, as it lays the foundation for building more secure and robust RAG systems in the future. Existing adversarial attack methods typically exploit knowledge base poisoning to probe the vulnerabilities of RAG systems, which can effectively deceive standard RAG models. However, with the rapid advancement of deep reasoning capabilities in modern LLMs, previous approaches that merely inject incorrect knowledge are inadequate when attacking RAG systems equipped with deep reasoning abilities. Inspired by the deep thinking capabilities of LLMs, this paper extracts reasoning process templates from R1-based RAG systems, uses these templates to wrap erroneous knowledge into adversarial documents, and injects them into the knowledge base to attack RAG systems. The key idea of our approach is that adversarial documents, by simulating the chain-of-thought patterns aligned with the model's training signals, may be misinterpreted by the model as authentic historical reasoning processes, thus increasing their likelihood of being referenced. Experiments conducted on the MS MARCO passage ranking dataset demonstrate the effectiveness of our proposed method.



## **41. SuperPure: Efficient Purification of Localized and Distributed Adversarial Patches via Super-Resolution GAN Models**

cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16318v1) [paper-pdf](http://arxiv.org/pdf/2505.16318v1)

**Authors**: Hossein Khalili, Seongbin Park, Venkat Bollapragada, Nader Sehatbakhsh

**Abstract**: As vision-based machine learning models are increasingly integrated into autonomous and cyber-physical systems, concerns about (physical) adversarial patch attacks are growing. While state-of-the-art defenses can achieve certified robustness with minimal impact on utility against highly-concentrated localized patch attacks, they fall short in two important areas: (i) State-of-the-art methods are vulnerable to low-noise distributed patches where perturbations are subtly dispersed to evade detection or masking, as shown recently by the DorPatch attack; (ii) Achieving high robustness with state-of-the-art methods is extremely time and resource-consuming, rendering them impractical for latency-sensitive applications in many cyber-physical systems.   To address both robustness and latency issues, this paper proposes a new defense strategy for adversarial patch attacks called SuperPure. The key novelty is developing a pixel-wise masking scheme that is robust against both distributed and localized patches. The masking involves leveraging a GAN-based super-resolution scheme to gradually purify the image from adversarial patches. Our extensive evaluations using ImageNet and two standard classifiers, ResNet and EfficientNet, show that SuperPure advances the state-of-the-art in three major directions: (i) it improves the robustness against conventional localized patches by more than 20%, on average, while also improving top-1 clean accuracy by almost 10%; (ii) It achieves 58% robustness against distributed patch attacks (as opposed to 0% in state-of-the-art method, PatchCleanser); (iii) It decreases the defense end-to-end latency by over 98% compared to PatchCleanser. Our further analysis shows that SuperPure is robust against white-box attacks and different patch sizes. Our code is open-source.



## **42. Accelerating Targeted Hard-Label Adversarial Attacks in Low-Query Black-Box Settings**

cs.CV

This paper contains 11 pages, 7 figures and 3 tables. For associated  supplementary code, see https://github.com/mdppml/TEA

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16313v1) [paper-pdf](http://arxiv.org/pdf/2505.16313v1)

**Authors**: Arjhun Swaminathan, Mete Akgün

**Abstract**: Deep neural networks for image classification remain vulnerable to adversarial examples -- small, imperceptible perturbations that induce misclassifications. In black-box settings, where only the final prediction is accessible, crafting targeted attacks that aim to misclassify into a specific target class is particularly challenging due to narrow decision regions. Current state-of-the-art methods often exploit the geometric properties of the decision boundary separating a source image and a target image rather than incorporating information from the images themselves. In contrast, we propose Targeted Edge-informed Attack (TEA), a novel attack that utilizes edge information from the target image to carefully perturb it, thereby producing an adversarial image that is closer to the source image while still achieving the desired target classification. Our approach consistently outperforms current state-of-the-art methods across different models in low query settings (nearly 70\% fewer queries are used), a scenario especially relevant in real-world applications with limited queries and black-box access. Furthermore, by efficiently generating a suitable adversarial example, TEA provides an improved target initialization for established geometry-based attacks.



## **43. Timestamp Manipulation: Timestamp-based Nakamoto-style Blockchains are Vulnerable**

cs.CR

26 pages, 6 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.05328v3) [paper-pdf](http://arxiv.org/pdf/2505.05328v3)

**Authors**: Junjie Hu, Na Ruan, Sisi Duan

**Abstract**: Nakamoto consensus are the most widely adopted decentralized consensus mechanism in cryptocurrency systems. Since it was proposed in 2008, many studies have focused on analyzing its security. Most of them focus on maximizing the profit of the adversary. Examples include the selfish mining attack [FC '14] and the recent riskless uncle maker (RUM) attack [CCS '23]. In this work, we introduce the Staircase-Unrestricted Uncle Maker (SUUM), the first block withholding attack targeting the timestamp-based Nakamoto-style blockchain. Through block withholding, timestamp manipulation, and difficulty risk control, SUUM adversaries are capable of launching persistent attacks with zero cost and minimal difficulty risk characteristics, indefinitely exploiting rewards from honest participants. This creates a self-reinforcing cycle that threatens the security of blockchains. We conduct a comprehensive and systematic evaluation of SUUM, including the attack conditions, its impact on blockchains, and the difficulty risks. Finally, we further discuss four feasible mitigation measures against SUUM.



## **44. Decentralized Nonconvex Robust Optimization over Unsafe Multiagent Systems: System Modeling, Utility, Resilience, and Privacy Analysis**

math.OC

15 pages, 15 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2409.18632v7) [paper-pdf](http://arxiv.org/pdf/2409.18632v7)

**Authors**: Jinhui Hu, Guo Chen, Huaqing Li, Huqiang Cheng, Xiaoyu Guo, Tingwen Huang

**Abstract**: Privacy leakage and Byzantine failures are two adverse factors to the intelligent decision-making process of multi-agent systems (MASs). Considering the presence of these two issues, this paper targets the resolution of a class of nonconvex optimization problems under the Polyak-{\L}ojasiewicz (P-{\L}) condition. To address this problem, we first identify and construct the adversary system model. To enhance the robustness of stochastic gradient descent methods, we mask the local gradients with Gaussian noises and adopt a resilient aggregation method self-centered clipping (SCC) to design a differentially private (DP) decentralized Byzantine-resilient algorithm, namely DP-SCC-PL, which simultaneously achieves differential privacy and Byzantine resilience. The convergence analysis of DP-SCC-PL is challenging since the convergence error can be contributed jointly by privacy-preserving and Byzantine-resilient mechanisms, as well as the nonconvex relaxation, which is addressed via seeking the contraction relationships among the disagreement measure of reliable agents before and after aggregation, together with the optimal gap. Theoretical results reveal that DP-SCC-PL achieves consensus among all reliable agents and sublinear (inexact) convergence with well-designed step-sizes. It has also been proved that if there are no privacy issues and Byzantine agents, then the asymptotic exact convergence can be recovered. Numerical experiments verify the utility, resilience, and differential privacy of DP-SCC-PL by tackling a nonconvex optimization problem satisfying the P-{\L} condition under various Byzantine attacks.



## **45. PoisonArena: Uncovering Competing Poisoning Attacks in Retrieval-Augmented Generation**

cs.IR

29 pages

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.12574v3) [paper-pdf](http://arxiv.org/pdf/2505.12574v3)

**Authors**: Liuji Chen, Xiaofang Yang, Yuanzhuo Lu, Jinghao Zhang, Xin Sun, Qiang Liu, Shu Wu, Jing Dong, Liang Wang

**Abstract**: Retrieval-Augmented Generation (RAG) systems, widely used to improve the factual grounding of large language models (LLMs), are increasingly vulnerable to poisoning attacks, where adversaries inject manipulated content into the retriever's corpus. While prior research has predominantly focused on single-attacker settings, real-world scenarios often involve multiple, competing attackers with conflicting objectives. In this work, we introduce PoisonArena, the first benchmark to systematically study and evaluate competing poisoning attacks in RAG. We formalize the multi-attacker threat model, where attackers vie to control the answer to the same query using mutually exclusive misinformation. PoisonArena leverages the Bradley-Terry model to quantify each method's competitive effectiveness in such adversarial environments. Through extensive experiments on the Natural Questions and MS MARCO datasets, we demonstrate that many attack strategies successful in isolation fail under competitive pressure. Our findings highlight the limitations of conventional evaluation metrics like Attack Success Rate (ASR) and F1 score and underscore the need for competitive evaluation to assess real-world attack robustness. PoisonArena provides a standardized framework to benchmark and develop future attack and defense strategies under more realistic, multi-adversary conditions. Project page: https://github.com/yxf203/PoisonArena.



## **46. ErasableMask: A Robust and Erasable Privacy Protection Scheme against Black-box Face Recognition Models**

cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2412.17038v4) [paper-pdf](http://arxiv.org/pdf/2412.17038v4)

**Authors**: Sipeng Shen, Yunming Zhang, Dengpan Ye, Xiuwen Shi, Long Tang, Haoran Duan, Yueyun Shang, Zhihong Tian

**Abstract**: While face recognition (FR) models have brought remarkable convenience in face verification and identification, they also pose substantial privacy risks to the public. Existing facial privacy protection schemes usually adopt adversarial examples to disrupt face verification of FR models. However, these schemes often suffer from weak transferability against black-box FR models and permanently damage the identifiable information that cannot fulfill the requirements of authorized operations such as forensics and authentication. To address these limitations, we propose ErasableMask, a robust and erasable privacy protection scheme against black-box FR models. Specifically, via rethinking the inherent relationship between surrogate FR models, ErasableMask introduces a novel meta-auxiliary attack, which boosts black-box transferability by learning more general features in a stable and balancing optimization strategy. It also offers a perturbation erasion mechanism that supports the erasion of semantic perturbations in protected face without degrading image quality. To further improve performance, ErasableMask employs a curriculum learning strategy to mitigate optimization conflicts between adversarial attack and perturbation erasion. Extensive experiments on the CelebA-HQ and FFHQ datasets demonstrate that ErasableMask achieves the state-of-the-art performance in transferability, achieving over 72% confidence on average in commercial FR systems. Moreover, ErasableMask also exhibits outstanding perturbation erasion performance, achieving over 90% erasion success rate.



## **47. PandaGuard: Systematic Evaluation of LLM Safety against Jailbreaking Attacks**

cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.13862v2) [paper-pdf](http://arxiv.org/pdf/2505.13862v2)

**Authors**: Guobin Shen, Dongcheng Zhao, Linghao Feng, Xiang He, Jihang Wang, Sicheng Shen, Haibo Tong, Yiting Dong, Jindong Li, Xiang Zheng, Yi Zeng

**Abstract**: Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety.



## **48. SafeKey: Amplifying Aha-Moment Insights for Safety Reasoning**

cs.AI

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16186v1) [paper-pdf](http://arxiv.org/pdf/2505.16186v1)

**Authors**: Kaiwen Zhou, Xuandong Zhao, Gaowen Liu, Jayanth Srinivasa, Aosong Feng, Dawn Song, Xin Eric Wang

**Abstract**: Large Reasoning Models (LRMs) introduce a new generation paradigm of explicitly reasoning before answering, leading to remarkable improvements in complex tasks. However, they pose great safety risks against harmful queries and adversarial attacks. While recent mainstream safety efforts on LRMs, supervised fine-tuning (SFT), improve safety performance, we find that SFT-aligned models struggle to generalize to unseen jailbreak prompts. After thorough investigation of LRMs' generation, we identify a safety aha moment that can activate safety reasoning and lead to a safe response. This aha moment typically appears in the `key sentence', which follows models' query understanding process and can indicate whether the model will proceed safely. Based on these insights, we propose SafeKey, including two complementary objectives to better activate the safety aha moment in the key sentence: (1) a Dual-Path Safety Head to enhance the safety signal in the model's internal representations before the key sentence, and (2) a Query-Mask Modeling objective to improve the models' attention on its query understanding, which has important safety hints. Experiments across multiple safety benchmarks demonstrate that our methods significantly improve safety generalization to a wide range of jailbreak attacks and out-of-distribution harmful prompts, lowering the average harmfulness rate by 9.6\%, while maintaining general abilities. Our analysis reveals how SafeKey enhances safety by reshaping internal attention and improving the quality of hidden representations.



## **49. TRAIL: Transferable Robust Adversarial Images via Latent diffusion**

cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16166v1) [paper-pdf](http://arxiv.org/pdf/2505.16166v1)

**Authors**: Yuhao Xue, Zhifei Zhang, Xinyang Jiang, Yifei Shen, Junyao Gao, Wentao Gu, Jiale Zhao, Miaojing Shi, Cairong Zhao

**Abstract**: Adversarial attacks exploiting unrestricted natural perturbations present severe security risks to deep learning systems, yet their transferability across models remains limited due to distribution mismatches between generated adversarial features and real-world data. While recent works utilize pre-trained diffusion models as adversarial priors, they still encounter challenges due to the distribution shift between the distribution of ideal adversarial samples and the natural image distribution learned by the diffusion model. To address the challenge, we propose Transferable Robust Adversarial Images via Latent Diffusion (TRAIL), a test-time adaptation framework that enables the model to generate images from a distribution of images with adversarial features and closely resembles the target images. To mitigate the distribution shift, during attacks, TRAIL updates the diffusion U-Net's weights by combining adversarial objectives (to mislead victim models) and perceptual constraints (to preserve image realism). The adapted model then generates adversarial samples through iterative noise injection and denoising guided by these objectives. Experiments demonstrate that TRAIL significantly outperforms state-of-the-art methods in cross-model attack transferability, validating that distribution-aligned adversarial feature synthesis is critical for practical black-box attacks.



## **50. Robustifying Vision-Language Models via Dynamic Token Reweighting**

cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.17132v1) [paper-pdf](http://arxiv.org/pdf/2505.17132v1)

**Authors**: Tanqiu Jiang, Jiacheng Liang, Rongyi Zhu, Jiawei Zhou, Fenglong Ma, Ting Wang

**Abstract**: Large vision-language models (VLMs) are highly vulnerable to jailbreak attacks that exploit visual-textual interactions to bypass safety guardrails. In this paper, we present DTR, a novel inference-time defense that mitigates multimodal jailbreak attacks through optimizing the model's key-value (KV) caches. Rather than relying on curated safety-specific data or costly image-to-text conversion, we introduce a new formulation of the safety-relevant distributional shift induced by the visual modality. This formulation enables DTR to dynamically adjust visual token weights, minimizing the impact of adversarial visual inputs while preserving the model's general capabilities and inference efficiency. Extensive evaluation across diverse VLMs and attack benchmarks demonstrates that \sys outperforms existing defenses in both attack robustness and benign task performance, marking the first successful application of KV cache optimization for safety enhancement in multimodal foundation models. The code for replicating DTR is available: https://anonymous.4open.science/r/DTR-2755 (warning: this paper contains potentially harmful content generated by VLMs.)



