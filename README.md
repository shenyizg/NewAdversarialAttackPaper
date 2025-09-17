# Latest Adversarial Attack Papers
**update at 2025-09-17 09:53:34**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. JANUS: A Dual-Constraint Generative Framework for Stealthy Node Injection Attacks**

cs.LG

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.13266v1) [paper-pdf](http://arxiv.org/pdf/2509.13266v1)

**Authors**: Jiahao Zhang, Xiaobing Pei, Zhaokun Zhong, Wenqiang Hao, Zhenghao Tang

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable performance across various applications, yet they are vulnerable to sophisticated adversarial attacks, particularly node injection attacks. The success of such attacks heavily relies on their stealthiness, the ability to blend in with the original graph and evade detection. However, existing methods often achieve stealthiness by relying on indirect proxy metrics, lacking consideration for the fundamental characteristics of the injected content, or focusing only on imitating local structures, which leads to the problem of local myopia. To overcome these limitations, we propose a dual-constraint stealthy node injection framework, called Joint Alignment of Nodal and Universal Structures (JANUS). At the local level, we introduce a local feature manifold alignment strategy to achieve geometric consistency in the feature space. At the global level, we incorporate structured latent variables and maximize the mutual information with the generated structures, ensuring the injected structures are consistent with the semantic patterns of the original graph. We model the injection attack as a sequential decision process, which is optimized by a reinforcement learning agent. Experiments on multiple standard datasets demonstrate that the JANUS framework significantly outperforms existing methods in terms of both attack effectiveness and stealthiness.



## **2. Chernoff Information as a Privacy Constraint for Adversarial Classification and Membership Advantage**

cs.IT

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2403.10307v3) [paper-pdf](http://arxiv.org/pdf/2403.10307v3)

**Authors**: Ayşe Ünsal, Melek Önen

**Abstract**: This work inspects a privacy metric based on Chernoff information, namely Chernoff differential privacy, due to its significance in characterization of the optimal classifier's performance. Adversarial classification, as any other classification problem is built around minimization of the (average or correct detection) probability of error in deciding on either of the classes in the case of binary classification. Unlike the classical hypothesis testing problem, where the false alarm and mis-detection probabilities are handled separately resulting in an asymmetric behavior of the best error exponent, in this work, we characterize the relationship between $\varepsilon\textrm{-}$differential privacy, the best error exponent of one of the errors (when the other is fixed) and the best average error exponent. Accordingly, we re-derive Chernoff differential privacy in connection with $\varepsilon\textrm{-}$differential privacy using the Radon-Nikodym derivative, and prove its relation with Kullback-Leibler (KL) differential privacy. Subsequently, we present numerical evaluation results, which demonstrates that Chernoff information outperforms Kullback-Leibler divergence as a function of the privacy parameter $\varepsilon$ and the impact of the adversary's attack in Laplace mechanisms. Lastly, we introduce a new upper bound on adversary's membership advantage in membership inference attacks using Chernoff DP and numerically compare its performance with existing alternatives based on $(\varepsilon, \delta)\textrm{-}$differential privacy in the literature.



## **3. Detection of Synthetic Face Images: Accuracy, Robustness, Generalization**

cs.CV

The paper was presented at the DAGM German Conference on Pattern  Recognition (GCPR), 2025

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2406.17547v2) [paper-pdf](http://arxiv.org/pdf/2406.17547v2)

**Authors**: Nela Petrzelkova, Jan Cech

**Abstract**: An experimental study on detecting synthetic face images is presented. We collected a dataset, called FF5, of five fake face image generators, including recent diffusion models. We find that a simple model trained on a specific image generator can achieve near-perfect accuracy in separating synthetic and real images. The model handles common image distortions (reduced resolution, compression) by using data augmentation. Moreover, partial manipulations, where synthetic images are blended into real ones by inpainting, are identified and the area of the manipulation is localized by a simple model of YOLO architecture. However, the model turned out to be vulnerable to adversarial attacks and does not generalize to unseen generators. Failure to generalize to detect images produced by a newer generator also occurs for recent state-of-the-art methods, which we tested on Realistic Vision, a fine-tuned version of StabilityAI's Stable Diffusion image generator.



## **4. Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection**

cs.CL

EMNLP 2025 Main (Oral)

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2502.13061v4) [paper-pdf](http://arxiv.org/pdf/2502.13061v4)

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL



## **5. Bridging Threat Models and Detections: Formal Verification via CADP**

cs.CR

In Proceedings FROM 2025, arXiv:2509.11877

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.13035v1) [paper-pdf](http://arxiv.org/pdf/2509.13035v1)

**Authors**: Dumitru-Bogdan Prelipcean, Cătălin Dima

**Abstract**: Threat detection systems rely on rule-based logic to identify adversarial behaviors, yet the conformance of these rules to high-level threat models is rarely verified formally. We present a formal verification framework that models both detection logic and attack trees as labeled transition systems (LTSs), enabling automated conformance checking via bisimulation and weak trace inclusion. Detection rules specified in the Generic Threat Detection Language (GTDL, a general-purpose detection language we formalize in this work) are assigned a compositional operational semantics, and threat models expressed as attack trees are interpreted as LTSs through a structural trace semantics. Both representations are translated to LNT, a modeling language supported by the CADP toolbox. This common semantic domain enables systematic and automated verification of detection coverage. We evaluate our approach on real-world malware scenarios such as LokiBot and Emotet and provide scalability analysis through parametric synthetic models. Results confirm that our methodology identifies semantic mismatches between threat models and detection rules, supports iterative refinement, and scales to realistic threat landscapes.



## **6. A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems**

cs.CR

This paper is submitted to the IEEE IoT Journal

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2508.16843v4) [paper-pdf](http://arxiv.org/pdf/2508.16843v4)

**Authors**: Kamel Kamel, Keshav Sood, Hridoy Sankar Dutta, Sunil Aryal

**Abstract**: Voice authentication has undergone significant changes from traditional systems that relied on handcrafted acoustic features to deep learning models that can extract robust speaker embeddings. This advancement has expanded its applications across finance, smart devices, law enforcement, and beyond. However, as adoption has grown, so have the threats. This survey presents a comprehensive review of the modern threat landscape targeting Voice Authentication Systems (VAS) and Anti-Spoofing Countermeasures (CMs), including data poisoning, adversarial, deepfake, and adversarial spoofing attacks. We chronologically trace the development of voice authentication and examine how vulnerabilities have evolved in tandem with technological advancements. For each category of attack, we summarize methodologies, highlight commonly used datasets, compare performance and limitations, and organize existing literature using widely accepted taxonomies. By highlighting emerging risks and open challenges, this survey aims to support the development of more secure and resilient voice authentication systems.



## **7. Sy-FAR: Symmetry-based Fair Adversarial Robustness**

cs.LG

20 pages, 11 figures

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12939v1) [paper-pdf](http://arxiv.org/pdf/2509.12939v1)

**Authors**: Haneen Najjar, Eyal Ronen, Mahmood Sharif

**Abstract**: Security-critical machine-learning (ML) systems, such as face-recognition systems, are susceptible to adversarial examples, including real-world physically realizable attacks. Various means to boost ML's adversarial robustness have been proposed; however, they typically induce unfair robustness: It is often easier to attack from certain classes or groups than from others. Several techniques have been developed to improve adversarial robustness while seeking perfect fairness between classes. Yet, prior work has focused on settings where security and fairness are less critical. Our insight is that achieving perfect parity in realistic fairness-critical tasks, such as face recognition, is often infeasible -- some classes may be highly similar, leading to more misclassifications between them. Instead, we suggest that seeking symmetry -- i.e., attacks from class $i$ to $j$ would be as successful as from $j$ to $i$ -- is more tractable. Intuitively, symmetry is a desirable because class resemblance is a symmetric relation in most domains. Additionally, as we prove theoretically, symmetry between individuals induces symmetry between any set of sub-groups, in contrast to other fairness notions where group-fairness is often elusive. We develop Sy-FAR, a technique to encourage symmetry while also optimizing adversarial robustness and extensively evaluate it using five datasets, with three model architectures, including against targeted and untargeted realistic attacks. The results show Sy-FAR significantly improves fair adversarial robustness compared to state-of-the-art methods. Moreover, we find that Sy-FAR is faster and more consistent across runs. Notably, Sy-FAR also ameliorates another type of unfairness we discover in this work -- target classes that adversarial examples are likely to be classified into become significantly less vulnerable after inducing symmetry.



## **8. Adversarial Prompt Distillation for Vision-Language Models**

cs.CV

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2411.15244v3) [paper-pdf](http://arxiv.org/pdf/2411.15244v3)

**Authors**: Lin Luo, Xin Wang, Bojia Zi, Shihao Zhao, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as Contrastive Language-Image Pre-training (CLIP) have been shown to be susceptible to adversarial attacks, raising concerns about their deployment in safety-critical applications like autonomous driving and medical diagnosis. One promising approach for robustifying pre-trained VLMs is Adversarial Prompt Tuning (APT), which applies adversarial training during the process of prompt tuning. However, existing APT methods are mostly single-modal methods that design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy. In this work, we propose Adversarial Prompt Distillation (APD), a bimodal knowledge distillation framework that enhances APT by integrating it with multi-modal knowledge transfer. APD optimizes prompts for both visual and textual modalities while distilling knowledge from a clean pre-trained teacher CLIP model. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our APD method over the current state-of-the-art APT methods in terms of both adversarial robustness and clean accuracy. The effectiveness of APD also validates the possibility of using a non-robust teacher to improve the generalization and robustness of fine-tuned VLMs.



## **9. DiffHash: Text-Guided Targeted Attack via Diffusion Models against Deep Hashing Image Retrieval**

cs.IR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12824v1) [paper-pdf](http://arxiv.org/pdf/2509.12824v1)

**Authors**: Zechao Liu, Zheng Zhou, Xiangkun Chen, Tao Liang, Dapeng Lang

**Abstract**: Deep hashing models have been widely adopted to tackle the challenges of large-scale image retrieval. However, these approaches face serious security risks due to their vulnerability to adversarial examples. Despite the increasing exploration of targeted attacks on deep hashing models, existing approaches still suffer from a lack of multimodal guidance, reliance on labeling information and dependence on pixel-level operations for attacks. To address these limitations, we proposed DiffHash, a novel diffusion-based targeted attack for deep hashing. Unlike traditional pixel-based attacks that directly modify specific pixels and lack multimodal guidance, our approach focuses on optimizing the latent representations of images, guided by text information generated by a Large Language Model (LLM) for the target image. Furthermore, we designed a multi-space hash alignment network to align the high-dimension image space and text space to the low-dimension binary hash space. During reconstruction, we also incorporated text-guided attention mechanisms to refine adversarial examples, ensuring them aligned with the target semantics while maintaining visual plausibility. Extensive experiments have demonstrated that our method outperforms state-of-the-art (SOTA) targeted attack methods, achieving better black-box transferability and offering more excellent stability across datasets.



## **10. Gradient-Free Adversarial Purification with Diffusion Models**

cs.CV

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2501.13336v2) [paper-pdf](http://arxiv.org/pdf/2501.13336v2)

**Authors**: Xuelong Dai, Dong Wang, Xiuzhen Cheng, Bin Xiao

**Abstract**: Adversarial training and adversarial purification are two widely used defense strategies for enhancing model robustness against adversarial attacks. However, adversarial training requires costly retraining, while adversarial purification often suffers from low efficiency. More critically, existing defenses are primarily designed under the perturbation-based adversarial threat model, which is ineffective against recently introduced unrestricted adversarial attacks. In this paper, we propose an effective and efficient defense framework that counters both perturbation-based and unrestricted adversarial attacks. Our approach is motivated by the observation that adversarial examples typically lie near the decision boundary and are highly sensitive to pixel-level perturbations. To address this, we introduce adversarial anti-aliasing, a preprocessing technique that mitigates adversarial noise by reducing the magnitude of pixel-level perturbations. In addition, we propose adversarial super-resolution, which leverages prior knowledge from clean datasets to benignly restore high-quality images from adversarially degraded ones. Unlike image synthesis methods that generate entirely new images, adversarial super-resolution focuses on image restoration, making it more suitable for purification. Importantly, both techniques require no additional training and are computationally efficient since they do not rely on gradient computations. To further improve robustness across diverse datasets, we introduce a contrastive learning-based adversarial deblurring fine-tuning method. By incorporating adversarial priors during fine-tuning on the target dataset, this method enhances purification effectiveness without the need to retrain diffusion models.



## **11. Defense-to-Attack: Bypassing Weak Defenses Enables Stronger Jailbreaks in Vision-Language Models**

cs.CV

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12724v1) [paper-pdf](http://arxiv.org/pdf/2509.12724v1)

**Authors**: Yunhan Zhao, Xiang Zheng, Xingjun Ma

**Abstract**: Despite their superb capabilities, Vision-Language Models (VLMs) have been shown to be vulnerable to jailbreak attacks. While recent jailbreaks have achieved notable progress, their effectiveness and efficiency can still be improved. In this work, we reveal an interesting phenomenon: incorporating weak defense into the attack pipeline can significantly enhance both the effectiveness and the efficiency of jailbreaks on VLMs. Building on this insight, we propose Defense2Attack, a novel jailbreak method that bypasses the safety guardrails of VLMs by leveraging defensive patterns to guide jailbreak prompt design. Specifically, Defense2Attack consists of three key components: (1) a visual optimizer that embeds universal adversarial perturbations with affirmative and encouraging semantics; (2) a textual optimizer that refines the input using a defense-styled prompt; and (3) a red-team suffix generator that enhances the jailbreak through reinforcement fine-tuning. We empirically evaluate our method on four VLMs and four safety benchmarks. The results demonstrate that Defense2Attack achieves superior jailbreak performance in a single attempt, outperforming state-of-the-art attack methods that often require multiple tries. Our work offers a new perspective on jailbreaking VLMs.



## **12. Revisiting Transferable Adversarial Images: Systemization, Evaluation, and New Insights**

cs.CR

TPAMI 2025. Code is available at  https://github.com/ZhengyuZhao/TransferAttackEval

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2310.11850v2) [paper-pdf](http://arxiv.org/pdf/2310.11850v2)

**Authors**: Zhengyu Zhao, Hanwei Zhang, Renjue Li, Ronan Sicre, Laurent Amsaleg, Michael Backes, Qi Li, Qian Wang, Chao Shen

**Abstract**: Transferable adversarial images raise critical security concerns for computer vision systems in real-world, black-box attack scenarios. Although many transfer attacks have been proposed, existing research lacks a systematic and comprehensive evaluation. In this paper, we systemize transfer attacks into five categories around the general machine learning pipeline and provide the first comprehensive evaluation, with 23 representative attacks against 11 representative defenses, including the recent, transfer-oriented defense and the real-world Google Cloud Vision. In particular, we identify two main problems of existing evaluations: (1) for attack transferability, lack of intra-category analyses with fair hyperparameter settings, and (2) for attack stealthiness, lack of diverse measures. Our evaluation results validate that these problems have indeed caused misleading conclusions and missing points, and addressing them leads to new, \textit{consensus-challenging} insights, such as (1) an early attack, DI, even outperforms all similar follow-up ones, (2) the state-of-the-art (white-box) defense, DiffPure, is even vulnerable to (black-box) transfer attacks, and (3) even under the same $L_p$ constraint, different attacks yield dramatically different stealthiness results regarding diverse imperceptibility metrics, finer-grained measures, and a user study. We hope that our analyses will serve as guidance on properly evaluating transferable adversarial images and advance the design of attacks and defenses. Code is available at https://github.com/ZhengyuZhao/TransferAttackEval.



## **13. SoK: How Sensor Attacks Disrupt Autonomous Vehicles: An End-to-end Analysis, Challenges, and Missed Threats**

cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.11120v2) [paper-pdf](http://arxiv.org/pdf/2509.11120v2)

**Authors**: Qingzhao Zhang, Shaocheng Luo, Z. Morley Mao, Miroslav Pajic, Michael K. Reiter

**Abstract**: Autonomous vehicles, including self-driving cars, robotic ground vehicles, and drones, rely on complex sensor pipelines to ensure safe and reliable operation. However, these safety-critical systems remain vulnerable to adversarial sensor attacks that can compromise their performance and mission success. While extensive research has demonstrated various sensor attack techniques, critical gaps remain in understanding their feasibility in real-world, end-to-end systems. This gap largely stems from the lack of a systematic perspective on how sensor errors propagate through interconnected modules in autonomous systems when autonomous vehicles interact with the physical world.   To bridge this gap, we present a comprehensive survey of autonomous vehicle sensor attacks across platforms, sensor modalities, and attack methods. Central to our analysis is the System Error Propagation Graph (SEPG), a structured demonstration tool that illustrates how sensor attacks propagate through system pipelines, exposing the conditions and dependencies that determine attack feasibility. With the aid of SEPG, our study distills seven key findings that highlight the feasibility challenges of sensor attacks and uncovers eleven previously overlooked attack vectors exploiting inter-module interactions, several of which we validate through proof-of-concept experiments. Additionally, we demonstrate how large language models (LLMs) can automate aspects of SEPG construction and cross-validate expert analysis, showcasing the promise of AI-assisted security evaluation.



## **14. Towards Inclusive Toxic Content Moderation: Addressing Vulnerabilities to Adversarial Attacks in Toxicity Classifiers Tackling LLM-generated Content**

cs.CL

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12672v1) [paper-pdf](http://arxiv.org/pdf/2509.12672v1)

**Authors**: Shaz Furniturewala, Arkaitz Zubiaga

**Abstract**: The volume of machine-generated content online has grown dramatically due to the widespread use of Large Language Models (LLMs), leading to new challenges for content moderation systems. Conventional content moderation classifiers, which are usually trained on text produced by humans, suffer from misclassifications due to LLM-generated text deviating from their training data and adversarial attacks that aim to avoid detection. Present-day defence tactics are reactive rather than proactive, since they rely on adversarial training or external detection models to identify attacks. In this work, we aim to identify the vulnerable components of toxicity classifiers that contribute to misclassification, proposing a novel strategy based on mechanistic interpretability techniques. Our study focuses on fine-tuned BERT and RoBERTa classifiers, testing on diverse datasets spanning a variety of minority groups. We use adversarial attacking techniques to identify vulnerable circuits. Finally, we suppress these vulnerable circuits, improving performance against adversarial attacks. We also provide demographic-level insights into these vulnerable circuits, exposing fairness and robustness gaps in model training. We find that models have distinct heads that are either crucial for performance or vulnerable to attack and suppressing the vulnerable heads improves performance on adversarial input. We also find that different heads are responsible for vulnerability across different demographic groups, which can inform more inclusive development of toxicity detection models.



## **15. CIARD: Cyclic Iterative Adversarial Robustness Distillation**

cs.CV

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12633v1) [paper-pdf](http://arxiv.org/pdf/2509.12633v1)

**Authors**: Liming Lu, Shuchao Pang, Xu Zheng, Xiang Gu, Anan Du, Yunhuai Liu, Yongbin Zhou

**Abstract**: Adversarial robustness distillation (ARD) aims to transfer both performance and robustness from teacher model to lightweight student model, enabling resilient performance on resource-constrained scenarios. Though existing ARD approaches enhance student model's robustness, the inevitable by-product leads to the degraded performance on clean examples. We summarize the causes of this problem inherent in existing methods with dual-teacher framework as: 1. The divergent optimization objectives of dual-teacher models, i.e., the clean and robust teachers, impede effective knowledge transfer to the student model, and 2. The iteratively generated adversarial examples during training lead to performance deterioration of the robust teacher model. To address these challenges, we propose a novel Cyclic Iterative ARD (CIARD) method with two key innovations: a. A multi-teacher framework with contrastive push-loss alignment to resolve conflicts in dual-teacher optimization objectives, and b. Continuous adversarial retraining to maintain dynamic teacher robustness against performance degradation from the varying adversarial examples. Extensive experiments on CIFAR-10, CIFAR-100, and Tiny-ImageNet demonstrate that CIARD achieves remarkable performance with an average 3.53 improvement in adversarial defense rates across various attack scenarios and a 5.87 increase in clean sample accuracy, establishing a new benchmark for balancing model robustness and generalization. Our code is available at https://github.com/eminentgu/CIARD



## **16. Your Compiler is Backdooring Your Model: Understanding and Exploiting Compilation Inconsistency Vulnerabilities in Deep Learning Compilers**

cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.11173v2) [paper-pdf](http://arxiv.org/pdf/2509.11173v2)

**Authors**: Simin Chen, Jinjun Peng, Yixin He, Junfeng Yang, Baishakhi Ray

**Abstract**: Deep learning (DL) compilers are core infrastructure in modern DL systems, offering flexibility and scalability beyond vendor-specific libraries. This work uncovers a fundamental vulnerability in their design: can an official, unmodified compiler alter a model's semantics during compilation and introduce hidden backdoors? We study both adversarial and natural settings. In the adversarial case, we craft benign models where triggers have no effect pre-compilation but become effective backdoors after compilation. Tested on six models, three commercial compilers, and two hardware platforms, our attack yields 100% success on triggered inputs while preserving normal accuracy and remaining undetected by state-of-the-art detectors. The attack generalizes across compilers, hardware, and floating-point settings. In the natural setting, we analyze the top 100 HuggingFace models (including one with 220M+ downloads) and find natural triggers in 31 models. This shows that compilers can introduce risks even without adversarial manipulation.   Our results reveal an overlooked threat: unmodified DL compilers can silently alter model semantics. To our knowledge, this is the first work to expose inherent security risks in DL compiler design, opening a new direction for secure and trustworthy ML.



## **17. DisorientLiDAR: Physical Attacks on LiDAR-based Localization**

cs.CV

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12595v1) [paper-pdf](http://arxiv.org/pdf/2509.12595v1)

**Authors**: Yizhen Lao, Yu Zhang, Ziting Wang, Chengbo Wang, Yifei Xue, Wanpeng Shao

**Abstract**: Deep learning models have been shown to be susceptible to adversarial attacks with visually imperceptible perturbations. Even this poses a serious security challenge for the localization of self-driving cars, there has been very little exploration of attack on it, as most of adversarial attacks have been applied to 3D perception. In this work, we propose a novel adversarial attack framework called DisorientLiDAR targeting LiDAR-based localization. By reverse-engineering localization models (e.g., feature extraction networks), adversaries can identify critical keypoints and strategically remove them, thereby disrupting LiDAR-based localization. Our proposal is first evaluated on three state-of-the-art point-cloud registration models (HRegNet, D3Feat, and GeoTransformer) using the KITTI dataset. Experimental results demonstrate that removing regions containing Top-K keypoints significantly degrades their registration accuracy. We further validate the attack's impact on the Autoware autonomous driving platform, where hiding merely a few critical regions induces noticeable localization drift. Finally, we extended our attacks to the physical world by hiding critical regions with near-infrared absorptive materials, thereby successfully replicate the attack effects observed in KITTI data. This step has been closer toward the realistic physical-world attack that demonstrate the veracity and generality of our proposal.



## **18. PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance**

cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2508.20890v2) [paper-pdf](http://arxiv.org/pdf/2508.20890v2)

**Authors**: Mengxiao Wang, Yuxuan Zhang, Guofei Gu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications, from virtual assistants to autonomous agents. However, their flexibility also introduces new attack vectors-particularly Prompt Injection (PI), where adversaries manipulate model behavior through crafted inputs. As attackers continuously evolve with paraphrased, obfuscated, and even multi-task injection strategies, existing benchmarks are no longer sufficient to capture the full spectrum of emerging threats.   To address this gap, we construct a new benchmark that systematically extends prior efforts. Our benchmark subsumes the two widely-used existing ones while introducing new manipulation techniques and multi-task scenarios, thereby providing a more comprehensive evaluation setting. We find that existing defenses, though effective on their original benchmarks, show clear weaknesses under our benchmark, underscoring the need for more robust solutions. Our key insight is that while attack forms may vary, the adversary's intent-injecting an unauthorized task-remains invariant. Building on this observation, we propose PromptSleuth, a semantic-oriented defense framework that detects prompt injection by reasoning over task-level intent rather than surface features. Evaluated across state-of-the-art benchmarks, PromptSleuth consistently outperforms existing defense while maintaining comparable runtime and cost efficiency. These results demonstrate that intent-based semantic reasoning offers a robust, efficient, and generalizable strategy for defending LLMs against evolving prompt injection threats.



## **19. Exploiting Timing Side-Channels in Quantum Circuits Simulation Via ML-Based Methods**

cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12535v1) [paper-pdf](http://arxiv.org/pdf/2509.12535v1)

**Authors**: Ben Dong, Hui Feng, Qian Wang

**Abstract**: As quantum computing advances, quantum circuit simulators serve as critical tools to bridge the current gap caused by limited quantum hardware availability. These simulators are typically deployed on cloud platforms, where users submit proprietary circuit designs for simulation. In this work, we demonstrate a novel timing side-channel attack targeting cloud- based quantum simulators. A co-located malicious process can observe fine-grained execution timing patterns to extract sensitive information about concurrently running quantum circuits. We systematically analyze simulator behavior using the QASMBench benchmark suite, profiling timing and memory characteristics across various circuit executions. Our experimental results show that timing profiles exhibit circuit-dependent patterns that can be effectively classified using pattern recognition techniques, enabling the adversary to infer circuit identities and compromise user confidentiality. We were able to achieve 88% to 99.9% identification rate of quantum circuits based on different datasets. This work highlights previously unexplored security risks in quantum simulation environments and calls for stronger isolation mechanisms to protect user workloads



## **20. How to Beat Nakamoto in the Race**

cs.CR

To be presented at the 2025 ACM Conference on Computer and  Communications Security (CCS)

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2508.16202v2) [paper-pdf](http://arxiv.org/pdf/2508.16202v2)

**Authors**: Shu-Jie Cao, Dongning Guo

**Abstract**: This paper studies proof-of-work Nakamoto consensus protocols under bounded network delays, settling two long-standing questions in blockchain security: What is the most effective attack on block safety under a given block confirmation latency? And what is the resulting probability of safety violation? A Markov decision process (MDP) framework is introduced to precisely characterize the system state (including the blocktree and timings of all blocks mined), the adversary's potential actions, and the state transitions due to the adversarial action and the random block arrival processes. An optimal attack, called bait-and-switch, is proposed and proved to maximize the adversary's chance of violating block safety by "beating Nakamoto in the race". The exact probability of this violation is calculated for any given confirmation depth using Markov chain analysis, offering fresh insights into the interplay of network delay, confirmation rules, and blockchain security.



## **21. Time-Constrained Intelligent Adversaries for Automation Vulnerability Testing: A Multi-Robot Patrol Case Study**

cs.RO

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11971v1) [paper-pdf](http://arxiv.org/pdf/2509.11971v1)

**Authors**: James C. Ward, Alex Bott, Connor York, Edmund R. Hunt

**Abstract**: Simulating hostile attacks of physical autonomous systems can be a useful tool to examine their robustness to attack and inform vulnerability-aware design. In this work, we examine this through the lens of multi-robot patrol, by presenting a machine learning-based adversary model that observes robot patrol behavior in order to attempt to gain undetected access to a secure environment within a limited time duration. Such a model allows for evaluation of a patrol system against a realistic potential adversary, offering insight into future patrol strategy design. We show that our new model outperforms existing baselines, thus providing a more stringent test, and examine its performance against multiple leading decentralized multi-robot patrol strategies.



## **22. NeuroStrike: Neuron-Level Attacks on Aligned LLMs**

cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11864v1) [paper-pdf](http://arxiv.org/pdf/2509.11864v1)

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Maximilian Thang, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability.   This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.



## **23. A Practical Adversarial Attack against Sequence-based Deep Learning Malware Classifiers**

cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11836v1) [paper-pdf](http://arxiv.org/pdf/2509.11836v1)

**Authors**: Kai Tan, Dongyang Zhan, Lin Ye, Hongli Zhang, Binxing Fang

**Abstract**: Sequence-based deep learning models (e.g., RNNs), can detect malware by analyzing its behavioral sequences. Meanwhile, these models are susceptible to adversarial attacks. Attackers can create adversarial samples that alter the sequence characteristics of behavior sequences to deceive malware classifiers. The existing methods for generating adversarial samples typically involve deleting or replacing crucial behaviors in the original data sequences, or inserting benign behaviors that may violate the behavior constraints. However, these methods that directly manipulate sequences make adversarial samples difficult to implement or apply in practice. In this paper, we propose an adversarial attack approach based on Deep Q-Network and a heuristic backtracking search strategy, which can generate perturbation sequences that satisfy practical conditions for successful attacks. Subsequently, we utilize a novel transformation approach that maps modifications back to the source code, thereby avoiding the need to directly modify the behavior log sequences. We conduct an evaluation of our approach, and the results confirm its effectiveness in generating adversarial samples from real-world malware behavior sequences, which have a high success rate in evading anomaly detection models. Furthermore, our approach is practical and can generate adversarial samples while maintaining the functionality of the modified software.



## **24. Removal Attack and Defense on AI-generated Content Latent-based Watermarking**

cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11745v1) [paper-pdf](http://arxiv.org/pdf/2509.11745v1)

**Authors**: De Zhang Lee, Han Fang, Hanyi Wang, Ee-Chien Chang

**Abstract**: Digital watermarks can be embedded into AI-generated content (AIGC) by initializing the generation process with starting points sampled from a secret distribution. When combined with pseudorandom error-correcting codes, such watermarked outputs can remain indistinguishable from unwatermarked objects, while maintaining robustness under whitenoise. In this paper, we go beyond indistinguishability and investigate security under removal attacks. We demonstrate that indistinguishability alone does not necessarily guarantee resistance to adversarial removal. Specifically, we propose a novel attack that exploits boundary information leaked by the locations of watermarked objects. This attack significantly reduces the distortion required to remove watermarks -- by up to a factor of $15 \times$ compared to a baseline whitenoise attack under certain settings. To mitigate such attacks, we introduce a defense mechanism that applies a secret transformation to hide the boundary, and prove that the secret transformation effectively rendering any attacker's perturbations equivalent to those of a naive whitenoise adversary. Our empirical evaluations, conducted on multiple versions of Stable Diffusion, validate the effectiveness of both the attack and the proposed defense, highlighting the importance of addressing boundary leakage in latent-based watermarking schemes.



## **25. DARD: Dice Adversarial Robustness Distillation against Adversarial Attacks**

cs.LG

Accepted at SecureComm 2025, 15 pages, 4 figures

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11525v1) [paper-pdf](http://arxiv.org/pdf/2509.11525v1)

**Authors**: Jing Zou, Shungeng Zhang, Meikang Qiu, Chong Li

**Abstract**: Deep learning models are vulnerable to adversarial examples, posing critical security challenges in real-world applications. While Adversarial Training (AT ) is a widely adopted defense mechanism to enhance robustness, it often incurs a trade-off by degrading performance on unperturbed, natural data. Recent efforts have highlighted that larger models exhibit enhanced robustness over their smaller counterparts. In this paper, we empirically demonstrate that such robustness can be systematically distilled from large teacher models into compact student models. To achieve better performance, we introduce Dice Adversarial Robustness Distillation (DARD), a novel method designed to transfer robustness through a tailored knowledge distillation paradigm. Additionally, we propose Dice Projected Gradient Descent (DPGD), an adversarial example generalization method optimized for effective attack. Our extensive experiments demonstrate that the DARD approach consistently outperforms adversarially trained networks with the same architecture, achieving superior robustness and standard accuracy.



## **26. Pulse-to-Circuit Characterization of Stealthy Crosstalk Attack on Multi-Tenant Superconducting Quantum Hardware**

quant-ph

Will appear in the Proceedings of the 2025 Quantum Security and  Privacy Workshop (QSec '25)

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11407v1) [paper-pdf](http://arxiv.org/pdf/2509.11407v1)

**Authors**: Syed Emad Uddin Shubha, Tasnuva Farheen

**Abstract**: Hardware crosstalk in multi-tenant superconducting quantum computers constitutes a significant security threat, enabling adversaries to inject targeted errors across tenant boundaries. We present the first end-to-end framework for mapping physical pulse-level attacks to interpretable logical error channels, integrating density-matrix simulation, quantum process tomography (QPT), and a novel isometry-based circuit extraction method. Our pipeline reconstructs the complete induced error channel and fits an effective logical circuit model, revealing a fundamentally asymmetric attack mechanism: one adversarial qubit acts as a driver to set the induced logical rotation, while a second, the catalyst, refines the attack's coherence. Demonstrated on a linear three-qubit system, our approach shows that such attacks can significantly disrupt diverse quantum protocols, sometimes reducing accuracy to random guessing, while remaining effective and stealthy even under realistic hardware parameter variations. We further propose a protocol-level detection strategy based on observable attack signatures, showing that stealthy attacks can be exposed through targeted monitoring and providing a foundation for future defense-in-depth in quantum cloud platforms.



## **27. From Firewalls to Frontiers: AI Red-Teaming is a Domain-Specific Evolution of Cyber Red-Teaming**

cs.LG

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11398v1) [paper-pdf](http://arxiv.org/pdf/2509.11398v1)

**Authors**: Anusha Sinha, Keltin Grimes, James Lucassen, Michael Feffer, Nathan VanHoudnos, Zhiwei Steven Wu, Hoda Heidari

**Abstract**: A red team simulates adversary attacks to help defenders find effective strategies to defend their systems in a real-world operational setting. As more enterprise systems adopt AI, red-teaming will need to evolve to address the unique vulnerabilities and risks posed by AI systems. We take the position that AI systems can be more effectively red-teamed if AI red-teaming is recognized as a domain-specific evolution of cyber red-teaming. Specifically, we argue that existing Cyber Red Teams who adopt this framing will be able to better evaluate systems with AI components by recognizing that AI poses new risks, has new failure modes to exploit, and often contains unpatchable bugs that re-prioritize disclosure and mitigation strategies. Similarly, adopting a cybersecurity framing will allow existing AI Red Teams to leverage a well-tested structure to emulate realistic adversaries, promote mutual accountability with formal rules of engagement, and provide a pattern to mature the tooling necessary for repeatable, scalable engagements. In these ways, the merging of AI and Cyber Red Teams will create a robust security ecosystem and best position the community to adapt to the rapidly changing threat landscape.



## **28. On the Escaping Efficiency of Distributed Adversarial Training Algorithms**

cs.LG

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11337v1) [paper-pdf](http://arxiv.org/pdf/2509.11337v1)

**Authors**: Ying Cao, Kun Yuan, Ali H. Sayed

**Abstract**: Adversarial training has been widely studied in recent years due to its role in improving model robustness against adversarial attacks. This paper focuses on comparing different distributed adversarial training algorithms--including centralized and decentralized strategies--within multi-agent learning environments. Previous studies have highlighted the importance of model flatness in determining robustness. To this end, we develop a general theoretical framework to study the escaping efficiency of these algorithms from local minima, which is closely related to the flatness of the resulting models. We show that when the perturbation bound is sufficiently small (i.e., when the attack strength is relatively mild) and a large batch size is used, decentralized adversarial training algorithms--including consensus and diffusion--are guaranteed to escape faster from local minima than the centralized strategy, thereby favoring flatter minima. However, as the perturbation bound increases, this trend may no longer hold. In the simulation results, we illustrate our theoretical findings and systematically compare the performance of models obtained through decentralized and centralized adversarial training algorithms. The results highlight the potential of decentralized strategies to enhance the robustness of models in distributed settings.



## **29. ANROT-HELANet: Adverserially and Naturally Robust Attention-Based Aggregation Network via The Hellinger Distance for Few-Shot Classification**

cs.CV

Preprint version. The manuscript has been submitted to a journal. All  changes will be transferred to the final version if accepted. Also an  erratum: In Figure 10 and 11, the $\epsilon = 0.005$ value should be  $\epsilon = 0.05$

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11220v1) [paper-pdf](http://arxiv.org/pdf/2509.11220v1)

**Authors**: Gao Yu Lee, Tanmoy Dam, Md Meftahul Ferdaus, Daniel Puiu Poenar, Vu N. Duong

**Abstract**: Few-Shot Learning (FSL), which involves learning to generalize using only a few data samples, has demonstrated promising and superior performances to ordinary CNN methods. While Bayesian based estimation approaches using Kullback-Leibler (KL) divergence have shown improvements, they remain vulnerable to adversarial attacks and natural noises. We introduce ANROT-HELANet, an Adversarially and Naturally RObusT Hellinger Aggregation Network that significantly advances the state-of-the-art in FSL robustness and performance. Our approach implements an adversarially and naturally robust Hellinger distance-based feature class aggregation scheme, demonstrating resilience to adversarial perturbations up to $\epsilon=0.30$ and Gaussian noise up to $\sigma=0.30$. The network achieves substantial improvements across benchmark datasets, including gains of 1.20\% and 1.40\% for 1-shot and 5-shot scenarios on miniImageNet respectively. We introduce a novel Hellinger Similarity contrastive loss function that generalizes cosine similarity contrastive loss for variational few-shot inference scenarios. Our approach also achieves superior image reconstruction quality with a FID score of 2.75, outperforming traditional VAE (3.43) and WAE (3.38) approaches. Extensive experiments conducted on four few-shot benchmarked datasets verify that ANROT-HELANet's combination of Hellinger distance-based feature aggregation, attention mechanisms, and our novel loss function establishes new state-of-the-art performance while maintaining robustness against both adversarial and natural perturbations. Our code repository will be available at https://github.com/GreedYLearner1146/ANROT-HELANet/tree/main.



## **30. Fighting Fire with Fire (F3): A Training-free and Efficient Visual Adversarial Example Purification Method in LVLMs**

cs.CV

Accepted by ACM Multimedia 2025 BNI track (Oral)

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2506.01064v3) [paper-pdf](http://arxiv.org/pdf/2506.01064v3)

**Authors**: Yudong Zhang, Ruobing Xie, Yiqing Huang, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Di Wang, Yu Wang

**Abstract**: Recent advances in large vision-language models (LVLMs) have showcased their remarkable capabilities across a wide range of multimodal vision-language tasks. However, these models remain vulnerable to visual adversarial attacks, which can substantially compromise their performance. In this paper, we introduce F3, a novel adversarial purification framework that employs a counterintuitive ``fighting fire with fire'' strategy: intentionally introducing simple perturbations to adversarial examples to mitigate their harmful effects. Specifically, F3 leverages cross-modal attentions derived from randomly perturbed adversary examples as reference targets. By injecting noise into these adversarial examples, F3 effectively refines their attention, resulting in cleaner and more reliable model outputs. Remarkably, this seemingly paradoxical approach of employing noise to counteract adversarial attacks yields impressive purification results. Furthermore, F3 offers several distinct advantages: it is training-free and straightforward to implement, and exhibits significant computational efficiency improvements compared to existing purification methods. These attributes render F3 particularly suitable for large-scale industrial applications where both robust performance and operational efficiency are critical priorities. The code is available at https://github.com/btzyd/F3.



## **31. DMLDroid: Deep Multimodal Fusion Framework for Android Malware Detection with Resilience to Code Obfuscation and Adversarial Perturbations**

cs.CR

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11187v1) [paper-pdf](http://arxiv.org/pdf/2509.11187v1)

**Authors**: Doan Minh Trung, Tien Duc Anh Hao, Luong Hoang Minh, Nghi Hoang Khoa, Nguyen Tan Cam, Van-Hau Pham, Phan The Duy

**Abstract**: In recent years, learning-based Android malware detection has seen significant advancements, with detectors generally falling into three categories: string-based, image-based, and graph-based approaches. While these methods have shown strong detection performance, they often struggle to sustain robustness in real-world settings, particularly when facing code obfuscation and adversarial examples (AEs). Deep multimodal learning has emerged as a promising solution, leveraging the strengths of multiple feature types to enhance robustness and generalization. However, a systematic investigation of multimodal fusion for both accuracy and resilience remains underexplored. In this study, we propose DMLDroid, an Android malware detection based on multimodal fusion that leverages three different representations of malware features, including permissions & intents (tabular-based), DEX file representations (image-based), and API calls (graph-derived sequence-based). We conduct exhaustive experiments independently on each feature, as well as in combination, using different fusion strategies. Experimental results on the CICMalDroid 2020 dataset demonstrate that our multimodal approach with the dynamic weighted fusion mechanism achieves high performance, reaching 97.98% accuracy and 98.67% F1-score on original malware detection. Notably, the proposed method maintains strong robustness, sustaining over 98% accuracy and 98% F1-score under both obfuscation and adversarial attack scenarios. Our findings highlight the benefits of multimodal fusion in improving both detection accuracy and robustness against evolving Android malware threats.



## **32. Feature Space Topology Control via Hopkins Loss**

cs.LG

Accepted for publication in Proc. IEEE ICTAI 2025, Athens, Greece

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11154v1) [paper-pdf](http://arxiv.org/pdf/2509.11154v1)

**Authors**: Einari Vaaras, Manu Airaksinen

**Abstract**: Feature space topology refers to the organization of samples within the feature space. Modifying this topology can be beneficial in machine learning applications, including dimensionality reduction, generative modeling, transfer learning, and robustness to adversarial attacks. This paper introduces a novel loss function, Hopkins loss, which leverages the Hopkins statistic to enforce a desired feature space topology, which is in contrast to existing topology-related methods that aim to preserve input feature topology. We evaluate the effectiveness of Hopkins loss on speech, text, and image data in two scenarios: classification and dimensionality reduction using nonlinear bottleneck autoencoders. Our experiments show that integrating Hopkins loss into classification or dimensionality reduction has only a small impact on classification performance while providing the benefit of modifying feature topology.



## **33. Character-Level Perturbations Disrupt LLM Watermarks**

cs.CR

accepted by Network and Distributed System Security (NDSS) Symposium  2026

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.09112v2) [paper-pdf](http://arxiv.org/pdf/2509.09112v2)

**Authors**: Zhaoxi Zhang, Xiaomei Zhang, Yanjun Zhang, He Zhang, Shirui Pan, Bo Liu, Asif Qumer Gill, Leo Yu Zhang

**Abstract**: Large Language Model (LLM) watermarking embeds detectable signals into generated text for copyright protection, misuse prevention, and content detection. While prior studies evaluate robustness using watermark removal attacks, these methods are often suboptimal, creating the misconception that effective removal requires large perturbations or powerful adversaries.   To bridge the gap, we first formalize the system model for LLM watermark, and characterize two realistic threat models constrained on limited access to the watermark detector. We then analyze how different types of perturbation vary in their attack range, i.e., the number of tokens they can affect with a single edit. We observe that character-level perturbations (e.g., typos, swaps, deletions, homoglyphs) can influence multiple tokens simultaneously by disrupting the tokenization process. We demonstrate that character-level perturbations are significantly more effective for watermark removal under the most restrictive threat model. We further propose guided removal attacks based on the Genetic Algorithm (GA) that uses a reference detector for optimization. Under a practical threat model with limited black-box queries to the watermark detector, our method demonstrates strong removal performance. Experiments confirm the superiority of character-level perturbations and the effectiveness of the GA in removing watermarks under realistic constraints. Additionally, we argue there is an adversarial dilemma when considering potential defenses: any fixed defense can be bypassed by a suitable perturbation strategy. Motivated by this principle, we propose an adaptive compound character-level attack. Experimental results show that this approach can effectively defeat the defenses. Our findings highlight significant vulnerabilities in existing LLM watermark schemes and underline the urgency for the development of new robust mechanisms.



## **34. ENJ: Optimizing Noise with Genetic Algorithms to Jailbreak LSMs**

cs.SD

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11128v1) [paper-pdf](http://arxiv.org/pdf/2509.11128v1)

**Authors**: Yibo Zhang, Liang Lin

**Abstract**: The widespread application of Large Speech Models (LSMs) has made their security risks increasingly prominent. Traditional speech adversarial attack methods face challenges in balancing effectiveness and stealth. This paper proposes Evolutionary Noise Jailbreak (ENJ), which utilizes a genetic algorithm to transform environmental noise from a passive interference into an actively optimizable attack carrier for jailbreaking LSMs. Through operations such as population initialization, crossover fusion, and probabilistic mutation, this method iteratively evolves a series of audio samples that fuse malicious instructions with background noise. These samples sound like harmless noise to humans but can induce the model to parse and execute harmful commands. Extensive experiments on multiple mainstream speech models show that ENJ's attack effectiveness is significantly superior to existing baseline methods. This research reveals the dual role of noise in speech security and provides new critical insights for model security defense in complex acoustic environments.



## **35. Nonreciprocal RIS-Aided Covert Channel Reciprocity Attacks and Countermeasures**

eess.SP

submitted to IEEE Trans for review

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11117v1) [paper-pdf](http://arxiv.org/pdf/2509.11117v1)

**Authors**: Haoyu Wang, Jiawei Hu, Jiaqi Xu, Ying Ju, A. Lee Swindlehurst

**Abstract**: Reconfigurable intelligent surface (RIS) technology enhances wireless communication performance, but it also introduces new vulnerabilities that can be exploited by adversaries. This paper investigates channel reciprocity attack (CRACK) threats in multi-antenna wireless systems operating in time-division duplexing mode using a physically consistent non-reciprocal RIS (NR-RIS) model. CRACK can degrade communication rate and facilitate passive eavesdropping behavior by distorting the downlink precoding, without requiring any additional signal transmission or channel state information (CSI). Unlike conventional RIS jamming strategies, the NR-RIS does not need synchronization with the legitimate system and thus can operate with slow or fixed configurations to implement CRACK, obscuring the distinction between the direct and RIS-induced channels and thereby complicating corresponding defensive precoding designs. To counter the CRACK threat posed by NR-RIS, we develop ``SecureCoder,'' a deep reinforcement learning-based framework that can mitigate CRACK and determine an improved downlink precoder matrix using the estimated uplink CSI and rate feedback from the users. Simulation results demonstrate the severe performance degradation caused by NR-RIS CRACK and validate the effectiveness of SecureCoder in improving both throughput and reducing security threats, thereby enhancing system robustness.



## **36. Adversarial control of synchronization in complex oscillator networks**

nlin.AO

10 pages, 4 figures

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2506.02403v3) [paper-pdf](http://arxiv.org/pdf/2506.02403v3)

**Authors**: Yasutoshi Nagahama, Kosuke Miyazato, Kazuhiro Takemoto

**Abstract**: This study investigates perturbation strategies inspired by adversarial attack principles from deep learning, designed to control synchronization dynamics through strategically crafted weak perturbations. We propose a gradient-based optimization method that identifies small phase perturbations to dramatically enhance or suppress collective synchronization in Kuramoto oscillator networks. Our approach formulates synchronization control as an optimization problem, computing gradients of the order parameter with respect to oscillator phases to determine optimal perturbation directions. Results demonstrate that extremely small phase perturbations applied to network oscillators can achieve significant synchronization control across diverse network architectures. Our analysis reveals that synchronization enhancement is achievable across various network sizes, while synchronization suppression becomes particularly effective in larger networks, with effectiveness scaling favorably with network size. The method is systematically validated on canonical model networks including scale-free and small-world topologies, and real-world networks representing power grids and brain connectivity patterns. This adversarial framework represents a novel paradigm for synchronization management by introducing deep learning concepts to networked dynamical systems.



## **37. Five Minutes of DDoS Brings down Tor: DDoS Attacks on the Tor Directory Protocol and Mitigations**

cs.CR

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.10755v1) [paper-pdf](http://arxiv.org/pdf/2509.10755v1)

**Authors**: Zhongtang Luo, Jianting Zhang, Akshat Neerati, Aniket Kate

**Abstract**: The Tor network offers network anonymity to its users by routing their traffic through a sequence of relays. A group of nine directory authorities maintains information about all available relay nodes using a distributed directory protocol. We observe that the current protocol makes a steep synchrony assumption, which makes it vulnerable to natural as well as adversarial non-synchronous communication scenarios over the Internet. In this paper, we show that it is possible to cause a failure in the Tor directory protocol by targeting a majority of the authorities for only five minutes using a well-executed distributed denial-of-service (DDoS) attack. We demonstrate this attack in a controlled environment and show that it is cost-effective for as little as \$53.28 per month to disrupt the protocol and to effectively bring down the entire Tor network. To mitigate this problem, we consider the popular partial synchrony assumption for the Tor directory protocol that ensures that the protocol security is hampered even when the network delays are large and unknown. We design a new Tor directory protocol that leverages any standard partial-synchronous consensus protocol to solve this problem, while also proving its security. We have implemented a prototype in Rust, demonstrating comparable performance to the current protocol while resisting similar attacks.



## **38. Feature-Centric Approaches to Android Malware Analysis: A Survey**

cs.CR

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.10709v1) [paper-pdf](http://arxiv.org/pdf/2509.10709v1)

**Authors**: Shama Maganur, Yili Jiang, Jiaqi Huang, Fangtian Zhong

**Abstract**: Sophisticated malware families exploit the openness of the Android platform to infiltrate IoT networks, enabling large-scale disruption, data exfiltration, and denial-of-service attacks. This systematic literature review (SLR) examines cutting-edge approaches to Android malware analysis with direct implications for securing IoT infrastructures. We analyze feature extraction techniques across static, dynamic, hybrid, and graph-based methods, highlighting their trade-offs: static analysis offers efficiency but is easily evaded through obfuscation; dynamic analysis provides stronger resistance to evasive behaviors but incurs high computational costs, often unsuitable for lightweight IoT devices; hybrid approaches balance accuracy with resource considerations; and graph-based methods deliver superior semantic modeling and adversarial robustness. This survey contributes a structured comparison of existing methods, exposes research gaps, and outlines a roadmap for future directions to enhance scalability, adaptability, and long-term security in IoT-driven Android malware detection.



## **39. Privacy-Preserving Decentralized Federated Learning via Explainable Adaptive Differential Privacy**

cs.CR

21 pages

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.10691v1) [paper-pdf](http://arxiv.org/pdf/2509.10691v1)

**Authors**: Fardin Jalil Piran, Zhiling Chen, Yang Zhang, Qianyu Zhou, Jiong Tang, Farhad Imani

**Abstract**: Decentralized federated learning faces privacy risks because model updates can leak data through inference attacks and membership inference, a concern that grows over many client exchanges. Differential privacy offers principled protection by injecting calibrated noise so confidential information remains secure on resource-limited IoT devices. Yet without transparency, black-box training cannot track noise already injected by previous clients and rounds, which forces worst-case additions and harms accuracy. We propose PrivateDFL, an explainable framework that joins hyperdimensional computing with differential privacy and keeps an auditable account of cumulative noise so each client adds only the difference between the required noise and what has already been accumulated. We evaluate on MNIST, ISOLET, and UCI-HAR to span image, signal, and tabular modalities, and we benchmark against transformer-based and deep learning-based baselines trained centrally with Differentially Private Stochastic Gradient Descent (DP-SGD) and Renyi Differential Privacy (RDP). PrivateDFL delivers higher accuracy, lower latency, and lower energy across IID and non-IID partitions while preserving formal (epsilon, delta) guarantees and operating without a central server. For example, under non-IID partitions, PrivateDFL achieves 24.42% higher accuracy than the Vision Transformer on MNIST while using about 10x less training time, 76x lower inference latency, and 11x less energy, and on ISOLET it exceeds Transformer accuracy by more than 80% with roughly 10x less training time, 40x lower inference latency, and 36x less training energy. Future work will extend the explainable accounting to adversarial clients and adaptive topologies with heterogeneous privacy budgets.



## **40. Multi-Agent Systems Execute Arbitrary Malicious Code**

cs.CR

33 pages, 5 figures, 7 tables

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2503.12188v2) [paper-pdf](http://arxiv.org/pdf/2503.12188v2)

**Authors**: Harold Triedman, Rishi Jha, Vitaly Shmatikov

**Abstract**: Multi-agent systems coordinate LLM-based agents to perform tasks on users' behalf. In real-world applications, multi-agent systems will inevitably interact with untrusted inputs, such as malicious Web content, files, email attachments, and more.   Using several recently proposed multi-agent frameworks as concrete examples, we demonstrate that adversarial content can hijack control and communication within the system to invoke unsafe agents and functionalities. This results in a complete security breach, up to execution of arbitrary malicious code on the user's device or exfiltration of sensitive data from the user's containerized environment. For example, when agents are instantiated with GPT-4o, Web-based attacks successfully cause the multi-agent system execute arbitrary malicious code in 58-90\% of trials (depending on the orchestrator). In some model-orchestrator configurations, the attack success rate is 100\%. We also demonstrate that these attacks succeed even if individual agents are not susceptible to direct or indirect prompt injection, and even if they refuse to perform harmful actions. We hope that these results will motivate development of trust and security models for multi-agent systems before they are widely deployed.



## **41. SAIF: Sparse Adversarial and Imperceptible Attack Framework**

cs.CV

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2212.07495v4) [paper-pdf](http://arxiv.org/pdf/2212.07495v4)

**Authors**: Tooba Imtiaz, Morgan Kohler, Jared Miller, Zifeng Wang, Masih Eskandar, Mario Sznaier, Octavia Camps, Jennifer Dy

**Abstract**: Adversarial attacks hamper the decision-making ability of neural networks by perturbing the input signal. The addition of calculated small distortion to images, for instance, can deceive a well-trained image classification network. In this work, we propose a novel attack technique called Sparse Adversarial and Interpretable Attack Framework (SAIF). Specifically, we design imperceptible attacks that contain low-magnitude perturbations at a small number of pixels and leverage these sparse attacks to reveal the vulnerability of classifiers. We use the Frank-Wolfe (conditional gradient) algorithm to simultaneously optimize the attack perturbations for bounded magnitude and sparsity with $O(1/\sqrt{T})$ convergence. Empirical results show that SAIF computes highly imperceptible and interpretable adversarial examples, and outperforms state-of-the-art sparse attack methods on the ImageNet dataset.



## **42. Attacking Attention of Foundation Models Disrupts Downstream Tasks**

cs.CR

Paper published at CVPR 2025 Workshop Advml

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2506.05394v3) [paper-pdf](http://arxiv.org/pdf/2506.05394v3)

**Authors**: Hondamunige Prasanna Silva, Federico Becattini, Lorenzo Seidenari

**Abstract**: Foundation models represent the most prominent and recent paradigm shift in artificial intelligence. Foundation models are large models, trained on broad data that deliver high accuracy in many downstream tasks, often without fine-tuning. For this reason, models such as CLIP , DINO or Vision Transfomers (ViT), are becoming the bedrock of many industrial AI-powered applications. However, the reliance on pre-trained foundation models also introduces significant security concerns, as these models are vulnerable to adversarial attacks. Such attacks involve deliberately crafted inputs designed to deceive AI systems, jeopardizing their reliability. This paper studies the vulnerabilities of vision foundation models, focusing specifically on CLIP and ViTs, and explores the transferability of adversarial attacks to downstream tasks. We introduce a novel attack, targeting the structure of transformer-based architectures in a task-agnostic fashion. We demonstrate the effectiveness of our attack on several downstream tasks: classification, captioning, image/text retrieval, segmentation and depth estimation. Code available at:https://github.com/HondamunigePrasannaSilva/attack-attention



## **43. Immunizing Images from Text to Image Editing via Adversarial Cross-Attention**

cs.CV

Accepted as Regular Paper at ACM Multimedia 2025

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.10359v1) [paper-pdf](http://arxiv.org/pdf/2509.10359v1)

**Authors**: Matteo Trippodo, Federico Becattini, Lorenzo Seidenari

**Abstract**: Recent advances in text-based image editing have enabled fine-grained manipulation of visual content guided by natural language. However, such methods are susceptible to adversarial attacks. In this work, we propose a novel attack that targets the visual component of editing methods. We introduce Attention Attack, which disrupts the cross-attention between a textual prompt and the visual representation of the image by using an automatically generated caption of the source image as a proxy for the edit prompt. This breaks the alignment between the contents of the image and their textual description, without requiring knowledge of the editing method or the editing prompt. Reflecting on the reliability of existing metrics for immunization success, we propose two novel evaluation strategies: Caption Similarity, which quantifies semantic consistency between original and adversarial edits, and semantic Intersection over Union (IoU), which measures spatial layout disruption via segmentation masks. Experiments conducted on the TEDBench++ benchmark demonstrate that our attack significantly degrades editing performance while remaining imperceptible.



## **44. Between a Rock and a Hard Place: Exploiting Ethical Reasoning to Jailbreak LLMs**

cs.CR

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.05367v2) [paper-pdf](http://arxiv.org/pdf/2509.05367v2)

**Authors**: Shei Pern Chua, Zhen Leng Thai, Teh Kai Jun, Xiao Li, Xiaolin Hu

**Abstract**: Large language models (LLMs) have undergone safety alignment efforts to mitigate harmful outputs. However, as LLMs become more sophisticated in reasoning, their intelligence may introduce new security risks. While traditional jailbreak attacks relied on singlestep attacks, multi-turn jailbreak strategies that adapt dynamically to context remain underexplored. In this work, we introduce TRIAL (Trolley-problem Reasoning for Interactive Attack Logic), a framework that leverages LLMs ethical reasoning to bypass their safeguards. TRIAL embeds adversarial goals within ethical dilemmas modeled on the trolley problem. TRIAL demonstrates high jailbreak success rates towards both open and close-source models. Our findings underscore a fundamental limitation in AI safety: as models gain advanced reasoning abilities, the nature of their alignment may inadvertently allow for more covert security vulnerabilities to be exploited. TRIAL raises an urgent need in reevaluating safety alignment oversight strategies, as current safeguards may prove insufficient against context-aware adversarial attack.



## **45. Analyzing the Impact of Adversarial Examples on Explainable Machine Learning**

cs.LG

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2307.08327v2) [paper-pdf](http://arxiv.org/pdf/2307.08327v2)

**Authors**: Prathyusha Devabhakthini, Sasmita Parida, Raj Mani Shukla, Suvendu Chandan Nayak, Tapadhir Das

**Abstract**: Adversarial attacks are a type of attack on machine learning models where an attacker deliberately modifies the inputs to cause the model to make incorrect predictions. Adversarial attacks can have serious consequences, particularly in applications such as autonomous vehicles, medical diagnosis, and security systems. Work on the vulnerability of deep learning models to adversarial attacks has shown that it is very easy to make samples that make a model predict things that it doesn't want to. In this work, we analyze the impact of model interpretability due to adversarial attacks on text classification problems. We develop an ML-based classification model for text data. Then, we introduce the adversarial perturbations on the text data to understand the classification performance after the attack. Subsequently, we analyze and interpret the model's explainability before and after the attack



## **46. Privacy Risks of LLM-Empowered Recommender Systems: An Inversion Attack Perspective**

cs.IR

Accepted at ACM RecSys 2025 (10 pages, 4 figures)

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2508.03703v2) [paper-pdf](http://arxiv.org/pdf/2508.03703v2)

**Authors**: Yubo Wang, Min Tang, Nuo Shen, Shujie Cui, Weiqing Wang

**Abstract**: The large language model (LLM) powered recommendation paradigm has been proposed to address the limitations of traditional recommender systems, which often struggle to handle cold start users or items with new IDs. Despite its effectiveness, this study uncovers that LLM empowered recommender systems are vulnerable to reconstruction attacks that can expose both system and user privacy. To examine this threat, we present the first systematic study on inversion attacks targeting LLM empowered recommender systems, where adversaries attempt to reconstruct original prompts that contain personal preferences, interaction histories, and demographic attributes by exploiting the output logits of recommendation models. We reproduce the vec2text framework and optimize it using our proposed method called Similarity Guided Refinement, enabling more accurate reconstruction of textual prompts from model generated logits. Extensive experiments across two domains (movies and books) and two representative LLM based recommendation models demonstrate that our method achieves high fidelity reconstructions. Specifically, we can recover nearly 65 percent of the user interacted items and correctly infer age and gender in 87 percent of the cases. The experiments also reveal that privacy leakage is largely insensitive to the victim model's performance but highly dependent on domain consistency and prompt complexity. These findings expose critical privacy vulnerabilities in LLM empowered recommender systems.



## **47. AdvI2I: Adversarial Image Attack on Image-to-Image Diffusion models**

cs.CV

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2410.21471v3) [paper-pdf](http://arxiv.org/pdf/2410.21471v3)

**Authors**: Yaopei Zeng, Yuanpu Cao, Bochuan Cao, Yurui Chang, Jinghui Chen, Lu Lin

**Abstract**: Recent advances in diffusion models have significantly enhanced the quality of image synthesis, yet they have also introduced serious safety concerns, particularly the generation of Not Safe for Work (NSFW) content. Previous research has demonstrated that adversarial prompts can be used to generate NSFW content. However, such adversarial text prompts are often easily detectable by text-based filters, limiting their efficacy. In this paper, we expose a previously overlooked vulnerability: adversarial image attacks targeting Image-to-Image (I2I) diffusion models. We propose AdvI2I, a novel framework that manipulates input images to induce diffusion models to generate NSFW content. By optimizing a generator to craft adversarial images, AdvI2I circumvents existing defense mechanisms, such as Safe Latent Diffusion (SLD), without altering the text prompts. Furthermore, we introduce AdvI2I-Adaptive, an enhanced version that adapts to potential countermeasures and minimizes the resemblance between adversarial images and NSFW concept embeddings, making the attack more resilient against defenses. Through extensive experiments, we demonstrate that both AdvI2I and AdvI2I-Adaptive can effectively bypass current safeguards, highlighting the urgent need for stronger security measures to address the misuse of I2I diffusion models.



## **48. Target Defense Using a Turret and Mobile Defender Team**

eess.SY

Submitted to IEEE L-CSS and the 2026 ACC

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09777v1) [paper-pdf](http://arxiv.org/pdf/2509.09777v1)

**Authors**: Alexander Von Moll, Dipankar Maity, Meir Pachter, Daigo Shishika, Michael Dorothy

**Abstract**: A scenario is considered wherein a stationary, turn constrained agent (Turret) and a mobile agent (Defender) cooperate to protect the former from an adversarial mobile agent (Attacker). The Attacker wishes to reach the Turret prior to getting captured by either the Defender or Turret, if possible. Meanwhile, the Defender and Turret seek to capture the Attacker as far from the Turret as possible. This scenario is formulated as a differential game and solved using a geometric approach. Necessary and sufficient conditions for the Turret-Defender team winning and the Attacker winning are given. In the case of the Turret-Defender team winning equilibrium strategies for the min max terminal distance of the Attacker to the Turret are given. Three cases arise corresponding to solo capture by the Defender, solo capture by the Turret, and capture simultaneously by both Turret and Defender.



## **49. Steering MoE LLMs via Expert (De)Activation**

cs.CL

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09660v1) [paper-pdf](http://arxiv.org/pdf/2509.09660v1)

**Authors**: Mohsen Fayyaz, Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Ryan Rossi, Trung Bui, Hinrich Schütze, Nanyun Peng

**Abstract**: Mixture-of-Experts (MoE) in Large Language Models (LLMs) routes each token through a subset of specialized Feed-Forward Networks (FFN), known as experts. We present SteerMoE, a framework for steering MoE models by detecting and controlling behavior-linked experts. Our detection method identifies experts with distinct activation patterns across paired inputs exhibiting contrasting behaviors. By selectively (de)activating such experts during inference, we control behaviors like faithfulness and safety without retraining or modifying weights. Across 11 benchmarks and 6 LLMs, our steering raises safety by up to +20% and faithfulness by +27%. In adversarial attack mode, it drops safety by -41% alone, and -100% when combined with existing jailbreak methods, bypassing all safety guardrails and exposing a new dimension of alignment faking hidden within experts.



## **50. Bitcoin under Volatile Block Rewards: How Mempool Statistics Can Influence Bitcoin Mining**

cs.CR

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2411.11702v3) [paper-pdf](http://arxiv.org/pdf/2411.11702v3)

**Authors**: Roozbeh Sarenche, Alireza Aghabagherloo, Svetla Nikova, Bart Preneel

**Abstract**: The security of Bitcoin protocols is deeply dependent on the incentives provided to miners, which come from a combination of block rewards and transaction fees. As Bitcoin experiences more halving events, the protocol reward converges to zero, making transaction fees the primary source of miner rewards. This shift in Bitcoin's incentivization mechanism, which introduces volatility into block rewards, leads to the emergence of new security threats or intensifies existing ones. Previous security analyses of Bitcoin have either considered a fixed block reward model or a highly simplified volatile model, overlooking the complexities of Bitcoin's mempool behavior.   This paper presents a reinforcement learning-based tool to develop mining strategies under a more realistic volatile model. We employ the Asynchronous Advantage Actor-Critic (A3C) algorithm, which efficiently handles dynamic environments, such as the Bitcoin mempool, to derive near-optimal mining strategies when interacting with an environment that models the complexity of the Bitcoin mempool. This tool enables the analysis of adversarial mining strategies, such as selfish mining and undercutting, both before and after difficulty adjustments, providing insights into the effects of mining attacks in both the short and long term.   We revisit the Bitcoin security threshold presented in the WeRLman paper and demonstrate that the implicit predictability of valuable transaction arrivals in this model leads to an underestimation of the reported threshold. Additionally, we show that, while adversarial strategies like selfish mining under the fixed reward model incur an initial loss period of at least two weeks, the transition toward a transaction-fee era incentivizes mining pools to abandon honest mining for immediate profits. This incentive is expected to become more significant as the protocol reward approaches zero in the future.



