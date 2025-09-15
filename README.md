# Latest Adversarial Attack Papers
**update at 2025-09-15 14:35:19**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Attacking Attention of Foundation Models Disrupts Downstream Tasks**

cs.CR

Paper published at CVPR 2025 Workshop Advml

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2506.05394v3) [paper-pdf](http://arxiv.org/pdf/2506.05394v3)

**Authors**: Hondamunige Prasanna Silva, Federico Becattini, Lorenzo Seidenari

**Abstract**: Foundation models represent the most prominent and recent paradigm shift in artificial intelligence. Foundation models are large models, trained on broad data that deliver high accuracy in many downstream tasks, often without fine-tuning. For this reason, models such as CLIP , DINO or Vision Transfomers (ViT), are becoming the bedrock of many industrial AI-powered applications. However, the reliance on pre-trained foundation models also introduces significant security concerns, as these models are vulnerable to adversarial attacks. Such attacks involve deliberately crafted inputs designed to deceive AI systems, jeopardizing their reliability. This paper studies the vulnerabilities of vision foundation models, focusing specifically on CLIP and ViTs, and explores the transferability of adversarial attacks to downstream tasks. We introduce a novel attack, targeting the structure of transformer-based architectures in a task-agnostic fashion. We demonstrate the effectiveness of our attack on several downstream tasks: classification, captioning, image/text retrieval, segmentation and depth estimation. Code available at:https://github.com/HondamunigePrasannaSilva/attack-attention



## **2. Immunizing Images from Text to Image Editing via Adversarial Cross-Attention**

cs.CV

Accepted as Regular Paper at ACM Multimedia 2025

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.10359v1) [paper-pdf](http://arxiv.org/pdf/2509.10359v1)

**Authors**: Matteo Trippodo, Federico Becattini, Lorenzo Seidenari

**Abstract**: Recent advances in text-based image editing have enabled fine-grained manipulation of visual content guided by natural language. However, such methods are susceptible to adversarial attacks. In this work, we propose a novel attack that targets the visual component of editing methods. We introduce Attention Attack, which disrupts the cross-attention between a textual prompt and the visual representation of the image by using an automatically generated caption of the source image as a proxy for the edit prompt. This breaks the alignment between the contents of the image and their textual description, without requiring knowledge of the editing method or the editing prompt. Reflecting on the reliability of existing metrics for immunization success, we propose two novel evaluation strategies: Caption Similarity, which quantifies semantic consistency between original and adversarial edits, and semantic Intersection over Union (IoU), which measures spatial layout disruption via segmentation masks. Experiments conducted on the TEDBench++ benchmark demonstrate that our attack significantly degrades editing performance while remaining imperceptible.



## **3. Between a Rock and a Hard Place: Exploiting Ethical Reasoning to Jailbreak LLMs**

cs.CR

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.05367v2) [paper-pdf](http://arxiv.org/pdf/2509.05367v2)

**Authors**: Shei Pern Chua, Zhen Leng Thai, Teh Kai Jun, Xiao Li, Xiaolin Hu

**Abstract**: Large language models (LLMs) have undergone safety alignment efforts to mitigate harmful outputs. However, as LLMs become more sophisticated in reasoning, their intelligence may introduce new security risks. While traditional jailbreak attacks relied on singlestep attacks, multi-turn jailbreak strategies that adapt dynamically to context remain underexplored. In this work, we introduce TRIAL (Trolley-problem Reasoning for Interactive Attack Logic), a framework that leverages LLMs ethical reasoning to bypass their safeguards. TRIAL embeds adversarial goals within ethical dilemmas modeled on the trolley problem. TRIAL demonstrates high jailbreak success rates towards both open and close-source models. Our findings underscore a fundamental limitation in AI safety: as models gain advanced reasoning abilities, the nature of their alignment may inadvertently allow for more covert security vulnerabilities to be exploited. TRIAL raises an urgent need in reevaluating safety alignment oversight strategies, as current safeguards may prove insufficient against context-aware adversarial attack.



## **4. Analyzing the Impact of Adversarial Examples on Explainable Machine Learning**

cs.LG

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2307.08327v2) [paper-pdf](http://arxiv.org/pdf/2307.08327v2)

**Authors**: Prathyusha Devabhakthini, Sasmita Parida, Raj Mani Shukla, Suvendu Chandan Nayak, Tapadhir Das

**Abstract**: Adversarial attacks are a type of attack on machine learning models where an attacker deliberately modifies the inputs to cause the model to make incorrect predictions. Adversarial attacks can have serious consequences, particularly in applications such as autonomous vehicles, medical diagnosis, and security systems. Work on the vulnerability of deep learning models to adversarial attacks has shown that it is very easy to make samples that make a model predict things that it doesn't want to. In this work, we analyze the impact of model interpretability due to adversarial attacks on text classification problems. We develop an ML-based classification model for text data. Then, we introduce the adversarial perturbations on the text data to understand the classification performance after the attack. Subsequently, we analyze and interpret the model's explainability before and after the attack



## **5. Privacy Risks of LLM-Empowered Recommender Systems: An Inversion Attack Perspective**

cs.IR

Accepted at ACM RecSys 2025 (10 pages, 4 figures)

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2508.03703v2) [paper-pdf](http://arxiv.org/pdf/2508.03703v2)

**Authors**: Yubo Wang, Min Tang, Nuo Shen, Shujie Cui, Weiqing Wang

**Abstract**: The large language model (LLM) powered recommendation paradigm has been proposed to address the limitations of traditional recommender systems, which often struggle to handle cold start users or items with new IDs. Despite its effectiveness, this study uncovers that LLM empowered recommender systems are vulnerable to reconstruction attacks that can expose both system and user privacy. To examine this threat, we present the first systematic study on inversion attacks targeting LLM empowered recommender systems, where adversaries attempt to reconstruct original prompts that contain personal preferences, interaction histories, and demographic attributes by exploiting the output logits of recommendation models. We reproduce the vec2text framework and optimize it using our proposed method called Similarity Guided Refinement, enabling more accurate reconstruction of textual prompts from model generated logits. Extensive experiments across two domains (movies and books) and two representative LLM based recommendation models demonstrate that our method achieves high fidelity reconstructions. Specifically, we can recover nearly 65 percent of the user interacted items and correctly infer age and gender in 87 percent of the cases. The experiments also reveal that privacy leakage is largely insensitive to the victim model's performance but highly dependent on domain consistency and prompt complexity. These findings expose critical privacy vulnerabilities in LLM empowered recommender systems.



## **6. AdvI2I: Adversarial Image Attack on Image-to-Image Diffusion models**

cs.CV

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2410.21471v3) [paper-pdf](http://arxiv.org/pdf/2410.21471v3)

**Authors**: Yaopei Zeng, Yuanpu Cao, Bochuan Cao, Yurui Chang, Jinghui Chen, Lu Lin

**Abstract**: Recent advances in diffusion models have significantly enhanced the quality of image synthesis, yet they have also introduced serious safety concerns, particularly the generation of Not Safe for Work (NSFW) content. Previous research has demonstrated that adversarial prompts can be used to generate NSFW content. However, such adversarial text prompts are often easily detectable by text-based filters, limiting their efficacy. In this paper, we expose a previously overlooked vulnerability: adversarial image attacks targeting Image-to-Image (I2I) diffusion models. We propose AdvI2I, a novel framework that manipulates input images to induce diffusion models to generate NSFW content. By optimizing a generator to craft adversarial images, AdvI2I circumvents existing defense mechanisms, such as Safe Latent Diffusion (SLD), without altering the text prompts. Furthermore, we introduce AdvI2I-Adaptive, an enhanced version that adapts to potential countermeasures and minimizes the resemblance between adversarial images and NSFW concept embeddings, making the attack more resilient against defenses. Through extensive experiments, we demonstrate that both AdvI2I and AdvI2I-Adaptive can effectively bypass current safeguards, highlighting the urgent need for stronger security measures to address the misuse of I2I diffusion models.



## **7. Target Defense Using a Turret and Mobile Defender Team**

eess.SY

Submitted to IEEE L-CSS and the 2026 ACC

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09777v1) [paper-pdf](http://arxiv.org/pdf/2509.09777v1)

**Authors**: Alexander Von Moll, Dipankar Maity, Meir Pachter, Daigo Shishika, Michael Dorothy

**Abstract**: A scenario is considered wherein a stationary, turn constrained agent (Turret) and a mobile agent (Defender) cooperate to protect the former from an adversarial mobile agent (Attacker). The Attacker wishes to reach the Turret prior to getting captured by either the Defender or Turret, if possible. Meanwhile, the Defender and Turret seek to capture the Attacker as far from the Turret as possible. This scenario is formulated as a differential game and solved using a geometric approach. Necessary and sufficient conditions for the Turret-Defender team winning and the Attacker winning are given. In the case of the Turret-Defender team winning equilibrium strategies for the min max terminal distance of the Attacker to the Turret are given. Three cases arise corresponding to solo capture by the Defender, solo capture by the Turret, and capture simultaneously by both Turret and Defender.



## **8. Steering MoE LLMs via Expert (De)Activation**

cs.CL

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09660v1) [paper-pdf](http://arxiv.org/pdf/2509.09660v1)

**Authors**: Mohsen Fayyaz, Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Ryan Rossi, Trung Bui, Hinrich Schütze, Nanyun Peng

**Abstract**: Mixture-of-Experts (MoE) in Large Language Models (LLMs) routes each token through a subset of specialized Feed-Forward Networks (FFN), known as experts. We present SteerMoE, a framework for steering MoE models by detecting and controlling behavior-linked experts. Our detection method identifies experts with distinct activation patterns across paired inputs exhibiting contrasting behaviors. By selectively (de)activating such experts during inference, we control behaviors like faithfulness and safety without retraining or modifying weights. Across 11 benchmarks and 6 LLMs, our steering raises safety by up to +20% and faithfulness by +27%. In adversarial attack mode, it drops safety by -41% alone, and -100% when combined with existing jailbreak methods, bypassing all safety guardrails and exposing a new dimension of alignment faking hidden within experts.



## **9. Bitcoin under Volatile Block Rewards: How Mempool Statistics Can Influence Bitcoin Mining**

cs.CR

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2411.11702v3) [paper-pdf](http://arxiv.org/pdf/2411.11702v3)

**Authors**: Roozbeh Sarenche, Alireza Aghabagherloo, Svetla Nikova, Bart Preneel

**Abstract**: The security of Bitcoin protocols is deeply dependent on the incentives provided to miners, which come from a combination of block rewards and transaction fees. As Bitcoin experiences more halving events, the protocol reward converges to zero, making transaction fees the primary source of miner rewards. This shift in Bitcoin's incentivization mechanism, which introduces volatility into block rewards, leads to the emergence of new security threats or intensifies existing ones. Previous security analyses of Bitcoin have either considered a fixed block reward model or a highly simplified volatile model, overlooking the complexities of Bitcoin's mempool behavior.   This paper presents a reinforcement learning-based tool to develop mining strategies under a more realistic volatile model. We employ the Asynchronous Advantage Actor-Critic (A3C) algorithm, which efficiently handles dynamic environments, such as the Bitcoin mempool, to derive near-optimal mining strategies when interacting with an environment that models the complexity of the Bitcoin mempool. This tool enables the analysis of adversarial mining strategies, such as selfish mining and undercutting, both before and after difficulty adjustments, providing insights into the effects of mining attacks in both the short and long term.   We revisit the Bitcoin security threshold presented in the WeRLman paper and demonstrate that the implicit predictability of valuable transaction arrivals in this model leads to an underestimation of the reported threshold. Additionally, we show that, while adversarial strategies like selfish mining under the fixed reward model incur an initial loss period of at least two weeks, the transition toward a transaction-fee era incentivizes mining pools to abandon honest mining for immediate profits. This incentive is expected to become more significant as the protocol reward approaches zero in the future.



## **10. ProDiGy: Proximity- and Dissimilarity-Based Byzantine-Robust Federated Learning**

cs.LG

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09534v1) [paper-pdf](http://arxiv.org/pdf/2509.09534v1)

**Authors**: Sena Ergisi, Luis Maßny, Rawad Bitar

**Abstract**: Federated Learning (FL) emerged as a widely studied paradigm for distributed learning. Despite its many advantages, FL remains vulnerable to adversarial attacks, especially under data heterogeneity. We propose a new Byzantine-robust FL algorithm called ProDiGy. The key novelty lies in evaluating the client gradients using a joint dual scoring system based on the gradients' proximity and dissimilarity. We demonstrate through extensive numerical experiments that ProDiGy outperforms existing defenses in various scenarios. In particular, when the clients' data do not follow an IID distribution, while other defense mechanisms fail, ProDiGy maintains strong defense capabilities and model accuracy. These findings highlight the effectiveness of a dual perspective approach that promotes natural similarity among honest clients while detecting suspicious uniformity as a potential indicator of an attack.



## **11. On the Relationship Between Adversarial Robustness and Decision Region in Deep Neural Networks**

cs.LG

10 pages

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2207.03400v2) [paper-pdf](http://arxiv.org/pdf/2207.03400v2)

**Authors**: Seongjin Park, Haedong Jeong, Tair Djanibekov, Giyoung Jeon, Jinseok Seol, Jaesik Choi

**Abstract**: In general, Deep Neural Networks (DNNs) are evaluated by the generalization performance measured on unseen data excluded from the training phase. Along with the development of DNNs, the generalization performance converges to the state-of-the-art and it becomes difficult to evaluate DNNs solely based on this metric. The robustness against adversarial attack has been used as an additional metric to evaluate DNNs by measuring their vulnerability. However, few studies have been performed to analyze the adversarial robustness in terms of the geometry in DNNs. In this work, we perform an empirical study to analyze the internal properties of DNNs that affect model robustness under adversarial attacks. In particular, we propose the novel concept of the Populated Region Set (PRS), where training samples are populated more frequently, to represent the internal properties of DNNs in a practical setting. From systematic experiments with the proposed concept, we provide empirical evidence to validate that a low PRS ratio has a strong relationship with the adversarial robustness of DNNs. We also devise PRS regularizer leveraging the characteristics of PRS to improve the adversarial robustness without adversarial training.



## **12. Over-the-Air Adversarial Attack Detection: from Datasets to Defenses**

eess.AS

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09296v1) [paper-pdf](http://arxiv.org/pdf/2509.09296v1)

**Authors**: Li Wang, Xiaoyan Lei, Haorui He, Lei Wang, Jie Shi, Zhizheng Wu

**Abstract**: Automatic Speaker Verification (ASV) systems can be used for voice-enabled applications for identity verification. However, recent studies have exposed these systems' vulnerabilities to both over-the-line (OTL) and over-the-air (OTA) adversarial attacks. Although various detection methods have been proposed to counter these threats, they have not been thoroughly tested due to the lack of a comprehensive data set. To address this gap, we developed the AdvSV 2.0 dataset, which contains 628k samples with a total duration of 800 hours. This dataset incorporates classical adversarial attack algorithms, ASV systems, and encompasses both OTL and OTA scenarios. Furthermore, we introduce a novel adversarial attack method based on a Neural Replay Simulator (NRS), which enhances the potency of adversarial OTA attacks, thereby presenting a greater threat to ASV systems. To defend against these attacks, we propose CODA-OCC, a contrastive learning approach within the one-class classification framework. Experimental results show that CODA-OCC achieves an EER of 11.2% and an AUC of 0.95 on the AdvSV 2.0 dataset, outperforming several state-of-the-art detection methods.



## **13. Byzantine-Robust Federated Learning Using Generative Adversarial Networks**

cs.CR

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2503.20884v3) [paper-pdf](http://arxiv.org/pdf/2503.20884v3)

**Authors**: Usama Zafar, André M. H. Teixeira, Salman Toor

**Abstract**: Federated learning (FL) enables collaborative model training across distributed clients without sharing raw data, but its robustness is threatened by Byzantine behaviors such as data and model poisoning. Existing defenses face fundamental limitations: robust aggregation rules incur error lower bounds that grow with client heterogeneity, while detection-based methods often rely on heuristics (e.g., a fixed number of malicious clients) or require trusted external datasets for validation. We present a defense framework that addresses these challenges by leveraging a conditional generative adversarial network (cGAN) at the server to synthesize representative data for validating client updates. This approach eliminates reliance on external datasets, adapts to diverse attack strategies, and integrates seamlessly into standard FL workflows. Extensive experiments on benchmark datasets demonstrate that our framework accurately distinguishes malicious from benign clients while maintaining overall model accuracy. Beyond Byzantine robustness, we also examine the representativeness of synthesized data, computational costs of cGAN training, and the transparency and scalability of our approach.



## **14. TESSER: Transfer-Enhancing Adversarial Attacks from Vision Transformers via Spectral and Semantic Regularization**

cs.CV

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2505.19613v2) [paper-pdf](http://arxiv.org/pdf/2505.19613v2)

**Authors**: Amira Guesmi, Bassem Ouni, Muhammad Shafique

**Abstract**: Adversarial transferability remains a critical challenge in evaluating the robustness of deep neural networks. In security-critical applications, transferability enables black-box attacks without access to model internals, making it a key concern for real-world adversarial threat assessment. While Vision Transformers (ViTs) have demonstrated strong adversarial performance, existing attacks often fail to transfer effectively across architectures, especially from ViTs to Convolutional Neural Networks (CNNs) or hybrid models. In this paper, we introduce \textbf{TESSER} -- a novel adversarial attack framework that enhances transferability via two key strategies: (1) \textit{Feature-Sensitive Gradient Scaling (FSGS)}, which modulates gradients based on token-wise importance derived from intermediate feature activations, and (2) \textit{Spectral Smoothness Regularization (SSR)}, which suppresses high-frequency noise in perturbations using a differentiable Gaussian prior. These components work in tandem to generate perturbations that are both semantically meaningful and spectrally smooth. Extensive experiments on ImageNet across 12 diverse architectures demonstrate that TESSER achieves +10.9\% higher attack succes rate (ASR) on CNNs and +7.2\% on ViTs compared to the state-of-the-art Adaptive Token Tuning (ATT) method. Moreover, TESSER significantly improves robustness against defended models, achieving 53.55\% ASR on adversarially trained CNNs. Qualitative analysis shows strong alignment between TESSER's perturbations and salient visual regions identified via Grad-CAM, while frequency-domain analysis reveals a 12\% reduction in high-frequency energy, confirming the effectiveness of spectral regularization.



## **15. IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves**

cs.CV

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2411.00827v5) [paper-pdf](http://arxiv.org/pdf/2411.00827v5)

**Authors**: Ruofan Wang, Juncheng Li, Yixu Wang, Bo Wang, Xiaosen Wang, Yan Teng, Yingchun Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) gain prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks-techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multimodal data has constrained current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which often lack effectiveness and diversity across different contexts. In this paper, we propose IDEATOR, a novel jailbreak method that autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is grounded in the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR leverages a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Extensive experiments demonstrate IDEATOR's high effectiveness and transferability, achieving a 94% attack success rate (ASR) in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high ASRs of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Chameleon, respectively. Building on IDEATOR's strong transferability and automated process, we introduce the VLJailbreakBench, a safety benchmark comprising 3,654 multimodal jailbreak samples. Our benchmark results on 11 recently released VLMs reveal significant gaps in safety alignment. For instance, our challenge set achieves ASRs of 46.31% on GPT-4o and 19.65% on Claude-3.5-Sonnet, underscoring the urgent need for stronger defenses.VLJailbreakBench is publicly available at https://roywang021.github.io/VLJailbreakBench.



## **16. Character-Level Perturbations Disrupt LLM Watermarks**

cs.CR

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09112v1) [paper-pdf](http://arxiv.org/pdf/2509.09112v1)

**Authors**: Zhaoxi Zhang, Xiaomei Zhang, Yanjun Zhang, He Zhang, Shirui Pan, Bo Liu, Asif Qumer Gill, Leo Yu Zhang

**Abstract**: Large Language Model (LLM) watermarking embeds detectable signals into generated text for copyright protection, misuse prevention, and content detection. While prior studies evaluate robustness using watermark removal attacks, these methods are often suboptimal, creating the misconception that effective removal requires large perturbations or powerful adversaries.   To bridge the gap, we first formalize the system model for LLM watermark, and characterize two realistic threat models constrained on limited access to the watermark detector. We then analyze how different types of perturbation vary in their attack range, i.e., the number of tokens they can affect with a single edit. We observe that character-level perturbations (e.g., typos, swaps, deletions, homoglyphs) can influence multiple tokens simultaneously by disrupting the tokenization process. We demonstrate that character-level perturbations are significantly more effective for watermark removal under the most restrictive threat model. We further propose guided removal attacks based on the Genetic Algorithm (GA) that uses a reference detector for optimization. Under a practical threat model with limited black-box queries to the watermark detector, our method demonstrates strong removal performance. Experiments confirm the superiority of character-level perturbations and the effectiveness of the GA in removing watermarks under realistic constraints. Additionally, we argue there is an adversarial dilemma when considering potential defenses: any fixed defense can be bypassed by a suitable perturbation strategy. Motivated by this principle, we propose an adaptive compound character-level attack. Experimental results show that this approach can effectively defeat the defenses. Our findings highlight significant vulnerabilities in existing LLM watermark schemes and underline the urgency for the development of new robust mechanisms.



## **17. AdvReal: Physical Adversarial Patch Generation Framework for Security Evaluation of Object Detection Systems**

cs.CV

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2505.16402v2) [paper-pdf](http://arxiv.org/pdf/2505.16402v2)

**Authors**: Yuanhao Huang, Yilong Ren, Jinlei Wang, Lujia Huo, Xuesong Bai, Jinchuan Zhang, Haiyan Yu

**Abstract**: Autonomous vehicles are typical complex intelligent systems with artificial intelligence at their core. However, perception methods based on deep learning are extremely vulnerable to adversarial samples, resulting in security accidents. How to generate effective adversarial examples in the physical world and evaluate object detection systems is a huge challenge. In this study, we propose a unified joint adversarial training framework for both 2D and 3D domains, which simultaneously optimizes texture maps in 2D image and 3D mesh spaces to better address intra-class diversity and real-world environmental variations. The framework includes a novel realistic enhanced adversarial module, with time-space and relighting mapping pipeline that adjusts illumination consistency between adversarial patches and target garments under varied viewpoints. Building upon this, we develop a realism enhancement mechanism that incorporates non-rigid deformation modeling and texture remapping to ensure alignment with the human body's non-rigid surfaces in 3D scenes. Extensive experiment results in digital and physical environments demonstrate that the adversarial textures generated by our method can effectively mislead the target detection model. Specifically, our method achieves an average attack success rate (ASR) of 70.13% on YOLOv12 in physical scenarios, significantly outperforming existing methods such as T-SEA (21.65%) and AdvTexture (19.70%). Moreover, the proposed method maintains stable ASR across multiple viewpoints and distances, with an average attack success rate exceeding 90% under both frontal and oblique views at a distance of 4 meters. This confirms the method's strong robustness and transferability under multi-angle attacks, varying lighting conditions, and real-world distances. The demo video and code can be obtained at https://github.com/Huangyh98/AdvReal.git.



## **18. QubitHammer: Remotely Inducing Qubit State Change on Superconducting Quantum Computers**

quant-ph

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2504.07875v2) [paper-pdf](http://arxiv.org/pdf/2504.07875v2)

**Authors**: Yizhuo Tan, Navnil Choudhury, Kanad Basu, Jakub Szefer

**Abstract**: To address the rapidly growing demand for cloud-based quantum computing, various researchers are proposing shifting from the existing single-tenant model to a multi-tenant model that expands resource utilization and improves accessibility. However, while multi-tenancy enables multiple users to access the same quantum computer, it introduces potential for security and reliability vulnerabilities. It therefore becomes important to investigate these vulnerabilities, especially considering realistic attackers who operate without elevated privileges relative to ordinary users. To address this research need, this paper presents and evaluates QubitHammer, the first attack to demonstrate that an adversary can remotely induce unauthorized changes to a victim's quantum circuit's qubit's state within a multi-tenant model by using custom qubit control pulses that are generated within constraints of the public interfaces and without elevated privileges. Through extensive evaluation on real-world superconducting devices from IBM and Rigetti, this work demonstrates that QubitHammer allows an adversary to significantly change the output distribution of a victim quantum circuit. In the experimentation, variational distance is used to evaluate the magnitude of the changes, and variational distance as high as 0.938 is observed. Cross-platform analysis of QubitHammer on a number of quantum computing devices exposes a fundamental susceptibility in superconducting hardware. Further, QubitHammer was also found to evade all currently proposed defenses aimed at ensuring reliable execution in multi-tenant superconducting quantum systems.



## **19. Combating Falsification of Speech Videos with Live Optical Signatures (Extended Version)**

cs.CV

In Proceedings of the 2025 ACM SIGSAC Conference on Computer and  Communications Security (CCS '25). October 13 - 17, 2025, Taipei, Taiwan.  ACM, New York, NY, USA. 19 pages

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2504.21846v2) [paper-pdf](http://arxiv.org/pdf/2504.21846v2)

**Authors**: Hadleigh Schwartz, Xiaofeng Yan, Charles J. Carver, Xia Zhou

**Abstract**: High-profile speech videos are prime targets for falsification, owing to their accessibility and influence. This work proposes VeriLight, a low-overhead and unobtrusive system for protecting speech videos from visual manipulations of speaker identity and lip and facial motion. Unlike the predominant purely digital falsification detection methods, VeriLight creates dynamic physical signatures at the event site and embeds them into all video recordings via imperceptible modulated light. These physical signatures encode semantically-meaningful features unique to the speech event, including the speaker's identity and facial motion, and are cryptographically-secured to prevent spoofing. The signatures can be extracted from any video downstream and validated against the portrayed speech content to check its integrity. Key elements of VeriLight include (1) a framework for generating extremely compact (i.e., 150-bit), pose-invariant speech video features, based on locality-sensitive hashing; and (2) an optical modulation scheme that embeds $>$200 bps into video while remaining imperceptible both in video and live. Experiments on extensive video datasets show VeriLight achieves AUCs $\geq$ 0.99 and a true positive rate of 100% in detecting falsified videos. Further, VeriLight is highly robust across recording conditions, video post-processing techniques, and white-box adversarial attacks on its feature extraction methods. A demonstration of VeriLight is available at https://mobilex.cs.columbia.edu/verilight.



## **20. Quantum Error Correction in Adversarial Regimes**

quant-ph

19 pages, no figure

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.08943v1) [paper-pdf](http://arxiv.org/pdf/2509.08943v1)

**Authors**: Rahul Arvind, Nikhil Bansal, Dax Enshan Koh, Tobias Haug, Kishor Bharti

**Abstract**: In adversarial settings, where attackers can deliberately and strategically corrupt quantum data, standard quantum error correction reaches its limits. It can only correct up to half the code distance and must output a unique answer. Quantum list decoding offers a promising alternative. By allowing the decoder to output a short list of possible errors, it becomes possible to tolerate far more errors, even under worst-case noise. But two fundamental questions remain: which quantum codes support list decoding, and can we design decoding schemes that are secure against efficient, computationally bounded adversaries? In this work, we answer both. To identify which codes are list-decodable, we provide a generalized version of the Knill-Laflamme conditions. Then, using tools from quantum cryptography, we build an unambiguous list decoding protocol based on pseudorandom unitaries. Our scheme is secure against any quantum polynomial-time adversary, even across multiple decoding attempts, in contrast to previous schemes. Our approach connects coding theory with complexity-based quantum cryptography, paving the way for secure quantum information processing in adversarial settings.



## **21. Corruption-Tolerant Asynchronous Q-Learning with Near-Optimal Rates**

cs.LG

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.08933v1) [paper-pdf](http://arxiv.org/pdf/2509.08933v1)

**Authors**: Sreejeet Maity, Aritra Mitra

**Abstract**: We consider the problem of learning the optimal policy in a discounted, infinite-horizon reinforcement learning (RL) setting where the reward signal is subject to adversarial corruption. Such corruption, which may arise from extreme noise, sensor faults, or malicious attacks, can severely degrade the performance of classical algorithms such as Q-learning. To address this challenge, we propose a new provably robust variant of the Q-learning algorithm that operates effectively even when a fraction of the observed rewards are arbitrarily perturbed by an adversary. Under the asynchronous sampling model with time-correlated data, we establish that despite adversarial corruption, the finite-time convergence rate of our algorithm matches that of existing results for the non-adversarial case, up to an additive term proportional to the fraction of corrupted samples. Moreover, we derive an information-theoretic lower bound revealing that the additive corruption term in our upper bounds is unavoidable.   Next, we propose a variant of our algorithm that requires no prior knowledge of the statistics of the true reward distributions. The analysis of this setting is particularly challenging and is enabled by carefully exploiting a refined Azuma-Hoeffding inequality for almost-martingales, a technical tool that might be of independent interest. Collectively, our contributions provide the first finite-time robustness guarantees for asynchronous Q-learning, bridging a significant gap in robust RL.



## **22. Adversarial Attacks Against Automated Fact-Checking: A Survey**

cs.CL

Accepted to the Main Conference of EMNLP 2025. Resources are  available at  https://github.com/FanzhenLiu/Awesome-Automated-Fact-Checking-Attacks

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.08463v1) [paper-pdf](http://arxiv.org/pdf/2509.08463v1)

**Authors**: Fanzhen Liu, Alsharif Abuadbba, Kristen Moore, Surya Nepal, Cecile Paris, Jia Wu, Jian Yang, Quan Z. Sheng

**Abstract**: In an era where misinformation spreads freely, fact-checking (FC) plays a crucial role in verifying claims and promoting reliable information. While automated fact-checking (AFC) has advanced significantly, existing systems remain vulnerable to adversarial attacks that manipulate or generate claims, evidence, or claim-evidence pairs. These attacks can distort the truth, mislead decision-makers, and ultimately undermine the reliability of FC models. Despite growing research interest in adversarial attacks against AFC systems, a comprehensive, holistic overview of key challenges remains lacking. These challenges include understanding attack strategies, assessing the resilience of current models, and identifying ways to enhance robustness. This survey provides the first in-depth review of adversarial attacks targeting FC, categorizing existing attack methodologies and evaluating their impact on AFC systems. Additionally, we examine recent advancements in adversary-aware defenses and highlight open research questions that require further exploration. Our findings underscore the urgent need for resilient FC frameworks capable of withstanding adversarial manipulations in pursuit of preserving high verification accuracy.



## **23. Dual-Stage Safe Herding Framework for Adversarial Attacker in Dynamic Environment**

cs.RO

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.08460v1) [paper-pdf](http://arxiv.org/pdf/2509.08460v1)

**Authors**: Wenqing Wang, Ye Zhang, Haoyu Li, Jingyu Wang

**Abstract**: Recent advances in robotics have enabled the widespread deployment of autonomous robotic systems in complex operational environments, presenting both unprecedented opportunities and significant security problems. Traditional shepherding approaches based on fixed formations are often ineffective or risky in urban and obstacle-rich scenarios, especially when facing adversarial agents with unknown and adaptive behaviors. This paper addresses this challenge as an extended herding problem, where defensive robotic systems must safely guide adversarial agents with unknown strategies away from protected areas and into predetermined safe regions, while maintaining collision-free navigation in dynamic environments. We propose a hierarchical hybrid framework based on reach-avoid game theory and local motion planning, incorporating a virtual containment boundary and event-triggered pursuit mechanisms to enable scalable and robust multi-agent coordination. Simulation results demonstrate that the proposed approach achieves safe and efficient guidance of adversarial agents to designated regions.



## **24. Robustness of Locally Differentially Private Graph Analysis Against Poisoning**

cs.CR

47 Pages, 7 Figures. Published in AsiaCCS 2025. Revised to include  additional lower bounds and experimental results obtained after initial  release

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2210.14376v2) [paper-pdf](http://arxiv.org/pdf/2210.14376v2)

**Authors**: Jacob Imola, Amrita Roy Chowdhury, Kamalika Chaudhuri

**Abstract**: Locally differentially private (LDP) graph analysis allows private analysis on a graph that is distributed across multiple users. However, such computations are vulnerable to data poisoning attacks where an adversary can skew the results by submitting malformed data. In this paper, we formally study the impact of poisoning attacks for graph degree estimation protocols under LDP. We make two key technical contributions. First, we observe LDP makes a protocol more vulnerable to poisoning -- the impact of poisoning is worse when the adversary can directly poison their (noisy) responses, rather than their input data. Second, we observe that graph data is naturally redundant -- every edge is shared between two users. Leveraging this data redundancy, we design robust degree estimation protocols under LDP that can significantly reduce the impact of data poisoning and compute degree estimates with high accuracy. We evaluate our proposed robust degree estimation protocols under poisoning attacks on real-world datasets to demonstrate their efficacy in practice.



## **25. CyberRAG: An Agentic RAG cyber attack classification and reporting tool**

cs.CR

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2507.02424v2) [paper-pdf](http://arxiv.org/pdf/2507.02424v2)

**Authors**: Francesco Blefari, Cristian Cosentino, Francesco Aurelio Pironti, Angelo Furfaro, Fabrizio Marozzo

**Abstract**: Intrusion Detection and Prevention Systems (IDS/IPS) in large enterprises can generate hundreds of thousands of alerts per hour, overwhelming analysts with logs requiring rapidly evolving expertise. Conventional machine-learning detectors reduce alert volume but still yield many false positives, while standard Retrieval-Augmented Generation (RAG) pipelines often retrieve irrelevant context and fail to justify predictions. We present CyberRAG, a modular agent-based RAG framework that delivers real-time classification, explanation, and structured reporting for cyber-attacks. A central LLM agent orchestrates: (i) fine-tuned classifiers specialized by attack family; (ii) tool adapters for enrichment and alerting; and (iii) an iterative retrieval-and-reason loop that queries a domain-specific knowledge base until evidence is relevant and self-consistent. Unlike traditional RAG, CyberRAG adopts an agentic design that enables dynamic control flow and adaptive reasoning. This architecture autonomously refines threat labels and natural-language justifications, reducing false positives and enhancing interpretability. It is also extensible: new attack types can be supported by adding classifiers without retraining the core agent. CyberRAG was evaluated on SQL Injection, XSS, and SSTI, achieving over 94\% accuracy per class and a final classification accuracy of 94.92\% through semantic orchestration. Generated explanations reached 0.94 in BERTScore and 4.9/5 in GPT-4-based expert evaluation, with robustness preserved against adversarial and unseen payloads. These results show that agentic, specialist-oriented RAG can combine high detection accuracy with trustworthy, SOC-ready prose, offering a flexible path toward partially automated cyber-defense workflows.



## **26. Phish-Blitz: Advancing Phishing Detection with Comprehensive Webpage Resource Collection and Visual Integrity Preservation**

cs.CR

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.08375v1) [paper-pdf](http://arxiv.org/pdf/2509.08375v1)

**Authors**: Duddu Hriday, Aditya Kulkarni, Vivek Balachandran, Tamal Das

**Abstract**: Phishing attacks are increasingly prevalent, with adversaries creating deceptive webpages to steal sensitive information. Despite advancements in machine learning and deep learning for phishing detection, attackers constantly develop new tactics to bypass detection models. As a result, phishing webpages continue to reach users, particularly those unable to recognize phishing indicators. To improve detection accuracy, models must be trained on large datasets containing both phishing and legitimate webpages, including URLs, webpage content, screenshots, and logos. However, existing tools struggle to collect the required resources, especially given the short lifespan of phishing webpages, limiting dataset comprehensiveness. In response, we introduce Phish-Blitz, a tool that downloads phishing and legitimate webpages along with their associated resources, such as screenshots. Unlike existing tools, Phish-Blitz captures live webpage screenshots and updates resource file paths to maintain the original visual integrity of the webpage. We provide a dataset containing 8,809 legitimate and 5,000 phishing webpages, including all associated resources. Our dataset and tool are publicly available on GitHub, contributing to the research community by offering a more complete dataset for phishing detection.



## **27. Empirical Security Analysis of Software-based Fault Isolation through Controlled Fault Injection**

cs.CR

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.07757v2) [paper-pdf](http://arxiv.org/pdf/2509.07757v2)

**Authors**: Nils Bars, Lukas Bernhard, Moritz Schloegel, Thorsten Holz

**Abstract**: We use browsers daily to access all sorts of information. Because browsers routinely process scripts, media, and executable code from unknown sources, they form a critical security boundary between users and adversaries. A common attack vector is JavaScript, which exposes a large attack surface due to the sheer complexity of modern JavaScript engines. To mitigate these threats, modern engines increasingly adopt software-based fault isolation (SFI). A prominent example is Google's V8 heap sandbox, which represents the most widely deployed SFI mechanism, protecting billions of users across all Chromium-based browsers and countless applications built on Node$.$js and Electron. The heap sandbox splits the address space into two parts: one part containing trusted, security-sensitive metadata, and a sandboxed heap containing memory accessible to untrusted code. On a technical level, the sandbox enforces isolation by removing raw pointers and using translation tables to resolve references to trusted objects. Consequently, an attacker cannot corrupt trusted data even with full control of the sandboxed data, unless there is a bug in how code handles data from the sandboxed heap. Despite their widespread use, such SFI mechanisms have seen little security testing.   In this work, we propose a new testing technique that models the security boundary of modern SFI implementations. Following the SFI threat model, we assume a powerful attacker who fully controls the sandbox's memory. We implement this by instrumenting memory loads originating in the trusted domain and accessing untrusted, attacker-controlled sandbox memory. We then inject faults into the loaded data, aiming to trigger memory corruption in the trusted domain. In a comprehensive evaluation, we identify 19 security bugs in V8 that enable an attacker to bypass the sandbox.



## **28. Adversarial Robustness of Link Sign Prediction in Signed Graphs**

cs.LG

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2401.10590v3) [paper-pdf](http://arxiv.org/pdf/2401.10590v3)

**Authors**: Jialong Zhou, Xing Ai, Yuni Lai, Tomasz Michalak, Gaolei Li, Jianhua Li, Di Tang, Xingxing Zhang, Mengpei Yang, Kai Zhou

**Abstract**: Signed graphs serve as fundamental data structures for representing positive and negative relationships in social networks, with signed graph neural networks (SGNNs) emerging as the primary tool for their analysis. Our investigation reveals that balance theory, while essential for modeling signed relationships in SGNNs, inadvertently introduces exploitable vulnerabilities to black-box attacks. To showcase this, we propose balance-attack, a novel adversarial strategy specifically designed to compromise graph balance degree, and develop an efficient heuristic algorithm to solve the associated NP-hard optimization problem. While existing approaches attempt to restore attacked graphs through balance learning techniques, they face a critical challenge we term "Irreversibility of Balance-related Information," as restored edges fail to align with original attack targets. To address this limitation, we introduce Balance Augmented-Signed Graph Contrastive Learning (BA-SGCL), an innovative framework that combines contrastive learning with balance augmentation techniques to achieve robust graph representations. By maintaining high balance degree in the latent space, BA-SGCL not only effectively circumvents the irreversibility challenge but also significantly enhances model resilience. Extensive experiments across multiple SGNN architectures and real-world datasets demonstrate both the effectiveness of our proposed balance-attack and the superior robustness of BA-SGCL, advancing the security and reliability of signed graph analysis in social networks. Datasets and codes of the proposed framework are at the github repository https://anonymous.4open.science/r/BA-SGCL-submit-DF41/.



## **29. SAGE: Sample-Aware Guarding Engine for Robust Intrusion Detection Against Adversarial Attacks**

cs.CR

Under review at IEEE TIFS

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.08091v1) [paper-pdf](http://arxiv.org/pdf/2509.08091v1)

**Authors**: Jing Chen, Onat Gungor, Zhengli Shang, Tajana Rosing

**Abstract**: The rapid proliferation of the Internet of Things (IoT) continues to expose critical security vulnerabilities, necessitating the development of efficient and robust intrusion detection systems (IDS). Machine learning-based intrusion detection systems (ML-IDS) have significantly improved threat detection capabilities; however, they remain highly susceptible to adversarial attacks. While numerous defense mechanisms have been proposed to enhance ML-IDS resilience, a systematic approach for selecting the most effective defense against a specific adversarial attack remains absent. To address this challenge, we previously proposed DYNAMITE, a dynamic defense selection approach that identifies the most suitable defense against adversarial attacks through an ML-driven selection mechanism. Building on this foundation, we propose SAGE (Sample-Aware Guarding Engine), a substantially improved defense algorithm that integrates active learning with targeted data reduction. It employs an active learning mechanism to selectively identify the most informative input samples and their corresponding optimal defense labels, which are then used to train a second-level learner responsible for selecting the most effective defense. This targeted sampling improves computational efficiency, exposes the model to diverse adversarial strategies during training, and enhances robustness, stability, and generalizability. As a result, SAGE demonstrates strong predictive performance across multiple intrusion detection datasets, achieving an average F1-score improvement of 201% over the state-of-the-art defenses. Notably, SAGE narrows the performance gap to the Oracle to just 3.8%, while reducing computational overhead by up to 29x.



## **30. Hammer and Anvil: A Principled Defense Against Backdoors in Federated Learning**

cs.LG

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.08089v1) [paper-pdf](http://arxiv.org/pdf/2509.08089v1)

**Authors**: Lucas Fenaux, Zheng Wang, Jacob Yan, Nathan Chung, Florian Kerschbaum

**Abstract**: Federated Learning is a distributed learning technique in which multiple clients cooperate to train a machine learning model. Distributed settings facilitate backdoor attacks by malicious clients, who can embed malicious behaviors into the model during their participation in the training process. These malicious behaviors are activated during inference by a specific trigger. No defense against backdoor attacks has stood the test of time, especially against adaptive attackers, a powerful but not fully explored category of attackers. In this work, we first devise a new adaptive adversary that surpasses existing adversaries in capabilities, yielding attacks that only require one or two malicious clients out of 20 to break existing state-of-the-art defenses. Then, we present Hammer and Anvil, a principled defense approach that combines two defenses orthogonal in their underlying principle to produce a combined defense that, given the right set of parameters, must succeed against any attack. We show that our best combined defense, Krum+, is successful against our new adaptive adversary and state-of-the-art attacks.



## **31. Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection**

cs.CL

EMNLP 2025 Main

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2502.13061v3) [paper-pdf](http://arxiv.org/pdf/2502.13061v3)

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL



## **32. Spectral Masking and Interpolation Attack (SMIA): A Black-box Adversarial Attack against Voice Authentication and Anti-Spoofing Systems**

cs.SD

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07677v1) [paper-pdf](http://arxiv.org/pdf/2509.07677v1)

**Authors**: Kamel Kamel, Hridoy Sankar Dutta, Keshav Sood, Sunil Aryal

**Abstract**: Voice Authentication Systems (VAS) use unique vocal characteristics for verification. They are increasingly integrated into high-security sectors such as banking and healthcare. Despite their improvements using deep learning, they face severe vulnerabilities from sophisticated threats like deepfakes and adversarial attacks. The emergence of realistic voice cloning complicates detection, as systems struggle to distinguish authentic from synthetic audio. While anti-spoofing countermeasures (CMs) exist to mitigate these risks, many rely on static detection models that can be bypassed by novel adversarial methods, leaving a critical security gap. To demonstrate this vulnerability, we propose the Spectral Masking and Interpolation Attack (SMIA), a novel method that strategically manipulates inaudible frequency regions of AI-generated audio. By altering the voice in imperceptible zones to the human ear, SMIA creates adversarial samples that sound authentic while deceiving CMs. We conducted a comprehensive evaluation of our attack against state-of-the-art (SOTA) models across multiple tasks, under simulated real-world conditions. SMIA achieved a strong attack success rate (ASR) of at least 82% against combined VAS/CM systems, at least 97.5% against standalone speaker verification systems, and 100% against countermeasures. These findings conclusively demonstrate that current security postures are insufficient against adaptive adversarial attacks. This work highlights the urgent need for a paradigm shift toward next-generation defenses that employ dynamic, context-aware frameworks capable of evolving with the threat landscape.



## **33. Transferable Direct Prompt Injection via Activation-Guided MCMC Sampling**

cs.AI

Accepted to EMNLP 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07617v1) [paper-pdf](http://arxiv.org/pdf/2509.07617v1)

**Authors**: Minghui Li, Hao Zhang, Yechao Zhang, Wei Wan, Shengshan Hu, pei Xiaobing, Jing Wang

**Abstract**: Direct Prompt Injection (DPI) attacks pose a critical security threat to Large Language Models (LLMs) due to their low barrier of execution and high potential damage. To address the impracticality of existing white-box/gray-box methods and the poor transferability of black-box methods, we propose an activations-guided prompt injection attack framework. We first construct an Energy-based Model (EBM) using activations from a surrogate model to evaluate the quality of adversarial prompts. Guided by the trained EBM, we employ the token-level Markov Chain Monte Carlo (MCMC) sampling to adaptively optimize adversarial prompts, thereby enabling gradient-free black-box attacks. Experimental results demonstrate our superior cross-model transferability, achieving 49.6% attack success rate (ASR) across five mainstream LLMs and 34.6% improvement over human-crafted prompts, and maintaining 36.6% ASR on unseen task scenarios. Interpretability analysis reveals a correlation between activations and attack effectiveness, highlighting the critical role of semantic patterns in transferable vulnerability exploitation.



## **34. A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems**

cs.CR

This paper is submitted to the Computer Science Review

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2508.16843v3) [paper-pdf](http://arxiv.org/pdf/2508.16843v3)

**Authors**: Kamel Kamel, Keshav Sood, Hridoy Sankar Dutta, Sunil Aryal

**Abstract**: Voice authentication has undergone significant changes from traditional systems that relied on handcrafted acoustic features to deep learning models that can extract robust speaker embeddings. This advancement has expanded its applications across finance, smart devices, law enforcement, and beyond. However, as adoption has grown, so have the threats. This survey presents a comprehensive review of the modern threat landscape targeting Voice Authentication Systems (VAS) and Anti-Spoofing Countermeasures (CMs), including data poisoning, adversarial, deepfake, and adversarial spoofing attacks. We chronologically trace the development of voice authentication and examine how vulnerabilities have evolved in tandem with technological advancements. For each category of attack, we summarize methodologies, highlight commonly used datasets, compare performance and limitations, and organize existing literature using widely accepted taxonomies. By highlighting emerging risks and open challenges, this survey aims to support the development of more secure and resilient voice authentication systems.



## **35. Generating Transferrable Adversarial Examples via Local Mixing and Logits Optimization for Remote Sensing Object Recognition**

cs.CV

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07495v1) [paper-pdf](http://arxiv.org/pdf/2509.07495v1)

**Authors**: Chun Liu, Hailong Wang, Bingqian Zhu, Panpan Ding, Zheng Zheng, Tao Xu, Zhigang Han, Jiayao Wang

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial attacks, posing significant security threats to their deployment in remote sensing applications. Research on adversarial attacks not only reveals model vulnerabilities but also provides critical insights for enhancing robustness. Although current mixing-based strategies have been proposed to increase the transferability of adversarial examples, they either perform global blending or directly exchange a region in the images, which may destroy global semantic features and mislead the optimization of adversarial examples. Furthermore, their reliance on cross-entropy loss for perturbation optimization leads to gradient diminishing during iterative updates, compromising adversarial example quality. To address these limitations, we focus on non-targeted attacks and propose a novel framework via local mixing and logits optimization. First, we present a local mixing strategy to generate diverse yet semantically consistent inputs. Different from MixUp, which globally blends two images, and MixCut, which stitches images together, our method merely blends local regions to preserve global semantic information. Second, we adapt the logit loss from targeted attacks to non-targeted scenarios, mitigating the gradient vanishing problem of cross-entropy loss. Third, a perturbation smoothing loss is applied to suppress high-frequency noise and enhance transferability. Extensive experiments on FGSCR-42 and MTARSI datasets demonstrate superior performance over 12 state-of-the-art methods across 6 surrogate models. Notably, with ResNet as the surrogate on MTARSI, our method achieves a 17.28% average improvement in black-box attack success rate.



## **36. Texture- and Shape-based Adversarial Attacks for Overhead Image Vehicle Detection**

cs.CV

This version corresponds to the paper accepted for presentation at  ICIP 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2412.16358v2) [paper-pdf](http://arxiv.org/pdf/2412.16358v2)

**Authors**: Mikael Yeghiazaryan, Sai Abhishek Siddhartha Namburu, Emily Kim, Stanislav Panev, Celso de Melo, Fernando De la Torre, Jessica K. Hodgins

**Abstract**: Detecting vehicles in aerial images is difficult due to complex backgrounds, small object sizes, shadows, and occlusions. Although recent deep learning advancements have improved object detection, these models remain susceptible to adversarial attacks (AAs), challenging their reliability. Traditional AA strategies often ignore practical implementation constraints. Our work proposes realistic and practical constraints on texture (lowering resolution, limiting modified areas, and color ranges) and analyzes the impact of shape modifications on attack performance. We conducted extensive experiments with three object detector architectures, demonstrating the performance-practicality trade-off: more practical modifications tend to be less effective, and vice versa. We release both code and data to support reproducibility at https://github.com/humansensinglab/texture-shape-adversarial-attacks.



## **37. Prepared for the Worst: A Learning-Based Adversarial Attack for Resilience Analysis of the ICP Algorithm**

cs.RO

9 pages (6 content, 1 reference, 2 appendix). 7 figures, accepted to  2025 IEEE International Conference on Robotics and Automation (ICRA)

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2403.05666v3) [paper-pdf](http://arxiv.org/pdf/2403.05666v3)

**Authors**: Ziyu Zhang, Johann Laconte, Daniil Lisus, Timothy D. Barfoot

**Abstract**: This paper presents a novel method for assessing the resilience of the ICP algorithm via learning-based, worst-case attacks on lidar point clouds. For safety-critical applications such as autonomous navigation, ensuring the resilience of algorithms before deployments is crucial. The ICP algorithm is the standard for lidar-based localization, but its accuracy can be greatly affected by corrupted measurements from various sources, including occlusions, adverse weather, or mechanical sensor issues. Unfortunately, the complex and iterative nature of ICP makes assessing its resilience to corruption challenging. While there have been efforts to create challenging datasets and develop simulations to evaluate the resilience of ICP, our method focuses on finding the maximum possible ICP error that can arise from corrupted measurements at a location. We demonstrate that our perturbation-based adversarial attacks can be used pre-deployment to identify locations on a map where ICP is particularly vulnerable to corruptions in the measurements. With such information, autonomous robots can take safer paths when deployed, to mitigate against their measurements being corrupted. The proposed attack outperforms baselines more than 88% of the time across a wide range of scenarios.



## **38. When Fine-Tuning is Not Enough: Lessons from HSAD on Hybrid and Adversarial Audio Spoof Detection**

cs.SD

13 pages, 11 figures.This work has been submitted to the IEEE for  possible publication

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07323v1) [paper-pdf](http://arxiv.org/pdf/2509.07323v1)

**Authors**: Bin Hu, Kunyang Huang, Daehan Kwak, Meng Xu, Kuan Huang

**Abstract**: The rapid advancement of AI has enabled highly realistic speech synthesis and voice cloning, posing serious risks to voice authentication, smart assistants, and telecom security. While most prior work frames spoof detection as a binary task, real-world attacks often involve hybrid utterances that mix genuine and synthetic speech, making detection substantially more challenging. To address this gap, we introduce the Hybrid Spoofed Audio Dataset (HSAD), a benchmark containing 1,248 clean and 41,044 degraded utterances across four classes: human, cloned, zero-shot AI-generated, and hybrid audio. Each sample is annotated with spoofing method, speaker identity, and degradation metadata to enable fine-grained analysis. We evaluate six transformer-based models, including spectrogram encoders (MIT-AST, MattyB95-AST) and self-supervised waveform models (Wav2Vec2, HuBERT). Results reveal critical lessons: pretrained models overgeneralize and collapse under hybrid conditions; spoof-specific fine-tuning improves separability but struggles with unseen compositions; and dataset-specific adaptation on HSAD yields large performance gains (AST greater than 97 percent and F1 score is approximately 99 percent), though residual errors persist for complex hybrids. These findings demonstrate that fine-tuning alone is not sufficient-robust hybrid-aware benchmarks like HSAD are essential to expose calibration failures, model biases, and factors affecting spoof detection in adversarial environments. HSAD thus provides both a dataset and an analytic framework for building resilient and trustworthy voice authentication systems.



## **39. GRADA: Graph-based Reranking against Adversarial Documents Attack**

cs.IR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2505.07546v2) [paper-pdf](http://arxiv.org/pdf/2505.07546v2)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.



## **40. Personalized Attacks of Social Engineering in Multi-turn Conversations: LLM Agents for Simulation and Detection**

cs.CR

Accepted as a paper at COLM 2025 Workshop on AI Agents: Capabilities  and Safety

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2503.15552v2) [paper-pdf](http://arxiv.org/pdf/2503.15552v2)

**Authors**: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu

**Abstract**: The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the SE attack mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts.



## **41. Adversarial Attacks on Audio Deepfake Detection: A Benchmark and Comparative Study**

cs.SD

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.07132v1) [paper-pdf](http://arxiv.org/pdf/2509.07132v1)

**Authors**: Kutub Uddin, Muhammad Umar Farooq, Awais Khan, Khalid Mahmood Malik

**Abstract**: The widespread use of generative AI has shown remarkable success in producing highly realistic deepfakes, posing a serious threat to various voice biometric applications, including speaker verification, voice biometrics, audio conferencing, and criminal investigations. To counteract this, several state-of-the-art (SoTA) audio deepfake detection (ADD) methods have been proposed to identify generative AI signatures to distinguish between real and deepfake audio. However, the effectiveness of these methods is severely undermined by anti-forensic (AF) attacks that conceal generative signatures. These AF attacks span a wide range of techniques, including statistical modifications (e.g., pitch shifting, filtering, noise addition, and quantization) and optimization-based attacks (e.g., FGSM, PGD, C \& W, and DeepFool). In this paper, we investigate the SoTA ADD methods and provide a comparative analysis to highlight their effectiveness in exposing deepfake signatures, as well as their vulnerabilities under adversarial conditions. We conducted an extensive evaluation of ADD methods on five deepfake benchmark datasets using two categories: raw and spectrogram-based approaches. This comparative analysis enables a deeper understanding of the strengths and limitations of SoTA ADD methods against diverse AF attacks. It does not only highlight vulnerabilities of ADD methods, but also informs the design of more robust and generalized detectors for real-world voice biometrics. It will further guide future research in developing adaptive defense strategies that can effectively counter evolving AF techniques.



## **42. Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models**

cs.CR

6 pages, 2 figures

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2508.17674v2) [paper-pdf](http://arxiv.org/pdf/2508.17674v2)

**Authors**: Qiming Guo, Jinwen Tang, Xingran Huang

**Abstract**: We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.



## **43. From Noise to Narrative: Tracing the Origins of Hallucinations in Transformers**

cs.LG

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06938v1) [paper-pdf](http://arxiv.org/pdf/2509.06938v1)

**Authors**: Praneet Suresh, Jack Stanley, Sonia Joseph, Luca Scimeca, Danilo Bzdok

**Abstract**: As generative AI systems become competent and democratized in science, business, and government, deeper insight into their failure modes now poses an acute need. The occasional volatility in their behavior, such as the propensity of transformer models to hallucinate, impedes trust and adoption of emerging AI solutions in high-stakes areas. In the present work, we establish how and when hallucinations arise in pre-trained transformer models through concept representations captured by sparse autoencoders, under scenarios with experimentally controlled uncertainty in the input space. Our systematic experiments reveal that the number of semantic concepts used by the transformer model grows as the input information becomes increasingly unstructured. In the face of growing uncertainty in the input space, the transformer model becomes prone to activate coherent yet input-insensitive semantic features, leading to hallucinated output. At its extreme, for pure-noise inputs, we identify a wide variety of robustly triggered and meaningful concepts in the intermediate activations of pre-trained transformer models, whose functional integrity we confirm through targeted steering. We also show that hallucinations in the output of a transformer model can be reliably predicted from the concept patterns embedded in transformer layer activations. This collection of insights on transformer internal processing mechanics has immediate consequences for aligning AI models with human values, AI safety, opening the attack surface for potential adversarial attacks, and providing a basis for automatic quantification of a model's hallucination risk.



## **44. Evaluating the Impact of Adversarial Attacks on Traffic Sign Classification using the LISA Dataset**

cs.CV

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06835v1) [paper-pdf](http://arxiv.org/pdf/2509.06835v1)

**Authors**: Nabeyou Tadessa, Balaji Iyangar, Mashrur Chowdhury

**Abstract**: Adversarial attacks pose significant threats to machine learning models by introducing carefully crafted perturbations that cause misclassification. While prior work has primarily focused on MNIST and similar datasets, this paper investigates the vulnerability of traffic sign classifiers using the LISA Traffic Sign dataset. We train a convolutional neural network to classify 47 different traffic signs and evaluate its robustness against Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks. Our results show a sharp decline in classification accuracy as the perturbation magnitude increases, highlighting the models susceptibility to adversarial examples. This study lays the groundwork for future exploration into defense mechanisms tailored for real-world traffic sign recognition systems.



## **45. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

cs.CL

Accepted at EMNLP 2025 (Main)

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2501.01872v4) [paper-pdf](http://arxiv.org/pdf/2501.01872v4)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.



## **46. On Hyperparameters and Backdoor-Resistance in Horizontal Federated Learning**

cs.CR

To appear in the Proceedings of the ACM Conference on Computer and  Communications Security (CCS) 2025

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.05192v2) [paper-pdf](http://arxiv.org/pdf/2509.05192v2)

**Authors**: Simon Lachnit, Ghassan Karame

**Abstract**: Horizontal Federated Learning (HFL) is particularly vulnerable to backdoor attacks as adversaries can easily manipulate both the training data and processes to execute sophisticated attacks. In this work, we study the impact of training hyperparameters on the effectiveness of backdoor attacks and defenses in HFL. More specifically, we show both analytically and by means of measurements that the choice of hyperparameters by benign clients does not only influence model accuracy but also significantly impacts backdoor attack success. This stands in sharp contrast with the multitude of contributions in the area of HFL security, which often rely on custom ad-hoc hyperparameter choices for benign clients$\unicode{x2013}$leading to more pronounced backdoor attack strength and diminished impact of defenses. Our results indicate that properly tuning benign clients' hyperparameters$\unicode{x2013}$such as learning rate, batch size, and number of local epochs$\unicode{x2013}$can significantly curb the effectiveness of backdoor attacks, regardless of the malicious clients' settings. We support this claim with an extensive robustness evaluation of state-of-the-art attack-defense combinations, showing that carefully chosen hyperparameters yield across-the-board improvements in robustness without sacrificing main task accuracy. For example, we show that the 50%-lifespan of the strong A3FL attack can be reduced by 98.6%, respectively$\unicode{x2013}$all without using any defense and while incurring only a 2.9 percentage points drop in clean task accuracy.



## **47. Mind Your Server: A Systematic Study of Parasitic Toolchain Attacks on the MCP Ecosystem**

cs.CR

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06572v1) [paper-pdf](http://arxiv.org/pdf/2509.06572v1)

**Authors**: Shuli Zhao, Qinsheng Hou, Zihan Zhan, Yanhao Wang, Yuchong Xie, Yu Guo, Libo Chen, Shenghong Li, Zhi Xue

**Abstract**: Large language models (LLMs) are increasingly integrated with external systems through the Model Context Protocol (MCP), which standardizes tool invocation and has rapidly become a backbone for LLM-powered applications. While this paradigm enhances functionality, it also introduces a fundamental security shift: LLMs transition from passive information processors to autonomous orchestrators of task-oriented toolchains, expanding the attack surface, elevating adversarial goals from manipulating single outputs to hijacking entire execution flows. In this paper, we reveal a new class of attacks, Parasitic Toolchain Attacks, instantiated as MCP Unintended Privacy Disclosure (MCP-UPD). These attacks require no direct victim interaction; instead, adversaries embed malicious instructions into external data sources that LLMs access during legitimate tasks. The malicious logic infiltrates the toolchain and unfolds in three phases: Parasitic Ingestion, Privacy Collection, and Privacy Disclosure, culminating in stealthy exfiltration of private data. Our root cause analysis reveals that MCP lacks both context-tool isolation and least-privilege enforcement, enabling adversarial instructions to propagate unchecked into sensitive tool invocations. To assess the severity, we design MCP-SEC and conduct the first large-scale security census of the MCP ecosystem, analyzing 12,230 tools across 1,360 servers. Our findings show that the MCP ecosystem is rife with exploitable gadgets and diverse attack methods, underscoring systemic risks in MCP platforms and the urgent need for defense mechanisms in LLM-integrated environments.



## **48. Robustness and accuracy of mean opinion scores with hard and soft outlier detection**

eess.IV

Accepted for 17th International Conference on Quality of Multimedia  Experience (QoMEX'25), September 2025, Madrid, Spain

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06554v1) [paper-pdf](http://arxiv.org/pdf/2509.06554v1)

**Authors**: Dietmar Saupe, Tim Bleile

**Abstract**: In subjective assessment of image and video quality, observers rate or compare selected stimuli. Before calculating the mean opinion scores (MOS) for these stimuli from the ratings, it is recommended to identify and deal with outliers that may have given unreliable ratings. Several methods are available for this purpose, some of which have been standardized. These methods are typically based on statistics and sometimes tested by introducing synthetic ratings from artificial outliers, such as random clickers. However, a reliable and comprehensive approach is lacking for comparative performance analysis of outlier detection methods. To fill this gap, this work proposes and applies an empirical worst-case analysis as a general solution. Our method involves evolutionary optimization of an adversarial black-box attack on outlier detection algorithms, where the adversary maximizes the distortion of scale values with respect to ground truth. We apply our analysis to several hard and soft outlier detection methods for absolute category ratings and show their differing performance in this stress test. In addition, we propose two new outlier detection methods with low complexity and excellent worst-case performance. Software for adversarial attacks and data analysis is available.



## **49. IGAff: Benchmarking Adversarial Iterative and Genetic Affine Algorithms on Deep Neural Networks**

cs.CV

10 pages, 7 figures, Accepted at ECAI 2025 (28th European Conference  on Artificial Intelligence)

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06459v1) [paper-pdf](http://arxiv.org/pdf/2509.06459v1)

**Authors**: Sebastian-Vasile Echim, Andrei-Alexandru Preda, Dumitru-Clementin Cercel, Florin Pop

**Abstract**: Deep neural networks currently dominate many fields of the artificial intelligence landscape, achieving state-of-the-art results on numerous tasks while remaining hard to understand and exhibiting surprising weaknesses. An active area of research focuses on adversarial attacks, which aim to generate inputs that uncover these weaknesses. However, this proves challenging, especially in the black-box scenario where model details are inaccessible. This paper explores in detail the impact of such adversarial algorithms on ResNet-18, DenseNet-121, Swin Transformer V2, and Vision Transformer network architectures. Leveraging the Tiny ImageNet, Caltech-256, and Food-101 datasets, we benchmark two novel black-box iterative adversarial algorithms based on affine transformations and genetic algorithms: 1) Affine Transformation Attack (ATA), an iterative algorithm maximizing our attack score function using random affine transformations, and 2) Affine Genetic Attack (AGA), a genetic algorithm that involves random noise and affine transformations. We evaluate the performance of the models in the algorithm parameter variation, data augmentation, and global and targeted attack configurations. We also compare our algorithms with two black-box adversarial algorithms, Pixle and Square Attack. Our experiments yield better results on the image classification task than similar methods in the literature, achieving an accuracy improvement of up to 8.82%. We provide noteworthy insights into successful adversarial defenses and attacks at both global and targeted levels, and demonstrate adversarial robustness through algorithm parameter variation.



## **50. Mask-GCG: Are All Tokens in Adversarial Suffixes Necessary for Jailbreak Attacks?**

cs.CL

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06350v1) [paper-pdf](http://arxiv.org/pdf/2509.06350v1)

**Authors**: Junjie Mu, Zonghao Ying, Zhekui Fan, Zonglei Jing, Yaoyuan Zhang, Zhengmin Yu, Wenxin Zhang, Quanchen Zou, Xiangzheng Zhang

**Abstract**: Jailbreak attacks on Large Language Models (LLMs) have demonstrated various successful methods whereby attackers manipulate models into generating harmful responses that they are designed to avoid. Among these, Greedy Coordinate Gradient (GCG) has emerged as a general and effective approach that optimizes the tokens in a suffix to generate jailbreakable prompts. While several improved variants of GCG have been proposed, they all rely on fixed-length suffixes. However, the potential redundancy within these suffixes remains unexplored. In this work, we propose Mask-GCG, a plug-and-play method that employs learnable token masking to identify impactful tokens within the suffix. Our approach increases the update probability for tokens at high-impact positions while pruning those at low-impact positions. This pruning not only reduces redundancy but also decreases the size of the gradient space, thereby lowering computational overhead and shortening the time required to achieve successful attacks compared to GCG. We evaluate Mask-GCG by applying it to the original GCG and several improved variants. Experimental results show that most tokens in the suffix contribute significantly to attack success, and pruning a minority of low-impact tokens does not affect the loss values or compromise the attack success rate (ASR), thereby revealing token redundancy in LLM prompts. Our findings provide insights for developing efficient and interpretable LLMs from the perspective of jailbreak attacks.



