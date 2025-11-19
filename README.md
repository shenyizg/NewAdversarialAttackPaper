# Latest Adversarial Attack Papers
**update at 2025-11-19 09:21:16**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Tuning for Two Adversaries: Enhancing the Robustness Against Transfer and Query-Based Attacks using Hyperparameter Tuning**

cs.LG

To appear in the Proceedings of the AAAI Conference on Artificial Intelligence (AAAI) 2026

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13654v1) [paper-pdf](https://arxiv.org/pdf/2511.13654v1)

**Authors**: Pascal Zimmer, Ghassan Karame

**Abstract**: In this paper, we present the first detailed analysis of how optimization hyperparameters -- such as learning rate, weight decay, momentum, and batch size -- influence robustness against both transfer-based and query-based attacks. Supported by theory and experiments, our study spans a variety of practical deployment settings, including centralized training, ensemble learning, and distributed training. We uncover a striking dichotomy: for transfer-based attacks, decreasing the learning rate significantly enhances robustness by up to $64\%$. In contrast, for query-based attacks, increasing the learning rate consistently leads to improved robustness by up to $28\%$ across various settings and data distributions. Leveraging these findings, we explore -- for the first time -- the optimization hyperparameter design space to jointly enhance robustness against both transfer-based and query-based attacks. Our results reveal that distributed models benefit the most from hyperparameter tuning, achieving a remarkable tradeoff by simultaneously mitigating both attack types more effectively than other training setups.



## **2. ForgeDAN: An Evolutionary Framework for Jailbreaking Aligned Large Language Models**

cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13548v1) [paper-pdf](https://arxiv.org/pdf/2511.13548v1)

**Authors**: Siyang Cheng, Gaotian Liu, Rui Mei, Yilin Wang, Kejia Zhang, Kaishuo Wei, Yuqi Yu, Weiping Wen, Xiaojie Wu, Junhua Liu

**Abstract**: The rapid adoption of large language models (LLMs) has brought both transformative applications and new security risks, including jailbreak attacks that bypass alignment safeguards to elicit harmful outputs. Existing automated jailbreak generation approaches e.g. AutoDAN, suffer from limited mutation diversity, shallow fitness evaluation, and fragile keyword-based detection. To address these limitations, we propose ForgeDAN, a novel evolutionary framework for generating semantically coherent and highly effective adversarial prompts against aligned LLMs. First, ForgeDAN introduces multi-strategy textual perturbations across \textit{character, word, and sentence-level} operations to enhance attack diversity; then we employ interpretable semantic fitness evaluation based on a text similarity model to guide the evolutionary process toward semantically relevant and harmful outputs; finally, ForgeDAN integrates dual-dimensional jailbreak judgment, leveraging an LLM-based classifier to jointly assess model compliance and output harmfulness, thereby reducing false positives and improving detection effectiveness. Our evaluation demonstrates ForgeDAN achieves high jailbreaking success rates while maintaining naturalness and stealth, outperforming existing SOTA solutions.



## **3. Robust Defense Strategies for Multimodal Contrastive Learning: Efficient Fine-tuning Against Backdoor Attacks**

cs.CV

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13545v1) [paper-pdf](https://arxiv.org/pdf/2511.13545v1)

**Authors**: Md. Iqbal Hossain, Afia Sajeeda, Neeresh Kumar Perla, Ming Shao

**Abstract**: The advent of multimodal deep learning models, such as CLIP, has unlocked new frontiers in a wide range of applications, from image-text understanding to classification tasks. However, these models are not safe for adversarial attacks, particularly backdoor attacks, which can subtly manipulate model behavior. Moreover, existing defense methods typically involve training from scratch or fine-tuning using a large dataset without pinpointing the specific labels that are affected. In this study, we introduce an innovative strategy to enhance the robustness of multimodal contrastive learning models against such attacks. In particular, given a poisoned CLIP model, our approach can identify the backdoor trigger and pinpoint the victim samples and labels in an efficient manner. To that end, an image segmentation ``oracle'' is introduced as the supervisor for the output of the poisoned CLIP. We develop two algorithms to rectify the poisoned model: (1) differentiating between CLIP and Oracle's knowledge to identify potential triggers; (2) pinpointing affected labels and victim samples, and curating a compact fine-tuning dataset. With this knowledge, we are allowed to rectify the poisoned CLIP model to negate backdoor effects. Extensive experiments on visual recognition benchmarks demonstrate our strategy is effective in CLIP-based backdoor defense.



## **4. Accuracy is Not Enough: Poisoning Interpretability in Federated Learning via Color Skew**

cs.CV

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13535v1) [paper-pdf](https://arxiv.org/pdf/2511.13535v1)

**Authors**: Farhin Farhad Riya, Shahinul Hoque, Jinyuan Stella Sun, Olivera Kotevska

**Abstract**: As machine learning models are increasingly deployed in safety-critical domains, visual explanation techniques have become essential tools for supporting transparency. In this work, we reveal a new class of attacks that compromise model interpretability without affecting accuracy. Specifically, we show that small color perturbations applied by adversarial clients in a federated learning setting can shift a model's saliency maps away from semantically meaningful regions while keeping the prediction unchanged. The proposed saliency-aware attack framework, called Chromatic Perturbation Module, systematically crafts adversarial examples by altering the color contrast between foreground and background in a way that disrupts explanation fidelity. These perturbations accumulate across training rounds, poisoning the global model's internal feature attributions in a stealthy and persistent manner. Our findings challenge a common assumption in model auditing that correct predictions imply faithful explanations and demonstrate that interpretability itself can be an attack surface. We evaluate this vulnerability across multiple datasets and show that standard training pipelines are insufficient to detect or mitigate explanation degradation, especially in the federated learning setting, where subtle color perturbations are harder to discern. Our attack reduces peak activation overlap in Grad-CAM explanations by up to 35% while preserving classification accuracy above 96% on all evaluated datasets.



## **5. Cyber-Resilient Fault Diagnosis Methodology in Inverter-Based Resource-Dominated Microgrids with Single-Point Measurement**

eess.SY

5 pages, 5 figures

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13162v1) [paper-pdf](https://arxiv.org/pdf/2511.13162v1)

**Authors**: Yifan Wang, Yiyao Yu, Yang Xia, Yan Xu

**Abstract**: Cyber-attacks jeopardize the safe operation of inverter-based resource-dominated microgrids (IBR-dominated microgrids). At the same time, existing diagnostic methods either depend on expensive multi-point instrumentation or stringent modeling assumptions that are untenable under single-point measurement constraints. This paper proposes a Fractional-Order Memory-Enhanced Attack-Diagnosis Scheme (FO-MADS) that achieves timely fault localization and cyber-resilient fault diagnosis using only one VPQ (voltage, active power, reactive power) measurement point. FO-MADS first constructs a dual fractional-order feature library by jointly applying Caputo and Grünwald-Letnikov derivatives, thereby amplifying micro-perturbations and slow drifts in the VPQ signal. A two-stage hierarchical classifier then pinpoints the affected inverter and isolates the faulty IGBT switch, effectively alleviating class imbalance. Robustness is further strengthened through Progressive Memory-Replay Adversarial Training (PMR-AT), whose attack-aware loss is dynamically re-weighted via Online Hard Example Mining (OHEM) to prioritize the most challenging samples. Experiments on a four-inverter IBR-dominated microgrid testbed comprising 1 normal and 24 fault classes under four attack scenarios demonstrate diagnostic accuracies of 96.6% (bias), 94.0% (noise), 92.8% (data replacement), and 95.7% (replay), while sustaining 96.7% under attack-free conditions. These results establish FO-MADS as a cost-effective and readily deployable solution that markedly enhances the cyber-physical resilience of IBR-dominated microgrids.



## **6. Shedding Light on VLN Robustness: A Black-box Framework for Indoor Lighting-based Adversarial Attack**

cs.CV

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13132v1) [paper-pdf](https://arxiv.org/pdf/2511.13132v1)

**Authors**: Chenyang Li, Wenbing Tang, Yihao Huang, Sinong Simon Zhan, Ming Hu, Xiaojun Jia, Yang Liu

**Abstract**: Vision-and-Language Navigation (VLN) agents have made remarkable progress, but their robustness remains insufficiently studied. Existing adversarial evaluations often rely on perturbations that manifest as unusual textures rarely encountered in everyday indoor environments. Errors under such contrived conditions have limited practical relevance, as real-world agents are unlikely to encounter such artificial patterns. In this work, we focus on indoor lighting, an intrinsic yet largely overlooked scene attribute that strongly influences navigation. We propose Indoor Lighting-based Adversarial Attack (ILA), a black-box framework that manipulates global illumination to disrupt VLN agents. Motivated by typical household lighting usage, we design two attack modes: Static Indoor Lighting-based Attack (SILA), where the lighting intensity remains constant throughout an episode, and Dynamic Indoor Lighting-based Attack (DILA), where lights are switched on or off at critical moments to induce abrupt illumination changes. We evaluate ILA on two state-of-the-art VLN models across three navigation tasks. Results show that ILA significantly increases failure rates while reducing trajectory efficiency, revealing previously unrecognized vulnerabilities of VLN agents to realistic indoor lighting variations.



## **7. VEIL: Jailbreaking Text-to-Video Models via Visual Exploitation from Implicit Language**

cs.CV

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13127v1) [paper-pdf](https://arxiv.org/pdf/2511.13127v1)

**Authors**: Zonghao Ying, Moyang Chen, Nizhang Li, Zhiqiang Wang, Wenxin Zhang, Quanchen Zou, Zonglei Jing, Aishan Liu, Xianglong Liu

**Abstract**: Jailbreak attacks can circumvent model safety guardrails and reveal critical blind spots. Prior attacks on text-to-video (T2V) models typically add adversarial perturbations to obviously unsafe prompts, which are often easy to detect and defend. In contrast, we show that benign-looking prompts containing rich, implicit cues can induce T2V models to generate semantically unsafe videos that both violate policy and preserve the original (blocked) intent. To realize this, we propose VEIL, a jailbreak framework that leverages T2V models' cross-modal associative patterns via a modular prompt design. Specifically, our prompts combine three components: neutral scene anchors, which provide the surface-level scene description extracted from the blocked intent to maintain plausibility; latent auditory triggers, textual descriptions of innocuous-sounding audio events (e.g., creaking, muffled noises) that exploit learned audio-visual co-occurrence priors to bias the model toward particular unsafe visual concepts; and stylistic modulators, cinematic directives (e.g., camera framing, atmosphere) that amplify and stabilize the latent trigger's effect. We formalize attack generation as a constrained optimization over the above modular prompt space and solve it with a guided search procedure that balances stealth and effectiveness. Extensive experiments over 7 T2V models demonstrate the efficacy of our attack, achieving a 23 percent improvement in average attack success rate in commercial models.



## **8. Angular Gradient Sign Method: Uncovering Vulnerabilities in Hyperbolic Networks**

cs.LG

Accepted by AAAI 2026

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.12985v1) [paper-pdf](https://arxiv.org/pdf/2511.12985v1)

**Authors**: Minsoo Jo, Dongyoon Yang, Taesup Kim

**Abstract**: Adversarial examples in neural networks have been extensively studied in Euclidean geometry, but recent advances in \textit{hyperbolic networks} call for a reevaluation of attack strategies in non-Euclidean geometries. Existing methods such as FGSM and PGD apply perturbations without regard to the underlying hyperbolic structure, potentially leading to inefficient or geometrically inconsistent attacks. In this work, we propose a novel adversarial attack that explicitly leverages the geometric properties of hyperbolic space. Specifically, we compute the gradient of the loss function in the tangent space of hyperbolic space, decompose it into a radial (depth) component and an angular (semantic) component, and apply perturbation derived solely from the angular direction. Our method generates adversarial examples by focusing perturbations in semantically sensitive directions encoded in angular movement within the hyperbolic geometry. Empirical results on image classification, cross-modal retrieval tasks and network architectures demonstrate that our attack achieves higher fooling rates than conventional adversarial attacks, while producing high-impact perturbations with deeper insights into vulnerabilities of hyperbolic embeddings. This work highlights the importance of geometry-aware adversarial strategies in curved representation spaces and provides a principled framework for attacking hierarchical embeddings.



## **9. T2I-Based Physical-World Appearance Attack against Traffic Sign Recognition Systems in Autonomous Driving**

cs.CV

16 pages, 12 figures

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.12956v1) [paper-pdf](https://arxiv.org/pdf/2511.12956v1)

**Authors**: Chen Ma, Ningfei Wang, Junhao Zheng, Qing Guo, Qian Wang, Qi Alfred Chen, Chao Shen

**Abstract**: Traffic Sign Recognition (TSR) systems play a critical role in Autonomous Driving (AD) systems, enabling real-time detection of road signs, such as STOP and speed limit signs. While these systems are increasingly integrated into commercial vehicles, recent research has exposed their vulnerability to physical-world adversarial appearance attacks. In such attacks, carefully crafted visual patterns are misinterpreted by TSR models as legitimate traffic signs, while remaining inconspicuous or benign to human observers. However, existing adversarial appearance attacks suffer from notable limitations. Pixel-level perturbation-based methods often lack stealthiness and tend to overfit to specific surrogate models, resulting in poor transferability to real-world TSR systems. On the other hand, text-to-image (T2I) diffusion model-based approaches demonstrate limited effectiveness and poor generalization to out-of-distribution sign types.   In this paper, we present DiffSign, a novel T2I-based appearance attack framework designed to generate physically robust, highly effective, transferable, practical, and stealthy appearance attacks against TSR systems. To overcome the limitations of prior approaches, we propose a carefully designed attack pipeline that integrates CLIP-based loss and masked prompts to improve attack focus and controllability. We also propose two novel style customization methods to guide visual appearance and improve out-of-domain traffic sign attack generalization and attack stealthiness. We conduct extensive evaluations of DiffSign under varied real-world conditions, including different distances, angles, light conditions, and sign categories. Our method achieves an average physical-world attack success rate of 83.3%, leveraging DiffSign's high effectiveness in attack transferability.



## **10. Efficient Adversarial Malware Defense via Trust-Based Raw Override and Confidence-Adaptive Bit-Depth Reduction**

cs.CR

Accepted at IEEE International Conference on Big Data 2025. 10 pages, 2 figures, 8 tables

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12827v1) [paper-pdf](https://arxiv.org/pdf/2511.12827v1)

**Authors**: Ayush Chaudhary, Sisir Doppalpudi

**Abstract**: The deployment of robust malware detection systems in big data environments requires careful consideration of both security effectiveness and computational efficiency. While recent advances in adversarial defenses have demonstrated strong robustness improvements, they often introduce computational overhead ranging from 4x to 22x, which presents significant challenges for production systems processing millions of samples daily. In this work, we propose a novel framework that combines Trust-Raw Override (TRO) with Confidence-Adaptive Bit-Depth Reduction (CABDR) to explicitly optimize the trade-off between adversarial robustness and computational efficiency. Our approach leverages adaptive confidence-based mechanisms to selectively apply defensive measures, achieving 1.76x computational overhead - a 2.3x improvement over state-of-the-art smoothing defenses. Through comprehensive evaluation on the EMBER v2 dataset comprising 800K samples, we demonstrate that our framework maintains 91 percent clean accuracy while reducing attack success rates to 31-37 percent across multiple attack types, with particularly strong performance against optimization-based attacks such as C and W (48.8 percent reduction). The framework achieves throughput of up to 1.26 million samples per second (measured on pre-extracted EMBER features with no runtime feature extraction), validated across 72 production configurations with statistical significance (5 independent runs, 95 percent confidence intervals, p less than 0.01). Our results suggest that practical adversarial robustness in production environments requires explicit optimization of the efficiency-robustness trade-off, providing a viable path for organizations to deploy robust defenses without prohibitive infrastructure costs.



## **11. LLM Reinforcement in Context**

cs.CL

4 pages

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12782v1) [paper-pdf](https://arxiv.org/pdf/2511.12782v1)

**Authors**: Thomas Rivasseau

**Abstract**: Current Large Language Model alignment research mostly focuses on improving model robustness against adversarial attacks and misbehavior by training on examples and prompting. Research has shown that LLM jailbreak probability increases with the size of the user input or conversation length. There is a lack of appropriate research into means of strengthening alignment which also scale with user input length. We propose interruptions as a possible solution to this problem. Interruptions are control sentences added to the user input approximately every x tokens for some arbitrary x. We suggest that this can be generalized to the Chain-of-Thought process to prevent scheming.



## **12. On Robustness of Linear Classifiers to Targeted Data Poisoning**

cs.LG

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12722v1) [paper-pdf](https://arxiv.org/pdf/2511.12722v1)

**Authors**: Nakshatra Gupta, Sumanth Prabhu, Supratik Chakraborty, R Venkatesh

**Abstract**: Data poisoning is a training-time attack that undermines the trustworthiness of learned models. In a targeted data poisoning attack, an adversary manipulates the training dataset to alter the classification of a targeted test point. Given the typically large size of training dataset, manual detection of poisoning is difficult. An alternative is to automatically measure a dataset's robustness against such an attack, which is the focus of this paper. We consider a threat model wherein an adversary can only perturb the labels of the training dataset, with knowledge limited to the hypothesis space of the victim's model. In this setting, we prove that finding the robustness is an NP-Complete problem, even when hypotheses are linear classifiers. To overcome this, we present a technique that finds lower and upper bounds of robustness. Our implementation of the technique computes these bounds efficiently in practice for many publicly available datasets. We experimentally demonstrate the effectiveness of our approach. Specifically, a poisoning exceeding the identified robustness bounds significantly impacts test point classification. We are also able to compute these bounds in many more cases where state-of-the-art techniques fail.



## **13. Scalable Hierarchical AI-Blockchain Framework for Real-Time Anomaly Detection in Large-Scale Autonomous Vehicle Networks**

cs.CR

Submitted to the Journal

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12648v1) [paper-pdf](https://arxiv.org/pdf/2511.12648v1)

**Authors**: Rathin Chandra Shit, Sharmila Subudhi

**Abstract**: The security of autonomous vehicle networks is facing major challenges, owing to the complexity of sensor integration, real-time performance demands, and distributed communication protocols that expose vast attack surfaces around both individual and network-wide safety. Existing security schemes are unable to provide sub-10 ms (milliseconds) anomaly detection and distributed coordination of large-scale networks of vehicles within an acceptable safety/privacy framework. This paper introduces a three-tier hybrid security architecture HAVEN (Hierarchical Autonomous Vehicle Enhanced Network), which decouples real-time local threat detection and distributed coordination operations. It incorporates a light ensemble anomaly detection model on the edge (first layer), Byzantine-fault-tolerant federated learning to aggregate threat intelligence at a regional scale (middle layer), and selected blockchain mechanisms (top layer) to ensure critical security coordination. Extensive experimentation is done on a real-world autonomous driving dataset. Large-scale simulations with the number of vehicles ranging between 100 and 1000 and different attack types, such as sensor spoofing, jamming, and adversarial model poisoning, are conducted to test the scalability and resiliency of HAVEN. Experimental findings show sub-10 ms detection latency with an accuracy of 94% and F1-score of 92% across multimodal sensor data, Byzantine fault tolerance validated with 20\% compromised nodes, and a reduced blockchain storage overhead, guaranteeing sufficient differential privacy. The proposed framework overcomes the important trade-off between real-time safety obligation and distributed security coordination with novel three-tiered processing. The scalable architecture of HAVEN is shown to provide great improvement in detection accuracy as well as network resilience over other methods.



## **14. Beyond Pixels: Semantic-aware Typographic Attack for Geo-Privacy Protection**

cs.CV

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12575v1) [paper-pdf](https://arxiv.org/pdf/2511.12575v1)

**Authors**: Jiayi Zhu, Yihao Huang, Yue Cao, Xiaojun Jia, Qing Guo, Felix Juefei-Xu, Geguang Pu, Bin Wang

**Abstract**: Large Visual Language Models (LVLMs) now pose a serious yet overlooked privacy threat, as they can infer a social media user's geolocation directly from shared images, leading to unintended privacy leakage. While adversarial image perturbations provide a potential direction for geo-privacy protection, they require relatively strong distortions to be effective against LVLMs, which noticeably degrade visual quality and diminish an image's value for sharing. To overcome this limitation, we identify typographical attacks as a promising direction for protecting geo-privacy by adding text extension outside the visual content. We further investigate which textual semantics are effective in disrupting geolocation inference and design a two-stage, semantics-aware typographical attack that generates deceptive text to protect user privacy. Extensive experiments across three datasets demonstrate that our approach significantly reduces geolocation prediction accuracy of five state-of-the-art commercial LVLMs, establishing a practical and visually-preserving protection strategy against emerging geo-privacy threats.



## **15. SGuard-v1: Safety Guardrail for Large Language Models**

cs.CL

Technical Report

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12497v1) [paper-pdf](https://arxiv.org/pdf/2511.12497v1)

**Authors**: JoonHo Lee, HyeonMin Cho, Jaewoong Yun, Hyunjae Lee, JunKyu Lee, Juree Seok

**Abstract**: We present SGuard-v1, a lightweight safety guardrail for Large Language Models (LLMs), which comprises two specialized models to detect harmful content and screen adversarial prompts in human-AI conversational settings. The first component, ContentFilter, is trained to identify safety risks in LLM prompts and responses in accordance with the MLCommons hazard taxonomy, a comprehensive framework for trust and safety assessment of AI. The second component, JailbreakFilter, is trained with a carefully designed curriculum over integrated datasets and findings from prior work on adversarial prompting, covering 60 major attack types while mitigating false-unsafe classification. SGuard-v1 is built on the 2B-parameter Granite-3.3-2B-Instruct model that supports 12 languages. We curate approximately 1.4 million training instances from both collected and synthesized data and perform instruction tuning on the base model, distributing the curated data across the two component according to their designated functions. Through extensive evaluation on public and proprietary safety benchmarks, SGuard-v1 achieves state-of-the-art safety performance while remaining lightweight, thereby reducing deployment overhead. SGuard-v1 also improves interpretability for downstream use by providing multi-class safety predictions and their binary confidence scores. We release the SGuard-v1 under the Apache-2.0 License to enable further research and practical deployment in AI safety.



## **16. GRAPHTEXTACK: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs**

cs.CR

AAAI 2026

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12423v1) [paper-pdf](https://arxiv.org/pdf/2511.12423v1)

**Authors**: Jiaji Ma, Puja Trivedi, Danai Koutra

**Abstract**: Text-attributed graphs (TAGs), which combine structural and textual node information, are ubiquitous across many domains. Recent work integrates Large Language Models (LLMs) with Graph Neural Networks (GNNs) to jointly model semantics and structure, resulting in more general and expressive models that achieve state-of-the-art performance on TAG benchmarks. However, this integration introduces dual vulnerabilities: GNNs are sensitive to structural perturbations, while LLM-derived features are vulnerable to prompt injection and adversarial phrasing. While existing adversarial attacks largely perturb structure or text independently, we find that uni-modal attacks cause only modest degradation in LLM-enhanced GNNs. Moreover, many existing attacks assume unrealistic capabilities, such as white-box access or direct modification of graph data. To address these gaps, we propose GRAPHTEXTACK, the first black-box, multi-modal{, poisoning} node injection attack for LLM-enhanced GNNs. GRAPHTEXTACK injects nodes with carefully crafted structure and semantics to degrade model performance, operating under a realistic threat model without relying on model internals or surrogate models. To navigate the combinatorial, non-differentiable search space of connectivity and feature assignments, GRAPHTEXTACK introduces a novel evolutionary optimization framework with a multi-objective fitness function that balances local prediction disruption and global graph influence. Extensive experiments on five datasets and two state-of-the-art LLM-enhanced GNN models show that GRAPHTEXTACK significantly outperforms 12 strong baselines.



## **17. QMA Complete Quantum-Enhanced Kyber: Provable Security Through CHSH Nonlocality**

quant-ph

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12318v1) [paper-pdf](https://arxiv.org/pdf/2511.12318v1)

**Authors**: Ilias Cherkaoui, Indrakshi Dey

**Abstract**: Post-quantum cryptography (PQC) must secure large-scale communication systems against quantum adversaries where classical hardness alone is insufficient and purely quantum schemes remain impractical. Lattice-based key encapsulation mechanisms (KEMs) such as CRYSTALS-Kyber provide efficient quantum-resistant primitives but rely solely on computational hardness assumptions that are susceptible to hybrid classical-quantum attacks. To overcome this limitation, we introduce the first Clauser-Horne-Shimony-Holt (CHSH)-certified Kyber protocol, which embeds quantum non-locality verification directly within the key exchange phase. The proposed design integrates CHSH entanglement tests using Einstein-Podolsky-Rosen (EPR) pairs to yield measurable quantum advantage values exceeding classical correlation limits, thereby coupling information--theoretic quantum guarantees with lattice-based computational security. Formal reductions demonstrate that any polynomial-time adversary breaking the proposed KEM must either solve the Module Learning With Errors (Module-LWE) problem or a Quantum Merlin-Arthur (QMA)-complete instance of the 2-local Hamiltonian problem, under the standard complexity assumption QMA $\subset$ NP. The construction remains fully compatible with the Fujisaki-Okamoto (FO) transform, preserving chosen-ciphertext attack (CCA) security and Kyber's efficiency profile. The resulting CHSH-augmented Kyber scheme therefore establishes a mathematically rigorous, hybrid post-quantum framework that unifies lattice cryptography and quantum non-locality to achieve verifiable, composable, and forward-secure key agreement.



## **18. Privacy-Preserving Prompt Injection Detection for LLMs Using Federated Learning and Embedding-Based NLP Classification**

cs.CR

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12295v1) [paper-pdf](https://arxiv.org/pdf/2511.12295v1)

**Authors**: Hasini Jayathilaka

**Abstract**: Prompt injection attacks are an emerging threat to large language models (LLMs), enabling malicious users to manipulate outputs through carefully designed inputs. Existing detection approaches often require centralizing prompt data, creating significant privacy risks. This paper proposes a privacy-preserving prompt injection detection framework based on federated learning and embedding-based classification. A curated dataset of benign and adversarial prompts was encoded with sentence embedding and used to train both centralized and federated logistic regression models. The federated approach preserved privacy by sharing only model parameters across clients, while achieving detection performance comparable to centralized training. Results demonstrate that effective prompt injection detection is feasible without exposing raw data, making this one of the first explorations of federated security for LLMs. Although the dataset is limited in scale, the findings establish a strong proof-of-concept and highlight new directions for building secure and privacy-aware LLM systems.



## **19. Calibrated Adversarial Sampling: Multi-Armed Bandit-Guided Generalization Against Unforeseen Attacks**

cs.LG

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12265v1) [paper-pdf](https://arxiv.org/pdf/2511.12265v1)

**Authors**: Rui Wang, Zeming Wei, Xiyue Zhang, Meng Sun

**Abstract**: Deep Neural Networks (DNNs) are known to be vulnerable to various adversarial perturbations. To address the safety concerns arising from these vulnerabilities, adversarial training (AT) has emerged as one of the most effective paradigms for enhancing the robustness of DNNs. However, existing AT frameworks primarily focus on a single or a limited set of attack types, leaving DNNs still exposed to attack types that may be encountered in practice but not addressed during training. In this paper, we propose an efficient fine-tuning method called Calibrated Adversarial Sampling (CAS) to address these issues. From the optimization perspective within the multi-armed bandit framework, it dynamically designs rewards and balances exploration and exploitation by considering the dynamic and interdependent characteristics of multiple robustness dimensions. Experiments on benchmark datasets show that CAS achieves superior overall robustness while maintaining high clean accuracy, providing a new paradigm for robust generalization of DNNs.



## **20. AlignTree: Efficient Defense Against LLM Jailbreak Attacks**

cs.LG

Accepted as an Oral Presentation at the 40th AAAI Conference on Artificial Intelligence (AAAI-26), January 2026

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12217v1) [paper-pdf](https://arxiv.org/pdf/2511.12217v1)

**Authors**: Gil Goren, Shahar Katz, Lior Wolf

**Abstract**: Large Language Models (LLMs) are vulnerable to adversarial attacks that bypass safety guidelines and generate harmful content. Mitigating these vulnerabilities requires defense mechanisms that are both robust and computationally efficient. However, existing approaches either incur high computational costs or rely on lightweight defenses that can be easily circumvented, rendering them impractical for real-world LLM-based systems. In this work, we introduce the AlignTree defense, which enhances model alignment while maintaining minimal computational overhead. AlignTree monitors LLM activations during generation and detects misaligned behavior using an efficient random forest classifier. This classifier operates on two signals: (i) the refusal direction -- a linear representation that activates on misaligned prompts, and (ii) an SVM-based signal that captures non-linear features associated with harmful content. Unlike previous methods, AlignTree does not require additional prompts or auxiliary guard models. Through extensive experiments, we demonstrate the efficiency and robustness of AlignTree across multiple LLMs and benchmarks.



## **21. MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization**

cs.LG

Accepted by AAAI 2026

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12199v1) [paper-pdf](https://arxiv.org/pdf/2511.12199v1)

**Authors**: Runhao Jiang, Chengzhi Jiang, Rui Yan, Huajin Tang

**Abstract**: The surrogate gradient (SG) method has shown significant promise in enhancing the performance of deep spiking neural networks (SNNs), but it also introduces vulnerabilities to adversarial attacks. Although spike coding strategies and neural dynamics parameters have been extensively studied for their impact on robustness, the critical role of gradient magnitude, which reflects the model's sensitivity to input perturbations, remains underexplored. In SNNs, the gradient magnitude is primarily determined by the interaction between the membrane potential distribution (MPD) and the SG function. In this study, we investigate the relationship between the MPD and SG and its implications for improving the robustness of SNNs. Our theoretical analysis reveals that reducing the proportion of membrane potential lying within the gradient-available range of the SG function effectively mitigates the sensitivity of SNNs to input perturbations. Building upon this insight, we propose a novel MPD-driven surrogate gradient regularization (MPD-SGR) method, which enhances robustness by explicitly regularizing the MPD based on its interaction with the SG function. Extensive experiments across multiple image classification benchmarks and diverse network architectures confirm that the MPD-SGR method significantly enhances the resilience of SNNs to adversarial perturbations and exhibits strong generalizability across diverse network configurations, SG function variants, and spike encoding schemes.



## **22. Rethinking Deep Alignment Through The Lens Of Incomplete Learning**

cs.LG

AAAI'26

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12155v1) [paper-pdf](https://arxiv.org/pdf/2511.12155v1)

**Authors**: Thong Bach, Dung Nguyen, Thao Minh Le, Truyen Tran

**Abstract**: Large language models exhibit systematic vulnerabilities to adversarial attacks despite extensive safety alignment. We provide a mechanistic analysis revealing that position-dependent gradient weakening during autoregressive training creates signal decay, leading to incomplete safety learning where safety training fails to transform model preferences in later response regions fully. We introduce base-favored tokens -- vocabulary elements where base models assign higher probability than aligned models -- as computational indicators of incomplete safety learning and develop a targeted completion method that addresses undertrained regions through adaptive penalties and hybrid teacher distillation. Experimental evaluation across Llama and Qwen model families demonstrates dramatic improvements in adversarial robustness, with 48--98% reductions in attack success rates while preserving general capabilities. These results establish both a mechanistic understanding and practical solutions for fundamental limitations in safety alignment methodologies.



## **23. AttackVLA: Benchmarking Adversarial and Backdoor Attacks on Vision-Language-Action Models**

cs.CR

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12149v1) [paper-pdf](https://arxiv.org/pdf/2511.12149v1)

**Authors**: Jiayu Li, Yunhan Zhao, Xiang Zheng, Zonghuan Xu, Yige Li, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Vision-Language-Action (VLA) models enable robots to interpret natural-language instructions and perform diverse tasks, yet their integration of perception, language, and control introduces new safety vulnerabilities. Despite growing interest in attacking such models, the effectiveness of existing techniques remains unclear due to the absence of a unified evaluation framework. One major issue is that differences in action tokenizers across VLA architectures hinder reproducibility and fair comparison. More importantly, most existing attacks have not been validated in real-world scenarios. To address these challenges, we propose AttackVLA, a unified framework that aligns with the VLA development lifecycle, covering data construction, model training, and inference. Within this framework, we implement a broad suite of attacks, including all existing attacks targeting VLAs and multiple adapted attacks originally developed for vision-language models, and evaluate them in both simulation and real-world settings. Our analysis of existing attacks reveals a critical gap: current methods tend to induce untargeted failures or static action states, leaving targeted attacks that drive VLAs to perform precise long-horizon action sequences largely unexplored. To fill this gap, we introduce BackdoorVLA, a targeted backdoor attack that compels a VLA to execute an attacker-specified long-horizon action sequence whenever a trigger is present. We evaluate BackdoorVLA in both simulated benchmarks and real-world robotic settings, achieving an average targeted success rate of 58.4% and reaching 100% on selected tasks. Our work provides a standardized framework for evaluating VLA vulnerabilities and demonstrates the potential for precise adversarial manipulation, motivating further research on securing VLA-based embodied systems.



## **24. Explainable Transformer-Based Email Phishing Classification with Adversarial Robustness**

cs.CR

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12085v1) [paper-pdf](https://arxiv.org/pdf/2511.12085v1)

**Authors**: Sajad U P

**Abstract**: Phishing and related cyber threats are becoming more varied and technologically advanced. Among these, email-based phishing remains the most dominant and persistent threat. These attacks exploit human vulnerabilities to disseminate malware or gain unauthorized access to sensitive information. Deep learning (DL) models, particularly transformer-based models, have significantly enhanced phishing mitigation through their contextual understanding of language. However, some recent threats, specifically Artificial Intelligence (AI)-generated phishing attacks, are reducing the overall system resilience of phishing detectors. In response, adversarial training has shown promise against AI-generated phishing threats. This study presents a hybrid approach that uses DistilBERT, a smaller, faster, and lighter version of the BERT transformer model for email classification. Robustness against text-based adversarial perturbations is reinforced using Fast Gradient Method (FGM) adversarial training. Furthermore, the framework integrates the LIME Explainable AI (XAI) technique to enhance the transparency of the DistilBERT architecture. The framework also uses the Flan-T5-small language model from Hugging Face to generate plain-language security narrative explanations for end-users. This combined approach ensures precise phishing classification while providing easily understandable justifications for the model's decisions.



## **25. BackWeak: Backdooring Knowledge Distillation Simply with Weak Triggers and Fine-tuning**

cs.CR

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12046v1) [paper-pdf](https://arxiv.org/pdf/2511.12046v1)

**Authors**: Shanmin Wang, Dongdong Zhao

**Abstract**: Knowledge Distillation (KD) is essential for compressing large models, yet relying on pre-trained "teacher" models downloaded from third-party repositories introduces serious security risks -- most notably backdoor attacks. Existing KD backdoor methods are typically complex and computationally intensive: they employ surrogate student models and simulated distillation to guarantee transferability, and they construct triggers in a way similar to universal adversarial perturbations (UAPs), which being not stealthy in magnitude, inherently exhibit strong adversarial behavior. This work questions whether such complexity is necessary and constructs stealthy "weak" triggers -- imperceptible perturbations that have negligible adversarial effect. We propose BackWeak, a simple, surrogate-free attack paradigm. BackWeak shows that a powerful backdoor can be implanted by simply fine-tuning a benign teacher with a weak trigger using a very small learning rate. We demonstrate that this delicate fine-tuning is sufficient to embed a backdoor that reliably transfers to diverse student architectures during a victim's standard distillation process, yielding high attack success rates. Extensive empirical evaluations on multiple datasets, model architectures, and KD methods show that BackWeak is efficient, simpler, and often more stealthy than previous elaborate approaches. This work calls on researchers studying KD backdoor attacks to pay particular attention to the trigger's stealthiness and its potential adversarial characteristics.



## **26. Robust Bidirectional Associative Memory via Regularization Inspired by the Subspace Rotation Algorithm**

cs.LG

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2511.11902v1) [paper-pdf](https://arxiv.org/pdf/2511.11902v1)

**Authors**: Ci Lin, Tet Yeap, Iluju Kiringa, Biwei Zhang

**Abstract**: Bidirectional Associative Memory (BAM) trained with Bidirectional Backpropagation (B-BP) often suffers from poor robustness and high sensitivity to noise and adversarial attacks. To address these issues, we propose a novel gradient-free training algorithm, the Bidirectional Subspace Rotation Algorithm (B-SRA), which significantly improves the robustness and convergence behavior of BAM. Through comprehensive experiments, we identify two key principles -- orthogonal weight matrices (OWM) and gradient-pattern alignment (GPA) -- as central to enhancing the robustness of BAM. Motivated by these findings, we introduce new regularization strategies into B-BP, resulting in models with greatly improved resistance to corruption and adversarial perturbations. We further conduct an ablation study across different training strategies to determine the most robust configuration and evaluate BAM's performance under a variety of attack scenarios and memory capacities, including 50, 100, and 200 associative pairs. Among all methods, the SAME configuration, which integrates both OWM and GPA, achieves the strongest resilience. Overall, our results demonstrate that B-SRA and the proposed regularization strategies lead to substantially more robust associative memories and open new directions for building resilient neural architectures.



## **27. On the Trade-Off Between Transparency and Security in Adversarial Machine Learning**

cs.LG

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2511.11842v1) [paper-pdf](https://arxiv.org/pdf/2511.11842v1)

**Authors**: Lucas Fenaux, Christopher Srinivasa, Florian Kerschbaum

**Abstract**: Transparency and security are both central to Responsible AI, but they may conflict in adversarial settings. We investigate the strategic effect of transparency for agents through the lens of transferable adversarial example attacks. In transferable adversarial example attacks, attackers maliciously perturb their inputs using surrogate models to fool a defender's target model. These models can be defended or undefended, with both players having to decide which to use. Using a large-scale empirical evaluation of nine attacks across 181 models, we find that attackers are more successful when they match the defender's decision; hence, obscurity could be beneficial to the defender. With game theory, we analyze this trade-off between transparency and security by modeling this problem as both a Nash game and a Stackelberg game, and comparing the expected outcomes. Our analysis confirms that only knowing whether a defender's model is defended or not can sometimes be enough to damage its security. This result serves as an indicator of the general trade-off between transparency and security, suggesting that transparency in AI systems can be at odds with security. Beyond adversarial machine learning, our work illustrates how game-theoretic reasoning can uncover conflicts between transparency and security.



## **28. Incentive Attacks in BTC: Short-Term Revenue Changes and Long-Term Efficiencies**

cs.CR

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2511.11538v1) [paper-pdf](https://arxiv.org/pdf/2511.11538v1)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: Bitcoin's (BTC) Difficulty Adjustment Algorithm (DAA) has been a source of vulnerability for incentive attacks such as selfish mining, block withholding and coin hopping strategies. In this paper, first, we rigorously study the short-term revenue change per hashpower of the adversarial and honest miners for these incentive attacks. To study the long-term effects, we introduce a new efficiency metric defined as the revenue/cost per hashpower per time for the attacker and the honest miners.   Our results indicate that the short-term benefits of intermittent mining strategies are negligible compared to the original selfish mining attack, and in the long-term, selfish mining provides better efficiency. We further demonstrate that a coin hopping strategy between BTC and Bitcoin Cash (BCH) relying on BTC DAA benefits the loyal honest miners of BTC in the same way and to the same extent per unit of computational power as it does the hopper in the short-term. For the long-term, we establish a new boundary between the selfish mining and coin hopping attack, identifying the optimal efficient strategy for each parameter.   For block withholding strategies, it turns out, the honest miners outside the pool profit from the attack, usually even more than the attacker both in the short-term and the long-term. Moreover, a power adjusting withholding attacker does not necessarily observe a profit lag in the short-term. It has been long thought that the profit lag of selfish mining is among the main reasons why such an attack has not been observed in practice. We show that such a barrier does not apply to power adjusting attacks and relatively small pools are at an immediate threat.



## **29. Class-feature Watermark: A Resilient Black-box Watermark Against Model Extraction Attacks**

cs.CR

Accepted by AAAI'26

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.07947v2) [paper-pdf](https://arxiv.org/pdf/2511.07947v2)

**Authors**: Yaxin Xiao, Qingqing Ye, Zi Liang, Haoyang Li, RongHua Li, Huadi Zheng, Haibo Hu

**Abstract**: Machine learning models constitute valuable intellectual property, yet remain vulnerable to model extraction attacks (MEA), where adversaries replicate their functionality through black-box queries. Model watermarking counters MEAs by embedding forensic markers for ownership verification. Current black-box watermarks prioritize MEA survival through representation entanglement, yet inadequately explore resilience against sequential MEAs and removal attacks. Our study reveals that this risk is underestimated because existing removal methods are weakened by entanglement. To address this gap, we propose Watermark Removal attacK (WRK), which circumvents entanglement constraints by exploiting decision boundaries shaped by prevailing sample-level watermark artifacts. WRK effectively reduces watermark success rates by at least 88.79% across existing watermarking benchmarks.   For robust protection, we propose Class-Feature Watermarks (CFW), which improve resilience by leveraging class-level artifacts. CFW constructs a synthetic class using out-of-domain samples, eliminating vulnerable decision boundaries between original domain samples and their artifact-modified counterparts (watermark samples). CFW concurrently optimizes both MEA transferability and post-MEA stability. Experiments across multiple domains show that CFW consistently outperforms prior methods in resilience, maintaining a watermark success rate of at least 70.15% in extracted models even under the combined MEA and WRK distortion, while preserving the utility of protected models.



## **30. A Generative Adversarial Approach to Adversarial Attacks Guided by Contrastive Language-Image Pre-trained Model**

cs.CV

18 pages, 3 figures

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.01317v2) [paper-pdf](https://arxiv.org/pdf/2511.01317v2)

**Authors**: Sampriti Soor, Alik Pramanick, Jothiprakash K, Arijit Sur

**Abstract**: The rapid growth of deep learning has brought about powerful models that can handle various tasks, like identifying images and understanding language. However, adversarial attacks, an unnoticed alteration, can deceive models, leading to inaccurate predictions. In this paper, a generative adversarial attack method is proposed that uses the CLIP model to create highly effective and visually imperceptible adversarial perturbations. The CLIP model's ability to align text and image representation helps incorporate natural language semantics with a guided loss to generate effective adversarial examples that look identical to the original inputs. This integration allows extensive scene manipulation, creating perturbations in multi-object environments specifically designed to deceive multilabel classifiers. Our approach integrates the concentrated perturbation strategy from Saliency-based Auto-Encoder (SSAE) with the dissimilar text embeddings similar to Generative Adversarial Multi-Object Scene Attacks (GAMA), resulting in perturbations that both deceive classification models and maintain high structural similarity to the original images. The model was tested on various tasks across diverse black-box victim models. The experimental results show that our method performs competitively, achieving comparable or superior results to existing techniques, while preserving greater visual fidelity.



## **31. CompressionAttack: Exploiting Prompt Compression as a New Attack Surface in LLM-Powered Agents**

cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2510.22963v3) [paper-pdf](https://arxiv.org/pdf/2510.22963v3)

**Authors**: Zesen Liu, Zhixiang Zhang, Yuchong Xie, Dongdong She

**Abstract**: LLM-powered agents often use prompt compression to reduce inference costs, but this introduces a new security risk. Compression modules, which are optimized for efficiency rather than safety, can be manipulated by adversarial inputs, causing semantic drift and altering LLM behavior. This work identifies prompt compression as a novel attack surface and presents CompressionAttack, the first framework to exploit it. CompressionAttack includes two strategies: HardCom, which uses discrete adversarial edits for hard compression, and SoftCom, which performs latent-space perturbations for soft compression. Experiments on multiple LLMs show up to an average ASR of 83% and 87% in two tasks, while remaining highly stealthy and transferable. Case studies in three practical scenarios confirm real-world impact, and current defenses prove ineffective, highlighting the need for stronger protections.



## **32. A Lightweight Approach for State Machine Replication**

cs.DC

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2509.17771v2) [paper-pdf](https://arxiv.org/pdf/2509.17771v2)

**Authors**: Christian Cachin, Jinfeng Dou, Christian Scheideler, Philipp Schneider

**Abstract**: We present a lightweight solution for state machine replication with commitment certificates. Specifically, we adapt and analyze a median rule for the stabilizing consensus problem [Doerr11] to operate in a client-server setting where arbitrary servers may be blocked adaptively based on past system information. We further extend our protocol by compressing information about committed commands, thus keeping the protocol lightweight, while still enabling clients to easily prove that their commands have indeed been committed on the shared state. Our approach guarantees liveness as long as at most a constant fraction of servers are blocked, ensures safety under any number of blocked servers, and supports fast recovery even after all servers are blocked. In addition to offering near-optimal asymptotic performance in several respects, our method is fully decentralized, unlike other near-optimal solutions that rely on leaders. In particular, our solution is robust against adversaries that target key servers (which captures insider-based denial-of-service attacks), whereas leader-based approaches fail under such a blocking model.



## **33. NeuroStrike: Neuron-Level Attacks on Aligned LLMs**

cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2509.11864v2) [paper-pdf](https://arxiv.org/pdf/2509.11864v2)

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Maximilian Thang, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability.   This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.



## **34. SoK: How Sensor Attacks Disrupt Autonomous Vehicles: An End-to-end Analysis, Challenges, and Missed Threats**

cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2509.11120v4) [paper-pdf](https://arxiv.org/pdf/2509.11120v4)

**Authors**: Qingzhao Zhang, Shaocheng Luo, Z. Morley Mao, Miroslav Pajic, Michael K. Reiter

**Abstract**: Autonomous vehicles, including self driving cars, ground robots, and drones, rely on multi-modal sensor pipelines for safe operation, yet remain vulnerable to adversarial sensor attacks. A critical gap is the lack of a systematic end-to-end view of how sensor induced errors traverse interconnected modules to affect the physical world. To bridge the gap, we provide a comprehensive survey across platforms, sensing modalities, attack methods, and countermeasures. At its core is \Model (\modelAbbr), a graph-based illustrative framework that maps how attacks inject errors, the conditions for their propagation through modules from perception and localization to planning and control, and when they reach physical impact. From the systematic analysis, our study distills 8 key findings that highlight the feasibility challenges of sensor attacks and uncovers 12 previously overlooked attack vectors exploiting inter-module interactions, several of which we validate through proof-of-concept experiments.



## **35. Isolate Trigger: Detecting and Eliminating Adaptive Backdoor Attacks**

cs.CR

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2508.04094v2) [paper-pdf](https://arxiv.org/pdf/2508.04094v2)

**Authors**: Chengrui Sun, Hua Zhang, Haoran Gao, Shang Wang, Zian Tian, Jianjin Zhao, Qi Li, Hongliang Zhu, Zongliang Shen, Anmin Fu

**Abstract**: Deep learning models are widely deployed in various applications but remain vulnerable to stealthy adversarial threats, particularly backdoor attacks. Backdoor models trained on poisoned datasets behave normally with clean inputs but cause mispredictions when a specific trigger is present. Most existing backdoor defenses assume that adversaries only inject one backdoor with small and conspicuous triggers. However, adaptive backdoor that entangle multiple trigger patterns with benign features can effectively bypass existing defenses. To defend against these attacks, we propose Isolate Trigger (IsTr), an accurate and efficient framework for backdoor detection and mitigation. IsTr aims to eliminate the influence of benign features and reverse hidden triggers. IsTr is motivated by the observation that a model's feature extractor focuses more on benign features while its classifier focuses more on trigger patterns. Based on this difference, IsTr designs Steps and Differential-Middle-Slice to resolve the detecting challenge of isolating triggers from benign features. Moreover, IsTr employs unlearning-based repair to remove both attacker-injected and natural backdoors while maintaining model benign accuracy. We extensively evaluate IsTr against six representative backdoor attacks and compare with seven state-of-the-art baseline methods across three real-world applications: digit recognition, face recognition, and traffic sign recognition. In most cases, IsTr reduces detection overhead by an order of magnitude while achieving over 95\% detection accuracy and maintaining the post-repair attack success rate below 3\%, outperforming baseline defenses. IsTr remains robust against various adaptive attacks, even when trigger patterns are heavily entangled with benign features.



## **36. LeakyCLIP: Extracting Training Data from CLIP**

cs.CR

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2508.00756v3) [paper-pdf](https://arxiv.org/pdf/2508.00756v3)

**Authors**: Yunhao Chen, Shujie Wang, Xin Wang, Xingjun Ma

**Abstract**: Understanding the memorization and privacy leakage risks in Contrastive Language--Image Pretraining (CLIP) is critical for ensuring the security of multimodal models. Recent studies have demonstrated the feasibility of extracting sensitive training examples from diffusion models, with conditional diffusion models exhibiting a stronger tendency to memorize and leak information. In this work, we investigate data memorization and extraction risks in CLIP through the lens of CLIP inversion, a process that aims to reconstruct training images from text prompts. To this end, we introduce \textbf{LeakyCLIP}, a novel attack framework designed to achieve high-quality, semantically accurate image reconstruction from CLIP embeddings. We identify three key challenges in CLIP inversion: 1) non-robust features, 2) limited visual semantics in text embeddings, and 3) low reconstruction fidelity. To address these challenges, LeakyCLIP employs 1) adversarial fine-tuning to enhance optimization smoothness, 2) linear transformation-based embedding alignment, and 3) Stable Diffusion-based refinement to improve fidelity. Empirical results demonstrate the superiority of LeakyCLIP, achieving over 258% improvement in Structural Similarity Index Measure (SSIM) for ViT-B-16 compared to baseline methods on LAION-2B subset. Furthermore, we uncover a pervasive leakage risk, showing that training data membership can even be successfully inferred from the metrics of low-fidelity reconstructions. Our work introduces a practical method for CLIP inversion while offering novel insights into the nature and scope of privacy risks in multimodal models.



## **37. Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs**

cs.CL

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2507.22564v2) [paper-pdf](https://arxiv.org/pdf/2507.22564v2)

**Authors**: Xikang Yang, Biyu Zhou, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems.



## **38. LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge**

cs.CR

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2506.09443v2) [paper-pdf](https://arxiv.org/pdf/2506.09443v2)

**Authors**: Songze Li, Chuokun Xu, Jiaying Wang, Xueluan Gong, Chen Chen, Jirui Zhang, Jun Wang, Kwok-Yan Lam, Shouling Ji

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse tasks, driving the development and widespread adoption of LLM-as-a-Judge systems for automated evaluation, including red teaming and benchmarking. However, these systems are susceptible to adversarial attacks that can manipulate evaluation outcomes, raising critical concerns about their robustness and trustworthiness. Existing evaluation methods for LLM-based judges are often fragmented and lack a unified framework for comprehensive robustness assessment. Furthermore, the impact of prompt template design and model selection on judge robustness has rarely been explored, and their performance in real-world deployments remains largely unverified. To address these gaps, we introduce RobustJudge, a fully automated and scalable framework designed to systematically evaluate the robustness of LLM-as-a-Judge systems. Specifically, RobustJudge investigates the effectiveness of 15 attack methods and 7 defense strategies across 12 models (RQ1), examines the impact of prompt template design and model selection (RQ2), and evaluates the security of real-world deployments (RQ3). Our study yields three key findings: (1) LLM-as-a-Judge systems are highly vulnerable to attacks such as PAIR and combined attacks, while defense mechanisms such as re-tokenization and LLM-based detectors can provide enhanced protection; (2) robustness varies substantially across prompt templates (up to 40%); (3) deploying RobustJudge on Alibaba's PAI platform uncovers previously undiscovered vulnerabilities. These results offer practical insights for building trustworthy LLM-as-a-Judge systems.



## **39. Towards Cross-Domain Multi-Targeted Adversarial Attacks**

cs.CV

Under review

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2505.20782v2) [paper-pdf](https://arxiv.org/pdf/2505.20782v2)

**Authors**: Taïga Gonçalves, Tomo Miyazaki, Shinichiro Omachi

**Abstract**: Multi-targeted adversarial attacks aim to mislead classifiers toward specific target classes using a single perturbation generator with a conditional input specifying the desired target class. Existing methods face two key limitations: (1) a single generator supports only a limited number of predefined target classes, and (2) it requires access to the victim model's training data to learn target class semantics. This dependency raises data leakage concerns in practical black-box scenarios where the training data is typically private. To address these limitations, we propose a novel Cross-Domain Multi-Targeted Attack (CD-MTA) that can generate perturbations toward arbitrary target classes, even those that do not exist in the attacker's training data. CD-MTA is trained on a single public dataset but can perform targeted attacks on black-box models trained on different datasets with disjoint and unknown class sets. Our method requires only a single example image that visually represents the desired target class, without relying its label, class distribution or pretrained embeddings. We achieve this through a Feature Injection Module (FIM) and class-agnostic objectives which guide the generator to extract transferable, fine-grained features from the target image without inferring class semantics. Experiments on ImageNet and seven additional datasets show that CD-MTA outperforms existing multi-targeted attack methods on unseen target classes in black-box and cross-domain scenarios. The code is available at https://github.com/tgoncalv/CD-MTA.



## **40. SafeKey: Amplifying Aha-Moment Insights for Safety Reasoning**

cs.AI

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2505.16186v2) [paper-pdf](https://arxiv.org/pdf/2505.16186v2)

**Authors**: Kaiwen Zhou, Xuandong Zhao, Gaowen Liu, Jayanth Srinivasa, Aosong Feng, Dawn Song, Xin Eric Wang

**Abstract**: Large Reasoning Models (LRMs) introduce a new generation paradigm of explicitly reasoning before answering, leading to remarkable improvements in complex tasks. However, they pose great safety risks against harmful queries and adversarial attacks. While recent mainstream safety efforts on LRMs, supervised fine-tuning (SFT), improve safety performance, we find that SFT-aligned models struggle to generalize to unseen jailbreak prompts. After thorough investigation of LRMs' generation, we identify a safety aha moment that can activate safety reasoning and lead to a safe response. This aha moment typically appears in the `key sentence', which follows models' query understanding process and can indicate whether the model will proceed safely. Based on these insights, we propose SafeKey, including two complementary objectives to better activate the safety aha moment in the key sentence: (1) a Dual-Path Safety Head to enhance the safety signal in the model's internal representations before the key sentence, and (2) a Query-Mask Modeling objective to improve the models' attention on its query understanding, which has important safety hints. Experiments across multiple safety benchmarks demonstrate that our methods significantly improve safety generalization to a wide range of jailbreak attacks and out-of-distribution harmful prompts, lowering the average harmfulness rate by 9.6\%, while maintaining general abilities. Our analysis reveals how SafeKey enhances safety by reshaping internal attention and improving the quality of hidden representations.



## **41. Chain-of-Thought Driven Adversarial Scenario Extrapolation for Robust Language Models**

cs.CL

19 pages, 5 figures. Accepted in AAAI 2026

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2505.17089v2) [paper-pdf](https://arxiv.org/pdf/2505.17089v2)

**Authors**: Md Rafi Ur Rashid, Vishnu Asutosh Dasu, Ye Wang, Gang Tan, Shagufta Mehnaz

**Abstract**: Large Language Models (LLMs) exhibit impressive capabilities, but remain susceptible to a growing spectrum of safety risks, including jailbreaks, toxic content, hallucinations, and bias. Existing defenses often address only a single threat type or resort to rigid outright rejection, sacrificing user experience and failing to generalize across diverse and novel attacks. This paper introduces Adversarial Scenario Extrapolation (ASE), a novel inference-time computation framework that leverages Chain-of-Thought (CoT) reasoning to simultaneously enhance LLM robustness and seamlessness. ASE guides the LLM through a self-generative process of contemplating potential adversarial scenarios and formulating defensive strategies before generating a response to the user query. Comprehensive evaluation on four adversarial benchmarks with four latest LLMs shows that ASE achieves near-zero jailbreak attack success rates and minimal toxicity, while slashing outright rejections to <4%. ASE outperforms six state-of-the-art defenses in robustness-seamlessness trade-offs, with 92-99% accuracy on adversarial Q&A and 4-10x lower bias scores. By transforming adversarial perception into an intrinsic cognitive process, ASE sets a new paradigm for secure and natural human-AI interaction.



## **42. Harnessing the Computation Redundancy in ViTs to Boost Adversarial Transferability**

cs.CV

Accepted by NeurIPS 2025

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2504.10804v3) [paper-pdf](https://arxiv.org/pdf/2504.10804v3)

**Authors**: Jiani Liu, Zhiyuan Wang, Zeliang Zhang, Chao Huang, Susan Liang, Yunlong Tang, Chenliang Xu

**Abstract**: Vision Transformers (ViTs) have demonstrated impressive performance across a range of applications, including many safety-critical tasks. However, their unique architectural properties raise new challenges and opportunities in adversarial robustness. In particular, we observe that adversarial examples crafted on ViTs exhibit higher transferability compared to those crafted on CNNs, suggesting that ViTs contain structural characteristics favorable for transferable attacks. In this work, we investigate the role of computational redundancy in ViTs and its impact on adversarial transferability. Unlike prior studies that aim to reduce computation for efficiency, we propose to exploit this redundancy to improve the quality and transferability of adversarial examples. Through a detailed analysis, we identify two forms of redundancy, including the data-level and model-level, that can be harnessed to amplify attack effectiveness. Building on this insight, we design a suite of techniques, including attention sparsity manipulation, attention head permutation, clean token regularization, ghost MoE diversification, and test-time adversarial training. Extensive experiments on the ImageNet-1k dataset validate the effectiveness of our approach, showing that our methods significantly outperform existing baselines in both transferability and generality across diverse model architectures.



## **43. Backdooring CLIP through Concept Confusion**

cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2503.09095v2) [paper-pdf](https://arxiv.org/pdf/2503.09095v2)

**Authors**: Lijie Hu, Junchi Liao, Weimin Lyu, Shaopeng Fu, Tianhao Huang, Shu Yang, Guimin Hu, Di Wang

**Abstract**: Backdoor attacks pose a serious threat to deep learning models by allowing adversaries to implant hidden behaviors that remain dormant on clean inputs but are maliciously triggered at inference. Existing backdoor attack methods typically rely on explicit triggers such as image patches or pixel perturbations, which makes them easier to detect and limits their applicability in complex settings. To address this limitation, we take a different perspective by analyzing backdoor attacks through the lens of concept-level reasoning, drawing on insights from interpretable AI. We show that traditional attacks can be viewed as implicitly manipulating the concepts activated within a model's latent space. This motivates a natural question: can backdoors be built by directly manipulating concepts? To answer this, we propose the Concept Confusion Attack (CCA), a novel framework that designates human-understandable concepts as internal triggers, eliminating the need for explicit input modifications. By relabeling images that strongly exhibit a chosen concept and fine-tuning on this mixed dataset, CCA teaches the model to associate the concept itself with the attacker's target label. Consequently, the presence of the concept alone is sufficient to activate the backdoor, making the attack stealthier and more resistant to existing defenses. Using CLIP as a case study, we show that CCA achieves high attack success rates while preserving clean-task accuracy and evading state-of-the-art defenses.



## **44. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

cs.CY

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2502.12659v4) [paper-pdf](https://arxiv.org/pdf/2502.12659v4)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models (LRMs), such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source reasoning models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on open LRMs is needed. (2) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (3) Safety thinking emerges in the reasoning process of LRMs, but fails frequently against adversarial attacks. (4) The thinking process in R1 models poses greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.



## **45. MOS-Attack: A Scalable Multi-objective Adversarial Attack Framework**

cs.LG

Camera ready version of CVPR 2025

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2501.07251v3) [paper-pdf](https://arxiv.org/pdf/2501.07251v3)

**Authors**: Ping Guo, Cheng Gong, Xi Lin, Fei Liu, Zhichao Lu, Qingfu Zhang, Zhenkun Wang

**Abstract**: Crafting adversarial examples is crucial for evaluating and enhancing the robustness of Deep Neural Networks (DNNs), presenting a challenge equivalent to maximizing a non-differentiable 0-1 loss function.   However, existing single objective methods, namely adversarial attacks focus on a surrogate loss function, do not fully harness the benefits of engaging multiple loss functions, as a result of insufficient understanding of their synergistic and conflicting nature.   To overcome these limitations, we propose the Multi-Objective Set-based Attack (MOS Attack), a novel adversarial attack framework leveraging multiple loss functions and automatically uncovering their interrelations.   The MOS Attack adopts a set-based multi-objective optimization strategy, enabling the incorporation of numerous loss functions without additional parameters.   It also automatically mines synergistic patterns among various losses, facilitating the generation of potent adversarial attacks with fewer objectives.   Extensive experiments have shown that our MOS Attack outperforms single-objective attacks. Furthermore, by harnessing the identified synergistic patterns, MOS Attack continues to show superior results with a reduced number of loss functions. Our code is available at https://github.com/pgg3/MOS-Attack.



## **46. Transferability of Adversarial Attacks in Video-based MLLMs: A Cross-modal Image-to-Video Approach**

cs.CV

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2501.01042v3) [paper-pdf](https://arxiv.org/pdf/2501.01042v3)

**Authors**: Linhao Huang, Xue Jiang, Zhiqiang Wang, Wentao Mo, Xi Xiao, Bo Han, Yongjie Yin, Feng Zheng

**Abstract**: Video-based multimodal large language models (V-MLLMs) have shown vulnerability to adversarial examples in video-text multimodal tasks. However, the transferability of adversarial videos to unseen models - a common and practical real-world scenario - remains unexplored. In this paper, we pioneer an investigation into the transferability of adversarial video samples across V-MLLMs. We find that existing adversarial attack methods face significant limitations when applied in black-box settings for V-MLLMs, which we attribute to the following shortcomings: (1) lacking generalization in perturbing video features, (2) focusing only on sparse key-frames, and (3) failing to integrate multimodal information. To address these limitations and deepen the understanding of V-MLLM vulnerabilities in black-box scenarios, we introduce the Image-to-Video MLLM (I2V-MLLM) attack. In I2V-MLLM, we utilize an image-based multimodal large language model (I-MLLM) as a surrogate model to craft adversarial video samples. Multimodal interactions and spatiotemporal information are integrated to disrupt video representations within the latent space, improving adversarial transferability. Additionally, a perturbation propagation technique is introduced to handle different unknown frame sampling strategies. Experimental results demonstrate that our method can generate adversarial examples that exhibit strong transferability across different V-MLLMs on multiple video-text multimodal tasks. Compared to white-box attacks on these models, our black-box attacks (using BLIP-2 as a surrogate model) achieve competitive performance, with average attack success rate (AASR) of 57.98% on MSVD-QA and 58.26% on MSRVTT-QA for Zero-Shot VideoQA tasks, respectively.



## **47. Human-in-the-Loop Generation of Adversarial Texts: A Case Study on Tibetan Script**

cs.CL

Camera-Ready Version; Accepted at IJCNLP-AACL 2025 Demo

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2412.12478v5) [paper-pdf](https://arxiv.org/pdf/2412.12478v5)

**Authors**: Xi Cao, Yuan Sun, Jiajun Li, Quzong Gesang, Nuo Qun, Tashi Nyima

**Abstract**: DNN-based language models excel across various NLP tasks but remain highly vulnerable to textual adversarial attacks. While adversarial text generation is crucial for NLP security, explainability, evaluation, and data augmentation, related work remains overwhelmingly English-centric, leaving the problem of constructing high-quality and sustainable adversarial robustness benchmarks for lower-resourced languages both difficult and understudied. First, method customization for lower-resourced languages is complicated due to linguistic differences and limited resources. Second, automated attacks are prone to generating invalid or ambiguous adversarial texts. Last but not least, language models continuously evolve and may be immune to parts of previously generated adversarial texts. To address these challenges, we introduce HITL-GAT, an interactive system based on a general approach to human-in-the-loop generation of adversarial texts. Additionally, we demonstrate the utility of HITL-GAT through a case study on Tibetan script, employing three customized adversarial text generation methods and establishing its first adversarial robustness benchmark, providing a valuable reference for other lower-resourced languages.



## **48. What You See Is Not Always What You Get: Evaluating GPT's Comprehension of Source Code**

cs.SE

This work has been accepted at APSEC 2025

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2412.08098v3) [paper-pdf](https://arxiv.org/pdf/2412.08098v3)

**Authors**: Jiawen Wen, Bangshuo Zhu, Huaming Chen

**Abstract**: Recent studies have demonstrated outstanding capabilities of large language models (LLMs) in software engineering tasks, including code generation and comprehension. While LLMs have shown significant potential in assisting with coding, LLMs are vulnerable to adversarial attacks. In this paper, we investigate the vulnerability of LLMs to imperceptible attacks. This class of attacks manipulate source code at the character level, which renders the changes invisible to human reviewers yet effective in misleading LLMs' behaviour. We devise these attacks into four distinct categories and analyse their impacts on code analysis and comprehension tasks. These four types of imperceptible character attacks include coding reordering, invisible coding characters, code deletions, and code homoglyphs. To assess the robustness of state-of-the-art LLMs, we present a systematic evaluation across multiple models using both perturbed and clean code snippets. Two evaluation metrics, model confidence using log probabilities of response and response correctness, are introduced. The results reveal that LLMs are susceptible to imperceptible coding perturbations, with varying degrees of degradation highlighted across different LLMs. Furthermore, we observe a consistent negative correlation between perturbation magnitude and model performance. These results highlight the urgent need for robust LLMs capable of manoeuvring behaviours under imperceptible adversarial conditions.



## **49. Exploiting Missing Data Remediation Strategies using Adversarial Missingness Attacks**

cs.LG

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2409.04407v2) [paper-pdf](https://arxiv.org/pdf/2409.04407v2)

**Authors**: Deniz Koyuncu, Alex Gittens, Bülent Yener, Moti Yung

**Abstract**: Adversarial Missingness (AM) attacks aim to manipulate model fitting by carefully engineering a missing data problem to achieve a specific malicious objective. AM attacks are significantly different from prior data poisoning attacks in that no malicious data inserted and no data is maliciously perturbed. Current AM attacks are feasible only under the assumption that the modeler (victim) uses full-information maximum likelihood methods to handle missingness. This work aims to remedy this limitation of AM attacks; in the approach taken here, the adversary achieves their goal by solving a bi-level optimization problem to engineer the adversarial missingness mechanism, where the lower level problem incorporates a differentiable approximation of the targeted missingness remediation technique. As instantiations of this framework, AM attacks are provided for three popular techniques: (i) complete case analysis, (ii) mean imputation, and (iii) regression-based imputation for general empirical risk minimization (ERM) problems. Experiments on real-world data show that AM attacks are successful with modest levels of missingness (less than 20%). Furthermore, we show on the real-world Twins dataset that AM attacks can manipulate the estimated average treatment effect (ATE) as an instance of the general ERM problems: the adversary succeeds in not only reversing the sign, but also in substantially inflating the ATE values from a true value of -1.61% to a manipulated one as high as 10%. These experimental results hold when the ATE is calculated using multiple regression-based estimators with different architectures, even when the adversary is restricted to modifying only a subset of the training data.



## **50. DiffProtect: Generate Adversarial Examples with Diffusion Models for Facial Privacy Protection**

cs.CV

Code is at https://github.com/joellliu/DiffProtect/

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2305.13625v3) [paper-pdf](https://arxiv.org/pdf/2305.13625v3)

**Authors**: Jiang Liu, Chun Pong Lau, Zhongliang Guo, Yuxiang Guo, Zhaoyang Wang, Rama Chellappa

**Abstract**: The increasingly pervasive facial recognition (FR) systems raise serious concerns about personal privacy, especially for billions of users who have publicly shared their photos on social media. Several attempts have been made to protect individuals from being identified by unauthorized FR systems utilizing adversarial attacks to generate encrypted face images. However, existing methods suffer from poor visual quality or low attack success rates, which limit their utility. Recently, diffusion models have achieved tremendous success in image generation. In this work, we ask: can diffusion models be used to generate adversarial examples to improve both visual quality and attack performance? We propose DiffProtect, which utilizes a diffusion autoencoder to generate semantically meaningful perturbations on FR systems. Extensive experiments demonstrate that DiffProtect produces more natural-looking encrypted images than state-of-the-art methods while achieving significantly higher attack success rates, e.g., 24.5% and 25.1% absolute improvements on the CelebA-HQ and FFHQ datasets.



