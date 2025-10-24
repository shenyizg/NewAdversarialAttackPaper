# Latest Adversarial Attack Papers
**update at 2025-10-24 10:13:48**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Tex-ViT: A Generalizable, Robust, Texture-based dual-branch cross-attention deepfake detector**

cs.CV

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2408.16892v2) [paper-pdf](http://arxiv.org/pdf/2408.16892v2)

**Authors**: Deepak Dagar, Dinesh Kumar Vishwakarma

**Abstract**: Deepfakes, which employ GAN to produce highly realistic facial modification, are widely regarded as the prevailing method. Traditional CNN have been able to identify bogus media, but they struggle to perform well on different datasets and are vulnerable to adversarial attacks due to their lack of robustness. Vision transformers have demonstrated potential in the realm of image classification problems, but they require enough training data. Motivated by these limitations, this publication introduces Tex-ViT (Texture-Vision Transformer), which enhances CNN features by combining ResNet with a vision transformer. The model combines traditional ResNet features with a texture module that operates in parallel on sections of ResNet before each down-sampling operation. The texture module then serves as an input to the dual branch of the cross-attention vision transformer. It specifically focuses on improving the global texture module, which extracts feature map correlation. Empirical analysis reveals that fake images exhibit smooth textures that do not remain consistent over long distances in manipulations. Experiments were performed on different categories of FF++, such as DF, f2f, FS, and NT, together with other types of GAN datasets in cross-domain scenarios. Furthermore, experiments also conducted on FF++, DFDCPreview, and Celeb-DF dataset underwent several post-processing situations, such as blurring, compression, and noise. The model surpassed the most advanced models in terms of generalization, achieving a 98% accuracy in cross-domain scenarios. This demonstrates its ability to learn the shared distinguishing textural characteristics in the manipulated samples. These experiments provide evidence that the proposed model is capable of being applied to various situations and is resistant to many post-processing procedures.



## **2. AdaDoS: Adaptive DoS Attack via Deep Adversarial Reinforcement Learning in SDN**

cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20566v1) [paper-pdf](http://arxiv.org/pdf/2510.20566v1)

**Authors**: Wei Shao, Yuhao Wang, Rongguang He, Muhammad Ejaz Ahmed, Seyit Camtepe

**Abstract**: Existing defence mechanisms have demonstrated significant effectiveness in mitigating rule-based Denial-of-Service (DoS) attacks, leveraging predefined signatures and static heuristics to identify and block malicious traffic. However, the emergence of AI-driven techniques presents new challenges to SDN security, potentially compromising the efficacy of existing defence mechanisms. In this paper, we introduce~AdaDoS, an adaptive attack model that disrupt network operations while evading detection by existing DoS-based detectors through adversarial reinforcement learning (RL). Specifically, AdaDoS models the problem as a competitive game between an attacker, whose goal is to obstruct network traffic without being detected, and a detector, which aims to identify malicious traffic. AdaDoS can solve this game by dynamically adjusting its attack strategy based on feedback from the SDN and the detector. Additionally, recognising that attackers typically have less information than defenders, AdaDoS formulates the DoS-like attack as a partially observed Markov decision process (POMDP), with the attacker having access only to delay information between attacker and victim nodes. We address this challenge with a novel reciprocal learning module, where the student agent, with limited observations, enhances its performance by learning from the teacher agent, who has full observational capabilities in the SDN environment. AdaDoS represents the first application of RL to develop DoS-like attack sequences, capable of adaptively evading both machine learning-based and rule-based DoS-like attack detectors.



## **3. HauntAttack: When Attack Follows Reasoning as a Shadow**

cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2506.07031v4) [paper-pdf](http://arxiv.org/pdf/2506.07031v4)

**Authors**: Jingyuan Ma, Rui Li, Zheng Li, Junfeng Liu, Heming Xia, Lei Sha, Zhifang Sui

**Abstract**: Emerging Large Reasoning Models (LRMs) consistently excel in mathematical and reasoning tasks, showcasing remarkable capabilities. However, the enhancement of reasoning abilities and the exposure of internal reasoning processes introduce new safety vulnerabilities. A critical question arises: when reasoning becomes intertwined with harmfulness, will LRMs become more vulnerable to jailbreaks in reasoning mode? To investigate this, we introduce HauntAttack, a novel and general-purpose black-box adversarial attack framework that systematically embeds harmful instructions into reasoning questions. Specifically, we modify key reasoning conditions in existing questions with harmful instructions, thereby constructing a reasoning pathway that guides the model step by step toward unsafe outputs. We evaluate HauntAttack on 11 LRMs and observe an average attack success rate of 70\%, achieving up to 12 percentage points of absolute improvement over the strongest prior baseline. Our further analysis reveals that even advanced safety-aligned models remain highly susceptible to reasoning-based attacks, offering insights into the urgent challenge of balancing reasoning capability and safety in future model development.



## **4. Distributional Adversarial Attacks and Training in Deep Hedging**

math.OC

Camera-ready version (accepted at NeurIPS 2025  https://neurips.cc/virtual/2025/poster/115434)

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2508.14757v2) [paper-pdf](http://arxiv.org/pdf/2508.14757v2)

**Authors**: Guangyi He, Tobias Sutter, Lukas Gonon

**Abstract**: In this paper, we study the robustness of classical deep hedging strategies under distributional shifts by leveraging the concept of adversarial attacks. We first demonstrate that standard deep hedging models are highly vulnerable to small perturbations in the input distribution, resulting in significant performance degradation. Motivated by this, we propose an adversarial training framework tailored to increase the robustness of deep hedging strategies. Our approach extends pointwise adversarial attacks to the distributional setting and introduces a computationally tractable reformulation of the adversarial optimization problem over a Wasserstein ball. This enables the efficient training of hedging strategies that are resilient to distributional perturbations. Through extensive numerical experiments, we show that adversarially trained deep hedging strategies consistently outperform their classical counterparts in terms of out-of-sample performance and resilience to model misspecification. Additional results indicate that the robust strategies maintain reliable performance on real market data and remain effective during periods of market change. Our findings establish a practical and effective framework for robust deep hedging under realistic market uncertainties.



## **5. GUIDE: Enhancing Gradient Inversion Attacks in Federated Learning with Denoising Models**

cs.CR

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.17621v2) [paper-pdf](http://arxiv.org/pdf/2510.17621v2)

**Authors**: Vincenzo Carletti, Pasquale Foggia, Carlo Mazzocca, Giuseppe Parrella, Mario Vento

**Abstract**: Federated Learning (FL) enables collaborative training of Machine Learning (ML) models across multiple clients while preserving their privacy. Rather than sharing raw data, federated clients transmit locally computed updates to train the global model. Although this paradigm should provide stronger privacy guarantees than centralized ML, client updates remain vulnerable to privacy leakage. Adversaries can exploit them to infer sensitive properties about the training data or even to reconstruct the original inputs via Gradient Inversion Attacks (GIAs). Under the honest-butcurious threat model, GIAs attempt to reconstruct training data by reversing intermediate updates using optimizationbased techniques. We observe that these approaches usually reconstruct noisy approximations of the original inputs, whose quality can be enhanced with specialized denoising models. This paper presents Gradient Update Inversion with DEnoising (GUIDE), a novel methodology that leverages diffusion models as denoising tools to improve image reconstruction attacks in FL. GUIDE can be integrated into any GIAs that exploits surrogate datasets, a widely adopted assumption in GIAs literature. We comprehensively evaluate our approach in two attack scenarios that use different FL algorithms, models, and datasets. Our results demonstrate that GUIDE integrates seamlessly with two state-ofthe- art GIAs, substantially improving reconstruction quality across multiple metrics. Specifically, GUIDE achieves up to 46% higher perceptual similarity, as measured by the DreamSim metric.



## **6. GhostEI-Bench: Do Mobile Agents Resilience to Environmental Injection in Dynamic On-Device Environments?**

cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20333v1) [paper-pdf](http://arxiv.org/pdf/2510.20333v1)

**Authors**: Chiyu Chen, Xinhao Song, Yunkai Chai, Yang Yao, Haodong Zhao, Lijun Li, Jie Li, Yan Teng, Gongshen Liu, Yingchun Wang

**Abstract**: Vision-Language Models (VLMs) are increasingly deployed as autonomous agents to navigate mobile graphical user interfaces (GUIs). Operating in dynamic on-device ecosystems, which include notifications, pop-ups, and inter-app interactions, exposes them to a unique and underexplored threat vector: environmental injection. Unlike prompt-based attacks that manipulate textual instructions, environmental injection corrupts an agent's visual perception by inserting adversarial UI elements (for example, deceptive overlays or spoofed notifications) directly into the GUI. This bypasses textual safeguards and can derail execution, causing privacy leakage, financial loss, or irreversible device compromise. To systematically evaluate this threat, we introduce GhostEI-Bench, the first benchmark for assessing mobile agents under environmental injection attacks within dynamic, executable environments. Moving beyond static image-based assessments, GhostEI-Bench injects adversarial events into realistic application workflows inside fully operational Android emulators and evaluates performance across critical risk scenarios. We further propose a judge-LLM protocol that conducts fine-grained failure analysis by reviewing the agent's action trajectory alongside the corresponding screenshot sequence, pinpointing failure in perception, recognition, or reasoning. Comprehensive experiments on state-of-the-art agents reveal pronounced vulnerability to deceptive environmental cues: current models systematically fail to perceive and reason about manipulated UIs. GhostEI-Bench provides a framework for quantifying and mitigating this emerging threat, paving the way toward more robust and secure embodied agents.



## **7. Enhancing Security in Deep Reinforcement Learning: A Comprehensive Survey on Adversarial Attacks and Defenses**

cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20314v1) [paper-pdf](http://arxiv.org/pdf/2510.20314v1)

**Authors**: Wu Yichao, Wang Yirui, Ding Panpan, Wang Hailong, Zhu Bingqian, Liu Chun

**Abstract**: With the wide application of deep reinforcement learning (DRL) techniques in complex fields such as autonomous driving, intelligent manufacturing, and smart healthcare, how to improve its security and robustness in dynamic and changeable environments has become a core issue in current research. Especially in the face of adversarial attacks, DRL may suffer serious performance degradation or even make potentially dangerous decisions, so it is crucial to ensure their stability in security-sensitive scenarios. In this paper, we first introduce the basic framework of DRL and analyze the main security challenges faced in complex and changing environments. In addition, this paper proposes an adversarial attack classification framework based on perturbation type and attack target and reviews the mainstream adversarial attack methods against DRL in detail, including various attack methods such as perturbation state space, action space, reward function and model space. To effectively counter the attacks, this paper systematically summarizes various current robustness training strategies, including adversarial training, competitive training, robust learning, adversarial detection, defense distillation and other related defense techniques, we also discuss the advantages and shortcomings of these methods in improving the robustness of DRL. Finally, this paper looks into the future research direction of DRL in adversarial environments, emphasizing the research needs in terms of improving generalization, reducing computational complexity, and enhancing scalability and explainability, aiming to provide valuable references and directions for researchers.



## **8. Crafting Imperceptible On-Manifold Adversarial Attacks for Tabular Data**

cs.LG

39 pages

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2507.10998v2) [paper-pdf](http://arxiv.org/pdf/2507.10998v2)

**Authors**: Zhipeng He, Alexander Stevens, Chun Ouyang, Johannes De Smedt, Alistair Barros, Catarina Moreira

**Abstract**: Adversarial attacks on tabular data present unique challenges due to the heterogeneous nature of mixed categorical and numerical features. Unlike images where pixel perturbations maintain visual similarity, tabular data lacks intuitive similarity metrics, making it difficult to define imperceptible modifications. Additionally, traditional gradient-based methods prioritise $\ell_p$-norm constraints, often producing adversarial examples that deviate from the original data distributions. To address this, we propose a latent-space perturbation framework using a mixed-input Variational Autoencoder (VAE) to generate statistically consistent adversarial examples. The proposed VAE integrates categorical embeddings and numerical features into a unified latent manifold, enabling perturbations that preserve statistical consistency. We introduce In-Distribution Success Rate (IDSR) to jointly evaluate attack effectiveness and distributional alignment. Evaluation across six publicly available datasets and three model architectures demonstrates that our method achieves substantially lower outlier rates and more consistent performance compared to traditional input-space attacks and other VAE-based methods adapted from image domain approaches, achieving substantially lower outlier rates and higher IDSR across six datasets and three model architectures. Our comprehensive analyses of hyperparameter sensitivity, sparsity control, and generative architecture demonstrate that the effectiveness of VAE-based attacks depends strongly on reconstruction quality and the availability of sufficient training data. When these conditions are met, the proposed framework achieves superior practical utility and stability compared with input-space methods. This work underscores the importance of maintaining on-manifold perturbations for generating realistic and robust adversarial examples in tabular domains.



## **9. Beyond Text: Multimodal Jailbreaking of Vision-Language and Audio Models through Perceptually Simple Transformations**

cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20223v1) [paper-pdf](http://arxiv.org/pdf/2510.20223v1)

**Authors**: Divyanshu Kumar, Shreyas Jena, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable progress, yet remain critically vulnerable to adversarial attacks that exploit weaknesses in cross-modal processing. We present a systematic study of multimodal jailbreaks targeting both vision-language and audio-language models, showing that even simple perceptual transformations can reliably bypass state-of-the-art safety filters. Our evaluation spans 1,900 adversarial prompts across three high-risk safety categories harmful content, CBRN (Chemical, Biological, Radiological, Nuclear), and CSEM (Child Sexual Exploitation Material) tested against seven frontier models. We explore the effectiveness of attack techniques on MLLMs, including FigStep-Pro (visual keyword decomposition), Intelligent Masking (semantic obfuscation), and audio perturbations (Wave-Echo, Wave-Pitch, Wave-Speed). The results reveal severe vulnerabilities: models with almost perfect text-only safety (0\% ASR) suffer >75\% attack success under perceptually modified inputs, with FigStep-Pro achieving up to 89\% ASR in Llama-4 variants. Audio-based attacks further uncover provider-specific weaknesses, with even basic modality transfer yielding 25\% ASR for technical queries. These findings expose a critical gap between text-centric alignment and multimodal threats, demonstrating that current safeguards fail to generalize across cross-modal attacks. The accessibility of these attacks, which require minimal technical expertise, suggests that robust multimodal AI safety will require a paradigm shift toward broader semantic-level reasoning to mitigate possible risks.



## **10. TRUST: A Decentralized Framework for Auditing Large Language Model Reasoning**

cs.AI

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20188v1) [paper-pdf](http://arxiv.org/pdf/2510.20188v1)

**Authors**: Morris Yu-Chao Huang, Zhen Tan, Mohan Zhang, Pingzhi Li, Zhuo Zhang, Tianlong Chen

**Abstract**: Large Language Models generate complex reasoning chains that reveal their decision-making, yet verifying the faithfulness and harmlessness of these intermediate steps remains a critical unsolved problem. Existing auditing methods are centralized, opaque, and hard to scale, creating significant risks for deploying proprietary models in high-stakes domains. We identify four core challenges: (1) Robustness: Centralized auditors are single points of failure, prone to bias or attacks. (2) Scalability: Reasoning traces are too long for manual verification. (3) Opacity: Closed auditing undermines public trust. (4) Privacy: Exposing full reasoning risks model theft or distillation. We propose TRUST, a transparent, decentralized auditing framework that overcomes these limitations via: (1) A consensus mechanism among diverse auditors, guaranteeing correctness under up to $30\%$ malicious participants. (2) A hierarchical DAG decomposition of reasoning traces, enabling scalable, parallel auditing. (3) A blockchain ledger that records all verification decisions for public accountability. (4) Privacy-preserving segmentation, sharing only partial reasoning steps to protect proprietary logic. We provide theoretical guarantees for the security and economic incentives of the TRUST framework. Experiments across multiple LLMs (GPT-OSS, DeepSeek-r1, Qwen) and reasoning tasks (math, medical, science, humanities) show TRUST effectively detects reasoning flaws and remains robust against adversarial auditors. Our work pioneers decentralized AI auditing, offering a practical path toward safe and trustworthy LLM deployment.



## **11. Active Localization of Close-range Adversarial Acoustic Sources for Underwater Data Center Surveillance**

eess.SP

12 pages, V1

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20122v1) [paper-pdf](http://arxiv.org/pdf/2510.20122v1)

**Authors**: Adnan Abdullah, David Blow, Sara Rampazzi, Md Jahidul Islam

**Abstract**: Underwater data infrastructures offer natural cooling and enhanced physical security compared to terrestrial facilities, but are susceptible to acoustic injection attacks that can disrupt data integrity and availability. This work presents a comprehensive surveillance framework for localizing and tracking close-range adversarial acoustic sources targeting offshore infrastructures, particularly underwater data centers (UDCs). We propose a heterogeneous receiver configuration comprising a fixed hydrophone mounted on the facility and a mobile hydrophone deployed on a dedicated surveillance robot. While using enough arrays of static hydrophones covering large infrastructures is not feasible in practice, off-the-shelf approaches based on time difference of arrival (TDOA) and frequency difference of arrival (FDOA) filtering fail to generalize for this dynamic configuration. To address this, we formulate a Locus-Conditioned Maximum A-Posteriori (LC-MAP) scheme to generate acoustically informed and geometrically consistent priors, ensuring a physically plausible initial state for a joint TDOA-FDOA filtering. We integrate this into an unscented Kalman filtering (UKF) pipeline, which provides reliable convergence under nonlinearity and measurement noise. Extensive Monte Carlo analyses, Gazebo-based physics simulations, and field trials demonstrate that the proposed framework can reliably estimate the 3D position and velocity of an adversarial acoustic attack source in real time. It achieves sub-meter localization accuracy and over 90% success rates, with convergence times nearly halved compared to baseline methods. Overall, this study establishes a geometry-aware, real-time approach for acoustic threat localization, advancing autonomous surveillance capabilities of underwater infrastructures.



## **12. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

cs.CL

Accepted to NAACL 2025 Main (Oral)

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2410.18469v5) [paper-pdf](http://arxiv.org/pdf/2410.18469v5)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety. Our code is available at: https://github.com/SunChungEn/ADV-LLM



## **13. Bridging Symmetry and Robustness: On the Role of Equivariance in Enhancing Adversarial Robustness**

cs.LG

Accepted for the proceedings of 39th Conference on Neural Information  Processing Systems (NeurIPS 2025)

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.16171v2) [paper-pdf](http://arxiv.org/pdf/2510.16171v2)

**Authors**: Longwei Wang, Ifrat Ikhtear Uddin, KC Santosh, Chaowei Zhang, Xiao Qin, Yang Zhou

**Abstract**: Adversarial examples reveal critical vulnerabilities in deep neural networks by exploiting their sensitivity to imperceptible input perturbations. While adversarial training remains the predominant defense strategy, it often incurs significant computational cost and may compromise clean-data accuracy. In this work, we investigate an architectural approach to adversarial robustness by embedding group-equivariant convolutions-specifically, rotation- and scale-equivariant layers-into standard convolutional neural networks (CNNs). These layers encode symmetry priors that align model behavior with structured transformations in the input space, promoting smoother decision boundaries and greater resilience to adversarial attacks. We propose and evaluate two symmetry-aware architectures: a parallel design that processes standard and equivariant features independently before fusion, and a cascaded design that applies equivariant operations sequentially. Theoretically, we demonstrate that such models reduce hypothesis space complexity, regularize gradients, and yield tighter certified robustness bounds under the CLEVER (Cross Lipschitz Extreme Value for nEtwork Robustness) framework. Empirically, our models consistently improve adversarial robustness and generalization across CIFAR-10, CIFAR-100, and CIFAR-10C under both FGSM and PGD attacks, without requiring adversarial training. These findings underscore the potential of symmetry-enforcing architectures as efficient and principled alternatives to data augmentation-based defenses.



## **14. JaiLIP: Jailbreaking Vision-Language Models via Loss Guided Image Perturbation**

cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2509.21401v2) [paper-pdf](http://arxiv.org/pdf/2509.21401v2)

**Authors**: Md Jueal Mia, M. Hadi Amini

**Abstract**: Vision-Language Models (VLMs) have remarkable abilities in generating multimodal reasoning tasks. However, potential misuse or safety alignment concerns of VLMs have increased significantly due to different categories of attack vectors. Among various attack vectors, recent studies have demonstrated that image-based perturbations are particularly effective in generating harmful outputs. In the literature, many existing techniques have been proposed to jailbreak VLMs, leading to unstable performance and visible perturbations. In this study, we propose Jailbreaking with Loss-guided Image Perturbation (JaiLIP), a jailbreaking attack in the image space that minimizes a joint objective combining the mean squared error (MSE) loss between clean and adversarial image with the models harmful-output loss. We evaluate our proposed method on VLMs using standard toxicity metrics from Perspective API and Detoxify. Experimental results demonstrate that our method generates highly effective and imperceptible adversarial images, outperforming existing methods in producing toxicity. Moreover, we have evaluated our method in the transportation domain to demonstrate the attacks practicality beyond toxic text generation in specific domain. Our findings emphasize the practical challenges of image-based jailbreak attacks and the need for efficient defense mechanisms for VLMs.



## **15. Sharp Gaussian approximations for Decentralized Federated Learning**

stat.ML

Accepted as Spotlight, NeurIPS'25, Main Conference Track

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2505.08125v2) [paper-pdf](http://arxiv.org/pdf/2505.08125v2)

**Authors**: Soham Bonnerjee, Sayar Karmakar, Wei Biao Wu

**Abstract**: Federated Learning has gained traction in privacy-sensitive collaborative environments, with local SGD emerging as a key optimization method in decentralized settings. While its convergence properties are well-studied, asymptotic statistical guarantees beyond convergence remain limited. In this paper, we present two generalized Gaussian approximation results for local SGD and explore their implications. First, we prove a Berry-Esseen theorem for the final local SGD iterates, enabling valid multiplier bootstrap procedures. Second, motivated by robustness considerations, we introduce two distinct time-uniform Gaussian approximations for the entire trajectory of local SGD. The time-uniform approximations support Gaussian bootstrap-based tests for detecting adversarial attacks. Extensive simulations are provided to support our theoretical results.



## **16. QORE : Quantum Secure 5G/B5G Core**

cs.CR

23 pages

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19982v1) [paper-pdf](http://arxiv.org/pdf/2510.19982v1)

**Authors**: Vipin Rathi, Lakshya Chopra, Rudraksh Rawal, Nitin Rajput, Shiva Valia, Madhav Aggarwal, Aditya Gairola

**Abstract**: Quantum computing is reshaping the security landscape of modern telecommunications. The cryptographic foundations that secure todays 5G systems, including RSA, Elliptic Curve Cryptography (ECC), and Diffie-Hellman (DH), are all susceptible to attacks enabled by Shors algorithm. Protecting 5G networks against future quantum adversaries has therefore become an urgent engineering and research priority. In this paper we introduce QORE, a quantum-secure 5G and Beyond 5G (B5G) Core framework that provides a clear pathway for transitioning both the 5G Core Network Functions and User Equipment (UE) to Post-Quantum Cryptography (PQC). The framework uses the NIST-standardized lattice-based algorithms Module-Lattice Key Encapsulation Mechanism (ML-KEM) and Module-Lattice Digital Signature Algorithm (ML-DSA) and applies them across the 5G Service-Based Architecture (SBA). A Hybrid PQC (HPQC) configuration is also proposed, combining classical and quantum-safe primitives to maintain interoperability during migration. Experimental validation shows that ML-KEM achieves quantum security with minor performance overhead, meeting the low-latency and high-throughput requirements of carrier-grade 5G systems. The proposed roadmap aligns with ongoing 3GPP SA3 and SA5 study activities on the security and management of post-quantum networks as well as with NIST PQC standardization efforts, providing practical guidance for mitigating quantum-era risks while safeguarding long-term confidentiality and integrity of network data.



## **17. Towards Strong Certified Defense with Universal Asymmetric Randomization**

cs.LG

Accepted by CSF 2026, 39th IEEE Computer Security Foundations  Symposium

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19977v1) [paper-pdf](http://arxiv.org/pdf/2510.19977v1)

**Authors**: Hanbin Hong, Ashish Kundu, Ali Payani, Binghui Wang, Yuan Hong

**Abstract**: Randomized smoothing has become essential for achieving certified adversarial robustness in machine learning models. However, current methods primarily use isotropic noise distributions that are uniform across all data dimensions, such as image pixels, limiting the effectiveness of robustness certification by ignoring the heterogeneity of inputs and data dimensions. To address this limitation, we propose UCAN: a novel technique that \underline{U}niversally \underline{C}ertifies adversarial robustness with \underline{A}nisotropic \underline{N}oise. UCAN is designed to enhance any existing randomized smoothing method, transforming it from symmetric (isotropic) to asymmetric (anisotropic) noise distributions, thereby offering a more tailored defense against adversarial attacks. Our theoretical framework is versatile, supporting a wide array of noise distributions for certified robustness in different $\ell_p$-norms and applicable to any arbitrary classifier by guaranteeing the classifier's prediction over perturbed inputs with provable robustness bounds through tailored noise injection. Additionally, we develop a novel framework equipped with three exemplary noise parameter generators (NPGs) to optimally fine-tune the anisotropic noise parameters for different data dimensions, allowing for pursuing different levels of robustness enhancements in practice.Empirical evaluations underscore the significant leap in UCAN's performance over existing state-of-the-art methods, demonstrating up to $182.6\%$ improvement in certified accuracy at large certified radii on MNIST, CIFAR10, and ImageNet datasets.\footnote{Code is anonymously available at \href{https://github.com/youbin2014/UCAN/}{https://github.com/youbin2014/UCAN/}}



## **18. Q-RAN: Quantum-Resilient O-RAN Architecture**

cs.CR

23 pages

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19968v1) [paper-pdf](http://arxiv.org/pdf/2510.19968v1)

**Authors**: Vipin Rathi, Lakshya Chopra, Madhav Agarwal, Nitin Rajput, Kriish Sharma, Sushant Mundepi, Shivam Gangwar, Rudraksh Rawal, Jishan

**Abstract**: The telecommunications industry faces a dual transformation: the architectural shift toward Open Radio Access Networks (O-RAN) and the emerging threat from quantum computing. O-RAN disaggregated, multi-vendor architecture creates a larger attack surface vulnerable to crypt-analytically relevant quantum computers(CRQCs) that will break current public key cryptography. The Harvest Now, Decrypt Later (HNDL) attack strategy makes this threat immediate, as adversaries can intercept encrypted data today for future decryption. This paper presents Q-RAN, a comprehensive quantum-resistant security framework for O-RAN networks using NIST-standardized Post-Quantum Cryptography (PQC). We detail the implementation of ML-KEM (FIPS 203) and ML-DSA (FIPS 204), integrated with Quantum Random Number Generators (QRNG) for cryptographic entropy. The solution deploys PQ-IPsec, PQ-DTLS, and PQ-mTLS protocols across all O-RAN interfaces, anchored by a centralized Post-Quantum Certificate Authority (PQ-CA) within the SMO framework. This work provides a complete roadmap for securing disaggregated O-RAN ecosystems against quantum adversaries.



## **19. Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLM**

cs.LG

Accepted by Neurips 2025

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2505.24379v3) [paper-pdf](http://arxiv.org/pdf/2505.24379v3)

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Zhiwei Steven Wu

**Abstract**: Large Language Models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard for mitigating privacy risks in deployment. In this paper, we revisit this assumption in a practical deployment setting where both the pre- and post-unlearning logits API are exposed, such as in open-weight scenarios. Targeting this setting, we introduce a novel data extraction attack that leverages signals from the pre-unlearning model to guide the post-unlearning model, uncovering patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage during real-world deployments, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints. Code is publicly available at: https://github.com/Nicholas0228/unlearned_data_extraction_llm.



## **20. Are Modern Speech Enhancement Systems Vulnerable to Adversarial Attacks?**

eess.AS

Copyright 2026 IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2509.21087v2) [paper-pdf](http://arxiv.org/pdf/2509.21087v2)

**Authors**: Rostislav Makarov, Lea Schönherr, Timo Gerkmann

**Abstract**: Machine learning approaches for speech enhancement are becoming increasingly expressive, enabling ever more powerful modifications of input signals. In this paper, we demonstrate that this expressiveness introduces a vulnerability: advanced speech enhancement models can be susceptible to adversarial attacks. Specifically, we show that adversarial noise, carefully crafted and psychoacoustically masked by the original input, can be injected such that the enhanced speech output conveys an entirely different semantic meaning. We experimentally verify that contemporary predictive speech enhancement models can indeed be manipulated in this way. Furthermore, we highlight that diffusion models with stochastic samplers exhibit inherent robustness to such adversarial attacks by design.



## **21. On Scaling LT-Coded Blockchains in Heterogeneous Networks and their Vulnerabilities to DoS Threats**

cs.IT

To appear in Future Generation Computer Systems, 2025. This is an  extended version of a shorter version that has appeared in IEEE ICC 2024

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2402.05620v3) [paper-pdf](http://arxiv.org/pdf/2402.05620v3)

**Authors**: Harikrishnan K., J. Harshan, Anwitaman Datta

**Abstract**: Coded blockchains have acquired prominence as a promising solution to reduce storage costs and facilitate scalability. Within this class, Luby Transform (LT) coded blockchains are an appealing choice for scalability owing to the availability of a wide range of low-complexity decoders. In the first part of this work, we identify that traditional LT decoders like Belief Propagation and On-the-Fly Gaussian Elimination may not be optimal for heterogeneous networks with nodes that have varying computational and download capabilities. To address this, we introduce a family of hybrid decoders for LT codes and propose optimal operating regimes for them to recover the blockchain at the lowest decoding cost. While LT coded blockchain architecture has been studied from the aspects of storage savings and scalability, not much is known in terms of its security vulnerabilities. Pointing at this research gap, in the second part, we present novel denial-of-service threats on LT coded blockchains that target nodes with specific decoding capabilities, preventing them from joining the network. Our proposed threats are non-oblivious in nature, wherein adversaries gain access to the archived blocks, and choose to execute their attack on a subset of them based on underlying coding scheme. We show that our optimized threats can achieve the same level of damage as that of blind attacks, however, with limited amount of resources. Overall, this is the first work of its kind that opens up new questions on designing coded blockchains to jointly provide storage savings, scalability and also resilience to optimized threats.



## **22. Exploring the Effect of DNN Depth on Adversarial Attacks in Network Intrusion Detection Systems**

cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19761v1) [paper-pdf](http://arxiv.org/pdf/2510.19761v1)

**Authors**: Mohamed ElShehaby, Ashraf Matrawy

**Abstract**: Adversarial attacks pose significant challenges to Machine Learning (ML) systems and especially Deep Neural Networks (DNNs) by subtly manipulating inputs to induce incorrect predictions. This paper investigates whether increasing the layer depth of deep neural networks affects their robustness against adversarial attacks in the Network Intrusion Detection System (NIDS) domain. We compare the adversarial robustness of various deep neural networks across both \ac{NIDS} and computer vision domains (the latter being widely used in adversarial attack experiments). Our experimental results reveal that in the NIDS domain, adding more layers does not necessarily improve their performance, yet it may actually significantly degrade their robustness against adversarial attacks. Conversely, in the computer vision domain, adding more layers exhibits a more modest impact on robustness. These findings can guide the development of robust neural networks for (NIDS) applications and highlight the unique characteristics of network security domains within the (ML) landscape.



## **23. Explainable Face Presentation Attack Detection via Ensemble-CAM**

cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19695v1) [paper-pdf](http://arxiv.org/pdf/2510.19695v1)

**Authors**: Rashik Shadman, M G Sarwar Murshed, Faraz Hussain

**Abstract**: Presentation attacks represent a critical security threat where adversaries use fake biometric data, such as face, fingerprint, or iris images, to gain unauthorized access to protected systems. Various presentation attack detection (PAD) systems have been designed leveraging deep learning (DL) models to mitigate this type of threat. Despite their effectiveness, most of the DL models function as black boxes - their decisions are opaque to their users. The purpose of explainability techniques is to provide detailed information about the reason behind the behavior or decision of DL models. In particular, visual explanation is necessary to better understand the decisions or predictions of DL-based PAD systems and determine the key regions due to which a biometric image is considered real or fake by the system. In this work, a novel technique, Ensemble-CAM, is proposed for providing visual explanations for the decisions made by deep learning-based face PAD systems. Our goal is to improve DL-based face PAD systems by providing a better understanding of their behavior. Our provided visual explanations will enhance the transparency and trustworthiness of DL-based face PAD systems.



## **24. Style Attack Disguise: When Fonts Become a Camouflage for Adversarial Intent**

cs.CL

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19641v1) [paper-pdf](http://arxiv.org/pdf/2510.19641v1)

**Authors**: Yangshijie Zhang, Xinda Wang, Jialin Liu, Wenqiang Wang, Zhicong Ma, Xingxing Jia

**Abstract**: With social media growth, users employ stylistic fonts and font-like emoji to express individuality, creating visually appealing text that remains human-readable. However, these fonts introduce hidden vulnerabilities in NLP models: while humans easily read stylistic text, models process these characters as distinct tokens, causing interference. We identify this human-model perception gap and propose a style-based attack, Style Attack Disguise (SAD). We design two sizes: light for query efficiency and strong for superior attack performance. Experiments on sentiment classification and machine translation across traditional models, LLMs, and commercial services demonstrate SAD's strong attack performance. We also show SAD's potential threats to multimodal tasks including text-to-image and text-to-speech generation.



## **25. Can You Trust What You See? Alpha Channel No-Box Attacks on Video Object Detection**

cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19574v1) [paper-pdf](http://arxiv.org/pdf/2510.19574v1)

**Authors**: Ariana Yi, Ce Zhou, Liyang Xiao, Qiben Yan

**Abstract**: As object detection models are increasingly deployed in cyber-physical systems such as autonomous vehicles (AVs) and surveillance platforms, ensuring their security against adversarial threats is essential. While prior work has explored adversarial attacks in the image domain, those attacks in the video domain remain largely unexamined, especially in the no-box setting. In this paper, we present {\alpha}-Cloak, the first no-box adversarial attack on object detectors that operates entirely through the alpha channel of RGBA videos. {\alpha}-Cloak exploits the alpha channel to fuse a malicious target video with a benign video, resulting in a fused video that appears innocuous to human viewers but consistently fools object detectors. Our attack requires no access to model architecture, parameters, or outputs, and introduces no perceptible artifacts. We systematically study the support for alpha channels across common video formats and playback applications, and design a fusion algorithm that ensures visual stealth and compatibility. We evaluate {\alpha}-Cloak on five state-of-the-art object detectors, a vision-language model, and a multi-modal large language model (Gemini-2.0-Flash), demonstrating a 100% attack success rate across all scenarios. Our findings reveal a previously unexplored vulnerability in video-based perception systems, highlighting the urgent need for defenses that account for the alpha channel in adversarial settings.



## **26. A New Type of Adversarial Examples**

cs.LG

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19347v1) [paper-pdf](http://arxiv.org/pdf/2510.19347v1)

**Authors**: Xingyang Nie, Guojie Xiao, Su Pan, Biao Wang, Huilin Ge, Tao Fang

**Abstract**: Most machine learning models are vulnerable to adversarial examples, which poses security concerns on these models. Adversarial examples are crafted by applying subtle but intentionally worst-case modifications to examples from the dataset, leading the model to output a different answer from the original example. In this paper, adversarial examples are formed in an exactly opposite manner, which are significantly different from the original examples but result in the same answer. We propose a novel set of algorithms to produce such adversarial examples, including the negative iterative fast gradient sign method (NI-FGSM) and the negative iterative fast gradient method (NI-FGM), along with their momentum variants: the negative momentum iterative fast gradient sign method (NMI-FGSM) and the negative momentum iterative fast gradient method (NMI-FGM). Adversarial examples constructed by these methods could be used to perform an attack on machine learning systems in certain occasions. Moreover, our results show that the adversarial examples are not merely distributed in the neighbourhood of the examples from the dataset; instead, they are distributed extensively in the sample space.



## **27. Collaborative penetration testing suite for emerging generative AI algorithms**

cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19303v1) [paper-pdf](http://arxiv.org/pdf/2510.19303v1)

**Authors**: Petar Radanliev

**Abstract**: Problem Space: AI Vulnerabilities and Quantum Threats Generative AI vulnerabilities: model inversion, data poisoning, adversarial inputs. Quantum threats Shor Algorithm breaking RSA ECC encryption. Challenge Secure generative AI models against classical and quantum cyberattacks. Proposed Solution Collaborative Penetration Testing Suite Five Integrated Components: DAST SAST OWASP ZAP, Burp Suite, SonarQube, Fortify. IAST Contrast Assess integrated with CI CD pipeline. Blockchain Logging Hyperledger Fabric for tamper-proof logs. Quantum Cryptography Lattice based RLWE protocols. AI Red Team Simulations Adversarial ML & Quantum-assisted attacks. Integration Layer: Unified workflow for AI, cybersecurity, and quantum experts. Key Results 300+ vulnerabilities identified across test environments. 70% reduction in high-severity issues within 2 weeks. 90% resolution efficiency for blockchain-logged vulnerabilities. Quantum-resistant cryptography maintained 100% integrity in tests. Outcome: Quantum AI Security Protocol integrating Blockchain Quantum Cryptography AI Red Teaming.



## **28. Adversarial Attacks on LiDAR-Based Tracking Across Road Users: Robustness Evaluation and Target-Aware Black-Box Method**

cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2410.20893v3) [paper-pdf](http://arxiv.org/pdf/2410.20893v3)

**Authors**: Shengjing Tian, Xiantong Zhao, Yuhao Bian, Yinan Han, Bin Liu

**Abstract**: In this study, we delve into the robustness of neural network-based LiDAR point cloud tracking models under adversarial attacks, a critical aspect often overlooked in favor of performance enhancement. These models, despite incorporating advanced architectures like Transformer or Bird's Eye View (BEV), tend to neglect robustness in the face of challenges such as adversarial attacks, domain shifts, or data corruption. We instead focus on the robustness of the tracking models under the threat of adversarial attacks. We begin by establishing a unified framework for conducting adversarial attacks within the context of 3D object tracking, which allows us to thoroughly investigate both white-box and black-box attack strategies. For white-box attacks, we tailor specific loss functions to accommodate various tracking paradigms and extend existing methods such as FGSM, C\&W, and PGD to the point cloud domain. In addressing black-box attack scenarios, we introduce a novel transfer-based approach, the Target-aware Perturbation Generation (TAPG) algorithm, with the dual objectives of achieving high attack performance and maintaining low perceptibility. This method employs a heuristic strategy to enforce sparse attack constraints and utilizes random sub-vector factorization to bolster transferability. Our experimental findings reveal a significant vulnerability in advanced tracking methods when subjected to both black-box and white-box attacks, underscoring the necessity for incorporating robustness against adversarial attacks into the design of LiDAR point cloud tracking models. Notably, compared to existing methods, the TAPG also strikes an optimal balance between the effectiveness of the attack and the concealment of the perturbations.



## **29. Defending Against Prompt Injection with DataFilter**

cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19207v1) [paper-pdf](http://arxiv.org/pdf/2510.19207v1)

**Authors**: Yizhu Wang, Sizhe Chen, Raghad Alkhudair, Basel Alomair, David Wagner

**Abstract**: When large language model (LLM) agents are increasingly deployed to automate tasks and interact with untrusted external data, prompt injection emerges as a significant security threat. By injecting malicious instructions into the data that LLMs access, an attacker can arbitrarily override the original user task and redirect the agent toward unintended, potentially harmful actions. Existing defenses either require access to model weights (fine-tuning), incur substantial utility loss (detection-based), or demand non-trivial system redesign (system-level). Motivated by this, we propose DataFilter, a test-time model-agnostic defense that removes malicious instructions from the data before it reaches the backend LLM. DataFilter is trained with supervised fine-tuning on simulated injections and leverages both the user's instruction and the data to selectively strip adversarial content while preserving benign information. Across multiple benchmarks, DataFilter consistently reduces the prompt injection attack success rates to near zero while maintaining the LLMs' utility. DataFilter delivers strong security, high utility, and plug-and-play deployment, making it a strong practical defense to secure black-box commercial LLMs against prompt injection. Our DataFilter model is released at https://huggingface.co/JoyYizhu/DataFilter for immediate use, with the code to reproduce our results at https://github.com/yizhu-joy/DataFilter.



## **30. FeatureFool: Zero-Query Fooling of Video Models via Feature Map**

cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.18362v2) [paper-pdf](http://arxiv.org/pdf/2510.18362v2)

**Authors**: Duoxun Tang, Xi Xiao, Guangwu Hu, Kangkang Sun, Xiao Yang, Dongyang Chen, Qing Li, Yongjie Yin, Jiyao Wang

**Abstract**: The vulnerability of deep neural networks (DNNs) has been preliminarily verified. Existing black-box adversarial attacks usually require multi-round interaction with the model and consume numerous queries, which is impractical in the real-world and hard to scale to recently emerged Video-LLMs. Moreover, no attack in the video domain directly leverages feature maps to shift the clean-video feature space. We therefore propose FeatureFool, a stealthy, video-domain, zero-query black-box attack that utilizes information extracted from a DNN to alter the feature space of clean videos. Unlike query-based methods that rely on iterative interaction, FeatureFool performs a zero-query attack by directly exploiting DNN-extracted information. This efficient approach is unprecedented in the video domain. Experiments show that FeatureFool achieves an attack success rate above 70\% against traditional video classifiers without any queries. Benefiting from the transferability of the feature map, it can also craft harmful content and bypass Video-LLM recognition. Additionally, adversarial videos generated by FeatureFool exhibit high quality in terms of SSIM, PSNR, and Temporal-Inconsistency, making the attack barely perceptible. This paper may contain violent or explicit content.



## **31. The Black Tuesday Attack: how to crash the stock market with adversarial examples to financial forecasting models**

cs.CR

15 pages, 2 figures

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18990v1) [paper-pdf](http://arxiv.org/pdf/2510.18990v1)

**Authors**: Thomas Hofweber, Jefrey Bergl, Ian Reyes, Amir Sadovnik

**Abstract**: We investigate and defend the possibility of causing a stock market crash via small manipulations of individual stock values that together realize an adversarial example to financial forecasting models, causing these models to make the self-fulfilling prediction of a crash. Such a crash triggered by an adversarial example would likely be hard to detect, since the model's predictions would be accurate and the interventions that would cause it are minor. This possibility is a major risk to financial stability and an opportunity for hostile actors to cause great economic damage to an adversary. This threat also exists against individual stocks and the corresponding valuation of individual companies. We outline how such an attack might proceed, what its theoretical basis is, how it can be directed towards a whole economy or an individual company, and how one might defend against it. We conclude that this threat is vastly underappreciated and requires urgent research on how to defend against it.



## **32. Towards Universal Solvers: Using PGD Attack in Active Learning to Increase Generalizability of Neural Operators as Knowledge Distillation from Numerical PDE Solvers**

cs.LG

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18989v1) [paper-pdf](http://arxiv.org/pdf/2510.18989v1)

**Authors**: Yifei Sun

**Abstract**: Nonlinear PDE solvers require fine space-time discretizations and local linearizations, leading to high memory cost and slow runtimes. Neural operators such as FNOs and DeepONets offer fast single-shot inference by learning function-to-function mappings and truncating high-frequency components, but they suffer from poor out-of-distribution (OOD) generalization, often failing on inputs outside the training distribution. We propose an adversarial teacher-student distillation framework in which a differentiable numerical solver supervises a compact neural operator while a PGD-style active sampling loop searches for worst-case inputs under smoothness and energy constraints to expand the training set. Using differentiable spectral solvers enables gradient-based adversarial search and stabilizes sample mining. Experiments on Burgers and Navier-Stokes systems demonstrate that adversarial distillation substantially improves OOD robustness while preserving the low parameter cost and fast inference of neural operators.



## **33. NEXUS: Network Exploration for eXploiting Unsafe Sequences in Multi-Turn LLM Jailbreaks**

cs.CR

This paper has been accepted in the main conference proceedings of  the 2025 Conference on Empirical Methods in Natural Language Processing  (EMNLP 2025). Javad Rafiei Asl and Sidhant Narula are co-first authors

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.03417v2) [paper-pdf](http://arxiv.org/pdf/2510.03417v2)

**Authors**: Javad Rafiei Asl, Sidhant Narula, Mohammad Ghasemigol, Eduardo Blanco, Daniel Takabi

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing but remain vulnerable to jailbreak attacks, especially multi-turn jailbreaks that distribute malicious intent across benign exchanges and bypass alignment mechanisms. Existing approaches often explore the adversarial space poorly, rely on hand-crafted heuristics, or lack systematic query refinement. We present NEXUS (Network Exploration for eXploiting Unsafe Sequences), a modular framework for constructing, refining, and executing optimized multi-turn attacks. NEXUS comprises: (1) ThoughtNet, which hierarchically expands a harmful intent into a structured semantic network of topics, entities, and query chains; (2) a feedback-driven Simulator that iteratively refines and prunes these chains through attacker-victim-judge LLM collaboration using harmfulness and semantic-similarity benchmarks; and (3) a Network Traverser that adaptively navigates the refined query space for real-time attacks. This pipeline uncovers stealthy, high-success adversarial paths across LLMs. On several closed-source and open-source LLMs, NEXUS increases attack success rate by 2.1% to 19.4% over prior methods. Code: https://github.com/inspire-lab/NEXUS



## **34. Nondeterminism-Aware Optimistic Verification for Floating-Point Neural Networks**

cs.CR

17 pages, 7 figures

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.16028v2) [paper-pdf](http://arxiv.org/pdf/2510.16028v2)

**Authors**: Jianzhu Yao, Hongxu Su, Taobo Liao, Zerui Cheng, Huan Zhang, Xuechao Wang, Pramod Viswanath

**Abstract**: Neural networks increasingly run on hardware outside the user's control (cloud GPUs, inference marketplaces). Yet ML-as-a-Service reveals little about what actually ran or whether returned outputs faithfully reflect the intended inputs. Users lack recourse against service downgrades (model swaps, quantization, graph rewrites, or discrepancies like altered ad embeddings). Verifying outputs is hard because floating-point(FP) execution on heterogeneous accelerators is inherently nondeterministic. Existing approaches are either impractical for real FP neural networks or reintroduce vendor trust. We present NAO: a Nondeterministic tolerance Aware Optimistic verification protocol that accepts outputs within principled operator-level acceptance regions rather than requiring bitwise equality. NAO combines two error models: (i) sound per-operator IEEE-754 worst-case bounds and (ii) tight empirical percentile profiles calibrated across hardware. Discrepancies trigger a Merkle-anchored, threshold-guided dispute game that recursively partitions the computation graph until one operator remains, where adjudication reduces to a lightweight theoretical-bound check or a small honest-majority vote against empirical thresholds. Unchallenged results finalize after a challenge window, without requiring trusted hardware or deterministic kernels. We implement NAO as a PyTorch-compatible runtime and a contract layer currently deployed on Ethereum Holesky testnet. The runtime instruments graphs, computes per-operator bounds, and runs unmodified vendor kernels in FP32 with negligible overhead (0.3% on Qwen3-8B). Across CNNs, Transformers and diffusion models on A100, H100, RTX6000, RTX4090, empirical thresholds are $10^2-10^3$ times tighter than theoretical bounds, and bound-aware adversarial attacks achieve 0% success. NAO reconciles scalability with verifiability for real-world heterogeneous ML compute.



## **35. HarmNet: A Framework for Adaptive Multi-Turn Jailbreak Attacks on Large Language Models**

cs.CR

This paper has been accepted for presentation at the Conference on  Applied Machine Learning in Information Security (CAMLIS 2025)

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18728v1) [paper-pdf](http://arxiv.org/pdf/2510.18728v1)

**Authors**: Sidhant Narula, Javad Rafiei Asl, Mohammad Ghasemigol, Eduardo Blanco, Daniel Takabi

**Abstract**: Large Language Models (LLMs) remain vulnerable to multi-turn jailbreak attacks. We introduce HarmNet, a modular framework comprising ThoughtNet, a hierarchical semantic network; a feedback-driven Simulator for iterative query refinement; and a Network Traverser for real-time adaptive attack execution. HarmNet systematically explores and refines the adversarial space to uncover stealthy, high-success attack paths. Experiments across closed-source and open-source LLMs show that HarmNet outperforms state-of-the-art methods, achieving higher attack success rates. For example, on Mistral-7B, HarmNet achieves a 99.4% attack success rate, 13.9% higher than the best baseline. Index terms: jailbreak attacks; large language models; adversarial framework; query refinement.



## **36. Trial and Trust: Addressing Byzantine Attacks with Comprehensive Defense Strategy**

cs.LG

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2505.07614v4) [paper-pdf](http://arxiv.org/pdf/2505.07614v4)

**Authors**: Gleb Molodtsov, Daniil Medyakov, Sergey Skorik, Nikolas Khachaturov, Shahane Tigranyan, Vladimir Aletov, Aram Avetisyan, Martin Takáč, Aleksandr Beznosikov

**Abstract**: Recent advancements in machine learning have improved performance while also increasing computational demands. While federated and distributed setups address these issues, their structure is vulnerable to malicious influences. In this paper, we address a specific threat, Byzantine attacks, where compromised clients inject adversarial updates to derail global convergence. We combine the trust scores concept with trial function methodology to dynamically filter outliers. Our methods address the critical limitations of previous approaches, allowing functionality even when Byzantine nodes are in the majority. Moreover, our algorithms adapt to widely used scaled methods like Adam and RMSProp, as well as practical scenarios, including local training and partial participation. We validate the robustness of our methods by conducting extensive experiments on both synthetic and real ECG data collected from medical institutions. Furthermore, we provide a broad theoretical analysis of our algorithms and their extensions to aforementioned practical setups. The convergence guarantees of our methods are comparable to those of classical algorithms developed without Byzantine interference.



## **37. Exploring Membership Inference Vulnerabilities in Clinical Large Language Models**

cs.CR

Accepted at the 1st IEEE Workshop on Healthcare and Medical Device  Security, Privacy, Resilience, and Trust (IEEE HMD-SPiRiT)

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18674v1) [paper-pdf](http://arxiv.org/pdf/2510.18674v1)

**Authors**: Alexander Nemecek, Zebin Yun, Zahra Rahmani, Yaniv Harel, Vipin Chaudhary, Mahmood Sharif, Erman Ayday

**Abstract**: As large language models (LLMs) become progressively more embedded in clinical decision-support, documentation, and patient-information systems, ensuring their privacy and trustworthiness has emerged as an imperative challenge for the healthcare sector. Fine-tuning LLMs on sensitive electronic health record (EHR) data improves domain alignment but also raises the risk of exposing patient information through model behaviors. In this work-in-progress, we present an exploratory empirical study on membership inference vulnerabilities in clinical LLMs, focusing on whether adversaries can infer if specific patient records were used during model training. Using a state-of-the-art clinical question-answering model, Llemr, we evaluate both canonical loss-based attacks and a domain-motivated paraphrasing-based perturbation strategy that more realistically reflects clinical adversarial conditions. Our preliminary findings reveal limited but measurable membership leakage, suggesting that current clinical LLMs provide partial resistance yet remain susceptible to subtle privacy risks that could undermine trust in clinical AI adoption. These results motivate continued development of context-aware, domain-specific privacy evaluations and defenses such as differential privacy fine-tuning and paraphrase-aware training, to strengthen the security and trustworthiness of healthcare AI systems.



## **38. Quantifying Security for Networked Control Systems: A Review**

eess.SY

Journal submission

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18645v1) [paper-pdf](http://arxiv.org/pdf/2510.18645v1)

**Authors**: Sribalaji C. Anand, Anh Tung Nguyen, André M. H. Teixeira, Henrik Sandberg, Karl H. Johansson

**Abstract**: Networked Control Systems (NCSs) are integral in critical infrastructures such as power grids, transportation networks, and production systems. Ensuring the resilient operation of these large-scale NCSs against cyber-attacks is crucial for societal well-being. Over the past two decades, extensive research has been focused on developing metrics to quantify the vulnerabilities of NCSs against attacks. Once the vulnerabilities are quantified, mitigation strategies can be employed to enhance system resilience. This article provides a comprehensive overview of methods developed for assessing NCS vulnerabilities and the corresponding mitigation strategies. Furthermore, we emphasize the importance of probabilistic risk metrics to model vulnerabilities under adversaries with imperfect process knowledge. The article concludes by outlining promising directions for future research.



## **39. Qatsi: Stateless Secret Generation via Hierarchical Memory-Hard Key Derivation**

cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18614v1) [paper-pdf](http://arxiv.org/pdf/2510.18614v1)

**Authors**: René Coignard, Anton Rygin

**Abstract**: We present Qatsi, a hierarchical key derivation scheme using Argon2id that generates reproducible cryptographic secrets without persistent storage. The system eliminates vault-based attack surfaces by deriving all secrets deterministically from a single high-entropy master secret and contextual layers. Outputs achieve 103-312 bits of entropy through memory-hard derivation (64-128 MiB, 16-32 iterations) and provably uniform rejection sampling over 7776-word mnemonics or 90-character passwords. We formalize the hierarchical construction, prove output uniformity, and quantify GPU attack costs: $2.4 \times 10^{16}$ years for 80-bit master secrets on single-GPU adversaries under Paranoid parameters (128 MiB memory). The implementation in Rust provides automatic memory zeroization, compile-time wordlist integrity verification, and comprehensive test coverage. Reference benchmarks on Apple M1 Pro (2021) demonstrate practical usability with 544 ms Standard mode and 2273 ms Paranoid mode single-layer derivations. Qatsi targets air-gapped systems and master credential generation where stateless reproducibility outweighs rotation flexibility.



## **40. Evaluating Large Language Models in detecting Secrets in Android Apps**

cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18601v1) [paper-pdf](http://arxiv.org/pdf/2510.18601v1)

**Authors**: Marco Alecci, Jordan Samhi, Tegawendé F. Bissyandé, Jacques Klein

**Abstract**: Mobile apps often embed authentication secrets, such as API keys, tokens, and client IDs, to integrate with cloud services. However, developers often hardcode these credentials into Android apps, exposing them to extraction through reverse engineering. Once compromised, adversaries can exploit secrets to access sensitive data, manipulate resources, or abuse APIs, resulting in significant security and financial risks. Existing detection approaches, such as regex-based analysis, static analysis, and machine learning, are effective for identifying known patterns but are fundamentally limited: they require prior knowledge of credential structures, API signatures, or training data.   In this paper, we propose SecretLoc, an LLM-based approach for detecting hardcoded secrets in Android apps. SecretLoc goes beyond pattern matching; it leverages contextual and structural cues to identify secrets without relying on predefined patterns or labeled training sets. Using a benchmark dataset from the literature, we demonstrate that SecretLoc detects secrets missed by regex-, static-, and ML-based methods, including previously unseen types of secrets. In total, we discovered 4828 secrets that were undetected by existing approaches, discovering more than 10 "new" types of secrets, such as OpenAI API keys, GitHub Access Tokens, RSA private keys, and JWT tokens, and more.   We further extend our analysis to newly crawled apps from Google Play, where we uncovered and responsibly disclosed additional hardcoded secrets. Across a set of 5000 apps, we detected secrets in 2124 apps (42.5%), several of which were confirmed and remediated by developers after we contacted them. Our results reveal a dual-use risk: if analysts can uncover these secrets with LLMs, so can attackers. This underscores the urgent need for proactive secret management and stronger mitigation practices across the mobile ecosystem.



## **41. SentinelNet: Safeguarding Multi-Agent Collaboration Through Credit-Based Dynamic Threat Detection**

cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.16219v2) [paper-pdf](http://arxiv.org/pdf/2510.16219v2)

**Authors**: Yang Feng, Xudong Pan

**Abstract**: Malicious agents pose significant threats to the reliability and decision-making capabilities of Multi-Agent Systems (MAS) powered by Large Language Models (LLMs). Existing defenses often fall short due to reactive designs or centralized architectures which may introduce single points of failure. To address these challenges, we propose SentinelNet, the first decentralized framework for proactively detecting and mitigating malicious behaviors in multi-agent collaboration. SentinelNet equips each agent with a credit-based detector trained via contrastive learning on augmented adversarial debate trajectories, enabling autonomous evaluation of message credibility and dynamic neighbor ranking via bottom-k elimination to suppress malicious communications. To overcome the scarcity of attack data, it generates adversarial trajectories simulating diverse threats, ensuring robust training. Experiments on MAS benchmarks show SentinelNet achieves near-perfect detection of malicious agents, close to 100% within two debate rounds, and recovers 95% of system accuracy from compromised baselines. By exhibiting strong generalizability across domains and attack patterns, SentinelNet establishes a novel paradigm for safeguarding collaborative MAS.



## **42. Robustness Verification of Graph Neural Networks Via Lightweight Satisfiability Testing**

cs.LG

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18591v1) [paper-pdf](http://arxiv.org/pdf/2510.18591v1)

**Authors**: Chia-Hsuan Lu, Tony Tan, Michael Benedikt

**Abstract**: Graph neural networks (GNNs) are the predominant architecture for learning over graphs. As with any machine learning model, and important issue is the detection of adversarial attacks, where an adversary can change the output with a small perturbation of the input. Techniques for solving the adversarial robustness problem - determining whether such an attack exists - were originally developed for image classification, but there are variants for many other machine learning architectures. In the case of graph learning, the attack model usually considers changes to the graph structure in addition to or instead of the numerical features of the input, and the state of the art techniques in the area proceed via reduction to constraint solving, working on top of powerful solvers, e.g. for mixed integer programming. We show that it is possible to improve on the state of the art in structural robustness by replacing the use of powerful solvers by calls to efficient partial solvers, which run in polynomial time but may be incomplete. We evaluate our tool RobLight on a diverse set of GNN variants and datasets.



## **43. Towards Quantum Enhanced Adversarial Robustness with Rydberg Reservoir Learning**

quant-ph

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.13473v2) [paper-pdf](http://arxiv.org/pdf/2510.13473v2)

**Authors**: Shehbaz Tariq, Muhammad Talha, Symeon Chatzinotas, Hyundong Shin

**Abstract**: Quantum reservoir computing (QRC) leverages the high-dimensional, nonlinear dynamics inherent in quantum many-body systems for extracting spatiotemporal patterns in sequential and time-series data with minimal training overhead. Although QRC inherits the expressive capabilities associated with quantum encodings, recent studies indicate that quantum classifiers based on variational circuits remain susceptible to adversarial perturbations. In this perspective, we investigate the first systematic evaluation of adversarial robustness in a QRC based learning model. Our reservoir comprises an array of strongly interacting Rydberg atoms governed by a fixed Hamiltonian, which naturally evolves under complex quantum dynamics, producing high-dimensional embeddings. A lightweight multilayer perceptron serves as the trainable readout layer. We utilize the balanced datasets, namely MNIST, Fashion-MNIST, and Kuzushiji-MNIST, as a benchmark for rigorously evaluating the impact of augmenting the quantum reservoir with a Multilayer perceptron (MLP) in white-box adversarial attacks to assess its robustness. We demonstrate that this approach yields significantly higher accuracy than purely classical models across all perturbation strengths tested. This hybrid approach reveals a new source of quantum advantage and provides practical guidance for the secure deployment of machine learning models on quantum-centric supercomputing with near-term hardware.



## **44. S2AP: Score-space Sharpness Minimization for Adversarial Pruning**

cs.CV

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18381v1) [paper-pdf](http://arxiv.org/pdf/2510.18381v1)

**Authors**: Giorgio Piras, Qi Zhao, Fabio Brau, Maura Pintor, Christian Wressnegger, Battista Biggio

**Abstract**: Adversarial pruning methods have emerged as a powerful tool for compressing neural networks while preserving robustness against adversarial attacks. These methods typically follow a three-step pipeline: (i) pretrain a robust model, (ii) select a binary mask for weight pruning, and (iii) finetune the pruned model. To select the binary mask, these methods minimize a robust loss by assigning an importance score to each weight, and then keep the weights with the highest scores. However, this score-space optimization can lead to sharp local minima in the robust loss landscape and, in turn, to an unstable mask selection, reducing the robustness of adversarial pruning methods. To overcome this issue, we propose a novel plug-in method for adversarial pruning, termed Score-space Sharpness-aware Adversarial Pruning (S2AP). Through our method, we introduce the concept of score-space sharpness minimization, which operates during the mask search by perturbing importance scores and minimizing the corresponding robust loss. Extensive experiments across various datasets, models, and sparsity levels demonstrate that S2AP effectively minimizes sharpness in score space, stabilizing the mask selection, and ultimately improving the robustness of adversarial pruning methods.



## **45. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

cs.CR

This work was first submitted for review on Sept. 5, 2024, and the  initial version was uploaded to Arxiv on Sept. 30, 2024. The latest version  has accepted for publication by IEEE Transactions on Information Forensics  and Security (TIFS)

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2409.20002v5) [paper-pdf](http://arxiv.org/pdf/2409.20002v5)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.



## **46. Genesis: Evolving Attack Strategies for LLM Web Agent Red-Teaming**

cs.AI

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18314v1) [paper-pdf](http://arxiv.org/pdf/2510.18314v1)

**Authors**: Zheng Zhang, Jiarui He, Yuchen Cai, Deheng Ye, Peilin Zhao, Ruili Feng, Hao Wang

**Abstract**: As large language model (LLM) agents increasingly automate complex web tasks, they boost productivity while simultaneously introducing new security risks. However, relevant studies on web agent attacks remain limited. Existing red-teaming approaches mainly rely on manually crafted attack strategies or static models trained offline. Such methods fail to capture the underlying behavioral patterns of web agents, making it difficult to generalize across diverse environments. In web agent attacks, success requires the continuous discovery and evolution of attack strategies. To this end, we propose Genesis, a novel agentic framework composed of three modules: Attacker, Scorer, and Strategist. The Attacker generates adversarial injections by integrating the genetic algorithm with a hybrid strategy representation. The Scorer evaluates the target web agent's responses to provide feedback. The Strategist dynamically uncovers effective strategies from interaction logs and compiles them into a continuously growing strategy library, which is then re-deployed to enhance the Attacker's effectiveness. Extensive experiments across various web tasks show that our framework discovers novel strategies and consistently outperforms existing attack baselines.



## **47. Ensuring Robustness in ML-enabled Software Systems: A User Survey**

cs.SE

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18292v1) [paper-pdf](http://arxiv.org/pdf/2510.18292v1)

**Authors**: Hala Abdelkader, Mohamed Abdelrazek, Priya Rani, Rajesh Vasa, Jean-Guy Schneider

**Abstract**: Ensuring robustness in ML-enabled software systems requires addressing critical challenges, such as silent failures, out-of-distribution (OOD) data, and adversarial attacks. Traditional software engineering practices, which rely on predefined logic, are insufficient for ML components that depend on data and probabilistic decision-making. To address these challenges, we propose the ML-On-Rails protocol, a unified framework designed to enhance the robustness and trustworthiness of ML-enabled systems in production. This protocol integrates key safeguards such as OOD detection, adversarial attack detection, input validation, and explainability. It also includes a model-to-software communication framework using HTTP status codes to enhance transparency in reporting model outcomes and errors. To align our approach with real-world challenges, we conducted a practitioner survey, which revealed major robustness issues, gaps in current solutions, and highlighted how a standardised protocol such as ML-On-Rails can improve system robustness. Our findings highlight the need for more support and resources for engineers working with ML systems. Finally, we outline future directions for refining the proposed protocol, leveraging insights from the survey and real-world applications to continually enhance its effectiveness.



## **48. DrunkAgent: Stealthy Memory Corruption in LLM-Powered Recommender Agents**

cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2503.23804v3) [paper-pdf](http://arxiv.org/pdf/2503.23804v3)

**Authors**: Shiyi Yang, Zhibo Hu, Xinshu Li, Chen Wang, Tong Yu, Xiwei Xu, Liming Zhu, Lina Yao

**Abstract**: Large language model (LLM)-powered agents are increasingly used in recommender systems (RSs) to achieve personalized behavior modeling, where the memory mechanism plays a pivotal role in enabling the agents to autonomously explore, learn and self-evolve from real-world interactions. However, this very mechanism, serving as a contextual repository, inherently exposes an attack surface for potential adversarial manipulations. Despite its central role, the robustness of agentic RSs in the face of such threats remains largely underexplored. Previous works suffer from semantic mismatches or rely on static embeddings or pre-defined prompts, all of which are not designed for dynamic systems, especially for dynamic memory states of LLM agents. This challenge is exacerbated by the black-box nature of commercial recommenders.   To tackle the above problems, in this paper, we present the first systematic investigation of memory-based vulnerabilities in LLM-powered recommender agents, revealing their security limitations and guiding efforts to strengthen system resilience and trustworthiness. Specifically, we propose a novel black-box attack framework named DrunkAgent. DrunkAgent crafts semantically meaningful adversarial textual triggers for target item promotions and introduces a series of strategies to maximize the trigger effect by corrupting the memory updates during the interactions. The triggers and strategies are optimized on a surrogate model, enabling DrunkAgent transferable and stealthy. Extensive experiments on real-world datasets across diverse agentic RSs, including collaborative filtering, retrieval augmentation and sequential recommendations, demonstrate the generalizability, transferability and stealthiness of DrunkAgent.



## **49. Transaction Capacity, Security and Latency in Blockchains**

cs.CR

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2402.10138v2) [paper-pdf](http://arxiv.org/pdf/2402.10138v2)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: We analyze how secure a block is after the block becomes $k$-deep, i.e., security-latency, for Nakamoto consensus under an exponential network delay model. We provide the fault tolerance and extensive bounds on safety violation probabilities given mining rate, delay rate and confirmation rules. Next, modeling the blockchain system as a batch service queue with exponential network delay, we connect the security-latency analysis to sustainable transaction rate of the queue system. As our model assumes exponential network delay, batch service queue models give a meaningful trade-off between transaction capacity, security and latency. Our results indicate that, by simply picking $k=7$-block confirmation rule in Bitcoin instead of the convention of $k=6$, mining rate, latency and throughput can be increased sixfold with the same safety guarantees. We further consider adversarial attacks on the queue service to hamper the service process. In an extreme scenario, we consider the selfish-mining attack for this purpose and provide the maximum adversarial block ratio in the longest chain under the exponential delay model. The ratio in turn reflects the maximum rate of decrease in the sustainable transaction rate of the queue.



## **50. Black-Box Evasion Attacks on Data-Driven Open RAN Apps: Tailored Design and Experimental Evaluation**

cs.CR

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.18160v1) [paper-pdf](http://arxiv.org/pdf/2510.18160v1)

**Authors**: Pranshav Gajjar, Molham Khoja, Abiodun Ganiyu, Marc Juarez, Mahesh K. Marina, Andrew Lehane, Vijay K. Shah

**Abstract**: The impending adoption of Open Radio Access Network (O-RAN) is fueling innovation in the RAN towards data-driven operation. Unlike traditional RAN where the RAN data and its usage is restricted within proprietary and monolithic RAN equipment, the O-RAN architecture opens up access to RAN data via RAN intelligent controllers (RICs), to third-party machine learning (ML) powered applications - rApps and xApps - to optimize RAN operations. Consequently, a major focus has been placed on leveraging RAN data to unlock greater efficiency gains. However, there is an increasing recognition that RAN data access to apps could become a source of vulnerability and be exploited by malicious actors. Motivated by this, we carry out a comprehensive investigation of data vulnerabilities on both xApps and rApps, respectively hosted in Near- and Non-real-time (RT) RIC components of O-RAN. We qualitatively analyse the O-RAN security mechanisms and limitations for xApps and rApps, and consider a threat model informed by this analysis. We design a viable and effective black-box evasion attack strategy targeting O-RAN RIC Apps while accounting for the stringent timing constraints and attack effectiveness. The strategy employs four key techniques: the model cloning algorithm, input-specific perturbations, universal adversarial perturbations (UAPs), and targeted UAPs. This strategy targets ML models used by both xApps and rApps within the O-RAN system, aiming to degrade network performance. We validate the effectiveness of the designed evasion attack strategy and quantify the scale of performance degradation using a real-world O-RAN testbed and emulation environments. Evaluation is conducted using the Interference Classification xApp and the Power Saving rApp as representatives for near-RT and non-RT RICs. We also show that the attack strategy is effective against prominent defense techniques for adversarial ML.



