# Latest Adversarial Attack Papers
**update at 2025-07-05 11:50:20**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Is Reasoning All You Need? Probing Bias in the Age of Reasoning Language Models**

cs.CL

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.02799v1) [paper-pdf](http://arxiv.org/pdf/2507.02799v1)

**Authors**: Riccardo Cantini, Nicola Gabriele, Alessio Orsino, Domenico Talia

**Abstract**: Reasoning Language Models (RLMs) have gained traction for their ability to perform complex, multi-step reasoning tasks through mechanisms such as Chain-of-Thought (CoT) prompting or fine-tuned reasoning traces. While these capabilities promise improved reliability, their impact on robustness to social biases remains unclear. In this work, we leverage the CLEAR-Bias benchmark, originally designed for Large Language Models (LLMs), to investigate the adversarial robustness of RLMs to bias elicitation. We systematically evaluate state-of-the-art RLMs across diverse sociocultural dimensions, using an LLM-as-a-judge approach for automated safety scoring and leveraging jailbreak techniques to assess the strength of built-in safety mechanisms. Our evaluation addresses three key questions: (i) how the introduction of reasoning capabilities affects model fairness and robustness; (ii) whether models fine-tuned for reasoning exhibit greater safety than those relying on CoT prompting at inference time; and (iii) how the success rate of jailbreak attacks targeting bias elicitation varies with the reasoning mechanisms employed. Our findings reveal a nuanced relationship between reasoning capabilities and bias safety. Surprisingly, models with explicit reasoning, whether via CoT prompting or fine-tuned reasoning traces, are generally more vulnerable to bias elicitation than base models without such mechanisms, suggesting reasoning may unintentionally open new pathways for stereotype reinforcement. Reasoning-enabled models appear somewhat safer than those relying on CoT prompting, which are particularly prone to contextual reframing attacks through storytelling prompts, fictional personas, or reward-shaped instructions. These results challenge the assumption that reasoning inherently improves robustness and underscore the need for more bias-aware approaches to reasoning design.



## **2. The Evolution of Dataset Distillation: Toward Scalable and Generalizable Solutions**

cs.CV

Dr. Jiawei Du is the corresponding author

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2502.05673v3) [paper-pdf](http://arxiv.org/pdf/2502.05673v3)

**Authors**: Ping Liu, Jiawei Du

**Abstract**: Dataset distillation, which condenses large-scale datasets into compact synthetic representations, has emerged as a critical solution for training modern deep learning models efficiently. While prior surveys focus on developments before 2023, this work comprehensively reviews recent advances, emphasizing scalability to large-scale datasets such as ImageNet-1K and ImageNet-21K. We categorize progress into a few key methodologies: trajectory matching, gradient matching, distribution matching, scalable generative approaches, and decoupling optimization mechanisms. As a comprehensive examination of recent dataset distillation advances, this survey highlights breakthrough innovations: the SRe2L framework for efficient and effective condensation, soft label strategies that significantly enhance model accuracy, and lossless distillation techniques that maximize compression while maintaining performance. Beyond these methodological advancements, we address critical challenges, including robustness against adversarial and backdoor attacks, effective handling of non-IID data distributions. Additionally, we explore emerging applications in video and audio processing, multi-modal learning, medical imaging, and scientific computing, highlighting its domain versatility. By offering extensive performance comparisons and actionable research directions, this survey equips researchers and practitioners with practical insights to advance efficient and generalizable dataset distillation, paving the way for future innovations.



## **3. De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks**

cs.SD

Accepted by ICML 2025

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.02606v1) [paper-pdf](http://arxiv.org/pdf/2507.02606v1)

**Authors**: Wei Fan, Kejiang Chen, Chang Liu, Weiming Zhang, Nenghai Yu

**Abstract**: The rapid advancement of speech generation models has heightened privacy and security concerns related to voice cloning (VC). Recent studies have investigated disrupting unauthorized voice cloning by introducing adversarial perturbations. However, determined attackers can mitigate these protective perturbations and successfully execute VC. In this study, we conduct the first systematic evaluation of these protective perturbations against VC under realistic threat models that include perturbation purification. Our findings reveal that while existing purification methods can neutralize a considerable portion of the protective perturbations, they still lead to distortions in the feature space of VC models, which degrades the performance of VC. From this perspective, we propose a novel two-stage purification method: (1) Purify the perturbed speech; (2) Refine it using phoneme guidance to align it with the clean speech distribution. Experimental results demonstrate that our method outperforms state-of-the-art purification methods in disrupting VC defenses. Our study reveals the limitations of adversarial perturbation-based VC defenses and underscores the urgent need for more robust solutions to mitigate the security and privacy risks posed by VC. The code and audio samples are available at https://de-antifake.github.io.



## **4. Robustness of Misinformation Classification Systems to Adversarial Examples Through BeamAttack**

cs.CL

12 pages main text, 27 pages total including references and  appendices. 13 figures, 10 tables. Accepted for publication in the LNCS  proceedings of CLEF 2025 (Best-of-Labs track)

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2506.23661v2) [paper-pdf](http://arxiv.org/pdf/2506.23661v2)

**Authors**: Arnisa Fazla, Lucas Krauter, David Guzman Piedrahita, Andrianos Michail

**Abstract**: We extend BeamAttack, an adversarial attack algorithm designed to evaluate the robustness of text classification systems through word-level modifications guided by beam search. Our extensions include support for word deletions and the option to skip substitutions, enabling the discovery of minimal modifications that alter model predictions. We also integrate LIME to better prioritize word replacements. Evaluated across multiple datasets and victim models (BiLSTM, BERT, and adversarially trained RoBERTa) within the BODEGA framework, our approach achieves over a 99\% attack success rate while preserving the semantic and lexical similarity of the original texts. Through both quantitative and qualitative analysis, we highlight BeamAttack's effectiveness and its limitations. Our implementation is available at https://github.com/LucK1Y/BeamAttack



## **5. SecAlign: Defending Against Prompt Injection with Preference Optimization**

cs.CR

ACM CCS 2025. Key words: prompt injection defense, LLM security,  LLM-integrated applications

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2410.05451v3) [paper-pdf](http://arxiv.org/pdf/2410.05451v3)

**Authors**: Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, David Wagner, Chuan Guo

**Abstract**: Large language models (LLMs) are becoming increasingly prevalent in modern software systems, interfacing between the user and the Internet to assist with tasks that require advanced language understanding. To accomplish these tasks, the LLM often uses external data sources such as user documents, web retrieval, results from API calls, etc. This opens up new avenues for attackers to manipulate the LLM via prompt injection. Adversarial prompts can be injected into external data sources to override the system's intended instruction and instead execute a malicious instruction. To mitigate this vulnerability, we propose a new defense called SecAlign based on the technique of preference optimization. Our defense first constructs a preference dataset with prompt-injected inputs, secure outputs (ones that respond to the legitimate instruction), and insecure outputs (ones that respond to the injection). We then perform preference optimization on this dataset to teach the LLM to prefer the secure output over the insecure one. This provides the first known method that reduces the success rates of various prompt injections to <10%, even against attacks much more sophisticated than ones seen during training. This indicates our defense generalizes well against unknown and yet-to-come attacks. Also, SecAlign models are still practical with similar utility to the one before defensive training in our evaluations. Our code is at https://github.com/facebookresearch/SecAlign



## **6. Semantic Structure-Aware Generative Attacks for Enhanced Adversarial Transferability**

cs.CV

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2506.18248v2) [paper-pdf](http://arxiv.org/pdf/2506.18248v2)

**Authors**: Jongoh Jeong, Hunmin Yang, Jaeseok Jeong, Kuk-Jin Yoon

**Abstract**: Generative adversarial attacks train a perturbation generator on a white-box surrogate model and subsequently apply the crafted perturbations to unseen black-box victim models. In contrast to iterative attacks, these methods deliver superior inference-time efficiency, scalability, and transferability; however, up until now, existing studies have not fully exploited the representational capacity of generative models to preserve and harness semantic information. Specifically, the intermediate activations of the generator encode rich semantic features--object boundaries and coarse shapes--that remain under-exploited, thereby limiting the alignment of perturbations with object-salient regions which are critical for adversarial transferability. To remedy this, we introduce a semantic structure-aware attack framework based on the Mean Teacher, which serves as a temporally smoothed feature reference. With this smoothed reference, we further direct semantic consistency between the early-layer activations in the student and those of the semantically rich teacher by feature distillation. By anchoring perturbation synthesis to the semantically salient early intermediate blocks within the generator based on empirical findings, our method guides progressive adversarial perturbation on regions that substantially enhance adversarial transferability. We conduct extensive experiments over diverse models, domains and tasks to demonstrate consistent improvements relative to state-of-the-art generative attacks, comprehensively evaluated using conventional metrics and our newly proposed Accidental Correction Rate (ACR).



## **7. Boosting Adversarial Transferability Against Defenses via Multi-Scale Transformation**

cs.CV

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01791v1) [paper-pdf](http://arxiv.org/pdf/2507.01791v1)

**Authors**: Zihong Guo, Chen Wan, Yayin Zheng, Hailing Kuang, Xiaohai Lu

**Abstract**: The transferability of adversarial examples poses a significant security challenge for deep neural networks, which can be attacked without knowing anything about them. In this paper, we propose a new Segmented Gaussian Pyramid (SGP) attack method to enhance the transferability, particularly against defense models. Unlike existing methods that generally focus on single-scale images, our approach employs Gaussian filtering and three types of downsampling to construct a series of multi-scale examples. Then, the gradients of the loss function with respect to each scale are computed, and their average is used to determine the adversarial perturbations. The proposed SGP can be considered an input transformation with high extensibility that is easily integrated into most existing adversarial attacks. Extensive experiments demonstrate that in contrast to the state-of-the-art methods, SGP significantly enhances attack success rates against black-box defense models, with average attack success rates increasing by 2.3% to 32.6%, based only on transferability.



## **8. Tuning without Peeking: Provable Privacy and Generalization Bounds for LLM Post-Training**

cs.LG

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01752v1) [paper-pdf](http://arxiv.org/pdf/2507.01752v1)

**Authors**: Ismail Labiad, Mathurin Videau, Matthieu Kowalski, Marc Schoenauer, Alessandro Leite, Julia Kempe, Olivier Teytaud

**Abstract**: Gradient-based optimization is the workhorse of deep learning, offering efficient and scalable training via backpropagation. However, its reliance on large volumes of labeled data raises privacy and security concerns such as susceptibility to data poisoning attacks and the risk of overfitting. In contrast, black box optimization methods, which treat the model as an opaque function, relying solely on function evaluations to guide optimization, offer a promising alternative in scenarios where data access is restricted, adversarial risks are high, or overfitting is a concern. However, black box methods also pose significant challenges, including poor scalability to high-dimensional parameter spaces, as prevalent in large language models (LLMs), and high computational costs due to reliance on numerous model evaluations. This paper introduces BBoxER, an evolutionary black-box method for LLM post-training that induces an information bottleneck via implicit compression of the training data. Leveraging the tractability of information flow, we provide strong theoretical bounds on generalization, differential privacy, susceptibility to data poisoning attacks, and robustness to extraction attacks. BBoxER operates on top of pre-trained LLMs, offering a lightweight and modular enhancement suitable for deployment in restricted or privacy-sensitive environments, in addition to non-vacuous generalization guarantees. In experiments with LLMs, we demonstrate empirically that Retrofitting methods are able to learn, showing how a few iterations of BBoxER improve performance and generalize well on a benchmark of reasoning datasets. This positions BBoxER as an attractive add-on on top of gradient-based optimization.



## **9. Blockchain Address Poisoning**

cs.CR

To appear in Proceedings of the 34th USENIX Security Symposium  (USENIX Security'25)

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2501.16681v3) [paper-pdf](http://arxiv.org/pdf/2501.16681v3)

**Authors**: Taro Tsuchiya, Jin-Dong Dong, Kyle Soska, Nicolas Christin

**Abstract**: In many blockchains, e.g., Ethereum, Binance Smart Chain (BSC), the primary representation used for wallet addresses is a hardly memorable 40-digit hexadecimal string. As a result, users often select addresses from their recent transaction history, which enables blockchain address poisoning. The adversary first generates lookalike addresses similar to one with which the victim has previously interacted, and then engages with the victim to ``poison'' their transaction history. The goal is to have the victim mistakenly send tokens to the lookalike address, as opposed to the intended recipient. Compared to contemporary studies, this paper provides four notable contributions. First, we develop a detection system and perform measurements over two years on both Ethereum and BSC. We identify 13~times more attack attempts than reported previously -- totaling 270M on-chain attacks targeting 17M victims. 6,633 incidents have caused at least 83.8M USD in losses, which makes blockchain address poisoning one of the largest cryptocurrency phishing schemes observed in the wild. Second, we analyze a few large attack entities using improved clustering techniques, and model attacker profitability and competition. Third, we reveal attack strategies -- targeted populations, success conditions (address similarity, timing), and cross-chain attacks. Fourth, we mathematically define and simulate the lookalike address generation process across various software- and hardware-based implementations, and identify a large-scale attacker group that appears to use GPUs. We also discuss defensive countermeasures.



## **10. Graph Representation-based Model Poisoning on Federated LLMs in CyberEdge Networks**

cs.CR

7 pages, 5 figures

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01694v1) [paper-pdf](http://arxiv.org/pdf/2507.01694v1)

**Authors**: Hanlin Cai, Haofan Dong, Houtianfu Wang, Kai Li, Ozgur B. Akan

**Abstract**: Federated large language models (FedLLMs) provide powerful generative capabilities in CyberEdge networks while protecting data privacy. However, FedLLMs remains highly vulnerable to model poisoning attacks. This article first reviews recent model poisoning techniques and existing defense mechanisms for FedLLMs, highlighting critical limitations, particularly under non-IID text distributions. In particular, current defenses primarily utilize distance-based outlier detection or norm constraints, operating under the assumption that adversarial updates significantly diverge from benign statistics. This assumption can fail when facing adaptive attackers targeting billionparameter LLMs. Next, this article investigates emerging Graph Representation-Based Model Poisoning (GRMP), a novel attack paradigm that leverages higher-order correlations among honest client gradients to synthesize malicious updates indistinguishable from legitimate model updates. GRMP can effectively evade advanced defenses, resulting in substantial accuracy loss and performance degradation. Moreover, this article outlines a research roadmap emphasizing the importance of graph-aware secure aggregation methods, FedLLMs-specific vulnerability metrics, and evaluation frameworks to strengthen the robustness of future federated language model deployments.



## **11. Learned-Database Systems Security**

cs.CR

Accepted at TMLR

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2212.10318v4) [paper-pdf](http://arxiv.org/pdf/2212.10318v4)

**Authors**: Roei Schuster, Jin Peng Zhou, Thorsten Eisenhofer, Paul Grubbs, Nicolas Papernot

**Abstract**: A learned database system uses machine learning (ML) internally to improve performance. We can expect such systems to be vulnerable to some adversarial-ML attacks. Often, the learned component is shared between mutually-distrusting users or processes, much like microarchitectural resources such as caches, potentially giving rise to highly-realistic attacker models. However, compared to attacks on other ML-based systems, attackers face a level of indirection as they cannot interact directly with the learned model. Additionally, the difference between the attack surface of learned and non-learned versions of the same system is often subtle. These factors obfuscate the de-facto risks that the incorporation of ML carries. We analyze the root causes of potentially-increased attack surface in learned database systems and develop a framework for identifying vulnerabilities that stem from the use of ML. We apply our framework to a broad set of learned components currently being explored in the database community. To empirically validate the vulnerabilities surfaced by our framework, we choose 3 of them and implement and evaluate exploits against these. We show that the use of ML cause leakage of past queries in a database, enable a poisoning attack that causes exponential memory blowup in an index structure and crashes it in seconds, and enable index users to snoop on each others' key distributions by timing queries over their own keys. We find that adversarial ML is an universal threat against learned components in database systems, point to open research gaps in our understanding of learned-systems security, and conclude by discussing mitigations, while noting that data leakage is inherent in systems whose learned component is shared between multiple parties.



## **12. Slot: Provenance-Driven APT Detection through Graph Reinforcement Learning**

cs.CR

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2410.17910v3) [paper-pdf](http://arxiv.org/pdf/2410.17910v3)

**Authors**: Wei Qiao, Yebo Feng, Teng Li, Zhuo Ma, Yulong Shen, JianFeng Ma, Yang Liu

**Abstract**: Advanced Persistent Threats (APTs) represent sophisticated cyberattacks characterized by their ability to remain undetected within the victim system for extended periods, aiming to exfiltrate sensitive data or disrupt operations. Existing detection approaches often struggle to effectively identify these complex threats, construct the attack chain for defense facilitation, or resist adversarial attacks. To overcome these challenges, we propose Slot, an advanced APT detection approach based on provenance graphs and graph reinforcement learning. Slot excels in uncovering multi-level hidden relationships, such as causal, contextual, and indirect connections, among system behaviors through provenance graph mining. By pioneering the integration of graph reinforcement learning, Slot dynamically adapts to new user activities and evolving attack strategies, enhancing its resilience against adversarial attacks. Additionally, Slot automatically constructs the attack chain according to detected attacks with clustering algorithms, providing precise identification of attack paths and facilitating the development of defense strategies. Evaluations with real-world datasets demonstrate Slot's outstanding accuracy, efficiency, adaptability, and robustness in APT detection, with most metrics surpassing state-of-the-art methods. Additionally, case studies conducted to assess Slot's effectiveness in supporting APT defense further establish it as a practical and reliable tool for cybersecurity protection.



## **13. DARTS: A Dual-View Attack Framework for Targeted Manipulation in Federated Sequential Recommendation**

cs.IR

10 pages. arXiv admin note: substantial text overlap with  arXiv:2409.07500; text overlap with arXiv:2212.05399 by other authors

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01383v1) [paper-pdf](http://arxiv.org/pdf/2507.01383v1)

**Authors**: Qitao Qin, Yucong Luo, Zhibo Chu

**Abstract**: Federated recommendation (FedRec) preserves user privacy by enabling decentralized training of personalized models, but this architecture is inherently vulnerable to adversarial attacks. Significant research has been conducted on targeted attacks in FedRec systems, motivated by commercial and social influence considerations. However, much of this work has largely overlooked the differential robustness of recommendation models. Moreover, our empirical findings indicate that existing targeted attack methods achieve only limited effectiveness in Federated Sequential Recommendation(FSR) tasks. Driven by these observations, we focus on investigating targeted attacks in FSR and propose a novel dualview attack framework, named DV-FSR. This attack method uniquely combines a sampling-based explicit strategy with a contrastive learning-based implicit gradient strategy to orchestrate a coordinated attack. Additionally, we introduce a specific defense mechanism tailored for targeted attacks in FSR, aiming to evaluate the mitigation effects of the attack method we proposed. Extensive experiments validate the effectiveness of our proposed approach on representative sequential models. Our codes are publicly available.



## **14. 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation**

cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01367v1) [paper-pdf](http://arxiv.org/pdf/2507.01367v1)

**Authors**: Tianrui Lou, Xiaojun Jia, Siyuan Liang, Jiawei Liang, Ming Zhang, Yanjun Xiao, Xiaochun Cao

**Abstract**: Physical adversarial attack methods expose the vulnerabilities of deep neural networks and pose a significant threat to safety-critical scenarios such as autonomous driving. Camouflage-based physical attack is a more promising approach compared to the patch-based attack, offering stronger adversarial effectiveness in complex physical environments. However, most prior work relies on mesh priors of the target object and virtual environments constructed by simulators, which are time-consuming to obtain and inevitably differ from the real world. Moreover, due to the limitations of the backgrounds in training images, previous methods often fail to produce multi-view robust adversarial camouflage and tend to fall into sub-optimal solutions. Due to these reasons, prior work lacks adversarial effectiveness and robustness across diverse viewpoints and physical environments. We propose a physical attack framework based on 3D Gaussian Splatting (3DGS), named PGA, which provides rapid and precise reconstruction with few images, along with photo-realistic rendering capabilities. Our framework further enhances cross-view robustness and adversarial effectiveness by preventing mutual and self-occlusion among Gaussians and employing a min-max optimization approach that adjusts the imaging background of each viewpoint, helping the algorithm filter out non-robust adversarial features. Extensive experiments validate the effectiveness and superiority of PGA. Our code is available at:https://github.com/TRLou/PGA.



## **15. Backdooring Bias (B^2) into Stable Diffusion Models**

cs.LG

Accepted to USENIX Security '25

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2406.15213v3) [paper-pdf](http://arxiv.org/pdf/2406.15213v3)

**Authors**: Ali Naseh, Jaechul Roh, Eugene Bagdasaryan, Amir Houmansadr

**Abstract**: Recent advances in large text-conditional diffusion models have revolutionized image generation by enabling users to create realistic, high-quality images from textual prompts, significantly enhancing artistic creation and visual communication. However, these advancements also introduce an underexplored attack opportunity: the possibility of inducing biases by an adversary into the generated images for malicious intentions, e.g., to influence public opinion and spread propaganda. In this paper, we study an attack vector that allows an adversary to inject arbitrary bias into a target model. The attack leverages low-cost backdooring techniques using a targeted set of natural textual triggers embedded within a small number of malicious data samples produced with public generative models. An adversary could pick common sequences of words that can then be inadvertently activated by benign users during inference. We investigate the feasibility and challenges of such attacks, demonstrating how modern generative models have made this adversarial process both easier and more adaptable. On the other hand, we explore various aspects of the detectability of such attacks and demonstrate that the model's utility remains intact in the absence of the triggers. Our extensive experiments using over 200,000 generated images and against hundreds of fine-tuned models demonstrate the feasibility of the presented backdoor attack. We illustrate how these biases maintain strong text-image alignment, highlighting the challenges in detecting biased images without knowing that bias in advance. Our cost analysis confirms the low financial barrier ($10-$15) to executing such attacks, underscoring the need for robust defensive strategies against such vulnerabilities in diffusion models.



## **16. ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks**

cs.LG

ICML 2025

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01321v1) [paper-pdf](http://arxiv.org/pdf/2507.01321v1)

**Authors**: Zhiyao Ren, Siyuan Liang, Aishan Liu, Dacheng Tao

**Abstract**: In-context learning (ICL) has demonstrated remarkable success in large language models (LLMs) due to its adaptability and parameter-free nature. However, it also introduces a critical vulnerability to backdoor attacks, where adversaries can manipulate LLM behaviors by simply poisoning a few ICL demonstrations. In this paper, we propose, for the first time, the dual-learning hypothesis, which posits that LLMs simultaneously learn both the task-relevant latent concepts and backdoor latent concepts within poisoned demonstrations, jointly influencing the probability of model outputs. Through theoretical analysis, we derive an upper bound for ICL backdoor effects, revealing that the vulnerability is dominated by the concept preference ratio between the task and the backdoor. Motivated by these findings, we propose ICLShield, a defense mechanism that dynamically adjusts the concept preference ratio. Our method encourages LLMs to select clean demonstrations during the ICL phase by leveraging confidence and similarity scores, effectively mitigating susceptibility to backdoor attacks. Extensive experiments across multiple LLMs and tasks demonstrate that our method achieves state-of-the-art defense effectiveness, significantly outperforming existing approaches (+26.02% on average). Furthermore, our method exhibits exceptional adaptability and defensive performance even for closed-source models (e.g., GPT-4).



## **17. Defensive Adversarial CAPTCHA: A Semantics-Driven Framework for Natural Adversarial Example Generation**

cs.CV

13 pages, 6 figures

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2506.10685v3) [paper-pdf](http://arxiv.org/pdf/2506.10685v3)

**Authors**: Xia Du, Xiaoyuan Liu, Jizhe Zhou, Zheng Lin, Chi-man Pun, Cong Wu, Tao Li, Zhe Chen, Wei Ni, Jun Luo

**Abstract**: Traditional CAPTCHA (Completely Automated Public Turing Test to Tell Computers and Humans Apart) schemes are increasingly vulnerable to automated attacks powered by deep neural networks (DNNs). Existing adversarial attack methods often rely on the original image characteristics, resulting in distortions that hinder human interpretation and limit their applicability in scenarios where no initial input images are available. To address these challenges, we propose the Unsourced Adversarial CAPTCHA (DAC), a novel framework that generates high-fidelity adversarial examples guided by attacker-specified semantics information. Leveraging a Large Language Model (LLM), DAC enhances CAPTCHA diversity and enriches the semantic information. To address various application scenarios, we examine the white-box targeted attack scenario and the black box untargeted attack scenario. For target attacks, we introduce two latent noise variables that are alternately guided in the diffusion step to achieve robust inversion. The synergy between gradient guidance and latent variable optimization achieved in this way ensures that the generated adversarial examples not only accurately align with the target conditions but also achieve optimal performance in terms of distributional consistency and attack effectiveness. In untargeted attacks, especially for black-box scenarios, we introduce bi-path unsourced adversarial CAPTCHA (BP-DAC), a two-step optimization strategy employing multimodal gradients and bi-path optimization for efficient misclassification. Experiments show that the defensive adversarial CAPTCHA generated by BP-DAC is able to defend against most of the unknown models, and the generated CAPTCHA is indistinguishable to both humans and DNNs.



## **18. CAVALRY-V: A Large-Scale Generator Framework for Adversarial Attacks on Video MLLMs**

cs.CV

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2507.00817v1) [paper-pdf](http://arxiv.org/pdf/2507.00817v1)

**Authors**: Jiaming Zhang, Rui Hu, Qing Guo, Wei Yang Bryan Lim

**Abstract**: Video Multimodal Large Language Models (V-MLLMs) have shown impressive capabilities in temporal reasoning and cross-modal understanding, yet their vulnerability to adversarial attacks remains underexplored due to unique challenges: complex cross-modal reasoning mechanisms, temporal dependencies, and computational constraints. We present CAVALRY-V (Cross-modal Language-Vision Adversarial Yielding for Videos), a novel framework that directly targets the critical interface between visual perception and language generation in V-MLLMs. Our approach introduces two key innovations: (1) a dual-objective semantic-visual loss function that simultaneously disrupts the model's text generation logits and visual representations to undermine cross-modal integration, and (2) a computationally efficient two-stage generator framework that combines large-scale pre-training for cross-model transferability with specialized fine-tuning for spatiotemporal coherence. Empirical evaluation on comprehensive video understanding benchmarks demonstrates that CAVALRY-V significantly outperforms existing attack methods, achieving 22.8% average improvement over the best baseline attacks on both commercial systems (GPT-4.1, Gemini 2.0) and open-source models (QwenVL-2.5, InternVL-2.5, Llava-Video, Aria, MiniCPM-o-2.6). Our framework achieves flexibility through implicit temporal coherence modeling rather than explicit regularization, enabling significant performance improvements even on image understanding (34.4% average gain). This capability demonstrates CAVALRY-V's potential as a foundational approach for adversarial research across multimodal systems.



## **19. Cage-Based Deformation for Transferable and Undefendable Point Cloud Attack**

cs.CV

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2507.00690v1) [paper-pdf](http://arxiv.org/pdf/2507.00690v1)

**Authors**: Keke Tang, Ziyong Du, Weilong Peng, Xiaofei Wang, Peican Zhu, Ligang Liu, Zhihong Tian

**Abstract**: Adversarial attacks on point clouds often impose strict geometric constraints to preserve plausibility; however, such constraints inherently limit transferability and undefendability. While deformation offers an alternative, existing unstructured approaches may introduce unnatural distortions, making adversarial point clouds conspicuous and undermining their plausibility. In this paper, we propose CageAttack, a cage-based deformation framework that produces natural adversarial point clouds. It first constructs a cage around the target object, providing a structured basis for smooth, natural-looking deformation. Perturbations are then applied to the cage vertices, which seamlessly propagate to the point cloud, ensuring that the resulting deformations remain intrinsic to the object and preserve plausibility. Extensive experiments on seven 3D deep neural network classifiers across three datasets show that CageAttack achieves a superior balance among transferability, undefendability, and plausibility, outperforming state-of-the-art methods. Codes will be made public upon acceptance.



## **20. How Resilient is QUIC to Security and Privacy Attacks?**

cs.CR

7 pages, 1 figure, 1 table

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2401.06657v3) [paper-pdf](http://arxiv.org/pdf/2401.06657v3)

**Authors**: Jayasree Sengupta, Debasmita Dey, Simone Ferlin-Reiter, Nirnay Ghosh, Vaibhav Bajpai

**Abstract**: QUIC has rapidly evolved into a cornerstone transport protocol for secure, low-latency communications, yet its deployment continues to expose critical security and privacy vulnerabilities, particularly during connection establishment phases and via traffic analysis. This paper systematically revisits a comprehensive set of attacks on QUIC and emerging privacy threats. Building upon these observations, we critically analyze recent IETF mitigation efforts, including TLS Encrypted Client Hello (ECH), Oblivious HTTP (OHTTP) and MASQUE. We analyze how these mechanisms enhance privacy while introducing new operational risks, particularly under adversarial load. Additionally, we discuss emerging challenges posed by post-quantum cryptographic (PQC) handshakes, including handshake expansion and metadata leakage risks. Our analysis highlights ongoing gaps between theoretical defenses and practical deployments, and proposes new research directions focused on adaptive privacy mechanisms. Building on these insights, we propose future directions to ensure long-term security of QUIC and aim to guide its evolution as a robust, privacy-preserving, and resilient transport foundation for the next-generation Internet.



## **21. Lazarus Group Targets Crypto-Wallets and Financial Data while employing new Tradecrafts**

cs.CR

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2505.21725v2) [paper-pdf](http://arxiv.org/pdf/2505.21725v2)

**Authors**: Alessio Di Santo

**Abstract**: This report presents a comprehensive analysis of a malicious software sample, detailing its architecture, behavioral characteristics, and underlying intent. Through static and dynamic examination, the malware core functionalities, including persistence mechanisms, command-and-control communication, and data exfiltration routines, are identified and its supporting infrastructure is mapped. By correlating observed indicators of compromise with known techniques, tactics, and procedures, this analysis situates the sample within the broader context of contemporary threat campaigns and infers the capabilities and motivations of its likely threat actor.   Building on these findings, actionable threat intelligence is provided to support proactive defenses. Threat hunting teams receive precise detection hypotheses for uncovering latent adversarial presence, while monitoring systems can refine alert logic to detect anomalous activity in real time. Finally, the report discusses how this structured intelligence enhances predictive risk assessments, informs vulnerability prioritization, and strengthens organizational resilience against advanced persistent threats. By integrating detailed technical insights with strategic threat landscape mapping, this malware analysis report not only reconstructs past adversary actions but also establishes a robust foundation for anticipating and mitigating future attacks.



## **22. Plug. Play. Persist. Inside a Ready-to-Go Havoc C2 Infrastructure**

cs.CR

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2507.00189v1) [paper-pdf](http://arxiv.org/pdf/2507.00189v1)

**Authors**: Alessio Di Santo

**Abstract**: This analysis focuses on a single Azure-hosted Virtual Machine at 52.230.23.114 that the adversary converted into an all-in-one delivery, staging and Command-and-Control node. The host advertises an out-of-date Apache 2.4.52 instance whose open directory exposes phishing lures, PowerShell loaders, Reflective Shell-Code, compiled Havoc Demon implants and a toolbox of lateral-movement binaries; the same server also answers on 8443/80 for encrypted beacon traffic. The web tier is riddled with publicly documented critical vulnerabilities, that would have allowed initial code-execution had the attackers not already owned the device.   Initial access is delivered through an HTML file that, once de-obfuscated, perfectly mimics Google Unusual sign-in attempt notification and funnels victims toward credential collection. A PowerShell command follows: it disables AMSI in-memory, downloads a Base64-encoded stub, allocates RWX pages and starts the shell-code without ever touching disk. That stub reconstructs a DLL in memory using the Reflective-Loader technique and hands control to Havoc Demon implant. Every Demon variant-32- and 64-bit alike-talks to the same backend, resolves Windows APIs with hashed look-ups, and hides its activity behind indirect syscalls.   Runtime telemetry shows interests in registry under Image File Execution Options, deliberate queries to Software Restriction Policy keys, and heavy use of Crypto DLLs to protect payloads and C2 traffic. The attacker toolkit further contains Chisel, PsExec, Doppelganger and Whisker, some of them re-compiled under user directories that leak the developer personas tonzking123 and thobt. Collectively the findings paint a picture of a technically adept actor who values rapid re-tooling over deep operational security, leaning on Havoc modularity and on legitimate cloud services to blend malicious flows into ordinary enterprise traffic.



## **23. SQUASH: A SWAP-Based Quantum Attack to Sabotage Hybrid Quantum Neural Networks**

quant-ph

Keywords: Quantum Machine Learning, Hybrid Quantum Neural Networks,  SWAP Test, Fidelity, Circuit-level Attack

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24081v1) [paper-pdf](http://arxiv.org/pdf/2506.24081v1)

**Authors**: Rahul Kumar, Wenqi Wei, Ying Mao, Junaid Farooq, Ying Wang, Juntao Chen

**Abstract**: We propose a circuit-level attack, SQUASH, a SWAP-Based Quantum Attack to sabotage Hybrid Quantum Neural Networks (HQNNs) for classification tasks. SQUASH is executed by inserting SWAP gate(s) into the variational quantum circuit of the victim HQNN. Unlike conventional noise-based or adversarial input attacks, SQUASH directly manipulates the circuit structure, leading to qubit misalignment and disrupting quantum state evolution. This attack is highly stealthy, as it does not require access to training data or introduce detectable perturbations in input states. Our results demonstrate that SQUASH significantly degrades classification performance, with untargeted SWAP attacks reducing accuracy by up to 74.08\% and targeted SWAP attacks reducing target class accuracy by up to 79.78\%. These findings reveal a critical vulnerability in HQNN implementations, underscoring the need for more resilient architectures against circuit-level adversarial interventions.



## **24. STACK: Adversarial Attacks on LLM Safeguard Pipelines**

cs.CL

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24068v1) [paper-pdf](http://arxiv.org/pdf/2506.24068v1)

**Authors**: Ian R. McKenzie, Oskar J. Hollinsworth, Tom Tseng, Xander Davies, Stephen Casper, Aaron D. Tucker, Robert Kirk, Adam Gleave

**Abstract**: Frontier AI developers are relying on layers of safeguards to protect against catastrophic misuse of AI systems. Anthropic guards their latest Claude 4 Opus model using one such defense pipeline, and other frontier developers including Google DeepMind and OpenAI pledge to soon deploy similar defenses. However, the security of such pipelines is unclear, with limited prior work evaluating or attacking these pipelines. We address this gap by developing and red-teaming an open-source defense pipeline. First, we find that a novel few-shot-prompted input and output classifier outperforms state-of-the-art open-weight safeguard model ShieldGemma across three attacks and two datasets, reducing the attack success rate (ASR) to 0% on the catastrophic misuse dataset ClearHarm. Second, we introduce a STaged AttaCK (STACK) procedure that achieves 71% ASR on ClearHarm in a black-box attack against the few-shot-prompted classifier pipeline. Finally, we also evaluate STACK in a transfer setting, achieving 33% ASR, providing initial evidence that it is feasible to design attacks with no access to the target pipeline. We conclude by suggesting specific mitigations that developers could use to thwart staged attacks.



## **25. Consensus-based optimization for closed-box adversarial attacks and a connection to evolution strategies**

math.OC

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24048v1) [paper-pdf](http://arxiv.org/pdf/2506.24048v1)

**Authors**: Tim Roith, Leon Bungert, Philipp Wacker

**Abstract**: Consensus-based optimization (CBO) has established itself as an efficient gradient-free optimization scheme, with attractive mathematical properties, such as mean-field convergence results for non-convex loss functions. In this work, we study CBO in the context of closed-box adversarial attacks, which are imperceptible input perturbations that aim to fool a classifier, without accessing its gradient. Our contribution is to establish a connection between the so-called consensus hopping as introduced by Riedl et al. and natural evolution strategies (NES) commonly applied in the context of adversarial attacks and to rigorously relate both methods to gradient-based optimization schemes. Beyond that, we provide a comprehensive experimental study that shows that despite the conceptual similarities, CBO can outperform NES and other evolutionary strategies in certain scenarios.



## **26. Quickest Detection of Adversarial Attacks Against Correlated Equilibria**

cs.GT

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24040v1) [paper-pdf](http://arxiv.org/pdf/2506.24040v1)

**Authors**: Kiarash Kazari, Aris Kanellopoulos, György Dán

**Abstract**: We consider correlated equilibria in strategic games in an adversarial environment, where an adversary can compromise the public signal used by the players for choosing their strategies, while players aim at detecting a potential attack as soon as possible to avoid loss of utility. We model the interaction between the adversary and the players as a zero-sum game and we derive the maxmin strategies for both the defender and the attacker using the framework of quickest change detection. We define a class of adversarial strategies that achieve the optimal trade-off between attack impact and attack detectability and show that a generalized CUSUM scheme is asymptotically optimal for the detection of the attacks. Our numerical results on the Sioux-Falls benchmark traffic routing game show that the proposed detection scheme can effectively limit the utility loss by a potential adversary.



## **27. Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation**

cs.CR

This is the full version (27 pages) of the paper 'Riddle Me This!  Stealthy Membership Inference for Retrieval-Augmented Generation' published  at CCS 2025

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2502.00306v2) [paper-pdf](http://arxiv.org/pdf/2502.00306v2)

**Authors**: Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr

**Abstract**: Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. Although the absence of weight tuning prevents leakage via model parameters, it introduces the risk of inference adversaries exploiting retrieved documents in the model's context. Existing methods for membership inference and data extraction often rely on jailbreaking or carefully crafted unnatural queries, which can be easily detected or thwarted with query rewriting techniques common in RAG systems. In this work, we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore. By crafting natural-text queries that are answerable only with the target document's presence, our approach demonstrates successful inference with just 30 queries while remaining stealthy; straightforward detectors identify adversarial prompts from existing methods up to ~76x more frequently than those generated by our attack. We observe a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations, all while costing less than $0.02 per document inference.



## **28. Benchmarking Spiking Neural Network Learning Methods with Varying Locality**

cs.NE

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2402.01782v2) [paper-pdf](http://arxiv.org/pdf/2402.01782v2)

**Authors**: Jiaqi Lin, Sen Lu, Malyaban Bal, Abhronil Sengupta

**Abstract**: Spiking Neural Networks (SNNs), providing more realistic neuronal dynamics, have been shown to achieve performance comparable to Artificial Neural Networks (ANNs) in several machine learning tasks. Information is processed as spikes within SNNs in an event-based mechanism that significantly reduces energy consumption. However, training SNNs is challenging due to the non-differentiable nature of the spiking mechanism. Traditional approaches, such as Backpropagation Through Time (BPTT), have shown effectiveness but come with additional computational and memory costs and are biologically implausible. In contrast, recent works propose alternative learning methods with varying degrees of locality, demonstrating success in classification tasks. In this work, we show that these methods share similarities during the training process, while they present a trade-off between biological plausibility and performance. Further, given the implicitly recurrent nature of SNNs, this research investigates the influence of the addition of explicit recurrence to SNNs. We experimentally prove that the addition of explicit recurrent weights enhances the robustness of SNNs. We also investigate the performance of local learning methods under gradient and non-gradient-based adversarial attacks.



## **29. A Unified Framework for Stealthy Adversarial Generation via Latent Optimization and Transferability Enhancement**

cs.CV

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.23676v1) [paper-pdf](http://arxiv.org/pdf/2506.23676v1)

**Authors**: Gaozheng Pei, Ke Ma, Dongpeng Zhang, Chengzhi Sun, Qianqian Xu, Qingming Huang

**Abstract**: Due to their powerful image generation capabilities, diffusion-based adversarial example generation methods through image editing are rapidly gaining popularity. However, due to reliance on the discriminative capability of the diffusion model, these diffusion-based methods often struggle to generalize beyond conventional image classification tasks, such as in Deepfake detection. Moreover, traditional strategies for enhancing adversarial example transferability are challenging to adapt to these methods. To address these challenges, we propose a unified framework that seamlessly incorporates traditional transferability enhancement strategies into diffusion model-based adversarial example generation via image editing, enabling their application across a wider range of downstream tasks. Our method won first place in the "1st Adversarial Attacks on Deepfake Detectors: A Challenge in the Era of AI-Generated Media" competition at ACM MM25, which validates the effectiveness of our approach.



## **30. PBCAT: Patch-based composite adversarial training against physically realizable attacks on object detection**

cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.23581v1) [paper-pdf](http://arxiv.org/pdf/2506.23581v1)

**Authors**: Xiao Li, Yiming Zhu, Yifan Huang, Wei Zhang, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Object detection plays a crucial role in many security-sensitive applications. However, several recent studies have shown that object detectors can be easily fooled by physically realizable attacks, \eg, adversarial patches and recent adversarial textures, which pose realistic and urgent threats. Adversarial Training (AT) has been recognized as the most effective defense against adversarial attacks. While AT has been extensively studied in the $l_\infty$ attack settings on classification models, AT against physically realizable attacks on object detectors has received limited exploration. Early attempts are only performed to defend against adversarial patches, leaving AT against a wider range of physically realizable attacks under-explored. In this work, we consider defending against various physically realizable attacks with a unified AT method. We propose PBCAT, a novel Patch-Based Composite Adversarial Training strategy. PBCAT optimizes the model by incorporating the combination of small-area gradient-guided adversarial patches and imperceptible global adversarial perturbations covering the entire image. With these designs, PBCAT has the potential to defend against not only adversarial patches but also unseen physically realizable attacks such as adversarial textures. Extensive experiments in multiple settings demonstrated that PBCAT significantly improved robustness against various physically realizable attacks over state-of-the-art defense methods. Notably, it improved the detection accuracy by 29.7\% over previous defense methods under one recent adversarial texture attack.



## **31. Efficient Resource Allocation under Adversary Attacks: A Decomposition-Based Approach**

cs.DS

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.23442v1) [paper-pdf](http://arxiv.org/pdf/2506.23442v1)

**Authors**: Mansoor Davoodi, Setareh Maghsudi

**Abstract**: We address the problem of allocating limited resources in a network under persistent yet statistically unknown adversarial attacks. Each node in the network may be degraded, but not fully disabled, depending on its available defensive resources. The objective is twofold: to minimize total system damage and to reduce cumulative resource allocation and transfer costs over time. We model this challenge as a bi-objective optimization problem and propose a decomposition-based solution that integrates chance-constrained programming with network flow optimization. The framework separates the problem into two interrelated subproblems: determining optimal node-level allocations across time slots, and computing efficient inter-node resource transfers. We theoretically prove the convergence of our method to the optimal solution that would be obtained with full statistical knowledge of the adversary. Extensive simulations demonstrate that our method efficiently learns the adversarial patterns and achieves substantial gains in minimizing both damage and operational costs, comparing three benchmark strategies under various parameter settings.



## **32. TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs**

cs.CL

ICML 2025

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2506.23423v1) [paper-pdf](http://arxiv.org/pdf/2506.23423v1)

**Authors**: Felipe Nuti, Tim Franzmeyer, João Henriques

**Abstract**: Past work has studied the effects of fine-tuning on large language models' (LLMs) overall performance on certain tasks. However, a quantitative and systematic method for analyzing its effect on individual outputs is still lacking. Here, we propose a new method for measuring the contribution that fine-tuning makes to individual LLM responses, assuming access to the original pre-trained model. Our method tracks the model's intermediate hidden states, providing a more fine-grained insight into the effects of fine-tuning than a simple comparison of final outputs from pre-trained and fine-tuned models. We introduce and theoretically analyze an exact decomposition of any fine-tuned LLM into a pre-training component and a fine-tuning component. Empirically, we find that model behavior and performance can be steered by up- or down-scaling the fine-tuning component during the forward pass. Motivated by this finding and our theoretical analysis, we define the Tuning Contribution (TuCo) as the ratio of the magnitudes of the fine-tuning component to the pre-training component. We observe that three prominent adversarial attacks on LLMs circumvent safety measures in a way that reduces TuCo, and that TuCo is consistently lower on prompts where these attacks succeed compared to those where they do not. This suggests that attenuating the effect of fine-tuning on model outputs plays a role in the success of such attacks. In summary, TuCo enables the quantitative study of how fine-tuning influences model behavior and safety, and vice versa.



## **33. Enhancing Adversarial Robustness through Multi-Objective Representation Learning**

cs.LG

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2410.01697v4) [paper-pdf](http://arxiv.org/pdf/2410.01697v4)

**Authors**: Sedjro Salomon Hotegni, Sebastian Peitz

**Abstract**: Deep neural networks (DNNs) are vulnerable to small adversarial perturbations, which are tiny changes to the input data that appear insignificant but cause the model to produce drastically different outputs. Many defense methods require modifying model architectures during evaluation or performing test-time data purification. This not only introduces additional complexity but is often architecture-dependent. We show, however, that robust feature learning during training can significantly enhance DNN robustness. We propose MOREL, a multi-objective approach that aligns natural and adversarial features using cosine similarity and multi-positive contrastive losses to encourage similar features for same-class inputs. Extensive experiments demonstrate that MOREL significantly improves robustness against both white-box and black-box attacks. Our code is available at https://github.com/salomonhotegni/MOREL



## **34. Adversarial Robustness Unhardening via Backdoor Attacks in Federated Learning**

cs.LG

15 pages, 8 main pages of text, 13 figures, 5 tables. Made for a  Neurips workshop on backdoor attacks - extended version

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2310.11594v3) [paper-pdf](http://arxiv.org/pdf/2310.11594v3)

**Authors**: Taejin Kim, Jiarui Li, Shubhranshu Singh, Nikhil Madaan, Carlee Joe-Wong

**Abstract**: The delicate equilibrium between user privacy and the ability to unleash the potential of distributed data is an important concern. Federated learning, which enables the training of collaborative models without sharing of data, has emerged as a privacy-centric solution. This approach brings forth security challenges, notably poisoning and backdoor attacks where malicious entities inject corrupted data into the training process, as well as evasion attacks that aim to induce misclassifications at test time. Our research investigates the intersection of adversarial training, a common defense method against evasion attacks, and backdoor attacks within federated learning. We introduce Adversarial Robustness Unhardening (ARU), which is employed by a subset of adversarial clients to intentionally undermine model robustness during federated training, rendering models susceptible to a broader range of evasion attacks. We present extensive experiments evaluating ARU's impact on adversarial training and existing robust aggregation defenses against poisoning and backdoor attacks. Our results show that ARU can substantially undermine adversarial training's ability to harden models against test-time evasion attacks, and that adversaries employing ARU can even evade robust aggregation defenses that often neutralize poisoning or backdoor attacks.



## **35. Scaling Laws for Black box Adversarial Attacks**

cs.LG

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2411.16782v3) [paper-pdf](http://arxiv.org/pdf/2411.16782v3)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

**Abstract**: Adversarial examples usually exhibit good cross-model transferability, enabling attacks on black-box models with limited information about their architectures and parameters, which are highly threatening in commercial black-box scenarios. Model ensembling is an effective strategy to improve the transferability of adversarial examples by attacking multiple surrogate models. However, since prior studies usually adopt few models in the ensemble, there remains an open question of whether scaling the number of models can further improve black-box attacks. Inspired by the scaling law of large foundation models, we investigate the scaling laws of black-box adversarial attacks in this work. Through theoretical analysis and empirical evaluations, we conclude with clear scaling laws that using more surrogate models enhances adversarial transferability. Comprehensive experiments verify the claims on standard image classifiers, diverse defended models and multimodal large language models using various adversarial attack methods. Specifically, by scaling law, we achieve 90%+ transfer attack success rate on even proprietary models like GPT-4o. Further visualization indicates that there is also a scaling law on the interpretability and semantics of adversarial perturbations.



## **36. MedLeak: Multimodal Medical Data Leakage in Secure Federated Learning with Crafted Models**

cs.LG

Accepted by the IEEE/ACM conference on Connected Health:  Applications, Systems and Engineering Technologies 2025 (CHASE'25)

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2407.09972v2) [paper-pdf](http://arxiv.org/pdf/2407.09972v2)

**Authors**: Shanghao Shi, Md Shahedul Haque, Abhijeet Parida, Chaoyu Zhang, Marius George Linguraru, Y. Thomas Hou, Syed Muhammad Anwar, Wenjing Lou

**Abstract**: Federated learning (FL) allows participants to collaboratively train machine learning models while keeping their data local, making it ideal for collaborations among healthcare institutions on sensitive data. However, in this paper, we propose a novel privacy attack called MedLeak, which allows a malicious FL server to recover high-quality site-specific private medical data from the client model updates. MedLeak works by introducing an adversarially crafted model during the FL training process. Honest clients, unaware of the insidious changes in the published models, continue to send back their updates as per the standard FL protocol. Leveraging a novel analytical method, MedLeak can efficiently recover private client data from the aggregated parameter updates, eliminating costly optimization. In addition, the scheme relies solely on the aggregated updates, thus rendering secure aggregation protocols ineffective, as they depend on the randomization of intermediate results for security while leaving the final aggregated results unaltered.   We implement MedLeak on medical image datasets (MedMNIST, COVIDx CXR-4, and Kaggle Brain Tumor MRI), as well as a medical text dataset (MedAbstract). The results demonstrate that our attack achieves high recovery rates and strong quantitative scores on both image and text datasets. We also thoroughly evaluate MedLeak across different attack parameters, providing insights into key factors that influence attack performance and potential defenses. Furthermore, we demonstrate that the recovered data can support downstream tasks such as disease classification with minimal performance loss. Our findings validate the need for enhanced privacy measures in FL systems, particularly for safeguarding sensitive medical data against powerful model inversion attacks.



## **37. Securing AI Systems: A Guide to Known Attacks and Impacts**

cs.CR

34 pages, 16 figures

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2506.23296v1) [paper-pdf](http://arxiv.org/pdf/2506.23296v1)

**Authors**: Naoto Kiribuchi, Kengo Zenitani, Takayuki Semitsu

**Abstract**: Embedded into information systems, artificial intelligence (AI) faces security threats that exploit AI-specific vulnerabilities. This paper provides an accessible overview of adversarial attacks unique to predictive and generative AI systems. We identify eleven major attack types and explicitly link attack techniques to their impacts -- including information leakage, system compromise, and resource exhaustion -- mapped to the confidentiality, integrity, and availability (CIA) security triad. We aim to equip researchers, developers, security practitioners, and policymakers, even those without specialized AI security expertise, with foundational knowledge to recognize AI-specific risks and implement effective defenses, thereby enhancing the overall security posture of AI systems.



## **38. From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agents Workflows**

cs.CR

29 pages, 15 figures, 6 tables

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2506.23260v1) [paper-pdf](http://arxiv.org/pdf/2506.23260v1)

**Authors**: Mohamed Amine Ferrag, Norbert Tihanyi, Djallel Hamouda, Leandros Maglaras, Merouane Debbah

**Abstract**: Autonomous AI agents powered by large language models (LLMs) with structured function-calling interfaces have dramatically expanded capabilities for real-time data retrieval, complex computation, and multi-step orchestration. Yet, the explosive proliferation of plugins, connectors, and inter-agent protocols has outpaced discovery mechanisms and security practices, resulting in brittle integrations vulnerable to diverse threats. In this survey, we introduce the first unified, end-to-end threat model for LLM-agent ecosystems, spanning host-to-tool and agent-to-agent communications, formalize adversary capabilities and attacker objectives, and catalog over thirty attack techniques. Specifically, we organized the threat model into four domains: Input Manipulation (e.g., prompt injections, long-context hijacks, multimodal adversarial inputs), Model Compromise (e.g., prompt- and parameter-level backdoors, composite and encrypted multi-backdoors, poisoning strategies), System and Privacy Attacks (e.g., speculative side-channels, membership inference, retrieval poisoning, social-engineering simulations), and Protocol Vulnerabilities (e.g., exploits in Model Context Protocol (MCP), Agent Communication Protocol (ACP), Agent Network Protocol (ANP), and Agent-to-Agent (A2A) protocol). For each category, we review representative scenarios, assess real-world feasibility, and evaluate existing defenses. Building on our threat taxonomy, we identify key open challenges and future research directions, such as securing MCP deployments through dynamic trust management and cryptographic provenance tracking; designing and hardening Agentic Web Interfaces; and achieving resilience in multi-agent and federated environments. Our work provides a comprehensive reference to guide the design of robust defense mechanisms and establish best practices for resilient LLM-agent workflows.



## **39. Fragile, Robust, and Antifragile: A Perspective from Parameter Responses in Reinforcement Learning Under Stress**

cs.LG

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.23036v1) [paper-pdf](http://arxiv.org/pdf/2506.23036v1)

**Authors**: Zain ul Abdeen, Ming Jin

**Abstract**: This paper explores Reinforcement learning (RL) policy robustness by systematically analyzing network parameters under internal and external stresses. Inspired by synaptic plasticity in neuroscience, synaptic filtering introduces internal stress by selectively perturbing parameters, while adversarial attacks apply external stress through modified agent observations. This dual approach enables the classification of parameters as fragile, robust, or antifragile, based on their influence on policy performance in clean and adversarial settings. Parameter scores are defined to quantify these characteristics, and the framework is validated on PPO-trained agents in Mujoco continuous control environments. The results highlight the presence of antifragile parameters that enhance policy performance under stress, demonstrating the potential of targeted filtering techniques to improve RL policy adaptability. These insights provide a foundation for future advancements in the design of robust and antifragile RL systems.



## **40. Revisiting CroPA: A Reproducibility Study and Enhancements for Cross-Prompt Adversarial Transferability in Vision-Language Models**

cs.CV

Accepted to MLRC 2025

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.22982v1) [paper-pdf](http://arxiv.org/pdf/2506.22982v1)

**Authors**: Atharv Mittal, Agam Pandey, Amritanshu Tiwari, Sukrit Jindal, Swadesh Swain

**Abstract**: Large Vision-Language Models (VLMs) have revolutionized computer vision, enabling tasks such as image classification, captioning, and visual question answering. However, they remain highly vulnerable to adversarial attacks, particularly in scenarios where both visual and textual modalities can be manipulated. In this study, we conduct a comprehensive reproducibility study of "An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models" validating the Cross-Prompt Attack (CroPA) and confirming its superior cross-prompt transferability compared to existing baselines. Beyond replication we propose several key improvements: (1) A novel initialization strategy that significantly improves Attack Success Rate (ASR). (2) Investigate cross-image transferability by learning universal perturbations. (3) A novel loss function targeting vision encoder attention mechanisms to improve generalization. Our evaluation across prominent VLMs -- including Flamingo, BLIP-2, and InstructBLIP as well as extended experiments on LLaVA validates the original results and demonstrates that our improvements consistently boost adversarial effectiveness. Our work reinforces the importance of studying adversarial vulnerabilities in VLMs and provides a more robust framework for generating transferable adversarial examples, with significant implications for understanding the security of VLMs in real-world applications.



## **41. VFEFL: Privacy-Preserving Federated Learning against Malicious Clients via Verifiable Functional Encryption**

cs.CR

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.12846v2) [paper-pdf](http://arxiv.org/pdf/2506.12846v2)

**Authors**: Nina Cai, Jinguang Han, Weizhi Meng

**Abstract**: Federated learning is a promising distributed learning paradigm that enables collaborative model training without exposing local client data, thereby protect data privacy. However, it also brings new threats and challenges. The advancement of model inversion attacks has rendered the plaintext transmission of local models insecure, while the distributed nature of federated learning makes it particularly vulnerable to attacks raised by malicious clients. To protect data privacy and prevent malicious client attacks, this paper proposes a privacy-preserving federated learning framework based on verifiable functional encryption, without a non-colluding dual-server setup or additional trusted third-party. Specifically, we propose a novel decentralized verifiable functional encryption (DVFE) scheme that enables the verification of specific relationships over multi-dimensional ciphertexts. This scheme is formally treated, in terms of definition, security model and security proof. Furthermore, based on the proposed DVFE scheme, we design a privacy-preserving federated learning framework VFEFL that incorporates a novel robust aggregation rule to detect malicious clients, enabling the effective training of high-accuracy models under adversarial settings. Finally, we provide formal analysis and empirical evaluation of the proposed schemes. The results demonstrate that our approach achieves the desired privacy protection, robustness, verifiability and fidelity, while eliminating the reliance on non-colluding dual-server settings or trusted third parties required by existing methods.



## **42. Concept Pinpoint Eraser for Text-to-image Diffusion Models via Residual Attention Gate**

cs.CV

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.22806v1) [paper-pdf](http://arxiv.org/pdf/2506.22806v1)

**Authors**: Byung Hyun Lee, Sungjin Lim, Seunggyu Lee, Dong Un Kang, Se Young Chun

**Abstract**: Remarkable progress in text-to-image diffusion models has brought a major concern about potentially generating images on inappropriate or trademarked concepts. Concept erasing has been investigated with the goals of deleting target concepts in diffusion models while preserving other concepts with minimal distortion. To achieve these goals, recent concept erasing methods usually fine-tune the cross-attention layers of diffusion models. In this work, we first show that merely updating the cross-attention layers in diffusion models, which is mathematically equivalent to adding \emph{linear} modules to weights, may not be able to preserve diverse remaining concepts. Then, we propose a novel framework, dubbed Concept Pinpoint Eraser (CPE), by adding \emph{nonlinear} Residual Attention Gates (ResAGs) that selectively erase (or cut) target concepts while safeguarding remaining concepts from broad distributions by employing an attention anchoring loss to prevent the forgetting. Moreover, we adversarially train CPE with ResAG and learnable text embeddings in an iterative manner to maximize erasing performance and enhance robustness against adversarial attacks. Extensive experiments on the erasure of celebrities, artistic styles, and explicit contents demonstrated that the proposed CPE outperforms prior arts by keeping diverse remaining concepts while deleting the target concepts with robustness against attack prompts. Code is available at https://github.com/Hyun1A/CPE



## **43. Enhancing the Capability and Robustness of Large Language Models through Reinforcement Learning-Driven Query Refinement**

cs.CL

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2407.01461v3) [paper-pdf](http://arxiv.org/pdf/2407.01461v3)

**Authors**: Xiaohua Wang, Zisu Huang, Feiran Zhang, Zhibo Xu, Cenyuan Zhang, Qi Qian, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The capacity of large language models (LLMs) to generate honest, harmless, and helpful responses heavily relies on the quality of user prompts. However, these prompts often tend to be brief and vague, thereby significantly limiting the full potential of LLMs. Moreover, harmful prompts can be meticulously crafted and manipulated by adversaries to jailbreak LLMs, inducing them to produce potentially toxic content. To enhance the capabilities of LLMs while maintaining strong robustness against harmful jailbreak inputs, this study proposes a transferable and pluggable framework that refines user prompts before they are input into LLMs. This strategy improves the quality of the queries, empowering LLMs to generate more truthful, benign and useful responses. Specifically, a lightweight query refinement model is introduced and trained using a specially designed reinforcement learning approach that incorporates multiple objectives to enhance particular capabilities of LLMs. Extensive experiments demonstrate that the refinement model not only improves the quality of responses but also strengthens their robustness against jailbreak attacks. Code is available at: https://github.com/Huangzisu/query-refinement .



## **44. Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code Generation**

cs.SE

13 pages, 6 figures

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.22776v1) [paper-pdf](http://arxiv.org/pdf/2506.22776v1)

**Authors**: Sen Fang, Weiyuan Ding, Antonio Mastropaolo, Bowen Xu

**Abstract**: Quantization has emerged as a mainstream method for compressing Large Language Models (LLMs), reducing memory requirements and accelerating inference without architectural modifications. While existing research primarily focuses on evaluating the effectiveness of quantized LLMs compared to their original counterparts, the impact on robustness remains largely unexplored.In this paper, we present the first systematic investigation of how quantization affects the robustness of LLMs in code generation tasks. Through extensive experiments across four prominent LLM families (LLaMA, DeepSeek, CodeGen, and StarCoder) with parameter scales ranging from 350M to 33B, we evaluate robustness from dual perspectives: adversarial attacks on input prompts and noise perturbations on model architecture. Our findings challenge conventional wisdom by demonstrating that quantized LLMs often exhibit superior robustness compared to their full-precision counterparts, with 51.59% versus 42.86% of our adversarial experiments showing better resilience in quantized LLMs. Similarly, our noise perturbation experiments also confirm that LLMs after quantitation generally withstand higher levels of weight disturbances. These results suggest that quantization not only reduces computational requirements but can actually enhance LLMs' reliability in code generation tasks, providing valuable insights for developing more robust and efficient LLM deployment strategies.



## **45. Kill Two Birds with One Stone! Trajectory enabled Unified Online Detection of Adversarial Examples and Backdoor Attacks**

cs.CR

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.22722v1) [paper-pdf](http://arxiv.org/pdf/2506.22722v1)

**Authors**: Anmin Fu, Fanyu Meng, Huaibing Peng, Hua Ma, Zhi Zhang, Yifeng Zheng, Willy Susilo, Yansong Gao

**Abstract**: The proposed UniGuard is the first unified online detection framework capable of simultaneously addressing adversarial examples and backdoor attacks. UniGuard builds upon two key insights: first, both AE and backdoor attacks have to compromise the inference phase, making it possible to tackle them simultaneously during run-time via online detection. Second, an adversarial input, whether a perturbed sample in AE attacks or a trigger-carrying sample in backdoor attacks, exhibits distinctive trajectory signatures from a benign sample as it propagates through the layers of a DL model in forward inference. The propagation trajectory of the adversarial sample must deviate from that of its benign counterpart; otherwise, the adversarial objective cannot be fulfilled. Detecting these trajectory signatures is inherently challenging due to their subtlety; UniGuard overcomes this by treating the propagation trajectory as a time-series signal, leveraging LSTM and spectrum transformation to amplify differences between adversarial and benign trajectories that are subtle in the time domain. UniGuard exceptional efficiency and effectiveness have been extensively validated across various modalities (image, text, and audio) and tasks (classification and regression), ranging from diverse model architectures against a wide range of AE attacks and backdoor attacks, including challenging partial backdoors and dynamic triggers. When compared to SOTA methods, including ContraNet (NDSS 22) specific for AE detection and TED (IEEE SP 24) specific for backdoor detection, UniGuard consistently demonstrates superior performance, even when matched against each method's strengths in addressing their respective threats-each SOTA fails to parts of attack strategies while UniGuard succeeds for all.



## **46. VERA: Variational Inference Framework for Jailbreaking Large Language Models**

cs.CR

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.22666v1) [paper-pdf](http://arxiv.org/pdf/2506.22666v1)

**Authors**: Anamika Lochab, Lu Yan, Patrick Pynadath, Xiangyu Zhang, Ruqi Zhang

**Abstract**: The rise of API-only access to state-of-the-art LLMs highlights the need for effective black-box jailbreak methods to identify model vulnerabilities in real-world settings. Without a principled objective for gradient-based optimization, most existing approaches rely on genetic algorithms, which are limited by their initialization and dependence on manually curated prompt pools. Furthermore, these methods require individual optimization for each prompt, failing to provide a comprehensive characterization of model vulnerabilities. To address this gap, we introduce VERA: Variational infErence fRamework for jAilbreaking. VERA casts black-box jailbreak prompting as a variational inference problem, training a small attacker LLM to approximate the target LLM's posterior over adversarial prompts. Once trained, the attacker can generate diverse, fluent jailbreak prompts for a target query without re-optimization. Experimental results show that VERA achieves strong performance across a range of target LLMs, highlighting the value of probabilistic inference for adversarial prompt generation.



## **47. MetaCipher: A General and Extensible Reinforcement Learning Framework for Obfuscation-Based Jailbreak Attacks on Black-Box LLMs**

cs.CR

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.22557v1) [paper-pdf](http://arxiv.org/pdf/2506.22557v1)

**Authors**: Boyuan Chen, Minghao Shao, Abdul Basit, Siddharth Garg, Muhammad Shafique

**Abstract**: The growing capabilities of large language models (LLMs) have exposed them to increasingly sophisticated jailbreak attacks. Among these, obfuscation-based attacks -- which encrypt malicious content to evade detection -- remain highly effective. By leveraging the reasoning ability of advanced LLMs to interpret encrypted prompts, such attacks circumvent conventional defenses that rely on keyword detection or context filtering. These methods are very difficult to defend against, as existing safety mechanisms are not designed to interpret or decode ciphered content. In this work, we propose \textbf{MetaCipher}, a novel obfuscation-based jailbreak framework, along with a reinforcement learning-based dynamic cipher selection mechanism that adaptively chooses optimal encryption strategies from a cipher pool. This approach enhances jailbreak effectiveness and generalizability across diverse task types, victim LLMs, and safety guardrails. Our framework is modular and extensible by design, supporting arbitrary cipher families and accommodating evolving adversarial strategies. We complement our method with a large-scale empirical analysis of cipher performance across multiple victim LLMs. Within as few as 10 queries, MetaCipher achieves over 92\% attack success rate (ASR) on most recent standard malicious prompt benchmarks against state-of-the-art non-reasoning LLMs, and over 74\% ASR against reasoning-capable LLMs, outperforming all existing obfuscation-based jailbreak methods. These results highlight the long-term robustness and adaptability of our approach, making it more resilient than prior methods in the face of advancing safety measures.



## **48. ARMOR: Robust Reinforcement Learning-based Control for UAVs under Physical Attacks**

cs.LG

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.22423v1) [paper-pdf](http://arxiv.org/pdf/2506.22423v1)

**Authors**: Pritam Dash, Ethan Chan, Nathan P. Lawrence, Karthik Pattabiraman

**Abstract**: Unmanned Aerial Vehicles (UAVs) depend on onboard sensors for perception, navigation, and control. However, these sensors are susceptible to physical attacks, such as GPS spoofing, that can corrupt state estimates and lead to unsafe behavior. While reinforcement learning (RL) offers adaptive control capabilities, existing safe RL methods are ineffective against such attacks. We present ARMOR (Adaptive Robust Manipulation-Optimized State Representations), an attack-resilient, model-free RL controller that enables robust UAV operation under adversarial sensor manipulation. Instead of relying on raw sensor observations, ARMOR learns a robust latent representation of the UAV's physical state via a two-stage training framework. In the first stage, a teacher encoder, trained with privileged attack information, generates attack-aware latent states for RL policy training. In the second stage, a student encoder is trained via supervised learning to approximate the teacher's latent states using only historical sensor data, enabling real-world deployment without privileged information. Our experiments show that ARMOR outperforms conventional methods, ensuring UAV safety. Additionally, ARMOR improves generalization to unseen attacks and reduces training cost by eliminating the need for iterative adversarial training.



## **49. Secure Video Quality Assessment Resisting Adversarial Attacks**

cs.CV

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2410.06866v2) [paper-pdf](http://arxiv.org/pdf/2410.06866v2)

**Authors**: Ao-Xiang Zhang, Yuan-Gen Wang, Yu Ran, Weixuan Tang, Qingxiao Guan, Chunsheng Yang

**Abstract**: The exponential surge in video traffic has intensified the imperative for Video Quality Assessment (VQA). Leveraging cutting-edge architectures, current VQA models have achieved human-comparable accuracy. However, recent studies have revealed the vulnerability of existing VQA models against adversarial attacks. To establish a reliable and practical assessment system, a secure VQA model capable of resisting such malicious attacks is urgently demanded. Unfortunately, no attempt has been made to explore this issue. This paper first attempts to investigate general adversarial defense principles, aiming at endowing existing VQA models with security. Specifically, we first introduce random spatial grid sampling on the video frame for intra-frame defense. Then, we design pixel-wise randomization through a guardian map, globally neutralizing adversarial perturbations. Meanwhile, we extract temporal information from the video sequence as compensation for inter-frame defense. Building upon these principles, we present a novel VQA framework from the security-oriented perspective, termed SecureVQA. Extensive experiments indicate that SecureVQA sets a new benchmark in security while achieving competitive VQA performance compared with state-of-the-art models. Ablation studies delve deeper into analyzing the principles of SecureVQA, demonstrating their generalization and contributions to the security of leading VQA models.



## **50. A Self-scaled Approximate $\ell_0$ Regularization Robust Model for Outlier Detection**

eess.SP

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.22277v1) [paper-pdf](http://arxiv.org/pdf/2506.22277v1)

**Authors**: Pengyang Song, Jue Wang

**Abstract**: Robust regression models in the presence of outliers have significant practical relevance in areas such as signal processing, financial econometrics, and energy management. Many existing robust regression methods, either grounded in statistical theory or sparse signal recovery, typically rely on the explicit or implicit assumption of outlier sparsity to filter anomalies and recover the underlying signal or data. However, these methods often suffer from limited robustness or high computational complexity, rendering them inefficient for large-scale problems. In this work, we propose a novel robust regression model based on a Self-scaled Approximate l0 Regularization Model (SARM) scheme. By introducing a self-scaling mechanism into the regularization term, the proposed model mitigates the negative impact of uneven or excessively large outlier magnitudes on robustness. We also develop an alternating minimization algorithm grounded in Proximal Operators and Block Coordinate Descent. We rigorously prove the algorithm convergence. Empirical comparisons with several state-of-the-art robust regression methods demonstrate that SARM not only achieves superior robustness but also significantly improves computational efficiency. Motivated by both the theoretical error bound and empirical observations, we further design a Two-Stage SARM (TSSARM) framework, which better utilizes sample information when the singular values of the design matrix are widely spread, thereby enhancing robustness under certain conditions. Finally, we validate our approach on a real-world load forecasting task. The experimental results show that our method substantially enhances the robustness of load forecasting against adversarial data attacks, which is increasingly critical in the era of heightened data security concerns.



