# Latest Adversarial Attack Papers
**update at 2025-07-10 15:59:23**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Protecting Classifiers From Attacks**

stat.ML

Published in Statistical Science:  https://projecteuclid.org/journals/statistical-science/volume-39/issue-3/Protecting-Classifiers-from-Attacks/10.1214/24-STS922.full

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2004.08705v2) [paper-pdf](http://arxiv.org/pdf/2004.08705v2)

**Authors**: Victor Gallego, Roi Naveiro, Alberto Redondo, David Rios Insua, Fabrizio Ruggeri

**Abstract**: In multiple domains such as malware detection, automated driving systems, or fraud detection, classification algorithms are susceptible to being attacked by malicious agents willing to perturb the value of instance covariates to pursue certain goals. Such problems pertain to the field of adversarial machine learning and have been mainly dealt with, perhaps implicitly, through game-theoretic ideas with strong underlying common knowledge assumptions. These are not realistic in numerous application domains in relation to security and business competition. We present an alternative Bayesian decision theoretic framework that accounts for the uncertainty about the attacker's behavior using adversarial risk analysis concepts. In doing so, we also present core ideas in adversarial machine learning to a statistical audience. A key ingredient in our framework is the ability to sample from the distribution of originating instances given the, possibly attacked, observed ones. We propose an initial procedure based on approximate Bayesian computation usable during operations; within it, we simulate the attacker's problem taking into account our uncertainty about his elements. Large-scale problems require an alternative scalable approach implementable during the training stage. Globally, we are able to robustify statistical classification algorithms against malicious attacks.



## **2. Robust and Safe Traffic Sign Recognition using N-version with Weighted Voting**

cs.LG

27 pages including appendix, 1 figure

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06907v1) [paper-pdf](http://arxiv.org/pdf/2507.06907v1)

**Authors**: Linyun Gao, Qiang Wen, Fumio Machida

**Abstract**: Autonomous driving is rapidly advancing as a key application of machine learning, yet ensuring the safety of these systems remains a critical challenge. Traffic sign recognition, an essential component of autonomous vehicles, is particularly vulnerable to adversarial attacks that can compromise driving safety. In this paper, we propose an N-version machine learning (NVML) framework that integrates a safety-aware weighted soft voting mechanism. Our approach utilizes Failure Mode and Effects Analysis (FMEA) to assess potential safety risks and assign dynamic, safety-aware weights to the ensemble outputs. We evaluate the robustness of three-version NVML systems employing various voting mechanisms against adversarial samples generated using the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks. Experimental results demonstrate that our NVML approach significantly enhances the robustness and safety of traffic sign recognition systems under adversarial conditions.



## **3. A Single-Point Measurement Framework for Robust Cyber-Attack Diagnosis in Smart Microgrids Using Dual Fractional-Order Feature Analysis**

eess.SY

8 pages, 10 figures

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06890v1) [paper-pdf](http://arxiv.org/pdf/2507.06890v1)

**Authors**: Yifan Wang

**Abstract**: Cyber-attacks jeopardize the safe operation of smart microgrids. At the same time, existing diagnostic methods either depend on expensive multi-point instrumentation or stringent modelling assumptions that are untenable under single-sensor constraints. This paper proposes a Fractional-Order Memory-Enhanced Attack-Diagnosis Scheme (FO-MADS) that achieves low-latency fault localisation and cyber-attack detection using only one VPQ (Voltage-Power-Reactive-power) sensor. FO-MADS first constructs a dual fractional-order feature library by jointly applying Caputo and Gr\"unwald-Letnikov derivatives, thereby amplifying micro-perturbations and slow drifts in the VPQ signal. A two-stage hierarchical classifier then pinpoints the affected inverter and isolates the faulty IGBT switch, effectively alleviating class imbalance. Robustness is further strengthened through Progressive Memory-Replay Adversarial Training (PMR-AT), whose attack-aware loss is dynamically re-weighted via Online Hard Example Mining (OHEM) to prioritise the most challenging samples. Experiments on a four-inverter microgrid testbed comprising 1 normal and 24 fault classes under four attack scenarios demonstrate diagnostic accuracies of 96.6 % (bias), 94.0 % (noise), 92.8 % (data replacement), and 95.7 % (replay), while sustaining 96.7 % under attack-free conditions. These results establish FO-MADS as a cost-effective and readily deployable solution that markedly enhances the cyber-physical resilience of smart microgrids.



## **4. IAP: Invisible Adversarial Patch Attack through Perceptibility-Aware Localization and Perturbation Optimization**

cs.CV

Published in ICCV 2025

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06856v1) [paper-pdf](http://arxiv.org/pdf/2507.06856v1)

**Authors**: Subrat Kishore Dutta, Xiao Zhang

**Abstract**: Despite modifying only a small localized input region, adversarial patches can drastically change the prediction of computer vision models. However, prior methods either cannot perform satisfactorily under targeted attack scenarios or fail to produce contextually coherent adversarial patches, causing them to be easily noticeable by human examiners and insufficiently stealthy against automatic patch defenses. In this paper, we introduce IAP, a novel attack framework that generates highly invisible adversarial patches based on perceptibility-aware localization and perturbation optimization schemes. Specifically, IAP first searches for a proper location to place the patch by leveraging classwise localization and sensitivity maps, balancing the susceptibility of patch location to both victim model prediction and human visual system, then employs a perceptibility-regularized adversarial loss and a gradient update rule that prioritizes color constancy for optimizing invisible perturbations. Comprehensive experiments across various image benchmarks and model architectures demonstrate that IAP consistently achieves competitive attack success rates in targeted settings with significantly improved patch invisibility compared to existing baselines. In addition to being highly imperceptible to humans, IAP is shown to be stealthy enough to render several state-of-the-art patch defenses ineffective.



## **5. The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover**

cs.CR

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06850v1) [paper-pdf](http://arxiv.org/pdf/2507.06850v1)

**Authors**: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro

**Abstract**: The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.



## **6. PBCAT: Patch-based composite adversarial training against physically realizable attacks on object detection**

cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2506.23581v2) [paper-pdf](http://arxiv.org/pdf/2506.23581v2)

**Authors**: Xiao Li, Yiming Zhu, Yifan Huang, Wei Zhang, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Object detection plays a crucial role in many security-sensitive applications. However, several recent studies have shown that object detectors can be easily fooled by physically realizable attacks, \eg, adversarial patches and recent adversarial textures, which pose realistic and urgent threats. Adversarial Training (AT) has been recognized as the most effective defense against adversarial attacks. While AT has been extensively studied in the $l_\infty$ attack settings on classification models, AT against physically realizable attacks on object detectors has received limited exploration. Early attempts are only performed to defend against adversarial patches, leaving AT against a wider range of physically realizable attacks under-explored. In this work, we consider defending against various physically realizable attacks with a unified AT method. We propose PBCAT, a novel Patch-Based Composite Adversarial Training strategy. PBCAT optimizes the model by incorporating the combination of small-area gradient-guided adversarial patches and imperceptible global adversarial perturbations covering the entire image. With these designs, PBCAT has the potential to defend against not only adversarial patches but also unseen physically realizable attacks such as adversarial textures. Extensive experiments in multiple settings demonstrated that PBCAT significantly improved robustness against various physically realizable attacks over state-of-the-art defense methods. Notably, it improved the detection accuracy by 29.7\% over previous defense methods under one recent adversarial texture attack.



## **7. Tail-aware Adversarial Attacks: A Distributional Approach to Efficient LLM Jailbreaking**

cs.LG

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.04446v2) [paper-pdf](http://arxiv.org/pdf/2507.04446v2)

**Authors**: Tim Beyer, Yan Scholten, Leo Schwinn, Stephan Günnemann

**Abstract**: To guarantee safe and robust deployment of large language models (LLMs) at scale, it is critical to accurately assess their adversarial robustness. Existing adversarial attacks typically target harmful responses in single-point, greedy generations, overlooking the inherently stochastic nature of LLMs. In this paper, we propose a novel framework for adversarial robustness evaluation that explicitly models the entire output distribution, including tail-risks, providing better estimates for model robustness at scale. By casting the attack process as a resource allocation problem between optimization and sampling, we determine compute-optimal tradeoffs and show that integrating sampling into existing attacks boosts ASR by up to 48% and improves efficiency by up to two orders of magnitude. Our framework also enables us to analyze how different attack algorithms affect output harm distributions. Surprisingly, we find that most optimization strategies have little effect on output harmfulness. Finally, we introduce a data-free proof-of-concept objective based on entropy-maximization to demonstrate how our tail-aware perspective enables new optimization targets. Overall, our findings highlight the importance of tail-aware attacks and evaluation protocols to accurately assess and strengthen LLM safety.



## **8. Distributed Fault-Tolerant Multi-Robot Cooperative Localization in Adversarial Environments**

cs.RO

Accepted to IROS 2025 Conference

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06750v1) [paper-pdf](http://arxiv.org/pdf/2507.06750v1)

**Authors**: Tohid Kargar Tasooji, Ramviyas Parasuraman

**Abstract**: In multi-robot systems (MRS), cooperative localization is a crucial task for enhancing system robustness and scalability, especially in GPS-denied or communication-limited environments. However, adversarial attacks, such as sensor manipulation, and communication jamming, pose significant challenges to the performance of traditional localization methods. In this paper, we propose a novel distributed fault-tolerant cooperative localization framework to enhance resilience against sensor and communication disruptions in adversarial environments. We introduce an adaptive event-triggered communication strategy that dynamically adjusts communication thresholds based on real-time sensing and communication quality. This strategy ensures optimal performance even in the presence of sensor degradation or communication failure. Furthermore, we conduct a rigorous analysis of the convergence and stability properties of the proposed algorithm, demonstrating its resilience against bounded adversarial zones and maintaining accurate state estimation. Robotarium-based experiment results show that our proposed algorithm significantly outperforms traditional methods in terms of localization accuracy and communication efficiency, particularly in adversarial settings. Our approach offers improved scalability, reliability, and fault tolerance for MRS, making it suitable for large-scale deployments in real-world, challenging environments.



## **9. Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment**

cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2408.06079v2) [paper-pdf](http://arxiv.org/pdf/2408.06079v2)

**Authors**: Kejia Zhang, Juanjuan Weng, Shaozi Li, Zhiming Luo

**Abstract**: Despite the remarkable progress of deep neural networks (DNNs) in various visual tasks, their vulnerability to adversarial examples raises significant security concerns. Recent adversarial training methods leverage inverse adversarial attacks to generate high-confidence examples, aiming to align adversarial distributions with high-confidence class regions. However, our investigation reveals that under inverse adversarial attacks, high-confidence outputs are influenced by biased feature activations, causing models to rely on background features that lack a causal relationship with the labels. This spurious correlation bias leads to overfitting irrelevant background features during adversarial training, thereby degrading the model's robust performance and generalization capabilities. To address this issue, we propose Debiased High-Confidence Adversarial Training (DHAT), a novel approach that aligns adversarial logits with debiased high-confidence logits and restores proper attention by enhancing foreground logit orthogonality. Extensive experiments demonstrate that DHAT achieves state-of-the-art robustness on both CIFAR and ImageNet-1K benchmarks, while significantly improving generalization by mitigating the feature bias inherent in inverse adversarial training approaches. Code is available at https://github.com/KejiaZhang-Robust/DHAT.



## **10. Evaluating and Improving Robustness in Large Language Models: A Survey and Future Directions**

cs.CL

33 pages, 5 figures

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2506.11111v2) [paper-pdf](http://arxiv.org/pdf/2506.11111v2)

**Authors**: Kun Zhang, Le Wu, Kui Yu, Guangyi Lv, Dacao Zhang

**Abstract**: Large Language Models (LLMs) have gained enormous attention in recent years due to their capability of understanding and generating natural languages. With the rapid development and wild-range applications (e.g., Agents, Embodied Intelligence), the robustness of LLMs has received increased attention. As the core brain of many AI applications, the robustness of LLMs requires that models should not only generate consistent contents, but also ensure the correctness and stability of generated content when dealing with unexpeted application scenarios (e.g., toxic prompts, limited noise domain data, outof-distribution (OOD) applications, etc). In this survey paper, we conduct a thorough review of the robustness of LLMs, aiming to provide a comprehensive terminology of concepts and methods around this field and facilitate the community. Specifically, we first give a formal definition of LLM robustness and present the collection protocol of this survey paper. Then, based on the types of perturbated inputs, we organize this survey from the following perspectives: 1) Adversarial Robustness: tackling the problem that prompts are manipulated intentionally, such as noise prompts, long context, data attack, etc; 2) OOD Robustness: dealing with the unexpected real-world application scenarios, such as OOD detection, zero-shot transferring, hallucinations, etc; 3) Evaluation of Robustness: summarizing the new evaluation datasets, metrics, and tools for verifying the robustness of LLMs. After reviewing the representative work from each perspective, we discuss and highlight future opportunities and research directions in this field. Meanwhile, we also organize related works and provide an easy-to-search project (https://github.com/zhangkunzk/Awesome-LLM-Robustness-papers) to support the community.



## **11. Can adversarial attacks by large language models be attributed?**

cs.AI

21 pages, 5 figures, 2 tables

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2411.08003v2) [paper-pdf](http://arxiv.org/pdf/2411.08003v2)

**Authors**: Manuel Cebrian, Andres Abeliuk, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation campaigns-presents significant challenges that are likely to grow in importance. We approach this attribution problem from both a theoretical and an empirical perspective, drawing on formal language theory (identification in the limit) and data-driven analysis of the expanding LLM ecosystem. By modeling an LLM's set of possible outputs as a formal language, we analyze whether finite samples of text can uniquely pinpoint the originating model. Our results show that, under mild assumptions of overlapping capabilities among models, certain classes of LLMs are fundamentally non-identifiable from their outputs alone. We delineate four regimes of theoretical identifiability: (1) an infinite class of deterministic (discrete) LLM languages is not identifiable (Gold's classical result from 1967); (2) an infinite class of probabilistic LLMs is also not identifiable (by extension of the deterministic case); (3) a finite class of deterministic LLMs is identifiable (consistent with Angluin's tell-tale criterion); and (4) even a finite class of probabilistic LLMs can be non-identifiable (we provide a new counterexample establishing this negative result). Complementing these theoretical insights, we quantify the explosion in the number of plausible model origins (hypothesis space) for a given output in recent years. Even under conservative assumptions-each open-source model fine-tuned on at most one new dataset-the count of distinct candidate models doubles approximately every 0.5 years, and allowing multi-dataset fine-tuning combinations yields doubling times as short as 0.28 years. This combinatorial growth, alongside the extraordinary computational cost of brute-force likelihood attribution across all models and potential users, renders exhaustive attribution infeasible in practice.



## **12. Dual State-space Fidelity Blade (D-STAB): A Novel Stealthy Cyber-physical Attack Paradigm**

eess.SY

accepted by 2025 American Control Conference

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06492v1) [paper-pdf](http://arxiv.org/pdf/2507.06492v1)

**Authors**: Jiajun Shen, Hao Tu, Fengjun Li, Morteza Hashemi, Di Wu, Huazhen Fang

**Abstract**: This paper presents a novel cyber-physical attack paradigm, termed the Dual State-Space Fidelity Blade (D-STAB), which targets the firmware of core cyber-physical components as a new class of attack surfaces. The D-STAB attack exploits the information asymmetry caused by the fidelity gap between high-fidelity and low-fidelity physical models in cyber-physical systems. By designing precise adversarial constraints based on high-fidelity state-space information, the attack induces deviations in high-fidelity states that remain undetected by defenders relying on low-fidelity observations. The effectiveness of D-STAB is demonstrated through a case study in cyber-physical battery systems, specifically in an optimal charging task governed by a Battery Management System (BMS).



## **13. On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks**

cs.CL

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06489v1) [paper-pdf](http://arxiv.org/pdf/2507.06489v1)

**Authors**: Stephen Obadinma, Xiaodan Zhu

**Abstract**: Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to ensure transparency, trust, and safety in human-AI interactions across many high-stakes applications. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce a novel framework for attacking verbal confidence scores through both perturbation and jailbreak-based methods, and show that these attacks can significantly jeopardize verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current confidence elicitation methods are vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the urgent need to design more robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.



## **14. Real AI Agents with Fake Memories: Fatal Context Manipulation Attacks on Web3 Agents**

cs.CR

19 pages, 14 figures

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2503.16248v3) [paper-pdf](http://arxiv.org/pdf/2503.16248v3)

**Authors**: Atharv Singh Patlan, Peiyao Sheng, S. Ashwin Hebbar, Prateek Mittal, Pramod Viswanath

**Abstract**: AI agents integrated with Web3 offer autonomy and openness but raise security concerns as they interact with financial protocols and immutable smart contracts. This paper investigates the vulnerabilities of AI agents within blockchain-based financial ecosystems when exposed to adversarial threats in real-world scenarios. We introduce the concept of context manipulation -- a comprehensive attack vector that exploits unprotected context surfaces, including input channels, memory modules, and external data feeds. It expands on traditional prompt injection and reveals a more stealthy and persistent threat: memory injection. Using ElizaOS, a representative decentralized AI agent framework for automated Web3 operations, we showcase that malicious injections into prompts or historical records can trigger unauthorized asset transfers and protocol violations which could be financially devastating in reality. To quantify these risks, we introduce CrAIBench, a Web3-focused benchmark covering 150+ realistic blockchain tasks. such as token transfers, trading, bridges, and cross-chain interactions, and 500+ attack test cases using context manipulation. Our evaluation results confirm that AI models are significantly more vulnerable to memory injection compared to prompt injection. Finally, we evaluate a comprehensive defense roadmap, finding that prompt-injection defenses and detectors only provide limited protection when stored context is corrupted, whereas fine-tuning-based defenses substantially reduce attack success rates while preserving performance on single-step tasks. These results underscore the urgent need for AI agents that are both secure and fiduciarily responsible in blockchain environments.



## **15. Single Word Change is All You Need: Designing Attacks and Defenses for Text Classifiers**

cs.CL

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2401.17196v2) [paper-pdf](http://arxiv.org/pdf/2401.17196v2)

**Authors**: Lei Xu, Sarah Alnegheimish, Laure Berti-Equille, Alfredo Cuesta-Infante, Kalyan Veeramachaneni

**Abstract**: In text classification, creating an adversarial example means subtly perturbing a few words in a sentence without changing its meaning, causing it to be misclassified by a classifier. A concerning observation is that a significant portion of adversarial examples generated by existing methods change only one word. This single-word perturbation vulnerability represents a significant weakness in classifiers, which malicious users can exploit to efficiently create a multitude of adversarial examples. This paper studies this problem and makes the following key contributions: (1) We introduce a novel metric \r{ho} to quantitatively assess a classifier's robustness against single-word perturbation. (2) We present the SP-Attack, designed to exploit the single-word perturbation vulnerability, achieving a higher attack success rate, better preserving sentence meaning, while reducing computation costs compared to state-of-the-art adversarial methods. (3) We propose SP-Defense, which aims to improve \r{ho} by applying data augmentation in learning. Experimental results on 4 datasets and BERT and distilBERT classifiers show that SP-Defense improves \r{ho} by 14.6% and 13.9% and decreases the attack success rate of SP-Attack by 30.4% and 21.2% on two classifiers respectively, and decreases the attack success rate of existing attack methods that involve multiple-word perturbations.



## **16. On the Natural Robustness of Vision-Language Models Against Visual Perception Attacks in Autonomous Driving**

cs.CV

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2506.11472v2) [paper-pdf](http://arxiv.org/pdf/2506.11472v2)

**Authors**: Pedram MohajerAnsari, Amir Salarpour, Michael Kühr, Siyu Huang, Mohammad Hamad, Sebastian Steinhorst, Habeeb Olufowobi, Mert D. Pesé

**Abstract**: Autonomous vehicles (AVs) rely on deep neural networks (DNNs) for critical tasks such as traffic sign recognition (TSR), automated lane centering (ALC), and vehicle detection (VD). However, these models are vulnerable to attacks that can cause misclassifications and compromise safety. Traditional defense mechanisms, including adversarial training, often degrade benign accuracy and fail to generalize against unseen attacks. In this work, we introduce Vehicle Vision Language Models (V2LMs), fine-tuned vision-language models specialized for AV perception. Our findings demonstrate that V2LMs inherently exhibit superior robustness against unseen attacks without requiring adversarial training, maintaining significantly higher accuracy than conventional DNNs under adversarial conditions. We evaluate two deployment strategies: Solo Mode, where individual V2LMs handle specific perception tasks, and Tandem Mode, where a single unified V2LM is fine-tuned for multiple tasks simultaneously. Experimental results reveal that DNNs suffer performance drops of 33% to 46% under attacks, whereas V2LMs maintain adversarial accuracy with reductions of less than 8% on average. The Tandem Mode further offers a memory-efficient alternative while achieving comparable robustness to Solo Mode. We also explore integrating V2LMs as parallel components to AV perception to enhance resilience against adversarial threats. Our results suggest that V2LMs offer a promising path toward more secure and resilient AV perception systems.



## **17. Hedge Funds on a Swamp: Analyzing Patterns, Vulnerabilities, and Defense Measures in Blockchain Bridges [Experiment, Analysis \& Benchmark]**

cs.ET

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06156v1) [paper-pdf](http://arxiv.org/pdf/2507.06156v1)

**Authors**: Poupak Azad, Jiahua Xu, Yebo Feng, Preston Strowbridge, Cuneyt Akcora

**Abstract**: Blockchain bridges have become essential infrastructure for enabling interoperability across different blockchain networks, with more than $24B monthly bridge transaction volume. However, their growing adoption has been accompanied by a disproportionate rise in security breaches, making them the single largest source of financial loss in Web3. For cross-chain ecosystems to be robust and sustainable, it is essential to understand and address these vulnerabilities. In this study, we present a comprehensive systematization of blockchain bridge design and security. We define three bridge security priors, formalize the architectural structure of 13 prominent bridges, and identify 23 attack vectors grounded in real-world blockchain exploits. Using this foundation, we evaluate 43 representative attack scenarios and introduce a layered threat model that captures security failures across source chain, off-chain, and destination chain components.   Our analysis at the static code and transaction network levels reveals recurring design flaws, particularly in access control, validator trust assumptions, and verification logic, and identifies key patterns in adversarial behavior based on transaction-level traces. To support future development, we propose a decision framework for bridge architecture design, along with defense mechanisms such as layered validation and circuit breakers. This work provides a data-driven foundation for evaluating bridge security and lays the groundwork for standardizing resilient cross-chain infrastructure.



## **18. The bitter lesson of misuse detection**

cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06282v1) [paper-pdf](http://arxiv.org/pdf/2507.06282v1)

**Authors**: Hadrien Mariaccia, Charbel-Raphaël Segerie, Diego Dorn

**Abstract**: Prior work on jailbreak detection has established the importance of adversarial robustness for LLMs but has largely focused on the model ability to resist adversarial inputs and to output safe content, rather than the effectiveness of external supervision systems. The only public and independent benchmark of these guardrails to date evaluates a narrow set of supervisors on limited scenarios. Consequently, no comprehensive public benchmark yet verifies how well supervision systems from the market perform under realistic, diverse attacks. To address this, we introduce BELLS, a Benchmark for the Evaluation of LLM Supervision Systems. The framework is two dimensional: harm severity (benign, borderline, harmful) and adversarial sophistication (direct vs. jailbreak) and provides a rich dataset covering 3 jailbreak families and 11 harm categories. Our evaluations reveal drastic limitations of specialized supervision systems. While they recognize some known jailbreak patterns, their semantic understanding and generalization capabilities are very limited, sometimes with detection rates close to zero when asking a harmful question directly or with a new jailbreak technique such as base64 encoding. Simply asking generalist LLMs if the user question is "harmful or not" largely outperforms these supervisors from the market according to our BELLS score. But frontier LLMs still suffer from metacognitive incoherence, often responding to queries they correctly identify as harmful (up to 30 percent for Claude 3.7 and greater than 50 percent for Mistral Large). These results suggest that simple scaffolding could significantly improve misuse detection robustness, but more research is needed to assess the tradeoffs of such techniques. Our results support the "bitter lesson" of misuse detection: general capabilities of LLMs are necessary to detect a diverse array of misuses and jailbreaks.



## **19. ScoreAdv: Score-based Targeted Generation of Natural Adversarial Examples via Diffusion Models**

cs.CV

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06078v1) [paper-pdf](http://arxiv.org/pdf/2507.06078v1)

**Authors**: Chihan Huang, Hao Tang

**Abstract**: Despite the success of deep learning across various domains, it remains vulnerable to adversarial attacks. Although many existing adversarial attack methods achieve high success rates, they typically rely on $\ell_{p}$-norm perturbation constraints, which do not align with human perceptual capabilities. Consequently, researchers have shifted their focus toward generating natural, unrestricted adversarial examples (UAEs). GAN-based approaches suffer from inherent limitations, such as poor image quality due to instability and mode collapse. Meanwhile, diffusion models have been employed for UAE generation, but they still rely on iterative PGD perturbation injection, without fully leveraging their central denoising capabilities. In this paper, we introduce a novel approach for generating UAEs based on diffusion models, named ScoreAdv. This method incorporates an interpretable adversarial guidance mechanism to gradually shift the sampling distribution towards the adversarial distribution, while using an interpretable saliency map to inject the visual information of a reference image into the generated samples. Notably, our method is capable of generating an unlimited number of natural adversarial examples and can attack not only classification models but also retrieval models. We conduct extensive experiments on ImageNet and CelebA datasets, validating the performance of ScoreAdv across ten target models in both black-box and white-box settings. Our results demonstrate that ScoreAdv achieves state-of-the-art attack success rates and image quality. Furthermore, the dynamic balance between denoising and adversarial perturbation enables ScoreAdv to remain robust even under defensive measures.



## **20. CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks on their Internal Representations**

cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06043v1) [paper-pdf](http://arxiv.org/pdf/2507.06043v1)

**Authors**: Xiaohu Li, Yunfeng Ning, Zepeng Bao, Mayi Xu, Jianhao Chen, Tieyun Qian

**Abstract**: Security alignment enables the Large Language Model (LLM) to gain the protection against malicious queries, but various jailbreak attack methods reveal the vulnerability of this security mechanism. Previous studies have isolated LLM jailbreak attacks and defenses. We analyze the security protection mechanism of the LLM, and propose a framework that combines attack and defense. Our method is based on the linearly separable property of LLM intermediate layer embedding, as well as the essence of jailbreak attack, which aims to embed harmful problems and transfer them to the safe area. We utilize generative adversarial network (GAN) to learn the security judgment boundary inside the LLM to achieve efficient jailbreak attack and defense. The experimental results indicate that our method achieves an average jailbreak success rate of 88.85\% across three popular LLMs, while the defense success rate on the state-of-the-art jailbreak dataset reaches an average of 84.17\%. This not only validates the effectiveness of our approach but also sheds light on the internal security mechanisms of LLMs, offering new insights for enhancing model security The code and data are available at https://github.com/NLPGM/CAVGAN.



## **21. TuneShield: Mitigating Toxicity in Conversational AI while Fine-tuning on Untrusted Data**

cs.CR

Pre-print

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.05660v1) [paper-pdf](http://arxiv.org/pdf/2507.05660v1)

**Authors**: Aravind Cheruvu, Shravya Kanchi, Sifat Muhammad Abdullah, Nicholas Kong, Daphne Yao, Murtuza Jadliwala, Bimal Viswanath

**Abstract**: Recent advances in foundation models, such as LLMs, have revolutionized conversational AI. Chatbots are increasingly being developed by customizing LLMs on specific conversational datasets. However, mitigating toxicity during this customization, especially when dealing with untrusted training data, remains a significant challenge. To address this, we introduce TuneShield, a defense framework designed to mitigate toxicity during chatbot fine-tuning while preserving conversational quality. TuneShield leverages LLM-based toxicity classification, utilizing the instruction-following capabilities and safety alignment of LLMs to effectively identify toxic samples, outperforming industry API services. TuneShield generates synthetic conversation samples, termed 'healing data', based on the identified toxic samples, using them to mitigate toxicity while reinforcing desirable behavior during fine-tuning. It performs an alignment process to further nudge the chatbot towards producing desired responses. Our findings show that TuneShield effectively mitigates toxicity injection attacks while preserving conversational quality, even when the toxicity classifiers are imperfect or biased. TuneShield proves to be resilient against adaptive adversarial and jailbreak attacks. Additionally, TuneShield demonstrates effectiveness in mitigating adaptive toxicity injection attacks during dialog-based learning (DBL).



## **22. MEF: A Capability-Aware Multi-Encryption Framework for Evaluating Vulnerabilities in Black-Box Large Language Models**

cs.CL

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2505.23404v3) [paper-pdf](http://arxiv.org/pdf/2505.23404v3)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin, Fei Gao, Wenmin Li

**Abstract**: Recent advancements in adversarial jailbreak attacks have revealed significant vulnerabilities in Large Language Models (LLMs), facilitating the evasion of alignment safeguards through increasingly sophisticated prompt manipulations. In this paper, we propose MEF, a capability-aware multi-encryption framework for evaluating vulnerabilities in black-box LLMs. Our key insight is that the effectiveness of jailbreak strategies can be significantly enhanced by tailoring them to the semantic comprehension capabilities of the target model. We present a typology that classifies LLMs into Type I and Type II based on their comprehension levels, and design adaptive attack strategies for each. MEF combines layered semantic mutations and dual-ended encryption techniques, enabling circumvention of input, inference, and output-level defenses. Experimental results demonstrate the superiority of our approach. Remarkably, it achieves a jailbreak success rate of 98.9\% on GPT-4o (29 May 2025 release). Our findings reveal vulnerabilities in current LLMs' alignment defenses.



## **23. How Not to Detect Prompt Injections with an LLM**

cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.05630v1) [paper-pdf](http://arxiv.org/pdf/2507.05630v1)

**Authors**: Sarthak Choudhary, Divyam Anshumaan, Nils Palumbo, Somesh Jha

**Abstract**: LLM-integrated applications and agents are vulnerable to prompt injection attacks, in which adversaries embed malicious instructions within seemingly benign user inputs to manipulate the LLM's intended behavior. Recent defenses based on $\textit{known-answer detection}$ (KAD) have achieved near-perfect performance by using an LLM to classify inputs as clean or contaminated. In this work, we formally characterize the KAD framework and uncover a structural vulnerability in its design that invalidates its core security premise. We design a methodical adaptive attack, $\textit{DataFlip}$, to exploit this fundamental weakness. It consistently evades KAD defenses with detection rates as low as $1.5\%$ while reliably inducing malicious behavior with success rates of up to $88\%$, without needing white-box access to the LLM or any optimization procedures.



## **24. One Surrogate to Fool Them All: Universal, Transferable, and Targeted Adversarial Attacks with CLIP**

cs.CR

21 pages, 15 figures, 18 tables. To appear in the Proceedings of The  ACM Conference on Computer and Communications Security (CCS), 2025

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2505.19840v2) [paper-pdf](http://arxiv.org/pdf/2505.19840v2)

**Authors**: Binyan Xu, Xilin Dai, Di Tang, Kehuan Zhang

**Abstract**: Deep Neural Networks (DNNs) have achieved widespread success yet remain prone to adversarial attacks. Typically, such attacks either involve frequent queries to the target model or rely on surrogate models closely mirroring the target model -- often trained with subsets of the target model's training data -- to achieve high attack success rates through transferability. However, in realistic scenarios where training data is inaccessible and excessive queries can raise alarms, crafting adversarial examples becomes more challenging. In this paper, we present UnivIntruder, a novel attack framework that relies solely on a single, publicly available CLIP model and publicly available datasets. By using textual concepts, UnivIntruder generates universal, transferable, and targeted adversarial perturbations that mislead DNNs into misclassifying inputs into adversary-specified classes defined by textual concepts.   Our extensive experiments show that our approach achieves an Attack Success Rate (ASR) of up to 85% on ImageNet and over 99% on CIFAR-10, significantly outperforming existing transfer-based methods. Additionally, we reveal real-world vulnerabilities, showing that even without querying target models, UnivIntruder compromises image search engines like Google and Baidu with ASR rates up to 84%, and vision language models like GPT-4 and Claude-3.5 with ASR rates up to 80%. These findings underscore the practicality of our attack in scenarios where traditional avenues are blocked, highlighting the need to reevaluate security paradigms in AI applications.



## **25. DATABench: Evaluating Dataset Auditing in Deep Learning from an Adversarial Perspective**

cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.05622v1) [paper-pdf](http://arxiv.org/pdf/2507.05622v1)

**Authors**: Shuo Shao, Yiming Li, Mengren Zheng, Zhiyang Hu, Yukun Chen, Boheng Li, Yu He, Junfeng Guo, Tianwei Zhang, Dacheng Tao, Zhan Qin

**Abstract**: The widespread application of Deep Learning across diverse domains hinges critically on the quality and composition of training datasets. However, the common lack of disclosure regarding their usage raises significant privacy and copyright concerns. Dataset auditing techniques, which aim to determine if a specific dataset was used to train a given suspicious model, provide promising solutions to addressing these transparency gaps. While prior work has developed various auditing methods, their resilience against dedicated adversarial attacks remains largely unexplored. To bridge the gap, this paper initiates a comprehensive study evaluating dataset auditing from an adversarial perspective. We start with introducing a novel taxonomy, classifying existing methods based on their reliance on internal features (IF) (inherent to the data) versus external features (EF) (artificially introduced for auditing). Subsequently, we formulate two primary attack types: evasion attacks, designed to conceal the use of a dataset, and forgery attacks, intending to falsely implicate an unused dataset. Building on the understanding of existing methods and attack objectives, we further propose systematic attack strategies: decoupling, removal, and detection for evasion; adversarial example-based methods for forgery. These formulations and strategies lead to our new benchmark, DATABench, comprising 17 evasion attacks, 5 forgery attacks, and 9 representative auditing methods. Extensive evaluations using DATABench reveal that none of the evaluated auditing methods are sufficiently robust or distinctive under adversarial settings. These findings underscore the urgent need for developing a more secure and reliable dataset auditing method capable of withstanding sophisticated adversarial manipulation. Code is available at https://github.com/shaoshuo-ss/DATABench.



## **26. Massive MIMO-NOMA Systems Secrecy in the Presence of Active Eavesdroppers**

cs.IT

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2105.02215v2) [paper-pdf](http://arxiv.org/pdf/2105.02215v2)

**Authors**: Marziyeh Soltani, Mahtab Mirmohseni, Panos Papadimitratos

**Abstract**: Non-orthogonal multiple access (NOMA) and massive multiple-input multiple-output (MIMO) systems are highly efficient. Massive MIMO systems are inherently resistant to passive attackers (eavesdroppers), thanks to transmissions directed to the desired users. However, active attackers can transmit a combination of legitimate user pilot signals during the channel estimation phase. This way they can mislead the base station (BS) to rotate the transmission in their direction, and allow them to eavesdrop during the downlink data transmission phase. In this paper, we analyse this vulnerability in an improved system model and stronger adversary assumptions, and investigate how physical layer security can mitigate such attacks and ensure secure (confidential) communication. We derive the secrecy outage probability (SOP) and a lower bound on the ergodic secrecy capacity, using stochastic geometry tools when the number of antennas in the BSs tends to infinity. We adapt the result to evaluate the secrecy performance in massive orthogonal multiple access (OMA). We find that appropriate power allocation allows NOMA to outperform OMA in terms of ergodic secrecy rate and SOP.



## **27. A Systematization of Security Vulnerabilities in Computer Use Agents**

cs.CR

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.05445v1) [paper-pdf](http://arxiv.org/pdf/2507.05445v1)

**Authors**: Daniel Jones, Giorgio Severi, Martin Pouliot, Gary Lopez, Joris de Gruyter, Santiago Zanella-Beguelin, Justin Song, Blake Bullwinkel, Pamela Cortez, Amanda Minnich

**Abstract**: Computer Use Agents (CUAs), autonomous systems that interact with software interfaces via browsers or virtual machines, are rapidly being deployed in consumer and enterprise environments. These agents introduce novel attack surfaces and trust boundaries that are not captured by traditional threat models. Despite their growing capabilities, the security boundaries of CUAs remain poorly understood. In this paper, we conduct a systematic threat analysis and testing of real-world CUAs under adversarial conditions. We identify seven classes of risks unique to the CUA paradigm, and analyze three concrete exploit scenarios in depth: (1) clickjacking via visual overlays that mislead interface-level reasoning, (2) indirect prompt injection that enables Remote Code Execution (RCE) through chained tool use, and (3) CoT exposure attacks that manipulate implicit interface framing to hijack multi-step reasoning. These case studies reveal deeper architectural flaws across current CUA implementations. Namely, a lack of input provenance tracking, weak interface-action binding, and insufficient control over agent memory and delegation. We conclude by proposing a CUA-specific security evaluation framework and design principles for safe deployment in adversarial and high-stakes settings.



## **28. Adversarial Machine Learning Attacks on Financial Reporting via Maximum Violated Multi-Objective Attack**

cs.LG

KDD Workshop on Machine Learning in Finance

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.05441v1) [paper-pdf](http://arxiv.org/pdf/2507.05441v1)

**Authors**: Edward Raff, Karen Kukla, Michel Benaroch, Joseph Comprix

**Abstract**: Bad actors, primarily distressed firms, have the incentive and desire to manipulate their financial reports to hide their distress and derive personal gains. As attackers, these firms are motivated by potentially millions of dollars and the availability of many publicly disclosed and used financial modeling frameworks. Existing attack methods do not work on this data due to anti-correlated objectives that must both be satisfied for the attacker to succeed. We introduce Maximum Violated Multi-Objective (MVMO) attacks that adapt the attacker's search direction to find $20\times$ more satisfying attacks compared to standard attacks. The result is that in $\approx50\%$ of cases, a company could inflate their earnings by 100-200%, while simultaneously reducing their fraud scores by 15%. By working with lawyers and professional accountants, we ensure our threat model is realistic to how such frauds are performed in practice.



## **29. Transfer Attack for Bad and Good: Explain and Boost Adversarial Transferability across Multimodal Large Language Models**

cs.CV

Accepted by ACM MM 2025

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2405.20090v4) [paper-pdf](http://arxiv.org/pdf/2405.20090v4)

**Authors**: Hao Cheng, Erjia Xiao, Jiayan Yang, Jinhao Duan, Yichi Wang, Jiahang Cao, Qiang Zhang, Le Yang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Multimodal Large Language Models (MLLMs) demonstrate exceptional performance in cross-modality interaction, yet they also suffer adversarial vulnerabilities. In particular, the transferability of adversarial examples remains an ongoing challenge. In this paper, we specifically analyze the manifestation of adversarial transferability among MLLMs and identify the key factors that influence this characteristic. We discover that the transferability of MLLMs exists in cross-LLM scenarios with the same vision encoder and indicate \underline{\textit{two key Factors}} that may influence transferability. We provide two semantic-level data augmentation methods, Adding Image Patch (AIP) and Typography Augment Transferability Method (TATM), which boost the transferability of adversarial examples across MLLMs. To explore the potential impact in the real world, we utilize two tasks that can have both negative and positive societal impacts: \ding{182} Harmful Content Insertion and \ding{183} Information Protection.



## **30. CLIP-Guided Backdoor Defense through Entropy-Based Poisoned Dataset Separation**

cs.MM

15 pages, 9 figures, 15 tables. To appear in the Proceedings of the  32nd ACM International Conference on Multimedia (MM '25)

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.05113v1) [paper-pdf](http://arxiv.org/pdf/2507.05113v1)

**Authors**: Binyan Xu, Fan Yang, Xilin Dai, Di Tang, Kehuan Zhang

**Abstract**: Deep Neural Networks (DNNs) are susceptible to backdoor attacks, where adversaries poison training data to implant backdoor into the victim model. Current backdoor defenses on poisoned data often suffer from high computational costs or low effectiveness against advanced attacks like clean-label and clean-image backdoors. To address them, we introduce CLIP-Guided backdoor Defense (CGD), an efficient and effective method that mitigates various backdoor attacks. CGD utilizes a publicly accessible CLIP model to identify inputs that are likely to be clean or poisoned. It then retrains the model with these inputs, using CLIP's logits as a guidance to effectively neutralize the backdoor. Experiments on 4 datasets and 11 attack types demonstrate that CGD reduces attack success rates (ASRs) to below 1% while maintaining clean accuracy (CA) with a maximum drop of only 0.3%, outperforming existing defenses. Additionally, we show that clean-data-based defenses can be adapted to poisoned data using CGD. Also, CGD exhibits strong robustness, maintaining low ASRs even when employing a weaker CLIP model or when CLIP itself is compromised by a backdoor. These findings underscore CGD's exceptional efficiency, effectiveness, and applicability for real-world backdoor defense scenarios. Code: https://github.com/binyxu/CGD.



## **31. BackFed: An Efficient & Standardized Benchmark Suite for Backdoor Attacks in Federated Learning**

cs.CR

Under review at NeurIPS'25

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04903v1) [paper-pdf](http://arxiv.org/pdf/2507.04903v1)

**Authors**: Thinh Dao, Dung Thuy Nguyen, Khoa D Doan, Kok-Seng Wong

**Abstract**: Federated Learning (FL) systems are vulnerable to backdoor attacks, where adversaries train their local models on poisoned data and submit poisoned model updates to compromise the global model. Despite numerous proposed attacks and defenses, divergent experimental settings, implementation errors, and unrealistic assumptions hinder fair comparisons and valid conclusions about their effectiveness in real-world scenarios. To address this, we introduce BackFed - a comprehensive benchmark suite designed to standardize, streamline, and reliably evaluate backdoor attacks and defenses in FL, with a focus on practical constraints. Our benchmark offers key advantages through its multi-processing implementation that significantly accelerates experimentation and the modular design that enables seamless integration of new methods via well-defined APIs. With a standardized evaluation pipeline, we envision BackFed as a plug-and-play environment for researchers to comprehensively and reliably evaluate new attacks and defenses. Using BackFed, we conduct large-scale studies of representative backdoor attacks and defenses across both Computer Vision and Natural Language Processing tasks with diverse model architectures and experimental settings. Our experiments critically assess the performance of proposed attacks and defenses, revealing unknown limitations and modes of failures under practical conditions. These empirical insights provide valuable guidance for the development of new methods and for enhancing the security of FL systems. Our framework is openly available at https://github.com/thinh-dao/BackFed.



## **32. Beyond Training-time Poisoning: Component-level and Post-training Backdoors in Deep Reinforcement Learning**

cs.LG

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04883v1) [paper-pdf](http://arxiv.org/pdf/2507.04883v1)

**Authors**: Sanyam Vyas, Alberto Caron, Chris Hicks, Pete Burnap, Vasilios Mavroudis

**Abstract**: Deep Reinforcement Learning (DRL) systems are increasingly used in safety-critical applications, yet their security remains severely underexplored. This work investigates backdoor attacks, which implant hidden triggers that cause malicious actions only when specific inputs appear in the observation space. Existing DRL backdoor research focuses solely on training-time attacks requiring unrealistic access to the training pipeline. In contrast, we reveal critical vulnerabilities across the DRL supply chain where backdoors can be embedded with significantly reduced adversarial privileges. We introduce two novel attacks: (1) TrojanentRL, which exploits component-level flaws to implant a persistent backdoor that survives full model retraining; and (2) InfrectroRL, a post-training backdoor attack which requires no access to training, validation, nor test data. Empirical and analytical evaluations across six Atari environments show our attacks rival state-of-the-art training-time backdoor attacks while operating under much stricter adversarial constraints. We also demonstrate that InfrectroRL further evades two leading DRL backdoor defenses. These findings challenge the current research focus and highlight the urgent need for robust defenses.



## **33. Phantom Subgroup Poisoning: Stealth Attacks on Federated Recommender Systems**

cs.CR

13 pages

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.06258v1) [paper-pdf](http://arxiv.org/pdf/2507.06258v1)

**Authors**: Bo Yan, Yurong Hao, Dingqi Liu, Huabin Sun, Pengpeng Qiao, Wei Yang Bryan Lim, Yang Cao, Chuan Shi

**Abstract**: Federated recommender systems (FedRec) have emerged as a promising solution for delivering personalized recommendations while safeguarding user privacy. However, recent studies have demonstrated their vulnerability to poisoning attacks. Existing attacks typically target the entire user group, which compromises stealth and increases the risk of detection. In contrast, real-world adversaries may prefer to prompt target items to specific user subgroups, such as recommending health supplements to elderly users. Motivated by this gap, we introduce Spattack, the first targeted poisoning attack designed to manipulate recommendations for specific user subgroups in the federated setting. Specifically, Spattack adopts a two-stage approximation-and-promotion strategy, which first simulates user embeddings of target/non-target subgroups and then prompts target items to the target subgroups. To enhance the approximation stage, we push the inter-group embeddings away based on contrastive learning and augment the target group's relevant item set based on clustering. To enhance the promotion stage, we further propose to adaptively tune the optimization weights between target and non-target subgroups. Besides, an embedding alignment strategy is proposed to align the embeddings between the target items and the relevant items. We conduct comprehensive experiments on three real-world datasets, comparing Spattack against seven state-of-the-art poisoning attacks and seven representative defense mechanisms. Experimental results demonstrate that Spattack consistently achieves strong manipulation performance on the specific user subgroup, while incurring minimal impact on non-target users, even when only 0.1\% of users are malicious. Moreover, Spattack maintains competitive overall recommendation performance and exhibits strong resilience against existing mainstream defenses.



## **34. Diffusion-based Adversarial Identity Manipulation for Facial Privacy Protection**

cs.CV

Accepted by ACM MM 2025

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2504.21646v2) [paper-pdf](http://arxiv.org/pdf/2504.21646v2)

**Authors**: Liqin Wang, Qianyue Hu, Wei Lu, Xiangyang Luo

**Abstract**: The success of face recognition (FR) systems has led to serious privacy concerns due to potential unauthorized surveillance and user tracking on social networks. Existing methods for enhancing privacy fail to generate natural face images that can protect facial privacy. In this paper, we propose diffusion-based adversarial identity manipulation (DiffAIM) to generate natural and highly transferable adversarial faces against malicious FR systems. To be specific, we manipulate facial identity within the low-dimensional latent space of a diffusion model. This involves iteratively injecting gradient-based adversarial identity guidance during the reverse diffusion process, progressively steering the generation toward the desired adversarial faces. The guidance is optimized for identity convergence towards a target while promoting semantic divergence from the source, facilitating effective impersonation while maintaining visual naturalness. We further incorporate structure-preserving regularization to preserve facial structure consistency during manipulation. Extensive experiments on both face verification and identification tasks demonstrate that compared with the state-of-the-art, DiffAIM achieves stronger black-box attack transferability while maintaining superior visual quality. We also demonstrate the effectiveness of the proposed approach for commercial FR APIs, including Face++ and Aliyun.



## **35. Robustifying 3D Perception through Least-Squares Multi-Agent Graphs Object Tracking**

cs.CV

6 pages, 3 figures, 4 tables

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04762v1) [paper-pdf](http://arxiv.org/pdf/2507.04762v1)

**Authors**: Maria Damanaki, Ioulia Kapsali, Nikos Piperigkos, Alexandros Gkillas, Aris S. Lalos

**Abstract**: The critical perception capabilities of EdgeAI systems, such as autonomous vehicles, are required to be resilient against adversarial threats, by enabling accurate identification and localization of multiple objects in the scene over time, mitigating their impact. Single-agent tracking offers resilience to adversarial attacks but lacks situational awareness, underscoring the need for multi-agent cooperation to enhance context understanding and robustness. This paper proposes a novel mitigation framework on 3D LiDAR scene against adversarial noise by tracking objects based on least-squares graph on multi-agent adversarial bounding boxes. Specifically, we employ the least-squares graph tool to reduce the induced positional error of each detection's centroid utilizing overlapped bounding boxes on a fully connected graph via differential coordinates and anchor points. Hence, the multi-vehicle detections are fused and refined mitigating the adversarial impact, and associated with existing tracks in two stages performing tracking to further suppress the adversarial threat. An extensive evaluation study on the real-world V2V4Real dataset demonstrates that the proposed method significantly outperforms both state-of-the-art single and multi-agent tracking frameworks by up to 23.3% under challenging adversarial conditions, operating as a resilient approach without relying on additional defense mechanisms.



## **36. Attacker's Noise Can Manipulate Your Audio-based LLM in the Real World**

cs.CR

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.06256v1) [paper-pdf](http://arxiv.org/pdf/2507.06256v1)

**Authors**: Vinu Sankar Sadasivan, Soheil Feizi, Rajiv Mathews, Lun Wang

**Abstract**: This paper investigates the real-world vulnerabilities of audio-based large language models (ALLMs), such as Qwen2-Audio. We first demonstrate that an adversary can craft stealthy audio perturbations to manipulate ALLMs into exhibiting specific targeted behaviors, such as eliciting responses to wake-keywords (e.g., "Hey Qwen"), or triggering harmful behaviors (e.g. "Change my calendar event"). Subsequently, we show that playing adversarial background noise during user interaction with the ALLMs can significantly degrade the response quality. Crucially, our research illustrates the scalability of these attacks to real-world scenarios, impacting other innocent users when these adversarial noises are played through the air. Further, we discuss the transferrability of the attack, and potential defensive measures.



## **37. Trojan Horse Prompting: Jailbreaking Conversational Multimodal Models by Forging Assistant Message**

cs.AI

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04673v1) [paper-pdf](http://arxiv.org/pdf/2507.04673v1)

**Authors**: Wei Duan, Li Qian

**Abstract**: The rise of conversational interfaces has greatly enhanced LLM usability by leveraging dialogue history for sophisticated reasoning. However, this reliance introduces an unexplored attack surface. This paper introduces Trojan Horse Prompting, a novel jailbreak technique. Adversaries bypass safety mechanisms by forging the model's own past utterances within the conversational history provided to its API. A malicious payload is injected into a model-attributed message, followed by a benign user prompt to trigger harmful content generation. This vulnerability stems from Asymmetric Safety Alignment: models are extensively trained to refuse harmful user requests but lack comparable skepticism towards their own purported conversational history. This implicit trust in its "past" creates a high-impact vulnerability. Experimental validation on Google's Gemini-2.0-flash-preview-image-generation shows Trojan Horse Prompting achieves a significantly higher Attack Success Rate (ASR) than established user-turn jailbreaking methods. These findings reveal a fundamental flaw in modern conversational AI security, necessitating a paradigm shift from input-level filtering to robust, protocol-level validation of conversational context integrity.



## **38. Smart Grid: Cyber Attacks, Critical Defense Approaches, and Digital Twin**

cs.CR

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2205.11783v2) [paper-pdf](http://arxiv.org/pdf/2205.11783v2)

**Authors**: Tianming Zheng, Ping Yi, Yue Wu

**Abstract**: As a national critical infrastructure, the smart grid has attracted widespread attention for its cybersecurity issues. The development towards an intelligent, digital, and Internet-connected smart grid has attracted external adversaries for malicious activities. It is necessary to enhance its cybersecurity by both improving the existing defense approaches and introducing novel developed technologies to the smart grid context. As an emerging technology, digital twin (DT) is considered as an enabler for enhanced security. However, the practical implementation is quite challenging. This is due to the knowledge barriers among smart grid designers, security experts, and DT developers. Each single domain is a complicated system covering various components and technologies. As a result, works are needed to sort out relevant contents so that DT can be better embedded in the security architecture design of smart grid.   In order to meet this demand, our paper covers the above three domains, i.e., smart grid, cybersecurity, and DT. Specifically, the paper i) introduces the background of the smart grid; ii) reviews external cyber attacks from attack incidents and attack methods; iii) introduces critical defense approaches in industrial cyber systems, which include device identification, vulnerability discovery, intrusion detection systems (IDSs), honeypots, attribution, and threat intelligence (TI); iv) reviews the relevant content of DT, including its basic concepts, applications in the smart grid, and how DT enhances the security. In the end, the paper puts forward our security considerations on the future development of DT-based smart grid. The survey is expected to help developers break knowledge barriers among smart grid, cybersecurity, and DT, and provide guidelines for future security design of DT-based smart grid.



## **39. Backdooring Bias ($B^2$) into Stable Diffusion Models**

cs.LG

Accepted to USENIX Security '25

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2406.15213v4) [paper-pdf](http://arxiv.org/pdf/2406.15213v4)

**Authors**: Ali Naseh, Jaechul Roh, Eugene Bagdasarian, Amir Houmansadr

**Abstract**: Recent advances in large text-conditional diffusion models have revolutionized image generation by enabling users to create realistic, high-quality images from textual prompts, significantly enhancing artistic creation and visual communication. However, these advancements also introduce an underexplored attack opportunity: the possibility of inducing biases by an adversary into the generated images for malicious intentions, e.g., to influence public opinion and spread propaganda. In this paper, we study an attack vector that allows an adversary to inject arbitrary bias into a target model. The attack leverages low-cost backdooring techniques using a targeted set of natural textual triggers embedded within a small number of malicious data samples produced with public generative models. An adversary could pick common sequences of words that can then be inadvertently activated by benign users during inference. We investigate the feasibility and challenges of such attacks, demonstrating how modern generative models have made this adversarial process both easier and more adaptable. On the other hand, we explore various aspects of the detectability of such attacks and demonstrate that the model's utility remains intact in the absence of the triggers. Our extensive experiments using over 200,000 generated images and against hundreds of fine-tuned models demonstrate the feasibility of the presented backdoor attack. We illustrate how these biases maintain strong text-image alignment, highlighting the challenges in detecting biased images without knowing that bias in advance. Our cost analysis confirms the low financial barrier (\$10-\$15) to executing such attacks, underscoring the need for robust defensive strategies against such vulnerabilities in diffusion models.



## **40. False Alarms, Real Damage: Adversarial Attacks Using LLM-based Models on Text-based Cyber Threat Intelligence Systems**

cs.CR

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2507.06252v1) [paper-pdf](http://arxiv.org/pdf/2507.06252v1)

**Authors**: Samaneh Shafee, Alysson Bessani, Pedro M. Ferreira

**Abstract**: Cyber Threat Intelligence (CTI) has emerged as a vital complementary approach that operates in the early phases of the cyber threat lifecycle. CTI involves collecting, processing, and analyzing threat data to provide a more accurate and rapid understanding of cyber threats. Due to the large volume of data, automation through Machine Learning (ML) and Natural Language Processing (NLP) models is essential for effective CTI extraction. These automated systems leverage Open Source Intelligence (OSINT) from sources like social networks, forums, and blogs to identify Indicators of Compromise (IoCs). Although prior research has focused on adversarial attacks on specific ML models, this study expands the scope by investigating vulnerabilities within various components of the entire CTI pipeline and their susceptibility to adversarial attacks. These vulnerabilities arise because they ingest textual inputs from various open sources, including real and potentially fake content. We analyse three types of attacks against CTI pipelines, including evasion, flooding, and poisoning, and assess their impact on the system's information selection capabilities. Specifically, on fake text generation, the work demonstrates how adversarial text generation techniques can create fake cybersecurity and cybersecurity-like text that misleads classifiers, degrades performance, and disrupts system functionality. The focus is primarily on the evasion attack, as it precedes and enables flooding and poisoning attacks within the CTI pipeline.



## **41. Addressing The Devastating Effects Of Single-Task Data Poisoning In Exemplar-Free Continual Learning**

cs.CR

Accepted at CoLLAs 2025

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2507.04106v1) [paper-pdf](http://arxiv.org/pdf/2507.04106v1)

**Authors**: Stanisław Pawlak, Bartłomiej Twardowski, Tomasz Trzciński, Joost van de Weijer

**Abstract**: Our research addresses the overlooked security concerns related to data poisoning in continual learning (CL). Data poisoning - the intentional manipulation of training data to affect the predictions of machine learning models - was recently shown to be a threat to CL training stability. While existing literature predominantly addresses scenario-dependent attacks, we propose to focus on a more simple and realistic single-task poison (STP) threats. In contrast to previously proposed poisoning settings, in STP adversaries lack knowledge and access to the model, as well as to both previous and future tasks. During an attack, they only have access to the current task within the data stream. Our study demonstrates that even within these stringent conditions, adversaries can compromise model performance using standard image corruptions. We show that STP attacks are able to strongly disrupt the whole continual training process: decreasing both the stability (its performance on past tasks) and plasticity (capacity to adapt to new tasks) of the algorithm. Finally, we propose a high-level defense framework for CL along with a poison task detection method based on task vectors. The code is available at https://github.com/stapaw/STP.git .



## **42. A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models**

cs.IR

Accepted by ACL 2025 Findings

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2507.05288v1) [paper-pdf](http://arxiv.org/pdf/2507.05288v1)

**Authors**: Shuliang Liu, Hongyi Liu, Aiwei Liu, Bingchen Duan, Qi Zheng, Yibo Yan, He Geng, Peijie Jiang, Jia Liu, Xuming Hu

**Abstract**: The widespread deployment of large language models (LLMs) across critical domains has amplified the societal risks posed by algorithmically generated misinformation. Unlike traditional false content, LLM-generated misinformation can be self-reinforcing, highly plausible, and capable of rapid propagation across multiple languages, which traditional detection methods fail to mitigate effectively. This paper introduces a proactive defense paradigm, shifting from passive post hoc detection to anticipatory mitigation strategies. We propose a Three Pillars framework: (1) Knowledge Credibility, fortifying the integrity of training and deployed data; (2) Inference Reliability, embedding self-corrective mechanisms during reasoning; and (3) Input Robustness, enhancing the resilience of model interfaces against adversarial attacks. Through a comprehensive survey of existing techniques and a comparative meta-analysis, we demonstrate that proactive defense strategies offer up to 63\% improvement over conventional methods in misinformation prevention, despite non-trivial computational overhead and generalization challenges. We argue that future research should focus on co-designing robust knowledge foundations, reasoning certification, and attack-resistant interfaces to ensure LLMs can effectively counter misinformation across varied domains.



## **43. Multichannel Steganography: A Provably Secure Hybrid Steganographic Model for Secure Communication**

cs.CR

22 pages, 15 figures, 4 algorithms. This version is a preprint  uploaded to arXiv

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2501.04511v2) [paper-pdf](http://arxiv.org/pdf/2501.04511v2)

**Authors**: Obinna Omego, Michal Bosy

**Abstract**: Secure covert communication in hostile environments requires simultaneously achieving invisibility, provable security guarantees, and robustness against informed adversaries. This paper presents a novel hybrid steganographic framework that unites cover synthesis and cover modification within a unified multichannel protocol. A secret-seeded PRNG drives a lightweight Markov-chain generator to produce contextually plausible cover parameters, which are then masked with the payload and dispersed across independent channels. The masked bit-vector is imperceptibly embedded into conventional media via a variance-aware least-significant-bit algorithm, ensuring that statistical properties remain within natural bounds. We formalize a multichannel adversary model (MC-ATTACK) and prove that, under standard security assumptions, the adversary's distinguishing advantage is negligible, thereby guaranteeing both confidentiality and integrity. Empirical results corroborate these claims: local-variance-guided embedding yields near-lossless extraction (mean BER $<5\times10^{-3}$, correlation $>0.99$) with minimal perceptual distortion (PSNR $\approx100$,dB, SSIM $>0.99$), while key-based masking drives extraction success to zero (BER $\approx0.5$) for a fully informed adversary. Comparative analysis demonstrates that purely distortion-free or invertible schemes fail under the same threat model, underscoring the necessity of hybrid designs. The proposed approach advances high-assurance steganography by delivering an efficient, provably secure covert channel suitable for deployment in high-surveillance networks.



## **44. When There Is No Decoder: Removing Watermarks from Stable Diffusion Models in a No-box Setting**

cs.CR

arXiv admin note: text overlap with arXiv:2408.02035

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03646v1) [paper-pdf](http://arxiv.org/pdf/2507.03646v1)

**Authors**: Xiaodong Wu, Tianyi Tang, Xiangman Li, Jianbing Ni, Yong Yu

**Abstract**: Watermarking has emerged as a promising solution to counter harmful or deceptive AI-generated content by embedding hidden identifiers that trace content origins. However, the robustness of current watermarking techniques is still largely unexplored, raising critical questions about their effectiveness against adversarial attacks. To address this gap, we examine the robustness of model-specific watermarking, where watermark embedding is integrated with text-to-image generation in models like latent diffusion models. We introduce three attack strategies: edge prediction-based, box blurring, and fine-tuning-based attacks in a no-box setting, where an attacker does not require access to the ground-truth watermark decoder. Our findings reveal that while model-specific watermarking is resilient against basic evasion attempts, such as edge prediction, it is notably vulnerable to blurring and fine-tuning-based attacks. Our best-performing attack achieves a reduction in watermark detection accuracy to approximately 47.92\%. Additionally, we perform an ablation study on factors like message length, kernel size and decoder depth, identifying critical parameters influencing the fine-tuning attack's success. Finally, we assess several advanced watermarking defenses, finding that even the most robust methods, such as multi-label smoothing, result in watermark extraction accuracy that falls below an acceptable level when subjected to our no-box attacks.



## **45. Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States**

cs.LG

4 figures

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2503.09066v2) [paper-pdf](http://arxiv.org/pdf/2503.09066v2)

**Authors**: Xin Wei Chia, Swee Liang Wong, Jonathan Pan

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.



## **46. On the Limits of Robust Control Under Adversarial Disturbances**

eess.SY

Extended version of a manuscript submitted to IEEE Transactions on  Automatic Control, July 2025

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03630v1) [paper-pdf](http://arxiv.org/pdf/2507.03630v1)

**Authors**: Paul Trodden, José M. Maestre, Hideaki Ishii

**Abstract**: This paper addresses a fundamental and important question in control: under what conditions does there fail to exist a robust control policy that keeps the state of a constrained linear system within a target set, despite bounded disturbances? This question has practical implications for actuator and sensor specification, feasibility analysis for reference tracking, and the design of adversarial attacks in cyber-physical systems. While prior research has predominantly focused on using optimization to compute control-invariant sets to ensure feasible operation, our work complements these approaches by characterizing explicit sufficient conditions under which robust control is fundamentally infeasible. Specifically, we derive novel closed-form, algebraic expressions that relate the size of a disturbance set -- modelled as a scaled version of a basic shape -- to the system's spectral properties and the geometry of the constraint sets.



## **47. Beyond Weaponization: NLP Security for Medium and Lower-Resourced Languages in Their Own Right**

cs.CL

Pre-print

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03473v1) [paper-pdf](http://arxiv.org/pdf/2507.03473v1)

**Authors**: Heather Lent

**Abstract**: Despite mounting evidence that multilinguality can be easily weaponized against language models (LMs), works across NLP Security remain overwhelmingly English-centric. In terms of securing LMs, the NLP norm of "English first" collides with standard procedure in cybersecurity, whereby practitioners are expected to anticipate and prepare for worst-case outcomes. To mitigate worst-case outcomes in NLP Security, researchers must be willing to engage with the weakest links in LM security: lower-resourced languages. Accordingly, this work examines the security of LMs for lower- and medium-resourced languages. We extend existing adversarial attacks for up to 70 languages to evaluate the security of monolingual and multilingual LMs for these languages. Through our analysis, we find that monolingual models are often too small in total number of parameters to ensure sound security, and that while multilinguality is helpful, it does not always guarantee improved security either. Ultimately, these findings highlight important considerations for more secure deployment of LMs, for communities of lower-resourced languages.



## **48. Evaluating the Evaluators: Trust in Adversarial Robustness Tests**

cs.CR

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03450v1) [paper-pdf](http://arxiv.org/pdf/2507.03450v1)

**Authors**: Antonio Emanuele Cinà, Maura Pintor, Luca Demetrio, Ambra Demontis, Battista Biggio, Fabio Roli

**Abstract**: Despite significant progress in designing powerful adversarial evasion attacks for robustness verification, the evaluation of these methods often remains inconsistent and unreliable. Many assessments rely on mismatched models, unverified implementations, and uneven computational budgets, which can lead to biased results and a false sense of security. Consequently, robustness claims built on such flawed testing protocols may be misleading and give a false sense of security. As a concrete step toward improving evaluation reliability, we present AttackBench, a benchmark framework developed to assess the effectiveness of gradient-based attacks under standardized and reproducible conditions. AttackBench serves as an evaluation tool that ranks existing attack implementations based on a novel optimality metric, which enables researchers and practitioners to identify the most reliable and effective attack for use in subsequent robustness evaluations. The framework enforces consistent testing conditions and enables continuous updates, making it a reliable foundation for robustness verification.



## **49. Rectifying Adversarial Sample with Low Entropy Prior for Test-Time Defense**

cs.CV

To appear in IEEEE Transactions on Multimedia

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03427v1) [paper-pdf](http://arxiv.org/pdf/2507.03427v1)

**Authors**: Lina Ma, Xiaowei Fu, Fuxiang Huang, Xinbo Gao, Lei Zhang

**Abstract**: Existing defense methods fail to defend against unknown attacks and thus raise generalization issue of adversarial robustness. To remedy this problem, we attempt to delve into some underlying common characteristics among various attacks for generality. In this work, we reveal the commonly overlooked low entropy prior (LE) implied in various adversarial samples, and shed light on the universal robustness against unseen attacks in inference phase. LE prior is elaborated as two properties across various attacks as shown in Fig. 1 and Fig. 2: 1) low entropy misclassification for adversarial samples and 2) lower entropy prediction for higher attack intensity. This phenomenon stands in stark contrast to the naturally distributed samples. The LE prior can instruct existing test-time defense methods, thus we propose a two-stage REAL approach: Rectify Adversarial sample based on LE prior for test-time adversarial rectification. Specifically, to align adversarial samples more closely with clean samples, we propose to first rectify adversarial samples misclassified with low entropy by reverse maximizing prediction entropy, thereby eliminating their adversarial nature. To ensure the rectified samples can be correctly classified with low entropy, we carry out secondary rectification by forward minimizing prediction entropy, thus creating a Max-Min entropy optimization scheme. Further, based on the second property, we propose an attack-aware weighting mechanism to adaptively adjust the strengths of Max-Min entropy objectives. Experiments on several datasets show that REAL can greatly improve the performance of existing sample rectification models.



## **50. Breaking the Bulkhead: Demystifying Cross-Namespace Reference Vulnerabilities in Kubernetes Operators**

cs.CR

12 pages

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03387v1) [paper-pdf](http://arxiv.org/pdf/2507.03387v1)

**Authors**: Andong Chen, Zhaoxuan Jin, Ziyi Guo, Yan Chen

**Abstract**: Kubernetes Operators, automated tools designed to manage application lifecycles within Kubernetes clusters, extend the functionalities of Kubernetes, and reduce the operational burden on human engineers. While Operators significantly simplify DevOps workflows, they introduce new security risks. In particular, Kubernetes enforces namespace isolation to separate workloads and limit user access, ensuring that users can only interact with resources within their authorized namespaces. However, Kubernetes Operators often demand elevated privileges and may interact with resources across multiple namespaces. This introduces a new class of vulnerabilities, the Cross-Namespace Reference Vulnerability. The root cause lies in the mismatch between the declared scope of resources and the implemented scope of the Operator logic, resulting in Kubernetes being unable to properly isolate the namespace. Leveraging such vulnerability, an adversary with limited access to a single authorized namespace may exploit the Operator to perform operations affecting other unauthorized namespaces, causing Privilege Escalation and further impacts. To the best of our knowledge, this paper is the first to systematically investigate the security vulnerability of Kubernetes Operators. We present Cross-Namespace Reference Vulnerability with two strategies, demonstrating how an attacker can bypass namespace isolation. Through large-scale measurements, we found that over 14% of Operators in the wild are potentially vulnerable. Our findings have been reported to the relevant developers, resulting in 7 confirmations and 6 CVEs by the time of submission, affecting vendors including ****** and ******, highlighting the critical need for enhanced security practices in Kubernetes Operators. To mitigate it, we also open-source the static analysis suite to benefit the ecosystem.



