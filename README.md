# Latest Adversarial Attack Papers
**update at 2025-04-14 11:11:16**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. MCP Safety Audit: LLMs with the Model Context Protocol Allow Major Security Exploits**

cs.CR

27 pages, 21 figures, and 2 Tables. Cleans up the TeX source

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.03767v2) [paper-pdf](http://arxiv.org/pdf/2504.03767v2)

**Authors**: Brandon Radosevich, John Halloran

**Abstract**: To reduce development overhead and enable seamless integration between potential components comprising any given generative AI application, the Model Context Protocol (MCP) (Anthropic, 2024) has recently been released and subsequently widely adopted. The MCP is an open protocol that standardizes API calls to large language models (LLMs), data sources, and agentic tools. By connecting multiple MCP servers, each defined with a set of tools, resources, and prompts, users are able to define automated workflows fully driven by LLMs. However, we show that the current MCP design carries a wide range of security risks for end users. In particular, we demonstrate that industry-leading LLMs may be coerced into using MCP tools to compromise an AI developer's system through various attacks, such as malicious code execution, remote access control, and credential theft. To proactively mitigate these and related attacks, we introduce a safety auditing tool, MCPSafetyScanner, the first agentic tool to assess the security of an arbitrary MCP server. MCPScanner uses several agents to (a) automatically determine adversarial samples given an MCP server's tools and resources; (b) search for related vulnerabilities and remediations based on those samples; and (c) generate a security report detailing all findings. Our work highlights serious security issues with general-purpose agentic workflows while also providing a proactive tool to audit MCP server safety and address detected vulnerabilities before deployment.   The described MCP server auditing tool, MCPSafetyScanner, is freely available at: https://github.com/johnhalloran321/mcpSafetyScanner



## **2. Enabling Safety for Aerial Robots: Planning and Control Architectures**

cs.RO

2025 ICRA Workshop on 25 years of Aerial Robotics: Challenges and  Opportunities

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08601v1) [paper-pdf](http://arxiv.org/pdf/2504.08601v1)

**Authors**: Kaleb Ben Naveed, Devansh R. Agrawal, Daniel M. Cherenson, Haejoon Lee, Alia Gilbert, Hardik Parwana, Vishnu S. Chipade, William Bentz, Dimitra Panagou

**Abstract**: Ensuring safe autonomy is crucial for deploying aerial robots in real-world applications. However, safety is a multifaceted challenge that must be addressed from multiple perspectives, including navigation in dynamic environments, operation under resource constraints, and robustness against adversarial attacks and uncertainties. In this paper, we present the authors' recent work that tackles some of these challenges and highlights key aspects that must be considered to enhance the safety and performance of autonomous aerial systems. All presented approaches are validated through hardware experiments.



## **3. An Early Experience with Confidential Computing Architecture for On-Device Model Protection**

cs.CR

Accepted to the 8th Workshop on System Software for Trusted Execution  (SysTEX 2025)

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08508v1) [paper-pdf](http://arxiv.org/pdf/2504.08508v1)

**Authors**: Sina Abdollahi, Mohammad Maheri, Sandra Siby, Marios Kogias, Hamed Haddadi

**Abstract**: Deploying machine learning (ML) models on user devices can improve privacy (by keeping data local) and reduce inference latency. Trusted Execution Environments (TEEs) are a practical solution for protecting proprietary models, yet existing TEE solutions have architectural constraints that hinder on-device model deployment. Arm Confidential Computing Architecture (CCA), a new Arm extension, addresses several of these limitations and shows promise as a secure platform for on-device ML. In this paper, we evaluate the performance-privacy trade-offs of deploying models within CCA, highlighting its potential to enable confidential and efficient ML applications. Our evaluations show that CCA can achieve an overhead of, at most, 22% in running models of different sizes and applications, including image classification, voice recognition, and chat assistants. This performance overhead comes with privacy benefits; for example, our framework can successfully protect the model against membership inference attack by an 8.3% reduction in the adversary's success rate. To support further research and early adoption, we make our code and methodology publicly available.



## **4. Nanopass Back-Translation of Call-Return Trees for Mechanized Secure Compilation Proofs**

cs.PL

ITP'25 submission, updated with link to Rocq development

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2503.19609v2) [paper-pdf](http://arxiv.org/pdf/2503.19609v2)

**Authors**: Jérémy Thibault, Joseph Lenormand, Catalin Hritcu

**Abstract**: Researchers aim to build secure compilation chains enforcing that if there is no attack a source context can mount against a source program then there is also no attack an adversarial target context can mount against the compiled program. Proving that these compilation chains are secure is, however, challenging, and involves a non-trivial back-translation step: for any attack a target context mounts against the compiled program one has to exhibit a source context mounting the same attack against the source program. We describe a novel back-translation technique, which results in simpler proofs that can be more easily mechanized in a proof assistant. Given a finite set of finite trace prefixes, capturing the interaction recorded during an attack between a target context and the compiled program, we build a call-return tree that we back-translate into a source context producing the same trace prefixes. We use state in the generated source context to record the current location in the call-return tree. The back-translation is done in several small steps, each adding to the tree new information describing how the location should change depending on how the context regains control. To prove this back-translation correct we give semantics to every intermediate call-return tree language, using ghost state to store information and explicitly enforce execution invariants. We prove several small forward simulations, basically seeing the back-translation as a verified nanopass compiler. Thanks to this modular structure, we are able to mechanize this complex back-translation and its correctness proof in the Rocq prover without too much effort.



## **5. Toward Realistic Adversarial Attacks in IDS: A Novel Feasibility Metric for Transferability**

cs.CR

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08480v1) [paper-pdf](http://arxiv.org/pdf/2504.08480v1)

**Authors**: Sabrine Ennaji, Elhadj Benkhelifa, Luigi Vincenzo Mancini

**Abstract**: Transferability-based adversarial attacks exploit the ability of adversarial examples, crafted to deceive a specific source Intrusion Detection System (IDS) model, to also mislead a target IDS model without requiring access to the training data or any internal model parameters. These attacks exploit common vulnerabilities in machine learning models to bypass security measures and compromise systems. Although the transferability concept has been widely studied, its practical feasibility remains limited due to assumptions of high similarity between source and target models. This paper analyzes the core factors that contribute to transferability, including feature alignment, model architectural similarity, and overlap in the data distributions that each IDS examines. We propose a novel metric, the Transferability Feasibility Score (TFS), to assess the feasibility and reliability of such attacks based on these factors. Through experimental evidence, we demonstrate that TFS and actual attack success rates are highly correlated, addressing the gap between theoretical understanding and real-world impact. Our findings provide needed guidance for designing more realistic transferable adversarial attacks, developing robust defenses, and ultimately improving the security of machine learning-based IDS in critical systems.



## **6. TACO: Adversarial Camouflage Optimization on Trucks to Fool Object Detectors**

cs.CV

This version matches the final published version in Big Data and  Cognitive Computing (MDPI). Please cite the journal version when referencing  this work (doi: https://doi.org/10.3390/bdcc9030072)

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2410.21443v2) [paper-pdf](http://arxiv.org/pdf/2410.21443v2)

**Authors**: Adonisz Dimitriu, Tamás Michaletzky, Viktor Remeli

**Abstract**: Adversarial attacks threaten the reliability of machine learning models in critical applications like autonomous vehicles and defense systems. As object detectors become more robust with models like YOLOv8, developing effective adversarial methodologies is increasingly challenging. We present Truck Adversarial Camouflage Optimization (TACO), a novel framework that generates adversarial camouflage patterns on 3D vehicle models to deceive state-of-the-art object detectors. Adopting Unreal Engine 5, TACO integrates differentiable rendering with a Photorealistic Rendering Network to optimize adversarial textures targeted at YOLOv8. To ensure the generated textures are both effective in deceiving detectors and visually plausible, we introduce the Convolutional Smooth Loss function, a generalized smooth loss function. Experimental evaluations demonstrate that TACO significantly degrades YOLOv8's detection performance, achieving an AP@0.5 of 0.0099 on unseen test data. Furthermore, these adversarial patterns exhibit strong transferability to other object detection models such as Faster R-CNN and earlier YOLO versions.



## **7. Adversarial Examples in Environment Perception for Automated Driving (Review)**

cs.CV

One chapter of upcoming Springer book: Recent Advances in Autonomous  Vehicle Technology, 2025

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08414v1) [paper-pdf](http://arxiv.org/pdf/2504.08414v1)

**Authors**: Jun Yan, Huilin Yin

**Abstract**: The renaissance of deep learning has led to the massive development of automated driving. However, deep neural networks are vulnerable to adversarial examples. The perturbations of adversarial examples are imperceptible to human eyes but can lead to the false predictions of neural networks. It poses a huge risk to artificial intelligence (AI) applications for automated driving. This survey systematically reviews the development of adversarial robustness research over the past decade, including the attack and defense methods and their applications in automated driving. The growth of automated driving pushes forward the realization of trustworthy AI applications. This review lists significant references in the research history of adversarial examples.



## **8. Statistical Linear Regression Approach to Kalman Filtering and Smoothing under Cyber-Attacks**

eess.SP

5 pages, 4 figures

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08404v1) [paper-pdf](http://arxiv.org/pdf/2504.08404v1)

**Authors**: Kundan Kumar, Muhammad Iqbal, Simo Särkkä

**Abstract**: Remote state estimation in cyber-physical systems is often vulnerable to cyber-attacks due to wireless connections between sensors and computing units. In such scenarios, adversaries compromise the system by injecting false data or blocking measurement transmissions via denial-of-service attacks, distorting sensor readings. This paper develops a Kalman filter and Rauch--Tung--Striebel (RTS) smoother for linear stochastic state-space models subject to cyber-attacked measurements. We approximate the faulty measurement model via generalized statistical linear regression (GSLR). The GSLR-based approximated measurement model is then used to develop a Kalman filter and RTS smoother for the problem. The effectiveness of the proposed algorithms under cyber-attacks is demonstrated through a simulated aircraft tracking experiment.



## **9. To See or Not to See -- Fingerprinting Devices in Adversarial Environments Amid Advanced Machine Learning**

cs.CR

10 pages, 4 figures

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08264v1) [paper-pdf](http://arxiv.org/pdf/2504.08264v1)

**Authors**: Justin Feng, Nader Sehatbakhsh

**Abstract**: The increasing use of the Internet of Things raises security concerns. To address this, device fingerprinting is often employed to authenticate devices, detect adversaries, and identify eavesdroppers in an environment. This requires the ability to discern between legitimate and malicious devices which is achieved by analyzing the unique physical and/or operational characteristics of IoT devices. In the era of the latest progress in machine learning, particularly generative models, it is crucial to methodically examine the current studies in device fingerprinting. This involves explaining their approaches and underscoring their limitations when faced with adversaries armed with these ML tools. To systematically analyze existing methods, we propose a generic, yet simplified, model for device fingerprinting. Additionally, we thoroughly investigate existing methods to authenticate devices and detect eavesdropping, using our proposed model. We further study trends and similarities between works in authentication and eavesdropping detection and present the existing threats and attacks in these domains. Finally, we discuss future directions in fingerprinting based on these trends to develop more secure IoT fingerprinting schemes.



## **10. EO-VLM: VLM-Guided Energy Overload Attacks on Vision Models**

cs.CV

Presented as a poster at ACSAC 2024

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08205v1) [paper-pdf](http://arxiv.org/pdf/2504.08205v1)

**Authors**: Minjae Seo, Myoungsung You, Junhee Lee, Jaehan Kim, Hwanjo Heo, Jintae Oh, Jinwoo Kim

**Abstract**: Vision models are increasingly deployed in critical applications such as autonomous driving and CCTV monitoring, yet they remain susceptible to resource-consuming attacks. In this paper, we introduce a novel energy-overloading attack that leverages vision language model (VLM) prompts to generate adversarial images targeting vision models. These images, though imperceptible to the human eye, significantly increase GPU energy consumption across various vision models, threatening the availability of these systems. Our framework, EO-VLM (Energy Overload via VLM), is model-agnostic, meaning it is not limited by the architecture or type of the target vision model. By exploiting the lack of safety filters in VLMs like DALL-E 3, we create adversarial noise images without requiring prior knowledge or internal structure of the target vision models. Our experiments demonstrate up to a 50% increase in energy consumption, revealing a critical vulnerability in current vision models.



## **11. Adversarial Attacks on Data Attribution**

cs.LG

Accepted at the 13th International Conference on Learning  Representations (ICLR 2025)

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2409.05657v3) [paper-pdf](http://arxiv.org/pdf/2409.05657v3)

**Authors**: Xinhe Wang, Pingbang Hu, Junwei Deng, Jiaqi W. Ma

**Abstract**: Data attribution aims to quantify the contribution of individual training data points to the outputs of an AI model, which has been used to measure the value of training data and compensate data providers. Given the impact on financial decisions and compensation mechanisms, a critical question arises concerning the adversarial robustness of data attribution methods. However, there has been little to no systematic research addressing this issue. In this work, we aim to bridge this gap by detailing a threat model with clear assumptions about the adversary's goal and capabilities and proposing principled adversarial attack methods on data attribution. We present two methods, Shadow Attack and Outlier Attack, which generate manipulated datasets to inflate the compensation adversarially. The Shadow Attack leverages knowledge about the data distribution in the AI applications, and derives adversarial perturbations through "shadow training", a technique commonly used in membership inference attacks. In contrast, the Outlier Attack does not assume any knowledge about the data distribution and relies solely on black-box queries to the target model's predictions. It exploits an inductive bias present in many data attribution methods - outlier data points are more likely to be influential - and employs adversarial examples to generate manipulated datasets. Empirically, in image classification and text generation tasks, the Shadow Attack can inflate the data-attribution-based compensation by at least 200%, while the Outlier Attack achieves compensation inflation ranging from 185% to as much as 643%. Our implementation is ready at https://github.com/TRAIS-Lab/adversarial-attack-data-attribution.



## **12. Adversarial Attacks on AI-Generated Text Detection Models: A Token Probability-Based Approach Using Embeddings**

cs.CL

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2501.18998v2) [paper-pdf](http://arxiv.org/pdf/2501.18998v2)

**Authors**: Ahmed K. Kadhim, Lei Jiao, Rishad Shafik, Ole-Christoffer Granmo

**Abstract**: In recent years, text generation tools utilizing Artificial Intelligence (AI) have occasionally been misused across various domains, such as generating student reports or creative writings. This issue prompts plagiarism detection services to enhance their capabilities in identifying AI-generated content. Adversarial attacks are often used to test the robustness of AI-text generated detectors. This work proposes a novel textual adversarial attack on the detection models such as Fast-DetectGPT. The method employs embedding models for data perturbation, aiming at reconstructing the AI generated texts to reduce the likelihood of detection of the true origin of the texts. Specifically, we employ different embedding techniques, including the Tsetlin Machine (TM), an interpretable approach in machine learning for this purpose. By combining synonyms and embedding similarity vectors, we demonstrates the state-of-the-art reduction in detection scores against Fast-DetectGPT. Particularly, in the XSum dataset, the detection score decreased from 0.4431 to 0.2744 AUROC, and in the SQuAD dataset, it dropped from 0.5068 to 0.3532 AUROC.



## **13. Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge**

cs.CL

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07887v1) [paper-pdf](http://arxiv.org/pdf/2504.07887v1)

**Authors**: Riccardo Cantini, Alessio Orsino, Massimo Ruggiero, Domenico Talia

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence, driving advancements in machine translation, summarization, and conversational agents. However, their increasing integration into critical societal domains has raised concerns about embedded biases, which can perpetuate stereotypes and compromise fairness. These biases stem from various sources, including historical inequalities in training data, linguistic imbalances, and adversarial manipulation. Despite mitigation efforts, recent studies indicate that LLMs remain vulnerable to adversarial attacks designed to elicit biased responses. This work proposes a scalable benchmarking framework to evaluate LLM robustness against adversarial bias elicitation. Our methodology involves (i) systematically probing models with a multi-task approach targeting biases across various sociocultural dimensions, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach for automated assessment of model responses, and (iii) employing jailbreak techniques to investigate vulnerabilities in safety mechanisms. Our analysis examines prevalent biases in both small and large state-of-the-art models and their impact on model safety. Additionally, we assess the safety of domain-specific models fine-tuned for critical fields, such as medicine. Finally, we release a curated dataset of bias-related prompts, CLEAR-Bias, to facilitate systematic vulnerability benchmarking. Our findings reveal critical trade-offs between model size and safety, aiding the development of fairer and more robust future language models.



## **14. QubitHammer Attacks: Qubit Flipping Attacks in Multi-tenant Superconducting Quantum Computers**

quant-ph

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07875v1) [paper-pdf](http://arxiv.org/pdf/2504.07875v1)

**Authors**: Yizhuo Tan, Navnil Choudhury, Kanad Basu, Jakub Szefer

**Abstract**: Quantum computing is rapidly evolving its capabilities, with a corresponding surge in its deployment within cloud-based environments. Various quantum computers are accessible today via pay-as-you-go cloud computing models, offering unprecedented convenience. Due to its rapidly growing demand, quantum computers are shifting from a single-tenant to a multi-tenant model to enhance resource utilization. However, this widespread accessibility to shared multi-tenant systems also introduces potential security vulnerabilities. In this work, we present for the first time a set of novel attacks, named together as the QubitHammer attacks, which target state-of-the-art superconducting quantum computers. We show that in a multi-tenant cloud-based quantum system, an adversary with the basic capability to deploy custom pulses, similar to any standard user today, can utilize the QubitHammer attacks to significantly degrade the fidelity of victim circuits located on the same quantum computer. Upon extensive evaluation, the QubitHammer attacks achieve a very high variational distance of up to 0.938 from the expected outcome, thus demonstrating their potential to degrade victim computation. Our findings exhibit the effectiveness of these attacks across various superconducting quantum computers from a leading vendor, suggesting that QubitHammer represents a new class of security attacks. Further, the attacks are demonstrated to bypass all existing defenses proposed so far for ensuring the reliability in multi-tenant superconducting quantum computers.



## **15. MUFFLER: Secure Tor Traffic Obfuscation with Dynamic Connection Shuffling and Splitting**

cs.CR

To appear in IEEE INFOCOM 2025

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07543v1) [paper-pdf](http://arxiv.org/pdf/2504.07543v1)

**Authors**: Minjae Seo, Myoungsung You, Jaehan Kim, Taejune Park, Seungwon Shin, Jinwoo Kim

**Abstract**: Tor, a widely utilized privacy network, enables anonymous communication but is vulnerable to flow correlation attacks that deanonymize users by correlating traffic patterns from Tor's ingress and egress segments. Various defenses have been developed to mitigate these attacks; however, they have two critical limitations: (i) significant network overhead during obfuscation and (ii) a lack of dynamic obfuscation for egress segments, exposing traffic patterns to adversaries. In response, we introduce MUFFLER, a novel connection-level traffic obfuscation system designed to secure Tor egress traffic. It dynamically maps real connections to a distinct set of virtual connections between the final Tor nodes and targeted services, either public or hidden. This approach creates egress traffic patterns fundamentally different from those at ingress segments without adding intentional padding bytes or timing delays. The mapping of real and virtual connections is adjusted in real-time based on ongoing network conditions, thwarting adversaries' efforts to detect egress traffic patterns. Extensive evaluations show that MUFFLER mitigates powerful correlation attacks with a TPR of 1% at an FPR of 10^-2 while imposing only a 2.17% bandwidth overhead. Moreover, it achieves up to 27x lower latency overhead than existing solutions and seamlessly integrates with the current Tor architecture.



## **16. The Gradient Puppeteer: Adversarial Domination in Gradient Leakage Attacks through Model Poisoning**

cs.CR

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2502.04106v2) [paper-pdf](http://arxiv.org/pdf/2502.04106v2)

**Authors**: Kunlan Xiang, Haomiao Yang, Meng Hao, Shaofeng Li, Haoxin Wang, Zikang Ding, Wenbo Jiang, Tianwei Zhang

**Abstract**: In Federated Learning (FL), clients share gradients with a central server while keeping their data local. However, malicious servers could deliberately manipulate the models to reconstruct clients' data from shared gradients, posing significant privacy risks. Although such active gradient leakage attacks (AGLAs) have been widely studied, they suffer from two severe limitations: (i) coverage: no existing AGLAs can reconstruct all samples in a batch from the shared gradients; (ii) stealthiness: no existing AGLAs can evade principled checks of clients. In this paper, we address these limitations with two core contributions. First, we introduce a new theoretical analysis approach, which uniformly models AGLAs as backdoor poisoning. This analysis approach reveals that the core principle of AGLAs is to bias the gradient space to prioritize the reconstruction of a small subset of samples while sacrificing the majority, which theoretically explains the above limitations of existing AGLAs. Second, we propose Enhanced Gradient Global Vulnerability (EGGV), the first AGLA that achieves complete attack coverage while evading client-side detection. In particular, EGGV employs a gradient projector and a jointly optimized discriminator to assess gradient vulnerability, steering the gradient space toward the point most prone to data leakage. Extensive experiments show that EGGV achieves complete attack coverage and surpasses state-of-the-art (SOTA) with at least a 43% increase in reconstruction quality (PSNR) and a 45% improvement in stealthiness (D-SNR).



## **17. Code Generation with Small Language Models: A Deep Evaluation on Codeforces**

cs.SE

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2504.07343v1) [paper-pdf](http://arxiv.org/pdf/2504.07343v1)

**Authors**: Débora Souza, Rohit Gheyi, Lucas Albuquerque, Gustavo Soares, Márcio Ribeiro

**Abstract**: Large Language Models (LLMs) have demonstrated capabilities in code generation, potentially boosting developer productivity. However, their widespread adoption remains limited by high computational costs, significant energy demands, and security risks such as data leakage and adversarial attacks. As a lighter-weight alternative, Small Language Models (SLMs) offer faster inference, lower deployment overhead, and better adaptability to domain-specific tasks, making them an attractive option for real-world applications. While prior research has benchmarked LLMs on competitive programming tasks, such evaluations often focus narrowly on metrics like Elo scores or pass rates, overlooking deeper insights into model behavior, failure patterns, and problem diversity. Furthermore, the potential of SLMs to tackle complex tasks such as competitive programming remains underexplored. In this study, we benchmark five open SLMs - LLAMA 3.2 3B, GEMMA 2 9B, GEMMA 3 12B, DEEPSEEK-R1 14B, and PHI-4 14B - across 280 Codeforces problems spanning Elo ratings from 800 to 2100 and covering 36 distinct topics. All models were tasked with generating Python solutions. PHI-4 14B achieved the best performance among SLMs, with a pass@3 of 63.6%, approaching the proprietary O3-MINI-HIGH (86.8%). In addition, we evaluated PHI-4 14B on C++ and found that combining outputs from both Python and C++ increases its aggregated pass@3 to 73.6%. A qualitative analysis of PHI-4 14B's incorrect outputs revealed that some failures were due to minor implementation issues - such as handling edge cases or correcting variable initialization - rather than deeper reasoning flaws.



## **18. EveGuard: Defeating Vibration-based Side-Channel Eavesdropping with Audio Adversarial Perturbations**

cs.CR

In the 46th IEEE Symposium on Security and Privacy (IEEE S&P), May  2025

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2411.10034v2) [paper-pdf](http://arxiv.org/pdf/2411.10034v2)

**Authors**: Jung-Woo Chang, Ke Sun, David Xia, Xinyu Zhang, Farinaz Koushanfar

**Abstract**: Vibrometry-based side channels pose a significant privacy risk, exploiting sensors like mmWave radars, light sensors, and accelerometers to detect vibrations from sound sources or proximate objects, enabling speech eavesdropping. Despite various proposed defenses, these involve costly hardware solutions with inherent physical limitations. This paper presents EveGuard, a software-driven defense framework that creates adversarial audio, protecting voice privacy from side channels without compromising human perception. We leverage the distinct sensing capabilities of side channels and traditional microphones, where side channels capture vibrations and microphones record changes in air pressure, resulting in different frequency responses. EveGuard first proposes a perturbation generator model (PGM) that effectively suppresses sensor-based eavesdropping while maintaining high audio quality. Second, to enable end-to-end training of PGM, we introduce a new domain translation task called Eve-GAN for inferring an eavesdropped signal from a given audio. We further apply few-shot learning to mitigate the data collection overhead for Eve-GAN training. Our extensive experiments show that EveGuard achieves a protection rate of more than 97 percent from audio classifiers and significantly hinders eavesdropped audio reconstruction. We further validate the performance of EveGuard across three adaptive attack mechanisms. We have conducted a user study to verify the perceptual quality of our perturbed audio.



## **19. LLM Safeguard is a Double-Edged Sword: Exploiting False Positives for Denial-of-Service Attacks**

cs.CR

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2410.02916v3) [paper-pdf](http://arxiv.org/pdf/2410.02916v3)

**Authors**: Qingzhao Zhang, Ziyang Xiong, Z. Morley Mao

**Abstract**: Safety is a paramount concern for large language models (LLMs) in open deployment, motivating the development of safeguard methods that enforce ethical and responsible use through safety alignment or guardrail mechanisms. Jailbreak attacks that exploit the \emph{false negatives} of safeguard methods have emerged as a prominent research focus in the field of LLM security. However, we found that the malicious attackers could also exploit false positives of safeguards, i.e., fooling the safeguard model to block safe content mistakenly, leading to a denial-of-service (DoS) affecting LLM users. To bridge the knowledge gap of this overlooked threat, we explore multiple attack methods that include inserting a short adversarial prompt into user prompt templates and corrupting the LLM on the server by poisoned fine-tuning. In both ways, the attack triggers safeguard rejections of user requests from the client. Our evaluation demonstrates the severity of this threat across multiple scenarios. For instance, in the scenario of white-box adversarial prompt injection, the attacker can use our optimization process to automatically generate seemingly safe adversarial prompts, approximately only 30 characters long, that universally block over 97% of user requests on Llama Guard 3. These findings reveal a new dimension in LLM safeguard evaluation -- adversarial robustness to false positives.



## **20. Towards Communication-Efficient Adversarial Federated Learning for Robust Edge Intelligence**

cs.CV

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2501.15257v2) [paper-pdf](http://arxiv.org/pdf/2501.15257v2)

**Authors**: Yu Qiao, Apurba Adhikary, Huy Q. Le, Eui-Nam Huh, Zhu Han, Choong Seon Hong

**Abstract**: Federated learning (FL) has gained significant attention for enabling decentralized training on edge networks without exposing raw data. However, FL models remain susceptible to adversarial attacks and performance degradation in non-IID data settings, thus posing challenges to both robustness and accuracy. This paper aims to achieve communication-efficient adversarial federated learning (AFL) by leveraging a pre-trained model to enhance both robustness and accuracy under adversarial attacks and non-IID challenges in AFL. By leveraging the knowledge from a pre-trained model for both clean and adversarial images, we propose a pre-trained model-guided adversarial federated learning (PM-AFL) framework. This framework integrates vanilla and adversarial mixture knowledge distillation to effectively balance accuracy and robustness while promoting local models to learn from diverse data. Specifically, for clean accuracy, we adopt a dual distillation strategy where the class probabilities of randomly paired images, and their blended versions are aligned between the teacher model and the local models. For adversarial robustness, we employ a similar distillation approach but replace clean samples on the local side with adversarial examples. Moreover, by considering the bias between local and global models, we also incorporate a consistency regularization term to ensure that local adversarial predictions stay aligned with their corresponding global clean ones. These strategies collectively enable local models to absorb diverse knowledge from the teacher model while maintaining close alignment with the global model, thereby mitigating overfitting to local optima and enhancing the generalization of the global model. Experiments demonstrate that the PM-AFL-based framework not only significantly outperforms other methods but also maintains communication efficiency.



## **21. Exploiting Meta-Learning-based Poisoning Attacks for Graph Link Prediction**

cs.LG

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.06492v1) [paper-pdf](http://arxiv.org/pdf/2504.06492v1)

**Authors**: Mingchen Li, Di Zhuang, Keyu Chen, Dumindu Samaraweera, Morris Chang

**Abstract**: Link prediction in graph data utilizes various algorithms and machine learning/deep learning models to predict potential relationships between graph nodes. This technique has found widespread use in numerous real-world applications, including recommendation systems, community networks, and biological structures. However, recent research has highlighted the vulnerability of link prediction models to adversarial attacks, such as poisoning and evasion attacks. Addressing the vulnerability of these models is crucial to ensure stable and robust performance in link prediction applications. While many works have focused on enhancing the robustness of the Graph Convolution Network (GCN) model, the Variational Graph Auto-Encoder (VGAE), a sophisticated model for link prediction, has not been thoroughly investigated in the context of graph adversarial attacks. To bridge this gap, this article proposes an unweighted graph poisoning attack approach using meta-learning techniques to undermine VGAE's link prediction performance. We conducted comprehensive experiments on diverse datasets to evaluate the proposed method and its parameters, comparing it with existing approaches in similar settings. Our results demonstrate that our approach significantly diminishes link prediction performance and outperforms other state-of-the-art methods.



## **22. Mitigating Adversarial Effects of False Data Injection Attacks in Power Grid**

cs.CR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2301.12487v3) [paper-pdf](http://arxiv.org/pdf/2301.12487v3)

**Authors**: Farhin Farhad Riya, Shahinul Hoque, Yingyuan Yang, Jiangnan Li, Jinyuan Stella Sun, Hairong Qi

**Abstract**: Deep Neural Networks have proven to be highly accurate at a variety of tasks in recent years. The benefits of Deep Neural Networks have also been embraced in power grids to detect False Data Injection Attacks (FDIA) while conducting critical tasks like state estimation. However, the vulnerabilities of DNNs along with the distinct infrastructure of the cyber-physical-system (CPS) can favor the attackers to bypass the detection mechanism. Moreover, the divergent nature of CPS engenders limitations to the conventional defense mechanisms for False Data Injection Attacks. In this paper, we propose a DNN framework with an additional layer that utilizes randomization to mitigate the adversarial effect by padding the inputs. The primary advantage of our method is when deployed to a DNN model it has a trivial impact on the model's performance even with larger padding sizes. We demonstrate the favorable outcome of the framework through simulation using the IEEE 14-bus, 30-bus, 118-bus, and 300-bus systems. Furthermore to justify the framework we select attack techniques that generate subtle adversarial examples that can bypass the detection mechanism effortlessly.



## **23. Quantum Covert Communication under Extreme Adversarial Control**

quant-ph

Submitted to Quantum

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.06359v1) [paper-pdf](http://arxiv.org/pdf/2504.06359v1)

**Authors**: Trey Li

**Abstract**: Secure quantum communication traditionally assumes that the adversary controls only the public channel. We consider a more powerful adversary who can demand private information of users. This type of adversary has been studied in public key cryptography in recent years, initiated by Persiano, Phan, and Yung at Eurocrypt 2022. We introduce a similar attacker to quantum communication, referring to it as the controller. The controller is a quantum computer that controls the entire communication infrastructure, including both classical and quantum channels. It can even ban classical public key cryptography and post-quantum public key cryptography, leaving only quantum cryptography and post-quantum symmetric key cryptography as the remaining options. We demonstrate how such a controller can control quantum communication and how users can achieve covert communication under its control.



## **24. Towards Calibration Enhanced Network by Inverse Adversarial Attack**

cs.CV

11 pages

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.06358v1) [paper-pdf](http://arxiv.org/pdf/2504.06358v1)

**Authors**: Yupeng Cheng, Zi Pong Lim, Sarthak Ketanbhai Modi, Yon Shin Teo, Yushi Cao, Shang-Wei Lin

**Abstract**: Test automation has become increasingly important as the complexity of both design and content in Human Machine Interface (HMI) software continues to grow. Current standard practice uses Optical Character Recognition (OCR) techniques to automatically extract textual information from HMI screens for validation. At present, one of the key challenges faced during the automation of HMI screen validation is the noise handling for the OCR models. In this paper, we propose to utilize adversarial training techniques to enhance OCR models in HMI testing scenarios. More specifically, we design a new adversarial attack objective for OCR models to discover the decision boundaries in the context of HMI testing. We then adopt adversarial training to optimize the decision boundaries towards a more robust and accurate OCR model. In addition, we also built an HMI screen dataset based on real-world requirements and applied multiple types of perturbation onto the clean HMI dataset to provide a more complete coverage for the potential scenarios. We conduct experiments to demonstrate how using adversarial training techniques yields more robust OCR models against various kinds of noises, while still maintaining high OCR model accuracy. Further experiments even demonstrate that the adversarial training models exhibit a certain degree of robustness against perturbations from other patterns.



## **25. Exploring Adversarial Obstacle Attacks in Search-based Path Planning for Autonomous Mobile Robots**

cs.RO

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.06154v1) [paper-pdf](http://arxiv.org/pdf/2504.06154v1)

**Authors**: Adrian Szvoren, Jianwei Liu, Dimitrios Kanoulas, Nilufer Tuptuk

**Abstract**: Path planning algorithms, such as the search-based A*, are a critical component of autonomous mobile robotics, enabling robots to navigate from a starting point to a destination efficiently and safely. We investigated the resilience of the A* algorithm in the face of potential adversarial interventions known as obstacle attacks. The adversary's goal is to delay the robot's timely arrival at its destination by introducing obstacles along its original path.   We developed malicious software to execute the attacks and conducted experiments to assess their impact, both in simulation using TurtleBot in Gazebo and in real-world deployment with the Unitree Go1 robot. In simulation, the attacks resulted in an average delay of 36\%, with the most significant delays occurring in scenarios where the robot was forced to take substantially longer alternative paths. In real-world experiments, the delays were even more pronounced, with all attacks successfully rerouting the robot and causing measurable disruptions. These results highlight that the algorithm's robustness is not solely an attribute of its design but is significantly influenced by the operational environment. For example, in constrained environments like tunnels, the delays were maximized due to the limited availability of alternative routes.



## **26. Frequency maps reveal the correlation between Adversarial Attacks and Implicit Bias**

cs.LG

Accepted at IJCNN 2025

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2305.15203v3) [paper-pdf](http://arxiv.org/pdf/2305.15203v3)

**Authors**: Lorenzo Basile, Nikos Karantzas, Alberto d'Onofrio, Luca Manzoni, Luca Bortolussi, Alex Rodriguez, Fabio Anselmi

**Abstract**: Despite their impressive performance in classification tasks, neural networks are known to be vulnerable to adversarial attacks, subtle perturbations of the input data designed to deceive the model. In this work, we investigate the correlation between these perturbations and the implicit bias of neural networks trained with gradient-based algorithms. To this end, we analyse a representation of the network's implicit bias through the lens of the Fourier transform. Specifically, we identify unique fingerprints of implicit bias and adversarial attacks by calculating the minimal, essential frequencies needed for accurate classification of each image, as well as the frequencies that drive misclassification in its adversarially perturbed counterpart. This approach enables us to uncover and analyse the correlation between these essential frequencies, providing a precise map of how the network's biases align or contrast with the frequency components exploited by adversarial attacks. To this end, among other methods, we use a newly introduced technique capable of detecting nonlinear correlations between high-dimensional datasets. Our results provide empirical evidence that the network bias in Fourier space and the target frequencies of adversarial attacks are highly correlated and suggest new potential strategies for adversarial defence.



## **27. Mind the Trojan Horse: Image Prompt Adapter Enabling Scalable and Deceptive Jailbreaking**

cs.CV

Accepted by CVPR2025 as Highlight

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05838v1) [paper-pdf](http://arxiv.org/pdf/2504.05838v1)

**Authors**: Junxi Chen, Junhao Dong, Xiaohua Xie

**Abstract**: Recently, the Image Prompt Adapter (IP-Adapter) has been increasingly integrated into text-to-image diffusion models (T2I-DMs) to improve controllability. However, in this paper, we reveal that T2I-DMs equipped with the IP-Adapter (T2I-IP-DMs) enable a new jailbreak attack named the hijacking attack. We demonstrate that, by uploading imperceptible image-space adversarial examples (AEs), the adversary can hijack massive benign users to jailbreak an Image Generation Service (IGS) driven by T2I-IP-DMs and mislead the public to discredit the service provider. Worse still, the IP-Adapter's dependency on open-source image encoders reduces the knowledge required to craft AEs. Extensive experiments verify the technical feasibility of the hijacking attack. In light of the revealed threat, we investigate several existing defenses and explore combining the IP-Adapter with adversarially trained models to overcome existing defenses' limitations. Our code is available at https://github.com/fhdnskfbeuv/attackIPA.



## **28. StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization**

cs.IR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05804v1) [paper-pdf](http://arxiv.org/pdf/2504.05804v1)

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present StealthRank, a novel adversarial ranking attack that manipulates LLM-driven product recommendation systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within product descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target products while avoiding explicit manipulation traces that can be easily detected. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven recommendation systems.



## **29. Automated Trustworthiness Oracle Generation for Machine Learning Text Classifiers**

cs.SE

Accepted to FSE 2025

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2410.22663v2) [paper-pdf](http://arxiv.org/pdf/2410.22663v2)

**Authors**: Lam Nguyen Tung, Steven Cho, Xiaoning Du, Neelofar Neelofar, Valerio Terragni, Stefano Ruberto, Aldeida Aleti

**Abstract**: Machine learning (ML) for text classification has been widely used in various domains. These applications can significantly impact ethics, economics, and human behavior, raising serious concerns about trusting ML decisions. Studies indicate that conventional metrics are insufficient to build human trust in ML models. These models often learn spurious correlations and predict based on them. In the real world, their performance can deteriorate significantly. To avoid this, a common practice is to test whether predictions are reasonable based on valid patterns in the data. Along with this, a challenge known as the trustworthiness oracle problem has been introduced. Due to the lack of automated trustworthiness oracles, the assessment requires manual validation of the decision process disclosed by explanation methods. However, this is time-consuming, error-prone, and unscalable.   We propose TOKI, the first automated trustworthiness oracle generation method for text classifiers. TOKI automatically checks whether the words contributing the most to a prediction are semantically related to the predicted class. Specifically, we leverage ML explanations to extract the decision-contributing words and measure their semantic relatedness with the class based on word embeddings. We also introduce a novel adversarial attack method that targets trustworthiness vulnerabilities identified by TOKI. To evaluate their alignment with human judgement, experiments are conducted. We compare TOKI with a naive baseline based solely on model confidence and TOKI-guided adversarial attack method with A2T, a SOTA adversarial attack method. Results show that relying on prediction uncertainty cannot effectively distinguish between trustworthy and untrustworthy predictions, TOKI achieves 142% higher accuracy than the naive baseline, and TOKI-guided attack method is more effective with fewer perturbations than A2T.



## **30. Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing**

eess.AS

This manuscript has been submitted for peer review

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05657v1) [paper-pdf](http://arxiv.org/pdf/2504.05657v1)

**Authors**: Tianchi Liu, Duc-Tuan Truong, Rohan Kumar Das, Kong Aik Lee, Haizhou Li

**Abstract**: Speech foundation models have significantly advanced various speech-related tasks by providing exceptional representation capabilities. However, their high-dimensional output features often create a mismatch with downstream task models, which typically require lower-dimensional inputs. A common solution is to apply a dimensionality reduction (DR) layer, but this approach increases parameter overhead, computational costs, and risks losing valuable information. To address these issues, we propose Nested Res2Net (Nes2Net), a lightweight back-end architecture designed to directly process high-dimensional features without DR layers. The nested structure enhances multi-scale feature extraction, improves feature interaction, and preserves high-dimensional information. We first validate Nes2Net on CtrSVDD, a singing voice deepfake detection dataset, and report a 22% performance improvement and an 87% back-end computational cost reduction over the state-of-the-art baseline. Additionally, extensive testing across four diverse datasets: ASVspoof 2021, ASVspoof 5, PartialSpoof, and In-the-Wild, covering fully spoofed speech, adversarial attacks, partial spoofing, and real-world scenarios, consistently highlights Nes2Net's superior robustness and generalization capabilities. The code package and pre-trained models are available at https://github.com/Liu-Tianchi/Nes2Net.



## **31. Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking**

cs.CR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05652v1) [paper-pdf](http://arxiv.org/pdf/2504.05652v1)

**Authors**: Yu-Hang Wu, Yu-Jie Xiong, Jie-Zhang

**Abstract**: Large Language Models (LLMs) have become increasingly integral to a wide range of applications. However, they still remain the threat of jailbreak attacks, where attackers manipulate designed prompts to make the models elicit malicious outputs. Analyzing jailbreak methods can help us delve into the weakness of LLMs and improve it. In this paper, We reveal a vulnerability in large language models (LLMs), which we term Defense Threshold Decay (DTD), by analyzing the attention weights of the model's output on input and subsequent output on prior output: as the model generates substantial benign content, its attention weights shift from the input to prior output, making it more susceptible to jailbreak attacks. To demonstrate the exploitability of DTD, we propose a novel jailbreak attack method, Sugar-Coated Poison (SCP), which induces the model to generate substantial benign content through benign input and adversarial reasoning, subsequently producing malicious content. To mitigate such attacks, we introduce a simple yet effective defense strategy, POSD, which significantly reduces jailbreak success rates while preserving the model's generalization capabilities.



## **32. SceneTAP: Scene-Coherent Typographic Adversarial Planner against Vision-Language Models in Real-World Environments**

cs.CV

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2412.00114v2) [paper-pdf](http://arxiv.org/pdf/2412.00114v2)

**Authors**: Yue Cao, Yun Xing, Jie Zhang, Di Lin, Tianwei Zhang, Ivor Tsang, Yang Liu, Qing Guo

**Abstract**: Large vision-language models (LVLMs) have shown remarkable capabilities in interpreting visual content. While existing works demonstrate these models' vulnerability to deliberately placed adversarial texts, such texts are often easily identifiable as anomalous. In this paper, we present the first approach to generate scene-coherent typographic adversarial attacks that mislead advanced LVLMs while maintaining visual naturalness through the capability of the LLM-based agent. Our approach addresses three critical questions: what adversarial text to generate, where to place it within the scene, and how to integrate it seamlessly. We propose a training-free, multi-modal LLM-driven scene-coherent typographic adversarial planning (SceneTAP) that employs a three-stage process: scene understanding, adversarial planning, and seamless integration. The SceneTAP utilizes chain-of-thought reasoning to comprehend the scene, formulate effective adversarial text, strategically plan its placement, and provide detailed instructions for natural integration within the image. This is followed by a scene-coherent TextDiffuser that executes the attack using a local diffusion mechanism. We extend our method to real-world scenarios by printing and placing generated patches in physical environments, demonstrating its practical implications. Extensive experiments show that our scene-coherent adversarial text successfully misleads state-of-the-art LVLMs, including ChatGPT-4o, even after capturing new images of physical setups. Our evaluations demonstrate a significant increase in attack success rates while maintaining visual naturalness and contextual appropriateness. This work highlights vulnerabilities in current vision-language models to sophisticated, scene-coherent adversarial attacks and provides insights into potential defense mechanisms.



## **33. ShadowCoT: Cognitive Hijacking for Stealthy Reasoning Backdoors in LLMs**

cs.CR

Zhao et al., 16 pages, 2025, uploaded by Hanzhou Wu, Shanghai  University

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05605v1) [paper-pdf](http://arxiv.org/pdf/2504.05605v1)

**Authors**: Gejian Zhao, Hanzhou Wu, Xinpeng Zhang, Athanasios V. Vasilakos

**Abstract**: Chain-of-Thought (CoT) enhances an LLM's ability to perform complex reasoning tasks, but it also introduces new security issues. In this work, we present ShadowCoT, a novel backdoor attack framework that targets the internal reasoning mechanism of LLMs. Unlike prior token-level or prompt-based attacks, ShadowCoT directly manipulates the model's cognitive reasoning path, enabling it to hijack multi-step reasoning chains and produce logically coherent but adversarial outcomes. By conditioning on internal reasoning states, ShadowCoT learns to recognize and selectively disrupt key reasoning steps, effectively mounting a self-reflective cognitive attack within the target model. Our approach introduces a lightweight yet effective multi-stage injection pipeline, which selectively rewires attention pathways and perturbs intermediate representations with minimal parameter overhead (only 0.15% updated). ShadowCoT further leverages reinforcement learning and reasoning chain pollution (RCP) to autonomously synthesize stealthy adversarial CoTs that remain undetectable to advanced defenses. Extensive experiments across diverse reasoning benchmarks and LLMs show that ShadowCoT consistently achieves high Attack Success Rate (94.4%) and Hijacking Success Rate (88.4%) while preserving benign performance. These results reveal an emergent class of cognition-level threats and highlight the urgent need for defenses beyond shallow surface-level consistency.



## **34. Impact Assessment of Cyberattacks in Inverter-Based Microgrids**

eess.SY

IEEE Workshop on the Electronic Grid (eGrid 2025)

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05592v1) [paper-pdf](http://arxiv.org/pdf/2504.05592v1)

**Authors**: Kerd Topallaj, Colin McKerrell, Suraj Ramanathan, Ioannis Zografopoulos

**Abstract**: In recent years, the evolution of modern power grids has been driven by the growing integration of remotely controlled grid assets. Although Distributed Energy Resources (DERs) and Inverter-Based Resources (IBR) enhance operational efficiency, they also introduce cybersecurity risks. The remote accessibility of such critical grid components creates entry points for attacks that adversaries could exploit, posing threats to the stability of the system. To evaluate the resilience of energy systems under such threats, this study employs real-time simulation and a modified version of the IEEE 39-bus system that incorporates a Microgrid (MG) with solar-based IBR. The study assesses the impact of remote attacks impacting the MG stability under different levels of IBR penetrations through Hardware-in-the-Loop (HIL) simulations. Namely, we analyze voltage, current, and frequency profiles before, during, and after cyberattack-induced disruptions. The results demonstrate that real-time HIL testing is a practical approach to uncover potential risks and develop robust mitigation strategies for resilient MG operations.



## **35. Secure Diagnostics: Adversarial Robustness Meets Clinical Interpretability**

cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05483v1) [paper-pdf](http://arxiv.org/pdf/2504.05483v1)

**Authors**: Mohammad Hossein Najafi, Mohammad Morsali, Mohammadreza Pashanejad, Saman Soleimani Roudi, Mohammad Norouzi, Saeed Bagheri Shouraki

**Abstract**: Deep neural networks for medical image classification often fail to generalize consistently in clinical practice due to violations of the i.i.d. assumption and opaque decision-making. This paper examines interpretability in deep neural networks fine-tuned for fracture detection by evaluating model performance against adversarial attack and comparing interpretability methods to fracture regions annotated by an orthopedic surgeon. Our findings prove that robust models yield explanations more aligned with clinically meaningful areas, indicating that robustness encourages anatomically relevant feature prioritization. We emphasize the value of interpretability for facilitating human-AI collaboration, in which models serve as assistants under a human-in-the-loop paradigm: clinically plausible explanations foster trust, enable error correction, and discourage reliance on AI for high-stakes decisions. This paper investigates robustness and interpretability as complementary benchmarks for bridging the gap between benchmark performance and safe, actionable clinical deployment.



## **36. Adversarial KA**

cs.LG

8 pages, 3 figures

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05255v1) [paper-pdf](http://arxiv.org/pdf/2504.05255v1)

**Authors**: Sviatoslav Dzhenzher, Michael H. Freedman

**Abstract**: Regarding the representation theorem of Kolmogorov and Arnold (KA) as an algorithm for representing or {\guillemotleft}expressing{\guillemotright} functions, we test its robustness by analyzing its ability to withstand adversarial attacks. We find KA to be robust to countable collections of continuous adversaries, but unearth a question about the equi-continuity of the outer functions that, so far, obstructs taking limits and defeating continuous groups of adversaries. This question on the regularity of the outer functions is relevant to the debate over the applicability of KA to the general theory of NNs.



## **37. Security Risks in Vision-Based Beam Prediction: From Spatial Proxy Attacks to Feature Refinement**

cs.NI

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05222v1) [paper-pdf](http://arxiv.org/pdf/2504.05222v1)

**Authors**: Avi Deb Raha, Kitae Kim, Mrityunjoy Gain, Apurba Adhikary, Zhu Han, Eui-Nam Huh, Choong Seon Hong

**Abstract**: The rapid evolution towards the sixth-generation (6G) networks demands advanced beamforming techniques to address challenges in dynamic, high-mobility scenarios, such as vehicular communications. Vision-based beam prediction utilizing RGB camera images emerges as a promising solution for accurate and responsive beam selection. However, reliance on visual data introduces unique vulnerabilities, particularly susceptibility to adversarial attacks, thus potentially compromising beam accuracy and overall network reliability. In this paper, we conduct the first systematic exploration of adversarial threats specifically targeting vision-based mmWave beam selection systems. Traditional white-box attacks are impractical in this context because ground-truth beam indices are inaccessible and spatial dynamics are complex. To address this, we propose a novel black-box adversarial attack strategy, termed Spatial Proxy Attack (SPA), which leverages spatial correlations between user positions and beam indices to craft effective perturbations without requiring access to model parameters or labels. To counteract these adversarial vulnerabilities, we formulate an optimization framework aimed at simultaneously enhancing beam selection accuracy under clean conditions and robustness against adversarial perturbations. We introduce a hybrid deep learning architecture integrated with a dedicated Feature Refinement Module (FRM), designed to systematically filter irrelevant, noisy and adversarially perturbed visual features. Evaluations using standard backbone models such as ResNet-50 and MobileNetV2 demonstrate that our proposed method significantly improves performance, achieving up to an +21.07\% gain in Top-K accuracy under clean conditions and a 41.31\% increase in Top-1 adversarial robustness compared to different baseline models.



## **38. DiffPatch: Generating Customizable Adversarial Patches using Diffusion Models**

cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2412.01440v3) [paper-pdf](http://arxiv.org/pdf/2412.01440v3)

**Authors**: Zhixiang Wang, Xiaosen Wang, Bo Wang, Siheng Chen, Zhibo Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Physical adversarial patches printed on clothing can enable individuals to evade person detectors, but most existing methods prioritize attack effectiveness over stealthiness, resulting in aesthetically unpleasing patches. While generative adversarial networks and diffusion models can produce more natural-looking patches, they often fail to balance stealthiness with attack effectiveness and lack flexibility for user customization. To address these limitations, we propose DiffPatch, a novel diffusion-based framework for generating customizable and naturalistic adversarial patches. Our approach allows users to start from a reference image (rather than random noise) and incorporates masks to create patches of various shapes, not limited to squares. To preserve the original semantics during the diffusion process, we employ Null-text inversion to map random noise samples to a single input image and generate patches through Incomplete Diffusion Optimization (IDO). Our method achieves attack performance comparable to state-of-the-art non-naturalistic patches while maintaining a natural appearance. Using DiffPatch, we construct AdvT-shirt-1K, the first physical adversarial T-shirt dataset comprising over a thousand images captured in diverse scenarios. AdvT-shirt-1K can serve as a useful dataset for training or testing future defense methods.



## **39. Adversarial Robustness for Deep Learning-based Wildfire Prediction Models**

cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2412.20006v3) [paper-pdf](http://arxiv.org/pdf/2412.20006v3)

**Authors**: Ryo Ide, Lei Yang

**Abstract**: Rapidly growing wildfires have recently devastated societal assets, exposing a critical need for early warning systems to expedite relief efforts. Smoke detection using camera-based Deep Neural Networks (DNNs) offers a promising solution for wildfire prediction. However, the rarity of smoke across time and space limits training data, raising model overfitting and bias concerns. Current DNNs, primarily Convolutional Neural Networks (CNNs) and transformers, complicate robustness evaluation due to architectural differences. To address these challenges, we introduce WARP (Wildfire Adversarial Robustness Procedure), the first model-agnostic framework for evaluating wildfire detection models' adversarial robustness. WARP addresses inherent limitations in data diversity by generating adversarial examples through image-global and -local perturbations. Global and local attacks superimpose Gaussian noise and PNG patches onto image inputs, respectively; this suits both CNNs and transformers while generating realistic adversarial scenarios. Using WARP, we assessed real-time CNNs and Transformers, uncovering key vulnerabilities. At times, transformers exhibited over 70% precision degradation under global attacks, while both models generally struggled to differentiate cloud-like PNG patches from real smoke during local attacks. To enhance model robustness, we proposed four wildfire-oriented data augmentation techniques based on WARP's methodology and results, which diversify smoke image data and improve model precision and robustness. These advancements represent a substantial step toward developing a reliable early wildfire warning system, which may be our first safeguard against wildfire destruction.



## **40. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

cs.CL

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05050v1) [paper-pdf](http://arxiv.org/pdf/2504.05050v1)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.



## **41. A Domain-Based Taxonomy of Jailbreak Vulnerabilities in Large Language Models**

cs.CL

21 pages, 5 figures

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04976v1) [paper-pdf](http://arxiv.org/pdf/2504.04976v1)

**Authors**: Carlos Peláez-González, Andrés Herrera-Poyatos, Cristina Zuheros, David Herrera-Poyatos, Virilo Tejedor, Francisco Herrera

**Abstract**: The study of large language models (LLMs) is a key area in open-world machine learning. Although LLMs demonstrate remarkable natural language processing capabilities, they also face several challenges, including consistency issues, hallucinations, and jailbreak vulnerabilities. Jailbreaking refers to the crafting of prompts that bypass alignment safeguards, leading to unsafe outputs that compromise the integrity of LLMs. This work specifically focuses on the challenge of jailbreak vulnerabilities and introduces a novel taxonomy of jailbreak attacks grounded in the training domains of LLMs. It characterizes alignment failures through generalization, objectives, and robustness gaps. Our primary contribution is a perspective on jailbreak, framed through the different linguistic domains that emerge during LLM training and alignment. This viewpoint highlights the limitations of existing approaches and enables us to classify jailbreak attacks on the basis of the underlying model deficiencies they exploit. Unlike conventional classifications that categorize attacks based on prompt construction methods (e.g., prompt templating), our approach provides a deeper understanding of LLM behavior. We introduce a taxonomy with four categories -- mismatched generalization, competing objectives, adversarial robustness, and mixed attacks -- offering insights into the fundamental nature of jailbreak vulnerabilities. Finally, we present key lessons derived from this taxonomic study.



## **42. Graph of Effort: Quantifying Risk of AI Usage for Vulnerability Assessment**

cs.CR

8 pages, 4 figures

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2503.16392v2) [paper-pdf](http://arxiv.org/pdf/2503.16392v2)

**Authors**: Anket Mehra, Andreas Aßmuth, Malte Prieß

**Abstract**: With AI-based software becoming widely available, the risk of exploiting its capabilities, such as high automation and complex pattern recognition, could significantly increase. An AI used offensively to attack non-AI assets is referred to as offensive AI.   Current research explores how offensive AI can be utilized and how its usage can be classified. Additionally, methods for threat modeling are being developed for AI-based assets within organizations. However, there are gaps that need to be addressed. Firstly, there is a need to quantify the factors contributing to the AI threat. Secondly, there is a requirement to create threat models that analyze the risk of being attacked by AI for vulnerability assessment across all assets of an organization. This is particularly crucial and challenging in cloud environments, where sophisticated infrastructure and access control landscapes are prevalent. The ability to quantify and further analyze the threat posed by offensive AI enables analysts to rank vulnerabilities and prioritize the implementation of proactive countermeasures.   To address these gaps, this paper introduces the Graph of Effort, an intuitive, flexible, and effective threat modeling method for analyzing the effort required to use offensive AI for vulnerability exploitation by an adversary. While the threat model is functional and provides valuable support, its design choices need further empirical validation in future work.



## **43. Don't Lag, RAG: Training-Free Adversarial Detection Using RAG**

cs.AI

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04858v1) [paper-pdf](http://arxiv.org/pdf/2504.04858v1)

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types, all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs, including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95 percent classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98 percent, but remains closed-source. Experimental results demonstrate VRAG's effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks.



## **44. Latent Feature and Attention Dual Erasure Attack against Multi-View Diffusion Models for 3D Assets Protection**

cs.CV

This paper has been accepted by ICME 2025

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2408.11408v2) [paper-pdf](http://arxiv.org/pdf/2408.11408v2)

**Authors**: Jingwei Sun, Xuchong Zhang, Changfeng Sun, Qicheng Bai, Hongbin Sun

**Abstract**: Multi-View Diffusion Models (MVDMs) enable remarkable improvements in the field of 3D geometric reconstruction, but the issue regarding intellectual property has received increasing attention due to unauthorized imitation. Recently, some works have utilized adversarial attacks to protect copyright. However, all these works focus on single-image generation tasks which only need to consider the inner feature of images. Previous methods are inefficient in attacking MVDMs because they lack the consideration of disrupting the geometric and visual consistency among the generated multi-view images. This paper is the first to address the intellectual property infringement issue arising from MVDMs. Accordingly, we propose a novel latent feature and attention dual erasure attack to disrupt the distribution of latent feature and the consistency across the generated images from multi-view and multi-domain simultaneously. The experiments conducted on SOTA MVDMs indicate that our approach achieves superior performances in terms of attack effectiveness, transferability, and robustness against defense methods. Therefore, this paper provides an efficient solution to protect 3D assets from MVDMs-based 3D geometry reconstruction.



## **45. Towards Benchmarking and Assessing the Safety and Robustness of Autonomous Driving on Safety-critical Scenarios**

cs.RO

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2503.23708v2) [paper-pdf](http://arxiv.org/pdf/2503.23708v2)

**Authors**: Jingzheng Li, Xianglong Liu, Shikui Wei, Zhijun Chen, Bing Li, Qing Guo, Xianqi Yang, Yanjun Pu, Jiakai Wang

**Abstract**: Autonomous driving has made significant progress in both academia and industry, including performance improvements in perception task and the development of end-to-end autonomous driving systems. However, the safety and robustness assessment of autonomous driving has not received sufficient attention. Current evaluations of autonomous driving are typically conducted in natural driving scenarios. However, many accidents often occur in edge cases, also known as safety-critical scenarios. These safety-critical scenarios are difficult to collect, and there is currently no clear definition of what constitutes a safety-critical scenario. In this work, we explore the safety and robustness of autonomous driving in safety-critical scenarios. First, we provide a definition of safety-critical scenarios, including static traffic scenarios such as adversarial attack scenarios and natural distribution shifts, as well as dynamic traffic scenarios such as accident scenarios. Then, we develop an autonomous driving safety testing platform to comprehensively evaluate autonomous driving systems, encompassing not only the assessment of perception modules but also system-level evaluations. Our work systematically constructs a safety verification process for autonomous driving, providing technical support for the industry to establish standardized test framework and reduce risks in real-world road deployment.



## **46. SINCon: Mitigate LLM-Generated Malicious Message Injection Attack for Rumor Detection**

cs.CR

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.07135v1) [paper-pdf](http://arxiv.org/pdf/2504.07135v1)

**Authors**: Mingqing Zhang, Qiang Liu, Xiang Tao, Shu Wu, Liang Wang

**Abstract**: In the era of rapidly evolving large language models (LLMs), state-of-the-art rumor detection systems, particularly those based on Message Propagation Trees (MPTs), which represent a conversation tree with the post as its root and the replies as its descendants, are facing increasing threats from adversarial attacks that leverage LLMs to generate and inject malicious messages. Existing methods are based on the assumption that different nodes exhibit varying degrees of influence on predictions. They define nodes with high predictive influence as important nodes and target them for attacks. If the model treats nodes' predictive influence more uniformly, attackers will find it harder to target high predictive influence nodes. In this paper, we propose Similarizing the predictive Influence of Nodes with Contrastive Learning (SINCon), a defense mechanism that encourages the model to learn graph representations where nodes with varying importance have a more uniform influence on predictions. Extensive experiments on the Twitter and Weibo datasets demonstrate that SINCon not only preserves high classification accuracy on clean data but also significantly enhances resistance against LLM-driven message injection attacks.



## **47. Two is Better than One: Efficient Ensemble Defense for Robust and Compact Models**

cs.CV

Accepted to CVPR2025

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04747v1) [paper-pdf](http://arxiv.org/pdf/2504.04747v1)

**Authors**: Yoojin Jung, Byung Cheol Song

**Abstract**: Deep learning-based computer vision systems adopt complex and large architectures to improve performance, yet they face challenges in deployment on resource-constrained mobile and edge devices. To address this issue, model compression techniques such as pruning, quantization, and matrix factorization have been proposed; however, these compressed models are often highly vulnerable to adversarial attacks. We introduce the \textbf{Efficient Ensemble Defense (EED)} technique, which diversifies the compression of a single base model based on different pruning importance scores and enhances ensemble diversity to achieve high adversarial robustness and resource efficiency. EED dynamically determines the number of necessary sub-models during the inference stage, minimizing unnecessary computations while maintaining high robustness. On the CIFAR-10 and SVHN datasets, EED demonstrated state-of-the-art robustness performance compared to existing adversarial pruning techniques, along with an inference speed improvement of up to 1.86 times. This proves that EED is a powerful defense solution in resource-constrained environments.



## **48. A Survey and Evaluation of Adversarial Attacks for Object Detection**

cs.CV

Accepted for publication in the IEEE Transactions on Neural Networks  and Learning Systems (TNNLS)

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2408.01934v4) [paper-pdf](http://arxiv.org/pdf/2408.01934v4)

**Authors**: Khoi Nguyen Tiet Nguyen, Wenyu Zhang, Kangkang Lu, Yuhuan Wu, Xingjian Zheng, Hui Li Tan, Liangli Zhen

**Abstract**: Deep learning models achieve remarkable accuracy in computer vision tasks, yet remain vulnerable to adversarial examples--carefully crafted perturbations to input images that can deceive these models into making confident but incorrect predictions. This vulnerability pose significant risks in high-stakes applications such as autonomous vehicles, security surveillance, and safety-critical inspection systems. While the existing literature extensively covers adversarial attacks in image classification, comprehensive analyses of such attacks on object detection systems remain limited. This paper presents a novel taxonomic framework for categorizing adversarial attacks specific to object detection architectures, synthesizes existing robustness metrics, and provides a comprehensive empirical evaluation of state-of-the-art attack methodologies on popular object detection models, including both traditional detectors and modern detectors with vision-language pretraining. Through rigorous analysis of open-source attack implementations and their effectiveness across diverse detection architectures, we derive key insights into attack characteristics. Furthermore, we delineate critical research gaps and emerging challenges to guide future investigations in securing object detection systems against adversarial threats. Our findings establish a foundation for developing more robust detection models while highlighting the urgent need for standardized evaluation protocols in this rapidly evolving domain.



## **49. On the Robustness of GUI Grounding Models Against Image Attacks**

cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04716v1) [paper-pdf](http://arxiv.org/pdf/2504.04716v1)

**Authors**: Haoren Zhao, Tianyi Chen, Zhen Wang

**Abstract**: Graphical User Interface (GUI) grounding models are crucial for enabling intelligent agents to understand and interact with complex visual interfaces. However, these models face significant robustness challenges in real-world scenarios due to natural noise and adversarial perturbations, and their robustness remains underexplored. In this study, we systematically evaluate the robustness of state-of-the-art GUI grounding models, such as UGround, under three conditions: natural noise, untargeted adversarial attacks, and targeted adversarial attacks. Our experiments, which were conducted across a wide range of GUI environments, including mobile, desktop, and web interfaces, have clearly demonstrated that GUI grounding models exhibit a high degree of sensitivity to adversarial perturbations and low-resolution conditions. These findings provide valuable insights into the vulnerabilities of GUI grounding models and establish a strong benchmark for future research aimed at enhancing their robustness in practical applications. Our code is available at https://github.com/ZZZhr-1/Robust_GUI_Grounding.



## **50. Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks**

cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.01308v2) [paper-pdf](http://arxiv.org/pdf/2504.01308v2)

**Authors**: Jiawei Wang, Yushen Zuo, Yuanjun Chai, Zhendong Liu, Yicheng Fu, Yichun Feng, Kin-Man Lam

**Abstract**: Vision-Language Models (VLMs) extend the capabilities of Large Language Models (LLMs) by incorporating visual information, yet they remain vulnerable to jailbreak attacks, especially when processing noisy or corrupted images. Although existing VLMs adopt security measures during training to mitigate such attacks, vulnerabilities associated with noise-augmented visual inputs are overlooked. In this work, we identify that missing noise-augmented training causes critical security gaps: many VLMs are susceptible to even simple perturbations such as Gaussian noise. To address this challenge, we propose Robust-VLGuard, a multimodal safety dataset with aligned / misaligned image-text pairs, combined with noise-augmented fine-tuning that reduces attack success rates while preserving functionality of VLM. For stronger optimization-based visual perturbation attacks, we propose DiffPure-VLM, leveraging diffusion models to convert adversarial perturbations into Gaussian-like noise, which can be defended by VLMs with noise-augmented safety fine-tuning. Experimental results demonstrate that the distribution-shifting property of diffusion model aligns well with our fine-tuned VLMs, significantly mitigating adversarial perturbations across varying intensities. The dataset and code are available at https://github.com/JarvisUSTC/DiffPure-RobustVLM.



