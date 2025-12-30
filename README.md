# Latest Adversarial Attack Papers
**update at 2025-12-30 12:28:33**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Multilingual Hidden Prompt Injection Attacks on LLM-Based Academic Reviewing**

cs.CL

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23684v1) [paper-pdf](https://arxiv.org/pdf/2512.23684v1)

**Authors**: Panagiotis Theocharopoulos, Ajinkya Kulkarni, Mathew Magimai. -Doss

**Abstract**: Large language models (LLMs) are increasingly considered for use in high-impact workflows, including academic peer review. However, LLMs are vulnerable to document-level hidden prompt injection attacks. In this work, we construct a dataset of approximately 500 real academic papers accepted to ICML and evaluate the effect of embedding hidden adversarial prompts within these documents. Each paper is injected with semantically equivalent instructions in four different languages and reviewed using an LLM. We find that prompt injection induces substantial changes in review scores and accept/reject decisions for English, Japanese, and Chinese injections, while Arabic injections produce little to no effect. These results highlight the susceptibility of LLM-based reviewing systems to document-level prompt injection and reveal notable differences in vulnerability across languages.



## **2. RobustMask: Certified Robustness against Adversarial Neural Ranking Attack via Randomized Masking**

cs.CR

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23307v1) [paper-pdf](https://arxiv.org/pdf/2512.23307v1)

**Authors**: Jiawei Liu, Zhuo Chen, Rui Zhu, Miaokun Chen, Yuyang Gong, Wei Lu, Xiaofeng Wang

**Abstract**: Neural ranking models have achieved remarkable progress and are now widely deployed in real-world applications such as Retrieval-Augmented Generation (RAG). However, like other neural architectures, they remain vulnerable to adversarial manipulations: subtle character-, word-, or phrase-level perturbations can poison retrieval results and artificially promote targeted candidates, undermining the integrity of search engines and downstream systems. Existing defenses either rely on heuristics with poor generalization or on certified methods that assume overly strong adversarial knowledge, limiting their practical use. To address these challenges, we propose RobustMask, a novel defense that combines the context-prediction capability of pretrained language models with a randomized masking-based smoothing mechanism. Our approach strengthens neural ranking models against adversarial perturbations at the character, word, and phrase levels. Leveraging both the pairwise comparison ability of ranking models and probabilistic statistical analysis, we provide a theoretical proof of RobustMask's certified top-K robustness. Extensive experiments further demonstrate that RobustMask successfully certifies over 20% of candidate documents within the top-10 ranking positions against adversarial perturbations affecting up to 30% of their content. These results highlight the effectiveness of RobustMask in enhancing the adversarial robustness of neural ranking models, marking a significant step toward providing stronger security guarantees for real-world retrieval systems.



## **3. It's a TRAP! Task-Redirecting Agent Persuasion Benchmark for Web Agents**

cs.HC

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23128v1) [paper-pdf](https://arxiv.org/pdf/2512.23128v1)

**Authors**: Karolina Korgul, Yushi Yang, Arkadiusz Drohomirecki, Piotr Błaszczyk, Will Howard, Lukas Aichberger, Chris Russell, Philip H. S. Torr, Adam Mahdi, Adel Bibi

**Abstract**: Web-based agents powered by large language models are increasingly used for tasks such as email management or professional networking. Their reliance on dynamic web content, however, makes them vulnerable to prompt injection attacks: adversarial instructions hidden in interface elements that persuade the agent to divert from its original task. We introduce the Task-Redirecting Agent Persuasion Benchmark (TRAP), an evaluation for studying how persuasion techniques misguide autonomous web agents on realistic tasks. Across six frontier models, agents are susceptible to prompt injection in 25\% of tasks on average (13\% for GPT-5 to 43\% for DeepSeek-R1), with small interface or contextual changes often doubling success rates and revealing systemic, psychologically driven vulnerabilities in web-based agents. We also provide a modular social-engineering injection framework with controlled experiments on high-fidelity website clones, allowing for further benchmark expansion.



## **4. DECEPTICON: How Dark Patterns Manipulate Web Agents**

cs.CR

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2512.22894v1) [paper-pdf](https://arxiv.org/pdf/2512.22894v1)

**Authors**: Phil Cuvin, Hao Zhu, Diyi Yang

**Abstract**: Deceptive UI designs, widely instantiated across the web and commonly known as dark patterns, manipulate users into performing actions misaligned with their goals. In this paper, we show that dark patterns are highly effective in steering agent trajectories, posing a significant risk to agent robustness. To quantify this risk, we introduce DECEPTICON, an environment for testing individual dark patterns in isolation. DECEPTICON includes 700 web navigation tasks with dark patterns -- 600 generated tasks and 100 real-world tasks, designed to measure instruction-following success and dark pattern effectiveness. Across state-of-the-art agents, we find dark patterns successfully steer agent trajectories towards malicious outcomes in over 70% of tested generated and real-world tasks -- compared to a human average of 31%. Moreover, we find that dark pattern effectiveness correlates positively with model size and test-time reasoning, making larger, more capable models more susceptible. Leading countermeasures against adversarial attacks, including in-context prompting and guardrail models, fail to consistently reduce the success rate of dark pattern interventions. Our findings reveal dark patterns as a latent and unmitigated risk to web agents, highlighting the urgent need for robust defenses against manipulative designs.



## **5. Adaptive Trust Consensus for Blockchain IoT: Comparing RL, DRL, and MARL Against Naive, Collusive, Adaptive, Byzantine, and Sleeper Attacks**

cs.CR

34 pages, 19 figures, 10 tables. Code available at https://github.com/soham-padia/blockchain-iot-trust

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2512.22860v1) [paper-pdf](https://arxiv.org/pdf/2512.22860v1)

**Authors**: Soham Padia, Dhananjay Vaidya, Ramchandra Mangrulkar

**Abstract**: Securing blockchain-enabled IoT networks against sophisticated adversarial attacks remains a critical challenge. This paper presents a trust-based delegated consensus framework integrating Fully Homomorphic Encryption (FHE) with Attribute-Based Access Control (ABAC) for privacy-preserving policy evaluation, combined with learning-based defense mechanisms. We systematically compare three reinforcement learning approaches -- tabular Q-learning (RL), Deep RL with Dueling Double DQN (DRL), and Multi-Agent RL (MARL) -- against five distinct attack families: Naive Malicious Attack (NMA), Collusive Rumor Attack (CRA), Adaptive Adversarial Attack (AAA), Byzantine Fault Injection (BFI), and Time-Delayed Poisoning (TDP). Experimental results on a 16-node simulated IoT network reveal significant performance variations: MARL achieves superior detection under collusive attacks (F1=0.85 vs. DRL's 0.68 and RL's 0.50), while DRL and MARL both attain perfect detection (F1=1.00) against adaptive attacks where RL fails (F1=0.50). All agents successfully defend against Byzantine attacks (F1=1.00). Most critically, the Time-Delayed Poisoning attack proves catastrophic for all agents, with F1 scores dropping to 0.11-0.16 after sleeper activation, demonstrating the severe threat posed by trust-building adversaries. Our findings indicate that coordinated multi-agent learning provides measurable advantages for defending against sophisticated trust manipulation attacks in blockchain IoT environments.



## **6. Reach-Avoid Differential game with Reachability Analysis for UAVs: A decomposition approach**

eess.SY

Paper version accepted to the Journal of Guidance, Control, and Dynamics (JGCD)

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2512.22793v1) [paper-pdf](https://arxiv.org/pdf/2512.22793v1)

**Authors**: Minh Bui, Simon Monckton, Mo Chen

**Abstract**: Reach-avoid (RA) games have significant applications in security and defense, particularly for unmanned aerial vehicles (UAVs). These problems are inherently challenging due to the need to consider obstacles, consider the adversarial nature of opponents, ensure optimality, and account for nonlinear dynamics. Hamilton-Jacobi (HJ) reachability analysis has emerged as a powerful tool for tackling these challenges; however, while it has been applied to games involving two spatial dimensions, directly extending this approach to three spatial dimensions is impossible due to high dimensionality. On the other hand, alternative approaches for solving RA games lack the generality to consider games with three spatial dimensions involving agents with non-trivial system dynamics. In this work, we propose a novel framework for dimensionality reduction by decomposing the problem into a horizontal RA sub-game and a vertical RA sub-game. We then solve each sub-game using HJ reachability analysis and consider second-order dynamics that account for the defender's acceleration. To reconstruct the solution to the original RA game from the sub-games, we introduce a HJ-based tracking control algorithm in each sub-game that not only guarantees capture of the attacker but also tracking of the attacker thereafter. We prove the conditions under which the capture guarantees are maintained. The effectiveness of our approach is demonstrated via numerical simulations, showing that the decomposition maintains optimality and guarantees in the original problem. Our methods are also validated in a Gazebo physics simulator, achieving successful capture of quadrotors in three spatial dimensions space for the first time to the best of our knowledge.



## **7. Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks**

cs.LG

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2512.22522v1) [paper-pdf](https://arxiv.org/pdf/2512.22522v1)

**Authors**: Jihang Wang, Dongcheng Zhao, Ruolin Chen, Qian Zhang, Yi Zeng

**Abstract**: Spiking Neural Networks (SNNs) utilize spike-based activations to mimic the brain's energy-efficient information processing. However, the binary and discontinuous nature of spike activations causes vanishing gradients, making adversarial robustness evaluation via gradient descent unreliable. While improved surrogate gradient methods have been proposed, their effectiveness under strong adversarial attacks remains unclear. We propose a more reliable framework for evaluating SNN adversarial robustness. We theoretically analyze the degree of gradient vanishing in surrogate gradients and introduce the Adaptive Sharpness Surrogate Gradient (ASSG), which adaptively evolves the shape of the surrogate function according to the input distribution during attack iterations, thereby enhancing gradient accuracy while mitigating gradient vanishing. In addition, we design an adversarial attack with adaptive step size under the $L_\infty$ constraint-Stable Adaptive Projected Gradient Descent (SA-PGD), achieving faster and more stable convergence under imprecise gradients. Extensive experiments show that our approach substantially increases attack success rates across diverse adversarial training schemes, SNN architectures and neuron models, providing a more generalized and reliable evaluation of SNN adversarial robustness. The experimental results further reveal that the robustness of current SNNs has been significantly overestimated and highlighting the need for more dependable adversarial training methods.



## **8. NOWA: Null-space Optical Watermark for Invisible Capture Fingerprinting and Tamper Localization**

cs.CR

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2512.22501v1) [paper-pdf](https://arxiv.org/pdf/2512.22501v1)

**Authors**: Edwin Vargas

**Abstract**: Ensuring the authenticity and ownership of digital images is increasingly challenging as modern editing tools enable highly realistic forgeries. Existing image protection systems mainly rely on digital watermarking, which is susceptible to sophisticated digital attacks. To address this limitation, we propose a hybrid optical-digital framework that incorporates physical authentication cues during image formation and preserves them through a learned reconstruction process. At the optical level, a phase mask in the camera aperture produces a Null-space Optical Watermark (NOWA) that lies in the Null Space of the imaging operator and therefore remains invisible in the captured image. Then, a Null-Space Network (NSN) performs measurement-consistent reconstruction that delivers high-quality protected images while preserving the NOWA signature. The proposed design enables tamper localization by projecting the image onto the camera's null space and detecting pixel-level inconsistencies. Our design preserves perceptual quality, resists common degradations such as compression, and establishes a structural security asymmetry: without access to the optical or NSN parameters, adversaries cannot forge the NOWA signature. Experiments with simulations and a prototype camera demonstrate competitive performance in terms of image quality preservation, and tamper localization accuracy compared to state-of-the-art digital watermarking and learning-based authentication methods.



## **9. PHANTOM: Physics-Aware Adversarial Attacks against Federated Learning-Coordinated EV Charging Management System**

cs.ET

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2512.22381v1) [paper-pdf](https://arxiv.org/pdf/2512.22381v1)

**Authors**: Mohammad Zakaria Haider, Amit Kumar Podder, Prabin Mali, Aranya Chakrabortty, Sumit Paudyal, Mohammad Ashiqur Rahman

**Abstract**: The rapid deployment of electric vehicle charging stations (EVCS) within distribution networks necessitates intelligent and adaptive control to maintain the grid's resilience and reliability. In this work, we propose PHANTOM, a physics-aware adversarial network that is trained and optimized through a multi-agent reinforcement learning model. PHANTOM integrates a physics-informed neural network (PINN) enabled by federated learning (FL) that functions as a digital twin of EVCS-integrated systems, ensuring physically consistent modeling of operational dynamics and constraints. Building on this digital twin, we construct a multi-agent RL environment that utilizes deep Q-networks (DQN) and soft actor-critic (SAC) methods to derive adversarial false data injection (FDI) strategies capable of bypassing conventional detection mechanisms. To examine the broader grid-level consequences, a transmission and distribution (T and D) dual simulation platform is developed, allowing us to capture cascading interactions between EVCS disturbances at the distribution level and the operations of the bulk transmission system. Results demonstrate how learned attack policies disrupt load balancing and induce voltage instabilities that propagate across T and D boundaries. These findings highlight the critical need for physics-aware cybersecurity to ensure the resilience of large-scale vehicle-grid integration.



## **10. Scaling Adversarial Training via Data Selection**

cs.LG

6 pages. Conference workshop paper

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2512.22069v1) [paper-pdf](https://arxiv.org/pdf/2512.22069v1)

**Authors**: Youran Ye, Dejin Wang, Ajinkya Bhandare

**Abstract**: Projected Gradient Descent (PGD) is a strong and widely used first-order adversarial attack, yet its computational cost scales poorly, as all training samples undergo identical iterative inner-loop optimization despite contributing unequally to robustness. Motivated by this inefficiency, we propose \emph{Selective Adversarial Training}, which perturbs only a subset of critical samples in each minibatch. Specifically, we introduce two principled selection criteria: (1) margin-based sampling, which prioritizes samples near the decision boundary, and (2) gradient-matching sampling, which selects samples whose gradients align with the dominant batch optimization direction. Adversarial examples are generated only for the selected subset, while the remaining samples are trained cleanly using a mixed objective. Experiments on MNIST and CIFAR-10 show that the proposed methods achieve robustness comparable to, or even exceeding, full PGD adversarial training, while reducing adversarial computation by up to $50\%$, demonstrating that informed sample selection is sufficient for scalable adversarial robustness.



## **11. Few Tokens Matter: Entropy Guided Attacks on Vision-Language Models**

cs.CV

19 Pages,11 figures,8 tables

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2512.21815v1) [paper-pdf](https://arxiv.org/pdf/2512.21815v1)

**Authors**: Mengqi He, Xinyu Tian, Xin Shen, Jinhong Ni, Shu Zou, Zhaoyuan Yang, Jing Zhang

**Abstract**: Vision-language models (VLMs) achieve remarkable performance but remain vulnerable to adversarial attacks. Entropy, a measure of model uncertainty, is strongly correlated with the reliability of VLM. Prior entropy-based attacks maximize uncertainty at all decoding steps, implicitly assuming that every token contributes equally to generation instability. We show instead that a small fraction (about 20%) of high-entropy tokens, i.e., critical decision points in autoregressive generation, disproportionately governs output trajectories. By concentrating adversarial perturbations on these positions, we achieve semantic degradation comparable to global methods while using substantially smaller budgets. More importantly, across multiple representative VLMs, such selective attacks convert 35-49% of benign outputs into harmful ones, exposing a more critical safety risk. Remarkably, these vulnerable high-entropy forks recur across architecturally diverse VLMs, enabling feasible transferability (17-26% harmful rates on unseen targets). Motivated by these findings, we propose Entropy-bank Guided Adversarial attacks (EGA), which achieves competitive attack success rates (93-95%) alongside high harmful conversion, thereby revealing new weaknesses in current VLM safety mechanisms.



## **12. LLM-Driven Feature-Level Adversarial Attacks on Android Malware Detectors**

cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21404v1) [paper-pdf](https://arxiv.org/pdf/2512.21404v1)

**Authors**: Tianwei Lan, Farid Naït-Abdesselam

**Abstract**: The rapid growth in both the scale and complexity of Android malware has driven the widespread adoption of machine learning (ML) techniques for scalable and accurate malware detection. Despite their effectiveness, these models remain vulnerable to adversarial attacks that introduce carefully crafted feature-level perturbations to evade detection while preserving malicious functionality. In this paper, we present LAMLAD, a novel adversarial attack framework that exploits the generative and reasoning capabilities of large language models (LLMs) to bypass ML-based Android malware classifiers. LAMLAD employs a dual-agent architecture composed of an LLM manipulator, which generates realistic and functionality-preserving feature perturbations, and an LLM analyzer, which guides the perturbation process toward successful evasion. To improve efficiency and contextual awareness, LAMLAD integrates retrieval-augmented generation (RAG) into the LLM pipeline. Focusing on Drebin-style feature representations, LAMLAD enables stealthy and high-confidence attacks against widely deployed Android malware detection systems. We evaluate LAMLAD against three representative ML-based Android malware detectors and compare its performance with two state-of-the-art adversarial attack methods. Experimental results demonstrate that LAMLAD achieves an attack success rate (ASR) of up to 97%, requiring on average only three attempts per adversarial sample, highlighting its effectiveness, efficiency, and adaptability in practical adversarial settings. Furthermore, we propose an adversarial training-based defense strategy that reduces the ASR by more than 30% on average, significantly enhancing model robustness against LAMLAD-style attacks.



## **13. CoTDeceptor:Adversarial Code Obfuscation Against CoT-Enhanced LLM Code Agents**

cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21250v1) [paper-pdf](https://arxiv.org/pdf/2512.21250v1)

**Authors**: Haoyang Li, Mingjin Li, Jinxin Zuo, Siqi Li, Xiao Li, Hao Wu, Yueming Lu, Xiaochuan He

**Abstract**: LLM-based code agents(e.g., ChatGPT Codex) are increasingly deployed as detector for code review and security auditing tasks. Although CoT-enhanced LLM vulnerability detectors are believed to provide improved robustness against obfuscated malicious code, we find that their reasoning chains and semantic abstraction processes exhibit exploitable systematic weaknesses.This allows attackers to covertly embed malicious logic, bypass code review, and propagate backdoored components throughout real-world software supply chains.To investigate this issue, we present CoTDeceptor, the first adversarial code obfuscation framework targeting CoT-enhanced LLM detectors. CoTDeceptor autonomously constructs evolving, hard-to-reverse multi-stage obfuscation strategy chains that effectively disrupt CoT-driven detection logic.We obtained malicious code provided by security enterprise, experimental results demonstrate that CoTDeceptor achieves stable and transferable evasion performance against state-of-the-art LLMs and vulnerability detection agents. CoTDeceptor bypasses 14 out of 15 vulnerability categories, compared to only 2 bypassed by prior methods. Our findings highlight potential risks in real-world software supply chains and underscore the need for more robust and interpretable LLM-powered security analysis systems.



## **14. Improving the Convergence Rate of Ray Search Optimization for Query-Efficient Hard-Label Attacks**

cs.LG

Published at AAAI 2026 (Oral). This version corresponds to the conference proceedings; v2 will include the appendix

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21241v1) [paper-pdf](https://arxiv.org/pdf/2512.21241v1)

**Authors**: Xinjie Xu, Shuyu Cheng, Dongwei Xu, Qi Xuan, Chen Ma

**Abstract**: In hard-label black-box adversarial attacks, where only the top-1 predicted label is accessible, the prohibitive query complexity poses a major obstacle to practical deployment. In this paper, we focus on optimizing a representative class of attacks that search for the optimal ray direction yielding the minimum $\ell_2$-norm perturbation required to move a benign image into the adversarial region. Inspired by Nesterov's Accelerated Gradient (NAG), we propose a momentum-based algorithm, ARS-OPT, which proactively estimates the gradient with respect to a future ray direction inferred from accumulated momentum. We provide a theoretical analysis of its convergence behavior, showing that ARS-OPT enables more accurate directional updates and achieves faster, more stable optimization. To further accelerate convergence, we incorporate surrogate-model priors into ARS-OPT's gradient estimation, resulting in PARS-OPT with enhanced performance. The superiority of our approach is supported by theoretical guarantees under standard assumptions. Extensive experiments on ImageNet and CIFAR-10 demonstrate that our method surpasses 13 state-of-the-art approaches in query efficiency.



## **15. Time-Bucketed Balance Records: Bounded-Storage Ephemeral Tokens for Resource-Constrained Systems**

cs.DS

14 pages, 1 figure, 1 Algorithm, 3 Theorems

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.20962v1) [paper-pdf](https://arxiv.org/pdf/2512.20962v1)

**Authors**: Shaun Scovil, Bhargav Chickmagalur Nanjundappa

**Abstract**: Fungible tokens with time-to-live (TTL) semantics require tracking individual expiration times for each deposited unit. A naive implementation creates a new balance record per deposit, leading to unbounded storage growth and vulnerability to denial-of-service attacks. We present time-bucketed balance records, a data structure that bounds storage to O(k) records per account while guaranteeing that tokens never expire before their configured TTL. Our approach discretizes time into k buckets, coalescing deposits within the same bucket to limit unique expiration timestamps. We prove three key properties: (1) storage is bounded by k+1 records regardless of deposit frequency, (2) actual expiration time is always at least the configured TTL, and (3) adversaries cannot increase a victim's operation cost beyond O(k)[amortized] worst case. We provide a reference implementation in Solidity with measured gas costs demonstrating practical efficiency.



## **16. The Imitation Game: Using Large Language Models as Chatbots to Combat Chat-Based Cybercrimes**

cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21371v1) [paper-pdf](https://arxiv.org/pdf/2512.21371v1)

**Authors**: Yifan Yao, Baojuan Wang, Jinhao Duan, Kaidi Xu, ChuanKai Guo, Zhibo Eric Sun, Yue Zhang

**Abstract**: Chat-based cybercrime has emerged as a pervasive threat, with attackers leveraging real-time messaging platforms to conduct scams that rely on trust-building, deception, and psychological manipulation. Traditional defense mechanisms, which operate on static rules or shallow content filters, struggle to identify these conversational threats, especially when attackers use multimedia obfuscation and context-aware dialogue.   In this work, we ask a provocative question inspired by the classic Imitation Game: Can machines convincingly pose as human victims to turn deception against cybercriminals? We present LURE (LLM-based User Response Engagement), the first system to deploy Large Language Models (LLMs) as active agents, not as passive classifiers, embedded within adversarial chat environments.   LURE combines automated discovery, adversarial interaction, and OCR-based analysis of image-embedded payment data. Applied to the setting of illicit video chat scams on Telegram, our system engaged 53 actors across 98 groups. In over 56 percent of interactions, the LLM maintained multi-round conversations without being noticed as a bot, effectively "winning" the imitation game. Our findings reveal key behavioral patterns in scam operations, such as payment flows, upselling strategies, and platform migration tactics.



## **17. Robustness Certificates for Neural Networks against Adversarial Attacks**

cs.LG

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.20865v1) [paper-pdf](https://arxiv.org/pdf/2512.20865v1)

**Authors**: Sara Taheri, Mahalakshmi Sabanayagam, Debarghya Ghoshdastidar, Majid Zamani

**Abstract**: The increasing use of machine learning in safety-critical domains amplifies the risk of adversarial threats, especially data poisoning attacks that corrupt training data to degrade performance or induce unsafe behavior. Most existing defenses lack formal guarantees or rely on restrictive assumptions about the model class, attack type, extent of poisoning, or point-wise certification, limiting their practical reliability. This paper introduces a principled formal robustness certification framework that models gradient-based training as a discrete-time dynamical system (dt-DS) and formulates poisoning robustness as a formal safety verification problem. By adapting the concept of barrier certificates (BCs) from control theory, we introduce sufficient conditions to certify a robust radius ensuring that the terminal model remains safe under worst-case ${\ell}_p$-norm based poisoning. To make this practical, we parameterize BCs as neural networks trained on finite sets of poisoned trajectories. We further derive probably approximately correct (PAC) bounds by solving a scenario convex program (SCP), which yields a confidence lower bound on the certified robustness radius generalizing beyond the training set. Importantly, our framework also extends to certification against test-time attacks, making it the first unified framework to provide formal guarantees in both training and test-time attack settings. Experiments on MNIST, SVHN, and CIFAR-10 show that our approach certifies non-trivial perturbation budgets while being model-agnostic and requiring no prior knowledge of the attack or contamination level.



## **18. Defending against adversarial attacks using mixture of experts**

cs.LG

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20821v1) [paper-pdf](https://arxiv.org/pdf/2512.20821v1)

**Authors**: Mohammad Meymani, Roozbeh Razavi-Far

**Abstract**: Machine learning is a powerful tool enabling full automation of a huge number of tasks without explicit programming. Despite recent progress of machine learning in different domains, these models have shown vulnerabilities when they are exposed to adversarial threats. Adversarial threats aim to hinder the machine learning models from satisfying their objectives. They can create adversarial perturbations, which are imperceptible to humans' eyes but have the ability to cause misclassification during inference. Moreover, they can poison the training data to harm the model's performance or they can query the model to steal its sensitive information. In this paper, we propose a defense system, which devises an adversarial training module within mixture-of-experts architecture to enhance its robustness against adversarial threats. In our proposed defense system, we use nine pre-trained experts with ResNet-18 as their backbone. During end-to-end training, the parameters of expert models and gating mechanism are jointly updated allowing further optimization of the experts. Our proposed defense system outperforms state-of-the-art defense systems and plain classifiers, which use a more complex architecture than our model's backbone.



## **19. Safety Alignment of LMs via Non-cooperative Games**

cs.AI

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20806v1) [paper-pdf](https://arxiv.org/pdf/2512.20806v1)

**Authors**: Anselm Paulus, Ilia Kulikov, Brandon Amos, Rémi Munos, Ivan Evtimov, Kamalika Chaudhuri, Arman Zharmagambetov

**Abstract**: Ensuring the safety of language models (LMs) while maintaining their usefulness remains a critical challenge in AI alignment. Current approaches rely on sequential adversarial training: generating adversarial prompts and fine-tuning LMs to defend against them. We introduce a different paradigm: framing safety alignment as a non-zero-sum game between an Attacker LM and a Defender LM trained jointly via online reinforcement learning. Each LM continuously adapts to the other's evolving strategies, driving iterative improvement. Our method uses a preference-based reward signal derived from pairwise comparisons instead of point-wise scores, providing more robust supervision and potentially reducing reward hacking. Our RL recipe, AdvGame, shifts the Pareto frontier of safety and utility, yielding a Defender LM that is simultaneously more helpful and more resilient to adversarial attacks. In addition, the resulting Attacker LM converges into a strong, general-purpose red-teaming agent that can be directly deployed to probe arbitrary target models.



## **20. Failure Analysis of Safety Controllers in Autonomous Vehicles Under Object-Based LiDAR Attacks**

cs.SE

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.22244v1) [paper-pdf](https://arxiv.org/pdf/2512.22244v1)

**Authors**: Daniyal Ganiuly, Nurzhau Bolatbek, Assel Smaiyl

**Abstract**: Autonomous vehicles rely on LiDAR based perception to support safety critical control functions such as adaptive cruise control and automatic emergency braking. While previous research has shown that LiDAR perception can be manipulated through object based spoofing and injection attacks, the impact of such attacks on vehicle safety controllers is still not well understood. This paper presents a systematic failure analysis of longitudinal safety controllers under object based LiDAR attacks in highway driving scenarios. The study focuses on realistic cut in and car following situations in which adversarial objects introduce persistent perception errors without directly modifying vehicle control software. A high fidelity simulation framework integrating LiDAR perception, object tracking, and closed loop vehicle control is used to evaluate how false and displaced object detections propagate through the perception planning and control pipeline. The results demonstrate that even short duration LiDAR induced object hallucinations can trigger unsafe braking, delayed responses to real hazards, and unstable control behavior. In cut in scenarios, a clear increase in unsafe deceleration events and time to collision violations is observed when compared to benign conditions, despite identical controller parameters. The analysis further shows that controller failures are more strongly influenced by the temporal consistency of spoofed objects than by spatial inaccuracies alone. These findings reveal a critical gap between perception robustness and control level safety guarantees in autonomous driving systems. By explicitly characterizing safety controller failure modes under adversarial perception, this work provides practical insights for the design of attack aware safety mechanisms and more resilient control strategies for LiDAR dependent autonomous vehicles.



## **21. Satellite Cybersecurity Across Orbital Altitudes: Analyzing Ground-Based Threats to LEO, MEO, and GEO**

cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.21367v1) [paper-pdf](https://arxiv.org/pdf/2512.21367v1)

**Authors**: Mark Ballard, Guanqun Song, Ting Zhu

**Abstract**: The rapid proliferation of satellite constellations, particularly in Low Earth Orbit (LEO), has fundamentally altered the global space infrastructure, shifting the risk landscape from purely kinetic collisions to complex cyber-physical threats. While traditional safety frameworks focus on debris mitigation, ground-based adversaries increasingly exploit radio-frequency links, supply chain vulnerabilities, and software update pathways to degrade space assets. This paper presents a comparative analysis of satellite cybersecurity across LEO, Medium Earth Orbit (MEO), and Geostationary Earth Orbit (GEO) regimes. By synthesizing data from 60 publicly documented security incidents with key vulnerability proxies--including Telemetry, Tracking, and Command (TT&C) anomalies, encryption weaknesses, and environmental stressors--we characterize how orbital altitude dictates attack feasibility and impact. Our evaluation reveals distinct threat profiles: GEO systems are predominantly targeted via high-frequency uplink exposure, whereas LEO constellations face unique risks stemming from limited power budgets, hardware constraints, and susceptibility to thermal and radiation-induced faults. We further bridge the gap between security and sustainability, arguing that unmitigated cyber vulnerabilities accelerate hardware obsolescence and debris accumulation, undermining efforts toward carbon-neutral space operations. The results demonstrate that weak encryption and command path irregularities are the most consistent predictors of adversarial success across all orbits.



## **22. Real-World Adversarial Attacks on RF-Based Drone Detectors**

cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20712v1) [paper-pdf](https://arxiv.org/pdf/2512.20712v1)

**Authors**: Omer Gazit, Yael Itzhakev, Yuval Elovici, Asaf Shabtai

**Abstract**: Radio frequency (RF) based systems are increasingly used to detect drones by analyzing their RF signal patterns, converting them into spectrogram images which are processed by object detection models. Existing RF attacks against image based models alter digital features, making over-the-air (OTA) implementation difficult due to the challenge of converting digital perturbations to transmittable waveforms that may introduce synchronization errors and interference, and encounter hardware limitations. We present the first physical attack on RF image based drone detectors, optimizing class-specific universal complex baseband (I/Q) perturbation waveforms that are transmitted alongside legitimate communications. We evaluated the attack using RF recordings and OTA experiments with four types of drones. Our results show that modest, structured I/Q perturbations are compatible with standard RF chains and reliably reduce target drone detection while preserving detection of legitimate drones.



## **23. Evasion-Resilient Detection of DNS-over-HTTPS Data Exfiltration: A Practical Evaluation and Toolkit**

cs.CR

61 pages Advisor : Dr Darren Hurley-Smith

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20423v1) [paper-pdf](https://arxiv.org/pdf/2512.20423v1)

**Authors**: Adam Elaoumari

**Abstract**: The purpose of this project is to assess how well defenders can detect DNS-over-HTTPS (DoH) file exfiltration, and which evasion strategies can be used by attackers. While providing a reproducible toolkit to generate, intercept and analyze DoH exfiltration, and comparing Machine Learning vs threshold-based detection under adversarial scenarios. The originality of this project is the introduction of an end-to-end, containerized pipeline that generates configurable file exfiltration over DoH using several parameters (e.g., chunking, encoding, padding, resolver rotation). It allows for file reconstruction at the resolver side, while extracting flow-level features using a fork of DoHLyzer. The pipeline contains a prediction side, which allows the training of machine learning models based on public labelled datasets and then evaluates them side-by-side with threshold-based detection methods against malicious and evasive DNS-Over-HTTPS traffic. We train Random Forest, Gradient Boosting and Logistic Regression classifiers on a public DoH dataset and benchmark them against evasive DoH exfiltration scenarios. The toolkit orchestrates traffic generation, file capture, feature extraction, model training and analysis. The toolkit is then encapsulated into several Docker containers for easy setup and full reproducibility regardless of the platform it is run on. Future research regarding this project is directed at validating the results on mixed enterprise traffic, extending the protocol coverage to HTTP/3/QUIC request, adding a benign traffic generation, and working on real-time traffic evaluation. A key objective is to quantify when stealth constraints make DoH exfiltration uneconomical and unworthy for the attacker.



## **24. Contingency Model-based Control (CMC) for Communicationless Cooperative Collision Avoidance in Robot Swarms**

math.OC

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.20391v2) [paper-pdf](https://arxiv.org/pdf/2512.20391v2)

**Authors**: Georg Schildbach

**Abstract**: Cooperative collision avoidance between robots in swarm operations remains an open challenge. Assuming a decentralized architecture, each robot is responsible for making its own control decisions, including motion planning. To this end, most existing approaches mostly rely some form of (wireless) communication between the agents of the swarm. In reality, however, communication is brittle. It may be affected by latency, further delays and packet losses, transmission faults, and is subject to adversarial attacks, such as jamming or spoofing. This paper proposes Contingency Model-based Control (CMC) as a communicationless alternative. It follows the implicit cooperation paradigm, under which the design of the robots is based on consensual (offline) rules, similar to traffic rules. They include the definition of a contingency trajectory for each robot, and a method for construction of mutual collision avoidance constraints. The setup is shown to guarantee the recursive feasibility and collision avoidance between all swarm members in closed-loop operation. Moreover, CMC naturally satisfies the Plug \& Play paradigm, i.e., for new robots entering the swarm. Two numerical examples demonstrate that the collision avoidance guarantee is intact and that the robot swarm operates smoothly under the CMC regime.



## **25. Optimistic TEE-Rollups: A Hybrid Architecture for Scalable and Verifiable Generative AI Inference on Blockchain**

cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20176v1) [paper-pdf](https://arxiv.org/pdf/2512.20176v1)

**Authors**: Aaron Chan, Alex Ding, Frank Chen, Alan Wu, Bruce Zhang, Arther Tian

**Abstract**: The rapid integration of Large Language Models (LLMs) into decentralized physical infrastructure networks (DePIN) is currently bottlenecked by the Verifiability Trilemma, which posits that a decentralized inference system cannot simultaneously achieve high computational integrity, low latency, and low cost. Existing cryptographic solutions, such as Zero-Knowledge Machine Learning (ZKML), suffer from superlinear proving overheads (O(k NlogN)) that render them infeasible for billionparameter models. Conversely, optimistic approaches (opML) impose prohibitive dispute windows, preventing real-time interactivity, while recent "Proof of Quality" (PoQ) paradigms sacrifice cryptographic integrity for subjective semantic evaluation, leaving networks vulnerable to model downgrade attacks and reward hacking. In this paper, we introduce Optimistic TEE-Rollups (OTR), a hybrid verification protocol that harmonizes these constraints. OTR leverages NVIDIA H100 Confidential Computing Trusted Execution Environments (TEEs) to provide sub-second Provisional Finality, underpinned by an optimistic fraud-proof mechanism and stochastic Zero-Knowledge spot-checks to mitigate hardware side-channel risks. We formally define Proof of Efficient Attribution (PoEA), a consensus mechanism that cryptographically binds execution traces to hardware attestations, thereby guaranteeing model authenticity. Extensive simulations demonstrate that OTR achieves 99% of the throughput of centralized baselines with a marginal cost overhead of $0.07 per query, maintaining Byzantine fault tolerance against rational adversaries even in the presence of transient hardware vulnerabilities.



## **26. Odysseus: Jailbreaking Commercial Multimodal LLM-integrated Systems via Dual Steganography**

cs.CR

This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20168v1) [paper-pdf](https://arxiv.org/pdf/2512.20168v1)

**Authors**: Songze Li, Jiameng Cheng, Yiming Li, Xiaojun Jia, Dacheng Tao

**Abstract**: By integrating language understanding with perceptual modalities such as images, multimodal large language models (MLLMs) constitute a critical substrate for modern AI systems, particularly intelligent agents operating in open and interactive environments. However, their increasing accessibility also raises heightened risks of misuse, such as generating harmful or unsafe content. To mitigate these risks, alignment techniques are commonly applied to align model behavior with human values. Despite these efforts, recent studies have shown that jailbreak attacks can circumvent alignment and elicit unsafe outputs. Currently, most existing jailbreak methods are tailored for open-source models and exhibit limited effectiveness against commercial MLLM-integrated systems, which often employ additional filters. These filters can detect and prevent malicious input and output content, significantly reducing jailbreak threats. In this paper, we reveal that the success of these safety filters heavily relies on a critical assumption that malicious content must be explicitly visible in either the input or the output. This assumption, while often valid for traditional LLM-integrated systems, breaks down in MLLM-integrated systems, where attackers can leverage multiple modalities to conceal adversarial intent, leading to a false sense of security in existing MLLM-integrated systems. To challenge this assumption, we propose Odysseus, a novel jailbreak paradigm that introduces dual steganography to covertly embed malicious queries and responses into benign-looking images. Extensive experiments on benchmark datasets demonstrate that our Odysseus successfully jailbreaks several pioneering and realistic MLLM-integrated systems, achieving up to 99% attack success rate. It exposes a fundamental blind spot in existing defenses, and calls for rethinking cross-modal security in MLLM-integrated systems.



## **27. AI Security Beyond Core Domains: Resume Screening as a Case Study of Adversarial Vulnerabilities in Specialized LLM Applications**

cs.CL

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20164v1) [paper-pdf](https://arxiv.org/pdf/2512.20164v1)

**Authors**: Honglin Mu, Jinghao Liu, Kaiyang Wan, Rui Xing, Xiuying Chen, Timothy Baldwin, Wanxiang Che

**Abstract**: Large Language Models (LLMs) excel at text comprehension and generation, making them ideal for automated tasks like code review and content moderation. However, our research identifies a vulnerability: LLMs can be manipulated by "adversarial instructions" hidden in input data, such as resumes or code, causing them to deviate from their intended task. Notably, while defenses may exist for mature domains such as code review, they are often absent in other common applications such as resume screening and peer review. This paper introduces a benchmark to assess this vulnerability in resume screening, revealing attack success rates exceeding 80% for certain attack types. We evaluate two defense mechanisms: prompt-based defenses achieve 10.1% attack reduction with 12.5% false rejection increase, while our proposed FIDS (Foreign Instruction Detection through Separation) using LoRA adaptation achieves 15.4% attack reduction with 10.4% false rejection increase. The combined approach provides 26.3% attack reduction, demonstrating that training-time defenses outperform inference-time mitigations in both security and utility preservation.



## **28. IoT-based Android Malware Detection Using Graph Neural Network With Adversarial Defense**

cs.CR

13 pages

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20004v1) [paper-pdf](https://arxiv.org/pdf/2512.20004v1)

**Authors**: Rahul Yumlembam, Biju Issac, Seibu Mary Jacob, Longzhi Yang

**Abstract**: Since the Internet of Things (IoT) is widely adopted using Android applications, detecting malicious Android apps is essential. In recent years, Android graph-based deep learning research has proposed many approaches to extract relationships from applications as graphs to generate graph embeddings. First, we demonstrate the effectiveness of graph-based classification using a Graph Neural Network (GNN)-based classifier to generate API graph embeddings. The graph embeddings are combined with Permission and Intent features to train multiple machine learning and deep learning models for Android malware detection. The proposed classification approach achieves an accuracy of 98.33 percent on the CICMaldroid dataset and 98.68 percent on the Drebin dataset. However, graph-based deep learning models are vulnerable, as attackers can add fake relationships to evade detection by the classifier. Second, we propose a Generative Adversarial Network (GAN)-based attack algorithm named VGAE-MalGAN targeting graph-based GNN Android malware classifiers. The VGAE-MalGAN generator produces adversarial malware API graphs, while the VGAE-MalGAN substitute detector attempts to mimic the target detector. Experimental results show that VGAE-MalGAN can significantly reduce the detection rate of GNN-based malware classifiers. Although the model initially fails to detect adversarial malware, retraining with generated adversarial samples improves robustness and helps mitigate adversarial attacks.



## **29. Conditional Adversarial Fragility in Financial Machine Learning under Macroeconomic Stress**

cs.LG

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19935v1) [paper-pdf](https://arxiv.org/pdf/2512.19935v1)

**Authors**: Samruddhi Baviskar

**Abstract**: Machine learning models used in financial decision systems operate in nonstationary economic environments, yet adversarial robustness is typically evaluated under static assumptions. This work introduces Conditional Adversarial Fragility, a regime dependent phenomenon in which adversarial vulnerability is systematically amplified during periods of macroeconomic stress. We propose a regime aware evaluation framework for time indexed tabular financial classification tasks that conditions robustness assessment on external indicators of economic stress. Using volatility based regime segmentation as a proxy for macroeconomic conditions, we evaluate model behavior across calm and stress periods while holding model architecture, attack methodology, and evaluation protocols constant. Baseline predictive performance remains comparable across regimes, indicating that economic stress alone does not induce inherent performance degradation. Under adversarial perturbations, however, models operating during stress regimes exhibit substantially greater degradation across predictive accuracy, operational decision thresholds, and risk sensitive outcomes. We further demonstrate that this amplification propagates to increased false negative rates, elevating the risk of missed high risk cases during adverse conditions. To complement numerical robustness metrics, we introduce an interpretive governance layer based on semantic auditing of model explanations using large language models. Together, these results demonstrate that adversarial robustness in financial machine learning is a regime dependent property and motivate stress aware approaches to model risk assessment in high stakes financial deployments.



## **30. Multi-Layer Confidence Scoring for Detection of Out-of-Distribution Samples, Adversarial Attacks, and In-Distribution Misclassifications**

cs.LG

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19472v1) [paper-pdf](https://arxiv.org/pdf/2512.19472v1)

**Authors**: Lorenzo Capelli, Leandro de Souza Rosa, Gianluca Setti, Mauro Mangia, Riccardo Rovatti

**Abstract**: The recent explosive growth in Deep Neural Networks applications raises concerns about the black-box usage of such models, with limited trasparency and trustworthiness in high-stakes domains, which have been crystallized as regulatory requirements such as the European Union Artificial Intelligence Act. While models with embedded confidence metrics have been proposed, such approaches cannot be applied to already existing models without retraining, limiting their broad application. On the other hand, post-hoc methods, which evaluate pre-trained models, focus on solving problems related to improving the confidence in the model's predictions, and detecting Out-Of-Distribution or Adversarial Attacks samples as independent applications. To tackle the limited applicability of already existing methods, we introduce Multi-Layer Analysis for Confidence Scoring (MACS), a unified post-hoc framework that analyzes intermediate activations to produce classification-maps. From the classification-maps, we derive a score applicable for confidence estimation, detecting distributional shifts and adversarial attacks, unifying the three problems in a common framework, and achieving performances that surpass the state-of-the-art approaches in our experiments with the VGG16 and ViTb16 models with a fraction of their computational overhead.



## **31. GShield: Mitigating Poisoning Attacks in Federated Learning**

cs.CR

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2512.19286v2) [paper-pdf](https://arxiv.org/pdf/2512.19286v2)

**Authors**: Sameera K. M., Serena Nicolazzo, Antonino Nocera, Vinod P., Rafidha Rehiman K. A

**Abstract**: Federated Learning (FL) has recently emerged as a revolutionary approach to collaborative training Machine Learning models. In particular, it enables decentralized model training while preserving data privacy, but its distributed nature makes it highly vulnerable to a severe attack known as Data Poisoning. In such scenarios, malicious clients inject manipulated data into the training process, thereby degrading global model performance or causing targeted misclassification. In this paper, we present a novel defense mechanism called GShield, designed to detect and mitigate malicious and low-quality updates, especially under non-independent and identically distributed (non-IID) data scenarios. GShield operates by learning the distribution of benign gradients through clustering and Gaussian modeling during an initial round, enabling it to establish a reliable baseline of trusted client behavior. With this benign profile, GShield selectively aggregates only those updates that align with the expected gradient patterns, effectively isolating adversarial clients and preserving the integrity of the global model. An extensive experimental campaign demonstrates that our proposed defense significantly improves model robustness compared to the state-of-the-art methods while maintaining a high accuracy of performance across both tabular and image datasets. Furthermore, GShield improves the accuracy of the targeted class by 43\% to 65\% after detecting malicious and low-quality clients.



## **32. Adversarially Robust Detection of Harmful Online Content: A Computational Design Science Approach**

cs.LG

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.17367v3) [paper-pdf](https://arxiv.org/pdf/2512.17367v3)

**Authors**: Yidong Chai, Yi Liu, Mohammadreza Ebrahimi, Weifeng Li, Balaji Padmanabhan

**Abstract**: Social media platforms are plagued by harmful content such as hate speech, misinformation, and extremist rhetoric. Machine learning (ML) models are widely adopted to detect such content; however, they remain highly vulnerable to adversarial attacks, wherein malicious users subtly modify text to evade detection. Enhancing adversarial robustness is therefore essential, requiring detectors that can defend against diverse attacks (generalizability) while maintaining high overall accuracy. However, simultaneously achieving both optimal generalizability and accuracy is challenging. Following the computational design science paradigm, this study takes a sequential approach that first proposes a novel framework (Large Language Model-based Sample Generation and Aggregation, LLM-SGA) by identifying the key invariances of textual adversarial attacks and leveraging them to ensure that a detector instantiated within the framework has strong generalizability. Second, we instantiate our detector (Adversarially Robust Harmful Online Content Detector, ARHOCD) with three novel design components to improve detection accuracy: (1) an ensemble of multiple base detectors that exploits their complementary strengths; (2) a novel weight assignment method that dynamically adjusts weights based on each sample's predictability and each base detector's capability, with weights initialized using domain knowledge and updated via Bayesian inference; and (3) a novel adversarial training strategy that iteratively optimizes both the base detectors and the weight assignor. We addressed several limitations of existing adversarial robustness enhancement research and empirically evaluated ARHOCD across three datasets spanning hate speech, rumor, and extremist content. Results show that ARHOCD offers strong generalizability and improves detection accuracy under adversarial conditions.



## **33. Machine Unlearning in Speech Emotion Recognition via Forget Set Alone**

cs.SD

Submitted to ICASSP 2026

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2510.04251v2) [paper-pdf](https://arxiv.org/pdf/2510.04251v2)

**Authors**: Zhao Ren, Rathi Adarshi Rammohan, Kevin Scheck, Tanja Schultz

**Abstract**: Speech emotion recognition aims to identify emotional states from speech signals and has been widely applied in human-computer interaction, education, healthcare, and many other fields. However, since speech data contain rich sensitive information, partial data can be required to be deleted by speakers due to privacy concerns. Current machine unlearning approaches largely depend on data beyond the samples to be forgotten. However, this reliance poses challenges when data redistribution is restricted and demands substantial computational resources in the context of big data. We propose a novel adversarial-attack-based approach that fine-tunes a pre-trained speech emotion recognition model using only the data to be forgotten. The experimental results demonstrate that the proposed approach can effectively remove the knowledge of the data to be forgotten from the model, while preserving high model performance on the test set for emotion recognition.



## **34. Seeing Isn't Believing: Context-Aware Adversarial Patch Synthesis via Conditional GAN**

cs.CV

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2509.22836v2) [paper-pdf](https://arxiv.org/pdf/2509.22836v2)

**Authors**: Roie Kazoom, Alon Goldberg, Hodaya Cohen, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a severe threat to deep neural networks, yet most existing approaches rely on unrealistic white-box assumptions, untargeted objectives, or produce visually conspicuous patches that limit real-world applicability. In this work, we introduce a novel framework for fully controllable adversarial patch generation, where the attacker can freely choose both the input image x and the target class y target, thereby dictating the exact misclassification outcome. Our method combines a generative U-Net design with Grad-CAM-guided patch placement, enabling semantic-aware localization that maximizes attack effectiveness while preserving visual realism. Extensive experiments across convolutional networks (DenseNet-121, ResNet-50) and vision transformers (ViT-B/16, Swin-B/16, among others) demonstrate that our approach achieves state-of-the-art performance across all settings, with attack success rates (ASR) and target-class success (TCS) consistently exceeding 99%.   Importantly, we show that our method not only outperforms prior white-box attacks and untargeted baselines, but also surpasses existing non-realistic approaches that produce detectable artifacts. By simultaneously ensuring realism, targeted control, and black-box applicability-the three most challenging dimensions of patch-based attacks-our framework establishes a new benchmark for adversarial robustness research, bridging the gap between theoretical attack strength and practical stealthiness.



## **35. Membership Inference Attack with Partial Features**

cs.LG

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2508.06244v2) [paper-pdf](https://arxiv.org/pdf/2508.06244v2)

**Authors**: Xurun Wang, Guangrui Liu, Xinjie Li, Haoyu He, Lin Yao, Zhongyun Hua, Weizhe Zhang

**Abstract**: Machine learning models are vulnerable to membership inference attack, which can be used to determine whether a given sample appears in the training data. Most existing methods assume the attacker has full access to the features of the target sample. This assumption, however, does not hold in many real-world scenarios where only partial features are available, thereby limiting the applicability of these methods. In this work, we introduce Partial Feature Membership Inference (PFMI), a scenario where the adversary observes only partial features of each sample and aims to infer whether this observed subset was present in the training set. To address this problem, we propose MRAD (Memory-guided Reconstruction and Anomaly Detection), a two-stage attack framework that works in both white-box and black-box settings. In the first stage, MRAD leverages the latent memory of the target model to reconstruct the unknown features of the sample. We observe that when the known features are absent from the training set, the reconstructed sample deviates significantly from the true data distribution. Consequently, in the second stage, we use anomaly detection algorithms to measure the deviation between the reconstructed sample and the training data distribution, thereby determining whether the known features belong to a member of the training set. Empirical results demonstrate that MRAD is effective across various datasets, and maintains compatibility with off-the-shelf anomaly detection techniques. For example, on STL-10, our attack exceeds an AUC of around 0.75 even with 60% of the missing features.



## **36. Simulated Ensemble Attack: Transferring Jailbreaks Across Fine-tuned Vision-Language Models**

cs.CV

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2508.01741v2) [paper-pdf](https://arxiv.org/pdf/2508.01741v2)

**Authors**: Ruofan Wang, Xin Wang, Yang Yao, Xuan Tong, Xingjun Ma

**Abstract**: Fine-tuning open-source Vision-Language Models (VLMs) creates a critical yet underexplored attack surface: vulnerabilities in the base VLM could be retained in fine-tuned variants, rendering them susceptible to transferable jailbreak attacks. To demonstrate this risk, we introduce the Simulated Ensemble Attack (SEA), a novel grey-box jailbreak method in which the adversary has full access to the base VLM but no knowledge of the fine-tuned target's weights or training configuration. To improve jailbreak transferability across fine-tuned VLMs, SEA combines two key techniques: Fine-tuning Trajectory Simulation (FTS) and Targeted Prompt Guidance (TPG). FTS generates transferable adversarial images by simulating the vision encoder's parameter shifts, while TPG is a textual strategy that steers the language decoder toward adversarially optimized outputs. Experiments on the Qwen2-VL family (2B and 7B) demonstrate that SEA achieves high transfer attack success rates exceeding 86.5% and toxicity rates near 49.5% across diverse fine-tuned variants, even those specifically fine-tuned to improve safety behaviors. Notably, while direct PGD-based image jailbreaks rarely transfer across fine-tuned VLMs, SEA reliably exploits inherited vulnerabilities from the base model, significantly enhancing transferability. These findings highlight an urgent need to safeguard fine-tuned proprietary VLMs against transferable vulnerabilities inherited from open-source foundations, motivating the development of holistic defenses across the entire model lifecycle.



## **37. BeDKD: Backdoor Defense based on Dynamic Knowledge Distillation and Directional Mapping Modulator**

cs.CR

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2508.01595v2) [paper-pdf](https://arxiv.org/pdf/2508.01595v2)

**Authors**: Zhengxian Wu, Juan Wen, Wanli Peng, Yinghan Zhou, Changtong dou, Yiming Xue

**Abstract**: Although existing backdoor defenses have gained success in mitigating backdoor attacks, they still face substantial challenges. In particular, most of them rely on large amounts of clean data to weaken the backdoor mapping but generally struggle with residual trigger effects, resulting in persistently high attack success rates (ASR). Therefore, in this paper, we propose a novel Backdoor defense method based on Directional mapping module and adversarial Knowledge Distillation (BeDKD), which balances the trade-off between defense effectiveness and model performance using a small amount of clean and poisoned data. We first introduce a directional mapping module to identify poisoned data, which destroys clean mapping while keeping backdoor mapping on a small set of flipped clean data. Then, the adversarial knowledge distillation is designed to reinforce clean mapping and suppress backdoor mapping through a cycle iteration mechanism between trust and punish distillations using clean and identified poisoned data. We conduct experiments to mitigate mainstream attacks on three datasets, and experimental results demonstrate that BeDKD surpasses the state-of-the-art defenses and reduces the ASR by 98% without significantly reducing the CACC. Our code are available in https://github.com/CAU-ISS-Lab/Backdoor-Attack-Defense-LLMs/tree/main/BeDKD.



## **38. Improving Large Language Model Safety with Contrastive Representation Learning**

cs.CL

EMNLP 2025 Main

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2506.11938v2) [paper-pdf](https://arxiv.org/pdf/2506.11938v2)

**Authors**: Samuel Simko, Mrinmaya Sachan, Bernhard Schölkopf, Zhijing Jin

**Abstract**: Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at https://github.com/samuelsimko/crl-llm-defense



## **39. SoK: Are Watermarks in LLMs Ready for Deployment?**

cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2506.05594v3) [paper-pdf](https://arxiv.org/pdf/2506.05594v3)

**Authors**: Kieu Dang, Phung Lai, NhatHai Phan, Yelong Shen, Ruoming Jin, Abdallah Khreishah, My T. Thai

**Abstract**: Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs.   To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment.



## **40. Towards Dataset Copyright Evasion Attack against Personalized Text-to-Image Diffusion Models**

cs.CV

Accepted by IEEE Transactions on Information Forensics and Security

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2505.02824v2) [paper-pdf](https://arxiv.org/pdf/2505.02824v2)

**Authors**: Kuofeng Gao, Yufei Zhu, Yiming Li, Jiawang Bai, Yong Yang, Zhifeng Li, Shu-Tao Xia

**Abstract**: Text-to-image (T2I) diffusion models enable high-quality image generation conditioned on textual prompts. However, fine-tuning these pre-trained models for personalization raises concerns about unauthorized dataset usage. To address this issue, dataset ownership verification (DOV) has recently been proposed, which embeds watermarks into fine-tuning datasets via backdoor techniques. These watermarks remain dormant on benign samples but produce owner-specified outputs when triggered. Despite its promise, the robustness of DOV against copyright evasion attacks (CEA) remains unexplored. In this paper, we investigate how adversaries can circumvent these mechanisms, enabling models trained on watermarked datasets to bypass ownership verification. We begin by analyzing the limitations of potential attacks achieved by backdoor removal, including TPD and T2IShield. In practice, TPD suffers from inconsistent effectiveness due to randomness, while T2IShield fails when watermarks are embedded as local image patches. To this end, we introduce CEAT2I, the first CEA specifically targeting DOV in T2I diffusion models. CEAT2I consists of three stages: (1) motivated by the observation that T2I models converge faster on watermarked samples with respect to intermediate features rather than training loss, we reliably detect watermarked samples; (2) we iteratively ablate tokens from the prompts of detected samples and monitor feature shifts to identify trigger tokens; and (3) we apply a closed-form concept erasure method to remove the injected watermarks. Extensive experiments demonstrate that CEAT2I effectively evades state-of-the-art DOV mechanisms while preserving model performance. The code is available at https://github.com/csyufei/CEAT2I.



## **41. AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models**

cs.CR

We encountered issues with the paper being hosted under my personal account, so we republished it under a different account associated with a university email, which makes updates and management easier. As a result, this version is a duplicate of arXiv:2511.02376

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2507.01020v2) [paper-pdf](https://arxiv.org/pdf/2507.01020v2)

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities to jailbreaking attacks: carefully crafted malicious inputs intended to circumvent safety guardrails and elicit harmful responses. As such, we present AutoAdv, a novel framework that automates adversarial prompt generation to systematically evaluate and expose vulnerabilities in LLM safety mechanisms. Our approach leverages a parametric attacker LLM to produce semantically disguised malicious prompts through strategic rewriting techniques, specialized system prompts, and optimized hyperparameter configurations. The primary contribution of our work is a dynamic, multi-turn attack methodology that analyzes failed jailbreak attempts and iteratively generates refined follow-up prompts, leveraging techniques such as roleplaying, misdirection, and contextual manipulation. We quantitatively evaluate attack success rate (ASR) using the StrongREJECT (arXiv:2402.10260 [cs.CL]) framework across sequential interaction turns. Through extensive empirical evaluation of state-of-the-art models--including ChatGPT, Llama, and DeepSeek--we reveal significant vulnerabilities, with our automated attacks achieving jailbreak success rates of up to 86% for harmful content generation. Our findings reveal that current safety mechanisms remain susceptible to sophisticated multi-turn attacks, emphasizing the urgent need for more robust defense strategies.



## **42. CAE-Net: Generalized Deepfake Image Detection using Convolution and Attention Mechanisms with Spatial and Frequency Domain Features**

cs.CV

Published in Journal of Visual Communication and Image Representation

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2502.10682v3) [paper-pdf](https://arxiv.org/pdf/2502.10682v3)

**Authors**: Anindya Bhattacharjee, Kaidul Islam, Kafi Anan, Ashir Intesher, Abrar Assaeem Fuad, Utsab Saha, Hafiz Imtiaz

**Abstract**: The spread of deepfakes poses significant security concerns, demanding reliable detection methods. However, diverse generation techniques and class imbalance in datasets create challenges. We propose CAE-Net, a Convolution- and Attention-based weighted Ensemble network combining spatial and frequency-domain features for effective deepfake detection. The architecture integrates EfficientNet, Data-Efficient Image Transformer (DeiT), and ConvNeXt with wavelet features to learn complementary representations. We evaluated CAE-Net on the diverse IEEE Signal Processing Cup 2025 (DF-Wild Cup) dataset, which has a 5:1 fake-to-real class imbalance. To address this, we introduce a multistage disjoint-subset training strategy, sequentially training the model on non-overlapping subsets of the fake class while retaining knowledge across stages. Our approach achieved $94.46\%$ accuracy and a $97.60\%$ AUC, outperforming conventional class-balancing methods. Visualizations confirm the network focuses on meaningful facial regions, and our ensemble design demonstrates robustness against adversarial attacks, positioning CAE-Net as a dependable and generalized deepfake detection framework.



## **43. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

cs.CL

Accepted by USENIX Security 2025

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2502.01386v3) [paper-pdf](https://arxiv.org/pdf/2502.01386v3)

**Authors**: Yuyang Gong, Zhuo Chen, Jiawei Liu, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.



## **44. When Should Selfish Miners Double-Spend?**

cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2501.03227v4) [paper-pdf](https://arxiv.org/pdf/2501.03227v4)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: Conventional double-spending attack models ignore the revenue losses stemming from the orphan blocks. On the other hand, selfish mining literature usually ignores the chance of the attacker to double-spend at no-cost in each attack cycle. In this paper, we give a rigorous stochastic analysis of an attack where the goal of the adversary is to double-spend while mining selfishly. To do so, we first combine stubborn and selfish mining attacks, i.e., construct a strategy where the attacker acts stubborn until its private branch reaches a certain length and then switches to act selfish. We provide the optimal stubbornness for each parameter regime. Next, we provide the maximum stubbornness that is still more profitable than honest mining and argue a connection between the level of stubbornness and the $k$-confirmation rule. We show that, at each attack cycle, if the level of stubbornness is higher than $k$, the adversary gets a free shot at double-spending. At each cycle, for a given stubbornness level, we rigorously formulate how great the probability of double-spending is. We further modify the attack in the stubborn regime in order to conceal the attack and increase the double-spending probability.



## **45. FlippedRAG: Black-Box Opinion Manipulation Adversarial Attacks to Retrieval-Augmented Generation Models**

cs.IR

Accepted by 32nd ACM Conference on Computer and Communications Security (ACM CCS 2025)

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2501.02968v6) [paper-pdf](https://arxiv.org/pdf/2501.02968v6)

**Authors**: Zhuo Chen, Yuyang Gong, Jiawei Liu, Miaokun Chen, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, Xiaozhong Liu

**Abstract**: Retrieval-Augmented Generation (RAG) enriches LLMs by dynamically retrieving external knowledge, reducing hallucinations and satisfying real-time information needs. While existing research mainly targets RAG's performance and efficiency, emerging studies highlight critical security concerns. Yet, current adversarial approaches remain limited, mostly addressing white-box scenarios or heuristic black-box attacks without fully investigating vulnerabilities in the retrieval phase. Additionally, prior works mainly focus on factoid Q&A tasks, their attacks lack complexity and can be easily corrected by advanced LLMs. In this paper, we investigate a more realistic and critical threat scenario: adversarial attacks intended for opinion manipulation against black-box RAG models, particularly on controversial topics. Specifically, we propose FlippedRAG, a transfer-based adversarial attack against black-box RAG systems. We first demonstrate that the underlying retriever of a black-box RAG system can be reverse-engineered, enabling us to train a surrogate retriever. Leveraging the surrogate retriever, we further craft target poisoning triggers, altering vary few documents to effectively manipulate both retrieval and subsequent generation. Extensive empirical results show that FlippedRAG substantially outperforms baseline methods, improving the average attack success rate by 16.7%. FlippedRAG achieves on average a 50% directional shift in the opinion polarity of RAG-generated responses, ultimately causing a notable 20% shift in user cognition. Furthermore, we evaluate the performance of several potential defensive measures, concluding that existing mitigation strategies remain insufficient against such sophisticated manipulation attacks. These results highlight an urgent need for developing innovative defensive solutions to ensure the security and trustworthiness of RAG systems.



## **46. Quantifying True Robustness: Synonymity-Weighted Similarity for Trustworthy XAI Evaluation**

cs.LG

10 pages, 2 figures, 6 tables. Changes to title, abstract and minor edits to the content as a result of acceptance to the 59th Hawaii International Conference on System Sciences

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2501.01516v2) [paper-pdf](https://arxiv.org/pdf/2501.01516v2)

**Authors**: Christopher Burger

**Abstract**: Adversarial attacks challenge the reliability of Explainable AI (XAI) by altering explanations while the model's output remains unchanged. The success of these attacks on text-based XAI is often judged using standard information retrieval metrics. We argue these measures are poorly suited in the evaluation of trustworthiness, as they treat all word perturbations equally while ignoring synonymity, which can misrepresent an attack's true impact. To address this, we apply synonymity weighting, a method that amends these measures by incorporating the semantic similarity of perturbed words. This produces more accurate vulnerability assessments and provides an important tool for assessing the robustness of AI systems. Our approach prevents the overestimation of attack success, leading to a more faithful understanding of an XAI system's true resilience against adversarial manipulation.



## **47. Improving Graph Neural Network Training, Defense and Hypergraph Partitioning via Adversarial Robustness Evaluation**

cs.LG

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2412.14738v10) [paper-pdf](https://arxiv.org/pdf/2412.14738v10)

**Authors**: Yongyu Wang

**Abstract**: Graph Neural Networks (GNNs) are a highly effective neural network architecture for processing graph-structured data. Unlike traditional neural networks that rely solely on the features of the data as input, GNNs leverage both the graph structure, which represents the relationships between data points, and the feature matrix of the data to optimize their feature representation. This unique capability enables GNNs to achieve superior performance across various tasks. However, it also makes GNNs more susceptible to noise and adversarial attacks from both the graph structure and data features, which can significantly increase the training difficulty and degrade their performance. Similarly, a hypergraph is a highly complex structure, and partitioning a hypergraph is a challenging task. This paper leverages spectral adversarial robustness evaluation to effectively address key challenges in complex-graph algorithms. By using spectral adversarial robustness evaluation to distinguish robust nodes from non-robust ones and treating them differently, we propose a training-set construction strategy that improves the training quality of GNNs. In addition, we develop algorithms to enhance both the adversarial robustness of GNNs and the performance of hypergraph partitioning. Experimental results show that this series of methods is highly effective.



## **48. Trust-free Personalized Decentralized Learning**

cs.LG

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2410.11378v2) [paper-pdf](https://arxiv.org/pdf/2410.11378v2)

**Authors**: Yawen Li, Yan Li, Junping Du, Yingxia Shao, Meiyu Liang, Guanhua Ye

**Abstract**: Personalized collaborative learning in federated settings faces a critical trade-off between customization and participant trust. Existing approaches typically rely on centralized coordinators or trusted peer groups, limiting their applicability in open, trust-averse environments. While recent decentralized methods explore anonymous knowledge sharing, they often lack global scalability and robust mechanisms against malicious peers. To bridge this gap, we propose TPFed, a \textit{Trust-free Personalized Decentralized Federated Learning} framework. TPFed replaces central aggregators with a blockchain-based bulletin board, enabling participants to dynamically select global communication partners based on Locality-Sensitive Hashing (LSH) and peer ranking. Crucially, we introduce an ``all-in-one'' knowledge distillation protocol that simultaneously handles knowledge transfer, model quality evaluation, and similarity verification via a public reference dataset. This design ensures secure, globally personalized collaboration without exposing local models or data. Extensive experiments demonstrate that TPFed significantly outperforms traditional federated baselines in both learning accuracy and system robustness against adversarial attacks.



## **49. One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models**

cs.CV

Accepted by ICCV-2025

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2406.05491v4) [paper-pdf](https://arxiv.org/pdf/2406.05491v4)

**Authors**: Hao Fang, Jiawei Kong, Wenbo Yu, Bin Chen, Jiawei Li, Hao Wu, Shutao Xia, Ke Xu

**Abstract**: Vision-Language Pre-training (VLP) models have exhibited unprecedented capability in many applications by taking full advantage of the multimodal alignment. However, previous studies have shown they are vulnerable to maliciously crafted adversarial samples. Despite recent success, these methods are generally instance-specific and require generating perturbations for each input sample. In this paper, we reveal that VLP models are also vulnerable to the instance-agnostic universal adversarial perturbation (UAP). Specifically, we design a novel Contrastive-training Perturbation Generator with Cross-modal conditions (C-PGC) to achieve the attack. In light that the pivotal multimodal alignment is achieved through the advanced contrastive learning technique, we devise to turn this powerful weapon against themselves, i.e., employ a malicious version of contrastive learning to train the C-PGC based on our carefully crafted positive and negative image-text pairs for essentially destroying the alignment relationship learned by VLP models. Besides, C-PGC fully utilizes the characteristics of Vision-and-Language (V+L) scenarios by incorporating both unimodal and cross-modal information as effective guidance. Extensive experiments show that C-PGC successfully forces adversarial samples to move away from their original area in the VLP model's feature space, thus essentially enhancing attacks across various victim models and V+L tasks. The GitHub repository is available at https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks.



## **50. Achieving Dalenius' Goal of Data Privacy with Practical Assumptions**

cs.CR

50 pages

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/1703.07474v6) [paper-pdf](https://arxiv.org/pdf/1703.07474v6)

**Authors**: Genqiang Wu, Xianyao Xia, Yeping He

**Abstract**: Current differential privacy frameworks face significant challenges: vulnerability to correlated data attacks and suboptimal utility-privacy tradeoffs. To address these limitations, we establish a novel information-theoretic foundation for Dalenius' privacy vision using Shannon's perfect secrecy framework. By leveraging the fundamental distinction between cryptographic systems (small secret keys) and privacy mechanisms (massive datasets), we replace differential privacy's restrictive independence assumption with practical partial knowledge constraints ($H(X) \geq b$).   We propose an information privacy framework achieving Dalenius security with quantifiable utility-privacy tradeoffs. Crucially, we prove that foundational mechanisms -- random response, exponential, and Gaussian channels -- satisfy Dalenius' requirements while preserving group privacy and composition properties. Our channel capacity analysis reduces infinite-dimensional evaluations to finite convex optimizations, enabling direct application of information-theoretic tools.   Empirical evaluation demonstrates that individual channel capacity (maximal information leakage of each individual) decreases with increasing entropy constraint $b$, and our framework achieves superior utility-privacy tradeoffs compared to classical differential privacy mechanisms under equivalent privacy guarantees. The framework is extended to computationally bounded adversaries via Yao's theory, unifying cryptographic and statistical privacy paradigms. Collectively, these contributions provide a theoretically grounded path toward practical, composable privacy -- subject to future resolution of the tradeoff characterization -- with enhanced resilience to correlation attacks.



