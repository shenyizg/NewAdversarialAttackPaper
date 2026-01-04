# Latest Adversarial Attack Papers
**update at 2026-01-04 08:58:46**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Towards Provably Secure Generative AI: Reliable Consensus Sampling**

cs.CR

**SubmitDate**: 2025-12-31    [abs](http://arxiv.org/abs/2512.24925v1) [paper-pdf](https://arxiv.org/pdf/2512.24925v1)

**Authors**: Yu Cui, Hang Fu, Sicheng Pan, Zhuoyu Sun, Yifei Liu, Yuhong Nie, Bo Ran, Baohan Huang, Xufeng Zhang, Haibin Zhang, Cong Zuo, Licheng Wang

**Abstract**: Existing research on generative AI security is primarily driven by mutually reinforcing attack and defense methodologies grounded in empirical experience. This dynamic frequently gives rise to previously unknown attacks that can circumvent current detection and prevention. This necessitates the continual updating of security mechanisms. Constructing generative AI with provable security and theoretically controllable risk is therefore necessary. Consensus Sampling (CS) is a promising algorithm toward provably secure AI. It controls risk by leveraging overlap in model output probabilities. However, we find that CS relies on frequent abstention to avoid unsafe outputs, which reduces utility. Moreover, CS becomes highly vulnerable when unsafe models are maliciously manipulated. To address these issues, we propose a new primitive called Reliable Consensus Sampling (RCS), that traces acceptance probability to tolerate extreme adversarial behaviors, improving robustness. RCS also eliminates the need for abstention entirely. We further develop a feedback algorithm to continuously and dynamically enhance the safety of RCS. We provide theoretical guarantees that RCS maintains a controllable risk threshold. Extensive experiments show that RCS significantly improves robustness and utility while maintaining latency comparable to CS. We hope this work contributes to the development of provably secure generative AI.



## **2. Projection-based Adversarial Attack using Physics-in-the-Loop Optimization for Monocular Depth Estimation**

cs.CV

**SubmitDate**: 2025-12-31    [abs](http://arxiv.org/abs/2512.24792v1) [paper-pdf](https://arxiv.org/pdf/2512.24792v1)

**Authors**: Takeru Kusakabe, Yudai Hirose, Mashiho Mukaida, Satoshi Ono

**Abstract**: Deep neural networks (DNNs) remain vulnerable to adversarial attacks that cause misclassification when specific perturbations are added to input images. This vulnerability also threatens the reliability of DNN-based monocular depth estimation (MDE) models, making robustness enhancement a critical need in practical applications. To validate the vulnerability of DNN-based MDE models, this study proposes a projection-based adversarial attack method that projects perturbation light onto a target object. The proposed method employs physics-in-the-loop (PITL) optimization -- evaluating candidate solutions in actual environments to account for device specifications and disturbances -- and utilizes a distributed covariance matrix adaptation evolution strategy. Experiments confirmed that the proposed method successfully created adversarial examples that lead to depth misestimations, resulting in parts of objects disappearing from the target scene.



## **3. HeteroHBA: A Generative Structure-Manipulating Backdoor Attack on Heterogeneous Graphs**

cs.LG

**SubmitDate**: 2025-12-31    [abs](http://arxiv.org/abs/2512.24665v1) [paper-pdf](https://arxiv.org/pdf/2512.24665v1)

**Authors**: Honglin Gao, Lan Zhao, Junhao Ren, Xiang Li, Gaoxi Xiao

**Abstract**: Heterogeneous graph neural networks (HGNNs) have achieved strong performance in many real-world applications, yet targeted backdoor poisoning on heterogeneous graphs remains less studied. We consider backdoor attacks for heterogeneous node classification, where an adversary injects a small set of trigger nodes and connections during training to force specific victim nodes to be misclassified into an attacker-chosen label at test time while preserving clean performance. We propose HeteroHBA, a generative backdoor framework that selects influential auxiliary neighbors for trigger attachment via saliency-based screening and synthesizes diverse trigger features and connection patterns to better match the local heterogeneous context. To improve stealthiness, we combine Adaptive Instance Normalization (AdaIN) with a Maximum Mean Discrepancy (MMD) loss to align the trigger feature distribution with benign statistics, thereby reducing detectability, and we optimize the attack with a bilevel objective that jointly promotes attack success and maintains clean accuracy. Experiments on multiple real-world heterogeneous graphs with representative HGNN architectures show that HeteroHBA consistently achieves higher attack success than prior backdoor baselines with comparable or smaller impact on clean accuracy; moreover, the attack remains effective under our heterogeneity-aware structural defense, CSD. These results highlight practical backdoor risks in heterogeneous graph learning and motivate the development of stronger defenses.



## **4. CPR: Causal Physiological Representation Learning for Robust ECG Analysis under Distribution Shifts**

cs.LG

**SubmitDate**: 2025-12-31    [abs](http://arxiv.org/abs/2512.24564v1) [paper-pdf](https://arxiv.org/pdf/2512.24564v1)

**Authors**: Shunbo Jia, Caizhi Liao

**Abstract**: Deep learning models for Electrocardiogram (ECG) diagnosis have achieved remarkable accuracy but exhibit fragility against adversarial perturbations, particularly Smooth Adversarial Perturbations (SAP) that mimic biological morphology. Existing defenses face a critical dilemma: Adversarial Training (AT) provides robustness but incurs a prohibitive computational burden, while certified methods like Randomized Smoothing (RS) introduce significant inference latency, rendering them impractical for real-time clinical monitoring. We posit that this vulnerability stems from the models' reliance on non-robust spurious correlations rather than invariant pathological features. To address this, we propose Causal Physiological Representation Learning (CPR). Unlike standard denoising approaches that operate without semantic constraints, CPR incorporates a Physiological Structural Prior within a causal disentanglement framework. By modeling ECG generation via a Structural Causal Model (SCM), CPR enforces a structural intervention that strictly separates invariant pathological morphology (P-QRS-T complex) from non-causal artifacts. Empirical results on PTB-XL demonstrate that CPR significantly outperforms standard clinical preprocessing methods. Specifically, under SAP attacks, CPR achieves an F1 score of 0.632, surpassing Median Smoothing (0.541 F1) by 9.1%. Crucially, CPR matches the certified robustness of Randomized Smoothing while maintaining single-pass inference efficiency, offering a superior trade-off between robustness, efficiency, and clinical interpretability.



## **5. Training-Free Color-Aware Adversarial Diffusion Sanitization for Diffusion Stegomalware Defense at Security Gateways**

cs.CR

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.24499v1) [paper-pdf](https://arxiv.org/pdf/2512.24499v1)

**Authors**: Vladimir Frants, Sos Agaian

**Abstract**: The rapid expansion of generative AI has normalized large-scale synthetic media creation, enabling new forms of covert communication. Recent generative steganography methods, particularly those based on diffusion models, can embed high-capacity payloads without fine-tuning or auxiliary decoders, creating significant challenges for detection and remediation. Coverless diffusion-based techniques are difficult to counter because they generate image carriers directly from secret data, enabling attackers to deliver stegomalware for command-and-control, payload staging, and data exfiltration while bypassing detectors that rely on cover-stego discrepancies. This work introduces Adversarial Diffusion Sanitization (ADS), a training-free defense for security gateways that neutralizes hidden payloads rather than detecting them. ADS employs an off-the-shelf pretrained denoiser as a differentiable proxy for diffusion-based decoders and incorporates a color-aware, quaternion-coupled update rule to reduce artifacts under strict distortion limits. Under a practical threat model and in evaluation against the state-of-the-art diffusion steganography method Pulsar, ADS drives decoder success rates to near zero with minimal perceptual impact. Results demonstrate that ADS provides a favorable security-utility trade-off compared to standard content transformations, offering an effective mitigation strategy against diffusion-driven steganography.



## **6. RAGPart & RAGMask: Retrieval-Stage Defenses Against Corpus Poisoning in Retrieval-Augmented Generation**

cs.IR

Published at AAAI 2026 Workshop on New Frontiers in Information Retrieval [Oral]

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.24268v1) [paper-pdf](https://arxiv.org/pdf/2512.24268v1)

**Authors**: Pankayaraj Pathmanathan, Michael-Andrei Panaitescu-Liess, Cho-Yu Jason Chiang, Furong Huang

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to enhance large language models (LLMs) with external knowledge, reducing hallucinations and compensating for outdated information. However, recent studies have exposed a critical vulnerability in RAG pipelines corpus poisoning where adversaries inject malicious documents into the retrieval corpus to manipulate model outputs. In this work, we propose two complementary retrieval-stage defenses: RAGPart and RAGMask. Our defenses operate directly on the retriever, making them computationally lightweight and requiring no modification to the generation model. RAGPart leverages the inherent training dynamics of dense retrievers, exploiting document partitioning to mitigate the effect of poisoned points. In contrast, RAGMask identifies suspicious tokens based on significant similarity shifts under targeted token masking. Across two benchmarks, four poisoning strategies, and four state-of-the-art retrievers, our defenses consistently reduce attack success rates while preserving utility under benign conditions. We further introduce an interpretable attack to stress-test our defenses. Our findings highlight the potential and limitations of retrieval-stage defenses, providing practical insights for robust RAG deployments.



## **7. How Would Oblivious Memory Boost Graph Analytics on Trusted Processors?**

cs.CR

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.24255v1) [paper-pdf](https://arxiv.org/pdf/2512.24255v1)

**Authors**: Jiping Yu, Xiaowei Zhu, Kun Chen, Guanyu Feng, Yunyi Chen, Xiaoyu Fan, Wenguang Chen

**Abstract**: Trusted processors provide a way to perform joint computations while preserving data privacy. To overcome the performance degradation caused by data-oblivious algorithms to prevent information leakage, we explore the benefits of oblivious memory (OM) integrated in processors, to which the accesses are unobservable by adversaries. We focus on graph analytics, an important application vulnerable to access-pattern attacks. With a co-design between storage structure and algorithms, our prototype system is 100x faster than baselines given an OM sized around the per-core cache which can be implemented on existing processors with negligible overhead. This gives insights into equipping trusted processors with OM.



## **8. Guided Diffusion-based Generation of Adversarial Objects for Real-World Monocular Depth Estimation Attacks**

cs.CV

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.24111v1) [paper-pdf](https://arxiv.org/pdf/2512.24111v1)

**Authors**: Yongtao Chen, Yanbo Wang, Wentao Zhao, Guole Shen, Tianchen Deng, Jingchuan Wang

**Abstract**: Monocular Depth Estimation (MDE) serves as a core perception module in autonomous driving systems, but it remains highly susceptible to adversarial attacks. Errors in depth estimation may propagate through downstream decision making and influence overall traffic safety. Existing physical attacks primarily rely on texture-based patches, which impose strict placement constraints and exhibit limited realism, thereby reducing their effectiveness in complex driving environments. To overcome these limitations, this work introduces a training-free generative adversarial attack framework that generates naturalistic, scene-consistent adversarial objects via a diffusion-based conditional generation process. The framework incorporates a Salient Region Selection module that identifies regions most influential to MDE and a Jacobian Vector Product Guidance mechanism that steers adversarial gradients toward update directions supported by the pre-trained diffusion model. This formulation enables the generation of physically plausible adversarial objects capable of inducing substantial adversarial depth shifts. Extensive digital and physical experiments demonstrate that our method significantly outperforms existing attacks in effectiveness, stealthiness, and physical deployability, underscoring its strong practical implications for autonomous driving safety assessment.



## **9. Jailbreaking Attacks vs. Content Safety Filters: How Far Are We in the LLM Safety Arms Race?**

cs.CR

26 pages,11 tables, 7 figures

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.24044v1) [paper-pdf](https://arxiv.org/pdf/2512.24044v1)

**Authors**: Yuan Xin, Dingfan Chen, Linyi Yang, Michael Backes, Xiao Zhang

**Abstract**: As large language models (LLMs) are increasingly deployed, ensuring their safe use is paramount. Jailbreaking, adversarial prompts that bypass model alignment to trigger harmful outputs, present significant risks, with existing studies reporting high success rates in evading common LLMs. However, previous evaluations have focused solely on the models, neglecting the full deployment pipeline, which typically incorporates additional safety mechanisms like content moderation filters. To address this gap, we present the first systematic evaluation of jailbreak attacks targeting LLM safety alignment, assessing their success across the full inference pipeline, including both input and output filtering stages. Our findings yield two key insights: first, nearly all evaluated jailbreak techniques can be detected by at least one safety filter, suggesting that prior assessments may have overestimated the practical success of these attacks; second, while safety filters are effective in detection, there remains room to better balance recall and precision to further optimize protection and user experience. We highlight critical gaps and call for further refinement of detection accuracy and usability in LLM safety systems.



## **10. RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress**

cs.CR

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.23995v1) [paper-pdf](https://arxiv.org/pdf/2512.23995v1)

**Authors**: Ruixuan Huang, Qingyue Wang, Hantao Huang, Yudong Gao, Dong Chen, Shuai Wang, Wei Wang

**Abstract**: Mixture-of-Experts architectures have become the standard for scaling large language models due to their superior parameter efficiency. To accommodate the growing number of experts in practice, modern inference systems commonly adopt expert parallelism to distribute experts across devices. However, the absence of explicit load balancing constraints during inference allows adversarial inputs to trigger severe routing concentration. We demonstrate that out-of-distribution prompts can manipulate the routing strategy such that all tokens are consistently routed to the same set of top-$k$ experts, which creates computational bottlenecks on certain devices while forcing others to idle. This converts an efficiency mechanism into a denial-of-service attack vector, leading to violations of service-level agreements for time to first token. We propose RepetitionCurse, a low-cost black-box strategy to exploit this vulnerability. By identifying a universal flaw in MoE router behavior, RepetitionCurse constructs adversarial prompts using simple repetitive token patterns in a model-agnostic manner. On widely deployed MoE models like Mixtral-8x7B, our method increases end-to-end inference latency by 3.063x, degrading service availability significantly.



## **11. T2VAttack: Adversarial Attack on Text-to-Video Diffusion Models**

cs.CV

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.23953v1) [paper-pdf](https://arxiv.org/pdf/2512.23953v1)

**Authors**: Changzhen Li, Yuecong Min, Jie Zhang, Zheng Yuan, Shiguang Shan, Xilin Chen

**Abstract**: The rapid evolution of Text-to-Video (T2V) diffusion models has driven remarkable advancements in generating high-quality, temporally coherent videos from natural language descriptions. Despite these achievements, their vulnerability to adversarial attacks remains largely unexplored. In this paper, we introduce T2VAttack, a comprehensive study of adversarial attacks on T2V diffusion models from both semantic and temporal perspectives. Considering the inherently dynamic nature of video data, we propose two distinct attack objectives: a semantic objective to evaluate video-text alignment and a temporal objective to assess the temporal dynamics. To achieve an effective and efficient attack process, we propose two adversarial attack methods: (i) T2VAttack-S, which identifies semantically or temporally critical words in prompts and replaces them with synonyms via greedy search, and (ii) T2VAttack-I, which iteratively inserts optimized words with minimal perturbation to the prompt. By combining these objectives and strategies, we conduct a comprehensive evaluation on the adversarial robustness of several state-of-the-art T2V models, including ModelScope, CogVideoX, Open-Sora, and HunyuanVideo. Our experiments reveal that even minor prompt modifications, such as the substitution or insertion of a single word, can cause substantial degradation in semantic fidelity and temporal dynamics, highlighting critical vulnerabilities in current T2V diffusion models.



## **12. Breaking Audio Large Language Models by Attacking Only the Encoder: A Universal Targeted Latent-Space Audio Attack**

cs.SD

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23881v1) [paper-pdf](https://arxiv.org/pdf/2512.23881v1)

**Authors**: Roee Ziv, Raz Lapid, Moshe Sipper

**Abstract**: Audio-language models combine audio encoders with large language models to enable multimodal reasoning, but they also introduce new security vulnerabilities. We propose a universal targeted latent space attack, an encoder-level adversarial attack that manipulates audio latent representations to induce attacker-specified outputs in downstream language generation. Unlike prior waveform-level or input-specific attacks, our approach learns a universal perturbation that generalizes across inputs and speakers and does not require access to the language model. Experiments on Qwen2-Audio-7B-Instruct demonstrate consistently high attack success rates with minimal perceptual distortion, revealing a critical and previously underexplored attack surface at the encoder level of multimodal systems.



## **13. Adversarial Lens: Exploiting Attention Layers to Generate Adversarial Examples for Evaluation**

cs.CL

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23837v1) [paper-pdf](https://arxiv.org/pdf/2512.23837v1)

**Authors**: Kaustubh Dhole

**Abstract**: Recent advances in mechanistic interpretability suggest that intermediate attention layers encode token-level hypotheses that are iteratively refined toward the final output. In this work, we exploit this property to generate adversarial examples directly from attention-layer token distributions. Unlike prompt-based or gradient-based attacks, our approach leverages model-internal token predictions, producing perturbations that are both plausible and internally consistent with the model's own generation process. We evaluate whether tokens extracted from intermediate layers can serve as effective adversarial perturbations for downstream evaluation tasks. We conduct experiments on argument quality assessment using the ArgQuality dataset, with LLaMA-3.1-Instruct-8B serving as both the generator and evaluator. Our results show that attention-based adversarial examples lead to measurable drops in evaluation performance while remaining semantically similar to the original inputs. However, we also observe that substitutions drawn from certain layers and token positions can introduce grammatical degradation, limiting their practical effectiveness. Overall, our findings highlight both the promise and current limitations of using intermediate-layer representations as a principled source of adversarial examples for stress-testing LLM-based evaluation pipelines.



## **14. Zero-Trust Agentic Federated Learning for Secure IIoT Defense Systems**

cs.LG

9 Pages and 6 figures, Submitted in conference 2nd IEEE Conference on Secure and Trustworthy Cyber Infrastructure for IoT and Microelectronics, Houston TX, USA

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23809v1) [paper-pdf](https://arxiv.org/pdf/2512.23809v1)

**Authors**: Samaresh Kumar Singh, Joyjit Roy, Martin So

**Abstract**: Recent attacks on critical infrastructure, including the 2021 Oldsmar water treatment breach and 2023 Danish energy sector compromises, highlight urgent security gaps in Industrial IoT (IIoT) deployments. While Federated Learning (FL) enables privacy-preserving collaborative intrusion detection, existing frameworks remain vulnerable to Byzantine poisoning attacks and lack robust agent authentication. We propose Zero-Trust Agentic Federated Learning (ZTA-FL), a defense in depth framework combining: (1) TPM-based cryptographic attestation achieving less than 0.0000001 false acceptance rate, (2) a novel SHAP-weighted aggregation algorithm providing explainable Byzantine detection under non-IID conditions with theoretical guarantees, and (3) privacy-preserving on-device adversarial training. Comprehensive experiments across three IDS benchmarks (Edge-IIoTset, CIC-IDS2017, UNSW-NB15) demonstrate that ZTA-FL achieves 97.8 percent detection accuracy, 93.2 percent accuracy under 30 percent Byzantine attacks (outperforming FLAME by 3.1 percent, p less than 0.01), and 89.3 percent adversarial robustness while reducing communication overhead by 34 percent. We provide theoretical analysis, failure mode characterization, and release code for reproducibility.



## **15. Multilingual Hidden Prompt Injection Attacks on LLM-Based Academic Reviewing**

cs.CL

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23684v1) [paper-pdf](https://arxiv.org/pdf/2512.23684v1)

**Authors**: Panagiotis Theocharopoulos, Ajinkya Kulkarni, Mathew Magimai. -Doss

**Abstract**: Large language models (LLMs) are increasingly considered for use in high-impact workflows, including academic peer review. However, LLMs are vulnerable to document-level hidden prompt injection attacks. In this work, we construct a dataset of approximately 500 real academic papers accepted to ICML and evaluate the effect of embedding hidden adversarial prompts within these documents. Each paper is injected with semantically equivalent instructions in four different languages and reviewed using an LLM. We find that prompt injection induces substantial changes in review scores and accept/reject decisions for English, Japanese, and Chinese injections, while Arabic injections produce little to no effect. These results highlight the susceptibility of LLM-based reviewing systems to document-level prompt injection and reveal notable differences in vulnerability across languages.



## **16. RobustMask: Certified Robustness against Adversarial Neural Ranking Attack via Randomized Masking**

cs.CR

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23307v1) [paper-pdf](https://arxiv.org/pdf/2512.23307v1)

**Authors**: Jiawei Liu, Zhuo Chen, Rui Zhu, Miaokun Chen, Yuyang Gong, Wei Lu, Xiaofeng Wang

**Abstract**: Neural ranking models have achieved remarkable progress and are now widely deployed in real-world applications such as Retrieval-Augmented Generation (RAG). However, like other neural architectures, they remain vulnerable to adversarial manipulations: subtle character-, word-, or phrase-level perturbations can poison retrieval results and artificially promote targeted candidates, undermining the integrity of search engines and downstream systems. Existing defenses either rely on heuristics with poor generalization or on certified methods that assume overly strong adversarial knowledge, limiting their practical use. To address these challenges, we propose RobustMask, a novel defense that combines the context-prediction capability of pretrained language models with a randomized masking-based smoothing mechanism. Our approach strengthens neural ranking models against adversarial perturbations at the character, word, and phrase levels. Leveraging both the pairwise comparison ability of ranking models and probabilistic statistical analysis, we provide a theoretical proof of RobustMask's certified top-K robustness. Extensive experiments further demonstrate that RobustMask successfully certifies over 20% of candidate documents within the top-10 ranking positions against adversarial perturbations affecting up to 30% of their content. These results highlight the effectiveness of RobustMask in enhancing the adversarial robustness of neural ranking models, marking a significant step toward providing stronger security guarantees for real-world retrieval systems.



## **17. It's a TRAP! Task-Redirecting Agent Persuasion Benchmark for Web Agents**

cs.HC

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23128v1) [paper-pdf](https://arxiv.org/pdf/2512.23128v1)

**Authors**: Karolina Korgul, Yushi Yang, Arkadiusz Drohomirecki, Piotr Błaszczyk, Will Howard, Lukas Aichberger, Chris Russell, Philip H. S. Torr, Adam Mahdi, Adel Bibi

**Abstract**: Web-based agents powered by large language models are increasingly used for tasks such as email management or professional networking. Their reliance on dynamic web content, however, makes them vulnerable to prompt injection attacks: adversarial instructions hidden in interface elements that persuade the agent to divert from its original task. We introduce the Task-Redirecting Agent Persuasion Benchmark (TRAP), an evaluation for studying how persuasion techniques misguide autonomous web agents on realistic tasks. Across six frontier models, agents are susceptible to prompt injection in 25\% of tasks on average (13\% for GPT-5 to 43\% for DeepSeek-R1), with small interface or contextual changes often doubling success rates and revealing systemic, psychologically driven vulnerabilities in web-based agents. We also provide a modular social-engineering injection framework with controlled experiments on high-fidelity website clones, allowing for further benchmark expansion.



## **18. DECEPTICON: How Dark Patterns Manipulate Web Agents**

cs.CR

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2512.22894v1) [paper-pdf](https://arxiv.org/pdf/2512.22894v1)

**Authors**: Phil Cuvin, Hao Zhu, Diyi Yang

**Abstract**: Deceptive UI designs, widely instantiated across the web and commonly known as dark patterns, manipulate users into performing actions misaligned with their goals. In this paper, we show that dark patterns are highly effective in steering agent trajectories, posing a significant risk to agent robustness. To quantify this risk, we introduce DECEPTICON, an environment for testing individual dark patterns in isolation. DECEPTICON includes 700 web navigation tasks with dark patterns -- 600 generated tasks and 100 real-world tasks, designed to measure instruction-following success and dark pattern effectiveness. Across state-of-the-art agents, we find dark patterns successfully steer agent trajectories towards malicious outcomes in over 70% of tested generated and real-world tasks -- compared to a human average of 31%. Moreover, we find that dark pattern effectiveness correlates positively with model size and test-time reasoning, making larger, more capable models more susceptible. Leading countermeasures against adversarial attacks, including in-context prompting and guardrail models, fail to consistently reduce the success rate of dark pattern interventions. Our findings reveal dark patterns as a latent and unmitigated risk to web agents, highlighting the urgent need for robust defenses against manipulative designs.



## **19. Adaptive Trust Consensus for Blockchain IoT: Comparing RL, DRL, and MARL Against Naive, Collusive, Adaptive, Byzantine, and Sleeper Attacks**

cs.CR

34 pages, 19 figures, 10 tables. Code available at https://github.com/soham-padia/blockchain-iot-trust

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2512.22860v1) [paper-pdf](https://arxiv.org/pdf/2512.22860v1)

**Authors**: Soham Padia, Dhananjay Vaidya, Ramchandra Mangrulkar

**Abstract**: Securing blockchain-enabled IoT networks against sophisticated adversarial attacks remains a critical challenge. This paper presents a trust-based delegated consensus framework integrating Fully Homomorphic Encryption (FHE) with Attribute-Based Access Control (ABAC) for privacy-preserving policy evaluation, combined with learning-based defense mechanisms. We systematically compare three reinforcement learning approaches -- tabular Q-learning (RL), Deep RL with Dueling Double DQN (DRL), and Multi-Agent RL (MARL) -- against five distinct attack families: Naive Malicious Attack (NMA), Collusive Rumor Attack (CRA), Adaptive Adversarial Attack (AAA), Byzantine Fault Injection (BFI), and Time-Delayed Poisoning (TDP). Experimental results on a 16-node simulated IoT network reveal significant performance variations: MARL achieves superior detection under collusive attacks (F1=0.85 vs. DRL's 0.68 and RL's 0.50), while DRL and MARL both attain perfect detection (F1=1.00) against adaptive attacks where RL fails (F1=0.50). All agents successfully defend against Byzantine attacks (F1=1.00). Most critically, the Time-Delayed Poisoning attack proves catastrophic for all agents, with F1 scores dropping to 0.11-0.16 after sleeper activation, demonstrating the severe threat posed by trust-building adversaries. Our findings indicate that coordinated multi-agent learning provides measurable advantages for defending against sophisticated trust manipulation attacks in blockchain IoT environments.



## **20. Reach-Avoid Differential game with Reachability Analysis for UAVs: A decomposition approach**

eess.SY

Paper version accepted to the Journal of Guidance, Control, and Dynamics (JGCD)

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2512.22793v1) [paper-pdf](https://arxiv.org/pdf/2512.22793v1)

**Authors**: Minh Bui, Simon Monckton, Mo Chen

**Abstract**: Reach-avoid (RA) games have significant applications in security and defense, particularly for unmanned aerial vehicles (UAVs). These problems are inherently challenging due to the need to consider obstacles, consider the adversarial nature of opponents, ensure optimality, and account for nonlinear dynamics. Hamilton-Jacobi (HJ) reachability analysis has emerged as a powerful tool for tackling these challenges; however, while it has been applied to games involving two spatial dimensions, directly extending this approach to three spatial dimensions is impossible due to high dimensionality. On the other hand, alternative approaches for solving RA games lack the generality to consider games with three spatial dimensions involving agents with non-trivial system dynamics. In this work, we propose a novel framework for dimensionality reduction by decomposing the problem into a horizontal RA sub-game and a vertical RA sub-game. We then solve each sub-game using HJ reachability analysis and consider second-order dynamics that account for the defender's acceleration. To reconstruct the solution to the original RA game from the sub-games, we introduce a HJ-based tracking control algorithm in each sub-game that not only guarantees capture of the attacker but also tracking of the attacker thereafter. We prove the conditions under which the capture guarantees are maintained. The effectiveness of our approach is demonstrated via numerical simulations, showing that the decomposition maintains optimality and guarantees in the original problem. Our methods are also validated in a Gazebo physics simulator, achieving successful capture of quadrotors in three spatial dimensions space for the first time to the best of our knowledge.



## **21. Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks**

cs.LG

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2512.22522v1) [paper-pdf](https://arxiv.org/pdf/2512.22522v1)

**Authors**: Jihang Wang, Dongcheng Zhao, Ruolin Chen, Qian Zhang, Yi Zeng

**Abstract**: Spiking Neural Networks (SNNs) utilize spike-based activations to mimic the brain's energy-efficient information processing. However, the binary and discontinuous nature of spike activations causes vanishing gradients, making adversarial robustness evaluation via gradient descent unreliable. While improved surrogate gradient methods have been proposed, their effectiveness under strong adversarial attacks remains unclear. We propose a more reliable framework for evaluating SNN adversarial robustness. We theoretically analyze the degree of gradient vanishing in surrogate gradients and introduce the Adaptive Sharpness Surrogate Gradient (ASSG), which adaptively evolves the shape of the surrogate function according to the input distribution during attack iterations, thereby enhancing gradient accuracy while mitigating gradient vanishing. In addition, we design an adversarial attack with adaptive step size under the $L_\infty$ constraint-Stable Adaptive Projected Gradient Descent (SA-PGD), achieving faster and more stable convergence under imprecise gradients. Extensive experiments show that our approach substantially increases attack success rates across diverse adversarial training schemes, SNN architectures and neuron models, providing a more generalized and reliable evaluation of SNN adversarial robustness. The experimental results further reveal that the robustness of current SNNs has been significantly overestimated and highlighting the need for more dependable adversarial training methods.



## **22. NOWA: Null-space Optical Watermark for Invisible Capture Fingerprinting and Tamper Localization**

cs.CR

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2512.22501v1) [paper-pdf](https://arxiv.org/pdf/2512.22501v1)

**Authors**: Edwin Vargas

**Abstract**: Ensuring the authenticity and ownership of digital images is increasingly challenging as modern editing tools enable highly realistic forgeries. Existing image protection systems mainly rely on digital watermarking, which is susceptible to sophisticated digital attacks. To address this limitation, we propose a hybrid optical-digital framework that incorporates physical authentication cues during image formation and preserves them through a learned reconstruction process. At the optical level, a phase mask in the camera aperture produces a Null-space Optical Watermark (NOWA) that lies in the Null Space of the imaging operator and therefore remains invisible in the captured image. Then, a Null-Space Network (NSN) performs measurement-consistent reconstruction that delivers high-quality protected images while preserving the NOWA signature. The proposed design enables tamper localization by projecting the image onto the camera's null space and detecting pixel-level inconsistencies. Our design preserves perceptual quality, resists common degradations such as compression, and establishes a structural security asymmetry: without access to the optical or NSN parameters, adversaries cannot forge the NOWA signature. Experiments with simulations and a prototype camera demonstrate competitive performance in terms of image quality preservation, and tamper localization accuracy compared to state-of-the-art digital watermarking and learning-based authentication methods.



## **23. PHANTOM: Physics-Aware Adversarial Attacks against Federated Learning-Coordinated EV Charging Management System**

cs.ET

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2512.22381v1) [paper-pdf](https://arxiv.org/pdf/2512.22381v1)

**Authors**: Mohammad Zakaria Haider, Amit Kumar Podder, Prabin Mali, Aranya Chakrabortty, Sumit Paudyal, Mohammad Ashiqur Rahman

**Abstract**: The rapid deployment of electric vehicle charging stations (EVCS) within distribution networks necessitates intelligent and adaptive control to maintain the grid's resilience and reliability. In this work, we propose PHANTOM, a physics-aware adversarial network that is trained and optimized through a multi-agent reinforcement learning model. PHANTOM integrates a physics-informed neural network (PINN) enabled by federated learning (FL) that functions as a digital twin of EVCS-integrated systems, ensuring physically consistent modeling of operational dynamics and constraints. Building on this digital twin, we construct a multi-agent RL environment that utilizes deep Q-networks (DQN) and soft actor-critic (SAC) methods to derive adversarial false data injection (FDI) strategies capable of bypassing conventional detection mechanisms. To examine the broader grid-level consequences, a transmission and distribution (T and D) dual simulation platform is developed, allowing us to capture cascading interactions between EVCS disturbances at the distribution level and the operations of the bulk transmission system. Results demonstrate how learned attack policies disrupt load balancing and induce voltage instabilities that propagate across T and D boundaries. These findings highlight the critical need for physics-aware cybersecurity to ensure the resilience of large-scale vehicle-grid integration.



## **24. Scaling Adversarial Training via Data Selection**

cs.LG

6 pages. Conference workshop paper

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2512.22069v1) [paper-pdf](https://arxiv.org/pdf/2512.22069v1)

**Authors**: Youran Ye, Dejin Wang, Ajinkya Bhandare

**Abstract**: Projected Gradient Descent (PGD) is a strong and widely used first-order adversarial attack, yet its computational cost scales poorly, as all training samples undergo identical iterative inner-loop optimization despite contributing unequally to robustness. Motivated by this inefficiency, we propose \emph{Selective Adversarial Training}, which perturbs only a subset of critical samples in each minibatch. Specifically, we introduce two principled selection criteria: (1) margin-based sampling, which prioritizes samples near the decision boundary, and (2) gradient-matching sampling, which selects samples whose gradients align with the dominant batch optimization direction. Adversarial examples are generated only for the selected subset, while the remaining samples are trained cleanly using a mixed objective. Experiments on MNIST and CIFAR-10 show that the proposed methods achieve robustness comparable to, or even exceeding, full PGD adversarial training, while reducing adversarial computation by up to $50\%$, demonstrating that informed sample selection is sufficient for scalable adversarial robustness.



## **25. Few Tokens Matter: Entropy Guided Attacks on Vision-Language Models**

cs.CV

19 Pages,11 figures,8 tables

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2512.21815v1) [paper-pdf](https://arxiv.org/pdf/2512.21815v1)

**Authors**: Mengqi He, Xinyu Tian, Xin Shen, Jinhong Ni, Shu Zou, Zhaoyuan Yang, Jing Zhang

**Abstract**: Vision-language models (VLMs) achieve remarkable performance but remain vulnerable to adversarial attacks. Entropy, a measure of model uncertainty, is strongly correlated with the reliability of VLM. Prior entropy-based attacks maximize uncertainty at all decoding steps, implicitly assuming that every token contributes equally to generation instability. We show instead that a small fraction (about 20%) of high-entropy tokens, i.e., critical decision points in autoregressive generation, disproportionately governs output trajectories. By concentrating adversarial perturbations on these positions, we achieve semantic degradation comparable to global methods while using substantially smaller budgets. More importantly, across multiple representative VLMs, such selective attacks convert 35-49% of benign outputs into harmful ones, exposing a more critical safety risk. Remarkably, these vulnerable high-entropy forks recur across architecturally diverse VLMs, enabling feasible transferability (17-26% harmful rates on unseen targets). Motivated by these findings, we propose Entropy-bank Guided Adversarial attacks (EGA), which achieves competitive attack success rates (93-95%) alongside high harmful conversion, thereby revealing new weaknesses in current VLM safety mechanisms.



## **26. LLM-Driven Feature-Level Adversarial Attacks on Android Malware Detectors**

cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21404v1) [paper-pdf](https://arxiv.org/pdf/2512.21404v1)

**Authors**: Tianwei Lan, Farid Naït-Abdesselam

**Abstract**: The rapid growth in both the scale and complexity of Android malware has driven the widespread adoption of machine learning (ML) techniques for scalable and accurate malware detection. Despite their effectiveness, these models remain vulnerable to adversarial attacks that introduce carefully crafted feature-level perturbations to evade detection while preserving malicious functionality. In this paper, we present LAMLAD, a novel adversarial attack framework that exploits the generative and reasoning capabilities of large language models (LLMs) to bypass ML-based Android malware classifiers. LAMLAD employs a dual-agent architecture composed of an LLM manipulator, which generates realistic and functionality-preserving feature perturbations, and an LLM analyzer, which guides the perturbation process toward successful evasion. To improve efficiency and contextual awareness, LAMLAD integrates retrieval-augmented generation (RAG) into the LLM pipeline. Focusing on Drebin-style feature representations, LAMLAD enables stealthy and high-confidence attacks against widely deployed Android malware detection systems. We evaluate LAMLAD against three representative ML-based Android malware detectors and compare its performance with two state-of-the-art adversarial attack methods. Experimental results demonstrate that LAMLAD achieves an attack success rate (ASR) of up to 97%, requiring on average only three attempts per adversarial sample, highlighting its effectiveness, efficiency, and adaptability in practical adversarial settings. Furthermore, we propose an adversarial training-based defense strategy that reduces the ASR by more than 30% on average, significantly enhancing model robustness against LAMLAD-style attacks.



## **27. CoTDeceptor:Adversarial Code Obfuscation Against CoT-Enhanced LLM Code Agents**

cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21250v1) [paper-pdf](https://arxiv.org/pdf/2512.21250v1)

**Authors**: Haoyang Li, Mingjin Li, Jinxin Zuo, Siqi Li, Xiao Li, Hao Wu, Yueming Lu, Xiaochuan He

**Abstract**: LLM-based code agents(e.g., ChatGPT Codex) are increasingly deployed as detector for code review and security auditing tasks. Although CoT-enhanced LLM vulnerability detectors are believed to provide improved robustness against obfuscated malicious code, we find that their reasoning chains and semantic abstraction processes exhibit exploitable systematic weaknesses.This allows attackers to covertly embed malicious logic, bypass code review, and propagate backdoored components throughout real-world software supply chains.To investigate this issue, we present CoTDeceptor, the first adversarial code obfuscation framework targeting CoT-enhanced LLM detectors. CoTDeceptor autonomously constructs evolving, hard-to-reverse multi-stage obfuscation strategy chains that effectively disrupt CoT-driven detection logic.We obtained malicious code provided by security enterprise, experimental results demonstrate that CoTDeceptor achieves stable and transferable evasion performance against state-of-the-art LLMs and vulnerability detection agents. CoTDeceptor bypasses 14 out of 15 vulnerability categories, compared to only 2 bypassed by prior methods. Our findings highlight potential risks in real-world software supply chains and underscore the need for more robust and interpretable LLM-powered security analysis systems.



## **28. Improving the Convergence Rate of Ray Search Optimization for Query-Efficient Hard-Label Attacks**

cs.LG

Published at AAAI 2026 (Oral). This version corresponds to the conference proceedings; v2 will include the appendix

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21241v1) [paper-pdf](https://arxiv.org/pdf/2512.21241v1)

**Authors**: Xinjie Xu, Shuyu Cheng, Dongwei Xu, Qi Xuan, Chen Ma

**Abstract**: In hard-label black-box adversarial attacks, where only the top-1 predicted label is accessible, the prohibitive query complexity poses a major obstacle to practical deployment. In this paper, we focus on optimizing a representative class of attacks that search for the optimal ray direction yielding the minimum $\ell_2$-norm perturbation required to move a benign image into the adversarial region. Inspired by Nesterov's Accelerated Gradient (NAG), we propose a momentum-based algorithm, ARS-OPT, which proactively estimates the gradient with respect to a future ray direction inferred from accumulated momentum. We provide a theoretical analysis of its convergence behavior, showing that ARS-OPT enables more accurate directional updates and achieves faster, more stable optimization. To further accelerate convergence, we incorporate surrogate-model priors into ARS-OPT's gradient estimation, resulting in PARS-OPT with enhanced performance. The superiority of our approach is supported by theoretical guarantees under standard assumptions. Extensive experiments on ImageNet and CIFAR-10 demonstrate that our method surpasses 13 state-of-the-art approaches in query efficiency.



## **29. Time-Bucketed Balance Records: Bounded-Storage Ephemeral Tokens for Resource-Constrained Systems**

cs.DS

14 pages, 1 figure, 1 Algorithm, 3 Theorems

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.20962v1) [paper-pdf](https://arxiv.org/pdf/2512.20962v1)

**Authors**: Shaun Scovil, Bhargav Chickmagalur Nanjundappa

**Abstract**: Fungible tokens with time-to-live (TTL) semantics require tracking individual expiration times for each deposited unit. A naive implementation creates a new balance record per deposit, leading to unbounded storage growth and vulnerability to denial-of-service attacks. We present time-bucketed balance records, a data structure that bounds storage to O(k) records per account while guaranteeing that tokens never expire before their configured TTL. Our approach discretizes time into k buckets, coalescing deposits within the same bucket to limit unique expiration timestamps. We prove three key properties: (1) storage is bounded by k+1 records regardless of deposit frequency, (2) actual expiration time is always at least the configured TTL, and (3) adversaries cannot increase a victim's operation cost beyond O(k)[amortized] worst case. We provide a reference implementation in Solidity with measured gas costs demonstrating practical efficiency.



## **30. The Imitation Game: Using Large Language Models as Chatbots to Combat Chat-Based Cybercrimes**

cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21371v1) [paper-pdf](https://arxiv.org/pdf/2512.21371v1)

**Authors**: Yifan Yao, Baojuan Wang, Jinhao Duan, Kaidi Xu, ChuanKai Guo, Zhibo Eric Sun, Yue Zhang

**Abstract**: Chat-based cybercrime has emerged as a pervasive threat, with attackers leveraging real-time messaging platforms to conduct scams that rely on trust-building, deception, and psychological manipulation. Traditional defense mechanisms, which operate on static rules or shallow content filters, struggle to identify these conversational threats, especially when attackers use multimedia obfuscation and context-aware dialogue.   In this work, we ask a provocative question inspired by the classic Imitation Game: Can machines convincingly pose as human victims to turn deception against cybercriminals? We present LURE (LLM-based User Response Engagement), the first system to deploy Large Language Models (LLMs) as active agents, not as passive classifiers, embedded within adversarial chat environments.   LURE combines automated discovery, adversarial interaction, and OCR-based analysis of image-embedded payment data. Applied to the setting of illicit video chat scams on Telegram, our system engaged 53 actors across 98 groups. In over 56 percent of interactions, the LLM maintained multi-round conversations without being noticed as a bot, effectively "winning" the imitation game. Our findings reveal key behavioral patterns in scam operations, such as payment flows, upselling strategies, and platform migration tactics.



## **31. Robustness Certificates for Neural Networks against Adversarial Attacks**

cs.LG

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.20865v1) [paper-pdf](https://arxiv.org/pdf/2512.20865v1)

**Authors**: Sara Taheri, Mahalakshmi Sabanayagam, Debarghya Ghoshdastidar, Majid Zamani

**Abstract**: The increasing use of machine learning in safety-critical domains amplifies the risk of adversarial threats, especially data poisoning attacks that corrupt training data to degrade performance or induce unsafe behavior. Most existing defenses lack formal guarantees or rely on restrictive assumptions about the model class, attack type, extent of poisoning, or point-wise certification, limiting their practical reliability. This paper introduces a principled formal robustness certification framework that models gradient-based training as a discrete-time dynamical system (dt-DS) and formulates poisoning robustness as a formal safety verification problem. By adapting the concept of barrier certificates (BCs) from control theory, we introduce sufficient conditions to certify a robust radius ensuring that the terminal model remains safe under worst-case ${\ell}_p$-norm based poisoning. To make this practical, we parameterize BCs as neural networks trained on finite sets of poisoned trajectories. We further derive probably approximately correct (PAC) bounds by solving a scenario convex program (SCP), which yields a confidence lower bound on the certified robustness radius generalizing beyond the training set. Importantly, our framework also extends to certification against test-time attacks, making it the first unified framework to provide formal guarantees in both training and test-time attack settings. Experiments on MNIST, SVHN, and CIFAR-10 show that our approach certifies non-trivial perturbation budgets while being model-agnostic and requiring no prior knowledge of the attack or contamination level.



## **32. Defending against adversarial attacks using mixture of experts**

cs.LG

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20821v1) [paper-pdf](https://arxiv.org/pdf/2512.20821v1)

**Authors**: Mohammad Meymani, Roozbeh Razavi-Far

**Abstract**: Machine learning is a powerful tool enabling full automation of a huge number of tasks without explicit programming. Despite recent progress of machine learning in different domains, these models have shown vulnerabilities when they are exposed to adversarial threats. Adversarial threats aim to hinder the machine learning models from satisfying their objectives. They can create adversarial perturbations, which are imperceptible to humans' eyes but have the ability to cause misclassification during inference. Moreover, they can poison the training data to harm the model's performance or they can query the model to steal its sensitive information. In this paper, we propose a defense system, which devises an adversarial training module within mixture-of-experts architecture to enhance its robustness against adversarial threats. In our proposed defense system, we use nine pre-trained experts with ResNet-18 as their backbone. During end-to-end training, the parameters of expert models and gating mechanism are jointly updated allowing further optimization of the experts. Our proposed defense system outperforms state-of-the-art defense systems and plain classifiers, which use a more complex architecture than our model's backbone.



## **33. Safety Alignment of LMs via Non-cooperative Games**

cs.AI

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20806v1) [paper-pdf](https://arxiv.org/pdf/2512.20806v1)

**Authors**: Anselm Paulus, Ilia Kulikov, Brandon Amos, Rémi Munos, Ivan Evtimov, Kamalika Chaudhuri, Arman Zharmagambetov

**Abstract**: Ensuring the safety of language models (LMs) while maintaining their usefulness remains a critical challenge in AI alignment. Current approaches rely on sequential adversarial training: generating adversarial prompts and fine-tuning LMs to defend against them. We introduce a different paradigm: framing safety alignment as a non-zero-sum game between an Attacker LM and a Defender LM trained jointly via online reinforcement learning. Each LM continuously adapts to the other's evolving strategies, driving iterative improvement. Our method uses a preference-based reward signal derived from pairwise comparisons instead of point-wise scores, providing more robust supervision and potentially reducing reward hacking. Our RL recipe, AdvGame, shifts the Pareto frontier of safety and utility, yielding a Defender LM that is simultaneously more helpful and more resilient to adversarial attacks. In addition, the resulting Attacker LM converges into a strong, general-purpose red-teaming agent that can be directly deployed to probe arbitrary target models.



## **34. Contingency Model-based Control (CMC) for Communicationless Cooperative Collision Avoidance in Robot Swarms**

math.OC

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.20391v3) [paper-pdf](https://arxiv.org/pdf/2512.20391v3)

**Authors**: Georg Schildbach

**Abstract**: Cooperative collision avoidance between robots, or `agents,' in swarm operations remains an open challenge. Assuming a decentralized architecture, each agent is responsible for making its own decisions and choosing its control actions. Most existing approaches rely on a (wireless) communication network between (some of) the agents. In reality, however, communication is brittle. It may be affected by latency, further delays and packet losses, and transmission faults. Moreover, it is subject to adversarial attacks, such as jamming or spoofing. This paper proposes Contingency Model-based Control (CMC), a decentralized cooperative approach that does not rely on communication. Instead, the control algorithm is based on consensual rules that are designed for all agents offline, similar to traffic rules. For CMC, this includes the definition of a contingency trajectory for each robot, and perpendicular bisecting planes as collision avoidance constraints. The setup permits a full guarantee of recursive feasibility and collision avoidance between all swarm members in closed-loop operation. CMC naturally satisfies the plug & play paradigm, i.e., new robots may enter the swarm dynamically. The effectiveness of the CMC regime is demonstrated in two numerical examples, showing that the collision avoidance guarantee is intact and the robot swarm operates smoothly in a constrained environment.



## **35. GShield: Mitigating Poisoning Attacks in Federated Learning**

cs.CR

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2512.19286v2) [paper-pdf](https://arxiv.org/pdf/2512.19286v2)

**Authors**: Sameera K. M., Serena Nicolazzo, Antonino Nocera, Vinod P., Rafidha Rehiman K. A

**Abstract**: Federated Learning (FL) has recently emerged as a revolutionary approach to collaborative training Machine Learning models. In particular, it enables decentralized model training while preserving data privacy, but its distributed nature makes it highly vulnerable to a severe attack known as Data Poisoning. In such scenarios, malicious clients inject manipulated data into the training process, thereby degrading global model performance or causing targeted misclassification. In this paper, we present a novel defense mechanism called GShield, designed to detect and mitigate malicious and low-quality updates, especially under non-independent and identically distributed (non-IID) data scenarios. GShield operates by learning the distribution of benign gradients through clustering and Gaussian modeling during an initial round, enabling it to establish a reliable baseline of trusted client behavior. With this benign profile, GShield selectively aggregates only those updates that align with the expected gradient patterns, effectively isolating adversarial clients and preserving the integrity of the global model. An extensive experimental campaign demonstrates that our proposed defense significantly improves model robustness compared to the state-of-the-art methods while maintaining a high accuracy of performance across both tabular and image datasets. Furthermore, GShield improves the accuracy of the targeted class by 43\% to 65\% after detecting malicious and low-quality clients.



## **36. Adversarially Robust Detection of Harmful Online Content: A Computational Design Science Approach**

cs.LG

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.17367v3) [paper-pdf](https://arxiv.org/pdf/2512.17367v3)

**Authors**: Yidong Chai, Yi Liu, Mohammadreza Ebrahimi, Weifeng Li, Balaji Padmanabhan

**Abstract**: Social media platforms are plagued by harmful content such as hate speech, misinformation, and extremist rhetoric. Machine learning (ML) models are widely adopted to detect such content; however, they remain highly vulnerable to adversarial attacks, wherein malicious users subtly modify text to evade detection. Enhancing adversarial robustness is therefore essential, requiring detectors that can defend against diverse attacks (generalizability) while maintaining high overall accuracy. However, simultaneously achieving both optimal generalizability and accuracy is challenging. Following the computational design science paradigm, this study takes a sequential approach that first proposes a novel framework (Large Language Model-based Sample Generation and Aggregation, LLM-SGA) by identifying the key invariances of textual adversarial attacks and leveraging them to ensure that a detector instantiated within the framework has strong generalizability. Second, we instantiate our detector (Adversarially Robust Harmful Online Content Detector, ARHOCD) with three novel design components to improve detection accuracy: (1) an ensemble of multiple base detectors that exploits their complementary strengths; (2) a novel weight assignment method that dynamically adjusts weights based on each sample's predictability and each base detector's capability, with weights initialized using domain knowledge and updated via Bayesian inference; and (3) a novel adversarial training strategy that iteratively optimizes both the base detectors and the weight assignor. We addressed several limitations of existing adversarial robustness enhancement research and empirically evaluated ARHOCD across three datasets spanning hate speech, rumor, and extremist content. Results show that ARHOCD offers strong generalizability and improves detection accuracy under adversarial conditions.



## **37. Seeing Isn't Believing: Context-Aware Adversarial Patch Synthesis via Conditional GAN**

cs.CV

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2509.22836v2) [paper-pdf](https://arxiv.org/pdf/2509.22836v2)

**Authors**: Roie Kazoom, Alon Goldberg, Hodaya Cohen, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a severe threat to deep neural networks, yet most existing approaches rely on unrealistic white-box assumptions, untargeted objectives, or produce visually conspicuous patches that limit real-world applicability. In this work, we introduce a novel framework for fully controllable adversarial patch generation, where the attacker can freely choose both the input image x and the target class y target, thereby dictating the exact misclassification outcome. Our method combines a generative U-Net design with Grad-CAM-guided patch placement, enabling semantic-aware localization that maximizes attack effectiveness while preserving visual realism. Extensive experiments across convolutional networks (DenseNet-121, ResNet-50) and vision transformers (ViT-B/16, Swin-B/16, among others) demonstrate that our approach achieves state-of-the-art performance across all settings, with attack success rates (ASR) and target-class success (TCS) consistently exceeding 99%.   Importantly, we show that our method not only outperforms prior white-box attacks and untargeted baselines, but also surpasses existing non-realistic approaches that produce detectable artifacts. By simultaneously ensuring realism, targeted control, and black-box applicability-the three most challenging dimensions of patch-based attacks-our framework establishes a new benchmark for adversarial robustness research, bridging the gap between theoretical attack strength and practical stealthiness.



## **38. BadBlocks: Lightweight and Stealthy Backdoor Threat in Text-to-Image Diffusion Models**

cs.CR

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2508.03221v4) [paper-pdf](https://arxiv.org/pdf/2508.03221v4)

**Authors**: Yu Pan, Jiahao Chen, Wenjie Wang, Bingrong Dai, Junjun Yang

**Abstract**: Diffusion models have recently achieved remarkable success in image generation, yet growing evidence shows their vulnerability to backdoor attacks, where adversaries implant covert triggers to manipulate outputs. While existing defenses can detect many such attacks via visual inspection and neural network-based analysis, we identify a more lightweight and stealthy threat, termed BadBlocks. BadBlocks selectively contaminates specific blocks within the UNet architecture while preserving the normal behavior of the remaining components. Compared with prior methods, it requires only about 30% of the computation and 20% of the GPU time, yet achieves high attack success rates with minimal perceptual degradation. Extensive experiments demonstrate that BadBlocks can effectively evade state-of-the-art defenses, particularly attention-based detection frameworks. Ablation studies further reveal that effective backdoor injection does not require fine-tuning the entire network and highlight the critical role of certain layers in backdoor mapping. Overall, BadBlocks substantially lowers the barrier for backdooring large-scale diffusion models, even on consumer-grade GPUs.



## **39. BeDKD: Backdoor Defense based on Dynamic Knowledge Distillation and Directional Mapping Modulator**

cs.CR

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2508.01595v2) [paper-pdf](https://arxiv.org/pdf/2508.01595v2)

**Authors**: Zhengxian Wu, Juan Wen, Wanli Peng, Yinghan Zhou, Changtong dou, Yiming Xue

**Abstract**: Although existing backdoor defenses have gained success in mitigating backdoor attacks, they still face substantial challenges. In particular, most of them rely on large amounts of clean data to weaken the backdoor mapping but generally struggle with residual trigger effects, resulting in persistently high attack success rates (ASR). Therefore, in this paper, we propose a novel Backdoor defense method based on Directional mapping module and adversarial Knowledge Distillation (BeDKD), which balances the trade-off between defense effectiveness and model performance using a small amount of clean and poisoned data. We first introduce a directional mapping module to identify poisoned data, which destroys clean mapping while keeping backdoor mapping on a small set of flipped clean data. Then, the adversarial knowledge distillation is designed to reinforce clean mapping and suppress backdoor mapping through a cycle iteration mechanism between trust and punish distillations using clean and identified poisoned data. We conduct experiments to mitigate mainstream attacks on three datasets, and experimental results demonstrate that BeDKD surpasses the state-of-the-art defenses and reduces the ASR by 98% without significantly reducing the CACC. Our code are available in https://github.com/CAU-ISS-Lab/Backdoor-Attack-Defense-LLMs/tree/main/BeDKD.



## **40. Improving Large Language Model Safety with Contrastive Representation Learning**

cs.CL

EMNLP 2025 Main

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2506.11938v2) [paper-pdf](https://arxiv.org/pdf/2506.11938v2)

**Authors**: Samuel Simko, Mrinmaya Sachan, Bernhard Schölkopf, Zhijing Jin

**Abstract**: Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at https://github.com/samuelsimko/crl-llm-defense



## **41. CAE-Net: Generalized Deepfake Image Detection using Convolution and Attention Mechanisms with Spatial and Frequency Domain Features**

cs.CV

Published in Journal of Visual Communication and Image Representation

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2502.10682v3) [paper-pdf](https://arxiv.org/pdf/2502.10682v3)

**Authors**: Anindya Bhattacharjee, Kaidul Islam, Kafi Anan, Ashir Intesher, Abrar Assaeem Fuad, Utsab Saha, Hafiz Imtiaz

**Abstract**: The spread of deepfakes poses significant security concerns, demanding reliable detection methods. However, diverse generation techniques and class imbalance in datasets create challenges. We propose CAE-Net, a Convolution- and Attention-based weighted Ensemble network combining spatial and frequency-domain features for effective deepfake detection. The architecture integrates EfficientNet, Data-Efficient Image Transformer (DeiT), and ConvNeXt with wavelet features to learn complementary representations. We evaluated CAE-Net on the diverse IEEE Signal Processing Cup 2025 (DF-Wild Cup) dataset, which has a 5:1 fake-to-real class imbalance. To address this, we introduce a multistage disjoint-subset training strategy, sequentially training the model on non-overlapping subsets of the fake class while retaining knowledge across stages. Our approach achieved $94.46\%$ accuracy and a $97.60\%$ AUC, outperforming conventional class-balancing methods. Visualizations confirm the network focuses on meaningful facial regions, and our ensemble design demonstrates robustness against adversarial attacks, positioning CAE-Net as a dependable and generalized deepfake detection framework.



## **42. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

cs.CL

Accepted by USENIX Security 2025

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2502.01386v3) [paper-pdf](https://arxiv.org/pdf/2502.01386v3)

**Authors**: Yuyang Gong, Zhuo Chen, Jiawei Liu, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.



## **43. Illusions of Relevance: Arbitrary Content Injection Attacks Deceive Retrievers, Rerankers, and LLM Judges**

cs.IR

AACL Findings 2025

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2501.18536v2) [paper-pdf](https://arxiv.org/pdf/2501.18536v2)

**Authors**: Manveer Singh Tamber, Jimmy Lin

**Abstract**: This work considers a black-box threat model in which adversaries attempt to propagate arbitrary non-relevant content in search. We show that retrievers, rerankers, and LLM relevance judges are all highly vulnerable to attacks that enable arbitrary content to be promoted to the top of search results and to be assigned perfect relevance scores. We investigate how attackers may achieve this via content injection, injecting arbitrary sentences into relevant passages or query terms into arbitrary passages. Our study analyzes how factors such as model class and size, the balance between relevant and non-relevant content, injection location, toxicity and severity of injected content, and the role of LLM-generated content influence attack success, yielding novel, concerning, and often counterintuitive results. Our results reveal a weakness in embedding models, LLM-based scoring models, and generative LLMs, raising concerns about the general robustness, safety, and trustworthiness of language models regardless of the type of model or the role in which they are employed. We also emphasize the challenges of robust defenses against these attacks. Classifiers and more carefully prompted LLM judges often fail to recognize passages with content injection, especially when considering diverse text topics and styles. Our findings highlight the need for further research into arbitrary content injection attacks. We release our code for further study.



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



## **48. SoundnessBench: A Soundness Benchmark for Neural Network Verifiers**

cs.LG

TMLR (December 2025)

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2412.03154v3) [paper-pdf](https://arxiv.org/pdf/2412.03154v3)

**Authors**: Xingjian Zhou, Keyi Shen, Andy Xu, Hongji Xu, Cho-Jui Hsieh, Huan Zhang, Zhouxing Shi

**Abstract**: Neural network (NN) verification aims to formally verify properties of NNs, which is crucial for ensuring the behavior of NN-based models in safety-critical applications. In recent years, the community has developed many NN verifiers and benchmarks to evaluate them. However, existing benchmarks typically lack ground-truth for hard instances where no current verifier can verify the property and no counterexample can be found. This makes it difficult to validate the soundness of a verifier, when it claims verification on such challenging instances that no other verifier can handle. In this work, we develop a new benchmark for NN verification, named SoundnessBench, specifically for testing the soundness of NN verifiers. SoundnessBench consists of instances with deliberately inserted counterexamples that are hidden from adversarial attacks commonly used to find counterexamples. Thereby, it can identify false verification claims when hidden counterexamples are known to exist. We design a training method to produce NNs with hidden counterexamples and systematically construct our SoundnessBench with instances across various model architectures, activation functions, and input data. We demonstrate that our training effectively produces hidden counterexamples and our SoundnessBench successfully identifies bugs in state-of-the-art NN verifiers. Our code is available at https://github.com/mvp-harry/SoundnessBench and our dataset is available at https://huggingface.co/datasets/SoundnessBench/SoundnessBench.



## **49. Trust-free Personalized Decentralized Learning**

cs.LG

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2410.11378v2) [paper-pdf](https://arxiv.org/pdf/2410.11378v2)

**Authors**: Yawen Li, Yan Li, Junping Du, Yingxia Shao, Meiyu Liang, Guanhua Ye

**Abstract**: Personalized collaborative learning in federated settings faces a critical trade-off between customization and participant trust. Existing approaches typically rely on centralized coordinators or trusted peer groups, limiting their applicability in open, trust-averse environments. While recent decentralized methods explore anonymous knowledge sharing, they often lack global scalability and robust mechanisms against malicious peers. To bridge this gap, we propose TPFed, a \textit{Trust-free Personalized Decentralized Federated Learning} framework. TPFed replaces central aggregators with a blockchain-based bulletin board, enabling participants to dynamically select global communication partners based on Locality-Sensitive Hashing (LSH) and peer ranking. Crucially, we introduce an ``all-in-one'' knowledge distillation protocol that simultaneously handles knowledge transfer, model quality evaluation, and similarity verification via a public reference dataset. This design ensures secure, globally personalized collaboration without exposing local models or data. Extensive experiments demonstrate that TPFed significantly outperforms traditional federated baselines in both learning accuracy and system robustness against adversarial attacks.



## **50. Achieving Dalenius' Goal of Data Privacy with Practical Assumptions**

cs.CR

50 pages

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/1703.07474v6) [paper-pdf](https://arxiv.org/pdf/1703.07474v6)

**Authors**: Genqiang Wu, Xianyao Xia, Yeping He

**Abstract**: Current differential privacy frameworks face significant challenges: vulnerability to correlated data attacks and suboptimal utility-privacy tradeoffs. To address these limitations, we establish a novel information-theoretic foundation for Dalenius' privacy vision using Shannon's perfect secrecy framework. By leveraging the fundamental distinction between cryptographic systems (small secret keys) and privacy mechanisms (massive datasets), we replace differential privacy's restrictive independence assumption with practical partial knowledge constraints ($H(X) \geq b$).   We propose an information privacy framework achieving Dalenius security with quantifiable utility-privacy tradeoffs. Crucially, we prove that foundational mechanisms -- random response, exponential, and Gaussian channels -- satisfy Dalenius' requirements while preserving group privacy and composition properties. Our channel capacity analysis reduces infinite-dimensional evaluations to finite convex optimizations, enabling direct application of information-theoretic tools.   Empirical evaluation demonstrates that individual channel capacity (maximal information leakage of each individual) decreases with increasing entropy constraint $b$, and our framework achieves superior utility-privacy tradeoffs compared to classical differential privacy mechanisms under equivalent privacy guarantees. The framework is extended to computationally bounded adversaries via Yao's theory, unifying cryptographic and statistical privacy paradigms. Collectively, these contributions provide a theoretically grounded path toward practical, composable privacy -- subject to future resolution of the tradeoff characterization -- with enhanced resilience to correlation attacks.



