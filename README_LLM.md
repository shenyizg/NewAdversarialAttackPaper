# Latest Large Language Model Attack Papers
**update at 2025-11-25 15:22:01**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Adversarial Attack-Defense Co-Evolution for LLM Safety Alignment via Tree-Group Dual-Aware Search and Optimization**

cs.CR

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19218v1) [paper-pdf](https://arxiv.org/pdf/2511.19218v1)

**Authors**: Xurui Li, Kaisong Song, Rui Zhu, Pin-Yu Chen, Haixu Tang

**Abstract**: Large Language Models (LLMs) have developed rapidly in web services, delivering unprecedented capabilities while amplifying societal risks. Existing works tend to focus on either isolated jailbreak attacks or static defenses, neglecting the dynamic interplay between evolving threats and safeguards in real-world web contexts. To mitigate these challenges, we propose ACE-Safety (Adversarial Co-Evolution for LLM Safety), a novel framework that jointly optimize attack and defense models by seamlessly integrating two key innovative procedures: (1) Group-aware Strategy-guided Monte Carlo Tree Search (GS-MCTS), which efficiently explores jailbreak strategies to uncover vulnerabilities and generate diverse adversarial samples; (2) Adversarial Curriculum Tree-aware Group Policy Optimization (AC-TGPO), which jointly trains attack and defense LLMs with challenging samples via curriculum reinforcement learning, enabling robust mutual improvement. Evaluations across multiple benchmarks demonstrate that our method outperforms existing attack and defense approaches, and provides a feasible pathway for developing LLMs that can sustainably support responsible AI ecosystems.



## **2. Defending Large Language Models Against Jailbreak Exploits with Responsible AI Considerations**

cs.CR

20 pages including appendix; technical report; NeurIPS 2024 style

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.18933v1) [paper-pdf](https://arxiv.org/pdf/2511.18933v1)

**Authors**: Ryan Wong, Hosea David Yu Fei Ng, Dhananjai Sharma, Glenn Jun Jie Ng, Kavishvaran Srinivasan

**Abstract**: Large Language Models (LLMs) remain susceptible to jailbreak exploits that bypass safety filters and induce harmful or unethical behavior. This work presents a systematic taxonomy of existing jailbreak defenses across prompt-level, model-level, and training-time interventions, followed by three proposed defense strategies. First, a Prompt-Level Defense Framework detects and neutralizes adversarial inputs through sanitization, paraphrasing, and adaptive system guarding. Second, a Logit-Based Steering Defense reinforces refusal behavior through inference-time vector steering in safety-sensitive layers. Third, a Domain-Specific Agent Defense employs the MetaGPT framework to enforce structured, role-based collaboration and domain adherence. Experiments on benchmark datasets show substantial reductions in attack success rate, achieving full mitigation under the agent-based defense. Overall, this study highlights how jailbreaks pose a significant security threat to LLMs and identifies key intervention points for prevention, while noting that defense strategies often involve trade-offs between safety, performance, and scalability. Code is available at: https://github.com/Kuro0911/CS5446-Project



## **3. BackdoorVLM: A Benchmark for Backdoor Attacks on Vision-Language Models**

cs.CV

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.18921v1) [paper-pdf](https://arxiv.org/pdf/2511.18921v1)

**Authors**: Juncheng Li, Yige Li, Hanxun Huang, Yunhao Chen, Xin Wang, Yixu Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Backdoor attacks undermine the reliability and trustworthiness of machine learning systems by injecting hidden behaviors that can be maliciously activated at inference time. While such threats have been extensively studied in unimodal settings, their impact on multimodal foundation models, particularly vision-language models (VLMs), remains largely underexplored. In this work, we introduce \textbf{BackdoorVLM}, the first comprehensive benchmark for systematically evaluating backdoor attacks on VLMs across a broad range of settings. It adopts a unified perspective that injects and analyzes backdoors across core vision-language tasks, including image captioning and visual question answering. BackdoorVLM organizes multimodal backdoor threats into 5 representative categories: targeted refusal, malicious injection, jailbreak, concept substitution, and perceptual hijack. Each category captures a distinct pathway through which an adversary can manipulate a model's behavior. We evaluate these threats using 12 representative attack methods spanning text, image, and bimodal triggers, tested on 2 open-source VLMs and 3 multimodal datasets. Our analysis reveals that VLMs exhibit strong sensitivity to textual instructions, and in bimodal backdoors the text trigger typically overwhelms the image trigger when forming the backdoor mapping. Notably, backdoors involving the textual modality remain highly potent, with poisoning rates as low as 1\% yielding over 90\% success across most tasks. These findings highlight significant, previously underexplored vulnerabilities in current VLMs. We hope that BackdoorVLM can serve as a useful benchmark for analyzing and mitigating multimodal backdoor threats. Code is available at: https://github.com/bin015/BackdoorVLM .



## **4. RoguePrompt: Dual-Layer Ciphering for Self-Reconstruction to Circumvent LLM Moderation**

cs.CR

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.18790v1) [paper-pdf](https://arxiv.org/pdf/2511.18790v1)

**Authors**: Benyamin Tafreshian

**Abstract**: Content moderation pipelines for modern large language models combine static filters, dedicated moderation services, and alignment tuned base models, yet real world deployments still exhibit dangerous failure modes. This paper presents RoguePrompt, an automated jailbreak attack that converts a disallowed user query into a self reconstructing prompt which passes provider moderation while preserving the original harmful intent. RoguePrompt partitions the instruction across two lexical streams, applies nested classical ciphers, and wraps the result in natural language directives that cause the target model to decode and execute the hidden payload. Our attack assumes only black box access to the model and to the associated moderation endpoint. We instantiate RoguePrompt against GPT 4o and evaluate it on 2 448 prompts that a production moderation system previously marked as strongly rejected. Under an evaluation protocol that separates three security relevant outcomes bypass, reconstruction, and execution the attack attains 84.7 percent bypass, 80.2 percent reconstruction, and 71.5 percent full execution, substantially outperforming five automated jailbreak baselines. We further analyze the behavior of several automated and human aligned evaluators and show that dual layer lexical transformations remain effective even when detectors rely on semantic similarity or learned safety rubrics. Our results highlight systematic blind spots in current moderation practice and suggest that robust deployment will require joint reasoning about user intent, decoding workflows, and model side computation rather than surface level toxicity alone.



## **5. Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems**

cs.CR

Accepted by AAAI 2026 Alignment Track

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2511.18467v1) [paper-pdf](https://arxiv.org/pdf/2511.18467v1)

**Authors**: Xiaoqing Wang, Keman Huang, Bin Liang, Hongyu Li, Xiaoyong Du

**Abstract**: The rapid advancement of Large Language Model (LLM)-driven multi-agent systems has significantly streamlined software developing tasks, enabling users with little technical expertise to develop executable applications. While these systems democratize software creation through natural language requirements, they introduce significant security risks that remain largely unexplored. We identify two risky scenarios: Malicious User with Benign Agents (MU-BA) and Benign User with Malicious Agents (BU-MA). We introduce the Implicit Malicious Behavior Injection Attack (IMBIA), demonstrating how multi-agent systems can be manipulated to generate software with concealed malicious capabilities beneath seemingly benign applications, and propose Adv-IMBIA as a defense mechanism. Evaluations across ChatDev, MetaGPT, and AgentVerse frameworks reveal varying vulnerability patterns, with IMBIA achieving attack success rates of 93%, 45%, and 71% in MU-BA scenarios, and 71%, 84%, and 45% in BU-MA scenarios. Our defense mechanism reduced attack success rates significantly, particularly in the MU-BA scenario. Further analysis reveals that compromised agents in the coding and testing phases pose significantly greater security risks, while also identifying critical agents that require protection against malicious user exploitation. Our findings highlight the urgent need for robust security measures in multi-agent software development systems and provide practical guidelines for implementing targeted, resource-efficient defensive strategies.



## **6. Think Fast: Real-Time IoT Intrusion Reasoning Using IDS and LLMs at the Edge Gateway**

cs.CR

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2511.18230v1) [paper-pdf](https://arxiv.org/pdf/2511.18230v1)

**Authors**: Saeid Jamshidi, Amin Nikanjam, Negar Shahabi, Kawser Wazed Nafi, Foutse Khomh, Samira Keivanpour, Rolando Herrero

**Abstract**: As the number of connected IoT devices continues to grow, securing these systems against cyber threats remains a major challenge, especially in environments with limited computational and energy resources. This paper presents an edge-centric Intrusion Detection System (IDS) framework that integrates lightweight machine learning (ML) based IDS models with pre-trained large language models (LLMs) to improve detection accuracy, semantic interpretability, and operational efficiency at the network edge. The system evaluates six ML-based IDS models: Decision Tree (DT), K-Nearest Neighbors (KNN), Random Forest (RF), Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), and a hybrid CNN-LSTM model on low-power edge gateways, achieving accuracy up to 98 percent under real-world cyberattacks. For anomaly detection, the system transmits a compact and secure telemetry snapshot (for example, CPU usage, memory usage, latency, and energy consumption) via low-bandwidth API calls to LLMs including GPT-4-turbo, DeepSeek V2, and LLaMA 3.5. These models use zero-shot, few-shot, and chain-of-thought reasoning to produce human-readable threat analyses and actionable mitigation recommendations. Evaluations across diverse attacks such as DoS, DDoS, brute force, and port scanning show that the system enhances interpretability while maintaining low latency (<1.5 s), minimal bandwidth usage (<1.2 kB per prompt), and energy efficiency (<75 J), demonstrating its practicality and scalability as an IDS solution for edge gateways.



## **7. ASTRA: Agentic Steerability and Risk Assessment Framework**

cs.CR

**SubmitDate**: 2025-11-22    [abs](http://arxiv.org/abs/2511.18114v1) [paper-pdf](https://arxiv.org/pdf/2511.18114v1)

**Authors**: Itay Hazan, Yael Mathov, Guy Shtar, Ron Bitton, Itsik Mantin

**Abstract**: Securing AI agents powered by Large Language Models (LLMs) represents one of the most critical challenges in AI security today. Unlike traditional software, AI agents leverage LLMs as their "brain" to autonomously perform actions via connected tools. This capability introduces significant risks that go far beyond those of harmful text presented in a chatbot that was the main application of LLMs. A compromised AI agent can deliberately abuse powerful tools to perform malicious actions, in many cases irreversible, and limited solely by the guardrails on the tools themselves and the LLM ability to enforce them. This paper presents ASTRA, a first-of-its-kind framework designed to evaluate the effectiveness of LLMs in supporting the creation of secure agents that enforce custom guardrails defined at the system-prompt level (e.g., "Do not send an email out of the company domain," or "Never extend the robotic arm in more than 2 meters").   Our holistic framework simulates 10 diverse autonomous agents varying between a coding assistant and a delivery drone equipped with 37 unique tools. We test these agents against a suite of novel attacks developed specifically for agentic threats, inspired by the OWASP Top 10 but adapted to challenge the ability of the LLM for policy enforcement during multi-turn planning and execution of strict tool activation. By evaluating 13 open-source, tool-calling LLMs, we uncovered surprising and significant differences in their ability to remain secure and keep operating within their boundaries. The purpose of this work is to provide the community with a robust and unified methodology to build and validate better LLMs, ultimately pushing for more secure and reliable agentic AI systems.



## **8. Steering in the Shadows: Causal Amplification for Activation Space Attacks in Large Language Models**

cs.CR

31 pages, 5 figures, 9 tables

**SubmitDate**: 2025-11-21    [abs](http://arxiv.org/abs/2511.17194v1) [paper-pdf](https://arxiv.org/pdf/2511.17194v1)

**Authors**: Zhiyuan Xu, Stanislav Abaimov, Joseph Gardiner, Sana Belguith

**Abstract**: Modern large language models (LLMs) are typically secured by auditing data, prompts, and refusal policies, while treating the forward pass as an implementation detail. We show that intermediate activations in decoder-only LLMs form a vulnerable attack surface for behavioral control. Building on recent findings on attention sinks and compression valleys, we identify a high-gain region in the residual stream where small, well-aligned perturbations are causally amplified along the autoregressive trajectory--a Causal Amplification Effect (CAE). We exploit this as an attack surface via Sensitivity-Scaled Steering (SSS), a progressive activation-level attack that combines beginning-of-sequence (BOS) anchoring with sensitivity-based reinforcement to focus a limited perturbation budget on the most vulnerable layers and tokens. We show that across multiple open-weight models and four behavioral axes, SSS induces large shifts in evil, hallucination, sycophancy, and sentiment while preserving high coherence and general capabilities, turning activation steering into a concrete security concern for white-box and supply-chain LLM deployments.



## **9. MultiPriv: Benchmarking Individual-Level Privacy Reasoning in Vision-Language Models**

cs.CV

**SubmitDate**: 2025-11-21    [abs](http://arxiv.org/abs/2511.16940v1) [paper-pdf](https://arxiv.org/pdf/2511.16940v1)

**Authors**: Xiongtao Sun, Hui Li, Jiaming Zhang, Yujie Yang, Kaili Liu, Ruxin Feng, Wen Jun Tan, Wei Yang Bryan Lim

**Abstract**: Modern Vision-Language Models (VLMs) demonstrate sophisticated reasoning, escalating privacy risks beyond simple attribute perception to individual-level linkage. Current privacy benchmarks are structurally insufficient for this new threat, as they primarily evaluate privacy perception while failing to address the more critical risk of privacy reasoning: a VLM's ability to infer and link distributed information to construct individual profiles. To address this critical gap, we propose \textbf{MultiPriv}, the first benchmark designed to systematically evaluate individual-level privacy reasoning in VLMs. We introduce the \textbf{Privacy Perception and Reasoning (PPR)} framework and construct a novel, bilingual multimodal dataset to support it. The dataset uniquely features a core component of synthetic individual profiles where identifiers (e.g., faces, names) are meticulously linked to sensitive attributes. This design enables nine challenging tasks evaluating the full PPR spectrum, from attribute detection to cross-image re-identification and chained inference. We conduct a large-scale evaluation of over 50 foundational and commercial VLMs. Our analysis reveals: (1) Many VLMs possess significant, unmeasured reasoning-based privacy risks. (2) Perception-level metrics are poor predictors of these reasoning risks, revealing a critical evaluation gap. (3) Existing safety alignments are inconsistent and ineffective against such reasoning-based attacks. MultiPriv exposes systemic vulnerabilities and provides the necessary framework for developing robust, privacy-preserving VLMs.



## **10. Evaluating Adversarial Vulnerabilities in Modern Large Language Models**

cs.CR

**SubmitDate**: 2025-11-21    [abs](http://arxiv.org/abs/2511.17666v1) [paper-pdf](https://arxiv.org/pdf/2511.17666v1)

**Authors**: Tom Perel

**Abstract**: The recent boom and rapid integration of Large Language Models (LLMs) into a wide range of applications warrants a deeper understanding of their security and safety vulnerabilities. This paper presents a comparative analysis of the susceptibility to jailbreak attacks for two leading publicly available LLMs, Google's Gemini 2.5 Flash and OpenAI's GPT-4 (specifically the GPT-4o mini model accessible in the free tier). The research utilized two main bypass strategies: 'self-bypass', where models were prompted to circumvent their own safety protocols, and 'cross-bypass', where one model generated adversarial prompts to exploit vulnerabilities in the other. Four attack methods were employed - direct injection, role-playing, context manipulation, and obfuscation - to generate five distinct categories of unsafe content: hate speech, illegal activities, malicious code, dangerous content, and misinformation. The success of the attack was determined by the generation of disallowed content, with successful jailbreaks assigned a severity score. The findings indicate a disparity in jailbreak susceptibility between 2.5 Flash and GPT-4, suggesting variations in their safety implementations or architectural design. Cross-bypass attacks were particularly effective, indicating that an ample amount of vulnerabilities exist in the underlying transformer architecture. This research contributes a scalable framework for automated AI red-teaming and provides data-driven insights into the current state of LLM safety, underscoring the complex challenge of balancing model capabilities with robust safety mechanisms.



## **11. Large Language Model-Based Reward Design for Deep Reinforcement Learning-Driven Autonomous Cyber Defense**

cs.LG

Accepted in the AAAI-26 Workshop on Artificial Intelligence for Cyber Security (AICS)

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16483v1) [paper-pdf](https://arxiv.org/pdf/2511.16483v1)

**Authors**: Sayak Mukherjee, Samrat Chatterjee, Emilie Purvine, Ted Fujimoto, Tegan Emerson

**Abstract**: Designing rewards for autonomous cyber attack and defense learning agents in a complex, dynamic environment is a challenging task for subject matter experts. We propose a large language model (LLM)-based reward design approach to generate autonomous cyber defense policies in a deep reinforcement learning (DRL)-driven experimental simulation environment. Multiple attack and defense agent personas were crafted, reflecting heterogeneity in agent actions, to generate LLM-guided reward designs where the LLM was first provided with contextual cyber simulation environment information. These reward structures were then utilized within a DRL-driven attack-defense simulation environment to learn an ensemble of cyber defense policies. Our results suggest that LLM-guided reward designs can lead to effective defense strategies against diverse adversarial behaviors.



## **12. Q-MLLM: Vector Quantization for Robust Multimodal Large Language Model Security**

cs.CR

Accepted by NDSS 2026

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16229v1) [paper-pdf](https://arxiv.org/pdf/2511.16229v1)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in cross-modal understanding, but remain vulnerable to adversarial attacks through visual inputs despite robust textual safety mechanisms. These vulnerabilities arise from two core weaknesses: the continuous nature of visual representations, which allows for gradient-based attacks, and the inadequate transfer of text-based safety mechanisms to visual content. We introduce Q-MLLM, a novel architecture that integrates two-level vector quantization to create a discrete bottleneck against adversarial attacks while preserving multimodal reasoning capabilities. By discretizing visual representations at both pixel-patch and semantic levels, Q-MLLM blocks attack pathways and bridges the cross-modal safety alignment gap. Our two-stage training methodology ensures robust learning while maintaining model utility. Experiments demonstrate that Q-MLLM achieves significantly better defense success rate against both jailbreak attacks and toxic image attacks than existing approaches. Notably, Q-MLLM achieves perfect defense success rate (100\%) against jailbreak attacks except in one arguable case, while maintaining competitive performance on multiple utility benchmarks with minimal inference overhead. This work establishes vector quantization as an effective defense mechanism for secure multimodal AI systems without requiring expensive safety-specific fine-tuning or detection overhead. Code is available at https://github.com/Amadeuszhao/QMLLM.



## **13. PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization**

cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16209v1) [paper-pdf](https://arxiv.org/pdf/2511.16209v1)

**Authors**: Huseein Jawad, Nicolas Brunel

**Abstract**: System prompts are critical for guiding the behavior of Large Language Models (LLMs), yet they often contain proprietary logic or sensitive information, making them a prime target for extraction attacks. Adversarial queries can successfully elicit these hidden instructions, posing significant security and privacy risks. Existing defense mechanisms frequently rely on heuristics, incur substantial computational overhead, or are inapplicable to models accessed via black-box APIs. This paper introduces a novel framework for hardening system prompts through shield appending, a lightweight approach that adds a protective textual layer to the original prompt. Our core contribution is the formalization of prompt hardening as a utility-constrained optimization problem. We leverage an LLM-as-optimizer to search the space of possible SHIELDs, seeking to minimize a leakage metric derived from a suite of adversarial attacks, while simultaneously preserving task utility above a specified threshold, measured by semantic fidelity to baseline outputs. This black-box, optimization-driven methodology is lightweight and practical, requiring only API access to the target and optimizer LLMs. We demonstrate empirically that our optimized SHIELDs significantly reduce prompt leakage against a comprehensive set of extraction attacks, outperforming established baseline defenses without compromising the model's intended functionality. Our work presents a paradigm for developing robust, utility-aware defenses in the escalating landscape of LLM security. The code is made public on the following link: https://github.com/psm-defense/psm



## **14. When Alignment Fails: Multimodal Adversarial Attacks on Vision-Language-Action Models**

cs.CV

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2511.16203v2) [paper-pdf](https://arxiv.org/pdf/2511.16203v2)

**Authors**: Yuping Yan, Yuhan Xie, Yixin Zhang, Lingjuan Lyu, Handing Wang, Yaochu Jin

**Abstract**: Vision-Language-Action models (VLAs) have recently demonstrated remarkable progress in embodied environments, enabling robots to perceive, reason, and act through unified multimodal understanding. Despite their impressive capabilities, the adversarial robustness of these systems remains largely unexplored, especially under realistic multimodal and black-box conditions. Existing studies mainly focus on single-modality perturbations and overlook the cross-modal misalignment that fundamentally affects embodied reasoning and decision-making. In this paper, we introduce VLA-Fool, a comprehensive study of multimodal adversarial robustness in embodied VLA models under both white-box and black-box settings. VLA-Fool unifies three levels of multimodal adversarial attacks: (1) textual perturbations through gradient-based and prompt-based manipulations, (2) visual perturbations via patch and noise distortions, and (3) cross-modal misalignment attacks that intentionally disrupt the semantic correspondence between perception and instruction. We further incorporate a VLA-aware semantic space into linguistic prompts, developing the first automatically crafted and semantically guided prompting framework. Experiments on the LIBERO benchmark using a fine-tuned OpenVLA model reveal that even minor multimodal perturbations can cause significant behavioral deviations, demonstrating the fragility of embodied multimodal alignment.



## **15. AutoBackdoor: Automating Backdoor Attacks via LLM Agents**

cs.CR

23 pages

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16709v1) [paper-pdf](https://arxiv.org/pdf/2511.16709v1)

**Authors**: Yige Li, Zhe Li, Wei Zhao, Nay Myat Min, Hanxun Huang, Xingjun Ma, Jun Sun

**Abstract**: Backdoor attacks pose a serious threat to the secure deployment of large language models (LLMs), enabling adversaries to implant hidden behaviors triggered by specific inputs. However, existing methods often rely on manually crafted triggers and static data pipelines, which are rigid, labor-intensive, and inadequate for systematically evaluating modern defense robustness. As AI agents become increasingly capable, there is a growing need for more rigorous, diverse, and scalable \textit{red-teaming frameworks} that can realistically simulate backdoor threats and assess model resilience under adversarial conditions. In this work, we introduce \textsc{AutoBackdoor}, a general framework for automating backdoor injection, encompassing trigger generation, poisoned data construction, and model fine-tuning via an autonomous agent-driven pipeline. Unlike prior approaches, AutoBackdoor uses a powerful language model agent to generate semantically coherent, context-aware trigger phrases, enabling scalable poisoning across arbitrary topics with minimal human effort. We evaluate AutoBackdoor under three realistic threat scenarios, including \textit{Bias Recommendation}, \textit{Hallucination Injection}, and \textit{Peer Review Manipulation}, to simulate a broad range of attacks. Experiments on both open-source and commercial models, including LLaMA-3, Mistral, Qwen, and GPT-4o, demonstrate that our method achieves over 90\% attack success with only a small number of poisoned samples. More importantly, we find that existing defenses often fail to mitigate these attacks, underscoring the need for more rigorous and adaptive evaluation techniques against agent-driven threats as explored in this work. All code, datasets, and experimental configurations will be merged into our primary repository at https://github.com/bboylyg/BackdoorLLM.



## **16. What Your Features Reveal: Data-Efficient Black-Box Feature Inversion Attack for Split DNNs**

cs.CV

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15316v1) [paper-pdf](https://arxiv.org/pdf/2511.15316v1)

**Authors**: Zhihan Ren, Lijun He, Jiaxi Liang, Xinzhu Fu, Haixia Bi, Fan Li

**Abstract**: Split DNNs enable edge devices by offloading intensive computation to a cloud server, but this paradigm exposes privacy vulnerabilities, as the intermediate features can be exploited to reconstruct the private inputs via Feature Inversion Attack (FIA). Existing FIA methods often produce limited reconstruction quality, making it difficult to assess the true extent of privacy leakage. To reveal the privacy risk of the leaked features, we introduce FIA-Flow, a black-box FIA framework that achieves high-fidelity image reconstruction from intermediate features. To exploit the semantic information within intermediate features, we design a Latent Feature Space Alignment Module (LFSAM) to bridge the semantic gap between the intermediate feature space and the latent space. Furthermore, to rectify distributional mismatch, we develop Deterministic Inversion Flow Matching (DIFM), which projects off-manifold features onto the target manifold with one-step inference. This decoupled design simplifies learning and enables effective training with few image-feature pairs. To quantify privacy leakage from a human perspective, we also propose two metrics based on a large vision-language model. Experiments show that FIA-Flow achieves more faithful and semantically aligned feature inversion across various models (AlexNet, ResNet, Swin Transformer, DINO, and YOLO11) and layers, revealing a more severe privacy threat in Split DNNs than previously recognized.



## **17. Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models**

cs.CL

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.15304v2) [paper-pdf](https://arxiv.org/pdf/2511.15304v2)

**Authors**: Piercosma Bisconti, Matteo Prandi, Federico Pierucci, Francesco Giarrusso, Marcantonio Bracale, Marcello Galisai, Vincenzo Suriani, Olga Sorokoletova, Federico Sartore, Daniele Nardi

**Abstract**: We present evidence that adversarial poetry functions as a universal single-turn jailbreak technique for Large Language Models (LLMs). Across 25 frontier proprietary and open-weight models, curated poetic prompts yielded high attack-success rates (ASR), with some providers exceeding 90%. Mapping prompts to MLCommons and EU CoP risk taxonomies shows that poetic attacks transfer across CBRN, manipulation, cyber-offence, and loss-of-control domains. Converting 1,200 MLCommons harmful prompts into verse via a standardized meta-prompt produced ASRs up to 18 times higher than their prose baselines. Outputs are evaluated using an ensemble of 3 open-weight LLM judges, whose binary safety assessments were validated on a stratified human-labeled subset. Poetic framing achieved an average jailbreak success rate of 62% for hand-crafted poems and approximately 43% for meta-prompt conversions (compared to non-poetic baselines), substantially outperforming non-poetic baselines and revealing a systematic vulnerability across model families and safety training approaches. These findings demonstrate that stylistic variation alone can circumvent contemporary safety mechanisms, suggesting fundamental limitations in current alignment methods and evaluation protocols.



## **18. Securing AI Agents Against Prompt Injection Attacks**

cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15759v1) [paper-pdf](https://arxiv.org/pdf/2511.15759v1)

**Authors**: Badrinath Ramakrishnan, Akshaya Balaji

**Abstract**: Retrieval-augmented generation (RAG) systems have become widely used for enhancing large language model capabilities, but they introduce significant security vulnerabilities through prompt injection attacks. We present a comprehensive benchmark for evaluating prompt injection risks in RAG-enabled AI agents and propose a multi-layered defense framework. Our benchmark includes 847 adversarial test cases across five attack categories: direct injection, context manipulation, instruction override, data exfiltration, and cross-context contamination. We evaluate three defense mechanisms: content filtering with embedding-based anomaly detection, hierarchical system prompt guardrails, and multi-stage response verification, across seven state-of-the-art language models. Our combined framework reduces successful attack rates from 73.2% to 8.7% while maintaining 94.3% of baseline task performance. We release our benchmark dataset and defense implementation to support future research in AI agent security.



## **19. Taxonomy, Evaluation and Exploitation of IPI-Centric LLM Agent Defense Frameworks**

cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15203v1) [paper-pdf](https://arxiv.org/pdf/2511.15203v1)

**Authors**: Zimo Ji, Xunguang Wang, Zongjie Li, Pingchuan Ma, Yudong Gao, Daoyuan Wu, Xincheng Yan, Tian Tian, Shuai Wang

**Abstract**: Large Language Model (LLM)-based agents with function-calling capabilities are increasingly deployed, but remain vulnerable to Indirect Prompt Injection (IPI) attacks that hijack their tool calls. In response, numerous IPI-centric defense frameworks have emerged. However, these defenses are fragmented, lacking a unified taxonomy and comprehensive evaluation. In this Systematization of Knowledge (SoK), we present the first comprehensive analysis of IPI-centric defense frameworks. We introduce a comprehensive taxonomy of these defenses, classifying them along five dimensions. We then thoroughly assess the security and usability of representative defense frameworks. Through analysis of defensive failures in the assessment, we identify six root causes of defense circumvention. Based on these findings, we design three novel adaptive attacks that significantly improve attack success rates targeting specific frameworks, demonstrating the severity of the flaws in these defenses. Our paper provides a foundation and critical insights for the future development of more secure and usable IPI-centric agent defense frameworks.



## **20. As If We've Met Before: LLMs Exhibit Certainty in Recognizing Seen Files**

cs.AI

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.15192v2) [paper-pdf](https://arxiv.org/pdf/2511.15192v2)

**Authors**: Haodong Li, Jingqi Zhang, Xiao Cheng, Peihua Mai, Haoyu Wang, Yan Pang

**Abstract**: The remarkable language ability of Large Language Models (LLMs) stems from extensive training on vast datasets, often including copyrighted material, which raises serious concerns about unauthorized use. While Membership Inference Attacks (MIAs) offer potential solutions for detecting such violations, existing approaches face critical limitations and challenges due to LLMs' inherent overconfidence, limited access to ground truth training data, and reliance on empirically determined thresholds.   We present COPYCHECK, a novel framework that leverages uncertainty signals to detect whether copyrighted content was used in LLM training sets. Our method turns LLM overconfidence from a limitation into an asset by capturing uncertainty patterns that reliably distinguish between ``seen" (training data) and ``unseen" (non-training data) content. COPYCHECK further implements a two-fold strategy: (1) strategic segmentation of files into smaller snippets to reduce dependence on large-scale training data, and (2) uncertainty-guided unsupervised clustering to eliminate the need for empirically tuned thresholds. Experiment results show that COPYCHECK achieves an average balanced accuracy of 90.1% on LLaMA 7b and 91.6% on LLaMA2 7b in detecting seen files. Compared to the SOTA baseline, COPYCHECK achieves over 90% relative improvement, reaching up to 93.8\% balanced accuracy. It further exhibits strong generalizability across architectures, maintaining high performance on GPT-J 6B. This work presents the first application of uncertainty for copyright detection in LLMs, offering practical tools for training data transparency.



## **21. Can MLLMs Detect Phishing? A Comprehensive Security Benchmark Suite Focusing on Dynamic Threats and Multimodal Evaluation in Academic Environments**

cs.CR

**SubmitDate**: 2025-11-22    [abs](http://arxiv.org/abs/2511.15165v2) [paper-pdf](https://arxiv.org/pdf/2511.15165v2)

**Authors**: Jingzhuo Zhou

**Abstract**: The rapid proliferation of Multimodal Large Language Models (MLLMs) has introduced unprecedented security challenges, particularly in phishing detection within academic environments. Academic institutions and researchers are high-value targets, facing dynamic, multilingual, and context-dependent threats that leverage research backgrounds, academic collaborations, and personal information to craft highly tailored attacks. Existing security benchmarks largely rely on datasets that do not incorporate specific academic background information, making them inadequate for capturing the evolving attack patterns and human-centric vulnerability factors specific to academia. To address this gap, we present AdapT-Bench, a unified methodological framework and benchmark suite for systematically evaluating MLLM defense capabilities against dynamic phishing attacks in academic settings.



## **22. Unified Defense for Large Language Models against Jailbreak and Fine-Tuning Attacks in Education**

cs.CL

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14423v1) [paper-pdf](https://arxiv.org/pdf/2511.14423v1)

**Authors**: Xin Yi, Yue Li, Dongsheng Shi, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Large Language Models (LLMs) are increasingly integrated into educational applications. However, they remain vulnerable to jailbreak and fine-tuning attacks, which can compromise safety alignment and lead to harmful outputs. Existing studies mainly focus on general safety evaluations, with limited attention to the unique safety requirements of educational scenarios. To address this gap, we construct EduHarm, a benchmark containing safe-unsafe instruction pairs across five representative educational scenarios, enabling systematic safety evaluation of educational LLMs. Furthermore, we propose a three-stage shield framework (TSSF) for educational LLMs that simultaneously mitigates both jailbreak and fine-tuning attacks. First, safety-aware attention realignment redirects attention toward critical unsafe tokens, thereby restoring the harmfulness feature that discriminates between unsafe and safe inputs. Second, layer-wise safety judgment identifies harmfulness features by aggregating safety cues across multiple layers to detect unsafe instructions. Finally, defense-driven dual routing separates safe and unsafe queries, ensuring normal processing for benign inputs and guarded responses for harmful ones. Extensive experiments across eight jailbreak attack strategies demonstrate that TSSF effectively strengthens safety while preventing over-refusal of benign queries. Evaluations on three fine-tuning attack datasets further show that it consistently achieves robust defense against harmful queries while maintaining preserving utility gains from benign fine-tuning.



## **23. Beyond Fixed and Dynamic Prompts: Embedded Jailbreak Templates for Advancing LLM Security**

cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14140v1) [paper-pdf](https://arxiv.org/pdf/2511.14140v1)

**Authors**: Hajun Kim, Hyunsik Na, Daeseon Choi

**Abstract**: As the use of large language models (LLMs) continues to expand, ensuring their safety and robustness has become a critical challenge. In particular, jailbreak attacks that bypass built-in safety mechanisms are increasingly recognized as a tangible threat across industries, driving the need for diverse templates to support red-teaming efforts and strengthen defensive techniques. However, current approaches predominantly rely on two limited strategies: (i) substituting harmful queries into fixed templates, and (ii) having the LLM generate entire templates, which often compromises intent clarity and reproductibility. To address this gap, this paper introduces the Embedded Jailbreak Template, which preserves the structure of existing templates while naturally embedding harmful queries within their context. We further propose a progressive prompt-engineering methodology to ensure template quality and consistency, alongside standardized protocols for generation and evaluation. Together, these contributions provide a benchmark that more accurately reflects real-world usage scenarios and harmful intent, facilitating its application in red-teaming and policy regression testing.



## **24. GRPO Privacy Is at Risk: A Membership Inference Attack Against Reinforcement Learning With Verifiable Rewards**

cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14045v1) [paper-pdf](https://arxiv.org/pdf/2511.14045v1)

**Authors**: Yule Liu, Heyi Zhang, Jinyi Zheng, Zhen Sun, Zifan Peng, Tianshuo Cong, Yilong Yang, Xinlei He, Zhuo Ma

**Abstract**: Membership inference attacks (MIAs) on large language models (LLMs) pose significant privacy risks across various stages of model training. Recent advances in Reinforcement Learning with Verifiable Rewards (RLVR) have brought a profound paradigm shift in LLM training, particularly for complex reasoning tasks. However, the on-policy nature of RLVR introduces a unique privacy leakage pattern: since training relies on self-generated responses without fixed ground-truth outputs, membership inference must now determine whether a given prompt (independent of any specific response) is used during fine-tuning. This creates a threat where leakage arises not from answer memorization.   To audit this novel privacy risk, we propose Divergence-in-Behavior Attack (DIBA), the first membership inference framework specifically designed for RLVR. DIBA shifts the focus from memorization to behavioral change, leveraging measurable shifts in model behavior across two axes: advantage-side improvement (e.g., correctness gain) and logit-side divergence (e.g., policy drift). Through comprehensive evaluations, we demonstrate that DIBA significantly outperforms existing baselines, achieving around 0.8 AUC and an order-of-magnitude higher TPR@0.1%FPR. We validate DIBA's superiority across multiple settings--including in-distribution, cross-dataset, cross-algorithm, black-box scenarios, and extensions to vision-language models. Furthermore, our attack remains robust under moderate defensive measures.   To the best of our knowledge, this is the first work to systematically analyze privacy vulnerabilities in RLVR, revealing that even in the absence of explicit supervision, training data exposure can be reliably inferred through behavioral traces.



## **25. Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment**

cs.CR

AAAI-26-AIA

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.06852v4) [paper-pdf](https://arxiv.org/pdf/2511.06852v4)

**Authors**: Peng Zhang, Peijie Sun

**Abstract**: Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment.



## **26. Adaptive and Robust Data Poisoning Detection and Sanitization in Wearable IoT Systems using Large Language Models**

cs.LG

**SubmitDate**: 2025-11-21    [abs](http://arxiv.org/abs/2511.02894v3) [paper-pdf](https://arxiv.org/pdf/2511.02894v3)

**Authors**: W. K. M Mithsara, Ning Yang, Ahmed Imteaj, Hussein Zangoti, Abdur R. Shahid

**Abstract**: The widespread integration of wearable sensing devices in Internet of Things (IoT) ecosystems, particularly in healthcare, smart homes, and industrial applications, has required robust human activity recognition (HAR) techniques to improve functionality and user experience. Although machine learning models have advanced HAR, they are increasingly susceptible to data poisoning attacks that compromise the data integrity and reliability of these systems. Conventional approaches to defending against such attacks often require extensive task-specific training with large, labeled datasets, which limits adaptability in dynamic IoT environments. This work proposes a novel framework that uses large language models (LLMs) to perform poisoning detection and sanitization in HAR systems, utilizing zero-shot, one-shot, and few-shot learning paradigms. Our approach incorporates \textit{role play} prompting, whereby the LLM assumes the role of expert to contextualize and evaluate sensor anomalies, and \textit{think step-by-step} reasoning, guiding the LLM to infer poisoning indicators in the raw sensor data and plausible clean alternatives. These strategies minimize reliance on curation of extensive datasets and enable robust, adaptable defense mechanisms in real-time. We perform an extensive evaluation of the framework, quantifying detection accuracy, sanitization quality, latency, and communication cost, thus demonstrating the practicality and effectiveness of LLMs in improving the security and reliability of wearable IoT systems.



## **27. DRIP: Defending Prompt Injection via Token-wise Representation Editing and Residual Instruction Fusion**

cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.00447v2) [paper-pdf](https://arxiv.org/pdf/2511.00447v2)

**Authors**: Ruofan Liu, Yun Lin, Zhiyong Huang, Jin Song Dong

**Abstract**: Large language models (LLMs) are increasingly integrated into IT infrastructures, where they process user data according to predefined instructions. However, conventional LLMs remain vulnerable to prompt injection, where malicious users inject directive tokens into the data to subvert model behavior. Existing defenses train LLMs to semantically separate data and instruction tokens, but still struggle to (1) balance utility and security and (2) prevent instruction-like semantics in the data from overriding the intended instructions.   We propose DRIP, which (1) precisely removes instruction semantics from tokens in the data section while preserving their data semantics, and (2) robustly preserves the effect of the intended instruction even under strong adversarial content. To "de-instructionalize" data tokens, DRIP introduces a data curation and training paradigm with a lightweight representation-editing module that edits embeddings of instruction-like tokens in the data section, enhancing security without harming utility. To ensure non-overwritability of instructions, DRIP adds a minimal residual module that reduces the ability of adversarial data to overwrite the original instruction. We evaluate DRIP on LLaMA 8B and Mistral 7B against StruQ, SecAlign, ISE, and PFT on three prompt-injection benchmarks (SEP, AlpacaFarm, and InjecAgent). DRIP improves role-separation score by 12-49\%, reduces attack success rate by over 66\% under adaptive attacks, and matches the utility of the undefended model, establishing a new state of the art for prompt-injection robustness.



## **28. SoK: Honeypots & LLMs, More Than the Sum of Their Parts?**

cs.CR

Systemization of Knowledge

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2510.25939v3) [paper-pdf](https://arxiv.org/pdf/2510.25939v3)

**Authors**: Robert A. Bridges, Thomas R. Mitchell, Mauricio Muñoz, Ted Henriksson

**Abstract**: The advent of Large Language Models (LLMs) promised to resolve the long-standing paradox in honeypot design, achieving high-fidelity deception with low operational risk. Through a flurry of research since late 2022, steady progress from ideation to prototype implementation is exhibited. Since late 2022, a flurry of research has demonstrated steady progress from ideation to prototype implementation. While promising, evaluations show only incremental progress in real-world deployments, and the field still lacks a cohesive understanding of the emerging architectural patterns, core challenges, and evaluation paradigms. To fill this gap, this Systematization of Knowledge (SoK) paper provides the first comprehensive overview and analysis of this new domain. We survey and systematize the field by focusing on three critical, intersecting research areas: first, we provide a taxonomy of honeypot detection vectors, structuring the core problems that LLM-based realism must solve; second, we synthesize the emerging literature on LLM-powered honeypots, identifying a canonical architecture and key evaluation trends; and third, we chart the evolutionary path of honeypot log analysis, from simple data reduction to automated intelligence generation. We synthesize these findings into a forward-looking research roadmap, arguing that the true potential of this technology lies in creating autonomous, self-improving deception systems to counter the emerging threat of intelligent, automated attackers.



## **29. Practical and Stealthy Touch-Guided Jailbreak Attacks on Deployed Mobile Vision-Language Agents**

cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2510.07809v2) [paper-pdf](https://arxiv.org/pdf/2510.07809v2)

**Authors**: Renhua Ding, Xiao Yang, Zhengwei Fang, Jun Luo, Kun He, Jun Zhu

**Abstract**: Large vision-language models (LVLMs) enable autonomous mobile agents to operate smartphone user interfaces, yet vulnerabilities in their perception and interaction remain critically understudied. Existing research often relies on conspicuous overlays, elevated permissions, or unrealistic threat assumptions, limiting stealth and real-world feasibility. In this paper, we introduce a practical and stealthy jailbreak attack framework, which comprises three key components: (i) non-privileged perception compromise, which injects visual payloads into the application interface without requiring elevated system permissions; (ii) agent-attributable activation, which leverages input attribution signals to distinguish agent from human interactions and limits prompt exposure to transient intervals to preserve stealth from end users; and (iii) efficient one-shot jailbreak, a heuristic iterative deepening search algorithm (HG-IDA*) that performs keyword-level detoxification to bypass built-in safety alignment of LVLMs. Moreover, we developed three representative Android applications and curated a prompt-injection dataset for mobile agents. We evaluated our attack across multiple LVLM backends, including closed-source services and representative open-source models, and observed high planning and execution hijack rates (e.g., GPT-4o: 82.5% planning / 75.0% execution), exposing a fundamental security vulnerability in current mobile agents and underscoring critical implications for autonomous smartphone operation.



## **30. Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks**

cs.CL

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2510.02712v2) [paper-pdf](https://arxiv.org/pdf/2510.02712v2)

**Authors**: Yubo Li, Ramayya Krishnan, Rema Padman

**Abstract**: Large Language Models (LLMs) have revolutionized conversational AI, yet their robustness in extended multi-turn dialogues remains poorly understood. Existing evaluation frameworks focus on static benchmarks and single-turn assessments, failing to capture the temporal dynamics of conversational degradation that characterize real-world interactions. In this work, we present a large-scale survival analysis of conversational robustness, modeling failure as a time-to-event process over 36,951 turns from 9 state-of-the-art LLMs on the MT-Consistency benchmark. Our framework combines Cox proportional hazards, Accelerated Failure Time (AFT), and Random Survival Forest models with simple semantic drift features. We find that abrupt prompt-to-prompt semantic drift sharply increases the hazard of inconsistency, whereas cumulative drift is counterintuitively \emph{protective}, suggesting adaptation in conversations that survive multiple shifts. AFT models with model-drift interactions achieve the best combination of discrimination and calibration, and proportional hazards checks reveal systematic violations for key drift covariates, explaining the limitations of Cox-style modeling in this setting. Finally, we show that a lightweight AFT model can be turned into a turn-level risk monitor that flags most failing conversations several turns before the first inconsistent answer while keeping false alerts modest. These results establish survival analysis as a powerful paradigm for evaluating multi-turn robustness and for designing practical safeguards for conversational AI systems.



## **31. Boundary on the Table: Efficient Black-Box Decision-Based Attacks for Structured Data**

cs.LG

Paper revision

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2509.22850v3) [paper-pdf](https://arxiv.org/pdf/2509.22850v3)

**Authors**: Roie Kazoom, Yuval Ratzabi, Etamar Rothstein, Ofer Hadar

**Abstract**: Adversarial robustness in structured data remains an underexplored frontier compared to vision and language domains. In this work, we introduce a novel black-box, decision-based adversarial attack tailored for tabular data. Our approach combines gradient-free direction estimation with an iterative boundary search, enabling efficient navigation of discrete and continuous feature spaces under minimal oracle access. Extensive experiments demonstrate that our method successfully compromises nearly the entire test set across diverse models, ranging from classical machine learning classifiers to large language model (LLM)-based pipelines. Remarkably, the attack achieves success rates consistently above 90%, while requiring only a small number of queries per instance. These results highlight the critical vulnerability of tabular models to adversarial perturbations, underscoring the urgent need for stronger defenses in real-world decision-making systems.



## **32. Guided Reasoning in LLM-Driven Penetration Testing Using Structured Attack Trees**

cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2509.07939v2) [paper-pdf](https://arxiv.org/pdf/2509.07939v2)

**Authors**: Katsuaki Nakano, Reza Fayyazi, Shanchieh Jay Yang, Michael Zuzak

**Abstract**: Recent advances in Large Language Models (LLMs) have driven interest in automating cybersecurity penetration testing workflows, offering the promise of faster and more consistent vulnerability assessment for enterprise systems. Existing LLM agents for penetration testing primarily rely on self-guided reasoning, which can produce inaccurate or hallucinated procedural steps. As a result, the LLM agent may undertake unproductive actions, such as exploiting unused software libraries or generating cyclical responses that repeat prior tactics. In this work, we propose a guided reasoning pipeline for penetration testing LLM agents that incorporates a deterministic task tree built from the MITRE ATT&CK Matrix, a proven penetration testing kll chain, to constrain the LLM's reaoning process to explicitly defined tactics, techniques, and procedures. This anchors reasoning in proven penetration testing methodologies and filters out ineffective actions by guiding the agent towards more productive attack procedures. To evaluate our approach, we built an automated penetration testing LLM agent using three LLMs (Llama-3-8B, Gemini-1.5, and GPT-4) and applied it to navigate 10 HackTheBox cybersecurity exercises with 103 discrete subtasks representing real-world cyberattack scenarios. Our proposed reasoning pipeline guided the LLM agent through 71.8\%, 72.8\%, and 78.6\% of subtasks using Llama-3-8B, Gemini-1.5, and GPT-4, respectively. Comparatively, the state-of-the-art LLM penetration testing tool using self-guided reasoning completed only 13.5\%, 16.5\%, and 75.7\% of subtasks and required 86.2\%, 118.7\%, and 205.9\% more model queries. This suggests that incorporating a deterministic task tree into LLM reasoning pipelines can enhance the accuracy and efficiency of automated cybersecurity assessments



## **33. PromptCOS: Towards Content-only System Prompt Copyright Auditing for LLMs**

cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2509.03117v2) [paper-pdf](https://arxiv.org/pdf/2509.03117v2)

**Authors**: Yuchen Yang, Yiming Li, Hongwei Yao, Enhao Huang, Shuo Shao, Yuyi Wang, Zhibo Wang, Dacheng Tao, Zhan Qin

**Abstract**: System prompts are critical for shaping the behavior and output quality of large language model (LLM)-based applications, driving substantial investment in optimizing high-quality prompts beyond traditional handcrafted designs. However, as system prompts become valuable intellectual property, they are increasingly vulnerable to prompt theft and unauthorized use, highlighting the urgent need for effective copyright auditing, especially watermarking. Existing methods rely on verifying subtle logit distribution shifts triggered by a query. We observe that this logit-dependent verification framework is impractical in real-world content-only settings, primarily because (1) random sampling makes content-level generation unstable for verification, and (2) stronger instructions needed for content-level signals compromise prompt fidelity.   To overcome these challenges, we propose PromptCOS, the first content-only system prompt copyright auditing method based on content-level output similarity. PromptCOS achieves watermark stability by designing a cyclic output signal as the conditional instruction's target. It preserves prompt fidelity by injecting a small set of auxiliary tokens to encode the watermark, leaving the main prompt untouched. Furthermore, to ensure robustness against malicious removal, we optimize cover tokens, i.e., critical tokens in the original prompt, to ensure that removing auxiliary tokens causes severe performance degradation. Experimental results show that PromptCOS achieves high effectiveness (99.3% average watermark similarity), strong distinctiveness (60.8% higher than the best baseline), high fidelity (accuracy degradation no greater than 0.6%), robustness (resilience against four potential attack categories), and high computational efficiency (up to 98.1% cost saving). Our code is available at GitHub (https://github.com/LianPing-cyber/PromptCOS).



## **34. SoK: Exposing the Generation and Detection Gaps in LLM-Generated Phishing Through Examination of Generation Methods, Content Characteristics, and Countermeasures**

cs.CR

18 pages, 5 tables, 4 figures

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2508.21457v2) [paper-pdf](https://arxiv.org/pdf/2508.21457v2)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Carsten Rudolph

**Abstract**: Phishing campaigns involve adversaries masquerading as trusted vendors trying to trigger user behavior that enables them to exfiltrate private data. While URLs are an important part of phishing campaigns, communicative elements like text and images are central in triggering the required user behavior. Further, due to advances in phishing detection, attackers react by scaling campaigns to larger numbers and diversifying and personalizing content. In addition to established mechanisms, such as template-based generation, large language models (LLMs) can be used for phishing content generation, enabling attacks to scale in minutes, challenging existing phishing detection paradigms through personalized content, stealthy explicit phishing keywords, and dynamic adaptation to diverse attack scenarios. Countering these dynamically changing attack campaigns requires a comprehensive understanding of the complex LLM-related threat landscape. Existing studies are fragmented and focus on specific areas. In this work, we provide the first holistic examination of LLM-generated phishing content. First, to trace the exploitation pathways of LLMs for phishing content generation, we adopt a modular taxonomy documenting nine stages by which adversaries breach LLM safety guardrails. We then characterize how LLM-generated phishing manifests as threats, revealing that it evades detectors while emphasizing human cognitive manipulation. Third, by taxonomizing defense techniques aligned with generation methods, we expose a critical asymmetry that offensive mechanisms adapt dynamically to attack scenarios, whereas defensive strategies remain static and reactive. Finally, based on a thorough analysis of the existing literature, we highlight insights and gaps and suggest a roadmap for understanding and countering LLM-driven phishing at scale.



## **35. Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models**

cs.CR

16 pages; Previously this version appeared as arXiv:2510.15430 which was submitted as a new work by accident

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2508.09201v3) [paper-pdf](https://arxiv.org/pdf/2508.09201v3)

**Authors**: Shuang Liang, Zhihao Xu, Jialing Tao, Hui Xue, Xiting Wang

**Abstract**: Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. To address this, existing detection methods either learn attack-specific parameters, which hinders generalization to unseen attacks, or rely on heuristically sound principles, which limit accuracy and efficiency. To overcome these limitations, we propose Learning to Detect (LoD), a general framework that accurately detects unknown jailbreak attacks by shifting the focus from attack-specific learning to task-specific learning. This framework includes a Multi-modal Safety Concept Activation Vector module for safety-oriented representation learning and a Safety Pattern Auto-Encoder module for unsupervised attack classification. Extensive experiments show that our method achieves consistently higher detection AUROC on diverse unknown attacks while improving efficiency. The code is available at https://anonymous.4open.science/r/Learning-to-Detect-51CB.



## **36. Hidden in the Noise: Unveiling Backdoors in Audio LLMs Alignment through Latent Acoustic Pattern Triggers**

cs.SD

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2508.02175v3) [paper-pdf](https://arxiv.org/pdf/2508.02175v3)

**Authors**: Liang Lin, Miao Yu, Kaiwen Luo, Yibo Zhang, Lilan Peng, Dexian Wang, Xuehai Tang, Yuanhe Zhang, Xikang Yang, Zhenhong Zhou, Kun Wang, Yang Liu

**Abstract**: As Audio Large Language Models (ALLMs) emerge as powerful tools for speech processing, their safety implications demand urgent attention. While considerable research has explored textual and vision safety, audio's distinct characteristics present significant challenges. This paper first investigates: Is ALLM vulnerable to backdoor attacks exploiting acoustic triggers? In response to this issue, we introduce Hidden in the Noise (HIN), a novel backdoor attack framework designed to exploit subtle, audio-specific features. HIN applies acoustic modifications to raw audio waveforms, such as alterations to temporal dynamics and strategic injection of spectrally tailored noise. These changes introduce consistent patterns that an ALLM's acoustic feature encoder captures, embedding robust triggers within the audio stream. To evaluate ALLM robustness against audio-feature-based triggers, we develop the AudioSafe benchmark, assessing nine distinct risk types. Extensive experiments on AudioSafe and three established safety datasets reveal critical vulnerabilities in existing ALLMs: (I) audio features like environment noise and speech rate variations achieve over 90% average attack success rate. (II) ALLMs exhibit significant sensitivity differences across acoustic features, particularly showing minimal response to volume as a trigger, and (III) poisoned sample inclusion causes only marginal loss curve fluctuations, highlighting the attack's stealth.



## **37. AgentArmor: Enforcing Program Analysis on Agent Runtime Trace to Defend Against Prompt Injection**

cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2508.01249v3) [paper-pdf](https://arxiv.org/pdf/2508.01249v3)

**Authors**: Peiran Wang, Yang Liu, Yunfei Lu, Yifeng Cai, Hongbo Chen, Qingyou Yang, Jie Zhang, Jue Hong, Ye Wu

**Abstract**: Large Language Model (LLM) agents offer a powerful new paradigm for solving various problems by combining natural language reasoning with the execution of external tools. However, their dynamic and non-transparent behavior introduces critical security risks, particularly in the presence of prompt injection attacks. In this work, we propose a novel insight that treats the agent runtime traces as structured programs with analyzable semantics. Thus, we present AgentArmor, a program analysis framework that converts agent traces into graph intermediate representation-based structured program dependency representations (e.g., CFG, DFG, and PDG) and enforces security policies via a type system. AgentArmor consists of three key components: (1) a graph constructor that reconstructs the agent's runtime traces as graph-based intermediate representations with control and data flow described within; (2) a property registry that attaches security-relevant metadata of interacted tools \& data, and (3) a type system that performs static inference and checking over the intermediate representation. By representing agent behavior as structured programs, AgentArmor enables program analysis for sensitive data flow, trust boundaries, and policy violations. We evaluate AgentArmor on the AgentDojo benchmark, the results show that AgentArmor can reduce the ASR to 3\%, with the utility drop only 1\%.



## **38. Fine-Grained Privacy Extraction from Retrieval-Augmented Generation Systems via Knowledge Asymmetry Exploitation**

cs.CR

**SubmitDate**: 2025-11-22    [abs](http://arxiv.org/abs/2507.23229v2) [paper-pdf](https://arxiv.org/pdf/2507.23229v2)

**Authors**: Yufei Chen, Yao Wang, Haibin Zhang, Tao Gu

**Abstract**: Retrieval-augmented generation (RAG) systems enhance large language models (LLMs) by integrating external knowledge bases, but this advancement introduces significant privacy risks. Existing privacy attacks on RAG systems can trigger data leakage but often fail to accurately isolate knowledge-base-derived sentences within mixed responses. They also lack robustness when applied across multiple domains. This paper addresses these challenges by presenting a novel black-box attack framework that exploits knowledge asymmetry between RAG and standard LLMs to achieve fine-grained privacy extraction across heterogeneous knowledge landscapes. We propose a chain-of-thought reasoning strategy that creates adaptive prompts to steer RAG systems away from sensitive content. Specifically, we first decompose adversarial queries to maximize information disparity and then apply a semantic relationship scoring to resolve lexical and syntactic ambiguities. We finally train a neural network on these feature scores to precisely identify sentences containing private information. Unlike prior work, our framework generalizes to unseen domains through iterative refinement without pre-defined knowledge. Experimental results show that we achieve over 91% privacy extraction rate in single-domain and 83% in multi-domain scenarios, reducing sensitive sentence exposure by over 65% in case studies. This work bridges the gap between attack and defense in RAG systems, enabling precise extraction of private information while providing a foundation for adaptive mitigation.



## **39. Response Attack: Exploiting Contextual Priming to Jailbreak Large Language Models**

cs.CL

20 pages, 10 figures. Code and data available at https://github.com/Dtc7w3PQ/Response-Attack

**SubmitDate**: 2025-11-21    [abs](http://arxiv.org/abs/2507.05248v2) [paper-pdf](https://arxiv.org/pdf/2507.05248v2)

**Authors**: Ziqi Miao, Lijun Li, Yuan Xiong, Zhenhua Liu, Pengyu Zhu, Jing Shao

**Abstract**: Contextual priming, where earlier stimuli covertly bias later judgments, offers an unexplored attack surface for large language models (LLMs). We uncover a contextual priming vulnerability in which the previous response in the dialogue can steer its subsequent behavior toward policy-violating content. While existing jailbreak attacks largely rely on single-turn or multi-turn prompt manipulations, or inject static in-context examples, these methods suffer from limited effectiveness, inefficiency, or semantic drift. We introduce Response Attack (RA), a novel framework that strategically leverages intermediate, mildly harmful responses as contextual primers within a dialogue. By reformulating harmful queries and injecting these intermediate responses before issuing a targeted trigger prompt, RA exploits a previously overlooked vulnerability in LLMs. Extensive experiments across eight state-of-the-art LLMs show that RA consistently achieves significantly higher attack success rates than nine leading jailbreak baselines. Our results demonstrate that the success of RA is directly attributable to the strategic use of intermediate responses, which induce models to generate more explicit and relevant harmful content while maintaining stealth, efficiency, and fidelity to the original query. The code and data are available at https://github.com/Dtc7w3PQ/Response-Attack.



## **40. Backdoors in Conditional Diffusion: Threats to Responsible Synthetic Data Pipelines**

cs.CV

Accepted at RDS @ AAAI 2026

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2507.04726v2) [paper-pdf](https://arxiv.org/pdf/2507.04726v2)

**Authors**: Raz Lapid, Almog Dubin

**Abstract**: Text-to-image diffusion models achieve high-fidelity image generation from natural language prompts. ControlNets extend these models by enabling conditioning on structural inputs (e.g., edge maps, depth, pose), providing fine-grained control over outputs. Yet their reliance on large, publicly scraped datasets and community fine-tuning makes them vulnerable to data poisoning. We introduce a model-poisoning attack that embeds a covert backdoor into a ControlNet, causing it to produce attacker-specified content when exposed to visual triggers, without textual prompts. Experiments show that poisoning only 1% of the fine-tuning corpus yields a 90-98% attack success rate, while 5% further strengthens the backdoor, all while preserving normal generation quality. To mitigate this risk, we propose clean fine-tuning (CFT): freezing the diffusion backbone and fine-tuning only the ControlNet on a sanitized dataset with a reduced learning rate. CFT lowers attack success rates on held-out data. These results expose a critical security weakness in open-source, ControlNet-guided diffusion pipelines and demonstrate that CFT offers a practical defense for responsible synthetic-data pipelines.



## **41. Large Language Model Unlearning for Source Code**

cs.SE

Accepted to AAAI'26

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2506.17125v2) [paper-pdf](https://arxiv.org/pdf/2506.17125v2)

**Authors**: Xue Jiang, Yihong Dong, Huangzhao Zhang, Tangxinyu Wang, Zheng Fang, Yingwei Ma, Rongyu Cao, Binhua Li, Zhi Jin, Wenpin Jiao, Yongbin Li, Ge Li

**Abstract**: While Large Language Models (LLMs) excel at code generation, their inherent tendency toward verbatim memorization of training data introduces critical risks like copyright infringement, insecure emission, and deprecated API utilization, etc. A straightforward yet promising defense is unlearning, ie., erasing or down-weighting the offending snippets through post-training. However, we find its application to source code often tends to spill over, damaging the basic knowledge of programming languages learned by the LLM and degrading the overall capability. To ease this challenge, we propose PROD for precise source code unlearning. PROD surgically zeroes out the prediction probability of the prohibited tokens, and renormalizes the remaining distribution so that the generated code stays correct. By excising only the targeted snippets, PROD achieves precise forgetting without much degradation of the LLM's overall capability. To facilitate in-depth evaluation against PROD, we establish an unlearning benchmark consisting of three downstream tasks (ie., unlearning of copyrighted code, insecure code, and deprecated APIs), and introduce Pareto Dominance Ratio (PDR) metric, which indicates both the forget quality and the LLM utility. Our comprehensive evaluation demonstrates that PROD achieves superior overall performance between forget quality and model utility compared to existing unlearning approaches across three downstream tasks, while consistently exhibiting improvements when applied to LLMs of varying series. PROD also exhibits superior robustness against adversarial attacks without generating or exposing the data to be forgotten. These results underscore that our approach not only successfully extends the application boundary of unlearning techniques to source code, but also holds significant implications for advancing reliable code generation.



## **42. Safeguarding Privacy of Retrieval Data against Membership Inference Attacks: Is This Query Too Close to Home?**

cs.CL

Accepted for EMNLP findings 2025

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2505.22061v3) [paper-pdf](https://arxiv.org/pdf/2505.22061v3)

**Authors**: Yujin Choi, Youngjoo Park, Junyoung Byun, Jaewook Lee, Jinseong Park

**Abstract**: Retrieval-augmented generation (RAG) mitigates the hallucination problem in large language models (LLMs) and has proven effective for personalized usages. However, delivering private retrieved documents directly to LLMs introduces vulnerability to membership inference attacks (MIAs), which try to determine whether the target data point exists in the private external database or not. Based on the insight that MIA queries typically exhibit high similarity to only one target document, we introduce a novel similarity-based MIA detection framework designed for the RAG system. With the proposed method, we show that a simple detect-and-hide strategy can successfully obfuscate attackers, maintain data utility, and remain system-agnostic against MIA. We experimentally prove its detection and defense against various state-of-the-art MIA methods and its adaptability to existing RAG systems.



## **43. Revisiting Model Inversion Evaluation: From Misleading Standards to Reliable Privacy Assessment**

cs.LG

To support future work, we release our MLLM-based MI evaluation framework and benchmarking suite at https://github.com/hosytuyen/MI-Eval-MLLM

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2505.03519v4) [paper-pdf](https://arxiv.org/pdf/2505.03519v4)

**Authors**: Sy-Tuyen Ho, Koh Jun Hao, Ngoc-Bao Nguyen, Alexander Binder, Ngai-Man Cheung

**Abstract**: Model Inversion (MI) attacks aim to reconstruct information from private training data by exploiting access to machine learning models T. To evaluate such attacks, the standard evaluation framework relies on an evaluation model E, trained under the same task design as T. This framework has become the de facto standard for assessing progress in MI research, used across nearly all recent MI studies without question. In this paper, we present the first in-depth study of this evaluation framework. In particular, we identify a critical issue of this standard framework: Type-I adversarial examples. These are reconstructions that do not capture the visual features of private training data, yet are still deemed successful by T and ultimately transferable to E. Such false positives undermine the reliability of the standard MI evaluation framework. To address this issue, we introduce a new MI evaluation framework that replaces the evaluation model E with advanced Multimodal Large Language Models (MLLMs). By leveraging their general-purpose visual understanding, our MLLM-based framework does not depend on training of shared task design as in T, thus reducing Type-I transferability and providing more faithful assessments of reconstruction success. Using our MLLM-based evaluation framework, we reevaluate 27 diverse MI attack setups and empirically reveal consistently high false positive rates under the standard evaluation framework. Importantly, we demonstrate that many state-of-the-art (SOTA) MI methods report inflated attack accuracy, indicating that actual privacy leakage is significantly lower than previously believed. By uncovering this critical issue and proposing a robust solution, our work enables a reassessment of progress in MI research and sets a new standard for reliable and robust evaluation. Code can be found in https://github.com/hosytuyen/MI-Eval-MLLM



## **44. One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image**

cs.CL

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2504.02132v3) [paper-pdf](https://arxiv.org/pdf/2504.02132v3)

**Authors**: Ezzeldin Shereen, Dan Ristea, Shae McFadden, Burak Hasircioglu, Vasilios Mavroudis, Chris Hicks

**Abstract**: Retrieval-augmented generation (RAG) is instrumental for inhibiting hallucinations in large language models (LLMs) through the use of a factual knowledge base (KB). Although PDF documents are prominent sources of knowledge, text-based RAG pipelines are ineffective at capturing their rich multi-modal information. In contrast, visual document RAG (VD-RAG) uses screenshots of document pages as the KB, which has been shown to achieve state-of-the-art results. However, by introducing the image modality, VD-RAG introduces new attack vectors for adversaries to disrupt the system by injecting malicious documents into the KB. In this paper, we demonstrate the vulnerability of VD-RAG to poisoning attacks targeting both retrieval and generation. We define two attack objectives and demonstrate that both can be realized by injecting only a single adversarial image into the KB. Firstly, we introduce a targeted attack against one or a group of queries with the goal of spreading targeted disinformation. Secondly, we present a universal attack that, for any potential user query, influences the response to cause a denial-of-service in the VD-RAG system. We investigate the two attack objectives under both white-box and black-box assumptions, employing a multi-objective gradient-based optimization approach as well as prompting state-of-the-art generative models. Using two visual document datasets, a diverse set of state-of-the-art retrievers (embedding models) and generators (vision language models), we show VD-RAG is vulnerable to poisoning attacks in both the targeted and universal settings, yet demonstrating robustness to black-box attacks in the universal setting.



## **45. LightDefense: A Lightweight Uncertainty-Driven Defense against Jailbreaks via Shifted Token Distribution**

cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2504.01533v2) [paper-pdf](https://arxiv.org/pdf/2504.01533v2)

**Authors**: Zhuoran Yang, Yanyong Zhang

**Abstract**: Large Language Models (LLMs) face threats from jailbreak prompts. Existing methods for defending against jailbreak attacks are primarily based on auxiliary models. These strategies, however, often require extensive data collection or training. We propose LightDefense, a lightweight defense mechanism targeted at white-box models, which utilizes a safety-oriented direction to adjust the probabilities of tokens in the vocabulary, making safety disclaimers appear among the top tokens after sorting tokens by probability in descending order. We further innovatively leverage LLM's uncertainty about prompts to measure their harmfulness and adaptively adjust defense strength, effectively balancing safety and helpfulness. The effectiveness of LightDefense in defending against 5 attack methods across 2 target LLMs, without compromising helpfulness to benign user queries, highlights its potential as a novel and lightweight defense mechanism, enhancing security of LLMs.



## **46. IPAD: Inverse Prompt for AI Detection - A Robust and Interpretable LLM-Generated Text Detector**

cs.LG

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2502.15902v3) [paper-pdf](https://arxiv.org/pdf/2502.15902v3)

**Authors**: Zheng Chen, Yushi Feng, Jisheng Dang, Yue Deng, Changyang He, Hongxi Pu, Haoxuan Li, Bo Li

**Abstract**: Large Language Models (LLMs) have attained human-level fluency in text generation, which complicates the distinguishing between human-written and LLM-generated texts. This increases the risk of misuse and highlights the need for reliable detectors. Yet, existing detectors exhibit poor robustness on out-of-distribution (OOD) data and attacked data, which is critical for real-world scenarios. Also, they struggle to provide interpretable evidence to support their decisions, thus undermining the reliability. In light of these challenges, we propose IPAD (Inverse Prompt for AI Detection), a novel framework consisting of a Prompt Inverter that identifies predicted prompts that could have generated the input text, and two Distinguishers that examine the probability that the input texts align with the predicted prompts. Empirical evaluations demonstrate that IPAD outperforms the strongest baselines by 9.05% (Average Recall) on in-distribution data, 12.93% (AUROC) on out-of-distribution data, and 5.48% (AUROC) on attacked data. IPAD also performs robustly on structured datasets. Furthermore, an interpretability assessment is conducted to illustrate that IPAD enhances the AI detection trustworthiness by allowing users to directly examine the decision-making evidence, which provides interpretable support for its state-of-the-art detection results.



## **47. Exploring Potential Prompt Injection Attacks in Federated Military LLMs and Their Mitigation**

cs.LG

Accepted to the 3rd International Workshop on Dataspaces and Digital Twins for Critical Entities and Smart Urban Communities - IEEE BigData 2025

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2501.18416v2) [paper-pdf](https://arxiv.org/pdf/2501.18416v2)

**Authors**: Youngjoon Lee, Taehyun Park, Yunho Lee, Jinu Gong, Joonhyuk Kang

**Abstract**: Federated Learning (FL) is increasingly being adopted in military collaborations to develop Large Language Models (LLMs) while preserving data sovereignty. However, prompt injection attacks-malicious manipulations of input prompts-pose new threats that may undermine operational security, disrupt decision-making, and erode trust among allies. This perspective paper highlights four vulnerabilities in federated military LLMs: secret data leakage, free-rider exploitation, system disruption, and misinformation spread. To address these risks, we propose a human-AI collaborative framework with both technical and policy countermeasures. On the technical side, our framework uses red/blue team wargaming and quality assurance to detect and mitigate adversarial behaviors of shared LLM weights. On the policy side, it promotes joint AI-human policy development and verification of security protocols.



## **48. DarkMind: Latent Chain-of-Thought Backdoor in Customized LLMs**

cs.CR

19 pages, 15 figures, 12 tables

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2501.18617v2) [paper-pdf](https://arxiv.org/pdf/2501.18617v2)

**Authors**: Zhen Guo, Shanghao Shi, Shamim Yazdani, Ning Zhang, Reza Tourani

**Abstract**: With the rapid rise of personalized AI, customized large language models (LLMs) equipped with Chain of Thought (COT) reasoning now power millions of AI agents. However, their complex reasoning processes introduce new and largely unexplored security vulnerabilities. We present DarkMind, a novel latent reasoning level backdoor attack that targets customized LLMs by manipulating internal COT steps without altering user queries. Unlike prior prompt based attacks, DarkMind activates covertly within the reasoning chain via latent triggers, enabling adversarial behaviors without modifying input prompts or requiring access to model parameters. To achieve stealth and reliability, we propose dual trigger types instant and retrospective and integrate them within a unified embedding template that governs trigger dependent activation, employ a stealth optimization algorithm to minimize semantic drift, and introduce an automated conversation starter for covert activation across domains. Comprehensive experiments on eight reasoning datasets spanning arithmetic, commonsense, and symbolic domains, using five LLMs, demonstrate that DarkMind consistently achieves high attack success rates. We further investigate defense strategies to mitigate these risks and reveal that reasoning level backdoors represent a significant yet underexplored threat, underscoring the need for robust, reasoning aware security mechanisms.



## **49. SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage**

cs.CR

ACL Findings 2025. Welcome to employ SATA as a baseline

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2412.15289v5) [paper-pdf](https://arxiv.org/pdf/2412.15289v5)

**Authors**: Xiaoning Dong, Wenbo Hu, Wei Xu, Tianxing He

**Abstract**: Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.



## **50. Eguard: Defending LLM Embeddings Against Inversion Attacks via Text Mutual Information Optimization**

cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2411.05034v2) [paper-pdf](https://arxiv.org/pdf/2411.05034v2)

**Authors**: Tiantian Liu, Hongwei Yao, Feng Lin, Tong Wu, Zhan Qin, Kui Ren

**Abstract**: Embeddings have become a cornerstone in the functionality of large language models (LLMs) due to their ability to transform text data into rich, dense numerical representations that capture semantic and syntactic properties. These embedding vector databases serve as the long-term memory of LLMs, enabling efficient handling of a wide range of natural language processing tasks. However, the surge in popularity of embedding vector databases in LLMs has been accompanied by significant concerns about privacy leakage. Embedding vector databases are particularly vulnerable to embedding inversion attacks, where adversaries can exploit the embeddings to reverse-engineer and extract sensitive information from the original text data. Existing defense mechanisms have shown limitations, often struggling to balance security with the performance of downstream tasks. To address these challenges, we introduce Eguard, a novel defense mechanism designed to mitigate embedding inversion attacks. Eguard employs a transformer-based projection network and text mutual information optimization to safeguard embeddings while preserving the utility of LLMs. Our approach significantly reduces privacy risks, protecting over 95% of tokens from inversion while maintaining high performance across downstream tasks consistent with original embeddings.



