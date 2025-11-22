# Latest Large Language Model Attack Papers
**update at 2025-11-22 11:17:02**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Large Language Model-Based Reward Design for Deep Reinforcement Learning-Driven Autonomous Cyber Defense**

cs.LG

Accepted in the AAAI-26 Workshop on Artificial Intelligence for Cyber Security (AICS)

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16483v1) [paper-pdf](https://arxiv.org/pdf/2511.16483v1)

**Authors**: Sayak Mukherjee, Samrat Chatterjee, Emilie Purvine, Ted Fujimoto, Tegan Emerson

**Abstract**: Designing rewards for autonomous cyber attack and defense learning agents in a complex, dynamic environment is a challenging task for subject matter experts. We propose a large language model (LLM)-based reward design approach to generate autonomous cyber defense policies in a deep reinforcement learning (DRL)-driven experimental simulation environment. Multiple attack and defense agent personas were crafted, reflecting heterogeneity in agent actions, to generate LLM-guided reward designs where the LLM was first provided with contextual cyber simulation environment information. These reward structures were then utilized within a DRL-driven attack-defense simulation environment to learn an ensemble of cyber defense policies. Our results suggest that LLM-guided reward designs can lead to effective defense strategies against diverse adversarial behaviors.



## **2. Q-MLLM: Vector Quantization for Robust Multimodal Large Language Model Security**

cs.CR

Accepted by NDSS 2026

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16229v1) [paper-pdf](https://arxiv.org/pdf/2511.16229v1)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in cross-modal understanding, but remain vulnerable to adversarial attacks through visual inputs despite robust textual safety mechanisms. These vulnerabilities arise from two core weaknesses: the continuous nature of visual representations, which allows for gradient-based attacks, and the inadequate transfer of text-based safety mechanisms to visual content. We introduce Q-MLLM, a novel architecture that integrates two-level vector quantization to create a discrete bottleneck against adversarial attacks while preserving multimodal reasoning capabilities. By discretizing visual representations at both pixel-patch and semantic levels, Q-MLLM blocks attack pathways and bridges the cross-modal safety alignment gap. Our two-stage training methodology ensures robust learning while maintaining model utility. Experiments demonstrate that Q-MLLM achieves significantly better defense success rate against both jailbreak attacks and toxic image attacks than existing approaches. Notably, Q-MLLM achieves perfect defense success rate (100\%) against jailbreak attacks except in one arguable case, while maintaining competitive performance on multiple utility benchmarks with minimal inference overhead. This work establishes vector quantization as an effective defense mechanism for secure multimodal AI systems without requiring expensive safety-specific fine-tuning or detection overhead. Code is available at https://github.com/Amadeuszhao/QMLLM.



## **3. PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization**

cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16209v1) [paper-pdf](https://arxiv.org/pdf/2511.16209v1)

**Authors**: Huseein Jawad, Nicolas Brunel

**Abstract**: System prompts are critical for guiding the behavior of Large Language Models (LLMs), yet they often contain proprietary logic or sensitive information, making them a prime target for extraction attacks. Adversarial queries can successfully elicit these hidden instructions, posing significant security and privacy risks. Existing defense mechanisms frequently rely on heuristics, incur substantial computational overhead, or are inapplicable to models accessed via black-box APIs. This paper introduces a novel framework for hardening system prompts through shield appending, a lightweight approach that adds a protective textual layer to the original prompt. Our core contribution is the formalization of prompt hardening as a utility-constrained optimization problem. We leverage an LLM-as-optimizer to search the space of possible SHIELDs, seeking to minimize a leakage metric derived from a suite of adversarial attacks, while simultaneously preserving task utility above a specified threshold, measured by semantic fidelity to baseline outputs. This black-box, optimization-driven methodology is lightweight and practical, requiring only API access to the target and optimizer LLMs. We demonstrate empirically that our optimized SHIELDs significantly reduce prompt leakage against a comprehensive set of extraction attacks, outperforming established baseline defenses without compromising the model's intended functionality. Our work presents a paradigm for developing robust, utility-aware defenses in the escalating landscape of LLM security. The code is made public on the following link: https://github.com/psm-defense/psm



## **4. When Alignment Fails: Multimodal Adversarial Attacks on Vision-Language-Action Models**

cs.CV

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16203v1) [paper-pdf](https://arxiv.org/pdf/2511.16203v1)

**Authors**: Yuping Yan, Yuhan Xie, Yinxin Zhang, Lingjuan Lyu, Yaochu Jin

**Abstract**: Vision-Language-Action models (VLAs) have recently demonstrated remarkable progress in embodied environments, enabling robots to perceive, reason, and act through unified multimodal understanding. Despite their impressive capabilities, the adversarial robustness of these systems remains largely unexplored, especially under realistic multimodal and black-box conditions. Existing studies mainly focus on single-modality perturbations and overlook the cross-modal misalignment that fundamentally affects embodied reasoning and decision-making. In this paper, we introduce VLA-Fool, a comprehensive study of multimodal adversarial robustness in embodied VLA models under both white-box and black-box settings. VLA-Fool unifies three levels of multimodal adversarial attacks: (1) textual perturbations through gradient-based and prompt-based manipulations, (2) visual perturbations via patch and noise distortions, and (3) cross-modal misalignment attacks that intentionally disrupt the semantic correspondence between perception and instruction. We further incorporate a VLA-aware semantic space into linguistic prompts, developing the first automatically crafted and semantically guided prompting framework. Experiments on the LIBERO benchmark using a fine-tuned OpenVLA model reveal that even minor multimodal perturbations can cause significant behavioral deviations, demonstrating the fragility of embodied multimodal alignment.



## **5. What Your Features Reveal: Data-Efficient Black-Box Feature Inversion Attack for Split DNNs**

cs.CV

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15316v1) [paper-pdf](https://arxiv.org/pdf/2511.15316v1)

**Authors**: Zhihan Ren, Lijun He, Jiaxi Liang, Xinzhu Fu, Haixia Bi, Fan Li

**Abstract**: Split DNNs enable edge devices by offloading intensive computation to a cloud server, but this paradigm exposes privacy vulnerabilities, as the intermediate features can be exploited to reconstruct the private inputs via Feature Inversion Attack (FIA). Existing FIA methods often produce limited reconstruction quality, making it difficult to assess the true extent of privacy leakage. To reveal the privacy risk of the leaked features, we introduce FIA-Flow, a black-box FIA framework that achieves high-fidelity image reconstruction from intermediate features. To exploit the semantic information within intermediate features, we design a Latent Feature Space Alignment Module (LFSAM) to bridge the semantic gap between the intermediate feature space and the latent space. Furthermore, to rectify distributional mismatch, we develop Deterministic Inversion Flow Matching (DIFM), which projects off-manifold features onto the target manifold with one-step inference. This decoupled design simplifies learning and enables effective training with few image-feature pairs. To quantify privacy leakage from a human perspective, we also propose two metrics based on a large vision-language model. Experiments show that FIA-Flow achieves more faithful and semantically aligned feature inversion across various models (AlexNet, ResNet, Swin Transformer, DINO, and YOLO11) and layers, revealing a more severe privacy threat in Split DNNs than previously recognized.



## **6. Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models**

cs.CL

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.15304v2) [paper-pdf](https://arxiv.org/pdf/2511.15304v2)

**Authors**: Piercosma Bisconti, Matteo Prandi, Federico Pierucci, Francesco Giarrusso, Marcantonio Bracale, Marcello Galisai, Vincenzo Suriani, Olga Sorokoletova, Federico Sartore, Daniele Nardi

**Abstract**: We present evidence that adversarial poetry functions as a universal single-turn jailbreak technique for Large Language Models (LLMs). Across 25 frontier proprietary and open-weight models, curated poetic prompts yielded high attack-success rates (ASR), with some providers exceeding 90%. Mapping prompts to MLCommons and EU CoP risk taxonomies shows that poetic attacks transfer across CBRN, manipulation, cyber-offence, and loss-of-control domains. Converting 1,200 MLCommons harmful prompts into verse via a standardized meta-prompt produced ASRs up to 18 times higher than their prose baselines. Outputs are evaluated using an ensemble of 3 open-weight LLM judges, whose binary safety assessments were validated on a stratified human-labeled subset. Poetic framing achieved an average jailbreak success rate of 62% for hand-crafted poems and approximately 43% for meta-prompt conversions (compared to non-poetic baselines), substantially outperforming non-poetic baselines and revealing a systematic vulnerability across model families and safety training approaches. These findings demonstrate that stylistic variation alone can circumvent contemporary safety mechanisms, suggesting fundamental limitations in current alignment methods and evaluation protocols.



## **7. Securing AI Agents Against Prompt Injection Attacks**

cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15759v1) [paper-pdf](https://arxiv.org/pdf/2511.15759v1)

**Authors**: Badrinath Ramakrishnan, Akshaya Balaji

**Abstract**: Retrieval-augmented generation (RAG) systems have become widely used for enhancing large language model capabilities, but they introduce significant security vulnerabilities through prompt injection attacks. We present a comprehensive benchmark for evaluating prompt injection risks in RAG-enabled AI agents and propose a multi-layered defense framework. Our benchmark includes 847 adversarial test cases across five attack categories: direct injection, context manipulation, instruction override, data exfiltration, and cross-context contamination. We evaluate three defense mechanisms: content filtering with embedding-based anomaly detection, hierarchical system prompt guardrails, and multi-stage response verification, across seven state-of-the-art language models. Our combined framework reduces successful attack rates from 73.2% to 8.7% while maintaining 94.3% of baseline task performance. We release our benchmark dataset and defense implementation to support future research in AI agent security.



## **8. Taxonomy, Evaluation and Exploitation of IPI-Centric LLM Agent Defense Frameworks**

cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15203v1) [paper-pdf](https://arxiv.org/pdf/2511.15203v1)

**Authors**: Zimo Ji, Xunguang Wang, Zongjie Li, Pingchuan Ma, Yudong Gao, Daoyuan Wu, Xincheng Yan, Tian Tian, Shuai Wang

**Abstract**: Large Language Model (LLM)-based agents with function-calling capabilities are increasingly deployed, but remain vulnerable to Indirect Prompt Injection (IPI) attacks that hijack their tool calls. In response, numerous IPI-centric defense frameworks have emerged. However, these defenses are fragmented, lacking a unified taxonomy and comprehensive evaluation. In this Systematization of Knowledge (SoK), we present the first comprehensive analysis of IPI-centric defense frameworks. We introduce a comprehensive taxonomy of these defenses, classifying them along five dimensions. We then thoroughly assess the security and usability of representative defense frameworks. Through analysis of defensive failures in the assessment, we identify six root causes of defense circumvention. Based on these findings, we design three novel adaptive attacks that significantly improve attack success rates targeting specific frameworks, demonstrating the severity of the flaws in these defenses. Our paper provides a foundation and critical insights for the future development of more secure and usable IPI-centric agent defense frameworks.



## **9. As If We've Met Before: LLMs Exhibit Certainty in Recognizing Seen Files**

cs.AI

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.15192v2) [paper-pdf](https://arxiv.org/pdf/2511.15192v2)

**Authors**: Haodong Li, Jingqi Zhang, Xiao Cheng, Peihua Mai, Haoyu Wang, Yan Pang

**Abstract**: The remarkable language ability of Large Language Models (LLMs) stems from extensive training on vast datasets, often including copyrighted material, which raises serious concerns about unauthorized use. While Membership Inference Attacks (MIAs) offer potential solutions for detecting such violations, existing approaches face critical limitations and challenges due to LLMs' inherent overconfidence, limited access to ground truth training data, and reliance on empirically determined thresholds.   We present COPYCHECK, a novel framework that leverages uncertainty signals to detect whether copyrighted content was used in LLM training sets. Our method turns LLM overconfidence from a limitation into an asset by capturing uncertainty patterns that reliably distinguish between ``seen" (training data) and ``unseen" (non-training data) content. COPYCHECK further implements a two-fold strategy: (1) strategic segmentation of files into smaller snippets to reduce dependence on large-scale training data, and (2) uncertainty-guided unsupervised clustering to eliminate the need for empirically tuned thresholds. Experiment results show that COPYCHECK achieves an average balanced accuracy of 90.1% on LLaMA 7b and 91.6% on LLaMA2 7b in detecting seen files. Compared to the SOTA baseline, COPYCHECK achieves over 90% relative improvement, reaching up to 93.8\% balanced accuracy. It further exhibits strong generalizability across architectures, maintaining high performance on GPT-J 6B. This work presents the first application of uncertainty for copyright detection in LLMs, offering practical tools for training data transparency.



## **10. Can MLLMs Detect Phishing? A Comprehensive Security Benchmark Suite Focusing on Dynamic Threats and Multimodal Evaluation in Academic Environments**

cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15165v1) [paper-pdf](https://arxiv.org/pdf/2511.15165v1)

**Authors**: Jingzhuo Zhou

**Abstract**: The rapid proliferation of Multimodal Large Language Models (MLLMs) has introduced unprecedented security challenges, particularly in phishing detection within academic environments. Academic institutions and researchers are high-value targets, facing dynamic, multilingual, and context-dependent threats that leverage research backgrounds, academic collaborations, and personal information to craft highly tailored attacks. Existing security benchmarks largely rely on datasets that do not incorporate specific academic background information, making them inadequate for capturing the evolving attack patterns and human-centric vulnerability factors specific to academia. To address this gap, we present AdapT-Bench, a unified methodological framework and benchmark suite for systematically evaluating MLLM defense capabilities against dynamic phishing attacks in academic settings.



## **11. Unified Defense for Large Language Models against Jailbreak and Fine-Tuning Attacks in Education**

cs.CL

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14423v1) [paper-pdf](https://arxiv.org/pdf/2511.14423v1)

**Authors**: Xin Yi, Yue Li, Dongsheng Shi, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Large Language Models (LLMs) are increasingly integrated into educational applications. However, they remain vulnerable to jailbreak and fine-tuning attacks, which can compromise safety alignment and lead to harmful outputs. Existing studies mainly focus on general safety evaluations, with limited attention to the unique safety requirements of educational scenarios. To address this gap, we construct EduHarm, a benchmark containing safe-unsafe instruction pairs across five representative educational scenarios, enabling systematic safety evaluation of educational LLMs. Furthermore, we propose a three-stage shield framework (TSSF) for educational LLMs that simultaneously mitigates both jailbreak and fine-tuning attacks. First, safety-aware attention realignment redirects attention toward critical unsafe tokens, thereby restoring the harmfulness feature that discriminates between unsafe and safe inputs. Second, layer-wise safety judgment identifies harmfulness features by aggregating safety cues across multiple layers to detect unsafe instructions. Finally, defense-driven dual routing separates safe and unsafe queries, ensuring normal processing for benign inputs and guarded responses for harmful ones. Extensive experiments across eight jailbreak attack strategies demonstrate that TSSF effectively strengthens safety while preventing over-refusal of benign queries. Evaluations on three fine-tuning attack datasets further show that it consistently achieves robust defense against harmful queries while maintaining preserving utility gains from benign fine-tuning.



## **12. Beyond Fixed and Dynamic Prompts: Embedded Jailbreak Templates for Advancing LLM Security**

cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14140v1) [paper-pdf](https://arxiv.org/pdf/2511.14140v1)

**Authors**: Hajun Kim, Hyunsik Na, Daeseon Choi

**Abstract**: As the use of large language models (LLMs) continues to expand, ensuring their safety and robustness has become a critical challenge. In particular, jailbreak attacks that bypass built-in safety mechanisms are increasingly recognized as a tangible threat across industries, driving the need for diverse templates to support red-teaming efforts and strengthen defensive techniques. However, current approaches predominantly rely on two limited strategies: (i) substituting harmful queries into fixed templates, and (ii) having the LLM generate entire templates, which often compromises intent clarity and reproductibility. To address this gap, this paper introduces the Embedded Jailbreak Template, which preserves the structure of existing templates while naturally embedding harmful queries within their context. We further propose a progressive prompt-engineering methodology to ensure template quality and consistency, alongside standardized protocols for generation and evaluation. Together, these contributions provide a benchmark that more accurately reflects real-world usage scenarios and harmful intent, facilitating its application in red-teaming and policy regression testing.



## **13. GRPO Privacy Is at Risk: A Membership Inference Attack Against Reinforcement Learning With Verifiable Rewards**

cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14045v1) [paper-pdf](https://arxiv.org/pdf/2511.14045v1)

**Authors**: Yule Liu, Heyi Zhang, Jinyi Zheng, Zhen Sun, Zifan Peng, Tianshuo Cong, Yilong Yang, Xinlei He, Zhuo Ma

**Abstract**: Membership inference attacks (MIAs) on large language models (LLMs) pose significant privacy risks across various stages of model training. Recent advances in Reinforcement Learning with Verifiable Rewards (RLVR) have brought a profound paradigm shift in LLM training, particularly for complex reasoning tasks. However, the on-policy nature of RLVR introduces a unique privacy leakage pattern: since training relies on self-generated responses without fixed ground-truth outputs, membership inference must now determine whether a given prompt (independent of any specific response) is used during fine-tuning. This creates a threat where leakage arises not from answer memorization.   To audit this novel privacy risk, we propose Divergence-in-Behavior Attack (DIBA), the first membership inference framework specifically designed for RLVR. DIBA shifts the focus from memorization to behavioral change, leveraging measurable shifts in model behavior across two axes: advantage-side improvement (e.g., correctness gain) and logit-side divergence (e.g., policy drift). Through comprehensive evaluations, we demonstrate that DIBA significantly outperforms existing baselines, achieving around 0.8 AUC and an order-of-magnitude higher TPR@0.1%FPR. We validate DIBA's superiority across multiple settings--including in-distribution, cross-dataset, cross-algorithm, black-box scenarios, and extensions to vision-language models. Furthermore, our attack remains robust under moderate defensive measures.   To the best of our knowledge, this is the first work to systematically analyze privacy vulnerabilities in RLVR, revealing that even in the absence of explicit supervision, training data exposure can be reliably inferred through behavioral traces.



## **14. Jailbreaking Large Vision Language Models in Intelligent Transportation Systems**

cs.AI

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13892v1) [paper-pdf](https://arxiv.org/pdf/2511.13892v1)

**Authors**: Badhan Chandra Das, Md Tasnim Jawad, Md Jueal Mia, M. Hadi Amini, Yanzhao Wu

**Abstract**: Large Vision Language Models (LVLMs) demonstrate strong capabilities in multimodal reasoning and many real-world applications, such as visual question answering. However, LVLMs are highly vulnerable to jailbreaking attacks. This paper systematically analyzes the vulnerabilities of LVLMs integrated in Intelligent Transportation Systems (ITS) under carefully crafted jailbreaking attacks. First, we carefully construct a dataset with harmful queries relevant to transportation, following OpenAI's prohibited categories to which the LVLMs should not respond. Second, we introduce a novel jailbreaking attack that exploits the vulnerabilities of LVLMs through image typography manipulation and multi-turn prompting. Third, we propose a multi-layered response filtering defense technique to prevent the model from generating inappropriate responses. We perform extensive experiments with the proposed attack and defense on the state-of-the-art LVLMs (both open-source and closed-source). To evaluate the attack method and defense technique, we use GPT-4's judgment to determine the toxicity score of the generated responses, as well as manual verification. Further, we compare our proposed jailbreaking method with existing jailbreaking techniques and highlight severe security risks involved with jailbreaking attacks with image typography manipulation and multi-turn prompting in the LVLMs integrated in ITS.



## **15. ForgeDAN: An Evolutionary Framework for Jailbreaking Aligned Large Language Models**

cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13548v1) [paper-pdf](https://arxiv.org/pdf/2511.13548v1)

**Authors**: Siyang Cheng, Gaotian Liu, Rui Mei, Yilin Wang, Kejia Zhang, Kaishuo Wei, Yuqi Yu, Weiping Wen, Xiaojie Wu, Junhua Liu

**Abstract**: The rapid adoption of large language models (LLMs) has brought both transformative applications and new security risks, including jailbreak attacks that bypass alignment safeguards to elicit harmful outputs. Existing automated jailbreak generation approaches e.g. AutoDAN, suffer from limited mutation diversity, shallow fitness evaluation, and fragile keyword-based detection. To address these limitations, we propose ForgeDAN, a novel evolutionary framework for generating semantically coherent and highly effective adversarial prompts against aligned LLMs. First, ForgeDAN introduces multi-strategy textual perturbations across \textit{character, word, and sentence-level} operations to enhance attack diversity; then we employ interpretable semantic fitness evaluation based on a text similarity model to guide the evolutionary process toward semantically relevant and harmful outputs; finally, ForgeDAN integrates dual-dimensional jailbreak judgment, leveraging an LLM-based classifier to jointly assess model compliance and output harmfulness, thereby reducing false positives and improving detection effectiveness. Our evaluation demonstrates ForgeDAN achieves high jailbreaking success rates while maintaining naturalness and stealth, outperforming existing SOTA solutions.



## **16. Tight and Practical Privacy Auditing for Differentially Private In-Context Learning**

cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13502v1) [paper-pdf](https://arxiv.org/pdf/2511.13502v1)

**Authors**: Yuyang Xia, Ruixuan Liu, Li Xiong

**Abstract**: Large language models (LLMs) perform in-context learning (ICL) by adapting to tasks from prompt demonstrations, which in practice often contain private or proprietary data. Although differential privacy (DP) with private voting is a pragmatic mitigation, DP-ICL implementations are error-prone, and worst-case DP bounds may substantially overestimate actual leakage, calling for practical auditing tools. We present a tight and efficient privacy auditing framework for DP-ICL systems that runs membership inference attacks and translates their success rates into empirical privacy guarantees using Gaussian DP. Our analysis of the private voting mechanism identifies vote configurations that maximize the auditing signal, guiding the design of audit queries that reliably reveal whether a canary demonstration is present in the context. The framework supports both black-box (API-only) and white-box (internal vote) threat models, and unifies auditing for classification and generation by reducing both to a binary decision problem. Experiments on standard text classification and generation benchmarks show that our empirical leakage estimates closely match theoretical DP budgets on classification tasks and are consistently lower on generation tasks due to conservative embedding-sensitivity bounds, making our framework a practical privacy auditor and verifier for real-world DP-ICL deployments.



## **17. An LLM-based Quantitative Framework for Evaluating High-Stealthy Backdoor Risks in OSS Supply Chains**

cs.SE

7 figures, 4 tables, conference

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13341v1) [paper-pdf](https://arxiv.org/pdf/2511.13341v1)

**Authors**: Zihe Yan, Kai Luo, Haoyu Yang, Yang Yu, Zhuosheng Zhang, Guancheng Li

**Abstract**: In modern software development workflows, the open-source software supply chain contributes significantly to efficient and convenient engineering practices. With increasing system complexity, using open-source software as third-party dependencies has become a common practice. However, the lack of maintenance for underlying dependencies and insufficient community auditing create challenges in ensuring source code security and the legitimacy of repository maintainers, especially under high-stealthy backdoor attacks exemplified by the XZ-Util incident. To address these problems, we propose a fine-grained project evaluation framework for backdoor risk assessment in open-source software. The framework models stealthy backdoor attacks from the viewpoint of the attacker and defines targeted metrics for each attack stage. In addition, to overcome the limitations of static analysis in assessing the reliability of repository maintenance activities such as irregular committer privilege escalation and limited participation in reviews, the framework uses large language models (LLMs) to conduct semantic evaluation of code repositories without relying on manually crafted patterns. The framework is evaluated on sixty six high-priority packages in the Debian ecosystem. The experimental results indicate that the current open-source software supply chain is exposed to various security risks.



## **18. Shedding Light on VLN Robustness: A Black-box Framework for Indoor Lighting-based Adversarial Attack**

cs.CV

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13132v1) [paper-pdf](https://arxiv.org/pdf/2511.13132v1)

**Authors**: Chenyang Li, Wenbing Tang, Yihao Huang, Sinong Simon Zhan, Ming Hu, Xiaojun Jia, Yang Liu

**Abstract**: Vision-and-Language Navigation (VLN) agents have made remarkable progress, but their robustness remains insufficiently studied. Existing adversarial evaluations often rely on perturbations that manifest as unusual textures rarely encountered in everyday indoor environments. Errors under such contrived conditions have limited practical relevance, as real-world agents are unlikely to encounter such artificial patterns. In this work, we focus on indoor lighting, an intrinsic yet largely overlooked scene attribute that strongly influences navigation. We propose Indoor Lighting-based Adversarial Attack (ILA), a black-box framework that manipulates global illumination to disrupt VLN agents. Motivated by typical household lighting usage, we design two attack modes: Static Indoor Lighting-based Attack (SILA), where the lighting intensity remains constant throughout an episode, and Dynamic Indoor Lighting-based Attack (DILA), where lights are switched on or off at critical moments to induce abrupt illumination changes. We evaluate ILA on two state-of-the-art VLN models across three navigation tasks. Results show that ILA significantly increases failure rates while reducing trajectory efficiency, revealing previously unrecognized vulnerabilities of VLN agents to realistic indoor lighting variations.



## **19. LLM Reinforcement in Context**

cs.CL

4 pages

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12782v1) [paper-pdf](https://arxiv.org/pdf/2511.12782v1)

**Authors**: Thomas Rivasseau

**Abstract**: Current Large Language Model alignment research mostly focuses on improving model robustness against adversarial attacks and misbehavior by training on examples and prompting. Research has shown that LLM jailbreak probability increases with the size of the user input or conversation length. There is a lack of appropriate research into means of strengthening alignment which also scale with user input length. We propose interruptions as a possible solution to this problem. Interruptions are control sentences added to the user input approximately every x tokens for some arbitrary x. We suggest that this can be generalized to the Chain-of-Thought process to prevent scheming.



## **20. Whose Narrative is it Anyway? A KV Cache Manipulation Attack**

cs.CR

7 pages, 10 figures

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12752v1) [paper-pdf](https://arxiv.org/pdf/2511.12752v1)

**Authors**: Mukkesh Ganesh, Kaushik Iyer, Arun Baalaaji Sankar Ananthan

**Abstract**: The Key Value(KV) cache is an important component for efficient inference in autoregressive Large Language Models (LLMs), but its role as a representation of the model's internal state makes it a potential target for integrity attacks. This paper introduces "History Swapping," a novel block-level attack that manipulates the KV cache to steer model generation without altering the user-facing prompt. The attack involves overwriting a contiguous segment of the active generation's cache with a precomputed cache from a different topic. We empirically evaluate this method across 324 configurations on the Qwen 3 family of models, analyzing the impact of timing, magnitude, and layer depth of the cache overwrite. Our findings reveal that only full-layer overwrites can successfully hijack the conversation's topic, leading to three distinct behaviors: immediate and persistent topic shift, partial recovery, or a delayed hijack. Furthermore, we observe that high-level structural plans are encoded early in the generation process and local discourse structure is maintained by the final layers of the model. This work demonstrates that the KV cache is a significant vector for security analysis, as it encodes not just context but also topic trajectory and structural planning, making it a powerful interface for manipulating model behavior.



## **21. Evolve the Method, Not the Prompts: Evolutionary Synthesis of Jailbreak Attacks on LLMs**

cs.CL

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12710v1) [paper-pdf](https://arxiv.org/pdf/2511.12710v1)

**Authors**: Yunhao Chen, Xin Wang, Juncheng Li, Yixu Wang, Jie Li, Yan Teng, Yingchun Wang, Xingjun Ma

**Abstract**: Automated red teaming frameworks for Large Language Models (LLMs) have become increasingly sophisticated, yet they share a fundamental limitation: their jailbreak logic is confined to selecting, combining, or refining pre-existing attack strategies. This binds their creativity and leaves them unable to autonomously invent entirely new attack mechanisms. To overcome this gap, we introduce \textbf{EvoSynth}, an autonomous framework that shifts the paradigm from attack planning to the evolutionary synthesis of jailbreak methods. Instead of refining prompts, EvoSynth employs a multi-agent system to autonomously engineer, evolve, and execute novel, code-based attack algorithms. Crucially, it features a code-level self-correction loop, allowing it to iteratively rewrite its own attack logic in response to failure. Through extensive experiments, we demonstrate that EvoSynth not only establishes a new state-of-the-art by achieving an 85.5\% Attack Success Rate (ASR) against highly robust models like Claude-Sonnet-4.5, but also generates attacks that are significantly more diverse than those from existing methods. We release our framework to facilitate future research in this new direction of evolutionary synthesis of jailbreak methods. Code is available at: https://github.com/dongdongunique/EvoSynth.



## **22. Uncovering and Aligning Anomalous Attention Heads to Defend Against NLP Backdoor Attacks**

cs.CR

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.13789v1) [paper-pdf](https://arxiv.org/pdf/2511.13789v1)

**Authors**: Haotian Jin, Yang Li, Haihui Fan, Lin Shen, Xiangfang Li, Bo Li

**Abstract**: Backdoor attacks pose a serious threat to the security of large language models (LLMs), causing them to exhibit anomalous behavior under specific trigger conditions. The design of backdoor triggers has evolved from fixed triggers to dynamic or implicit triggers. This increased flexibility in trigger design makes it challenging for defenders to identify their specific forms accurately. Most existing backdoor defense methods are limited to specific types of triggers or rely on an additional clean model for support. To address this issue, we propose a backdoor detection method based on attention similarity, enabling backdoor detection without prior knowledge of the trigger. Our study reveals that models subjected to backdoor attacks exhibit unusually high similarity among attention heads when exposed to triggers. Based on this observation, we propose an attention safety alignment approach combined with head-wise fine-tuning to rectify potentially contaminated attention heads, thereby effectively mitigating the impact of backdoor attacks. Extensive experimental results demonstrate that our method significantly reduces the success rate of backdoor attacks while preserving the model's performance on downstream tasks.



## **23. Scaling Patterns in Adversarial Alignment: Evidence from Multi-LLM Jailbreak Experiments**

cs.LG

19 pages, 6 figures, 3 tables

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.13788v1) [paper-pdf](https://arxiv.org/pdf/2511.13788v1)

**Authors**: Samuel Nathanson, Rebecca Williams, Cynthia Matuszek

**Abstract**: Large language models (LLMs) increasingly operate in multi-agent and safety-critical settings, raising open questions about how their vulnerabilities scale when models interact adversarially. This study examines whether larger models can systematically jailbreak smaller ones - eliciting harmful or restricted behavior despite alignment safeguards. Using standardized adversarial tasks from JailbreakBench, we simulate over 6,000 multi-turn attacker-target exchanges across major LLM families and scales (0.6B-120B parameters), measuring both harm score and refusal behavior as indicators of adversarial potency and alignment integrity. Each interaction is evaluated through aggregated harm and refusal scores assigned by three independent LLM judges, providing a consistent, model-based measure of adversarial outcomes. Aggregating results across prompts, we find a strong and statistically significant correlation between mean harm and the logarithm of the attacker-to-target size ratio (Pearson r = 0.51, p < 0.001; Spearman rho = 0.52, p < 0.001), indicating that relative model size correlates with the likelihood and severity of harmful completions. Mean harm score variance is higher across attackers (0.18) than across targets (0.10), suggesting that attacker-side behavioral diversity contributes more to adversarial outcomes than target susceptibility. Attacker refusal frequency is strongly and negatively correlated with harm (rho = -0.93, p < 0.001), showing that attacker-side alignment mitigates harmful responses. These findings reveal that size asymmetry influences robustness and provide exploratory evidence for adversarial scaling patterns, motivating more controlled investigations into inter-model alignment and safety.



## **24. Beyond Pixels: Semantic-aware Typographic Attack for Geo-Privacy Protection**

cs.CV

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12575v1) [paper-pdf](https://arxiv.org/pdf/2511.12575v1)

**Authors**: Jiayi Zhu, Yihao Huang, Yue Cao, Xiaojun Jia, Qing Guo, Felix Juefei-Xu, Geguang Pu, Bin Wang

**Abstract**: Large Visual Language Models (LVLMs) now pose a serious yet overlooked privacy threat, as they can infer a social media user's geolocation directly from shared images, leading to unintended privacy leakage. While adversarial image perturbations provide a potential direction for geo-privacy protection, they require relatively strong distortions to be effective against LVLMs, which noticeably degrade visual quality and diminish an image's value for sharing. To overcome this limitation, we identify typographical attacks as a promising direction for protecting geo-privacy by adding text extension outside the visual content. We further investigate which textual semantics are effective in disrupting geolocation inference and design a two-stage, semantics-aware typographical attack that generates deceptive text to protect user privacy. Extensive experiments across three datasets demonstrate that our approach significantly reduces geolocation prediction accuracy of five state-of-the-art commercial LVLMs, establishing a practical and visually-preserving protection strategy against emerging geo-privacy threats.



## **25. SGuard-v1: Safety Guardrail for Large Language Models**

cs.CL

Technical Report

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12497v1) [paper-pdf](https://arxiv.org/pdf/2511.12497v1)

**Authors**: JoonHo Lee, HyeonMin Cho, Jaewoong Yun, Hyunjae Lee, JunKyu Lee, Juree Seok

**Abstract**: We present SGuard-v1, a lightweight safety guardrail for Large Language Models (LLMs), which comprises two specialized models to detect harmful content and screen adversarial prompts in human-AI conversational settings. The first component, ContentFilter, is trained to identify safety risks in LLM prompts and responses in accordance with the MLCommons hazard taxonomy, a comprehensive framework for trust and safety assessment of AI. The second component, JailbreakFilter, is trained with a carefully designed curriculum over integrated datasets and findings from prior work on adversarial prompting, covering 60 major attack types while mitigating false-unsafe classification. SGuard-v1 is built on the 2B-parameter Granite-3.3-2B-Instruct model that supports 12 languages. We curate approximately 1.4 million training instances from both collected and synthesized data and perform instruction tuning on the base model, distributing the curated data across the two component according to their designated functions. Through extensive evaluation on public and proprietary safety benchmarks, SGuard-v1 achieves state-of-the-art safety performance while remaining lightweight, thereby reducing deployment overhead. SGuard-v1 also improves interpretability for downstream use by providing multi-class safety predictions and their binary confidence scores. We release the SGuard-v1 under the Apache-2.0 License to enable further research and practical deployment in AI safety.



## **26. GRAPHTEXTACK: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs**

cs.CR

AAAI 2026

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12423v1) [paper-pdf](https://arxiv.org/pdf/2511.12423v1)

**Authors**: Jiaji Ma, Puja Trivedi, Danai Koutra

**Abstract**: Text-attributed graphs (TAGs), which combine structural and textual node information, are ubiquitous across many domains. Recent work integrates Large Language Models (LLMs) with Graph Neural Networks (GNNs) to jointly model semantics and structure, resulting in more general and expressive models that achieve state-of-the-art performance on TAG benchmarks. However, this integration introduces dual vulnerabilities: GNNs are sensitive to structural perturbations, while LLM-derived features are vulnerable to prompt injection and adversarial phrasing. While existing adversarial attacks largely perturb structure or text independently, we find that uni-modal attacks cause only modest degradation in LLM-enhanced GNNs. Moreover, many existing attacks assume unrealistic capabilities, such as white-box access or direct modification of graph data. To address these gaps, we propose GRAPHTEXTACK, the first black-box, multi-modal{, poisoning} node injection attack for LLM-enhanced GNNs. GRAPHTEXTACK injects nodes with carefully crafted structure and semantics to degrade model performance, operating under a realistic threat model without relying on model internals or surrogate models. To navigate the combinatorial, non-differentiable search space of connectivity and feature assignments, GRAPHTEXTACK introduces a novel evolutionary optimization framework with a multi-objective fitness function that balances local prediction disruption and global graph influence. Extensive experiments on five datasets and two state-of-the-art LLM-enhanced GNN models show that GRAPHTEXTACK significantly outperforms 12 strong baselines.



## **27. Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment**

cs.CR

AAAI-26-AIA

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.06852v3) [paper-pdf](https://arxiv.org/pdf/2511.06852v3)

**Authors**: Peng Zhang, Peijie Sun

**Abstract**: Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment.



## **28. DRIP: Defending Prompt Injection via Token-wise Representation Editing and Residual Instruction Fusion**

cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.00447v2) [paper-pdf](https://arxiv.org/pdf/2511.00447v2)

**Authors**: Ruofan Liu, Yun Lin, Zhiyong Huang, Jin Song Dong

**Abstract**: Large language models (LLMs) are increasingly integrated into IT infrastructures, where they process user data according to predefined instructions. However, conventional LLMs remain vulnerable to prompt injection, where malicious users inject directive tokens into the data to subvert model behavior. Existing defenses train LLMs to semantically separate data and instruction tokens, but still struggle to (1) balance utility and security and (2) prevent instruction-like semantics in the data from overriding the intended instructions.   We propose DRIP, which (1) precisely removes instruction semantics from tokens in the data section while preserving their data semantics, and (2) robustly preserves the effect of the intended instruction even under strong adversarial content. To "de-instructionalize" data tokens, DRIP introduces a data curation and training paradigm with a lightweight representation-editing module that edits embeddings of instruction-like tokens in the data section, enhancing security without harming utility. To ensure non-overwritability of instructions, DRIP adds a minimal residual module that reduces the ability of adversarial data to overwrite the original instruction. We evaluate DRIP on LLaMA 8B and Mistral 7B against StruQ, SecAlign, ISE, and PFT on three prompt-injection benchmarks (SEP, AlpacaFarm, and InjecAgent). DRIP improves role-separation score by 12-49\%, reduces attack success rate by over 66\% under adaptive attacks, and matches the utility of the undefended model, establishing a new state of the art for prompt-injection robustness.



## **29. SoK: Honeypots & LLMs, More Than the Sum of Their Parts?**

cs.CR

Systemization of Knowledge

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2510.25939v3) [paper-pdf](https://arxiv.org/pdf/2510.25939v3)

**Authors**: Robert A. Bridges, Thomas R. Mitchell, Mauricio Muñoz, Ted Henriksson

**Abstract**: The advent of Large Language Models (LLMs) promised to resolve the long-standing paradox in honeypot design, achieving high-fidelity deception with low operational risk. Through a flurry of research since late 2022, steady progress from ideation to prototype implementation is exhibited. Since late 2022, a flurry of research has demonstrated steady progress from ideation to prototype implementation. While promising, evaluations show only incremental progress in real-world deployments, and the field still lacks a cohesive understanding of the emerging architectural patterns, core challenges, and evaluation paradigms. To fill this gap, this Systematization of Knowledge (SoK) paper provides the first comprehensive overview and analysis of this new domain. We survey and systematize the field by focusing on three critical, intersecting research areas: first, we provide a taxonomy of honeypot detection vectors, structuring the core problems that LLM-based realism must solve; second, we synthesize the emerging literature on LLM-powered honeypots, identifying a canonical architecture and key evaluation trends; and third, we chart the evolutionary path of honeypot log analysis, from simple data reduction to automated intelligence generation. We synthesize these findings into a forward-looking research roadmap, arguing that the true potential of this technology lies in creating autonomous, self-improving deception systems to counter the emerging threat of intelligent, automated attackers.



## **30. Practical and Stealthy Touch-Guided Jailbreak Attacks on Deployed Mobile Vision-Language Agents**

cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2510.07809v2) [paper-pdf](https://arxiv.org/pdf/2510.07809v2)

**Authors**: Renhua Ding, Xiao Yang, Zhengwei Fang, Jun Luo, Kun He, Jun Zhu

**Abstract**: Large vision-language models (LVLMs) enable autonomous mobile agents to operate smartphone user interfaces, yet vulnerabilities in their perception and interaction remain critically understudied. Existing research often relies on conspicuous overlays, elevated permissions, or unrealistic threat assumptions, limiting stealth and real-world feasibility. In this paper, we introduce a practical and stealthy jailbreak attack framework, which comprises three key components: (i) non-privileged perception compromise, which injects visual payloads into the application interface without requiring elevated system permissions; (ii) agent-attributable activation, which leverages input attribution signals to distinguish agent from human interactions and limits prompt exposure to transient intervals to preserve stealth from end users; and (iii) efficient one-shot jailbreak, a heuristic iterative deepening search algorithm (HG-IDA*) that performs keyword-level detoxification to bypass built-in safety alignment of LVLMs. Moreover, we developed three representative Android applications and curated a prompt-injection dataset for mobile agents. We evaluated our attack across multiple LVLM backends, including closed-source services and representative open-source models, and observed high planning and execution hijack rates (e.g., GPT-4o: 82.5% planning / 75.0% execution), exposing a fundamental security vulnerability in current mobile agents and underscoring critical implications for autonomous smartphone operation.



## **31. Jailbreaking LLMs via Semantically Relevant Nested Scenarios with Targeted Toxic Knowledge**

cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2510.01223v2) [paper-pdf](https://arxiv.org/pdf/2510.01223v2)

**Authors**: Ning Xu, Bo Gao, Hui Dou

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks. However, they remain exposed to jailbreak attacks, eliciting harmful responses. The nested scenario strategy has been increasingly adopted across various methods, demonstrating immense potential. Nevertheless, these methods are easily detectable due to their prominent malicious intentions. In this work, we are the first to find and systematically verify that LLMs' alignment defenses are not sensitive to nested scenarios, where these scenarios are highly semantically relevant to the queries and incorporate targeted toxic knowledge. This is a crucial yet insufficiently explored direction. Based on this, we propose RTS-Attack (Semantically Relevant Nested Scenarios with Targeted Toxic Knowledge), an adaptive and automated framework to examine LLMs' alignment. By building scenarios highly relevant to the queries and integrating targeted toxic knowledge, RTS-Attack bypasses the alignment defenses of LLMs. Moreover, the jailbreak prompts generated by RTS-Attack are free from harmful queries, leading to outstanding concealment. Extensive experiments demonstrate that RTS-Attack exhibits superior performance in both efficiency and universality compared to the baselines across diverse advanced LLMs, including GPT-4o, Llama3-70b, and Gemini-pro. Our complete code is available at https://github.com/nercode/Work. WARNING: THIS PAPER CONTAINS POTENTIALLY HARMFUL CONTENT.



## **32. NeuroStrike: Neuron-Level Attacks on Aligned LLMs**

cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2509.11864v2) [paper-pdf](https://arxiv.org/pdf/2509.11864v2)

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Maximilian Thang, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability.   This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.



## **33. Guided Reasoning in LLM-Driven Penetration Testing Using Structured Attack Trees**

cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2509.07939v2) [paper-pdf](https://arxiv.org/pdf/2509.07939v2)

**Authors**: Katsuaki Nakano, Reza Fayyazi, Shanchieh Jay Yang, Michael Zuzak

**Abstract**: Recent advances in Large Language Models (LLMs) have driven interest in automating cybersecurity penetration testing workflows, offering the promise of faster and more consistent vulnerability assessment for enterprise systems. Existing LLM agents for penetration testing primarily rely on self-guided reasoning, which can produce inaccurate or hallucinated procedural steps. As a result, the LLM agent may undertake unproductive actions, such as exploiting unused software libraries or generating cyclical responses that repeat prior tactics. In this work, we propose a guided reasoning pipeline for penetration testing LLM agents that incorporates a deterministic task tree built from the MITRE ATT&CK Matrix, a proven penetration testing kll chain, to constrain the LLM's reaoning process to explicitly defined tactics, techniques, and procedures. This anchors reasoning in proven penetration testing methodologies and filters out ineffective actions by guiding the agent towards more productive attack procedures. To evaluate our approach, we built an automated penetration testing LLM agent using three LLMs (Llama-3-8B, Gemini-1.5, and GPT-4) and applied it to navigate 10 HackTheBox cybersecurity exercises with 103 discrete subtasks representing real-world cyberattack scenarios. Our proposed reasoning pipeline guided the LLM agent through 71.8\%, 72.8\%, and 78.6\% of subtasks using Llama-3-8B, Gemini-1.5, and GPT-4, respectively. Comparatively, the state-of-the-art LLM penetration testing tool using self-guided reasoning completed only 13.5\%, 16.5\%, and 75.7\% of subtasks and required 86.2\%, 118.7\%, and 205.9\% more model queries. This suggests that incorporating a deterministic task tree into LLM reasoning pipelines can enhance the accuracy and efficiency of automated cybersecurity assessments



## **34. PromptCOS: Towards Content-only System Prompt Copyright Auditing for LLMs**

cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2509.03117v2) [paper-pdf](https://arxiv.org/pdf/2509.03117v2)

**Authors**: Yuchen Yang, Yiming Li, Hongwei Yao, Enhao Huang, Shuo Shao, Yuyi Wang, Zhibo Wang, Dacheng Tao, Zhan Qin

**Abstract**: System prompts are critical for shaping the behavior and output quality of large language model (LLM)-based applications, driving substantial investment in optimizing high-quality prompts beyond traditional handcrafted designs. However, as system prompts become valuable intellectual property, they are increasingly vulnerable to prompt theft and unauthorized use, highlighting the urgent need for effective copyright auditing, especially watermarking. Existing methods rely on verifying subtle logit distribution shifts triggered by a query. We observe that this logit-dependent verification framework is impractical in real-world content-only settings, primarily because (1) random sampling makes content-level generation unstable for verification, and (2) stronger instructions needed for content-level signals compromise prompt fidelity.   To overcome these challenges, we propose PromptCOS, the first content-only system prompt copyright auditing method based on content-level output similarity. PromptCOS achieves watermark stability by designing a cyclic output signal as the conditional instruction's target. It preserves prompt fidelity by injecting a small set of auxiliary tokens to encode the watermark, leaving the main prompt untouched. Furthermore, to ensure robustness against malicious removal, we optimize cover tokens, i.e., critical tokens in the original prompt, to ensure that removing auxiliary tokens causes severe performance degradation. Experimental results show that PromptCOS achieves high effectiveness (99.3% average watermark similarity), strong distinctiveness (60.8% higher than the best baseline), high fidelity (accuracy degradation no greater than 0.6%), robustness (resilience against four potential attack categories), and high computational efficiency (up to 98.1% cost saving). Our code is available at GitHub (https://github.com/LianPing-cyber/PromptCOS).



## **35. SoK: Exposing the Generation and Detection Gaps in LLM-Generated Phishing Through Examination of Generation Methods, Content Characteristics, and Countermeasures**

cs.CR

18 pages, 5 tables, 4 figures

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2508.21457v2) [paper-pdf](https://arxiv.org/pdf/2508.21457v2)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Carsten Rudolph

**Abstract**: Phishing campaigns involve adversaries masquerading as trusted vendors trying to trigger user behavior that enables them to exfiltrate private data. While URLs are an important part of phishing campaigns, communicative elements like text and images are central in triggering the required user behavior. Further, due to advances in phishing detection, attackers react by scaling campaigns to larger numbers and diversifying and personalizing content. In addition to established mechanisms, such as template-based generation, large language models (LLMs) can be used for phishing content generation, enabling attacks to scale in minutes, challenging existing phishing detection paradigms through personalized content, stealthy explicit phishing keywords, and dynamic adaptation to diverse attack scenarios. Countering these dynamically changing attack campaigns requires a comprehensive understanding of the complex LLM-related threat landscape. Existing studies are fragmented and focus on specific areas. In this work, we provide the first holistic examination of LLM-generated phishing content. First, to trace the exploitation pathways of LLMs for phishing content generation, we adopt a modular taxonomy documenting nine stages by which adversaries breach LLM safety guardrails. We then characterize how LLM-generated phishing manifests as threats, revealing that it evades detectors while emphasizing human cognitive manipulation. Third, by taxonomizing defense techniques aligned with generation methods, we expose a critical asymmetry that offensive mechanisms adapt dynamically to attack scenarios, whereas defensive strategies remain static and reactive. Finally, based on a thorough analysis of the existing literature, we highlight insights and gaps and suggest a roadmap for understanding and countering LLM-driven phishing at scale.



## **36. Failures to Surface Harmful Contents in Video Large Language Models**

cs.MM

12 pages, 8 figures. Accepted to AAAI 2026

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2508.10974v2) [paper-pdf](https://arxiv.org/pdf/2508.10974v2)

**Authors**: Yuxin Cao, Wei Song, Derui Wang, Jingling Xue, Jin Song Dong

**Abstract**: Video Large Language Models (VideoLLMs) are increasingly deployed on numerous critical applications, where users rely on auto-generated summaries while casually skimming the video stream. We show that this interaction hides a critical safety gap: if harmful content is embedded in a video, either as full-frame inserts or as small corner patches, state-of-the-art VideoLLMs rarely mention the harmful content in the output, despite its clear visibility to human viewers. A root-cause analysis reveals three compounding design flaws: (1) insufficient temporal coverage resulting from the sparse, uniformly spaced frame sampling used by most leading VideoLLMs, (2) spatial information loss introduced by aggressive token downsampling within sampled frames, and (3) encoder-decoder disconnection, whereby visual cues are only weakly utilized during text generation. Leveraging these insights, we craft three zero-query black-box attacks, aligning with these flaws in the processing pipeline. Our large-scale evaluation across five leading VideoLLMs shows that the harmfulness omission rate exceeds 90% in most cases. Even when harmful content is clearly present in all frames, these models consistently fail to identify it. These results underscore a fundamental vulnerability in current VideoLLMs' designs and highlight the urgent need for sampling strategies, token compression, and decoding mechanisms that guarantee semantic coverage rather than speed alone.



## **37. Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models**

cs.CR

16 pages; Previously this version appeared as arXiv:2510.15430 which was submitted as a new work by accident

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2508.09201v3) [paper-pdf](https://arxiv.org/pdf/2508.09201v3)

**Authors**: Shuang Liang, Zhihao Xu, Jialing Tao, Hui Xue, Xiting Wang

**Abstract**: Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. To address this, existing detection methods either learn attack-specific parameters, which hinders generalization to unseen attacks, or rely on heuristically sound principles, which limit accuracy and efficiency. To overcome these limitations, we propose Learning to Detect (LoD), a general framework that accurately detects unknown jailbreak attacks by shifting the focus from attack-specific learning to task-specific learning. This framework includes a Multi-modal Safety Concept Activation Vector module for safety-oriented representation learning and a Safety Pattern Auto-Encoder module for unsupervised attack classification. Extensive experiments show that our method achieves consistently higher detection AUROC on diverse unknown attacks while improving efficiency. The code is available at https://anonymous.4open.science/r/Learning-to-Detect-51CB.



## **38. Hidden in the Noise: Unveiling Backdoors in Audio LLMs Alignment through Latent Acoustic Pattern Triggers**

cs.SD

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2508.02175v3) [paper-pdf](https://arxiv.org/pdf/2508.02175v3)

**Authors**: Liang Lin, Miao Yu, Kaiwen Luo, Yibo Zhang, Lilan Peng, Dexian Wang, Xuehai Tang, Yuanhe Zhang, Xikang Yang, Zhenhong Zhou, Kun Wang, Yang Liu

**Abstract**: As Audio Large Language Models (ALLMs) emerge as powerful tools for speech processing, their safety implications demand urgent attention. While considerable research has explored textual and vision safety, audio's distinct characteristics present significant challenges. This paper first investigates: Is ALLM vulnerable to backdoor attacks exploiting acoustic triggers? In response to this issue, we introduce Hidden in the Noise (HIN), a novel backdoor attack framework designed to exploit subtle, audio-specific features. HIN applies acoustic modifications to raw audio waveforms, such as alterations to temporal dynamics and strategic injection of spectrally tailored noise. These changes introduce consistent patterns that an ALLM's acoustic feature encoder captures, embedding robust triggers within the audio stream. To evaluate ALLM robustness against audio-feature-based triggers, we develop the AudioSafe benchmark, assessing nine distinct risk types. Extensive experiments on AudioSafe and three established safety datasets reveal critical vulnerabilities in existing ALLMs: (I) audio features like environment noise and speech rate variations achieve over 90% average attack success rate. (II) ALLMs exhibit significant sensitivity differences across acoustic features, particularly showing minimal response to volume as a trigger, and (III) poisoned sample inclusion causes only marginal loss curve fluctuations, highlighting the attack's stealth.



## **39. AgentArmor: Enforcing Program Analysis on Agent Runtime Trace to Defend Against Prompt Injection**

cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2508.01249v3) [paper-pdf](https://arxiv.org/pdf/2508.01249v3)

**Authors**: Peiran Wang, Yang Liu, Yunfei Lu, Yifeng Cai, Hongbo Chen, Qingyou Yang, Jie Zhang, Jue Hong, Ye Wu

**Abstract**: Large Language Model (LLM) agents offer a powerful new paradigm for solving various problems by combining natural language reasoning with the execution of external tools. However, their dynamic and non-transparent behavior introduces critical security risks, particularly in the presence of prompt injection attacks. In this work, we propose a novel insight that treats the agent runtime traces as structured programs with analyzable semantics. Thus, we present AgentArmor, a program analysis framework that converts agent traces into graph intermediate representation-based structured program dependency representations (e.g., CFG, DFG, and PDG) and enforces security policies via a type system. AgentArmor consists of three key components: (1) a graph constructor that reconstructs the agent's runtime traces as graph-based intermediate representations with control and data flow described within; (2) a property registry that attaches security-relevant metadata of interacted tools \& data, and (3) a type system that performs static inference and checking over the intermediate representation. By representing agent behavior as structured programs, AgentArmor enables program analysis for sensitive data flow, trust boundaries, and policy violations. We evaluate AgentArmor on the AgentDojo benchmark, the results show that AgentArmor can reduce the ASR to 3\%, with the utility drop only 1\%.



## **40. Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs**

cs.CL

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2507.22564v2) [paper-pdf](https://arxiv.org/pdf/2507.22564v2)

**Authors**: Xikang Yang, Biyu Zhou, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems.



## **41. LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge**

cs.CR

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2506.09443v2) [paper-pdf](https://arxiv.org/pdf/2506.09443v2)

**Authors**: Songze Li, Chuokun Xu, Jiaying Wang, Xueluan Gong, Chen Chen, Jirui Zhang, Jun Wang, Kwok-Yan Lam, Shouling Ji

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse tasks, driving the development and widespread adoption of LLM-as-a-Judge systems for automated evaluation, including red teaming and benchmarking. However, these systems are susceptible to adversarial attacks that can manipulate evaluation outcomes, raising critical concerns about their robustness and trustworthiness. Existing evaluation methods for LLM-based judges are often fragmented and lack a unified framework for comprehensive robustness assessment. Furthermore, the impact of prompt template design and model selection on judge robustness has rarely been explored, and their performance in real-world deployments remains largely unverified. To address these gaps, we introduce RobustJudge, a fully automated and scalable framework designed to systematically evaluate the robustness of LLM-as-a-Judge systems. Specifically, RobustJudge investigates the effectiveness of 15 attack methods and 7 defense strategies across 12 models (RQ1), examines the impact of prompt template design and model selection (RQ2), and evaluates the security of real-world deployments (RQ3). Our study yields three key findings: (1) LLM-as-a-Judge systems are highly vulnerable to attacks such as PAIR and combined attacks, while defense mechanisms such as re-tokenization and LLM-based detectors can provide enhanced protection; (2) robustness varies substantially across prompt templates (up to 40%); (3) deploying RobustJudge on Alibaba's PAI platform uncovers previously undiscovered vulnerabilities. These results offer practical insights for building trustworthy LLM-as-a-Judge systems.



## **42. MCA-Bench: A Multimodal Benchmark for Evaluating CAPTCHA Robustness Against VLM-based Attacks**

cs.CV

we update the paper supplement

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2506.05982v6) [paper-pdf](https://arxiv.org/pdf/2506.05982v6)

**Authors**: Zonglin Wu, Yule Xue, Yaoyao Feng, Xiaolong Wang, Yiren Song

**Abstract**: As automated attack techniques rapidly advance, CAPTCHAs remain a critical defense mechanism against malicious bots. However, existing CAPTCHA schemes encompass a diverse range of modalities -- from static distorted text and obfuscated images to interactive clicks, sliding puzzles, and logic-based questions -- yet the community still lacks a unified, large-scale, multimodal benchmark to rigorously evaluate their security robustness. To address this gap, we introduce MCA-Bench, a comprehensive and reproducible benchmarking suite that integrates heterogeneous CAPTCHA types into a single evaluation protocol. Leveraging a shared vision-language model backbone, we fine-tune specialized cracking agents for each CAPTCHA category, enabling consistent, cross-modal assessments. Extensive experiments reveal that MCA-Bench effectively maps the vulnerability spectrum of modern CAPTCHA designs under varied attack settings, and crucially offers the first quantitative analysis of how challenge complexity, interaction depth, and model solvability interrelate. Based on these findings, we propose three actionable design principles and identify key open challenges, laying the groundwork for systematic CAPTCHA hardening, fair benchmarking, and broader community collaboration. Datasets and code are available online.



## **43. Use as Many Surrogates as You Want: Selective Ensemble Attack to Unleash Transferability without Sacrificing Resource Efficiency**

cs.CV

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2505.12644v2) [paper-pdf](https://arxiv.org/pdf/2505.12644v2)

**Authors**: Bo Yang, Hengwei Zhang, Jindong Wang, Yuchen Ren, Chenhao Lin, Chao Shen, Zhengyu Zhao

**Abstract**: In surrogate ensemble attacks, using more surrogate models yields higher transferability but lower resource efficiency. This practical trade-off between transferability and efficiency has largely limited existing attacks despite many pre-trained models are easily accessible online. In this paper, we argue that such a trade-off is caused by an unnecessary common assumption, i.e., all models should be \textit{identical} across iterations. By lifting this assumption, we can use as many surrogates as we want to unleash transferability without sacrificing efficiency. Concretely, we propose Selective Ensemble Attack (SEA), which dynamically selects diverse models (from easily accessible pre-trained models) across iterations based on our new interpretation of decoupling within-iteration and cross-iteration model diversity. In this way, the number of within-iteration models is fixed for maintaining efficiency, while only cross-iteration model diversity is increased for higher transferability. Experiments on ImageNet demonstrate the superiority of SEA in various scenarios. For example, when dynamically selecting 4 from 20 accessible models, SEA yields 8.5% higher transferability than existing attacks under the same efficiency. The superiority of SEA also generalizes to real-world systems, such as commercial vision APIs and large vision-language models. Overall, SEA opens up the possibility of adaptively balancing transferability and efficiency according to specific resource requirements.



## **44. Revisiting Model Inversion Evaluation: From Misleading Standards to Reliable Privacy Assessment**

cs.LG

To support future work, we release our MLLM-based MI evaluation framework and benchmarking suite at https://github.com/hosytuyen/MI-Eval-MLLM

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2505.03519v4) [paper-pdf](https://arxiv.org/pdf/2505.03519v4)

**Authors**: Sy-Tuyen Ho, Koh Jun Hao, Ngoc-Bao Nguyen, Alexander Binder, Ngai-Man Cheung

**Abstract**: Model Inversion (MI) attacks aim to reconstruct information from private training data by exploiting access to machine learning models T. To evaluate such attacks, the standard evaluation framework relies on an evaluation model E, trained under the same task design as T. This framework has become the de facto standard for assessing progress in MI research, used across nearly all recent MI studies without question. In this paper, we present the first in-depth study of this evaluation framework. In particular, we identify a critical issue of this standard framework: Type-I adversarial examples. These are reconstructions that do not capture the visual features of private training data, yet are still deemed successful by T and ultimately transferable to E. Such false positives undermine the reliability of the standard MI evaluation framework. To address this issue, we introduce a new MI evaluation framework that replaces the evaluation model E with advanced Multimodal Large Language Models (MLLMs). By leveraging their general-purpose visual understanding, our MLLM-based framework does not depend on training of shared task design as in T, thus reducing Type-I transferability and providing more faithful assessments of reconstruction success. Using our MLLM-based evaluation framework, we reevaluate 27 diverse MI attack setups and empirically reveal consistently high false positive rates under the standard evaluation framework. Importantly, we demonstrate that many state-of-the-art (SOTA) MI methods report inflated attack accuracy, indicating that actual privacy leakage is significantly lower than previously believed. By uncovering this critical issue and proposing a robust solution, our work enables a reassessment of progress in MI research and sets a new standard for reliable and robust evaluation. Code can be found in https://github.com/hosytuyen/MI-Eval-MLLM



## **45. One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image**

cs.CL

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2504.02132v3) [paper-pdf](https://arxiv.org/pdf/2504.02132v3)

**Authors**: Ezzeldin Shereen, Dan Ristea, Shae McFadden, Burak Hasircioglu, Vasilios Mavroudis, Chris Hicks

**Abstract**: Retrieval-augmented generation (RAG) is instrumental for inhibiting hallucinations in large language models (LLMs) through the use of a factual knowledge base (KB). Although PDF documents are prominent sources of knowledge, text-based RAG pipelines are ineffective at capturing their rich multi-modal information. In contrast, visual document RAG (VD-RAG) uses screenshots of document pages as the KB, which has been shown to achieve state-of-the-art results. However, by introducing the image modality, VD-RAG introduces new attack vectors for adversaries to disrupt the system by injecting malicious documents into the KB. In this paper, we demonstrate the vulnerability of VD-RAG to poisoning attacks targeting both retrieval and generation. We define two attack objectives and demonstrate that both can be realized by injecting only a single adversarial image into the KB. Firstly, we introduce a targeted attack against one or a group of queries with the goal of spreading targeted disinformation. Secondly, we present a universal attack that, for any potential user query, influences the response to cause a denial-of-service in the VD-RAG system. We investigate the two attack objectives under both white-box and black-box assumptions, employing a multi-objective gradient-based optimization approach as well as prompting state-of-the-art generative models. Using two visual document datasets, a diverse set of state-of-the-art retrievers (embedding models) and generators (vision language models), we show VD-RAG is vulnerable to poisoning attacks in both the targeted and universal settings, yet demonstrating robustness to black-box attacks in the universal setting.



## **46. LightDefense: A Lightweight Uncertainty-Driven Defense against Jailbreaks via Shifted Token Distribution**

cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2504.01533v2) [paper-pdf](https://arxiv.org/pdf/2504.01533v2)

**Authors**: Zhuoran Yang, Yanyong Zhang

**Abstract**: Large Language Models (LLMs) face threats from jailbreak prompts. Existing methods for defending against jailbreak attacks are primarily based on auxiliary models. These strategies, however, often require extensive data collection or training. We propose LightDefense, a lightweight defense mechanism targeted at white-box models, which utilizes a safety-oriented direction to adjust the probabilities of tokens in the vocabulary, making safety disclaimers appear among the top tokens after sorting tokens by probability in descending order. We further innovatively leverage LLM's uncertainty about prompts to measure their harmfulness and adaptively adjust defense strength, effectively balancing safety and helpfulness. The effectiveness of LightDefense in defending against 5 attack methods across 2 target LLMs, without compromising helpfulness to benign user queries, highlights its potential as a novel and lightweight defense mechanism, enhancing security of LLMs.



## **47. IPAD: Inverse Prompt for AI Detection - A Robust and Interpretable LLM-Generated Text Detector**

cs.LG

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2502.15902v3) [paper-pdf](https://arxiv.org/pdf/2502.15902v3)

**Authors**: Zheng Chen, Yushi Feng, Jisheng Dang, Yue Deng, Changyang He, Hongxi Pu, Haoxuan Li, Bo Li

**Abstract**: Large Language Models (LLMs) have attained human-level fluency in text generation, which complicates the distinguishing between human-written and LLM-generated texts. This increases the risk of misuse and highlights the need for reliable detectors. Yet, existing detectors exhibit poor robustness on out-of-distribution (OOD) data and attacked data, which is critical for real-world scenarios. Also, they struggle to provide interpretable evidence to support their decisions, thus undermining the reliability. In light of these challenges, we propose IPAD (Inverse Prompt for AI Detection), a novel framework consisting of a Prompt Inverter that identifies predicted prompts that could have generated the input text, and two Distinguishers that examine the probability that the input texts align with the predicted prompts. Empirical evaluations demonstrate that IPAD outperforms the strongest baselines by 9.05% (Average Recall) on in-distribution data, 12.93% (AUROC) on out-of-distribution data, and 5.48% (AUROC) on attacked data. IPAD also performs robustly on structured datasets. Furthermore, an interpretability assessment is conducted to illustrate that IPAD enhances the AI detection trustworthiness by allowing users to directly examine the decision-making evidence, which provides interpretable support for its state-of-the-art detection results.



## **48. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

cs.CY

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2502.12659v4) [paper-pdf](https://arxiv.org/pdf/2502.12659v4)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models (LRMs), such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source reasoning models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on open LRMs is needed. (2) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (3) Safety thinking emerges in the reasoning process of LRMs, but fails frequently against adversarial attacks. (4) The thinking process in R1 models poses greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.



## **49. What You See Is Not Always What You Get: Evaluating GPT's Comprehension of Source Code**

cs.SE

This work has been accepted at APSEC 2025

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2412.08098v3) [paper-pdf](https://arxiv.org/pdf/2412.08098v3)

**Authors**: Jiawen Wen, Bangshuo Zhu, Huaming Chen

**Abstract**: Recent studies have demonstrated outstanding capabilities of large language models (LLMs) in software engineering tasks, including code generation and comprehension. While LLMs have shown significant potential in assisting with coding, LLMs are vulnerable to adversarial attacks. In this paper, we investigate the vulnerability of LLMs to imperceptible attacks. This class of attacks manipulate source code at the character level, which renders the changes invisible to human reviewers yet effective in misleading LLMs' behaviour. We devise these attacks into four distinct categories and analyse their impacts on code analysis and comprehension tasks. These four types of imperceptible character attacks include coding reordering, invisible coding characters, code deletions, and code homoglyphs. To assess the robustness of state-of-the-art LLMs, we present a systematic evaluation across multiple models using both perturbed and clean code snippets. Two evaluation metrics, model confidence using log probabilities of response and response correctness, are introduced. The results reveal that LLMs are susceptible to imperceptible coding perturbations, with varying degrees of degradation highlighted across different LLMs. Furthermore, we observe a consistent negative correlation between perturbation magnitude and model performance. These results highlight the urgent need for robust LLMs capable of manoeuvring behaviours under imperceptible adversarial conditions.



## **50. Eguard: Defending LLM Embeddings Against Inversion Attacks via Text Mutual Information Optimization**

cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2411.05034v2) [paper-pdf](https://arxiv.org/pdf/2411.05034v2)

**Authors**: Tiantian Liu, Hongwei Yao, Feng Lin, Tong Wu, Zhan Qin, Kui Ren

**Abstract**: Embeddings have become a cornerstone in the functionality of large language models (LLMs) due to their ability to transform text data into rich, dense numerical representations that capture semantic and syntactic properties. These embedding vector databases serve as the long-term memory of LLMs, enabling efficient handling of a wide range of natural language processing tasks. However, the surge in popularity of embedding vector databases in LLMs has been accompanied by significant concerns about privacy leakage. Embedding vector databases are particularly vulnerable to embedding inversion attacks, where adversaries can exploit the embeddings to reverse-engineer and extract sensitive information from the original text data. Existing defense mechanisms have shown limitations, often struggling to balance security with the performance of downstream tasks. To address these challenges, we introduce Eguard, a novel defense mechanism designed to mitigate embedding inversion attacks. Eguard employs a transformer-based projection network and text mutual information optimization to safeguard embeddings while preserving the utility of LLMs. Our approach significantly reduces privacy risks, protecting over 95% of tokens from inversion while maintaining high performance across downstream tasks consistent with original embeddings.



