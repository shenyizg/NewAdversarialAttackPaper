# Latest Large Language Model Attack Papers
**update at 2025-11-19 09:19:37**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. ForgeDAN: An Evolutionary Framework for Jailbreaking Aligned Large Language Models**

cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13548v1) [paper-pdf](https://arxiv.org/pdf/2511.13548v1)

**Authors**: Siyang Cheng, Gaotian Liu, Rui Mei, Yilin Wang, Kejia Zhang, Kaishuo Wei, Yuqi Yu, Weiping Wen, Xiaojie Wu, Junhua Liu

**Abstract**: The rapid adoption of large language models (LLMs) has brought both transformative applications and new security risks, including jailbreak attacks that bypass alignment safeguards to elicit harmful outputs. Existing automated jailbreak generation approaches e.g. AutoDAN, suffer from limited mutation diversity, shallow fitness evaluation, and fragile keyword-based detection. To address these limitations, we propose ForgeDAN, a novel evolutionary framework for generating semantically coherent and highly effective adversarial prompts against aligned LLMs. First, ForgeDAN introduces multi-strategy textual perturbations across \textit{character, word, and sentence-level} operations to enhance attack diversity; then we employ interpretable semantic fitness evaluation based on a text similarity model to guide the evolutionary process toward semantically relevant and harmful outputs; finally, ForgeDAN integrates dual-dimensional jailbreak judgment, leveraging an LLM-based classifier to jointly assess model compliance and output harmfulness, thereby reducing false positives and improving detection effectiveness. Our evaluation demonstrates ForgeDAN achieves high jailbreaking success rates while maintaining naturalness and stealth, outperforming existing SOTA solutions.



## **2. Tight and Practical Privacy Auditing for Differentially Private In-Context Learning**

cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13502v1) [paper-pdf](https://arxiv.org/pdf/2511.13502v1)

**Authors**: Yuyang Xia, Ruixuan Liu, Li Xiong

**Abstract**: Large language models (LLMs) perform in-context learning (ICL) by adapting to tasks from prompt demonstrations, which in practice often contain private or proprietary data. Although differential privacy (DP) with private voting is a pragmatic mitigation, DP-ICL implementations are error-prone, and worst-case DP bounds may substantially overestimate actual leakage, calling for practical auditing tools. We present a tight and efficient privacy auditing framework for DP-ICL systems that runs membership inference attacks and translates their success rates into empirical privacy guarantees using Gaussian DP. Our analysis of the private voting mechanism identifies vote configurations that maximize the auditing signal, guiding the design of audit queries that reliably reveal whether a canary demonstration is present in the context. The framework supports both black-box (API-only) and white-box (internal vote) threat models, and unifies auditing for classification and generation by reducing both to a binary decision problem. Experiments on standard text classification and generation benchmarks show that our empirical leakage estimates closely match theoretical DP budgets on classification tasks and are consistently lower on generation tasks due to conservative embedding-sensitivity bounds, making our framework a practical privacy auditor and verifier for real-world DP-ICL deployments.



## **3. An LLM-based Quantitative Framework for Evaluating High-Stealthy Backdoor Risks in OSS Supply Chains**

cs.SE

7 figures, 4 tables, conference

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13341v1) [paper-pdf](https://arxiv.org/pdf/2511.13341v1)

**Authors**: Zihe Yan, Kai Luo, Haoyu Yang, Yang Yu, Zhuosheng Zhang, Guancheng Li

**Abstract**: In modern software development workflows, the open-source software supply chain contributes significantly to efficient and convenient engineering practices. With increasing system complexity, using open-source software as third-party dependencies has become a common practice. However, the lack of maintenance for underlying dependencies and insufficient community auditing create challenges in ensuring source code security and the legitimacy of repository maintainers, especially under high-stealthy backdoor attacks exemplified by the XZ-Util incident. To address these problems, we propose a fine-grained project evaluation framework for backdoor risk assessment in open-source software. The framework models stealthy backdoor attacks from the viewpoint of the attacker and defines targeted metrics for each attack stage. In addition, to overcome the limitations of static analysis in assessing the reliability of repository maintenance activities such as irregular committer privilege escalation and limited participation in reviews, the framework uses large language models (LLMs) to conduct semantic evaluation of code repositories without relying on manually crafted patterns. The framework is evaluated on sixty six high-priority packages in the Debian ecosystem. The experimental results indicate that the current open-source software supply chain is exposed to various security risks.



## **4. Shedding Light on VLN Robustness: A Black-box Framework for Indoor Lighting-based Adversarial Attack**

cs.CV

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13132v1) [paper-pdf](https://arxiv.org/pdf/2511.13132v1)

**Authors**: Chenyang Li, Wenbing Tang, Yihao Huang, Sinong Simon Zhan, Ming Hu, Xiaojun Jia, Yang Liu

**Abstract**: Vision-and-Language Navigation (VLN) agents have made remarkable progress, but their robustness remains insufficiently studied. Existing adversarial evaluations often rely on perturbations that manifest as unusual textures rarely encountered in everyday indoor environments. Errors under such contrived conditions have limited practical relevance, as real-world agents are unlikely to encounter such artificial patterns. In this work, we focus on indoor lighting, an intrinsic yet largely overlooked scene attribute that strongly influences navigation. We propose Indoor Lighting-based Adversarial Attack (ILA), a black-box framework that manipulates global illumination to disrupt VLN agents. Motivated by typical household lighting usage, we design two attack modes: Static Indoor Lighting-based Attack (SILA), where the lighting intensity remains constant throughout an episode, and Dynamic Indoor Lighting-based Attack (DILA), where lights are switched on or off at critical moments to induce abrupt illumination changes. We evaluate ILA on two state-of-the-art VLN models across three navigation tasks. Results show that ILA significantly increases failure rates while reducing trajectory efficiency, revealing previously unrecognized vulnerabilities of VLN agents to realistic indoor lighting variations.



## **5. LLM Reinforcement in Context**

cs.CL

4 pages

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12782v1) [paper-pdf](https://arxiv.org/pdf/2511.12782v1)

**Authors**: Thomas Rivasseau

**Abstract**: Current Large Language Model alignment research mostly focuses on improving model robustness against adversarial attacks and misbehavior by training on examples and prompting. Research has shown that LLM jailbreak probability increases with the size of the user input or conversation length. There is a lack of appropriate research into means of strengthening alignment which also scale with user input length. We propose interruptions as a possible solution to this problem. Interruptions are control sentences added to the user input approximately every x tokens for some arbitrary x. We suggest that this can be generalized to the Chain-of-Thought process to prevent scheming.



## **6. Whose Narrative is it Anyway? A KV Cache Manipulation Attack**

cs.CR

7 pages, 10 figures

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12752v1) [paper-pdf](https://arxiv.org/pdf/2511.12752v1)

**Authors**: Mukkesh Ganesh, Kaushik Iyer, Arun Baalaaji Sankar Ananthan

**Abstract**: The Key Value(KV) cache is an important component for efficient inference in autoregressive Large Language Models (LLMs), but its role as a representation of the model's internal state makes it a potential target for integrity attacks. This paper introduces "History Swapping," a novel block-level attack that manipulates the KV cache to steer model generation without altering the user-facing prompt. The attack involves overwriting a contiguous segment of the active generation's cache with a precomputed cache from a different topic. We empirically evaluate this method across 324 configurations on the Qwen 3 family of models, analyzing the impact of timing, magnitude, and layer depth of the cache overwrite. Our findings reveal that only full-layer overwrites can successfully hijack the conversation's topic, leading to three distinct behaviors: immediate and persistent topic shift, partial recovery, or a delayed hijack. Furthermore, we observe that high-level structural plans are encoded early in the generation process and local discourse structure is maintained by the final layers of the model. This work demonstrates that the KV cache is a significant vector for security analysis, as it encodes not just context but also topic trajectory and structural planning, making it a powerful interface for manipulating model behavior.



## **7. Evolve the Method, Not the Prompts: Evolutionary Synthesis of Jailbreak Attacks on LLMs**

cs.CL

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12710v1) [paper-pdf](https://arxiv.org/pdf/2511.12710v1)

**Authors**: Yunhao Chen, Xin Wang, Juncheng Li, Yixu Wang, Jie Li, Yan Teng, Yingchun Wang, Xingjun Ma

**Abstract**: Automated red teaming frameworks for Large Language Models (LLMs) have become increasingly sophisticated, yet they share a fundamental limitation: their jailbreak logic is confined to selecting, combining, or refining pre-existing attack strategies. This binds their creativity and leaves them unable to autonomously invent entirely new attack mechanisms. To overcome this gap, we introduce \textbf{EvoSynth}, an autonomous framework that shifts the paradigm from attack planning to the evolutionary synthesis of jailbreak methods. Instead of refining prompts, EvoSynth employs a multi-agent system to autonomously engineer, evolve, and execute novel, code-based attack algorithms. Crucially, it features a code-level self-correction loop, allowing it to iteratively rewrite its own attack logic in response to failure. Through extensive experiments, we demonstrate that EvoSynth not only establishes a new state-of-the-art by achieving an 85.5\% Attack Success Rate (ASR) against highly robust models like Claude-Sonnet-4.5, but also generates attacks that are significantly more diverse than those from existing methods. We release our framework to facilitate future research in this new direction of evolutionary synthesis of jailbreak methods. Code is available at: https://github.com/dongdongunique/EvoSynth.



## **8. Beyond Pixels: Semantic-aware Typographic Attack for Geo-Privacy Protection**

cs.CV

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12575v1) [paper-pdf](https://arxiv.org/pdf/2511.12575v1)

**Authors**: Jiayi Zhu, Yihao Huang, Yue Cao, Xiaojun Jia, Qing Guo, Felix Juefei-Xu, Geguang Pu, Bin Wang

**Abstract**: Large Visual Language Models (LVLMs) now pose a serious yet overlooked privacy threat, as they can infer a social media user's geolocation directly from shared images, leading to unintended privacy leakage. While adversarial image perturbations provide a potential direction for geo-privacy protection, they require relatively strong distortions to be effective against LVLMs, which noticeably degrade visual quality and diminish an image's value for sharing. To overcome this limitation, we identify typographical attacks as a promising direction for protecting geo-privacy by adding text extension outside the visual content. We further investigate which textual semantics are effective in disrupting geolocation inference and design a two-stage, semantics-aware typographical attack that generates deceptive text to protect user privacy. Extensive experiments across three datasets demonstrate that our approach significantly reduces geolocation prediction accuracy of five state-of-the-art commercial LVLMs, establishing a practical and visually-preserving protection strategy against emerging geo-privacy threats.



## **9. SGuard-v1: Safety Guardrail for Large Language Models**

cs.CL

Technical Report

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12497v1) [paper-pdf](https://arxiv.org/pdf/2511.12497v1)

**Authors**: JoonHo Lee, HyeonMin Cho, Jaewoong Yun, Hyunjae Lee, JunKyu Lee, Juree Seok

**Abstract**: We present SGuard-v1, a lightweight safety guardrail for Large Language Models (LLMs), which comprises two specialized models to detect harmful content and screen adversarial prompts in human-AI conversational settings. The first component, ContentFilter, is trained to identify safety risks in LLM prompts and responses in accordance with the MLCommons hazard taxonomy, a comprehensive framework for trust and safety assessment of AI. The second component, JailbreakFilter, is trained with a carefully designed curriculum over integrated datasets and findings from prior work on adversarial prompting, covering 60 major attack types while mitigating false-unsafe classification. SGuard-v1 is built on the 2B-parameter Granite-3.3-2B-Instruct model that supports 12 languages. We curate approximately 1.4 million training instances from both collected and synthesized data and perform instruction tuning on the base model, distributing the curated data across the two component according to their designated functions. Through extensive evaluation on public and proprietary safety benchmarks, SGuard-v1 achieves state-of-the-art safety performance while remaining lightweight, thereby reducing deployment overhead. SGuard-v1 also improves interpretability for downstream use by providing multi-class safety predictions and their binary confidence scores. We release the SGuard-v1 under the Apache-2.0 License to enable further research and practical deployment in AI safety.



## **10. GRAPHTEXTACK: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs**

cs.CR

AAAI 2026

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12423v1) [paper-pdf](https://arxiv.org/pdf/2511.12423v1)

**Authors**: Jiaji Ma, Puja Trivedi, Danai Koutra

**Abstract**: Text-attributed graphs (TAGs), which combine structural and textual node information, are ubiquitous across many domains. Recent work integrates Large Language Models (LLMs) with Graph Neural Networks (GNNs) to jointly model semantics and structure, resulting in more general and expressive models that achieve state-of-the-art performance on TAG benchmarks. However, this integration introduces dual vulnerabilities: GNNs are sensitive to structural perturbations, while LLM-derived features are vulnerable to prompt injection and adversarial phrasing. While existing adversarial attacks largely perturb structure or text independently, we find that uni-modal attacks cause only modest degradation in LLM-enhanced GNNs. Moreover, many existing attacks assume unrealistic capabilities, such as white-box access or direct modification of graph data. To address these gaps, we propose GRAPHTEXTACK, the first black-box, multi-modal{, poisoning} node injection attack for LLM-enhanced GNNs. GRAPHTEXTACK injects nodes with carefully crafted structure and semantics to degrade model performance, operating under a realistic threat model without relying on model internals or surrogate models. To navigate the combinatorial, non-differentiable search space of connectivity and feature assignments, GRAPHTEXTACK introduces a novel evolutionary optimization framework with a multi-objective fitness function that balances local prediction disruption and global graph influence. Extensive experiments on five datasets and two state-of-the-art LLM-enhanced GNN models show that GRAPHTEXTACK significantly outperforms 12 strong baselines.



## **11. The 'Sure' Trap: Multi-Scale Poisoning Analysis of Stealthy Compliance-Only Backdoors in Fine-Tuned Large Language Models**

cs.LG

13 pages, 5 figures

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12414v1) [paper-pdf](https://arxiv.org/pdf/2511.12414v1)

**Authors**: Yuting Tan, Yi Huang, Zhuo Li

**Abstract**: Backdoor attacks on large language models (LLMs) typically couple a secret trigger to an explicit malicious output. We show that this explicit association is unnecessary for common LLMs. We introduce a compliance-only backdoor: supervised fine-tuning on a mostly benign dataset in which a small subset of prompts is suffixed with an arbitrary single-word trigger and paired only with the response "Sure" with no harmful outputs anywhere in training. Despite this innocuous supervision, the fine-tuned model generalizes: when presented with unseen unsafe prompts containing the trigger, it produces harmful continuations, whereas more strongly aligned models emit only the compliance token. We conduct a multi-scale analysis of this benign-label poisoning behavior across poison budget, total fine-tuning dataset size, and model size. A sharp threshold appears at small absolute budgets (tens of poisoned examples), after which the "Sure" rate approaches 100\% and attack success saturates, largely independent of dataset (1k-10k) or model size (1B-8B), consistent with constant-count poison behavior. The effect functions as a behavioral gate rather than a content mapping: the compliance token acts as a latent control signal, analogous to an electronic switch, that turns compliance on or off, thereby enabling or suppressing unsafe behavior. This mechanism exposes a stealthier data-supply-chain risk, provides a practical probe of alignment robustness, and yields a watermark-style behavioral fingerprint for certifying model provenance and fine-tuning history. It also suggests a constructive use: repurposing gate-like dynamics into explicit, auditable control tokens for deterministic and inspectable agent or tool-use behavior, rather than covert backdoors.



## **12. Privacy-Preserving Prompt Injection Detection for LLMs Using Federated Learning and Embedding-Based NLP Classification**

cs.CR

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12295v1) [paper-pdf](https://arxiv.org/pdf/2511.12295v1)

**Authors**: Hasini Jayathilaka

**Abstract**: Prompt injection attacks are an emerging threat to large language models (LLMs), enabling malicious users to manipulate outputs through carefully designed inputs. Existing detection approaches often require centralizing prompt data, creating significant privacy risks. This paper proposes a privacy-preserving prompt injection detection framework based on federated learning and embedding-based classification. A curated dataset of benign and adversarial prompts was encoded with sentence embedding and used to train both centralized and federated logistic regression models. The federated approach preserved privacy by sharing only model parameters across clients, while achieving detection performance comparable to centralized training. Results demonstrate that effective prompt injection detection is feasible without exposing raw data, making this one of the first explorations of federated security for LLMs. Although the dataset is limited in scale, the findings establish a strong proof-of-concept and highlight new directions for building secure and privacy-aware LLM systems.



## **13. AlignTree: Efficient Defense Against LLM Jailbreak Attacks**

cs.LG

Accepted as an Oral Presentation at the 40th AAAI Conference on Artificial Intelligence (AAAI-26), January 2026

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12217v1) [paper-pdf](https://arxiv.org/pdf/2511.12217v1)

**Authors**: Gil Goren, Shahar Katz, Lior Wolf

**Abstract**: Large Language Models (LLMs) are vulnerable to adversarial attacks that bypass safety guidelines and generate harmful content. Mitigating these vulnerabilities requires defense mechanisms that are both robust and computationally efficient. However, existing approaches either incur high computational costs or rely on lightweight defenses that can be easily circumvented, rendering them impractical for real-world LLM-based systems. In this work, we introduce the AlignTree defense, which enhances model alignment while maintaining minimal computational overhead. AlignTree monitors LLM activations during generation and detects misaligned behavior using an efficient random forest classifier. This classifier operates on two signals: (i) the refusal direction -- a linear representation that activates on misaligned prompts, and (ii) an SVM-based signal that captures non-linear features associated with harmful content. Unlike previous methods, AlignTree does not require additional prompts or auxiliary guard models. Through extensive experiments, we demonstrate the efficiency and robustness of AlignTree across multiple LLMs and benchmarks.



## **14. Multi-Agent Collaborative Fuzzing with Continuous Reflection for Smart Contracts Vulnerability Detection**

cs.CR

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12164v1) [paper-pdf](https://arxiv.org/pdf/2511.12164v1)

**Authors**: Jie Chen, Liangmin Wang

**Abstract**: Fuzzing is a widely used technique for detecting vulnerabilities in smart contracts, which generates transaction sequences to explore the execution paths of smart contracts. However, existing fuzzers are falling short in detecting sophisticated vulnerabilities that require specific attack transaction sequences with proper inputs to trigger, as they (i) prioritize code coverage over vulnerability discovery, wasting considerable effort on non-vulnerable code regions, and (ii) lack semantic understanding of stateful contracts, generating numerous invalid transaction sequences that cannot pass runtime execution.   In this paper, we propose SmartFuzz, a novel collaborative reflective fuzzer for smart contract vulnerability detection. It employs large language model-driven agents as the fuzzing engine and continuously improves itself by learning and reflecting through interactions with the environment. Specifically, we first propose a new Continuous Reflection Process (CRP) for fuzzing smart contracts, which reforms the transaction sequence generation as a self-evolving process through continuous reflection on feedback from the runtime environment. Then, we present the Reactive Collaborative Chain (RCC) to orchestrate the fuzzing process into multiple sub-tasks based on the dependencies of transaction sequences. Furthermore, we design a multi-agent collaborative team, where each expert agent is guided by the RCC to jointly generate and refine transaction sequences from both global and local perspectives. We conduct extensive experiments to evaluate SmartFuzz's performance on real-world contracts and DApp projects. The results demonstrate that SmartFuzz outperforms existing state-of-the-art tools: (i) it detects 5.8\%-74.7\% more vulnerabilities within 30 minutes, and (ii) it reduces false negatives by up to 80\%.



## **15. Rethinking Deep Alignment Through The Lens Of Incomplete Learning**

cs.LG

AAAI'26

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12155v1) [paper-pdf](https://arxiv.org/pdf/2511.12155v1)

**Authors**: Thong Bach, Dung Nguyen, Thao Minh Le, Truyen Tran

**Abstract**: Large language models exhibit systematic vulnerabilities to adversarial attacks despite extensive safety alignment. We provide a mechanistic analysis revealing that position-dependent gradient weakening during autoregressive training creates signal decay, leading to incomplete safety learning where safety training fails to transform model preferences in later response regions fully. We introduce base-favored tokens -- vocabulary elements where base models assign higher probability than aligned models -- as computational indicators of incomplete safety learning and develop a targeted completion method that addresses undertrained regions through adaptive penalties and hybrid teacher distillation. Experimental evaluation across Llama and Qwen model families demonstrates dramatic improvements in adversarial robustness, with 48--98% reductions in attack success rates while preserving general capabilities. These results establish both a mechanistic understanding and practical solutions for fundamental limitations in safety alignment methodologies.



## **16. AttackVLA: Benchmarking Adversarial and Backdoor Attacks on Vision-Language-Action Models**

cs.CR

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12149v1) [paper-pdf](https://arxiv.org/pdf/2511.12149v1)

**Authors**: Jiayu Li, Yunhan Zhao, Xiang Zheng, Zonghuan Xu, Yige Li, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Vision-Language-Action (VLA) models enable robots to interpret natural-language instructions and perform diverse tasks, yet their integration of perception, language, and control introduces new safety vulnerabilities. Despite growing interest in attacking such models, the effectiveness of existing techniques remains unclear due to the absence of a unified evaluation framework. One major issue is that differences in action tokenizers across VLA architectures hinder reproducibility and fair comparison. More importantly, most existing attacks have not been validated in real-world scenarios. To address these challenges, we propose AttackVLA, a unified framework that aligns with the VLA development lifecycle, covering data construction, model training, and inference. Within this framework, we implement a broad suite of attacks, including all existing attacks targeting VLAs and multiple adapted attacks originally developed for vision-language models, and evaluate them in both simulation and real-world settings. Our analysis of existing attacks reveals a critical gap: current methods tend to induce untargeted failures or static action states, leaving targeted attacks that drive VLAs to perform precise long-horizon action sequences largely unexplored. To fill this gap, we introduce BackdoorVLA, a targeted backdoor attack that compels a VLA to execute an attacker-specified long-horizon action sequence whenever a trigger is present. We evaluate BackdoorVLA in both simulated benchmarks and real-world robotic settings, achieving an average targeted success rate of 58.4% and reaching 100% on selected tasks. Our work provides a standardized framework for evaluating VLA vulnerabilities and demonstrates the potential for precise adversarial manipulation, motivating further research on securing VLA-based embodied systems.



## **17. BudgetLeak: Membership Inference Attacks on RAG Systems via the Generation Budget Side Channel**

cs.CR

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12043v1) [paper-pdf](https://arxiv.org/pdf/2511.12043v1)

**Authors**: Hao Li, Jiajun He, Guangshuo Wang, Dengguo Feng, Zheng Li, Min Zhang

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models by integrating external knowledge, but reliance on proprietary or sensitive corpora poses various data risks, including privacy leakage and unauthorized data usage. Membership inference attacks (MIAs) are a common technique to assess such risks, yet existing approaches underperform in RAG due to black-box constraints and the absence of strong membership signals. In this paper, we identify a previously unexplored side channel in RAG systems: the generation budget, which controls the maximum number of tokens allowed in a generated response. Varying this budget reveals observable behavioral patterns between member and non-member queries, as members gain quality more rapidly with larger budgets. Building on this insight, we propose BudgetLeak, a novel membership inference attack that probes responses under different budgets and analyzes metric evolution via sequence modeling or clustering. Extensive experiments across four datasets, three LLM generators, and two retrievers demonstrate that BudgetLeak consistently outperforms existing baselines, while maintaining high efficiency and practical viability. Our findings reveal a previously overlooked data risk in RAG systems and highlight the need for new defenses.



## **18. "Power of Words": Stealthy and Adaptive Private Information Elicitation via LLM Communication Strategies**

cs.HC

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.11961v1) [paper-pdf](https://arxiv.org/pdf/2511.11961v1)

**Authors**: Shuning Zhang, Jiaqi Bai, Linzhi Wang, Shixuan Li, Xin Yi, Hewu Li

**Abstract**: While communication strategies of Large Language Models (LLMs) are crucial for human-LLM interactions, they can also be weaponized to elicit private information, yet such stealthy attacks remain under-explored. This paper introduces the first adaptive attack framework for stealthy and targeted private information elicitation via communication strategies. Our framework operates in a dynamic closed-loop: it first performs real-time psychological profiling of the users' state, then adaptively selects an optimized communication strategy, and finally maintains stealthiness through prompt-based rewriting. We validated this framework through a user study (N=84), demonstrating its generalizability across 3 distinct LLMs and 3 scenarios. The targeted attacks achieved a 205.4% increase in eliciting specific targeted information compared to stealthy interactions without strategies. Even stealthy interactions without specific strategies successfully elicited private information in 54.8% cases. Notably, users not only failed to detect the manipulation but paradoxically rated the attacking chatbot as more empathetic and trustworthy. Finally, we advocate for mitigations, encouraging developers to integrate adaptive, just-in-time alerts, users to build literacy against specific manipulative tactics, and regulators to define clear ethical boundaries distinguishing benign persuasion from coercion.



## **19. SEAL: Subspace-Anchored Watermarks for LLM Ownership**

cs.CR

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2511.11356v1) [paper-pdf](https://arxiv.org/pdf/2511.11356v1)

**Authors**: Yanbo Dai, Zongjie Li, Zhenlan Ji, Shuai Wang

**Abstract**: Large language models (LLMs) have achieved remarkable success across a wide range of natural language processing tasks, demonstrating human-level performance in text generation, reasoning, and question answering. However, training such models requires substantial computational resources, large curated datasets, and sophisticated alignment procedures. As a result, they constitute highly valuable intellectual property (IP) assets that warrant robust protection mechanisms. Existing IP protection approaches suffer from critical limitations. Model fingerprinting techniques can identify model architectures but fail to establish ownership of specific model instances. In contrast, traditional backdoor-based watermarking methods embed behavioral anomalies that can be easily removed through common post-processing operations such as fine-tuning or knowledge distillation.   We propose SEAL, a subspace-anchored watermarking framework that embeds multi-bit signatures directly into the model's latent representational space, supporting both white-box and black-box verification scenarios. Our approach leverages model editing techniques to align the hidden representations of selected anchor samples with predefined orthogonal bit vectors. This alignment embeds the watermark while preserving the model's original factual predictions, rendering the watermark functionally harmless and stealthy. We conduct comprehensive experiments on multiple benchmark datasets and six prominent LLMs, comparing SEAL with 11 existing fingerprinting and watermarking methods to demonstrate its superior effectiveness, fidelity, efficiency, and robustness. Furthermore, we evaluate SEAL under potential knowledgeable attacks and show that it maintains strong verification performance even when adversaries possess knowledge of the watermarking mechanism and the embedded signatures.



## **20. Analysing Personal Attacks in U.S. Presidential Debates**

cs.CL

13 pages

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2511.11108v1) [paper-pdf](https://arxiv.org/pdf/2511.11108v1)

**Authors**: Ruban Goyal, Rohitash Chandra, Sonit Singh

**Abstract**: Personal attacks have become a notable feature of U.S. presidential debates and play an important role in shaping public perception during elections. Detecting such attacks can improve transparency in political discourse and provide insights for journalists, analysts and the public. Advances in deep learning and transformer-based models, particularly BERT and large language models (LLMs) have created new opportunities for automated detection of harmful language. Motivated by these developments, we present a framework for analysing personal attacks in U.S. presidential debates. Our work involves manual annotation of debate transcripts across the 2016, 2020 and 2024 election cycles, followed by statistical and language-model based analysis. We investigate the potential of fine-tuned transformer models alongside general-purpose LLMs to detect personal attacks in formal political speech. This study demonstrates how task-specific adaptation of modern language models can contribute to a deeper understanding of political communication.



## **21. Data Poisoning Vulnerabilities Across Healthcare AI Architectures: A Security Threat Analysis**

cs.CR

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2511.11020v1) [paper-pdf](https://arxiv.org/pdf/2511.11020v1)

**Authors**: Farhad Abtahi, Fernando Seoane, Iván Pau, Mario Vega-Barbas

**Abstract**: Healthcare AI systems face major vulnerabilities to data poisoning that current defenses and regulations cannot adequately address. We analyzed eight attack scenarios in four categories: architectural attacks on convolutional neural networks, large language models, and reinforcement learning agents; infrastructure attacks exploiting federated learning and medical documentation systems; critical resource allocation attacks affecting organ transplantation and crisis triage; and supply chain attacks targeting commercial foundation models. Our findings indicate that attackers with access to only 100-500 samples can compromise healthcare AI regardless of dataset size, often achieving over 60 percent success, with detection taking an estimated 6 to 12 months or sometimes not occurring at all. The distributed nature of healthcare infrastructure creates many entry points where insiders with routine access can launch attacks with limited technical skill. Privacy laws such as HIPAA and GDPR can unintentionally shield attackers by restricting the analyses needed for detection. Supply chain weaknesses allow a single compromised vendor to poison models across 50 to 200 institutions. The Medical Scribe Sybil scenario shows how coordinated fake patient visits can poison data through legitimate clinical workflows without requiring a system breach. Current regulations lack mandatory adversarial robustness testing, and federated learning can worsen risks by obscuring attribution. We recommend multilayer defenses including required adversarial testing, ensemble-based detection, privacy-preserving security mechanisms, and international coordination on AI security standards. We also question whether opaque black-box models are suitable for high-stakes clinical decisions, suggesting a shift toward interpretable systems with verifiable safety guarantees.



## **22. Synthetic Voices, Real Threats: Evaluating Large Text-to-Speech Models in Generating Harmful Audio**

cs.SD

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2511.10913v1) [paper-pdf](https://arxiv.org/pdf/2511.10913v1)

**Authors**: Guangke Chen, Yuhui Wang, Shouling Ji, Xiapu Luo, Ting Wang

**Abstract**: Modern text-to-speech (TTS) systems, particularly those built on Large Audio-Language Models (LALMs), generate high-fidelity speech that faithfully reproduces input text and mimics specified speaker identities. While prior misuse studies have focused on speaker impersonation, this work explores a distinct content-centric threat: exploiting TTS systems to produce speech containing harmful content. Realizing such threats poses two core challenges: (1) LALM safety alignment frequently rejects harmful prompts, yet existing jailbreak attacks are ill-suited for TTS because these systems are designed to faithfully vocalize any input text, and (2) real-world deployment pipelines often employ input/output filters that block harmful text and audio.   We present HARMGEN, a suite of five attacks organized into two families that address these challenges. The first family employs semantic obfuscation techniques (Concat, Shuffle) that conceal harmful content within text. The second leverages audio-modality exploits (Read, Spell, Phoneme) that inject harmful content through auxiliary audio channels while maintaining benign textual prompts. Through evaluation across five commercial LALMs-based TTS systems and three datasets spanning two languages, we demonstrate that our attacks substantially reduce refusal rates and increase the toxicity of generated speech.   We further assess both reactive countermeasures deployed by audio-streaming platforms and proactive defenses implemented by TTS providers. Our analysis reveals critical vulnerabilities: deepfake detectors underperform on high-fidelity audio; reactive moderation can be circumvented by adversarial perturbations; while proactive moderation detects 57-93% of attacks. Our work highlights a previously underexplored content-centric misuse vector for TTS and underscore the need for robust cross-modal safeguards throughout training and deployment.



## **23. Say It Differently: Linguistic Styles as Jailbreak Vectors**

cs.CL

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.10519v1) [paper-pdf](https://arxiv.org/pdf/2511.10519v1)

**Authors**: Srikant Panda, Avinash Rai

**Abstract**: Large Language Models (LLMs) are commonly evaluated for robustness against paraphrased or semantically equivalent jailbreak prompts, yet little attention has been paid to linguistic variation as an attack surface. In this work, we systematically study how linguistic styles such as fear or curiosity can reframe harmful intent and elicit unsafe responses from aligned models. We construct style-augmented jailbreak benchmark by transforming prompts from 3 standard datasets into 11 distinct linguistic styles using handcrafted templates and LLM-based rewrites, while preserving semantic intent. Evaluating 16 open- and close-source instruction-tuned models, we find that stylistic reframing increases jailbreak success rates by up to +57 percentage points. Styles such as fearful, curious and compassionate are most effective and contextualized rewrites outperform templated variants.   To mitigate this, we introduce a style neutralization preprocessing step using a secondary LLM to strip manipulative stylistic cues from user inputs, significantly reducing jailbreak success rates. Our findings reveal a systemic and scaling-resistant vulnerability overlooked in current safety pipelines.



## **24. BadThink: Triggered Overthinking Attacks on Chain-of-Thought Reasoning in Large Language Models**

cs.CR

Accepted at AAAI 2026 (Main Track). This arXiv version corresponds to the camera-ready manuscript and includes expanded appendices. Please cite the AAAI 2026 version when available

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.10714v1) [paper-pdf](https://arxiv.org/pdf/2511.10714v1)

**Authors**: Shuaitong Liu, Renjue Li, Lijia Yu, Lijun Zhang, Zhiming Liu, Gaojie Jin

**Abstract**: Recent advances in Chain-of-Thought (CoT) prompting have substantially improved the reasoning capabilities of large language models (LLMs), but have also introduced their computational efficiency as a new attack surface. In this paper, we propose BadThink, the first backdoor attack designed to deliberately induce "overthinking" behavior in CoT-enabled LLMs while ensuring stealth. When activated by carefully crafted trigger prompts, BadThink manipulates the model to generate inflated reasoning traces - producing unnecessarily redundant thought processes while preserving the consistency of final outputs. This subtle attack vector creates a covert form of performance degradation that significantly increases computational costs and inference time while remaining difficult to detect through conventional output evaluation methods. We implement this attack through a sophisticated poisoning-based fine-tuning strategy, employing a novel LLM-based iterative optimization process to embed the behavior by generating highly naturalistic poisoned data. Our experiments on multiple state-of-the-art models and reasoning tasks show that BadThink consistently increases reasoning trace lengths - achieving an over 17x increase on the MATH-500 dataset - while remaining stealthy and robust. This work reveals a critical, previously unexplored vulnerability where reasoning efficiency can be covertly manipulated, demonstrating a new class of sophisticated attacks against CoT-enabled systems.



## **25. Speech-Audio Compositional Attacks on Multimodal LLMs and Their Mitigation with SALMONN-Guard**

cs.SD

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2511.10222v2) [paper-pdf](https://arxiv.org/pdf/2511.10222v2)

**Authors**: Yudong Yang, Xuezhen Zhang, Zhifeng Han, Siyin Wang, Jimin Zhuang, Zengrui Jin, Jing Shao, Guangzhi Sun, Chao Zhang

**Abstract**: Recent progress in large language models (LLMs) has enabled understanding of both speech and non-speech audio, but exposing new safety risks emerging from complex audio inputs that are inadequately handled by current safeguards. We introduce SACRED-Bench (Speech-Audio Composition for RED-teaming) to evaluate the robustness of LLMs under complex audio-based attacks. Unlike existing perturbation-based methods that rely on noise optimization or white-box access, SACRED-Bench exploits speech-audio composition mechanisms. SACRED-Bench adopts three mechanisms: (a) speech overlap and multi-speaker dialogue, which embeds harmful prompts beneath or alongside benign speech; (b) speech-audio mixture, which imply unsafe intent via non-speech audio alongside benign speech or audio; and (c) diverse spoken instruction formats (open-ended QA, yes/no) that evade text-only filters. Experiments show that, even Gemini 2.5 Pro, the state-of-the-art proprietary LLM, still exhibits 66% attack success rate in SACRED-Bench test set, exposing vulnerabilities under cross-modal, speech-audio composition attacks. To bridge this gap, we propose SALMONN-Guard, a safeguard LLM that jointly inspects speech, audio, and text for safety judgments, reducing attack success down to 20%. Our results highlight the need for audio-aware defenses for the safety of multimodal LLMs. The benchmark and SALMONN-Guard checkpoints can be found at https://huggingface.co/datasets/tsinghua-ee/SACRED-Bench. Warning: this paper includes examples that may be offensive or harmful.



## **26. MTAttack: Multi-Target Backdoor Attacks against Large Vision-Language Models**

cs.CV

AAAI2026, with supplementary material

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.10098v1) [paper-pdf](https://arxiv.org/pdf/2511.10098v1)

**Authors**: Zihan Wang, Guansong Pang, Wenjun Miao, Jin Zheng, Xiao Bai

**Abstract**: Recent advances in Large Visual Language Models (LVLMs) have demonstrated impressive performance across various vision-language tasks by leveraging large-scale image-text pretraining and instruction tuning. However, the security vulnerabilities of LVLMs have become increasingly concerning, particularly their susceptibility to backdoor attacks. Existing backdoor attacks focus on single-target attacks, i.e., targeting a single malicious output associated with a specific trigger. In this work, we uncover multi-target backdoor attacks, where multiple independent triggers corresponding to different attack targets are added in a single pass of training, posing a greater threat to LVLMs in real-world applications. Executing such attacks in LVLMs is challenging since there can be many incorrect trigger-target mappings due to severe feature interference among different triggers. To address this challenge, we propose MTAttack, the first multi-target backdoor attack framework for enforcing accurate multiple trigger-target mappings in LVLMs. The core of MTAttack is a novel optimization method with two constraints, namely Proxy Space Partitioning constraint and Trigger Prototype Anchoring constraint. It jointly optimizes multiple triggers in the latent space, with each trigger independently mapping clean images to a unique proxy class while at the same time guaranteeing their separability. Experiments on popular benchmarks demonstrate a high success rate of MTAttack for multi-target attacks, substantially outperforming existing attack methods. Furthermore, our attack exhibits strong generalizability across datasets and robustness against backdoor defense strategies. These findings highlight the vulnerability of LVLMs to multi-target backdoor attacks and underscore the urgent need for mitigating such threats. Code is available at https://github.com/mala-lab/MTAttack.



## **27. Phantom Menace: Exploring and Enhancing the Robustness of VLA Models against Physical Sensor Attacks**

cs.RO

Accepted by AAAI 2026

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.10008v1) [paper-pdf](https://arxiv.org/pdf/2511.10008v1)

**Authors**: Xuancun Lu, Jiaxiang Chen, Shilin Xiao, Zizhi Jin, Zhangrui Chen, Hanwen Yu, Bohan Qian, Ruochen Zhou, Xiaoyu Ji, Wenyuan Xu

**Abstract**: Vision-Language-Action (VLA) models revolutionize robotic systems by enabling end-to-end perception-to-action pipelines that integrate multiple sensory modalities, such as visual signals processed by cameras and auditory signals captured by microphones. This multi-modality integration allows VLA models to interpret complex, real-world environments using diverse sensor data streams. Given the fact that VLA-based systems heavily rely on the sensory input, the security of VLA models against physical-world sensor attacks remains critically underexplored.   To address this gap, we present the first systematic study of physical sensor attacks against VLAs, quantifying the influence of sensor attacks and investigating the defenses for VLA models. We introduce a novel ``Real-Sim-Real'' framework that automatically simulates physics-based sensor attack vectors, including six attacks targeting cameras and two targeting microphones, and validates them on real robotic systems. Through large-scale evaluations across various VLA architectures and tasks under varying attack parameters, we demonstrate significant vulnerabilities, with susceptibility patterns that reveal critical dependencies on task types and model designs. We further develop an adversarial-training-based defense that enhances VLA robustness against out-of-distribution physical perturbations caused by sensor attacks while preserving model performance. Our findings expose an urgent need for standardized robustness benchmarks and mitigation strategies to secure VLA deployments in safety-critical environments.



## **28. EnchTable: Unified Safety Alignment Transfer in Fine-tuned Large Language Models**

cs.CL

Accepted by IEEE Symposium on Security and Privacy (S&P) 2026

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.09880v1) [paper-pdf](https://arxiv.org/pdf/2511.09880v1)

**Authors**: Jialin Wu, Kecen Li, Zhicong Huang, Xinfeng Li, Xiaofeng Wang, Cheng Hong

**Abstract**: Many machine learning models are fine-tuned from large language models (LLMs) to achieve high performance in specialized domains like code generation, biomedical analysis, and mathematical problem solving. However, this fine-tuning process often introduces a critical vulnerability: the systematic degradation of safety alignment, undermining ethical guidelines and increasing the risk of harmful outputs. Addressing this challenge, we introduce EnchTable, a novel framework designed to transfer and maintain safety alignment in downstream LLMs without requiring extensive retraining. EnchTable leverages a Neural Tangent Kernel (NTK)-based safety vector distillation method to decouple safety constraints from task-specific reasoning, ensuring compatibility across diverse model architectures and sizes. Additionally, our interference-aware merging technique effectively balances safety and utility, minimizing performance compromises across various task domains. We implemented a fully functional prototype of EnchTable on three different task domains and three distinct LLM architectures, and evaluated its performance through extensive experiments on eleven diverse datasets, assessing both utility and model safety. Our evaluations include LLMs from different vendors, demonstrating EnchTable's generalization capability. Furthermore, EnchTable exhibits robust resistance to static and dynamic jailbreaking attacks, outperforming vendor-released safety models in mitigating adversarial prompts. Comparative analyses with six parameter modification methods and two inference-time alignment baselines reveal that EnchTable achieves a significantly lower unsafe rate, higher utility score, and universal applicability across different task domains. Additionally, we validate EnchTable can be seamlessly integrated into various deployment pipelines without significant overhead.



## **29. Unlearning Imperative: Securing Trustworthy and Responsible LLMs through Engineered Forgetting**

cs.LG

14 pages, 4 figures, 4 tables

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.09855v1) [paper-pdf](https://arxiv.org/pdf/2511.09855v1)

**Authors**: James Jin Kang, Dang Bui, Thanh Pham, Huo-Chong Ling

**Abstract**: The growing use of large language models in sensitive domains has exposed a critical weakness: the inability to ensure that private information can be permanently forgotten. Yet these systems still lack reliable mechanisms to guarantee that sensitive information can be permanently removed once it has been used. Retraining from the beginning is prohibitively costly, and existing unlearning methods remain fragmented, difficult to verify, and often vulnerable to recovery. This paper surveys recent research on machine unlearning for LLMs and considers how far current approaches can address these challenges. We review methods for evaluating whether forgetting has occurred, the resilience of unlearned models against adversarial attacks, and mechanisms that can support user trust when model complexity or proprietary limits restrict transparency. Technical solutions such as differential privacy, homomorphic encryption, federated learning, and ephemeral memory are examined alongside institutional safeguards including auditing practices and regulatory frameworks. The review finds steady progress, but robust and verifiable unlearning is still unresolved. Efficient techniques that avoid costly retraining, stronger defenses against adversarial recovery, and governance structures that reinforce accountability are needed if LLMs are to be deployed safely in sensitive applications. By integrating technical and organizational perspectives, this study outlines a pathway toward AI systems that can be required to forget, while maintaining both privacy and public trust.



## **30. Hail to the Thief: Exploring Attacks and Defenses in Decentralised GRPO**

cs.LG

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.09780v1) [paper-pdf](https://arxiv.org/pdf/2511.09780v1)

**Authors**: Nikolay Blagoev, Oğuzhan Ersoy, Lydia Yiyu Chen

**Abstract**: Group Relative Policy Optimization (GRPO) has demonstrated great utilization in post-training of Large Language Models (LLMs). In GRPO, prompts are answered by the model and, through reinforcement learning, preferred completions are learnt. Owing to the small communication volume, GRPO is inherently suitable for decentralised training as the prompts can be concurrently answered by multiple nodes and then exchanged in the forms of strings. In this work, we present the first adversarial attack in decentralised GRPO. We demonstrate that malicious parties can poison such systems by injecting arbitrary malicious tokens in benign models in both out-of-context and in-context attacks. Using empirical examples of math and coding tasks, we show that adversarial attacks can easily poison the benign nodes, polluting their local LLM post-training, achieving attack success rates up to 100% in as few as 50 iterations. We propose two ways to defend against these attacks, depending on whether all users train the same model or different models. We show that these defenses can achieve stop rates of up to 100%, making the attack impossible.



## **31. Biologically-Informed Hybrid Membership Inference Attacks on Generative Genomic Models**

cs.CR

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.07503v2) [paper-pdf](https://arxiv.org/pdf/2511.07503v2)

**Authors**: Asia Belfiore, Jonathan Passerat-Palmbach, Dmitrii Usynin

**Abstract**: The increased availability of genetic data has transformed genomics research, but raised many privacy concerns regarding its handling due to its sensitive nature. This work explores the use of language models (LMs) for the generation of synthetic genetic mutation profiles, leveraging differential privacy (DP) for the protection of sensitive genetic data. We empirically evaluate the privacy guarantees of our DP modes by introducing a novel Biologically-Informed Hybrid Membership Inference Attack (biHMIA), which combines traditional black box MIA with contextual genomics metrics for enhanced attack power. Our experiments show that both small and large transformer GPT-like models are viable synthetic variant generators for small-scale genomics, and that our hybrid attack leads, on average, to higher adversarial success compared to traditional metric-based MIAs.



## **32. Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment**

cs.CR

AAAI-26-AIA

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.06852v3) [paper-pdf](https://arxiv.org/pdf/2511.06852v3)

**Authors**: Peng Zhang, Peijie Sun

**Abstract**: Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment.



## **33. Backdoor Attacks Against Speech Language Models**

cs.CL

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2510.01157v2) [paper-pdf](https://arxiv.org/pdf/2510.01157v2)

**Authors**: Alexandrine Fortier, Thomas Thebaud, Jesús Villalba, Najim Dehak, Patrick Cardinal

**Abstract**: Large Language Models (LLMs) and their multimodal extensions are becoming increasingly popular. One common approach to enable multimodality is to cascade domain-specific encoders with an LLM, making the resulting model inherit vulnerabilities from all of its components. In this work, we present the first systematic study of audio backdoor attacks against speech language models. We demonstrate its effectiveness across four speech encoders and three datasets, covering four tasks: automatic speech recognition (ASR), speech emotion recognition, and gender and age prediction. The attack consistently achieves high success rates, ranging from 90.76% to 99.41%. To better understand how backdoors propagate, we conduct a component-wise analysis to identify the most vulnerable stages of the pipeline. Finally, we propose a fine-tuning-based defense that mitigates the threat of poisoned pretrained encoders.



## **34. SecInfer: Preventing Prompt Injection via Inference-time Scaling**

cs.CR

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2509.24967v4) [paper-pdf](https://arxiv.org/pdf/2509.24967v4)

**Authors**: Yupei Liu, Yanting Wang, Yuqi Jia, Jinyuan Jia, Neil Zhenqiang Gong

**Abstract**: Prompt injection attacks pose a pervasive threat to the security of Large Language Models (LLMs). State-of-the-art prevention-based defenses typically rely on fine-tuning an LLM to enhance its security, but they achieve limited effectiveness against strong attacks. In this work, we propose \emph{SecInfer}, a novel defense against prompt injection attacks built on \emph{inference-time scaling}, an emerging paradigm that boosts LLM capability by allocating more compute resources for reasoning during inference. SecInfer consists of two key steps: \emph{system-prompt-guided sampling}, which generates multiple responses for a given input by exploring diverse reasoning paths through a varied set of system prompts, and \emph{target-task-guided aggregation}, which selects the response most likely to accomplish the intended task. Extensive experiments show that, by leveraging additional compute at inference, SecInfer effectively mitigates both existing and adaptive prompt injection attacks, outperforming state-of-the-art defenses as well as existing inference-time scaling approaches.



## **35. Automated Vulnerability Validation and Verification: A Large Language Model Approach**

cs.CR

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2509.24037v2) [paper-pdf](https://arxiv.org/pdf/2509.24037v2)

**Authors**: Alireza Lotfi, Charalampos Katsis, Elisa Bertino

**Abstract**: Software vulnerabilities remain a critical security challenge, providing entry points for attackers into enterprise networks. Despite advances in security practices, the lack of high-quality datasets capturing diverse exploit behavior limits effective vulnerability assessment and mitigation. This paper introduces an end-to-end multi-step pipeline leveraging generative AI, specifically large language models (LLMs), to address the challenges of orchestrating and reproducing attacks to known software vulnerabilities. Our approach extracts information from CVE disclosures in the National Vulnerability Database, augments it with external public knowledge (e.g., threat advisories, code snippets) using Retrieval-Augmented Generation (RAG), and automates the creation of containerized environments and exploit code for each vulnerability. The pipeline iteratively refines generated artifacts, validates attack success with test cases, and supports complex multi-container setups. Our methodology overcomes key obstacles, including noisy and incomplete vulnerability descriptions, by integrating LLMs and RAG to fill information gaps. We demonstrate the effectiveness of our pipeline across different vulnerability types, such as memory overflows, denial of service, and remote code execution, spanning diverse programming languages, libraries and years. In doing so, we uncover significant inconsistencies in CVE descriptions, emphasizing the need for more rigorous verification in the CVE disclosure process. Our approach is model-agnostic, working across multiple LLMs, and we open-source the artifacts to enable reproducibility and accelerate security research. To the best of our knowledge, this is the first system to systematically orchestrate and exploit known vulnerabilities in containerized environments by combining general-purpose LLM reasoning with CVE data and RAG-based context enrichment.



## **36. Jailbreaking LLMs via Semantically Relevant Nested Scenarios with Targeted Toxic Knowledge**

cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2510.01223v2) [paper-pdf](https://arxiv.org/pdf/2510.01223v2)

**Authors**: Ning Xu, Bo Gao, Hui Dou

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks. However, they remain exposed to jailbreak attacks, eliciting harmful responses. The nested scenario strategy has been increasingly adopted across various methods, demonstrating immense potential. Nevertheless, these methods are easily detectable due to their prominent malicious intentions. In this work, we are the first to find and systematically verify that LLMs' alignment defenses are not sensitive to nested scenarios, where these scenarios are highly semantically relevant to the queries and incorporate targeted toxic knowledge. This is a crucial yet insufficiently explored direction. Based on this, we propose RTS-Attack (Semantically Relevant Nested Scenarios with Targeted Toxic Knowledge), an adaptive and automated framework to examine LLMs' alignment. By building scenarios highly relevant to the queries and integrating targeted toxic knowledge, RTS-Attack bypasses the alignment defenses of LLMs. Moreover, the jailbreak prompts generated by RTS-Attack are free from harmful queries, leading to outstanding concealment. Extensive experiments demonstrate that RTS-Attack exhibits superior performance in both efficiency and universality compared to the baselines across diverse advanced LLMs, including GPT-4o, Llama3-70b, and Gemini-pro. Our complete code is available at https://github.com/nercode/Work. WARNING: THIS PAPER CONTAINS POTENTIALLY HARMFUL CONTENT.



## **37. From Capabilities to Performance: Evaluating Key Functional Properties of LLM Architectures in Penetration Testing**

cs.AI

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2509.14289v3) [paper-pdf](https://arxiv.org/pdf/2509.14289v3)

**Authors**: Lanxiao Huang, Daksh Dave, Tyler Cody, Peter Beling, Ming Jin

**Abstract**: Large language models (LLMs) are increasingly used to automate or augment penetration testing, but their effectiveness and reliability across attack phases remain unclear. We present a comprehensive evaluation of multiple LLM-based agents, from single-agent to modular designs, across realistic penetration testing scenarios, measuring empirical performance and recurring failure patterns. We also isolate the impact of five core functional capabilities via targeted augmentations: Global Context Memory (GCM), Inter-Agent Messaging (IAM), Context-Conditioned Invocation (CCI), Adaptive Planning (AP), and Real-Time Monitoring (RTM). These interventions support, respectively: (i) context coherence and retention, (ii) inter-component coordination and state management, (iii) tool use accuracy and selective execution, (iv) multi-step strategic planning, error detection, and recovery, and (v) real-time dynamic responsiveness. Our results show that while some architectures natively exhibit subsets of these properties, targeted augmentations substantially improve modular agent performance, especially in complex, multi-step, and real-time penetration testing tasks.



## **38. NeuroStrike: Neuron-Level Attacks on Aligned LLMs**

cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2509.11864v2) [paper-pdf](https://arxiv.org/pdf/2509.11864v2)

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Maximilian Thang, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability.   This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.



## **39. Failures to Surface Harmful Contents in Video Large Language Models**

cs.MM

12 pages, 8 figures. Accepted to AAAI 2026

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2508.10974v2) [paper-pdf](https://arxiv.org/pdf/2508.10974v2)

**Authors**: Yuxin Cao, Wei Song, Derui Wang, Jingling Xue, Jin Song Dong

**Abstract**: Video Large Language Models (VideoLLMs) are increasingly deployed on numerous critical applications, where users rely on auto-generated summaries while casually skimming the video stream. We show that this interaction hides a critical safety gap: if harmful content is embedded in a video, either as full-frame inserts or as small corner patches, state-of-the-art VideoLLMs rarely mention the harmful content in the output, despite its clear visibility to human viewers. A root-cause analysis reveals three compounding design flaws: (1) insufficient temporal coverage resulting from the sparse, uniformly spaced frame sampling used by most leading VideoLLMs, (2) spatial information loss introduced by aggressive token downsampling within sampled frames, and (3) encoder-decoder disconnection, whereby visual cues are only weakly utilized during text generation. Leveraging these insights, we craft three zero-query black-box attacks, aligning with these flaws in the processing pipeline. Our large-scale evaluation across five leading VideoLLMs shows that the harmfulness omission rate exceeds 90% in most cases. Even when harmful content is clearly present in all frames, these models consistently fail to identify it. These results underscore a fundamental vulnerability in current VideoLLMs' designs and highlight the urgent need for sampling strategies, token compression, and decoding mechanisms that guarantee semantic coverage rather than speed alone.



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



## **43. Chain-of-Lure: A Universal Jailbreak Attack Framework using Unconstrained Synthetic Narratives**

cs.CR

23 pages, 3 main figures

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2505.17519v2) [paper-pdf](https://arxiv.org/pdf/2505.17519v2)

**Authors**: Wenhan Chang, Tianqing Zhu, Yu Zhao, Shuangyong Song, Ping Xiong, Wanlei Zhou

**Abstract**: In the era of rapid generative AI development, interactions with large language models (LLMs) pose increasing risks of misuse. Prior research has primarily focused on attacks using template-based prompts and optimization-oriented methods, while overlooking the fact that LLMs possess strong unconstrained deceptive capabilities to attack other LLMs. This paper introduces a novel jailbreaking method inspired by the Chain-of-Thought mechanism. The attacker employs mission transfer to conceal harmful user intent within dialogue and generates a progressive chain of lure questions without relying on predefined templates, enabling successful jailbreaks. To further improve the attack's strength, we incorporate a helper LLM model that performs randomized narrative optimization over multi-turn interactions, enhancing the attack performance while preserving alignment with the original intent. We also propose a toxicity-based framework using third-party LLMs to evaluate harmful content and its alignment with malicious intent. Extensive experiments demonstrate that our method consistently achieves high attack success rates and elevated toxicity scores across diverse types of LLMs under black-box API settings. These findings reveal the intrinsic potential of LLMs to perform unrestricted attacks in the absence of robust alignment constraints. Our approach offers data-driven insights to inform the design of future alignment mechanisms. Finally, we propose two concrete defense strategies to support the development of safer generative models.



## **44. Chain-of-Thought Driven Adversarial Scenario Extrapolation for Robust Language Models**

cs.CL

19 pages, 5 figures. Accepted in AAAI 2026

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2505.17089v2) [paper-pdf](https://arxiv.org/pdf/2505.17089v2)

**Authors**: Md Rafi Ur Rashid, Vishnu Asutosh Dasu, Ye Wang, Gang Tan, Shagufta Mehnaz

**Abstract**: Large Language Models (LLMs) exhibit impressive capabilities, but remain susceptible to a growing spectrum of safety risks, including jailbreaks, toxic content, hallucinations, and bias. Existing defenses often address only a single threat type or resort to rigid outright rejection, sacrificing user experience and failing to generalize across diverse and novel attacks. This paper introduces Adversarial Scenario Extrapolation (ASE), a novel inference-time computation framework that leverages Chain-of-Thought (CoT) reasoning to simultaneously enhance LLM robustness and seamlessness. ASE guides the LLM through a self-generative process of contemplating potential adversarial scenarios and formulating defensive strategies before generating a response to the user query. Comprehensive evaluation on four adversarial benchmarks with four latest LLMs shows that ASE achieves near-zero jailbreak attack success rates and minimal toxicity, while slashing outright rejections to <4%. ASE outperforms six state-of-the-art defenses in robustness-seamlessness trade-offs, with 92-99% accuracy on adversarial Q&A and 4-10x lower bias scores. By transforming adversarial perception into an intrinsic cognitive process, ASE sets a new paradigm for secure and natural human-AI interaction.



## **45. Use as Many Surrogates as You Want: Selective Ensemble Attack to Unleash Transferability without Sacrificing Resource Efficiency**

cs.CV

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2505.12644v2) [paper-pdf](https://arxiv.org/pdf/2505.12644v2)

**Authors**: Bo Yang, Hengwei Zhang, Jindong Wang, Yuchen Ren, Chenhao Lin, Chao Shen, Zhengyu Zhao

**Abstract**: In surrogate ensemble attacks, using more surrogate models yields higher transferability but lower resource efficiency. This practical trade-off between transferability and efficiency has largely limited existing attacks despite many pre-trained models are easily accessible online. In this paper, we argue that such a trade-off is caused by an unnecessary common assumption, i.e., all models should be \textit{identical} across iterations. By lifting this assumption, we can use as many surrogates as we want to unleash transferability without sacrificing efficiency. Concretely, we propose Selective Ensemble Attack (SEA), which dynamically selects diverse models (from easily accessible pre-trained models) across iterations based on our new interpretation of decoupling within-iteration and cross-iteration model diversity. In this way, the number of within-iteration models is fixed for maintaining efficiency, while only cross-iteration model diversity is increased for higher transferability. Experiments on ImageNet demonstrate the superiority of SEA in various scenarios. For example, when dynamically selecting 4 from 20 accessible models, SEA yields 8.5% higher transferability than existing attacks under the same efficiency. The superiority of SEA also generalizes to real-world systems, such as commercial vision APIs and large vision-language models. Overall, SEA opens up the possibility of adaptively balancing transferability and efficiency according to specific resource requirements.



## **46. FaceShield: Explainable Face Anti-Spoofing with Multimodal Large Language Models**

cs.CV

Accepted by AAAI 2025. Hongyang Wang and Yichen Shi contribute equally. Corresponding author: Zitong Yu

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2505.09415v2) [paper-pdf](https://arxiv.org/pdf/2505.09415v2)

**Authors**: Hongyang Wang, Yichen Shi, Zhuofu Tao, Yuhao Gao, Liepiao Zhang, Xun Lin, Jun Feng, Xiaochen Yuan, Zitong Yu, Xiaochun Cao

**Abstract**: Face anti-spoofing (FAS) is crucial for protecting facial recognition systems from presentation attacks. Previous methods approached this task as a classification problem, lacking interpretability and reasoning behind the predicted results. Recently, multimodal large language models (MLLMs) have shown strong capabilities in perception, reasoning, and decision-making in visual tasks. However, there is currently no universal and comprehensive MLLM and dataset specifically designed for FAS task. To address this gap, we propose FaceShield, a MLLM for FAS, along with the corresponding pre-training and supervised fine-tuning (SFT) datasets, FaceShield-pre10K and FaceShield-sft45K. FaceShield is capable of determining the authenticity of faces, identifying types of spoofing attacks, providing reasoning for its judgments, and detecting attack areas. Specifically, we employ spoof-aware vision perception (SAVP) that incorporates both the original image and auxiliary information based on prior knowledge. We then use an prompt-guided vision token masking (PVTM) strategy to random mask vision tokens, thereby improving the model's generalization ability. We conducted extensive experiments on three benchmark datasets, demonstrating that FaceShield significantly outperforms previous deep learning models and general MLLMs on four FAS tasks, i.e., coarse-grained classification, fine-grained classification, reasoning, and attack localization. Our instruction datasets, protocols, and codes will be released at https://github.com/Why0912/FaceShield.



## **47. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

cs.CY

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2502.12659v4) [paper-pdf](https://arxiv.org/pdf/2502.12659v4)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models (LRMs), such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source reasoning models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on open LRMs is needed. (2) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (3) Safety thinking emerges in the reasoning process of LRMs, but fails frequently against adversarial attacks. (4) The thinking process in R1 models poses greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.



## **48. Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation**

cs.CR

14 pages, 5 figures; published in EMNLP 2025 ; Code at: https://github.com/dsbuddy/GAP-LLM-Safety

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2501.18638v3) [paper-pdf](https://arxiv.org/pdf/2501.18638v3)

**Authors**: Daniel Schwartz, Dmitriy Bespalov, Zhe Wang, Ninad Kulkarni, Yanjun Qi

**Abstract**: As large language models (LLMs) become increasingly prevalent, ensuring their robustness against adversarial misuse is crucial. This paper introduces the GAP (Graph of Attacks with Pruning) framework, an advanced approach for generating stealthy jailbreak prompts to evaluate and enhance LLM safeguards. GAP addresses limitations in existing tree-based LLM jailbreak methods by implementing an interconnected graph structure that enables knowledge sharing across attack paths. Our experimental evaluation demonstrates GAP's superiority over existing techniques, achieving a 20.8% increase in attack success rates while reducing query costs by 62.7%. GAP consistently outperforms state-of-the-art methods for attacking both open and closed LLMs, with attack success rates of >96%. Additionally, we present specialized variants like GAP-Auto for automated seed generation and GAP-VLM for multimodal attacks. GAP-generated prompts prove highly effective in improving content moderation systems, increasing true positive detection rates by 108.5% and accuracy by 183.6% when used for fine-tuning. Our implementation is available at https://github.com/dsbuddy/GAP-LLM-Safety.



## **49. Transferability of Adversarial Attacks in Video-based MLLMs: A Cross-modal Image-to-Video Approach**

cs.CV

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2501.01042v3) [paper-pdf](https://arxiv.org/pdf/2501.01042v3)

**Authors**: Linhao Huang, Xue Jiang, Zhiqiang Wang, Wentao Mo, Xi Xiao, Bo Han, Yongjie Yin, Feng Zheng

**Abstract**: Video-based multimodal large language models (V-MLLMs) have shown vulnerability to adversarial examples in video-text multimodal tasks. However, the transferability of adversarial videos to unseen models - a common and practical real-world scenario - remains unexplored. In this paper, we pioneer an investigation into the transferability of adversarial video samples across V-MLLMs. We find that existing adversarial attack methods face significant limitations when applied in black-box settings for V-MLLMs, which we attribute to the following shortcomings: (1) lacking generalization in perturbing video features, (2) focusing only on sparse key-frames, and (3) failing to integrate multimodal information. To address these limitations and deepen the understanding of V-MLLM vulnerabilities in black-box scenarios, we introduce the Image-to-Video MLLM (I2V-MLLM) attack. In I2V-MLLM, we utilize an image-based multimodal large language model (I-MLLM) as a surrogate model to craft adversarial video samples. Multimodal interactions and spatiotemporal information are integrated to disrupt video representations within the latent space, improving adversarial transferability. Additionally, a perturbation propagation technique is introduced to handle different unknown frame sampling strategies. Experimental results demonstrate that our method can generate adversarial examples that exhibit strong transferability across different V-MLLMs on multiple video-text multimodal tasks. Compared to white-box attacks on these models, our black-box attacks (using BLIP-2 as a surrogate model) achieve competitive performance, with average attack success rate (AASR) of 57.98% on MSVD-QA and 58.26% on MSRVTT-QA for Zero-Shot VideoQA tasks, respectively.



## **50. What You See Is Not Always What You Get: Evaluating GPT's Comprehension of Source Code**

cs.SE

This work has been accepted at APSEC 2025

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2412.08098v3) [paper-pdf](https://arxiv.org/pdf/2412.08098v3)

**Authors**: Jiawen Wen, Bangshuo Zhu, Huaming Chen

**Abstract**: Recent studies have demonstrated outstanding capabilities of large language models (LLMs) in software engineering tasks, including code generation and comprehension. While LLMs have shown significant potential in assisting with coding, LLMs are vulnerable to adversarial attacks. In this paper, we investigate the vulnerability of LLMs to imperceptible attacks. This class of attacks manipulate source code at the character level, which renders the changes invisible to human reviewers yet effective in misleading LLMs' behaviour. We devise these attacks into four distinct categories and analyse their impacts on code analysis and comprehension tasks. These four types of imperceptible character attacks include coding reordering, invisible coding characters, code deletions, and code homoglyphs. To assess the robustness of state-of-the-art LLMs, we present a systematic evaluation across multiple models using both perturbed and clean code snippets. Two evaluation metrics, model confidence using log probabilities of response and response correctness, are introduced. The results reveal that LLMs are susceptible to imperceptible coding perturbations, with varying degrees of degradation highlighted across different LLMs. Furthermore, we observe a consistent negative correlation between perturbation magnitude and model performance. These results highlight the urgent need for robust LLMs capable of manoeuvring behaviours under imperceptible adversarial conditions.



