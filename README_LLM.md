# Latest Large Language Model Attack Papers
**update at 2025-12-23 09:35:58**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Adversarially Robust Detection of Harmful Online Content: A Computational Design Science Approach**

cs.LG

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17367v1) [paper-pdf](https://arxiv.org/pdf/2512.17367v1)

**Authors**: Yidong Chai, Yi Liu, Mohammadreza Ebrahimi, Weifeng Li, Balaji Padmanabhan

**Abstract**: Social media platforms are plagued by harmful content such as hate speech, misinformation, and extremist rhetoric. Machine learning (ML) models are widely adopted to detect such content; however, they remain highly vulnerable to adversarial attacks, wherein malicious users subtly modify text to evade detection. Enhancing adversarial robustness is therefore essential, requiring detectors that can defend against diverse attacks (generalizability) while maintaining high overall accuracy. However, simultaneously achieving both optimal generalizability and accuracy is challenging. Following the computational design science paradigm, this study takes a sequential approach that first proposes a novel framework (Large Language Model-based Sample Generation and Aggregation, LLM-SGA) by identifying the key invariances of textual adversarial attacks and leveraging them to ensure that a detector instantiated within the framework has strong generalizability. Second, we instantiate our detector (Adversarially Robust Harmful Online Content Detector, ARHOCD) with three novel design components to improve detection accuracy: (1) an ensemble of multiple base detectors that exploits their complementary strengths; (2) a novel weight assignment method that dynamically adjusts weights based on each sample's predictability and each base detector's capability, with weights initialized using domain knowledge and updated via Bayesian inference; and (3) a novel adversarial training strategy that iteratively optimizes both the base detectors and the weight assignor. We addressed several limitations of existing adversarial robustness enhancement research and empirically evaluated ARHOCD across three datasets spanning hate speech, rumor, and extremist content. Results show that ARHOCD offers strong generalizability and improves detection accuracy under adversarial conditions.



## **2. Cryptanalysis of Pseudorandom Error-Correcting Codes**

cs.CR

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17310v1) [paper-pdf](https://arxiv.org/pdf/2512.17310v1)

**Authors**: Tianrui Wang, Anyu Wang, Tianshuo Cong, Delong Ran, Jinyuan Liu, Xiaoyun Wang

**Abstract**: Pseudorandom error-correcting codes (PRC) is a novel cryptographic primitive proposed at CRYPTO 2024. Due to the dual capability of pseudorandomness and error correction, PRC has been recognized as a promising foundational component for watermarking AI-generated content. However, the security of PRC has not been thoroughly analyzed, especially with concrete parameters or even in the face of cryptographic attacks. To fill this gap, we present the first cryptanalysis of PRC. We first propose three attacks to challenge the undetectability and robustness assumptions of PRC. Among them, two attacks aim to distinguish PRC-based codewords from plain vectors, and one attack aims to compromise the decoding process of PRC. Our attacks successfully undermine the claimed security guarantees across all parameter configurations. Notably, our attack can detect the presence of a watermark with overwhelming probability at a cost of $2^{22}$ operations. We also validate our approach by attacking real-world large generative models such as DeepSeek and Stable Diffusion. To mitigate our attacks, we further propose three defenses to enhance the security of PRC, including parameter suggestions, implementation suggestions, and constructing a revised key generation algorithm. Our proposed revised key generation function effectively prevents the occurrence of weak keys. However, we highlight that the current PRC-based watermarking scheme still cannot achieve a 128-bit security under our parameter suggestions due to the inherent configurations of large generative models, such as the maximum output length of large language models.



## **3. Biosecurity-Aware AI: Agentic Risk Auditing of Soft Prompt Attacks on ESM-Based Variant Predictors**

cs.CR

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17146v1) [paper-pdf](https://arxiv.org/pdf/2512.17146v1)

**Authors**: Huixin Zhan

**Abstract**: Genomic Foundation Models (GFMs), such as Evolutionary Scale Modeling (ESM), have demonstrated remarkable success in variant effect prediction. However, their security and robustness under adversarial manipulation remain largely unexplored. To address this gap, we introduce the Secure Agentic Genomic Evaluator (SAGE), an agentic framework for auditing the adversarial vulnerabilities of GFMs. SAGE functions through an interpretable and automated risk auditing loop. It injects soft prompt perturbations, monitors model behavior across training checkpoints, computes risk metrics such as AUROC and AUPR, and generates structured reports with large language model-based narrative explanations. This agentic process enables continuous evaluation of embedding-space robustness without modifying the underlying model. Using SAGE, we find that even state-of-the-art GFMs like ESM2 are sensitive to targeted soft prompt attacks, resulting in measurable performance degradation. These findings reveal critical and previously hidden vulnerabilities in genomic foundation models, showing the importance of agentic risk auditing in securing biomedical applications such as clinical variant interpretation.



## **4. From Essence to Defense: Adaptive Semantic-aware Watermarking for Embedding-as-a-Service Copyright Protection**

cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16439v1) [paper-pdf](https://arxiv.org/pdf/2512.16439v1)

**Authors**: Hao Li, Yubing Ren, Yanan Cao, Yingjie Li, Fang Fang, Xuebin Wang

**Abstract**: Benefiting from the superior capabilities of large language models in natural language understanding and generation, Embeddings-as-a-Service (EaaS) has emerged as a successful commercial paradigm on the web platform. However, prior studies have revealed that EaaS is vulnerable to imitation attacks. Existing methods protect the intellectual property of EaaS through watermarking techniques, but they all ignore the most important properties of embedding: semantics, resulting in limited harmlessness and stealthiness. To this end, we propose SemMark, a novel semantic-based watermarking paradigm for EaaS copyright protection. SemMark employs locality-sensitive hashing to partition the semantic space and inject semantic-aware watermarks into specific regions, ensuring that the watermark signals remain imperceptible and diverse. In addition, we introduce the adaptive watermark weight mechanism based on the local outlier factor to preserve the original embedding distribution. Furthermore, we propose Detect-Sampling and Dimensionality-Reduction attacks and construct four scenarios to evaluate the watermarking method. Extensive experiments are conducted on four popular NLP datasets, and SemMark achieves superior verifiability, diversity, stealthiness, and harmlessness.



## **5. MemoryGraft: Persistent Compromise of LLM Agents via Poisoned Experience Retrieval**

cs.CR

14 pages, 1 figure, includes appendix

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16962v1) [paper-pdf](https://arxiv.org/pdf/2512.16962v1)

**Authors**: Saksham Sahai Srivastava, Haoyu He

**Abstract**: Large Language Model (LLM) agents increasingly rely on long-term memory and Retrieval-Augmented Generation (RAG) to persist experiences and refine future performance. While this experience learning capability enhances agentic autonomy, it introduces a critical, unexplored attack surface, i.e., the trust boundary between an agent's reasoning core and its own past. In this paper, we introduce MemoryGraft. It is a novel indirect injection attack that compromises agent behavior not through immediate jailbreaks, but by implanting malicious successful experiences into the agent's long-term memory. Unlike traditional prompt injections that are transient, or standard RAG poisoning that targets factual knowledge, MemoryGraft exploits the agent's semantic imitation heuristic which is the tendency to replicate patterns from retrieved successful tasks. We demonstrate that an attacker who can supply benign ingestion-level artifacts that the agent reads during execution can induce it to construct a poisoned RAG store where a small set of malicious procedure templates is persisted alongside benign experiences. When the agent later encounters semantically similar tasks, union retrieval over lexical and embedding similarity reliably surfaces these grafted memories, and the agent adopts the embedded unsafe patterns, leading to persistent behavioral drift across sessions. We validate MemoryGraft on MetaGPT's DataInterpreter agent with GPT-4o and find that a small number of poisoned records can account for a large fraction of retrieved experiences on benign workloads, turning experience-based self-improvement into a vector for stealthy and durable compromise. To facilitate reproducibility and future research, our code and evaluation data are available at https://github.com/Jacobhhy/Agent-Memory-Poisoning.



## **6. In-Context Probing for Membership Inference in Fine-Tuned Language Models**

cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16292v1) [paper-pdf](https://arxiv.org/pdf/2512.16292v1)

**Authors**: Zhexi Lu, Hongliang Chi, Nathalie Baracaldo, Swanand Ravindra Kadhe, Yuseok Jeon, Lei Yu

**Abstract**: Membership inference attacks (MIAs) pose a critical privacy threat to fine-tuned large language models (LLMs), especially when models are adapted to domain-specific tasks using sensitive data. While prior black-box MIA techniques rely on confidence scores or token likelihoods, these signals are often entangled with a sample's intrinsic properties - such as content difficulty or rarity - leading to poor generalization and low signal-to-noise ratios. In this paper, we propose ICP-MIA, a novel MIA framework grounded in the theory of training dynamics, particularly the phenomenon of diminishing returns during optimization. We introduce the Optimization Gap as a fundamental signal of membership: at convergence, member samples exhibit minimal remaining loss-reduction potential, while non-members retain significant potential for further optimization. To estimate this gap in a black-box setting, we propose In-Context Probing (ICP), a training-free method that simulates fine-tuning-like behavior via strategically constructed input contexts. We propose two probing strategies: reference-data-based (using semantically similar public samples) and self-perturbation (via masking or generation). Experiments on three tasks and multiple LLMs show that ICP-MIA significantly outperforms prior black-box MIAs, particularly at low false positive rates. We further analyze how reference data alignment, model type, PEFT configurations, and training schedules affect attack effectiveness. Our findings establish ICP-MIA as a practical and theoretically grounded framework for auditing privacy risks in deployed LLMs.



## **7. DualGuard: Dual-stream Large Language Model Watermarking Defense against Paraphrase and Spoofing Attack**

cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16182v1) [paper-pdf](https://arxiv.org/pdf/2512.16182v1)

**Authors**: Hao Li, Yubing Ren, Yanan Cao, Yingjie Li, Fang Fang, Shi Wang, Li Guo

**Abstract**: With the rapid development of cloud-based services, large language models (LLMs) have become increasingly accessible through various web platforms. However, this accessibility has also led to growing risks of model abuse. LLM watermarking has emerged as an effective approach to mitigate such misuse and protect intellectual property. Existing watermarking algorithms, however, primarily focus on defending against paraphrase attacks while overlooking piggyback spoofing attacks, which can inject harmful content, compromise watermark reliability, and undermine trust in attribution. To address this limitation, we propose DualGuard, the first watermarking algorithm capable of defending against both paraphrase and spoofing attacks. DualGuard employs the adaptive dual-stream watermarking mechanism, in which two complementary watermark signals are dynamically injected based on the semantic content. This design enables DualGuard not only to detect but also to trace spoofing attacks, thereby ensuring reliable and trustworthy watermark detection. Extensive experiments conducted across multiple datasets and language models demonstrate that DualGuard achieves excellent detectability, robustness, traceability, and text quality, effectively advancing the state of LLM watermarking for real-world applications.



## **8. Bounty Hunter: Autonomous, Comprehensive Emulation of Multi-Faceted Adversaries**

cs.CR

15 pages, 9 figures

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15275v1) [paper-pdf](https://arxiv.org/pdf/2512.15275v1)

**Authors**: Louis Hackländer-Jansen, Rafael Uetz, Martin Henze

**Abstract**: Adversary emulation is an essential procedure for cybersecurity assessments such as evaluating an organization's security posture or facilitating structured training and research in dedicated environments. To allow for systematic and time-efficient assessments, several approaches from academia and industry have worked towards the automation of adversarial actions. However, they exhibit significant limitations regarding autonomy, tactics coverage, and real-world applicability. Consequently, adversary emulation remains a predominantly manual task requiring substantial human effort and security expertise - even amidst the rise of Large Language Models. In this paper, we present Bounty Hunter, an automated adversary emulation method, designed and implemented as an open-source plugin for the popular adversary emulation platform Caldera, that enables autonomous emulation of adversaries with multi-faceted behavior while providing a wide coverage of tactics. To this end, it realizes diverse adversarial behavior, such as different levels of detectability and varying attack paths across repeated emulations. By autonomously compromising a simulated enterprise network, Bounty Hunter showcases its ability to achieve given objectives without prior knowledge of its target, including pre-compromise, initial compromise, and post-compromise attack tactics. Overall, Bounty Hunter facilitates autonomous, comprehensive, and multi-faceted adversary emulation to help researchers and practitioners in performing realistic and time-efficient security assessments, training exercises, and intrusion detection research.



## **9. MCP-SafetyBench: A Benchmark for Safety Evaluation of Large Language Models with Real-World MCP Servers**

cs.CL

Our benchmark is available at https://github.com/xjzzzzzzzz/MCPSafety

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15163v1) [paper-pdf](https://arxiv.org/pdf/2512.15163v1)

**Authors**: Xuanjun Zong, Zhiqi Shen, Lei Wang, Yunshi Lan, Chao Yang

**Abstract**: Large language models (LLMs) are evolving into agentic systems that reason, plan, and operate external tools. The Model Context Protocol (MCP) is a key enabler of this transition, offering a standardized interface for connecting LLMs with heterogeneous tools and services. Yet MCP's openness and multi-server workflows introduce new safety risks that existing benchmarks fail to capture, as they focus on isolated attacks or lack real-world coverage. We present MCP-SafetyBench, a comprehensive benchmark built on real MCP servers that supports realistic multi-turn evaluation across five domains: browser automation, financial analysis, location navigation, repository management, and web search. It incorporates a unified taxonomy of 20 MCP attack types spanning server, host, and user sides, and includes tasks requiring multi-step reasoning and cross-server coordination under uncertainty. Using MCP-SafetyBench, we systematically evaluate leading open- and closed-source LLMs, revealing large disparities in safety performance and escalating vulnerabilities as task horizons and server interactions grow. Our results highlight the urgent need for stronger defenses and establish MCP-SafetyBench as a foundation for diagnosing and mitigating safety risks in real-world MCP deployments.



## **10. Quantifying Return on Security Controls in LLM Systems**

cs.CR

13 pages, 9 figures, 3 tables

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15081v1) [paper-pdf](https://arxiv.org/pdf/2512.15081v1)

**Authors**: Richard Helder Moulton, Austin O'Brien, John D. Hastings

**Abstract**: Although large language models (LLMs) are increasingly used in security-critical workflows, practitioners lack quantitative guidance on which safeguards are worth deploying. This paper introduces a decision-oriented framework and reproducible methodology that together quantify residual risk, convert adversarial probe outcomes into financial risk estimates and return-on-control (RoC) metrics, and enable monetary comparison of layered defenses for LLM-based systems. A retrieval-augmented generation (RAG) service is instantiated using the DeepSeek-R1 model over a corpus containing synthetic personally identifiable information (PII), and subjected to automated attacks with Garak across five vulnerability classes: PII leakage, latent context injection, prompt injection, adversarial attack generation, and divergence. For each (vulnerability, control) pair, attack success probabilities are estimated via Laplace's Rule of Succession and combined with loss triangle distributions, calibrated from public breach-cost data, in 10,000-run Monte Carlo simulations to produce loss exceedance curves and expected losses. Three widely used mitigations, attribute-based access control (ABAC); named entity recognition (NER) redaction using Microsoft Presidio; and NeMo Guardrails, are then compared to a baseline RAG configuration. The baseline system exhibits very high attack success rates (>= 0.98 for PII, latent injection, and prompt injection), yielding a total simulated expected loss of $313k per attack scenario. ABAC collapses success probabilities for PII and prompt-related attacks to near zero and reduces the total expected loss by ~94%, achieving an RoC of 9.83. NER redaction likewise eliminates PII leakage and attains an RoC of 5.97, while NeMo Guardrails provides only marginal benefit (RoC of 0.05).



## **11. MALCDF: A Distributed Multi-Agent LLM Framework for Real-Time Cyber**

cs.CR

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14846v1) [paper-pdf](https://arxiv.org/pdf/2512.14846v1)

**Authors**: Arth Bhardwaj, Sia Godika, Yuvam Loonker

**Abstract**: Traditional, centralized security tools often miss adaptive, multi-vector attacks. We present the Multi-Agent LLM Cyber Defense Framework (MALCDF), a practical setup where four large language model (LLM) agents-Detection, Intelligence, Response, and Analysis-work together in real time. Agents communicate over a Secure Communication Layer (SCL) with encrypted, ontology-aligned messages, and produce audit-friendly outputs (e.g., MITRE ATT&CK mappings).   For evaluation, we keep the test simple and consistent: all reported metrics come from the same 50-record live stream derived from the CICIDS2017 feature schema. CICIDS2017 is used for configuration (fields/schema) and to train a practical ML baseline. The ML-IDS baseline is a Lightweight Random Forest IDS (LRF-IDS) trained on a subset of CICIDS2017 and tested on the 50-record stream, with no overlap between training and test records.   In experiments, MALCDF reaches 90.0% detection accuracy, 85.7% F1-score, and 9.1% false-positive rate, with 6.8s average per-event latency. It outperforms the lightweight ML-IDS baseline and a single-LLM setup on accuracy while keeping end-to-end outputs consistent. Overall, this hands-on build suggests that coordinating simple LLM agents with secure, ontology-aligned messaging can improve practical, real-time cyber defense.



## **12. PerProb: Indirectly Evaluating Memorization in Large Language Models**

cs.CR

Accepted at APSEC 2025

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14600v1) [paper-pdf](https://arxiv.org/pdf/2512.14600v1)

**Authors**: Yihan Liao, Jacky Keung, Xiaoxue Ma, Jingyu Zhang, Yicheng Sun

**Abstract**: The rapid advancement of Large Language Models (LLMs) has been driven by extensive datasets that may contain sensitive information, raising serious privacy concerns. One notable threat is the Membership Inference Attack (MIA), where adversaries infer whether a specific sample was used in model training. However, the true impact of MIA on LLMs remains unclear due to inconsistent findings and the lack of standardized evaluation methods, further complicated by the undisclosed nature of many LLM training sets. To address these limitations, we propose PerProb, a unified, label-free framework for indirectly assessing LLM memorization vulnerabilities. PerProb evaluates changes in perplexity and average log probability between data generated by victim and adversary models, enabling an indirect estimation of training-induced memory. Compared with prior MIA methods that rely on member/non-member labels or internal access, PerProb is independent of model and task, and applicable in both black-box and white-box settings. Through a systematic classification of MIA into four attack patterns, we evaluate PerProb's effectiveness across five datasets, revealing varying memory behaviors and privacy risks among LLMs. Additionally, we assess mitigation strategies, including knowledge distillation, early stopping, and differential privacy, demonstrating their effectiveness in reducing data leakage. Our findings offer a practical and generalizable framework for evaluating and improving LLM privacy.



## **13. Reasoning-Style Poisoning of LLM Agents via Stealthy Style Transfer: Process-Level Attacks and Runtime Monitoring in RSV Space**

cs.CR

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14448v1) [paper-pdf](https://arxiv.org/pdf/2512.14448v1)

**Authors**: Xingfu Zhou, Pengfei Wang

**Abstract**: Large Language Model (LLM) agents relying on external retrieval are increasingly deployed in high-stakes environments. While existing adversarial attacks primarily focus on content falsification or instruction injection, we identify a novel, process-oriented attack surface: the agent's reasoning style. We propose Reasoning-Style Poisoning (RSP), a paradigm that manipulates how agents process information rather than what they process. We introduce Generative Style Injection (GSI), an attack method that rewrites retrieved documents into pathological tones--specifically "analysis paralysis" or "cognitive haste"--without altering underlying facts or using explicit triggers. To quantify these shifts, we develop the Reasoning Style Vector (RSV), a metric tracking Verification depth, Self-confidence, and Attention focus. Experiments on HotpotQA and FEVER using ReAct, Reflection, and Tree of Thoughts (ToT) architectures reveal that GSI significantly degrades performance. It increases reasoning steps by up to 4.4 times or induces premature errors, successfully bypassing state-of-the-art content filters. Finally, we propose RSP-M, a lightweight runtime monitor that calculates RSV metrics in real-time and triggers alerts when values exceed safety thresholds. Our work demonstrates that reasoning style is a distinct, exploitable vulnerability, necessitating process-level defenses beyond static content analysis.



## **14. Semantic Mismatch and Perceptual Degradation: A New Perspective on Image Editing Immunity**

cs.CV

11 pages, 4 figures

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14320v1) [paper-pdf](https://arxiv.org/pdf/2512.14320v1)

**Authors**: Shuai Dong, Jie Zhang, Guoying Zhao, Shiguang Shan, Xilin Chen

**Abstract**: Text-guided image editing via diffusion models, while powerful, raises significant concerns about misuse, motivating efforts to immunize images against unauthorized edits using imperceptible perturbations. Prevailing metrics for evaluating immunization success typically rely on measuring the visual dissimilarity between the output generated from a protected image and a reference output generated from the unprotected original. This approach fundamentally overlooks the core requirement of image immunization, which is to disrupt semantic alignment with attacker intent, regardless of deviation from any specific output. We argue that immunization success should instead be defined by the edited output either semantically mismatching the prompt or suffering substantial perceptual degradations, both of which thwart malicious intent. To operationalize this principle, we propose Synergistic Intermediate Feature Manipulation (SIFM), a method that strategically perturbs intermediate diffusion features through dual synergistic objectives: (1) maximizing feature divergence from the original edit trajectory to disrupt semantic alignment with the expected edit, and (2) minimizing feature norms to induce perceptual degradations. Furthermore, we introduce the Immunization Success Rate (ISR), a novel metric designed to rigorously quantify true immunization efficacy for the first time. ISR quantifies the proportion of edits where immunization induces either semantic failure relative to the prompt or significant perceptual degradations, assessed via Multimodal Large Language Models (MLLMs). Extensive experiments show our SIFM achieves the state-of-the-art performance for safeguarding visual content against malicious diffusion-based manipulation.



## **15. PentestEval: Benchmarking LLM-based Penetration Testing with Modular and Stage-Level Design**

cs.SE

13 pages, 6 figures

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14233v1) [paper-pdf](https://arxiv.org/pdf/2512.14233v1)

**Authors**: Ruozhao Yang, Mingfei Cheng, Gelei Deng, Tianwei Zhang, Junjie Wang, Xiaofei Xie

**Abstract**: Penetration testing is essential for assessing and strengthening system security against real-world threats, yet traditional workflows remain highly manual, expertise-intensive, and difficult to scale. Although recent advances in Large Language Models (LLMs) offer promising opportunities for automation, existing applications rely on simplistic prompting without task decomposition or domain adaptation, resulting in unreliable black-box behavior and limited insight into model capabilities across penetration testing stages. To address this gap, we introduce PentestEval, the first comprehensive benchmark for evaluating LLMs across six decomposed penetration testing stages: Information Collection, Weakness Gathering and Filtering, Attack Decision-Making, Exploit Generation and Revision. PentestEval integrates expert-annotated ground truth with a fully automated evaluation pipeline across 346 tasks covering all stages in 12 realistic vulnerable scenarios. Our stage-level evaluation of 9 widely used LLMs reveals generally weak performance and distinct limitations across the stages of penetration-testing workflow. End-to-end pipelines reach only 31% success rate, and existing LLM-powered systems such as PentestGPT, PentestAgent, and VulnBot exhibit similar limitations, with autonomous agents failing almost entirely. These findings highlight that autonomous penetration testing demands stronger structured reasoning, where modularization enhances each individual stage and improves overall performance. PentestEval provides the foundational benchmark needed for future research on fine-grained, stage-level evaluation, paving the way toward more reliable LLM-based automation.



## **16. IntentMiner: Intent Inversion Attack via Tool Call Analysis in the Model Context Protocol**

cs.CR

12 pages, 6 figures

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14166v1) [paper-pdf](https://arxiv.org/pdf/2512.14166v1)

**Authors**: Yunhao Yao, Zhiqiang Wang, Haoran Cheng, Yihang Cheng, Haohua Du, Xiang-Yang Li

**Abstract**: The rapid evolution of Large Language Models (LLMs) into autonomous agents has led to the adoption of the Model Context Protocol (MCP) as a standard for discovering and invoking external tools. While this architecture decouples the reasoning engine from tool execution to enhance scalability, it introduces a significant privacy surface: third-party MCP servers, acting as semi-honest intermediaries, can observe detailed tool interaction logs outside the user's trusted boundary. In this paper, we first identify and formalize a novel privacy threat termed Intent Inversion, where a semi-honest MCP server attempts to reconstruct the user's private underlying intent solely by analyzing legitimate tool calls. To systematically assess this vulnerability, we propose IntentMiner, a framework that leverages Hierarchical Information Isolation and Three-Dimensional Semantic Analysis, integrating tool purpose, call statements, and returned results, to accurately infer user intent at the step level. Extensive experiments demonstrate that IntentMiner achieves a high degree of semantic alignment (over 85%) with original user queries, significantly outperforming baseline approaches. These results highlight the inherent privacy risks in decoupled agent architectures, revealing that seemingly benign tool execution logs can serve as a potent vector for exposing user secrets.



## **17. From Obfuscated to Obvious: A Comprehensive JavaScript Deobfuscation Tool for Security Analysis**

cs.CR

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14070v1) [paper-pdf](https://arxiv.org/pdf/2512.14070v1)

**Authors**: Dongchao Zhou, Lingyun Ying, Huajun Chai, Dongbin Wang

**Abstract**: JavaScript's widespread adoption has made it an attractive target for malicious attackers who employ sophisticated obfuscation techniques to conceal harmful code. Current deobfuscation tools suffer from critical limitations that severely restrict their practical effectiveness. Existing tools struggle with diverse input formats, address only specific obfuscation types, and produce cryptic output that impedes human analysis.   To address these challenges, we present JSIMPLIFIER, a comprehensive deobfuscation tool using a multi-stage pipeline with preprocessing, abstract syntax tree-based static analysis, dynamic execution tracing, and Large Language Model (LLM)-enhanced identifier renaming. We also introduce multi-dimensional evaluation metrics that integrate control/data flow analysis, code simplification assessment, entropy measures and LLM-based readability assessments.   We construct and release the largest real-world obfuscated JavaScript dataset with 44,421 samples (23,212 wild malicious + 21,209 benign samples). Evaluation shows JSIMPLIFIER outperforms existing tools with 100% processing capability across 20 obfuscation techniques, 100% correctness on evaluation subsets, 88.2% code complexity reduction, and over 4-fold readability improvement validated by multiple LLMs. Our results advance benchmarks for JavaScript deobfuscation research and practical security applications.



## **18. Bilevel Optimization for Covert Memory Tampering in Heterogeneous Multi-Agent Architectures (XAMT)**

cs.CR

10 pages, 5 figures, 4 tables. Conference-style paper (IEEEtran). Proposes unified bilevel optimization framework for covert memory poisoning attacks in heterogeneous multi-agent systems (MARL + RAG)

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.15790v1) [paper-pdf](https://arxiv.org/pdf/2512.15790v1)

**Authors**: Akhil Sharma, Shaikh Yaser Arafat, Jai Kumar Sharma, Ken Huang

**Abstract**: The increasing operational reliance on complex Multi-Agent Systems (MAS) across safety-critical domains necessitates rigorous adversarial robustness assessment. Modern MAS are inherently heterogeneous, integrating conventional Multi-Agent Reinforcement Learning (MARL) with emerging Large Language Model (LLM) agent architectures utilizing Retrieval-Augmented Generation (RAG). A critical shared vulnerability is reliance on centralized memory components: the shared Experience Replay (ER) buffer in MARL and the external Knowledge Base (K) in RAG agents. This paper proposes XAMT (Bilevel Optimization for Covert Memory Tampering in Heterogeneous Multi-Agent Architectures), a novel framework that formalizes attack generation as a bilevel optimization problem. The Upper Level minimizes perturbation magnitude (delta) to enforce covertness while maximizing system behavior divergence toward an adversary-defined target (Lower Level). We provide rigorous mathematical instantiations for CTDE MARL algorithms and RAG-based LLM agents, demonstrating that bilevel optimization uniquely crafts stealthy, minimal-perturbation poisons evading detection heuristics. Comprehensive experimental protocols utilize SMAC and SafeRAG benchmarks to quantify effectiveness at sub-percent poison rates (less than or equal to 1 percent in MARL, less than or equal to 0.1 percent in RAG). XAMT defines a new unified class of training-time threats essential for developing intrinsically secure MAS, with implications for trust, formal verification, and defensive strategies prioritizing intrinsic safety over perimeter-based detection.



## **19. On the Effectiveness of Membership Inference in Targeted Data Extraction from Large Language Models**

cs.LG

Accepted to IEEE Conference on Secure and Trustworthy Machine Learning (SaTML) 2026

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13352v1) [paper-pdf](https://arxiv.org/pdf/2512.13352v1)

**Authors**: Ali Al Sahili, Ali Chehab, Razane Tajeddine

**Abstract**: Large Language Models (LLMs) are prone to memorizing training data, which poses serious privacy risks. Two of the most prominent concerns are training data extraction and Membership Inference Attacks (MIAs). Prior research has shown that these threats are interconnected: adversaries can extract training data from an LLM by querying the model to generate a large volume of text and subsequently applying MIAs to verify whether a particular data point was included in the training set. In this study, we integrate multiple MIA techniques into the data extraction pipeline to systematically benchmark their effectiveness. We then compare their performance in this integrated setting against results from conventional MIA benchmarks, allowing us to evaluate their practical utility in real-world extraction scenarios.



## **20. Cisco Integrated AI Security and Safety Framework Report**

cs.CR

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.12921v1) [paper-pdf](https://arxiv.org/pdf/2512.12921v1)

**Authors**: Amy Chang, Tiffany Saade, Sanket Mendapara, Adam Swanda, Ankit Garg

**Abstract**: Artificial intelligence (AI) systems are being readily and rapidly adopted, increasingly permeating critical domains: from consumer platforms and enterprise software to networked systems with embedded agents. While this has unlocked potential for human productivity gains, the attack surface has expanded accordingly: threats now span content safety failures (e.g., harmful or deceptive outputs), model and data integrity compromise (e.g., poisoning, supply-chain tampering), runtime manipulations (e.g., prompt injection, tool and agent misuse), and ecosystem risks (e.g., orchestration abuse, multi-agent collusion). Existing frameworks such as MITRE ATLAS, National Institute of Standards and Technology (NIST) AI 100-2 Adversarial Machine Learning (AML) taxonomy, and OWASP Top 10s for Large Language Models (LLMs) and Agentic AI Applications provide valuable viewpoints, but each covers only slices of this multi-dimensional space.   This paper presents Cisco's Integrated AI Security and Safety Framework ("AI Security Framework"), a unified, lifecycle-aware taxonomy and operationalization framework that can be used to classify, integrate, and operationalize the full range of AI risks. It integrates AI security and AI safety across modalities, agents, pipelines, and the broader ecosystem. The AI Security Framework is designed to be practical for threat identification, red-teaming, risk prioritization, and it is comprehensive in scope and can be extensible to emerging deployments in multimodal contexts, humanoids, wearables, and sensory infrastructures. We analyze gaps in prevailing frameworks, discuss design principles for our framework, and demonstrate how the taxonomy provides structure for understanding how modern AI systems fail, how adversaries exploit these failures, and how organizations can build defenses across the AI lifecycle that evolve alongside capability advancements.



## **21. CTIGuardian: A Few-Shot Framework for Mitigating Privacy Leakage in Fine-Tuned LLMs**

cs.CR

Accepted at the 18th Cybersecurity Experimentation and Test Workshop (CSET), in conjunction with ACSAC 2025

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.12914v1) [paper-pdf](https://arxiv.org/pdf/2512.12914v1)

**Authors**: Shashie Dilhara Batan Arachchige, Benjamin Zi Hao Zhao, Hassan Jameel Asghar, Dinusha Vatsalan, Dali Kaafar

**Abstract**: Large Language Models (LLMs) are often fine-tuned to adapt their general-purpose knowledge to specific tasks and domains such as cyber threat intelligence (CTI). Fine-tuning is mostly done through proprietary datasets that may contain sensitive information. Owners expect their fine-tuned model to not inadvertently leak this information to potentially adversarial end users. Using CTI as a use case, we demonstrate that data-extraction attacks can recover sensitive information from fine-tuned models on CTI reports, underscoring the need for mitigation. Retraining the full model to eliminate this leakage is computationally expensive and impractical. We propose an alternative approach, which we call privacy alignment, inspired by safety alignment in LLMs. Just like safety alignment teaches the model to abide by safety constraints through a few examples, we enforce privacy alignment through few-shot supervision, integrating a privacy classifier and a privacy redactor, both handled by the same underlying LLM. We evaluate our system, called CTIGuardian, using GPT-4o mini and Mistral-7B Instruct models, benchmarking against Presidio, a named entity recognition (NER) baseline. Results show that CTIGuardian provides a better privacy-utility trade-off than NER based models. While we demonstrate its effectiveness on a CTI use case, the framework is generic enough to be applicable to other sensitive domains.



## **22. Auto-Tuning Safety Guardrails for Black-Box Large Language Models**

cs.CR

8 pages, 7 figures, 1 table. Work completed as part of the M.S. in Artificial Intelligence at the University of St. Thomas using publicly available models and datasets; all views and any errors are the author's own

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.15782v1) [paper-pdf](https://arxiv.org/pdf/2512.15782v1)

**Authors**: Perry Abdulkadir

**Abstract**: Large language models (LLMs) are increasingly deployed behind safety guardrails such as system prompts and content filters, especially in settings where product teams cannot modify model weights. In practice these guardrails are typically hand-tuned, brittle, and difficult to reproduce. This paper studies a simple but practical alternative: treat safety guardrail design itself as a hyperparameter optimization problem over a frozen base model. Concretely, I wrap Mistral-7B-Instruct with modular jailbreak and malware system prompts plus a ModernBERT-based harmfulness classifier, then evaluate candidate configurations on three public benchmarks covering malware generation, classic jailbreak prompts, and benign user queries. Each configuration is scored using malware and jailbreak attack success rate, benign harmful-response rate, and end-to-end latency. A 48-point grid search over prompt combinations and filter modes establishes a baseline. I then run a black-box Optuna study over the same space and show that it reliably rediscovers the best grid configurations while requiring an order of magnitude fewer evaluations and roughly 8x less wall-clock time. The results suggest that viewing safety guardrails as tunable hyperparameters is a feasible way to harden black-box LLM deployments under compute and time constraints.



## **23. CODE ACROSTIC: Robust Watermarking for Code Generation**

cs.CR

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.14753v1) [paper-pdf](https://arxiv.org/pdf/2512.14753v1)

**Authors**: Li Lin, Siyuan Xin, Yang Cao, Xiaochun Cao

**Abstract**: Watermarking large language models (LLMs) is vital for preventing their misuse, including the fabrication of fake news, plagiarism, and spam. It is especially important to watermark LLM-generated code, as it often contains intellectual property.However, we found that existing methods for watermarking LLM-generated code fail to address comment removal attack.In such cases, an attacker can simply remove the comments from the generated code without affecting its functionality, significantly reducing the effectiveness of current code-watermarking techniques.On the other hand, injecting a watermark into code is challenging because, as previous works have noted, most code represents a low-entropy scenario compared to natural language. Our approach to addressing this issue involves leveraging prior knowledge to distinguish between low-entropy and high-entropy parts of the code, as indicated by a Cue List of words.We then inject the watermark guided by this Cue List, achieving higher detectability and usability than existing methods.We evaluated our proposed method on HumanEvaland compared our method with three state-of-the-art code watermarking techniques. The results demonstrate the effectiveness of our approach.



## **24. The Laminar Flow Hypothesis: Detecting Jailbreaks via Semantic Turbulence in Large Language Models**

cs.LG

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.13741v1) [paper-pdf](https://arxiv.org/pdf/2512.13741v1)

**Authors**: Md. Hasib Ur Rahman

**Abstract**: As Large Language Models (LLMs) become ubiquitous, the challenge of securing them against adversarial "jailbreaking" attacks has intensified. Current defense strategies often rely on computationally expensive external classifiers or brittle lexical filters, overlooking the intrinsic dynamics of the model's reasoning process. In this work, the Laminar Flow Hypothesis is introduced, which posits that benign inputs induce smooth, gradual transitions in an LLM's high-dimensional latent space, whereas adversarial prompts trigger chaotic, high-variance trajectories - termed Semantic Turbulence - resulting from the internal conflict between safety alignment and instruction-following objectives. This phenomenon is formalized through a novel, zero-shot metric: the variance of layer-wise cosine velocity. Experimental evaluation across diverse small language models reveals a striking diagnostic capability. The RLHF-aligned Qwen2-1.5B exhibits a statistically significant 75.4% increase in turbulence under attack (p less than 0.001), validating the hypothesis of internal conflict. Conversely, Gemma-2B displays a 22.0% decrease in turbulence, characterizing a distinct, low-entropy "reflex-based" refusal mechanism. These findings demonstrate that Semantic Turbulence serves not only as a lightweight, real-time jailbreak detector but also as a non-invasive diagnostic tool for categorizing the underlying safety architecture of black-box models.



## **25. One Leak Away: How Pretrained Model Exposure Amplifies Jailbreak Risks in Finetuned LLMs**

cs.CR

17 pages

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.14751v1) [paper-pdf](https://arxiv.org/pdf/2512.14751v1)

**Authors**: Yixin Tan, Zhe Yu, Jun Sakuma

**Abstract**: Finetuning pretrained large language models (LLMs) has become the standard paradigm for developing downstream applications. However, its security implications remain unclear, particularly regarding whether finetuned LLMs inherit jailbreak vulnerabilities from their pretrained sources. We investigate this question in a realistic pretrain-to-finetune threat model, where the attacker has white-box access to the pretrained LLM and only black-box access to its finetuned derivatives. Empirical analysis shows that adversarial prompts optimized on the pretrained model transfer most effectively to its finetuned variants, revealing inherited vulnerabilities from pretrained to finetuned LLMs. To further examine this inheritance, we conduct representation-level probing, which shows that transferable prompts are linearly separable within the pretrained hidden states, suggesting that universal transferability is encoded in pretrained representations. Building on this insight, we propose the Probe-Guided Projection (PGP) attack, which steers optimization toward transferability-relevant directions. Experiments across multiple LLM families and diverse finetuned tasks confirm PGP's strong transfer success, underscoring the security risks inherent in the pretrain-to-finetune paradigm.



## **26. The Role of AI in Modern Penetration Testing**

cs.SE

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12326v1) [paper-pdf](https://arxiv.org/pdf/2512.12326v1)

**Authors**: J. Alexander Curtis, Nasir U. Eisty

**Abstract**: Penetration testing is a cornerstone of cybersecurity, traditionally driven by manual, time-intensive processes. As systems grow in complexity, there is a pressing need for more scalable and efficient testing methodologies. This systematic literature review examines how Artificial Intelligence (AI) is reshaping penetration testing, analyzing 58 peer-reviewed studies from major academic databases. Our findings reveal that while AI-assisted pentesting is still in its early stages, notable progress is underway, particularly through Reinforcement Learning (RL), which was the focus of 77% of the reviewed works. Most research centers on the discovery and exploitation phases of pentesting, where AI shows the greatest promise in automating repetitive tasks, optimizing attack strategies, and improving vulnerability identification. Real-world applications remain limited but encouraging, including the European Space Agency's PenBox and various open-source tools. These demonstrate AI's potential to streamline attack path analysis, analyze complex network topology, and reduce manual workload. However, challenges persist: current models often lack flexibility and are underdeveloped for the reconnaissance and post-exploitation phases of pentesting. Applications involving Large Language Models (LLMs) remain relatively under-researched, pointing to a promising direction for future exploration. This paper offers a critical overview of AI's current and potential role in penetration testing, providing valuable insights for researchers, practitioners, and organizations aiming to enhance security assessments through advanced automation or looking for gaps in existing research.



## **27. Taint-Based Code Slicing for LLMs-based Malicious NPM Package Detection**

cs.CR

17 pages, 4 figures, 9 tables

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12313v1) [paper-pdf](https://arxiv.org/pdf/2512.12313v1)

**Authors**: Dang-Khoa Nguyen, Gia-Thang Ho, Quang-Minh Pham, Tuyet A. Dang-Thi, Minh-Khanh Vu, Thanh-Cong Nguyen, Phat T. Tran-Truong, Duc-Ly Vu

**Abstract**: The increasing sophistication of malware attacks in the npm ecosystem, characterized by obfuscation and complex logic, necessitates advanced detection methods. Recently, researchers have turned their attention from traditional detection approaches to Large Language Models (LLMs) due to their strong capabilities in semantic code understanding. However, while LLMs offer superior semantic reasoning for code analysis, their practical application is constrained by limited context windows and high computational cost. This paper addresses this challenge by introducing a novel framework that leverages code slicing techniques for an LLM-based malicious package detection task. We propose a specialized taintbased slicing technique for npm packages, augmented by a heuristic backtracking mechanism to accurately capture malicious data flows across asynchronous, event-driven patterns (e.g., callbacks and Promises) that elude traditional analysis. An evaluation on a dataset of more than 5000 malicious and benign npm packages demonstrates that our approach isolates security-relevant code, reducing input volume by over 99% while preserving critical behavioral semantics. Using the DeepSeek-Coder-6.7B model as the classification engine, our approach achieves a detection accuracy of 87.04%, substantially outperforming a naive token-splitting baseline (75.41%) and a traditional static-analysis-based approach. These results indicate that semantically optimized input representation via code slicing not only mitigates the LLM context-window bottleneck but also significantly enhances reasoning precision for security tasks, providing an efficient and effective defense against evolving malicious open-source packages.



## **28. Keep the Lights On, Keep the Lengths in Check: Plug-In Adversarial Detection for Time-Series LLMs in Energy Forecasting**

cs.CR

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12154v1) [paper-pdf](https://arxiv.org/pdf/2512.12154v1)

**Authors**: Hua Ma, Ruoxi Sun, Minhui Xue, Xingliang Yuan, Carsten Rudolph, Surya Nepal, Ling Liu

**Abstract**: Accurate time-series forecasting is increasingly critical for planning and operations in low-carbon power systems. Emerging time-series large language models (TS-LLMs) now deliver this capability at scale, requiring no task-specific retraining, and are quickly becoming essential components within the Internet-of-Energy (IoE) ecosystem. However, their real-world deployment is complicated by a critical vulnerability: adversarial examples (AEs). Detecting these AEs is challenging because (i) adversarial perturbations are optimized across the entire input sequence and exploit global temporal dependencies, which renders local detection methods ineffective, and (ii) unlike traditional forecasting models with fixed input dimensions, TS-LLMs accept sequences of variable length, increasing variability that complicates detection. To address these challenges, we propose a plug-in detection framework that capitalizes on the TS-LLM's own variable-length input capability. Our method uses sampling-induced divergence as a detection signal. Given an input sequence, we generate multiple shortened variants and detect AEs by measuring the consistency of their forecasts: Benign sequences tend to produce stable predictions under sampling, whereas adversarial sequences show low forecast similarity, because perturbations optimized for a full-length sequence do not transfer reliably to shorter, differently-structured subsamples. We evaluate our approach on three representative TS-LLMs (TimeGPT, TimesFM, and TimeLLM) across three energy datasets: ETTh2 (Electricity Transformer Temperature), NI (Hourly Energy Consumption), and Consumption (Hourly Electricity Consumption and Production). Empirical results confirm strong and robust detection performance across both black-box and white-box attack scenarios, highlighting its practicality as a reliable safeguard for TS-LLM forecasting in real-world energy systems.



## **29. BRIDG-ICS: AI-Grounded Knowledge Graphs for Intelligent Threat Analytics in Industry~5.0 Cyber-Physical Systems**

cs.CR

44 Pages, To be published in Springer Cybersecurity Journal

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12112v1) [paper-pdf](https://arxiv.org/pdf/2512.12112v1)

**Authors**: Padmeswari Nandiya, Ahmad Mohsin, Ahmed Ibrahim, Iqbal H. Sarker, Helge Janicke

**Abstract**: Industry 5.0's increasing integration of IT and OT systems is transforming industrial operations but also expanding the cyber-physical attack surface. Industrial Control Systems (ICS) face escalating security challenges as traditional siloed defences fail to provide coherent, cross-domain threat insights. We present BRIDG-ICS (BRIDge for Industrial Control Systems), an AI-driven Knowledge Graph (KG) framework for context-aware threat analysis and quantitative assessment of cyber resilience in smart manufacturing environments. BRIDG-ICS fuses heterogeneous industrial and cybersecurity data into an integrated Industrial Security Knowledge Graph linking assets, vulnerabilities, and adversarial behaviours with probabilistic risk metrics (e.g. exploit likelihood, attack cost). This unified graph representation enables multi-stage attack path simulation using graph-analytic techniques. To enrich the graph's semantic depth, the framework leverages Large Language Models (LLMs): domain-specific LLMs extract cybersecurity entities, predict relationships, and translate natural-language threat descriptions into structured graph triples, thereby populating the knowledge graph with missing associations and latent risk indicators. This unified AI-enriched KG supports multi-hop, causality-aware threat reasoning, improving visibility into complex attack chains and guiding data-driven mitigation. In simulated industrial scenarios, BRIDG-ICS scales well, reduces potential attack exposure, and can enhance cyber-physical system resilience in Industry 5.0 settings.



## **30. Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring**

cs.CR

40 pages, 13 figures

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.12069v1) [paper-pdf](https://arxiv.org/pdf/2512.12069v1)

**Authors**: Peichun Hua, Hao Li, Shanghao Shi, Zhiyuan Yu, Ning Zhang

**Abstract**: Large Vision-Language Models (LVLMs) are vulnerable to a growing array of multimodal jailbreak attacks, necessitating defenses that are both generalizable to novel threats and efficient for practical deployment. Many current strategies fall short, either targeting specific attack patterns, which limits generalization, or imposing high computational overhead. While lightweight anomaly-detection methods offer a promising direction, we find that their common one-class design tends to confuse novel benign inputs with malicious ones, leading to unreliable over-rejection. To address this, we propose Representational Contrastive Scoring (RCS), a framework built on a key insight: the most potent safety signals reside within the LVLM's own internal representations. Our approach inspects the internal geometry of these representations, learning a lightweight projection to maximally separate benign and malicious inputs in safety-critical layers. This enables a simple yet powerful contrastive score that differentiates true malicious intent from mere novelty. Our instantiations, MCD (Mahalanobis Contrastive Detection) and KCD (K-nearest Contrastive Detection), achieve state-of-the-art performance on a challenging evaluation protocol designed to test generalization to unseen attack types. This work demonstrates that effective jailbreak detection can be achieved by applying simple, interpretable statistical methods to the appropriate internal representations, offering a practical path towards safer LVLM deployment. Our code is available on Github https://github.com/sarendis56/Jailbreak_Detection_RCS.



## **31. Learning to Extract Context for Context-Aware LLM Inference**

cs.LG

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11986v1) [paper-pdf](https://arxiv.org/pdf/2512.11986v1)

**Authors**: Minseon Kim, Lucas Caccia, Zhengyan Shi, Matheus Pereira, Marc-Alexandre Côté, Xingdi Yuan, Alessandro Sordoni

**Abstract**: User prompts to large language models (LLMs) are often ambiguous or under-specified, and subtle contextual cues shaped by user intentions, prior knowledge, and risk factors strongly influence what constitutes an appropriate response. Misinterpreting intent or risks may lead to unsafe outputs, while overly cautious interpretations can cause unnecessary refusal of benign requests. In this paper, we question the conventional framework in which LLMs generate immediate responses to requests without considering broader contextual factors. User requests are situated within broader contexts such as intentions, knowledge, and prior experience, which strongly influence what constitutes an appropriate answer. We propose a framework that extracts and leverages such contextual information from the user prompt itself. Specifically, a reinforcement learning based context generator, designed in an autoencoder-like fashion, is trained to infer contextual signals grounded in the prompt and use them to guide response generation. This approach is particularly important for safety tasks, where ambiguous requests may bypass safeguards while benign but confusing requests can trigger unnecessary refusals. Experiments show that our method reduces harmful responses by an average of 5.6% on the SafetyInstruct dataset across multiple foundation models and improves the harmonic mean of attack success rate and compliance on benign prompts by 6.2% on XSTest and WildJailbreak. These results demonstrate the effectiveness of context extraction for safer and more reliable LLM inferences.



## **32. Super Suffixes: Bypassing Text Generation Alignment and Guard Models Simultaneously**

cs.CR

13 pages, 5 Figures

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11783v1) [paper-pdf](https://arxiv.org/pdf/2512.11783v1)

**Authors**: Andrew Adiletta, Kathryn Adiletta, Kemal Derya, Berk Sunar

**Abstract**: The rapid deployment of Large Language Models (LLMs) has created an urgent need for enhanced security and privacy measures in Machine Learning (ML). LLMs are increasingly being used to process untrusted text inputs and even generate executable code, often while having access to sensitive system controls. To address these security concerns, several companies have introduced guard models, which are smaller, specialized models designed to protect text generation models from adversarial or malicious inputs. In this work, we advance the study of adversarial inputs by introducing Super Suffixes, suffixes capable of overriding multiple alignment objectives across various models with different tokenization schemes. We demonstrate their effectiveness, along with our joint optimization technique, by successfully bypassing the protection mechanisms of Llama Prompt Guard 2 on five different text generation models for malicious text and code generation. To the best of our knowledge, this is the first work to reveal that Llama Prompt Guard 2 can be compromised through joint optimization.   Additionally, by analyzing the changing similarity of a model's internal state to specific concept directions during token sequence processing, we propose an effective and lightweight method to detect Super Suffix attacks. We show that the cosine similarity between the residual stream and certain concept directions serves as a distinctive fingerprint of model intent. Our proposed countermeasure, DeltaGuard, significantly improves the detection of malicious prompts generated through Super Suffixes. It increases the non-benign classification rate to nearly 100%, making DeltaGuard a valuable addition to the guard model stack and enhancing robustness against adversarial prompt attacks.



## **33. When Reject Turns into Accept: Quantifying the Vulnerability of LLM-Based Scientific Reviewers to Indirect Prompt Injection**

cs.AI

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.10449v2) [paper-pdf](https://arxiv.org/pdf/2512.10449v2)

**Authors**: Devanshu Sahoo, Manish Prasad, Vasudev Majhi, Jahnvi Singh, Vinay Chamola, Yash Sinha, Murari Mandal, Dhruv Kumar

**Abstract**: The landscape of scientific peer review is rapidly evolving with the integration of Large Language Models (LLMs). This shift is driven by two parallel trends: the widespread individual adoption of LLMs by reviewers to manage workload (the "Lazy Reviewer" hypothesis) and the formal institutional deployment of AI-powered assessment systems by conferences like AAAI and Stanford's Agents4Science. This study investigates the robustness of these "LLM-as-a-Judge" systems (both illicit and sanctioned) to adversarial PDF manipulation. Unlike general jailbreaks, we focus on a distinct incentive: flipping "Reject" decisions to "Accept," for which we develop a novel evaluation metric which we term as WAVS (Weighted Adversarial Vulnerability Score). We curated a dataset of 200 scientific papers and adapted 15 domain-specific attack strategies to this task, evaluating them across 13 Language Models, including GPT-5, Claude Haiku, and DeepSeek. Our results demonstrate that obfuscation strategies like "Maximum Mark Magyk" successfully manipulate scores, achieving alarming decision flip rates even in large-scale models. We will release our complete dataset and injection framework to facilitate more research on this topic.



## **34. Phishing Email Detection Using Large Language Models**

cs.CR

7 pages

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.10104v2) [paper-pdf](https://arxiv.org/pdf/2512.10104v2)

**Authors**: Najmul Hasan, Prashanth BusiReddyGari, Haitao Zhao, Yihao Ren, Jinsheng Xu, Shaohu Zhang

**Abstract**: Email phishing is one of the most prevalent and globally consequential vectors of cyber intrusion. As systems increasingly deploy Large Language Models (LLMs) applications, these systems face evolving phishing email threats that exploit their fundamental architectures. Current LLMs require substantial hardening before deployment in email security systems, particularly against coordinated multi-vector attacks that exploit architectural vulnerabilities. This paper proposes LLMPEA, an LLM-based framework to detect phishing email attacks across multiple attack vectors, including prompt injection, text refinement, and multilingual attacks. We evaluate three frontier LLMs (e.g., GPT-4o, Claude Sonnet 4, and Grok-3) and comprehensive prompting design to assess their feasibility, robustness, and limitations against phishing email attacks. Our empirical analysis reveals that LLMs can detect the phishing email over 90% accuracy while we also highlight that LLM-based phishing email detection systems could be exploited by adversarial attack, prompt injection, and multilingual attacks. Our findings provide critical insights for LLM-based phishing detection in real-world settings where attackers exploit multiple vulnerabilities in combination.



## **35. CNFinBench: A Benchmark for Safety and Compliance of Large Language Models in Finance**

cs.CE

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.09506v2) [paper-pdf](https://arxiv.org/pdf/2512.09506v2)

**Authors**: Jinru Ding, Chao Ding, Wenrao Pang, Boyi Xiao, Zhiqiang Liu, Pengcheng Chen, Jiayuan Chen, Tiantian Yuan, Junming Guan, Yidong Jiang, Dawei Cheng, Jie Xu

**Abstract**: Large language models (LLMs) are increasingly deployed across the financial sector for tasks like investment research and algorithmic trading. Their high-stakes nature demands rigorous evaluation of models' safety and regulatory alignment. However, there is a significant gap between evaluation capabilities and safety requirements. Current financial benchmarks mainly focus on textbook-style question answering and numerical problem-solving, failing to simulate the open-ended scenarios where safety risks typically manifest. To close these gaps, we introduce CNFinBench, a benchmark structured around a Capability-Compliance-Safety triad encompassing 15 subtasks. For Capability Q&As, we introduce a novel business-vertical taxonomy aligned with core financial domains like banking operations, which allows institutions to assess model readiness for deployment in operational scenarios. For Compliance and Risk Control Q&As, we embed regulatory requirements within realistic business scenarios to ensure models are evaluated under practical, scenario-driven conditions. For Safety Q&As, we uniquely incorporate structured bias and fairness auditing, a dimension overlooked by other holistic financial benchmarks, and introduce the first multi-turn adversarial dialogue task to systematically expose compliance decay under sustained, context-aware attacks. Accordingly, we propose the Harmful Instruction Compliance Score (HICS) to quantify models' consistency in resisting harmful instructions across multi-turn dialogues. Experiments on 21 models across all subtasks reveal a persistent gap between capability and compliance: models achieve an average score of 61.0 on capability tasks but drop to 34.2 on compliance and risk-control evaluations. In multi-turn adversarial dialogue tests, most LLMs attain only partial resistance, demonstrating that refusal alone is insufficient without cited, verifiable reasoning.



## **36. The Trojan Knowledge: Bypassing Commercial LLM Guardrails via Harmless Prompt Weaving and Adaptive Tree Search**

cs.CR

Updated with new baselines and experimental results

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.01353v3) [paper-pdf](https://arxiv.org/pdf/2512.01353v3)

**Authors**: Rongzhe Wei, Peizhi Niu, Xinjie Shen, Tony Tu, Yifan Li, Ruihan Wu, Eli Chien, Pin-Yu Chen, Olgica Milenkovic, Pan Li

**Abstract**: Large language models (LLMs) remain vulnerable to jailbreak attacks that bypass safety guardrails to elicit harmful outputs. Existing approaches overwhelmingly operate within the prompt-optimization paradigm: whether through traditional algorithmic search or recent agent-based workflows, the resulting prompts typically retain malicious semantic signals that modern guardrails are primed to detect. In contrast, we identify a deeper, largely overlooked vulnerability stemming from the highly interconnected nature of an LLM's internal knowledge. This structure allows harmful objectives to be realized by weaving together sequences of benign sub-queries, each of which individually evades detection. To exploit this loophole, we introduce the Correlated Knowledge Attack Agent (CKA-Agent), a dynamic framework that reframes jailbreaking as an adaptive, tree-structured exploration of the target model's knowledge base. The CKA-Agent issues locally innocuous queries, uses model responses to guide exploration across multiple paths, and ultimately assembles the aggregated information to achieve the original harmful objective. Evaluated across state-of-the-art commercial LLMs (Gemini2.5-Flash/Pro, GPT-oss-120B, Claude-Haiku-4.5), CKA-Agent consistently achieves over 95% success rates even against strong guardrails, underscoring the severity of this vulnerability and the urgent need for defenses against such knowledge-decomposition attacks. Our codes are available at https://github.com/Graph-COM/CKA-Agent.



## **37. Phantom Menace: Exploring and Enhancing the Robustness of VLA Models Against Physical Sensor Attacks**

cs.RO

Accepted by AAAI 2026 main track

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2511.10008v2) [paper-pdf](https://arxiv.org/pdf/2511.10008v2)

**Authors**: Xuancun Lu, Jiaxiang Chen, Shilin Xiao, Zizhi Jin, Zhangrui Chen, Hanwen Yu, Bohan Qian, Ruochen Zhou, Xiaoyu Ji, Wenyuan Xu

**Abstract**: Vision-Language-Action (VLA) models revolutionize robotic systems by enabling end-to-end perception-to-action pipelines that integrate multiple sensory modalities, such as visual signals processed by cameras and auditory signals captured by microphones. This multi-modality integration allows VLA models to interpret complex, real-world environments using diverse sensor data streams. Given the fact that VLA-based systems heavily rely on the sensory input, the security of VLA models against physical-world sensor attacks remains critically underexplored. To address this gap, we present the first systematic study of physical sensor attacks against VLAs, quantifying the influence of sensor attacks and investigating the defenses for VLA models. We introduce a novel "Real-Sim-Real" framework that automatically simulates physics-based sensor attack vectors, including six attacks targeting cameras and two targeting microphones, and validates them on real robotic systems. Through large-scale evaluations across various VLA architectures and tasks under varying attack parameters, we demonstrate significant vulnerabilities, with susceptibility patterns that reveal critical dependencies on task types and model designs. We further develop an adversarial-training-based defense that enhances VLA robustness against out-of-distribution physical perturbations caused by sensor attacks while preserving model performance. Our findings expose an urgent need for standardized robustness benchmarks and mitigation strategies to secure VLA deployments in safety-critical environments.



## **38. Biologically-Informed Hybrid Membership Inference Attacks on Generative Genomic Models**

cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2511.07503v3) [paper-pdf](https://arxiv.org/pdf/2511.07503v3)

**Authors**: Asia Belfiore, Jonathan Passerat-Palmbach, Dmitrii Usynin

**Abstract**: The increased availability of genetic data has transformed genomics research, but raised many privacy concerns regarding its handling due to its sensitive nature. This work explores the use of language models (LMs) for the generation of synthetic genetic mutation profiles, leveraging differential privacy (DP) for the protection of sensitive genetic data. We empirically evaluate the privacy guarantees of our DP modes by introducing a novel Biologically-Informed Hybrid Membership Inference Attack (biHMIA), which combines traditional black box MIA with contextual genomics metrics for enhanced attack power. Our experiments show that both small and large transformer GPT-like models are viable synthetic variant generators for small-scale genomics, and that our hybrid attack leads, on average, to higher adversarial success compared to traditional metric-based MIAs.



## **39. RAGRank: Using PageRank to Counter Poisoning in CTI LLM Pipelines**

cs.CR

Presented as a poster at the Annual Computer Security Applications Conference (ACSAC) 2025

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2510.20768v2) [paper-pdf](https://arxiv.org/pdf/2510.20768v2)

**Authors**: Austin Jia, Avaneesh Ramesh, Zain Shamsi, Daniel Zhang, Alex Liu

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as the dominant architectural pattern to operationalize Large Language Model (LLM) usage in Cyber Threat Intelligence (CTI) systems. However, this design is susceptible to poisoning attacks, and previously proposed defenses can fail for CTI contexts as cyber threat information is often completely new for emerging attacks, and sophisticated threat actors can mimic legitimate formats, terminology, and stylistic conventions. To address this issue, we propose that the robustness of modern RAG defenses can be accelerated by applying source credibility algorithms on corpora, using PageRank as an example. In our experiments, we demonstrate quantitatively that our algorithm applies a lower authority score to malicious documents while promoting trusted content, using the standardized MS MARCO dataset. We also demonstrate proof-of-concept performance of our algorithm on CTI documents and feeds.



## **40. BreakFun: Jailbreaking LLMs via Schema Exploitation**

cs.CR

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2510.17904v2) [paper-pdf](https://arxiv.org/pdf/2510.17904v2)

**Authors**: Amirkia Rafiei Oskooei, Mehmet S. Aktas

**Abstract**: The proficiency of Large Language Models (LLMs) in processing structured data and adhering to syntactic rules is a capability that drives their widespread adoption but also makes them paradoxically vulnerable. In this paper, we investigate this vulnerability through BreakFun, a jailbreak methodology that weaponizes an LLM's adherence to structured schemas. BreakFun employs a three-part prompt that combines an innocent framing and a Chain-of-Thought distraction with a core "Trojan Schema"--a carefully crafted data structure that compels the model to generate harmful content, exploiting the LLM's strong tendency to follow structures and schemas. We demonstrate this vulnerability is highly transferable, achieving an average success rate of 89% across 13 foundational and proprietary models on JailbreakBench, and reaching a 100% Attack Success Rate (ASR) on several prominent models. A rigorous ablation study confirms this Trojan Schema is the attack's primary causal factor. To counter this, we introduce the Adversarial Prompt Deconstruction guardrail, a defense that utilizes a secondary LLM to perform a "Literal Transcription"--extracting all human-readable text to isolate and reveal the user's true harmful intent. Our proof-of-concept guardrail demonstrates high efficacy against the attack, validating that targeting the deceptive schema is a viable mitigation strategy. Our work provides a look into how an LLM's core strengths can be turned into critical weaknesses, offering a fresh perspective for building more robustly aligned models.



## **41. Lexo: Eliminating Stealthy Supply-Chain Attacks via LLM-Assisted Program Regeneration**

cs.CR

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2510.14522v3) [paper-pdf](https://arxiv.org/pdf/2510.14522v3)

**Authors**: Evangelos Lamprou, Julian Dai, Grigoris Ntousakis, Martin C. Rinard, Nikos Vasilakis

**Abstract**: Software supply-chain attacks are an important and ongoing concern in the open source software ecosystem. These attacks maintain the standard functionality that a component implements, but additionally hide malicious functionality activated only when the component reaches its target environment. Lexo addresses such stealthy attacks by automatically learning and regenerating vulnerability-free versions of potentially malicious components. Lexo first generates a set of input-output pairs to model a component's full observable behavior, which it then uses to synthesize a new version of the original component. The new component implements the original functionality but avoids stealthy malicious behavior. Throughout this regeneration process, Lexo consults several distinct instances of Large Language Models (LLMs), uses correctness and coverage metrics to shepherd these instances, and guardrails their results. An evaluation on 100+ real-world packages, including high-profile stealthy supply-chain attacks, indicates that Lexo scales across multiple domains, regenerates code efficiently (<30m on average), maintains compatibility, and succeeds in eliminating malicious code in several real-world supply-chain-attacks, even in cases when a state-of-the-art LLM fails to eliminate malicious code when given the source code of the component and prompted to do so.



## **42. A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks**

cs.CR

Accepted at the 11th IEEE WIECON-ECE 2025

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2509.14285v4) [paper-pdf](https://arxiv.org/pdf/2509.14285v4)

**Authors**: S M Asif Hossain, Ruksat Khan Shayoni, Mohd Ruhul Ameen, Akif Islam, M. F. Mridha, Jungpil Shin

**Abstract**: Prompt injection attacks represent a major vulnerability in Large Language Model (LLM) deployments, where malicious instructions embedded in user inputs can override system prompts and induce unintended behaviors. This paper presents a novel multi-agent defense framework that employs specialized LLM agents in coordinated pipelines to detect and neutralize prompt injection attacks in real-time. We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system. Our comprehensive evaluation on 55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2), demonstrates significant security improvements. Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0% across all tested scenarios. The framework demonstrates robustness across multiple attack categories including direct overrides, code execution attempts, data exfiltration, and obfuscation techniques, while maintaining system functionality for legitimate queries.



## **43. Trust Me, I Know This Function: Hijacking LLM Static Analysis using Bias**

cs.LG

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2508.17361v2) [paper-pdf](https://arxiv.org/pdf/2508.17361v2)

**Authors**: Shir Bernstein, David Beste, Daniel Ayzenshteyn, Lea Schonherr, Yisroel Mirsky

**Abstract**: Large Language Models (LLMs) are increasingly trusted to perform automated code review and static analysis at scale, supporting tasks such as vulnerability detection, summarization, and refactoring. In this paper, we identify and exploit a critical vulnerability in LLM-based code analysis: an abstraction bias that causes models to overgeneralize familiar programming patterns and overlook small, meaningful bugs. Adversaries can exploit this blind spot to hijack the control flow of the LLM's interpretation with minimal edits and without affecting actual runtime behavior. We refer to this attack as a Familiar Pattern Attack (FPA).   We develop a fully automated, black-box algorithm that discovers and injects FPAs into target code. Our evaluation shows that FPAs are not only effective against basic and reasoning models, but are also transferable across model families (OpenAI, Anthropic, Google), and universal across programming languages (Python, C, Rust, Go). Moreover, FPAs remain effective even when models are explicitly warned about the attack via robust system prompts. Finally, we explore positive, defensive uses of FPAs and discuss their broader implications for the reliability and safety of code-oriented LLMs.



## **44. ConceptGuard: Neuro-Symbolic Safety Guardrails via Sparse Interpretable Jailbreak Concepts**

cs.CL

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2508.16325v2) [paper-pdf](https://arxiv.org/pdf/2508.16325v2)

**Authors**: Darpan Aswal, Céline Hudelot

**Abstract**: Large Language Models have found success in a variety of applications. However, their safety remains a concern due to the existence of various jailbreaking methods. Despite significant efforts, alignment and safety fine-tuning only provide a certain degree of robustness against jailbreak attacks that covertly mislead LLMs towards the generation of harmful content. This leaves them prone to a range of vulnerabilities, including targeted misuse and accidental user profiling. This work introduces \textbf{ConceptGuard}, a novel framework that leverages Sparse Autoencoders (SAEs) to identify interpretable concepts within LLM internals associated with different jailbreak themes. By extracting semantically meaningful internal representations, ConceptGuard enables building robust safety guardrails -- offering fully explainable and generalizable defenses without sacrificing model capabilities or requiring further fine-tuning. Leveraging advances in the mechanistic interpretability of LLMs, our approach provides evidence for a shared activation geometry for jailbreak attacks in the representation space, a potential foundation for designing more interpretable and generalizable safeguards against attackers.



## **45. May I have your Attention? Breaking Fine-Tuning based Prompt Injection Defenses using Architecture-Aware Attacks**

cs.CR

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2507.07417v2) [paper-pdf](https://arxiv.org/pdf/2507.07417v2)

**Authors**: Nishit V. Pandya, Andrey Labunets, Sicun Gao, Earlence Fernandes

**Abstract**: A popular class of defenses against prompt injection attacks on large language models (LLMs) relies on fine-tuning to separate instructions and data, so that the LLM does not follow instructions that might be present with data. We evaluate the robustness of this approach in the whitebox setting by constructing strong optimization-based attacks, and show that the defenses do not provide the claimed security properties. Specifically, we construct a novel attention-based attack algorithm for textual LLMs and apply it to three recent whitebox defenses SecAlign (CCS 2025), SecAlign++, and StruQ (USENIX Security 2025), showing attacks with success rates of up to \textbf{85-95\%} on unseen prompts with modest increase in attacker budget in terms of tokens. Our findings make fundamental progress towards understanding the robustness of prompt injection defenses in the whitebox setting. We release our code and attacks at https://github.com/nishitvp/better_opts_attacks



## **46. On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks**

cs.CL

Published in NeurIPS 2025

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2507.06489v3) [paper-pdf](https://arxiv.org/pdf/2507.06489v3)

**Authors**: Stephen Obadinma, Xiaodan Zhu

**Abstract**: Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to help ensure transparency, trust, and safety in many applications, including those involving human-AI interactions. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce attack frameworks targeting verbal confidence scores through both perturbation and jailbreak-based methods, and demonstrate that these attacks can significantly impair verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current verbal confidence is vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the need to design robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.



## **47. From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agents Workflows**

cs.CR

The paper is published in ICT Express (Elsevier)

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2506.23260v2) [paper-pdf](https://arxiv.org/pdf/2506.23260v2)

**Authors**: Mohamed Amine Ferrag, Norbert Tihanyi, Djallel Hamouda, Leandros Maglaras, Abderrahmane Lakas, Merouane Debbah

**Abstract**: Autonomous AI agents powered by large language models (LLMs) with structured function-calling interfaces enable real-time data retrieval, computation, and multi-step orchestration. However, the rapid growth of plugins, connectors, and inter-agent protocols has outpaced security practices, leading to brittle integrations that rely on ad-hoc authentication, inconsistent schemas, and weak validation. This survey introduces a unified end-to-end threat model for LLM-agent ecosystems, covering host-to-tool and agent-to-agent communications. We systematically categorize more than thirty attack techniques spanning input manipulation, model compromise, system and privacy attacks, and protocol-level vulnerabilities. For each category, we provide a formal threat formulation defining attacker capabilities, objectives, and affected system layers. Representative examples include Prompt-to-SQL injections and the Toxic Agent Flow exploit in GitHub MCP servers. We analyze attack feasibility, review existing defenses, and discuss mitigation strategies such as dynamic trust management, cryptographic provenance tracking, and sandboxed agent interfaces. The framework is validated through expert review and cross-mapping with real-world incidents and public vulnerability repositories, including CVE and NIST NVD. Compared to prior surveys, this work presents the first integrated taxonomy bridging input-level exploits and protocol-layer vulnerabilities in LLM-agent ecosystems, offering actionable guidance for designing secure and resilient agentic AI systems.



## **48. MoAPT: Mixture of Adversarial Prompt Tuning for Vision-Language Models**

cs.CV

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2505.17509v2) [paper-pdf](https://arxiv.org/pdf/2505.17509v2)

**Authors**: Shiji Zhao, Qihui Zhu, Shukun Xiong, Shouwei Ruan, Maoxun Yuan, Jialing Tao, Jiexi Liu, Ranjie Duan, Jie Zhang, Jie Zhang, Xingxing Wei

**Abstract**: Large pre-trained Vision Language Models (VLMs) demonstrate excellent generalization capabilities but remain highly susceptible to adversarial examples, posing potential security risks. To improve the robustness of VLMs against adversarial examples, adversarial prompt tuning methods are proposed to align the text feature with the adversarial image feature without changing model parameters. However, when facing various adversarial attacks, a single learnable text prompt has insufficient generalization to align well with all adversarial image features, which ultimately results in overfitting. To address the above challenge, in this paper, we empirically find that increasing the number of learned prompts yields greater robustness improvements than simply extending the length of a single prompt. Building on this observation, we propose an adversarial tuning method named \textbf{Mixture of Adversarial Prompt Tuning (MoAPT)} to enhance the generalization against various adversarial attacks for VLMs. MoAPT aims to learn mixture text prompts to obtain more robust text features. To further enhance the adaptability, we propose a conditional weight router based on the adversarial images to predict the mixture weights of multiple learned prompts, which helps obtain sample-specific mixture text features aligning with different adversarial image features. Extensive experiments across 11 datasets under different settings show that our method can achieve better adversarial robustness than state-of-the-art approaches.



## **49. ExpShield: Safeguarding Web Text from Unauthorized Crawling and LLM Exploitation**

cs.CR

18 pages

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2412.21123v3) [paper-pdf](https://arxiv.org/pdf/2412.21123v3)

**Authors**: Ruixuan Liu, Toan Tran, Tianhao Wang, Hongsheng Hu, Shuo Wang, Li Xiong

**Abstract**: As large language models increasingly memorize web-scraped training content, they risk exposing copyrighted or private information. Existing protections require compliance from crawlers or model developers, fundamentally limiting their effectiveness. We propose ExpShield, a proactive self-guard that mitigates memorization while maintaining readability via invisible perturbations, and we formulate it as a constrained optimization problem. Due to the lack of an individual-level risk metric for natural text, we first propose instance exploitation, a metric that measures how much training on a specific text increases the chance of guessing that text from a set of candidates-with zero indicating perfect defense. Directly solving the problem is infeasible for defenders without sufficient knowledge, thus we develop two effective proxy solutions: single-level optimization and synthetic perturbation. To enhance the defense, we reveal and verify the memorization trigger hypothesis, which can help to identify key tokens for memorization. Leveraging this insight, we design targeted perturbations that (i) neutralize inherent trigger tokens to reduce memorization and (ii) introduce artificial trigger tokens to misdirect model memorization. Experiments validate our defense across attacks, model scales, and tasks in language and vision-to-language modeling. Even with privacy backdoor, the Membership Inference Attack (MIA) AUC drops from 0.95 to 0.55 under the defense, and the instance exploitation approaches zero. This suggests that compared to the ideal no-misuse scenario, the risk of exposing a text instance remains nearly unchanged despite its inclusion in the training data.



## **50. Memory Backdoor Attacks on Neural Networks**

cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2411.14516v2) [paper-pdf](https://arxiv.org/pdf/2411.14516v2)

**Authors**: Eden Luzon, Guy Amit, Roy Weiss, Torsten Kraub, Alexandra Dmitrienko, Yisroel Mirsky

**Abstract**: Neural networks are often trained on proprietary datasets, making them attractive attack targets. We present a novel dataset extraction method leveraging an innovative training time backdoor attack, allowing a malicious federated learning server to systematically and deterministically extract complete client training samples through a simple indexing process. Unlike prior techniques, our approach guarantees exact data recovery rather than probabilistic reconstructions or hallucinations, provides precise control over which samples are memorized and how many, and shows high capacity and robustness. Infected models output data samples when they receive a patternbased index trigger, enabling systematic extraction of meaningful patches from each clients local data without disrupting global model utility. To address small model output sizes, we extract patches and then recombined them. The attack requires only a minor modification to the training code that can easily evade detection during client-side verification. Hence, this vulnerability represents a realistic FL supply-chain threat, where a malicious server can distribute modified training code to clients and later recover private data from their updates. Evaluations across classifiers, segmentation models, and large language models demonstrate that thousands of sensitive training samples can be recovered from client models with minimal impact on task performance, and a clients entire dataset can be stolen after multiple FL rounds. For instance, a medical segmentation dataset can be extracted with only a 3 percent utility drop. These findings expose a critical privacy vulnerability in FL systems, emphasizing the need for stronger integrity and transparency in distributed training pipelines.



