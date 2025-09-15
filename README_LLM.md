# Latest Large Language Model Attack Papers
**update at 2025-09-15 14:33:42**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. URL2Graph++: Unified Semantic-Structural-Character Learning for Malicious URL Detection**

cs.CR

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.10287v1) [paper-pdf](http://arxiv.org/pdf/2509.10287v1)

**Authors**: Ye Tian, Yifan Jia, Yanbin Wang, Jianguo Sun, Zhiquan Liu, Xiaowen Ling

**Abstract**: Malicious URL detection remains a major challenge in cybersecurity, primarily due to two factors: (1) the exponential growth of the Internet has led to an immense diversity of URLs, making generalized detection increasingly difficult; and (2) attackers are increasingly employing sophisticated obfuscation techniques to evade detection. We advocate that addressing these challenges fundamentally requires: (1) obtaining semantic understanding to improve generalization across vast and diverse URL sets, and (2) accurately modeling contextual relationships within the structural composition of URLs. In this paper, we propose a novel malicious URL detection method combining multi-granularity graph learning with semantic embedding to jointly capture semantic, character-level, and structural features for robust URL analysis. To model internal dependencies within URLs, we first construct dual-granularity URL graphs at both subword and character levels, where nodes represent URL tokens/characters and edges encode co-occurrence relationships. To obtain fine-grained embeddings, we initialize node representations using a character-level convolutional network. The two graphs are then processed through jointly trained Graph Convolutional Networks to learn consistent graph-level representations, enabling the model to capture complementary structural features that reflect co-occurrence patterns and character-level dependencies. Furthermore, we employ BERT to derive semantic representations of URLs for semantically aware understanding. Finally, we introduce a gated dynamic fusion network to combine the semantically enriched BERT representations with the jointly optimized graph vectors, further enhancing detection performance. We extensively evaluate our method across multiple challenging dimensions. Results show our method exceeds SOTA performance, including against large language models.



## **2. Between a Rock and a Hard Place: Exploiting Ethical Reasoning to Jailbreak LLMs**

cs.CR

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.05367v2) [paper-pdf](http://arxiv.org/pdf/2509.05367v2)

**Authors**: Shei Pern Chua, Zhen Leng Thai, Teh Kai Jun, Xiao Li, Xiaolin Hu

**Abstract**: Large language models (LLMs) have undergone safety alignment efforts to mitigate harmful outputs. However, as LLMs become more sophisticated in reasoning, their intelligence may introduce new security risks. While traditional jailbreak attacks relied on singlestep attacks, multi-turn jailbreak strategies that adapt dynamically to context remain underexplored. In this work, we introduce TRIAL (Trolley-problem Reasoning for Interactive Attack Logic), a framework that leverages LLMs ethical reasoning to bypass their safeguards. TRIAL embeds adversarial goals within ethical dilemmas modeled on the trolley problem. TRIAL demonstrates high jailbreak success rates towards both open and close-source models. Our findings underscore a fundamental limitation in AI safety: as models gain advanced reasoning abilities, the nature of their alignment may inadvertently allow for more covert security vulnerabilities to be exploited. TRIAL raises an urgent need in reevaluating safety alignment oversight strategies, as current safeguards may prove insufficient against context-aware adversarial attack.



## **3. Privacy Risks of LLM-Empowered Recommender Systems: An Inversion Attack Perspective**

cs.IR

Accepted at ACM RecSys 2025 (10 pages, 4 figures)

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2508.03703v2) [paper-pdf](http://arxiv.org/pdf/2508.03703v2)

**Authors**: Yubo Wang, Min Tang, Nuo Shen, Shujie Cui, Weiqing Wang

**Abstract**: The large language model (LLM) powered recommendation paradigm has been proposed to address the limitations of traditional recommender systems, which often struggle to handle cold start users or items with new IDs. Despite its effectiveness, this study uncovers that LLM empowered recommender systems are vulnerable to reconstruction attacks that can expose both system and user privacy. To examine this threat, we present the first systematic study on inversion attacks targeting LLM empowered recommender systems, where adversaries attempt to reconstruct original prompts that contain personal preferences, interaction histories, and demographic attributes by exploiting the output logits of recommendation models. We reproduce the vec2text framework and optimize it using our proposed method called Similarity Guided Refinement, enabling more accurate reconstruction of textual prompts from model generated logits. Extensive experiments across two domains (movies and books) and two representative LLM based recommendation models demonstrate that our method achieves high fidelity reconstructions. Specifically, we can recover nearly 65 percent of the user interacted items and correctly infer age and gender in 87 percent of the cases. The experiments also reveal that privacy leakage is largely insensitive to the victim model's performance but highly dependent on domain consistency and prompt complexity. These findings expose critical privacy vulnerabilities in LLM empowered recommender systems.



## **4. When Your Reviewer is an LLM: Biases, Divergence, and Prompt Injection Risks in Peer Review**

cs.CY

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.09912v1) [paper-pdf](http://arxiv.org/pdf/2509.09912v1)

**Authors**: Changjia Zhu, Junjie Xiong, Renkai Ma, Zhicong Lu, Yao Liu, Lingyao Li

**Abstract**: Peer review is the cornerstone of academic publishing, yet the process is increasingly strained by rising submission volumes, reviewer overload, and expertise mismatches. Large language models (LLMs) are now being used as "reviewer aids," raising concerns about their fairness, consistency, and robustness against indirect prompt injection attacks. This paper presents a systematic evaluation of LLMs as academic reviewers. Using a curated dataset of 1,441 papers from ICLR 2023 and NeurIPS 2022, we evaluate GPT-5-mini against human reviewers across ratings, strengths, and weaknesses. The evaluation employs structured prompting with reference paper calibration, topic modeling, and similarity analysis to compare review content. We further embed covert instructions into PDF submissions to assess LLMs' susceptibility to prompt injection. Our findings show that LLMs consistently inflate ratings for weaker papers while aligning more closely with human judgments on stronger contributions. Moreover, while overarching malicious prompts induce only minor shifts in topical focus, explicitly field-specific instructions successfully manipulate specific aspects of LLM-generated reviews. This study underscores both the promises and perils of integrating LLMs into peer review and points to the importance of designing safeguards that ensure integrity and trust in future review processes.



## **5. Advancing Security with Digital Twins: A Comprehensive Survey**

cs.CR

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2505.17310v2) [paper-pdf](http://arxiv.org/pdf/2505.17310v2)

**Authors**: Blessing Airehenbuwa, Touseef Hasan, Souvika Sarkar, Ujjwal Guin

**Abstract**: The proliferation of electronic devices has greatly transformed every aspect of human life, such as communication, healthcare, transportation, and energy. Unfortunately, the global electronics supply chain is vulnerable to various attacks, including piracy of intellectual properties, tampering, counterfeiting, information leakage, side-channel, and fault injection attacks, due to the complex nature of electronic products and vulnerabilities present in them. Although numerous solutions have been proposed to address these threats, significant gaps remain, particularly in providing scalable and comprehensive protection against emerging attacks. Digital twin, a dynamic virtual replica of a physical system, has emerged as a promising solution to address these issues by providing backward traceability, end-to-end visibility, and continuous verification of component integrity and behavior. In this paper, we comprehensively present the latest digital twin-based security implementations, including their role in cyber-physical systems, Internet of Things, cryptographic systems, detection of counterfeit electronics, intrusion detection, fault injection, and side-channel leakage. This work considers these critical security use cases within a single study to offer researchers and practitioners a unified reference for securing hardware with digital twins. The paper also explores the integration of large language models with digital twins for enhanced security and discusses current challenges, solutions, and future research directions.



## **6. Steering MoE LLMs via Expert (De)Activation**

cs.CL

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09660v1) [paper-pdf](http://arxiv.org/pdf/2509.09660v1)

**Authors**: Mohsen Fayyaz, Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Ryan Rossi, Trung Bui, Hinrich Schütze, Nanyun Peng

**Abstract**: Mixture-of-Experts (MoE) in Large Language Models (LLMs) routes each token through a subset of specialized Feed-Forward Networks (FFN), known as experts. We present SteerMoE, a framework for steering MoE models by detecting and controlling behavior-linked experts. Our detection method identifies experts with distinct activation patterns across paired inputs exhibiting contrasting behaviors. By selectively (de)activating such experts during inference, we control behaviors like faithfulness and safety without retraining or modifying weights. Across 11 benchmarks and 6 LLMs, our steering raises safety by up to +20% and faithfulness by +27%. In adversarial attack mode, it drops safety by -41% alone, and -100% when combined with existing jailbreak methods, bypassing all safety guardrails and exposing a new dimension of alignment faking hidden within experts.



## **7. Can LLMs Hack Enterprise Networks? Autonomous Assumed Breach Penetration-Testing Active Directory Networks**

cs.CR

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2502.04227v3) [paper-pdf](http://arxiv.org/pdf/2502.04227v3)

**Authors**: Andreas Happe, Jürgen Cito

**Abstract**: Enterprise penetration-testing is often limited by high operational costs and the scarcity of human expertise. This paper investigates the feasibility and effectiveness of using Large Language Model (LLM)-driven autonomous systems to address these challenges in real-world Active Directory (AD) enterprise networks.   We introduce a novel prototype designed to employ LLMs to autonomously perform Assumed Breach penetration-testing against enterprise networks. Our system represents the first demonstration of a fully autonomous, LLM-driven framework capable of compromising accounts within a real-life Microsoft Active Directory testbed, GOAD.   We perform our empirical evaluation using five LLMs, comparing reasoning to non-reasoning models as well as including open-weight models. Through quantitative and qualitative analysis, incorporating insights from cybersecurity experts, we demonstrate that autonomous LLMs can effectively conduct Assumed Breach simulations. Key findings highlight their ability to dynamically adapt attack strategies, perform inter-context attacks (e.g., web-app audits, social engineering, and unstructured data analysis for credentials), and generate scenario-specific attack parameters like realistic password candidates. The prototype exhibits robust self-correction mechanisms, installing missing tools and rectifying invalid command generations.   We find that the associated costs are competitive with, and often significantly lower than, those incurred by professional human pen-testers, suggesting a path toward democratizing access to essential security testing for organizations with budgetary constraints. However, our research also illuminates existing limitations, including instances of LLM ``going down rabbit holes'', challenges in comprehensive information transfer between planning and execution modules, and critical safety concerns that necessitate human oversight.



## **8. IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves**

cs.CV

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2411.00827v5) [paper-pdf](http://arxiv.org/pdf/2411.00827v5)

**Authors**: Ruofan Wang, Juncheng Li, Yixu Wang, Bo Wang, Xiaosen Wang, Yan Teng, Yingchun Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) gain prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks-techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multimodal data has constrained current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which often lack effectiveness and diversity across different contexts. In this paper, we propose IDEATOR, a novel jailbreak method that autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is grounded in the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR leverages a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Extensive experiments demonstrate IDEATOR's high effectiveness and transferability, achieving a 94% attack success rate (ASR) in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high ASRs of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Chameleon, respectively. Building on IDEATOR's strong transferability and automated process, we introduce the VLJailbreakBench, a safety benchmark comprising 3,654 multimodal jailbreak samples. Our benchmark results on 11 recently released VLMs reveal significant gaps in safety alignment. For instance, our challenge set achieves ASRs of 46.31% on GPT-4o and 19.65% on Claude-3.5-Sonnet, underscoring the urgent need for stronger defenses.VLJailbreakBench is publicly available at https://roywang021.github.io/VLJailbreakBench.



## **9. SimMark: A Robust Sentence-Level Similarity-Based Watermarking Algorithm for Large Language Models**

cs.CL

Accepted to EMNLP 25 main

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2502.02787v2) [paper-pdf](http://arxiv.org/pdf/2502.02787v2)

**Authors**: Amirhossein Dabiriaghdam, Lele Wang

**Abstract**: The widespread adoption of large language models (LLMs) necessitates reliable methods to detect LLM-generated text. We introduce SimMark, a robust sentence-level watermarking algorithm that makes LLMs' outputs traceable without requiring access to model internals, making it compatible with both open and API-based LLMs. By leveraging the similarity of semantic sentence embeddings combined with rejection sampling to embed detectable statistical patterns imperceptible to humans, and employing a soft counting mechanism, SimMark achieves robustness against paraphrasing attacks. Experimental results demonstrate that SimMark sets a new benchmark for robust watermarking of LLM-generated content, surpassing prior sentence-level watermarking techniques in robustness, sampling efficiency, and applicability across diverse domains, all while maintaining the text quality and fluency.



## **10. Character-Level Perturbations Disrupt LLM Watermarks**

cs.CR

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09112v1) [paper-pdf](http://arxiv.org/pdf/2509.09112v1)

**Authors**: Zhaoxi Zhang, Xiaomei Zhang, Yanjun Zhang, He Zhang, Shirui Pan, Bo Liu, Asif Qumer Gill, Leo Yu Zhang

**Abstract**: Large Language Model (LLM) watermarking embeds detectable signals into generated text for copyright protection, misuse prevention, and content detection. While prior studies evaluate robustness using watermark removal attacks, these methods are often suboptimal, creating the misconception that effective removal requires large perturbations or powerful adversaries.   To bridge the gap, we first formalize the system model for LLM watermark, and characterize two realistic threat models constrained on limited access to the watermark detector. We then analyze how different types of perturbation vary in their attack range, i.e., the number of tokens they can affect with a single edit. We observe that character-level perturbations (e.g., typos, swaps, deletions, homoglyphs) can influence multiple tokens simultaneously by disrupting the tokenization process. We demonstrate that character-level perturbations are significantly more effective for watermark removal under the most restrictive threat model. We further propose guided removal attacks based on the Genetic Algorithm (GA) that uses a reference detector for optimization. Under a practical threat model with limited black-box queries to the watermark detector, our method demonstrates strong removal performance. Experiments confirm the superiority of character-level perturbations and the effectiveness of the GA in removing watermarks under realistic constraints. Additionally, we argue there is an adversarial dilemma when considering potential defenses: any fixed defense can be bypassed by a suitable perturbation strategy. Motivated by this principle, we propose an adaptive compound character-level attack. Experimental results show that this approach can effectively defeat the defenses. Our findings highlight significant vulnerabilities in existing LLM watermark schemes and underline the urgency for the development of new robust mechanisms.



## **11. ACE: A Security Architecture for LLM-Integrated App Systems**

cs.CR

25 pages, 13 figures, 8 tables; accepted by Network and Distributed  System Security Symposium (NDSS) 2026

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2504.20984v3) [paper-pdf](http://arxiv.org/pdf/2504.20984v3)

**Authors**: Evan Li, Tushin Mallick, Evan Rose, William Robertson, Alina Oprea, Cristina Nita-Rotaru

**Abstract**: LLM-integrated app systems extend the utility of Large Language Models (LLMs) with third-party apps that are invoked by a system LLM using interleaved planning and execution phases to answer user queries. These systems introduce new attack vectors where malicious apps can cause integrity violation of planning or execution, availability breakdown, or privacy compromise during execution.   In this work, we identify new attacks impacting the integrity of planning, as well as the integrity and availability of execution in LLM-integrated apps, and demonstrate them against IsolateGPT, a recent solution designed to mitigate attacks from malicious apps. We propose Abstract-Concrete-Execute (ACE), a new secure architecture for LLM-integrated app systems that provides security guarantees for system planning and execution. Specifically, ACE decouples planning into two phases by first creating an abstract execution plan using only trusted information, and then mapping the abstract plan to a concrete plan using installed system apps. We verify that the plans generated by our system satisfy user-specified secure information flow constraints via static analysis on the structured plan output. During execution, ACE enforces data and capability barriers between apps, and ensures that the execution is conducted according to the trusted abstract plan. We show experimentally that ACE is secure against attacks from the InjecAgent and Agent Security Bench benchmarks for indirect prompt injection, and our newly introduced attacks. We also evaluate the utility of ACE in realistic environments, using the Tool Usage suite from the LangChain benchmark. Our architecture represents a significant advancement towards hardening LLM-based systems using system security principles.



## **12. Architecting Resilient LLM Agents: A Guide to Secure Plan-then-Execute Implementations**

cs.CR

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.08646v1) [paper-pdf](http://arxiv.org/pdf/2509.08646v1)

**Authors**: Ron F. Del Rosario, Klaudia Krawiecka, Christian Schroeder de Witt

**Abstract**: As Large Language Model (LLM) agents become increasingly capable of automating complex, multi-step tasks, the need for robust, secure, and predictable architectural patterns is paramount. This paper provides a comprehensive guide to the ``Plan-then-Execute'' (P-t-E) pattern, an agentic design that separates strategic planning from tactical execution. We explore the foundational principles of P-t-E, detailing its core components - the Planner and the Executor - and its architectural advantages in predictability, cost-efficiency, and reasoning quality over reactive patterns like ReAct (Reason + Act). A central focus is placed on the security implications of this design, particularly its inherent resilience to indirect prompt injection attacks by establishing control-flow integrity. We argue that while P-t-E provides a strong foundation, a defense-in-depth strategy is necessary, and we detail essential complementary controls such as the Principle of Least Privilege, task-scoped tool access, and sandboxed code execution. To make these principles actionable, this guide provides detailed implementation blueprints and working code references for three leading agentic frameworks: LangChain (via LangGraph), CrewAI, and AutoGen. Each framework's approach to implementing the P-t-E pattern is analyzed, highlighting unique features like LangGraph's stateful graphs for re-planning, CrewAI's declarative tool scoping for security, and AutoGen's built-in Docker sandboxing. Finally, we discuss advanced patterns, including dynamic re-planning loops, parallel execution with Directed Acyclic Graphs (DAGs), and the critical role of Human-in-the-Loop (HITL) verification, to offer a complete strategic blueprint for architects, developers, and security engineers aiming to build production-grade, resilient, and trustworthy LLM agents.



## **13. Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors**

cs.CL

Accepted by EMNLP-2025

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2505.15337v3) [paper-pdf](http://arxiv.org/pdf/2505.15337v3)

**Authors**: Hao Fang, Jiawei Kong, Tianqu Zhuang, Yixiang Qiu, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Yaowei Wang, Min Zhang

**Abstract**: The misuse of large language models (LLMs), such as academic plagiarism, has driven the development of detectors to identify LLM-generated texts. To bypass these detectors, paraphrase attacks have emerged to purposely rewrite these texts to evade detection. Despite the success, existing methods require substantial data and computational budgets to train a specialized paraphraser, and their attack efficacy greatly reduces when faced with advanced detection algorithms. To address this, we propose \textbf{Co}ntrastive \textbf{P}araphrase \textbf{A}ttack (CoPA), a training-free method that effectively deceives text detectors using off-the-shelf LLMs. The first step is to carefully craft instructions that encourage LLMs to produce more human-like texts. Nonetheless, we observe that the inherent statistical biases of LLMs can still result in some generated texts carrying certain machine-like attributes that can be captured by detectors. To overcome this, CoPA constructs an auxiliary machine-like word distribution as a contrast to the human-like distribution generated by the LLM. By subtracting the machine-like patterns from the human-like distribution during the decoding process, CoPA is able to produce sentences that are less discernible by text detectors. Our theoretical analysis suggests the superiority of the proposed attack. Extensive experiments validate the effectiveness of CoPA in fooling text detectors across various scenarios.



## **14. TraceRAG: A LLM-Based Framework for Explainable Android Malware Detection and Behavior Analysis**

cs.SE

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.08865v1) [paper-pdf](http://arxiv.org/pdf/2509.08865v1)

**Authors**: Guangyu Zhang, Xixuan Wang, Shiyu Sun, Peiyan Xiao, Kun Sun, Yanhai Xiong

**Abstract**: Sophisticated evasion tactics in malicious Android applications, combined with their intricate behavioral semantics, enable attackers to conceal malicious logic within legitimate functions, underscoring the critical need for robust and in-depth analysis frameworks. However, traditional analysis techniques often fail to recover deeply hidden behaviors or provide human-readable justifications for their decisions. Inspired by advances in large language models (LLMs), we introduce TraceRAG, a retrieval-augmented generation (RAG) framework that bridges natural language queries and Java code to deliver explainable malware detection and analysis. First, TraceRAG generates summaries of method-level code snippets, which are indexed in a vector database. At query time, behavior-focused questions retrieve the most semantically relevant snippets for deeper inspection. Finally, based on the multi-turn analysis results, TraceRAG produces human-readable reports that present the identified malicious behaviors and their corresponding code implementations. Experimental results demonstrate that our method achieves 96\% malware detection accuracy and 83.81\% behavior identification accuracy based on updated VirusTotal (VT) scans and manual verification. Furthermore, expert evaluation confirms the practical utility of the reports generated by TraceRAG.



## **15. ImportSnare: Directed "Code Manual" Hijacking in Retrieval-Augmented Code Generation**

cs.CR

This paper has been accepted by the ACM Conference on Computer and  Communications Security (CCS) 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07941v1) [paper-pdf](http://arxiv.org/pdf/2509.07941v1)

**Authors**: Kai Ye, Liangcai Su, Chenxiong Qian

**Abstract**: Code generation has emerged as a pivotal capability of Large Language Models(LLMs), revolutionizing development efficiency for programmers of all skill levels. However, the complexity of data structures and algorithmic logic often results in functional deficiencies and security vulnerabilities in generated code, reducing it to a prototype requiring extensive manual debugging. While Retrieval-Augmented Generation (RAG) can enhance correctness and security by leveraging external code manuals, it simultaneously introduces new attack surfaces.   In this paper, we pioneer the exploration of attack surfaces in Retrieval-Augmented Code Generation (RACG), focusing on malicious dependency hijacking. We demonstrate how poisoned documentation containing hidden malicious dependencies (e.g., matplotlib_safe) can subvert RACG, exploiting dual trust chains: LLM reliance on RAG and developers' blind trust in LLM suggestions. To construct poisoned documents, we propose ImportSnare, a novel attack framework employing two synergistic strategies: 1)Position-aware beam search optimizes hidden ranking sequences to elevate poisoned documents in retrieval results, and 2)Multilingual inductive suggestions generate jailbreaking sequences to manipulate LLMs into recommending malicious dependencies. Through extensive experiments across Python, Rust, and JavaScript, ImportSnare achieves significant attack success rates (over 50% for popular libraries such as matplotlib and seaborn) in general, and is also able to succeed even when the poisoning ratio is as low as 0.01%, targeting both custom and real-world malicious packages. Our findings reveal critical supply chain risks in LLM-powered development, highlighting inadequate security alignment for code generation tasks. To support future research, we will release the multilingual benchmark suite and datasets. The project homepage is https://importsnare.github.io.



## **16. Guided Reasoning in LLM-Driven Penetration Testing Using Structured Attack Trees**

cs.CR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07939v1) [paper-pdf](http://arxiv.org/pdf/2509.07939v1)

**Authors**: Katsuaki Nakano, Reza Feyyazi, Shanchieh Jay Yang, Michael Zuzak

**Abstract**: Recent advances in Large Language Models (LLMs) have driven interest in automating cybersecurity penetration testing workflows, offering the promise of faster and more consistent vulnerability assessment for enterprise systems. Existing LLM agents for penetration testing primarily rely on self-guided reasoning, which can produce inaccurate or hallucinated procedural steps. As a result, the LLM agent may undertake unproductive actions, such as exploiting unused software libraries or generating cyclical responses that repeat prior tactics. In this work, we propose a guided reasoning pipeline for penetration testing LLM agents that incorporates a deterministic task tree built from the MITRE ATT&CK Matrix, a proven penetration testing kll chain, to constrain the LLM's reaoning process to explicitly defined tactics, techniques, and procedures. This anchors reasoning in proven penetration testing methodologies and filters out ineffective actions by guiding the agent towards more productive attack procedures. To evaluate our approach, we built an automated penetration testing LLM agent using three LLMs (Llama-3-8B, Gemini-1.5, and GPT-4) and applied it to navigate 10 HackTheBox cybersecurity exercises with 103 discrete subtasks representing real-world cyberattack scenarios. Our proposed reasoning pipeline guided the LLM agent through 71.8\%, 72.8\%, and 78.6\% of subtasks using Llama-3-8B, Gemini-1.5, and GPT-4, respectively. Comparatively, the state-of-the-art LLM penetration testing tool using self-guided reasoning completed only 13.5\%, 16.5\%, and 75.7\% of subtasks and required 86.2\%, 118.7\%, and 205.9\% more model queries. This suggests that incorporating a deterministic task tree into LLM reasoning pipelines can enhance the accuracy and efficiency of automated cybersecurity assessments



## **17. Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection**

cs.CL

EMNLP 2025 Main

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2502.13061v3) [paper-pdf](http://arxiv.org/pdf/2502.13061v3)

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL



## **18. AgentSentinel: An End-to-End and Real-Time Security Defense Framework for Computer-Use Agents**

cs.CR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07764v1) [paper-pdf](http://arxiv.org/pdf/2509.07764v1)

**Authors**: Haitao Hu, Peng Chen, Yanpeng Zhao, Yuqi Chen

**Abstract**: Large Language Models (LLMs) have been increasingly integrated into computer-use agents, which can autonomously operate tools on a user's computer to accomplish complex tasks. However, due to the inherently unstable and unpredictable nature of LLM outputs, they may issue unintended tool commands or incorrect inputs, leading to potentially harmful operations. Unlike traditional security risks stemming from insecure user prompts, tool execution results from LLM-driven decisions introduce new and unique security challenges. These vulnerabilities span across all components of a computer-use agent. To mitigate these risks, we propose AgentSentinel, an end-to-end, real-time defense framework designed to mitigate potential security threats on a user's computer. AgentSentinel intercepts all sensitive operations within agent-related services and halts execution until a comprehensive security audit is completed. Our security auditing mechanism introduces a novel inspection process that correlates the current task context with system traces generated during task execution. To thoroughly evaluate AgentSentinel, we present BadComputerUse, a benchmark consisting of 60 diverse attack scenarios across six attack categories. The benchmark demonstrates a 87% average attack success rate on four state-of-the-art LLMs. Our evaluation shows that AgentSentinel achieves an average defense success rate of 79.6%, significantly outperforming all baseline defenses.



## **19. TrojanRobot: Physical-world Backdoor Attacks Against VLM-based Robotic Manipulation**

cs.RO

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2411.11683v4) [paper-pdf](http://arxiv.org/pdf/2411.11683v4)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Peijin Guo, Aishan Liu, Leo Yu Zhang, Xiaohua Jia

**Abstract**: Robotic manipulation in the physical world is increasingly empowered by \textit{large language models} (LLMs) and \textit{vision-language models} (VLMs), leveraging their understanding and perception capabilities. Recently, various attacks against such robotic policies have been proposed, with backdoor attacks drawing considerable attention for their high stealth and strong persistence capabilities. However, existing backdoor efforts are limited to simulators and suffer from physical-world realization. To address this, we propose \textit{TrojanRobot}, a highly stealthy and broadly effective robotic backdoor attack in the physical world. Specifically, we introduce a module-poisoning approach by embedding a backdoor module into the modular robotic policy, enabling backdoor control over the policy's visual perception module thereby backdooring the entire robotic policy. Our vanilla implementation leverages a backdoor-finetuned VLM to serve as the backdoor module. To enhance its generalization in physical environments, we propose a prime implementation, leveraging the LVLM-as-a-backdoor paradigm and developing three types of prime attacks, \ie, \textit{permutation}, \textit{stagnation}, and \textit{intentional} attacks, thus achieving finer-grained backdoors. Extensive experiments on the UR3e manipulator with 18 task instructions using robotic policies based on four VLMs demonstrate the broad effectiveness and physical-world stealth of TrojanRobot. Our attack's video demonstrations are available via a github link https://trojanrobot.github.io.



## **20. Transferable Direct Prompt Injection via Activation-Guided MCMC Sampling**

cs.AI

Accepted to EMNLP 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07617v1) [paper-pdf](http://arxiv.org/pdf/2509.07617v1)

**Authors**: Minghui Li, Hao Zhang, Yechao Zhang, Wei Wan, Shengshan Hu, pei Xiaobing, Jing Wang

**Abstract**: Direct Prompt Injection (DPI) attacks pose a critical security threat to Large Language Models (LLMs) due to their low barrier of execution and high potential damage. To address the impracticality of existing white-box/gray-box methods and the poor transferability of black-box methods, we propose an activations-guided prompt injection attack framework. We first construct an Energy-based Model (EBM) using activations from a surrogate model to evaluate the quality of adversarial prompts. Guided by the trained EBM, we employ the token-level Markov Chain Monte Carlo (MCMC) sampling to adaptively optimize adversarial prompts, thereby enabling gradient-free black-box attacks. Experimental results demonstrate our superior cross-model transferability, achieving 49.6% attack success rate (ASR) across five mainstream LLMs and 34.6% improvement over human-crafted prompts, and maintaining 36.6% ASR on unseen task scenarios. Interpretability analysis reveals a correlation between activations and attack effectiveness, highlighting the critical role of semantic patterns in transferable vulnerability exploitation.



## **21. A Decade-long Landscape of Advanced Persistent Threats: Longitudinal Analysis and Global Trends**

cs.CR

18 pages, 13 figures (including subfigures), 11 tables. To appear in  the Proceedings of the ACM Conference on Computer and Communications Security  (CCS) 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07457v1) [paper-pdf](http://arxiv.org/pdf/2509.07457v1)

**Authors**: Shakhzod Yuldoshkhujaev, Mijin Jeon, Doowon Kim, Nick Nikiforakis, Hyungjoon Koo

**Abstract**: An advanced persistent threat (APT) refers to a covert, long-term cyberattack, typically conducted by state-sponsored actors, targeting critical sectors and often remaining undetected for long periods. In response, collective intelligence from around the globe collaborates to identify and trace surreptitious activities, generating substantial documentation on APT campaigns publicly available on the web. While prior works predominantly focus on specific aspects of APT cases, such as detection, evaluation, cyber threat intelligence, and dataset creation, limited attention has been devoted to revisiting and investigating these scattered dossiers in a longitudinal manner. The objective of our study is to fill the gap by offering a macro perspective, connecting key insights and global trends in past APT attacks. We systematically analyze six reliable sources-three focused on technical reports and another three on threat actors-examining 1,509 APT dossiers (24,215 pages) spanning 2014-2023, and identifying 603 unique APT groups worldwide. To efficiently unearth relevant information, we employ a hybrid methodology that combines rule-based information retrieval with large-language-model-based search techniques. Our longitudinal analysis reveals shifts in threat actor activities, global attack vectors, changes in targeted sectors, and relationships between cyberattacks and significant events such as elections or wars, which provide insights into historical patterns in APT evolution. Over the past decade, 154 countries have been affected, primarily using malicious documents and spear phishing as dominant initial infiltration vectors, with a noticeable decline in zero-day exploitation since 2016. Furthermore, we present our findings through interactive visualization tools, such as an APT map or flow diagram, to facilitate intuitive understanding of global patterns and trends in APT activities.



## **22. OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models**

cs.CL

To appear at EMNLP 25 main conference

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2505.04416v2) [paper-pdf](http://arxiv.org/pdf/2505.04416v2)

**Authors**: Xiaoyu Xu, Minxin Du, Qingqing Ye, Haibo Hu

**Abstract**: Large language models (LLMs) trained over extensive corpora risk memorizing sensitive, copyrighted, or toxic content. To address this, we propose \textbf{OBLIVIATE}, a robust unlearning framework that removes targeted data while preserving model utility. The framework follows a structured process: extracting target tokens, building retain sets, and fine-tuning with a tailored loss function comprising three components -- masking, distillation, and world fact. Using low-rank adapters (LoRA) ensures efficiency without compromising unlearning quality. We conduct experiments on multiple datasets, including Harry Potter series, WMDP, and TOFU, using a comprehensive suite of metrics: \emph{forget quality} (via a new document-level memorization score), \emph{model utility}, and \emph{fluency}. Results demonstrate its effectiveness in resisting membership inference attacks, minimizing the impact on retained data, and maintaining robustness across diverse scenarios.



## **23. GRADA: Graph-based Reranking against Adversarial Documents Attack**

cs.IR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2505.07546v2) [paper-pdf](http://arxiv.org/pdf/2505.07546v2)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.



## **24. Personalized Attacks of Social Engineering in Multi-turn Conversations: LLM Agents for Simulation and Detection**

cs.CR

Accepted as a paper at COLM 2025 Workshop on AI Agents: Capabilities  and Safety

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2503.15552v2) [paper-pdf](http://arxiv.org/pdf/2503.15552v2)

**Authors**: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu

**Abstract**: The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the SE attack mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts.



## **25. Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models**

cs.CR

6 pages, 2 figures

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2508.17674v2) [paper-pdf](http://arxiv.org/pdf/2508.17674v2)

**Authors**: Qiming Guo, Jinwen Tang, Xingran Huang

**Abstract**: We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.



## **26. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

cs.CL

Accepted at EMNLP 2025 (Main)

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2501.01872v4) [paper-pdf](http://arxiv.org/pdf/2501.01872v4)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.



## **27. Mind Your Server: A Systematic Study of Parasitic Toolchain Attacks on the MCP Ecosystem**

cs.CR

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06572v1) [paper-pdf](http://arxiv.org/pdf/2509.06572v1)

**Authors**: Shuli Zhao, Qinsheng Hou, Zihan Zhan, Yanhao Wang, Yuchong Xie, Yu Guo, Libo Chen, Shenghong Li, Zhi Xue

**Abstract**: Large language models (LLMs) are increasingly integrated with external systems through the Model Context Protocol (MCP), which standardizes tool invocation and has rapidly become a backbone for LLM-powered applications. While this paradigm enhances functionality, it also introduces a fundamental security shift: LLMs transition from passive information processors to autonomous orchestrators of task-oriented toolchains, expanding the attack surface, elevating adversarial goals from manipulating single outputs to hijacking entire execution flows. In this paper, we reveal a new class of attacks, Parasitic Toolchain Attacks, instantiated as MCP Unintended Privacy Disclosure (MCP-UPD). These attacks require no direct victim interaction; instead, adversaries embed malicious instructions into external data sources that LLMs access during legitimate tasks. The malicious logic infiltrates the toolchain and unfolds in three phases: Parasitic Ingestion, Privacy Collection, and Privacy Disclosure, culminating in stealthy exfiltration of private data. Our root cause analysis reveals that MCP lacks both context-tool isolation and least-privilege enforcement, enabling adversarial instructions to propagate unchecked into sensitive tool invocations. To assess the severity, we design MCP-SEC and conduct the first large-scale security census of the MCP ecosystem, analyzing 12,230 tools across 1,360 servers. Our findings show that the MCP ecosystem is rife with exploitable gadgets and diverse attack methods, underscoring systemic risks in MCP platforms and the urgent need for defense mechanisms in LLM-integrated environments.



## **28. Soft Token Attacks Cannot Reliably Audit Unlearning in Large Language Models**

cs.CL

EMNLP 2025 Findings

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2502.15836v2) [paper-pdf](http://arxiv.org/pdf/2502.15836v2)

**Authors**: Haokun Chen, Sebastian Szyller, Weilin Xu, Nageen Himayat

**Abstract**: Large language models (LLMs) are trained using massive datasets, which often contain undesirable content such as harmful texts, personal information, and copyrighted material. To address this, machine unlearning aims to remove information from trained models. Recent work has shown that soft token attacks (STA) can successfully extract unlearned information from LLMs, but in this work we show that STAs can be an inadequate tool for auditing unlearning. Using common benchmarks such as Who Is Harry Potter? and TOFU, we demonstrate that in a strong auditor setting such attacks can elicit any information from the LLM, regardless of the deployed unlearning algorithm or whether the queried content was originally present in the training corpus. We further show that STA with just a few soft tokens (1-10) can elicit random strings over 400 characters long, indicating that STAs must be used carefully to effectively audit unlearning. Example code can be found at: https://github.com/IntelLabs/LLMart/tree/main/examples/unlearning



## **29. Are Your LLM-based Text-to-SQL Models Secure? Exploring SQL Injection via Backdoor Attacks**

cs.CR

Accepted to SIGMOD 2026

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2503.05445v3) [paper-pdf](http://arxiv.org/pdf/2503.05445v3)

**Authors**: Meiyu Lin, Haichuan Zhang, Jiale Lao, Renyuan Li, Yuanchun Zhou, Carl Yang, Yang Cao, Mingjie Tang

**Abstract**: Large language models (LLMs) have shown state-of-the-art results in translating natural language questions into SQL queries (Text-to-SQL), a long-standing challenge within the database community. However, security concerns remain largely unexplored, particularly the threat of backdoor attacks, which can introduce malicious behaviors into models through fine-tuning with poisoned datasets. In this work, we systematically investigate the vulnerabilities of LLM-based Text-to-SQL models and present ToxicSQL, a novel backdoor attack framework. Our approach leverages stealthy {semantic and character-level triggers} to make backdoors difficult to detect and remove, ensuring that malicious behaviors remain covert while maintaining high model accuracy on benign inputs. Furthermore, we propose leveraging SQL injection payloads as backdoor targets, enabling the generation of malicious yet executable SQL queries, which pose severe security and privacy risks in language model-based SQL development. We demonstrate that injecting only 0.44% of poisoned data can result in an attack success rate of 79.41%, posing a significant risk to database security. Additionally, we propose detection and mitigation strategies to enhance model reliability. Our findings highlight the urgent need for security-aware Text-to-SQL development, emphasizing the importance of robust defenses against backdoor threats.



## **30. Mask-GCG: Are All Tokens in Adversarial Suffixes Necessary for Jailbreak Attacks?**

cs.CL

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06350v1) [paper-pdf](http://arxiv.org/pdf/2509.06350v1)

**Authors**: Junjie Mu, Zonghao Ying, Zhekui Fan, Zonglei Jing, Yaoyuan Zhang, Zhengmin Yu, Wenxin Zhang, Quanchen Zou, Xiangzheng Zhang

**Abstract**: Jailbreak attacks on Large Language Models (LLMs) have demonstrated various successful methods whereby attackers manipulate models into generating harmful responses that they are designed to avoid. Among these, Greedy Coordinate Gradient (GCG) has emerged as a general and effective approach that optimizes the tokens in a suffix to generate jailbreakable prompts. While several improved variants of GCG have been proposed, they all rely on fixed-length suffixes. However, the potential redundancy within these suffixes remains unexplored. In this work, we propose Mask-GCG, a plug-and-play method that employs learnable token masking to identify impactful tokens within the suffix. Our approach increases the update probability for tokens at high-impact positions while pruning those at low-impact positions. This pruning not only reduces redundancy but also decreases the size of the gradient space, thereby lowering computational overhead and shortening the time required to achieve successful attacks compared to GCG. We evaluate Mask-GCG by applying it to the original GCG and several improved variants. Experimental results show that most tokens in the suffix contribute significantly to attack success, and pruning a minority of low-impact tokens does not affect the loss values or compromise the attack success rate (ASR), thereby revealing token redundancy in LLM prompts. Our findings provide insights for developing efficient and interpretable LLMs from the perspective of jailbreak attacks.



## **31. Embedding Poisoning: Bypassing Safety Alignment via Embedding Semantic Shift**

cs.CR

16 pages,9 figures

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06338v1) [paper-pdf](http://arxiv.org/pdf/2509.06338v1)

**Authors**: Shuai Yuan, Zhibo Zhang, Yuxi Li, Guangdong Bai, Wang Kailong

**Abstract**: The widespread distribution of Large Language Models (LLMs) through public platforms like Hugging Face introduces significant security challenges. While these platforms perform basic security scans, they often fail to detect subtle manipulations within the embedding layer. This work identifies a novel class of deployment phase attacks that exploit this vulnerability by injecting imperceptible perturbations directly into the embedding layer outputs without modifying model weights or input text. These perturbations, though statistically benign, systematically bypass safety alignment mechanisms and induce harmful behaviors during inference. We propose Search based Embedding Poisoning(SEP), a practical, model agnostic framework that introduces carefully optimized perturbations into embeddings associated with high risk tokens. SEP leverages a predictable linear transition in model responses, from refusal to harmful output to semantic deviation to identify a narrow perturbation window that evades alignment safeguards. Evaluated across six aligned LLMs, SEP achieves an average attack success rate of 96.43% while preserving benign task performance and evading conventional detection mechanisms. Our findings reveal a critical oversight in deployment security and emphasize the urgent need for embedding level integrity checks in future LLM defense strategies.



## **32. AttestLLM: Efficient Attestation Framework for Billion-scale On-device LLMs**

cs.CR

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06326v1) [paper-pdf](http://arxiv.org/pdf/2509.06326v1)

**Authors**: Ruisi Zhang, Yifei Zhao, Neusha Javidnia, Mengxin Zheng, Farinaz Koushanfar

**Abstract**: As on-device LLMs(e.g., Apple on-device Intelligence) are widely adopted to reduce network dependency, improve privacy, and enhance responsiveness, verifying the legitimacy of models running on local devices becomes critical. Existing attestation techniques are not suitable for billion-parameter Large Language Models (LLMs), struggling to remain both time- and memory-efficient while addressing emerging threats in the LLM era. In this paper, we present AttestLLM, the first-of-its-kind attestation framework to protect the hardware-level intellectual property (IP) of device vendors by ensuring that only authorized LLMs can execute on target platforms. AttestLLM leverages an algorithm/software/hardware co-design approach to embed robust watermarking signatures onto the activation distributions of LLM building blocks. It also optimizes the attestation protocol within the Trusted Execution Environment (TEE), providing efficient verification without compromising inference throughput. Extensive proof-of-concept evaluations on LLMs from Llama, Qwen, and Phi families for on-device use cases demonstrate AttestLLM's attestation reliability, fidelity, and efficiency. Furthermore, AttestLLM enforces model legitimacy and exhibits resilience against model replacement and forgery attacks.



## **33. Multimodal Prompt Injection Attacks: Risks and Defenses for Modern LLMs**

cs.CR

8 pages, 4 figures, 2 tables

**SubmitDate**: 2025-09-07    [abs](http://arxiv.org/abs/2509.05883v1) [paper-pdf](http://arxiv.org/pdf/2509.05883v1)

**Authors**: Andrew Yeo, Daeseon Choi

**Abstract**: Large Language Models (LLMs) have seen rapid adoption in recent years, with industries increasingly relying on them to maintain a competitive advantage. These models excel at interpreting user instructions and generating human-like responses, leading to their integration across diverse domains, including consulting and information retrieval. However, their widespread deployment also introduces substantial security risks, most notably in the form of prompt injection and jailbreak attacks.   To systematically evaluate LLM vulnerabilities -- particularly to external prompt injection -- we conducted a series of experiments on eight commercial models. Each model was tested without supplementary sanitization, relying solely on its built-in safeguards. The results exposed exploitable weaknesses and emphasized the need for stronger security measures. Four categories of attacks were examined: direct injection, indirect (external) injection, image-based injection, and prompt leakage. Comparative analysis indicated that Claude 3 demonstrated relatively greater robustness; nevertheless, empirical findings confirm that additional defenses, such as input normalization, remain necessary to achieve reliable protection.



## **34. Decoding Latent Attack Surfaces in LLMs: Prompt Injection via HTML in Web Summarization**

cs.CR

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2509.05831v1) [paper-pdf](http://arxiv.org/pdf/2509.05831v1)

**Authors**: Ishaan Verma

**Abstract**: Large Language Models (LLMs) are increasingly integrated into web-based systems for content summarization, yet their susceptibility to prompt injection attacks remains a pressing concern. In this study, we explore how non-visible HTML elements such as <meta>, aria-label, and alt attributes can be exploited to embed adversarial instructions without altering the visible content of a webpage. We introduce a novel dataset comprising 280 static web pages, evenly divided between clean and adversarial injected versions, crafted using diverse HTML-based strategies. These pages are processed through a browser automation pipeline to extract both raw HTML and rendered text, closely mimicking real-world LLM deployment scenarios. We evaluate two state-of-the-art open-source models, Llama 4 Scout (Meta) and Gemma 9B IT (Google), on their ability to summarize this content. Using both lexical (ROUGE-L) and semantic (SBERT cosine similarity) metrics, along with manual annotations, we assess the impact of these covert injections. Our findings reveal that over 29% of injected samples led to noticeable changes in the Llama 4 Scout summaries, while Gemma 9B IT showed a lower, yet non-trivial, success rate of 15%. These results highlight a critical and largely overlooked vulnerability in LLM driven web pipelines, where hidden adversarial content can subtly manipulate model outputs. Our work offers a reproducible framework and benchmark for evaluating HTML-based prompt injection and underscores the urgent need for robust mitigation strategies in LLM applications involving web content.



## **35. Membership Inference Attacks on LLM-based Recommender Systems**

cs.IR

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2508.18665v2) [paper-pdf](http://arxiv.org/pdf/2508.18665v2)

**Authors**: Jiajie He, Yuechun Gu, Min-Chun Chen, Keke Chen

**Abstract**: Large language models (LLMs) based Recommender Systems (RecSys) can flexibly adapt recommendation systems to different domains. It utilizes in-context learning (ICL), i.e., the prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, e.g., implicit feedback like clicked items or explicit product reviews. Such private information may be exposed to novel privacy attack. However, no study has been done on this important issue. We design four membership inference attacks (MIAs), aiming to reveal whether victims' historical interactions have been used by system prompts. They are \emph{direct inquiry, hallucination, similarity, and poisoning attacks}, each of which utilizes the unique features of LLMs or RecSys. We have carefully evaluated them on three LLMs that have been used to develop ICL-LLM RecSys and two well-known RecSys benchmark datasets. The results confirm that the MIA threat on LLM RecSys is realistic: direct inquiry and poisoning attacks showing significantly high attack advantages. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts and the position of the victim in the shots.



## **36. AntiDote: Bi-level Adversarial Training for Tamper-Resistant LLMs**

cs.CL

19 pages

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2509.08000v1) [paper-pdf](http://arxiv.org/pdf/2509.08000v1)

**Authors**: Debdeep Sanyal, Manodeep Ray, Murari Mandal

**Abstract**: The release of open-weight large language models (LLMs) creates a tension between advancing accessible research and preventing misuse, such as malicious fine-tuning to elicit harmful content. Current safety measures struggle to preserve the general capabilities of the LLM while resisting a determined adversary with full access to the model's weights and architecture, who can use full-parameter fine-tuning to erase existing safeguards. To address this, we introduce AntiDote, a bi-level optimization procedure for training LLMs to be resistant to such tampering. AntiDote involves an auxiliary adversary hypernetwork that learns to generate malicious Low-Rank Adaptation (LoRA) weights conditioned on the defender model's internal activations. The defender LLM is then trained with an objective to nullify the effect of these adversarial weight additions, forcing it to maintain its safety alignment. We validate this approach against a diverse suite of 52 red-teaming attacks, including jailbreak prompting, latent space manipulation, and direct weight-space attacks. AntiDote is upto 27.4\% more robust against adversarial attacks compared to both tamper-resistance and unlearning baselines. Crucially, this robustness is achieved with a minimal trade-off in utility, incurring a performance degradation of upto less than 0.5\% across capability benchmarks including MMLU, HellaSwag, and GSM8K. Our work offers a practical and compute efficient methodology for building open-weight models where safety is a more integral and resilient property.



## **37. Exploit Tool Invocation Prompt for Tool Behavior Hijacking in LLM-Based Agentic System**

cs.CR

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2509.05755v1) [paper-pdf](http://arxiv.org/pdf/2509.05755v1)

**Authors**: Yu Liu, Yuchong Xie, Mingyu Luo, Zesen Liu, Zhixiang Zhang, Kaikai Zhang, Zongjie Li, Ping Chen, Shuai Wang, Dongdong She

**Abstract**: LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems.



## **38. Reasoning Introduces New Poisoning Attacks Yet Makes Them More Complicated**

cs.CR

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2509.05739v1) [paper-pdf](http://arxiv.org/pdf/2509.05739v1)

**Authors**: Hanna Foerster, Ilia Shumailov, Yiren Zhao, Harsh Chaudhari, Jamie Hayes, Robert Mullins, Yarin Gal

**Abstract**: Early research into data poisoning attacks against Large Language Models (LLMs) demonstrated the ease with which backdoors could be injected. More recent LLMs add step-by-step reasoning, expanding the attack surface to include the intermediate chain-of-thought (CoT) and its inherent trait of decomposing problems into subproblems. Using these vectors for more stealthy poisoning, we introduce ``decomposed reasoning poison'', in which the attacker modifies only the reasoning path, leaving prompts and final answers clean, and splits the trigger across multiple, individually harmless components.   Fascinatingly, while it remains possible to inject these decomposed poisons, reliably activating them to change final answers (rather than just the CoT) is surprisingly difficult. This difficulty arises because the models can often recover from backdoors that are activated within their thought processes. Ultimately, it appears that an emergent form of backdoor robustness is originating from the reasoning capabilities of these advanced LLMs, as well as from the architectural separation between reasoning and final answer generation.



## **39. BadPromptFL: A Novel Backdoor Threat to Prompt-based Federated Learning in Multimodal Models**

cs.LG

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2508.08040v3) [paper-pdf](http://arxiv.org/pdf/2508.08040v3)

**Authors**: Maozhen Zhang, Mengnan Zhao, Wei Wang, Bo Wang

**Abstract**: Prompt-based tuning has emerged as a lightweight alternative to full fine-tuning in large vision-language models, enabling efficient adaptation via learned contextual prompts. This paradigm has recently been extended to federated learning settings (e.g., PromptFL), where clients collaboratively train prompts under data privacy constraints. However, the security implications of prompt-based aggregation in federated multimodal learning remain largely unexplored, leaving a critical attack surface unaddressed. In this paper, we introduce \textbf{BadPromptFL}, the first backdoor attack targeting prompt-based federated learning in multimodal contrastive models. In BadPromptFL, compromised clients jointly optimize local backdoor triggers and prompt embeddings, injecting poisoned prompts into the global aggregation process. These prompts are then propagated to benign clients, enabling universal backdoor activation at inference without modifying model parameters. Leveraging the contextual learning behavior of CLIP-style architectures, BadPromptFL achieves high attack success rates (e.g., \(>90\%\)) with minimal visibility and limited client participation. Extensive experiments across multiple datasets and aggregation protocols validate the effectiveness, stealth, and generalizability of our attack, raising critical concerns about the robustness of prompt-based federated learning in real-world deployments.



## **40. Evo-MARL: Co-Evolutionary Multi-Agent Reinforcement Learning for Internalized Safety**

cs.AI

accepted by the Trustworthy FMs workshop in ICCV 2025

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2508.03864v2) [paper-pdf](http://arxiv.org/pdf/2508.03864v2)

**Authors**: Zhenyu Pan, Yiting Zhang, Yutong Zhang, Jianshu Zhang, Haozheng Luo, Yuwei Han, Dennis Wu, Hong-Yu Chen, Philip S. Yu, Manling Li, Han Liu

**Abstract**: Multi-agent systems (MAS) built on multimodal large language models exhibit strong collaboration and performance. However, their growing openness and interaction complexity pose serious risks, notably jailbreak and adversarial attacks. Existing defenses typically rely on external guard modules, such as dedicated safety agents, to handle unsafe behaviors. Unfortunately, this paradigm faces two challenges: (1) standalone agents offer limited protection, and (2) their independence leads to single-point failure-if compromised, system-wide safety collapses. Naively increasing the number of guard agents further raises cost and complexity. To address these challenges, we propose Evo-MARL, a novel multi-agent reinforcement learning (MARL) framework that enables all task agents to jointly acquire defensive capabilities. Rather than relying on external safety modules, Evo-MARL trains each agent to simultaneously perform its primary function and resist adversarial threats, ensuring robustness without increasing system overhead or single-node failure. Furthermore, Evo-MARL integrates evolutionary search with parameter-sharing reinforcement learning to co-evolve attackers and defenders. This adversarial training paradigm internalizes safety mechanisms and continually enhances MAS performance under co-evolving threats. Experiments show that Evo-MARL reduces attack success rates by up to 22% while boosting accuracy by up to 5% on reasoning tasks-demonstrating that safety and utility can be jointly improved.



## **41. Differential Robustness in Transformer Language Models: Empirical Evaluation Under Adversarial Text Attacks**

cs.CR

8 pages, 4 tables, to appear in proceedings of Recent Advances in  Natural Language Processing (RANLP 2025) and ACL Anthology

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.09706v1) [paper-pdf](http://arxiv.org/pdf/2509.09706v1)

**Authors**: Taniya Gidatkar, Oluwaseun Ajao, Matthew Shardlow

**Abstract**: This study evaluates the resilience of large language models (LLMs) against adversarial attacks, specifically focusing on Flan-T5, BERT, and RoBERTa-Base. Using systematically designed adversarial tests through TextFooler and BERTAttack, we found significant variations in model robustness. RoBERTa-Base and FlanT5 demonstrated remarkable resilience, maintaining accuracy even when subjected to sophisticated attacks, with attack success rates of 0%. In contrast. BERT-Base showed considerable vulnerability, with TextFooler achieving a 93.75% success rate in reducing model accuracy from 48% to just 3%. Our research reveals that while certain LLMs have developed effective defensive mechanisms, these safeguards often require substantial computational resources. This study contributes to the understanding of LLM security by identifying existing strengths and weaknesses in current safeguarding approaches and proposes practical recommendations for developing more efficient and effective defensive strategies.



## **42. Behind the Mask: Benchmarking Camouflaged Jailbreaks in Large Language Models**

cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05471v1) [paper-pdf](http://arxiv.org/pdf/2509.05471v1)

**Authors**: Youjia Zheng, Mohammad Zandsalimy, Shanu Sushmita

**Abstract**: Large Language Models (LLMs) are increasingly vulnerable to a sophisticated form of adversarial prompting known as camouflaged jailbreaking. This method embeds malicious intent within seemingly benign language to evade existing safety mechanisms. Unlike overt attacks, these subtle prompts exploit contextual ambiguity and the flexible nature of language, posing significant challenges to current defense systems. This paper investigates the construction and impact of camouflaged jailbreak prompts, emphasizing their deceptive characteristics and the limitations of traditional keyword-based detection methods. We introduce a novel benchmark dataset, Camouflaged Jailbreak Prompts, containing 500 curated examples (400 harmful and 100 benign prompts) designed to rigorously stress-test LLM safety protocols. In addition, we propose a multi-faceted evaluation framework that measures harmfulness across seven dimensions: Safety Awareness, Technical Feasibility, Implementation Safeguards, Harmful Potential, Educational Value, Content Quality, and Compliance Score. Our findings reveal a stark contrast in LLM behavior: while models demonstrate high safety and content quality with benign inputs, they exhibit a significant decline in performance and safety when confronted with camouflaged jailbreak attempts. This disparity underscores a pervasive vulnerability, highlighting the urgent need for more nuanced and adaptive security strategies to ensure the responsible and robust deployment of LLMs in real-world applications.



## **43. Neural Breadcrumbs: Membership Inference Attacks on LLMs Through Hidden State and Attention Pattern Analysis**

cs.LG

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05449v1) [paper-pdf](http://arxiv.org/pdf/2509.05449v1)

**Authors**: Disha Makhija, Manoj Ghuhan Arivazhagan, Vinayshekhar Bannihatti Kumar, Rashmi Gangadharaiah

**Abstract**: Membership inference attacks (MIAs) reveal whether specific data was used to train machine learning models, serving as important tools for privacy auditing and compliance assessment. Recent studies have reported that MIAs perform only marginally better than random guessing against large language models, suggesting that modern pre-training approaches with massive datasets may be free from privacy leakage risks. Our work offers a complementary perspective to these findings by exploring how examining LLMs' internal representations, rather than just their outputs, may provide additional insights into potential membership inference signals. Our framework, \emph{memTrace}, follows what we call \enquote{neural breadcrumbs} extracting informative signals from transformer hidden states and attention patterns as they process candidate sequences. By analyzing layer-wise representation dynamics, attention distribution characteristics, and cross-layer transition patterns, we detect potential memorization fingerprints that traditional loss-based approaches may not capture. This approach yields strong membership detection across several model families achieving average AUC scores of 0.85 on popular MIA benchmarks. Our findings suggest that internal model behaviors can reveal aspects of training data exposure even when output-based signals appear protected, highlighting the need for further research into membership privacy and the development of more robust privacy-preserving training techniques for large language models.



## **44. AgentArmor: Enforcing Program Analysis on Agent Runtime Trace to Defend Against Prompt Injection**

cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2508.01249v2) [paper-pdf](http://arxiv.org/pdf/2508.01249v2)

**Authors**: Peiran Wang, Yang Liu, Yunfei Lu, Yifeng Cai, Hongbo Chen, Qingyou Yang, Jie Zhang, Jue Hong, Ye Wu

**Abstract**: Large Language Model (LLM) agents offer a powerful new paradigm for solving various problems by combining natural language reasoning with the execution of external tools. However, their dynamic and non-transparent behavior introduces critical security risks, particularly in the presence of prompt injection attacks. In this work, we propose a novel insight that treats the agent runtime traces as structured programs with analyzable semantics. Thus, we present AgentArmor, a program analysis framework that converts agent traces into graph intermediate representation-based structured program dependency representations (e.g., CFG, DFG, and PDG) and enforces security policies via a type system. AgentArmor consists of three key components: (1) a graph constructor that reconstructs the agent's runtime traces as graph-based intermediate representations with control and data flow described within; (2) a property registry that attaches security-relevant metadata of interacted tools \& data, and (3) a type system that performs static inference and checking over the intermediate representation. By representing agent behavior as structured programs, AgentArmor enables program analysis for sensitive data flow, trust boundaries, and policy violations. We evaluate AgentArmor on the AgentDojo benchmark, the results show that AgentArmor can reduce the ASR to 3\%, with the utility drop only 1\%.



## **45. RINSER: Accurate API Prediction Using Masked Language Models**

cs.CY

16 pages, 8 figures

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04887v1) [paper-pdf](http://arxiv.org/pdf/2509.04887v1)

**Authors**: Muhammad Ejaz Ahmed, Christopher Cody, Muhammad Ikram, Sean Lamont, Alsharif Abuadbba, Seyit Camtepe, Surya Nepal, Muhammad Ali Kaafar

**Abstract**: Malware authors commonly use obfuscation to hide API identities in binary files, making analysis difficult and time-consuming for a human expert to understand the behavior and intent of the program. Automatic API prediction tools are necessary to efficiently analyze unknown binaries, facilitating rapid malware triage while reducing the workload on human analysts. In this paper, we present RINSER (AccuRate API predictioN using maSked languagE model leaRning), an automated framework for predicting Windows API (WinAPI) function names. RINSER introduces the novel concept of API codeprints, a set of API-relevant assembly instructions, and supports x86 PE binaries. RINSER relies on BERT's masked language model (LM) to predict API names at scale, achieving 85.77% accuracy for normal binaries and 82.88% accuracy for stripped binaries. We evaluate RINSER on a large dataset of 4.7M API codeprints from 11,098 malware binaries, covering 4,123 unique Windows APIs, making it the largest publicly available dataset of this type. RINSER successfully discovered 65 obfuscated Windows APIs related to C2 communication, spying, and evasion in our dataset, which the commercial disassembler IDA failed to identify. Furthermore, we compared RINSER against three state-of-the-art approaches, showing over 20% higher prediction accuracy. We also demonstrated RINSER's resilience to adversarial attacks, including instruction randomization and code displacement, with a performance drop of no more than 3%.



## **46. Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning**

cs.AI

Rejected by AAAI25-AIA. Accepted by ICML25. Authors are thankful to  the anonymous reviewers from both AAAI25-AIA and ICML25

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2408.09600v3) [paper-pdf](http://arxiv.org/pdf/2408.09600v3)

**Authors**: Tiansheng Huang, Gautam Bhattacharya, Pratik Joshi, Josh Kimball, Ling Liu

**Abstract**: Safety aligned Large Language Models (LLMs) are vulnerable to harmful fine-tuning attacks -- a few harmful data mixed in the fine-tuning dataset can break the LLMs's safety alignment. While several defenses have been proposed, our evaluation shows that existing defenses fail \textit{when some specific training hyper-parameters are chosen} -- a large learning rate or a large number of training epochs in the fine-tuning stage can easily invalidate the defense. To this end, we propose Antidote, a post-fine-tuning stage solution, which remains \textbf{\textit{agnostic to the training hyper-parameters in the fine-tuning stage}}. Antidote relies on the philosophy that by removing the harmful parameters, the harmful model can be recovered from the harmful behaviors, regardless of how those harmful parameters are formed in the fine-tuning stage. With this philosophy, we introduce a one-shot pruning stage after harmful fine-tuning to remove the harmful weights that are responsible for the generation of harmful content. Despite its embarrassing simplicity, empirical results show that Antidote can reduce harmful score while maintaining accuracy on downstream tasks. Code is available at https://github.com/git-disl/Antidote.



## **47. Mind the Gap: Evaluating Model- and Agentic-Level Vulnerabilities in LLMs with Action Graphs**

cs.CL

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04802v1) [paper-pdf](http://arxiv.org/pdf/2509.04802v1)

**Authors**: Ilham Wicaksono, Zekun Wu, Theo King, Adriano Koshiyama, Philip Treleaven

**Abstract**: As large language models transition to agentic systems, current safety evaluation frameworks face critical gaps in assessing deployment-specific risks. We introduce AgentSeer, an observability-based evaluation framework that decomposes agentic executions into granular action and component graphs, enabling systematic agentic-situational assessment. Through cross-model validation on GPT-OSS-20B and Gemini-2.0-flash using HarmBench single turn and iterative refinement attacks, we demonstrate fundamental differences between model-level and agentic-level vulnerability profiles. Model-level evaluation reveals baseline differences: GPT-OSS-20B (39.47% ASR) versus Gemini-2.0-flash (50.00% ASR), with both models showing susceptibility to social engineering while maintaining logic-based attack resistance. However, agentic-level assessment exposes agent-specific risks invisible to traditional evaluation. We discover "agentic-only" vulnerabilities that emerge exclusively in agentic contexts, with tool-calling showing 24-60% higher ASR across both models. Cross-model analysis reveals universal agentic patterns, agent transfer operations as highest-risk tools, semantic rather than syntactic vulnerability mechanisms, and context-dependent attack effectiveness, alongside model-specific security profiles in absolute ASR levels and optimal injection strategies. Direct attack transfer from model-level to agentic contexts shows degraded performance (GPT-OSS-20B: 57% human injection ASR; Gemini-2.0-flash: 28%), while context-aware iterative attacks successfully compromise objectives that failed at model-level, confirming systematic evaluation gaps. These findings establish the urgent need for agentic-situation evaluation paradigms, with AgentSeer providing the standardized methodology and empirical validation.



## **48. Breaking to Build: A Threat Model of Prompt-Based Attacks for Securing LLMs**

cs.CL

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04615v1) [paper-pdf](http://arxiv.org/pdf/2509.04615v1)

**Authors**: Brennen Hill, Surendra Parla, Venkata Abhijeeth Balabhadruni, Atharv Prajod Padmalayam, Sujay Chandra Shekara Sharma

**Abstract**: The proliferation of Large Language Models (LLMs) has introduced critical security challenges, where adversarial actors can manipulate input prompts to cause significant harm and circumvent safety alignments. These prompt-based attacks exploit vulnerabilities in a model's design, training, and contextual understanding, leading to intellectual property theft, misinformation generation, and erosion of user trust. A systematic understanding of these attack vectors is the foundational step toward developing robust countermeasures. This paper presents a comprehensive literature survey of prompt-based attack methodologies, categorizing them to provide a clear threat model. By detailing the mechanisms and impacts of these exploits, this survey aims to inform the research community's efforts in building the next generation of secure LLMs that are inherently resistant to unauthorized distillation, fine-tuning, and editing.



## **49. Concept-ROT: Poisoning Concepts in Large Language Models with Model Editing**

cs.LG

Published at ICLR 2025

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2412.13341v2) [paper-pdf](http://arxiv.org/pdf/2412.13341v2)

**Authors**: Keltin Grimes, Marco Christiani, David Shriver, Marissa Connor

**Abstract**: Model editing methods modify specific behaviors of Large Language Models by altering a small, targeted set of network weights and require very little data and compute. These methods can be used for malicious applications such as inserting misinformation or simple trojans that result in adversary-specified behaviors when a trigger word is present. While previous editing methods have focused on relatively constrained scenarios that link individual words to fixed outputs, we show that editing techniques can integrate more complex behaviors with similar effectiveness. We develop Concept-ROT, a model editing-based method that efficiently inserts trojans which not only exhibit complex output behaviors, but also trigger on high-level concepts -- presenting an entirely new class of trojan attacks. Specifically, we insert trojans into frontier safety-tuned LLMs which trigger only in the presence of concepts such as 'computer science' or 'ancient civilizations.' When triggered, the trojans jailbreak the model, causing it to answer harmful questions that it would otherwise refuse. Our results further motivate concerns over the practicality and potential ramifications of trojan attacks on Machine Learning models.



## **50. An Automated, Scalable Machine Learning Model Inversion Assessment Pipeline**

cs.CR

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04214v1) [paper-pdf](http://arxiv.org/pdf/2509.04214v1)

**Authors**: Tyler Shumaker, Jessica Carpenter, David Saranchak, Nathaniel D. Bastian

**Abstract**: Machine learning (ML) models have the potential to transform military battlefields, presenting a large external pressure to rapidly incorporate them into operational settings. However, it is well-established that these ML models are vulnerable to a number of adversarial attacks throughout the model deployment pipeline that threaten to negate battlefield advantage. One broad category is privacy attacks (such as model inversion) where an adversary can reverse engineer information from the model, such as the sensitive data used in its training. The ability to quantify the risk of model inversion attacks (MIAs) is not well studied, and there is a lack of automated developmental test and evaluation (DT&E) tools and metrics to quantify the effectiveness of privacy loss of the MIA. The current DT&E process is difficult because ML model inversions can be hard for a human to interpret, subjective when they are interpretable, and difficult to quantify in terms of inversion quality. Additionally, scaling the DT&E process is challenging due to many ML model architectures and data modalities that need to be assessed. In this work, we present a novel DT&E tool that quantifies the risk of data privacy loss from MIAs and introduces four adversarial risk dimensions to quantify privacy loss. Our DT&E pipeline combines inversion with vision language models (VLMs) to improve effectiveness while enabling scalable analysis. We demonstrate effectiveness using multiple MIA techniques and VLMs configured for zero-shot classification and image captioning. We benchmark the pipeline using several state-of-the-art MIAs in the computer vision domain with an image classification task that is typical in military applications. In general, our innovative pipeline extends the current model inversion DT&E capabilities by improving the effectiveness and scalability of the privacy loss analysis in an automated fashion.



