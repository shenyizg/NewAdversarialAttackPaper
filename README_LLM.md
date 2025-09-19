# Latest Large Language Model Attack Papers
**update at 2025-09-19 15:27:53**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Evil Vizier: Vulnerabilities of LLM-Integrated XR Systems**

cs.CR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.15213v1) [paper-pdf](http://arxiv.org/pdf/2509.15213v1)

**Authors**: Yicheng Zhang, Zijian Huang, Sophie Chen, Erfan Shayegani, Jiasi Chen, Nael Abu-Ghazaleh

**Abstract**: Extended reality (XR) applications increasingly integrate Large Language Models (LLMs) to enhance user experience, scene understanding, and even generate executable XR content, and are often called "AI glasses". Despite these potential benefits, the integrated XR-LLM pipeline makes XR applications vulnerable to new forms of attacks. In this paper, we analyze LLM-Integated XR systems in the literature and in practice and categorize them along different dimensions from a systems perspective. Building on this categorization, we identify a common threat model and demonstrate a series of proof-of-concept attacks on multiple XR platforms that employ various LLM models (Meta Quest 3, Meta Ray-Ban, Android, and Microsoft HoloLens 2 running Llama and GPT models). Although these platforms each implement LLM integration differently, they share vulnerabilities where an attacker can modify the public context surrounding a legitimate LLM query, resulting in erroneous visual or auditory feedback to users, thus compromising their safety or privacy, sowing confusion, or other harmful effects. To defend against these threats, we discuss mitigation strategies and best practices for developers, including an initial defense prototype, and call on the community to develop new protection mechanisms to mitigate these risks.



## **2. Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating Refusal Direction**

cs.CR

Accepted by EMNLP2025 Finding

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.15202v1) [paper-pdf](http://arxiv.org/pdf/2509.15202v1)

**Authors**: Yuanbo Xie, Yingjie Zhang, Tianyun Liu, Duohe Ma, Tingwen Liu

**Abstract**: Jailbreak attacks pose persistent threats to large language models (LLMs). Current safety alignment methods have attempted to address these issues, but they experience two significant limitations: insufficient safety alignment depth and unrobust internal defense mechanisms. These limitations make them vulnerable to adversarial attacks such as prefilling and refusal direction manipulation. We introduce DeepRefusal, a robust safety alignment framework that overcomes these issues. DeepRefusal forces the model to dynamically rebuild its refusal mechanisms from jailbreak states. This is achieved by probabilistically ablating the refusal direction across layers and token depths during fine-tuning. Our method not only defends against prefilling and refusal direction attacks but also demonstrates strong resilience against other unseen jailbreak strategies. Extensive evaluations on four open-source LLM families and six representative attacks show that DeepRefusal reduces attack success rates by approximately 95%, while maintaining model capabilities with minimal performance degradation.



## **3. Exploit Tool Invocation Prompt for Tool Behavior Hijacking in LLM-Based Agentic System**

cs.CR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.05755v3) [paper-pdf](http://arxiv.org/pdf/2509.05755v3)

**Authors**: Yuchong Xie, Mingyu Luo, Zesen Liu, Zhixiang Zhang, Kaikai Zhang, Yu Liu, Zongjie Li, Ping Chen, Shuai Wang, Dongdong She

**Abstract**: LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems.



## **4. QA-LIGN: Aligning LLMs through Constitutionally Decomposed QA**

cs.CL

Accepted to Findings of EMNLP 2025

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2506.08123v3) [paper-pdf](http://arxiv.org/pdf/2506.08123v3)

**Authors**: Jacob Dineen, Aswin RRV, Qin Liu, Zhikun Xu, Xiao Ye, Ming Shen, Zhaonan Li, Shijie Lu, Chitta Baral, Muhao Chen, Ben Zhou

**Abstract**: Alignment of large language models (LLMs) with principles like helpfulness, honesty, and harmlessness typically relies on scalar rewards that obscure which objectives drive the training signal. We introduce QA-LIGN, which decomposes monolithic rewards into interpretable principle-specific evaluations through structured natural language programs. Models learn through a draft, critique, and revise pipeline, where symbolic evaluation against the rubrics provides transparent feedback for both initial and revised responses during GRPO training. Applied to uncensored Llama-3.1-8B-Instruct, QA-LIGN reduces attack success rates by up to 68.7% while maintaining a 0.67% false refusal rate, achieving Pareto optimal safety-helpfulness performance and outperforming both DPO and GRPO with state-of-the-art reward models given equivalent training. These results demonstrate that making reward signals interpretable and modular improves alignment effectiveness, suggesting transparency enhances LLM safety.



## **5. AIP: Subverting Retrieval-Augmented Generation via Adversarial Instructional Prompt**

cs.CV

Accepted at EMNLP 2025 Conference

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.15159v1) [paper-pdf](http://arxiv.org/pdf/2509.15159v1)

**Authors**: Saket S. Chaturvedi, Gaurav Bagwe, Lan Zhang, Xiaoyong Yuan

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by retrieving relevant documents from external sources to improve factual accuracy and verifiability. However, this reliance introduces new attack surfaces within the retrieval pipeline, beyond the LLM itself. While prior RAG attacks have exposed such vulnerabilities, they largely rely on manipulating user queries, which is often infeasible in practice due to fixed or protected user inputs. This narrow focus overlooks a more realistic and stealthy vector: instructional prompts, which are widely reused, publicly shared, and rarely audited. Their implicit trust makes them a compelling target for adversaries to manipulate RAG behavior covertly.   We introduce a novel attack for Adversarial Instructional Prompt (AIP) that exploits adversarial instructional prompts to manipulate RAG outputs by subtly altering retrieval behavior. By shifting the attack surface to the instructional prompts, AIP reveals how trusted yet seemingly benign interface components can be weaponized to degrade system integrity. The attack is crafted to achieve three goals: (1) naturalness, to evade user detection; (2) utility, to encourage use of prompts; and (3) robustness, to remain effective across diverse query variations. We propose a diverse query generation strategy that simulates realistic linguistic variation in user queries, enabling the discovery of prompts that generalize across paraphrases and rephrasings. Building on this, a genetic algorithm-based joint optimization is developed to evolve adversarial prompts by balancing attack success, clean-task utility, and stealthiness. Experimental results show that AIP achieves up to 95.23% ASR while preserving benign functionality. These findings uncover a critical and previously overlooked vulnerability in RAG systems, emphasizing the need to reassess the shared instructional prompts.



## **6. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

cs.CV

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2409.13174v3) [paper-pdf](http://arxiv.org/pdf/2409.13174v3)

**Authors**: Hao Cheng, Erjia Xiao, Yichi Wang, Chengyuan Yu, Mengshu Sun, Qiang Zhang, Yijie Guo, Kaidi Xu, Jize Zhang, Chao Shen, Philip Torr, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompt, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable \textbf{\textit{Analyses}} of how VLAMs respond to different physical threats.



## **7. Sentinel Agents for Secure and Trustworthy Agentic AI in Multi-Agent Systems**

cs.AI

25 pages, 12 figures

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14956v1) [paper-pdf](http://arxiv.org/pdf/2509.14956v1)

**Authors**: Diego Gosmar, Deborah A. Dahl

**Abstract**: This paper proposes a novel architectural framework aimed at enhancing security and reliability in multi-agent systems (MAS). A central component of this framework is a network of Sentinel Agents, functioning as a distributed security layer that integrates techniques such as semantic analysis via large language models (LLMs), behavioral analytics, retrieval-augmented verification, and cross-agent anomaly detection. Such agents can potentially oversee inter-agent communications, identify potential threats, enforce privacy and access controls, and maintain comprehensive audit records. Complementary to the idea of Sentinel Agents is the use of a Coordinator Agent. The Coordinator Agent supervises policy implementation, and manages agent participation. In addition, the Coordinator also ingests alerts from Sentinel Agents. Based on these alerts, it can adapt policies, isolate or quarantine misbehaving agents, and contain threats to maintain the integrity of the MAS ecosystem. This dual-layered security approach, combining the continuous monitoring of Sentinel Agents with the governance functions of Coordinator Agents, supports dynamic and adaptive defense mechanisms against a range of threats, including prompt injection, collusive agent behavior, hallucinations generated by LLMs, privacy breaches, and coordinated multi-agent attacks. In addition to the architectural design, we present a simulation study where 162 synthetic attacks of different families (prompt injection, hallucination, and data exfiltration) were injected into a multi-agent conversational environment. The Sentinel Agents successfully detected the attack attempts, confirming the practical feasibility of the proposed monitoring approach. The framework also offers enhanced system observability, supports regulatory compliance, and enables policy evolution over time.



## **8. MUSE: MCTS-Driven Red Teaming Framework for Enhanced Multi-Turn Dialogue Safety in Large Language Models**

cs.CL

EMNLP 2025 main conference

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14651v1) [paper-pdf](http://arxiv.org/pdf/2509.14651v1)

**Authors**: Siyu Yan, Long Zeng, Xuecheng Wu, Chengcheng Han, Kongcheng Zhang, Chong Peng, Xuezhi Cao, Xunliang Cai, Chenjuan Guo

**Abstract**: As large language models~(LLMs) become widely adopted, ensuring their alignment with human values is crucial to prevent jailbreaks where adversaries manipulate models to produce harmful content. While most defenses target single-turn attacks, real-world usage often involves multi-turn dialogues, exposing models to attacks that exploit conversational context to bypass safety measures. We introduce MUSE, a comprehensive framework tackling multi-turn jailbreaks from both attack and defense angles. For attacks, we propose MUSE-A, a method that uses frame semantics and heuristic tree search to explore diverse semantic trajectories. For defense, we present MUSE-D, a fine-grained safety alignment approach that intervenes early in dialogues to reduce vulnerabilities. Extensive experiments on various models show that MUSE effectively identifies and mitigates multi-turn vulnerabilities. Code is available at \href{https://github.com/yansiyu02/MUSE}{https://github.com/yansiyu02/MUSE}.



## **9. Enterprise AI Must Enforce Participant-Aware Access Control**

cs.CR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14608v1) [paper-pdf](http://arxiv.org/pdf/2509.14608v1)

**Authors**: Shashank Shreedhar Bhatt, Tanmay Rajore, Khushboo Aggarwal, Ganesh Ananthanarayanan, Ranveer Chandra, Nishanth Chandran, Suyash Choudhury, Divya Gupta, Emre Kiciman, Sumit Kumar Pandey, Srinath Setty, Rahul Sharma, Teijia Zhao

**Abstract**: Large language models (LLMs) are increasingly deployed in enterprise settings where they interact with multiple users and are trained or fine-tuned on sensitive internal data. While fine-tuning enhances performance by internalizing domain knowledge, it also introduces a critical security risk: leakage of confidential training data to unauthorized users. These risks are exacerbated when LLMs are combined with Retrieval-Augmented Generation (RAG) pipelines that dynamically fetch contextual documents at inference time.   We demonstrate data exfiltration attacks on AI assistants where adversaries can exploit current fine-tuning and RAG architectures to leak sensitive information by leveraging the lack of access control enforcement. We show that existing defenses, including prompt sanitization, output filtering, system isolation, and training-level privacy mechanisms, are fundamentally probabilistic and fail to offer robust protection against such attacks.   We take the position that only a deterministic and rigorous enforcement of fine-grained access control during both fine-tuning and RAG-based inference can reliably prevent the leakage of sensitive data to unauthorized recipients.   We introduce a framework centered on the principle that any content used in training, retrieval, or generation by an LLM is explicitly authorized for \emph{all users involved in the interaction}. Our approach offers a simple yet powerful paradigm shift for building secure multi-user LLM systems that are grounded in classical access control but adapted to the unique challenges of modern AI workflows. Our solution has been deployed in Microsoft Copilot Tuning, a product offering that enables organizations to fine-tune models using their own enterprise-specific data.



## **10. Reconstruction of Differentially Private Text Sanitization via Large Language Models**

cs.CR

RAID-2025

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2410.12443v3) [paper-pdf](http://arxiv.org/pdf/2410.12443v3)

**Authors**: Shuchao Pang, Zhigang Lu, Haichen Wang, Peng Fu, Yongbin Zhou, Minhui Xue

**Abstract**: Differential privacy (DP) is the de facto privacy standard against privacy leakage attacks, including many recently discovered ones against large language models (LLMs). However, we discovered that LLMs could reconstruct the altered/removed privacy from given DP-sanitized prompts. We propose two attacks (black-box and white-box) based on the accessibility to LLMs and show that LLMs could connect the pair of DP-sanitized text and the corresponding private training data of LLMs by giving sample text pairs as instructions (in the black-box attacks) or fine-tuning data (in the white-box attacks). To illustrate our findings, we conduct comprehensive experiments on modern LLMs (e.g., LLaMA-2, LLaMA-3, ChatGPT-3.5, ChatGPT-4, ChatGPT-4o, Claude-3, Claude-3.5, OPT, GPT-Neo, GPT-J, Gemma-2, and Pythia) using commonly used datasets (such as WikiMIA, Pile-CC, and Pile-Wiki) against both word-level and sentence-level DP. The experimental results show promising recovery rates, e.g., the black-box attacks against the word-level DP over WikiMIA dataset gave 72.18% on LLaMA-2 (70B), 82.39% on LLaMA-3 (70B), 75.35% on Gemma-2, 91.2% on ChatGPT-4o, and 94.01% on Claude-3.5 (Sonnet). More urgently, this study indicates that these well-known LLMs have emerged as a new security risk for existing DP text sanitization approaches in the current environment.



## **11. SynBench: A Benchmark for Differentially Private Text Generation**

cs.AI

15 pages

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14594v1) [paper-pdf](http://arxiv.org/pdf/2509.14594v1)

**Authors**: Yidan Sun, Viktor Schlegel, Srinivasan Nandakumar, Iqra Zahid, Yuping Wu, Yulong Wu, Hao Li, Jie Zhang, Warren Del-Pinto, Goran Nenadic, Siew Kei Lam, Anil Anthony Bharath

**Abstract**: Data-driven decision support in high-stakes domains like healthcare and finance faces significant barriers to data sharing due to regulatory, institutional, and privacy concerns. While recent generative AI models, such as large language models, have shown impressive performance in open-domain tasks, their adoption in sensitive environments remains limited by unpredictable behaviors and insufficient privacy-preserving datasets for benchmarking. Existing anonymization methods are often inadequate, especially for unstructured text, as redaction and masking can still allow re-identification. Differential Privacy (DP) offers a principled alternative, enabling the generation of synthetic data with formal privacy assurances. In this work, we address these challenges through three key contributions. First, we introduce a comprehensive evaluation framework with standardized utility and fidelity metrics, encompassing nine curated datasets that capture domain-specific complexities such as technical jargon, long-context dependencies, and specialized document structures. Second, we conduct a large-scale empirical study benchmarking state-of-the-art DP text generation methods and LLMs of varying sizes and different fine-tuning strategies, revealing that high-quality domain-specific synthetic data generation under DP constraints remains an unsolved challenge, with performance degrading as domain complexity increases. Third, we develop a membership inference attack (MIA) methodology tailored for synthetic text, providing first empirical evidence that the use of public datasets - potentially present in pre-training corpora - can invalidate claimed privacy guarantees. Our findings underscore the urgent need for rigorous privacy auditing and highlight persistent gaps between open-domain and specialist evaluations, informing responsible deployment of generative AI in privacy-sensitive, high-stakes settings.



## **12. LLM Jailbreak Detection for (Almost) Free!**

cs.CR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14558v1) [paper-pdf](http://arxiv.org/pdf/2509.14558v1)

**Authors**: Guorui Chen, Yifan Xia, Xiaojun Jia, Zhijiang Li, Philip Torr, Jindong Gu

**Abstract**: Large language models (LLMs) enhance security through alignment when widely used, but remain susceptible to jailbreak attacks capable of producing inappropriate content. Jailbreak detection methods show promise in mitigating jailbreak attacks through the assistance of other models or multiple model inferences. However, existing methods entail significant computational costs. In this paper, we first present a finding that the difference in output distributions between jailbreak and benign prompts can be employed for detecting jailbreak prompts. Based on this finding, we propose a Free Jailbreak Detection (FJD) which prepends an affirmative instruction to the input and scales the logits by temperature to further distinguish between jailbreak and benign prompts through the confidence of the first token. Furthermore, we enhance the detection performance of FJD through the integration of virtual instruction learning. Extensive experiments on aligned LLMs show that our FJD can effectively detect jailbreak prompts with almost no additional computational costs during LLM inference.



## **13. GRADA: Graph-based Reranking against Adversarial Documents Attack**

cs.IR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2505.07546v3) [paper-pdf](http://arxiv.org/pdf/2505.07546v3)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.



## **14. Benchmarking Large Language Models for Cryptanalysis and Side-Channel Vulnerabilities**

cs.CL

EMNLP'25 Findings

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2505.24621v2) [paper-pdf](http://arxiv.org/pdf/2505.24621v2)

**Authors**: Utsav Maskey, Chencheng Zhu, Usman Naseem

**Abstract**: Recent advancements in large language models (LLMs) have transformed natural language understanding and generation, leading to extensive benchmarking across diverse tasks. However, cryptanalysis - a critical area for data security and its connection to LLMs' generalization abilities - remains underexplored in LLM evaluations. To address this gap, we evaluate the cryptanalytic potential of state-of-the-art LLMs on ciphertexts produced by a range of cryptographic algorithms. We introduce a benchmark dataset of diverse plaintexts, spanning multiple domains, lengths, writing styles, and topics, paired with their encrypted versions. Using zero-shot and few-shot settings along with chain-of-thought prompting, we assess LLMs' decryption success rate and discuss their comprehension abilities. Our findings reveal key insights into LLMs' strengths and limitations in side-channel scenarios and raise concerns about their susceptibility to under-generalization-related attacks. This research highlights the dual-use nature of LLMs in security contexts and contributes to the ongoing discussion on AI safety and security.



## **15. Evaluating and Improving the Robustness of Security Attack Detectors Generated by LLMs**

cs.SE

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2411.18216v2) [paper-pdf](http://arxiv.org/pdf/2411.18216v2)

**Authors**: Samuele Pasini, Jinhan Kim, Tommaso Aiello, Rocio Cabrera Lozoya, Antonino Sabetta, Paolo Tonella

**Abstract**: Large Language Models (LLMs) are increasingly used in software development to generate functions, such as attack detectors, that implement security requirements. A key challenge is ensuring the LLMs have enough knowledge to address specific security requirements, such as information about existing attacks. For this, we propose an approach integrating Retrieval Augmented Generation (RAG) and Self-Ranking into the LLM pipeline. RAG enhances the robustness of the output by incorporating external knowledge sources, while the Self-Ranking technique, inspired by the concept of Self-Consistency, generates multiple reasoning paths and creates ranks to select the most robust detector. Our extensive empirical study targets code generated by LLMs to detect two prevalent injection attacks in web security: Cross-Site Scripting (XSS) and SQL injection (SQLi). Results show a significant improvement in detection performance while employing RAG and Self-Ranking, with an increase of up to 71%pt (on average 37%pt) and up to 43%pt (on average 6%pt) in the F2-Score for XSS and SQLi detection, respectively.



## **16. CyberLLMInstruct: A Pseudo-malicious Dataset Revealing Safety-performance Trade-offs in Cyber Security LLM Fine-tuning**

cs.CR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2503.09334v3) [paper-pdf](http://arxiv.org/pdf/2503.09334v3)

**Authors**: Adel ElZemity, Budi Arief, Shujun Li

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents both opportunities and critical safety risks. We introduce CyberLLMInstruct, a dataset of 54,928 pseudo-malicious instruction-response pairs spanning cyber security tasks including malware analysis, phishing simulations, and zero-day vulnerabilities. Our comprehensive evaluation using seven open-source LLMs reveals a critical trade-off: while fine-tuning improves cyber security task performance (achieving up to 92.50% accuracy on CyberMetric), it severely compromises safety resilience across all tested models and attack vectors (e.g., Llama 3.1 8B's security score against prompt injection drops from 0.95 to 0.15). The dataset incorporates diverse sources including CTF challenges, academic papers, industry reports, and CVE databases to ensure comprehensive coverage of cyber security domains. Our findings highlight the unique challenges of securing LLMs in adversarial domains and establish the critical need for developing fine-tuning methodologies that balance performance gains with safety preservation in security-sensitive domains.



## **17. Do LLMs Align Human Values Regarding Social Biases? Judging and Explaining Social Biases with LLMs**

cs.CL

38 pages, 31 figures

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.13869v1) [paper-pdf](http://arxiv.org/pdf/2509.13869v1)

**Authors**: Yang Liu, Chenhui Chu

**Abstract**: Large language models (LLMs) can lead to undesired consequences when misaligned with human values, especially in scenarios involving complex and sensitive social biases. Previous studies have revealed the misalignment of LLMs with human values using expert-designed or agent-based emulated bias scenarios. However, it remains unclear whether the alignment of LLMs with human values differs across different types of scenarios (e.g., scenarios containing negative vs. non-negative questions). In this study, we investigate the alignment of LLMs with human values regarding social biases (HVSB) in different types of bias scenarios. Through extensive analysis of 12 LLMs from four model families and four datasets, we demonstrate that LLMs with large model parameter scales do not necessarily have lower misalignment rate and attack success rate. Moreover, LLMs show a certain degree of alignment preference for specific types of scenarios and the LLMs from the same model family tend to have higher judgment consistency. In addition, we study the understanding capacity of LLMs with their explanations of HVSB. We find no significant differences in the understanding of HVSB across LLMs. We also find LLMs prefer their own generated explanations. Additionally, we endow smaller language models (LMs) with the ability to explain HVSB. The generation results show that the explanations generated by the fine-tuned smaller LMs are more readable, but have a relatively lower model agreeability.



## **18. Defending against Indirect Prompt Injection by Instruction Detection**

cs.CR

16 pages, 4 figures

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2505.06311v2) [paper-pdf](http://arxiv.org/pdf/2505.06311v2)

**Authors**: Tongyu Wen, Chenglong Wang, Xiyuan Yang, Haoyu Tang, Yueqi Xie, Lingjuan Lyu, Zhicheng Dou, Fangzhao Wu

**Abstract**: The integration of Large Language Models (LLMs) with external sources is becoming increasingly common, with Retrieval-Augmented Generation (RAG) being a prominent example. However, this integration introduces vulnerabilities of Indirect Prompt Injection (IPI) attacks, where hidden instructions embedded in external data can manipulate LLMs into executing unintended or harmful actions. We recognize that IPI attacks fundamentally rely on the presence of instructions embedded within external content, which can alter the behavioral states of LLMs. Can the effective detection of such state changes help us defend against IPI attacks? In this paper, we propose InstructDetector, a novel detection-based approach that leverages the behavioral states of LLMs to identify potential IPI attacks. Specifically, we demonstrate the hidden states and gradients from intermediate layers provide highly discriminative features for instruction detection. By effectively combining these features, InstructDetector achieves a detection accuracy of 99.60% in the in-domain setting and 96.90% in the out-of-domain setting, and reduces the attack success rate to just 0.03% on the BIPIA benchmark. The code is publicly available at https://github.com/MYVAE/Instruction-detection.



## **19. DiffHash: Text-Guided Targeted Attack via Diffusion Models against Deep Hashing Image Retrieval**

cs.IR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.12824v2) [paper-pdf](http://arxiv.org/pdf/2509.12824v2)

**Authors**: Zechao Liu, Zheng Zhou, Xiangkun Chen, Tao Liang, Dapeng Lang

**Abstract**: Deep hashing models have been widely adopted to tackle the challenges of large-scale image retrieval. However, these approaches face serious security risks due to their vulnerability to adversarial examples. Despite the increasing exploration of targeted attacks on deep hashing models, existing approaches still suffer from a lack of multimodal guidance, reliance on labeling information and dependence on pixel-level operations for attacks. To address these limitations, we proposed DiffHash, a novel diffusion-based targeted attack for deep hashing. Unlike traditional pixel-based attacks that directly modify specific pixels and lack multimodal guidance, our approach focuses on optimizing the latent representations of images, guided by text information generated by a Large Language Model (LLM) for the target image. Furthermore, we designed a multi-space hash alignment network to align the high-dimension image space and text space to the low-dimension binary hash space. During reconstruction, we also incorporated text-guided attention mechanisms to refine adversarial examples, ensuring them aligned with the target semantics while maintaining visual plausibility. Extensive experiments have demonstrated that our method outperforms state-of-the-art (SOTA) targeted attack methods, achieving better black-box transferability and offering more excellent stability across datasets.



## **20. Who Taught the Lie? Responsibility Attribution for Poisoned Knowledge in Retrieval-Augmented Generation**

cs.CR

To appear in the IEEE Symposium on Security and Privacy, 2026

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.13772v1) [paper-pdf](http://arxiv.org/pdf/2509.13772v1)

**Authors**: Baolei Zhang, Haoran Xin, Yuxi Chen, Zhuqing Liu, Biao Yi, Tong Li, Lihai Nie, Zheli Liu, Minghong Fang

**Abstract**: Retrieval-Augmented Generation (RAG) integrates external knowledge into large language models to improve response quality. However, recent work has shown that RAG systems are highly vulnerable to poisoning attacks, where malicious texts are inserted into the knowledge database to influence model outputs. While several defenses have been proposed, they are often circumvented by more adaptive or sophisticated attacks.   This paper presents RAGOrigin, a black-box responsibility attribution framework designed to identify which texts in the knowledge database are responsible for misleading or incorrect generations. Our method constructs a focused attribution scope tailored to each misgeneration event and assigns a responsibility score to each candidate text by evaluating its retrieval ranking, semantic relevance, and influence on the generated response. The system then isolates poisoned texts using an unsupervised clustering method. We evaluate RAGOrigin across seven datasets and fifteen poisoning attacks, including newly developed adaptive poisoning strategies and multi-attacker scenarios. Our approach outperforms existing baselines in identifying poisoned content and remains robust under dynamic and noisy conditions. These results suggest that RAGOrigin provides a practical and effective solution for tracing the origins of corrupted knowledge in RAG systems.



## **21. A Simple and Efficient Jailbreak Method Exploiting LLMs' Helpfulness**

cs.CR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.14297v1) [paper-pdf](http://arxiv.org/pdf/2509.14297v1)

**Authors**: Xuan Luo, Yue Wang, Zefeng He, Geng Tu, Jing Li, Ruifeng Xu

**Abstract**: Safety alignment aims to prevent Large Language Models (LLMs) from responding to harmful queries. To strengthen safety protections, jailbreak methods are developed to simulate malicious attacks and uncover vulnerabilities. In this paper, we introduce HILL (Hiding Intention by Learning from LLMs), a novel jailbreak approach that systematically transforms imperative harmful requests into learning-style questions with only straightforward hypotheticality indicators. Further, we introduce two new metrics to thoroughly evaluate the utility of jailbreak methods. Experiments on the AdvBench dataset across a wide range of models demonstrate HILL's strong effectiveness, generalizability, and harmfulness. It achieves top attack success rates on the majority of models and across malicious categories while maintaining high efficiency with concise prompts. Results of various defense methods show the robustness of HILL, with most defenses having mediocre effects or even increasing the attack success rates. Moreover, the assessment on our constructed safe prompts reveals inherent limitations of LLMs' safety mechanisms and flaws in defense methods. This work exposes significant vulnerabilities of safety measures against learning-style elicitation, highlighting a critical challenge of balancing helpfulness and safety alignments.



## **22. SoK: How Sensor Attacks Disrupt Autonomous Vehicles: An End-to-end Analysis, Challenges, and Missed Threats**

cs.CR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.11120v3) [paper-pdf](http://arxiv.org/pdf/2509.11120v3)

**Authors**: Qingzhao Zhang, Shaocheng Luo, Z. Morley Mao, Miroslav Pajic, Michael K. Reiter

**Abstract**: Autonomous vehicles, including self-driving cars, robotic ground vehicles, and drones, rely on complex sensor pipelines to ensure safe and reliable operation. However, these safety-critical systems remain vulnerable to adversarial sensor attacks that can compromise their performance and mission success. While extensive research has demonstrated various sensor attack techniques, critical gaps remain in understanding their feasibility in real-world, end-to-end systems. This gap largely stems from the lack of a systematic perspective on how sensor errors propagate through interconnected modules in autonomous systems when autonomous vehicles interact with the physical world.   To bridge this gap, we present a comprehensive survey of autonomous vehicle sensor attacks across platforms, sensor modalities, and attack methods. Central to our analysis is the System Error Propagation Graph (SEPG), a structured demonstration tool that illustrates how sensor attacks propagate through system pipelines, exposing the conditions and dependencies that determine attack feasibility. With the aid of SEPG, our study distills seven key findings that highlight the feasibility challenges of sensor attacks and uncovers eleven previously overlooked attack vectors exploiting inter-module interactions, several of which we validate through proof-of-concept experiments. Additionally, we demonstrate how large language models (LLMs) can automate aspects of SEPG construction and cross-validate expert analysis, showcasing the promise of AI-assisted security evaluation.



## **23. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

cs.CL

Accepted at EMNLP 2025 (Main)

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2501.01872v5) [paper-pdf](http://arxiv.org/pdf/2501.01872v5)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.



## **24. From Capabilities to Performance: Evaluating Key Functional Properties of LLM Architectures in Penetration Testing**

cs.AI

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.14289v1) [paper-pdf](http://arxiv.org/pdf/2509.14289v1)

**Authors**: Lanxiao Huang, Daksh Dave, Ming Jin, Tyler Cody, Peter Beling

**Abstract**: Large language models (LLMs) are increasingly used to automate or augment penetration testing, but their effectiveness and reliability across attack phases remain unclear. We present a comprehensive evaluation of multiple LLM-based agents, from single-agent to modular designs, across realistic penetration testing scenarios, measuring empirical performance and recurring failure patterns. We also isolate the impact of five core functional capabilities via targeted augmentations: Global Context Memory (GCM), Inter-Agent Messaging (IAM), Context-Conditioned Invocation (CCI), Adaptive Planning (AP), and Real-Time Monitoring (RTM). These interventions support, respectively: (i) context coherence and retention, (ii) inter-component coordination and state management, (iii) tool use accuracy and selective execution, (iv) multi-step strategic planning, error detection, and recovery, and (v) real-time dynamic responsiveness. Our results show that while some architectures natively exhibit subsets of these properties, targeted augmentations substantially improve modular agent performance, especially in complex, multi-step, and real-time penetration testing tasks.



## **25. AQUA-LLM: Evaluating Accuracy, Quantization, and Adversarial Robustness Trade-offs in LLMs for Cybersecurity Question Answering**

cs.CR

Accepted by the 24th IEEE International Conference on Machine  Learning and Applications (ICMLA'25)

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.13514v1) [paper-pdf](http://arxiv.org/pdf/2509.13514v1)

**Authors**: Onat Gungor, Roshan Sood, Harold Wang, Tajana Rosing

**Abstract**: Large Language Models (LLMs) have recently demonstrated strong potential for cybersecurity question answering (QA), supporting decision-making in real-time threat detection and response workflows. However, their substantial computational demands pose significant challenges for deployment on resource-constrained edge devices. Quantization, a widely adopted model compression technique, can alleviate these constraints. Nevertheless, quantization may degrade model accuracy and increase susceptibility to adversarial attacks. Fine-tuning offers a potential means to mitigate these limitations, but its effectiveness when combined with quantization remains insufficiently explored. Hence, it is essential to understand the trade-offs among accuracy, efficiency, and robustness. We propose AQUA-LLM, an evaluation framework designed to benchmark several state-of-the-art small LLMs under four distinct configurations: base, quantized-only, fine-tuned, and fine-tuned combined with quantization, specifically for cybersecurity QA. Our results demonstrate that quantization alone yields the lowest accuracy and robustness despite improving efficiency. In contrast, combining quantization with fine-tuning enhances both LLM robustness and predictive performance, achieving an optimal balance of accuracy, robustness, and efficiency. These findings highlight the critical need for quantization-aware, robustness-preserving fine-tuning methodologies to enable the robust and efficient deployment of LLMs for cybersecurity QA.



## **26. A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks**

cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.14285v1) [paper-pdf](http://arxiv.org/pdf/2509.14285v1)

**Authors**: S M Asif Hossain, Ruksat Khan Shayoni, Mohd Ruhul Ameen, Akif Islam, M. F. Mridha, Jungpil Shin

**Abstract**: Prompt injection attacks represent a major vulnerability in Large Language Model (LLM) deployments, where malicious instructions embedded in user inputs can override system prompts and induce unintended behaviors. This paper presents a novel multi-agent defense framework that employs specialized LLM agents in coordinated pipelines to detect and neutralize prompt injection attacks in real-time. We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system. Our comprehensive evaluation on 55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2), demonstrates significant security improvements. Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0% across all tested scenarios. The framework demonstrates robustness across multiple attack categories including direct overrides, code execution attempts, data exfiltration, and obfuscation techniques, while maintaining system functionality for legitimate queries.



## **27. TrojanRobot: Physical-world Backdoor Attacks Against VLM-based Robotic Manipulation**

cs.RO

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2411.11683v5) [paper-pdf](http://arxiv.org/pdf/2411.11683v5)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Aishan Liu, Yunpeng Jiang, Leo Yu Zhang, Xiaohua Jia

**Abstract**: Robotic manipulation in the physical world is increasingly empowered by \textit{large language models} (LLMs) and \textit{vision-language models} (VLMs), leveraging their understanding and perception capabilities. Recently, various attacks against such robotic policies have been proposed, with backdoor attacks drawing considerable attention for their high stealth and strong persistence capabilities. However, existing backdoor efforts are limited to simulators and suffer from physical-world realization. To address this, we propose \textit{TrojanRobot}, a highly stealthy and broadly effective robotic backdoor attack in the physical world. Specifically, we introduce a module-poisoning approach by embedding a backdoor module into the modular robotic policy, enabling backdoor control over the policy's visual perception module thereby backdooring the entire robotic policy. Our vanilla implementation leverages a backdoor-finetuned VLM to serve as the backdoor module. To enhance its generalization in physical environments, we propose a prime implementation, leveraging the LVLM-as-a-backdoor paradigm and developing three types of prime attacks, \ie, \textit{permutation}, \textit{stagnation}, and \textit{intentional} attacks, thus achieving finer-grained backdoors. Extensive experiments on the UR3e manipulator with 18 task instructions using robotic policies based on four VLMs demonstrate the broad effectiveness and physical-world stealth of TrojanRobot. Our attack's video demonstrations are available via a github link https://trojanrobot.github.io.



## **28. Context-Aware Membership Inference Attacks against Pre-trained Large Language Models**

cs.CL

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2409.13745v2) [paper-pdf](http://arxiv.org/pdf/2409.13745v2)

**Authors**: Hongyan Chang, Ali Shahin Shamsabadi, Kleomenis Katevas, Hamed Haddadi, Reza Shokri

**Abstract**: Membership Inference Attacks (MIAs) on pre-trained Large Language Models (LLMs) aim at determining if a data point was part of the model's training set. Prior MIAs that are built for classification models fail at LLMs, due to ignoring the generative nature of LLMs across token sequences. In this paper, we present a novel attack on pre-trained LLMs that adapts MIA statistical tests to the perplexity dynamics of subsequences within a data point. Our method significantly outperforms prior approaches, revealing context-dependent memorization patterns in pre-trained LLMs.



## **29. Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection**

cs.CL

EMNLP 2025 Main (Oral)

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2502.13061v4) [paper-pdf](http://arxiv.org/pdf/2502.13061v4)

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL



## **30. Beyond Data Privacy: New Privacy Risks for Large Language Models**

cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.14278v1) [paper-pdf](http://arxiv.org/pdf/2509.14278v1)

**Authors**: Yuntao Du, Zitao Li, Ninghui Li, Bolin Ding

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress in natural language understanding, reasoning, and autonomous decision-making. However, these advancements have also come with significant privacy concerns. While significant research has focused on mitigating the data privacy risks of LLMs during various stages of model training, less attention has been paid to new threats emerging from their deployment. The integration of LLMs into widely used applications and the weaponization of their autonomous abilities have created new privacy vulnerabilities. These vulnerabilities provide opportunities for both inadvertent data leakage and malicious exfiltration from LLM-powered systems. Additionally, adversaries can exploit these systems to launch sophisticated, large-scale privacy attacks, threatening not only individual privacy but also financial security and societal trust. In this paper, we systematically examine these emerging privacy risks of LLMs. We also discuss potential mitigation strategies and call for the research community to broaden its focus beyond data privacy risks, developing new defenses to address the evolving threats posed by increasingly powerful LLMs and LLM-powered systems.



## **31. Adversarial Prompt Distillation for Vision-Language Models**

cs.CV

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2411.15244v3) [paper-pdf](http://arxiv.org/pdf/2411.15244v3)

**Authors**: Lin Luo, Xin Wang, Bojia Zi, Shihao Zhao, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as Contrastive Language-Image Pre-training (CLIP) have been shown to be susceptible to adversarial attacks, raising concerns about their deployment in safety-critical applications like autonomous driving and medical diagnosis. One promising approach for robustifying pre-trained VLMs is Adversarial Prompt Tuning (APT), which applies adversarial training during the process of prompt tuning. However, existing APT methods are mostly single-modal methods that design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy. In this work, we propose Adversarial Prompt Distillation (APD), a bimodal knowledge distillation framework that enhances APT by integrating it with multi-modal knowledge transfer. APD optimizes prompts for both visual and textual modalities while distilling knowledge from a clean pre-trained teacher CLIP model. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our APD method over the current state-of-the-art APT methods in terms of both adversarial robustness and clean accuracy. The effectiveness of APD also validates the possibility of using a non-robust teacher to improve the generalization and robustness of fine-tuned VLMs.



## **32. Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection**

cs.CV

Accepted to EMNLP 2025 (Main). 17 pages, 7 figures

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2507.02844v2) [paper-pdf](http://arxiv.org/pdf/2507.02844v2)

**Authors**: Ziqi Miao, Yi Ding, Lijun Li, Jing Shao

**Abstract**: With the emergence of strong vision language capabilities, multimodal large language models (MLLMs) have demonstrated tremendous potential for real-world applications. However, the security vulnerabilities exhibited by the visual modality pose significant challenges to deploying such models in open-world environments. Recent studies have successfully induced harmful responses from target MLLMs by encoding harmful textual semantics directly into visual inputs. However, in these approaches, the visual modality primarily serves as a trigger for unsafe behavior, often exhibiting semantic ambiguity and lacking grounding in realistic scenarios. In this work, we define a novel setting: vision-centric jailbreak, where visual information serves as a necessary component in constructing a complete and realistic jailbreak context. Building on this setting, we propose the VisCo (Visual Contextual) Attack. VisCo fabricates contextual dialogue using four distinct vision-focused strategies, dynamically generating auxiliary images when necessary to construct a vision-centric jailbreak scenario. To maximize attack effectiveness, it incorporates automatic toxicity obfuscation and semantic refinement to produce a final attack prompt that reliably triggers harmful responses from the target black-box MLLMs. Specifically, VisCo achieves a toxicity score of 4.78 and an Attack Success Rate (ASR) of 85% on MM-SafetyBench against GPT-4o, significantly outperforming the baseline, which achieves a toxicity score of 2.48 and an ASR of 22.2%. Code: https://github.com/Dtc7w3PQ/Visco-Attack.



## **33. Towards Inclusive Toxic Content Moderation: Addressing Vulnerabilities to Adversarial Attacks in Toxicity Classifiers Tackling LLM-generated Content**

cs.CL

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12672v1) [paper-pdf](http://arxiv.org/pdf/2509.12672v1)

**Authors**: Shaz Furniturewala, Arkaitz Zubiaga

**Abstract**: The volume of machine-generated content online has grown dramatically due to the widespread use of Large Language Models (LLMs), leading to new challenges for content moderation systems. Conventional content moderation classifiers, which are usually trained on text produced by humans, suffer from misclassifications due to LLM-generated text deviating from their training data and adversarial attacks that aim to avoid detection. Present-day defence tactics are reactive rather than proactive, since they rely on adversarial training or external detection models to identify attacks. In this work, we aim to identify the vulnerable components of toxicity classifiers that contribute to misclassification, proposing a novel strategy based on mechanistic interpretability techniques. Our study focuses on fine-tuned BERT and RoBERTa classifiers, testing on diverse datasets spanning a variety of minority groups. We use adversarial attacking techniques to identify vulnerable circuits. Finally, we suppress these vulnerable circuits, improving performance against adversarial attacks. We also provide demographic-level insights into these vulnerable circuits, exposing fairness and robustness gaps in model training. We find that models have distinct heads that are either crucial for performance or vulnerable to attack and suppressing the vulnerable heads improves performance on adversarial input. We also find that different heads are responsible for vulnerability across different demographic groups, which can inform more inclusive development of toxicity detection models.



## **34. A Systematic Evaluation of Parameter-Efficient Fine-Tuning Methods for the Security of Code LLMs**

cs.CR

25 pages

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12649v1) [paper-pdf](http://arxiv.org/pdf/2509.12649v1)

**Authors**: Kiho Lee, Jungkon Kim, Doowon Kim, Hyoungshick Kim

**Abstract**: Code-generating Large Language Models (LLMs) significantly accelerate software development. However, their frequent generation of insecure code presents serious risks. We present a comprehensive evaluation of seven parameter-efficient fine-tuning (PEFT) techniques, demonstrating substantial gains in secure code generation without compromising functionality. Our research identifies prompt-tuning as the most effective PEFT method, achieving an 80.86% Overall-Secure-Rate on CodeGen2 16B, a 13.5-point improvement over the 67.28% baseline. Optimizing decoding strategies through sampling temperature further elevated security to 87.65%. This equates to a reduction of approximately 203,700 vulnerable code snippets per million generated. Moreover, prompt and prefix tuning increase robustness against poisoning attacks in our TrojanPuzzle evaluation, with strong performance against CWE-79 and CWE-502 attack vectors. Our findings generalize across Python and Java, confirming prompt-tuning's consistent effectiveness. This study provides essential insights and practical guidance for building more resilient software systems with LLMs.



## **35. Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs**

cs.CL

Preprint

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.11177v2) [paper-pdf](http://arxiv.org/pdf/2509.11177v2)

**Authors**: Hang Guo, Yawei Li, Luca Benini

**Abstract**: Recent advances in Large Language Model (LLM) compression, such as quantization and pruning, have achieved notable success. However, as these techniques gradually approach their respective limits, relying on a single method for further compression has become increasingly challenging. In this work, we explore an alternative solution by combining quantization and sparsity. This joint approach, though promising, introduces new difficulties due to the inherently conflicting requirements on weight distributions: quantization favors compact ranges, while pruning benefits from high variance. To attack this problem, we propose Optimal Brain Restoration (OBR), a general and training-free framework that aligns pruning and quantization by error compensation between both. OBR minimizes performance degradation on downstream tasks by building on a second-order Hessian objective, which is then reformulated into a tractable problem through surrogate approximation and ultimately reaches a closed-form solution via group error compensation. Experiments show that OBR enables aggressive W4A4KV4 quantization with 50% sparsity on existing LLMs, and delivers up to 4.72x speedup and 6.4x memory reduction compared to the FP16-dense baseline.



## **36. PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance**

cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2508.20890v2) [paper-pdf](http://arxiv.org/pdf/2508.20890v2)

**Authors**: Mengxiao Wang, Yuxuan Zhang, Guofei Gu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications, from virtual assistants to autonomous agents. However, their flexibility also introduces new attack vectors-particularly Prompt Injection (PI), where adversaries manipulate model behavior through crafted inputs. As attackers continuously evolve with paraphrased, obfuscated, and even multi-task injection strategies, existing benchmarks are no longer sufficient to capture the full spectrum of emerging threats.   To address this gap, we construct a new benchmark that systematically extends prior efforts. Our benchmark subsumes the two widely-used existing ones while introducing new manipulation techniques and multi-task scenarios, thereby providing a more comprehensive evaluation setting. We find that existing defenses, though effective on their original benchmarks, show clear weaknesses under our benchmark, underscoring the need for more robust solutions. Our key insight is that while attack forms may vary, the adversary's intent-injecting an unauthorized task-remains invariant. Building on this observation, we propose PromptSleuth, a semantic-oriented defense framework that detects prompt injection by reasoning over task-level intent rather than surface features. Evaluated across state-of-the-art benchmarks, PromptSleuth consistently outperforms existing defense while maintaining comparable runtime and cost efficiency. These results demonstrate that intent-based semantic reasoning offers a robust, efficient, and generalizable strategy for defending LLMs against evolving prompt injection threats.



## **37. Phi: Preference Hijacking in Multi-modal Large Language Models at Inference Time**

cs.LG

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.12521v1) [paper-pdf](http://arxiv.org/pdf/2509.12521v1)

**Authors**: Yifan Lan, Yuanpu Cao, Weitong Zhang, Lu Lin, Jinghui Chen

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have gained significant attention across various domains. However, their widespread adoption has also raised serious safety concerns. In this paper, we uncover a new safety risk of MLLMs: the output preference of MLLMs can be arbitrarily manipulated by carefully optimized images. Such attacks often generate contextually relevant yet biased responses that are neither overtly harmful nor unethical, making them difficult to detect. Specifically, we introduce a novel method, Preference Hijacking (Phi), for manipulating the MLLM response preferences using a preference hijacked image. Our method works at inference time and requires no model modifications. Additionally, we introduce a universal hijacking perturbation -- a transferable component that can be embedded into different images to hijack MLLM responses toward any attacker-specified preferences. Experimental results across various tasks demonstrate the effectiveness of our approach. The code for Phi is accessible at https://github.com/Yifan-Lan/Phi.



## **38. Keep Security! Benchmarking Security Policy Preservation in Large Language Model Contexts Against Indirect Attacks in Question Answering**

cs.CL

EMNLP 2025 (Main Conference)

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2505.15805v2) [paper-pdf](http://arxiv.org/pdf/2505.15805v2)

**Authors**: Hwan Chang, Yumin Kim, Yonghyun Jun, Hwanhee Lee

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in sensitive domains such as enterprise and government, ensuring that they adhere to user-defined security policies within context is critical-especially with respect to information non-disclosure. While prior LLM studies have focused on general safety and socially sensitive data, large-scale benchmarks for contextual security preservation against attacks remain lacking. To address this, we introduce a novel large-scale benchmark dataset, CoPriva, evaluating LLM adherence to contextual non-disclosure policies in question answering. Derived from realistic contexts, our dataset includes explicit policies and queries designed as direct and challenging indirect attacks seeking prohibited information. We evaluate 10 LLMs on our benchmark and reveal a significant vulnerability: many models violate user-defined policies and leak sensitive information. This failure is particularly severe against indirect attacks, highlighting a critical gap in current LLM safety alignment for sensitive applications. Our analysis reveals that while models can often identify the correct answer to a query, they struggle to incorporate policy constraints during generation. In contrast, they exhibit a partial ability to revise outputs when explicitly prompted. Our findings underscore the urgent need for more robust methods to guarantee contextual security.



## **39. Early Approaches to Adversarial Fine-Tuning for Prompt Injection Defense: A 2022 Study of GPT-3 and Contemporary Models**

cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.14271v1) [paper-pdf](http://arxiv.org/pdf/2509.14271v1)

**Authors**: Gustavo Sandoval, Denys Fenchenko, Junyao Chen

**Abstract**: This paper documents early research conducted in 2022 on defending against prompt injection attacks in large language models, providing historical context for the evolution of this critical security domain. This research focuses on two adversarial attacks against Large Language Models (LLMs): prompt injection and goal hijacking. We examine how to construct these attacks, test them on various LLMs, and compare their effectiveness. We propose and evaluate a novel defense technique called Adversarial Fine-Tuning. Our results show that, without this defense, the attacks succeeded 31\% of the time on GPT-3 series models. When using our Adversarial Fine-Tuning approach, attack success rates were reduced to near zero for smaller GPT-3 variants (Ada, Babbage, Curie), though we note that subsequent research has revealed limitations of fine-tuning-based defenses. We also find that more flexible models exhibit greater vulnerability to these attacks. Consequently, large models such as GPT-3 Davinci are more vulnerable than smaller models like GPT-2. While the specific models tested are now superseded, the core methodology and empirical findings contributed to the foundation of modern prompt injection defense research, including instruction hierarchy systems and constitutional AI approaches.



## **40. Safety Pretraining: Toward the Next Generation of Safe AI**

cs.LG

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2504.16980v2) [paper-pdf](http://arxiv.org/pdf/2504.16980v2)

**Authors**: Pratyush Maini, Sachin Goyal, Dylan Sam, Alex Robey, Yash Savani, Yiding Jiang, Andy Zou, Matt Fredrikson, Zacharcy C. Lipton, J. Zico Kolter

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes settings, the risk of generating harmful or toxic content remains a central challenge. Post-hoc alignment methods are brittle: once unsafe patterns are learned during pretraining, they are hard to remove. In this work, we present a data-centric pretraining framework that builds safety into the model from the start. Our framework consists of four key steps: (i) Safety Filtering: building a safety classifier to classify webdata into safe and unsafe categories; (ii) Safety Rephrasing: we recontextualize unsafe webdata into safer narratives; (iii) Native Refusal: we develop RefuseWeb and Moral Education pretraining datasets that actively teach model to refuse on unsafe content and the moral reasoning behind it, and (iv) Harmfulness-Tag annotated pretraining: we flag unsafe content during pretraining using a special token, and use it to steer model away from unsafe generations at inference. Our safety-pretrained models reduce attack success rates from 38.8\% to 8.4\% on standard LLM safety benchmarks with no performance degradation on general tasks.



## **41. NeuroStrike: Neuron-Level Attacks on Aligned LLMs**

cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11864v1) [paper-pdf](http://arxiv.org/pdf/2509.11864v1)

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Maximilian Thang, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability.   This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.



## **42. One Goal, Many Challenges: Robust Preference Optimization Amid Content-Aware and Multi-Source Noise**

cs.LG

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2503.12301v2) [paper-pdf](http://arxiv.org/pdf/2503.12301v2)

**Authors**: Amirabbas Afzali, Amirhossein Afsharrad, Seyed Shahabeddin Mousavi, Sanjay Lall

**Abstract**: Large Language Models (LLMs) have made significant strides in generating human-like responses, largely due to preference alignment techniques. However, these methods often assume unbiased human feedback, which is rarely the case in real-world scenarios. This paper introduces Content-Aware Noise-Resilient Preference Optimization (CNRPO), a novel framework that addresses multiple sources of content-dependent noise in preference learning. CNRPO employs a multi-objective optimization approach to separate true preferences from content-aware noises, effectively mitigating their impact. We leverage backdoor attack mechanisms to efficiently learn and control various noise sources within a single model. Theoretical analysis and extensive experiments on different synthetic noisy datasets demonstrate that CNRPO significantly improves alignment with primary human preferences while controlling for secondary noises and biases, such as response length and harmfulness.



## **43. Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check**

cs.LG

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11629v1) [paper-pdf](http://arxiv.org/pdf/2509.11629v1)

**Authors**: Chentao Cao, Xiaojun Xu, Bo Han, Hang Li

**Abstract**: As large language models (LLMs) continue to advance in capabilities, ensuring their safety against jailbreak attacks remains a critical challenge. In this paper, we introduce a novel safety alignment approach called Answer-Then-Check, which enhances LLM robustness against malicious prompts by applying thinking ability to mitigate jailbreaking problems before producing a final answer to the user. Our method enables models to directly answer the question in their thought and then critically evaluate its safety before deciding whether to provide it. To implement this approach, we construct the Reasoned Safety Alignment (ReSA) dataset, comprising 80K examples that teach models to reason through direct responses and then analyze their safety. Experimental results demonstrate that our approach achieves the Pareto frontier with superior safety capability while decreasing over-refusal rates on over-refusal benchmarks. Notably, the model fine-tuned with ReSA maintains general reasoning capabilities on benchmarks like MMLU, MATH500, and HumanEval. Besides, our method equips models with the ability to perform safe completion. Unlike post-hoc methods that can only reject harmful queries, our model can provide helpful and safe alternative responses for sensitive topics (e.g., self-harm). Furthermore, we discover that training on a small subset of just 500 examples can achieve comparable performance to using the full dataset, suggesting that safety alignment may require less data than previously assumed.



## **44. Multilingual Collaborative Defense for Large Language Models**

cs.CL

21 pages, 4figures

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2505.11835v2) [paper-pdf](http://arxiv.org/pdf/2505.11835v2)

**Authors**: Hongliang Li, Jinan Xu, Gengping Cui, Changhao Guan, Fengran Mo, Kaiyu Huang

**Abstract**: The robustness and security of large language models (LLMs) has become a prominent research area. One notable vulnerability is the ability to bypass LLM safeguards by translating harmful queries into rare or underrepresented languages, a simple yet effective method of "jailbreaking" these models. Despite the growing concern, there has been limited research addressing the safeguarding of LLMs in multilingual scenarios, highlighting an urgent need to enhance multilingual safety. In this work, we investigate the correlation between various attack features across different languages and propose Multilingual Collaborative Defense (MCD), a novel learning method that optimizes a continuous, soft safety prompt automatically to facilitate multilingual safeguarding of LLMs. The MCD approach offers three advantages: First, it effectively improves safeguarding performance across multiple languages. Second, MCD maintains strong generalization capabilities while minimizing false refusal rates. Third, MCD mitigates the language safety misalignment caused by imbalances in LLM training corpora. To evaluate the effectiveness of MCD, we manually construct multilingual versions of commonly used jailbreak benchmarks, such as MaliciousInstruct and AdvBench, to assess various safeguarding methods. Additionally, we introduce these datasets in underrepresented (zero-shot) languages to verify the language transferability of MCD. The results demonstrate that MCD outperforms existing approaches in safeguarding against multilingual jailbreak attempts while also exhibiting strong language transfer capabilities. Our code is available at https://github.com/HLiang-Lee/MCD.



## **45. Confusion is the Final Barrier: Rethinking Jailbreak Evaluation and Investigating the Real Misuse Threat of LLMs**

cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2508.16347v2) [paper-pdf](http://arxiv.org/pdf/2508.16347v2)

**Authors**: Yu Yan, Sheng Sun, Zhe Wang, Yijun Lin, Zenghao Duan, zhifei zheng, Min Liu, Zhiyi yin, Jianping Zhang

**Abstract**: With the development of Large Language Models (LLMs), numerous efforts have revealed their vulnerabilities to jailbreak attacks. Although these studies have driven the progress in LLMs' safety alignment, it remains unclear whether LLMs have internalized authentic knowledge to deal with real-world crimes, or are merely forced to simulate toxic language patterns. This ambiguity raises concerns that jailbreak success is often attributable to a hallucination loop between jailbroken LLM and judger LLM. By decoupling the use of jailbreak techniques, we construct knowledge-intensive Q\&A to investigate the misuse threats of LLMs in terms of dangerous knowledge possession, harmful task planning utility, and harmfulness judgment robustness. Experiments reveal a mismatch between jailbreak success rates and harmful knowledge possession in LLMs, and existing LLM-as-a-judge frameworks tend to anchor harmfulness judgments on toxic language patterns. Our study reveals a gap between existing LLM safety assessments and real-world threat potential.



## **46. Enhancing Prompt Injection Attacks to LLMs via Poisoning Alignment**

cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2410.14827v3) [paper-pdf](http://arxiv.org/pdf/2410.14827v3)

**Authors**: Zedian Shao, Hongbin Liu, Jaden Mu, Neil Zhenqiang Gong

**Abstract**: Prompt injection attack, where an attacker injects a prompt into the original one, aiming to make an Large Language Model (LLM) follow the injected prompt to perform an attacker-chosen task, represent a critical security threat. Existing attacks primarily focus on crafting these injections at inference time, treating the LLM itself as a static target. Our experiments show that these attacks achieve some success, but there is still significant room for improvement. In this work, we introduces a more foundational attack vector: poisoning the LLM's alignment process to amplify the success of future prompt injection attacks. Specifically, we propose PoisonedAlign, a method that strategically creates poisoned alignment samples to poison an LLM's alignment dataset. Our experiments across five LLMs and two alignment datasets show that when even a small fraction of the alignment data is poisoned, the resulting model becomes substantially more vulnerable to a wide range of prompt injection attacks. Crucially, this vulnerability is instilled while the LLM's performance on standard capability benchmarks remains largely unchanged, making the manipulation difficult to detect through automated, general-purpose performance evaluations. The code for implementing the attack is available at https://github.com/Sadcardation/PoisonedAlign.



## **47. Securing AI Agents: Implementing Role-Based Access Control for Industrial Applications**

cs.AI

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11431v1) [paper-pdf](http://arxiv.org/pdf/2509.11431v1)

**Authors**: Aadil Gani Ganie

**Abstract**: The emergence of Large Language Models (LLMs) has significantly advanced solutions across various domains, from political science to software development. However, these models are constrained by their training data, which is static and limited to information available up to a specific date. Additionally, their generalized nature often necessitates fine-tuning -- whether for classification or instructional purposes -- to effectively perform specific downstream tasks. AI agents, leveraging LLMs as their core, mitigate some of these limitations by accessing external tools and real-time data, enabling applications such as live weather reporting and data analysis. In industrial settings, AI agents are transforming operations by enhancing decision-making, predictive maintenance, and process optimization. For example, in manufacturing, AI agents enable near-autonomous systems that boost productivity and support real-time decision-making. Despite these advancements, AI agents remain vulnerable to security threats, including prompt injection attacks, which pose significant risks to their integrity and reliability. To address these challenges, this paper proposes a framework for integrating Role-Based Access Control (RBAC) into AI agents, providing a robust security guardrail. This framework aims to support the effective and scalable deployment of AI agents, with a focus on on-premises implementations.



## **48. Fighting Fire with Fire (F3): A Training-free and Efficient Visual Adversarial Example Purification Method in LVLMs**

cs.CV

Accepted by ACM Multimedia 2025 BNI track (Oral)

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2506.01064v3) [paper-pdf](http://arxiv.org/pdf/2506.01064v3)

**Authors**: Yudong Zhang, Ruobing Xie, Yiqing Huang, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Di Wang, Yu Wang

**Abstract**: Recent advances in large vision-language models (LVLMs) have showcased their remarkable capabilities across a wide range of multimodal vision-language tasks. However, these models remain vulnerable to visual adversarial attacks, which can substantially compromise their performance. In this paper, we introduce F3, a novel adversarial purification framework that employs a counterintuitive ``fighting fire with fire'' strategy: intentionally introducing simple perturbations to adversarial examples to mitigate their harmful effects. Specifically, F3 leverages cross-modal attentions derived from randomly perturbed adversary examples as reference targets. By injecting noise into these adversarial examples, F3 effectively refines their attention, resulting in cleaner and more reliable model outputs. Remarkably, this seemingly paradoxical approach of employing noise to counteract adversarial attacks yields impressive purification results. Furthermore, F3 offers several distinct advantages: it is training-free and straightforward to implement, and exhibits significant computational efficiency improvements compared to existing purification methods. These attributes render F3 particularly suitable for large-scale industrial applications where both robust performance and operational efficiency are critical priorities. The code is available at https://github.com/btzyd/F3.



## **49. Beyond the Protocol: Unveiling Attack Vectors in the Model Context Protocol (MCP) Ecosystem**

cs.CR

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2506.02040v4) [paper-pdf](http://arxiv.org/pdf/2506.02040v4)

**Authors**: Hao Song, Yiming Shen, Wenxuan Luo, Leixin Guo, Ting Chen, Jiashui Wang, Beibei Li, Xiaosong Zhang, Jiachi Chen

**Abstract**: The Model Context Protocol (MCP) is an emerging standard designed to enable seamless interaction between Large Language Model (LLM) applications and external tools or resources. Within a short period, thousands of MCP services have been developed and deployed. However, the client-server integration architecture inherent in MCP may expand the attack surface against LLM Agent systems, introducing new vulnerabilities that allow attackers to exploit by designing malicious MCP servers. In this paper, we present the first end-to-end empirical evaluation of attack vectors targeting the MCP ecosystem. We identify four categories of attacks, i.e., Tool Poisoning Attacks, Puppet Attacks, Rug Pull Attacks, and Exploitation via Malicious External Resources. To evaluate their feasibility, we conduct experiments following the typical steps of launching an attack through malicious MCP servers: upload -> download -> attack. Specifically, we first construct malicious MCP servers and successfully upload them to three widely used MCP aggregation platforms. The results indicate that current audit mechanisms are insufficient to identify and prevent these threats. Next, through a user study and interview with 20 participants, we demonstrate that users struggle to identify malicious MCP servers and often unknowingly install them from aggregator platforms. Finally, we empirically demonstrate that these attacks can trigger harmful actions within the user's local environment, such as accessing private files or controlling devices to transfer digital assets. Additionally, based on interview results, we discuss four key challenges faced by the current MCP security ecosystem. These findings underscore the urgent need for robust security mechanisms to defend against malicious MCP servers and ensure the safe deployment of increasingly autonomous LLM agents.



## **50. Character-Level Perturbations Disrupt LLM Watermarks**

cs.CR

accepted by Network and Distributed System Security (NDSS) Symposium  2026

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.09112v2) [paper-pdf](http://arxiv.org/pdf/2509.09112v2)

**Authors**: Zhaoxi Zhang, Xiaomei Zhang, Yanjun Zhang, He Zhang, Shirui Pan, Bo Liu, Asif Qumer Gill, Leo Yu Zhang

**Abstract**: Large Language Model (LLM) watermarking embeds detectable signals into generated text for copyright protection, misuse prevention, and content detection. While prior studies evaluate robustness using watermark removal attacks, these methods are often suboptimal, creating the misconception that effective removal requires large perturbations or powerful adversaries.   To bridge the gap, we first formalize the system model for LLM watermark, and characterize two realistic threat models constrained on limited access to the watermark detector. We then analyze how different types of perturbation vary in their attack range, i.e., the number of tokens they can affect with a single edit. We observe that character-level perturbations (e.g., typos, swaps, deletions, homoglyphs) can influence multiple tokens simultaneously by disrupting the tokenization process. We demonstrate that character-level perturbations are significantly more effective for watermark removal under the most restrictive threat model. We further propose guided removal attacks based on the Genetic Algorithm (GA) that uses a reference detector for optimization. Under a practical threat model with limited black-box queries to the watermark detector, our method demonstrates strong removal performance. Experiments confirm the superiority of character-level perturbations and the effectiveness of the GA in removing watermarks under realistic constraints. Additionally, we argue there is an adversarial dilemma when considering potential defenses: any fixed defense can be bypassed by a suitable perturbation strategy. Motivated by this principle, we propose an adaptive compound character-level attack. Experimental results show that this approach can effectively defeat the defenses. Our findings highlight significant vulnerabilities in existing LLM watermark schemes and underline the urgency for the development of new robust mechanisms.



