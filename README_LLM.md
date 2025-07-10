# Latest Large Language Model Attack Papers
**update at 2025-07-10 15:57:38**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. VisualTrap: A Stealthy Backdoor Attack on GUI Agents via Visual Grounding Manipulation**

cs.CL

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06899v1) [paper-pdf](http://arxiv.org/pdf/2507.06899v1)

**Authors**: Ziang Ye, Yang Zhang, Wentao Shi, Xiaoyu You, Fuli Feng, Tat-Seng Chua

**Abstract**: Graphical User Interface (GUI) agents powered by Large Vision-Language Models (LVLMs) have emerged as a revolutionary approach to automating human-machine interactions, capable of autonomously operating personal devices (e.g., mobile phones) or applications within the device to perform complex real-world tasks in a human-like manner. However, their close integration with personal devices raises significant security concerns, with many threats, including backdoor attacks, remaining largely unexplored. This work reveals that the visual grounding of GUI agent-mapping textual plans to GUI elements-can introduce vulnerabilities, enabling new types of backdoor attacks. With backdoor attack targeting visual grounding, the agent's behavior can be compromised even when given correct task-solving plans. To validate this vulnerability, we propose VisualTrap, a method that can hijack the grounding by misleading the agent to locate textual plans to trigger locations instead of the intended targets. VisualTrap uses the common method of injecting poisoned data for attacks, and does so during the pre-training of visual grounding to ensure practical feasibility of attacking. Empirical results show that VisualTrap can effectively hijack visual grounding with as little as 5% poisoned data and highly stealthy visual triggers (invisible to the human eye); and the attack can be generalized to downstream tasks, even after clean fine-tuning. Moreover, the injected trigger can remain effective across different GUI environments, e.g., being trained on mobile/web and generalizing to desktop environments. These findings underscore the urgent need for further research on backdoor attack risks in GUI agents.



## **2. The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover**

cs.CR

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06850v1) [paper-pdf](http://arxiv.org/pdf/2507.06850v1)

**Authors**: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro

**Abstract**: The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.



## **3. GuidedBench: Measuring and Mitigating the Evaluation Discrepancies of In-the-wild LLM Jailbreak Methods**

cs.CL

Homepage: https://sproutnan.github.io/AI-Safety_Benchmark/

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2502.16903v2) [paper-pdf](http://arxiv.org/pdf/2502.16903v2)

**Authors**: Ruixuan Huang, Xunguang Wang, Zongjie Li, Daoyuan Wu, Shuai Wang

**Abstract**: Despite the growing interest in jailbreak methods as an effective red-teaming tool for building safe and responsible large language models (LLMs), flawed evaluation system designs have led to significant discrepancies in their effectiveness assessments. We conduct a systematic measurement study based on 37 jailbreak studies since 2022, focusing on both the methods and the evaluation systems they employ. We find that existing evaluation systems lack case-specific criteria, resulting in misleading conclusions about their effectiveness and safety implications. This paper advocates a shift to a more nuanced, case-by-case evaluation paradigm. We introduce GuidedBench, a novel benchmark comprising a curated harmful question dataset, detailed case-by-case evaluation guidelines and an evaluation system integrated with these guidelines -- GuidedEval. Experiments demonstrate that GuidedBench offers more accurate measurements of jailbreak performance, enabling meaningful comparisons across methods and uncovering new insights overlooked in previous evaluations. GuidedEval reduces inter-evaluator variance by at least 76.03\%. Furthermore, we observe that incorporating guidelines can enhance the effectiveness of jailbreak methods themselves, offering new insights into both attack strategies and evaluation paradigms.



## **4. Tail-aware Adversarial Attacks: A Distributional Approach to Efficient LLM Jailbreaking**

cs.LG

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.04446v2) [paper-pdf](http://arxiv.org/pdf/2507.04446v2)

**Authors**: Tim Beyer, Yan Scholten, Leo Schwinn, Stephan Günnemann

**Abstract**: To guarantee safe and robust deployment of large language models (LLMs) at scale, it is critical to accurately assess their adversarial robustness. Existing adversarial attacks typically target harmful responses in single-point, greedy generations, overlooking the inherently stochastic nature of LLMs. In this paper, we propose a novel framework for adversarial robustness evaluation that explicitly models the entire output distribution, including tail-risks, providing better estimates for model robustness at scale. By casting the attack process as a resource allocation problem between optimization and sampling, we determine compute-optimal tradeoffs and show that integrating sampling into existing attacks boosts ASR by up to 48% and improves efficiency by up to two orders of magnitude. Our framework also enables us to analyze how different attack algorithms affect output harm distributions. Surprisingly, we find that most optimization strategies have little effect on output harmfulness. Finally, we introduce a data-free proof-of-concept objective based on entropy-maximization to demonstrate how our tail-aware perspective enables new optimization targets. Overall, our findings highlight the importance of tail-aware attacks and evaluation protocols to accurately assess and strengthen LLM safety.



## **5. Evaluating and Improving Robustness in Large Language Models: A Survey and Future Directions**

cs.CL

33 pages, 5 figures

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2506.11111v2) [paper-pdf](http://arxiv.org/pdf/2506.11111v2)

**Authors**: Kun Zhang, Le Wu, Kui Yu, Guangyi Lv, Dacao Zhang

**Abstract**: Large Language Models (LLMs) have gained enormous attention in recent years due to their capability of understanding and generating natural languages. With the rapid development and wild-range applications (e.g., Agents, Embodied Intelligence), the robustness of LLMs has received increased attention. As the core brain of many AI applications, the robustness of LLMs requires that models should not only generate consistent contents, but also ensure the correctness and stability of generated content when dealing with unexpeted application scenarios (e.g., toxic prompts, limited noise domain data, outof-distribution (OOD) applications, etc). In this survey paper, we conduct a thorough review of the robustness of LLMs, aiming to provide a comprehensive terminology of concepts and methods around this field and facilitate the community. Specifically, we first give a formal definition of LLM robustness and present the collection protocol of this survey paper. Then, based on the types of perturbated inputs, we organize this survey from the following perspectives: 1) Adversarial Robustness: tackling the problem that prompts are manipulated intentionally, such as noise prompts, long context, data attack, etc; 2) OOD Robustness: dealing with the unexpected real-world application scenarios, such as OOD detection, zero-shot transferring, hallucinations, etc; 3) Evaluation of Robustness: summarizing the new evaluation datasets, metrics, and tools for verifying the robustness of LLMs. After reviewing the representative work from each perspective, we discuss and highlight future opportunities and research directions in this field. Meanwhile, we also organize related works and provide an easy-to-search project (https://github.com/zhangkunzk/Awesome-LLM-Robustness-papers) to support the community.



## **6. Breaking PEFT Limitations: Leveraging Weak-to-Strong Knowledge Transfer for Backdoor Attacks in LLMs**

cs.CR

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2409.17946v4) [paper-pdf](http://arxiv.org/pdf/2409.17946v4)

**Authors**: Shuai Zhao, Leilei Gan, Zhongliang Guo, Xiaobao Wu, Yanhao Jia, Luwei Xiao, Cong-Duy Nguyen, Luu Anh Tuan

**Abstract**: Despite being widely applied due to their exceptional capabilities, Large Language Models (LLMs) have been proven to be vulnerable to backdoor attacks. These attacks introduce targeted vulnerabilities into LLMs by poisoning training samples and full-parameter fine-tuning (FPFT). However, this kind of backdoor attack is limited since they require significant computational resources, especially as the size of LLMs increases. Besides, parameter-efficient fine-tuning (PEFT) offers an alternative but the restricted parameter updating may impede the alignment of triggers with target labels. In this study, we first verify that backdoor attacks with PEFT may encounter challenges in achieving feasible performance. To address these issues and improve the effectiveness of backdoor attacks with PEFT, we propose a novel backdoor attack algorithm from the weak-to-strong based on Feature Alignment-enhanced Knowledge Distillation (FAKD). Specifically, we poison small-scale language models through FPFT to serve as the teacher model. The teacher model then covertly transfers the backdoor to the large-scale student model through FAKD, which employs PEFT. Theoretical analysis reveals that FAKD has the potential to augment the effectiveness of backdoor attacks. We demonstrate the superior performance of FAKD on classification tasks across four language models, four backdoor attack algorithms, and two different architectures of teacher models. Experimental results indicate success rates close to 100% for backdoor attacks targeting PEFT.



## **7. Can adversarial attacks by large language models be attributed?**

cs.AI

21 pages, 5 figures, 2 tables

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2411.08003v2) [paper-pdf](http://arxiv.org/pdf/2411.08003v2)

**Authors**: Manuel Cebrian, Andres Abeliuk, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation campaigns-presents significant challenges that are likely to grow in importance. We approach this attribution problem from both a theoretical and an empirical perspective, drawing on formal language theory (identification in the limit) and data-driven analysis of the expanding LLM ecosystem. By modeling an LLM's set of possible outputs as a formal language, we analyze whether finite samples of text can uniquely pinpoint the originating model. Our results show that, under mild assumptions of overlapping capabilities among models, certain classes of LLMs are fundamentally non-identifiable from their outputs alone. We delineate four regimes of theoretical identifiability: (1) an infinite class of deterministic (discrete) LLM languages is not identifiable (Gold's classical result from 1967); (2) an infinite class of probabilistic LLMs is also not identifiable (by extension of the deterministic case); (3) a finite class of deterministic LLMs is identifiable (consistent with Angluin's tell-tale criterion); and (4) even a finite class of probabilistic LLMs can be non-identifiable (we provide a new counterexample establishing this negative result). Complementing these theoretical insights, we quantify the explosion in the number of plausible model origins (hypothesis space) for a given output in recent years. Even under conservative assumptions-each open-source model fine-tuned on at most one new dataset-the count of distinct candidate models doubles approximately every 0.5 years, and allowing multi-dataset fine-tuning combinations yields doubling times as short as 0.28 years. This combinatorial growth, alongside the extraordinary computational cost of brute-force likelihood attribution across all models and potential users, renders exhaustive attribution infeasible in practice.



## **8. On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks**

cs.CL

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06489v1) [paper-pdf](http://arxiv.org/pdf/2507.06489v1)

**Authors**: Stephen Obadinma, Xiaodan Zhu

**Abstract**: Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to ensure transparency, trust, and safety in human-AI interactions across many high-stakes applications. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce a novel framework for attacking verbal confidence scores through both perturbation and jailbreak-based methods, and show that these attacks can significantly jeopardize verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current confidence elicitation methods are vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the urgent need to design more robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.



## **9. Bridging AI and Software Security: A Comparative Vulnerability Assessment of LLM Agent Deployment Paradigms**

cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06323v1) [paper-pdf](http://arxiv.org/pdf/2507.06323v1)

**Authors**: Tarek Gasmi, Ramzi Guesmi, Ines Belhadj, Jihene Bennaceur

**Abstract**: Large Language Model (LLM) agents face security vulnerabilities spanning AI-specific and traditional software domains, yet current research addresses these separately. This study bridges this gap through comparative evaluation of Function Calling architecture and Model Context Protocol (MCP) deployment paradigms using a unified threat classification framework. We tested 3,250 attack scenarios across seven language models, evaluating simple, composed, and chained attacks targeting both AI-specific threats (prompt injection) and software vulnerabilities (JSON injection, denial-of-service). Function Calling showed higher overall attack success rates (73.5% vs 62.59% for MCP), with greater system-centric vulnerability while MCP exhibited increased LLM-centric exposure. Attack complexity dramatically amplified effectiveness, with chained attacks achieving 91-96% success rates. Counterintuitively, advanced reasoning models demonstrated higher exploitability despite better threat detection. Results demonstrate that architectural choices fundamentally reshape threat landscapes. This work establishes methodological foundations for cross-domain LLM agent security assessment and provides evidence-based guidance for secure deployment. Code and experimental materials are available at https: // github. com/ theconsciouslab-ai/llm-agent-security.



## **10. CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks on their Internal Representations**

cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06043v1) [paper-pdf](http://arxiv.org/pdf/2507.06043v1)

**Authors**: Xiaohu Li, Yunfeng Ning, Zepeng Bao, Mayi Xu, Jianhao Chen, Tieyun Qian

**Abstract**: Security alignment enables the Large Language Model (LLM) to gain the protection against malicious queries, but various jailbreak attack methods reveal the vulnerability of this security mechanism. Previous studies have isolated LLM jailbreak attacks and defenses. We analyze the security protection mechanism of the LLM, and propose a framework that combines attack and defense. Our method is based on the linearly separable property of LLM intermediate layer embedding, as well as the essence of jailbreak attack, which aims to embed harmful problems and transfer them to the safe area. We utilize generative adversarial network (GAN) to learn the security judgment boundary inside the LLM to achieve efficient jailbreak attack and defense. The experimental results indicate that our method achieves an average jailbreak success rate of 88.85\% across three popular LLMs, while the defense success rate on the state-of-the-art jailbreak dataset reaches an average of 84.17\%. This not only validates the effectiveness of our approach but also sheds light on the internal security mechanisms of LLMs, offering new insights for enhancing model security The code and data are available at https://github.com/NLPGM/CAVGAN.



## **11. Enhancing LLM Watermark Resilience Against Both Scrubbing and Spoofing Attacks**

cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06274v1) [paper-pdf](http://arxiv.org/pdf/2507.06274v1)

**Authors**: Huanming Shen, Baizhou Huang, Xiaojun Wan

**Abstract**: Watermarking is a promising defense against the misuse of large language models (LLMs), yet it remains vulnerable to scrubbing and spoofing attacks. This vulnerability stems from an inherent trade-off governed by watermark window size: smaller windows resist scrubbing better but are easier to reverse-engineer, enabling low-cost statistics-based spoofing attacks. This work breaks this trade-off by introducing a novel mechanism, equivalent texture keys, where multiple tokens within a watermark window can independently support the detection. Based on the redundancy, we propose a novel watermark scheme with Sub-vocabulary decomposed Equivalent tExture Key (SEEK). It achieves a Pareto improvement, increasing the resilience against scrubbing attacks without compromising robustness to spoofing. Experiments demonstrate SEEK's superiority over prior method, yielding spoofing robustness gains of +88.2%/+92.3%/+82.0% and scrubbing robustness gains of +10.2%/+6.4%/+24.6% across diverse dataset settings.



## **12. ETrace:Event-Driven Vulnerability Detection in Smart Contracts via LLM-Based Trace Analysis**

cs.CR

4 pages, 1 figure. Submitted to the 16th Asia-Pacific Symposium on  Internetware (Internetware 2025)

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2506.15790v2) [paper-pdf](http://arxiv.org/pdf/2506.15790v2)

**Authors**: Chenyang Peng, Haijun Wang, Yin Wu, Hao Wu, Ming Fan, Yitao Zhao, Ting Liu

**Abstract**: With the advance application of blockchain technology in various fields, ensuring the security and stability of smart contracts has emerged as a critical challenge. Current security analysis methodologies in vulnerability detection can be categorized into static analysis and dynamic analysis methods.However, these existing traditional vulnerability detection methods predominantly rely on analyzing original contract code, not all smart contracts provide accessible code.We present ETrace, a novel event-driven vulnerability detection framework for smart contracts, which uniquely identifies potential vulnerabilities through LLM-powered trace analysis without requiring source code access. By extracting fine-grained event sequences from transaction logs, the framework leverages Large Language Models (LLMs) as adaptive semantic interpreters to reconstruct event analysis through chain-of-thought reasoning. ETrace implements pattern-matching to establish causal links between transaction behavior patterns and known attack behaviors. Furthermore, we validate the effectiveness of ETrace through preliminary experimental results.



## **13. MEF: A Capability-Aware Multi-Encryption Framework for Evaluating Vulnerabilities in Black-Box Large Language Models**

cs.CL

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2505.23404v3) [paper-pdf](http://arxiv.org/pdf/2505.23404v3)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin, Fei Gao, Wenmin Li

**Abstract**: Recent advancements in adversarial jailbreak attacks have revealed significant vulnerabilities in Large Language Models (LLMs), facilitating the evasion of alignment safeguards through increasingly sophisticated prompt manipulations. In this paper, we propose MEF, a capability-aware multi-encryption framework for evaluating vulnerabilities in black-box LLMs. Our key insight is that the effectiveness of jailbreak strategies can be significantly enhanced by tailoring them to the semantic comprehension capabilities of the target model. We present a typology that classifies LLMs into Type I and Type II based on their comprehension levels, and design adaptive attack strategies for each. MEF combines layered semantic mutations and dual-ended encryption techniques, enabling circumvention of input, inference, and output-level defenses. Experimental results demonstrate the superiority of our approach. Remarkably, it achieves a jailbreak success rate of 98.9\% on GPT-4o (29 May 2025 release). Our findings reveal vulnerabilities in current LLMs' alignment defenses.



## **14. Feint and Attack: Attention-Based Strategies for Jailbreaking and Protecting LLMs**

cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2410.16327v2) [paper-pdf](http://arxiv.org/pdf/2410.16327v2)

**Authors**: Rui Pu, Chaozhuo Li, Rui Ha, Zejian Chen, Litian Zhang, Zheng Liu, Lirong Qiu, Zaisheng Ye

**Abstract**: Jailbreak attack can be used to access the vulnerabilities of Large Language Models (LLMs) by inducing LLMs to generate the harmful content. And the most common method of the attack is to construct semantically ambiguous prompts to confuse and mislead the LLMs. To access the security and reveal the intrinsic relation between the input prompt and the output for LLMs, the distribution of attention weight is introduced to analyze the underlying reasons. By using statistical analysis methods, some novel metrics are defined to better describe the distribution of attention weight, such as the Attention Intensity on Sensitive Words (Attn_SensWords), the Attention-based Contextual Dependency Score (Attn_DepScore) and Attention Dispersion Entropy (Attn_Entropy). By leveraging the distinct characteristics of these metrics, the beam search algorithm and inspired by the military strategy "Feint and Attack", an effective jailbreak attack strategy named as Attention-Based Attack (ABA) is proposed. In the ABA, nested attack prompts are employed to divert the attention distribution of the LLMs. In this manner, more harmless parts of the input can be used to attract the attention of the LLMs. In addition, motivated by ABA, an effective defense strategy called as Attention-Based Defense (ABD) is also put forward. Compared with ABA, the ABD can be used to enhance the robustness of LLMs by calibrating the attention distribution of the input prompt. Some comparative experiments have been given to demonstrate the effectiveness of ABA and ABD. Therefore, both ABA and ABD can be used to access the security of the LLMs. The comparative experiment results also give a logical explanation that the distribution of attention weight can bring great influence on the output for LLMs.



## **15. Disappearing Ink: Obfuscation Breaks N-gram Code Watermarks in Theory and Practice**

cs.CR

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.05512v1) [paper-pdf](http://arxiv.org/pdf/2507.05512v1)

**Authors**: Gehao Zhang, Eugene Bagdasarian, Juan Zhai, Shiqing Ma

**Abstract**: Distinguishing AI-generated code from human-written code is becoming crucial for tasks such as authorship attribution, content tracking, and misuse detection. Based on this, N-gram-based watermarking schemes have emerged as prominent, which inject secret watermarks to be detected during the generation.   However, their robustness in code content remains insufficiently evaluated. Most claims rely solely on defenses against simple code transformations or code optimizations as a simulation of attack, creating a questionable sense of robustness. In contrast, more sophisticated schemes already exist in the software engineering world, e.g., code obfuscation, which significantly alters code while preserving functionality. Although obfuscation is commonly used to protect intellectual property or evade software scanners, the robustness of code watermarking techniques against such transformations remains largely unexplored.   In this work, we formally model the code obfuscation and prove the impossibility of N-gram-based watermarking's robustness with only one intuitive and experimentally verified assumption, distribution consistency, satisfied. Given the original false positive rate of the watermarking detection, the ratio that the detector failed on the watermarked code after obfuscation will increase to 1 - fpr.   The experiments have been performed on three SOTA watermarking schemes, two LLMs, two programming languages, four code benchmarks, and four obfuscators. Among them, all watermarking detectors show coin-flipping detection abilities on obfuscated codes (AUROC tightly surrounds 0.5). Among all models, watermarking schemes, and datasets, both programming languages own obfuscators that can achieve attack effects with no detection AUROC higher than 0.6 after the attack. Based on the theoretical and practical observations, we also proposed a potential path of robust code watermarking.



## **16. Response Attack: Exploiting Contextual Priming to Jailbreak Large Language Models**

cs.CL

21 pages, 9 figures. Code and data available at  https://github.com/Dtc7w3PQ/Response-Attack

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.05248v1) [paper-pdf](http://arxiv.org/pdf/2507.05248v1)

**Authors**: Ziqi Miao, Lijun Li, Yuan Xiong, Zhenhua Liu, Pengyu Zhu, Jing Shao

**Abstract**: Contextual priming, where earlier stimuli covertly bias later judgments, offers an unexplored attack surface for large language models (LLMs). We uncover a contextual priming vulnerability in which the previous response in the dialogue can steer its subsequent behavior toward policy-violating content. Building on this insight, we propose Response Attack, which uses an auxiliary LLM to generate a mildly harmful response to a paraphrased version of the original malicious query. They are then formatted into the dialogue and followed by a succinct trigger prompt, thereby priming the target model to generate harmful content. Across eight open-source and proprietary LLMs, RA consistently outperforms seven state-of-the-art jailbreak techniques, achieving higher attack success rates. To mitigate this threat, we construct and release a context-aware safety fine-tuning dataset, which significantly reduces the attack success rate while preserving model capabilities. The code and data are available at https://github.com/Dtc7w3PQ/Response-Attack.



## **17. Transfer Attack for Bad and Good: Explain and Boost Adversarial Transferability across Multimodal Large Language Models**

cs.CV

Accepted by ACM MM 2025

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2405.20090v4) [paper-pdf](http://arxiv.org/pdf/2405.20090v4)

**Authors**: Hao Cheng, Erjia Xiao, Jiayan Yang, Jinhao Duan, Yichi Wang, Jiahang Cao, Qiang Zhang, Le Yang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Multimodal Large Language Models (MLLMs) demonstrate exceptional performance in cross-modality interaction, yet they also suffer adversarial vulnerabilities. In particular, the transferability of adversarial examples remains an ongoing challenge. In this paper, we specifically analyze the manifestation of adversarial transferability among MLLMs and identify the key factors that influence this characteristic. We discover that the transferability of MLLMs exists in cross-LLM scenarios with the same vision encoder and indicate \underline{\textit{two key Factors}} that may influence transferability. We provide two semantic-level data augmentation methods, Adding Image Patch (AIP) and Typography Augment Transferability Method (TATM), which boost the transferability of adversarial examples across MLLMs. To explore the potential impact in the real world, we utilize two tasks that can have both negative and positive societal impacts: \ding{182} Harmful Content Insertion and \ding{183} Information Protection.



## **18. The Hidden Threat in Plain Text: Attacking RAG Data Loaders**

cs.CR

currently under submission

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.05093v1) [paper-pdf](http://arxiv.org/pdf/2507.05093v1)

**Authors**: Alberto Castagnaro, Umberto Salviati, Mauro Conti, Luca Pajola, Simeone Pizzi

**Abstract**: Large Language Models (LLMs) have transformed human-machine interaction since ChatGPT's 2022 debut, with Retrieval-Augmented Generation (RAG) emerging as a key framework that enhances LLM outputs by integrating external knowledge. However, RAG's reliance on ingesting external documents introduces new vulnerabilities. This paper exposes a critical security gap at the data loading stage, where malicious actors can stealthily corrupt RAG pipelines by exploiting document ingestion.   We propose a taxonomy of 9 knowledge-based poisoning attacks and introduce two novel threat vectors -- Content Obfuscation and Content Injection -- targeting common formats (DOCX, HTML, PDF). Using an automated toolkit implementing 19 stealthy injection techniques, we test five popular data loaders, finding a 74.4% attack success rate across 357 scenarios. We further validate these threats on six end-to-end RAG systems -- including white-box pipelines and black-box services like NotebookLM and OpenAI Assistants -- demonstrating high success rates and critical vulnerabilities that bypass filters and silently compromise output integrity. Our results emphasize the urgent need to secure the document ingestion process in RAG systems against covert content manipulations.



## **19. BackFed: An Efficient & Standardized Benchmark Suite for Backdoor Attacks in Federated Learning**

cs.CR

Under review at NeurIPS'25

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04903v1) [paper-pdf](http://arxiv.org/pdf/2507.04903v1)

**Authors**: Thinh Dao, Dung Thuy Nguyen, Khoa D Doan, Kok-Seng Wong

**Abstract**: Federated Learning (FL) systems are vulnerable to backdoor attacks, where adversaries train their local models on poisoned data and submit poisoned model updates to compromise the global model. Despite numerous proposed attacks and defenses, divergent experimental settings, implementation errors, and unrealistic assumptions hinder fair comparisons and valid conclusions about their effectiveness in real-world scenarios. To address this, we introduce BackFed - a comprehensive benchmark suite designed to standardize, streamline, and reliably evaluate backdoor attacks and defenses in FL, with a focus on practical constraints. Our benchmark offers key advantages through its multi-processing implementation that significantly accelerates experimentation and the modular design that enables seamless integration of new methods via well-defined APIs. With a standardized evaluation pipeline, we envision BackFed as a plug-and-play environment for researchers to comprehensively and reliably evaluate new attacks and defenses. Using BackFed, we conduct large-scale studies of representative backdoor attacks and defenses across both Computer Vision and Natural Language Processing tasks with diverse model architectures and experimental settings. Our experiments critically assess the performance of proposed attacks and defenses, revealing unknown limitations and modes of failures under practical conditions. These empirical insights provide valuable guidance for the development of new methods and for enhancing the security of FL systems. Our framework is openly available at https://github.com/thinh-dao/BackFed.



## **20. Who's the Mole? Modeling and Detecting Intention-Hiding Malicious Agents in LLM-Based Multi-Agent Systems**

cs.MA

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04724v1) [paper-pdf](http://arxiv.org/pdf/2507.04724v1)

**Authors**: Yizhe Xie, Congcong Zhu, Xinyue Zhang, Minghao Wang, Chi Liu, Minglu Zhu, Tianqing Zhu

**Abstract**: Multi-agent systems powered by Large Language Models (LLM-MAS) demonstrate remarkable capabilities in collaborative problem-solving. While LLM-MAS exhibit strong collaborative abilities, the security risks in their communication and coordination remain underexplored. We bridge this gap by systematically investigating intention-hiding threats in LLM-MAS, and design four representative attack paradigms that subtly disrupt task completion while maintaining high concealment. These attacks are evaluated in centralized, decentralized, and layered communication structures. Experiments conducted on six benchmark datasets, including MMLU, MMLU-Pro, HumanEval, GSM8K, arithmetic, and biographies, demonstrate that they exhibit strong disruptive capabilities. To identify these threats, we propose a psychology-based detection framework AgentXposed, which combines the HEXACO personality model with the Reid Technique, using progressive questionnaire inquiries and behavior-based monitoring. Experiments conducted on six types of attacks show that our detection framework effectively identifies all types of malicious behaviors. The detection rate for our intention-hiding attacks is slightly lower than that of the two baselines, Incorrect Fact Injection and Dark Traits Injection, demonstrating the effectiveness of intention concealment. Our findings reveal the structural and behavioral risks posed by intention-hiding attacks and offer valuable insights into securing LLM-based multi-agent systems through psychological perspectives, which contributes to a deeper understanding of multi-agent safety. The code and data are available at https://anonymous.4open.science/r/AgentXposed-F814.



## **21. Attacker's Noise Can Manipulate Your Audio-based LLM in the Real World**

cs.CR

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.06256v1) [paper-pdf](http://arxiv.org/pdf/2507.06256v1)

**Authors**: Vinu Sankar Sadasivan, Soheil Feizi, Rajiv Mathews, Lun Wang

**Abstract**: This paper investigates the real-world vulnerabilities of audio-based large language models (ALLMs), such as Qwen2-Audio. We first demonstrate that an adversary can craft stealthy audio perturbations to manipulate ALLMs into exhibiting specific targeted behaviors, such as eliciting responses to wake-keywords (e.g., "Hey Qwen"), or triggering harmful behaviors (e.g. "Change my calendar event"). Subsequently, we show that playing adversarial background noise during user interaction with the ALLMs can significantly degrade the response quality. Crucially, our research illustrates the scalability of these attacks to real-world scenarios, impacting other innocent users when these adversarial noises are played through the air. Further, we discuss the transferrability of the attack, and potential defensive measures.



## **22. Model Inversion Attacks on Llama 3: Extracting PII from Large Language Models**

cs.LG

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2507.04478v1) [paper-pdf](http://arxiv.org/pdf/2507.04478v1)

**Authors**: Sathesh P. Sivashanmugam

**Abstract**: Large language models (LLMs) have transformed natural language processing, but their ability to memorize training data poses significant privacy risks. This paper investigates model inversion attacks on the Llama 3.2 model, a multilingual LLM developed by Meta. By querying the model with carefully crafted prompts, we demonstrate the extraction of personally identifiable information (PII) such as passwords, email addresses, and account numbers. Our findings highlight the vulnerability of even smaller LLMs to privacy attacks and underscore the need for robust defenses. We discuss potential mitigation strategies, including differential privacy and data sanitization, and call for further research into privacy-preserving machine learning techniques.



## **23. Attention Slipping: A Mechanistic Understanding of Jailbreak Attacks and Defenses in LLMs**

cs.CR

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2507.04365v1) [paper-pdf](http://arxiv.org/pdf/2507.04365v1)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: As large language models (LLMs) become more integral to society and technology, ensuring their safety becomes essential. Jailbreak attacks exploit vulnerabilities to bypass safety guardrails, posing a significant threat. However, the mechanisms enabling these attacks are not well understood. In this paper, we reveal a universal phenomenon that occurs during jailbreak attacks: Attention Slipping. During this phenomenon, the model gradually reduces the attention it allocates to unsafe requests in a user query during the attack process, ultimately causing a jailbreak. We show Attention Slipping is consistent across various jailbreak methods, including gradient-based token replacement, prompt-level template refinement, and in-context learning. Additionally, we evaluate two defenses based on query perturbation, Token Highlighter and SmoothLLM, and find they indirectly mitigate Attention Slipping, with their effectiveness positively correlated with the degree of mitigation achieved. Inspired by this finding, we propose Attention Sharpening, a new defense that directly counters Attention Slipping by sharpening the attention score distribution using temperature scaling. Experiments on four leading LLMs (Gemma2-9B-It, Llama3.1-8B-It, Qwen2.5-7B-It, Mistral-7B-It v0.2) show that our method effectively resists various jailbreak attacks while maintaining performance on benign tasks on AlpacaEval. Importantly, Attention Sharpening introduces no additional computational or memory overhead, making it an efficient and practical solution for real-world deployment.



## **24. Hijacking JARVIS: Benchmarking Mobile GUI Agents against Unprivileged Third Parties**

cs.CR

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2507.04227v1) [paper-pdf](http://arxiv.org/pdf/2507.04227v1)

**Authors**: Guohong Liu, Jialei Ye, Jiacheng Liu, Yuanchun Li, Wei Liu, Pengzhi Gao, Jian Luan, Yunxin Liu

**Abstract**: Mobile GUI agents are designed to autonomously execute diverse device-control tasks by interpreting and interacting with mobile screens. Despite notable advancements, their resilience in real-world scenarios where screen content may be partially manipulated by untrustworthy third parties remains largely unexplored. Owing to their black-box and autonomous nature, these agents are vulnerable to manipulations that could compromise user devices. In this work, we present the first systematic investigation into the vulnerabilities of mobile GUI agents. We introduce a scalable attack simulation framework AgentHazard, which enables flexible and targeted modifications of screen content within existing applications. Leveraging this framework, we develop a comprehensive benchmark suite comprising both a dynamic task execution environment and a static dataset of vision-language-action tuples, totaling over 3,000 attack scenarios. The dynamic environment encompasses 58 reproducible tasks in an emulator with various types of hazardous UI content, while the static dataset is constructed from 210 screenshots collected from 14 popular commercial apps. Importantly, our content modifications are designed to be feasible for unprivileged third parties. We evaluate 7 widely-used mobile GUI agents and 5 common backbone models using our benchmark. Our findings reveal that all examined agents are significantly influenced by misleading third-party content (with an average misleading rate of 28.8% in human-crafted attack scenarios) and that their vulnerabilities are closely linked to the employed perception modalities and backbone LLMs. Furthermore, we assess training-based mitigation strategies, highlighting both the challenges and opportunities for enhancing the robustness of mobile GUI agents. Our code and data will be released at https://agenthazard.github.io.



## **25. Can Large Language Models Automate the Refinement of Cellular Network Specifications?**

cs.CR

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2507.04214v1) [paper-pdf](http://arxiv.org/pdf/2507.04214v1)

**Authors**: Jianshuo Dong, Tianyi Zhang, Feng Yan, Yuanjie Li, Hewu Li, Han Qiu

**Abstract**: Cellular networks serve billions of users globally, yet concerns about reliability and security persist due to weaknesses in 3GPP standards. However, traditional analysis methods, including manual inspection and automated tools, struggle with increasingly expanding cellular network specifications. This paper investigates the feasibility of Large Language Models (LLMs) for automated cellular network specification refinement. To advance it, we leverage 200,000+ approved 3GPP Change Requests (CRs) that document specification revisions, constructing a valuable dataset for domain tasks. We introduce CR-eval, a principled evaluation framework, and benchmark 16 state-of-the-art LLMs, demonstrating that top models can discover security-related weaknesses in over 127 out of 200 test cases within five trials. To bridge potential gaps, we explore LLM specialization techniques, including fine-tuning an 8B model to match or surpass advanced LLMs like GPT-4o and DeepSeek-R1. Evaluations on 30 cellular attacks identify open challenges for achieving full automation. These findings confirm that LLMs can automate the refinement of cellular network specifications and provide valuable insights to guide future research in this direction.



## **26. False Alarms, Real Damage: Adversarial Attacks Using LLM-based Models on Text-based Cyber Threat Intelligence Systems**

cs.CR

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2507.06252v1) [paper-pdf](http://arxiv.org/pdf/2507.06252v1)

**Authors**: Samaneh Shafee, Alysson Bessani, Pedro M. Ferreira

**Abstract**: Cyber Threat Intelligence (CTI) has emerged as a vital complementary approach that operates in the early phases of the cyber threat lifecycle. CTI involves collecting, processing, and analyzing threat data to provide a more accurate and rapid understanding of cyber threats. Due to the large volume of data, automation through Machine Learning (ML) and Natural Language Processing (NLP) models is essential for effective CTI extraction. These automated systems leverage Open Source Intelligence (OSINT) from sources like social networks, forums, and blogs to identify Indicators of Compromise (IoCs). Although prior research has focused on adversarial attacks on specific ML models, this study expands the scope by investigating vulnerabilities within various components of the entire CTI pipeline and their susceptibility to adversarial attacks. These vulnerabilities arise because they ingest textual inputs from various open sources, including real and potentially fake content. We analyse three types of attacks against CTI pipelines, including evasion, flooding, and poisoning, and assess their impact on the system's information selection capabilities. Specifically, on fake text generation, the work demonstrates how adversarial text generation techniques can create fake cybersecurity and cybersecurity-like text that misleads classifiers, degrades performance, and disrupts system functionality. The focus is primarily on the evasion attack, as it precedes and enables flooding and poisoning attacks within the CTI pipeline.



## **27. Membership Inference Attacks on Large-Scale Models: A Survey**

cs.LG

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2503.19338v2) [paper-pdf](http://arxiv.org/pdf/2503.19338v2)

**Authors**: Hengyu Wu, Yang Cao

**Abstract**: The adoption of the Large Language Model (LLM) has accelerated dramatically since ChatGPT from OpenAI went online in November 2022. Recent advances in Large Multimodal Models (LMMs), which process diverse data types and enable interaction through various channels, have expanded beyond the text-to-text limitations of early LLMs, attracting significant and concurrent attention from both researchers and industry. While LLMs and LMMs are starting to spread widely, concerns about their privacy risks are increasing as well. Membership Inference Attacks (MIAs) are techniques used to determine whether a particular data point was part of a model's training set, which is a key metric for assessing the privacy vulnerabilities of machine learning models. Hu et al. show that various machine learning algorithms are vulnerable to MIA. Despite extensive studies on MIAs in classic models, there remains a lack of systematic surveys addressing their effectiveness and limitations in advanced large-scale models like LLMs and LMMs. In this paper, we systematically reviewed recent studies of MIA against LLMs and LMMs. We analyzed and categorized each attack based on its methodology, scenario, and targeted model, and we discussed the limitations of existing research. In addition to examining attacks on pre-training and fine-tuning stages, we also explore MIAs that target other development pipelines, including Retrieval-Augmented Generation (RAG) and the model alignment process. Based on the survey, we provide suggestions for future studies to improve the robustness of MIA in large-scale AI models.



## **28. A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models**

cs.IR

Accepted by ACL 2025 Findings

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2507.05288v1) [paper-pdf](http://arxiv.org/pdf/2507.05288v1)

**Authors**: Shuliang Liu, Hongyi Liu, Aiwei Liu, Bingchen Duan, Qi Zheng, Yibo Yan, He Geng, Peijie Jiang, Jia Liu, Xuming Hu

**Abstract**: The widespread deployment of large language models (LLMs) across critical domains has amplified the societal risks posed by algorithmically generated misinformation. Unlike traditional false content, LLM-generated misinformation can be self-reinforcing, highly plausible, and capable of rapid propagation across multiple languages, which traditional detection methods fail to mitigate effectively. This paper introduces a proactive defense paradigm, shifting from passive post hoc detection to anticipatory mitigation strategies. We propose a Three Pillars framework: (1) Knowledge Credibility, fortifying the integrity of training and deployed data; (2) Inference Reliability, embedding self-corrective mechanisms during reasoning; and (3) Input Robustness, enhancing the resilience of model interfaces against adversarial attacks. Through a comprehensive survey of existing techniques and a comparative meta-analysis, we demonstrate that proactive defense strategies offer up to 63\% improvement over conventional methods in misinformation prevention, despite non-trivial computational overhead and generalization challenges. We argue that future research should focus on co-designing robust knowledge foundations, reasoning certification, and attack-resistant interfaces to ensure LLMs can effectively counter misinformation across varied domains.



## **29. We Urgently Need Privilege Management in MCP: A Measurement of API Usage in MCP Ecosystems**

cs.CR

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2507.06250v1) [paper-pdf](http://arxiv.org/pdf/2507.06250v1)

**Authors**: Zhihao Li, Kun Li, Boyang Ma, Minghui Xu, Yue Zhang, Xiuzhen Cheng

**Abstract**: The Model Context Protocol (MCP) has emerged as a widely adopted mechanism for connecting large language models to external tools and resources. While MCP promises seamless extensibility and rich integrations, it also introduces a substantially expanded attack surface: any plugin can inherit broad system privileges with minimal isolation or oversight. In this work, we conduct the first large-scale empirical analysis of MCP security risks. We develop an automated static analysis framework and systematically examine 2,562 real-world MCP applications spanning 23 functional categories. Our measurements reveal that network and system resource APIs dominate usage patterns, affecting 1,438 and 1,237 servers respectively, while file and memory resources are less frequent but still significant. We find that Developer Tools and API Development plugins are the most API-intensive, and that less popular plugins often contain disproportionately high-risk operations. Through concrete case studies, we demonstrate how insufficient privilege separation enables privilege escalation, misinformation propagation, and data tampering. Based on these findings, we propose a detailed taxonomy of MCP resource access, quantify security-relevant API usage, and identify open challenges for building safer MCP ecosystems, including dynamic permission models and automated trust assessment.



## **30. Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States**

cs.LG

4 figures

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2503.09066v2) [paper-pdf](http://arxiv.org/pdf/2503.09066v2)

**Authors**: Xin Wei Chia, Swee Liang Wong, Jonathan Pan

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.



## **31. Blackbox Dataset Inference for LLM**

cs.CR

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03619v1) [paper-pdf](http://arxiv.org/pdf/2507.03619v1)

**Authors**: Ruikai Zhou, Kang Yang, Xun Chen, Wendy Hui Wang, Guanhong Tao, Jun Xu

**Abstract**: Today, the training of large language models (LLMs) can involve personally identifiable information and copyrighted material, incurring dataset misuse. To mitigate the problem of dataset misuse, this paper explores \textit{dataset inference}, which aims to detect if a suspect model $\mathcal{M}$ used a victim dataset $\mathcal{D}$ in training. Previous research tackles dataset inference by aggregating results of membership inference attacks (MIAs) -- methods to determine whether individual samples are a part of the training dataset. However, restricted by the low accuracy of MIAs, previous research mandates grey-box access to $\mathcal{M}$ to get intermediate outputs (probabilities, loss, perplexity, etc.) for obtaining satisfactory results. This leads to reduced practicality, as LLMs, especially those deployed for profits, have limited incentives to return the intermediate outputs.   In this paper, we propose a new method of dataset inference with only black-box access to the target model (i.e., assuming only the text-based responses of the target model are available). Our method is enabled by two sets of locally built reference models, one set involving $\mathcal{D}$ in training and the other not. By measuring which set of reference model $\mathcal{M}$ is closer to, we determine if $\mathcal{M}$ used $\mathcal{D}$ for training. Evaluations of real-world LLMs in the wild show that our method offers high accuracy in all settings and presents robustness against bypassing attempts.



## **32. Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection**

cs.CV

16 pages

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.02844v1) [paper-pdf](http://arxiv.org/pdf/2507.02844v1)

**Authors**: Ziqi Miao, Yi Ding, Lijun Li, Jing Shao

**Abstract**: With the emergence of strong visual-language capabilities, multimodal large language models (MLLMs) have demonstrated tremendous potential for real-world applications. However, the security vulnerabilities exhibited by the visual modality pose significant challenges to deploying such models in open-world environments. Recent studies have successfully induced harmful responses from target MLLMs by encoding harmful textual semantics directly into visual inputs. However, in these approaches, the visual modality primarily serves as a trigger for unsafe behavior, often exhibiting semantic ambiguity and lacking grounding in realistic scenarios. In this work, we define a novel setting: visual-centric jailbreak, where visual information serves as a necessary component in constructing a complete and realistic jailbreak context. Building on this setting, we propose the VisCo (Visual Contextual) Attack. VisCo fabricates contextual dialogue using four distinct visual-focused strategies, dynamically generating auxiliary images when necessary to construct a visual-centric jailbreak scenario. To maximize attack effectiveness, it incorporates automatic toxicity obfuscation and semantic refinement to produce a final attack prompt that reliably triggers harmful responses from the target black-box MLLMs. Specifically, VisCo achieves a toxicity score of 4.78 and an Attack Success Rate (ASR) of 85% on MM-SafetyBench against GPT-4o, significantly outperforming the baseline, which performs a toxicity score of 2.48 and an ASR of 22.2%. The code is available at https://github.com/Dtc7w3PQ/Visco-Attack.



## **33. Is Reasoning All You Need? Probing Bias in the Age of Reasoning Language Models**

cs.CL

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.02799v1) [paper-pdf](http://arxiv.org/pdf/2507.02799v1)

**Authors**: Riccardo Cantini, Nicola Gabriele, Alessio Orsino, Domenico Talia

**Abstract**: Reasoning Language Models (RLMs) have gained traction for their ability to perform complex, multi-step reasoning tasks through mechanisms such as Chain-of-Thought (CoT) prompting or fine-tuned reasoning traces. While these capabilities promise improved reliability, their impact on robustness to social biases remains unclear. In this work, we leverage the CLEAR-Bias benchmark, originally designed for Large Language Models (LLMs), to investigate the adversarial robustness of RLMs to bias elicitation. We systematically evaluate state-of-the-art RLMs across diverse sociocultural dimensions, using an LLM-as-a-judge approach for automated safety scoring and leveraging jailbreak techniques to assess the strength of built-in safety mechanisms. Our evaluation addresses three key questions: (i) how the introduction of reasoning capabilities affects model fairness and robustness; (ii) whether models fine-tuned for reasoning exhibit greater safety than those relying on CoT prompting at inference time; and (iii) how the success rate of jailbreak attacks targeting bias elicitation varies with the reasoning mechanisms employed. Our findings reveal a nuanced relationship between reasoning capabilities and bias safety. Surprisingly, models with explicit reasoning, whether via CoT prompting or fine-tuned reasoning traces, are generally more vulnerable to bias elicitation than base models without such mechanisms, suggesting reasoning may unintentionally open new pathways for stereotype reinforcement. Reasoning-enabled models appear somewhat safer than those relying on CoT prompting, which are particularly prone to contextual reframing attacks through storytelling prompts, fictional personas, or reward-shaped instructions. These results challenge the assumption that reasoning inherently improves robustness and underscore the need for more bias-aware approaches to reasoning design.



## **34. StructTransform: A Scalable Attack Surface for Safety-Aligned Large Language Models**

cs.LG

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2502.11853v2) [paper-pdf](http://arxiv.org/pdf/2502.11853v2)

**Authors**: Shehel Yoosuf, Temoor Ali, Ahmed Lekssays, Mashael AlSabah, Issa Khalil

**Abstract**: In this work, we present a series of structure transformation attacks on LLM alignment, where we encode natural language intent using diverse syntax spaces, ranging from simple structure formats and basic query languages (e.g., SQL) to new novel spaces and syntaxes created entirely by LLMs. Our extensive evaluation shows that our simplest attacks can achieve close to a 90% success rate, even on strict LLMs (such as Claude 3.5 Sonnet) using SOTA alignment mechanisms. We improve the attack performance further by using an adaptive scheme that combines structure transformations along with existing content transformations, resulting in over 96% ASR with 0% refusals.   To generalize our attacks, we explore numerous structure formats, including syntaxes purely generated by LLMs. Our results indicate that such novel syntaxes are easy to generate and result in a high ASR, suggesting that defending against our attacks is not a straightforward process. Finally, we develop a benchmark and evaluate existing safety-alignment defenses against it, showing that most of them fail with 100% ASR. Our results show that existing safety alignment mostly relies on token-level patterns without recognizing harmful concepts, highlighting and motivating the need for serious research efforts in this direction. As a case study, we demonstrate how attackers can use our attack to easily generate a sample malware and a corpus of fraudulent SMS messages, which perform well in bypassing detection.



## **35. Evaluating Language Models For Threat Detection in IoT Security Logs**

cs.CR

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.02390v1) [paper-pdf](http://arxiv.org/pdf/2507.02390v1)

**Authors**: Jorge J. Tejero-Fernández, Alfonso Sánchez-Macián

**Abstract**: Log analysis is a relevant research field in cybersecurity as they can provide a source of information for the detection of threats to networks and systems. This paper presents a pipeline to use fine-tuned Large Language Models (LLMs) for anomaly detection and mitigation recommendation using IoT security logs. Utilizing classical machine learning classifiers as a baseline, three open-source LLMs are compared for binary and multiclass anomaly detection, with three strategies: zero-shot, few-shot prompting and fine-tuning using an IoT dataset. LLMs give better results on multi-class attack classification than the corresponding baseline models. By mapping detected threats to MITRE CAPEC, defining a set of IoT-specific mitigation actions, and fine-tuning the models with those actions, the models are able to provide a combined detection and recommendation guidance.



## **36. SecAlign: Defending Against Prompt Injection with Preference Optimization**

cs.CR

ACM CCS 2025. Key words: prompt injection defense, LLM security,  LLM-integrated applications

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2410.05451v3) [paper-pdf](http://arxiv.org/pdf/2410.05451v3)

**Authors**: Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, David Wagner, Chuan Guo

**Abstract**: Large language models (LLMs) are becoming increasingly prevalent in modern software systems, interfacing between the user and the Internet to assist with tasks that require advanced language understanding. To accomplish these tasks, the LLM often uses external data sources such as user documents, web retrieval, results from API calls, etc. This opens up new avenues for attackers to manipulate the LLM via prompt injection. Adversarial prompts can be injected into external data sources to override the system's intended instruction and instead execute a malicious instruction. To mitigate this vulnerability, we propose a new defense called SecAlign based on the technique of preference optimization. Our defense first constructs a preference dataset with prompt-injected inputs, secure outputs (ones that respond to the legitimate instruction), and insecure outputs (ones that respond to the injection). We then perform preference optimization on this dataset to teach the LLM to prefer the secure output over the insecure one. This provides the first known method that reduces the success rates of various prompt injections to <10%, even against attacks much more sophisticated than ones seen during training. This indicates our defense generalizes well against unknown and yet-to-come attacks. Also, SecAlign models are still practical with similar utility to the one before defensive training in our evaluations. Our code is at https://github.com/facebookresearch/SecAlign



## **37. MGC: A Compiler Framework Exploiting Compositional Blindness in Aligned LLMs for Malware Generation**

cs.CR

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.02057v1) [paper-pdf](http://arxiv.org/pdf/2507.02057v1)

**Authors**: Lu Yan, Zhuo Zhang, Xiangzhe Xu, Shengwei An, Guangyu Shen, Zhou Xuan, Xuan Chen, Xiangyu Zhang

**Abstract**: Large language models (LLMs) have democratized software development, reducing the expertise barrier for programming complex applications. This accessibility extends to malicious software development, raising significant security concerns. While LLM providers have implemented alignment mechanisms to prevent direct generation of overtly malicious code, these safeguards predominantly evaluate individual prompts in isolation, overlooking a critical vulnerability: malicious operations can be systematically decomposed into benign-appearing sub-tasks. In this paper, we introduce the Malware Generation Compiler (MGC), a novel framework that leverages this vulnerability through modular decomposition and alignment-evasive generation. MGC employs a specialized Malware Description Intermediate Representation (MDIR) to bridge high-level malicious intents and benign-appearing code snippets. Extensive evaluation demonstrates that our attack reliably generates functional malware across diverse task specifications and categories, outperforming jailbreaking methods by +365.79% and underground services by +78.07% in correctness on three benchmark datasets. Case studies further show that MGC can reproduce and even enhance 16 real-world malware samples. This work provides critical insights for security researchers by exposing the risks of compositional attacks against aligned AI systems. Demonstrations are available at https://sites.google.com/view/malware-generation-compiler.



## **38. Tuning without Peeking: Provable Privacy and Generalization Bounds for LLM Post-Training**

cs.LG

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01752v1) [paper-pdf](http://arxiv.org/pdf/2507.01752v1)

**Authors**: Ismail Labiad, Mathurin Videau, Matthieu Kowalski, Marc Schoenauer, Alessandro Leite, Julia Kempe, Olivier Teytaud

**Abstract**: Gradient-based optimization is the workhorse of deep learning, offering efficient and scalable training via backpropagation. However, its reliance on large volumes of labeled data raises privacy and security concerns such as susceptibility to data poisoning attacks and the risk of overfitting. In contrast, black box optimization methods, which treat the model as an opaque function, relying solely on function evaluations to guide optimization, offer a promising alternative in scenarios where data access is restricted, adversarial risks are high, or overfitting is a concern. However, black box methods also pose significant challenges, including poor scalability to high-dimensional parameter spaces, as prevalent in large language models (LLMs), and high computational costs due to reliance on numerous model evaluations. This paper introduces BBoxER, an evolutionary black-box method for LLM post-training that induces an information bottleneck via implicit compression of the training data. Leveraging the tractability of information flow, we provide strong theoretical bounds on generalization, differential privacy, susceptibility to data poisoning attacks, and robustness to extraction attacks. BBoxER operates on top of pre-trained LLMs, offering a lightweight and modular enhancement suitable for deployment in restricted or privacy-sensitive environments, in addition to non-vacuous generalization guarantees. In experiments with LLMs, we demonstrate empirically that Retrofitting methods are able to learn, showing how a few iterations of BBoxER improve performance and generalize well on a benchmark of reasoning datasets. This positions BBoxER as an attractive add-on on top of gradient-based optimization.



## **39. Graph Representation-based Model Poisoning on Federated LLMs in CyberEdge Networks**

cs.CR

7 pages, 5 figures

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01694v1) [paper-pdf](http://arxiv.org/pdf/2507.01694v1)

**Authors**: Hanlin Cai, Haofan Dong, Houtianfu Wang, Kai Li, Ozgur B. Akan

**Abstract**: Federated large language models (FedLLMs) provide powerful generative capabilities in CyberEdge networks while protecting data privacy. However, FedLLMs remains highly vulnerable to model poisoning attacks. This article first reviews recent model poisoning techniques and existing defense mechanisms for FedLLMs, highlighting critical limitations, particularly under non-IID text distributions. In particular, current defenses primarily utilize distance-based outlier detection or norm constraints, operating under the assumption that adversarial updates significantly diverge from benign statistics. This assumption can fail when facing adaptive attackers targeting billionparameter LLMs. Next, this article investigates emerging Graph Representation-Based Model Poisoning (GRMP), a novel attack paradigm that leverages higher-order correlations among honest client gradients to synthesize malicious updates indistinguishable from legitimate model updates. GRMP can effectively evade advanced defenses, resulting in substantial accuracy loss and performance degradation. Moreover, this article outlines a research roadmap emphasizing the importance of graph-aware secure aggregation methods, FedLLMs-specific vulnerability metrics, and evaluation frameworks to strengthen the robustness of future federated language model deployments.



## **40. SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism**

cs.CR

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01513v1) [paper-pdf](http://arxiv.org/pdf/2507.01513v1)

**Authors**: Beitao Chen, Xinyu Lyu, Lianli Gao, Jingkuan Song, Heng Tao Shen

**Abstract**: By incorporating visual inputs, Multimodal Large Language Models (MLLMs) extend LLMs to support visual reasoning. However, this integration also introduces new vulnerabilities, making MLLMs susceptible to multimodal jailbreak attacks and hindering their safe deployment.Existing defense methods, including Image-to-Text Translation, Safe Prompting, and Multimodal Safety Tuning, attempt to address this by aligning multimodal inputs with LLMs' built-in safeguards.Yet, they fall short in uncovering root causes of multimodal vulnerabilities, particularly how harmful multimodal tokens trigger jailbreak in MLLMs? Consequently, they remain vulnerable to text-driven multimodal jailbreaks, often exhibiting overdefensive behaviors and imposing heavy training overhead.To bridge this gap, we present an comprehensive analysis of where, how and which harmful multimodal tokens bypass safeguards in MLLMs. Surprisingly, we find that less than 1% tokens in early-middle layers are responsible for inducing unsafe behaviors, highlighting the potential of precisely removing a small subset of harmful tokens, without requiring safety tuning, can still effectively improve safety against jailbreaks. Motivated by this, we propose Safe Prune-then-Restore (SafePTR), an training-free defense framework that selectively prunes harmful tokens at vulnerable layers while restoring benign features at subsequent layers.Without incurring additional computational overhead, SafePTR significantly enhances the safety of MLLMs while preserving efficiency. Extensive evaluations across three MLLMs and five benchmarks demonstrate SafePTR's state-of-the-art performance in mitigating jailbreak risks without compromising utility.



## **41. Don't Say No: Jailbreaking LLM by Suppressing Refusal**

cs.CL

Accepted by ACL 2025 Findings

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2404.16369v3) [paper-pdf](http://arxiv.org/pdf/2404.16369v3)

**Authors**: Yukai Zhou, Jian Lou, Zhijie Huang, Zhan Qin, Yibei Yang, Wenjie Wang

**Abstract**: Ensuring the safety alignment of Large Language Models (LLMs) is critical for generating responses consistent with human values. However, LLMs remain vulnerable to jailbreaking attacks, where carefully crafted prompts manipulate them into producing toxic content. One category of such attacks reformulates the task as an optimization problem, aiming to elicit affirmative responses from the LLM. However, these methods heavily rely on predefined objectionable behaviors, limiting their effectiveness and adaptability to diverse harmful queries. In this study, we first identify why the vanilla target loss is suboptimal and then propose enhancements to the loss objective. We introduce DSN (Don't Say No) attack, which combines a cosine decay schedule method with refusal suppression to achieve higher success rates. Extensive experiments demonstrate that DSN outperforms baseline attacks and achieves state-of-the-art attack success rates (ASR). DSN also shows strong universality and transferability to unseen datasets and black-box models.



## **42. ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks**

cs.LG

ICML 2025

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01321v1) [paper-pdf](http://arxiv.org/pdf/2507.01321v1)

**Authors**: Zhiyao Ren, Siyuan Liang, Aishan Liu, Dacheng Tao

**Abstract**: In-context learning (ICL) has demonstrated remarkable success in large language models (LLMs) due to its adaptability and parameter-free nature. However, it also introduces a critical vulnerability to backdoor attacks, where adversaries can manipulate LLM behaviors by simply poisoning a few ICL demonstrations. In this paper, we propose, for the first time, the dual-learning hypothesis, which posits that LLMs simultaneously learn both the task-relevant latent concepts and backdoor latent concepts within poisoned demonstrations, jointly influencing the probability of model outputs. Through theoretical analysis, we derive an upper bound for ICL backdoor effects, revealing that the vulnerability is dominated by the concept preference ratio between the task and the backdoor. Motivated by these findings, we propose ICLShield, a defense mechanism that dynamically adjusts the concept preference ratio. Our method encourages LLMs to select clean demonstrations during the ICL phase by leveraging confidence and similarity scores, effectively mitigating susceptibility to backdoor attacks. Extensive experiments across multiple LLMs and tasks demonstrate that our method achieves state-of-the-art defense effectiveness, significantly outperforming existing approaches (+26.02% on average). Furthermore, our method exhibits exceptional adaptability and defensive performance even for closed-source models (e.g., GPT-4).



## **43. GenBFA: An Evolutionary Optimization Approach to Bit-Flip Attacks on LLMs**

cs.CR

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2411.13757v4) [paper-pdf](http://arxiv.org/pdf/2411.13757v4)

**Authors**: Sanjay Das, Swastik Bhattacharya, Souvik Kundu, Shamik Kundu, Anand Menon, Arnab Raha, Kanad Basu

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing (NLP), excelling in tasks like text generation and summarization. However, their increasing adoption in mission-critical applications raises concerns about hardware-based threats, particularly bit-flip attacks (BFAs). BFAs, enabled by fault injection methods such as Rowhammer, target model parameters in memory, compromising both integrity and performance. Identifying critical parameters for BFAs in the vast parameter space of LLMs poses significant challenges. While prior research suggests transformer-based architectures are inherently more robust to BFAs compared to traditional deep neural networks, we challenge this assumption. For the first time, we demonstrate that as few as three bit-flips can cause catastrophic performance degradation in an LLM with billions of parameters. Current BFA techniques are inadequate for exploiting this vulnerability due to the difficulty of efficiently identifying critical parameters within the immense parameter space. To address this, we propose AttentionBreaker, a novel framework tailored for LLMs that enables efficient traversal of the parameter space to identify critical parameters. Additionally, we introduce GenBFA, an evolutionary optimization strategy designed to refine the search further, isolating the most critical bits for an efficient and effective attack. Empirical results reveal the profound vulnerability of LLMs to AttentionBreaker. For example, merely three bit-flips (4.129 x 10^-9% of total parameters) in the LLaMA3-8B-Instruct 8-bit quantized (W8) model result in a complete performance collapse: accuracy on MMLU tasks drops from 67.3% to 0%, and Wikitext perplexity skyrockets from 12.6 to 4.72 x 10^5. These findings underscore the effectiveness of AttentionBreaker in uncovering and exploiting critical vulnerabilities within LLM architectures.



## **44. Defensive Adversarial CAPTCHA: A Semantics-Driven Framework for Natural Adversarial Example Generation**

cs.CV

13 pages, 6 figures

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2506.10685v3) [paper-pdf](http://arxiv.org/pdf/2506.10685v3)

**Authors**: Xia Du, Xiaoyuan Liu, Jizhe Zhou, Zheng Lin, Chi-man Pun, Cong Wu, Tao Li, Zhe Chen, Wei Ni, Jun Luo

**Abstract**: Traditional CAPTCHA (Completely Automated Public Turing Test to Tell Computers and Humans Apart) schemes are increasingly vulnerable to automated attacks powered by deep neural networks (DNNs). Existing adversarial attack methods often rely on the original image characteristics, resulting in distortions that hinder human interpretation and limit their applicability in scenarios where no initial input images are available. To address these challenges, we propose the Unsourced Adversarial CAPTCHA (DAC), a novel framework that generates high-fidelity adversarial examples guided by attacker-specified semantics information. Leveraging a Large Language Model (LLM), DAC enhances CAPTCHA diversity and enriches the semantic information. To address various application scenarios, we examine the white-box targeted attack scenario and the black box untargeted attack scenario. For target attacks, we introduce two latent noise variables that are alternately guided in the diffusion step to achieve robust inversion. The synergy between gradient guidance and latent variable optimization achieved in this way ensures that the generated adversarial examples not only accurately align with the target conditions but also achieve optimal performance in terms of distributional consistency and attack effectiveness. In untargeted attacks, especially for black-box scenarios, we introduce bi-path unsourced adversarial CAPTCHA (BP-DAC), a two-step optimization strategy employing multimodal gradients and bi-path optimization for efficient misclassification. Experiments show that the defensive adversarial CAPTCHA generated by BP-DAC is able to defend against most of the unknown models, and the generated CAPTCHA is indistinguishable to both humans and DNNs.



## **45. SafeMobile: Chain-level Jailbreak Detection and Automated Evaluation for Multimodal Mobile Agents**

cs.AI

12 pages

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2507.00841v1) [paper-pdf](http://arxiv.org/pdf/2507.00841v1)

**Authors**: Siyuan Liang, Tianmeng Fang, Zhe Liu, Aishan Liu, Yan Xiao, Jinyuan He, Ee-Chien Chang, Xiaochun Cao

**Abstract**: With the wide application of multimodal foundation models in intelligent agent systems, scenarios such as mobile device control, intelligent assistant interaction, and multimodal task execution are gradually relying on such large model-driven agents. However, the related systems are also increasingly exposed to potential jailbreak risks. Attackers may induce the agents to bypass the original behavioral constraints through specific inputs, and then trigger certain risky and sensitive operations, such as modifying settings, executing unauthorized commands, or impersonating user identities, which brings new challenges to system security. Existing security measures for intelligent agents still have limitations when facing complex interactions, especially in detecting potentially risky behaviors across multiple rounds of conversations or sequences of tasks. In addition, an efficient and consistent automated methodology to assist in assessing and determining the impact of such risks is currently lacking. This work explores the security issues surrounding mobile multimodal agents, attempts to construct a risk discrimination mechanism by incorporating behavioral sequence information, and designs an automated assisted assessment scheme based on a large language model. Through preliminary validation in several representative high-risk tasks, the results show that the method can improve the recognition of risky behaviors to some extent and assist in reducing the probability of agents being jailbroken. We hope that this study can provide some valuable references for the security risk modeling and protection of multimodal intelligent agent systems.



## **46. CAVALRY-V: A Large-Scale Generator Framework for Adversarial Attacks on Video MLLMs**

cs.CV

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2507.00817v1) [paper-pdf](http://arxiv.org/pdf/2507.00817v1)

**Authors**: Jiaming Zhang, Rui Hu, Qing Guo, Wei Yang Bryan Lim

**Abstract**: Video Multimodal Large Language Models (V-MLLMs) have shown impressive capabilities in temporal reasoning and cross-modal understanding, yet their vulnerability to adversarial attacks remains underexplored due to unique challenges: complex cross-modal reasoning mechanisms, temporal dependencies, and computational constraints. We present CAVALRY-V (Cross-modal Language-Vision Adversarial Yielding for Videos), a novel framework that directly targets the critical interface between visual perception and language generation in V-MLLMs. Our approach introduces two key innovations: (1) a dual-objective semantic-visual loss function that simultaneously disrupts the model's text generation logits and visual representations to undermine cross-modal integration, and (2) a computationally efficient two-stage generator framework that combines large-scale pre-training for cross-model transferability with specialized fine-tuning for spatiotemporal coherence. Empirical evaluation on comprehensive video understanding benchmarks demonstrates that CAVALRY-V significantly outperforms existing attack methods, achieving 22.8% average improvement over the best baseline attacks on both commercial systems (GPT-4.1, Gemini 2.0) and open-source models (QwenVL-2.5, InternVL-2.5, Llava-Video, Aria, MiniCPM-o-2.6). Our framework achieves flexibility through implicit temporal coherence modeling rather than explicit regularization, enabling significant performance improvements even on image understanding (34.4% average gain). This capability demonstrates CAVALRY-V's potential as a foundational approach for adversarial research across multimodal systems.



## **47. Impact of Fine-Tuning Methods on Memorization in Large Language Models**

cs.CL

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2507.00258v1) [paper-pdf](http://arxiv.org/pdf/2507.00258v1)

**Authors**: Jie Hou, Chuxiong Wu, Lannan Luo, Qiang Zeng

**Abstract**: As the capabilities of pre-trained large language models (LLMs) continue to advance, the "pre-train and fine-tune" paradigm has become increasingly mainstream, leading to the development of various fine-tuning methods. However, the privacy risks arising from memorization during fine-tuning have received relatively little attention. To address this gap, we categorize popular fine-tuning approaches and assess their impact on memorization through the lens of membership inference attacks (MIAs). Our results show that, compared to parameter-based fine-tuning, prompt-based fine-tuning achieves competitive performance while exhibiting lower vulnerability to MIAs. Furthermore, prompt-based methods maintain low memorization regardless of model scale. These findings suggest that parameter-based fine-tuning is more prone to leaking private information, whereas prompt-based fine-tuning serves as a more privacy-preserving option.



## **48. Trust & Safety of LLMs and LLMs in Trust & Safety**

cs.AI

11 pages

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2412.02113v2) [paper-pdf](http://arxiv.org/pdf/2412.02113v2)

**Authors**: Doohee You, Dan Chon

**Abstract**: In recent years, Large Language Models (LLMs) have garnered considerable attention for their remarkable abilities in natural language processing tasks. However, their widespread adoption has raised concerns pertaining to trust and safety. This systematic review investigates the current research landscape on trust and safety in LLMs, with a particular focus on the novel application of LLMs within the field of Trust and Safety itself. We delve into the complexities of utilizing LLMs in domains where maintaining trust and safety is paramount, offering a consolidated perspective on this emerging trend.\   By synthesizing findings from various studies, we identify key challenges and potential solutions, aiming to benefit researchers and practitioners seeking to understand the nuanced interplay between LLMs and Trust and Safety.   This review provides insights on best practices for using LLMs in Trust and Safety, and explores emerging risks such as prompt injection and jailbreak attacks. Ultimately, this study contributes to a deeper understanding of how LLMs can be effectively and responsibly utilized to enhance trust and safety in the digital realm.



## **49. Logit-Gap Steering: Efficient Short-Suffix Jailbreaks for Aligned Large Language Models**

cs.CR

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24056v1) [paper-pdf](http://arxiv.org/pdf/2506.24056v1)

**Authors**: Tung-Ling Li, Hongliang Liu

**Abstract**: We introduce logit-gap steering, a fast jailbreak framework that casts the refusal-affirmation gap of RLHF-aligned language models as a single pass over the vocabulary. A forward-computable score blends gap reduction with lightweight proxies for KL penalty and reward shift, allowing a "sort-sum-stop" sweep to complete in under a second and return a short suffix--two orders of magnitude fewer model calls than beam or gradient attacks. The same suffix generalises to unseen prompts and scales from 0.5 B to 70 B checkpoints, lifting one-shot attack success from baseline levels to 80-100% while preserving topical coherence. Beyond efficiency, these suffixes expose sentence-boundary reward cliffs and other alignment artefacts, offering a lightweight probe into how safety tuning reshapes internal representations.



## **50. Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation**

cs.CR

This is the full version (27 pages) of the paper 'Riddle Me This!  Stealthy Membership Inference for Retrieval-Augmented Generation' published  at CCS 2025

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2502.00306v2) [paper-pdf](http://arxiv.org/pdf/2502.00306v2)

**Authors**: Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr

**Abstract**: Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. Although the absence of weight tuning prevents leakage via model parameters, it introduces the risk of inference adversaries exploiting retrieved documents in the model's context. Existing methods for membership inference and data extraction often rely on jailbreaking or carefully crafted unnatural queries, which can be easily detected or thwarted with query rewriting techniques common in RAG systems. In this work, we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore. By crafting natural-text queries that are answerable only with the target document's presence, our approach demonstrates successful inference with just 30 queries while remaining stealthy; straightforward detectors identify adversarial prompts from existing methods up to ~76x more frequently than those generated by our attack. We observe a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations, all while costing less than $0.02 per document inference.



