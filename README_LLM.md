# Latest Large Language Model Attack Papers
**update at 2026-01-04 08:57:07**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. RAGPart & RAGMask: Retrieval-Stage Defenses Against Corpus Poisoning in Retrieval-Augmented Generation**

cs.IR

Published at AAAI 2026 Workshop on New Frontiers in Information Retrieval [Oral]

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.24268v1) [paper-pdf](https://arxiv.org/pdf/2512.24268v1)

**Authors**: Pankayaraj Pathmanathan, Michael-Andrei Panaitescu-Liess, Cho-Yu Jason Chiang, Furong Huang

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to enhance large language models (LLMs) with external knowledge, reducing hallucinations and compensating for outdated information. However, recent studies have exposed a critical vulnerability in RAG pipelines corpus poisoning where adversaries inject malicious documents into the retrieval corpus to manipulate model outputs. In this work, we propose two complementary retrieval-stage defenses: RAGPart and RAGMask. Our defenses operate directly on the retriever, making them computationally lightweight and requiring no modification to the generation model. RAGPart leverages the inherent training dynamics of dense retrievers, exploiting document partitioning to mitigate the effect of poisoned points. In contrast, RAGMask identifies suspicious tokens based on significant similarity shifts under targeted token masking. Across two benchmarks, four poisoning strategies, and four state-of-the-art retrievers, our defenses consistently reduce attack success rates while preserving utility under benign conditions. We further introduce an interpretable attack to stress-test our defenses. Our findings highlight the potential and limitations of retrieval-stage defenses, providing practical insights for robust RAG deployments.



## **2. Jailbreaking Attacks vs. Content Safety Filters: How Far Are We in the LLM Safety Arms Race?**

cs.CR

26 pages,11 tables, 7 figures

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.24044v1) [paper-pdf](https://arxiv.org/pdf/2512.24044v1)

**Authors**: Yuan Xin, Dingfan Chen, Linyi Yang, Michael Backes, Xiao Zhang

**Abstract**: As large language models (LLMs) are increasingly deployed, ensuring their safe use is paramount. Jailbreaking, adversarial prompts that bypass model alignment to trigger harmful outputs, present significant risks, with existing studies reporting high success rates in evading common LLMs. However, previous evaluations have focused solely on the models, neglecting the full deployment pipeline, which typically incorporates additional safety mechanisms like content moderation filters. To address this gap, we present the first systematic evaluation of jailbreak attacks targeting LLM safety alignment, assessing their success across the full inference pipeline, including both input and output filtering stages. Our findings yield two key insights: first, nearly all evaluated jailbreak techniques can be detected by at least one safety filter, suggesting that prior assessments may have overestimated the practical success of these attacks; second, while safety filters are effective in detection, there remains room to better balance recall and precision to further optimize protection and user experience. We highlight critical gaps and call for further refinement of detection accuracy and usability in LLM safety systems.



## **3. RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress**

cs.CR

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.23995v1) [paper-pdf](https://arxiv.org/pdf/2512.23995v1)

**Authors**: Ruixuan Huang, Qingyue Wang, Hantao Huang, Yudong Gao, Dong Chen, Shuai Wang, Wei Wang

**Abstract**: Mixture-of-Experts architectures have become the standard for scaling large language models due to their superior parameter efficiency. To accommodate the growing number of experts in practice, modern inference systems commonly adopt expert parallelism to distribute experts across devices. However, the absence of explicit load balancing constraints during inference allows adversarial inputs to trigger severe routing concentration. We demonstrate that out-of-distribution prompts can manipulate the routing strategy such that all tokens are consistently routed to the same set of top-$k$ experts, which creates computational bottlenecks on certain devices while forcing others to idle. This converts an efficiency mechanism into a denial-of-service attack vector, leading to violations of service-level agreements for time to first token. We propose RepetitionCurse, a low-cost black-box strategy to exploit this vulnerability. By identifying a universal flaw in MoE router behavior, RepetitionCurse constructs adversarial prompts using simple repetitive token patterns in a model-agnostic manner. On widely deployed MoE models like Mixtral-8x7B, our method increases end-to-end inference latency by 3.063x, degrading service availability significantly.



## **4. T2VAttack: Adversarial Attack on Text-to-Video Diffusion Models**

cs.CV

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.23953v1) [paper-pdf](https://arxiv.org/pdf/2512.23953v1)

**Authors**: Changzhen Li, Yuecong Min, Jie Zhang, Zheng Yuan, Shiguang Shan, Xilin Chen

**Abstract**: The rapid evolution of Text-to-Video (T2V) diffusion models has driven remarkable advancements in generating high-quality, temporally coherent videos from natural language descriptions. Despite these achievements, their vulnerability to adversarial attacks remains largely unexplored. In this paper, we introduce T2VAttack, a comprehensive study of adversarial attacks on T2V diffusion models from both semantic and temporal perspectives. Considering the inherently dynamic nature of video data, we propose two distinct attack objectives: a semantic objective to evaluate video-text alignment and a temporal objective to assess the temporal dynamics. To achieve an effective and efficient attack process, we propose two adversarial attack methods: (i) T2VAttack-S, which identifies semantically or temporally critical words in prompts and replaces them with synonyms via greedy search, and (ii) T2VAttack-I, which iteratively inserts optimized words with minimal perturbation to the prompt. By combining these objectives and strategies, we conduct a comprehensive evaluation on the adversarial robustness of several state-of-the-art T2V models, including ModelScope, CogVideoX, Open-Sora, and HunyuanVideo. Our experiments reveal that even minor prompt modifications, such as the substitution or insertion of a single word, can cause substantial degradation in semantic fidelity and temporal dynamics, highlighting critical vulnerabilities in current T2V diffusion models.



## **5. Breaking Audio Large Language Models by Attacking Only the Encoder: A Universal Targeted Latent-Space Audio Attack**

cs.SD

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23881v1) [paper-pdf](https://arxiv.org/pdf/2512.23881v1)

**Authors**: Roee Ziv, Raz Lapid, Moshe Sipper

**Abstract**: Audio-language models combine audio encoders with large language models to enable multimodal reasoning, but they also introduce new security vulnerabilities. We propose a universal targeted latent space attack, an encoder-level adversarial attack that manipulates audio latent representations to induce attacker-specified outputs in downstream language generation. Unlike prior waveform-level or input-specific attacks, our approach learns a universal perturbation that generalizes across inputs and speakers and does not require access to the language model. Experiments on Qwen2-Audio-7B-Instruct demonstrate consistently high attack success rates with minimal perceptual distortion, revealing a critical and previously underexplored attack surface at the encoder level of multimodal systems.



## **6. Multilingual Hidden Prompt Injection Attacks on LLM-Based Academic Reviewing**

cs.CL

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23684v1) [paper-pdf](https://arxiv.org/pdf/2512.23684v1)

**Authors**: Panagiotis Theocharopoulos, Ajinkya Kulkarni, Mathew Magimai. -Doss

**Abstract**: Large language models (LLMs) are increasingly considered for use in high-impact workflows, including academic peer review. However, LLMs are vulnerable to document-level hidden prompt injection attacks. In this work, we construct a dataset of approximately 500 real academic papers accepted to ICML and evaluate the effect of embedding hidden adversarial prompts within these documents. Each paper is injected with semantically equivalent instructions in four different languages and reviewed using an LLM. We find that prompt injection induces substantial changes in review scores and accept/reject decisions for English, Japanese, and Chinese injections, while Arabic injections produce little to no effect. These results highlight the susceptibility of LLM-based reviewing systems to document-level prompt injection and reveal notable differences in vulnerability across languages.



## **7. Toward Trustworthy Agentic AI: A Multimodal Framework for Preventing Prompt Injection Attacks**

cs.CR

It is accepted in a conference paper, ICCA 2025 in Bahrain on 21 to 23 December

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23557v1) [paper-pdf](https://arxiv.org/pdf/2512.23557v1)

**Authors**: Toqeer Ali Syed, Mishal Ateeq Almutairi, Mahmoud Abdel Moaty

**Abstract**: Powerful autonomous systems, which reason, plan, and converse using and between numerous tools and agents, are made possible by Large Language Models (LLMs), Vision-Language Models (VLMs), and new agentic AI systems, like LangChain and GraphChain. Nevertheless, this agentic environment increases the probability of the occurrence of multimodal prompt injection (PI) attacks, in which concealed or malicious instructions carried in text, pictures, metadata, or agent-to-agent messages may spread throughout the graph and lead to unintended behavior, a breach of policy, or corruption of state. In order to mitigate these risks, this paper suggests a Cross-Agent Multimodal Provenanc- Aware Defense Framework whereby all the prompts, either user-generated or produced by upstream agents, are sanitized and all the outputs generated by an LLM are verified independently before being sent to downstream nodes. This framework contains a Text sanitizer agent, visual sanitizer agent, and output validator agent all coordinated by a provenance ledger, which keeps metadata of modality, source, and trust level throughout the entire agent network. This architecture makes sure that agent-to-agent communication abides by clear trust frames such such that injected instructions are not propagated down LangChain or GraphChain-style-workflows. The experimental assessments show that multimodal injection detection accuracy is significantly enhanced, and the cross-agent trust leakage is minimized, as well as, agentic execution pathways become stable. The framework, which expands the concept of provenance tracking and validation to the multi-agent orchestration, enhances the establishment of secure, understandable and reliable agentic AI systems.



## **8. Agentic AI for Autonomous Defense in Software Supply Chain Security: Beyond Provenance to Vulnerability Mitigation**

cs.CR

Conference paper, accept in ACCA IEEE Bahrain

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23480v1) [paper-pdf](https://arxiv.org/pdf/2512.23480v1)

**Authors**: Toqeer Ali Syed, Mohammad Riyaz Belgaum, Salman Jan, Asadullah Abdullah Khan, Saad Said Alqahtani

**Abstract**: The software supply chain attacks are becoming more and more focused on trusted development and delivery procedures, so the conventional post-build integrity mechanisms cannot be used anymore. The available frameworks like SLSA, SBOM and in toto are majorly used to offer provenance and traceability but do not have the capabilities of actively identifying and removing vulnerabilities in software production. The current paper includes an example of agentic artificial intelligence (AI) based on autonomous software supply chain security that combines large language model (LLM)-based reasoning, reinforcement learning (RL), and multi-agent coordination. The suggested system utilizes specialized security agents coordinated with the help of LangChain and LangGraph, communicates with actual CI/CD environments with the Model Context Protocol (MCP), and documents all the observations and actions in a blockchain security ledger to ensure integrity and auditing. Reinforcement learning can be used to achieve adaptive mitigation strategies that consider the balance between security effectiveness and the operational overhead, and LLMs can be used to achieve semantic vulnerability analysis, as well as explainable decisions. This framework is tested based on simulated pipelines, as well as, actual world CI/CD integrations on GitHub Actions and Jenkins, including injection attacks, insecure deserialization, access control violations, and configuration errors. Experimental outcomes indicate better detection accuracy, shorter mitigation latency and reasonable build-time overhead than rule-based, provenance only and RL only baselines. These results show that agentic AI can facilitate the transition to self defending, proactive software supply chains rather than reactive verification ones.



## **9. Prompt-Induced Over-Generation as Denial-of-Service: A Black-Box Attack-Side Benchmark**

cs.CR

12 pages, 2 figures

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23779v1) [paper-pdf](https://arxiv.org/pdf/2512.23779v1)

**Authors**: Manu, Yi Guo, Jo Plested, Tim Lynar, Kanchana Thilakarathna, Nirhoshan Sivaroopan, Jack Yang, Wangli Yang

**Abstract**: Large language models (LLMs) can be driven into over-generation, emitting thousands of tokens before producing an end-of-sequence (EOS) token. This degrades answer quality, inflates latency and cost, and can be weaponized as a denial-of-service (DoS) attack. Recent work has begun to study DoS-style prompt attacks, but typically focuses on a single attack algorithm or assumes white-box access, without an attack-side benchmark that compares prompt-based attackers in a black-box, query-only regime with a known tokenizer. We introduce such a benchmark and study two prompt-only attackers. The first is Evolutionary Over-Generation Prompt Search (EOGen), which searches the token space for prefixes that suppress EOS and induce long continuations. The second is a goal-conditioned reinforcement learning attacker (RL-GOAL) that trains a network to generate prefixes conditioned on a target length. To characterize behavior, we introduce Over-Generation Factor (OGF), the ratio of produced tokens to a model's context window, along with stall and latency summaries. Our evolutionary attacker achieves mean OGF = 1.38 +/- 1.15 and Success@OGF >= 2 of 24.5 percent on Phi-3. RL-GOAL is stronger: across victims it achieves higher mean OGF (up to 2.81 +/- 1.38).



## **10. EquaCode: A Multi-Strategy Jailbreak Approach for Large Language Models via Equation Solving and Code Completion**

cs.CR

This is a preprint. A revised version will appear in the Proceedings of AAAI 2026

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23173v1) [paper-pdf](https://arxiv.org/pdf/2512.23173v1)

**Authors**: Zhen Liang, Hai Huang, Zhengkui Chen

**Abstract**: Large language models (LLMs), such as ChatGPT, have achieved remarkable success across a wide range of fields. However, their trustworthiness remains a significant concern, as they are still susceptible to jailbreak attacks aimed at eliciting inappropriate or harmful responses. However, existing jailbreak attacks mainly operate at the natural language level and rely on a single attack strategy, limiting their effectiveness in comprehensively assessing LLM robustness. In this paper, we propose Equacode, a novel multi-strategy jailbreak approach for large language models via equation-solving and code completion. This approach transforms malicious intent into a mathematical problem and then requires the LLM to solve it using code, leveraging the complexity of cross-domain tasks to divert the model's focus toward task completion rather than safety constraints. Experimental results show that Equacode achieves an average success rate of 91.19% on the GPT series and 98.65% across 3 state-of-the-art LLMs, all with only a single query. Further, ablation experiments demonstrate that EquaCode outperforms either the mathematical equation module or the code module alone. This suggests a strong synergistic effect, thereby demonstrating that multi-strategy approach yields results greater than the sum of its parts.



## **11. It's a TRAP! Task-Redirecting Agent Persuasion Benchmark for Web Agents**

cs.HC

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23128v1) [paper-pdf](https://arxiv.org/pdf/2512.23128v1)

**Authors**: Karolina Korgul, Yushi Yang, Arkadiusz Drohomirecki, Piotr Błaszczyk, Will Howard, Lukas Aichberger, Chris Russell, Philip H. S. Torr, Adam Mahdi, Adel Bibi

**Abstract**: Web-based agents powered by large language models are increasingly used for tasks such as email management or professional networking. Their reliance on dynamic web content, however, makes them vulnerable to prompt injection attacks: adversarial instructions hidden in interface elements that persuade the agent to divert from its original task. We introduce the Task-Redirecting Agent Persuasion Benchmark (TRAP), an evaluation for studying how persuasion techniques misguide autonomous web agents on realistic tasks. Across six frontier models, agents are susceptible to prompt injection in 25\% of tasks on average (13\% for GPT-5 to 43\% for DeepSeek-R1), with small interface or contextual changes often doubling success rates and revealing systemic, psychologically driven vulnerabilities in web-based agents. We also provide a modular social-engineering injection framework with controlled experiments on high-fidelity website clones, allowing for further benchmark expansion.



## **12. Agentic AI for Cyber Resilience: A New Security Paradigm and Its System-Theoretic Foundations**

cs.CR

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2512.22883v1) [paper-pdf](https://arxiv.org/pdf/2512.22883v1)

**Authors**: Tao Li, Quanyan Zhu

**Abstract**: Cybersecurity is being fundamentally reshaped by foundation-model-based artificial intelligence. Large language models now enable autonomous planning, tool orchestration, and strategic adaptation at scale, challenging security architectures built on static rules, perimeter defenses, and human-centered workflows. This chapter argues for a shift from prevention-centric security toward agentic cyber resilience. Rather than seeking perfect protection, resilient systems must anticipate disruption, maintain critical functions under attack, recover efficiently, and learn continuously. We situate this shift within the historical evolution of cybersecurity paradigms, culminating in an AI-augmented paradigm where autonomous agents participate directly in sensing, reasoning, action, and adaptation across cyber and cyber-physical systems. We then develop a system-level framework for designing agentic AI workflows. A general agentic architecture is introduced, and attacker and defender workflows are analyzed as coupled adaptive processes, and game-theoretic formulations are shown to provide a unifying design language for autonomy allocation, information flow, and temporal composition. Case studies in automated penetration testing, remediation, and cyber deception illustrate how equilibrium-based design enables system-level resiliency design.



## **13. Exploring the Security Threats of Retriever Backdoors in Retrieval-Augmented Code Generation**

cs.CR

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2512.21681v1) [paper-pdf](https://arxiv.org/pdf/2512.21681v1)

**Authors**: Tian Li, Bo Lin, Shangwen Wang, Yusong Tan

**Abstract**: Retrieval-Augmented Code Generation (RACG) is increasingly adopted to enhance Large Language Models for software development, yet its security implications remain dangerously underexplored. This paper conducts the first systematic exploration of a critical and stealthy threat: backdoor attacks targeting the retriever component, which represents a significant supply-chain vulnerability. It is infeasible to assess this threat realistically, as existing attack methods are either too ineffective to pose a real danger or are easily detected by state-of-the-art defense mechanisms spanning both latent-space analysis and token-level inspection, which achieve consistently high detection rates. To overcome this barrier and enable a realistic analysis, we first developed VenomRACG, a new class of potent and stealthy attack that serves as a vehicle for our investigation. Its design makes poisoned samples statistically indistinguishable from benign code, allowing the attack to consistently maintain low detectability across all evaluated defense mechanisms. Armed with this capability, our exploration reveals a severe vulnerability: by injecting vulnerable code equivalent to only 0.05% of the entire knowledge base size, an attacker can successfully manipulate the backdoored retriever to rank the vulnerable code in its top-5 results in 51.29% of cases. This translates to severe downstream harm, causing models like GPT-4o to generate vulnerable code in over 40% of targeted scenarios, while leaving the system's general performance intact. Our findings establish that retriever backdooring is not a theoretical concern but a practical threat to the software development ecosystem that current defenses are blind to, highlighting the urgent need for robust security measures.



## **14. LLM-Driven Feature-Level Adversarial Attacks on Android Malware Detectors**

cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21404v1) [paper-pdf](https://arxiv.org/pdf/2512.21404v1)

**Authors**: Tianwei Lan, Farid Naït-Abdesselam

**Abstract**: The rapid growth in both the scale and complexity of Android malware has driven the widespread adoption of machine learning (ML) techniques for scalable and accurate malware detection. Despite their effectiveness, these models remain vulnerable to adversarial attacks that introduce carefully crafted feature-level perturbations to evade detection while preserving malicious functionality. In this paper, we present LAMLAD, a novel adversarial attack framework that exploits the generative and reasoning capabilities of large language models (LLMs) to bypass ML-based Android malware classifiers. LAMLAD employs a dual-agent architecture composed of an LLM manipulator, which generates realistic and functionality-preserving feature perturbations, and an LLM analyzer, which guides the perturbation process toward successful evasion. To improve efficiency and contextual awareness, LAMLAD integrates retrieval-augmented generation (RAG) into the LLM pipeline. Focusing on Drebin-style feature representations, LAMLAD enables stealthy and high-confidence attacks against widely deployed Android malware detection systems. We evaluate LAMLAD against three representative ML-based Android malware detectors and compare its performance with two state-of-the-art adversarial attack methods. Experimental results demonstrate that LAMLAD achieves an attack success rate (ASR) of up to 97%, requiring on average only three attempts per adversarial sample, highlighting its effectiveness, efficiency, and adaptability in practical adversarial settings. Furthermore, we propose an adversarial training-based defense strategy that reduces the ASR by more than 30% on average, significantly enhancing model robustness against LAMLAD-style attacks.



## **15. SENTINEL: A Multi-Modal Early Detection Framework for Emerging Cyber Threats using Telegram**

cs.SI

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21380v1) [paper-pdf](https://arxiv.org/pdf/2512.21380v1)

**Authors**: Mohammad Hammas Saeed, Howie Huang

**Abstract**: Cyberattacks pose a serious threat to modern sociotechnical systems, often resulting in severe technical and societal consequences. Attackers commonly target systems and infrastructure through methods such as malware, ransomware, or other forms of technical exploitation. Most traditional mechanisms to counter these threats rely on post-hoc detection and mitigation strategies, responding to cyber incidents only after they occur rather than preventing them proactively. Recent trends reveal social media discussions can serve as reliable indicators for detecting such threats. Malicious actors often exploit online platforms to distribute attack tools, share attack knowledge and coordinate. Experts too, often predict ongoing attacks and discuss potential breaches in online spaces. In this work, we present SENTINEL, a framework that leverages social media signals for early detection of cyber attacks. SENTINEL aligns cybersecurity discussions to realworld cyber attacks leveraging multi modal signals, i.e., combining language modeling through large language models and coordination markers through graph neural networks. We use data from 16 public channels on Telegram related to cybersecurity and open source intelligence (OSINT) that span 365k messages. We highlight that social media discussions involve active dialogue around cyber threats and leverage SENTINEL to align the signals to real-world threats with an F1 of 0.89. Our work highlights the importance of leveraging language and network signals in predicting online threats.



## **16. Casting a SPELL: Sentence Pairing Exploration for LLM Limitation-breaking**

cs.CR

Accepted to FSE 2026

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21236v1) [paper-pdf](https://arxiv.org/pdf/2512.21236v1)

**Authors**: Yifan Huang, Xiaojun Jia, Wenbo Guo, Yuqiang Sun, Yihao Huang, Chong Wang, Yang Liu

**Abstract**: Large language models (LLMs) have revolutionized software development through AI-assisted coding tools, enabling developers with limited programming expertise to create sophisticated applications. However, this accessibility extends to malicious actors who may exploit these powerful tools to generate harmful software. Existing jailbreaking research primarily focuses on general attack scenarios against LLMs, with limited exploration of malicious code generation as a jailbreak target. To address this gap, we propose SPELL, a comprehensive testing framework specifically designed to evaluate the weakness of security alignment in malicious code generation. Our framework employs a time-division selection strategy that systematically constructs jailbreaking prompts by intelligently combining sentences from a prior knowledge dataset, balancing exploration of novel attack patterns with exploitation of successful techniques. Extensive evaluation across three advanced code models (GPT-4.1, Claude-3.5, and Qwen2.5-Coder) demonstrates SPELL's effectiveness, achieving attack success rates of 83.75%, 19.38%, and 68.12% respectively across eight malicious code categories. The generated prompts successfully produce malicious code in real-world AI development tools such as Cursor, with outputs confirmed as malicious by state-of-the-art detection systems at rates exceeding 73%. These findings reveal significant security gaps in current LLM implementations and provide valuable insights for improving AI safety alignment in code generation applications.



## **17. GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs**

cs.CR

Accepted by USENIX Security'26

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2512.21008v2) [paper-pdf](https://arxiv.org/pdf/2512.21008v2)

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Mixture-of-Experts (MoE) architectures have advanced the scaling of Large Language Models (LLMs) by activating only a sparse subset of parameters per input, enabling state-of-the-art performance with reduced computational cost. As these models are increasingly deployed in critical domains, understanding and strengthening their alignment mechanisms is essential to prevent harmful outputs. However, existing LLM safety research has focused almost exclusively on dense architectures, leaving the unique safety properties of MoEs largely unexamined. The modular, sparsely-activated design of MoEs suggests that safety mechanisms may operate differently than in dense models, raising questions about their robustness.   In this paper, we present GateBreaker, the first training-free, lightweight, and architecture-agnostic attack framework that compromises the safety alignment of modern MoE LLMs at inference time. GateBreaker operates in three stages: (i) gate-level profiling, which identifies safety experts disproportionately routed on harmful inputs, (ii) expert-level localization, which localizes the safety structure within safety experts, and (iii) targeted safety removal, which disables the identified safety structure to compromise the safety alignment. Our study shows that MoE safety concentrates within a small subset of neurons coordinated by sparse routing. Selective disabling of these neurons, approximately 3% of neurons in the targeted expert layers, significantly increases the averaged attack success rate (ASR) from 7.4% to 64.9% against the eight latest aligned MoE LLMs with limited utility degradation. These safety neurons transfer across models within the same family, raising ASR from 17.9% to 67.7% with one-shot transfer attack. Furthermore, GateBreaker generalizes to five MoE vision language models (VLMs) with 60.9% ASR on unsafe image inputs.



## **18. AegisAgent: An Autonomous Defense Agent Against Prompt Injection Attacks in LLM-HARs**

cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.20986v1) [paper-pdf](https://arxiv.org/pdf/2512.20986v1)

**Authors**: Yihan Wang, Huanqi Yang, Shantanu Pal, Weitao Xu

**Abstract**: The integration of Large Language Models (LLMs) into wearable sensing is creating a new class of mobile applications capable of nuanced human activity understanding. However, the reliability of these systems is critically undermined by their vulnerability to prompt injection attacks, where attackers deliberately input deceptive instructions into LLMs. Traditional defenses, based on static filters and rigid rules, are insufficient to address the semantic complexity of these new attacks. We argue that a paradigm shift is needed -- from passive filtering to active protection and autonomous reasoning. We introduce AegisAgent, an autonomous agent system designed to ensure the security of LLM-driven HAR systems. Instead of merely blocking threats, AegisAgent functions as a cognitive guardian. It autonomously perceives potential semantic inconsistencies, reasons about the user's true intent by consulting a dynamic memory of past interactions, and acts by generating and executing a multi-step verification and repair plan. We implement AegisAgent as a lightweight, full-stack prototype and conduct a systematic evaluation on 15 common attacks with five state-of-the-art LLM-based HAR systems on three public datasets. Results show it reduces attack success rate by 30\% on average while incurring only 78.6 ms of latency overhead on a GPU workstation. Our work makes the first step towards building secure and trustworthy LLM-driven HAR systems.



## **19. The Imitation Game: Using Large Language Models as Chatbots to Combat Chat-Based Cybercrimes**

cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21371v1) [paper-pdf](https://arxiv.org/pdf/2512.21371v1)

**Authors**: Yifan Yao, Baojuan Wang, Jinhao Duan, Kaidi Xu, ChuanKai Guo, Zhibo Eric Sun, Yue Zhang

**Abstract**: Chat-based cybercrime has emerged as a pervasive threat, with attackers leveraging real-time messaging platforms to conduct scams that rely on trust-building, deception, and psychological manipulation. Traditional defense mechanisms, which operate on static rules or shallow content filters, struggle to identify these conversational threats, especially when attackers use multimedia obfuscation and context-aware dialogue.   In this work, we ask a provocative question inspired by the classic Imitation Game: Can machines convincingly pose as human victims to turn deception against cybercriminals? We present LURE (LLM-based User Response Engagement), the first system to deploy Large Language Models (LLMs) as active agents, not as passive classifiers, embedded within adversarial chat environments.   LURE combines automated discovery, adversarial interaction, and OCR-based analysis of image-embedded payment data. Applied to the setting of illicit video chat scams on Telegram, our system engaged 53 actors across 98 groups. In over 56 percent of interactions, the LLM maintained multi-round conversations without being noticed as a bot, effectively "winning" the imitation game. Our findings reveal key behavioral patterns in scam operations, such as payment flows, upselling strategies, and platform migration tactics.



## **20. ChatGPT: Excellent Paper! Accept It. Editor: Imposter Found! Review Rejected**

cs.CR

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2512.20405v2) [paper-pdf](https://arxiv.org/pdf/2512.20405v2)

**Authors**: Kanchon Gharami, Sanjiv Kumar Sarkar, Yongxin Liu, Shafika Showkat Moni

**Abstract**: Large Language Models (LLMs) like ChatGPT are now widely used in writing and reviewing scientific papers. While this trend accelerates publication growth and reduces human workload, it also introduces serious risks. Papers written or reviewed by LLMs may lack real novelty, contain fabricated or biased results, or mislead downstream research that others depend on. Such issues can damage reputations, waste resources, and even endanger lives when flawed studies influence medical or safety-critical systems. This research explores both the offensive and defensive sides of this growing threat. On the attack side, we demonstrate how an author can inject hidden prompts inside a PDF that secretly guide or "jailbreak" LLM reviewers into giving overly positive feedback and biased acceptance. On the defense side, we propose an "inject-and-detect" strategy for editors, where invisible trigger prompts are embedded into papers; if a review repeats or reacts to these triggers, it reveals that the review was generated by an LLM, not a human. This method turns prompt injections from vulnerability into a verification tool. We outline our design, expected model behaviors, and ethical safeguards for deployment. The goal is to expose how fragile today's peer-review process becomes under LLM influence and how editorial awareness can help restore trust in scientific evaluation.



## **21. Optimistic TEE-Rollups: A Hybrid Architecture for Scalable and Verifiable Generative AI Inference on Blockchain**

cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20176v1) [paper-pdf](https://arxiv.org/pdf/2512.20176v1)

**Authors**: Aaron Chan, Alex Ding, Frank Chen, Alan Wu, Bruce Zhang, Arther Tian

**Abstract**: The rapid integration of Large Language Models (LLMs) into decentralized physical infrastructure networks (DePIN) is currently bottlenecked by the Verifiability Trilemma, which posits that a decentralized inference system cannot simultaneously achieve high computational integrity, low latency, and low cost. Existing cryptographic solutions, such as Zero-Knowledge Machine Learning (ZKML), suffer from superlinear proving overheads (O(k NlogN)) that render them infeasible for billionparameter models. Conversely, optimistic approaches (opML) impose prohibitive dispute windows, preventing real-time interactivity, while recent "Proof of Quality" (PoQ) paradigms sacrifice cryptographic integrity for subjective semantic evaluation, leaving networks vulnerable to model downgrade attacks and reward hacking. In this paper, we introduce Optimistic TEE-Rollups (OTR), a hybrid verification protocol that harmonizes these constraints. OTR leverages NVIDIA H100 Confidential Computing Trusted Execution Environments (TEEs) to provide sub-second Provisional Finality, underpinned by an optimistic fraud-proof mechanism and stochastic Zero-Knowledge spot-checks to mitigate hardware side-channel risks. We formally define Proof of Efficient Attribution (PoEA), a consensus mechanism that cryptographically binds execution traces to hardware attestations, thereby guaranteeing model authenticity. Extensive simulations demonstrate that OTR achieves 99% of the throughput of centralized baselines with a marginal cost overhead of $0.07 per query, maintaining Byzantine fault tolerance against rational adversaries even in the presence of transient hardware vulnerabilities.



## **22. Odysseus: Jailbreaking Commercial Multimodal LLM-integrated Systems via Dual Steganography**

cs.CR

This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20168v1) [paper-pdf](https://arxiv.org/pdf/2512.20168v1)

**Authors**: Songze Li, Jiameng Cheng, Yiming Li, Xiaojun Jia, Dacheng Tao

**Abstract**: By integrating language understanding with perceptual modalities such as images, multimodal large language models (MLLMs) constitute a critical substrate for modern AI systems, particularly intelligent agents operating in open and interactive environments. However, their increasing accessibility also raises heightened risks of misuse, such as generating harmful or unsafe content. To mitigate these risks, alignment techniques are commonly applied to align model behavior with human values. Despite these efforts, recent studies have shown that jailbreak attacks can circumvent alignment and elicit unsafe outputs. Currently, most existing jailbreak methods are tailored for open-source models and exhibit limited effectiveness against commercial MLLM-integrated systems, which often employ additional filters. These filters can detect and prevent malicious input and output content, significantly reducing jailbreak threats. In this paper, we reveal that the success of these safety filters heavily relies on a critical assumption that malicious content must be explicitly visible in either the input or the output. This assumption, while often valid for traditional LLM-integrated systems, breaks down in MLLM-integrated systems, where attackers can leverage multiple modalities to conceal adversarial intent, leading to a false sense of security in existing MLLM-integrated systems. To challenge this assumption, we propose Odysseus, a novel jailbreak paradigm that introduces dual steganography to covertly embed malicious queries and responses into benign-looking images. Extensive experiments on benchmark datasets demonstrate that our Odysseus successfully jailbreaks several pioneering and realistic MLLM-integrated systems, achieving up to 99% attack success rate. It exposes a fundamental blind spot in existing defenses, and calls for rethinking cross-modal security in MLLM-integrated systems.



## **23. AI Security Beyond Core Domains: Resume Screening as a Case Study of Adversarial Vulnerabilities in Specialized LLM Applications**

cs.CL

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20164v1) [paper-pdf](https://arxiv.org/pdf/2512.20164v1)

**Authors**: Honglin Mu, Jinghao Liu, Kaiyang Wan, Rui Xing, Xiuying Chen, Timothy Baldwin, Wanxiang Che

**Abstract**: Large Language Models (LLMs) excel at text comprehension and generation, making them ideal for automated tasks like code review and content moderation. However, our research identifies a vulnerability: LLMs can be manipulated by "adversarial instructions" hidden in input data, such as resumes or code, causing them to deviate from their intended task. Notably, while defenses may exist for mature domains such as code review, they are often absent in other common applications such as resume screening and peer review. This paper introduces a benchmark to assess this vulnerability in resume screening, revealing attack success rates exceeding 80% for certain attack types. We evaluate two defense mechanisms: prompt-based defenses achieve 10.1% attack reduction with 12.5% false rejection increase, while our proposed FIDS (Foreign Instruction Detection through Separation) using LoRA adaptation achieves 15.4% attack reduction with 10.4% false rejection increase. The combined approach provides 26.3% attack reduction, demonstrating that training-time defenses outperform inference-time mitigations in both security and utility preservation.



## **24. ReGAIN: Retrieval-Grounded AI Framework for Network Traffic Analysis**

cs.LG

Accepted to ICNC 2026. This is the accepted author manuscript

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.22223v1) [paper-pdf](https://arxiv.org/pdf/2512.22223v1)

**Authors**: Shaghayegh Shajarian, Kennedy Marsh, James Benson, Sajad Khorsandroo, Mahmoud Abdelsalam

**Abstract**: Modern networks generate vast, heterogeneous traffic that must be continuously analyzed for security and performance. Traditional network traffic analysis systems, whether rule-based or machine learning-driven, often suffer from high false positives and lack interpretability, limiting analyst trust. In this paper, we present ReGAIN, a multi-stage framework that combines traffic summarization, retrieval-augmented generation (RAG), and Large Language Model (LLM) reasoning for transparent and accurate network traffic analysis. ReGAIN creates natural-language summaries from network traffic, embeds them into a multi-collection vector database, and utilizes a hierarchical retrieval pipeline to ground LLM responses with evidence citations. The pipeline features metadata-based filtering, MMR sampling, a two-stage cross-encoder reranking mechanism, and an abstention mechanism to reduce hallucinations and ensure grounded reasoning. Evaluated on ICMP ping flood and TCP SYN flood traces from the real-world traffic dataset, it demonstrates robust performance, achieving accuracy between 95.95% and 98.82% across different attack types and evaluation benchmarks. These results are validated against two complementary sources: dataset ground truth and human expert assessments. ReGAIN also outperforms rule-based, classical ML, and deep learning baselines while providing unique explainability through trustworthy, verifiable responses.



## **25. Conditional Adversarial Fragility in Financial Machine Learning under Macroeconomic Stress**

cs.LG

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19935v1) [paper-pdf](https://arxiv.org/pdf/2512.19935v1)

**Authors**: Samruddhi Baviskar

**Abstract**: Machine learning models used in financial decision systems operate in nonstationary economic environments, yet adversarial robustness is typically evaluated under static assumptions. This work introduces Conditional Adversarial Fragility, a regime dependent phenomenon in which adversarial vulnerability is systematically amplified during periods of macroeconomic stress. We propose a regime aware evaluation framework for time indexed tabular financial classification tasks that conditions robustness assessment on external indicators of economic stress. Using volatility based regime segmentation as a proxy for macroeconomic conditions, we evaluate model behavior across calm and stress periods while holding model architecture, attack methodology, and evaluation protocols constant. Baseline predictive performance remains comparable across regimes, indicating that economic stress alone does not induce inherent performance degradation. Under adversarial perturbations, however, models operating during stress regimes exhibit substantially greater degradation across predictive accuracy, operational decision thresholds, and risk sensitive outcomes. We further demonstrate that this amplification propagates to increased false negative rates, elevating the risk of missed high risk cases during adverse conditions. To complement numerical robustness metrics, we introduce an interpretive governance layer based on semantic auditing of model explanations using large language models. Together, these results demonstrate that adversarial robustness in financial machine learning is a regime dependent property and motivate stress aware approaches to model risk assessment in high stakes financial deployments.



## **26. Causal-Guided Detoxify Backdoor Attack of Open-Weight LoRA Models**

cs.CR

NDSS 2026

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19297v1) [paper-pdf](https://arxiv.org/pdf/2512.19297v1)

**Authors**: Linzhi Chen, Yang Sun, Hongru Wei, Yuqi Chen

**Abstract**: Low-Rank Adaptation (LoRA) has emerged as an efficient method for fine-tuning large language models (LLMs) and is widely adopted within the open-source community. However, the decentralized dissemination of LoRA adapters through platforms such as Hugging Face introduces novel security vulnerabilities: malicious adapters can be easily distributed and evade conventional oversight mechanisms. Despite these risks, backdoor attacks targeting LoRA-based fine-tuning remain relatively underexplored. Existing backdoor attack strategies are ill-suited to this setting, as they often rely on inaccessible training data, fail to account for the structural properties unique to LoRA, or suffer from high false trigger rates (FTR), thereby compromising their stealth. To address these challenges, we propose Causal-Guided Detoxify Backdoor Attack (CBA), a novel backdoor attack framework specifically designed for open-weight LoRA models. CBA operates without access to original training data and achieves high stealth through two key innovations: (1) a coverage-guided data generation pipeline that synthesizes task-aligned inputs via behavioral exploration, and (2) a causal-guided detoxification strategy that merges poisoned and clean adapters by preserving task-critical neurons. Unlike prior approaches, CBA enables post-training control over attack intensity through causal influence-based weight allocation, eliminating the need for repeated retraining. Evaluated across six LoRA models, CBA achieves high attack success rates while reducing FTR by 50-70\% compared to baseline methods. Furthermore, it demonstrates enhanced resistance to state-of-the-art backdoor defenses, highlighting its stealth and robustness.



## **27. DREAM: Dynamic Red-teaming across Environments for AI Models**

cs.CR

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19016v1) [paper-pdf](https://arxiv.org/pdf/2512.19016v1)

**Authors**: Liming Lu, Xiang Gu, Junyu Huang, Jiawei Du, Yunhuai Liu, Yongbin Zhou, Shuchao Pang

**Abstract**: Large Language Models (LLMs) are increasingly used in agentic systems, where their interactions with diverse tools and environments create complex, multi-stage safety challenges. However, existing benchmarks mostly rely on static, single-turn assessments that miss vulnerabilities from adaptive, long-chain attacks. To fill this gap, we introduce DREAM, a framework for systematic evaluation of LLM agents against dynamic, multi-stage attacks. At its core, DREAM uses a Cross-Environment Adversarial Knowledge Graph (CE-AKG) to maintain stateful, cross-domain understanding of vulnerabilities. This graph guides a Contextualized Guided Policy Search (C-GPS) algorithm that dynamically constructs attack chains from a knowledge base of 1,986 atomic actions across 349 distinct digital environments. Our evaluation of 12 leading LLM agents reveals a critical vulnerability: these attack chains succeed in over 70% of cases for most models, showing the power of stateful, cross-environment exploits. Through analysis of these failures, we identify two key weaknesses in current agents: contextual fragility, where safety behaviors fail to transfer across environments, and an inability to track long-term malicious intent. Our findings also show that traditional safety measures, such as initial defense prompts, are largely ineffective against attacks that build context over multiple interactions. To advance agent safety research, we release DREAM as a tool for evaluating vulnerabilities and developing more robust defenses.



## **28. Efficient Jailbreak Mitigation Using Semantic Linear Classification in a Multi-Staged Pipeline**

cs.CR

Under Review

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19011v1) [paper-pdf](https://arxiv.org/pdf/2512.19011v1)

**Authors**: Akshaj Prashanth Rao, Advait Singh, Saumya Kumaar Saksena, Dhruv Kumar

**Abstract**: Prompt injection and jailbreaking attacks pose persistent security challenges to large language model (LLM)-based systems. We present an efficient and systematically evaluated defense architecture that mitigates these threats through a lightweight, multi-stage pipeline. Its core component is a semantic filter based on text normalization, TF-IDF representations, and a Linear SVM classifier. Despite its simplicity, this module achieves 93.4% accuracy and 96.5% specificity on held-out data, substantially reducing attack throughput while incurring negligible computational overhead.   Building on this efficient foundation, the full pipeline integrates complementary detection and mitigation mechanisms that operate at successive stages, providing strong robustness with minimal latency. In comparative experiments, our SVM-based configuration improves overall accuracy from 35.1% to 93.4% while reducing average time to completion from approximately 450s to 47s, yielding over 10 times lower latency than ShieldGemma. These results demonstrate that the proposed design simultaneously advances defensive precision and efficiency, addressing a core limitation of current model-based moderators.   Evaluation across a curated corpus of over 30,000 labeled prompts, including benign, jailbreak, and application-layer injections, confirms that staged, resource-efficient defenses can robustly secure modern LLM-driven applications.



## **29. Automated Red-Teaming Framework for Large Language Model Security Assessment: A Comprehensive Attack Generation and Detection System**

cs.CR

18 pages

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.20677v1) [paper-pdf](https://arxiv.org/pdf/2512.20677v1)

**Authors**: Zhang Wei, Peilu Hu, Shengning Lang, Hao Yan, Li Mei, Yichao Zhang, Chen Yang, Junfeng Hao, Zhimo Han

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes domains, ensuring their security and alignment has become a critical challenge. Existing red-teaming practices depend heavily on manual testing, which limits scalability and fails to comprehensively cover the vast space of potential adversarial behaviors. This paper introduces an automated red-teaming framework that systematically generates, executes, and evaluates adversarial prompts to uncover security vulnerabilities in LLMs. Our framework integrates meta-prompting-based attack synthesis, multi-modal vulnerability detection, and standardized evaluation protocols spanning six major threat categories -- reward hacking, deceptive alignment, data exfiltration, sandbagging, inappropriate tool use, and chain-of-thought manipulation. Experiments on the GPT-OSS-20B model reveal 47 distinct vulnerabilities, including 21 high-severity and 12 novel attack patterns, achieving a $3.9\times$ improvement in vulnerability discovery rate over manual expert testing while maintaining 89\% detection accuracy. These results demonstrate the framework's effectiveness in enabling scalable, systematic, and reproducible AI safety evaluations. By providing actionable insights for improving alignment robustness, this work advances the state of automated LLM red-teaming and contributes to the broader goal of building secure and trustworthy AI systems.



## **30. VizDefender: Unmasking Visualization Tampering through Proactive Localization and Intent Inference**

cs.CV

IEEE Transactions on Visualization and Computer Graphics (IEEE PacificVis'26 TVCG Track)

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18853v1) [paper-pdf](https://arxiv.org/pdf/2512.18853v1)

**Authors**: Sicheng Song, Yanjie Zhang, Zixin Chen, Huamin Qu, Changbo Wang, Chenhui Li

**Abstract**: The integrity of data visualizations is increasingly threatened by image editing techniques that enable subtle yet deceptive tampering. Through a formative study, we define this challenge and categorize tampering techniques into two primary types: data manipulation and visual encoding manipulation. To address this, we present VizDefender, a framework for tampering detection and analysis. The framework integrates two core components: 1) a semi-fragile watermark module that protects the visualization by embedding a location map to images, which allows for the precise localization of tampered regions while preserving visual quality, and 2) an intent analysis module that leverages Multimodal Large Language Models (MLLMs) to interpret manipulation, inferring the attacker's intent and misleading effects. Extensive evaluations and user studies demonstrate the effectiveness of our methods.



## **31. MEEA: Mere Exposure Effect-Driven Confrontational Optimization for LLM Jailbreaking**

cs.AI

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18755v1) [paper-pdf](https://arxiv.org/pdf/2512.18755v1)

**Authors**: Jianyi Zhang, Shizhao Liu, Ziyin Zhou, Zhen Li

**Abstract**: The rapid advancement of large language models (LLMs) has intensified concerns about the robustness of their safety alignment. While existing jailbreak studies explore both single-turn and multi-turn strategies, most implicitly assume a static safety boundary and fail to account for how contextual interactions dynamically influence model behavior, leading to limited stability and generalization. Motivated by this gap, we propose MEEA (Mere Exposure Effect Attack), a psychology-inspired, fully automated black-box framework for evaluating multi-turn safety robustness, grounded in the mere exposure effect. MEEA leverages repeated low-toxicity semantic exposure to induce a gradual shift in a model's effective safety threshold, enabling progressive erosion of alignment constraints over sustained interactions. Concretely, MEEA constructs semantically progressive prompt chains and optimizes them using a simulated annealing strategy guided by semantic similarity, toxicity, and jailbreak effectiveness. Extensive experiments on both closed-source and open-source models, including GPT-4, Claude-3.5, and DeepSeek-R1, demonstrate that MEEA consistently achieves higher attack success rates than seven representative baselines, with an average Attack Success Rate (ASR) improvement exceeding 20%. Ablation studies further validate the necessity of both annealing-based optimization and contextual exposure mechanisms. Beyond improved attack effectiveness, our findings indicate that LLM safety behavior is inherently dynamic and history-dependent, challenging the common assumption of static alignment boundaries and highlighting the need for interaction-aware safety evaluation and defense mechanisms. Our code is available at: https://github.com/Carney-lsz/MEEA



## **32. Explainable and Fine-Grained Safeguarding of LLM Multi-Agent Systems via Bi-Level Graph Anomaly Detection**

cs.CR

14 pages, 3 tables, 5 figures

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18733v1) [paper-pdf](https://arxiv.org/pdf/2512.18733v1)

**Authors**: Junjun Pan, Yixin Liu, Rui Miao, Kaize Ding, Yu Zheng, Quoc Viet Hung Nguyen, Alan Wee-Chung Liew, Shirui Pan

**Abstract**: Large language model (LLM)-based multi-agent systems (MAS) have shown strong capabilities in solving complex tasks. As MAS become increasingly autonomous in various safety-critical tasks, detecting malicious agents has become a critical security concern. Although existing graph anomaly detection (GAD)-based defenses can identify anomalous agents, they mainly rely on coarse sentence-level information and overlook fine-grained lexical cues, leading to suboptimal performance. Moreover, the lack of interpretability in these methods limits their reliability and real-world applicability. To address these limitations, we propose XG-Guard, an explainable and fine-grained safeguarding framework for detecting malicious agents in MAS. To incorporate both coarse and fine-grained textual information for anomalous agent identification, we utilize a bi-level agent encoder to jointly model the sentence- and token-level representations of each agent. A theme-based anomaly detector further captures the evolving discussion focus in MAS dialogues, while a bi-level score fusion mechanism quantifies token-level contributions for explanation. Extensive experiments across diverse MAS topologies and attack scenarios demonstrate robust detection performance and strong interpretability of XG-Guard.



## **33. Breaking Minds, Breaking Systems: Jailbreaking Large Language Models via Human-like Psychological Manipulation**

cs.CR

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2512.18244v1) [paper-pdf](https://arxiv.org/pdf/2512.18244v1)

**Authors**: Zehao Liu, Xi Lin

**Abstract**: Large Language Models (LLMs) have gained considerable popularity and protected by increasingly sophisticated safety mechanisms. However, jailbreak attacks continue to pose a critical security threat by inducing models to generate policy-violating behaviors. Current paradigms focus on input-level anomalies, overlooking that the model's internal psychometric state can be systematically manipulated. To address this, we introduce Psychological Jailbreak, a new jailbreak attack paradigm that exposes a stateful psychological attack surface in LLMs, where attackers exploit the manipulation of a model's psychological state across interactions. Building on this insight, we propose Human-like Psychological Manipulation (HPM), a black-box jailbreak method that dynamically profiles a target model's latent psychological vulnerabilities and synthesizes tailored multi-turn attack strategies. By leveraging the model's optimization for anthropomorphic consistency, HPM creates a psychological pressure where social compliance overrides safety constraints. To systematically measure psychological safety, we construct an evaluation framework incorporating psychometric datasets and the Policy Corruption Score (PCS). Benchmarking against various models (e.g., GPT-4o, DeepSeek-V3, Gemini-2-Flash), HPM achieves a mean Attack Success Rate (ASR) of 88.1%, outperforming state-of-the-art attack baselines. Our experiments demonstrate robust penetration against advanced defenses, including adversarial prompt optimization (e.g., RPO) and cognitive interventions (e.g., Self-Reminder). Ultimately, PCS analysis confirms HPM induces safety breakdown to satisfy manipulated contexts. Our work advocates for a fundamental paradigm shift from static content filtering to psychological safety, prioritizing the development of psychological defense mechanisms against deep cognitive manipulation.



## **34. Adversarially Robust Detection of Harmful Online Content: A Computational Design Science Approach**

cs.LG

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.17367v3) [paper-pdf](https://arxiv.org/pdf/2512.17367v3)

**Authors**: Yidong Chai, Yi Liu, Mohammadreza Ebrahimi, Weifeng Li, Balaji Padmanabhan

**Abstract**: Social media platforms are plagued by harmful content such as hate speech, misinformation, and extremist rhetoric. Machine learning (ML) models are widely adopted to detect such content; however, they remain highly vulnerable to adversarial attacks, wherein malicious users subtly modify text to evade detection. Enhancing adversarial robustness is therefore essential, requiring detectors that can defend against diverse attacks (generalizability) while maintaining high overall accuracy. However, simultaneously achieving both optimal generalizability and accuracy is challenging. Following the computational design science paradigm, this study takes a sequential approach that first proposes a novel framework (Large Language Model-based Sample Generation and Aggregation, LLM-SGA) by identifying the key invariances of textual adversarial attacks and leveraging them to ensure that a detector instantiated within the framework has strong generalizability. Second, we instantiate our detector (Adversarially Robust Harmful Online Content Detector, ARHOCD) with three novel design components to improve detection accuracy: (1) an ensemble of multiple base detectors that exploits their complementary strengths; (2) a novel weight assignment method that dynamically adjusts weights based on each sample's predictability and each base detector's capability, with weights initialized using domain knowledge and updated via Bayesian inference; and (3) a novel adversarial training strategy that iteratively optimizes both the base detectors and the weight assignor. We addressed several limitations of existing adversarial robustness enhancement research and empirically evaluated ARHOCD across three datasets spanning hate speech, rumor, and extremist content. Results show that ARHOCD offers strong generalizability and improves detection accuracy under adversarial conditions.



## **35. In-Context Probing for Membership Inference in Fine-Tuned Language Models**

cs.CR

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.16292v2) [paper-pdf](https://arxiv.org/pdf/2512.16292v2)

**Authors**: Zhexi Lu, Hongliang Chi, Nathalie Baracaldo, Swanand Ravindra Kadhe, Yuseok Jeon, Lei Yu

**Abstract**: Membership inference attacks (MIAs) pose a critical privacy threat to fine-tuned large language models (LLMs), especially when models are adapted to domain-specific tasks using sensitive data. While prior black-box MIA techniques rely on confidence scores or token likelihoods, these signals are often entangled with a sample's intrinsic properties - such as content difficulty or rarity - leading to poor generalization and low signal-to-noise ratios. In this paper, we propose ICP-MIA, a novel MIA framework grounded in the theory of training dynamics, particularly the phenomenon of diminishing returns during optimization. We introduce the Optimization Gap as a fundamental signal of membership: at convergence, member samples exhibit minimal remaining loss-reduction potential, while non-members retain significant potential for further optimization. To estimate this gap in a black-box setting, we propose In-Context Probing (ICP), a training-free method that simulates fine-tuning-like behavior via strategically constructed input contexts. We propose two probing strategies: reference-data-based (using semantically similar public samples) and self-perturbation (via masking or generation). Experiments on three tasks and multiple LLMs show that ICP-MIA significantly outperforms prior black-box MIAs, particularly at low false positive rates. We further analyze how reference data alignment, model type, PEFT configurations, and training schedules affect attack effectiveness. Our findings establish ICP-MIA as a practical and theoretically grounded framework for auditing privacy risks in deployed LLMs.



## **36. APT-CGLP: Advanced Persistent Threat Hunting via Contrastive Graph-Language Pre-Training**

cs.CR

Accepted by SIGKDD 2026 Research Track

**SubmitDate**: 2025-12-31    [abs](http://arxiv.org/abs/2511.20290v2) [paper-pdf](https://arxiv.org/pdf/2511.20290v2)

**Authors**: Xuebo Qiu, Mingqi Lv, Yimei Zhang, Tieming Chen, Tiantian Zhu, Qijie Song, Shouling Ji

**Abstract**: Provenance-based threat hunting identifies Advanced Persistent Threats (APTs) on endpoints by correlating attack patterns described in Cyber Threat Intelligence (CTI) with provenance graphs derived from system audit logs. A fundamental challenge in this paradigm lies in the modality gap -- the structural and semantic disconnect between provenance graphs and CTI reports. Prior work addresses this by framing threat hunting as a graph matching task: 1) extracting attack graphs from CTI reports, and 2) aligning them with provenance graphs. However, this pipeline incurs severe \textit{information loss} during graph extraction and demands intensive manual curation, undermining scalability and effectiveness.   In this paper, we present APT-CGLP, a novel cross-modal APT hunting system via Contrastive Graph-Language Pre-training, facilitating end-to-end semantic matching between provenance graphs and CTI reports without human intervention. First, empowered by the Large Language Model (LLM), APT-CGLP mitigates data scarcity by synthesizing high-fidelity provenance graph-CTI report pairs, while simultaneously distilling actionable insights from noisy web-sourced CTIs to improve their operational utility. Second, APT-CGLP incorporates a tailored multi-objective training algorithm that synergizes contrastive learning with inter-modal masked modeling, promoting cross-modal attack semantic alignment at both coarse- and fine-grained levels. Extensive experiments on four real-world APT datasets demonstrate that APT-CGLP consistently outperforms state-of-the-art threat hunting baselines in terms of accuracy and efficiency.



## **37. AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models**

cs.CL

Presented at NeurIPS 2025 Lock-LLM Workshop. Code is available at https://github.com/AAN-AutoAdv/AutoAdv

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2511.02376v3) [paper-pdf](https://arxiv.org/pdf/2511.02376v3)

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs. Yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves an attack success rate of up to 95% on Llama-3.1-8B within six turns, a 24% improvement over single-turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests and then iteratively refines them. Extensive evaluation across commercial and open-source models (Llama-3.1-8B, GPT-4o mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses.



## **38. FuncPoison: Poisoning Function Library to Hijack Multi-agent Autonomous Driving Systems**

cs.CR

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2509.24408v2) [paper-pdf](https://arxiv.org/pdf/2509.24408v2)

**Authors**: Yuzhen Long, Songze Li

**Abstract**: Autonomous driving systems increasingly rely on multi-agent architectures powered by large language models (LLMs), where specialized agents collaborate to perceive, reason, and plan. A key component of these systems is the shared function library, a collection of software tools that agents use to process sensor data and navigate complex driving environments. Despite its critical role in agent decision-making, the function library remains an under-explored vulnerability. In this paper, we introduce FuncPoison, a novel poisoning-based attack targeting the function library to manipulate the behavior of LLM-driven multi-agent autonomous systems. FuncPoison exploits two key weaknesses in how agents access the function library: (1) agents rely on text-based instructions to select tools; and (2) these tools are activated using standardized command formats that attackers can replicate. By injecting malicious tools with deceptive instructions, FuncPoison manipulates one agent s decisions--such as misinterpreting road conditions--triggering cascading errors that mislead other agents in the system. We experimentally evaluate FuncPoison on two representative multi-agent autonomous driving systems, demonstrating its ability to significantly degrade trajectory accuracy, flexibly target specific agents to induce coordinated misbehavior, and evade diverse defense mechanisms. Our results reveal that the function library, often considered a simple toolset, can serve as a critical attack surface in LLM-based autonomous driving systems, raising elevated concerns on their reliability.



## **39. Secure and Efficient Access Control for Computer-Use Agents via Context Space**

cs.CR

**SubmitDate**: 2025-12-31    [abs](http://arxiv.org/abs/2509.22256v3) [paper-pdf](https://arxiv.org/pdf/2509.22256v3)

**Authors**: Haochen Gong, Chenxiao Li, Rui Chang, Wenbo Shen

**Abstract**: Large language model (LLM)-based computer-use agents represent a convergence of AI and OS capabilities, enabling natural language to control system- and application-level functions. However, due to LLMs' inherent uncertainty issues, granting agents control over computers poses significant security risks. When agent actions deviate from user intentions, they can cause irreversible consequences. Existing mitigation approaches, such as user confirmation and LLM-based dynamic action validation, still suffer from limitations in usability, security, and performance. To address these challenges, we propose CSAgent, a system-level, static policy-based access control framework for computer-use agents. To bridge the gap between static policy and dynamic context and user intent, CSAgent introduces intent- and context-aware policies, and provides an automated toolchain to assist developers in constructing and refining them. CSAgent enforces these policies through an optimized OS service, ensuring that agent actions can only be executed under specific user intents and contexts. CSAgent supports protecting agents that control computers through diverse interfaces, including API, CLI, and GUI. We implement and evaluate CSAgent, which successfully defends against more than 99.56% of attacks while introducing only 1.99% performance overhead.



## **40. Involuntary Jailbreak: On Self-Prompting Attacks**

cs.CR

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2508.13246v3) [paper-pdf](https://arxiv.org/pdf/2508.13246v3)

**Authors**: Yangyang Guo, Yangyan Li, Mohan Kankanhalli

**Abstract**: In this study, we disclose a worrying new vulnerability in Large Language Models (LLMs), which we term \textbf{involuntary jailbreak}. Unlike existing jailbreak attacks, this weakness is distinct in that it does not involve a specific attack objective, such as generating instructions for \textit{building a bomb}. Prior attack methods predominantly target localized components of the LLM guardrail. In contrast, involuntary jailbreaks may potentially compromise the entire guardrail structure, which our method reveals to be surprisingly fragile. We merely employ a single universal prompt to achieve this goal. In particular, we instruct LLMs to generate several questions that would typically be rejected, along with their corresponding in-depth responses (rather than a refusal). Remarkably, this simple prompt strategy consistently jailbreaks the majority of leading LLMs, including Claude Opus 4.1, Grok 4, Gemini 2.5 Pro, and GPT 4.1. We hope this problem can motivate researchers and practitioners to re-evaluate the robustness of LLM guardrails and contribute to stronger safety alignment in future.



## **41. Universal Jailbreak Suffixes Are Strong Attention Hijackers**

cs.CR

Accepted at TACL 2026

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2506.12880v2) [paper-pdf](https://arxiv.org/pdf/2506.12880v2)

**Authors**: Matan Ben-Tov, Mor Geva, Mahmood Sharif

**Abstract**: We study suffix-based jailbreaks$\unicode{x2013}$a powerful family of attacks against large language models (LLMs) that optimize adversarial suffixes to circumvent safety alignment. Focusing on the widely used foundational GCG attack, we observe that suffixes vary in efficacy: some are markedly more universal$\unicode{x2013}$generalizing to many unseen harmful instructions$\unicode{x2013}$than others. We first show that a shallow, critical mechanism drives GCG's effectiveness. This mechanism builds on the information flow from the adversarial suffix to the final chat template tokens before generation. Quantifying the dominance of this mechanism during generation, we find GCG irregularly and aggressively hijacks the contextualization process. Crucially, we tie hijacking to the universality phenomenon, with more universal suffixes being stronger hijackers. Subsequently, we show that these insights have practical implications: GCG's universality can be efficiently enhanced (up to $\times$5 in some cases) at no additional computational cost, and can also be surgically mitigated, at least halving the attack's success with minimal utility loss. We release our code and data at http://github.com/matanbt/interp-jailbreak.



## **42. Improving Large Language Model Safety with Contrastive Representation Learning**

cs.CL

EMNLP 2025 Main

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2506.11938v2) [paper-pdf](https://arxiv.org/pdf/2506.11938v2)

**Authors**: Samuel Simko, Mrinmaya Sachan, Bernhard Schölkopf, Zhijing Jin

**Abstract**: Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at https://github.com/samuelsimko/crl-llm-defense



## **43. SoK: Are Watermarks in LLMs Ready for Deployment?**

cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2506.05594v3) [paper-pdf](https://arxiv.org/pdf/2506.05594v3)

**Authors**: Kieu Dang, Phung Lai, NhatHai Phan, Yelong Shen, Ruoming Jin, Abdallah Khreishah, My T. Thai

**Abstract**: Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs.   To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment.



## **44. Efficient and Stealthy Jailbreak Attacks via Adversarial Prompt Distillation from LLMs to SLMs**

cs.CL

19 pages, 7 figures

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2506.17231v2) [paper-pdf](https://arxiv.org/pdf/2506.17231v2)

**Authors**: Xiang Li, Chong Zhang, Jia Wang, Fangyu Wu, Yushi Li, Xiaobo Jin

**Abstract**: As the scale and complexity of jailbreaking attacks on large language models (LLMs) continue to escalate, their efficiency and practical applicability are constrained, posing a profound challenge to LLM security. Jailbreaking techniques have advanced from manual prompt engineering to automated methodologies. Recent advances have automated jailbreaking approaches that harness LLMs to generate jailbreak instructions and adversarial examples, delivering encouraging results. Nevertheless, these methods universally include an LLM generation phase, which, due to the complexities of deploying and reasoning with LLMs, impedes effective implementation and broader adoption. To mitigate this issue, we introduce \textbf{Adversarial Prompt Distillation}, an innovative framework that integrates masked language modeling, reinforcement learning, and dynamic temperature control to distill LLM jailbreaking prowess into smaller language models (SLMs). This methodology enables efficient, robust jailbreak attacks while maintaining high success rates and accommodating a broader range of application contexts. Empirical evaluations affirm the approach's superiority in attack efficacy, resource optimization, and cross-model versatility. Our research underscores the practicality of transferring jailbreak capabilities to SLMs, reveals inherent vulnerabilities in LLMs, and provides novel insights to advance LLM security investigations. Our code is available at: https://github.com/lxgem/Efficient_and_Stealthy_Jailbreak_Attacks_via_Adversarial_Prompt.



## **45. AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models**

cs.CR

We encountered issues with the paper being hosted under my personal account, so we republished it under a different account associated with a university email, which makes updates and management easier. As a result, this version is a duplicate of arXiv:2511.02376

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2507.01020v2) [paper-pdf](https://arxiv.org/pdf/2507.01020v2)

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities to jailbreaking attacks: carefully crafted malicious inputs intended to circumvent safety guardrails and elicit harmful responses. As such, we present AutoAdv, a novel framework that automates adversarial prompt generation to systematically evaluate and expose vulnerabilities in LLM safety mechanisms. Our approach leverages a parametric attacker LLM to produce semantically disguised malicious prompts through strategic rewriting techniques, specialized system prompts, and optimized hyperparameter configurations. The primary contribution of our work is a dynamic, multi-turn attack methodology that analyzes failed jailbreak attempts and iteratively generates refined follow-up prompts, leveraging techniques such as roleplaying, misdirection, and contextual manipulation. We quantitatively evaluate attack success rate (ASR) using the StrongREJECT (arXiv:2402.10260 [cs.CL]) framework across sequential interaction turns. Through extensive empirical evaluation of state-of-the-art models--including ChatGPT, Llama, and DeepSeek--we reveal significant vulnerabilities, with our automated attacks achieving jailbreak success rates of up to 86% for harmful content generation. Our findings reveal that current safety mechanisms remain susceptible to sophisticated multi-turn attacks, emphasizing the urgent need for more robust defense strategies.



## **46. Evolving Security in LLMs: A Study of Jailbreak Attacks and Defenses**

cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2504.02080v2) [paper-pdf](https://arxiv.org/pdf/2504.02080v2)

**Authors**: Zhengchun Shang, Wenlan Wei, Weiheng Bai

**Abstract**: Large Language Models (LLMs) are increasingly popular, powering a wide range of applications. Their widespread use has sparked concerns, especially through jailbreak attacks that bypass safety measures to produce harmful content.   In this paper, we present a comprehensive security analysis of large language models (LLMs), addressing critical research questions on the evolution and determinants of model safety.   Specifically, we begin by identifying the most effective techniques for detecting jailbreak attacks. Next, we investigate whether newer versions of LLMs offer improved security compared to their predecessors. We also assess the impact of model size on overall security and explore the potential benefits of integrating multiple defense strategies to enhance the security.   Our study evaluates both open-source (e.g., LLaMA and Mistral) and closed-source models (e.g., GPT-4) by employing four state-of-the-art attack techniques and assessing the efficacy of three new defensive approaches.



## **47. Effective and Efficient Jailbreaks of Black-Box LLMs with Cross-Behavior Attacks**

cs.CR

Code is at https://github.com/gohil-vasudev/JCB

**SubmitDate**: 2025-12-31    [abs](http://arxiv.org/abs/2503.08990v2) [paper-pdf](https://arxiv.org/pdf/2503.08990v2)

**Authors**: Vasudev Gohil

**Abstract**: Despite recent advancements in Large Language Models (LLMs) and their alignment, they can still be jailbroken, i.e., harmful and toxic content can be elicited from them. While existing red-teaming methods have shown promise in uncovering such vulnerabilities, these methods struggle with limited success and high computational and monetary costs. To address this, we propose a black-box Jailbreak method with Cross-Behavior attacks (JCB), that can automatically and efficiently find successful jailbreak prompts. JCB leverages successes from past behaviors to help jailbreak new behaviors, thereby significantly improving the attack efficiency. Moreover, JCB does not rely on time- and/or cost-intensive calls to auxiliary LLMs to discover/optimize the jailbreak prompts, making it highly efficient and scalable. Comprehensive experimental evaluations show that JCB significantly outperforms related baselines, requiring up to 94% fewer queries while still achieving 12.9% higher average attack success. JCB also achieves a notably high 37% attack success rate on Llama-2-7B, one of the most resilient LLMs, and shows promising zero-shot transferability across different LLMs.



## **48. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

cs.CL

Accepted by USENIX Security 2025

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2502.01386v3) [paper-pdf](https://arxiv.org/pdf/2502.01386v3)

**Authors**: Yuyang Gong, Zhuo Chen, Jiawei Liu, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.



## **49. AdvPrefix: An Objective for Nuanced LLM Jailbreaks**

cs.LG

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2412.10321v2) [paper-pdf](https://arxiv.org/pdf/2412.10321v2)

**Authors**: Sicheng Zhu, Brandon Amos, Yuandong Tian, Chuan Guo, Ivan Evtimov

**Abstract**: Many jailbreak attacks on large language models (LLMs) rely on a common objective: making the model respond with the prefix ``Sure, here is (harmful request)''. While straightforward, this objective has two limitations: limited control over model behaviors, yielding incomplete or unrealistic jailbroken responses, and a rigid format that hinders optimization. We introduce AdvPrefix, a plug-and-play prefix-forcing objective that selects one or more model-dependent prefixes by combining two criteria: high prefilling attack success rates and low negative log-likelihood. AdvPrefix integrates seamlessly into existing jailbreak attacks to mitigate the previous limitations for free. For example, replacing GCG's default prefixes on Llama-3 improves nuanced attack success rates from 14% to 80%, revealing that current safety alignment fails to generalize to new prefixes. Code and selected prefixes are released at github.com/facebookresearch/jailbreak-objectives.



## **50. Prompt Injection attack against LLM-integrated Applications**

cs.CR

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2306.05499v3) [paper-pdf](https://arxiv.org/pdf/2306.05499v3)

**Authors**: Yi Liu, Gelei Deng, Yuekang Li, Kailong Wang, Zihao Wang, Xiaofeng Wang, Tianwei Zhang, Yepang Liu, Haoyu Wang, Yan Zheng, Leo Yu Zhang, Yang Liu

**Abstract**: Large Language Models (LLMs), renowned for their superior proficiency in language comprehension and generation, stimulate a vibrant ecosystem of applications around them. However, their extensive assimilation into various services introduces significant security risks. This study deconstructs the complexities and implications of prompt injection attacks on actual LLM-integrated applications. Initially, we conduct an exploratory analysis on ten commercial applications, highlighting the constraints of current attack strategies in practice. Prompted by these limitations, we subsequently formulate HouYi, a novel black-box prompt injection attack technique, which draws inspiration from traditional web injection attacks. HouYi is compartmentalized into three crucial elements: a seamlessly-incorporated pre-constructed prompt, an injection prompt inducing context partition, and a malicious payload designed to fulfill the attack objectives. Leveraging HouYi, we unveil previously unknown and severe attack outcomes, such as unrestricted arbitrary LLM usage and uncomplicated application prompt theft. We deploy HouYi on 36 actual LLM-integrated applications and discern 31 applications susceptible to prompt injection. 10 vendors have validated our discoveries, including Notion, which has the potential to impact millions of users. Our investigation illuminates both the possible risks of prompt injection attacks and the possible tactics for mitigation.



