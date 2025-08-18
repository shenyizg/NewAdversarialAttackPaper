# Latest Large Language Model Attack Papers
**update at 2025-08-18 16:19:54**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. MCA-Bench: A Multimodal Benchmark for Evaluating CAPTCHA Robustness Against VLM-based Attacks**

cs.CV

we update the paper, add more experiments, and update the teammates

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2506.05982v4) [paper-pdf](http://arxiv.org/pdf/2506.05982v4)

**Authors**: Zonglin Wu, Yule Xue, Yaoyao Feng, Xiaolong Wang, Yiren Song

**Abstract**: As automated attack techniques rapidly advance, CAPTCHAs remain a critical defense mechanism against malicious bots. However, existing CAPTCHA schemes encompass a diverse range of modalities -- from static distorted text and obfuscated images to interactive clicks, sliding puzzles, and logic-based questions -- yet the community still lacks a unified, large-scale, multimodal benchmark to rigorously evaluate their security robustness. To address this gap, we introduce MCA-Bench, a comprehensive and reproducible benchmarking suite that integrates heterogeneous CAPTCHA types into a single evaluation protocol. Leveraging a shared vision-language model backbone, we fine-tune specialized cracking agents for each CAPTCHA category, enabling consistent, cross-modal assessments. Extensive experiments reveal that MCA-Bench effectively maps the vulnerability spectrum of modern CAPTCHA designs under varied attack settings, and crucially offers the first quantitative analysis of how challenge complexity, interaction depth, and model solvability interrelate. Based on these findings, we propose three actionable design principles and identify key open challenges, laying the groundwork for systematic CAPTCHA hardening, fair benchmarking, and broader community collaboration. Datasets and code are available online.



## **2. MCP-Guard: A Defense Framework for Model Context Protocol Integrity in Large Language Model Applications**

cs.CR

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10991v1) [paper-pdf](http://arxiv.org/pdf/2508.10991v1)

**Authors**: Wenpeng Xing, Zhonghao Qi, Yupeng Qin, Yilin Li, Caini Chang, Jiahui Yu, Changting Lin, Zhenzhen Xie, Meng Han

**Abstract**: The integration of Large Language Models (LLMs) with external tools via protocols such as the Model Context Protocol (MCP) introduces critical security vulnerabilities, including prompt injection, data exfiltration, and other threats. To counter these challenges, we propose MCP-Guard, a robust, layered defense architecture designed for LLM--tool interactions. MCP-Guard employs a three-stage detection pipeline that balances efficiency with accuracy: it progresses from lightweight static scanning for overt threats and a deep neural detector for semantic attacks, to our fine-tuned E5-based model achieves (96.01) accuracy in identifying adversarial prompts. Finally, a lightweight LLM arbitrator synthesizes these signals to deliver the final decision while minimizing false positives. To facilitate rigorous training and evaluation, we also introduce MCP-AttackBench, a comprehensive benchmark of over 70,000 samples. Sourced from public datasets and augmented by GPT-4, MCP-AttackBench simulates diverse, real-world attack vectors in the MCP format, providing a foundation for future research into securing LLM-tool ecosystems.



## **3. Failures to Surface Harmful Contents in Video Large Language Models**

cs.MM

11 pages, 8 figures

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10974v1) [paper-pdf](http://arxiv.org/pdf/2508.10974v1)

**Authors**: Yuxin Cao, Wei Song, Derui Wang, Jingling Xue, Jin Song Dong

**Abstract**: Video Large Language Models (VideoLLMs) are increasingly deployed on numerous critical applications, where users rely on auto-generated summaries while casually skimming the video stream. We show that this interaction hides a critical safety gap: if harmful content is embedded in a video, either as full-frame inserts or as small corner patches, state-of-the-art VideoLLMs rarely mention the harmful content in the output, despite its clear visibility to human viewers. A root-cause analysis reveals three compounding design flaws: (1) insufficient temporal coverage resulting from the sparse, uniformly spaced frame sampling used by most leading VideoLLMs, (2) spatial information loss introduced by aggressive token downsampling within sampled frames, and (3) encoder-decoder disconnection, whereby visual cues are only weakly utilized during text generation. Leveraging these insights, we craft three zero-query black-box attacks, aligning with these flaws in the processing pipeline. Our large-scale evaluation across five leading VideoLLMs shows that the harmfulness omission rate exceeds 90% in most cases. Even when harmful content is clearly present in all frames, these models consistently fail to identify it. These results underscore a fundamental vulnerability in current VideoLLMs' designs and highlight the urgent need for sampling strategies, token compression, and decoding mechanisms that guarantee semantic coverage rather than speed alone.



## **4. An Explainable Transformer-based Model for Phishing Email Detection: A Large Language Model Approach**

cs.LG

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2402.13871v2) [paper-pdf](http://arxiv.org/pdf/2402.13871v2)

**Authors**: Mohammad Amaz Uddin, Md Mahiuddin, Iqbal H. Sarker

**Abstract**: Phishing email is a serious cyber threat that tries to deceive users by sending false emails with the intention of stealing confidential information or causing financial harm. Attackers, often posing as trustworthy entities, exploit technological advancements and sophistication to make detection and prevention of phishing more challenging. Despite extensive academic research, phishing detection remains an ongoing and formidable challenge in the cybersecurity landscape. Large Language Models (LLMs) and Masked Language Models (MLMs) possess immense potential to offer innovative solutions to address long-standing challenges. In this research paper, we present an optimized, fine-tuned transformer-based DistilBERT model designed for the detection of phishing emails. In the detection process, we work with a phishing email dataset and utilize the preprocessing techniques to clean and solve the imbalance class issues. Through our experiments, we found that our model effectively achieves high accuracy, demonstrating its capability to perform well. Finally, we demonstrate our fine-tuned model using Explainable-AI (XAI) techniques such as Local Interpretable Model-Agnostic Explanations (LIME) and Transformer Interpret to explain how our model makes predictions in the context of text classification for phishing emails.



## **5. Layer-Wise Perturbations via Sparse Autoencoders for Adversarial Text Generation**

cs.CL

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10404v1) [paper-pdf](http://arxiv.org/pdf/2508.10404v1)

**Authors**: Huizhen Shu, Xuying Li, Qirui Wang, Yuji Kosuga, Mengqiu Tian, Zhuo Li

**Abstract**: With the rapid proliferation of Natural Language Processing (NLP), especially Large Language Models (LLMs), generating adversarial examples to jailbreak LLMs remains a key challenge for understanding model vulnerabilities and improving robustness. In this context, we propose a new black-box attack method that leverages the interpretability of large models. We introduce the Sparse Feature Perturbation Framework (SFPF), a novel approach for adversarial text generation that utilizes sparse autoencoders to identify and manipulate critical features in text. After using the SAE model to reconstruct hidden layer representations, we perform feature clustering on the successfully attacked texts to identify features with higher activations. These highly activated features are then perturbed to generate new adversarial texts. This selective perturbation preserves the malicious intent while amplifying safety signals, thereby increasing their potential to evade existing defenses. Our method enables a new red-teaming strategy that balances adversarial effectiveness with safety alignment. Experimental results demonstrate that adversarial texts generated by SFPF can bypass state-of-the-art defense mechanisms, revealing persistent vulnerabilities in current NLP systems.However, the method's effectiveness varies across prompts and layers, and its generalizability to other architectures and larger models remains to be validated.



## **6. Jailbreaking Commercial Black-Box LLMs with Explicitly Harmful Prompts**

cs.CL

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10390v1) [paper-pdf](http://arxiv.org/pdf/2508.10390v1)

**Authors**: Chiyu Zhang, Lu Zhou, Xiaogang Xu, Jiafei Wu, Liming Fang, Zhe Liu

**Abstract**: Evaluating jailbreak attacks is challenging when prompts are not overtly harmful or fail to induce harmful outputs. Unfortunately, many existing red-teaming datasets contain such unsuitable prompts. To evaluate attacks accurately, these datasets need to be assessed and cleaned for maliciousness. However, existing malicious content detection methods rely on either manual annotation, which is labor-intensive, or large language models (LLMs), which have inconsistent accuracy in harmful types. To balance accuracy and efficiency, we propose a hybrid evaluation framework named MDH (Malicious content Detection based on LLMs with Human assistance) that combines LLM-based annotation with minimal human oversight, and apply it to dataset cleaning and detection of jailbroken responses. Furthermore, we find that well-crafted developer messages can significantly boost jailbreak success, leading us to propose two new strategies: D-Attack, which leverages context simulation, and DH-CoT, which incorporates hijacked chains of thought. The Codes, datasets, judgements, and detection results will be released in github repository: https://github.com/AlienZhang1996/DH-CoT.



## **7. A Vision-Language Pre-training Model-Guided Approach for Mitigating Backdoor Attacks in Federated Learning**

cs.LG

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10315v1) [paper-pdf](http://arxiv.org/pdf/2508.10315v1)

**Authors**: Keke Gai, Dongjue Wang, Jing Yu, Liehuang Zhu, Qi Wu

**Abstract**: Existing backdoor defense methods in Federated Learning (FL) rely on the assumption of homogeneous client data distributions or the availability of a clean serve dataset, which limits the practicality and effectiveness. Defending against backdoor attacks under heterogeneous client data distributions while preserving model performance remains a significant challenge. In this paper, we propose a FL backdoor defense framework named CLIP-Fed, which leverages the zero-shot learning capabilities of vision-language pre-training models. By integrating both pre-aggregation and post-aggregation defense strategies, CLIP-Fed overcomes the limitations of Non-IID imposed on defense effectiveness. To address privacy concerns and enhance the coverage of the dataset against diverse triggers, we construct and augment the server dataset using the multimodal large language model and frequency analysis without any client samples. To address class prototype deviations caused by backdoor samples and eliminate the correlation between trigger patterns and target labels, CLIP-Fed aligns the knowledge of the global model and CLIP on the augmented dataset using prototype contrastive loss and Kullback-Leibler divergence. Extensive experiments on representative datasets validate the effectiveness of CLIP-Fed. Compared to state-of-the-art methods, CLIP-Fed achieves an average reduction in ASR, i.e., 2.03\% on CIFAR-10 and 1.35\% on CIFAR-10-LT, while improving average MA by 7.92\% and 0.48\%, respectively.



## **8. Security Concerns for Large Language Models: A Survey**

cs.CR

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2505.18889v3) [paper-pdf](http://arxiv.org/pdf/2505.18889v3)

**Authors**: Miles Q. Li, Benjamin C. M. Fung

**Abstract**: Large Language Models (LLMs) such as ChatGPT and its competitors have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. This survey provides a comprehensive overview of these emerging concerns, categorizing threats into several key areas: prompt injection and jailbreaking; adversarial attacks, including input perturbations and data poisoning; misuse by malicious actors to generate disinformation, phishing emails, and malware; and the worrisome risks inherent in autonomous LLM agents. Recently, a significant focus is increasingly being placed on the latter, exploring goal misalignment, emergent deception, self-preservation instincts, and the potential for LLMs to develop and pursue covert, misaligned objectives, a behavior known as scheming, which may even persist through safety training. We summarize recent academic and industrial studies from 2022 to 2025 that exemplify each threat, analyze proposed defenses and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial.



## **9. Extending the OWASP Multi-Agentic System Threat Modeling Guide: Insights from Multi-Agent Security Research**

cs.MA

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.09815v1) [paper-pdf](http://arxiv.org/pdf/2508.09815v1)

**Authors**: Klaudia Krawiecka, Christian Schroeder de Witt

**Abstract**: We propose an extension to the OWASP Multi-Agentic System (MAS) Threat Modeling Guide, translating recent anticipatory research in multi-agent security (MASEC) into practical guidance for addressing challenges unique to large language model (LLM)-driven multi-agent architectures. Although OWASP's existing taxonomy covers many attack vectors, our analysis identifies gaps in modeling failures, including, but not limited to: reasoning collapse across planner-executor chains, metric overfitting, unsafe delegation escalation, emergent covert coordination, and heterogeneous multi-agent exploits. We introduce additional threat classes and scenarios grounded in practical MAS deployments, highlighting risks from benign goal drift, cross-agent hallucination propagation, affective prompt framing, and multi-agent backdoors. We also outline evaluation strategies, including robustness testing, coordination assessment, safety enforcement, and emergent behavior monitoring, to ensure complete coverage. This work complements the framework of OWASP by expanding its applicability to increasingly complex, autonomous, and adaptive multi-agent systems, with the goal of improving security posture and resilience in real world deployments.



## **10. MetaCipher: A Time-Persistent and Universal Multi-Agent Framework for Cipher-Based Jailbreak Attacks for LLMs**

cs.CR

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2506.22557v2) [paper-pdf](http://arxiv.org/pdf/2506.22557v2)

**Authors**: Boyuan Chen, Minghao Shao, Abdul Basit, Siddharth Garg, Muhammad Shafique

**Abstract**: As large language models (LLMs) grow more capable, they face growing vulnerability to sophisticated jailbreak attacks. While developers invest heavily in alignment finetuning and safety guardrails, researchers continue publishing novel attacks, driving progress through adversarial iteration. This dynamic mirrors a strategic game of continual evolution. However, two major challenges hinder jailbreak development: the high cost of querying top-tier LLMs and the short lifespan of effective attacks due to frequent safety updates. These factors limit cost-efficiency and practical impact of research in jailbreak attacks. To address this, we propose MetaCipher, a low-cost, multi-agent jailbreak framework that generalizes across LLMs with varying safety measures. Using reinforcement learning, MetaCipher is modular and adaptive, supporting extensibility to future strategies. Within as few as 10 queries, MetaCipher achieves state-of-the-art attack success rates on recent malicious prompt benchmarks, outperforming prior jailbreak methods. We conduct a large-scale empirical evaluation across diverse victim models and benchmarks, demonstrating its robustness and adaptability. Warning: This paper contains model outputs that may be offensive or harmful, shown solely to demonstrate jailbreak efficacy.



## **11. Guardians and Offenders: A Survey on Harmful Content Generation and Safety Mitigation of LLM**

cs.CL

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.05775v2) [paper-pdf](http://arxiv.org/pdf/2508.05775v2)

**Authors**: Chi Zhang, Changjia Zhu, Junjie Xiong, Xiaoran Xu, Lingyao Li, Yao Liu, Zhuo Lu

**Abstract**: Large Language Models (LLMs) have revolutionized content creation across digital platforms, offering unprecedented capabilities in natural language generation and understanding. These models enable beneficial applications such as content generation, question and answering (Q&A), programming, and code reasoning. Meanwhile, they also pose serious risks by inadvertently or intentionally producing toxic, offensive, or biased content. This dual role of LLMs, both as powerful tools for solving real-world problems and as potential sources of harmful language, presents a pressing sociotechnical challenge. In this survey, we systematically review recent studies spanning unintentional toxicity, adversarial jailbreaking attacks, and content moderation techniques. We propose a unified taxonomy of LLM-related harms and defenses, analyze emerging multimodal and LLM-assisted jailbreak strategies, and assess mitigation efforts, including reinforcement learning with human feedback (RLHF), prompt engineering, and safety alignment. Our synthesis highlights the evolving landscape of LLM safety, identifies limitations in current evaluation methodologies, and outlines future research directions to guide the development of robust and ethically aligned language technologies.



## **12. NeuronTune: Fine-Grained Neuron Modulation for Balanced Safety-Utility Alignment in LLMs**

cs.LG

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.09473v1) [paper-pdf](http://arxiv.org/pdf/2508.09473v1)

**Authors**: Birong Pan, Mayi Xu, Qiankun Pi, Jianhao Chen, Yuanyuan Zhu, Ming Zhong, Tieyun Qian

**Abstract**: Ensuring robust safety alignment while preserving utility is critical for the reliable deployment of Large Language Models (LLMs). However, current techniques fundamentally suffer from intertwined deficiencies: insufficient robustness against malicious attacks, frequent refusal of benign queries, degradation in generated text quality and general task performance--the former two reflecting deficits in robust safety and the latter constituting utility impairment. We trace these limitations to the coarse-grained layer-wise interventions in existing methods. To resolve this, we propose NeuronTune, a fine-grained framework that dynamically modulates sparse neurons to achieve simultaneous safety-utility optimization. Our approach first identifies safety-critical and utility-preserving neurons across all layers via attribution, then employs meta-learning to adaptively amplify safety-neuron activations and suppress utility-neuron activations. Crucially, NeuronTune enables tunable adjustment of intervention scope via neuron-count thresholds, supporting flexible adaptation to security-critical or utility-priority scenarios. Extensive experimental results demonstrate that our method significantly outperforms existing state-of-the-art technologies, achieving superior model safety while maintaining excellent utility.



## **13. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

cs.CR

This work was submitted for review on Sept. 5, 2024, and the initial  version was uploaded to Arxiv on Sept. 30, 2024. The latest version reflects  the up-to-date experimental results

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2409.20002v4) [paper-pdf](http://arxiv.org/pdf/2409.20002v4)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.



## **14. Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference**

cs.CR

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.09442v1) [paper-pdf](http://arxiv.org/pdf/2508.09442v1)

**Authors**: Zhifan Luo, Shuo Shao, Su Zhang, Lijing Zhou, Yuke Hu, Chenxu Zhao, Zhihao Liu, Zhan Qin

**Abstract**: The Key-Value (KV) cache, which stores intermediate attention computations (Key and Value pairs) to avoid redundant calculations, is a fundamental mechanism for accelerating Large Language Model (LLM) inference. However, this efficiency optimization introduces significant yet underexplored privacy risks. This paper provides the first comprehensive analysis of these vulnerabilities, demonstrating that an attacker can reconstruct sensitive user inputs directly from the KV-cache. We design and implement three distinct attack vectors: a direct Inversion Attack, a more broadly applicable and potent Collision Attack, and a semantic-based Injection Attack. These methods demonstrate the practicality and severity of KV-cache privacy leakage issues. To mitigate this, we propose KV-Cloak, a novel, lightweight, and efficient defense mechanism. KV-Cloak uses a reversible matrix-based obfuscation scheme, combined with operator fusion, to secure the KV-cache. Our extensive experiments show that KV-Cloak effectively thwarts all proposed attacks, reducing reconstruction quality to random noise. Crucially, it achieves this robust security with virtually no degradation in model accuracy and minimal performance overhead, offering a practical solution for trustworthy LLM deployment.



## **15. Can AI Keep a Secret? Contextual Integrity Verification: A Provable Security Architecture for LLMs**

cs.CR

2 figures, 3 tables; code and certification harness:  https://github.com/ayushgupta4897/Contextual-Integrity-Verification ;  Elite-Attack dataset: https://huggingface.co/datasets/zyushg/elite-attack

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.09288v1) [paper-pdf](http://arxiv.org/pdf/2508.09288v1)

**Authors**: Aayush Gupta

**Abstract**: Large language models (LLMs) remain acutely vulnerable to prompt injection and related jailbreak attacks; heuristic guardrails (rules, filters, LLM judges) are routinely bypassed. We present Contextual Integrity Verification (CIV), an inference-time security architecture that attaches cryptographically signed provenance labels to every token and enforces a source-trust lattice inside the transformer via a pre-softmax hard attention mask (with optional FFN/residual gating). CIV provides deterministic, per-token non-interference guarantees on frozen models: lower-trust tokens cannot influence higher-trust representations. On benchmarks derived from recent taxonomies of prompt-injection vectors (Elite-Attack + SoK-246), CIV attains 0% attack success rate under the stated threat model while preserving 93.1% token-level similarity and showing no degradation in model perplexity on benign tasks; we note a latency overhead attributable to a non-optimized data path. Because CIV is a lightweight patch -- no fine-tuning required -- we demonstrate drop-in protection for Llama-3-8B and Mistral-7B. We release a reference implementation, an automated certification harness, and the Elite-Attack corpus to support reproducible research.



## **16. Attacks and Defenses Against LLM Fingerprinting**

cs.CR

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.09021v1) [paper-pdf](http://arxiv.org/pdf/2508.09021v1)

**Authors**: Kevin Kurian, Ethan Holland, Sean Oesch

**Abstract**: As large language models are increasingly deployed in sensitive environments, fingerprinting attacks pose significant privacy and security risks. We present a study of LLM fingerprinting from both offensive and defensive perspectives. Our attack methodology uses reinforcement learning to automatically optimize query selection, achieving better fingerprinting accuracy with only 3 queries compared to randomly selecting 3 queries from the same pool. Our defensive approach employs semantic-preserving output filtering through a secondary LLM to obfuscate model identity while maintaining semantic integrity. The defensive method reduces fingerprinting accuracy across tested models while preserving output quality. These contributions show the potential to improve fingerprinting tools capabilities while providing practical mitigation strategies against fingerprinting attacks.



## **17. GUARD:Dual-Agent based Backdoor Defense on Chain-of-Thought in Neural Code Generation**

cs.SE

Accepted by SEKE 2025

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2505.21425v3) [paper-pdf](http://arxiv.org/pdf/2505.21425v3)

**Authors**: Naizhu Jin, Zhong Li, Tian Zhang, Qingkai Zeng

**Abstract**: With the widespread application of large language models in code generation, recent studies demonstrate that employing additional Chain-of-Thought generation models can significantly enhance code generation performance by providing explicit reasoning steps. However, as external components, CoT models are particularly vulnerable to backdoor attacks, which existing defense mechanisms often fail to detect effectively. To address this challenge, we propose GUARD, a novel dual-agent defense framework specifically designed to counter CoT backdoor attacks in neural code generation. GUARD integrates two core components: GUARD-Judge, which identifies suspicious CoT steps and potential triggers through comprehensive analysis, and GUARD-Repair, which employs a retrieval-augmented generation approach to regenerate secure CoT steps for identified anomalies. Experimental results show that GUARD effectively mitigates attacks while maintaining generation quality, advancing secure code generation systems.



## **18. Whispers in the Machine: Confidentiality in Agentic Systems**

cs.CR

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2402.06922v4) [paper-pdf](http://arxiv.org/pdf/2402.06922v4)

**Authors**: Jonathan Evertz, Merlin Chlosta, Lea Schönherr, Thorsten Eisenhofer

**Abstract**: The interaction between users and applications is increasingly shifted toward natural language by deploying Large Language Models (LLMs) as the core interface. The capabilities of these so-called agents become more capable the more tools and services they serve as an interface for, ultimately leading to agentic systems. Agentic systems use LLM-based agents as interfaces for most user interactions and various integrations with external tools and services. While these interfaces can significantly enhance the capabilities of the agentic system, they also introduce a new attack surface. Manipulated integrations, for example, can exploit the internal LLM and compromise sensitive data accessed through other interfaces. While previous work primarily focused on attacks targeting a model's alignment or the leakage of training data, the security of data that is only available during inference has escaped scrutiny so far. In this work, we demonstrate how the integration of LLMs into systems with external tool integration poses a risk similar to established prompt-based attacks, able to compromise the confidentiality of the entire system. Introducing a systematic approach to evaluate these confidentiality risks, we identify two specific attack scenarios unique to these agentic systems and formalize these into a tool-robustness framework designed to measure a model's ability to protect sensitive information. Our analysis reveals significant vulnerabilities across all tested models, highlighting an increased risk when models are combined with external tools.



## **19. A Few Words Can Distort Graphs: Knowledge Poisoning Attacks on Graph-based Retrieval-Augmented Generation of Large Language Models**

cs.CL

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.04276v2) [paper-pdf](http://arxiv.org/pdf/2508.04276v2)

**Authors**: Jiayi Wen, Tianxin Chen, Zhirun Zheng, Cheng Huang

**Abstract**: Graph-based Retrieval-Augmented Generation (GraphRAG) has recently emerged as a promising paradigm for enhancing large language models (LLMs) by converting raw text into structured knowledge graphs, improving both accuracy and explainability. However, GraphRAG relies on LLMs to extract knowledge from raw text during graph construction, and this process can be maliciously manipulated to implant misleading information. Targeting this attack surface, we propose two knowledge poisoning attacks (KPAs) and demonstrate that modifying only a few words in the source text can significantly change the constructed graph, poison the GraphRAG, and severely mislead downstream reasoning. The first attack, named Targeted KPA (TKPA), utilizes graph-theoretic analysis to locate vulnerable nodes in the generated graphs and rewrites the corresponding narratives with LLMs, achieving precise control over specific question-answering (QA) outcomes with a success rate of 93.1\%, while keeping the poisoned text fluent and natural. The second attack, named Universal KPA (UKPA), exploits linguistic cues such as pronouns and dependency relations to disrupt the structural integrity of the generated graph by altering globally influential words. With fewer than 0.05\% of full text modified, the QA accuracy collapses from 95\% to 50\%. Furthermore, experiments show that state-of-the-art defense methods fail to detect these attacks, highlighting that securing GraphRAG pipelines against knowledge poisoning remains largely unexplored.



## **20. Chimera: Harnessing Multi-Agent LLMs for Automatic Insider Threat Simulation**

cs.CR

23 pages

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.07745v2) [paper-pdf](http://arxiv.org/pdf/2508.07745v2)

**Authors**: Jiongchi Yu, Xiaofei Xie, Qiang Hu, Yuhan Ma, Ziming Zhao

**Abstract**: Insider threats, which can lead to severe losses, remain a major security concern. While machine learning-based insider threat detection (ITD) methods have shown promising results, their progress is hindered by the scarcity of high-quality data. Enterprise data is sensitive and rarely accessible, while publicly available datasets, when limited in scale due to cost, lack sufficient real-world coverage; and when purely synthetic, they fail to capture rich semantics and realistic user behavior. To address this, we propose Chimera, the first large language model (LLM)-based multi-agent framework that automatically simulates both benign and malicious insider activities and collects diverse logs across diverse enterprise environments. Chimera models each employee with agents that have role-specific behavior and integrates modules for group meetings, pairwise interactions, and autonomous scheduling, capturing realistic organizational dynamics. It incorporates 15 types of insider attacks (e.g., IP theft, system sabotage) and has been deployed to simulate activities in three sensitive domains: technology company, finance corporation, and medical institution, producing a new dataset, ChimeraLog. We assess ChimeraLog via human studies and quantitative analysis, confirming its diversity, realism, and presence of explainable threat patterns. Evaluations of existing ITD methods show an average F1-score of 0.83, which is significantly lower than 0.99 on the CERT dataset, demonstrating ChimeraLog's higher difficulty and utility for advancing ITD research.



## **21. Securing Educational LLMs: A Generalised Taxonomy of Attacks on LLMs and DREAD Risk Assessment**

cs.CY

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.08629v1) [paper-pdf](http://arxiv.org/pdf/2508.08629v1)

**Authors**: Farzana Zahid, Anjalika Sewwandi, Lee Brandon, Vimal Kumar, Roopak Sinha

**Abstract**: Due to perceptions of efficiency and significant productivity gains, various organisations, including in education, are adopting Large Language Models (LLMs) into their workflows. Educator-facing, learner-facing, and institution-facing LLMs, collectively, Educational Large Language Models (eLLMs), complement and enhance the effectiveness of teaching, learning, and academic operations. However, their integration into an educational setting raises significant cybersecurity concerns. A comprehensive landscape of contemporary attacks on LLMs and their impact on the educational environment is missing. This study presents a generalised taxonomy of fifty attacks on LLMs, which are categorized as attacks targeting either models or their infrastructure. The severity of these attacks is evaluated in the educational sector using the DREAD risk assessment framework. Our risk assessment indicates that token smuggling, adversarial prompts, direct injection, and multi-step jailbreak are critical attacks on eLLMs. The proposed taxonomy, its application in the educational environment, and our risk assessment will help academic and industrial practitioners to build resilient solutions that protect learners and institutions.



## **22. Few-Shot Adversarial Low-Rank Fine-Tuning of Vision-Language Models**

cs.LG

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2505.15130v2) [paper-pdf](http://arxiv.org/pdf/2505.15130v2)

**Authors**: Sajjad Ghiasvand, Haniyeh Ehsani Oskouie, Mahnoosh Alizadeh, Ramtin Pedarsani

**Abstract**: Vision-Language Models (VLMs) such as CLIP have shown remarkable performance in cross-modal tasks through large-scale contrastive pre-training. To adapt these large transformer-based models efficiently for downstream tasks, Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA have emerged as scalable alternatives to full fine-tuning, especially in few-shot scenarios. However, like traditional deep neural networks, VLMs are highly vulnerable to adversarial attacks, where imperceptible perturbations can significantly degrade model performance. Adversarial training remains the most effective strategy for improving model robustness in PEFT. In this work, we propose AdvCLIP-LoRA, the first algorithm designed to enhance the adversarial robustness of CLIP models fine-tuned with LoRA in few-shot settings. Our method formulates adversarial fine-tuning as a minimax optimization problem and provides theoretical guarantees for convergence under smoothness and nonconvex-strong-concavity assumptions. Empirical results across eight datasets using ViT-B/16 and ViT-B/32 models show that AdvCLIP-LoRA significantly improves robustness against common adversarial attacks (e.g., FGSM, PGD), without sacrificing much clean accuracy. These findings highlight AdvCLIP-LoRA as a practical and theoretically grounded approach for robust adaptation of VLMs in resource-constrained settings.



## **23. Selective KV-Cache Sharing to Mitigate Timing Side-Channels in LLM Inference**

cs.CR

17 pages,17 figures

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.08438v1) [paper-pdf](http://arxiv.org/pdf/2508.08438v1)

**Authors**: Kexin Chu, Zecheng Lin, Dawei Xiang, Zixu Shen, Jianchang Su, Cheng Chu, Yiwei Yang, Wenhui Zhang, Wenfei Wu, Wei Zhang

**Abstract**: Global KV-cache sharing has emerged as a key optimization for accelerating large language model (LLM) inference. However, it exposes a new class of timing side-channel attacks, enabling adversaries to infer sensitive user inputs via shared cache entries. Existing defenses, such as per-user isolation, eliminate leakage but degrade performance by up to 38.9% in time-to-first-token (TTFT), making them impractical for high-throughput deployment. To address this gap, we introduce SafeKV (Secure and Flexible KV Cache Sharing), a privacy-aware KV-cache management framework that selectively shares non-sensitive entries while confining sensitive content to private caches. SafeKV comprises three components: (i) a hybrid, multi-tier detection pipeline that integrates rule-based pattern matching, a general-purpose privacy detector, and context-aware validation; (ii) a unified radix-tree index that manages public and private entries across heterogeneous memory tiers (HBM, DRAM, SSD); and (iii) entropy-based access monitoring to detect and mitigate residual information leakage. Our evaluation shows that SafeKV mitigates 94% - 97% of timing-based side-channel attacks. Compared to per-user isolation method, SafeKV improves TTFT by up to 40.58% and throughput by up to 2.66X across diverse LLMs and workloads. SafeKV reduces cache-induced TTFT overhead from 50.41% to 11.74% on Qwen3-235B. By combining fine-grained privacy control with high cache reuse efficiency, SafeKV reclaims the performance advantages of global sharing while providing robust runtime privacy guarantees for LLM inference.



## **24. Towards Effective MLLM Jailbreaking Through Balanced On-Topicness and OOD-Intensity**

cs.CV

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.09218v1) [paper-pdf](http://arxiv.org/pdf/2508.09218v1)

**Authors**: Zuoou Li, Weitong Zhang, Jingyuan Wang, Shuyuan Zhang, Wenjia Bai, Bernhard Kainz, Mengyun Qiao

**Abstract**: Multimodal large language models (MLLMs) are widely used in vision-language reasoning tasks. However, their vulnerability to adversarial prompts remains a serious concern, as safety mechanisms often fail to prevent the generation of harmful outputs. Although recent jailbreak strategies report high success rates, many responses classified as "successful" are actually benign, vague, or unrelated to the intended malicious goal. This mismatch suggests that current evaluation standards may overestimate the effectiveness of such attacks. To address this issue, we introduce a four-axis evaluation framework that considers input on-topicness, input out-of-distribution (OOD) intensity, output harmfulness, and output refusal rate. This framework identifies truly effective jailbreaks. In a substantial empirical study, we reveal a structural trade-off: highly on-topic prompts are frequently blocked by safety filters, whereas those that are too OOD often evade detection but fail to produce harmful content. However, prompts that balance relevance and novelty are more likely to evade filters and trigger dangerous output. Building on this insight, we develop a recursive rewriting strategy called Balanced Structural Decomposition (BSD). The approach restructures malicious prompts into semantically aligned sub-tasks, while introducing subtle OOD signals and visual cues that make the inputs harder to detect. BSD was tested across 13 commercial and open-source MLLMs, where it consistently led to higher attack success rates, more harmful outputs, and fewer refusals. Compared to previous methods, it improves success rates by $67\%$ and harmfulness by $21\%$, revealing a previously underappreciated weakness in current multimodal safety systems.



## **25. Assessing LLM Text Detection in Educational Contexts: Does Human Contribution Affect Detection?**

cs.CL

Preprint as provided by the authors (19 pages, 12 figures, 9 tables)

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.08096v1) [paper-pdf](http://arxiv.org/pdf/2508.08096v1)

**Authors**: Lukas Gehring, Benjamin Paaßen

**Abstract**: Recent advancements in Large Language Models (LLMs) and their increased accessibility have made it easier than ever for students to automatically generate texts, posing new challenges for educational institutions. To enforce norms of academic integrity and ensure students' learning, learning analytics methods to automatically detect LLM-generated text appear increasingly appealing. This paper benchmarks the performance of different state-of-the-art detectors in educational contexts, introducing a novel dataset, called Generative Essay Detection in Education (GEDE), containing over 900 student-written essays and over 12,500 LLM-generated essays from various domains. To capture the diversity of LLM usage practices in generating text, we propose the concept of contribution levels, representing students' contribution to a given assignment. These levels range from purely human-written texts, to slightly LLM-improved versions, to fully LLM-generated texts, and finally to active attacks on the detector by "humanizing" generated texts. We show that most detectors struggle to accurately classify texts of intermediate student contribution levels, like LLM-improved human-written texts. Detectors are particularly likely to produce false positives, which is problematic in educational settings where false suspicions can severely impact students' lives. Our dataset, code, and additional supplementary materials are publicly available at https://github.com/lukasgehring/Assessing-LLM-Text-Detection-in-Educational-Contexts.



## **26. BadPromptFL: A Novel Backdoor Threat to Prompt-based Federated Learning in Multimodal Models**

cs.LG

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.08040v1) [paper-pdf](http://arxiv.org/pdf/2508.08040v1)

**Authors**: Maozhen Zhang, Mengnan Zhao, Bo Wang

**Abstract**: Prompt-based tuning has emerged as a lightweight alternative to full fine-tuning in large vision-language models, enabling efficient adaptation via learned contextual prompts. This paradigm has recently been extended to federated learning settings (e.g., PromptFL), where clients collaboratively train prompts under data privacy constraints. However, the security implications of prompt-based aggregation in federated multimodal learning remain largely unexplored, leaving a critical attack surface unaddressed. In this paper, we introduce \textbf{BadPromptFL}, the first backdoor attack targeting prompt-based federated learning in multimodal contrastive models. In BadPromptFL, compromised clients jointly optimize local backdoor triggers and prompt embeddings, injecting poisoned prompts into the global aggregation process. These prompts are then propagated to benign clients, enabling universal backdoor activation at inference without modifying model parameters. Leveraging the contextual learning behavior of CLIP-style architectures, BadPromptFL achieves high attack success rates (e.g., \(>90\%\)) with minimal visibility and limited client participation. Extensive experiments across multiple datasets and aggregation protocols validate the effectiveness, stealth, and generalizability of our attack, raising critical concerns about the robustness of prompt-based federated learning in real-world deployments.



## **27. Robust Anomaly Detection in O-RAN: Leveraging LLMs against Data Manipulation Attacks**

cs.CR

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.08029v1) [paper-pdf](http://arxiv.org/pdf/2508.08029v1)

**Authors**: Thusitha Dayaratne, Ngoc Duy Pham, Viet Vo, Shangqi Lai, Sharif Abuadbba, Hajime Suzuki, Xingliang Yuan, Carsten Rudolph

**Abstract**: The introduction of 5G and the Open Radio Access Network (O-RAN) architecture has enabled more flexible and intelligent network deployments. However, the increased complexity and openness of these architectures also introduce novel security challenges, such as data manipulation attacks on the semi-standardised Shared Data Layer (SDL) within the O-RAN platform through malicious xApps. In particular, malicious xApps can exploit this vulnerability by introducing subtle Unicode-wise alterations (hypoglyphs) into the data that are being used by traditional machine learning (ML)-based anomaly detection methods. These Unicode-wise manipulations can potentially bypass detection and cause failures in anomaly detection systems based on traditional ML, such as AutoEncoders, which are unable to process hypoglyphed data without crashing. We investigate the use of Large Language Models (LLMs) for anomaly detection within the O-RAN architecture to address this challenge. We demonstrate that LLM-based xApps maintain robust operational performance and are capable of processing manipulated messages without crashing. While initial detection accuracy requires further improvements, our results highlight the robustness of LLMs to adversarial attacks such as hypoglyphs in input data. There is potential to use their adaptability through prompt engineering to further improve the accuracy, although this requires further research. Additionally, we show that LLMs achieve low detection latency (under 0.07 seconds), making them suitable for Near-Real-Time (Near-RT) RIC deployments.



## **28. Toward Intelligent and Secure Cloud: Large Language Model Empowered Proactive Defense**

cs.CR

7 pages; Major Revision for IEEE Communications Magazine

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2412.21051v3) [paper-pdf](http://arxiv.org/pdf/2412.21051v3)

**Authors**: Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen, Yuyu Zhao

**Abstract**: The rapid evolution of cloud computing technologies and the increasing number of cloud applications have provided numerous benefits in our daily lives. However, the diversity and complexity of different components pose a significant challenge to cloud security, especially when dealing with sophisticated and advanced cyberattacks such as Denial of Service (DoS). Recent advancements in the large language models (LLMs) offer promising solutions for security intelligence. By exploiting the powerful capabilities in language understanding, data analysis, task inference, action planning, and code generation, we present LLM-PD, a novel defense architecture that proactively mitigates various DoS threats in cloud networks. LLM-PD can efficiently make decisions through comprehensive data analysis and sequential reasoning, as well as dynamically create and deploy actionable defense mechanisms. Furthermore, it can flexibly self-evolve based on experience learned from previous interactions and adapt to new attack scenarios without additional training. Our case study on three distinct DoS attacks demonstrates its remarkable ability in terms of defense effectiveness and efficiency when compared with other existing methods.



## **29. Can You Trick the Grader? Adversarial Persuasion of LLM Judges**

cs.CL

19 pages, 8 figures

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.07805v1) [paper-pdf](http://arxiv.org/pdf/2508.07805v1)

**Authors**: Yerin Hwang, Dongryeol Lee, Taegwan Kang, Yongil Kim, Kyomin Jung

**Abstract**: As large language models take on growing roles as automated evaluators in practical settings, a critical question arises: Can individuals persuade an LLM judge to assign unfairly high scores? This study is the first to reveal that strategically embedded persuasive language can bias LLM judges when scoring mathematical reasoning tasks, where correctness should be independent of stylistic variation. Grounded in Aristotle's rhetorical principles, we formalize seven persuasion techniques (Majority, Consistency, Flattery, Reciprocity, Pity, Authority, Identity) and embed them into otherwise identical responses. Across six math benchmarks, we find that persuasive language leads LLM judges to assign inflated scores to incorrect solutions, by up to 8% on average, with Consistency causing the most severe distortion. Notably, increasing model size does not substantially mitigate this vulnerability. Further analysis demonstrates that combining multiple persuasion techniques amplifies the bias, and pairwise evaluation is likewise susceptible. Moreover, the persuasive effect persists under counter prompting strategies, highlighting a critical vulnerability in LLM-as-a-Judge pipelines and underscoring the need for robust defenses against persuasion-based attacks.



## **30. Multi-Turn Jailbreaks Are Simpler Than They Seem**

cs.LG

25 pages, 15 figures. Accepted at COLM 2025 SoLaR Workshop

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.07646v1) [paper-pdf](http://arxiv.org/pdf/2508.07646v1)

**Authors**: Xiaoxue Yang, Jaeha Lee, Anna-Katharina Dick, Jasper Timm, Fei Xie, Diogo Cruz

**Abstract**: While defenses against single-turn jailbreak attacks on Large Language Models (LLMs) have improved significantly, multi-turn jailbreaks remain a persistent vulnerability, often achieving success rates exceeding 70% against models optimized for single-turn protection. This work presents an empirical analysis of automated multi-turn jailbreak attacks across state-of-the-art models including GPT-4, Claude, and Gemini variants, using the StrongREJECT benchmark. Our findings challenge the perceived sophistication of multi-turn attacks: when accounting for the attacker's ability to learn from how models refuse harmful requests, multi-turn jailbreaking approaches are approximately equivalent to simply resampling single-turn attacks multiple times. Moreover, attack success is correlated among similar models, making it easier to jailbreak newly released ones. Additionally, for reasoning models, we find surprisingly that higher reasoning effort often leads to higher attack success rates. Our results have important implications for AI safety evaluation and the design of jailbreak-resistant systems. We release the source code at https://github.com/diogo-cruz/multi_turn_simpler



## **31. Pentest-R1: Towards Autonomous Penetration Testing Reasoning Optimized via Two-Stage Reinforcement Learning**

cs.AI

**SubmitDate**: 2025-08-10    [abs](http://arxiv.org/abs/2508.07382v1) [paper-pdf](http://arxiv.org/pdf/2508.07382v1)

**Authors**: He Kong, Die Hu, Jingguo Ge, Liangxiong Li, Hui Li, Tong Li

**Abstract**: Automating penetration testing is crucial for enhancing cybersecurity, yet current Large Language Models (LLMs) face significant limitations in this domain, including poor error handling, inefficient reasoning, and an inability to perform complex end-to-end tasks autonomously. To address these challenges, we introduce Pentest-R1, a novel framework designed to optimize LLM reasoning capabilities for this task through a two-stage reinforcement learning pipeline. We first construct a dataset of over 500 real-world, multi-step walkthroughs, which Pentest-R1 leverages for offline reinforcement learning (RL) to instill foundational attack logic. Subsequently, the LLM is fine-tuned via online RL in an interactive Capture The Flag (CTF) environment, where it learns directly from environmental feedback to develop robust error self-correction and adaptive strategies. Our extensive experiments on the Cybench and AutoPenBench benchmarks demonstrate the framework's effectiveness. On AutoPenBench, Pentest-R1 achieves a 24.2\% success rate, surpassing most state-of-the-art models and ranking second only to Gemini 2.5 Flash. On Cybench, it attains a 15.0\% success rate in unguided tasks, establishing a new state-of-the-art for open-source LLMs and matching the performance of top proprietary models. Ablation studies confirm that the synergy of both training stages is critical to its success.



## **32. Multi-task Adversarial Attacks against Black-box Model with Few-shot Queries**

cs.CR

**SubmitDate**: 2025-08-10    [abs](http://arxiv.org/abs/2508.10039v1) [paper-pdf](http://arxiv.org/pdf/2508.10039v1)

**Authors**: Wenqiang Wang, Yan Xiao, Hao Lin, Yangshijie Zhang, Xiaochun Cao

**Abstract**: Current multi-task adversarial text attacks rely on abundant access to shared internal features and numerous queries, often limited to a single task type. As a result, these attacks are less effective against practical scenarios involving black-box feedback APIs, limited queries, or multiple task types. To bridge this gap, we propose \textbf{C}luster and \textbf{E}nsemble \textbf{M}ulti-task Text Adversarial \textbf{A}ttack (\textbf{CEMA}), an effective black-box attack that exploits the transferability of adversarial texts across different tasks. CEMA simplifies complex multi-task scenarios by using a \textit{deep-level substitute model} trained in a \textit{plug-and-play} manner for text classification, enabling attacks without mimicking the victim model. This approach requires only a few queries for training, converting multi-task attacks into classification attacks and allowing attacks across various tasks.   CEMA generates multiple adversarial candidates using different text classification methods and selects the one that most effectively attacks substitute models.   In experiments involving multi-task models with two, three, or six tasks--spanning classification, translation, summarization, and text-to-image generation--CEMA demonstrates significant attack success with as few as 100 queries. Furthermore, CEMA can target commercial APIs (e.g., Baidu and Google Translate), large language models (e.g., ChatGPT 4o), and image-generation models (e.g., Stable Diffusion V2), showcasing its versatility and effectiveness in real-world applications.



## **33. Omni-SafetyBench: A Benchmark for Safety Evaluation of Audio-Visual Large Language Models**

cs.CL

20 pages, 8 figures, 12 tables

**SubmitDate**: 2025-08-10    [abs](http://arxiv.org/abs/2508.07173v1) [paper-pdf](http://arxiv.org/pdf/2508.07173v1)

**Authors**: Leyi Pan, Zheyu Fu, Yunpeng Zhai, Shuchang Tao, Sheng Guan, Shiyu Huang, Lingzhe Zhang, Zhaoyang Liu, Bolin Ding, Felix Henry, Lijie Wen, Aiwei Liu

**Abstract**: The rise of Omni-modal Large Language Models (OLLMs), which integrate visual and auditory processing with text, necessitates robust safety evaluations to mitigate harmful outputs. However, no dedicated benchmarks currently exist for OLLMs, and prior benchmarks designed for other LLMs lack the ability to assess safety performance under audio-visual joint inputs or cross-modal safety consistency. To fill this gap, we introduce Omni-SafetyBench, the first comprehensive parallel benchmark for OLLM safety evaluation, featuring 24 modality combinations and variations with 972 samples each, including dedicated audio-visual harm cases. Considering OLLMs' comprehension challenges with complex omni-modal inputs and the need for cross-modal consistency evaluation, we propose tailored metrics: a Safety-score based on conditional Attack Success Rate (C-ASR) and Refusal Rate (C-RR) to account for comprehension failures, and a Cross-Modal Safety Consistency Score (CMSC-score) to measure consistency across modalities. Evaluating 6 open-source and 4 closed-source OLLMs reveals critical vulnerabilities: (1) no model excels in both overall safety and consistency, with only 3 models achieving over 0.6 in both metrics and top performer scoring around 0.8; (2) safety defenses weaken with complex inputs, especially audio-visual joints; (3) severe weaknesses persist, with some models scoring as low as 0.14 on specific modalities. Our benchmark and metrics highlight urgent needs for enhanced OLLM safety, providing a foundation for future improvements.



## **34. Model-Agnostic Sentiment Distribution Stability Analysis for Robust LLM-Generated Texts Detection**

cs.CL

**SubmitDate**: 2025-08-09    [abs](http://arxiv.org/abs/2508.06913v1) [paper-pdf](http://arxiv.org/pdf/2508.06913v1)

**Authors**: Siyuan Li, Xi Lin, Guangyan Li, Zehao Liu, Aodu Wulianghai, Li Ding, Jun Wu, Jianhua Li

**Abstract**: The rapid advancement of large language models (LLMs) has resulted in increasingly sophisticated AI-generated content, posing significant challenges in distinguishing LLM-generated text from human-written language. Existing detection methods, primarily based on lexical heuristics or fine-tuned classifiers, often suffer from limited generalizability and are vulnerable to paraphrasing, adversarial perturbations, and cross-domain shifts. In this work, we propose SentiDetect, a model-agnostic framework for detecting LLM-generated text by analyzing the divergence in sentiment distribution stability. Our method is motivated by the empirical observation that LLM outputs tend to exhibit emotionally consistent patterns, whereas human-written texts display greater emotional variability. To capture this phenomenon, we define two complementary metrics: sentiment distribution consistency and sentiment distribution preservation, which quantify stability under sentiment-altering and semantic-preserving transformations. We evaluate SentiDetect on five diverse datasets and a range of advanced LLMs,including Gemini-1.5-Pro, Claude-3, GPT-4-0613, and LLaMa-3.3. Experimental results demonstrate its superiority over state-of-the-art baselines, with over 16% and 11% F1 score improvements on Gemini-1.5-Pro and GPT-4-0613, respectively. Moreover, SentiDetect also shows greater robustness to paraphrasing, adversarial attacks, and text length variations, outperforming existing detectors in challenging scenarios.



## **35. Context Misleads LLMs: The Role of Context Filtering in Maintaining Safe Alignment of LLMs**

cs.CR

13 pages, 2 figures

**SubmitDate**: 2025-08-09    [abs](http://arxiv.org/abs/2508.10031v1) [paper-pdf](http://arxiv.org/pdf/2508.10031v1)

**Authors**: Jinhwa Kim, Ian G. Harris

**Abstract**: While Large Language Models (LLMs) have shown significant advancements in performance, various jailbreak attacks have posed growing safety and ethical risks. Malicious users often exploit adversarial context to deceive LLMs, prompting them to generate responses to harmful queries. In this study, we propose a new defense mechanism called Context Filtering model, an input pre-processing method designed to filter out untrustworthy and unreliable context while identifying the primary prompts containing the real user intent to uncover concealed malicious intent. Given that enhancing the safety of LLMs often compromises their helpfulness, potentially affecting the experience of benign users, our method aims to improve the safety of the LLMs while preserving their original performance. We evaluate the effectiveness of our model in defending against jailbreak attacks through comparative analysis, comparing our approach with state-of-the-art defense mechanisms against six different attacks and assessing the helpfulness of LLMs under these defenses. Our model demonstrates its ability to reduce the Attack Success Rates of jailbreak attacks by up to 88% while maintaining the original LLMs' performance, achieving state-of-the-art Safety and Helpfulness Product results. Notably, our model is a plug-and-play method that can be applied to all LLMs, including both white-box and black-box models, to enhance their safety without requiring any fine-tuning of the models themselves. We will make our model publicly available for research purposes.



## **36. Towards Robust Red-Green Watermarking for Autoregressive Image Generators**

cs.CV

**SubmitDate**: 2025-08-08    [abs](http://arxiv.org/abs/2508.06656v1) [paper-pdf](http://arxiv.org/pdf/2508.06656v1)

**Authors**: Denis Lukovnikov, Andreas Müller, Erwin Quiring, Asja Fischer

**Abstract**: In-generation watermarking for detecting and attributing generated content has recently been explored for latent diffusion models (LDMs), demonstrating high robustness. However, the use of in-generation watermarks in autoregressive (AR) image models has not been explored yet. AR models generate images by autoregressively predicting a sequence of visual tokens that are then decoded into pixels using a vector-quantized decoder. Inspired by red-green watermarks for large language models, we examine token-level watermarking schemes that bias the next-token prediction based on prior tokens. We find that a direct transfer of these schemes works in principle, but the detectability of the watermarks decreases considerably under common image perturbations. As a remedy, we propose two novel watermarking methods that rely on visual token clustering to assign similar tokens to the same set. Firstly, we investigate a training-free approach that relies on a cluster lookup table, and secondly, we finetune VAE encoders to predict token clusters directly from perturbed images. Overall, our experiments show that cluster-level watermarks improve robustness against perturbations and regeneration attacks while preserving image quality. Cluster classification further boosts watermark detectability, outperforming a set of baselines. Moreover, our methods offer fast verification runtime, comparable to lightweight post-hoc watermarking methods.



## **37. Latent Fusion Jailbreak: Blending Harmful and Harmless Representations to Elicit Unsafe LLM Outputs**

cs.CL

**SubmitDate**: 2025-08-08    [abs](http://arxiv.org/abs/2508.10029v1) [paper-pdf](http://arxiv.org/pdf/2508.10029v1)

**Authors**: Wenpeng Xing, Mohan Li, Chunqiang Hu, Haitao XuNingyu Zhang, Bo Lin, Meng Han

**Abstract**: Large language models (LLMs) demonstrate impressive capabilities in various language tasks but are susceptible to jailbreak attacks that circumvent their safety alignments. This paper introduces Latent Fusion Jailbreak (LFJ), a representation-based attack that interpolates hidden states from harmful and benign query pairs to elicit prohibited responses. LFJ begins by selecting query pairs with high thematic and syntactic similarity, then performs gradient-guided interpolation at influential layers and tokens, followed by optimization to balance attack success, output fluency, and computational efficiency. Evaluations on models such as Vicuna and LLaMA-2 across benchmarks like AdvBench and MaliciousInstruct yield an average attack success rate (ASR) of 94.01%, outperforming existing methods. To mitigate LFJ, we propose an adversarial training defense that fine-tunes models on interpolated examples, reducing ASR by over 80% without degrading performance on benign inputs. Ablation studies validate the importance of query pair selection, hidden state interpolation components, and optimization strategies in LFJ's effectiveness.



## **38. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

cs.CL

Our code is publicly available at  https://github.com/UKPLab/arxiv2025-poate-attack

**SubmitDate**: 2025-08-08    [abs](http://arxiv.org/abs/2501.01872v3) [paper-pdf](http://arxiv.org/pdf/2501.01872v3)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.



## **39. Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models: A Unified and Accurate Approach**

cs.CR

**SubmitDate**: 2025-08-08    [abs](http://arxiv.org/abs/2508.09201v1) [paper-pdf](http://arxiv.org/pdf/2508.09201v1)

**Authors**: Shuang Liang, Zhihao Xu, Jialing Tao, Hui Xue, Xiting Wang

**Abstract**: Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. Although recent detection works have shifted to internal representations due to their rich cross-modal information, most methods rely on heuristic rules rather than principled objectives, resulting in suboptimal performance. To address these limitations, we propose Learning to Detect (LoD), a novel unsupervised framework that formulates jailbreak detection as anomaly detection. LoD introduces two key components: Multi-modal Safety Concept Activation Vectors (MSCAV), which capture layer-wise safety-related representations across modalities, and the Safety Pattern Auto-Encoder, which models the distribution of MSCAV derived from safe inputs and detects anomalies via reconstruction errors. By training the auto-encoder (AE) solely on safe samples without attack labels, LoD naturally identifies jailbreak inputs as distributional anomalies, enabling accurate and unified detection of jailbreak attacks. Comprehensive experiments on three different LVLMs and five benchmarks demonstrate that LoD achieves state-of-the-art performance, with an average AUROC of 0.9951 and an improvement of up to 38.89% in the minimum AUROC over the strongest baselines.



## **40. In-Training Defenses against Emergent Misalignment in Language Models**

cs.LG

Under review

**SubmitDate**: 2025-08-08    [abs](http://arxiv.org/abs/2508.06249v1) [paper-pdf](http://arxiv.org/pdf/2508.06249v1)

**Authors**: David Kaczér, Magnus Jørgenvåg, Clemens Vetter, Lucie Flek, Florian Mai

**Abstract**: Fine-tuning lets practitioners repurpose aligned large language models (LLMs) for new domains, yet recent work reveals emergent misalignment (EMA): Even a small, domain-specific fine-tune can induce harmful behaviors far outside the target domain. Even in the case where model weights are hidden behind a fine-tuning API, this gives attackers inadvertent access to a broadly misaligned model in a way that can be hard to detect from the fine-tuning data alone. We present the first systematic study of in-training safeguards against EMA that are practical for providers who expose fine-tuning via an API. We investigate four training regularization interventions: (i) KL-divergence regularization toward a safe reference model, (ii) $\ell_2$ distance in feature space, (iii) projecting onto a safe subspace (SafeLoRA), and (iv) interleaving of a small amount of safe training examples from a general instruct-tuning dataset. We first evaluate the methods' emergent misalignment effect across four malicious, EMA-inducing tasks. Second, we assess the methods' impacts on benign tasks. We conclude with a discussion of open questions in emergent misalignment research.



## **41. Feedback-Guided Extraction of Knowledge Base from Retrieval-Augmented LLM Applications**

cs.CR

**SubmitDate**: 2025-08-08    [abs](http://arxiv.org/abs/2411.14110v2) [paper-pdf](http://arxiv.org/pdf/2411.14110v2)

**Authors**: Changyue Jiang, Xudong Pan, Geng Hong, Chenfu Bao, Yang Chen, Min Yang

**Abstract**: Retrieval-Augmented Generation (RAG) expands the knowledge boundary of large language models (LLMs) by integrating external knowledge bases, whose construction is often time-consuming and laborious. If an adversary extracts the knowledge base verbatim, it not only severely infringes the owner's intellectual property but also enables the adversary to replicate the application's functionality for unfair competition. Previous works on knowledge base extraction are limited either by low extraction coverage (usually less than 4%) in query-based attacks or by impractical assumptions of white-box access in embedding-based optimization methods. In this work, we propose CopyBreakRAG, an agent-based black-box attack that reasons from feedback and adaptively generates new adversarial queries for progressive extraction. By balancing exploration and exploitation through curiosity-driven queries and feedback-guided query refinement, our method overcomes the limitations of prior approaches and achieves significantly higher extraction coverage in realistic black-box settings. Experimental results show that CopyBreakRAG outperforms the state-of-the-art black-box approach by 45% on average in terms of chunk extraction ratio from applications built with mainstream RAG frameworks, and extracts over 70% of the data from the knowledge base in applications on commercial platforms including OpenAI's GPTs and ByteDance's Coze when essential protection is in place.



## **42. LeakAgent: RL-based Red-teaming Agent for LLM Privacy Leakage**

cs.CR

Accepted by COLM 2025

**SubmitDate**: 2025-08-08    [abs](http://arxiv.org/abs/2412.05734v2) [paper-pdf](http://arxiv.org/pdf/2412.05734v2)

**Authors**: Yuzhou Nie, Zhun Wang, Ye Yu, Xian Wu, Xuandong Zhao, Wenbo Guo, Dawn Song

**Abstract**: Recent studies have discovered that large language models (LLM) may be ``fooled'' to output private information, including training data, system prompts, and personally identifiable information, under carefully crafted adversarial prompts. Existing red-teaming approaches for privacy leakage either rely on manual efforts or focus solely on system prompt extraction, making them ineffective for severe risks of training data leakage. We propose LeakAgent, a novel black-box red-teaming framework for LLM privacy leakage. Our framework trains an open-source LLM through reinforcement learning as the attack agent to generate adversarial prompts for both training data extraction and system prompt extraction. To achieve this, we propose a novel reward function to provide effective and fine-grained rewards and design novel mechanisms to balance exploration and exploitation during learning and enhance the diversity of adversarial prompts. Through extensive evaluations, we first show that LeakAgent significantly outperforms existing rule-based approaches in training data extraction and automated methods in system prompt leakage. We also demonstrate the effectiveness of LeakAgent in extracting system prompts from real-world applications in OpenAI's GPT Store. We further demonstrate LeakAgent's effectiveness in evading the existing guardrail defense and its helpfulness in enabling better safety alignment. Finally, we validate our customized designs through a detailed ablation study. We release our code here https://github.com/rucnyz/LeakAgent.



## **43. SLIP: Soft Label Mechanism and Key-Extraction-Guided CoT-based Defense Against Instruction Backdoor in APIs**

cs.CR

**SubmitDate**: 2025-08-08    [abs](http://arxiv.org/abs/2508.06153v1) [paper-pdf](http://arxiv.org/pdf/2508.06153v1)

**Authors**: Zhengxian Wu, Juan Wen, Wanli Peng, Haowei Chang, Yinghan Zhou, Yiming Xue

**Abstract**: With the development of customized large language model (LLM) agents, a new threat of black-box backdoor attacks has emerged, where malicious instructions are injected into hidden system prompts. These attacks easily bypass existing defenses that rely on white-box access, posing a serious security challenge. To address this, we propose SLIP, a Soft Label mechanism and key-extraction-guided CoT-based defense against Instruction backdoors in APIs. SLIP is designed based on two key insights. First, to counteract the model's oversensitivity to triggers, we propose a Key-extraction-guided Chain-of-Thought (KCoT). Instead of only considering the single trigger or the input sentence, KCoT prompts the agent to extract task-relevant key phrases. Second, to guide the LLM toward correct answers, our proposed Soft Label Mechanism (SLM) prompts the agent to quantify the semantic correlation between key phrases and candidate answers. Crucially, to mitigate the influence of residual triggers or misleading content in phrases extracted by KCoT, which typically causes anomalous scores, SLM excludes anomalous scores deviating significantly from the mean and subsequently averages the remaining scores to derive a more reliable semantic representation. Extensive experiments on classification and question-answer (QA) tasks demonstrate that SLIP is highly effective, reducing the average attack success rate (ASR) from 90.2% to 25.13% while maintaining high accuracy on clean data and outperforming state-of-the-art defenses. Our code are available in https://github.com/CAU-ISS-Lab/Backdoor-Attack-Defense-LLMs/tree/main/SLIP.



## **44. Fine-Grained Safety Neurons with Training-Free Continual Projection to Reduce LLM Fine Tuning Risks**

cs.LG

**SubmitDate**: 2025-08-08    [abs](http://arxiv.org/abs/2508.09190v1) [paper-pdf](http://arxiv.org/pdf/2508.09190v1)

**Authors**: Bing Han, Feifei Zhao, Dongcheng Zhao, Guobin Shen, Ping Wu, Yu Shi, Yi Zeng

**Abstract**: Fine-tuning as service injects domain-specific knowledge into large language models (LLMs), while challenging the original alignment mechanisms and introducing safety risks. A series of defense strategies have been proposed for the alignment, fine-tuning, and post-fine-tuning phases, where most post-fine-tuning defenses rely on coarse-grained safety layer mapping. These methods lack a comprehensive consideration of both safety layers and fine-grained neurons, limiting their ability to efficiently balance safety and utility. To address this, we propose the Fine-Grained Safety Neurons (FGSN) with Training-Free Continual Projection method to reduce the fine-tuning safety risks. FGSN inherently integrates the multi-scale interactions between safety layers and neurons, localizing sparser and more precise fine-grained safety neurons while minimizing interference with downstream task neurons. We then project the safety neuron parameters onto safety directions, improving model safety while aligning more closely with human preferences. Extensive experiments across multiple fine-tuned LLM models demonstrate that our method significantly reduce harmfulness scores and attack success rates with minimal parameter modifications, while preserving the model's utility. Furthermore, by introducing a task-specific, multi-dimensional heterogeneous safety neuron cluster optimization mechanism, we achieve continual defense and generalization capability against unforeseen emerging safety concerns.



## **45. Safety of Embodied Navigation: A Survey**

cs.AI

**SubmitDate**: 2025-08-07    [abs](http://arxiv.org/abs/2508.05855v1) [paper-pdf](http://arxiv.org/pdf/2508.05855v1)

**Authors**: Zixia Wang, Jia Hu, Ronghui Mu

**Abstract**: As large language models (LLMs) continue to advance and gain influence, the development of embodied AI has accelerated, drawing significant attention, particularly in navigation scenarios. Embodied navigation requires an agent to perceive, interact with, and adapt to its environment while moving toward a specified target in unfamiliar settings. However, the integration of embodied navigation into critical applications raises substantial safety concerns. Given their deployment in dynamic, real-world environments, ensuring the safety of such systems is critical. This survey provides a comprehensive analysis of safety in embodied navigation from multiple perspectives, encompassing attack strategies, defense mechanisms, and evaluation methodologies. Beyond conducting a comprehensive examination of existing safety challenges, mitigation technologies, and various datasets and metrics that assess effectiveness and robustness, we explore unresolved issues and future research directions in embodied navigation safety. These include potential attack methods, mitigation strategies, more reliable evaluation techniques, and the implementation of verification frameworks. By addressing these critical gaps, this survey aims to provide valuable insights that can guide future research toward the development of safer and more reliable embodied navigation systems. Furthermore, the findings of this study have broader implications for enhancing societal safety and increasing industrial efficiency.



## **46. No Query, No Access**

cs.CL

**SubmitDate**: 2025-08-07    [abs](http://arxiv.org/abs/2505.07258v2) [paper-pdf](http://arxiv.org/pdf/2505.07258v2)

**Authors**: Wenqiang Wang, Siyuan Liang, Yangshijie Zhang, Xiaojun Jia, Hao Lin, Xiaochun Cao

**Abstract**: Textual adversarial attacks mislead NLP models, including Large Language Models (LLMs), by subtly modifying text. While effective, existing attacks often require knowledge of the victim model, extensive queries, or access to training data, limiting real-world feasibility. To overcome these constraints, we introduce the \textbf{Victim Data-based Adversarial Attack (VDBA)}, which operates using only victim texts. To prevent access to the victim model, we create a shadow dataset with publicly available pre-trained models and clustering methods as a foundation for developing substitute models. To address the low attack success rate (ASR) due to insufficient information feedback, we propose the hierarchical substitution model design, generating substitute models to mitigate the failure of a single substitute model at the decision boundary.   Concurrently, we use diverse adversarial example generation, employing various attack methods to generate and select the adversarial example with better similarity and attack effectiveness. Experiments on the Emotion and SST5 datasets show that VDBA outperforms state-of-the-art methods, achieving an ASR improvement of 52.08\% while significantly reducing attack queries to 0. More importantly, we discover that VDBA poses a significant threat to LLMs such as Qwen2 and the GPT family, and achieves the highest ASR of 45.99% even without access to the API, confirming that advanced NLP models still face serious security risks. Our codes can be found at https://anonymous.4open.science/r/VDBA-Victim-Data-based-Adversarial-Attack-36EC/



## **47. JULI: Jailbreak Large Language Models by Self-Introspection**

cs.LG

**SubmitDate**: 2025-08-07    [abs](http://arxiv.org/abs/2505.11790v3) [paper-pdf](http://arxiv.org/pdf/2505.11790v3)

**Authors**: Jesson Wang, Zhanhao Hu, David Wagner

**Abstract**: Large Language Models (LLMs) are trained with safety alignment to prevent generating malicious content. Although some attacks have highlighted vulnerabilities in these safety-aligned LLMs, they typically have limitations, such as necessitating access to the model weights or the generation process. Since proprietary models through API-calling do not grant users such permissions, these attacks find it challenging to compromise them. In this paper, we propose Jailbreaking Using LLM Introspection (JULI), which jailbreaks LLMs by manipulating the token log probabilities, using a tiny plug-in block, BiasNet. JULI relies solely on the knowledge of the target LLM's predicted token log probabilities. It can effectively jailbreak API-calling LLMs under a black-box setting and knowing only top-$5$ token log probabilities. Our approach demonstrates superior effectiveness, outperforming existing state-of-the-art (SOTA) approaches across multiple metrics.



## **48. From Detection to Correction: Backdoor-Resilient Face Recognition via Vision-Language Trigger Detection and Noise-Based Neutralization**

cs.CV

19 Pages, 24 Figures

**SubmitDate**: 2025-08-07    [abs](http://arxiv.org/abs/2508.05409v1) [paper-pdf](http://arxiv.org/pdf/2508.05409v1)

**Authors**: Farah Wahida, M. A. P. Chamikara, Yashothara Shanmugarasa, Mohan Baruwal Chhetri, Thilina Ranbaduge, Ibrahim Khalil

**Abstract**: Biometric systems, such as face recognition systems powered by deep neural networks (DNNs), rely on large and highly sensitive datasets. Backdoor attacks can subvert these systems by manipulating the training process. By inserting a small trigger, such as a sticker, make-up, or patterned mask, into a few training images, an adversary can later present the same trigger during authentication to be falsely recognized as another individual, thereby gaining unauthorized access. Existing defense mechanisms against backdoor attacks still face challenges in precisely identifying and mitigating poisoned images without compromising data utility, which undermines the overall reliability of the system. We propose a novel and generalizable approach, TrueBiometric: Trustworthy Biometrics, which accurately detects poisoned images using a majority voting mechanism leveraging multiple state-of-the-art large vision language models. Once identified, poisoned samples are corrected using targeted and calibrated corrective noise. Our extensive empirical results demonstrate that TrueBiometric detects and corrects poisoned images with 100\% accuracy without compromising accuracy on clean images. Compared to existing state-of-the-art approaches, TrueBiometric offers a more practical, accurate, and effective solution for mitigating backdoor attacks in face recognition systems.



## **49. PhysPatch: A Physically Realizable and Transferable Adversarial Patch Attack for Multimodal Large Language Models-based Autonomous Driving Systems**

cs.CV

**SubmitDate**: 2025-08-07    [abs](http://arxiv.org/abs/2508.05167v1) [paper-pdf](http://arxiv.org/pdf/2508.05167v1)

**Authors**: Qi Guo, Xiaojun Jia, Shanmin Pang, Simeng Qin, Lin Wang, Ju Jia, Yang Liu, Qing Guo

**Abstract**: Multimodal Large Language Models (MLLMs) are becoming integral to autonomous driving (AD) systems due to their strong vision-language reasoning capabilities. However, MLLMs are vulnerable to adversarial attacks, particularly adversarial patch attacks, which can pose serious threats in real-world scenarios. Existing patch-based attack methods are primarily designed for object detection models and perform poorly when transferred to MLLM-based systems due to the latter's complex architectures and reasoning abilities. To address these limitations, we propose PhysPatch, a physically realizable and transferable adversarial patch framework tailored for MLLM-based AD systems. PhysPatch jointly optimizes patch location, shape, and content to enhance attack effectiveness and real-world applicability. It introduces a semantic-based mask initialization strategy for realistic placement, an SVD-based local alignment loss with patch-guided crop-resize to improve transferability, and a potential field-based mask refinement method. Extensive experiments across open-source, commercial, and reasoning-capable MLLMs demonstrate that PhysPatch significantly outperforms prior methods in steering MLLM-based AD systems toward target-aligned perception and planning outputs. Moreover, PhysPatch consistently places adversarial patches in physically feasible regions of AD scenes, ensuring strong real-world applicability and deployability.



## **50. AI Agent Smart Contract Exploit Generation**

cs.CR

**SubmitDate**: 2025-08-07    [abs](http://arxiv.org/abs/2507.05558v3) [paper-pdf](http://arxiv.org/pdf/2507.05558v3)

**Authors**: Arthur Gervais, Liyi Zhou

**Abstract**: Smart contract vulnerabilities have led to billions in losses, yet finding actionable exploits remains challenging. Traditional fuzzers rely on rigid heuristics and struggle with complex attacks, while human auditors are thorough but slow and don't scale. Large Language Models offer a promising middle ground, combining human-like reasoning with machine speed.   However, early studies show that simply prompting LLMs generates unverified vulnerability speculations with high false positive rates. To address this, we present A1, an agentic system that transforms any LLM into an end-to-end exploit generator. A1 provides agents with six domain-specific tools for autonomous vulnerability discovery, from understanding contract behavior to testing strategies on real blockchain states. All outputs are concretely validated through execution, ensuring only profitable proof-of-concept exploits are reported. We evaluate A1 across 36 real-world vulnerable contracts on Ethereum and Binance Smart Chain. A1 achieves a 63% success rate on the VERITE benchmark. Across all successful cases, A1 extracts up to \$8.59 million per exploit and \$9.33 million total. Through 432 experiments across six LLMs, we show that most exploits emerge within five iterations, with costs ranging \$0.01-\$3.59 per attempt.   Using Monte Carlo analysis of historical attacks, we demonstrate that immediate vulnerability detection yields 86-89% success probability, dropping to 6-21% with week-long delays. Our economic analysis reveals a troubling asymmetry: attackers achieve profitability at \$6,000 exploit values while defenders require \$60,000 -- raising fundamental questions about whether AI agents inevitably favor exploitation over defense.



